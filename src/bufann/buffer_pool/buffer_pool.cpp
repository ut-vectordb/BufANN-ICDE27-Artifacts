#include "bufann/buffer_pool/buffer_pool.h"

#include "ann_exception.h"
#include "aligned_file_reader.h"

#include <libaio.h>

#include <condition_variable>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace diskann {
namespace inplace {

namespace {

inline PageID invalid_page_id() {
    return {INVALID_FILE, INVALID_PAGE};
}

inline bool valid_page_id(PageID page_id) {
    return page_id.first != INVALID_FILE && page_id.second != INVALID_PAGE;
}

inline PageID frame_page_id(const PageFrame& frame) {
    return {frame.file_no.load(std::memory_order_acquire),
            frame.page_no.load(std::memory_order_acquire)};
}

inline void set_frame_page_id(PageFrame& frame, PageID page_id) {
    frame.file_no.store(page_id.first, std::memory_order_release);
    frame.page_no.store(page_id.second, std::memory_order_release);
}

inline uint64_t elapsed_ns(
    const std::chrono::steady_clock::time_point& start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

}  // namespace

BufferPool::BufferPool() = default;

BufferPool::~BufferPool() {
    stop_bg_flush();
}

PageGuard::PageGuard(PageGuard&& other) noexcept {
    *this = std::move(other);
}

PageGuard& PageGuard::operator=(PageGuard&& other) noexcept {
    if (this == &other) return *this;
    release();
    _pool = other._pool;
    _frame = other._frame;
    _mode = other._mode;
    _dirty = other._dirty;
    _discard = other._discard;
    other._pool = nullptr;
    other._frame = nullptr;
    other._dirty = false;
    other._discard = false;
    return *this;
}

PageGuard::~PageGuard() {
    release();
}

bool PageGuard::dirty() const {
    return _frame != nullptr && (_dirty || _frame->dirty.load(std::memory_order_acquire));
}

PageID PageGuard::page_id() const {
    return _frame == nullptr ? invalid_page_id() : _frame->page_id();
}

char* PageGuard::data() {
    return _frame == nullptr ? nullptr : _frame->data;
}

const char* PageGuard::data() const {
    return _frame == nullptr ? nullptr : _frame->data;
}

void PageGuard::release() {
    if (_pool == nullptr || _frame == nullptr) return;
    BufferPool* pool = _pool;
    PageFrame* frame = _frame;
    const bool dirty = _dirty;
    const bool discard = _discard;
    const AccessMode mode = _mode;
    _pool = nullptr;
    _frame = nullptr;
    _dirty = false;
    _discard = false;
    pool->unpin_frame(frame, dirty, mode);
    if (discard) pool->discard_frame(frame);
}



uint32_t BufferPool::shard_index(PageID page_id) const {
    return _num_shards == 0 ? 0 : (PageIDHash{}(page_id) % _num_shards);
}

BufferPool::ShardState& BufferPool::shard_for_page(PageID page_id) {
    return *_shards[shard_index(page_id)];
}

const BufferPool::ShardState& BufferPool::shard_for_page(PageID page_id) const {
    return *_shards[shard_index(page_id)];
}

void BufferPool::require_frames(const char* caller) const {
    if (!_frames.empty()) [[likely]] return;
    throw ANNException(
        std::string(caller) +
            ": buffer pool has no frames",
        -1, __FUNCSIG__, __FILE__, __LINE__);
}

void BufferPool::read_page(PageID page_id, char* buf) {
    _file_manager->read_page(page_id, buf);
    record_thread_physical_reads();
}

void BufferPool::read_batch(FileNo file_no, void* ctx_ptr, std::vector<AlignedRead>& reqs) {
    const auto ctx = static_cast<io_context_t>(ctx_ptr);
    const int fd = _file_manager->fd(file_no);
    const long n = static_cast<long>(reqs.size());
    if (n > static_cast<long>(kPinBatchMax)) {
        throw std::invalid_argument("BufferPool::read_batch: size exceeds kPinBatchMax (256)");
    }
    struct iocb cb[kPinBatchMax];
    struct iocb* cbs[kPinBatchMax];
    struct io_event evts[kPinBatchMax];
    for (long i = 0; i < n; ++i) {
        io_prep_pread(&cb[i], fd, reqs[i].buf, reqs[i].len, static_cast<long long>(reqs[i].offset));
        cbs[i] = &cb[i];
    }

    // In-flight reads target frame buffers, so every submitted read is reaped before any error is thrown.
    int error = 0;
    long submitted = 0;
    while (submitted < n) {
        const int ret = io_submit(ctx, n - submitted, cbs + submitted);
        if (ret == -EAGAIN || ret == -EINTR) continue;
        if (ret <= 0) { error = ret == 0 ? EIO : -ret; break; }
        submitted += ret;
    }

    long done = 0;
    while (done < submitted) {
        const int got = io_getevents(ctx, submitted - done, submitted - done, evts, nullptr);
        if (got == -EINTR) continue;
        if (got < 0) {
            throw std::runtime_error("BufferPool::read_batch: io_getevents failed: " + std::string(std::strerror(-got)));
        }
        for (int e = 0; e < got; ++e) {
            const long i = static_cast<long>(evts[e].obj - cb);
            const long res = static_cast<long>(evts[e].res);
            if (res < 0) {
                error = static_cast<int>(-res);
            } else if (static_cast<uint64_t>(res) < reqs[i].len) {
                memset(static_cast<char*>(reqs[i].buf) + res, 0, reqs[i].len - res);  // past EOF, as read_page does
            }
        }
        done += got;
    }
    if (error != 0) {
        throw std::runtime_error("BufferPool::read_batch: read failed: " + std::string(std::strerror(error)));
    }
}

void BufferPool::write_pages(const PageID* page_ids, const char* const* bufs, size_t n) {
    if (n == 0) return;
    thread_local io_context_t ctx = 0;
    size_t submitted = 0;
    if (ctx != 0 || io_setup(kPinBatchMax, &ctx) == 0) {
        struct iocb cb[kPinBatchMax];
        struct iocb* cbs[kPinBatchMax];
        struct io_event evts[kPinBatchMax];
        for (size_t i = 0; i < n; ++i) {
            io_prep_pwrite(&cb[i], _file_manager->fd(page_ids[i].first), const_cast<char*>(bufs[i]), _page_size,
                           static_cast<long long>(page_ids[i].second) * _page_size);
            cbs[i] = &cb[i];
        }
        while (submitted < n) {
            const int ret = io_submit(ctx, static_cast<long>(n - submitted), cbs + submitted);
            if (ret == -EAGAIN || ret == -EINTR) continue;
            if (ret <= 0) break;
            submitted += static_cast<size_t>(ret);
        }
        for (size_t done = 0; done < submitted;) {
            const int got = io_getevents(ctx, static_cast<long>(submitted - done), static_cast<long>(submitted - done),
                                         evts, nullptr);
            if (got == -EINTR) continue;
            if (got < 0) break;
            done += static_cast<size_t>(got);
        }
    } else {
        ctx = 0;
    }
    for (size_t i = submitted; i < n; ++i) {  // no io context, or whatever io_submit refused
        _file_manager->write_page(page_ids[i], bufs[i]);
    }
}

void BufferPool::finish_writeback(PageID victim) { shard_for_page(victim).page_table.erase(victim); }

uint32_t BufferPool::evict_one(ShardState& shard, bool* used_eviction_out, PageID* writeback_out) {
    if (writeback_out) *writeback_out = invalid_page_id();
    const auto evict_begin = std::chrono::steady_clock::now();
    const uint32_t shard_frames = shard.frame_end - shard.frame_begin;
    const uint32_t budget = std::max<uint32_t>(1, shard_frames) * 2;

    uint32_t idx = INVALID_PAGE;
    PageID victim_pid = invalid_page_id();
    bool evict = false;

    // Pick and exclusively latch a candidate under the cursor lock, but do any
    // writeback outside it so a disk write never serializes the shard.
    {
        std::lock_guard<SpinLock> clk(shard.cursor_lock);
        for (uint32_t scanned = 0; scanned < budget; ++scanned) {
            const uint32_t cand = shard.clock_hand;
            if (++shard.clock_hand >= shard.frame_end) {
                shard.clock_hand = shard.frame_begin;
            }
            PageFrame& f = _frames[cand];
            // try_lock fails if anyone holds the frame (shared or exclusive):
            // that is exactly "in use, do not evict".
            if (!f.latch.try_lock()) continue;
            const PageID pid = frame_page_id(f);
            if (!valid_page_id(pid)) {
                idx = cand;  // already-free frame; keep it latched
                break;
            }
            if (f.ref_bit.exchange(0, std::memory_order_acq_rel)) {
                f.latch.unlock();  // second chance
                continue;
            }
            idx = cand;
            victim_pid = pid;
            evict = true;
            break;
        }
    }

    // The scan steps over every frame twice, so it finds nothing only if every
    // one was latched throughout -- i.e. more live pins than frames, a caller
    // bug we cannot recover from.
    if (idx == INVALID_PAGE) {
        throw ANNException(
            "BufferPool::evict_one: no reusable frame; more live pins than "
            "frames in a shard",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    PageFrame& f = _frames[idx];
    if (!evict) {
        if (used_eviction_out) *used_eviction_out = false;
        return idx;  // held exclusive
    }

    // We hold f exclusive, so its bytes are stable. Write it back if dirty, then drop the old mapping.
    // The mapping stays until the write is done, so a concurrent miss waits on this frame's latch instead of reading
    // the stale copy from disk. Erasing while latched keeps the lock order frame-then-map.
    const bool was_dirty = f.dirty.exchange(false, std::memory_order_acq_rel);
    if (was_dirty) _dirty_frame_count.fetch_sub(1, std::memory_order_relaxed);
    if (_stats) {
        _stats->evictions.fetch_add(1);
        if (was_dirty) _stats->dirty_evictions.fetch_add(1);
    }
    if (was_dirty && writeback_out) {
        *writeback_out = victim_pid;  // the caller writes it and drops the mapping
        set_frame_page_id(f, invalid_page_id());
        f.ref_bit.store(0, std::memory_order_release);
        if (used_eviction_out) *used_eviction_out = true;
        return idx;
    }
    if (was_dirty) {
        const auto flush_begin = std::chrono::steady_clock::now();
        const char* buf = f.data;
        write_pages(&victim_pid, &buf, 1);
        if (_stats) {
            _stats->pages_flushed.fetch_add(1);
            _stats->dirty_page_bytes_flushed.fetch_add(_page_size);
            _stats->eviction_flush_ns.fetch_add(elapsed_ns(flush_begin),
                                                std::memory_order_relaxed);
        }
    }
    shard.page_table.erase(victim_pid);
    set_frame_page_id(f, invalid_page_id());
    f.ref_bit.store(0, std::memory_order_release);
    if (_stats) {
        _stats->eviction_total_ns.fetch_add(elapsed_ns(evict_begin),
                                            std::memory_order_relaxed);
    }
    if (used_eviction_out) *used_eviction_out = true;
    return idx;
}

PageFrame& BufferPool::pin_frame(PageID page_id, AccessMode mode) {
    require_frames("pin");
    ShardState& shard = shard_for_page(page_id);
    for (;;) {
        // Resident fast path.
        const uint32_t idx = shard.page_table.find(page_id);
        if (idx != INVALID_PAGE) {
            PageFrame& f = _frames[idx];
            if (kOptimisticReads && mode == AccessMode::READ) {
                if (_stats) record_query_cache_hit();
                return f;
            }
            mode == AccessMode::READ ? f.latch.lock_shared() : f.latch.lock();
            if (frame_page_id(f) == page_id) {
                f.ref_bit.store(1, std::memory_order_release);
                if (_stats) {
                    if (mode == AccessMode::READ) record_query_cache_hit();
                    else _stats->cache_hits.fetch_add(1);
                }
                return f;  // caller's guard now owns the latch
            }
            // Evicted and reused between lookup and latch; retry.
            mode == AccessMode::READ ? f.latch.unlock_shared() : f.latch.unlock();
            continue;
        }

        // Miss: obtain an exclusively-latched free frame and install it.
        const auto page_fault_begin = std::chrono::steady_clock::now();
        bool used_eviction = false;
        uint32_t frame_idx = evict_one(shard, &used_eviction);
        PageFrame& f = _frames[frame_idx];  // held exclusive by evict_one
        set_frame_page_id(f, page_id);
        f.dirty.store(false, std::memory_order_release);

        if (!shard.page_table.try_insert(page_id, frame_idx)) {
            // Another thread installed this page first. Release the frame and
            // retry; the next pass finds the page resident.
            set_frame_page_id(f, invalid_page_id());
            f.latch.unlock();
            continue;
        }

        try {
            const auto io_begin = std::chrono::high_resolution_clock::now();
            read_page(page_id, f.data);
            if (_stats) {
                const uint64_t page_in_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - io_begin)
                        .count());
                _stats->page_in_ns.fetch_add(page_in_ns, std::memory_order_relaxed);
                _stats->record_io_ns(mode == AccessMode::READ, page_in_ns);
            }
        } catch (...) {
            shard.page_table.erase(page_id);
            set_frame_page_id(f, invalid_page_id());
            f.latch.unlock();
            throw;
        }

        f.ref_bit.store(1, std::memory_order_release);
        if (_stats) {
            _stats->page_fault_total.fetch_add(1, std::memory_order_relaxed);
            _stats->cache_misses.fetch_add(1, std::memory_order_relaxed);
            if (used_eviction) _stats->page_fault_with_evict.fetch_add(1, std::memory_order_relaxed);
            else _stats->page_fault_no_evict.fetch_add(1, std::memory_order_relaxed);
            _stats->page_fault_total_ns.fetch_add(elapsed_ns(page_fault_begin),
                                                  std::memory_order_relaxed);
        }

        if (mode == AccessMode::WRITE) {
            return f;  // already exclusive, which is what a WRITE pin needs
        }
        if (kOptimisticReads) {
            f.latch.unlock();
            return f;
        }
        // A READ pin needs the latch shared, but we loaded under exclusive and
        // shared_mutex has no downgrade. Release and loop; the page is resident
        // now, so the next pass takes it shared. That pass also tallies a hit on
        // top of this miss -- a small inflation, and only here: the batched
        // query path latches shared directly and does not round-trip.
        f.latch.unlock();
    }
}

PageGuard BufferPool::pin(PageID page_id, AccessMode mode) {
    return PageGuard(this, &pin_frame(page_id, mode), mode);
}

bool BufferPool::is_resident(PageID page_id) const {
    return shard_for_page(page_id).page_table.find(page_id) != INVALID_PAGE;
}

PageGuard BufferPool::try_pin_resident(PageID page_id, AccessMode mode) {
    require_frames("try_pin_resident");
    const uint32_t idx = shard_for_page(page_id).page_table.find(page_id);
    if (idx == INVALID_PAGE) return PageGuard();
    PageFrame& f = _frames[idx];
    if (kOptimisticReads && mode == AccessMode::READ) return PageGuard(this, &f, mode);
    if (!(mode == AccessMode::READ ? f.latch.try_lock_shared() : f.latch.try_lock())) return PageGuard();
    if (frame_page_id(f) != page_id) {
        mode == AccessMode::READ ? f.latch.unlock_shared() : f.latch.unlock();
        return PageGuard();
    }
    f.ref_bit.store(1, std::memory_order_release);
    return PageGuard(this, &f, mode);
}

static_assert(BufferPool::kPinBatchMax <= 256,
              "kPinBatchMax must fit in MAX_EVENTS (256) of linux_aligned_file_reader");

void BufferPool::pin_batch(const PageID* page_ids, PageGuard* out_guards, size_t n, AccessMode mode,
                           bool account_sequential_read, bool* out_cache_hit) {
    pin_batch_frames(page_ids, out_guards, n, mode, account_sequential_read, /*transient=*/false, out_cache_hit);
}

void BufferPool::pin_batch_transient(const PageID* page_ids, PageGuard* out_guards,
                                     size_t n, AccessMode mode,
                                     bool account_sequential_read) {
    pin_batch_frames(page_ids, out_guards, n, mode, account_sequential_read, /*transient=*/true,
                     /*out_cache_hit=*/nullptr);
}

void BufferPool::pin_batch_frames(const PageID* page_ids, PageGuard* out_guards,
                                  size_t n, AccessMode mode,
                                  bool account_sequential_read, bool transient,
                                  bool* out_cache_hit) {
    require_frames("pin_batch");
    if (n == 0) return;
    if (n > kPinBatchMax) {
        throw std::invalid_argument(
            "BufferPool::pin_batch: size exceeds kPinBatchMax (256)");
    }
    // page_ids must be unique. Each then maps to a distinct frame, so this call
    // never has to take one frame's latch twice, which a shared_mutex forbids.
    const FileNo batch_file = page_ids[0].first;
    for (size_t i = 1; i < n; ++i) {
        if (page_ids[i].first != batch_file) {
            throw std::runtime_error("not implemented");
        }
    }

    AlignedFileReader& reader = _file_manager->reader(batch_file);
    IOContext bad = (IOContext)(-1);
    IOContext ctx = reader.get_ctx();
    if (ctx == bad) {
        reader.register_thread();
        ctx = reader.get_ctx();
    }
    if (ctx == bad) {
        throw std::runtime_error("BufferPool::pin_batch: io_setup failed for this thread");
    }

    uint64_t n_miss = 0, n_with_evict = 0, n_no_evict = 0, batch_ns = 0;

    // Two passes. The resolve pass records each slot's frame index -- from the
    // map for a hit, from evict_one for a miss -- and faults every miss in with
    // one batched read. A miss frame is held exclusive only across that read,
    // and only through evict_one's try_lock, so this pass never blocks on a
    // latch while already holding one.
    //
    // The latch pass then takes the frames in ascending frame-index order,
    // reusing the recorded indices. That fixed order keeps two concurrent
    // batches from deadlocking, and the reuse saves a second map lookup.
    struct Acq { size_t slot; uint32_t frame_idx; };

    bool cache_hit[kPinBatchMax];
    std::fill(cache_hit, cache_hit + n, true);
    for (;;) {
        n_miss = n_with_evict = n_no_evict = 0;
        batch_ns = 0;
        uint32_t slot_frame_idx[kPinBatchMax];
        PageFrame* miss_frames[kPinBatchMax];
        size_t n_miss_slots = 0;
        std::vector<AlignedRead> reqs;
        reqs.reserve(n);
        // Dirty victims of this batch's evictions, written together before the reads reuse their frames.
        PageID victims[kPinBatchMax];
        const char* victim_bufs[kPinBatchMax];
        size_t n_victims = 0;

        for (size_t i = 0; i < n; ++i) {
            const PageID pid = page_ids[i];
            ShardState& shard = shard_for_page(pid);
            for (;;) {
                const uint32_t idx = shard.page_table.find(pid);
                if (idx != INVALID_PAGE) {
                    slot_frame_idx[i] = idx;  // resident; validated later under latch
                    break;
                }
                bool used_evict = false;
                PageID victim;
                uint32_t fi = evict_one(shard, &used_evict, &victim);  // exclusive-latched
                PageFrame& f = _frames[fi];
                if (valid_page_id(victim)) {
                    victims[n_victims] = victim;
                    victim_bufs[n_victims++] = f.data;
                }
                set_frame_page_id(f, pid);
                f.dirty.store(false, std::memory_order_release);
                if (!shard.page_table.try_insert(pid, fi)) {
                    // The frame is released now, so its victim cannot wait for the batch's write.
                    if (n_victims > 0 && victim_bufs[n_victims - 1] == f.data) {
                        --n_victims;
                        write_pages(&victims[n_victims], &victim_bufs[n_victims], 1);
                        finish_writeback(victims[n_victims]);
                    }
                    set_frame_page_id(f, invalid_page_id());
                    f.latch.unlock();
                    continue;  // lost the race; it is resident now
                }
                slot_frame_idx[i] = fi;
                cache_hit[i] = false;
                miss_frames[n_miss_slots++] = &f;
                reqs.push_back(AlignedRead{
                    static_cast<uint64_t>(pid.second) *
                        static_cast<uint64_t>(_page_size),
                    static_cast<uint64_t>(_page_size), f.data});
                if (used_evict) ++n_with_evict; else ++n_no_evict;
                ++n_miss;
                break;
            }
        }

        if (n_victims > 0) {
            const auto flush_begin = std::chrono::steady_clock::now();
            write_pages(victims, victim_bufs, n_victims);
            for (size_t j = 0; j < n_victims; ++j) finish_writeback(victims[j]);
            if (_stats) {
                _stats->pages_flushed.fetch_add(n_victims);
                _stats->dirty_page_bytes_flushed.fetch_add(n_victims * _page_size);
                _stats->eviction_flush_ns.fetch_add(elapsed_ns(flush_begin), std::memory_order_relaxed);
            }
        }
        if (!reqs.empty()) {
            const auto io_begin = std::chrono::high_resolution_clock::now();
            try {
                read_batch(batch_file, ctx, reqs);
            } catch (...) {
                for (size_t j = 0; j < n_miss_slots; ++j) {
                    PageFrame* f = miss_frames[j];
                    const PageID pid = frame_page_id(*f);
                    shard_for_page(pid).page_table.erase(pid);
                    set_frame_page_id(*f, invalid_page_id());
                    f->latch.unlock();
                }
                throw;
            }
            batch_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - io_begin)
                    .count());
            for (size_t j = 0; j < n_miss_slots; ++j) {
                miss_frames[j]->ref_bit.store(1, std::memory_order_release);
                miss_frames[j]->latch.unlock();
            }
        }

        if (kOptimisticReads && mode == AccessMode::READ) {
            for (size_t i = 0; i < n; ++i) {
                out_guards[i] = PageGuard(this, &_frames[slot_frame_idx[i]], mode);
                if (transient && !cache_hit[i]) out_guards[i].discard_on_release();
            }
            break;
        }

        // Ordered-latch pass over the recorded indices.
        Acq acq[kPinBatchMax];
        for (size_t i = 0; i < n; ++i) acq[i] = Acq{i, slot_frame_idx[i]};
        std::sort(acq, acq + n, [](const Acq& a, const Acq& b) {
            return a.frame_idx < b.frame_idx;
        });
        size_t locked = 0;
        bool retry = false;
        for (size_t j = 0; j < n; ++j) {
            // A recorded hit's frame can be evicted and reused by a later miss
            // in this same batch, which shows up as two slots sharing a frame
            // index. Latching it twice from one thread is illegal, so restart
            // the whole batch (the stale page re-faults into a fresh frame).
            if (j > 0 && acq[j].frame_idx == acq[j - 1].frame_idx) {
                retry = true;
                break;
            }
            PageFrame& f = _frames[acq[j].frame_idx];
            mode == AccessMode::READ ? f.latch.lock_shared() : f.latch.lock();
            ++locked;
            if (frame_page_id(f) != page_ids[acq[j].slot]) {
                retry = true;  // evicted since we recorded the index
                break;
            }
        }
        if (retry) {
            for (size_t t = 0; t < locked; ++t) {
                PageFrame& uf = _frames[acq[t].frame_idx];
                mode == AccessMode::READ ? uf.latch.unlock_shared() : uf.latch.unlock();
            }
            continue;
        }
        for (size_t j = 0; j < n; ++j) {
            PageFrame& f = _frames[acq[j].frame_idx];
            f.ref_bit.store(1, std::memory_order_release);
            const size_t slot = acq[j].slot;
            out_guards[slot] = PageGuard(this, &f, mode);
            if (transient && !cache_hit[slot]) out_guards[slot].discard_on_release();
        }
        break;
    }
    if (out_cache_hit != nullptr) std::copy(cache_hit, cache_hit + n, out_cache_hit);

    if (n_miss > 0) record_thread_physical_reads(n_miss);
    if (_stats) {
        _stats->batch_pin_calls.fetch_add(1, std::memory_order_relaxed);
        const uint64_t logical_hits = static_cast<uint64_t>(n) - n_miss;
        if (logical_hits > 0) {
            if (mode == AccessMode::READ) record_query_cache_hit(logical_hits);
            else _stats->cache_hits.fetch_add(logical_hits, std::memory_order_relaxed);
        }
        if (n_miss > 0) {
            _stats->cache_misses.fetch_add(n_miss, std::memory_order_relaxed);
            const uint64_t read_bytes = n_miss * static_cast<uint64_t>(_page_size);
            _stats->batch_misses_total.fetch_add(n_miss, std::memory_order_relaxed);
            _stats->batch_with_evict.fetch_add(n_with_evict, std::memory_order_relaxed);
            _stats->batch_no_evict.fetch_add(n_no_evict, std::memory_order_relaxed);
            _stats->batch_io_ns_total.fetch_add(batch_ns, std::memory_order_relaxed);
            if (account_sequential_read) {
                _stats->sequential_read_ios.fetch_add(n_miss, std::memory_order_relaxed);
                _stats->sequential_read_bytes.fetch_add(read_bytes, std::memory_order_relaxed);
            }
        }
    }
}

void BufferPool::unpin_frame(PageFrame* frame, bool dirty, AccessMode mode) {
    if (frame == nullptr) return;
    if (dirty) {
        record_buffer_pool_write();
        mark_frame_dirty(*frame);
    }
    if (mode == AccessMode::WRITE) frame->latch.unlock();
    else if (!kOptimisticReads) frame->latch.unlock_shared();
}

void BufferPool::write_frames(PageFrame* const* frames, size_t n) {
    if (n == 0) return;
    PageID page_ids[kPinBatchMax];
    const char* bufs[kPinBatchMax];
    for (size_t i = 0; i < n; ++i) {
        page_ids[i] = frame_page_id(*frames[i]);
        bufs[i] = frames[i]->data;
    }
    write_pages(page_ids, bufs, n);
    for (size_t i = 0; i < n; ++i) {
        if (frames[i]->dirty.exchange(false, std::memory_order_acq_rel)) {
            _dirty_frame_count.fetch_sub(1, std::memory_order_relaxed);
        }
        frames[i]->latch.unlock_shared();
    }
    if (_stats) {
        _stats->pages_flushed.fetch_add(n);
        _stats->dirty_page_bytes_flushed.fetch_add(n * _page_size);
    }
}

// Best-effort reclaim of a transient page on release: if we can take the frame
// exclusively right now, evict it so it does not linger in cache. The caller
// has already released its own latch via unpin_frame.
void BufferPool::discard_frame(PageFrame* frame) {
    if (frame == nullptr) return;
    if (!frame->latch.try_lock()) return;  // still in use; leave it for CLOCK
    const PageID page_id = frame_page_id(*frame);
    if (!valid_page_id(page_id)) { frame->latch.unlock(); return; }

    // As in evict_one, the mapping stays until a dirty page is written back.
    const bool was_dirty = frame->dirty.exchange(false, std::memory_order_acq_rel);
    if (was_dirty) {
        _dirty_frame_count.fetch_sub(1, std::memory_order_relaxed);
        const char* buf = frame->data;
        write_pages(&page_id, &buf, 1);
        if (_stats) _stats->pages_flushed.fetch_add(1);
    }
    if (_stats) _stats->evictions.fetch_add(1);
    shard_for_page(page_id).page_table.erase(page_id);
    set_frame_page_id(*frame, invalid_page_id());
    frame->ref_bit.store(0, std::memory_order_release);
    frame->latch.unlock();
}

void BufferPool::unpin(PageID page_id, bool dirty, AccessMode mode) {
    require_frames("unpin");
    const uint32_t idx = shard_for_page(page_id).page_table.find(page_id);
    if (idx != INVALID_PAGE) unpin_frame(&_frames[idx], dirty, mode);
}

void BufferPool::mark_dirty(PageID page_id) {
    require_frames("mark_dirty");
    const uint32_t idx = shard_for_page(page_id).page_table.find(page_id);
    if (idx == INVALID_PAGE) return;
    record_buffer_pool_write();
    mark_frame_dirty(_frames[idx]);
}

void BufferPool::record_buffer_pool_write() {
    if (_stats == nullptr) return;
    _stats->buffer_pool_write_ios.fetch_add(1, std::memory_order_relaxed);
}

void BufferPool::mark_frame_dirty(PageFrame& frame) {
    if (!frame.dirty.exchange(true, std::memory_order_acq_rel)) {
        _dirty_frame_count.fetch_add(1, std::memory_order_relaxed);
    }
}
PageNo BufferPool::grow_heap(FileNo file_no, uint32_t batch_size) {
    const PageNo base = _file_manager->grow_file(file_no, batch_size);
    const uint64_t new_total_pages = static_cast<uint64_t>(base) + batch_size;
    if (new_total_pages > INVALID_PAGE) {
        throw ANNException("BufferPool::grow_heap: page count out of range",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    return base;
}

void BufferPool::set_total_pages(FileNo file_no, uint32_t used_pages) {
    _file_manager->set_total_pages(file_no, used_pages);
}

void BufferPool::bg_flush_loop(float high_wm, float low_wm) {
    while (_flush_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(_flush_wakeup_ms));
        uint32_t dirty_count = dirty_page_count();
        float ratio = (float)dirty_count / (float)(_num_frames ? _num_frames : 1);
        if (ratio <= high_wm) continue;
        for (auto& shard_ptr : _shards) {
            if (dirty_count <= static_cast<uint32_t>(low_wm * _num_frames)) break;
            auto frames = collect_dirty_frames(*shard_ptr, _flush_budget_pages_per_cycle);
            for (size_t i = 0; i < frames.size(); i += kPinBatchMax) {
                write_frames(frames.data() + i, std::min<size_t>(kPinBatchMax, frames.size() - i));
            }
            dirty_count = dirty_page_count();
        }
    }
}

void BufferPool::start_bg_flush(float high_wm, float low_wm) {
    if (_flush_running.load()) return;
    _flush_running.store(true);
    _flush_thread = std::thread(&BufferPool::bg_flush_loop, this,
                                high_wm, low_wm);
}

void BufferPool::stop_bg_flush() {
    _flush_running.store(false);
    if (_flush_thread.joinable()) _flush_thread.join();
}

void BufferPool::reset() {
    require_frames("reset");
    stop_bg_flush();
    for (auto& f : _frames) {
        set_frame_page_id(f, invalid_page_id());
        f.dirty.store(false, std::memory_order_release);
        f.ref_bit.store(0, std::memory_order_release);
    }
    _dirty_frame_count.store(0, std::memory_order_relaxed);
    for (auto& shard_ptr : _shards) {
        shard_ptr->page_table.clear();
    }
    for (auto& shard_ptr : _shards) {
        std::lock_guard<SpinLock> clk(shard_ptr->cursor_lock);
        shard_ptr->clock_hand = shard_ptr->frame_begin;
        shard_ptr->flush_cursor = shard_ptr->frame_begin;
    }
}


}  // namespace inplace
}  // namespace diskann
