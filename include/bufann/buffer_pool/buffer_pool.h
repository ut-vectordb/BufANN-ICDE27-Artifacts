#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include <parallel_hashmap/phmap.h>
#include <stdexcept>
#include <sys/mman.h>

#include "bufann/buffer_pool/file_manager.h"
#include "bufann/inplace_defs.h"
#include "bufann/inplace_io_stats.h"

struct AlignedRead;

namespace diskann {
namespace inplace {

class BufferPool;

struct SpinLock {
    std::atomic<bool> _locked{false};

    void lock() noexcept {
        for (;;) {
            if (!_locked.load(std::memory_order_relaxed) &&
                !_locked.exchange(true, std::memory_order_acquire))
                return;
            _mm_pause();
        }
    }

    void unlock() noexcept {
        _locked.store(false, std::memory_order_release);
    }

    bool try_lock() noexcept {
        return !_locked.load(std::memory_order_relaxed) &&
               !_locked.exchange(true, std::memory_order_acquire);
    }
};

struct PageFrame {
    std::atomic<FileNo> file_no{INVALID_FILE};
    std::atomic<PageNo> page_no{INVALID_PAGE};

    char*       data      = nullptr;
    std::atomic<bool> dirty{false};
    std::atomic<uint8_t> ref_bit{0};

    // The single per-frame lock. Holding it shared is a read pin; holding it
    // exclusive is a write pin, or a loader/evictor/flusher busy with the
    // frame. Residency follows from it: the evictor can only reclaim a frame
    // whose latch it can take exclusively, so a page anyone holds cannot be
    // evicted. There is no separate pin count.
    std::shared_mutex latch;
    uint32_t frame_index = INVALID_PAGE;

    ~PageFrame() { std::free(data); }

    PageID page_id() const {
        return {file_no.load(std::memory_order_acquire),
                page_no.load(std::memory_order_acquire)};
    }
};

struct PageIDHash {
    size_t operator()(const PageID& page_id) const noexcept {
        return (static_cast<size_t>(page_id.first) << 32) ^ static_cast<size_t>(page_id.second);
    }
};

// The page table maps each resident page to the index of its frame, or INVALID_PAGE. CMake's BUFANN_PAGE_TABLE picks
// it: "hash" (the default) is a hash map per shard, "array" is Calico's array translation, and "array_optimistic" is
// that plus Calico's relaxed reads (kOptimisticReads below).
class HashPageTable {
 public:
    uint32_t find(PageID page_id) const {
        uint32_t idx = INVALID_PAGE;
        _map.if_contains(page_id, [&](const auto& kv) { idx = kv.second; });
        return idx;
    }
    // Maps page_id to frame_idx unless it is mapped already, and returns whether it did.
    bool try_insert(PageID page_id, uint32_t frame_idx) {
        return _map.lazy_emplace_l(page_id, [](auto& /*kv*/) {}, [&](auto&& ctor) {
            ctor(std::piecewise_construct, std::make_tuple(page_id), std::make_tuple(frame_idx));
        });
    }
    void erase(PageID page_id) { _map.erase(page_id); }
    void clear() { _map.clear(); }

 private:
    phmap::parallel_flat_hash_map<PageID, uint32_t, PageIDHash, phmap::priv::hash_default_eq<PageID>,
                                  std::allocator<std::pair<const PageID, uint32_t>>, 8, std::shared_mutex>
        _map;
};

// Calico's array translation: a file's page number indexes an array of 8-byte entries holding the frame index plus
// one, so 0 means not resident. Each file's array is one anonymous mapping covering every page number, which the OS
// backs only where entries were written. All shards share the arrays; a shard still owns the frames its pages use.
class ArrayPageTable {
 public:
    static constexpr FileNo kMaxFiles = 4;
    static constexpr size_t kArrayBytes = (size_t{1} << 32) * sizeof(uint64_t);

    struct Arrays {
        std::atomic<uint64_t>* entries[kMaxFiles];
        Arrays() {
            for (auto& e : entries) {
                void* p = mmap(nullptr, kArrayBytes, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (p == MAP_FAILED) throw std::runtime_error("ArrayPageTable: mmap failed");
                e = static_cast<std::atomic<uint64_t>*>(p);
            }
        }
        ~Arrays() {
            for (auto* e : entries) munmap(e, kArrayBytes);
        }
    };

    void attach(std::shared_ptr<Arrays> arrays) { _arrays = std::move(arrays); }

    uint32_t find(PageID page_id) const {
        const uint64_t e = entry(page_id).load(std::memory_order_acquire);
        return e == 0 ? INVALID_PAGE : static_cast<uint32_t>(e - 1);
    }
    bool try_insert(PageID page_id, uint32_t frame_idx) {
        if (page_id.first >= kMaxFiles) throw std::runtime_error("ArrayPageTable: file number out of range");
        uint64_t expected = 0;
        return entry(page_id).compare_exchange_strong(expected, uint64_t{frame_idx} + 1, std::memory_order_acq_rel);
    }
    void erase(PageID page_id) { entry(page_id).store(0, std::memory_order_release); }
    // Gives every array's pages back to the OS, which reads them as zeros again.
    void clear() {
        for (auto* e : _arrays->entries) madvise(e, kArrayBytes, MADV_DONTNEED);
    }

 private:
    std::atomic<uint64_t>& entry(PageID page_id) const { return _arrays->entries[page_id.first][page_id.second]; }

    std::shared_ptr<Arrays> _arrays;
};

#ifdef BUFANN_ARRAY_PAGE_TABLE
using PageTable = ArrayPageTable;
#else
using PageTable = HashPageTable;
#endif

// With optimistic reads, a READ pin takes no latch, checks nothing and sets no reference bit, as Calico's relaxed
// reads do, with their version check left out as in its prototype: the evictor can reuse a frame while a reader still
// reads it, and the reader is never told. WRITE pins and the flusher still latch.
#ifdef BUFANN_OPTIMISTIC_READS
inline constexpr bool kOptimisticReads = true;
#else
inline constexpr bool kOptimisticReads = false;
#endif
inline constexpr const char* kPageTableName =
    kOptimisticReads ? "array_optimistic" : std::is_same_v<PageTable, ArrayPageTable> ? "array" : "hash";

class PageGuard {
 public:
    PageGuard() = default;
    PageGuard(const PageGuard&) = delete;
    PageGuard& operator=(const PageGuard&) = delete;
    PageGuard(PageGuard&& other) noexcept;
    PageGuard& operator=(PageGuard&& other) noexcept;
    ~PageGuard();

    bool valid() const { return _pool != nullptr && _frame != nullptr; }
    PageID page_id() const;
    char* data();
    const char* data() const;

    void mark_dirty() { _dirty = true; }
    void discard_on_release() { _discard = true; }
    // Whether the page holds changes not yet on disk, made through this guard or before it.
    bool dirty() const;
    void release();

 private:
    friend class BufferPool;
    PageGuard(BufferPool* pool, PageFrame* frame, AccessMode mode)
        : _pool(pool), _frame(frame), _mode(mode) {}

    BufferPool* _pool = nullptr;
    PageFrame* _frame = nullptr;
    AccessMode _mode = AccessMode::READ;
    bool _dirty = false;
    bool _discard = false;
};

class BufferPool {
    friend class PageGuard;
 public:

    void       flush_all_dirty();
    uint32_t   flush_dirty_budget(uint32_t max_pages);
    uint32_t   dirty_page_count() const;
    double     dirty_ratio() const;

    const std::string& heap_path(FileNo file_no) const { return _file_manager->path(file_no); }
    size_t configured_bytes() const noexcept;

    BufferPool();
    ~BufferPool();

    void init(uint32_t page_size, uint32_t num_frames, std::shared_ptr<FileManager> file_manager,
              FileNo default_file_no, InPlaceIOStats* stats,
              uint32_t flush_budget_pages_per_cycle = 32, uint32_t flush_wakeup_ms = 100);

    PageGuard pin(PageID page_id, AccessMode mode = AccessMode::READ);
    // Pins page_id only if it is resident and its latch is free right now, without blocking or reading from disk.
    // Returns an invalid guard otherwise.
    PageGuard try_pin_resident(PageID page_id, AccessMode mode);
    // Whether page_id is mapped to a frame right now, without latching it.
    bool is_resident(PageID page_id) const;

    static constexpr uint32_t kPinBatchMax = 256;
    // If out_cache_hit is non-null, out_cache_hit[i] is set to whether page_ids[i] was already cached.
    void pin_batch(const PageID* page_ids, PageGuard* out_guards,
                   size_t n, AccessMode mode = AccessMode::READ,
                   bool account_sequential_read = false, bool* out_cache_hit = nullptr);
    void pin_batch_transient(const PageID* page_ids, PageGuard* out_guards,
                             size_t n, AccessMode mode = AccessMode::READ,
                             bool account_sequential_read = false);
    void unpin(PageID page_id, bool dirty = false, AccessMode mode = AccessMode::READ);
    void mark_dirty(PageID page_id);
    PageNo grow_heap(FileNo file_no, uint32_t batch_size);
    void set_total_pages(FileNo file_no, uint32_t used_pages);

    void start_bg_flush(float high_wm = 0.70f, float low_wm = 0.30f);
    void stop_bg_flush();
    void reset();

    uint32_t page_size() const { return _page_size; }
    uint32_t num_frames() const { return _num_frames; }
    uint32_t total_pages(FileNo file_no) const { return _file_manager->total_pages(file_no); }
    FileNo default_file_no() const { return _default_file_no; }
    const std::shared_ptr<FileManager>& file_manager() const { return _file_manager; }

 private:
    struct ShardState {
        PageTable page_table;

        mutable SpinLock cursor_lock;
        uint32_t clock_hand   = 0;
        uint32_t flush_cursor = 0;
        uint32_t frame_begin   = 0;
        uint32_t frame_end     = 0;
    };

    void require_frames(const char* caller) const;

    // Returns dirty frames of the shard, each held with its latch shared, for write_frames.
    std::vector<PageFrame*> collect_dirty_frames(ShardState& shard, uint32_t max_frames);
    // Writes frames held shared, at most kPinBatchMax, in one batch, marks them clean once their writes are on disk
    // and releases their latches. Holding the latch until then keeps a concurrent miss from reading an older copy.
    void write_frames(PageFrame* const* frames, size_t n);

    std::deque<PageFrame> _frames;

    uint32_t shard_index(PageID page_id) const;
    ShardState& shard_for_page(PageID page_id);
    const ShardState& shard_for_page(PageID page_id) const;
    PageFrame& pin_frame(PageID page_id, AccessMode mode = AccessMode::READ);
    void pin_batch_frames(const PageID* page_ids, PageGuard* out_guards,
                          size_t n, AccessMode mode, bool account_sequential_read, bool transient,
                          bool* out_cache_hit);
    void unpin_frame(PageFrame* frame, bool dirty = false, AccessMode mode = AccessMode::READ);
    void discard_frame(PageFrame* frame);
    void record_buffer_pool_write();
    void mark_frame_dirty(PageFrame& frame);
    // Returns the index of a frame reclaimed for reuse, held with its latch
    // exclusively locked, page id invalid, and absent from any page table.
    // With writeback_out, a dirty victim is not written: its page id goes to *writeback_out, and the caller must write
    // the frame's bytes with write_pages and then finish_writeback it before reusing them. Its mapping stays until
    // then, so a concurrent miss waits instead of reading an older copy. Otherwise *writeback_out is invalid.
    uint32_t evict_one(ShardState& shard, bool* used_eviction_out = nullptr, PageID* writeback_out = nullptr);
    void finish_writeback(PageID victim);

    void read_page(PageID page_id, char* buf);
    // Reads all requests with one libaio batch and returns once all are in.
    void read_batch(FileNo file_no, void* ctx, std::vector<::AlignedRead>& reqs);
    // Writes n pages, at most kPinBatchMax, with one libaio batch, and returns once all are on disk. Every page write
    // of the pool goes through here.
    void write_pages(const PageID* page_ids, const char* const* bufs, size_t n);
    void bg_flush_loop(float high_wm, float low_wm);

    uint32_t _page_size = 0;
    uint32_t _num_frames = 0;

    std::vector<std::unique_ptr<ShardState>> _shards;
    std::shared_ptr<FileManager> _file_manager;
    FileNo _default_file_no = INVALID_FILE;
    uint32_t _num_shards = 1;
    uint32_t _flush_budget_pages_per_cycle = 32;
    uint32_t _flush_wakeup_ms = 100;
    std::atomic<uint32_t> _dirty_frame_count{0};

    std::thread _flush_thread;
    std::atomic<bool> _flush_running{false};
    std::atomic<uint32_t> _flush_shard_next{0};  // flush_dirty_budget's first shard

    InPlaceIOStats* _stats = nullptr;
};

}  // namespace inplace
}  // namespace diskann
