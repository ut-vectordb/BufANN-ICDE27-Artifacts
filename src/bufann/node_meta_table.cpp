#include "bufann/node_meta_table.h"
#include "ann_exception.h"
#include "threadpool.h"
#include "tcmalloc/malloc_extension.h"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>

namespace diskann {
namespace inplace {

void release_allocator_free_memory() noexcept {
    if (MallocExtension *ext = MallocExtension::instance()) {
        ext->ReleaseFreeMemory();
    }
}

namespace {

inline uint64_t elapsed_ns(const std::chrono::steady_clock::time_point& start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

inline uint64_t steady_us(const std::chrono::steady_clock::time_point& t) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count());
}

}  // namespace

NodeMetaTable::Arrays::Arrays(size_t count, uint32_t pq_chunks)
    : capacity(count), n_chunks(pq_chunks), tags(new TagType[count]), states(new std::atomic<NodeState>[count]()) {
    if (pq_chunks != 0) {
        if (count > std::numeric_limits<size_t>::max() / pq_chunks) {
            throw ANNException("metadata PQ capacity overflow", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        pq_codes.reset(new uint8_t[count * static_cast<size_t>(pq_chunks)]);
    }
}

// ===========================================================================
// Lifecycle
// ===========================================================================
NodeMetaTable::NodeMetaTable(InPlaceIOStats& stats) : _stats(stats) {
    for (auto& readers : _readers) readers.store(0, std::memory_order_relaxed);
    for (auto& writers : _writers) writers.store(0, std::memory_order_relaxed);
    for (auto& pending : _retired_pending) pending.store(false, std::memory_order_relaxed);
}

NodeMetaTable::~NodeMetaTable() {
    shutdown();
}

void NodeMetaTable::init(size_t capacity, uint32_t pq_chunks, uint32_t copy_threads) {
    _copy_pool.reset();
    _copy_threads = copy_threads;
    _owned = std::make_unique<Arrays>(capacity, pq_chunks);
    _current.store(_owned.get(), std::memory_order_release);
    // Drop any prior retired queue without leaving the reclaim worker's pending/count gauges stranded (init is rare; keep
    // gauges consistent).
    _retired.clear();
    _retired_count.store(0, std::memory_order_relaxed);
    for (auto& pending : _retired_pending) pending.store(false, std::memory_order_relaxed);
    _reader_epoch.store(0, std::memory_order_relaxed);
}

void NodeMetaTable::shutdown() noexcept {
    stop_reclaim_worker();
    // Tear the generations down here rather than in member destructors so ReleaseFreeMemory observes the freed spans.
    // Skip the allocator scan when there was no metadata.
    const bool had_meta = _owned != nullptr || !_retired.empty();
    _current.store(nullptr, std::memory_order_release);
    _owned.reset();
    _retired.clear();
    _retired_count.store(0, std::memory_order_relaxed);
    if (had_meta) {
        release_allocator_free_memory();
    }
}

size_t NodeMetaTable::capacity() const {
    Arrays* meta = current();
    return meta == nullptr ? 0 : meta->capacity;
}

// ===========================================================================
// Epoch tickets
// ===========================================================================
NodeMetaTable::Guard::Guard(Guard&& other) noexcept
    : _table(other._table), _epoch(other._epoch), _writer(other._writer) {
    other._table = nullptr;
}

NodeMetaTable::Guard& NodeMetaTable::Guard::operator=(Guard&& other) noexcept {
    if (this == &other) return *this;
    if (_table != nullptr) {
        _table->release_epoch(_epoch, _writer);
    }
    _table = other._table;
    _epoch = other._epoch;
    _writer = other._writer;
    other._table = nullptr;
    return *this;
}

NodeMetaTable::Guard::~Guard() {
    if (_table != nullptr) {
        _table->release_epoch(_epoch, _writer);
    }
}

NodeMetaTable::Guard NodeMetaTable::enter_read() {
    return enter_epoch(false);
}

NodeMetaTable::Guard NodeMetaTable::enter_write() {
    return enter_epoch(true);
}

void NodeMetaTable::release_epoch(uint8_t epoch, bool writer) noexcept {
    if (writer) {
        const bool last_writer = _writers[epoch].fetch_sub(1, std::memory_order_release) == 1;
        if (last_writer && _growth_active.load(std::memory_order_acquire)) {
            // The grower holds _growth_mtx across the copy. Never acquire it from an operation completing its epoch ticket.
            _growth_cv.notify_all();
        }
    }
    const bool last_reader = _readers[epoch].fetch_sub(1, std::memory_order_acq_rel) == 1;
    if (last_reader && _retired_pending[epoch].load(std::memory_order_acquire)) {
        request_reclaim(epoch);
    }
}

NodeMetaTable::Guard NodeMetaTable::enter_epoch(bool writer) {
    for (;;) {
        // The counter publication and the grower's admission close form a two-way handshake: either growth observes
        // this ticket, or this operation observes growth and retries. Keep those operations in one SC order;
        // acquire-only RMWs permit both sides to miss each other on weakly ordered architectures.
        if (_growth_active.load(std::memory_order_seq_cst)) {
            const auto wait_begin = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(_growth_mtx);
            // wait_for: writer notify cannot take this mutex (held across growth copy), so a pure wait() can lose the wakeup.
            while (_growth_active.load(std::memory_order_acquire)) {
                _growth_cv.wait_for(lock, std::chrono::milliseconds(1));
            }
            if (writer) {
                _stats.meta_growth_update_stall_ns.fetch_add(elapsed_ns(wait_begin), std::memory_order_relaxed);
            }
            continue;
        }
        const uint8_t epoch = _reader_epoch.load(std::memory_order_seq_cst);
        _readers[epoch].fetch_add(1, std::memory_order_seq_cst);
        if (writer) {
            _writers[epoch].fetch_add(1, std::memory_order_seq_cst);
        }
        if (!_growth_active.load(std::memory_order_seq_cst) && epoch == _reader_epoch.load(std::memory_order_seq_cst)) {
            return Guard(this, epoch, writer);
        }
        release_epoch(epoch, writer);
    }
}

// ===========================================================================
// Growth
// ===========================================================================
void NodeMetaTable::begin_growth_locked(std::unique_lock<std::mutex>& lock) {
    // Block new epoch admissions, then drain in-flight writers only. Live query readers may keep the retired source
    // table until reclaim.
    _growth_active.store(true, std::memory_order_seq_cst);
    // Last-writer notify_all cannot take this mutex (we hold it across the subsequent copy). Use wait_for so a lost
    // wakeup rechecks writers==0.
    while (_writers[0].load(std::memory_order_seq_cst) != 0 || _writers[1].load(std::memory_order_seq_cst) != 0) {
        _growth_cv.wait_for(lock, std::chrono::milliseconds(1));
    }
}

void NodeMetaTable::end_growth_locked() {
    _growth_active.store(false, std::memory_order_release);
    _growth_cv.notify_all();
}

NodeMetaTable::GrowthGuard::GrowthGuard(NodeMetaTable& table, std::unique_lock<std::mutex>& lock) : _table(table) {
    _table.begin_growth_locked(lock);
}

NodeMetaTable::GrowthGuard::~GrowthGuard() {
    _table.end_growth_locked();
}

void NodeMetaTable::reserve(size_t needed, uint32_t pq_chunks) {
    if (needed > static_cast<size_t>(INVALID_NODE)) {
        throw ANNException("reserve_meta_capacity: 32-bit internal node ID space exhausted", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    Arrays* current_view = current();
    size_t current_capacity = current_view == nullptr ? 0 : current_view->capacity;
    if (needed <= current_capacity) return;
    std::unique_lock<std::mutex> lock(_growth_mtx);
    current_view = current();
    current_capacity = current_view == nullptr ? 0 : current_view->capacity;
    if (needed <= current_capacity) return;
    const auto growth_start = std::chrono::steady_clock::now();
    _stats.meta_growth_last_started_us.store(steady_us(growth_start), std::memory_order_relaxed);
    GrowthGuard growth(*this, lock);
    const uint64_t drain_ns = elapsed_ns(growth_start);
    current_view = current();
    current_capacity = current_view == nullptr ? 0 : current_view->capacity;
    if (needed <= current_capacity) return;
    start_reclaim_worker_locked();
    if (current_view != nullptr) {
        // Retirement after publication must not allocate: otherwise a bad_alloc could destroy `old` while readers
        // still hold its epoch.
        _retired.reserve(_retired.size() + 1);
    }
    // Grow by 1.2x (current * 6 / 5). Still at least `needed`.
    size_t grown = current_capacity == 0 ? kDefaultCapacity : current_capacity + current_capacity / 5;
    grown = std::max(grown, needed);
    grown = std::min(grown, static_cast<size_t>(INVALID_NODE));
    const auto alloc_start = std::chrono::steady_clock::now();
    auto next = std::make_unique<Arrays>(grown, pq_chunks);
    const uint64_t alloc_ns = elapsed_ns(alloc_start);
    const auto copy_start = std::chrono::steady_clock::now();
    if (current_view != nullptr) {
        copy_prefix_parallel(*current_view, *next, current_capacity);
    }
    const uint64_t copy_ns = elapsed_ns(copy_start);
    publish_locked(std::move(next));
    const auto growth_end = std::chrono::steady_clock::now();
    _stats.meta_growth_count.fetch_add(1, std::memory_order_relaxed);
    _stats.meta_growth_drain_ns.fetch_add(drain_ns, std::memory_order_relaxed);
    _stats.meta_growth_alloc_ns.fetch_add(alloc_ns, std::memory_order_relaxed);
    _stats.meta_growth_copy_ns.fetch_add(copy_ns, std::memory_order_relaxed);
    _stats.meta_growth_total_ns.fetch_add(elapsed_ns(growth_start), std::memory_order_relaxed);
    _stats.meta_growth_last_from.store(current_capacity, std::memory_order_relaxed);
    _stats.meta_growth_last_to.store(grown, std::memory_order_relaxed);
    _stats.meta_growth_last_finished_us.store(steady_us(growth_end), std::memory_order_relaxed);
}

void NodeMetaTable::ensure_pq_chunks(uint32_t pq_chunks) {
    Arrays* view = current();
    if (view == nullptr) {
        throw ANNException("ensure_pq_storage: metadata arrays are not initialized", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (view->n_chunks == pq_chunks && view->pq_codes != nullptr) return;
    std::unique_lock<std::mutex> lock(_growth_mtx);
    view = current();
    if (view->n_chunks == pq_chunks && view->pq_codes != nullptr) return;
    GrowthGuard growth(*this, lock);
    view = current();
    if (view->n_chunks == pq_chunks && view->pq_codes != nullptr) return;
    start_reclaim_worker_locked();
    _retired.reserve(_retired.size() + 1);
    auto next = std::make_unique<Arrays>(view->capacity, pq_chunks);
    copy_prefix_parallel(*view, *next, view->capacity);
    publish_locked(std::move(next));
}

void NodeMetaTable::publish_locked(std::unique_ptr<Arrays> next) {
    const uint8_t old_epoch = _reader_epoch.load(std::memory_order_relaxed);
    std::unique_ptr<Arrays> old = std::move(_owned);
    _owned = std::move(next);
    _current.store(_owned.get(), std::memory_order_release);
    _reader_epoch.store(old_epoch ^ 1u, std::memory_order_release);
    if (old == nullptr) return;
    _retired.push_back(Retired{old_epoch, std::move(old)});
    _retired_count.fetch_add(1, std::memory_order_release);
    _retired_pending[old_epoch].store(true, std::memory_order_release);
    _stats.meta_retired_generations.fetch_add(1, std::memory_order_relaxed);
    notify_retirement();
}

void NodeMetaTable::copy_prefix_parallel(const Arrays& source, Arrays& destination, size_t count) {
    if (count == 0) return;
    if (count > source.capacity || count > destination.capacity) {
        throw ANNException("metadata copy exceeds array capacity", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    constexpr size_t kTargetRangeBytes = 4u * 1024u * 1024u;
    const bool copy_pq = source.pq_codes != nullptr && destination.pq_codes != nullptr && source.n_chunks == destination.n_chunks;
    const size_t pq_bytes_per_slot = copy_pq ? source.n_chunks : 0;
    const size_t bytes_per_slot = sizeof(TagType) + sizeof(std::atomic<NodeState>) + pq_bytes_per_slot;
    const size_t slots_per_range = std::max<size_t>(1, kTargetRangeBytes / bytes_per_slot);
    const size_t range_count = 1 + (count - 1) / slots_per_range;
    const uint32_t workers = static_cast<uint32_t>(
        std::min<size_t>(std::min<size_t>(_copy_threads, range_count), std::numeric_limits<uint32_t>::max()));
    _stats.meta_growth_last_workers.store(workers, std::memory_order_relaxed);
    _stats.meta_growth_last_copy_bytes.store(count * bytes_per_slot, std::memory_order_relaxed);

    auto copy_range = [&](size_t range_index) {
        const size_t begin = range_index * slots_per_range;
        const size_t end = std::min(count, begin + slots_per_range);
        for (size_t i = begin; i < end; ++i) {
            destination.states[i].store(source.states[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        memcpy(destination.tags.get() + begin, source.tags.get() + begin, (end - begin) * sizeof(TagType));
        if (copy_pq) {
            const size_t pq_begin = begin * pq_bytes_per_slot;
            memcpy(destination.pq_codes.get() + pq_begin, source.pq_codes.get() + pq_begin,
                   (end - begin) * pq_bytes_per_slot);
        }
    };

    if (workers <= 1) {
        for (size_t range = 0; range < range_count; ++range) {
            copy_range(range);
        }
        return;
    }

    // OpenMP outside an existing team; fixed helper pool when already nested (capacity miss on an insert worker).
    // Pool sized once to the configured max; each grow submits only `workers` strided tasks.
    if (!omp_in_parallel()) {
#pragma omp parallel for num_threads(workers) schedule(static)
        for (int64_t range = 0; range < static_cast<int64_t>(range_count); ++range) {
            copy_range(static_cast<size_t>(range));
        }
        return;
    }

    if (_copy_pool == nullptr) {
        _copy_pool = std::make_unique<::ThreadPool>(_copy_threads);
    }
    std::vector<std::future<void>> futures;
    futures.reserve(workers);
    try {
        for (uint32_t worker = 0; worker < workers; ++worker) {
            futures.emplace_back(_copy_pool->enqueue([&, worker]() {
                for (size_t range = worker; range < range_count; range += workers) {
                    copy_range(range);
                }
            }));
        }
    } catch (...) {
        for (auto& future : futures) future.wait();
        throw;
    }
    for (auto& future : futures) future.get();
}

// ===========================================================================
// Reclaim worker
// ===========================================================================
void NodeMetaTable::start_reclaim_worker_locked() {
    if (_reclaim_started) return;
    _reclaim_stop.store(false, std::memory_order_relaxed);
    _reclaim_thread = std::thread([this]() { reclaim_loop(); });
    _reclaim_started = true;
}

void NodeMetaTable::notify_retirement() {
    // Unlike last-reader release, growth is a rare slow path and can use the wait mutex. This makes the idle (no
    // retired arrays) notification lossless, allowing the worker to sleep without periodic polling.
    {
        std::lock_guard<std::mutex> wait_lock(_reclaim_wait_mtx);
        _reclaim_requested.store(true, std::memory_order_release);
    }
    _reclaim_cv.notify_one();
}

void NodeMetaTable::request_reclaim(uint8_t epoch) noexcept {
    if (!_retired_pending[epoch].load(std::memory_order_acquire) || _readers[epoch].load(std::memory_order_acquire) != 0) {
        return;
    }
    _reclaim_requested.store(true, std::memory_order_release);
    _reclaim_cv.notify_one();
}

std::unique_ptr<NodeMetaTable::Arrays> NodeMetaTable::take_reclaimable_locked() {
    std::unique_ptr<Arrays> reclaimed;
    for (auto it = _retired.begin(); it != _retired.end();) {
        if (_readers[it->reader_epoch].load(std::memory_order_acquire) != 0) {
            ++it;
            continue;
        }
        reclaimed = std::move(it->arrays);
        _retired.erase(it);
        _retired_count.fetch_sub(1, std::memory_order_release);
        break;
    }

    bool pending[2] = {false, false};
    for (const auto& retired : _retired) {
        pending[retired.reader_epoch] = true;
    }
    for (uint8_t epoch = 0; epoch < 2; ++epoch) {
        _retired_pending[epoch].store(pending[epoch], std::memory_order_release);
    }
    return reclaimed;
}

void NodeMetaTable::reclaim_loop() {
    for (;;) {
        {
            std::unique_lock<std::mutex> wait_lock(_reclaim_wait_mtx);
            auto wake_requested = [this]() {
                return _reclaim_stop.load(std::memory_order_acquire) || _reclaim_requested.load(std::memory_order_acquire);
            };
            if (_retired_count.load(std::memory_order_acquire) == 0) {
                // New retirements notify while holding wait_lock, so the idle worker needs no polling fallback.
                _reclaim_cv.wait(wait_lock, wake_requested);
            } else {
                // Last-reader notifications stay lock-free. A bounded wait closes their condition-variable check/sleep race.
                _reclaim_cv.wait_for(wait_lock, std::chrono::seconds(1), wake_requested);
            }
        }

        _reclaim_requested.store(false, std::memory_order_release);
        bool reclaimed_any = false;
        for (;;) {
            std::unique_ptr<Arrays> reclaimed;
            {
                std::lock_guard<std::mutex> growth_lock(_growth_mtx);
                reclaimed = take_reclaimable_locked();
            }
            if (reclaimed == nullptr) break;
            // Potentially tens of GiB are released here, outside both metadata locks and never on the query/update
            // thread that released the last old-generation reader.
            reclaimed.reset();
            reclaimed_any = true;
            _stats.meta_reclaimed_generations.fetch_add(1, std::memory_order_relaxed);
        }
        if (reclaimed_any) {
            release_allocator_free_memory();
        }

        if (_reclaim_stop.load(std::memory_order_acquire)) return;
    }
}

void NodeMetaTable::stop_reclaim_worker() noexcept {
    if (!_reclaim_started) return;
    {
        std::lock_guard<std::mutex> wait_lock(_reclaim_wait_mtx);
        _reclaim_stop.store(true, std::memory_order_release);
    }
    _reclaim_cv.notify_one();
    if (_reclaim_thread.joinable()) _reclaim_thread.join();
    _reclaim_started = false;
}

}  // namespace inplace
}  // namespace diskann
