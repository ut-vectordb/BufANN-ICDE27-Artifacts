#include "bufann/lazy_delete.h"

#include "bufann/inplace_backend.h"
#include "bufann/inplace_graph_ops.h"

#include <parallel_hashmap/phmap.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace diskann {
namespace inplace {

namespace {

constexpr uint16_t kSampledNbrs = 20;  // as in consolidate_deletes_freshdiskann
constexpr uint32_t kSweepRun = 64;     // consecutive pages a worker's sweep pins in one batch
constexpr size_t kMaxWorkers = 256;    // workers an episode can run, one queue each
// Threads that resolve the deleted tags at an episode's start, as many as FreshDiskANN's tag resolution uses; the
// samples are snapshotted on the episode's workers, as FreshDiskANN snapshots them on its merge threads.
constexpr uint32_t kResolveThreads = 8;

struct Sample {
    uint16_t count = 0;
    uint32_t ids[kSampledNbrs];
};

// Pages searches handed to one worker.
struct HandedQueue {
    std::mutex mu;
    std::vector<uint32_t> pages;
};

}  // namespace

struct LazyShared {
    std::mutex episode_mu;  // one episode at a time; a trigger during one waits for it, then runs its own
    std::atomic<bool> active{false};
    std::atomic<int> handing{0};  // searches handing pages over right now; an episode ends only once none is

    // Set before an episode becomes active and read-only while it is.
    uint32_t pages = 0;    // graph pages that existed when the episode started
    uint32_t workers = 0;  // repair workers, each owning queues[w]
    std::unique_ptr<std::atomic<uint64_t>[]> done;  // one bit per page: no edge on it points to a resolved node
    phmap::flat_hash_map<uint32_t, Sample> samples;  // resolved node -> its sampled live neighbors
    // One byte per node, 1 for a node resolved at the start, as FreshDiskANN's merge keeps: the repair checks every
    // neighbor of every node against it, so it reads a plain array rather than the store's node states. Nodes added
    // after the start are past its end and never resolved.
    std::vector<uint8_t> is_resolved;

    std::atomic<uint32_t> done_count{0};
    std::atomic<uint64_t> sweep_cursor{0};  // the sweep's next page, modulo pages, so it goes round until all are done
    std::mutex done_mu;
    std::condition_variable all_done_cv;  // the triggering thread waits on this for the last page

    // Never reallocated, so a search can hand pages to a queue while an episode starts or ends.
    std::array<HandedQueue, kMaxWorkers> queues;
    std::atomic<uint32_t> next_search_thread{0};  // numbers search threads, to give each one worker

    bool page_done(uint32_t page) const {
        if (page >= pages) return true;  // added after the start, so it only holds newer nodes
        return (done[page / 64].load(std::memory_order_acquire) >> (page % 64)) & 1;
    }

    // Sets the page's done bit and returns whether this call set it. The caller counts the pages it finished and adds
    // them to done_count with count_done once per batch, as FreshDiskANN's scan adds up its repairs per batch.
    bool mark_done(uint32_t page) {
        const uint64_t bit = uint64_t{1} << (page % 64);
        return !(done[page / 64].fetch_or(bit, std::memory_order_acq_rel) & bit);
    }

    void count_done(uint32_t n) {
        if (n != 0 && done_count.fetch_add(n, std::memory_order_acq_rel) + n == pages) {
            std::lock_guard<std::mutex> lk(done_mu);
            all_done_cv.notify_all();
        }
    }

    bool resolved(uint32_t id) const { return id < is_resolved.size() && is_resolved[id] != 0; }

    bool all_done() const { return done_count.load(std::memory_order_acquire) >= pages; }
};

LazyDeletes::LazyDeletes(unsigned R_, unsigned C_, float alpha_)
    : R(R_), C(C_), alpha(alpha_), shared(std::make_shared<LazyShared>()) {}

LazyDeletes::~LazyDeletes() = default;

void lazy_hand_pages(InPlaceGraphStore* store, const PageGuard* pages, size_t n_pages) {
    LazyDeletes* lazy = store->lazy_deletes();
    if (lazy == nullptr) return;
    LazyShared& sh = *lazy->shared;
    if (!sh.active.load(std::memory_order_acquire)) return;
    sh.handing.fetch_add(1, std::memory_order_seq_cst);
    if (!sh.active.load(std::memory_order_seq_cst)) {  // the episode ended in between
        sh.handing.fetch_sub(1, std::memory_order_seq_cst);
        return;
    }
    thread_local const uint32_t search_thread = sh.next_search_thread.fetch_add(1, std::memory_order_relaxed);
    uint32_t handed[BufferPool::kPinBatchMax];
    size_t n = 0;
    for (size_t i = 0; i < n_pages; ++i) {
        if (!pages[i].valid()) continue;
        const uint32_t page = store->graph_page_id(pages[i]);
        if (!sh.page_done(page)) handed[n++] = page;
    }
    if (n != 0) {
        HandedQueue& q = sh.queues[search_thread % sh.workers];
        {
            std::lock_guard<std::mutex> lk(q.mu);
            q.pages.insert(q.pages.end(), handed, handed + n);
        }
        lazy->stats.handed.fetch_add(n, std::memory_order_relaxed);
    }
    sh.handing.fetch_sub(1, std::memory_order_seq_cst);
}

namespace {

// One repair worker of an episode: its queue first, the sweep whenever the queue is empty.
template <typename T>
class LazyWorker {
 public:
    LazyWorker(InPlaceGraphStore* store, FixedChunkPQTable<T>* pq_table, uint32_t index)
        : _store(store), _lazy(store->lazy_deletes()), _sh(*_lazy->shared), _pq_table(pq_table),
          _queue(_sh.queues[index]) {}

    void run() {
        std::vector<uint32_t> handed;
        while (!_sh.all_done()) {
            {
                std::lock_guard<std::mutex> lk(_queue.mu);
                handed.swap(_queue.pages);
            }
            if (handed.empty()) {
                sweep_step();
                continue;
            }
            std::vector<uint32_t> busy;
            uint32_t finished = 0;
            for (const uint32_t page : handed) {
                if (_sh.page_done(page)) continue;
                PageGuard frame = _store->try_pin_page_resident(page, AccessMode::WRITE);
                if (frame.valid()) {
                    if (finish(page, frame)) ++finished;
                } else if (_store->page_resident(page)) {
                    busy.push_back(page);  // a search holds it; try again on the next round
                } else {
                    ++_evicted_handed;  // left for the sweep
                }
            }
            _pages_handed += finished;
            _sh.count_done(finished);
            const bool all_busy = busy.size() == handed.size();
            handed.clear();
            if (!busy.empty()) {
                std::lock_guard<std::mutex> lk(_queue.mu);
                _queue.pages.insert(_queue.pages.end(), busy.begin(), busy.end());
            }
            if (all_busy) std::this_thread::yield();  // nothing but pages searches hold: let them go first
        }
        // The counters go to the shared stats once, when the worker ends, rather than once per page or repair.
        _lazy->stats.pages_handed.fetch_add(_pages_handed, std::memory_order_relaxed);
        _lazy->stats.pages_swept.fetch_add(_pages_swept, std::memory_order_relaxed);
        _lazy->stats.evicted_handed.fetch_add(_evicted_handed, std::memory_order_relaxed);
        _lazy->stats.repaired.fetch_add(_repaired, std::memory_order_relaxed);
    }

 private:
    // The next run of pages in order, pinned with one blocking batch. The pins are transient, so the sweep's mostly
    // cold pages leave the pool instead of crowding out the searches' pages, and the worker writes back what it
    // repaired itself. Once past the last page the sweep goes round again for the pages still open. Like
    // FreshDiskANN's scan, it reads every page, empty ones included.
    void sweep_step() {
        uint32_t pages[kSweepRun];
        PageGuard frames[kSweepRun];
        size_t n = 0;
        const uint64_t start = _sh.sweep_cursor.fetch_add(kSweepRun, std::memory_order_relaxed);
        for (uint32_t i = 0; i < kSweepRun; ++i) {
            const uint32_t page = static_cast<uint32_t>((start + i) % _sh.pages);
            if (_sh.page_done(page)) continue;
            if (std::find(pages, pages + n, page) == pages + n) pages[n++] = page;  // a wrapped run may repeat
        }
        if (n == 0) return;
        _store->pin_pages_batch_transient(pages, n, AccessMode::WRITE, frames);
        uint32_t finished = 0;
        for (size_t i = 0; i < n; ++i) {
            if (frames[i].valid() && finish(pages[i], frames[i])) ++finished;
            frames[i].release();
        }
        _pages_swept += finished;
        _sh.count_done(finished);
    }

    // Repairs every node on a page held under its WRITE latch that points to a resolved node, then marks the page
    // done. Returns whether this call finished it.
    bool finish(uint32_t page, PageGuard& frame) {
        if (_sh.page_done(page)) return false;
        auto guard = _store->write_guard(frame);
        const uint32_t spp = _store->slots_per_page();
        for (uint32_t slot = 0; slot < spp; ++slot) {
            const uint32_t u = page * spp + slot;
            MutableNodeRef node = guard.write_slot(static_cast<uint16_t>(slot));
            if (!node.valid() || node.node_id() != u) continue;
            if (!_store->is_active(u)) continue;  // free, or deleted: deleted nodes are never repaired
            // The neighbor list and degree are read once per node, as FreshDiskANN's has_deleted_nbr takes them: both
            // are calls into another file, which in the loop would run once per neighbor.
            const uint32_t* nbrs = node.neighbors();
            const uint16_t degree = node.degree();
            bool points_to_deleted = false;
            for (uint16_t j = 0; j < degree && !points_to_deleted; ++j) points_to_deleted = _sh.resolved(nbrs[j]);
            if (points_to_deleted) repair(u, node);
        }
        return _sh.mark_done(page);
    }

    // Replaces each deleted neighbor with its snapshotted sample and prunes to R.
    void repair(uint32_t u, MutableNodeRef& node) {
        _ids.clear();
        const uint32_t* nbrs = node.neighbors();
        const uint16_t degree = node.degree();
        for (uint16_t j = 0; j < degree; ++j) {
            const uint32_t v = nbrs[j];
            if (!_sh.resolved(v)) {
                _ids.push_back(v);
                continue;
            }
            auto it = _sh.samples.find(v);
            if (it == _sh.samples.end()) continue;
            for (uint16_t k = 0; k < it->second.count; ++k) {
                const uint32_t w = it->second.ids[k];
                if (w != u && _store->is_active(w)) _ids.push_back(w);
            }
        }
        std::sort(_ids.begin(), _ids.end());
        _ids.erase(std::unique(_ids.begin(), _ids.end()), _ids.end());
        if (_ids.size() <= _lazy->R) {
            node.set_neighbors(_ids.data(), _ids.size());
        } else {
            prune(u, node);
        }
        ++_repaired;
    }

    void prune(uint32_t u, MutableNodeRef& node) {
        if (!_scratch) {
            _scratch = std::make_unique<InPlaceSearchScratch>();
            _scratch->init(0, _store->n_chunks(), sizeof(T), 1, _store->max_degree(), 1);
        }
        _ids.erase(std::remove_if(_ids.begin(), _ids.end(),
                                  [&](uint32_t id) { return _store->pq_codes_for_node(id) == nullptr; }),
                   _ids.end());
        _pool.clear();
        for (size_t begin = 0; begin < _ids.size();) {
            const uint32_t count = static_cast<uint32_t>(
                std::min<size_t>(_ids.size() - begin, InPlaceSearchScratch::MAX_SCRATCH_NODES));
            _store->compute_pq_dists_src(u, _ids.data() + begin, _scratch->dist_scratch, count,
                                         _scratch->pq_coord_scratch, *_pq_table);
            for (uint32_t j = 0; j < count; ++j) _pool.emplace_back(_ids[begin + j], _scratch->dist_scratch[j], true);
            begin += count;
        }
        graph_prune_neighbors_pq<T>(u, _pool, _lazy->R, _lazy->C, _lazy->alpha, _pruned, _store, _scratch.get(), 0,
                                    nullptr, DistanceScope::MAINTENANCE, _pq_table);
        node.set_neighbors(_pruned.data(), _pruned.size());
    }

    InPlaceGraphStore* _store;
    LazyDeletes* _lazy;
    LazyShared& _sh;
    FixedChunkPQTable<T>* _pq_table;
    HandedQueue& _queue;
    std::unique_ptr<InPlaceSearchScratch> _scratch;
    std::vector<uint32_t> _ids;
    std::vector<unsigned> _pruned;
    std::vector<Neighbor> _pool;
    uint64_t _pages_handed = 0, _pages_swept = 0, _evicted_handed = 0, _repaired = 0;  // flushed when run() ends
};

// Reads every resolved node's page once and keeps up to kSampledNbrs of its live neighbors.
// A resolved node's own list does not change during the episode: repairs skip deleted nodes and nothing links to them.
void snapshot_samples(InPlaceGraphStore* store, const std::vector<uint32_t>& ids, uint32_t num_threads,
                      phmap::flat_hash_map<uint32_t, Sample>& samples) {
    constexpr size_t kBatch = BufferPool::kPinBatchMax;
    const size_t n_batches = (ids.size() + kBatch - 1) / kBatch;
    std::vector<std::vector<std::pair<uint32_t, Sample>>> parts(n_batches);
    std::exception_ptr failure;
#pragma omp parallel for num_threads(static_cast<int>(std::max<uint32_t>(1, num_threads))) schedule(dynamic, 1)
    for (int64_t b = 0; b < static_cast<int64_t>(n_batches); ++b) {
        try {
            const size_t begin = static_cast<size_t>(b) * kBatch;
            const size_t n = std::min(kBatch, ids.size() - begin);
            PageGuard guards[kBatch];
            uint16_t node_to_guard[kBatch];
            store->pin_pages_for_nodes_batch(ids.data() + begin, n, READ, guards, node_to_guard);
            for (size_t i = 0; i < n; ++i) {
                if (node_to_guard[i] == InPlaceGraphStore::kNoPage) continue;
                ConstNodeRef node = store->read_guard(guards[node_to_guard[i]]).read_node(ids[begin + i]);
                if (!node.valid()) continue;
                Sample s;
                for (uint16_t j = 0; j < node.degree() && s.count < kSampledNbrs; ++j) {
                    if (!store->is_delete_pending(node.neighbors()[j])) s.ids[s.count++] = node.neighbors()[j];
                }
                parts[b].emplace_back(ids[begin + i], s);
            }
        } catch (...) {
#pragma omp critical(bufann_lazy_snapshot_failure)
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
    samples.clear();
    samples.reserve(ids.size());
    for (auto& part : parts) samples.insert(part.begin(), part.end());
}

}  // namespace

template <typename T>
size_t lazy_run_episode(InPlaceGraphStore* store, FixedChunkPQTable<T>* pq_table, uint32_t num_threads) {
    LazyDeletes* lazy = store->lazy_deletes();
    LazyShared& sh = *lazy->shared;
    std::lock_guard<std::mutex> episode_lk(sh.episode_mu);
    if (num_threads == 0 || num_threads > kMaxWorkers) {
        throw std::invalid_argument("lazy deletion needs between 1 and 256 repair workers");
    }

    // Resolve the deleted tags to nodes, which become DeletePending, and snapshot their neighbors.
    std::vector<uint32_t> ids;
    {
        auto meta_guard = store->enter_meta_write();
        ids = store->resolve_deleted_tags_lazy(kResolveThreads);
    }
    lazy->stats.resolved.fetch_add(ids.size(), std::memory_order_relaxed);
    if (ids.empty()) return 0;
    std::sort(ids.begin(), ids.end());
    const auto snapshot_begin = std::chrono::steady_clock::now();
    snapshot_samples(store, ids, num_threads, sh.samples);
    std::cout << "[BufANN] lazy deletion: snapshotted samples of " << ids.size() << " nodes in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - snapshot_begin).count() << " s with "
              << num_threads << " threads" << std::endl;

    sh.is_resolved.assign(store->meta_capacity(), 0);
    for (const uint32_t id : ids) {
        if (id < sh.is_resolved.size()) sh.is_resolved[id] = 1;
    }
    sh.pages = store->total_pages();
    sh.workers = num_threads;
    sh.done.reset(new std::atomic<uint64_t>[(sh.pages + 63) / 64]());
    sh.done_count.store(0, std::memory_order_relaxed);
    sh.sweep_cursor.store(0, std::memory_order_relaxed);
    for (uint32_t w = 0; w < sh.workers; ++w) {
        std::lock_guard<std::mutex> lk(sh.queues[w].mu);
        sh.queues[w].pages.clear();
    }
    sh.active.store(true, std::memory_order_release);

    std::vector<std::thread> workers;
    for (uint32_t w = 0; w < sh.workers; ++w) {
        workers.emplace_back([store, pq_table, w] { LazyWorker<T>(store, pq_table, w).run(); });
    }
    {
        std::unique_lock<std::mutex> lk(sh.done_mu);
        sh.all_done_cv.wait(lk, [&] { return sh.all_done(); });
    }
    for (auto& w : workers) w.join();
    // No search may still be handing pages over, reading the done bits, once the episode is over.
    sh.active.store(false, std::memory_order_seq_cst);
    while (sh.handing.load(std::memory_order_seq_cst) != 0) std::this_thread::yield();

    size_t freed = 0;
    {
        auto meta_guard = store->enter_meta_write();
        freed = store->free_lazy_deleted(ids, num_threads);
    }
    sh.samples.clear();
    std::vector<uint8_t>().swap(sh.is_resolved);
    lazy->stats.freed.fetch_add(freed, std::memory_order_relaxed);
    lazy->stats.episodes.fetch_add(1, std::memory_order_relaxed);
    return freed;
}

template size_t lazy_run_episode<float>(InPlaceGraphStore*, FixedChunkPQTable<float>*, uint32_t);
template size_t lazy_run_episode<uint8_t>(InPlaceGraphStore*, FixedChunkPQTable<uint8_t>*, uint32_t);
template size_t lazy_run_episode<int8_t>(InPlaceGraphStore*, FixedChunkPQTable<int8_t>*, uint32_t);

}  // namespace inplace
}  // namespace diskann
