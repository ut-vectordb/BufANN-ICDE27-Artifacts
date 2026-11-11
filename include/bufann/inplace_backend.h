#pragma once

#include <atomic>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <limits>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>

#include <parallel_hashmap/phmap.h>

#include "bufann/buffer_pool/buffer_pool.h"
#include "bufann/inplace_defs.h"
#include "bufann/inplace_io_stats.h"
#include "bufann/lazy_delete.h"
#include "bufann/node_meta_table.h"

#include "distance.h"
#include "neighbor.h"
#include "pq_table.h"
#include "utils.h"
#include "tsl/robin_set.h"

class AlignedFileReader;
class ThreadPool;

namespace diskann {
  template<typename T, typename TagT>
  class Index;
}

namespace diskann {
namespace inplace {

enum class DistanceScope : uint8_t { SEARCH = 0, INSERT = 1, MAINTENANCE = 2 };
enum class TraversalScope : uint8_t { SEARCH = 0, INSERT = 1 };

// ---------------------------------------------------------------------------
// PackedSlotHeader -- 8 bytes, on-disk
// ---------------------------------------------------------------------------
struct PackedSlotHeader {
    uint16_t degree;
    uint8_t  flags;
    uint8_t  _pad;
    uint32_t node_id;
};
static_assert(sizeof(PackedSlotHeader) == 8, "slot header must be 8 bytes");

// ---------------------------------------------------------------------------
// Frame Guards  (AccessMode lives in inplace_defs.h)
// ---------------------------------------------------------------------------
class InPlaceGraphStore;
class FrameReadGuard;
class FrameWriteGuard;
class ConstNodeRef;
class MutableNodeRef;

class ConstNodeRef {
 public:
    ConstNodeRef() = default;

    bool valid() const { return _slot_ptr != nullptr && _store != nullptr; }
    uint16_t degree() const;
    uint32_t node_id() const;
    const uint32_t* neighbors() const;
    const char* coords_bytes() const;
    template <typename T>
    const T* coords() const;

 protected:
    friend class FrameReadGuard;
    friend class FrameWriteGuard;
    friend class MutableNodeRef;
    friend class InPlaceGraphStore;
    ConstNodeRef(const InPlaceGraphStore* store, const char* slot_ptr);

    const InPlaceGraphStore* _store = nullptr;
    const char* _slot_ptr = nullptr;
};

class MutableNodeRef {
 public:
    MutableNodeRef() = default;

    bool valid() const { return _slot_ptr != nullptr && _store != nullptr; }
    uint16_t degree() const;
    uint32_t node_id() const;
    uint32_t* neighbors();
    const uint32_t* neighbors() const;
    char* coords_bytes();
    const char* coords_bytes() const;
    template <typename T>
    T* coords();
    template <typename T>
    const T* coords() const;
    void set_degree(uint16_t degree);
    void set_neighbors(const uint32_t* neighbors, size_t count);
    template <typename T>
    void set_coords(const T* coords);

 private:
    friend class FrameWriteGuard;
    MutableNodeRef(InPlaceGraphStore* store, char* slot_ptr,
                   FrameWriteGuard* guard);
    void mark_dirty();

    InPlaceGraphStore* _store = nullptr;
    char* _slot_ptr = nullptr;
    FrameWriteGuard* _guard = nullptr;
};

class FrameReadGuard {
 public:
    FrameReadGuard() = default;
    FrameReadGuard(const FrameReadGuard&) = delete;
    FrameReadGuard& operator=(const FrameReadGuard&) = delete;
    FrameReadGuard(FrameReadGuard&&) noexcept = default;
    FrameReadGuard& operator=(FrameReadGuard&&) noexcept = default;
    ~FrameReadGuard() = default;

    ConstNodeRef read_slot(uint16_t slot_idx) const;
    // Returns an invalid ref if node_id does not live on this page.
    ConstNodeRef read_node(uint32_t node_id) const;

 private:
    friend class InPlaceGraphStore;
    FrameReadGuard(InPlaceGraphStore* store, PageGuard* page);
    InPlaceGraphStore* _store = nullptr;
    PageGuard* _page = nullptr;
};

class FrameWriteGuard {
 public:
    FrameWriteGuard() = default;
    FrameWriteGuard(const FrameWriteGuard&) = delete;
    FrameWriteGuard& operator=(const FrameWriteGuard&) = delete;
    FrameWriteGuard(FrameWriteGuard&&) noexcept = default;
    FrameWriteGuard& operator=(FrameWriteGuard&&) noexcept = default;
    ~FrameWriteGuard() = default;

    ConstNodeRef read_slot(uint16_t slot_idx) const;
    ConstNodeRef read_node(uint32_t node_id) const;
    MutableNodeRef write_slot(uint16_t slot_idx);
    MutableNodeRef write_node(uint32_t node_id);

 private:
    friend class InPlaceGraphStore;
    friend class MutableNodeRef;
    FrameWriteGuard(InPlaceGraphStore* store, PageGuard* page);
    void mark_dirty();

    InPlaceGraphStore* _store = nullptr;
    PageGuard* _page = nullptr;
};

// ---------------------------------------------------------------------------
// InPlaceSearchScratch -- per-thread scratch buffers
// ---------------------------------------------------------------------------
struct InPlaceSearchScratch {
    float*   pq_dists         = nullptr;
    float*   dist_scratch     = nullptr;
    uint8_t* pq_coord_scratch = nullptr;
    std::vector<Neighbor> traversal_results;
    std::vector<Neighbor> traversal_retset;
    std::vector<Neighbor> frontier_nodes;
    std::vector<unsigned> frontier_ids;
    tsl::robin_set<unsigned> visited;
    std::vector<unsigned> frontier_neighbor_ids;
    std::vector<uint32_t> frontier_candidate_ids;
    uint64_t local_query_comparable_cpu_ns = 0;
    uint64_t local_query_comparable_cmps = 0;

    static constexpr uint32_t MAX_SCRATCH_NODES = 16384;

    void init(uint32_t aligned_dim, uint32_t n_chunks, uint32_t elem_size,
              uint32_t search_list_size, uint32_t max_degree,
              uint32_t beamwidth);
    void reset() {
        local_query_comparable_cpu_ns = 0;
        local_query_comparable_cmps = 0;
        traversal_results.clear();
        traversal_retset.clear();
        frontier_nodes.clear();
        frontier_ids.clear();
        visited.clear();
        frontier_neighbor_ids.clear();
        frontier_candidate_ids.clear();
    }
    inline void record_comparable_distance(DistanceScope scope, uint64_t ns, uint64_t ops) {
        if (scope != DistanceScope::SEARCH) return;
        local_query_comparable_cpu_ns += ns;
        local_query_comparable_cmps += ops;
    }
    void flush_distance_stats(InPlaceIOStats& stats);
    ~InPlaceSearchScratch();
};

// ---------------------------------------------------------------------------
// InPlaceGraphStore
// ---------------------------------------------------------------------------
class InPlaceGraphStore {
 public:
    struct DeletedNodeBatch {
        std::vector<uint32_t> repair_required;
    };

    // Per-node metadata lives in NodeMetaTable (node_meta_table.h); these
    // aliases and forwards keep the store's existing interface.
    using MetaReadGuard = NodeMetaTable::Guard;
    MetaReadGuard enter_meta_read() { return _meta.enter_read(); }
    MetaReadGuard enter_meta_write() { return _meta.enter_write(); }
    InPlaceGraphStore();
    ~InPlaceGraphStore();

    void init(uint32_t dim, uint32_t Mmax, uint32_t elem_size_bytes,
              uint32_t page_size, uint32_t buffer_pool_frames,
              const std::string& heap_path,
              uint32_t flush_budget_pages_per_cycle = 32,
              uint32_t flush_wakeup_ms = 100,
              bool truncate_heap = true,
              uint32_t max_nodes = 0,
              uint32_t meta_growth_copy_threads = 16,
              uint32_t initial_pq_chunks = 0);

    // PQ support
    void load_pq_codes_from_disk_index(const std::string& pq_prefix, uint32_t num_chunks);
    template<typename T>
    void compute_pq_dists_src(uint32_t src, const unsigned* ids,
                              float* dists_out, uint32_t count,
                              uint8_t* scratch,
                              FixedChunkPQTable<T>& pq_table);
    // Gather PQ codes for `n` node IDs into caller-supplied flat buffer
    // (must have capacity >= n * n_chunks bytes; missing nodes get zeros).
    void gather_pq_codes(const unsigned* ids, uint64_t n, uint8_t* out) const;
    // Raw pointer to the PQ codes of a single node, or nullptr if absent.
    const uint8_t* pq_codes_for_node(uint32_t node_id) const;
    uint32_t n_chunks() const { return _n_chunks; }
    // Whether searches skip PQ and score every neighbor with its full vector, read through the buffer pool.
    bool exact_search() const { return _exact_search; }
    void set_exact_search(bool exact) { _exact_search = exact; }

    // Buffer-pool pinning. A PageGuard holds the frame's latch for its whole
    // lifetime (shared for READ, exclusive for WRITE), which is both residency
    // and read/write serialization. FrameReadGuard/FrameWriteGuard are just
    // typed node-slot views over an already-pinned page; they take no lock.
    static constexpr uint16_t kNoPage = 0xFFFF;
    PageGuard pin_page(uint32_t page_id, AccessMode mode);
    PageGuard pin_page_for_node(uint32_t node_id, AccessMode mode);
    PageGuard try_pin_page_resident(uint32_t page_id, AccessMode mode);
    bool page_resident(uint32_t page_id) const;
    void pin_pages_batch(const uint32_t* page_ids, size_t n, AccessMode mode, PageGuard* out_pages);
    // As pin_pages_batch, but a page that was not cached leaves the pool when its guard is released.
    void pin_pages_batch_transient(const uint32_t* page_ids, size_t n, AccessMode mode, PageGuard* out_pages);

    // Batched analogue of pin_page_for_node, but page-grouped so no frame is
    // ever pinned twice in one call (a single latch cannot be re-taken). The
    // unique pages spanned by node_ids are pinned into out_page_guards[0..P);
    // node i resolves through out_page_guards[out_node_to_guard[i]]; a node that
    // is not present gets kNoPage instead. Returns P, the number of pages pinned.
    // n must be <= BufferPool::kPinBatchMax.
    size_t pin_pages_for_nodes_batch(const uint32_t* node_ids, size_t n,
                                     AccessMode mode,
                                     PageGuard* out_page_guards,
                                     uint16_t* out_node_to_guard);
    FrameReadGuard read_guard(PageGuard& page);
    FrameWriteGuard write_guard(PageGuard& page);
    uint32_t graph_page_id(const PageGuard& page) const { return page.page_id().second; }
    // A node ID is its slot address: node_id = page * slots_per_page + slot.
    uint32_t node_page(uint32_t node_id) const { return node_id / _slots_per_page; }
    uint16_t node_slot(uint32_t node_id) const { return static_cast<uint16_t>(node_id % _slots_per_page); }
    uint32_t slots_per_page() const { return _slots_per_page; }
    char* page_data(PageGuard& page) const { return page.data(); }
    const char* page_data(const PageGuard& page) const { return page.data(); }
    void     batch_fetch_coords(const std::vector<uint32_t>& ids, char* out_coords,
                                std::vector<uint8_t>& found);
    void     preload_pages(uint32_t max_pages, uint32_t num_threads);

    // Allocation
    // Claims a Free slot for a new node and returns its ID. The slot stays
    // Allocating until publish_node. The caller must not hold a metadata ticket,
    // because this may grow the metadata arrays.
    uint32_t reserve_node_id();
    // Writes the new node's page slot. node_id comes from reserve_node_id, or,
    // during bulk load and import, is any Free slot the caller picks.
    void     allocate_node(uint32_t node_id);
    void publish_node(uint32_t node_id);

    // Node state by node id.
    bool is_active(uint32_t node_id) const;          // Active
    bool is_delete_pending(uint32_t node_id) const;  // DeletePending
    bool node_present(uint32_t node_id) const;       // Active or DeletePending: the slot still holds it, so a search may expand it

    bool     set_node_tag(uint32_t node_id, TagType tag);
    TagType  node_tag(uint32_t node_id) const;
    bool mark_tag_deleted(TagType tag);
    bool     is_tag_deleted(TagType tag) const;
    size_t   deleted_tag_count() const;
    std::vector<TagType> deleted_tags_snapshot() const;
    void restore_deleted_tags(
        const std::vector<TagType>& tags);
    DeletedNodeBatch drain_deleted_tags_to_delete_pending();
    uint32_t get_page_id(uint32_t node_id) const;
    void     reserve_meta_capacity(size_t needed);

    // Page-level pin count query for safe reclamation

    // PQ encode for new inserts
    template<typename T>
    void encode_pq(uint32_t node_id, const T* coords, FixedChunkPQTable<T>& pq_table);

    uint32_t dirty_page_count() const;

    // FreshDiskANN-style consolidation: scans every graph page in order and bypasses deleted neighbors.
    // With `snapshot`, the deleted nodes' adjacency is copied into memory first; without it, it is read through the buffer pool.
    // Foreground updates must be paused until this pass and reclamation finish.
    // Nothing in bufann_cleanup_deleted_edges enforces that, because enter_meta_write() does not exclude searches or inserts.
    template<typename T>
    uint64_t consolidate_deletes_freshdiskann(unsigned R, unsigned C, float alpha,
                                             FixedChunkPQTable<T>* pq_table,
                                             uint32_t num_threads, bool snapshot);

    // Lazy deletion (lazy_delete.h).
    void enable_lazy_deletes(unsigned R, unsigned C, float alpha);
    LazyDeletes* lazy_deletes() const { return _lazy_deletes.get(); }
    // Lazy deletion's copy of drain_deleted_tags_to_delete_pending, with the scan over all nodes split across threads.
    // Returns the nodes it moved to DeletePending.
    std::vector<uint32_t> resolve_deleted_tags_lazy(uint32_t num_threads);
    // Frees the slots of DeletePending nodes that no edge points to any more, and returns how many.
    size_t free_lazy_deleted(const std::vector<uint32_t>& ids, uint32_t num_threads);

    // Foreground delete lifecycle transition. Returns true if the node
    // transitioned from Active to DeletePending.
    bool transition_active_to_delete_pending(uint32_t node_id);

    // Bulk load
    template<typename T, typename TagT>
    void bulk_load_from_index(diskann::Index<T, TagT>& mem_index, uint32_t n);

    // Warmup + lifecycle
    void warmup_bfs(uint32_t entry_point, uint32_t num_nodes);
    // Node-based level-synchronous BFS warmup from entry_point. The frontier
    // holds graph node IDs; each visited node's host page is pinned as a
    // side effect (via pin_page_for_node), and only that node's neighbors
    // expand the frontier — so the warmed set corresponds to the first few
    // graph hops around entry_point rather than the full slot population of
    // each touched page. Hops are fetched in parallel with num_threads OMP
    // threads. Budget is on *pages*: when a hop's frontier exceeds the
    // remaining page budget, we parallel-pin a `remaining`-node prefix
    // (likely undershoots due to node->page sharing), then tail-pin the rest
    // of the hop's frontier sequentially until visited_pages == num_pages.
    // If the frontier is exhausted before the budget is filled, terminates
    // undershot.
    void warmup_bfs_pages(uint32_t entry_point, uint32_t num_pages,
                          uint32_t num_threads);
    // Drain every thread's TLS hit buffer, then zero globals. Prefer over
    // bare stats().reset(), which leaves TLS to leak into the next phase.
    void reset_stats();
    void start_bg_flush(float high_wm = 0.70f, float low_wm = 0.30f);
    void stop_bg_flush();
    void flush();
    uint32_t flush_dirty_budget(uint32_t max_pages);
    double dirty_ratio() const;
    void set_entry_point(uint32_t entry_point);
    void seed_entry_pool_from_reservoir();
    void get_entry_points(std::vector<uint32_t>& out);
    void save_snapshot(const std::string& meta_path);
    void load_snapshot(const std::string& meta_path);
    void save_dense_pq_codes(const std::string& pq_prefix) const;
    // Stats + info
    InPlaceIOStats& stats();
    const InPlaceIOStats& stats() const { return _stats; }
    uint32_t entry_point() {
        std::vector<uint32_t> pts;
        get_entry_points(pts);
        return pts.empty() ? INVALID_NODE : pts[0];
    }
    uint32_t entry_point() const {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        return _entry_pool[0];
    }
    uint32_t num_active() const;
    // Current RCU-published metadata capacity. Grows geometrically by 1.2x
    // (at least `needed`); the configured max_dataset_size is only the
    // initial lower bound.
    size_t meta_capacity() const;
    bool meta_growth_active() const { return _meta.growth_active(); }
    size_t retired_meta_generation_count() const { return _meta.retired_count(); }
    // Empty-slot ratio over pages that have held at least one node. Virgin
    // reserve pages created by batched heap growth are excluded.
    // Fraction of ever-used pages that a perfectly packed live set can
    // reclaim. A densely packed graph has zero fragmentation even when its
    // final page is partially filled.
    double fragmentation_ratio() const;
    uint32_t aligned_dim() const { return _aligned_dim; }
    uint32_t slot_size() const { return _slot_size; }
    uint32_t page_size() const { return _page_size; }
    uint32_t max_degree() const { return _Mmax; }
    uint32_t total_pages() const { return _heap.total_pages(_graph_file_no); }
    // Heap file backing the graph.
    const std::string& heap_path() const { return _heap.heap_path(_graph_file_no); }
    size_t   memory_usage_bytes() const;
    // Safe to call without a meta writer ticket (may take MetaGrowthGuard).
    void ensure_pq_storage();

    // Memory breakdown for metrics
    // Configured graph buffer-pool budget.
    size_t buffer_pool_bytes() const;
    size_t locator_bytes() const;
    size_t pq_data_bytes() const;

 private:
    char* slot_ptr(PageGuard& page, uint16_t slot_idx) const;
    const char* slot_ptr(const PageGuard& page, uint16_t slot_idx) const;
    bool slot_occupied(const PageGuard& page, uint16_t slot_idx) const;

    struct PageDir {
        uint16_t slots_per_page;
        uint16_t num_occupied;
    };

    // Picks a page with a free slot and counts the new occupant. Caller holds _pages_mtx.
    uint32_t claim_page();
    // Counts a new occupant on page pid, growing the heap if pid is past its end. Caller holds _pages_mtx.
    void claim_slot_on_page_locked(uint32_t pid);
    void grow_pages_locked();
    PageID graph_page(PageNo page_no) const { return {_graph_file_no, page_no}; }
    void materialize_node(uint32_t node_id);
    void reset_metadata(size_t expected_entries = 0);
    void reset_node_slot(uint32_t node_id);
    std::vector<uint32_t> reclaim_delete_pending_ids(
        const std::vector<uint32_t>& node_ids, uint32_t num_threads);

    static constexpr size_t kEntryPoolSize = 10;
    static constexpr size_t kReservoirSize = 64;
    using MetaArrays = NodeMetaTable::Arrays;
    MetaArrays* meta_arrays() const { return _meta.current(); }
    const uint8_t* pending_pq_codes(uint32_t node_id) const;
    void clear_pending_pq_codes();

    // Stores only the reference to _stats (declared below); never touches it during construction.
    NodeMetaTable _meta{_stats};

    uint32_t _dim         = 0;
    uint32_t _aligned_dim = 0;
    uint32_t _Mmax        = 0;
    uint32_t _elem_size   = 0;
    uint32_t _page_size   = 0;
    uint32_t _slot_size   = 0;
    uint32_t _slots_per_page = 0;

    std::shared_ptr<FileManager> _file_manager;
    FileNo _graph_file_no = INVALID_FILE;
    BufferPool      _heap;
    InPlaceIOStats _stats;
    std::mutex _delete_pending_since_maintenance_mtx;
    std::vector<uint32_t> _delete_pending_since_maintenance;
    using DeletedTagSet = phmap::parallel_flat_hash_set_m<TagType>;
    mutable std::shared_mutex _deleted_tags_swap_mtx;
    DeletedTagSet _deleted_tags_current;
    DeletedTagSet _deleted_tags_draining;
    std::unique_ptr<LazyDeletes> _lazy_deletes;
    // One past the highest node ID ever materialized.
    std::atomic<uint32_t> _id_high_water{0};

    // PQ
    // Read-only staging for initial import; live mutable codes belong to MetaArrays.
    const uint8_t* _pending_pq_codes = nullptr;
    size_t _pending_pq_code_count = 0;
    void* _pending_pq_mmap_base = nullptr;
    size_t _pending_pq_mmap_bytes = 0;
    uint32_t _n_chunks   = 0;
    bool _exact_search = false;
    std::atomic<uint32_t> _num_active{0};

    // Page directory — protected by _pages_mtx
    mutable std::mutex    _pages_mtx;
    std::vector<PageDir>  _page_dir;
    std::vector<uint8_t>  _page_ever_used;
    std::atomic<uint64_t>  _ever_used_pages{0};
    std::atomic<uint64_t>  _occupied_slots{0};
    std::set<uint32_t>    _pages_with_space;
    // Fully-empty pages in the free suffix [begin, end). claim_page drains
    // this before growing the heap. Empty on the dense path.
    uint32_t              _empty_page_span_begin = 0;
    uint32_t              _empty_page_span_end = 0;
    uint32_t              _alloc_page_rr_cursor = 0;

    static constexpr uint32_t kPageAllocBatchSize = 256;

    mutable std::mutex _entry_mtx;
    std::array<uint32_t, kEntryPoolSize> _entry_pool;
    std::array<uint32_t, kReservoirSize> _node_reservoir;
    size_t _reservoir_cursor = 0;

    friend class ConstNodeRef;
    friend class MutableNodeRef;
    friend class FrameReadGuard;
    friend class FrameWriteGuard;
};

template <typename T>
const T* ConstNodeRef::coords() const {
    if (!valid()) return nullptr;
    if (_store->_elem_size != sizeof(T)) {
        throw std::runtime_error("ConstNodeRef::coords type size mismatch");
    }
    return reinterpret_cast<const T*>(_slot_ptr + sizeof(PackedSlotHeader));
}

template <typename T>
T* MutableNodeRef::coords() {
    if (!valid()) return nullptr;
    if (_store->_elem_size != sizeof(T)) {
        throw std::runtime_error("MutableNodeRef::coords type size mismatch");
    }
    return reinterpret_cast<T*>(_slot_ptr + sizeof(PackedSlotHeader));
}

template <typename T>
const T* MutableNodeRef::coords() const {
    if (!valid()) return nullptr;
    if (_store->_elem_size != sizeof(T)) {
        throw std::runtime_error("MutableNodeRef::coords type size mismatch");
    }
    return reinterpret_cast<const T*>(_slot_ptr + sizeof(PackedSlotHeader));
}

template <typename T>
void MutableNodeRef::set_coords(const T* coords_src) {
    if (!valid() || coords_src == nullptr) return;
    T* dst = coords<T>();
    std::memcpy(dst, coords_src, static_cast<size_t>(_store->_aligned_dim) * sizeof(T));
    mark_dirty();
}

}  // namespace inplace
}  // namespace diskann
