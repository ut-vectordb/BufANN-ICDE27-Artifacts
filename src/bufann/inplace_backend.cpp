#include "bufann/inplace_backend.h"
#include "index.h"
#include "logger.h"
#include "threadpool.h"

#include <cerrno>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace diskann {
namespace inplace {

static inline uint32_t page_header_bytes(uint32_t slots_per_page);

namespace {

// One snapshot record per present node. page_id and slot_idx follow from node_id; they stay so the file format
// does not change.
struct SerializedRID {
    uint32_t node_id;
    uint32_t page_id;
    uint16_t slot_idx;
    uint8_t _legacy[2];
};
static_assert(sizeof(SerializedRID) == 12, "snapshot RID record layout changed");

void* checked_aligned_alloc(size_t alignment, size_t size, const char* what) {
    void* ptr = nullptr;
    int rc = posix_memalign(&ptr, alignment, size);
    if (rc != 0) {
        throw std::runtime_error(std::string("posix_memalign failed for ") + what +
                                 " rc=" + std::to_string(rc));
    }
    return ptr;
}

#if defined(__GNUC__) || defined(__clang__)
struct PackedSlotHeaderView {
    uint16_t degree;
    uint8_t flags;
    uint8_t _pad;
    uint32_t node_id;
} __attribute__((packed, may_alias));
static_assert(sizeof(PackedSlotHeaderView) == sizeof(PackedSlotHeader),
              "packed slot view must match header size");

inline PackedSlotHeader load_slot_header(const char* slot_ptr) {
    const auto* view = reinterpret_cast<const PackedSlotHeaderView*>(slot_ptr);
    PackedSlotHeader hdr;
    hdr.degree = view->degree;
    hdr.flags = view->flags;
    hdr._pad = view->_pad;
    hdr.node_id = view->node_id;
    return hdr;
}
#else
inline PackedSlotHeader load_slot_header(const char* slot_ptr) {
    PackedSlotHeader hdr;
    std::memcpy(&hdr, slot_ptr, sizeof(hdr));
    return hdr;
}
#endif

}  // namespace

// ===========================================================================
// InPlaceSearchScratch
// ===========================================================================
void InPlaceSearchScratch::init(uint32_t aligned_dim, uint32_t n_chunks,
                                uint32_t elem_size,
                                uint32_t search_list_size,
                                uint32_t max_degree,
                                uint32_t beamwidth) {
    (void)aligned_dim;
    (void)elem_size;

    size_t dist_bytes = (size_t)MAX_SCRATCH_NODES * sizeof(float);
    dist_scratch = static_cast<float*>(checked_aligned_alloc(32, dist_bytes, "dist_scratch"));
    memset(dist_scratch, 0, dist_bytes);

    if (n_chunks > 0) {
        size_t pq_dist_bytes = (size_t)n_chunks * 256 * sizeof(float);
        pq_dists = static_cast<float*>(checked_aligned_alloc(32, pq_dist_bytes, "pq_dists"));
        memset(pq_dists, 0, pq_dist_bytes);

        size_t pq_coord_bytes = (size_t)n_chunks * MAX_SCRATCH_NODES;
        pq_coord_scratch = static_cast<uint8_t*>(
            checked_aligned_alloc(32, pq_coord_bytes, "pq_coord_scratch"));
        memset(pq_coord_scratch, 0, pq_coord_bytes);
    }

    const size_t l_cap = std::max<size_t>(1, search_list_size);
    const size_t degree_cap = std::max<size_t>(1, max_degree);
    const size_t beam_cap = std::max<size_t>(1, beamwidth);
    const size_t frontier_cap = std::max<size_t>(degree_cap * 2, l_cap * 2);
    const size_t visited_cap = std::min<size_t>(
        MAX_SCRATCH_NODES, std::max<size_t>(10 * l_cap, l_cap * degree_cap));

    traversal_results.reserve(10 * l_cap);
    traversal_retset.reserve(l_cap + 1);
    frontier_nodes.reserve(beam_cap);
    frontier_ids.reserve(beam_cap);
    visited.reserve(visited_cap);
    frontier_neighbor_ids.reserve(frontier_cap);
    frontier_candidate_ids.reserve(frontier_cap);
}

InPlaceSearchScratch::~InPlaceSearchScratch() {
    if (pq_dists)              free(pq_dists);
    if (dist_scratch)          free(dist_scratch);
    if (pq_coord_scratch)      free(pq_coord_scratch);
}

void InPlaceSearchScratch::flush_distance_stats(InPlaceIOStats& stats) {
    if (local_query_comparable_cpu_ns || local_query_comparable_cmps) {
        stats.query_comparable_cpu_ns.fetch_add(local_query_comparable_cpu_ns,
                                                std::memory_order_relaxed);
        stats.query_comparable_cmps.fetch_add(local_query_comparable_cmps,
                                              std::memory_order_relaxed);
    }
    local_query_comparable_cpu_ns = 0;
    local_query_comparable_cmps = 0;
    flush_query_hot_stats(stats);
}



// ===========================================================================
// InPlaceGraphStore
// ===========================================================================
InPlaceGraphStore::InPlaceGraphStore() {
    _entry_pool.fill(INVALID_NODE);
    _node_reservoir.fill(INVALID_NODE);
}

InPlaceGraphStore::~InPlaceGraphStore() {
    clear_pending_pq_codes();
    stop_bg_flush();
    _meta.shutdown();
}

void InPlaceGraphStore::ensure_pq_storage() {
    if (_n_chunks == 0) return;
    _meta.ensure_pq_chunks(_n_chunks);
}

const uint8_t* InPlaceGraphStore::pq_codes_for_node(uint32_t node_id) const {
    MetaArrays* meta = meta_arrays();
    if (meta != nullptr && meta->pq_codes != nullptr && node_present(node_id)) {
        return meta->pq_codes.get() + static_cast<size_t>(node_id) * _n_chunks;
    }
    return pending_pq_codes(node_id);
}

const uint8_t* InPlaceGraphStore::pending_pq_codes(uint32_t node_id) const {
    if (_pending_pq_codes == nullptr || _n_chunks == 0 ||
        node_id >= _pending_pq_code_count) {
        return nullptr;
    }
    return _pending_pq_codes + static_cast<size_t>(node_id) * _n_chunks;
}

void InPlaceGraphStore::clear_pending_pq_codes() {
    if (_pending_pq_mmap_base != nullptr) {
        munmap(_pending_pq_mmap_base, _pending_pq_mmap_bytes);
    }
    _pending_pq_codes = nullptr;
    _pending_pq_code_count = 0;
    _pending_pq_mmap_base = nullptr;
    _pending_pq_mmap_bytes = 0;
}

uint32_t InPlaceGraphStore::reserve_node_id() {
    // claim_page counts us as an occupant of pid, so pid keeps a Free slot for us.
    uint32_t pid = 0;
    {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        pid = claim_page();
    }
    const uint32_t base = pid * _slots_per_page;
    // Grow now, while this thread holds no writer ticket; growth waits for every writer.
    reserve_meta_capacity(static_cast<size_t>(base) + _slots_per_page);

    // Hold a writer ticket so that growth cannot copy the arrays while we change a state.
    MetaReadGuard writer = enter_meta_write();
    MetaArrays* meta = meta_arrays();
    for (int pass = 0; pass < 3; ++pass) {
        for (uint32_t slot = 0; slot < _slots_per_page; ++slot) {
            NodeState expected = NodeState::Free;
            if (meta->states[base + slot].compare_exchange_strong(expected, NodeState::Allocating,
                                                                  std::memory_order_acq_rel)) {
                return base + slot;
            }
        }
    }
    throw ANNException("reserve_node_id: claimed page " + std::to_string(pid) + " has no free slot", -1,
                       __FUNCSIG__, __FILE__, __LINE__);
}

void InPlaceGraphStore::reserve_meta_capacity(size_t needed) {
    _meta.reserve(needed, _n_chunks);
}

void InPlaceGraphStore::reset_metadata(size_t expected_entries) {
    const size_t initialized_slots = std::min<size_t>(
        _id_high_water.load(std::memory_order_relaxed),
        meta_arrays() == nullptr ? 0 : meta_arrays()->capacity);
    {
        std::lock_guard<std::mutex> lk(_delete_pending_since_maintenance_mtx);
        _delete_pending_since_maintenance.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
        _deleted_tags_current.clear();
        _deleted_tags_draining.clear();
    }
    {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        _empty_page_span_begin = 0;
        _empty_page_span_end = 0;
    }
    _num_active.store(0, std::memory_order_relaxed);
    _id_high_water.store(0, std::memory_order_relaxed);
    reserve_meta_capacity(expected_entries);
    MetaArrays* meta = meta_arrays();
    if (meta != nullptr) {
        meta->n_chunks = _n_chunks;
    }
    // IDs at or above the high-water mark were never used, so they are still Free.
    for (size_t i = 0; i < initialized_slots; ++i) {
        reset_node_slot(static_cast<uint32_t>(i));
    }
}

void InPlaceGraphStore::reset_node_slot(uint32_t node_id) {
    MetaArrays* meta = meta_arrays();
    if (meta == nullptr || node_id >= meta->capacity) return;
    meta->tags[node_id] = INVALID_TAG;
    if (_n_chunks > 0 && meta->pq_codes != nullptr) {
        memset(meta->pq_codes.get() +
                   static_cast<size_t>(node_id) * _n_chunks, 0,
               _n_chunks);
    }
    // Free last: reserve_node_id may take the slot once it sees Free.
    meta->states[node_id].store(NodeState::Free, std::memory_order_release);
}

bool InPlaceGraphStore::node_present(uint32_t node_id) const {
    MetaArrays* meta = meta_arrays();
    if (meta == nullptr || node_id >= meta->capacity) return false;
    const NodeState state = meta->states[node_id].load(std::memory_order_acquire);
    return state == NodeState::Active || state == NodeState::DeletePending;
}

void InPlaceGraphStore::materialize_node(uint32_t node_id) {
    // Capacity and PQ storage must already be reserved before a metadata write
    // epoch is held. Growing here would wait for this thread's own writer
    // ticket and deadlock.
    MetaArrays* meta = meta_arrays();
    if (_n_chunks > 0 &&
        (meta->n_chunks != _n_chunks || meta->pq_codes == nullptr)) {
        throw ANNException(
            "materialize_node: PQ metadata storage was not prepared",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    meta->tags[node_id] = static_cast<TagType>(node_id);
    if (_n_chunks > 0) {
        uint8_t* dst = meta->pq_codes.get() +
                       static_cast<size_t>(node_id) * _n_chunks;
        memset(dst, 0, _n_chunks);
        const uint8_t* src = pending_pq_codes(node_id);
        if (src != nullptr) memcpy(dst, src, _n_chunks);
    }
    uint32_t observed = _id_high_water.load(std::memory_order_relaxed);
    while (observed <= node_id &&
           !_id_high_water.compare_exchange_weak(
               observed, node_id + 1, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

ConstNodeRef::ConstNodeRef(const InPlaceGraphStore* store, const char* slot_ptr)
    : _store(store), _slot_ptr(slot_ptr) {}

uint16_t ConstNodeRef::degree() const {
    if (!valid()) return 0;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    return hdr.degree;
}

uint32_t ConstNodeRef::node_id() const {
    if (!valid()) return INVALID_NODE;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    return hdr.node_id;
}

const uint32_t* ConstNodeRef::neighbors() const {
    if (!valid()) return nullptr;
    return reinterpret_cast<const uint32_t*>(
        _slot_ptr + sizeof(PackedSlotHeader) +
        static_cast<size_t>(_store->_aligned_dim) * _store->_elem_size);
}

const char* ConstNodeRef::coords_bytes() const {
    if (!valid()) return nullptr;
    return _slot_ptr + sizeof(PackedSlotHeader);
}

MutableNodeRef::MutableNodeRef(InPlaceGraphStore* store, char* slot_ptr,
                               FrameWriteGuard* guard)
    : _store(store), _slot_ptr(slot_ptr), _guard(guard) {}

void MutableNodeRef::mark_dirty() {
    if (_guard != nullptr) _guard->mark_dirty();
}

uint16_t MutableNodeRef::degree() const {
    if (!valid()) return 0;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    return hdr.degree;
}

uint32_t MutableNodeRef::node_id() const {
    if (!valid()) return INVALID_NODE;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    return hdr.node_id;
}

uint32_t* MutableNodeRef::neighbors() {
    if (!valid()) return nullptr;
    return reinterpret_cast<uint32_t*>(
        _slot_ptr + sizeof(PackedSlotHeader) +
        static_cast<size_t>(_store->_aligned_dim) * _store->_elem_size);
}

const uint32_t* MutableNodeRef::neighbors() const {
    if (!valid()) return nullptr;
    return reinterpret_cast<const uint32_t*>(
        _slot_ptr + sizeof(PackedSlotHeader) +
        static_cast<size_t>(_store->_aligned_dim) * _store->_elem_size);
}

char* MutableNodeRef::coords_bytes() {
    if (!valid()) return nullptr;
    return _slot_ptr + sizeof(PackedSlotHeader);
}

const char* MutableNodeRef::coords_bytes() const {
    if (!valid()) return nullptr;
    return _slot_ptr + sizeof(PackedSlotHeader);
}

void MutableNodeRef::set_degree(uint16_t degree_value) {
    if (!valid()) return;
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    hdr.degree = static_cast<uint16_t>(
        std::min<size_t>(degree_value, _store->_Mmax));
    std::memcpy(_slot_ptr, &hdr, sizeof(hdr));
    mark_dirty();
}

void MutableNodeRef::set_neighbors(const uint32_t* neighbors_src, size_t count) {
    if (!valid() || (count != 0 && neighbors_src == nullptr)) return;
    const size_t n = std::min<size_t>(count, _store->_Mmax);
    if (n != 0) {
        std::memcpy(neighbors(), neighbors_src, n * sizeof(uint32_t));
    }
    PackedSlotHeader hdr = load_slot_header(_slot_ptr);
    hdr.degree = static_cast<uint16_t>(n);
    std::memcpy(_slot_ptr, &hdr, sizeof(hdr));
    mark_dirty();
}

char* InPlaceGraphStore::slot_ptr(PageGuard& page, uint16_t slot_idx) const {
    if (!page.valid() || slot_idx >= _slots_per_page) return nullptr;
    return page.data() + page_header_bytes(_slots_per_page) + static_cast<size_t>(slot_idx) * _slot_size;
}

const char* InPlaceGraphStore::slot_ptr(const PageGuard& page, uint16_t slot_idx) const {
    if (!page.valid() || slot_idx >= _slots_per_page) return nullptr;
    return page.data() + page_header_bytes(_slots_per_page) + static_cast<size_t>(slot_idx) * _slot_size;
}

bool InPlaceGraphStore::slot_occupied(const PageGuard& page, uint16_t slot_idx) const {
    if (!page.valid() || slot_idx >= _slots_per_page) return false;
    const uint8_t* bmap = reinterpret_cast<const uint8_t*>(page.data() + 8);
    return ((bmap[slot_idx / 8] >> (slot_idx % 8)) & 1u) != 0;
}

// No-lock view over a page already pinned READ; its PageGuard holds the frame
// latch shared, so reads stay safe for the pin's lifetime.
FrameReadGuard::FrameReadGuard(InPlaceGraphStore* store, PageGuard* page)
    : _store(store), _page(page) {}

ConstNodeRef FrameReadGuard::read_slot(uint16_t slot_idx) const {
    if (_store == nullptr || _page == nullptr || !_page->valid()) return ConstNodeRef();
    if (!_store->slot_occupied(*_page, slot_idx)) return ConstNodeRef();
    return ConstNodeRef(_store, _store->slot_ptr(*_page, slot_idx));
}

ConstNodeRef FrameReadGuard::read_node(uint32_t node_id) const {
    if (_store == nullptr || _page == nullptr || !_page->valid()) return ConstNodeRef();
    if (_store->node_page(node_id) != _store->graph_page_id(*_page)) return ConstNodeRef();
    return read_slot(_store->node_slot(node_id));
}

// No-lock view over a page already pinned WRITE; its PageGuard holds the frame
// latch exclusive, so writes stay serialized for the pin's lifetime.
FrameWriteGuard::FrameWriteGuard(InPlaceGraphStore* store, PageGuard* page)
    : _store(store), _page(page) {}

// Marks the PageGuard itself, so the page is dirty as soon as a slot is written, whatever order the view and the pin
// are released in.
void FrameWriteGuard::mark_dirty() {
    if (_page != nullptr) _page->mark_dirty();
}

ConstNodeRef FrameWriteGuard::read_slot(uint16_t slot_idx) const {
    if (_store == nullptr || _page == nullptr || !_page->valid()) return ConstNodeRef();
    if (!_store->slot_occupied(*_page, slot_idx)) return ConstNodeRef();
    return ConstNodeRef(_store, _store->slot_ptr(*_page, slot_idx));
}

ConstNodeRef FrameWriteGuard::read_node(uint32_t node_id) const {
    if (_store == nullptr || _page == nullptr || !_page->valid()) return ConstNodeRef();
    if (_store->node_page(node_id) != _store->graph_page_id(*_page)) return ConstNodeRef();
    return read_slot(_store->node_slot(node_id));
}

MutableNodeRef FrameWriteGuard::write_slot(uint16_t slot_idx) {
    if (_store == nullptr || _page == nullptr || !_page->valid()) return MutableNodeRef();
    if (!_store->slot_occupied(*_page, slot_idx)) return MutableNodeRef();
    return MutableNodeRef(_store, _store->slot_ptr(*_page, slot_idx), this);
}

MutableNodeRef FrameWriteGuard::write_node(uint32_t node_id) {
    if (_store == nullptr || _page == nullptr || !_page->valid()) return MutableNodeRef();
    if (_store->node_page(node_id) != _store->graph_page_id(*_page)) return MutableNodeRef();
    return write_slot(_store->node_slot(node_id));
}

FrameReadGuard InPlaceGraphStore::read_guard(PageGuard& page) {
    return FrameReadGuard(this, &page);
}

FrameWriteGuard InPlaceGraphStore::write_guard(PageGuard& page) {
    return FrameWriteGuard(this, &page);
}

PageGuard InPlaceGraphStore::pin_page(uint32_t page_id, AccessMode mode) {
    PageGuard page = _heap.pin(graph_page(page_id), mode);
    record_thread_page_accesses();
    return page;
}

PageGuard InPlaceGraphStore::try_pin_page_resident(uint32_t page_id, AccessMode mode) {
    return _heap.try_pin_resident(graph_page(page_id), mode);
}

bool InPlaceGraphStore::page_resident(uint32_t page_id) const {
    return _heap.is_resident(graph_page(page_id));
}

PageGuard InPlaceGraphStore::pin_page_for_node(uint32_t node_id, AccessMode mode) {
    // Allocating slots are allowed so the inserting thread can write its own node. Other callers check is_active.
    MetaArrays* meta = meta_arrays();
    if (meta == nullptr || node_id >= meta->capacity ||
        meta->states[node_id].load(std::memory_order_acquire) == NodeState::Free) {
        return PageGuard();
    }
    return pin_page(node_page(node_id), mode);
}

void InPlaceGraphStore::pin_pages_batch(const uint32_t* page_ids, size_t n, AccessMode mode, PageGuard* out_pages) {
    assert(n <= BufferPool::kPinBatchMax);
    if (n == 0) return;


    PageID buffer_page_ids[BufferPool::kPinBatchMax];
    for (size_t i = 0; i < n; ++i) {
        buffer_page_ids[i] = graph_page(page_ids[i]);
    }
    _heap.pin_batch(buffer_page_ids, out_pages, n, mode);
    record_thread_page_accesses(n);
}

void InPlaceGraphStore::pin_pages_batch_transient(const uint32_t* page_ids, size_t n, AccessMode mode,
                                                  PageGuard* out_pages) {
    assert(n <= BufferPool::kPinBatchMax);
    if (n == 0) return;
    PageID buffer_page_ids[BufferPool::kPinBatchMax];
    for (size_t i = 0; i < n; ++i) buffer_page_ids[i] = graph_page(page_ids[i]);
    _heap.pin_batch_transient(buffer_page_ids, out_pages, n, mode, /*account_sequential_read=*/true);
    record_thread_page_accesses(n);
}

size_t InPlaceGraphStore::pin_pages_for_nodes_batch(
    const uint32_t* node_ids, size_t n, AccessMode mode,
    PageGuard* out_page_guards, uint16_t* out_node_to_guard) {
    assert(n <= BufferPool::kPinBatchMax);
    if (n == 0) return 0;

    // Map each present node to its page, collecting every distinct page once.
    uint32_t unique_pages[BufferPool::kPinBatchMax];
    size_t num_pages = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!node_present(node_ids[i])) {
            out_node_to_guard[i] = kNoPage;
            continue;
        }
        const uint32_t page_id = node_page(node_ids[i]);
        size_t guard = 0;
        for (; guard < num_pages; ++guard) {
            if (unique_pages[guard] == page_id) break;
        }
        if (guard == num_pages) unique_pages[num_pages++] = page_id;
        out_node_to_guard[i] = static_cast<uint16_t>(guard);
    }

    if (num_pages == 0) return 0;

    pin_pages_batch(unique_pages, num_pages, mode, out_page_guards);
    return num_pages;
}

void InPlaceGraphStore::init(uint32_t dim, uint32_t Mmax,
                             uint32_t elem_size_bytes, uint32_t page_size,
                             uint32_t buffer_pool_frames,
                             const std::string& heap_path,
                             uint32_t flush_budget_pages_per_cycle,
                             uint32_t flush_wakeup_ms,
                             bool truncate_heap,
                             uint32_t max_nodes,
                             uint32_t meta_growth_copy_threads,
                             uint32_t initial_pq_chunks) {
    if (meta_growth_copy_threads == 0) {
        throw ANNException("metadata growth copy threads must be non-zero", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    _dim       = dim;
    _aligned_dim = (uint32_t)ROUND_UP(dim, 8);
    _Mmax      = Mmax;
    _elem_size = elem_size_bytes;
    _page_size = page_size;

    _slot_size = sizeof(PackedSlotHeader) +
                 _aligned_dim * _elem_size +
                 Mmax * sizeof(uint32_t);

    // Compute slots per page iteratively (page header + bitmap + slots)
    uint32_t avail = page_size - 8;  // page header is 8 bytes
    _slots_per_page = 0;
    for (uint32_t s = 1; s <= avail / _slot_size; s++) {
        uint32_t bmap = (s + 7) / 8;
        if (8 + bmap + s * _slot_size <= page_size) {
            _slots_per_page = s;
        }
    }
    if (_slots_per_page == 0) {
        throw ANNException("Slot size exceeds page size", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    _file_manager = std::make_shared<FileManager>(page_size);
    _graph_file_no = _file_manager->register_file(heap_path, truncate_heap);
    _heap.init(page_size, buffer_pool_frames, _file_manager, _graph_file_no, &_stats,
               flush_budget_pages_per_cycle, flush_wakeup_ms);
    _ever_used_pages.store(0, std::memory_order_relaxed);
    _occupied_slots.store(0, std::memory_order_relaxed);
    clear_pending_pq_codes();
    _n_chunks = initial_pq_chunks;
    _meta.init(max_nodes == 0 ? NodeMetaTable::kDefaultCapacity : max_nodes, initial_pq_chunks, meta_growth_copy_threads);
    reset_metadata();
    {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        _entry_pool.fill(INVALID_NODE);
        _node_reservoir.fill(INVALID_NODE);
        _reservoir_cursor = 0;
    }
}

static inline uint32_t page_bitmap_bytes(uint32_t slots_per_page) {
    return (slots_per_page + 7) / 8;
}

static inline uint32_t page_header_bytes(uint32_t slots_per_page) {
    return 8 + page_bitmap_bytes(slots_per_page);
}

static inline void set_page_bitmap(char* page_data, uint16_t slot_idx) {
    uint8_t* bmap = reinterpret_cast<uint8_t*>(page_data + 8);
    bmap[slot_idx / 8] |= (1u << (slot_idx % 8));
}

static inline bool test_page_bitmap(const char* page_data, uint16_t slot_idx) {
    const uint8_t* bmap = reinterpret_cast<const uint8_t*>(page_data + 8);
    return ((bmap[slot_idx / 8] >> (slot_idx % 8)) & 1u) != 0;
}


// Round-robin over the pages that have a free slot.
uint32_t InPlaceGraphStore::claim_page() {
    for (;;) {
        if (!_pages_with_space.empty()) {
            auto it = _pages_with_space.lower_bound(_alloc_page_rr_cursor);
            if (it == _pages_with_space.end()) it = _pages_with_space.begin();
            uint32_t pid = *it;
            PageDir& dir = _page_dir[pid];
            if (pid < _page_ever_used.size() &&
                _page_ever_used[pid] == 0) {
                _page_ever_used[pid] = 1;
                _ever_used_pages.fetch_add(1, std::memory_order_relaxed);
            }
            dir.num_occupied++;
            _occupied_slots.fetch_add(1, std::memory_order_relaxed);
            if (dir.num_occupied >= dir.slots_per_page) {
                _pages_with_space.erase(it);
            }
            _alloc_page_rr_cursor = pid + 1;
            return pid;
        }
        // Drain fully-empty free-suffix pages before growing past HWM.
        while (_empty_page_span_begin < _empty_page_span_end) {
            const uint32_t pid = _empty_page_span_begin;
            if (pid >= _page_dir.size()) {
                _empty_page_span_begin = _empty_page_span_end;
                break;
            }
            PageDir& dir = _page_dir[pid];
            if (dir.num_occupied >= dir.slots_per_page) {
                ++_empty_page_span_begin;
                continue;
            }
            if (pid < _page_ever_used.size() &&
                _page_ever_used[pid] == 0) {
                _page_ever_used[pid] = 1;
                _ever_used_pages.fetch_add(1, std::memory_order_relaxed);
            }
            dir.num_occupied++;
            _occupied_slots.fetch_add(1, std::memory_order_relaxed);
            if (dir.num_occupied >= dir.slots_per_page) {
                ++_empty_page_span_begin;
            }
            _alloc_page_rr_cursor = pid + 1;
            return pid;
        }
        grow_pages_locked();
    }
}

// Batch grow: _page_dir updated before _pages_with_space. Caller must hold _pages_mtx.
void InPlaceGraphStore::grow_pages_locked() {
    uint32_t base = _heap.grow_heap(_graph_file_no, kPageAllocBatchSize);
    _page_dir.resize(base + kPageAllocBatchSize,
                     PageDir{static_cast<uint16_t>(_slots_per_page), 0});
    _page_ever_used.resize(base + kPageAllocBatchSize, 0);
    for (uint32_t i = base; i < base + kPageAllocBatchSize; ++i) {
        _pages_with_space.insert(_pages_with_space.end(), i);
    }
}

void InPlaceGraphStore::claim_slot_on_page_locked(uint32_t pid) {
    while (pid >= _page_dir.size()) grow_pages_locked();
    PageDir& dir = _page_dir[pid];
    if (_page_ever_used[pid] == 0) {
        _page_ever_used[pid] = 1;
        _ever_used_pages.fetch_add(1, std::memory_order_relaxed);
    }
    dir.num_occupied++;
    _occupied_slots.fetch_add(1, std::memory_order_relaxed);
    if (dir.num_occupied >= dir.slots_per_page) {
        _pages_with_space.erase(pid);
    }
}

void InPlaceGraphStore::allocate_node(uint32_t node_id) {
    MetaArrays* meta = meta_arrays();
    if (meta == nullptr || static_cast<size_t>(node_id) >= meta->capacity) {
        throw ANNException(
            "allocate_node: metadata capacity was not reserved for node_id " +
                std::to_string(node_id),
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    const uint32_t pid = node_page(node_id);
    const uint16_t slot = node_slot(node_id);
    if (meta->states[node_id].load(std::memory_order_acquire) != NodeState::Allocating) {
        // A caller-chosen ID (bulk load / import): claim that exact slot.
        std::lock_guard<std::mutex> lk(_pages_mtx);
        NodeState expected = NodeState::Free;
        if (!meta->states[node_id].compare_exchange_strong(expected, NodeState::Allocating,
                                                           std::memory_order_acq_rel)) {
            throw ANNException("allocate_node: node_id " + std::to_string(node_id) + " is already allocated", -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        claim_slot_on_page_locked(pid);
    }

    PageGuard frame = pin_page(pid, WRITE);
    char* page_data = frame.data();
    if (test_page_bitmap(page_data, slot)) {
        throw ANNException("allocate_node: page bitmap already marks node_id " + std::to_string(node_id), -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    set_page_bitmap(page_data, slot);
    // Write slots_per_page to page header (idempotent; new pages are zero-filled by OS).
    uint16_t spp = static_cast<uint16_t>(_slots_per_page);
    memcpy(page_data, &spp, 2);
    PackedSlotHeader hdr;
    hdr.degree  = 0;
    hdr.flags   = 0;
    hdr._pad    = 0;
    hdr.node_id = node_id;
    memcpy(slot_ptr(frame, slot), &hdr, sizeof(hdr));
    frame.mark_dirty();
    materialize_node(node_id);
}

void InPlaceGraphStore::publish_node(uint32_t node_id) {
    MetaArrays* meta = meta_arrays();
    NodeState expected = NodeState::Allocating;
    if (meta->states[node_id].compare_exchange_strong(expected, NodeState::Active, std::memory_order_release)) {
        _num_active.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lk(_entry_mtx);
    _node_reservoir[_reservoir_cursor] = node_id;
    _reservoir_cursor = (_reservoir_cursor + 1) % kReservoirSize;
}

void InPlaceGraphStore::batch_fetch_coords(const std::vector<uint32_t>& ids,
                                           char* out_coords,
                                           std::vector<uint8_t>& found) {
    found.assign(ids.size(), 0);
    if (ids.empty() || out_coords == nullptr) return;

    struct BatchItem {
        uint32_t page_id;
        uint16_t slot_idx;
        uint32_t node_id;
        size_t out_idx;
    };

    std::vector<BatchItem> items;
    items.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t node_id = ids[i];
        if (!node_present(node_id)) continue;
        items.push_back(BatchItem{node_page(node_id), node_slot(node_id), node_id, i});
    }
    std::sort(items.begin(), items.end(), [](const BatchItem& a, const BatchItem& b) {
        if (a.page_id != b.page_id) return a.page_id < b.page_id;
        return a.slot_idx < b.slot_idx;
    });

    size_t cursor = 0;
    while (cursor < items.size()) {
        uint32_t page_id = items[cursor].page_id;
        PageGuard frame = pin_page(page_id, READ);
        if (!frame.valid()) {
            while (cursor < items.size() && items[cursor].page_id == page_id) {
                ++cursor;
            }
            continue;
        }
        auto guard = read_guard(frame);

        while (cursor < items.size() && items[cursor].page_id == page_id) {
            const auto& item = items[cursor];
            ConstNodeRef node = guard.read_slot(item.slot_idx);
            if (node.valid() &&
                node.node_id() == item.node_id &&
                is_active(item.node_id)) {
                const char* coords_ptr = node.coords_bytes();
                std::memcpy(
                    out_coords + item.out_idx * static_cast<size_t>(_aligned_dim) * _elem_size,
                    coords_ptr, static_cast<size_t>(_aligned_dim) * _elem_size);
                found[item.out_idx] = 1;
            }
            ++cursor;
        }
    }
}

bool InPlaceGraphStore::transition_active_to_delete_pending(uint32_t node_id) {
    if (!node_present(node_id)) return false;
    MetaArrays* meta = meta_arrays();
    NodeState expected = NodeState::Active;
    if (!meta->states[node_id].compare_exchange_strong(
            expected, NodeState::DeletePending,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    _num_active.fetch_sub(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(_delete_pending_since_maintenance_mtx);
        _delete_pending_since_maintenance.push_back(node_id);
    }
    _stats.total_deletes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void InPlaceGraphStore::get_entry_points(std::vector<uint32_t>& out) {
    out.clear();
    std::lock_guard<std::mutex> lk(_entry_mtx);
    for (size_t i = 0; i < kEntryPoolSize; ++i) {
        uint32_t node = _entry_pool[i];
        if (node == INVALID_NODE) continue;
        if (is_active(node)) {
            out.push_back(node);
        } else {
            // Lazy repair: find a live replacement from the reservoir.
            uint32_t replacement = INVALID_NODE;
            for (size_t step = 0; step < kReservoirSize; ++step) {
                size_t idx = (_reservoir_cursor + step) % kReservoirSize;
                uint32_t cand = _node_reservoir[idx];
                if (cand != INVALID_NODE && is_active(cand)) {
                    replacement = cand;
                    _reservoir_cursor = (idx + 1) % kReservoirSize;
                    break;
                }
            }
            _entry_pool[i] = replacement;
            if (replacement != INVALID_NODE) out.push_back(replacement);
        }
    }
}

void InPlaceGraphStore::seed_entry_pool_from_reservoir() {
    std::lock_guard<std::mutex> lk(_entry_mtx);
    // Fill slots 1..kEntryPoolSize-1 with random samples from the reservoir.
    // Slot 0 is reserved for the medoid (set via set_entry_point).
    size_t stride = std::max<size_t>(1, kReservoirSize / kEntryPoolSize);
    for (size_t i = 1; i < kEntryPoolSize; ++i) {
        size_t idx = ((i - 1) * stride) % kReservoirSize;
        uint32_t cand = _node_reservoir[idx];
        if (cand != INVALID_NODE && is_active(cand)) {
            _entry_pool[i] = cand;
        }
    }
}

bool InPlaceGraphStore::is_active(uint32_t node_id) const {
    MetaArrays* meta = meta_arrays();
    return meta != nullptr && node_id < meta->capacity &&
           meta->states[node_id].load(std::memory_order_acquire) ==
               NodeState::Active;
}

bool InPlaceGraphStore::is_delete_pending(uint32_t node_id) const {
    MetaArrays* meta = meta_arrays();
    return meta != nullptr && node_id < meta->capacity &&
           meta->states[node_id].load(std::memory_order_acquire) == NodeState::DeletePending;
}

void InPlaceGraphStore::enable_lazy_deletes(unsigned R, unsigned C, float alpha) {
    _lazy_deletes = std::make_unique<LazyDeletes>(R, C, alpha);
}

size_t InPlaceGraphStore::free_lazy_deleted(const std::vector<uint32_t>& ids, uint32_t num_threads) {
    const auto reclaimed = reclaim_delete_pending_ids(ids, num_threads);
    const std::unordered_set<uint32_t> reclaimed_set(reclaimed.begin(), reclaimed.end());
    {
        std::lock_guard<std::mutex> lk(_delete_pending_since_maintenance_mtx);
        auto& pending = _delete_pending_since_maintenance;
        pending.erase(std::remove_if(pending.begin(), pending.end(),
            [&](uint32_t id) { return reclaimed_set.count(id) != 0; }), pending.end());
    }
    return reclaimed.size();
}

bool InPlaceGraphStore::set_node_tag(uint32_t node_id, TagType tag) {
    if (tag == INVALID_TAG) return false;
    if (is_tag_deleted(tag)) return false;
    meta_arrays()->tags[node_id] = tag;
    return true;
}

TagType InPlaceGraphStore::node_tag(uint32_t node_id) const {
    return node_present(node_id) ? meta_arrays()->tags[node_id] : INVALID_TAG;
}


bool InPlaceGraphStore::mark_tag_deleted(TagType tag) {
    if (tag == INVALID_TAG) return false;
    std::shared_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
    return _deleted_tags_current.emplace(tag).second;
}

bool InPlaceGraphStore::is_tag_deleted(TagType tag) const {
    if (tag == INVALID_TAG) return false;
    std::shared_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
    return _deleted_tags_current.contains(tag) ||
           _deleted_tags_draining.contains(tag);
}

size_t InPlaceGraphStore::deleted_tag_count() const {
    std::shared_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
    return _deleted_tags_current.size() + _deleted_tags_draining.size();
}

std::vector<TagType> InPlaceGraphStore::deleted_tags_snapshot() const {
    std::shared_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
    std::vector<TagType> result;
    result.reserve(_deleted_tags_current.size() + _deleted_tags_draining.size());
    for (TagType tag : _deleted_tags_current) result.push_back(tag);
    for (TagType tag : _deleted_tags_draining) result.push_back(tag);
    return result;
}

void InPlaceGraphStore::restore_deleted_tags(const std::vector<TagType>& tags) {
    for (TagType tag : tags) mark_tag_deleted(tag);
}

InPlaceGraphStore::DeletedNodeBatch
InPlaceGraphStore::drain_deleted_tags_to_delete_pending() {
    auto elapsed_s = [](const std::chrono::steady_clock::time_point& begin) {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - begin).count();
    };

    size_t drained_tag_count = 0;
    {
        std::unique_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
        if (!_deleted_tags_current.empty()) {
            _deleted_tags_draining.clear();
            _deleted_tags_current.swap(_deleted_tags_draining);
            drained_tag_count = _deleted_tags_draining.size();
        }
    }

    const auto resolution_begin = std::chrono::steady_clock::now();
    MetaArrays* meta = meta_arrays();
    const size_t span = std::min<size_t>(
        _id_high_water.load(std::memory_order_acquire), meta->capacity);
    std::cout << "[BufANN] cleanup_deleted_edges(): started tag resolution"
              << " deleted_tags=" << drained_tag_count
              << " node_span=" << span << std::endl;

    DeletedNodeBatch result;
    result.repair_required.reserve(drained_tag_count);
    if (drained_tag_count != 0) {
        // As in resolve_deleted_tags_lazy: the draining set takes a lock on every lookup, so the scan reads a plain
        // copy of it, on 8 threads; sorting keeps the ids in ascending order, as a serial scan would give them.
        const phmap::flat_hash_set<TagType> deleted(_deleted_tags_draining.begin(), _deleted_tags_draining.end());
        constexpr int kThreads = 8;
        std::vector<std::vector<uint32_t>> found(kThreads);
#pragma omp parallel num_threads(kThreads)
        {
            std::vector<uint32_t>& mine = found[omp_get_thread_num()];
#pragma omp for schedule(static, 1 << 16)
            for (int64_t nid = 0; nid < static_cast<int64_t>(span); ++nid) {
                const uint32_t node_id = static_cast<uint32_t>(nid);
                if (!node_present(node_id)) continue;
                if (meta->states[node_id].load(std::memory_order_acquire) != NodeState::Active) continue;
                if (!deleted.contains(meta->tags[node_id])) continue;
                mine.push_back(node_id);
            }
        }
        for (auto& part : found) result.repair_required.insert(result.repair_required.end(), part.begin(), part.end());
        std::sort(result.repair_required.begin(), result.repair_required.end());
    }

    std::cout << "[BufANN] cleanup_deleted_edges(): finished tag resolution"
              << " elapsed_s=" << elapsed_s(resolution_begin)
              << " resolved_ids="
              << result.repair_required.size()
              << std::endl;

    const auto tombstone_begin = std::chrono::steady_clock::now();
    std::cout << "[BufANN] cleanup_deleted_edges(): started tombstone marking"
              << " resolved_ids="
              << result.repair_required.size()
              << std::endl;

    auto transition = [&](std::vector<uint32_t>& ids) {
        size_t write = 0;
        for (uint32_t node_id : ids) {
            if (transition_active_to_delete_pending(node_id)) {
                ids[write++] = node_id;
            }
        }
        ids.resize(write);
    };
    transition(result.repair_required);

    std::cout << "[BufANN] cleanup_deleted_edges(): finished tombstone marking"
              << " elapsed_s=" << elapsed_s(tombstone_begin)
              << " marked_ids="
              << result.repair_required.size()
              << std::endl;

    if (drained_tag_count != 0) {
        std::unique_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
        _deleted_tags_draining.clear();
    }
    return result;
}

std::vector<uint32_t> InPlaceGraphStore::resolve_deleted_tags_lazy(uint32_t num_threads) {
    {
        std::unique_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
        if (_deleted_tags_current.empty()) return {};
        _deleted_tags_draining.clear();
        _deleted_tags_current.swap(_deleted_tags_draining);
    }
    const auto begin = std::chrono::steady_clock::now();
    MetaArrays* meta = meta_arrays();
    const int64_t span = static_cast<int64_t>(std::min<size_t>(_id_high_water.load(std::memory_order_acquire),
                                                               meta->capacity));
    // The draining set takes a lock on every lookup, so the scan reads a plain copy of it.
    const phmap::flat_hash_set<TagType> deleted(_deleted_tags_draining.begin(), _deleted_tags_draining.end());
    const int threads = static_cast<int>(std::max<uint32_t>(1, num_threads));
    std::vector<std::vector<uint32_t>> found(threads);
#pragma omp parallel num_threads(threads)
    {
        std::vector<uint32_t>& mine = found[omp_get_thread_num()];
#pragma omp for schedule(static, 1 << 16)
        for (int64_t nid = 0; nid < span; ++nid) {
            const uint32_t node_id = static_cast<uint32_t>(nid);
            if (meta->states[node_id].load(std::memory_order_acquire) != NodeState::Active) continue;
            if (!deleted.contains(meta->tags[node_id])) continue;
            mine.push_back(node_id);
        }
    }
    // As in drain_deleted_tags_to_delete_pending, the nodes found become DeletePending on one thread afterwards: each
    // transition takes the delete-pending list's lock, which the scan's threads would otherwise contend on.
    std::vector<uint32_t> ids;
    for (auto& part : found) {
        for (const uint32_t node_id : part) {
            if (transition_active_to_delete_pending(node_id)) ids.push_back(node_id);
        }
    }
    {
        std::unique_lock<std::shared_mutex> lk(_deleted_tags_swap_mtx);
        _deleted_tags_draining.clear();
    }
    std::cout << "[BufANN] lazy deletion: resolved " << ids.size() << " deleted tags over " << span << " nodes in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() << " s with "
              << threads << " threads" << std::endl;
    return ids;
}

uint32_t InPlaceGraphStore::get_page_id(uint32_t node_id) const {
    return node_present(node_id) ? node_page(node_id) : INVALID_PAGE;
}

uint32_t InPlaceGraphStore::num_active() const {
    return _num_active.load(std::memory_order_relaxed);
}

size_t InPlaceGraphStore::meta_capacity() const {
    return _meta.capacity();
}

double InPlaceGraphStore::fragmentation_ratio() const {
    const uint64_t used_pages =
        _ever_used_pages.load(std::memory_order_relaxed);
    const uint64_t occupied =
        _occupied_slots.load(std::memory_order_relaxed);
    const uint64_t slots_per_page = _slots_per_page;
    if (used_pages == 0 || slots_per_page == 0) return 0.0;
    const uint64_t packed_pages =
        (occupied + slots_per_page - 1) / slots_per_page;
    const uint64_t reclaimable_pages =
        used_pages > packed_pages ? used_pages - packed_pages : 0;
    return static_cast<double>(reclaimable_pages) /
           static_cast<double>(used_pages);
}


// PQ support
void InPlaceGraphStore::load_pq_codes_from_disk_index(const std::string& pq_prefix,
                                                       uint32_t num_chunks) {
    std::string codes_path  = pq_prefix + "_pq_compressed.bin";

    if (_n_chunks != 0 && _n_chunks != num_chunks) {
        throw ANNException("PQ chunk mismatch while reloading: requested " +
                               std::to_string(num_chunks) + ", existing " +
                               std::to_string(_n_chunks),
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    _n_chunks = num_chunks;
    ensure_pq_storage();
    clear_pending_pq_codes();

    int fd = open(codes_path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw ANNException("Unable to open PQ code file: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw ANNException("Unable to stat PQ code file: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (st.st_size < static_cast<off_t>(2 * sizeof(int32_t))) {
        close(fd);
        throw ANNException("Invalid PQ code file header: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    void* mapping = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                         MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        throw ANNException("Unable to mmap PQ code file: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    const uint8_t* mapped_bytes = static_cast<const uint8_t*>(mapping);
    int32_t nr_i32 = 0, nc_i32 = 0;
    memcpy(&nr_i32, mapped_bytes, sizeof(int32_t));
    memcpy(&nc_i32, mapped_bytes + sizeof(int32_t), sizeof(int32_t));
    if (nr_i32 < 0 || nc_i32 < 0) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("Invalid PQ code file header: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    const size_t nr = static_cast<size_t>(nr_i32);
    const size_t nc = static_cast<size_t>(nc_i32);
    diskann::cout << "Reading PQ codes " << codes_path << " ..." << std::endl
                  << "Metadata: #pts = " << nr << ", #dims = " << nc
                  << "..." << std::endl;
    if (nc != num_chunks) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("PQ code chunk mismatch while loading " + codes_path,
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (nr > 0 &&
        static_cast<size_t>(nr) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(_n_chunks)) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("PQ code table too large to stage in memory", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    const size_t payload_bytes = nr * static_cast<size_t>(_n_chunks);
    const size_t payload_offset = 2 * sizeof(int32_t);
    if (payload_bytes >
        std::numeric_limits<size_t>::max() - payload_offset) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("PQ code table too large to map", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (static_cast<uint64_t>(st.st_size) <
        static_cast<uint64_t>(payload_offset + payload_bytes)) {
        munmap(mapping, static_cast<size_t>(st.st_size));
        throw ANNException("PQ code file is truncated: " + codes_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    _pending_pq_code_count = static_cast<size_t>(nr);
    _pending_pq_mmap_base = mapping;
    _pending_pq_mmap_bytes = static_cast<size_t>(st.st_size);
    _pending_pq_codes = mapped_bytes + payload_offset;

    diskann::cout << "InPlace: loaded PQ codes, n=" << nr
                  << ", n_chunks=" << _n_chunks << std::endl;
}



template<typename T>
void InPlaceGraphStore::compute_pq_dists_src(uint32_t src,
                                             const unsigned* ids,
                                             float* dists_out,
                                             uint32_t count,
                                             uint8_t* scratch,
                                             FixedChunkPQTable<T>& pq_table) {
    if (_n_chunks == 0) return;
    const uint8_t* src_ptr = pq_codes_for_node(src);
    if (src_ptr == nullptr) return;
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t* code = pq_codes_for_node(ids[i]);
        if (code != nullptr) {
            memcpy(scratch + i * _n_chunks, code, _n_chunks);
        } else {
            memset(scratch + i * _n_chunks, 0, _n_chunks);
        }
    }
    pq_table.compute_distances(src_ptr, scratch, dists_out, count);
}

template<typename T>
void InPlaceGraphStore::encode_pq(uint32_t node_id, const T* coords,
                                  FixedChunkPQTable<T>& pq_table) {
    if (_n_chunks == 0) return;
    // Callers that may hold a meta writer ticket must ensure PQ storage
    // before enter_meta_write. Growing here would deadlock on writers==0.
    MetaArrays* meta = meta_arrays();
    if (meta == nullptr || meta->pq_codes == nullptr ||
        meta->n_chunks != _n_chunks) {
        throw ANNException(
            "encode_pq: PQ storage not initialized (call ensure_pq_storage "
            "before enter_meta_write)",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    uint8_t* codes = meta->pq_codes.get() +
                     static_cast<size_t>(node_id) * _n_chunks;
    // No release store is needed here: publish_node's release store makes the code visible to any reader that sees
    // the node Active.
    pq_table.deflate_vec(coords, codes);
}

void InPlaceGraphStore::gather_pq_codes(
    const unsigned* ids, uint64_t n, uint8_t* out) const {
    if (_n_chunks == 0) return;
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t* code = pq_codes_for_node(ids[i]);
        if (code != nullptr) {
            memcpy(out + i * _n_chunks, code, _n_chunks);
        } else {
            memset(out + i * _n_chunks, 0, _n_chunks);
        }
    }
}

uint32_t InPlaceGraphStore::dirty_page_count() const {
    return _heap.dirty_page_count();
}

template<typename T, typename TagT>
void InPlaceGraphStore::bulk_load_from_index(diskann::Index<T, TagT>& mem_index,
                                              uint32_t n) {
    const auto* graph = mem_index.get_graph();
    const T*    data  = mem_index.data_by_location();

    for (uint32_t i = 0; i < n; i++) {
        allocate_node(i);
        PageGuard frame = pin_page_for_node(i, WRITE);
        if (!frame.valid()) continue;
        auto guard = write_guard(frame);
        MutableNodeRef node = guard.write_node(i);
        if (!node.valid()) continue;
        node.set_coords<T>(data + static_cast<size_t>(_aligned_dim) * i);
        const auto& nbrs = (*graph)[i];
        node.set_neighbors(nbrs.data(), nbrs.size());
        publish_node(i);
    }
    diskann::cout << "InPlace: bulk loaded " << n << " nodes" << std::endl;
}

// explicit instantiations
template void InPlaceGraphStore::bulk_load_from_index<float, uint32_t>(
    diskann::Index<float, uint32_t>&, uint32_t);
template void InPlaceGraphStore::bulk_load_from_index<uint8_t, uint32_t>(
    diskann::Index<uint8_t, uint32_t>&, uint32_t);
template void InPlaceGraphStore::bulk_load_from_index<int8_t, uint32_t>(
    diskann::Index<int8_t, uint32_t>&, uint32_t);

void InPlaceGraphStore::warmup_bfs(uint32_t entry_point, uint32_t num_nodes) {
    if (num_nodes == 0) return;
    tsl::robin_set<uint32_t> visited;
    std::queue<uint32_t> q;
    q.push(entry_point);
    visited.insert(entry_point);
    uint32_t count = 0;
    while (!q.empty() && count < num_nodes) {
        uint32_t cur = q.front();
        q.pop();
        PageGuard frame = pin_page_for_node(cur, READ);
        if (!frame.valid()) continue;
        auto guard = read_guard(frame);
        ConstNodeRef node = guard.read_node(cur);
        if (!node.valid()) continue;
        const uint32_t* nbrs = node.neighbors();
        for (uint16_t i = 0; i < node.degree(); i++) {
            uint32_t nbr = nbrs[i];
            if (visited.find(nbr) == visited.end()) {
                visited.insert(nbr);
                q.push(nbr);
            }
        }
        count++;
    }
    diskann::cout << "InPlace: warmed up " << count << " nodes" << std::endl;
}

void InPlaceGraphStore::warmup_bfs_pages(uint32_t entry_point, uint32_t num_pages,
                                         uint32_t num_threads) {
    if (num_pages == 0) return;
    if (!node_present(entry_point)) return;
    const int threads = static_cast<int>(std::max<uint32_t>(1, num_threads));

    // True node-based BFS from the entry point: the frontier holds graph
    // node IDs, and pages enter the buffer pool as a side effect of pinning
    // each node's host page. visited_nodes dedups the BFS expansion;
    // visited_pages tracks the actual buffer-pool side-effect and serves as
    // the termination budget.
    tsl::robin_set<uint32_t> visited_nodes;
    tsl::robin_set<uint32_t> visited_pages;
    visited_nodes.insert(entry_point);
    std::vector<uint32_t> current_frontier{entry_point};
    std::mt19937 urng{std::random_device{}()};
    uint32_t hop = 0;

    while (!current_frontier.empty() && visited_pages.size() < num_pages) {
        std::shuffle(current_frontier.begin(), current_frontier.end(), urng);

        const uint32_t remaining =
            static_cast<uint32_t>(num_pages - visited_pages.size());
        // When the frontier is larger than the remaining page budget, this is
        // the last hop: parallel-pin only the first `remaining` nodes (since
        // node->page is many-to-one, this pins <= remaining novel pages —
        // typically an undershoot), then sequentially tail-pin the rest of
        // the frontier until the budget is hit. We don't build a next
        // frontier in this case.
        const bool last_hop = (current_frontier.size() > remaining);
        const size_t parallel_count =
            last_hop ? remaining : current_frontier.size();

        // Per-thread accumulators; merged sequentially after the parallel
        // section to avoid locks in the hot loop.
        std::vector<std::vector<uint32_t>> tls_pages(threads);
        std::vector<std::vector<uint32_t>> tls_nbrs(threads);
        const int64_t n = static_cast<int64_t>(parallel_count);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 8)
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t node_id = current_frontier[i];
            PageGuard frame = pin_page_for_node(node_id, READ);
            if (!frame.valid()) continue;
            const int tid = omp_get_thread_num();
            tls_pages[tid].push_back(node_page(node_id));
            if (last_hop) continue;  // skip neighbor scan on last hop
            auto guard = read_guard(frame);
            ConstNodeRef node = guard.read_node(node_id);
            if (!node.valid()) continue;
            const uint32_t* nbrs = node.neighbors();
            const uint16_t deg = node.degree();
            auto& local = tls_nbrs[tid];
            for (uint16_t k = 0; k < deg; ++k) {
                local.push_back(nbrs[k]);
            }
        }

        // Merge pinned pages from the parallel section.
        for (auto& local : tls_pages) {
            for (uint32_t pid : local) {
                visited_pages.insert(pid);
            }
        }

        // Last-hop tail pin: parallel top-up to the page budget. Atomic
        // flag for early exit once full; mutex on the shared visited_pages.
        uint32_t tail_pinned = 0;
        if (last_hop && visited_pages.size() < num_pages) {
            std::mutex pages_mtx;
            std::atomic<bool> budget_full{false};
            std::atomic<uint32_t> tail_pinned_atomic{0};
            const int64_t tail_start = static_cast<int64_t>(parallel_count);
            const int64_t tail_end = static_cast<int64_t>(current_frontier.size());
#pragma omp parallel for num_threads(threads) schedule(dynamic, 8)
            for (int64_t i = tail_start; i < tail_end; ++i) {
                if (budget_full.load(std::memory_order_relaxed)) continue;
                PageGuard frame = pin_page_for_node(current_frontier[i], READ);
                if (!frame.valid()) continue;
                std::lock_guard<std::mutex> lk(pages_mtx);
                if (visited_pages.size() >= num_pages) {
                    budget_full.store(true, std::memory_order_relaxed);
                    continue;
                }
                if (visited_pages.insert(node_page(current_frontier[i])).second) {
                    tail_pinned_atomic.fetch_add(1, std::memory_order_relaxed);
                    if (visited_pages.size() >= num_pages) {
                        budget_full.store(true, std::memory_order_relaxed);
                    }
                }
            }
            tail_pinned = tail_pinned_atomic.load(std::memory_order_relaxed);
        }

        // Build the next frontier by deduping collected neighbors against
        // visited_nodes. Skipped on last hop.
        std::vector<uint32_t> next_frontier;
        if (!last_hop) {
            for (auto& local : tls_nbrs) {
                for (uint32_t nbr : local) {
                    if (visited_nodes.insert(nbr).second) {
                        next_frontier.push_back(nbr);
                    }
                }
            }
        }

        diskann::cout << "InPlace BFS warmup: hop=" << hop
                      << " nodes=" << current_frontier.size()
                      << " parallel=" << parallel_count
                      << " tail=" << tail_pinned
                      << " pages=" << visited_pages.size()
                      << " next_frontier=" << next_frontier.size() << std::endl;
        ++hop;
        current_frontier.swap(next_frontier);
    }
    diskann::cout << "InPlace: BFS warmed up " << visited_pages.size()
                  << " pages over " << hop << " hops" << std::endl;
}

void InPlaceGraphStore::reset_stats() {
    drain_query_hot_stats(_stats);
    _stats.reset();
}

InPlaceIOStats& InPlaceGraphStore::stats() {
    drain_query_hot_stats(_stats);
    return _stats;
}

void InPlaceGraphStore::preload_pages(uint32_t max_pages,
                                      uint32_t num_threads) {
    const uint32_t total = total_pages();
    const uint32_t limit = max_pages == 0 ? total : std::min(total, max_pages);
    if (limit == 0) return;

    const int threads = static_cast<int>(std::max<uint32_t>(1, num_threads));
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t page_id = 0; page_id < static_cast<int64_t>(limit); ++page_id) {
        PageGuard frame = _heap.pin(graph_page(static_cast<uint32_t>(page_id)), AccessMode::READ);
    }
}

void InPlaceGraphStore::start_bg_flush(float high_wm, float low_wm) { _heap.start_bg_flush(high_wm, low_wm); }
void InPlaceGraphStore::stop_bg_flush()  { _heap.stop_bg_flush(); }
uint32_t InPlaceGraphStore::flush_dirty_budget(uint32_t max_pages) {
    return _heap.flush_dirty_budget(max_pages);
}

double InPlaceGraphStore::dirty_ratio() const { return _heap.dirty_ratio(); }

void InPlaceGraphStore::flush() {
    _heap.flush_all_dirty();
}
void InPlaceGraphStore::set_entry_point(uint32_t entry_point) {
    if (!is_active(entry_point)) return;
    std::lock_guard<std::mutex> lk(_entry_mtx);
    _entry_pool[0] = entry_point;
}

void InPlaceGraphStore::save_snapshot(const std::string& meta_path) {
    struct SnapshotHeader {
        char magic[8];
        uint32_t version;
        uint32_t dim;
        uint32_t aligned_dim;
        uint32_t Mmax;
        uint32_t elem_size;
        uint32_t page_size;
        uint32_t slot_size;
        uint32_t slots_per_page;
        uint32_t _reserved;     // unused, was entry_point
        uint32_t max_nodes;     // unused
        uint32_t n_chunks;
        uint32_t num_active;
        uint32_t total_pages;
        uint64_t rid_size;
        uint64_t active_cap;
        uint64_t page_dir_size;
        uint64_t pages_with_space_size;
        uint64_t reservoir_cursor;
        uint64_t _reserved2;    // v5+: ever-used page bitmap size
    } hdr{};

    const std::vector<TagType> pending_deleted_tags =
        deleted_tags_snapshot();

    // IPGSNP7 only when an empty page span is set. v7 trailer:
    // free_span_begin/end, empty_page_span_begin/end, free_count, free_ids[].
    // The free-ID span and list are always written empty, because free IDs are now just the Free slots.
    const uint32_t free_span_begin = 0;
    const uint32_t free_span_end = 0;
    uint32_t empty_page_span_begin = 0;
    uint32_t empty_page_span_end = 0;
    {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        empty_page_span_begin = _empty_page_span_begin;
        empty_page_span_end = _empty_page_span_end;
    }
    const bool write_v7 = empty_page_span_begin < empty_page_span_end;
    if (write_v7) {
        memcpy(hdr.magic, "IPGSNP7", 8);
        hdr.version = 7;
    } else {
        memcpy(hdr.magic, "IPGSNP6", 8);
        hdr.version = 6;
    }
    hdr.dim = _dim;
    hdr.aligned_dim = _aligned_dim;
    hdr.Mmax = _Mmax;
    hdr.elem_size = _elem_size;
    hdr.page_size = _page_size;
    hdr.slot_size = _slot_size;
    hdr.slots_per_page = _slots_per_page;
    hdr.n_chunks = _n_chunks;
    hdr.num_active = _num_active.load(std::memory_order_relaxed);
    hdr.total_pages = _heap.total_pages(_graph_file_no);
    hdr.rid_size = 0;
    MetaArrays* meta = meta_arrays();
    hdr.active_cap = std::min<uint64_t>(
        _id_high_water.load(std::memory_order_acquire), meta->capacity);
    // save_snapshot is called at quiesce, so no synchronization is needed.
    for (size_t nid = 0; nid < hdr.active_cap; ++nid) {
        if (node_present(static_cast<uint32_t>(nid)))
            ++hdr.rid_size;
    }
    hdr.page_dir_size = _page_dir.size();
    hdr.pages_with_space_size = _pages_with_space.size();
    {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        hdr.reservoir_cursor = static_cast<uint64_t>(_reservoir_cursor);
    }
    hdr._reserved2 = _page_ever_used.size();

    std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw ANNException("Failed to open snapshot file: " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    // Write the present nodes as (node_id, page, slot) records.
    constexpr size_t kRidChunk = size_t{1} << 20;
    std::vector<SerializedRID> rid_chunk;
    rid_chunk.reserve(kRidChunk);
    for (size_t nid = 0; nid < hdr.active_cap; ++nid) {
        if (!node_present(static_cast<uint32_t>(nid)) ||
            nid == INVALID_NODE) continue;
        const uint32_t node_id = static_cast<uint32_t>(nid);
        rid_chunk.push_back(SerializedRID{node_id, node_page(node_id), node_slot(node_id), {0, 0}});
        if (rid_chunk.size() == kRidChunk) {
            out.write(reinterpret_cast<const char*>(rid_chunk.data()),
                      static_cast<std::streamsize>(
                          rid_chunk.size() * sizeof(SerializedRID)));
            rid_chunk.clear();
        }
    }
    if (!rid_chunk.empty()) {
        out.write(reinterpret_cast<const char*>(rid_chunk.data()),
                  static_cast<std::streamsize>(
                      rid_chunk.size() * sizeof(SerializedRID)));
    }
    constexpr size_t kSnapshotChunk = size_t{1} << 20;
    std::vector<uint8_t> active_flags(
        std::min<size_t>(kSnapshotChunk, hdr.active_cap));
    for (size_t begin = 0; begin < hdr.active_cap;
         begin += kSnapshotChunk) {
        const size_t count = std::min<size_t>(
            kSnapshotChunk, hdr.active_cap - begin);
        std::fill(active_flags.begin(),
                  active_flags.begin() + count, 0);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t nid = begin + offset;
            if (!node_present(static_cast<uint32_t>(nid)) ||
                nid == INVALID_NODE) continue;
            active_flags[offset] =
                meta->states[nid].load(std::memory_order_relaxed) ==
                    NodeState::Active ? 1 : 0;
        }
        out.write(reinterpret_cast<const char*>(active_flags.data()),
                  static_cast<std::streamsize>(count));
    }
    std::vector<TagType> tags(
        std::min<size_t>(kSnapshotChunk, hdr.active_cap));
    for (size_t begin = 0; begin < hdr.active_cap;
         begin += kSnapshotChunk) {
        const size_t count = std::min<size_t>(
            kSnapshotChunk, hdr.active_cap - begin);
        std::fill(tags.begin(), tags.begin() + count, INVALID_TAG);
        for (size_t offset = 0; offset < count; ++offset) {
            const size_t nid = begin + offset;
            if (!node_present(static_cast<uint32_t>(nid)) ||
                nid == INVALID_NODE) continue;
            tags[offset] = meta->tags[nid];
        }
        out.write(reinterpret_cast<const char*>(tags.data()),
                  static_cast<std::streamsize>(
                      count * sizeof(TagType)));
    }
    out.write(reinterpret_cast<const char*>(_page_dir.data()),
              static_cast<std::streamsize>(_page_dir.size() * sizeof(PageDir)));
    out.write(reinterpret_cast<const char*>(_page_ever_used.data()),
              static_cast<std::streamsize>(_page_ever_used.size()));
    for (uint32_t pid : _pages_with_space) {
        out.write(reinterpret_cast<const char*>(&pid), sizeof(pid));
    }
    // Entry pool and reservoir (fixed-size arrays).
    {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        out.write(reinterpret_cast<const char*>(_entry_pool.data()),
                  static_cast<std::streamsize>(kEntryPoolSize * sizeof(uint32_t)));
        out.write(reinterpret_cast<const char*>(_node_reservoir.data()),
                  static_cast<std::streamsize>(kReservoirSize * sizeof(uint32_t)));
    }
    // Preserve the v6+ tombstone record layout; the legacy flag is reserved.
    const uint64_t pending_count = pending_deleted_tags.size();
    out.write(reinterpret_cast<const char*>(&pending_count),
              sizeof(pending_count));
    for (const auto& item : pending_deleted_tags) {
        const uint8_t reserved = 0;
        out.write(reinterpret_cast<const char*>(&item),
                  sizeof(item));
        out.write(reinterpret_cast<const char*>(&reserved),
                  sizeof(reserved));
    }
    // v7 reusable section: free/page spans + optional small freelist.
    if (write_v7) {
        if (free_span_end < free_span_begin ||
            empty_page_span_end < empty_page_span_begin ||
            free_span_end > hdr.active_cap ||
            empty_page_span_end > hdr.total_pages) {
            throw ANNException("save_snapshot: invalid free/page span", -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        out.write(reinterpret_cast<const char*>(&free_span_begin),
                  sizeof(free_span_begin));
        out.write(reinterpret_cast<const char*>(&free_span_end),
                  sizeof(free_span_end));
        out.write(reinterpret_cast<const char*>(&empty_page_span_begin),
                  sizeof(empty_page_span_begin));
        out.write(reinterpret_cast<const char*>(&empty_page_span_end),
                  sizeof(empty_page_span_end));
        const uint64_t free_count = 0;
        out.write(reinterpret_cast<const char*>(&free_count),
                  sizeof(free_count));
    }
    clear_pending_pq_codes();
}

void InPlaceGraphStore::save_dense_pq_codes(
    const std::string& pq_prefix) const {
    MetaArrays* meta = meta_arrays();
    if (_n_chunks == 0 || meta == nullptr || meta->pq_codes == nullptr) return;
    const uint32_t count = _id_high_water.load(std::memory_order_acquire);
    std::ofstream out(pq_prefix + "_pq_compressed.bin",
                      std::ios::binary | std::ios::trunc);
    if (!out) {
        throw ANNException("save_dense_pq_codes: cannot create output", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&_n_chunks), sizeof(_n_chunks));
    const size_t total_bytes =
        static_cast<size_t>(count) * _n_chunks;
    constexpr size_t kWriteChunk = size_t{64} << 20;
    for (size_t offset = 0; offset < total_bytes;
         offset += kWriteChunk) {
        const size_t bytes =
            std::min(kWriteChunk, total_bytes - offset);
        out.write(
            reinterpret_cast<const char*>(meta->pq_codes.get() + offset),
            static_cast<std::streamsize>(bytes));
    }
    if (!out) {
        throw ANNException("save_dense_pq_codes: write failed", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
}

void InPlaceGraphStore::load_snapshot(const std::string& meta_path) {
    struct SnapshotHeader {
        char magic[8];
        uint32_t version;
        uint32_t dim;
        uint32_t aligned_dim;
        uint32_t Mmax;
        uint32_t elem_size;
        uint32_t page_size;
        uint32_t slot_size;
        uint32_t slots_per_page;
        uint32_t _reserved;
        uint32_t max_nodes;
        uint32_t n_chunks;
        uint32_t num_active;
        uint32_t total_pages;
        uint64_t rid_size;
        uint64_t active_cap;
        uint64_t page_dir_size;
        uint64_t pages_with_space_size;
        uint64_t reservoir_cursor;
        uint64_t _reserved2;
    } hdr{};

    std::ifstream in(meta_path, std::ios::binary);
    if (!in) {
        throw ANNException("Failed to open snapshot file: " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    const bool is_v2 = in && memcmp(hdr.magic, "IPGSNP2", 8) == 0 && hdr.version == 2;
    const bool is_v3 = in && memcmp(hdr.magic, "IPGSNP3", 8) == 0 && hdr.version == 3;
    const bool is_v4 = in && memcmp(hdr.magic, "IPGSNP4", 8) == 0 && hdr.version == 4;
    const bool is_v5 = in && memcmp(hdr.magic, "IPGSNP5", 8) == 0 && hdr.version == 5;
    const bool is_v6 = in && memcmp(hdr.magic, "IPGSNP6", 8) == 0 && hdr.version == 6;
    const bool is_v7 = in && memcmp(hdr.magic, "IPGSNP7", 8) == 0 && hdr.version == 7;
    if (!in || (!is_v2 && !is_v3 && !is_v4 && !is_v5 && !is_v6 && !is_v7)) {
        throw ANNException("Invalid snapshot header (expected IPGSNP7 v7, IPGSNP6 v6, IPGSNP5 v5, IPGSNP4 v4, IPGSNP3 v3, or IPGSNP2 v2): " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (hdr.dim != _dim || hdr.page_size != _page_size || hdr.Mmax != _Mmax ||
        hdr.elem_size != _elem_size || hdr.slot_size != _slot_size ||
        hdr.slots_per_page != _slots_per_page) {
        std::ostringstream oss;
        oss << "Snapshot layout mismatch: " << meta_path
            << " [snapshot vs runtime]"
            << " dim=" << hdr.dim << "/" << _dim
            << " page_size=" << hdr.page_size << "/" << _page_size
            << " Mmax=" << hdr.Mmax << "/" << _Mmax
            << " elem_size=" << hdr.elem_size << "/" << _elem_size
            << " slot_size=" << hdr.slot_size << "/" << _slot_size
            << " slots_per_page=" << hdr.slots_per_page << "/" << _slots_per_page;
        throw ANNException(oss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    // Keep runtime PQ mode from current configuration/load path.  We do not
    // auto-enable PQ from snapshot metadata so NoPQ stays zero-overhead.
    // NOTE: _num_active is restored *after* reset_metadata() below, because
    // reset_metadata() zeros it and the materialize_node loop does not
    // increment it back. Setting it here gets silently clobbered.
    _heap.set_total_pages(_graph_file_no, hdr.total_pages);
    diskann::cout << "InPlace: start streaming metadata load: active_cap = "
                  << hdr.active_cap << ", rid_entries=" << hdr.rid_size
                  << std::endl;
    reset_metadata(static_cast<size_t>(hdr.active_cap));
    MetaArrays* meta = meta_arrays();
    constexpr size_t kRidChunk = size_t{1} << 20;
    std::vector<SerializedRID> rid_chunk(kRidChunk);
    for (uint64_t begin = 0; begin < hdr.rid_size;
         begin += kRidChunk) {
        const size_t count = static_cast<size_t>(
            std::min<uint64_t>(kRidChunk, hdr.rid_size - begin));
        in.read(reinterpret_cast<char*>(rid_chunk.data()),
                static_cast<std::streamsize>(
                    count * sizeof(SerializedRID)));
        if (!in) {
            throw ANNException("Truncated snapshot RID entries", -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        for (size_t i = 0; i < count; ++i) {
            const SerializedRID& entry = rid_chunk[i];
            if (entry.node_id >= hdr.active_cap ||
                entry.node_id >= meta->capacity) {
                throw ANNException("Invalid snapshot RID entry", -1,
                                   __FUNCSIG__, __FILE__, __LINE__);
            }
            if (entry.page_id == INVALID_PAGE) continue;
            if (entry.page_id != node_page(entry.node_id) ||
                entry.slot_idx != node_slot(entry.node_id)) {
                throw ANNException(
                    "Snapshot node_id " + std::to_string(entry.node_id) +
                        " is not stored at its slot address (page " +
                        std::to_string(entry.page_id) + ", slot " +
                        std::to_string(entry.slot_idx) + ")",
                    -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            materialize_node(entry.node_id);
            // The active-flag pass below turns nodes that were inactive at save time into DeletePending.
            meta->states[entry.node_id].store(NodeState::Active, std::memory_order_relaxed);
        }
    }

    constexpr size_t kSnapshotChunk = size_t{1} << 20;
    std::vector<uint8_t> active_flags(
        std::min<size_t>(kSnapshotChunk, hdr.active_cap));
    size_t inactive_count = 0;
    uint32_t first_inactive = INVALID_NODE;
    for (size_t begin = 0; begin < hdr.active_cap;
         begin += kSnapshotChunk) {
        const size_t count = std::min<size_t>(
            kSnapshotChunk, hdr.active_cap - begin);
        in.read(reinterpret_cast<char*>(active_flags.data()),
                static_cast<std::streamsize>(count));
        if (!in) {
            throw ANNException("Truncated snapshot active flags", -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        for (size_t offset = 0; offset < count; ++offset) {
            const uint32_t node_id =
                static_cast<uint32_t>(begin + offset);
            if (!node_present(node_id)) continue;
            const uint8_t active = active_flags[offset] != 0 ? 1 : 0;
            meta->states[node_id].store(
                active != 0 ? NodeState::Active
                            : NodeState::DeletePending,
                std::memory_order_relaxed);
            if (active == 0) {
                if (first_inactive == INVALID_NODE)
                    first_inactive = node_id;
                ++inactive_count;
            }
        }
    }
    if (inactive_count != 0) {
        std::ostringstream oss;
        oss << "Snapshot contains " << inactive_count
            << " inactive/delete-pending node entries; first node_id="
            << first_inactive
            << ". Loading snapshots with pending deletes is unsupported; "
            << "run maintenance before saving.";
        throw ANNException(oss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (is_v4 || is_v5 || is_v6 || is_v7) {
        std::vector<TagType> tags(
            std::min<size_t>(kSnapshotChunk, hdr.active_cap));
        for (size_t begin = 0; begin < hdr.active_cap;
             begin += kSnapshotChunk) {
            const size_t count = std::min<size_t>(
                kSnapshotChunk, hdr.active_cap - begin);
            in.read(reinterpret_cast<char*>(tags.data()),
                    static_cast<std::streamsize>(
                        count * sizeof(TagType)));
            if (!in) {
                throw ANNException("Truncated snapshot tags", -1,
                                   __FUNCSIG__, __FILE__, __LINE__);
            }
            for (size_t offset = 0; offset < count; ++offset) {
                const uint32_t node_id =
                    static_cast<uint32_t>(begin + offset);
                if (node_present(node_id))
                    meta->tags[node_id] = tags[offset];
            }
        }
    } else {
        for (size_t node_id = 0; node_id < hdr.active_cap; ++node_id) {
            if (node_present(static_cast<uint32_t>(node_id))) {
                meta->tags[node_id] =
                    static_cast<TagType>(node_id);
            }
        }
    }
    diskann::cout << "InPlace: finished streaming metadata load"
                  << std::endl;
    // Restore num_active now that reset_metadata-induced zeroing is past.
    _num_active.store(hdr.num_active, std::memory_order_relaxed);
    clear_pending_pq_codes();
    _page_dir.assign(static_cast<size_t>(hdr.page_dir_size), PageDir{0, 0});
    if (!_page_dir.empty()) {
        in.read(reinterpret_cast<char*>(_page_dir.data()),
                static_cast<std::streamsize>(_page_dir.size() * sizeof(PageDir)));
    }
    if (is_v5 || is_v6 || is_v7) {
        if (hdr._reserved2 != _page_dir.size()) {
            throw ANNException(
                "Invalid snapshot ever-used page bitmap size: " + meta_path,
                -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        _page_ever_used.assign(static_cast<size_t>(hdr._reserved2), 0);
        if (!_page_ever_used.empty()) {
            in.read(reinterpret_cast<char*>(_page_ever_used.data()),
                    static_cast<std::streamsize>(_page_ever_used.size()));
        }
    } else {
        // v2-v4 snapshots predate the explicit ever-used bit. Heap allocation
        // is monotonic, so the occupied prefix is the best safe approximation.
        _page_ever_used.assign(_page_dir.size(), 0);
        size_t highest_occupied_plus_one = 0;
        for (size_t page_id = 0; page_id < _page_dir.size(); ++page_id) {
            if (_page_dir[page_id].num_occupied != 0) {
                highest_occupied_plus_one = page_id + 1;
            }
        }
        std::fill(_page_ever_used.begin(),
                  _page_ever_used.begin() + highest_occupied_plus_one, 1);
    }
    _ever_used_pages.store(
        static_cast<uint64_t>(std::count(
            _page_ever_used.begin(), _page_ever_used.end(), uint8_t{1})),
        std::memory_order_relaxed);
    uint64_t occupied_slots = 0;
    for (const PageDir& page : _page_dir) {
        occupied_slots += page.num_occupied;
    }
    _occupied_slots.store(occupied_slots, std::memory_order_relaxed);
    _pages_with_space.clear();
    for (uint64_t i = 0; i < hdr.pages_with_space_size; ++i) {
        uint32_t pid = 0;
        in.read(reinterpret_cast<char*>(&pid), sizeof(pid));
        _pages_with_space.insert(pid);
    }
    _alloc_page_rr_cursor = 0;

    {
        std::lock_guard<std::mutex> lk(_entry_mtx);
        if (is_v2) {
            // v2 stored: single entry_point in hdr._reserved, then a variable-length
            // _entry_candidates array (count in hdr.reservoir_cursor, which was
            // entry_candidates_count in v2).  Reconstruct the pool and reservoir.
            _entry_pool.fill(INVALID_NODE);
            _node_reservoir.fill(INVALID_NODE);
            _reservoir_cursor = 0;
            _entry_pool[0] = hdr._reserved;  // was entry_point
            uint64_t cand_count = hdr.reservoir_cursor;  // was entry_candidates_count
            for (uint64_t i = 0; i < cand_count; ++i) {
                uint32_t cand = INVALID_NODE;
                in.read(reinterpret_cast<char*>(&cand), sizeof(cand));
                if (i < kReservoirSize) _node_reservoir[i] = cand;
            }
            _reservoir_cursor = std::min<size_t>(static_cast<size_t>(cand_count), kReservoirSize);
        } else {
            _reservoir_cursor = static_cast<size_t>(hdr.reservoir_cursor);
            in.read(reinterpret_cast<char*>(_entry_pool.data()),
                    static_cast<std::streamsize>(kEntryPoolSize * sizeof(uint32_t)));
            in.read(reinterpret_cast<char*>(_node_reservoir.data()),
                    static_cast<std::streamsize>(kReservoirSize * sizeof(uint32_t)));
        }
    }
    if (is_v6 || is_v7) {
        uint64_t pending_count = 0;
        in.read(reinterpret_cast<char*>(&pending_count),
                sizeof(pending_count));
        if (!in || pending_count >
                       static_cast<uint64_t>(
                           std::numeric_limits<size_t>::max())) {
            throw ANNException("Invalid snapshot pending-delete section", -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        std::vector<TagType> pending;
        pending.reserve(static_cast<size_t>(pending_count));
        for (uint64_t i = 0; i < pending_count; ++i) {
            TagType tag = INVALID_TAG;
            uint8_t reserved = 0;
            in.read(reinterpret_cast<char*>(&tag), sizeof(tag));
            in.read(reinterpret_cast<char*>(&reserved),
                    sizeof(reserved));
            if (!in || tag == INVALID_TAG || reserved > 1) {
                throw ANNException(
                    "Invalid snapshot pending-delete entry", -1,
                    __FUNCSIG__, __FILE__, __LINE__);
            }
            pending.push_back(tag);
        }
        restore_deleted_tags(pending);
    }
    if (is_v7) {
        uint32_t free_span_begin = 0;
        uint32_t free_span_end = 0;
        uint32_t empty_page_span_begin = 0;
        uint32_t empty_page_span_end = 0;
        const std::streampos trailer_pos = in.tellg();
        in.read(reinterpret_cast<char*>(&free_span_begin),
                sizeof(free_span_begin));
        in.read(reinterpret_cast<char*>(&free_span_end),
                sizeof(free_span_end));
        in.read(reinterpret_cast<char*>(&empty_page_span_begin),
                sizeof(empty_page_span_begin));
        in.read(reinterpret_cast<char*>(&empty_page_span_end),
                sizeof(empty_page_span_end));
        const bool spanned =
            in && free_span_end >= free_span_begin &&
            empty_page_span_end >= empty_page_span_begin &&
            free_span_end <= hdr.active_cap &&
            empty_page_span_end <= hdr.total_pages &&
            (free_span_begin < free_span_end ||
             empty_page_span_begin < empty_page_span_end ||
             (free_span_begin == 0 && free_span_end == 0 &&
              empty_page_span_begin == 0 && empty_page_span_end == 0));
        if (!spanned) {
            // Trailer is not span-shaped; remainder is free_count + ids.
            in.clear();
            in.seekg(trailer_pos);
            free_span_begin = 0;
            free_span_end = 0;
            empty_page_span_begin = 0;
            empty_page_span_end = 0;
        }
        if (free_span_begin < free_span_end) {
            if (node_present(free_span_begin) ||
                (free_span_end > free_span_begin + 1 &&
                 node_present(free_span_end - 1))) {
                throw ANNException(
                    "Invalid snapshot free-span endpoints are present", -1,
                    __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        uint64_t free_count = 0;
        in.read(reinterpret_cast<char*>(&free_count), sizeof(free_count));
        if (!in || free_count > static_cast<uint64_t>(
                                   std::numeric_limits<size_t>::max()) ||
            free_count > hdr.active_cap) {
            throw ANNException("Invalid snapshot reusable-ID section", -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        std::vector<uint32_t> free_ids(static_cast<size_t>(free_count));
        if (free_count != 0) {
            in.read(reinterpret_cast<char*>(free_ids.data()),
                    static_cast<std::streamsize>(free_count * sizeof(uint32_t)));
        }
        std::sort(free_ids.begin(), free_ids.end());
        for (size_t i = 0; i < free_ids.size(); ++i) {
            if (!in || free_ids[i] >= hdr.active_cap ||
                node_present(free_ids[i]) ||
                (free_ids[i] >= free_span_begin &&
                 free_ids[i] < free_span_end) ||
                (i != 0 && free_ids[i - 1] == free_ids[i])) {
                throw ANNException("Invalid snapshot reusable-ID entry", -1,
                                   __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        // Nothing to restore for free IDs: their slots are already Free.
        {
            std::lock_guard<std::mutex> lk(_pages_mtx);
            _empty_page_span_begin = empty_page_span_begin;
            _empty_page_span_end = empty_page_span_end;
        }
    }
    if (is_v7) {
        // v7 active_cap is the logical HWM and may exceed max(present)+1.
        _id_high_water.store(static_cast<uint32_t>(hdr.active_cap),
                                std::memory_order_release);
    }
    if (is_v2) seed_entry_pool_from_reservoir();
}

size_t InPlaceGraphStore::memory_usage_bytes() const {
    size_t rss = 0;
    std::ifstream f("/proc/self/statm");
    if (f.is_open()) {
        size_t dummy;
        f >> dummy >> rss;
        rss *= sysconf(_SC_PAGESIZE);
    }
    return rss;
}

size_t InPlaceGraphStore::buffer_pool_bytes() const {
    return _heap.configured_bytes();
}





size_t InPlaceGraphStore::locator_bytes() const {
    MetaArrays* meta = meta_arrays();
    return meta == nullptr ? 0 :
        meta->capacity * (sizeof(TagType) + sizeof(std::atomic<NodeState>));
}

size_t InPlaceGraphStore::pq_data_bytes() const {
    if (_n_chunks == 0) return 0;

    MetaArrays* meta = meta_arrays();
    const size_t attached = meta == nullptr || meta->pq_codes == nullptr ? 0 :
        meta->capacity * static_cast<size_t>(_n_chunks);
    size_t staged = (_pending_pq_codes == nullptr) ? 0 :
                    _pending_pq_code_count * static_cast<size_t>(_n_chunks);
    return attached + staged;
}

template void InPlaceGraphStore::encode_pq<float>(
    uint32_t, const float*, FixedChunkPQTable<float>&);
template void InPlaceGraphStore::encode_pq<uint8_t>(
    uint32_t, const uint8_t*, FixedChunkPQTable<uint8_t>&);
template void InPlaceGraphStore::encode_pq<int8_t>(
    uint32_t, const int8_t*, FixedChunkPQTable<int8_t>&);
template void InPlaceGraphStore::compute_pq_dists_src<float>(
    uint32_t, const unsigned*, float*, uint32_t, uint8_t*,
    FixedChunkPQTable<float>&);
template void InPlaceGraphStore::compute_pq_dists_src<uint8_t>(
    uint32_t, const unsigned*, float*, uint32_t, uint8_t*,
    FixedChunkPQTable<uint8_t>&);
template void InPlaceGraphStore::compute_pq_dists_src<int8_t>(
    uint32_t, const unsigned*, float*, uint32_t, uint8_t*,
    FixedChunkPQTable<int8_t>&);

}
}
