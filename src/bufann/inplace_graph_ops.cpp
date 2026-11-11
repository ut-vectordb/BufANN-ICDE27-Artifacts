#include "bufann/inplace_graph_ops.h"
#include "bufann/inplace_backend.h"
#include "bufann/lazy_delete.h"
#include "logger.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <immintrin.h>

#include "tsl/robin_map.h"

namespace diskann {
namespace inplace {

namespace {

struct MaintenanceScratchCache {
    std::unique_ptr<InPlaceSearchScratch> scratch;
    uint32_t aligned_dim = 0;
    uint32_t n_chunks = 0;
    uint32_t elem_size = 0;
    uint32_t list_size = 0;
    uint32_t max_degree = 0;
    uint32_t beamwidth = 0;
};

InPlaceSearchScratch& maintenance_scratch_for(uint32_t aligned_dim,
                                              uint32_t n_chunks,
                                              uint32_t elem_size,
                                              uint32_t list_size,
                                              uint32_t max_degree,
                                              uint32_t beamwidth) {
    thread_local MaintenanceScratchCache cache;
    const bool needs_init =
        !cache.scratch || cache.aligned_dim != aligned_dim ||
        cache.n_chunks != n_chunks || cache.elem_size != elem_size ||
        cache.list_size != list_size || cache.max_degree != max_degree ||
        cache.beamwidth != beamwidth;
    if (needs_init) {
        cache.scratch.reset(new InPlaceSearchScratch());
        cache.scratch->init(aligned_dim, n_chunks, elem_size, list_size,
                            max_degree, beamwidth);
        cache.aligned_dim = aligned_dim;
        cache.n_chunks = n_chunks;
        cache.elem_size = elem_size;
        cache.list_size = list_size;
        cache.max_degree = max_degree;
        cache.beamwidth = beamwidth;
    } else {
        cache.scratch->reset();
    }
    return *cache.scratch;
}

static inline uint32_t page_bitmap_bytes(uint32_t slots_per_page) {
    return (slots_per_page + 7) / 8;
}

static inline uint32_t page_header_bytes(uint32_t slots_per_page) {
    return 8 + page_bitmap_bytes(slots_per_page);
}

struct PhaseIOStats {
    std::atomic<uint64_t> page_accesses{0};
    std::atomic<uint64_t> page_misses{0};
};

struct InsertIOStats {
    std::atomic<uint64_t> n_inserts{0};
    PhaseIOStats phases[InsertIOPhaseScope::NUM_PHASES];
};
InsertIOStats g_insert_io_stats;

struct PhaseIODelta {
    uint64_t page_accesses = 0;
    uint64_t page_misses = 0;
};

static inline PhaseIODelta load_and_reset(PhaseIOStats& phase) {
    return PhaseIODelta{
        phase.page_accesses.exchange(0, std::memory_order_relaxed),
        phase.page_misses.exchange(0, std::memory_order_relaxed)};
}

}  // namespace

InsertIOPhaseScope::InsertIOPhaseScope(Phase phase)
    : _phase(phase),
      _start_accesses(thread_page_access_count()),
      _start_misses(thread_physical_reads_count()) {}

InsertIOPhaseScope::~InsertIOPhaseScope() {
    auto& s = g_insert_io_stats.phases[_phase];
    s.page_accesses.fetch_add(thread_page_access_count() - _start_accesses,
                              std::memory_order_relaxed);
    s.page_misses.fetch_add(thread_physical_reads_count() - _start_misses,
                            std::memory_order_relaxed);
}

void insert_stats_increment_n() {
    g_insert_io_stats.n_inserts.fetch_add(1, std::memory_order_relaxed);
}

void dump_and_reset_insert_phase_stats() {
    uint64_t n = g_insert_io_stats.n_inserts.exchange(0, std::memory_order_relaxed);
    if (n == 0) return;
    double denom = static_cast<double>(n);

    auto print_phase = [denom](const char* label, const PhaseIODelta& io) {
        double miss_rate = io.page_accesses == 0
                               ? 0.0
                               : 100.0 * static_cast<double>(io.page_misses) /
                                     static_cast<double>(io.page_accesses);
        fprintf(stderr,
                "  %-20s accesses=%12llu  misses=%10llu  "
                "accesses/insert=%8.2f  misses/insert=%8.4f  miss_rate=%7.4f%%\n",
                label,
                (unsigned long long)io.page_accesses,
                (unsigned long long)io.page_misses,
                io.page_accesses / denom,
                io.page_misses / denom,
                miss_rate);
    };

    PhaseIODelta beam = load_and_reset(
        g_insert_io_stats.phases[InsertIOPhaseScope::BEAM_SEARCH]);
    PhaseIODelta prune = load_and_reset(
        g_insert_io_stats.phases[InsertIOPhaseScope::PRUNE]);
    PhaseIODelta write_new = load_and_reset(
        g_insert_io_stats.phases[InsertIOPhaseScope::WRITE_NEW_NODE]);
    PhaseIODelta bidir = load_and_reset(
        g_insert_io_stats.phases[InsertIOPhaseScope::BIDIRECTIONAL]);

    uint64_t n_dist_comp = diskann::collect_n_dist_comp_total_and_reset();
    fprintf(stderr,
            "=== insert buffer-pool page IO (totals over %llu inserts) ===\n",
            (unsigned long long)n);
    fprintf(stderr,
            "  pq_dist_comp        total=%12llu  per_insert=%12.2f\n",
            (unsigned long long)n_dist_comp,
            (double)n_dist_comp / denom);
    print_phase("step1 beam_search", beam);
    print_phase("step2 prune", prune);
    print_phase("step3 write_new_node", write_new);
    print_phase("step4 bidirectional", bidir);
}

// ===========================================================================
// PQ helpers (local copies from pq_flash_index.cpp anonymous namespace)
// ===========================================================================
static void pq_dist_lookup(const _u8* pq_ids, const _u64 n_pts,
                            const _u64 pq_nchunks, const float* pq_dists,
                            float* dists_out) {
    _mm_prefetch((char*)dists_out, _MM_HINT_T0);
    _mm_prefetch((char*)pq_ids, _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
        const float* chunk_dists = pq_dists + 256 * chunk;
        if (chunk < pq_nchunks - 1) {
            _mm_prefetch((char*)(chunk_dists + 256), _MM_HINT_T0);
        }
        for (_u64 idx = 0; idx < n_pts; idx++) {
            _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
            dists_out[idx] += chunk_dists[pq_centerid];
        }
    }
}

static inline uint64_t elapsed_ns(
    const std::chrono::steady_clock::time_point& start) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

// Full-vector distances from query to the n nodes in ids, each read from its page through the buffer pool. A node
// that is not present gets FLT_MAX.
template<typename T>
static void exact_distances(InPlaceGraphStore* store, const uint32_t* ids, uint32_t n, const T* query,
                            Distance<T>* dist_cmp, unsigned aligned_dim, float* dists_out) {
    std::array<PageGuard, BufferPool::kPinBatchMax> guards;
    std::array<uint16_t, BufferPool::kPinBatchMax> node_to_guard;
    for (uint32_t begin = 0; begin < n; begin += BufferPool::kPinBatchMax) {
        const uint32_t batch = std::min<uint32_t>(BufferPool::kPinBatchMax, n - begin);
        const size_t n_pages =
            store->pin_pages_for_nodes_batch(ids + begin, batch, READ, guards.data(), node_to_guard.data());
        for (uint32_t i = 0; i < batch; ++i) {
            dists_out[begin + i] = std::numeric_limits<float>::max();
            if (node_to_guard[i] == InPlaceGraphStore::kNoPage) continue;
            auto guard = store->read_guard(guards[node_to_guard[i]]);
            ConstNodeRef node = guard.read_node(ids[begin + i]);
            if (node.valid()) dists_out[begin + i] = dist_cmp->compare(query, node.coords<T>(), aligned_dim);
        }
        for (size_t p = 0; p < n_pages; ++p) guards[p].release();
    }
}

// ===========================================================================
// graph_iterate_to_fixed_point -- PQ two-tier search
// ===========================================================================
template<typename T>
static
std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point_impl(
    const T* query, unsigned L,
    const std::vector<unsigned>& init_ids,
    unsigned beamwidth,
    InPlaceGraphStore* store,
    unsigned aligned_dim,
    Distance<T>* dist_cmp,
    InPlaceSearchScratch* scratch,
    std::vector<Neighbor>& results,
    TraversalScope scope,
    FixedChunkPQTable<T>* pq_table) {
    const bool record_search_stats = scope == TraversalScope::SEARCH;
    const uint32_t n_chunks = store->n_chunks();
    // Searches without PQ score every new neighbor with its full vector, so an expansion pins the expanded node's page
    // and then every neighbor's: R+1 page accesses instead of one, as in Calico's DiskANN.
    const bool exact = store->exact_search() && scope == TraversalScope::SEARCH;
    if (!exact && (n_chunks == 0 || pq_table == nullptr)) {
        throw ANNException("graph_iterate_to_fixed_point: PQ codes are required", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (!exact) pq_table->populate_chunk_distances(query, scratch->pq_dists);

    // retset: beam-search working set, PQ-scored, size L.
    auto& retset = scratch->traversal_retset;
    retset.clear();
    retset.resize(L + 1);
    unsigned cur_list_size = 0;

    // visited: nodes already enqueued into retset (checked when processing
    // neighbors, not when inserting seeds).
    auto& visited = scratch->visited;
    visited.clear();

    // results: all expanded frontier nodes with exact full-vector distances.
    results.clear();
    if (results.capacity() < 4096) {
        results.reserve(4096);
    }

    // Seed initialization: PQ distance, no visited check.
    for (const unsigned id : init_ids) {
        if (!store->is_active(id)) continue;
        auto inserted = visited.insert(id);
        if (!inserted.second) continue;
        if (exact) {
            exact_distances(store, &id, 1, query, dist_cmp, aligned_dim, scratch->dist_scratch);
        } else {
            store->gather_pq_codes(&id, 1, scratch->pq_coord_scratch);
            pq_dist_lookup(scratch->pq_coord_scratch, 1, n_chunks,
                           scratch->pq_dists, scratch->dist_scratch);
        }
        const float seed_dist = scratch->dist_scratch[0];
        if (cur_list_size == 0) {
            retset[0] = Neighbor(id, seed_dist, true);
            cur_list_size = 1;
        } else {
            InsertIntoPoolNoDup(
                retset.data(), cur_list_size, Neighbor(id, seed_dist, true));
            if (cur_list_size < L) ++cur_list_size;
        }
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);

    auto& frontier_nodes = scratch->frontier_nodes;
    frontier_nodes.clear();
    if (frontier_nodes.capacity() < std::max<unsigned>(1, beamwidth)) {
        frontier_nodes.reserve(std::max<unsigned>(1, beamwidth));
    }
    uint32_t hops = 0;
    uint32_t cmps = 0;
    unsigned k = 0;

    while (k < cur_list_size) {
        unsigned nk = cur_list_size;
        frontier_nodes.clear();
        unsigned cursor = k;
        while (cursor < cur_list_size &&
               frontier_nodes.size() < std::max<unsigned>(1, beamwidth)) {
            if (retset[cursor].flag) {
                retset[cursor].flag = false;
                frontier_nodes.push_back(retset[cursor]);
            }
            cursor++;
        }
        if (frontier_nodes.empty()) {
            ++k;
            continue;
        }

        // Batched libaio prefetch+pin of all frontier pages, grouped by page so
        // no frame is pinned twice. The page pins live for the whole iteration
        // (multiple frontier nodes can share a page); they release together
        // when page_guards is destroyed at the end of the iteration.
        assert(frontier_nodes.size() <= BufferPool::kPinBatchMax);
        std::array<uint32_t,  BufferPool::kPinBatchMax> batch_ids;
        std::array<PageGuard, BufferPool::kPinBatchMax> page_guards;
        std::array<uint16_t,  BufferPool::kPinBatchMax> node_to_guard;
        for (size_t i = 0; i < frontier_nodes.size(); ++i) {
            batch_ids[i] = frontier_nodes[i].id;
        }
        const size_t n_pages = store->pin_pages_for_nodes_batch(
            batch_ids.data(), frontier_nodes.size(), READ,
            page_guards.data(), node_to_guard.data());
        hops += static_cast<uint32_t>(frontier_nodes.size());
        // Lazy deletion: during an episode, the pages just read go to this thread's repair worker.
        lazy_hand_pages(store, page_guards.data(), n_pages);

        for (size_t fi = 0; fi < frontier_nodes.size(); ++fi) {
            const Neighbor& frontier = frontier_nodes[fi];
            const unsigned n = frontier.id;
            // A deleted node still routes the search until its slot is freed, but only live nodes become results.
            if (!store->node_present(n)) continue;
            if (node_to_guard[fi] == InPlaceGraphStore::kNoPage) continue;
            PageGuard& frame = page_guards[node_to_guard[fi]];
            if (!frame.valid()) continue;
            auto guard = store->read_guard(frame);
            ConstNodeRef node = guard.read_node(n);
            if (!node.valid() || !store->node_present(n)) {
                continue;
            }

            // Results are also insert's candidates, so a deleted node never gains a new edge.
            if (store->is_active(n) && !store->is_tag_deleted(store->node_tag(n))) {
                results.emplace_back(n, dist_cmp->compare(query, node.coords<T>(), aligned_dim), true);
            }

            scratch->frontier_candidate_ids.clear();
            if (scratch->frontier_candidate_ids.capacity() <
                scratch->frontier_candidate_ids.size() + node.degree()) {
                scratch->frontier_candidate_ids.reserve(
                    scratch->frontier_candidate_ids.size() + node.degree());
            }
            const uint32_t* nbrs = node.neighbors();
            for (uint16_t j = 0; j < node.degree(); ++j) {
                const uint32_t nbr_id = nbrs[j];
                auto inserted = visited.insert(nbr_id);
                if (inserted.second) {
                    scratch->frontier_candidate_ids.push_back(nbr_id);
                }
            }

            const uint32_t expanded_nbr_count =
                static_cast<uint32_t>(scratch->frontier_candidate_ids.size());
            if (expanded_nbr_count == 0) {
                continue;
            }
            auto start = std::chrono::steady_clock::now();

            uint32_t processed = 0;
            while (processed < expanded_nbr_count) {
                const uint32_t batch = std::min<uint32_t>(
                    InPlaceSearchScratch::MAX_SCRATCH_NODES,
                    expanded_nbr_count - processed);
                if (exact) {
                    exact_distances(store, scratch->frontier_candidate_ids.data() + processed, batch, query,
                                    dist_cmp, aligned_dim, scratch->dist_scratch);
                } else {
                    store->gather_pq_codes(reinterpret_cast<const unsigned*>(
                            scratch->frontier_candidate_ids.data() + processed),
                        batch, scratch->pq_coord_scratch);
                    pq_dist_lookup(scratch->pq_coord_scratch, batch, n_chunks,
                                   scratch->pq_dists, scratch->dist_scratch);
                }
                for (uint32_t m = 0; m < batch; ++m) {
                    unsigned nbr_id = scratch->frontier_candidate_ids[processed + m];
                    cmps++;
                    float dist = scratch->dist_scratch[m];
                    if (dist >= retset[cur_list_size - 1].distance &&
                        cur_list_size == L)
                        continue;
                    unsigned r = InsertIntoPoolNoDup(
                        retset.data(), cur_list_size,
                        Neighbor(nbr_id, dist, true));
                    if (cur_list_size < L) ++cur_list_size;
                    if (r < nk) nk = r;
                }
                processed += batch;
            }
            if (record_search_stats) {
                scratch->record_comparable_distance(
                    DistanceScope::SEARCH, elapsed_ns(start), expanded_nbr_count);
            }
        }

        if (nk <= k) k = nk;
        else ++k;
    }

    if (scope != TraversalScope::INSERT) {
        std::sort(results.begin(), results.end());
    }
    return std::make_pair(hops, cmps);
}

template<typename T>
std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point(
    const T* query, unsigned L,
    const std::vector<unsigned>& init_ids,
    unsigned beamwidth,
    InPlaceGraphStore* store,
    unsigned aligned_dim,
    Distance<T>* dist_cmp,
    InPlaceSearchScratch* scratch,
    std::vector<Neighbor>& results,
    TraversalScope scope,
    FixedChunkPQTable<T>* pq_table) {
    return graph_iterate_to_fixed_point_impl<T>(
        query, L, init_ids, beamwidth, store, aligned_dim, dist_cmp,
        scratch, results, scope, pq_table);
}

// ===========================================================================
// graph_occlude_list_pq
// ===========================================================================
#ifdef __AVX512VBMI__
namespace {
// Sums rows[c][code] over the chunks for the 64 candidates of one transposed block (block[c * 64 + lane] is a
// candidate's code in chunk c) and returns the 64 sums as four vectors of 16 floats, times scale.
inline void block_sums_u8(const uint8_t* const* rows, const uint8_t* block, uint32_t n_chunks, float scale,
                          __m512 out[4]) {
    __m512i acc_lo = _mm512_setzero_si512(), acc_hi = _mm512_setzero_si512();
    for (uint32_t c = 0; c < n_chunks; ++c) {
        const uint8_t* r = rows[c];
        const __m512i idx = _mm512_loadu_si512(block + static_cast<size_t>(c) * 64);
        const __m512i lo = _mm512_permutex2var_epi8(_mm512_loadu_si512(r), idx, _mm512_loadu_si512(r + 64));
        const __m512i hi = _mm512_permutex2var_epi8(_mm512_loadu_si512(r + 128), idx, _mm512_loadu_si512(r + 192));
        const __m512i v = _mm512_mask_blend_epi8(_mm512_movepi8_mask(idx), lo, hi);
        acc_lo = _mm512_add_epi16(acc_lo, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v)));
        acc_hi = _mm512_add_epi16(acc_hi, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(v, 1)));
    }
    const __m512 s = _mm512_set1_ps(scale);
    out[0] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(acc_lo))), s);
    out[1] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(acc_lo, 1))), s);
    out[2] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(acc_hi))), s);
    out[3] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(acc_hi, 1))), s);
}
}  // namespace
#endif

template<typename T>
void graph_occlude_list_pq(
    std::vector<Neighbor>& pool,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<Neighbor>& result,
    std::vector<float>& occlude_factor,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    DistanceScope scope,
    FixedChunkPQTable<T>* pq_table) {
    if (pool.empty()) return;
    assert(std::is_sorted(pool.begin(), pool.end()));

    (void)scratch;
    (void)scope;

    const uint32_t n_cand = static_cast<uint32_t>(std::min<size_t>(pool.size(), maxc));
    const uint32_t n_chunks = store->n_chunks();
    thread_local std::vector<uint8_t> codes;
    codes.assign(static_cast<size_t>(n_cand) * n_chunks, 0);
    for (uint32_t t = 0; t < n_cand; ++t) {
        const uint8_t* code = store->pq_codes_for_node(pool[t].id);
        if (code != nullptr) std::memcpy(codes.data() + static_cast<size_t>(t) * n_chunks, code, n_chunks);
    }

// This path is disabled for fairness. Enabling this makes the compute-bound graph repair work about 2x faster. It works with 8-bit codes.
// #ifdef __AVX512VBMI__
#if 0
    // Quicker ADC (André et al., 2019): the selected candidate's rows of the PQ-PQ table, quantized to uint8, are
    // looked up for 64 candidates at a time with byte permutes.
    if (pq_table != nullptr) {
        const uint32_t n_blocks = (n_cand + 63) / 64;
        thread_local std::vector<uint8_t> blocks;
        thread_local std::vector<float> dist, occ;
        thread_local std::vector<const uint8_t*> rows;
        blocks.assign(static_cast<size_t>(n_blocks) * n_chunks * 64, 0);
        dist.assign(static_cast<size_t>(n_blocks) * 64, 0.0f);
        occ.assign(static_cast<size_t>(n_blocks) * 64, std::numeric_limits<float>::max());
        rows.resize(n_chunks);
        for (uint32_t t = 0; t < n_cand; ++t) {
            uint8_t* block = blocks.data() + static_cast<size_t>(t / 64) * n_chunks * 64 + t % 64;
            for (uint32_t c = 0; c < n_chunks; ++c) block[c * 64] = codes[static_cast<size_t>(t) * n_chunks + c];
            dist[t] = pool[t].distance;
            occ[t] = occlude_factor[t];
        }
        const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m512 alpha_v = _mm512_set1_ps(alpha);

        float cur_alpha = 1;
        while (cur_alpha <= alpha && result.size() < degree) {
            uint32_t start = 0;
            while (result.size() < degree && start < n_cand) {
                if (occ[start] > cur_alpha) {
                    start++;
                    continue;
                }
                occ[start] = std::numeric_limits<float>::max();
                result.push_back(pool[start]);
                const uint8_t* src = codes.data() + static_cast<size_t>(start) * n_chunks;
                for (uint32_t c = 0; c < n_chunks; ++c) {
                    rows[c] = pq_table->all_to_all_u8.data() + (static_cast<size_t>(src[c]) * n_chunks + c) * 256;
                }
                for (uint32_t b = (start + 1) / 64; b < n_blocks; ++b) {
                    __m512 d[4];
                    block_sums_u8(rows.data(), blocks.data() + static_cast<size_t>(b) * n_chunks * 64, n_chunks,
                                  pq_table->all_to_all_u8_scale, d);
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t t0 = b * 64 + 16 * k;
                        const __m512i t = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(t0)), lane);
                        const __m512 o = _mm512_loadu_ps(occ.data() + t0);
                        const __mmask16 m = _mm512_cmpgt_epi32_mask(t, _mm512_set1_epi32(static_cast<int>(start))) &
                                            _mm512_cmplt_epi32_mask(t, _mm512_set1_epi32(static_cast<int>(n_cand))) &
                                            _mm512_cmp_ps_mask(o, alpha_v, _CMP_LE_OQ);
                        const __m512 ratio = _mm512_div_ps(_mm512_loadu_ps(dist.data() + t0), d[k]);
                        _mm512_storeu_ps(occ.data() + t0, _mm512_mask_max_ps(o, m, o, ratio));
                        diskann::n_dist_comp_counter += __builtin_popcount(m);
                    }
                }
                start++;
            }
            cur_alpha *= 1.2f;
        }
        return;
    }
#endif

    // Without AVX-512 VBMI: float rows, summed in compute_distances' chunk order.
    thread_local std::vector<const float*> rows;
    rows.resize(n_chunks);
    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
        uint32_t start = 0;
        while (result.size() < degree && start < n_cand) {
            if (occlude_factor[start] > cur_alpha) {
                start++;
                continue;
            }
            occlude_factor[start] = std::numeric_limits<float>::max();
            result.push_back(pool[start]);
            if (pq_table != nullptr) {
                const uint8_t* src = codes.data() + static_cast<size_t>(start) * n_chunks;
                for (uint32_t c = 0; c < n_chunks; ++c) {
                    rows[c] = pq_table->all_to_all_dists + (static_cast<size_t>(src[c]) * n_chunks + c) * 256;
                }
                for (uint32_t t = start + 1; t < n_cand; ++t) {
                    if (occlude_factor[t] > alpha) continue;
                    const uint8_t* code = codes.data() + static_cast<size_t>(t) * n_chunks;
                    float djk = 0;
                    for (uint32_t c = 0; c < n_chunks; ++c) djk += rows[c][code[c]];
                    occlude_factor[t] = std::max(occlude_factor[t], pool[t].distance / djk);
                    ++diskann::n_dist_comp_counter;
                }
            }
            start++;
        }
        cur_alpha *= 1.2f;
    }
}

// ===========================================================================
// graph_occlude_list (full-precision)
// ===========================================================================
template<typename T>
void graph_occlude_list(
    std::vector<Neighbor>& pool,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<Neighbor>& result,
    std::vector<float>& occlude_factor,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    unsigned aligned_dim,
    Distance<T>* distance,
    DistanceScope scope) {
    (void)scratch;
    (void)scope;

    if (pool.empty()) return;
    assert(std::is_sorted(pool.begin(), pool.end()));

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
        uint32_t start = 0;
        while (result.size() < degree && start < pool.size() && start < maxc) {
            auto& p = pool[start];
            if (occlude_factor[start] > cur_alpha) {
                start++;
                continue;
            }
            occlude_factor[start] = std::numeric_limits<float>::max();
            result.push_back(p);

            // Copy p's coords out under its latch, then release before the
            // t-loop, which only reads other pages.
            std::vector<T> p_coords_copy;
            bool p_valid = false;
            const T* p_coords = nullptr;
            {
                PageGuard p_frame = store->pin_page_for_node(p.id, READ);
                if (p_frame.valid()) {
                    auto p_guard = store->read_guard(p_frame);
                    ConstNodeRef p_node = p_guard.read_node(p.id);
                    if (p_node.valid() && store->is_active(p.id)) {
                        const T* src = p_node.coords<T>();
                        p_coords_copy.assign(src, src + aligned_dim);
                        p_coords = p_coords_copy.data();
                        p_valid = true;
                    }
                }
            }

            for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                if (occlude_factor[t] > alpha) continue;
                if (!p_valid) continue;
                float djk = std::numeric_limits<float>::max();
                PageGuard t_frame = store->pin_page_for_node(pool[t].id, READ);
                if (t_frame.valid()) {
                    auto t_guard = store->read_guard(t_frame);
                    ConstNodeRef t_node = t_guard.read_node(pool[t].id);
                    if (t_node.valid() && store->is_active(pool[t].id)) {
                        djk = distance->compare(p_coords, t_node.coords<T>(), aligned_dim);
                    }
                }
                occlude_factor[t] =
                    std::max(occlude_factor[t], pool[t].distance / djk);
            }
            start++;
        }
        cur_alpha *= 1.2f;
    }
}

template<typename T>
void graph_prune_neighbors_cached_exact(
    unsigned location,
    std::vector<Neighbor>& sorted_pool,
    const std::vector<T>& sorted_coords,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<unsigned>& pruned_list,
    unsigned aligned_dim,
    Distance<T>* distance) {
    if (sorted_pool.empty()) return;
    assert(std::is_sorted(sorted_pool.begin(), sorted_pool.end()));

    std::vector<uint32_t> result_indices;
    result_indices.reserve(degree);
    std::vector<float> occlude_factor(sorted_pool.size(), 0);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result_indices.size() < degree) {
        uint32_t start = 0;
        while (result_indices.size() < degree &&
               start < sorted_pool.size() && start < maxc) {
            if (occlude_factor[start] > cur_alpha) {
                start++;
                continue;
            }
            occlude_factor[start] = std::numeric_limits<float>::max();
            result_indices.push_back(start);

            const T* p_coords =
                sorted_coords.data() + static_cast<size_t>(start) * aligned_dim;
            for (uint32_t t = start + 1; t < sorted_pool.size() && t < maxc; t++) {
                if (occlude_factor[t] > alpha) continue;
                const T* t_coords =
                    sorted_coords.data() + static_cast<size_t>(t) * aligned_dim;
                const float djk = distance->compare(p_coords, t_coords, aligned_dim);
                occlude_factor[t] =
                    std::max(occlude_factor[t], sorted_pool[t].distance / djk);
            }
            start++;
        }
        cur_alpha *= 1.2f;
    }

    pruned_list.clear();
    for (uint32_t idx : result_indices) {
        if (sorted_pool[idx].id != location) {
            pruned_list.push_back(sorted_pool[idx].id);
        }
    }
}

// ===========================================================================
// graph_prune_neighbors_pq
// ===========================================================================
template<typename T>
void graph_prune_neighbors_pq(
    unsigned location,
    std::vector<Neighbor>& pool,
    unsigned R, unsigned C, float alpha,
    std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    unsigned aligned_dim,
    Distance<T>* distance,
    DistanceScope scope,
    FixedChunkPQTable<T>* pq_table) {
    if (pool.empty()) return;
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(R);
    std::vector<float> occlude_factor(pool.size(), 0);

    if (store->n_chunks() > 0 && scratch && scratch->pq_coord_scratch &&
        pq_table != nullptr) {
        graph_occlude_list_pq<T>(pool, alpha, R, C, result, occlude_factor,
                                 store, scratch, scope, pq_table);
    } else if (distance && aligned_dim > 0) {
        graph_occlude_list<T>(pool, alpha, R, C, result, occlude_factor,
                              store, scratch, aligned_dim, distance, scope);
    } else {
        // No PQ and no distance function: just take top-R by distance
        for (auto& p : pool) {
            if (p.id != location) {
                result.push_back(p);
                if (result.size() >= R) break;
            }
        }
    }

    pruned_list.clear();
    for (auto& iter : result) {
        if (iter.id != location)
            pruned_list.push_back(iter.id);
    }
}

template<typename T>
void graph_inter_insert_immediate(
    unsigned new_node,
    const std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store,
    unsigned R, unsigned C, float alpha,
    unsigned aligned_dim,
    Distance<T>* dist,
    FixedChunkPQTable<T>* pq_table) {
    if (store == nullptr || dist == nullptr || !store->is_active(new_node) ||
        pruned_list.empty()) {
        return;
    }

    struct RepairTarget {
        uint32_t node_id;
        uint32_t page_id;
    };
    std::vector<RepairTarget> targets;
    targets.reserve(pruned_list.size());
    tsl::robin_set<uint32_t> seen_targets;
    seen_targets.reserve(pruned_list.size() * 2 + 1);
    for (uint32_t target : pruned_list) {
        if (target == new_node || !seen_targets.insert(target).second ||
            !store->is_active(target)) {
            continue;
        }
        targets.push_back(RepairTarget{target, store->get_page_id(target)});
    }
    if (targets.empty()) return;

    std::sort(targets.begin(), targets.end(),
              [](const RepairTarget& a, const RepairTarget& b) {
                  if (a.page_id != b.page_id) return a.page_id < b.page_id;
                  return a.node_id < b.node_id;
              });

    InPlaceSearchScratch& local_scratch =
        maintenance_scratch_for(aligned_dim, store->n_chunks(), sizeof(T),
                                C, store->max_degree(), 1);
    std::vector<unsigned> current_nbrs;
    current_nbrs.reserve(store->max_degree() + 1);
    tsl::robin_set<unsigned> existing;
    existing.reserve(store->max_degree() + 1);
    std::vector<Neighbor> pool;
    pool.reserve(store->max_degree() + 1);
    std::vector<unsigned> pruned;
    pruned.reserve(store->max_degree());
    std::vector<uint8_t> pq_scratch_buf;
    std::vector<float> dist_buf;
    std::vector<T> target_vec_copy;

    size_t ti = 0;
    while (ti < targets.size()) {
        const uint32_t cur_page = targets[ti].page_id;
        const size_t page_start = ti;
        while (ti < targets.size() && targets[ti].page_id == cur_page) {
            ++ti;
        }
        if (cur_page == INVALID_PAGE) continue;

        for (size_t idx = page_start; idx < ti; ++idx) {
            const uint32_t target = targets[idx].node_id;
            if (!store->is_active(target)) continue;

            current_nbrs.clear();
            target_vec_copy.clear();
            const bool use_pq_dist =
                store->n_chunks() > 0 && pq_table != nullptr &&
                store->pq_codes_for_node(target) != nullptr;
            bool needs_prune = false;

            // Hold target's write latch only while reading its neighbors (and,
            // for the exact-distance path, copying its vector out). The prune
            // below works over other pages, not target, so we release here and
            // re-pin to write the result. A concurrent writer can change target
            // in between; the graph is approximate, so a lost update is fine.
            {
                PageGuard frame = store->pin_page_for_node(target, WRITE);
                if (!frame.valid()) continue;
                auto guard = store->write_guard(frame);
                MutableNodeRef node = guard.write_node(target);
                if (!node.valid() || node.node_id() != target ||
                    !store->is_active(target)) {
                    continue;
                }

                const uint32_t* nbrs = node.neighbors();
                current_nbrs.reserve(std::max<size_t>(
                    current_nbrs.capacity(), node.degree() + 1));
                for (uint16_t j = 0; j < node.degree(); ++j) {
                    const uint32_t nbr = nbrs[j];
                    if (nbr != target && store->is_active(nbr)) {
                        current_nbrs.push_back(nbr);
                    }
                }

                existing.clear();
                existing.insert(current_nbrs.begin(), current_nbrs.end());
                if (existing.find(new_node) != existing.end()) {
                    continue;
                }
                current_nbrs.push_back(new_node);

                if (current_nbrs.size() <= static_cast<size_t>(R)) {
                    node.set_neighbors(current_nbrs.data(), current_nbrs.size());
                    continue;
                }

                needs_prune = true;
                if (!use_pq_dist) {
                    const T* target_coords = node.coords<T>();
                    target_vec_copy.assign(target_coords,
                                           target_coords + aligned_dim);
                }
            }

            if (!needs_prune) continue;

            pool.clear();
            pool.reserve(std::max<size_t>(pool.capacity(), current_nbrs.size()));
            if (use_pq_dist) {
                pq_scratch_buf.resize(current_nbrs.size() * store->n_chunks() + 64);
                dist_buf.resize(current_nbrs.size());
                store->compute_pq_dists_src(
                    target, current_nbrs.data(), dist_buf.data(),
                    static_cast<uint32_t>(current_nbrs.size()),
                    pq_scratch_buf.data(), *pq_table);
                for (size_t j = 0; j < current_nbrs.size(); ++j) {
                    pool.emplace_back(current_nbrs[j], dist_buf[j], true);
                }
            } else {
                const T* target_vec = target_vec_copy.data();
                for (uint32_t nbr : current_nbrs) {
                    float d = std::numeric_limits<float>::max();
                    PageGuard nbr_frame =
                        store->pin_page_for_node(nbr, READ);
                    if (nbr_frame.valid()) {
                        auto nbr_guard = store->read_guard(nbr_frame);
                        ConstNodeRef nbr_node = nbr_guard.read_node(nbr);
                        if (nbr_node.valid() &&
                            store->is_active(nbr)) {
                            d = dist->compare(target_vec, nbr_node.coords<T>(),
                                              aligned_dim);
                        }
                    }
                    pool.emplace_back(nbr, d, true);
                }
            }

            pruned.clear();
            graph_prune_neighbors_pq<T>(target, pool, R, C, alpha,
                                        pruned, store, &local_scratch,
                                        aligned_dim, dist,
                                        DistanceScope::MAINTENANCE,
                                        pq_table);

            {
                PageGuard frame = store->pin_page_for_node(target, WRITE);
                if (!frame.valid()) continue;
                auto guard = store->write_guard(frame);
                MutableNodeRef node = guard.write_node(target);
                if (!node.valid() || node.node_id() != target ||
                    !store->is_active(target)) {
                    continue;
                }
                node.set_neighbors(pruned.data(), pruned.size());
            }
        }
    }
}

template void graph_inter_insert_immediate<float>(
    unsigned, const std::vector<unsigned>&, InPlaceGraphStore*,
    unsigned, unsigned, float, unsigned, Distance<float>*,
    FixedChunkPQTable<float>*);
template void graph_inter_insert_immediate<uint8_t>(
    unsigned, const std::vector<unsigned>&, InPlaceGraphStore*,
    unsigned, unsigned, float, unsigned, Distance<uint8_t>*,
    FixedChunkPQTable<uint8_t>*);
template void graph_inter_insert_immediate<int8_t>(
    unsigned, const std::vector<unsigned>&, InPlaceGraphStore*,
    unsigned, unsigned, float, unsigned, Distance<int8_t>*,
    FixedChunkPQTable<int8_t>*);

// Removes deleted nodes from the graph: every live node that points to a deleted node points to that node's live
// neighbors (up to 20) instead, pruned to R.
// snapshot:  copy the deleted nodes' neighbor lists into RAM first.
// !snapshot: read them from the buffer pool as needed.
template<typename T>
uint64_t InPlaceGraphStore::consolidate_deletes_freshdiskann(
    unsigned R, unsigned C, float alpha, FixedChunkPQTable<T>* pq_table,
    uint32_t num_threads, bool snapshot) {
    std::vector<uint32_t> deleted_ids;
    {
        std::lock_guard<std::mutex> lk(_delete_pending_since_maintenance_mtx);
        deleted_ids = _delete_pending_since_maintenance;
    }
    if (deleted_ids.empty()) return 0;
    if (_n_chunks == 0 || pq_table == nullptr || R == 0 || R > _Mmax ||
        C == 0 || C > InPlaceSearchScratch::MAX_SCRATCH_NODES ||
        !std::isfinite(alpha) || alpha < 1.0f) {
        throw std::runtime_error("FreshDiskANN deletion requires PQ, valid R/alpha, "
                                 "and 1 <= C <= MAX_SCRATCH_NODES");
    }
    std::sort(deleted_ids.begin(), deleted_ids.end());
    deleted_ids.erase(std::unique(deleted_ids.begin(), deleted_ids.end()), deleted_ids.end());
    num_threads = std::max<uint32_t>(1, num_threads);

    const size_t capacity = meta_capacity();
    std::vector<uint8_t> is_deleted(capacity, 0);
    for (uint32_t id : deleted_ids) {
        if (id < capacity) is_deleted[id] = 1;
    }
    auto deleted = [&](uint32_t id) { return id < capacity && is_deleted[id] != 0; };

    std::exception_ptr failure;
    std::atomic<bool> failed{false};
    auto record_failure = [&]() {
#pragma omp critical(bufann_freshdiskann_failure)
        {
            if (!failure) failure = std::current_exception();
        }
        failed.store(true, std::memory_order_relaxed);
    };

    // Up to kSampledNbrs live out-neighbors of each deleted node are bypassed, as in DGAI's trigger_deletion(nthreads,
    // 20), which keeps the candidate unions small.
    constexpr uint16_t kSampledNbrs = 20;
    auto for_each_sampled_nbr = [&](const ConstNodeRef& node, auto&& fn) {
        const uint32_t* neighbors = node.neighbors();
        uint16_t kept = 0;
        for (uint16_t j = 0; j < node.degree() && kept < kSampledNbrs; ++j) {
            if (!deleted(neighbors[j])) {
                fn(neighbors[j]);
                ++kept;
            }
        }
    };

    // Pass 1 (snapshot only): copy the deleted nodes' sampled neighbors. deleted_ids is sorted, so node-id order is
    // page order and each batch pins a run of neighboring pages. Row i holds deleted_ids[i]'s sample.
    const auto snapshot_begin = std::chrono::steady_clock::now();
    const size_t n_deleted = deleted_ids.size();
    std::vector<uint32_t> adj;
    std::vector<uint16_t> adj_degree;
    if (snapshot) {
        adj.resize(n_deleted * static_cast<size_t>(_Mmax));
        adj_degree.assign(n_deleted, 0);
        constexpr size_t kSnapshotBatch = BufferPool::kPinBatchMax;
        const size_t snapshot_batches = (n_deleted + kSnapshotBatch - 1) / kSnapshotBatch;
#pragma omp parallel for num_threads(static_cast<int>(num_threads)) schedule(dynamic, 1)
        for (int64_t b = 0; b < static_cast<int64_t>(snapshot_batches); ++b) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const size_t begin = static_cast<size_t>(b) * kSnapshotBatch;
                const size_t n = std::min(kSnapshotBatch, n_deleted - begin);
                PageGuard guards[BufferPool::kPinBatchMax];
                uint16_t node_to_guard[BufferPool::kPinBatchMax];
                pin_pages_for_nodes_batch(deleted_ids.data() + begin, n, READ, guards, node_to_guard);
                for (size_t i = 0; i < n; ++i) {
                    const uint32_t d = deleted_ids[begin + i];
                    if (node_to_guard[i] == kNoPage) {
                        throw std::runtime_error("FreshDiskANN deletion: deleted node is not present");
                    }
                    auto view = read_guard(guards[node_to_guard[i]]);
                    ConstNodeRef node = view.read_node(d);
                    if (!node.valid() || node.node_id() != d) {
                        throw std::runtime_error("FreshDiskANN deletion: invalid deleted slot");
                    }
                    uint32_t* row = adj.data() + (begin + i) * _Mmax;
                    uint16_t kept = 0;
                    for_each_sampled_nbr(node, [&](uint32_t v) { row[kept++] = v; });
                    adj_degree[begin + i] = kept;
                }
            } catch (...) {
                record_failure();
            }
        }
        if (failure) std::rethrow_exception(failure);
    }
    const double snapshot_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - snapshot_begin).count();

    auto snapshot_nbrs = [&](uint32_t v, auto&& fn) {
        const size_t row = static_cast<size_t>(
            std::lower_bound(deleted_ids.begin(), deleted_ids.end(), v) - deleted_ids.begin());
        const uint32_t* dn = adj.data() + row * _Mmax;
        for (uint16_t k = 0; k < adj_degree[row]; ++k) fn(dn[k]);
    };

    auto has_deleted_nbr = [&](const uint32_t* neighbors, uint16_t degree) {
        for (uint16_t j = 0; j < degree; ++j) {
            if (deleted(neighbors[j])) return true;
        }
        return false;
    };

    // Plans u's new neighbor list into `pruned`. deleted_nbrs(v, fn) calls fn on each sampled neighbor of deleted v.
    auto plan_repair = [&](uint32_t u, const uint32_t* neighbors, uint16_t degree, auto&& deleted_nbrs,
                           std::vector<unsigned>& pruned) {
        thread_local std::vector<uint32_t> ids;
        thread_local std::vector<Neighbor> pool;
        ids.clear();
        auto add_live = [&](uint32_t v) {
            if (v != u && !deleted(v) && is_active(v)) ids.push_back(v);
        };
        for (uint16_t j = 0; j < degree; ++j) {
            const uint32_t v = neighbors[j];
            if (deleted(v)) {
                deleted_nbrs(v, add_live);
            } else {
                add_live(v);
            }
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        pruned.clear();
        if (ids.size() <= R) {
            pruned.assign(ids.begin(), ids.end());
            return;
        }
        auto& scratch = maintenance_scratch_for(0, _n_chunks, sizeof(T), 1, _Mmax, 1);
        if (pq_codes_for_node(u) == nullptr) {
            throw std::runtime_error("FreshDiskANN deletion: missing source PQ code");
        }
        for (uint32_t id : ids) {
            if (pq_codes_for_node(id) == nullptr) {
                throw std::runtime_error("FreshDiskANN deletion: missing candidate PQ code");
            }
        }
        pool.clear();
        pool.reserve(ids.size());
        // Candidate unions may exceed the fixed PQ scratch capacity.
        for (size_t begin = 0; begin < ids.size();) {
            const uint32_t count = static_cast<uint32_t>(std::min<size_t>(
                ids.size() - begin, InPlaceSearchScratch::MAX_SCRATCH_NODES));
            compute_pq_dists_src(u, ids.data() + begin, scratch.dist_scratch,
                                 count, scratch.pq_coord_scratch, *pq_table);
            for (uint32_t j = 0; j < count; ++j) {
                pool.emplace_back(ids[begin + j], scratch.dist_scratch[j], true);
            }
            begin += count;
        }
        graph_prune_neighbors_pq<T>(u, pool, R, C, alpha, pruned, this,
            &scratch, 0, nullptr, DistanceScope::MAINTENANCE, pq_table);
    };

    // Pass 2: sequential scan over every graph page.
    const auto scan_begin = std::chrono::steady_clock::now();
    std::vector<PageDir> pages;
    {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        pages = _page_dir;
    }
    const size_t page_count = pages.size();
    constexpr size_t kScanBatch = 64;
    const size_t scan_batches = (page_count + kScanBatch - 1) / kScanBatch;
    uint64_t repaired_nodes = 0;

    // Snapshot scan: one transient WRITE pin per page, repaired in place.
    auto scan_batch_snapshot = [&](const uint32_t* batch_pages, size_t batch_size) -> uint64_t {
        uint64_t batch_repaired = 0;
        thread_local std::vector<unsigned> pruned;
        PageGuard frames[kScanBatch];
        PageID buffer_page_ids[kScanBatch];
        for (size_t i = 0; i < batch_size; ++i) buffer_page_ids[i] = graph_page(batch_pages[i]);
        _heap.pin_batch_transient(buffer_page_ids, frames, batch_size, AccessMode::WRITE,
                                  true /* account_sequential_read */);
        record_thread_page_accesses(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            if (!frames[i].valid()) continue;
            auto guard = write_guard(frames[i]);
            for (uint16_t slot = 0; slot < pages[batch_pages[i]].slots_per_page; ++slot) {
                MutableNodeRef node = guard.write_slot(slot);
                if (!node.valid()) continue;
                const uint32_t u = node.node_id();
                if (u == INVALID_NODE || !is_active(u)) continue;
                if (!has_deleted_nbr(node.neighbors(), node.degree())) continue;
                plan_repair(u, node.neighbors(), node.degree(), snapshot_nbrs, pruned);
                node.set_neighbors(pruned.data(), pruned.size());
                ++batch_repaired;
            }
            frames[i].release();
        }
        return batch_repaired;
    };

    // No-snapshot scan: READ pin the batch, plan every repair, then WRITE re-pin only the pages that changed.
    // Blocking on another page's pin is safe because only READ pins are held at that point.
    auto scan_batch_nosnapshot = [&](const uint32_t* batch_pages, size_t batch_size) -> uint64_t {
        struct PlannedWrite {
            size_t page;      // index into batch_pages
            uint16_t slot;
            uint32_t node_id;
            size_t begin;     // range in planned_nbrs
            size_t count;
        };
        thread_local std::vector<PlannedWrite> writes;
        thread_local std::vector<unsigned> planned_nbrs;
        thread_local std::vector<unsigned> pruned;
        writes.clear();
        planned_nbrs.clear();

        PageGuard frames[kScanBatch];
        PageID buffer_page_ids[kScanBatch];
        bool cache_hit[kScanBatch];
        for (size_t i = 0; i < batch_size; ++i) buffer_page_ids[i] = graph_page(batch_pages[i]);
        _heap.pin_batch(buffer_page_ids, frames, batch_size, AccessMode::READ,
                        true /* account_sequential_read */, cache_hit);
        record_thread_page_accesses(batch_size);

        // A deleted neighbor on a page of this batch is read through the READ pin already held; any other is pinned.
        auto buffered_nbrs = [&](uint32_t v, auto&& fn) {
            const uint32_t v_page = node_page(v);
            const uint32_t* it = std::lower_bound(batch_pages, batch_pages + batch_size, v_page);
            PageGuard foreign;
            PageGuard* frame = nullptr;
            if (it != batch_pages + batch_size && *it == v_page && frames[it - batch_pages].valid()) {
                frame = &frames[it - batch_pages];
            } else {
                foreign = pin_page(v_page, READ);
                frame = &foreign;
            }
            auto view = read_guard(*frame);
            ConstNodeRef node = view.read_node(v);
            if (!node.valid() || node.node_id() != v) {
                throw std::runtime_error("FreshDiskANN deletion: invalid deleted slot");
            }
            for_each_sampled_nbr(node, fn);
        };

        for (size_t i = 0; i < batch_size; ++i) {
            if (!frames[i].valid()) continue;
            auto view = read_guard(frames[i]);
            for (uint16_t slot = 0; slot < pages[batch_pages[i]].slots_per_page; ++slot) {
                ConstNodeRef node = view.read_slot(slot);
                if (!node.valid()) continue;
                const uint32_t u = node.node_id();
                if (u == INVALID_NODE || !is_active(u)) continue;
                if (!has_deleted_nbr(node.neighbors(), node.degree())) continue;
                plan_repair(u, node.neighbors(), node.degree(), buffered_nbrs, pruned);
                writes.push_back(PlannedWrite{i, slot, u, planned_nbrs.size(), pruned.size()});
                planned_nbrs.insert(planned_nbrs.end(), pruned.begin(), pruned.end());
            }
        }

        // Release the READ pins. A page with nothing to write that this batch fetched from disk is discarded now; one
        // with writes is discarded after the WRITE pin below.
        uint32_t write_pages[kScanBatch];
        size_t write_page_index[kScanBatch];
        size_t n_write_pages = 0;
        for (size_t w = 0; w < writes.size(); ++w) {
            if (n_write_pages == 0 || write_page_index[n_write_pages - 1] != writes[w].page) {
                write_page_index[n_write_pages] = writes[w].page;
                write_pages[n_write_pages++] = batch_pages[writes[w].page];
            }
        }
        for (size_t i = 0, k = 0; i < batch_size; ++i) {
            const bool has_writes = k < n_write_pages && write_page_index[k] == i;
            if (has_writes) ++k;
            if (frames[i].valid() && !cache_hit[i] && !has_writes) frames[i].discard_on_release();
            frames[i].release();
        }
        if (writes.empty()) return 0;

        // Re-pins are not counted as page accesses again.
        PageGuard write_frames[kScanBatch];
        PageID write_page_ids[kScanBatch];
        for (size_t k = 0; k < n_write_pages; ++k) write_page_ids[k] = graph_page(write_pages[k]);
        _heap.pin_batch(write_page_ids, write_frames, n_write_pages, AccessMode::WRITE);
        for (size_t k = 0, w = 0; k < n_write_pages; ++k) {
            if (!cache_hit[write_page_index[k]]) write_frames[k].discard_on_release();
            auto guard = write_guard(write_frames[k]);
            for (; w < writes.size() && writes[w].page == write_page_index[k]; ++w) {
                MutableNodeRef node = guard.write_slot(writes[w].slot);
                if (!node.valid() || node.node_id() != writes[w].node_id) {
                    throw std::runtime_error("FreshDiskANN deletion: live slot changed");
                }
                node.set_neighbors(planned_nbrs.data() + writes[w].begin, writes[w].count);
            }
            write_frames[k].release();
        }
        return writes.size();
    };

#pragma omp parallel for num_threads(static_cast<int>(num_threads)) \
    schedule(dynamic, 1) reduction(+:repaired_nodes)
    for (int64_t b = 0; b < static_cast<int64_t>(scan_batches); ++b) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            const size_t begin_page = static_cast<size_t>(b) * kScanBatch;
            const size_t end_page = std::min(begin_page + kScanBatch, page_count);
            uint32_t batch_pages[kScanBatch];
            size_t batch_size = 0;
            for (size_t page_id = begin_page; page_id < end_page; ++page_id) {
                if (pages[page_id].slots_per_page == 0 || pages[page_id].num_occupied == 0) continue;
                batch_pages[batch_size++] = static_cast<uint32_t>(page_id);
            }
            if (batch_size == 0) continue;
            repaired_nodes += snapshot ? scan_batch_snapshot(batch_pages, batch_size)
                                       : scan_batch_nosnapshot(batch_pages, batch_size);
        } catch (...) {
            record_failure();
        }
    }
    if (failure) std::rethrow_exception(failure);
    const double scan_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - scan_begin).count();

    const auto reclaimed = reclaim_delete_pending_ids(deleted_ids, num_threads);
    const std::unordered_set<uint32_t> reclaimed_set(reclaimed.begin(), reclaimed.end());
    {
        std::lock_guard<std::mutex> lk(_delete_pending_since_maintenance_mtx);
        auto& pending = _delete_pending_since_maintenance;
        pending.erase(std::remove_if(pending.begin(), pending.end(),
            [&](uint32_t id) { return reclaimed_set.count(id) != 0; }), pending.end());
    }
    std::cout << "[BufANN] consolidate_deletes_freshdiskann(): deleted=" << n_deleted
              << " snapshot_s=" << snapshot_s << " scan_pages=" << page_count
              << " scan_s=" << scan_s << " repaired_nodes=" << repaired_nodes << std::endl;
    return repaired_nodes;
}

template uint64_t InPlaceGraphStore::consolidate_deletes_freshdiskann<float>(
    unsigned, unsigned, float, FixedChunkPQTable<float>*, uint32_t, bool);
template uint64_t InPlaceGraphStore::consolidate_deletes_freshdiskann<uint8_t>(
    unsigned, unsigned, float, FixedChunkPQTable<uint8_t>*, uint32_t, bool);
template uint64_t InPlaceGraphStore::consolidate_deletes_freshdiskann<int8_t>(
    unsigned, unsigned, float, FixedChunkPQTable<int8_t>*, uint32_t, bool);

std::vector<uint32_t> InPlaceGraphStore::reclaim_delete_pending_ids(
    const std::vector<uint32_t>& node_ids, uint32_t num_threads) {
    struct ReclaimItem {
      uint32_t node_id;
      uint32_t page_id;
      uint16_t slot_idx;
    };
    std::vector<ReclaimItem> reclaim_items;
    reclaim_items.reserve(node_ids.size());
    MetaArrays* meta = meta_arrays();

    // Collect valid reclaim items. DeletePending nodes still keep their slots
    // until this maintenance commit point, so readers cannot confuse a reused
    // internal ID with an old graph reference.
    {
        uint32_t page_dir_size = 0;
        {
            std::lock_guard<std::mutex> lk(_pages_mtx);
            page_dir_size = static_cast<uint32_t>(_page_dir.size());
        }
        for (uint32_t node_id : node_ids) {
            if (!node_present(node_id)) continue;
            if (meta->states[node_id].load(std::memory_order_acquire) !=
                NodeState::DeletePending) continue;
            if (node_page(node_id) >= page_dir_size) continue;
            reclaim_items.push_back(ReclaimItem{node_id, node_page(node_id), node_slot(node_id)});
        }
    }

    std::sort(reclaim_items.begin(), reclaim_items.end(),
              [](const ReclaimItem& a, const ReclaimItem& b) {
                  return a.node_id < b.node_id;
              });

    // page_begin[k] indexes the first item on the k-th distinct page.
    std::vector<size_t> page_begin;
    for (size_t i = 0; i < reclaim_items.size(); ++i) {
        if (i == 0 || reclaim_items[i].page_id != reclaim_items[i - 1].page_id) page_begin.push_back(i);
    }
    const size_t n_pages = page_begin.size();
    page_begin.push_back(reclaim_items.size());

    struct PageFreedCount { uint32_t page_id; uint16_t freed; };
    std::vector<PageFreedCount> freed_counts;
    std::vector<uint32_t> reclaimed_ids;
    reclaimed_ids.reserve(reclaim_items.size());
    std::exception_ptr failure;
    constexpr size_t kReclaimBatch = 64;
    const size_t n_batches = (n_pages + kReclaimBatch - 1) / kReclaimBatch;

#pragma omp parallel for num_threads(static_cast<int>(std::max<uint32_t>(1, num_threads))) schedule(dynamic, 1)
    for (size_t b = 0; b < n_batches; ++b) {
        std::vector<PageFreedCount> batch_freed;
        std::vector<uint32_t> batch_reclaimed;
        try {
            const size_t first_page = b * kReclaimBatch;
            const size_t n = std::min(kReclaimBatch, n_pages - first_page);
            PageID page_ids[kReclaimBatch];
            PageGuard frames[kReclaimBatch];
            for (size_t i = 0; i < n; ++i) {
                page_ids[i] = graph_page(reclaim_items[page_begin[first_page + i]].page_id);
            }
            _heap.pin_batch_transient(page_ids, frames, n, AccessMode::WRITE);
            record_thread_page_accesses(n);
            for (size_t i = 0; i < n; ++i) {
                PageGuard& frame = frames[i];
                if (!frame.valid()) continue;
                const size_t k = first_page + i;
                auto guard = write_guard(frame);
                uint8_t* bitmap = reinterpret_cast<uint8_t*>(frame.data() + 8);
                uint16_t freed_on_page = 0;
                for (size_t j = page_begin[k]; j < page_begin[k + 1]; ++j) {
                    const ReclaimItem& item = reclaim_items[j];
                    if (!node_present(item.node_id) ||
                        meta->states[item.node_id].load(std::memory_order_acquire) != NodeState::DeletePending) {
                        continue;
                    }
                    MutableNodeRef node = guard.write_node(item.node_id);
                    if (!node.valid() || node.node_id() != item.node_id) continue;
                    bitmap[item.slot_idx / 8] &= static_cast<uint8_t>(~(1u << (item.slot_idx % 8)));
                    reset_node_slot(item.node_id);
                    batch_reclaimed.push_back(item.node_id);
                    ++freed_on_page;
                }
                if (freed_on_page > 0) {
                    frame.mark_dirty();
                    batch_freed.push_back(PageFreedCount{reclaim_items[page_begin[k]].page_id, freed_on_page});
                }
                frame.release();
            }
        } catch (...) {
#pragma omp critical(bufann_reclaim_failure)
            if (!failure) failure = std::current_exception();
        }
#pragma omp critical(bufann_reclaim_merge)
        {
            freed_counts.insert(freed_counts.end(), batch_freed.begin(), batch_freed.end());
            reclaimed_ids.insert(reclaimed_ids.end(), batch_reclaimed.begin(), batch_reclaimed.end());
        }
    }
    if (failure) std::rethrow_exception(failure);

    if (!freed_counts.empty()) {
        std::lock_guard<std::mutex> lk(_pages_mtx);
        for (const auto& fc : freed_counts) {
            if (fc.page_id < _page_dir.size()) {
                PageDir& dir = _page_dir[fc.page_id];
                dir.num_occupied = (dir.num_occupied >= fc.freed)
                                       ? dir.num_occupied - fc.freed
                                       : 0;
                _occupied_slots.fetch_sub(
                    fc.freed, std::memory_order_relaxed);
                _pages_with_space.insert(fc.page_id);
            }
        }
    }

    // The reclaimed slots are Free now, so reserve_node_id can hand them out again.
    return reclaimed_ids;
}

// ===========================================================================
// Explicit template instantiations for graph ops
// ===========================================================================
template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<float>(
    const float*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<float>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&, TraversalScope,
    FixedChunkPQTable<float>*);

template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<uint8_t>(
    const uint8_t*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<uint8_t>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&, TraversalScope,
    FixedChunkPQTable<uint8_t>*);

template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<int8_t>(
    const int8_t*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<int8_t>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&, TraversalScope,
    FixedChunkPQTable<int8_t>*);

template void graph_occlude_list_pq<float>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope,
    FixedChunkPQTable<float>*);
template void graph_occlude_list_pq<uint8_t>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope,
    FixedChunkPQTable<uint8_t>*);
template void graph_occlude_list_pq<int8_t>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope,
    FixedChunkPQTable<int8_t>*);

template void graph_occlude_list<float>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, unsigned, Distance<float>*,
    DistanceScope);

template void graph_prune_neighbors_pq<float>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<float>*, DistanceScope, FixedChunkPQTable<float>*);
template void graph_prune_neighbors_pq<uint8_t>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<uint8_t>*, DistanceScope, FixedChunkPQTable<uint8_t>*);
template void graph_prune_neighbors_pq<int8_t>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<int8_t>*, DistanceScope, FixedChunkPQTable<int8_t>*);

}  // namespace inplace
}  // namespace diskann
