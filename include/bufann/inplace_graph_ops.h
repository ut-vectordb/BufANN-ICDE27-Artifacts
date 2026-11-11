#pragma once

#include <string>
#include <vector>
#include "distance.h"
#include "neighbor.h"
#include "pq_table.h"
#include "bufann/inplace_backend.h"

namespace diskann {
namespace inplace {

// Beam search over the in-place graph.
// retset (search queue) is scored by PQ distance; results receives all
// expanded frontier nodes with exact full-vector distances, sorted ascending.
// The caller must hold a metadata guard (enter_meta_read or enter_meta_write) for the whole call.
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
    TraversalScope scope = TraversalScope::SEARCH,
    FixedChunkPQTable<T>* pq_table = nullptr);

// PQ-based alpha-occlusion
template<typename T>
void graph_occlude_list_pq(
    std::vector<Neighbor>& pool,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<Neighbor>& result,
    std::vector<float>& occlude_factor,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    DistanceScope scope,
    FixedChunkPQTable<T>* pq_table = nullptr);

// Full-precision occlusion (for drain when coords are in cache)
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
    DistanceScope scope);

// Prune: uses PQ occlusion when available, full-precision fallback otherwise
template<typename T>
void graph_prune_neighbors_pq(
    unsigned location,
    std::vector<Neighbor>& pool,
    unsigned R, unsigned C, float alpha,
    std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    unsigned aligned_dim = 0,
    Distance<T>* distance = nullptr,
    DistanceScope scope = DistanceScope::MAINTENANCE,
    FixedChunkPQTable<T>* pq_table = nullptr);

// Per-phase buffer-pool page IO counters for insertion. Wrap each phase of
// bufann_insert in an InsertIOPhaseScope, then call insert_stats_increment_n()
// once per completed insert. Dump + reset via dump_and_reset_insert_phase_stats.
class InsertIOPhaseScope {
 public:
    enum Phase {
        BEAM_SEARCH = 0,
        PRUNE,
        WRITE_NEW_NODE,
        BIDIRECTIONAL,
        NUM_PHASES,
    };
    explicit InsertIOPhaseScope(Phase phase);
    ~InsertIOPhaseScope();
    InsertIOPhaseScope(const InsertIOPhaseScope&) = delete;
    InsertIOPhaseScope& operator=(const InsertIOPhaseScope&) = delete;

 private:
    Phase _phase;
    uint64_t _start_accesses;
    uint64_t _start_misses;
};

void insert_stats_increment_n();
void dump_and_reset_insert_phase_stats();

// Immediate reverse-edge repair for an inserted node. This updates the reverse
// neighbors directly and does not touch the deferred-edge buffer.
template<typename T>
void graph_inter_insert_immediate(
    unsigned new_node,
    const std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store,
    unsigned R, unsigned C, float alpha,
    unsigned aligned_dim,
    Distance<T>* dist,
    FixedChunkPQTable<T>* pq_table = nullptr);

}  // namespace inplace
}  // namespace diskann
