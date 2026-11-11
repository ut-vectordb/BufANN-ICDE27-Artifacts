// bufann_api.h
//
// Clean interface for building and querying a BufANN disk-resident graph index.
// Suitable for use by serverful database engines and external benchmarks.
//
// Template parameter T: float | uint8_t | int8_t

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "distance.h"
#include "pq_table.h"
#include "utils.h"
#include "bufann/inplace_backend.h"

namespace diskann {
namespace inplace {

template<typename T>
struct BufANNIndex;

template<typename T>
std::shared_ptr<InPlaceGraphStore> bufann_store_snapshot(
    const BufANNIndex<T>& idx);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

enum class DeleteAlgorithm : uint8_t {
    FreshDiskANN,            // deleted nodes' adjacency snapshotted in memory before the scan
    FreshDiskANNNoSnapshot,  // deleted nodes read through the buffer pool during the scan
    Lazy,                    // searches and inserts repair deleted neighbors while waiting on IO (lazy_delete.h)
};

struct BufANNConfig {
    // --- required ---
    uint32_t dim = 0;           // vector dimensionality

    // --- graph parameters ---
    uint32_t R           = 64;  // max graph degree (used for build and insert pruning)
    uint32_t L           = 100; // beam width for graph construction / default search beam
    uint32_t C           = 160; // candidate list cap for RNG pruning
    float    alpha       = 1.2f;
    bool     saturate_graph = false;

    // --- threading ---
    uint32_t build_threads = 8;

    // --- buffer pool ---
    uint32_t page_size                  = 4096;
    uint32_t buffer_pool_frames         = 16384;
    uint32_t flush_budget_pages_per_cycle = 32;
    uint32_t flush_wakeup_ms            = 100;

    // --- search ---
    uint32_t beamwidth = 4;
    // false scores every neighbor a search visits with its full vector, read from its page, instead of with PQ.
    bool search_pq = true;

    // --- product quantization ---
    uint32_t pq_chunks = 0;  // 0 = no PQ

    // --- metadata/PQ storage ---
    // Initial flat metadata capacity for build/load. 0 derives it from the
    // current dataset.
    // Not a permanent insertion limit: arrays grow geometrically and are
    // RCU-published on a rare capacity miss.
    uint32_t max_dataset_size = 0;
    // Maximum workers used by synchronous metadata growth copies. Work is
    // adaptively capped by the number of coarse copy ranges.
    uint32_t meta_growth_copy_threads = 16;

    // --- deletion ---
    DeleteAlgorithm delete_algorithm = DeleteAlgorithm::FreshDiskANN;

    // --- build mode ---
    // memory_budget_gb == 0: in-memory build (full dataset must fit in RAM)
    // memory_budget_gb  > 0: out-of-core disk-stream build via merged Vamana
    float memory_budget_gb = 0.0f;

    // --- metric ---
    diskann::Metric metric = diskann::Metric::L2;
};

// ---------------------------------------------------------------------------
// Index handle
// ---------------------------------------------------------------------------

template<typename T>
struct BufANNIndex {
    std::vector<unsigned> medoids;
    // Float centroids retained for on-disk format compatibility. Search now
    // starts from all live medoids rather than selecting one via its centroid.
    std::vector<float>    medoid_coords;
    BufANNConfig         config;
    diskann::Distance<T>* dist_cmp        = nullptr;
    diskann::Distance<float>* dist_cmp_float = nullptr;
    FixedChunkPQTable<T>  pq_table;
    uint32_t              aligned_dim = 0;
    std::string           pq_prefix;  // path prefix used for PQ files
    std::string           index_prefix;
    std::shared_ptr<InPlaceGraphStore> store = std::make_shared<InPlaceGraphStore>();
    // Shared steady-clock origin for correlating runtime_event JSON with
    // window metrics. Optional; zero means unset.
    std::chrono::steady_clock::time_point runtime_epoch{};

    BufANNIndex() = default;
    ~BufANNIndex() {
        delete dist_cmp; dist_cmp = nullptr;
        delete dist_cmp_float; dist_cmp_float = nullptr;
    }

    BufANNIndex(const BufANNIndex&)            = delete;
    BufANNIndex& operator=(const BufANNIndex&) = delete;
};

template<typename T>
std::shared_ptr<InPlaceGraphStore> bufann_store_snapshot(const BufANNIndex<T>& idx) {
    return idx.store;
}

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

// Build a new index from a raw data file.
//
// data_bin      - path to the flat binary file (.bin) containing base vectors.
//                 Format: [int32 npts][int32 dim][T * npts * dim].
// index_prefix  - path prefix for the heap file and metadata files written to
//                 disk (e.g. "/data/myindex/bufann").
// config        - build and runtime parameters (see BufANNConfig above).
// build_temp_dir- scratch directory for intermediate files when using the
//                 out-of-core disk-stream build mode. Ignored for in-memory build.
//
// Returns an owning pointer to the loaded index. Caller must free with
// bufann_free().
template<typename T>
BufANNIndex<T>* bufann_build(
    const std::string& data_bin,
    const std::string& index_prefix,
    const BufANNConfig& config,
    const std::string& build_temp_dir = "");

// Load an existing index that was previously built and saved to disk.
//
// index_prefix  - same prefix used when the index was built.
// config        - must supply at least dim, pq_chunks, page_size,
//                 buffer_pool_frames, and beamwidth. Build parameters
//                 (R, L, C, alpha) are re-used for maintenance operations.
//
// Returns an owning pointer. Caller must free with bufann_free().
template<typename T>
BufANNIndex<T>* bufann_load(
    const std::string& index_prefix,
    const BufANNConfig& config);

uint32_t bufann_snapshot_active_cap(const std::string& index_prefix);

// Insert a single node into the graph index.
//
// Synchronous: on return the node is live and reachable from search. Reverse
// edges are repaired in the foreground before return.
//
// tag      - external ID for the new point (must not already be active).
// coords   - raw vector with config.dim elements. Zero-padded internally
//            to the aligned dimension.
// search_L - beam width to use when finding the new node's neighbors.
//            Pass 0 to use config.L.
template<typename T>
void bufann_insert(
    BufANNIndex<T>& idx,
    TagType tag,
    const T* coords,
    uint32_t search_L = 0,
    uint32_t* internal_id_out = nullptr);

// Logically delete a node by external tag. Foreground delete records the tag in
// an exact deleted-tag set and returns immediately; query result emission filters
// that set so the tag is invisible before graph repair runs.
template<typename T>
void bufann_delete(
    BufANNIndex<T>& idx,
    TagType tag);

// Micro-batched logical delete: records every tag in `tags` in the deleted-tag
// set. Tag-to-internal-ID resolution and graph repair happen in maintenance.
template<typename T>
void bufann_delete_batch(
    BufANNIndex<T>& idx,
    const TagType* tags,
    size_t count);

// Maintenance pass for logical deletes since the previous maintenance run.
// Swaps out the current deleted-tag set, scans metadata arrays to resolve tags
// to active internal IDs, transitions those IDs to DeletePending, repairs their
// in-neighbors with FreshDiskANN consolidation, then reclaims slots/internal IDs.
template<typename T>
void bufann_cleanup_deleted_edges(BufANNIndex<T>& idx,
                                  uint32_t num_threads = 1);

// Flush all dirty pages currently held by the BufANN buffer pool.
template<typename T>
void bufann_flush_dirty(BufANNIndex<T>& idx);

// Flush up to max_pages dirty pages. max_pages=0 flushes all dirty pages.
template<typename T>
uint32_t bufann_flush_dirty_budget(BufANNIndex<T>& idx, uint32_t max_pages);

// Search for the topK approximate nearest neighbors of query_vec.
//
// query_vec - raw query vector with config.dim elements. Zero-padded
//             internally to the aligned dimension.
// topK      - number of results to return.
// search_L  - beam width. Pass 0 to use config.L.
// out_tags  - caller-provided buffer with capacity >= topK. When nullptr,
//             results are discarded and only the count is returned.
//
// Returns the number of result tags written to out_tags.
template<typename T>
uint32_t bufann_query_into(
    BufANNIndex<T>& idx,
    const T* query_vec,
    uint32_t topK,
    TagType* out_tags,
    uint32_t search_L = 0);

// Convenience wrapper that returns a vector of at most topK tags sorted
// by ascending distance.
template<typename T>
std::vector<TagType> bufann_query(
    BufANNIndex<T>& idx,
    const T* query_vec,
    uint32_t topK,
    uint32_t search_L = 0);

// Repair the graph after deletions.
//
// Performs up to `budget` pages of sweep-repair: for each live node whose
// neighbor list references a deleted node, the dead neighbor is replaced by
// candidates drawn from its neighborhood.
//
// Reverse-edge consistency is handled in insert foreground path.
//
// budget - number of pages to sweep per call. Pass 0 to sweep all pages.
template<typename T>
void bufann_cleanup(BufANNIndex<T>& idx);

// Release all resources held by the index. Sets the pointer to nullptr.
template<typename T>
void bufann_free(BufANNIndex<T>*& idx);

struct BufANNRuntimeSnapshot {
    uint32_t num_active = 0;
    size_t meta_capacity = 0;
    double fragmentation_ratio = 0.0;
    bool meta_growth_active = false;
    uint64_t meta_growth_count = 0;
    uint64_t meta_growth_last_from = 0;
    uint64_t meta_growth_last_to = 0;
    uint64_t meta_growth_total_ns = 0;
    uint64_t meta_growth_last_started_us = 0;
    uint64_t meta_growth_last_finished_us = 0;
    uint64_t meta_growth_update_stall_ns = 0;
    uint64_t meta_retired_pending = 0;
    uint64_t meta_retired_generations = 0;
    uint64_t meta_reclaimed_generations = 0;
};

template<typename T>
BufANNRuntimeSnapshot bufann_runtime_snapshot(const BufANNIndex<T>& idx);

// Optional origin for correlating event timestamps with driver window clocks.
template<typename T>
void bufann_set_runtime_epoch(
    BufANNIndex<T>& idx,
    std::chrono::steady_clock::time_point epoch);

}  // namespace inplace
}  // namespace diskann
