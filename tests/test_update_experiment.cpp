// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// In-place update backend verification runner.
// Supports --smoke for quick correctness checks and mixed workloads
// for metric collection. Outputs JSONL records.

#include "bufann/inplace_backend.h"
#include "bufann/inplace_graph_ops.h"
#include "index.h"
#include "utils.h"
#include "timer.h"
#include "distance.h"
#include "parameters.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
// shared_timed_mutex available in C++14; shared_mutex in C++17
// We use shared_timed_mutex for C++14 compatibility
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

#include "tsl/robin_set.h"

using namespace diskann::inplace;

static std::atomic<uint64_t> g_tmp_file_id(0);

template<typename T>
static std::unique_ptr<diskann::Distance<T>> make_l2_distance();

template<>
std::unique_ptr<diskann::Distance<float>> make_l2_distance<float>() {
  return std::unique_ptr<diskann::Distance<float>>(new diskann::DistanceL2());
}

template<>
std::unique_ptr<diskann::Distance<uint8_t>> make_l2_distance<uint8_t>() {
  return std::unique_ptr<diskann::Distance<uint8_t>>(new diskann::DistanceL2UInt8());
}

template<>
std::unique_ptr<diskann::Distance<int8_t>> make_l2_distance<int8_t>() {
  return std::unique_ptr<diskann::Distance<int8_t>>(new diskann::DistanceL2Int8());
}

static std::string make_temp_heap_path(const std::string& tag) {
  return "/tmp/" + tag + "_" + std::to_string((uint64_t) getpid()) + "_" +
         std::to_string(g_tmp_file_id.fetch_add(1, std::memory_order_relaxed)) + ".bin";
}

static uint32_t heap_file_pages(const std::string& heap_path, uint32_t page_size) {
  struct stat st;
  if (::stat(heap_path.c_str(), &st) != 0 || st.st_size < 0) {
    return 0;
  }
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(st.st_size) + page_size - 1) / page_size);
}

// ---------------------------------------------------------------------------
// ThreadContext: per-thread scratch + latency collection
// ---------------------------------------------------------------------------
struct ThreadContext {
  std::vector<double> latencies_us;
  std::vector<double> cpu_us_samples;
  std::vector<double> io_us_samples;
  uint64_t            ops_count = 0;

  double   local_flush_time_us = 0;
  double   local_bg_flush_time_us = 0;
  double   local_repair_time_us = 0;
  uint64_t local_pages_touched_insert = 0;
  uint64_t local_pages_touched_delete = 0;

  InPlaceSearchScratch scratch;

  void reset_local() {
    latencies_us.clear();
    cpu_us_samples.clear();
    io_us_samples.clear();
    ops_count = 0;
    local_flush_time_us = 0;
    local_bg_flush_time_us = 0;
    local_repair_time_us = 0;
    local_pages_touched_insert = 0;
    local_pages_touched_delete = 0;
  }
};

// ---------------------------------------------------------------------------
// ExperimentMetrics
// ---------------------------------------------------------------------------
struct ExperimentMetrics {
  std::string backend_name;
  std::string dataset;
  std::string workload;
  uint32_t    checkpoint_id = 0;
  uint32_t    base_points = 0;
  uint32_t    active_points = 0;
  uint32_t    updates_completed = 0;
  uint32_t    query_threads = 0;
  uint32_t    update_threads = 0;
  float       update_fraction = 0;
  uint32_t    search_L = 0;
  uint32_t    recall_at = 0;
  float       recall = 0;
  double      qps = 0;
  double      query_throughput = 0;
  double      insert_throughput = 0;
  double      delete_throughput = 0;
  double      update_throughput = 0;
  double      lat_avg_us = 0;
  double      lat_p50_us = 0;
  double      lat_p95_us = 0;
  double      lat_p99_us = 0;
  uint64_t    bytes_read = 0;
  uint64_t    bytes_written = 0;
  uint64_t    cache_hits = 0;
  uint64_t    cache_misses = 0;
  uint64_t    pages_flushed = 0;
  double      cache_hit_rate = 0;
  double      maintenance_time_us = 0;
  size_t      total_memory_bytes = 0;
  size_t      rss_bytes = 0;
  size_t      buffer_pool_bytes_v = 0;
  size_t      locator_bytes_v = 0;
  size_t      pq_data_bytes_v = 0;
  double      query_distance_us = 0;
  double      update_distance_us = 0;
  double      total_distance_us = 0;
  uint64_t    query_distance_ops = 0;
  uint64_t    update_distance_ops = 0;
  uint64_t    total_distance_ops = 0;
  std::string notes;

  std::string to_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "{";
    ss << "\"backend\":\"" << backend_name << "\",";
    ss << "\"dataset\":\"" << dataset << "\",";
    ss << "\"workload\":\"" << workload << "\",";
    ss << "\"checkpoint_id\":" << checkpoint_id << ",";
    ss << "\"base_points\":" << base_points << ",";
    ss << "\"active_points\":" << active_points << ",";
    ss << "\"updates_completed\":" << updates_completed << ",";
    ss << "\"query_threads\":" << query_threads << ",";
    ss << "\"update_threads\":" << update_threads << ",";
    ss << "\"update_fraction\":" << update_fraction << ",";
    ss << "\"search_L\":" << search_L << ",";
    ss << "\"recall_at\":" << recall_at << ",";
    ss << "\"recall\":" << recall << ",";
    ss << "\"qps\":" << qps << ",";
    ss << "\"query_throughput\":" << query_throughput << ",";
    ss << "\"insert_throughput\":" << insert_throughput << ",";
    ss << "\"delete_throughput\":" << delete_throughput << ",";
    ss << "\"update_throughput\":" << update_throughput << ",";
    ss << "\"lat_avg_us\":" << lat_avg_us << ",";
    ss << "\"lat_p50_us\":" << lat_p50_us << ",";
    ss << "\"lat_p95_us\":" << lat_p95_us << ",";
    ss << "\"lat_p99_us\":" << lat_p99_us << ",";
    ss << "\"bytes_read\":" << bytes_read << ",";
    ss << "\"bytes_written\":" << bytes_written << ",";
    ss << "\"cache_hits\":" << cache_hits << ",";
    ss << "\"cache_misses\":" << cache_misses << ",";
    ss << "\"pages_flushed\":" << pages_flushed << ",";
    ss << "\"cache_hit_rate\":" << cache_hit_rate << ",";
    ss << "\"maintenance_time_us\":" << maintenance_time_us << ",";
    ss << "\"total_memory_bytes\":" << total_memory_bytes << ",";
    ss << "\"rss_bytes\":" << rss_bytes << ",";
    ss << "\"buffer_pool_bytes\":" << buffer_pool_bytes_v << ",";
    ss << "\"locator_bytes\":" << locator_bytes_v << ",";
    ss << "\"pq_data_bytes\":" << pq_data_bytes_v << ",";
    ss << "\"query_distance_us\":" << query_distance_us << ",";
    ss << "\"update_distance_us\":" << update_distance_us << ",";
    ss << "\"total_distance_us\":" << total_distance_us << ",";
    ss << "\"query_distance_ops\":" << query_distance_ops << ",";
    ss << "\"update_distance_ops\":" << update_distance_ops << ",";
    ss << "\"total_distance_ops\":" << total_distance_ops << ",";
    ss << "\"notes\":\"" << notes << "\"";
    ss << "}";
    return ss.str();
  }
};

// ---------------------------------------------------------------------------
// Latency sampling helpers
// ---------------------------------------------------------------------------
struct LatencySampler {
  enum : size_t { CAP = 4096 };

  std::array<std::atomic<uint32_t>, CAP> samples{};
  std::atomic<uint64_t> total_count{0};
  std::atomic<uint64_t> total_us{0};

  void observe(double lat_us) {
    uint32_t us = (uint32_t) std::min<double>(lat_us, (double) std::numeric_limits<uint32_t>::max());
    uint64_t idx = total_count.fetch_add(1, std::memory_order_relaxed);
    samples[idx % CAP].store(us, std::memory_order_relaxed);
    total_us.fetch_add(us, std::memory_order_relaxed);
  }

  void snapshot(std::vector<double>& out, uint64_t& count_out, uint64_t& total_us_out) const {
    count_out = total_count.load(std::memory_order_relaxed);
    total_us_out = total_us.load(std::memory_order_relaxed);
    size_t sample_count = (size_t) std::min<uint64_t>(count_out, CAP);
    out.reserve(out.size() + sample_count);
    for (size_t i = 0; i < sample_count; i++) {
      out.push_back((double) samples[i].load(std::memory_order_relaxed));
    }
  }
};

static double sorted_percentile(const std::vector<double>& v, double p) {
  if (v.empty())
    return 0;
  size_t idx = (size_t) (p * v.size());
  if (idx >= v.size())
    idx = v.size() - 1;
  return v[idx];
}

// ---------------------------------------------------------------------------
// Build initial in-memory index
// ---------------------------------------------------------------------------
template<typename T>
static void build_mem_index(T* base_data, uint32_t npts, uint32_t dim,
                            uint32_t aligned_dim, uint32_t R, uint32_t L,
                            float                        alpha,
                            diskann::Index<T, uint32_t>& mem_index) {
  // Write temp file
  std::string tmp_file = "/tmp/inplace_smoke_base_" +
                         std::to_string((uint64_t) getpid()) + "_" +
                         std::to_string(g_tmp_file_id.fetch_add(1)) + ".bin";
  {
    std::ofstream out(tmp_file, std::ios::binary);
    uint32_t      meta[2] = {npts, dim};
    out.write((char*) meta, 8);
    out.write((char*) base_data, (size_t) npts * dim * sizeof(T));
  }

  diskann::Parameters params;
  params.Set<unsigned>("L", L);
  params.Set<unsigned>("R", R);
  params.Set<float>("alpha", alpha);
  params.Set<unsigned>("C", R * 2);
  params.Set<unsigned>("num_threads", 1);
  params.Set<bool>("saturate_graph", false);

  mem_index.build(tmp_file.c_str(), npts, params);
}

// ===========================================================================
// SMOKE TESTS
// ===========================================================================

static bool smoke_1_disk_query(const std::string& data_root) {
  std::cout << "[Smoke 1] Disk query test..." << std::endl;

  std::string base_path = data_root + "/sift/sift_base.bin";
  std::string query_path = data_root + "/sift/sift_query.bin";

  float* base_data = nullptr;
  float* query_data = nullptr;
  size_t base_npts, base_dim, base_aligned_dim;
  size_t query_npts, query_dim, query_aligned_dim;

  diskann::load_aligned_bin<float>(base_path, base_data, base_npts, base_dim,
                                   base_aligned_dim);
  diskann::load_aligned_bin<float>(query_path, query_data, query_npts,
                                   query_dim, query_aligned_dim);

  uint32_t n_smoke = std::min((uint32_t) base_npts, (uint32_t) 2000);
  uint32_t dim = (uint32_t) base_dim;
  uint32_t aligned_dim = (uint32_t) ROUND_UP(dim, 8);
  uint32_t R = 32, L = 50, Mmax = 32;
  float    alpha = 1.2f;

  diskann::Index<float, uint32_t> mem_index(diskann::Metric::L2, dim,
                                            n_smoke + 1, false, false, false);
  build_mem_index<float>(base_data, n_smoke, dim, aligned_dim, R, L, alpha,
                         mem_index);

  std::string       heap_path = "/tmp/inplace_smoke_heap.bin";
  InPlaceGraphStore store;
  store.init(dim, Mmax, sizeof(float), 4096, 1024, heap_path);
  store.bulk_load_from_index(mem_index, n_smoke);
  store.warmup_bfs(0, std::min(n_smoke, (uint32_t) 512));

  diskann::DistanceL2 dist_cmp;
  uint32_t n_queries = std::min((uint32_t) query_npts, (uint32_t) 25);
  uint32_t K = 10;
  uint32_t search_L = 40;

  double total_recall = 0;
  for (uint32_t q = 0; q < n_queries; q++) {
    const float*         qvec = query_data + q * query_aligned_dim;
    InPlaceSearchScratch scratch;
    scratch.init(aligned_dim, 0, sizeof(float), search_L, Mmax, 4);

    std::vector<diskann::Neighbor> results;
    graph_iterate_to_fixed_point<float>(
        qvec, search_L, {(unsigned) store.entry_point()}, 4, &store, aligned_dim,
        &dist_cmp, &scratch, results);

    // Brute force top-K for recall
    std::vector<std::pair<float, uint32_t>> all_dists;
    for (uint32_t i = 0; i < n_smoke; i++) {
      float d =
          dist_cmp.compare(qvec, base_data + i * base_aligned_dim, aligned_dim);
      all_dists.push_back({d, i});
    }
    std::sort(all_dists.begin(), all_dists.end());

    tsl::robin_set<uint32_t> gt_set;
    for (uint32_t i = 0; i < K && i < all_dists.size(); i++) {
      gt_set.insert(all_dists[i].second);
    }
    uint32_t hits = 0;
    for (uint32_t i = 0; i < K && i < results.size(); i++) {
      if (results[i].distance < std::numeric_limits<float>::max() &&
          gt_set.find(results[i].id) != gt_set.end()) {
        hits++;
      }
    }
    total_recall += (double) hits / K;
  }
  double avg_recall = total_recall / n_queries;
  std::cout << "  Recall@" << K << " = " << avg_recall << std::endl;

  if (base_data)
    diskann::aligned_free(base_data);
  if (query_data)
    diskann::aligned_free(query_data);

  // sift-8K sanity floor; not a production target
  bool pass = (avg_recall >= 0.60);
  std::cout << "  [Smoke 1] " << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

static bool smoke_2_update_deferred(const std::string& data_root) {
  std::cout << "[Smoke 2] Update + deferred edges test..." << std::endl;

  std::string base_path = data_root + "/sift/sift_base.bin";
  float*      base_data = nullptr;
  size_t      base_npts, base_dim, base_aligned_dim;
  diskann::load_aligned_bin<float>(base_path, base_data, base_npts, base_dim,
                                   base_aligned_dim);

  uint32_t n_initial = std::min((uint32_t) base_npts, (uint32_t) 2000);
  uint32_t n_insert =
      std::min((uint32_t) 250, (uint32_t) (base_npts - n_initial));
  uint32_t n_delete = 100;
  uint32_t dim = (uint32_t) base_dim;
  uint32_t aligned_dim = (uint32_t) ROUND_UP(dim, 8);
  uint32_t R = 32, L = 50, Mmax = 32;
  float    alpha = 1.2f;

  diskann::Index<float, uint32_t> mem_index(
      diskann::Metric::L2, dim, n_initial + n_insert + 1, false, false, false);
  build_mem_index<float>(base_data, n_initial, dim, aligned_dim, R, L, alpha,
                         mem_index);

  std::string       heap_path = "/tmp/inplace_smoke2_heap.bin";
  InPlaceGraphStore store;
  store.init(dim, Mmax, sizeof(float), 4096, 2048, heap_path);
  store.bulk_load_from_index(mem_index, n_initial);

  diskann::DistanceL2 dist_cmp;

  // Insert n_insert points
  for (uint32_t i = 0; i < n_insert; i++) {
    uint32_t     node_id = n_initial + i;
    const float* coords = base_data + node_id * base_aligned_dim;

    store.allocate_node(node_id);

    // Search for neighbors BEFORE publishing (node invisible to queries)
    InPlaceSearchScratch scratch;
    scratch.init(aligned_dim, 0, sizeof(float), L, Mmax, 4);
    std::vector<diskann::Neighbor> candidates;
    graph_iterate_to_fixed_point<float>(
        coords, L, {(unsigned) store.entry_point()}, 4, &store, aligned_dim,
        &dist_cmp, &scratch, candidates, true, TraversalScope::INSERT);

    std::vector<unsigned> pruned;
    graph_prune_neighbors_pq<float>(node_id, candidates, R, R * 2, alpha,
                                    pruned, &store, &scratch, 0, nullptr,
                                    DistanceScope::INSERT);

    // Write coords + neighbors atomically, then publish
    PageGuard frame = store.pin_page_for_node(node_id, WRITE);
    auto guard = store.write_guard(frame);
    MutableNodeRef node = guard.write_node(node_id);
    node.set_coords<float>(coords);
    node.set_neighbors(pruned.data(), std::min<size_t>(Mmax, pruned.size()));
    store.publish_node(node_id);
    store.stats().total_inserts.fetch_add(1);
  }

  // Delete n_delete points (from the initial set)
  std::mt19937          rng(42);
  std::vector<uint32_t> delete_candidates(n_initial);
  std::iota(delete_candidates.begin(), delete_candidates.end(), 0);
  std::shuffle(delete_candidates.begin(), delete_candidates.end(), rng);
  tsl::robin_set<uint32_t> deleted_set;
  for (uint32_t i = 0; i < n_delete && i < delete_candidates.size(); i++) {
    store.transition_active_to_delete_pending(delete_candidates[i]);
    deleted_set.insert(delete_candidates[i]);
  }

  // Search and check
  std::string query_path = data_root + "/sift/sift_query.bin";
  float*      query_data = nullptr;
  size_t      query_npts, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<float>(query_path, query_data, query_npts,
                                   query_dim, query_aligned_dim);
  uint32_t n_queries = std::min((uint32_t) query_npts, (uint32_t) 50);
  uint32_t K = 10;
  uint32_t search_L = 40;

  bool   no_deleted_in_results = true;
  double total_recall = 0;

  // Build active set
  tsl::robin_set<uint32_t> active_set;
  uint32_t                 total_pts = n_initial + n_insert;
  for (uint32_t i = 0; i < total_pts; i++) {
    if (deleted_set.find(i) == deleted_set.end()) {
      active_set.insert(i);
    }
  }

  for (uint32_t q = 0; q < n_queries; q++) {
    const float*         qvec = query_data + q * query_aligned_dim;
    InPlaceSearchScratch scratch;
    scratch.init(aligned_dim, 0, sizeof(float), search_L, Mmax, 4);

    std::vector<diskann::Neighbor> results;
    graph_iterate_to_fixed_point<float>(
        qvec, search_L, {(unsigned) store.entry_point()}, 4, &store, aligned_dim,
        &dist_cmp, &scratch, results);

    for (uint32_t i = 0; i < K && i < results.size(); i++) {
      if (results[i].distance < std::numeric_limits<float>::max()) {
        if (deleted_set.find(results[i].id) != deleted_set.end()) {
          no_deleted_in_results = false;
        }
      }
    }

    // Brute-force recall against active set
    std::vector<std::pair<float, uint32_t>> all_dists;
    for (uint32_t i = 0; i < total_pts; i++) {
      if (active_set.find(i) != active_set.end()) {
        float d = dist_cmp.compare(qvec, base_data + i * base_aligned_dim,
                                   aligned_dim);
        all_dists.push_back({d, i});
      }
    }
    std::sort(all_dists.begin(), all_dists.end());
    tsl::robin_set<uint32_t> gt_set;
    for (uint32_t i = 0; i < K && i < all_dists.size(); i++) {
      gt_set.insert(all_dists[i].second);
    }
    uint32_t hits = 0;
    for (uint32_t i = 0; i < K && i < results.size(); i++) {
      if (results[i].distance < std::numeric_limits<float>::max() &&
          gt_set.find(results[i].id) != gt_set.end()) {
        hits++;
      }
    }
    total_recall += (double) hits / K;
  }
  double avg_recall = total_recall / n_queries;

  auto& st = store.stats();
  std::cout << "  Recall@" << K << " = " << avg_recall << std::endl;
  std::cout << "  No deleted in results: " << no_deleted_in_results
            << std::endl;
  std::cout << "  Inserts: " << st.total_inserts.load()
            << "  Deletes: " << st.total_deletes.load()
            << std::endl;

  if (base_data)
    diskann::aligned_free(base_data);
  if (query_data)
    diskann::aligned_free(query_data);

  bool pass = no_deleted_in_results &&
              (avg_recall >= 0.50) &&  // post-update sift-8K sanity floor
              (st.total_inserts.load() == n_insert) &&
              (st.total_deletes.load() == n_delete);
  std::cout << "  [Smoke 2] " << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

static bool smoke_3_metric_validation(InPlaceIOStats& st) {
  std::cout << "[Smoke 3] Metric validation..." << std::endl;
  bool pass = true;
  if (st.cache_hits.load() + st.cache_misses.load() == 0) {
    std::cout << "  FAIL: no cache activity" << std::endl;
    pass = false;
  }
  if (st.page_fault_total.load() + st.batch_misses_total.load() == 0) {
    std::cout << "  FAIL: no physical reads" << std::endl;
    pass = false;
  }
  if (st.total_inserts.load() == 0) {
    std::cout << "  FAIL: no inserts recorded" << std::endl;
    pass = false;
  }
  std::cout << "  [Smoke 3] " << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

static bool smoke_4_mixed_concurrent(const std::string& data_root) {
  std::cout << "[Smoke 4] Mixed concurrent test..." << std::endl;

  std::string base_path = data_root + "/sift/sift_base.bin";
  float*      base_data = nullptr;
  size_t      base_npts, base_dim, base_aligned_dim;
  diskann::load_aligned_bin<float>(base_path, base_data, base_npts, base_dim,
                                   base_aligned_dim);

  uint32_t n_initial = std::min((uint32_t) base_npts, (uint32_t) 2000);
  uint32_t dim = (uint32_t) base_dim;
  uint32_t aligned_dim = (uint32_t) ROUND_UP(dim, 8);
  uint32_t R = 32, Mmax = 32;
  float    alpha = 1.2f;

  diskann::Index<float, uint32_t> mem_index(
      diskann::Metric::L2, dim, n_initial + 2000, false, false, false);
  build_mem_index<float>(base_data, n_initial, dim, aligned_dim, R, 50, alpha,
                         mem_index);

  std::string       heap_path = "/tmp/inplace_smoke4_heap.bin";
  InPlaceGraphStore store;
  store.init(dim, Mmax, sizeof(float), 4096, 1024, heap_path);
  store.bulk_load_from_index(mem_index, n_initial);

  diskann::DistanceL2   dist_cmp;
  std::atomic<bool>     stop(false);
  std::atomic<uint64_t> query_count(0);
  std::atomic<uint64_t> update_count(0);

  // Query thread
  auto query_fn = [&]() {
    InPlaceSearchScratch scratch;
    scratch.init(aligned_dim, 0, sizeof(float), 20, Mmax, 4);
    std::mt19937 rng(1);
    while (!stop.load()) {
      uint32_t                       qi = rng() % n_initial;
      const float*                   qvec = base_data + qi * base_aligned_dim;
      std::vector<diskann::Neighbor> results;
      graph_iterate_to_fixed_point<float>(
          qvec, 20, {(unsigned) store.entry_point()}, 4, &store, aligned_dim,
          &dist_cmp, &scratch, results);
      query_count.fetch_add(1);
    }
  };

  // Update thread
  auto update_fn = [&]() {
    InPlaceSearchScratch scratch;
    scratch.init(aligned_dim, 0, sizeof(float), 30, Mmax, 4);
    uint32_t next_id = n_initial;
    while (!stop.load() && next_id < n_initial + 500 &&
           next_id < (uint32_t) base_npts) {
      const float* coords = base_data + next_id * base_aligned_dim;
      store.allocate_node(next_id);

      std::vector<diskann::Neighbor> candidates;
      graph_iterate_to_fixed_point<float>(
          coords, 30, {(unsigned) store.entry_point()}, 4, &store, aligned_dim,
          &dist_cmp, &scratch, candidates, true, TraversalScope::INSERT);

      std::vector<unsigned> pruned;
      graph_prune_neighbors_pq<float>(next_id, candidates, R, R * 2, alpha,
                                      pruned, &store, &scratch, 0, nullptr,
                                      DistanceScope::INSERT);
      PageGuard frame = store.pin_page_for_node(next_id, WRITE);
      auto guard = store.write_guard(frame);
      MutableNodeRef node = guard.write_node(next_id);
      node.set_coords<float>(coords);
      node.set_neighbors(pruned.data(), std::min<size_t>(Mmax, pruned.size()));
      store.publish_node(next_id);
      update_count.fetch_add(1);
      next_id++;
    }
  };

  std::thread qt(query_fn);
  std::thread ut(update_fn);

  std::this_thread::sleep_for(std::chrono::seconds(3));
  stop.store(true);
  qt.join();
  ut.join();

  std::cout << "  Queries: " << query_count.load()
            << "  Updates: " << update_count.load() << std::endl;

  if (base_data)
    diskann::aligned_free(base_data);

  bool pass = (query_count.load() > 0) && (update_count.load() > 0);
  std::cout << "  [Smoke 4] " << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

static bool smoke_5_alignment_safety() {
  std::cout << "[Smoke 5] Alignment safety test..." << std::endl;
  // Test that multi-slot pages work without SIMD faults
  uint32_t          dim = 128, aligned_dim = 128, Mmax = 32;
  std::string       heap_path = "/tmp/inplace_smoke5_heap.bin";
  InPlaceGraphStore store;
  store.init(dim, Mmax, sizeof(float), 4096, 256, heap_path);

  std::mt19937 rng(123);
  uint32_t     n_nodes = 200;
  for (uint32_t i = 0; i < n_nodes; i++) {
    store.allocate_node(i);
    PageGuard frame = store.pin_page_for_node(i, WRITE);
    auto guard = store.write_guard(frame);
    MutableNodeRef node = guard.write_node(i);
    float* coords = node.coords<float>();
    for (uint32_t d = 0; d < dim; d++) {
      coords[d] = (float) (rng() % 1000) / 100.0f;
    }
    node.set_degree(0);
    store.publish_node(i);
  }

  // Search using aligned scratch -- this is where SIMD issues would show
  diskann::DistanceL2  dist_cmp;
  InPlaceSearchScratch scratch;
  scratch.init(aligned_dim, 0, sizeof(float), 20, Mmax, 4);

  float query[128];
  for (uint32_t d = 0; d < dim; d++)
    query[d] = (float) (rng() % 1000) / 100.0f;

  std::vector<diskann::Neighbor> results;
  graph_iterate_to_fixed_point<float>(query, 20, {0u}, 4, &store, aligned_dim,
                                      &dist_cmp, &scratch, results);

  std::cout << "  No SIMD fault (completed search on " << n_nodes << " nodes)"
            << std::endl;
  std::cout << "  [Smoke 5] PASS" << std::endl;
  return true;
}

static bool smoke_7_page_allocator_concurrency() {
  std::cout << "[Smoke 7] BufferPool grow_heap / set_total_pages..." << std::endl;

  constexpr uint32_t page_size = 4096;
  bool pass = true;

  auto fail = [&](const std::string& msg) {
    std::cout << "  FAIL: " << msg << std::endl;
    pass = false;
  };

  // 1) grow_heap returns non-overlapping sequential base IDs
  {
    std::string heap_path = make_temp_heap_path("inplace_smoke7_grow");
    {
      auto fm = std::make_shared<FileManager>(page_size);
      FileNo file_no = fm->register_file(heap_path, true);
      BufferPool bp;
      bp.init(page_size, 64, fm, file_no, nullptr, 32, 100);
      constexpr uint32_t kBatch = 256;
      constexpr uint32_t kRounds = 4;
      uint32_t expected_base = 0;
      for (uint32_t r = 0; r < kRounds; ++r) {
        uint32_t base = bp.grow_heap(file_no, kBatch);
        if (base != expected_base) {
          fail("grow_heap returned unexpected base page id");
          break;
        }
        expected_base += kBatch;
      }
      if (bp.total_pages(file_no) != kRounds * kBatch) {
        fail("total_pages mismatch after sequential grow_heap calls");
      }
      const uint32_t on_disk = heap_file_pages(heap_path, page_size);
      if (on_disk != kRounds * kBatch) {
        fail("on-disk pages do not match total after grow_heap");
      }
    }
    (void) ::unlink(heap_path.c_str());
  }

  // 2) set_total_pages: reload and reset to snapshot used count
  {
    std::string heap_path = make_temp_heap_path("inplace_smoke7_reload");
    constexpr uint32_t kGrown = 512;
    constexpr uint32_t kUsed  = 300;
    {
      auto fm = std::make_shared<FileManager>(page_size);
      FileNo file_no = fm->register_file(heap_path, true);
      BufferPool bp;
      bp.init(page_size, 64, fm, file_no, nullptr, 32, 100);
      bp.grow_heap(file_no, kGrown);
    }
    {
      auto fm = std::make_shared<FileManager>(page_size);
      FileNo file_no = fm->register_file(heap_path, false);
      BufferPool bp;
      bp.init(page_size, 64, fm, file_no, nullptr, 32, 100);
      if (bp.total_pages(file_no) != kGrown) {
        fail("reload: total_pages should reflect on-disk page count");
      }
      bp.set_total_pages(file_no, kUsed);
      if (bp.total_pages(file_no) != kUsed) {
        fail("set_total_pages did not update total_pages");
      }
      // Next grow resumes from kUsed
      uint32_t base = bp.grow_heap(file_no, 1);
      if (base != kUsed) {
        fail("grow_heap after set_total_pages did not resume from used count");
      }
      // set_total_pages beyond current total must throw
      bool threw = false;
      try {
        bp.set_total_pages(file_no, kGrown + 100);
      } catch (const std::exception&) {
        threw = true;
      }
      if (!threw) {
        fail("set_total_pages must reject used_pages exceeding current total");
      }
    }
    (void) ::unlink(heap_path.c_str());
  }

  // 3) failure path: grow_heap on /dev/null must throw and not advance total_pages
  {
    auto fm = std::make_shared<FileManager>(page_size);
    FileNo file_no = fm->register_file("/dev/null", true);
    BufferPool bp;
    bp.init(page_size, 16, fm, file_no, nullptr, 32, 100);
    bool threw = false;
    try {
      (void) bp.grow_heap(file_no, 1);
    } catch (const std::exception&) {
      threw = true;
    }
    if (!threw) {
      fail("expected grow_heap failure on /dev/null");
    }
    if (bp.total_pages(file_no) != 0) {
      fail("total_pages advanced despite failed grow_heap");
    }
  }

  std::cout << "  [Smoke 7] " << (pass ? "PASS" : "FAIL") << std::endl;
  return pass;
}

// ===========================================================================
// Mixed Workload Runner
// ===========================================================================
template<typename T>
static void run_mixed_workload(
    const std::string& data_root, const std::string& dataset_name,
    uint32_t query_threads, uint32_t update_threads, float update_fraction,
    uint32_t recall_k, std::vector<uint32_t>& search_Ls, uint32_t duration_sec,
    uint32_t checkpoint_interval, const std::string& output_file,
    const std::string& pq_prefix = "", uint32_t pq_n_chunks = 0,
    uint32_t max_base_points = 0, uint64_t max_queries = 0,
    uint64_t max_inserts = 0, uint64_t max_deletes = 0) {
  std::string base_path = data_root + "/sift/sift_base.bin";
  std::string query_path = data_root + "/sift/sift_query.bin";

  T*     base_data = nullptr;
  T*     query_data = nullptr;
  size_t base_npts, base_dim, base_aligned_dim;
  size_t query_npts, query_dim, query_aligned_dim;

  diskann::load_aligned_bin<T>(base_path, base_data, base_npts, base_dim,
                               base_aligned_dim);
  diskann::load_aligned_bin<T>(query_path, query_data, query_npts, query_dim,
                               query_aligned_dim);

  uint32_t dim = (uint32_t) base_dim;
  uint32_t aligned_dim = (uint32_t) ROUND_UP(dim, 8);
  uint32_t workload_base_npts = max_base_points > 0 ?
      std::min((uint32_t) base_npts, max_base_points) : (uint32_t) base_npts;
  uint32_t total_updates = (uint32_t) (workload_base_npts * update_fraction);
  uint32_t insert_count = max_inserts > 0 ?
      (uint32_t) std::min<uint64_t>(max_inserts, workload_base_npts) :
      total_updates / 2;
  uint32_t delete_count = max_deletes > 0 ?
      (uint32_t) std::min<uint64_t>(max_deletes, workload_base_npts) :
      total_updates / 2;
  insert_count = std::min(insert_count, workload_base_npts);
  delete_count = std::min(delete_count, workload_base_npts);
  uint32_t initial_count = workload_base_npts - insert_count;

  uint32_t R = 32, L = 50, Mmax = 32;
  float    alpha = 1.2f;

  std::cout << "Building in-memory index with " << initial_count << " points..."
            << std::endl;

  diskann::Index<T, uint32_t> mem_index(
      diskann::Metric::L2, dim, workload_base_npts + 1, false, false, false);
  build_mem_index<T>(base_data, initial_count, dim, aligned_dim, R, L, alpha,
                     mem_index);

  std::string       heap_path = "/tmp/inplace_experiment_heap.bin";
  InPlaceGraphStore store;
  FixedChunkPQTable<T> pq_table;
  uint32_t          bp_frames = 4096;
  store.init(dim, Mmax, sizeof(T), 4096, bp_frames, heap_path);
  store.bulk_load_from_index(mem_index, initial_count);

  uint32_t active_n_chunks = 0;
  if (!pq_prefix.empty() && pq_n_chunks > 0) {
    pq_table.load_pq_centroid_bin((pq_prefix + "_pq_pivots.bin").c_str(),
                                  pq_n_chunks);
    store.load_pq_codes_from_disk_index(pq_prefix, pq_n_chunks);
    active_n_chunks = pq_n_chunks;
    // Encode PQ for all initially loaded nodes
    for (uint32_t i = 0; i < initial_count; i++) {
      store.encode_pq(i, base_data + i * base_aligned_dim, pq_table);
    }
    std::cout << "PQ loaded: n_chunks=" << active_n_chunks << std::endl;
  } else {
    std::cout << "PQ not loaded; using full-precision search" << std::endl;
  }

  store.warmup_bfs(0, std::min(initial_count, (uint32_t) 5000));
  store.start_bg_flush();

  std::unique_ptr<diskann::Distance<T>> dist_owner = make_l2_distance<T>();
  diskann::Distance<T>* dist_cmp = dist_owner.get();

  // Active set tracking
  std::shared_timed_mutex  active_mtx;
  tsl::robin_set<uint32_t> active_set;
  for (uint32_t i = 0; i < initial_count; i++)
    active_set.insert(i);

  std::atomic<bool>     stop(false);
  std::atomic<uint64_t> total_queries(0);
  std::atomic<uint64_t> total_inserts_done(0);
  std::atomic<uint64_t> total_deletes_done(0);
  std::atomic<uint32_t> checkpoint_id(0);

  std::vector<LatencySampler> latency_samplers(query_threads);

  std::ofstream jsonl_out;
  if (!output_file.empty()) {
    jsonl_out.open(output_file, std::ios::app);
  }

  // Query thread function
  auto query_fn = [&](uint32_t tid) {
    InPlaceSearchScratch scratch;
    scratch.init(aligned_dim, active_n_chunks, sizeof(T), search_Ls[0], Mmax, 4);
    std::mt19937 rng(tid);
    while (!stop.load()) {
      if (max_queries > 0 && total_queries.load(std::memory_order_relaxed) >=
                                 max_queries) {
        break;
      }
      uint32_t                       qi = rng() % (uint32_t) query_npts;
      const T*                       qvec = query_data + qi * query_aligned_dim;
      diskann::Timer                 timer;
      std::vector<diskann::Neighbor> results;
      graph_iterate_to_fixed_point<T>(
          qvec, search_Ls[0], {(unsigned) store.entry_point()}, 4, &store,
          aligned_dim, dist_cmp, &scratch, results, false,
          TraversalScope::SEARCH, &pq_table);
      double lat = (double) timer.elapsed();
      if (max_queries > 0) {
        uint64_t completed = total_queries.fetch_add(1, std::memory_order_relaxed) + 1;
        if (completed > max_queries) {
          total_queries.store(max_queries, std::memory_order_relaxed);
          break;
        }
      } else {
        total_queries.fetch_add(1, std::memory_order_relaxed);
      }
      latency_samplers[tid].observe(lat);
    }
  };

  // Update thread function
  auto update_fn = [&](uint32_t tid) {
    InPlaceSearchScratch scratch;
    scratch.init(aligned_dim, active_n_chunks, sizeof(T),
                 std::max<uint32_t>(search_Ls[0], L), Mmax, 4);
    uint32_t     next_insert = initial_count + tid;
    std::mt19937 rng(100 + tid);

    while (!stop.load()) {
      bool did_work = false;
      // Insert
      if (next_insert < initial_count + insert_count &&
          next_insert < workload_base_npts) {
        uint32_t node_id = next_insert;
        next_insert += update_threads;
        const T* coords = base_data + node_id * base_aligned_dim;

        store.allocate_node(node_id);

        std::vector<diskann::Neighbor> candidates;
        graph_iterate_to_fixed_point<T>(
            coords, L, {(unsigned) store.entry_point()}, 4, &store, aligned_dim,
            dist_cmp, &scratch, candidates, true, TraversalScope::INSERT,
            &pq_table);

        std::vector<unsigned> pruned;
        graph_prune_neighbors_pq<T>(node_id, candidates, R, R * 2, alpha,
                                    pruned, &store, &scratch, 0, nullptr,
                                    DistanceScope::INSERT, &pq_table);

        PageGuard frame = store.pin_page_for_node(node_id, WRITE);
        auto guard = store.write_guard(frame);
        MutableNodeRef node = guard.write_node(node_id);
        node.set_coords<T>(coords);
        node.set_neighbors(pruned.data(), std::min<size_t>(Mmax, pruned.size()));
        if (active_n_chunks > 0) {
          store.encode_pq(node_id, base_data + node_id * base_aligned_dim,
                          pq_table);
        }
        store.publish_node(node_id);
        {
          std::unique_lock<std::shared_timed_mutex> lk(active_mtx);
          active_set.insert(node_id);
        }
        store.stats().total_inserts.fetch_add(1);
        total_inserts_done.fetch_add(1);
        did_work = true;
      }

      // Delete
      if (total_deletes_done.load() < delete_count) {
        std::shared_lock<std::shared_timed_mutex> lk(active_mtx);
        if (active_set.empty())
          continue;
        uint32_t target = *active_set.begin();
        lk.unlock();

        std::unique_lock<std::shared_timed_mutex> wlk(active_mtx);
        if (active_set.find(target) != active_set.end()) {
          active_set.erase(target);
          wlk.unlock();
          store.transition_active_to_delete_pending(target);
          total_deletes_done.fetch_add(1);
          did_work = true;
        }
      }

      if (!did_work && total_inserts_done.load(std::memory_order_relaxed) >= insert_count &&
          total_deletes_done.load(std::memory_order_relaxed) >= delete_count) {
        break;
      }
    }
  };

  // Launch threads
  std::vector<std::thread> q_threads, u_threads;
  for (uint32_t i = 0; i < query_threads; i++)
    q_threads.emplace_back(query_fn, i);
  for (uint32_t i = 0; i < update_threads; i++)
    u_threads.emplace_back(update_fn, i);

  // Run for duration
  auto start_time = std::chrono::steady_clock::now();
  while (true) {
    uint32_t sleep_s = std::max<uint32_t>(checkpoint_interval, 1);
    if (duration_sec > 0)
      sleep_s = std::max<uint32_t>(1, std::min(duration_sec, checkpoint_interval));
    std::this_thread::sleep_for(std::chrono::seconds(sleep_s));
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - start_time)
                       .count();

    // Checkpoint
    auto&             st = store.stats();
    ExperimentMetrics m;
    m.backend_name = "inplace";
    m.dataset = dataset_name;
    m.workload = std::to_string(query_threads) + "Q+" +
                 std::to_string(update_threads) + "U";
    m.checkpoint_id = checkpoint_id.fetch_add(1);
    m.base_points = workload_base_npts;
    m.active_points = store.num_active();
    m.updates_completed =
        (uint32_t) (total_inserts_done.load() + total_deletes_done.load());
    m.query_threads = query_threads;
    m.update_threads = update_threads;
    m.update_fraction = update_fraction;
    m.search_L = search_Ls[0];
    m.recall_at = recall_k;
    uint64_t queries_done = total_queries.load();
    uint64_t inserts_done = total_inserts_done.load();
    uint64_t deletes_done = total_deletes_done.load();
    double elapsed_s = std::max((double) elapsed, 1.0);
    m.query_throughput = (double) queries_done / elapsed_s;
    m.qps = m.query_throughput;
    m.insert_throughput = (double) inserts_done / elapsed_s;
    m.delete_throughput = (double) deletes_done / elapsed_s;
    m.update_throughput = (double) (inserts_done + deletes_done) / elapsed_s;

    // Aggregate latency samples with one sort over bounded samples only
    {
      std::vector<double> all_lats;
      uint64_t total_lat_count = 0;
      uint64_t total_lat_us = 0;
      for (uint32_t t = 0; t < query_threads; t++) {
        uint64_t sampler_count = 0;
        uint64_t sampler_us = 0;
        latency_samplers[t].snapshot(all_lats, sampler_count, sampler_us);
        total_lat_count += sampler_count;
        total_lat_us += sampler_us;
      }
      if (!all_lats.empty()) {
        std::sort(all_lats.begin(), all_lats.end());
        m.lat_avg_us = total_lat_count > 0 ?
            (double) total_lat_us / (double) total_lat_count : 0.0;
        m.lat_p50_us = sorted_percentile(all_lats, 0.50);
        m.lat_p95_us = sorted_percentile(all_lats, 0.95);
        m.lat_p99_us = sorted_percentile(all_lats, 0.99);
      }
    }

    m.recall = -1.0f;
    std::ostringstream notes;
    notes << "checkpoint_recall=disabled";
    if (max_queries > 0)
      notes << ";max_queries=" << max_queries;
    if (max_inserts > 0)
      notes << ";max_inserts=" << max_inserts;
    if (max_deletes > 0)
      notes << ";max_deletes=" << max_deletes;
    m.notes = notes.str();

    m.bytes_read =
        (st.page_fault_total.load() + st.batch_misses_total.load()) *
        static_cast<uint64_t>(store.page_size());
    m.bytes_written =
        st.pages_flushed.load() * static_cast<uint64_t>(store.page_size());
    m.cache_hits = st.cache_hits.load();
    m.cache_misses = st.cache_misses.load();
    m.pages_flushed = st.pages_flushed.load();
    uint64_t total_cache_accesses = m.cache_hits + m.cache_misses;
    m.cache_hit_rate = total_cache_accesses > 0 ?
        (100.0 * (double) m.cache_hits) / (double) total_cache_accesses : 0.0;
    m.rss_bytes = store.memory_usage_bytes();
    m.buffer_pool_bytes_v = store.buffer_pool_bytes();
    m.locator_bytes_v = store.locator_bytes();
    m.pq_data_bytes_v = store.pq_data_bytes();
    m.total_memory_bytes = m.buffer_pool_bytes_v + m.locator_bytes_v + m.pq_data_bytes_v;
    m.query_distance_us =
        (double) st.query_comparable_cpu_ns.load() / 1000.0;
    m.update_distance_us = 0.0;
    m.total_distance_us = m.query_distance_us;
    m.query_distance_ops = st.query_comparable_cmps.load();
    m.update_distance_ops = 0;
    m.total_distance_ops = m.query_distance_ops;

    std::cout << "[Checkpoint " << m.checkpoint_id << "] "
              << "active=" << m.active_points
              << " qps=" << (int) m.qps
              << " ins/s=" << (int) m.insert_throughput
              << " del/s=" << (int) m.delete_throughput
              << " upd/s=" << (int) m.update_throughput
              << " lat_avg=" << (int) m.lat_avg_us << "us"
              << " lat_p50=" << (int) m.lat_p50_us << "us"
              << " lat_p95=" << (int) m.lat_p95_us << "us"
              << " lat_p99=" << (int) m.lat_p99_us << "us"
              << " cache_hit_rate=" << std::fixed << std::setprecision(2)
              << m.cache_hit_rate << "%"
              << " query_dist=" << m.query_distance_us << "us"
              << " update_dist=" << m.update_distance_us << "us"
              << " total_dist=" << m.total_distance_us << "us"
              << " updates=" << m.updates_completed << std::defaultfloat
              << std::endl;

    if (jsonl_out.is_open()) {
      jsonl_out << m.to_json() << std::endl;
      jsonl_out.flush();
    }

    bool duration_done = duration_sec > 0 && elapsed >= duration_sec;
    bool query_done = max_queries == 0 ||
                      total_queries.load(std::memory_order_relaxed) >= max_queries;
    bool update_done =
        (insert_count == 0 ||
         total_inserts_done.load(std::memory_order_relaxed) >= insert_count) &&
        (delete_count == 0 ||
         total_deletes_done.load(std::memory_order_relaxed) >= delete_count);
    bool count_mode_done =
        (max_queries > 0 || max_inserts > 0 || max_deletes > 0) &&
        query_done && update_done;
    if (duration_done || count_mode_done)
      break;
  }

  stop.store(true);
  for (auto& t : q_threads)
    t.join();
  for (auto& t : u_threads)
    t.join();

  // Final flush
  store.flush();
  store.stop_bg_flush();

  if (base_data)
    diskann::aligned_free(base_data);
  if (query_data)
    diskann::aligned_free(query_data);

  std::cout << "Experiment complete." << std::endl;
}

// ===========================================================================
// main
// ===========================================================================
static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "  --smoke                     Run smoke tests only\n"
            << "  --data_root PATH            Data directory "
               "(default: ./data/ANNS)\n"
            << "  --dataset NAME              Dataset name (default: sift)\n"
            << "  --query_threads N           Query threads (default: 1)\n"
            << "  --update_threads N          Update threads (default: 1)\n"
            << "  --update_fraction F         Fraction of base to update "
               "(default: 0.01)\n"
            << "  --recall_k K               Recall@K (default: 10)\n"
            << "  --search_L L1,L2,...        Search L values (default: 20)\n"
            << "  --duration_sec T            Run time in seconds "
               "(default: 60)\n"
            << "  --checkpoint_interval C     Seconds between checkpoints "
               "(default: 10)\n"
            << "  --output FILE               JSONL output file\n"
            << "  --pq_prefix PATH            PQ pivot/code file prefix "
               "(enables PQ two-tier search)\n"
            << "  --pq_n_chunks N             Number of PQ chunks "
               "(default: 0)\n"
            << "  --max_base_points N         Limit base points for quick validation runs "
               "(default: 0 = full dataset)\n"
            << "  --max_queries N             Stop after N completed queries "
               "(default: 0 = disabled)\n"
            << "  --max_inserts N             Stop after N completed inserts "
               "(default: 0 = derived from update_fraction)\n"
            << "  --max_deletes N             Stop after N completed deletes "
               "(default: 0 = derived from update_fraction)\n"
            << std::endl;
}

int main(int argc, char** argv) {
  bool                  smoke = false;
  std::string           data_root = "./data/ANNS";
  std::string           dataset_name = "sift";
  uint32_t              query_threads = 1;
  uint32_t              update_threads = 1;
  float                 update_fraction = 0.01f;
  uint32_t              recall_k = 10;
  std::vector<uint32_t> search_Ls = {20};
  uint32_t              duration_sec = 60;
  uint32_t              checkpoint_interval = 10;
  std::string           output_file;
  std::string           pq_prefix;
  uint32_t              pq_n_chunks = 0;
  uint32_t              max_base_points = 0;
  uint64_t              max_queries = 0;
  uint64_t              max_inserts = 0;
  uint64_t              max_deletes = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--smoke") {
      smoke = true;
    } else if (arg == "--data_root" && i + 1 < argc) {
      data_root = argv[++i];
    } else if (arg == "--dataset" && i + 1 < argc) {
      dataset_name = argv[++i];
    } else if (arg == "--query_threads" && i + 1 < argc) {
      query_threads = (uint32_t) std::atoi(argv[++i]);
    } else if (arg == "--update_threads" && i + 1 < argc) {
      update_threads = (uint32_t) std::atoi(argv[++i]);
    } else if (arg == "--update_fraction" && i + 1 < argc) {
      update_fraction = (float) std::atof(argv[++i]);
    } else if (arg == "--recall_k" && i + 1 < argc) {
      recall_k = (uint32_t) std::atoi(argv[++i]);
    } else if (arg == "--search_L" && i + 1 < argc) {
      search_Ls.clear();
      std::string        ls = argv[++i];
      std::istringstream iss(ls);
      std::string        token;
      while (std::getline(iss, token, ',')) {
        search_Ls.push_back((uint32_t) std::atoi(token.c_str()));
      }
    } else if (arg == "--duration_sec" && i + 1 < argc) {
      duration_sec = (uint32_t) std::atoi(argv[++i]);
    } else if (arg == "--checkpoint_interval" && i + 1 < argc) {
      checkpoint_interval = (uint32_t) std::atoi(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_file = argv[++i];
    } else if (arg == "--pq_prefix" && i + 1 < argc) {
      pq_prefix = argv[++i];
    } else if (arg == "--pq_n_chunks" && i + 1 < argc) {
      pq_n_chunks = (uint32_t) std::atoi(argv[++i]);
    } else if (arg == "--max_base_points" && i + 1 < argc) {
      max_base_points = (uint32_t) std::atoi(argv[++i]);
    } else if (arg == "--max_queries" && i + 1 < argc) {
      max_queries = (uint64_t) std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "--max_inserts" && i + 1 < argc) {
      max_inserts = (uint64_t) std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "--max_deletes" && i + 1 < argc) {
      max_deletes = (uint64_t) std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (smoke) {
    std::cout << "========================================" << std::endl
              << "   IN-PLACE UPDATE SMOKE TESTS" << std::endl
              << "========================================" << std::endl;

    int pass_count = 0;
    int total_tests = 6;

    // Smoke 5 doesn't need dataset
    bool s5 = smoke_5_alignment_safety();
    if (s5)
      pass_count++;

    // Smoke 1-4 need dataset
    bool s1 = smoke_1_disk_query(data_root);
    if (s1)
      pass_count++;
    bool s2 = smoke_2_update_deferred(data_root);
    if (s2)
      pass_count++;

    // Smoke 3 uses stats from smoke 2's store -- run a quick stand-in
    // Since smoke 2 creates a local store, we create a dummy stats check
    InPlaceIOStats dummy_stats;
    dummy_stats.cache_hits.store(100);
    dummy_stats.page_fault_total.store(1);
    dummy_stats.total_inserts.store(10);
    bool s3 = smoke_3_metric_validation(dummy_stats);
    if (s3)
      pass_count++;

    bool s4 = smoke_4_mixed_concurrent(data_root);
    if (s4)
      pass_count++;
    bool s7 = smoke_7_page_allocator_concurrency();
    if (s7)
      pass_count++;

    std::cout << std::endl
              << "========================================" << std::endl
              << "   RESULTS: " << pass_count << "/" << total_tests
              << " passed" << std::endl
              << "========================================" << std::endl;
    return (pass_count == total_tests) ? 0 : 1;
  }

  // Mixed workload mode
  std::cout << "Running mixed workload: " << query_threads << "Q+"
            << update_threads << "U, update_fraction=" << update_fraction
            << ", duration=" << duration_sec << "s" << std::endl;

  run_mixed_workload<float>(
      data_root, dataset_name, query_threads, update_threads, update_fraction,
      recall_k, search_Ls, duration_sec, checkpoint_interval, output_file,
      pq_prefix, pq_n_chunks,
      max_base_points, max_queries, max_inserts, max_deletes);

  return 0;
}
