// Harness insert/delete/update driver for DGAI, ported from the benchmark harness
// PipeANN/tests/test_insert_or_delete.cpp (insert, delete) and test_mix_update.cpp (update) so the systems run
// identical workload semantics and emit the same ann_bench JSON records. Differences:
//   - DGAI's DynamicSSDIndex takes a search mode and strategy, and queries go
//     through the same rerank search as the query workload.
//   - Delete consolidation is trigger_deletion(nthreads, 20), which is what
//     the authors' overall_performance.cpp calls under USE_TOPO_DISK. It works
//     in place, so there is no merge prefix and no alternating index dir.
//   - The io_* fields come from hooks in DGAI's file reader (single-page requests count as random,
//     multi-page ones as sequential).
//   - Update mode interleaves inserts and lazy deletes through the shared WeightedUpdatePool, then
//     consolidates the deletes with trigger_deletion. Its ids_file argument is "<insert_ids>:<delete_ids>",
//     and count_per_iter is both the insert and the delete count of each round.
//   - With SKIP_UPDATE_SEARCH=0 (RQ2), update mode also runs PipeANN's concurrent queries: a query stream in
//     the pool sized by QUERY_RATIO, MERGE_QUERY_THREADS query threads beside a trigger_deletion that uses
//     MERGE_THREADS, and a post-merge tail that finishes MERGE_QUERY_BUDGET. These knobs come from the
//     environment, as MERGE_QUERY_BUDGET and UPDATE_POOL_MODE already do in PipeANN's driver, so the insert
//     and delete command lines are unchanged. TAIL_QUERY_BEGIN and TAIL_QUERY_END add base vectors
//     [begin, end) to the cyclic query pool, matching PipeANN's BASE_POINTS..TOTAL_POINTS.
//   - Merge-only mode is dropped.
#include "v2/dynamic_index.h"

#include <omp.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "distance.h"
#include "utils.h"
#include "aux_utils.h"
#include "percentile_stats.h"
#include "global_stats.h"
#include "ann_bench_metrics.h"
#include "streaming_query_source.h"
#include "weighted_update_pool.h"

namespace {

enum class WorkloadMode { INSERT, DELETE, UPDATE };

struct Config {
  WorkloadMode mode;
  std::string type;
  std::string full_data_bin;
  unsigned l_disk;
  std::string base_prefix;
  size_t n_iters;
  uint32_t count_per_iter;
  uint32_t range;
  uint32_t beamwidth;
  uint32_t nthreads;
  std::string ids_file;
  std::string query_bin;
  std::string gt_dir;  // dir of per-round GT: round_<k>_gt<recall_at>.bin
  uint32_t query_threads;
  uint32_t recall_at;
  uint32_t search_l;
  int search_mode;
  int strategy;
  // Update-mode concurrent queries (RQ2), read from the environment.
  bool concurrent_queries;
  uint32_t query_ratio;
  uint32_t merge_threads;
  uint32_t merge_query_threads;
  bool enable_interval_metrics;
  uint64_t tail_query_begin;
  uint64_t tail_query_end;
};

uint64_t parse_u64(const char *value, const std::string &name) {
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::invalid_argument("invalid integer for " + name + ": " + value);
  }
  return static_cast<uint64_t>(parsed);
}

uint32_t parse_u32(const char *value, const std::string &name) {
  uint64_t parsed = parse_u64(value, name);
  if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid uint32 for " + name + ": " + value);
  }
  return static_cast<uint32_t>(parsed);
}

WorkloadMode parse_mode(const std::string &mode) {
  if (mode == "insert") {
    return WorkloadMode::INSERT;
  }
  if (mode == "delete") {
    return WorkloadMode::DELETE;
  }
  if (mode == "update") {
    return WorkloadMode::UPDATE;
  }
  throw std::invalid_argument("mode must be insert, delete or update");
}

Config parse_args(int argc, char **argv) {
  if (argc != 19) {
    std::cerr << "Correct usage: " << argv[0]
              << " <mode[insert/delete/update]> <type[int8/uint8/float]> "
                 "<full_data_bin>"
              << " <L_disk> <index_prefix>"
              << " <n_iters> <count_per_iter> <range> <beamwidth> <nthreads>"
              << " <ids_file[update: insert_ids:delete_ids]>"
              << " <query_bin[\"null\" to skip]> <gt_dir> <query_threads>"
              << " <recall_at> <search_l> <search_mode> <strategy>" << std::endl;
    throw std::invalid_argument("wrong number of arguments");
  }

  int arg_no = 1;
  Config cfg;
  cfg.mode = parse_mode(argv[arg_no++]);
  cfg.type = argv[arg_no++];
  cfg.full_data_bin = argv[arg_no++];
  cfg.l_disk = parse_u32(argv[arg_no++], "L_disk");
  cfg.base_prefix = argv[arg_no++];
  cfg.n_iters = static_cast<size_t>(parse_u64(argv[arg_no++], "n_iters"));
  cfg.count_per_iter = parse_u32(argv[arg_no++], "count_per_iter");
  cfg.range = parse_u32(argv[arg_no++], "range");
  cfg.beamwidth = parse_u32(argv[arg_no++], "beamwidth");
  cfg.nthreads = parse_u32(argv[arg_no++], "nthreads");
  cfg.ids_file = argv[arg_no++];
  cfg.query_bin = argv[arg_no++];
  cfg.gt_dir = argv[arg_no++];
  cfg.query_threads = parse_u32(argv[arg_no++], "query_threads");
  cfg.recall_at = parse_u32(argv[arg_no++], "recall_at");
  cfg.search_l = parse_u32(argv[arg_no++], "search_l");
  cfg.search_mode = static_cast<int>(parse_u64(argv[arg_no++], "search_mode"));
  cfg.strategy = static_cast<int>(parse_u64(argv[arg_no++], "strategy"));
  if (cfg.n_iters == 0) {
    throw std::invalid_argument("n_iters must be positive");
  }

  cfg.concurrent_queries = cfg.mode == WorkloadMode::UPDATE && ann_bench::env_u64_or("SKIP_UPDATE_SEARCH", 1) == 0;
  cfg.query_ratio = 0;
  cfg.merge_threads = cfg.nthreads;
  cfg.merge_query_threads = 0;
  cfg.enable_interval_metrics = false;
  cfg.tail_query_begin = cfg.tail_query_end = 0;
  if (cfg.concurrent_queries) {
    cfg.query_ratio = static_cast<uint32_t>(ann_bench::env_u64_or("QUERY_RATIO", 0));
    cfg.merge_threads = static_cast<uint32_t>(ann_bench::env_u64_or("MERGE_THREADS", cfg.nthreads));
    cfg.merge_query_threads = static_cast<uint32_t>(ann_bench::env_u64_or("MERGE_QUERY_THREADS", 0));
    cfg.enable_interval_metrics = ann_bench::env_u64_or("ENABLE_INTERVAL_METRICS", 0) != 0;
    cfg.tail_query_begin = ann_bench::env_u64_or("TAIL_QUERY_BEGIN", 0);
    cfg.tail_query_end = ann_bench::env_u64_or("TAIL_QUERY_END", 0);
    if (cfg.query_ratio == 0 || cfg.query_ratio >= 100) {
      throw std::invalid_argument("QUERY_RATIO must be in [1, 99] when SKIP_UPDATE_SEARCH=0");
    }
    if (cfg.merge_threads == 0 || cfg.merge_threads + cfg.merge_query_threads > cfg.nthreads) {
      throw std::invalid_argument("need 0 < MERGE_THREADS and MERGE_THREADS + MERGE_QUERY_THREADS <= nthreads");
    }
    if (cfg.query_bin == "null") {
      throw std::invalid_argument("concurrent queries need a query_bin");
    }
  }
  return cfg;
}

void read_bin_metadata(const std::string &path, uint32_t &npts, uint32_t &dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open data file: " + path);
  }

  int32_t file_npts = 0;
  int32_t file_dim = 0;
  in.read(reinterpret_cast<char *>(&file_npts), sizeof(int32_t));
  in.read(reinterpret_cast<char *>(&file_dim), sizeof(int32_t));
  if (!in || file_npts <= 0 || file_dim <= 0) {
    throw std::runtime_error("invalid bin header: " + path);
  }
  npts = static_cast<uint32_t>(file_npts);
  dim = static_cast<uint32_t>(file_dim);
}

std::vector<uint32_t> load_ids(const std::string &path, size_t required_ids) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("cannot open IDs file: " + path);
  }

  const std::streamoff file_size = in.tellg();
  if (file_size < 0 || file_size % static_cast<std::streamoff>(sizeof(uint32_t)) != 0) {
    throw std::runtime_error("invalid IDs file size: " + path);
  }
  in.seekg(0, std::ios::beg);

  int32_t counted_size = 0;
  in.read(reinterpret_cast<char *>(&counted_size), sizeof(int32_t));
  if (!in) {
    throw std::runtime_error("short read from IDs file header: " + path);
  }

  const bool counted =
      counted_size >= 0 &&
      file_size == static_cast<std::streamoff>(sizeof(int32_t)) +
                       static_cast<std::streamoff>(counted_size) *
                           static_cast<std::streamoff>(sizeof(uint32_t));

  size_t count = 0;
  if (counted) {
    count = static_cast<size_t>(counted_size);
  } else {
    count = static_cast<size_t>(file_size / static_cast<std::streamoff>(sizeof(uint32_t)));
    in.clear();
    in.seekg(0, std::ios::beg);
  }

  if (count < required_ids) {
    throw std::runtime_error("IDs file does not contain enough IDs: need " + std::to_string(required_ids) +
                             ", found " + std::to_string(count));
  }

  std::vector<uint32_t> ids(count);
  if (!ids.empty()) {
    in.read(reinterpret_cast<char *>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    if (!in) {
      throw std::runtime_error("short read from IDs file payload: " + path);
    }
  }
  return ids;
}

template<typename T>
std::vector<T> load_insert_batch(const std::string &data_file, const std::vector<uint32_t> &ids, size_t begin,
                                 uint32_t count, uint32_t full_npts, uint32_t dim) {
  std::vector<T> batch(static_cast<size_t>(count) * static_cast<size_t>(dim));
  std::ifstream reader(data_file, std::ios::binary);
  if (!reader) {
    throw std::runtime_error("cannot open full-data file: " + data_file);
  }

  uint32_t file_npts = 0;
  uint32_t file_dim = 0;
  read_bin_metadata(data_file, file_npts, file_dim);
  if (file_npts != full_npts || file_dim != dim) {
    throw std::runtime_error("full-data metadata changed while reading: " + data_file);
  }

  for (uint32_t i = 0; i < count; i++) {
    uint32_t id = ids[begin + i];
    if (id >= full_npts) {
      throw std::runtime_error("insert ID out of range: " + std::to_string(id));
    }
    const uint64_t offset = 2 * sizeof(int32_t) +
                            static_cast<uint64_t>(id) * static_cast<uint64_t>(dim) * sizeof(T);
    reader.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    reader.read(reinterpret_cast<char *>(batch.data() + static_cast<size_t>(i) * dim),
                static_cast<std::streamsize>(dim * sizeof(T)));
    if (!reader) {
      throw std::runtime_error("failed to read vector ID " + std::to_string(id));
    }
  }
  return batch;
}

template<typename T>
void run_workload(const Config &cfg, pipeann::Distance<T> *dist_cmp) {
  uint32_t full_npts = 0;
  uint32_t dim = 0;
  read_bin_metadata(cfg.full_data_bin, full_npts, dim);

  // Same parameter names and values as the authors' overall_performance.cpp,
  // except R (the loader overrides it with the index's real degree anyway)
  // and the thread-data pool, which must cover every concurrent worker.
  pipeann::Parameters paras;
  paras.Set<unsigned>("L_disk", cfg.l_disk);
  paras.Set<unsigned>("R_disk", cfg.range);
  paras.Set<float>("alpha_disk", 1.2f);
  paras.Set<unsigned>("C", 160);
  paras.Set<unsigned>("beamwidth", cfg.beamwidth);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_threads", std::max(cfg.nthreads, cfg.query_threads));

  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, uint32_t> sync_index(paras, cfg.base_prefix, cfg.base_prefix, dist_cmp, metric,
                                                   cfg.search_mode, false, cfg.strategy);

  omp_set_dynamic(0);

  const size_t required_ids = cfg.n_iters * static_cast<size_t>(cfg.count_per_iter);
  const bool update_mode = cfg.mode == WorkloadMode::UPDATE;
  std::vector<uint32_t> ids, delete_ids;
  if (update_mode) {
    const size_t colon = cfg.ids_file.find(':');
    if (colon == std::string::npos) {
      throw std::invalid_argument("update mode needs ids_file as <insert_ids>:<delete_ids>");
    }
    ids = load_ids(cfg.ids_file.substr(0, colon), required_ids);
    delete_ids = load_ids(cfg.ids_file.substr(colon + 1), required_ids);
  } else {
    ids = load_ids(cfg.ids_file, required_ids);
  }
  // Operations per round and in total: an update round does count_per_iter inserts and as many deletes.
  const size_t ops_per_iter = static_cast<size_t>(cfg.count_per_iter) * (update_mode ? 2 : 1);
  const size_t total_ops = cfg.n_iters * ops_per_iter;
  const char *workload_name =
      cfg.mode == WorkloadMode::INSERT ? "insert" : cfg.mode == WorkloadMode::DELETE ? "delete" : "mixed_update";

  // Per-round recall: load queries once and search the live index (mem + disk)
  // after each round via sync_index.search at cfg.search_l, scoring against that
  // round's groundtruth <gt_dir>/round_<k>_gt<recall_at>.bin (generated by the
  // prepare step). The end-of-run final_query block keeps its own load.
  T *rq_query = nullptr;
  size_t rq_num = 0, rq_dim = 0;
  const bool rq_have_queries = (cfg.query_bin != "null");
  if (rq_have_queries) {
    pipeann::load_bin<T>(cfg.query_bin, rq_query, rq_num, rq_dim);
  }
  auto measure_recall = [&](size_t round_idx) -> double {
    if (!rq_have_queries || rq_num == 0) return -1.0;
    const std::string gt_path = cfg.gt_dir + "/round_" +
                                std::to_string(round_idx) + "_gt" +
                                std::to_string(cfg.recall_at) + ".bin";
    if (!file_exists(gt_path)) {
      std::cerr << "WARNING: missing per-round GT " << gt_path
                << "; skipping recall for round " << round_idx << std::endl;
      return -1.0;
    }
    unsigned *gt_ids = nullptr, *gt_eval = nullptr;
    float *gt_dists = nullptr;
    uint32_t *gt_tags = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    pipeann::load_truthset(gt_path, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
    if (gt_num != rq_num) {
      delete[] gt_ids; delete[] gt_dists; delete[] gt_tags;
      return -1.0;
    }
    gt_eval = (gt_tags != nullptr) ? reinterpret_cast<unsigned *>(gt_tags) : gt_ids;
    std::vector<uint32_t> res_tags(rq_num * cfg.recall_at);
    std::vector<float> res_dists(rq_num * cfg.recall_at);
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(cfg.query_threads))
    for (int64_t i = 0; i < static_cast<int64_t>(rq_num); i++) {
      pipeann::QueryStats st;
      sync_index.search(rq_query + static_cast<size_t>(i) * rq_dim, cfg.recall_at, 0,
                        cfg.search_l, cfg.beamwidth,
                        res_tags.data() + static_cast<size_t>(i) * cfg.recall_at,
                        res_dists.data() + static_cast<size_t>(i) * cfg.recall_at, &st);
    }
    const double r = pipeann::calculate_recall(
        static_cast<uint32_t>(rq_num), gt_eval, gt_dists,
        static_cast<uint32_t>(gt_dim), res_tags.data(),
        static_cast<uint32_t>(cfg.recall_at), static_cast<uint32_t>(cfg.recall_at));
    delete[] gt_ids; delete[] gt_dists; delete[] gt_tags;
    return r;
  };

  // Concurrent queries (update mode, SKIP_UPDATE_SEARCH=0), following PipeANN's test_mix_update.cpp.
  ann_bench::StreamingQuerySource<T> query_source;
  if (cfg.concurrent_queries) {
    query_source.load(cfg.query_bin, cfg.full_data_bin, cfg.tail_query_begin, cfg.tail_query_end);
    if (query_source.dim != dim) {
      throw std::runtime_error("query dim mismatch");
    }
  }
  const uint64_t merge_query_budget =
      cfg.concurrent_queries ? ann_bench::env_u64_or("MERGE_QUERY_BUDGET", 0) : 0;
  uint64_t total_query_count = 0;
  double total_query_time_s = 0.0;
  double post_merge_query_time_s = 0.0;
  ann_bench::LatencySampler query_latency_sampler;
  ann_bench::QueryCycleWindowMetrics interval_metrics;
  ann_bench::QueryCycleWindowMetrics *window_metrics_ptr =
      cfg.enable_interval_metrics ? &interval_metrics : nullptr;
  const auto workload_start = std::chrono::steady_clock::now();
  if (window_metrics_ptr != nullptr) {
    window_metrics_ptr->start("DGAI", ann_bench::latency_reserve_hint(static_cast<uint64_t>(query_source.query_n)));
  }
  // One concurrent query: sequence number seq of the cyclic pool, recorded under stage.
  auto run_concurrent_query = [&](uint64_t seq, size_t round_idx, const char *stage,
                                  typename ann_bench::StreamingQuerySource<T>::ThreadState &qstate) {
    qstate.result_tags.resize(cfg.recall_at);
    qstate.result_dists.resize(cfg.recall_at);
    pipeann::QueryStats qstats;
    const T *query = query_source.get(seq, qstate);
    const auto op_begin = std::chrono::steady_clock::now();
    sync_index.search(query, cfg.recall_at, 0, cfg.search_l, cfg.beamwidth, qstate.result_tags.data(),
                      qstate.result_dists.data(), &qstats);
    const double lat_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - op_begin).count();
    if (window_metrics_ptr != nullptr) {
      window_metrics_ptr->record_query(lat_us, round_idx, stage);
    } else {
      query_latency_sampler.record(seq, lat_us);
    }
  };

  double foreground_time_s = 0.0;
  double maintenance_time_s = 0.0;
  double last_recall = -1.0;
  std::vector<double> call_latencies_us;  // insert or delete calls; in update mode the insert calls
  std::vector<double> delete_call_latencies_us;  // update mode only
  call_latencies_us.reserve(required_ids);
  ann_bench::LogicalIoSnapshot logical_io_delta;
  ann_bench::RssSampler rss_sampler;
  for (size_t iter = 0; iter < cfg.n_iters; iter++) {
    const size_t batch_begin = iter * static_cast<size_t>(cfg.count_per_iter);
    std::cout << "ITER " << iter << ": "
              << (cfg.mode == WorkloadMode::INSERT ? "insert " : cfg.mode == WorkloadMode::DELETE ? "delete "
                                                                                                   : "insert+delete ")
              << cfg.count_per_iter << " IDs" << std::endl;

    double round_fg_s = 0.0, round_maint_s = 0.0;
    std::vector<double> batch_latencies_us(cfg.count_per_iter, 0.0);
    std::vector<double> batch_delete_latencies_us(update_mode ? cfg.count_per_iter : 0, 0.0);
    uint64_t round_query_count = 0, round_fg_query_count = 0, round_merge_query_count = 0;
    uint64_t round_post_merge_query_count = 0;
    double round_post_merge_s = 0.0;
    ann_bench::LogicalIoCounter::reset_and_enable();
    if (window_metrics_ptr != nullptr) {
      window_metrics_ptr->reset_io_baseline();
      window_metrics_ptr->resume_rss();
    } else {
      rss_sampler.resume();
    }
    if (gs != nullptr) {
      gs->clear_harness_counters();
    }

    if (cfg.mode == WorkloadMode::INSERT) {
      std::vector<T> batch =
          load_insert_batch<T>(cfg.full_data_bin, ids, batch_begin, cfg.count_per_iter, full_npts, dim);
      std::atomic<uint32_t> failures(0);
      const auto foreground_begin = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(cfg.nthreads))
      for (int64_t i = 0; i < static_cast<int64_t>(cfg.count_per_iter); i++) {
        const uint32_t tag = ids[batch_begin + static_cast<size_t>(i)];
        const auto op_begin = std::chrono::steady_clock::now();
        if (sync_index.insert(batch.data() + static_cast<size_t>(i) * dim, tag) < 0) {
          failures.fetch_add(1);
        }
        const auto op_end = std::chrono::steady_clock::now();
        batch_latencies_us[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::micro>(op_end - op_begin).count();
      }
      const auto foreground_end = std::chrono::steady_clock::now();
      round_fg_s =
          std::chrono::duration<double>(foreground_end - foreground_begin).count();
      if (failures.load() != 0) {
        throw std::runtime_error("failed to insert " + std::to_string(failures.load()) + " points");
      }
      // DGAI inserts in place, so insert-only rounds need no maintenance.
    } else if (update_mode) {
      std::vector<T> batch =
          load_insert_batch<T>(cfg.full_data_bin, ids, batch_begin, cfg.count_per_iter, full_npts, dim);
      const uint64_t query_n =
          cfg.concurrent_queries
              ? ann_bench::mixed_update_query_count(cfg.count_per_iter, cfg.count_per_iter, cfg.query_ratio)
              : 0;
      std::atomic<uint32_t> failures(0);
      std::function<void(uint64_t)> tasks[ann_bench::WeightedUpdatePool::NUM_OPS];
      tasks[ann_bench::WeightedUpdatePool::INSERT] = [&](uint64_t i) {
        const auto op_begin = std::chrono::steady_clock::now();
        if (sync_index.insert(batch.data() + static_cast<size_t>(i) * dim, ids[batch_begin + i]) < 0) {
          failures.fetch_add(1);
        }
        batch_latencies_us[i] =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - op_begin).count();
        if (window_metrics_ptr != nullptr) {
          window_metrics_ptr->record_insert(batch_latencies_us[i]);
        }
      };
      tasks[ann_bench::WeightedUpdatePool::DELETE] = [&](uint64_t i) {
        const auto op_begin = std::chrono::steady_clock::now();
        sync_index.lazy_delete(delete_ids[batch_begin + i]);
        batch_delete_latencies_us[i] =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - op_begin).count();
        if (window_metrics_ptr != nullptr) {
          window_metrics_ptr->record_delete(batch_delete_latencies_us[i]);
        }
      };
      tasks[ann_bench::WeightedUpdatePool::QUERY] = [&](uint64_t i) {
        thread_local typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
        run_concurrent_query(i, iter, "foreground", qstate);
      };
      const uint64_t counts[ann_bench::WeightedUpdatePool::NUM_OPS] = {
          static_cast<uint64_t>(cfg.count_per_iter), static_cast<uint64_t>(cfg.count_per_iter), query_n};
      const auto foreground_begin = std::chrono::steady_clock::now();
      if (cfg.concurrent_queries && ann_bench::env_equals("UPDATE_POOL_MODE", "round_robin")) {
        ann_bench::WeightedUpdatePool::run_rq2_round_robin_90_5_5(cfg.nthreads, counts, tasks);
      } else {
        ann_bench::WeightedUpdatePool::run(cfg.nthreads, counts, tasks);
      }
      const auto foreground_end = std::chrono::steady_clock::now();
      round_fg_s = std::chrono::duration<double>(foreground_end - foreground_begin).count();
      if (failures.load() != 0) {
        throw std::runtime_error("failed to insert " + std::to_string(failures.load()) + " points");
      }
      round_fg_query_count = query_n;
      if (window_metrics_ptr != nullptr) {
        window_metrics_ptr->flush(iter, "foreground");
        ann_bench::emit_phase_checkpoint(
            std::cout, "DGAI", "foreground", iter,
            std::chrono::duration<double>(foreground_end - workload_start).count(), cfg.count_per_iter,
            cfg.count_per_iter, query_n, round_fg_s);
      }

      // Consolidation, with MERGE_QUERY_THREADS query threads beside it until it finishes or the budget runs
      // out (budget 0 means unlimited). Query sequence numbers continue after the foreground's.
      std::atomic<bool> merge_done{false};
      std::atomic<uint64_t> next_merge_query{0};
      std::vector<uint64_t> merge_query_counts(cfg.merge_query_threads, 0);
      std::vector<std::thread> merge_query_workers;
      const auto maintenance_begin = std::chrono::steady_clock::now();
      std::thread merge_thread([&]() {
        sync_index.trigger_deletion(cfg.merge_threads, 20);
        merge_done.store(true, std::memory_order_release);
      });
      if (cfg.concurrent_queries) {
        for (uint32_t t = 0; t < cfg.merge_query_threads; t++) {
          merge_query_workers.emplace_back([&, t]() {
            typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
            uint64_t local = 0;
            uint64_t seq = query_n + t;
            while (!merge_done.load(std::memory_order_acquire)) {
              if (merge_query_budget > 0) {
                const uint64_t budget_idx = next_merge_query.fetch_add(1, std::memory_order_relaxed);
                if (budget_idx >= merge_query_budget) {
                  break;
                }
                seq = query_n + budget_idx;
              }
              run_concurrent_query(seq, iter, "merge", qstate);
              local++;
              if (merge_query_budget == 0) {
                seq += cfg.merge_query_threads;
              }
            }
            merge_query_counts[t] = local;
          });
        }
      }
      merge_thread.join();
      for (auto &th : merge_query_workers) th.join();
      const auto maintenance_end = std::chrono::steady_clock::now();
      round_maint_s = std::chrono::duration<double>(maintenance_end - maintenance_begin).count();
      for (uint64_t c : merge_query_counts) round_merge_query_count += c;
      if (window_metrics_ptr != nullptr) {
        window_metrics_ptr->flush(iter, "merge");
        ann_bench::emit_phase_checkpoint(
            std::cout, "DGAI", "merge", iter,
            std::chrono::duration<double>(maintenance_end - workload_start).count(), 0, 0,
            round_merge_query_count, round_maint_s);
      }

      // Post-merge tail: the rest of the budget on the whole pool.
      if (cfg.concurrent_queries && merge_query_budget > round_merge_query_count) {
        const uint64_t remaining = merge_query_budget - round_merge_query_count;
        std::atomic<uint64_t> next_tail{0};
        std::vector<std::thread> tail_workers;
        const auto tail_begin = std::chrono::steady_clock::now();
        for (uint32_t t = 0; t < cfg.nthreads; t++) {
          tail_workers.emplace_back([&]() {
            typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
            for (;;) {
              const uint64_t local = next_tail.fetch_add(1, std::memory_order_relaxed);
              if (local >= remaining) {
                break;
              }
              run_concurrent_query(query_n + round_merge_query_count + local, iter, "post_merge_query", qstate);
            }
          });
        }
        for (auto &th : tail_workers) th.join();
        round_post_merge_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tail_begin).count();
        round_post_merge_query_count = remaining;
        if (window_metrics_ptr != nullptr) {
          window_metrics_ptr->flush(iter, "post_merge_query");
          ann_bench::emit_phase_checkpoint(
              std::cout, "DGAI", "post_merge_query", iter,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - workload_start).count(), 0, 0,
              round_post_merge_query_count, round_post_merge_s);
        }
      }
      round_query_count = round_fg_query_count + round_merge_query_count + round_post_merge_query_count;
    } else {
      const auto foreground_begin = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(static_cast<int>(cfg.nthreads))
      for (int64_t i = 0; i < static_cast<int64_t>(cfg.count_per_iter); i++) {
        const auto op_begin = std::chrono::steady_clock::now();
        sync_index.lazy_delete(ids[batch_begin + static_cast<size_t>(i)]);
        const auto op_end = std::chrono::steady_clock::now();
        batch_latencies_us[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::micro>(op_end - op_begin).count();
      }
      const auto foreground_end = std::chrono::steady_clock::now();
      round_fg_s =
          std::chrono::duration<double>(foreground_end - foreground_begin).count();
      const auto maintenance_begin = std::chrono::steady_clock::now();
      sync_index.trigger_deletion(cfg.nthreads, 20);
      const auto maintenance_end = std::chrono::steady_clock::now();
      round_maint_s =
          std::chrono::duration<double>(maintenance_end - maintenance_begin).count();
    }

    const ann_bench::RssStats round_rss = window_metrics_ptr != nullptr
                                              ? window_metrics_ptr->pause_and_take_round_rss()
                                              : rss_sampler.stop_and_take_interval();
    const ann_bench::LogicalIoSnapshot round_io =
        ann_bench::LogicalIoCounter::snapshot_and_disable();

    // Roll the round into the workload-level aggregate.
    foreground_time_s += round_fg_s;
    maintenance_time_s += round_maint_s;
    post_merge_query_time_s += round_post_merge_s;
    total_query_count += round_query_count;
    if (cfg.concurrent_queries) {
      total_query_time_s += round_fg_s + round_maint_s + round_post_merge_s;
    }
    ann_bench::add_logical_io(logical_io_delta, round_io);
    call_latencies_us.insert(call_latencies_us.end(),
                             batch_latencies_us.begin(),
                             batch_latencies_us.end());
    delete_call_latencies_us.insert(delete_call_latencies_us.end(), batch_delete_latencies_us.begin(),
                                    batch_delete_latencies_us.end());

    // Per-round recall on the live index (excluded from the timed/IO window).
    const double round_recall = measure_recall(iter);
    if (round_recall >= 0.0) last_recall = round_recall;

    // Emit one record per round (phase "round").
    const double round_time_s = round_fg_s + round_maint_s + round_post_merge_s;
    const double round_ops_per_s = ann_bench::safe_div(
        static_cast<double>(ops_per_iter), round_time_s);
    const double round_per_stream_ops_per_s =
        ann_bench::safe_div(static_cast<double>(cfg.count_per_iter), round_time_s);
    const ann_bench::LatencyStats round_lat =
        ann_bench::summarize_latencies(batch_latencies_us);
    bool rfirst = true;
    std::cout << std::fixed << std::setprecision(6) << "{";
    ann_bench::json_kv(std::cout, rfirst, "baseline", "DGAI");
    ann_bench::json_kv(std::cout, rfirst, "workload", workload_name);
    ann_bench::json_kv(std::cout, rfirst, "phase", "round");
    ann_bench::json_kv(std::cout, rfirst, "round_index",
                       static_cast<uint64_t>(iter));
    if (cfg.mode == WorkloadMode::INSERT) {
      ann_bench::json_kv(std::cout, rfirst, "insert_count",
                         static_cast<uint64_t>(cfg.count_per_iter));
      ann_bench::json_kv(std::cout, rfirst, "delete_count", static_cast<uint64_t>(0));
      ann_bench::json_kv(std::cout, rfirst, "insert_ops_per_s", round_ops_per_s);
      ann_bench::json_kv(std::cout, rfirst, "delete_ops_per_s", 0.0);
      ann_bench::emit_latency(std::cout, rfirst, "insert_call", round_lat);
    } else if (update_mode) {
      ann_bench::json_kv(std::cout, rfirst, "insert_count", static_cast<uint64_t>(cfg.count_per_iter));
      ann_bench::json_kv(std::cout, rfirst, "delete_count", static_cast<uint64_t>(cfg.count_per_iter));
      ann_bench::json_kv(std::cout, rfirst, "query_count", round_query_count);
      ann_bench::json_kv(std::cout, rfirst, "foreground_query_count", round_fg_query_count);
      ann_bench::json_kv(std::cout, rfirst, "merge_query_count", round_merge_query_count);
      ann_bench::json_kv(std::cout, rfirst, "post_merge_query_count", round_post_merge_query_count);
      ann_bench::json_kv(std::cout, rfirst, "post_merge_query_time_s", round_post_merge_s);
      ann_bench::json_kv(std::cout, rfirst, "query_qps",
                         ann_bench::safe_div(static_cast<double>(round_query_count), round_time_s));
      ann_bench::json_kv(std::cout, rfirst, "insert_ops_per_s", round_per_stream_ops_per_s);
      ann_bench::json_kv(std::cout, rfirst, "delete_ops_per_s", round_per_stream_ops_per_s);
      ann_bench::emit_latency(std::cout, rfirst, "insert_call", round_lat);
      ann_bench::emit_latency(std::cout, rfirst, "delete_call",
                              ann_bench::summarize_latencies(batch_delete_latencies_us));
    } else {
      ann_bench::json_kv(std::cout, rfirst, "insert_count", static_cast<uint64_t>(0));
      ann_bench::json_kv(std::cout, rfirst, "delete_count",
                         static_cast<uint64_t>(cfg.count_per_iter));
      ann_bench::json_kv(std::cout, rfirst, "insert_ops_per_s", 0.0);
      ann_bench::json_kv(std::cout, rfirst, "delete_ops_per_s", round_ops_per_s);
      ann_bench::emit_latency(std::cout, rfirst, "delete_call", round_lat);
    }
    ann_bench::json_kv(std::cout, rfirst, "update_ops_per_s", round_ops_per_s);
    ann_bench::json_kv(std::cout, rfirst, "foreground_time_s", round_fg_s);
    ann_bench::json_kv(std::cout, rfirst, "maintenance_time_s", round_maint_s);
    ann_bench::json_kv(std::cout, rfirst, "workload_time_s", round_time_s);
    ann_bench::emit_rss_stats(std::cout, rfirst, round_rss);
    if (rq_have_queries) {
      ann_bench::json_kv(std::cout, rfirst, "recall", round_recall);
      ann_bench::json_kv(std::cout, rfirst, "recall_at",
                         static_cast<uint64_t>(cfg.recall_at));
      ann_bench::json_kv(std::cout, rfirst, "recall_search_L",
                         static_cast<uint64_t>(cfg.search_l));
    }
    ann_bench::emit_logical_io(std::cout, rfirst, round_io, round_time_s,
                               static_cast<uint64_t>(ops_per_iter) + round_query_count);
    if (gs != nullptr) {
      const double inserts = static_cast<double>(std::max<uint64_t>(gs->harness_inserts.load(), 1));
      ann_bench::json_kv(std::cout, rfirst, "topo_splits", gs->harness_topo_splits.load());
      ann_bench::json_kv(std::cout, rfirst, "pages_locked_per_insert", gs->harness_pages_locked.load() / inserts);
      ann_bench::json_kv(std::cout, rfirst, "lock_wait_s_per_thread",
                         gs->harness_lock_wait_ns.load() / 1e9 / cfg.nthreads);
      ann_bench::json_kv(std::cout, rfirst, "lock_wait_ms_per_insert", gs->harness_lock_wait_ns.load() / 1e6 / inserts);
    }
    std::cout << "}" << std::endl;
  }

  const double workload_time_s = foreground_time_s + maintenance_time_s + post_merge_query_time_s;
  if (window_metrics_ptr != nullptr) {
    interval_metrics.finalize(static_cast<uint64_t>(cfg.n_iters - 1), "workload");
  }
  interval_metrics.pause_rss();
  const ann_bench::LatencyStats query_lat = window_metrics_ptr != nullptr
                                                ? interval_metrics.query_latency_summary()
                                                : query_latency_sampler.summarize();
  const double ops_per_sec =
      ann_bench::safe_div(static_cast<double>(total_ops), workload_time_s);
  const double per_stream_ops_per_sec = ann_bench::safe_div(static_cast<double>(required_ids), workload_time_s);
  const ann_bench::LatencyStats call_lat =
      ann_bench::summarize_latencies(call_latencies_us);

  std::cout << std::fixed << std::setprecision(6);
  if (cfg.mode == WorkloadMode::INSERT) {
    std::cout << "total_inserts=" << required_ids << std::endl;
    std::cout << "insert_update_merge_time_sec=" << workload_time_s << std::endl;
    std::cout << "inserts_per_sec=" << ops_per_sec << std::endl;
  } else if (cfg.mode == WorkloadMode::DELETE) {
    std::cout << "total_deletes=" << required_ids << std::endl;
    std::cout << "delete_update_merge_time_sec=" << workload_time_s << std::endl;
    std::cout << "deletes_per_sec=" << ops_per_sec << std::endl;
  } else {
    std::cout << "total_inserts=" << required_ids << " total_deletes=" << required_ids << std::endl;
    std::cout << "update_merge_time_sec=" << workload_time_s << std::endl;
    std::cout << "updates_per_sec=" << ops_per_sec << std::endl;
    if (cfg.concurrent_queries) {
      std::cout << "query_count=" << total_query_count << std::endl;
      std::cout << "query_time_s=" << total_query_time_s << std::endl;
      std::cout << "queries_per_sec=" << ann_bench::safe_div(static_cast<double>(total_query_count), total_query_time_s)
                << std::endl;
    }
  }
  bool first = true;
  std::cout << "{";
  ann_bench::json_kv(std::cout, first, "baseline", "DGAI");
  ann_bench::json_kv(std::cout, first, "workload", workload_name);
  ann_bench::json_kv(std::cout, first, "phase", "workload");
  if (cfg.mode == WorkloadMode::INSERT) {
    ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(required_ids));
    ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(0));
    ann_bench::json_kv(std::cout, first, "insert_ops_per_s", ops_per_sec);
    ann_bench::json_kv(std::cout, first, "delete_ops_per_s", 0.0);
    ann_bench::emit_latency(std::cout, first, "insert_call", call_lat);
  } else if (update_mode) {
    double insert_fg_s = 0.0, delete_fg_s = 0.0;  // summed per-call latencies, as PipeANN's driver reports them
    for (double v : call_latencies_us) insert_fg_s += v / 1e6;
    for (double v : delete_call_latencies_us) delete_fg_s += v / 1e6;
    ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(required_ids));
    ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(required_ids));
    ann_bench::json_kv(std::cout, first, "query_count", total_query_count);
    ann_bench::json_kv(std::cout, first, "insert_foreground_time_s", insert_fg_s);
    ann_bench::json_kv(std::cout, first, "delete_foreground_time_s", delete_fg_s);
    ann_bench::json_kv(std::cout, first, "query_foreground_time_s", total_query_time_s);
    ann_bench::json_kv(std::cout, first, "query_time_s", total_query_time_s);
    ann_bench::json_kv(std::cout, first, "post_merge_query_time_s", post_merge_query_time_s);
    ann_bench::json_kv(std::cout, first, "query_qps",
                       ann_bench::safe_div(static_cast<double>(total_query_count), total_query_time_s));
    if (cfg.concurrent_queries) {
      ann_bench::emit_latency(std::cout, first, "query", query_lat);
    }
    ann_bench::json_kv(std::cout, first, "insert_ops_per_s", per_stream_ops_per_sec);
    ann_bench::json_kv(std::cout, first, "delete_ops_per_s", per_stream_ops_per_sec);
    ann_bench::emit_latency(std::cout, first, "insert_call", call_lat);
    ann_bench::emit_latency(std::cout, first, "delete_call", ann_bench::summarize_latencies(delete_call_latencies_us));
  } else {
    ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(0));
    ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(required_ids));
    ann_bench::json_kv(std::cout, first, "insert_ops_per_s", 0.0);
    ann_bench::json_kv(std::cout, first, "delete_ops_per_s", ops_per_sec);
    ann_bench::emit_latency(std::cout, first, "delete_call", call_lat);
  }
  ann_bench::json_kv(std::cout, first, "update_ops_per_s", ops_per_sec);
  ann_bench::json_kv(std::cout, first, "overall_ops_per_s",
                     ann_bench::safe_div(static_cast<double>(total_ops + total_query_count), workload_time_s));
  ann_bench::json_kv(std::cout, first, "foreground_time_s", foreground_time_s);
  ann_bench::json_kv(std::cout, first, "maintenance_time_s", maintenance_time_s);
  ann_bench::json_kv(std::cout, first, "workload_time_s", workload_time_s);
  ann_bench::json_kv(std::cout, first, "avg_rss_mb",
                     window_metrics_ptr != nullptr ? interval_metrics.avg_rss_mb() : rss_sampler.avg_mb());
  ann_bench::json_kv(std::cout, first, "peak_rss_mb",
                     window_metrics_ptr != nullptr ? interval_metrics.peak_rss_mb() : rss_sampler.peak_mb());
  if (rq_have_queries) {
    ann_bench::json_kv(std::cout, first, "recall", last_recall);
    ann_bench::json_kv(std::cout, first, "recall_at",
                       static_cast<uint64_t>(cfg.recall_at));
    ann_bench::json_kv(std::cout, first, "recall_search_L",
                       static_cast<uint64_t>(cfg.search_l));
  }
  ann_bench::emit_logical_io(std::cout, first, logical_io_delta,
                             workload_time_s,
                             static_cast<uint64_t>(total_ops) + total_query_count);
  std::cout << "}" << std::endl;

  if (cfg.query_bin != "null") {
    T *query = nullptr;
    size_t query_num = 0, query_dim = 0;
    pipeann::load_bin<T>(cfg.query_bin, query, query_num, query_dim);

    unsigned *gt_ids = nullptr;
    float *gt_dists = nullptr;
    uint32_t *gt_tags = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    unsigned *gt_eval_ids = nullptr;
    bool calc_recall = false;
    const std::string final_gt = cfg.gt_dir + "/round_" +
        std::to_string(cfg.n_iters > 0 ? cfg.n_iters - 1 : 0) + "_gt" +
        std::to_string(cfg.recall_at) + ".bin";
    if (file_exists(final_gt)) {
      pipeann::load_truthset(final_gt, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
      calc_recall = (gt_num == query_num);
      gt_eval_ids = (gt_tags != nullptr) ? reinterpret_cast<unsigned *>(gt_tags) : gt_ids;
    }

    std::vector<uint32_t> query_result_tags(cfg.recall_at * query_num);
    std::vector<float> query_result_dists(cfg.recall_at * query_num);
    std::vector<pipeann::QueryStats> query_stats(query_num);

    ann_bench::RssSampler query_rss;
    ann_bench::LogicalIoCounter::reset_and_enable();
    query_rss.start();

    omp_set_num_threads(static_cast<int>(cfg.query_threads));
    const auto query_begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(query_num); i++) {
      sync_index.search(query + i * query_dim, cfg.recall_at, 0, cfg.search_l, cfg.beamwidth,
                        query_result_tags.data() + i * cfg.recall_at,
                        query_result_dists.data() + i * cfg.recall_at,
                        query_stats.data() + i);
    }
    const auto query_end = std::chrono::high_resolution_clock::now();
    query_rss.stop();
    const auto query_io = ann_bench::LogicalIoCounter::snapshot_and_disable();
    const double query_time_s = std::chrono::duration<double>(query_end - query_begin).count();
    const double query_qps = ann_bench::safe_div(static_cast<double>(query_num), query_time_s);

    float recall = 0.0f;
    if (calc_recall) {
      recall = static_cast<float>(pipeann::calculate_recall(
          static_cast<uint32_t>(query_num), gt_eval_ids, gt_dists, static_cast<uint32_t>(gt_dim),
          query_result_tags.data(), static_cast<uint32_t>(cfg.recall_at),
          static_cast<uint32_t>(cfg.recall_at)));
    }

    float mean_lat = static_cast<float>(pipeann::get_mean_stats(query_stats.data(), query_num, [](const pipeann::QueryStats &s) { return s.total_us; }));
    float p50_lat = static_cast<float>(pipeann::get_percentile_stats(query_stats.data(), query_num, 0.50f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    float p90_lat = static_cast<float>(pipeann::get_percentile_stats(query_stats.data(), query_num, 0.90f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    float p95_lat = static_cast<float>(pipeann::get_percentile_stats(query_stats.data(), query_num, 0.95f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    float p99_lat = static_cast<float>(pipeann::get_percentile_stats(query_stats.data(), query_num, 0.99f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    float p999_lat = static_cast<float>(pipeann::get_percentile_stats(query_stats.data(), query_num, 0.999f, [](const pipeann::QueryStats &s) { return s.total_us; }));
    float mean_hops = static_cast<float>(pipeann::get_mean_stats(query_stats.data(), query_num, [](const pipeann::QueryStats &s) { return s.n_hops; }));
    float mean_ios = static_cast<float>(pipeann::get_mean_stats(query_stats.data(), query_num, [](const pipeann::QueryStats &s) { return s.n_ios; }));
    float mean_io_us = static_cast<float>(pipeann::get_mean_stats(query_stats.data(), query_num, [](const pipeann::QueryStats &s) { return s.io_us; }));

    std::cout << std::setprecision(2) << std::fixed;
    std::string recall_string = "Recall@" + std::to_string(cfg.recall_at);
    std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
              << "AvgLat(us)" << std::setw(12) << "P50(us)" << std::setw(12) << "P90(us)" << std::setw(12)
              << "P95(us)" << std::setw(12) << "P99(us)" << std::setw(12) << "P99.9(us)" << std::setw(12)
              << "Mean Hops" << std::setw(12) << "Mean IOs" << std::setw(12);
    if (calc_recall) {
      std::cout << std::setw(12) << recall_string;
    }
    std::cout << std::endl;
    std::cout << "=============================================="
                 "==========================================="
              << std::endl;
    std::cout << std::setw(6) << cfg.search_l << std::setw(12) << cfg.beamwidth << std::setw(12)
              << static_cast<float>(query_qps) << std::setw(12) << mean_lat << std::setw(12)
              << p50_lat << std::setw(12) << p90_lat << std::setw(12) << p95_lat << std::setw(12)
              << p99_lat << std::setw(12) << p999_lat << std::setw(12) << mean_hops << std::setw(12)
              << mean_ios;
    if (calc_recall) {
      std::cout << std::setw(12) << recall;
    }
    std::cout << std::endl;

    const char *phase_env = std::getenv("ANN_BENCH_PHASE");
    bool qfirst = true;
    std::cout << "{";
    ann_bench::json_kv(std::cout, qfirst, "baseline", "DGAI");
    // PipeANN's update workload labels its final pass "query"; insert and delete keep their own name.
    ann_bench::json_kv(std::cout, qfirst, "workload", update_mode ? "query" : workload_name);
    ann_bench::json_kv(std::cout, qfirst, "phase",
                       phase_env != nullptr && *phase_env != '\0' ? phase_env : "final_query");
    ann_bench::json_kv(std::cout, qfirst, "query_count", static_cast<uint64_t>(query_num));
    ann_bench::json_kv(std::cout, qfirst, "query_time_s", query_time_s);
    ann_bench::json_kv(std::cout, qfirst, "query_qps", query_qps);
    ann_bench::json_kv(std::cout, qfirst, "query_lat_avg_us", static_cast<double>(mean_lat));
    ann_bench::json_kv(std::cout, qfirst, "query_io_ns_total",
                       static_cast<uint64_t>(static_cast<double>(mean_io_us) *
                                             static_cast<double>(query_num) * 1000.0));
    ann_bench::json_kv(std::cout, qfirst, "query_io_avg_us_per_query",
                       static_cast<double>(mean_io_us));
    ann_bench::json_kv(std::cout, qfirst, "query_io_fraction_of_latency",
                       mean_lat > 0 ? static_cast<double>(mean_io_us) / static_cast<double>(mean_lat)
                                    : 0.0);
    ann_bench::json_kv(std::cout, qfirst, "query_lat_p50_us", static_cast<double>(p50_lat));
    ann_bench::json_kv(std::cout, qfirst, "query_lat_p90_us", static_cast<double>(p90_lat));
    ann_bench::json_kv(std::cout, qfirst, "query_lat_p95_us", static_cast<double>(p95_lat));
    ann_bench::json_kv(std::cout, qfirst, "query_lat_p99_us", static_cast<double>(p99_lat));
    ann_bench::json_kv(std::cout, qfirst, "recall", static_cast<double>(recall));
    ann_bench::json_kv(std::cout, qfirst, "avg_rss_mb", query_rss.avg_mb());
    ann_bench::json_kv(std::cout, qfirst, "peak_rss_mb", query_rss.peak_mb());
    ann_bench::emit_logical_io(std::cout, qfirst, query_io, query_time_s,
                               static_cast<uint64_t>(query_num));
    std::cout << "}" << std::endl;

    delete[] query;
    if (gt_ids) delete[] gt_ids;
    if (gt_dists) delete[] gt_dists;
    if (gt_tags) delete[] gt_tags;
  }

  // Per-round recall buffers (loaded once before the loop).
  if (rq_query) delete[] rq_query;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    Config cfg = parse_args(argc, argv);
    if (cfg.type == "float") {
      pipeann::DistanceL2 dist_cmp;
      run_workload<float>(cfg, &dist_cmp);
    } else if (cfg.type == "uint8") {
      pipeann::DistanceL2UInt8 dist_cmp;
      run_workload<uint8_t>(cfg, &dist_cmp);
    } else if (cfg.type == "int8") {
      pipeann::DistanceL2Int8 dist_cmp;
      run_workload<int8_t>(cfg, &dist_cmp);
    } else {
      throw std::invalid_argument("unsupported type: " + cfg.type);
    }
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
