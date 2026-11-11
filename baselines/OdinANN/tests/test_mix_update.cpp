// Mixed per-round inserts + deletes with merge (sift update workload).

#include "dynamic_index.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "distance.h"
#include "utils.h"
#include "../../src/ann_bench_metrics.h"
#include "../../src/streaming_query_source.h"
#include "../../src/weighted_update_pool.h"
#include <functional>
#include <numeric>

namespace {

struct Config {
  std::string type;
  std::string full_data_bin;
  unsigned l_disk;
  std::string base_prefix;
  std::string merge_prefix;
  size_t n_iters;
  uint32_t insert_count_per_iter;
  uint32_t delete_count_per_iter;
  uint32_t range;
  uint32_t beamwidth;
  uint32_t nthreads;          // unified insert/delete/query worker pool size
  uint32_t merge_threads;
  uint32_t merge_maxc;
  std::string insert_ids_file;
  std::string delete_ids_file;
  std::string query_bin;
  uint32_t query_k;
  uint32_t query_l;
  uint32_t skip_update_search;
  uint32_t query_ratio;
  uint32_t merge_query_threads;
  uint64_t tail_query_begin;
  uint64_t tail_query_end;
  std::string gt_rounds_dir;
  uint32_t recall_search_L;
  uint32_t enable_interval_metrics;  // 0/1
};

uint64_t parse_u64(const char *value, const std::string &name) {
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0') {
    throw std::invalid_argument("invalid integer for " + name + ": " + value);
  }
  return static_cast<uint64_t>(parsed);
}

uint32_t parse_u32(const char *value, const std::string &name, bool allow_zero = false) {
  uint64_t parsed = parse_u64(value, name);
  if ((!allow_zero && parsed == 0) || parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid uint32 for " + name + ": " + value);
  }
  return static_cast<uint32_t>(parsed);
}

Config parse_args(int argc, char **argv) {
  if (argc != 27) {
    std::cerr << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <full_data_bin> <L_disk> <base_prefix> <merge_prefix>"
              << " <n_iters> <insert_count_per_iter> <delete_count_per_iter> <range> <beamwidth> <nthreads>"
              << " <merge_threads> <merge_maxc>"
              << " <insert_ids_file> <delete_ids_file>"
              << " <query_bin|null> <query_k> <query_l>"
              << " <skip_update_search[0/1]> <query_ratio> <merge_query_threads>"
              << " <tail_query_begin> <tail_query_end> <gt_rounds_dir> <recall_search_L>"
              << " <enable_interval_metrics[0/1]>" << std::endl;
    throw std::invalid_argument("wrong number of arguments");
  }

  int arg_no = 1;
  Config cfg;
  cfg.type = argv[arg_no++];
  cfg.full_data_bin = argv[arg_no++];
  cfg.l_disk = parse_u32(argv[arg_no++], "L_disk");
  cfg.base_prefix = argv[arg_no++];
  cfg.merge_prefix = argv[arg_no++];
  cfg.n_iters = static_cast<size_t>(parse_u64(argv[arg_no++], "n_iters"));
  cfg.insert_count_per_iter = parse_u32(argv[arg_no++], "insert_count_per_iter", true);
  cfg.delete_count_per_iter = parse_u32(argv[arg_no++], "delete_count_per_iter", true);
  cfg.range = parse_u32(argv[arg_no++], "range");
  cfg.beamwidth = parse_u32(argv[arg_no++], "beamwidth");
  cfg.nthreads = parse_u32(argv[arg_no++], "nthreads");
  cfg.merge_threads = parse_u32(argv[arg_no++], "merge_threads");
  cfg.merge_maxc = parse_u32(argv[arg_no++], "merge_maxc");
  cfg.insert_ids_file = argv[arg_no++];
  cfg.delete_ids_file = argv[arg_no++];
  cfg.query_bin = argv[arg_no++];
  cfg.query_k = parse_u32(argv[arg_no++], "query_k");
  cfg.query_l = parse_u32(argv[arg_no++], "query_l");
  cfg.skip_update_search = parse_u32(argv[arg_no++], "skip_update_search", true);
  cfg.query_ratio = parse_u32(argv[arg_no++], "query_ratio", true);
  cfg.merge_query_threads = parse_u32(argv[arg_no++], "merge_query_threads", true);
  cfg.tail_query_begin = parse_u64(argv[arg_no++], "tail_query_begin");
  cfg.tail_query_end = parse_u64(argv[arg_no++], "tail_query_end");
  cfg.gt_rounds_dir = argv[arg_no++];
  cfg.recall_search_L = parse_u32(argv[arg_no++], "recall_search_L");
  cfg.enable_interval_metrics =
      parse_u32(argv[arg_no++], "enable_interval_metrics", true);
  if (cfg.n_iters == 0) {
    throw std::invalid_argument("n_iters must be positive");
  }
  if (cfg.insert_count_per_iter == 0 || cfg.delete_count_per_iter == 0) {
    throw std::invalid_argument("insert_count_per_iter and delete_count_per_iter must both be positive");
  }
  if (cfg.skip_update_search > 1 || cfg.enable_interval_metrics > 1) {
    throw std::invalid_argument(
        "skip_update_search and enable_interval_metrics must be 0 or 1");
  }
  if (cfg.query_ratio == 0 && cfg.skip_update_search == 0) {
    throw std::invalid_argument("query_ratio must be > 0 when skip_update_search=0");
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

std::vector<uint32_t> load_counted_ids(const std::string &path, size_t required_ids) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open IDs file: " + path);
  }

  int32_t count_i32 = 0;
  in.read(reinterpret_cast<char *>(&count_i32), sizeof(int32_t));
  if (!in || count_i32 < 0) {
    throw std::runtime_error("invalid IDs file header: " + path);
  }

  const size_t count = static_cast<size_t>(count_i32);
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

  int32_t file_npts_i = 0;
  int32_t file_dim_i = 0;
  reader.read(reinterpret_cast<char *>(&file_npts_i), sizeof(int32_t));
  reader.read(reinterpret_cast<char *>(&file_dim_i), sizeof(int32_t));
  if (!reader || file_npts_i <= 0 || file_dim_i <= 0) {
    throw std::runtime_error("invalid bin header: " + data_file);
  }
  if (static_cast<uint32_t>(file_npts_i) != full_npts ||
      static_cast<uint32_t>(file_dim_i) != dim) {
    throw std::runtime_error("full-data metadata mismatch: " + data_file);
  }

  for (uint32_t i = 0; i < count; i++) {
    const uint32_t id = ids[begin + i];
    if (id >= full_npts) {
      throw std::runtime_error("insert ID out of range: " + std::to_string(id));
    }
    const uint64_t offset = 2 * sizeof(int32_t) + static_cast<uint64_t>(id) * static_cast<uint64_t>(dim) * sizeof(T);
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

  const size_t required_insert_ids = cfg.n_iters * static_cast<size_t>(cfg.insert_count_per_iter);
  const size_t required_delete_ids = cfg.n_iters * static_cast<size_t>(cfg.delete_count_per_iter);
  std::vector<uint32_t> insert_ids = load_counted_ids(cfg.insert_ids_file, required_insert_ids);
  std::vector<uint32_t> delete_ids = load_counted_ids(cfg.delete_ids_file, required_delete_ids);
  ann_bench::StreamingQuerySource<T> query_source;
  if (cfg.skip_update_search == 0) {
    query_source.load(cfg.query_bin, cfg.full_data_bin,
                      cfg.tail_query_begin, cfg.tail_query_end);
    if (query_source.dim != dim) {
      throw std::runtime_error("query dim mismatch");
    }
  }

  pipeann::IndexBuildParameters paras;
  paras.set(0, cfg.l_disk, cfg.merge_maxc, 1.2f, cfg.nthreads, true, cfg.beamwidth);

  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, uint32_t> sync_index(paras, cfg.base_prefix, cfg.merge_prefix, dist_cmp, metric,
                                                   BEAM_SEARCH, false, "", false);
  uint64_t total_query_count = 0;
  double total_query_time_s = 0.0;
  double total_insert_time_s = 0.0;
  double total_delete_time_s = 0.0;
  double total_merge_time_s = 0.0;
  double total_foreground_time_s = 0.0;
  double total_post_merge_query_time_s = 0.0;
  ann_bench::LatencySampler query_latency_sampler;
  std::vector<double> insert_call_latencies_us;
  std::vector<double> delete_call_latencies_us;
  ann_bench::LogicalIoSnapshot logical_io_delta;
  ann_bench::RssSampler rss_sampler;

  T *rq_query = nullptr;
  size_t rq_num = 0, rq_dim = 0;
  const bool rq_have_queries =
      !cfg.query_bin.empty() && cfg.query_bin != "null";
  if (rq_have_queries) {
    pipeann::load_bin<T>(cfg.query_bin, rq_query, rq_num, rq_dim);
    if (rq_dim != dim) {
      throw std::runtime_error("query dim mismatch for per-round recall");
    }
  }
  auto measure_recall = [&](size_t round_idx) -> double {
    if (!rq_have_queries || rq_num == 0) return -1.0;
    const std::string gt_path = cfg.gt_rounds_dir + "/round_" +
                                std::to_string(round_idx) + "_gt" +
                                std::to_string(cfg.query_k) + ".bin";
    std::ifstream gt_check(gt_path);
    if (!gt_check.good()) {
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
      delete[] gt_ids;
      delete[] gt_dists;
      delete[] gt_tags;
      return -1.0;
    }
    gt_eval = (gt_tags != nullptr) ? reinterpret_cast<unsigned *>(gt_tags) : gt_ids;
    std::vector<uint32_t> res_tags(rq_num * cfg.query_k);
    std::vector<float> res_dists(rq_num * cfg.query_k);
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(cfg.nthreads))
    for (int64_t i = 0; i < static_cast<int64_t>(rq_num); i++) {
      pipeann::QueryStats st;
      sync_index.search(rq_query + static_cast<size_t>(i) * rq_dim, cfg.query_k, 0,
                        cfg.recall_search_L, cfg.beamwidth,
                        res_tags.data() + static_cast<size_t>(i) * cfg.query_k,
                        res_dists.data() + static_cast<size_t>(i) * cfg.query_k, &st);
    }
    const double r = pipeann::calculate_recall(
        static_cast<uint32_t>(rq_num), gt_eval, gt_dists,
        static_cast<uint32_t>(gt_dim), res_tags.data(),
        static_cast<uint32_t>(cfg.query_k), static_cast<uint32_t>(cfg.query_k));
    delete[] gt_ids;
    delete[] gt_dists;
    delete[] gt_tags;
    return r;
  };

  omp_set_dynamic(0);
  const auto start = std::chrono::steady_clock::now();
  ann_bench::QueryCycleWindowMetrics interval_metrics;
  const bool enable_interval_metrics =
      cfg.skip_update_search == 0 && cfg.enable_interval_metrics != 0;
  ann_bench::QueryCycleWindowMetrics *window_metrics_ptr =
      enable_interval_metrics ? &interval_metrics : nullptr;
  if (window_metrics_ptr != nullptr) {
    window_metrics_ptr->start(
        "PipeANN", ann_bench::latency_reserve_hint(
                       static_cast<uint64_t>(query_source.query_n)));
  }
  for (size_t iter = 0; iter < cfg.n_iters; iter++) {
    const size_t insert_begin = iter * static_cast<size_t>(cfg.insert_count_per_iter);
    const size_t delete_begin = iter * static_cast<size_t>(cfg.delete_count_per_iter);
    std::cout << "ITER " << iter << ": insert " << cfg.insert_count_per_iter
              << ", delete " << cfg.delete_count_per_iter
              << " (unified pool threads=" << cfg.nthreads
              << ", merge_threads=" << cfg.merge_threads << ")" << std::endl;

    std::vector<T> batch = load_insert_batch<T>(cfg.full_data_bin, insert_ids, insert_begin,
                                                cfg.insert_count_per_iter, full_npts, dim);

    // Concurrent (stress) queries drawn cyclically from the query pool, sharing
    // the worker pool with inserts and deletes. Recall is NOT computed here --
    // accuracy is measured by the standalone final query after the merge.
    const bool query_enabled =
        cfg.skip_update_search == 0 && query_source.pool_size() > 0 &&
        cfg.query_ratio > 0;
    const size_t insert_n = cfg.insert_count_per_iter;
    const size_t delete_n = cfg.delete_count_per_iter;
    const size_t query_n =
        query_enabled ? ann_bench::mixed_update_query_count(
                            insert_n, delete_n, cfg.query_ratio)
                      : 0;

    std::atomic<uint32_t> failures(0);
    std::vector<double> insert_lat(insert_n, 0.0);
    std::vector<double> delete_lat(delete_n, 0.0);

    std::function<void(uint64_t)> tasks[ann_bench::WeightedUpdatePool::NUM_OPS];
    tasks[ann_bench::WeightedUpdatePool::INSERT] = [&](uint64_t i) {
      const uint32_t tag = insert_ids[insert_begin + i];
      const auto op0 = std::chrono::steady_clock::now();
      if (sync_index.insert(batch.data() + static_cast<size_t>(i) * dim, tag) < 0) {
        failures.fetch_add(1);
      }
      const auto op1 = std::chrono::steady_clock::now();
      insert_lat[i] = std::chrono::duration<double, std::micro>(op1 - op0).count();
      if (window_metrics_ptr != nullptr) {
        window_metrics_ptr->record_insert(insert_lat[i]);
      }
    };
    tasks[ann_bench::WeightedUpdatePool::DELETE] = [&](uint64_t i) {
      const auto op0 = std::chrono::steady_clock::now();
      sync_index.lazy_delete(delete_ids[delete_begin + i]);
      const auto op1 = std::chrono::steady_clock::now();
      delete_lat[i] = std::chrono::duration<double, std::micro>(op1 - op0).count();
      if (window_metrics_ptr != nullptr) {
        window_metrics_ptr->record_delete(delete_lat[i]);
      }
    };
    tasks[ann_bench::WeightedUpdatePool::QUERY] = [&](uint64_t i) {
      thread_local typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
      qstate.result_tags.resize(cfg.query_k);
      qstate.result_dists.resize(cfg.query_k);
      pipeann::QueryStats qstats;
      const T *query = query_source.get(i, qstate);
      const auto op0 = std::chrono::steady_clock::now();
      sync_index.search(query, cfg.query_k, 0,
                        cfg.query_l, cfg.beamwidth,
                        qstate.result_tags.data(),
                        qstate.result_dists.data(),
                        &qstats, true);
      const auto op1 = std::chrono::steady_clock::now();
      const double lat_us = std::chrono::duration<double, std::micro>(op1 - op0).count();
      if (window_metrics_ptr != nullptr) {
        window_metrics_ptr->record_query(lat_us, iter, "foreground");
      } else {
        query_latency_sampler.record(i, lat_us);
      }
    };

    const uint64_t counts[ann_bench::WeightedUpdatePool::NUM_OPS] = {
        static_cast<uint64_t>(insert_n), static_cast<uint64_t>(delete_n),
        static_cast<uint64_t>(query_n)};

    ann_bench::LogicalIoCounter::reset_and_enable();
    if (window_metrics_ptr != nullptr) {
      window_metrics_ptr->reset_io_baseline();
    }
    if (window_metrics_ptr != nullptr) {
      window_metrics_ptr->resume_rss();
    } else {
      rss_sampler.resume();
    }
    const auto foreground_start = std::chrono::steady_clock::now();
    if (ann_bench::env_equals("UPDATE_POOL_MODE", "round_robin")) {
      ann_bench::WeightedUpdatePool::run_rq2_round_robin_90_5_5(
          cfg.nthreads, counts, tasks);
    } else {
      ann_bench::WeightedUpdatePool::run(cfg.nthreads, counts, tasks);
    }
    const auto foreground_end = std::chrono::steady_clock::now();
    const double round_wall_s =
        std::chrono::duration<double>(foreground_end - foreground_start).count();
    total_foreground_time_s += round_wall_s;
    if (window_metrics_ptr != nullptr) {
      window_metrics_ptr->flush(iter, "foreground");
      ann_bench::emit_phase_checkpoint(
          std::cout, "PipeANN", "foreground", iter,
          std::chrono::duration<double>(foreground_end - start).count(),
          static_cast<uint64_t>(insert_n), static_cast<uint64_t>(delete_n),
          static_cast<uint64_t>(query_n), round_wall_s);
    }

    if (window_metrics_ptr == nullptr) {
      insert_call_latencies_us.insert(insert_call_latencies_us.end(),
                                      insert_lat.begin(), insert_lat.end());
      delete_call_latencies_us.insert(delete_call_latencies_us.end(),
                                      delete_lat.begin(), delete_lat.end());
    }
    // Aggregate per-op busy time (sum of op latencies; >= wall under parallelism).
    total_insert_time_s +=
        std::accumulate(insert_lat.begin(), insert_lat.end(), 0.0) / 1000000.0;
    total_delete_time_s +=
        std::accumulate(delete_lat.begin(), delete_lat.end(), 0.0) / 1000000.0;
    if (query_enabled) {
      // Concurrent query QPS is measured over the full foreground window.
      total_query_count += query_n;
    }
    if (failures != 0) {
      throw std::runtime_error("failed to insert " + std::to_string(failures.load()) + " points");
    }

    auto merge_start = std::chrono::steady_clock::now();
    std::atomic<bool> merge_done{false};
    const uint64_t merge_query_budget =
        ann_bench::env_u64_or("MERGE_QUERY_BUDGET", 0);
    std::atomic<uint64_t> next_merge_query{0};
    std::thread merge_thread([&]() {
      sync_index.final_merge(cfg.merge_threads);
      merge_done.store(true, std::memory_order_release);
    });
    std::vector<std::thread> merge_query_threads;
    std::vector<uint64_t> merge_query_counts(cfg.merge_query_threads, 0);
    if (query_enabled && cfg.merge_query_threads > 0) {
      for (uint32_t t = 0; t < cfg.merge_query_threads; t++) {
        merge_query_threads.emplace_back([&, t]() {
          typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
          qstate.result_tags.resize(cfg.query_k);
          qstate.result_dists.resize(cfg.query_k);
          uint64_t local = 0;
          uint64_t seq = static_cast<uint64_t>(query_n) + t;
          while (!merge_done.load(std::memory_order_acquire)) {
            if (merge_query_budget > 0) {
              const uint64_t budget_idx =
                  next_merge_query.fetch_add(1, std::memory_order_relaxed);
              if (budget_idx >= merge_query_budget) {
                break;
              }
              seq = static_cast<uint64_t>(query_n) + budget_idx;
            }
            pipeann::QueryStats qstats;
            const T *query = query_source.get(seq, qstate);
            const auto op0 = std::chrono::steady_clock::now();
            sync_index.search(query, cfg.query_k, 0, cfg.query_l, cfg.beamwidth,
                              qstate.result_tags.data(), qstate.result_dists.data(),
                              &qstats, true);
            const auto op1 = std::chrono::steady_clock::now();
            const double lat_us =
                std::chrono::duration<double, std::micro>(op1 - op0).count();
            if (window_metrics_ptr != nullptr) {
              window_metrics_ptr->record_query(lat_us, iter, "merge");
            } else {
              query_latency_sampler.record(seq, lat_us);
            }
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
    for (auto &th : merge_query_threads) th.join();
    auto merge_end = std::chrono::steady_clock::now();
    const double round_merge_s = std::chrono::duration<double>(merge_end - merge_start).count();
    total_merge_time_s += round_merge_s;
    uint64_t merge_query_count = 0;
    for (uint64_t c : merge_query_counts) merge_query_count += c;
    uint64_t post_merge_query_count = 0;
    double post_merge_query_s = 0.0;
    if (query_enabled && merge_query_budget > merge_query_count) {
      const uint64_t remaining = merge_query_budget - merge_query_count;
      const uint32_t tail_threads = cfg.nthreads;
      const uint32_t effective_tail_threads = tail_threads == 0 ? 1 : tail_threads;
      const auto tail_begin = std::chrono::steady_clock::now();
      std::atomic<uint64_t> next_tail{0};
      std::vector<std::thread> tail_query_threads;
      tail_query_threads.reserve(effective_tail_threads);
      for (uint32_t t = 0; t < effective_tail_threads; t++) {
        tail_query_threads.emplace_back([&]() {
          typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
          qstate.result_tags.resize(cfg.query_k);
          qstate.result_dists.resize(cfg.query_k);
          for (;;) {
            const uint64_t local =
                next_tail.fetch_add(1, std::memory_order_relaxed);
            if (local >= remaining) {
              break;
            }
            const uint64_t seq =
                static_cast<uint64_t>(query_n) + merge_query_count + local;
            pipeann::QueryStats qstats;
            const T *query = query_source.get(seq, qstate);
            const auto op0 = std::chrono::steady_clock::now();
            sync_index.search(query, cfg.query_k, 0, cfg.query_l, cfg.beamwidth,
                              qstate.result_tags.data(), qstate.result_dists.data(),
                              &qstats, true);
            const auto op1 = std::chrono::steady_clock::now();
            const double lat_us =
                std::chrono::duration<double, std::micro>(op1 - op0).count();
            if (window_metrics_ptr != nullptr) {
              window_metrics_ptr->record_query(lat_us, iter, "post_merge_query");
            } else {
              query_latency_sampler.record(seq, lat_us);
            }
          }
        });
      }
      for (auto &th : tail_query_threads) th.join();
      const auto tail_end = std::chrono::steady_clock::now();
      post_merge_query_count = remaining;
      post_merge_query_s =
          std::chrono::duration<double>(tail_end - tail_begin).count();
    }
    total_query_count += merge_query_count + post_merge_query_count;
    total_post_merge_query_time_s += post_merge_query_s;
    if (window_metrics_ptr != nullptr) {
      window_metrics_ptr->flush(iter, "merge");
      ann_bench::emit_phase_checkpoint(
          std::cout, "PipeANN", "merge", iter,
          std::chrono::duration<double>(merge_end - start).count(),
          0, 0, merge_query_count, round_merge_s);
      if (post_merge_query_count > 0) {
        window_metrics_ptr->flush(iter, "post_merge_query");
        ann_bench::emit_phase_checkpoint(
            std::cout, "PipeANN", "post_merge_query", iter,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count(),
            0, 0, post_merge_query_count, post_merge_query_s);
      }
    }
    if (query_enabled) {
      total_query_time_s += round_wall_s + round_merge_s + post_merge_query_s;
    }
    ann_bench::add_logical_io(logical_io_delta,
                              ann_bench::LogicalIoCounter::snapshot_and_disable());
    const ann_bench::RssStats round_rss =
        window_metrics_ptr != nullptr
            ? window_metrics_ptr->pause_and_take_round_rss()
            : rss_sampler.stop_and_take_interval();

    bool rfirst = true;
    const uint64_t round_query_count =
        static_cast<uint64_t>(query_n) + merge_query_count + post_merge_query_count;
    const double round_time_s = round_wall_s + round_merge_s + post_merge_query_s;
    std::cout << std::fixed << std::setprecision(6) << "{";
    ann_bench::json_kv(std::cout, rfirst, "baseline", "PipeANN");
    ann_bench::json_kv(std::cout, rfirst, "workload", "mixed_update");
    ann_bench::json_kv(std::cout, rfirst, "phase", "round");
    ann_bench::json_kv(std::cout, rfirst, "round_index", static_cast<uint64_t>(iter));
    ann_bench::json_kv(std::cout, rfirst, "elapsed_s",
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start).count());
    ann_bench::json_kv(std::cout, rfirst, "insert_count", static_cast<uint64_t>(insert_n));
    ann_bench::json_kv(std::cout, rfirst, "delete_count", static_cast<uint64_t>(delete_n));
    ann_bench::json_kv(std::cout, rfirst, "query_count", round_query_count);
    ann_bench::json_kv(std::cout, rfirst, "foreground_query_count",
                       static_cast<uint64_t>(query_n));
    ann_bench::json_kv(std::cout, rfirst, "merge_query_count", merge_query_count);
    ann_bench::json_kv(std::cout, rfirst, "post_merge_query_count",
                       post_merge_query_count);
    ann_bench::json_kv(std::cout, rfirst, "foreground_time_s", round_wall_s);
    ann_bench::json_kv(std::cout, rfirst, "maintenance_time_s", round_merge_s);
    ann_bench::json_kv(std::cout, rfirst, "post_merge_query_time_s",
                       post_merge_query_s);
    ann_bench::json_kv(std::cout, rfirst, "workload_time_s", round_time_s);
    ann_bench::emit_rss_stats(std::cout, rfirst, round_rss);
    ann_bench::json_kv(std::cout, rfirst, "query_qps",
                       ann_bench::safe_div(static_cast<double>(round_query_count), round_time_s));
    ann_bench::json_kv(std::cout, rfirst, "insert_ops_per_s",
                       ann_bench::safe_div(static_cast<double>(insert_n), round_time_s));
    ann_bench::json_kv(std::cout, rfirst, "delete_ops_per_s",
                       ann_bench::safe_div(static_cast<double>(delete_n), round_time_s));
    ann_bench::json_kv(std::cout, rfirst, "update_ops_per_s",
                       ann_bench::safe_div(static_cast<double>(insert_n + delete_n), round_time_s));
    const double round_recall = measure_recall(iter);
    if (round_recall >= 0.0) {
      ann_bench::json_kv(std::cout, rfirst, "recall", round_recall);
      ann_bench::json_kv(std::cout, rfirst, "recall_at", static_cast<uint64_t>(cfg.query_k));
      ann_bench::json_kv(std::cout, rfirst, "recall_search_L",
                         static_cast<uint64_t>(cfg.recall_search_L));
    }
    std::cout << "}" << std::endl;
  }
  if (rq_query != nullptr) {
    delete[] rq_query;
  }
  const auto end = std::chrono::steady_clock::now();

  const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
  const double total_updates = static_cast<double>(required_insert_ids + required_delete_ids);
  const double maintenance_time_s = std::max(0.0, total_merge_time_s);
  const double workload_time_s =
      total_foreground_time_s + maintenance_time_s + total_post_merge_query_time_s;
  const double updates_per_sec = ann_bench::safe_div(total_updates, workload_time_s);
  const double insert_per_sec =
      ann_bench::safe_div(static_cast<double>(required_insert_ids), workload_time_s);
  const double delete_per_sec =
      ann_bench::safe_div(static_cast<double>(required_delete_ids), workload_time_s);
  const double query_qps =
      ann_bench::safe_div(static_cast<double>(total_query_count), total_query_time_s);
  const uint64_t total_ops =
      static_cast<uint64_t>(required_insert_ids + required_delete_ids) +
      total_query_count;
  const double overall_ops_per_s =
      ann_bench::safe_div(static_cast<double>(total_ops), workload_time_s);
  if (enable_interval_metrics) {
    const uint64_t last_round =
        cfg.n_iters > 0 ? static_cast<uint64_t>(cfg.n_iters - 1) : 0;
    interval_metrics.finalize(last_round, "workload");
  }
  interval_metrics.pause_rss();
  ann_bench::LatencyStats query_lat =
      enable_interval_metrics ? interval_metrics.query_latency_summary()
                              : query_latency_sampler.summarize();
  ann_bench::LatencyStats insert_call_lat =
      enable_interval_metrics
          ? interval_metrics.insert_latency_summary()
          : ann_bench::summarize_latencies(insert_call_latencies_us);
  ann_bench::LatencyStats delete_call_lat =
      enable_interval_metrics
          ? interval_metrics.delete_latency_summary()
          : ann_bench::summarize_latencies(delete_call_latencies_us);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "inserts=" << required_insert_ids << std::endl;
  std::cout << "deletes=" << required_delete_ids << std::endl;
  std::cout << "update_phase_time_sec=" << workload_time_s << std::endl;
  std::cout << "inserts_per_sec=" << insert_per_sec << std::endl;
  std::cout << "deletes_per_sec=" << delete_per_sec << std::endl;
  std::cout << "query_count=" << total_query_count << std::endl;
  std::cout << "query_time_s=" << total_query_time_s << std::endl;
  std::cout << "queries_per_sec=" << query_qps << std::endl;
  std::cout << "update_rounds=" << cfg.n_iters << std::endl;
  std::cout << "updates_per_sec=" << updates_per_sec << std::endl;
  bool first = true;
  std::cout << "{";
  ann_bench::json_kv(std::cout, first, "baseline", "PipeANN");
  ann_bench::json_kv(std::cout, first, "workload", "mixed_update");
  ann_bench::json_kv(std::cout, first, "phase", "workload");
  ann_bench::json_kv(std::cout, first, "elapsed_s", elapsed_seconds);
  ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(required_insert_ids));
  ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(required_delete_ids));
  ann_bench::json_kv(std::cout, first, "query_count", static_cast<uint64_t>(total_query_count));
  ann_bench::json_kv(std::cout, first, "foreground_time_s", total_foreground_time_s);
  ann_bench::json_kv(std::cout, first, "insert_foreground_time_s", total_insert_time_s);
  ann_bench::json_kv(std::cout, first, "delete_foreground_time_s", total_delete_time_s);
  ann_bench::json_kv(std::cout, first, "query_foreground_time_s", total_query_time_s);
  ann_bench::json_kv(std::cout, first, "query_time_s", total_query_time_s);
  ann_bench::json_kv(std::cout, first, "maintenance_time_s", maintenance_time_s);
  ann_bench::json_kv(std::cout, first, "post_merge_query_time_s",
                     total_post_merge_query_time_s);
  ann_bench::json_kv(std::cout, first, "workload_time_s", workload_time_s);
  ann_bench::json_kv(std::cout, first, "query_qps", query_qps);
  ann_bench::json_kv(std::cout, first, "insert_ops_per_s", insert_per_sec);
  ann_bench::json_kv(std::cout, first, "delete_ops_per_s", delete_per_sec);
  ann_bench::json_kv(std::cout, first, "update_ops_per_s", updates_per_sec);
  ann_bench::json_kv(std::cout, first, "overall_ops_per_s", overall_ops_per_s);
  ann_bench::emit_latency(std::cout, first, "query", query_lat);
  ann_bench::emit_latency(std::cout, first, "insert_call", insert_call_lat);
  ann_bench::emit_latency(std::cout, first, "delete_call", delete_call_lat);
  ann_bench::json_kv(std::cout, first, "avg_rss_mb",
                     enable_interval_metrics ? interval_metrics.avg_rss_mb()
                                             : rss_sampler.avg_mb());
  ann_bench::json_kv(std::cout, first, "peak_rss_mb",
                     enable_interval_metrics ? interval_metrics.peak_rss_mb()
                                             : rss_sampler.peak_mb());
  ann_bench::emit_logical_io(std::cout, first, logical_io_delta,
                             workload_time_s, total_ops);
  std::cout << "}" << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    Config cfg = parse_args(argc, argv);
    if (cfg.type == "float") {
      pipeann::DistanceL2Float dist_cmp;
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
