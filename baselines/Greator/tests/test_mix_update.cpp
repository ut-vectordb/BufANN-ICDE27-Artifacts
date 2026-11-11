// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Mixed insert/delete rounds with merge (benchmark harness update workload).

#include "v2/index_merger.h"
#include "v2/merge_insert.h"

#include <algorithm>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <cerrno>
#include <cstring>
#include <limits>
#include <ctime>
#include <timer.h>
#include <iomanip>
#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "aux_utils.h"
#include "utils.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "../../src/ann_bench_metrics.h"
#include "../../src/streaming_query_source.h"
#include "../../src/weighted_update_pool.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <pthread.h>
#include <sched.h>

struct Config {
  std::string           type;
  std::string           working_folder;
  std::string           base_prefix;
  std::string           merge_prefix;
  std::string           mem_prefix;
  unsigned              l_mem;
  float                 alpha_mem;
  unsigned              l_disk;
  float                 alpha_disk;
  std::string           full_data_bin;
  bool                  single_file;
  std::string           query_bin;
  size_t                n_iters;
  uint32_t              insert_count;
  uint32_t              delete_count;
  uint32_t              range;
  uint32_t              recall_k;
  uint32_t              pool_threads;   // unified insert/delete/query worker pool size
  uint32_t              merge_threads;
  std::vector<uint32_t> search_ls;
  std::string           insert_ids_file;
  std::string           delete_ids_file;
  uint32_t              nodes_to_cache;
  uint32_t              beamwidth;
  uint32_t              merge_maxc;
  uint32_t              skip_update_search;
  uint32_t              query_ratio;
  uint32_t              merge_query_threads;
  uint64_t              tail_query_begin;
  uint64_t              tail_query_end;
  std::string           gt_rounds_dir;
  uint32_t              recall_search_L;
  uint32_t              enable_interval_metrics;  // 0/1
};

tsl::robin_map<std::string, uint32_t> params;
float                                 mem_alpha, merge_alpha;
std::vector<uint32_t> Lvec;
diskann::Timer        global_timer;
std::string           all_points_file;
bool                  save_index_as_one_file;
std::string           query_file = "";
uint64_t              update_query_count = 0;
double                update_query_time_s = 0.0;
double                insert_phase_time_s = 0.0;
double                delete_phase_time_s = 0.0;
double                maintenance_phase_time_s = 0.0;
double                foreground_phase_time_s = 0.0;
double                post_merge_query_phase_time_s = 0.0;
ann_bench::LatencySampler query_latency_sampler;
ann_bench::QueryCycleWindowMetrics *phase_window_metrics = nullptr;
std::chrono::steady_clock::time_point workload_start_time;
std::vector<double>   insert_call_latencies_us;
std::vector<double>   delete_call_latencies_us;
ann_bench::RssSampler *phase_rss_sampler = nullptr;
ann_bench::LogicalIoSnapshot phase_logical_io_delta;
std::vector<uint32_t> scripted_insert_ids;
std::vector<uint32_t> scripted_delete_ids;
size_t                scripted_insert_cursor = 0;
size_t                scripted_delete_cursor = 0;

uint64_t parse_u64(const char *value, const std::string &name) {
  errno = 0;
  char              *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    throw std::invalid_argument("invalid integer for " + name + ": " + value);
  }
  return static_cast<uint64_t>(parsed);
}

uint32_t parse_u32(const char *value, const std::string &name,
                   bool allow_zero = false) {
  uint64_t parsed = parse_u64(value, name);
  if ((!allow_zero && parsed == 0) ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid uint32 for " + name + ": " + value);
  }
  return static_cast<uint32_t>(parsed);
}

float parse_float(const char *value, const std::string &name) {
  errno = 0;
  char *end = nullptr;
  float parsed = std::strtof(value, &end);
  if (errno != 0 || end == value || *end != '\0') {
    throw std::invalid_argument("invalid float for " + name + ": " + value);
  }
  return parsed;
}

Config parse_args(int argc, char **argv) {
  if (argc < 34) {
    std::cerr << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <WORKING_FOLDER> <base_prefix>"
              << " <merge_prefix> <mem_prefix> <L_mem> <alpha_mem> <L_disk>"
              << " <alpha_disk> <full_data_bin> <single_file[0/1]> <query_bin>"
              << " <n_iters> <insert_count_per_iter>"
              << " <delete_count_per_iter> <range> <recall_k> <pool_threads>"
              << " <merge_threads> <insert_ids_file>"
              << " <delete_ids_file>"
              << " <nodes_to_cache> <beamwidth> <merge_maxc>"
              << " <skip_update_search[0/1]>"
              << " <query_ratio> <merge_query_threads>"
              << " <tail_query_begin> <tail_query_end> <gt_rounds_dir> <recall_search_L>"
              << " <enable_interval_metrics[0/1]>"
              << " <search_L1> <search_L2> ..." << std::endl;
    throw std::invalid_argument("wrong number of arguments");
  }

  int    arg_no = 1;
  Config cfg;
  cfg.type = argv[arg_no++];
  cfg.working_folder = argv[arg_no++];
  cfg.base_prefix = argv[arg_no++];
  cfg.merge_prefix = argv[arg_no++];
  cfg.mem_prefix = argv[arg_no++];
  cfg.l_mem = parse_u32(argv[arg_no++], "L_mem");
  cfg.alpha_mem = parse_float(argv[arg_no++], "alpha_mem");
  cfg.l_disk = parse_u32(argv[arg_no++], "L_disk");
  cfg.alpha_disk = parse_float(argv[arg_no++], "alpha_disk");
  cfg.full_data_bin = argv[arg_no++];
  uint32_t single_file = parse_u32(argv[arg_no++], "single_file", true);
  if (single_file > 1) {
    throw std::invalid_argument("single_file must be 0 or 1");
  }
  cfg.single_file = single_file == 1;
  cfg.query_bin = argv[arg_no++];
  cfg.n_iters = static_cast<size_t>(parse_u64(argv[arg_no++], "n_iters"));
  cfg.insert_count = parse_u32(argv[arg_no++], "insert_count");
  cfg.delete_count = parse_u32(argv[arg_no++], "delete_count");
  cfg.range = parse_u32(argv[arg_no++], "range");
  cfg.recall_k = parse_u32(argv[arg_no++], "recall_k");
  cfg.pool_threads = parse_u32(argv[arg_no++], "pool_threads");
  cfg.merge_threads = parse_u32(argv[arg_no++], "merge_threads");
  cfg.insert_ids_file = argv[arg_no++];
  cfg.delete_ids_file = argv[arg_no++];
  cfg.nodes_to_cache = parse_u32(argv[arg_no++], "nodes_to_cache", true);
  cfg.beamwidth = parse_u32(argv[arg_no++], "beamwidth");
  cfg.merge_maxc = parse_u32(argv[arg_no++], "merge_maxc");
  cfg.skip_update_search =
      parse_u32(argv[arg_no++], "skip_update_search", true);
  cfg.query_ratio = parse_u32(argv[arg_no++], "query_ratio", true);
  cfg.merge_query_threads = parse_u32(argv[arg_no++], "merge_query_threads", true);
  cfg.tail_query_begin = parse_u64(argv[arg_no++], "tail_query_begin");
  cfg.tail_query_end = parse_u64(argv[arg_no++], "tail_query_end");
  cfg.gt_rounds_dir = argv[arg_no++];
  cfg.recall_search_L = parse_u32(argv[arg_no++], "recall_search_L");
  cfg.enable_interval_metrics =
      parse_u32(argv[arg_no++], "enable_interval_metrics", true);
  if (cfg.skip_update_search > 1 || cfg.enable_interval_metrics > 1) {
    throw std::invalid_argument(
        "skip_update_search and enable_interval_metrics must be 0 or 1");
  }
  for (int ctr = arg_no; ctr < argc; ctr++) {
    uint32_t cur_l = parse_u32(argv[ctr], "search_L");
    if (cur_l >= cfg.recall_k) {
      cfg.search_ls.push_back(cur_l);
    }
  }
  if (cfg.n_iters == 0) {
    throw std::invalid_argument("n_iters must be positive");
  }
  if (cfg.insert_count == 0 || cfg.delete_count == 0) {
    throw std::invalid_argument(
        "insert_count and delete_count must both be positive");
  }
  if (cfg.search_ls.empty()) {
    throw std::invalid_argument(
        "at least one search_L >= recall_k is required");
  }
  if (cfg.query_ratio == 0 && cfg.skip_update_search == 0) {
    throw std::invalid_argument("query_ratio must be > 0 when skip_update_search=0");
  }
  return cfg;
}

uint32_t get_merge_id_map() {
  const char *value = std::getenv("GREATOR_ID_MAP");
  if (value == nullptr || *value == '\0') {
    return 2;
  }
  return parse_u32(value, "GREATOR_ID_MAP");
}

bool should_output_merged_index_to_merge_prefix() {
  const char *value = std::getenv("GREATOR_OUTPUT_TO_MERGE_PREFIX");
  return value != nullptr && std::strcmp(value, "0") != 0 && *value != '\0';
}

std::vector<uint32_t> load_scripted_ids_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open scripted IDs file: " + path);
  }

  int32_t count_i32 = 0;
  input.read(reinterpret_cast<char *>(&count_i32), sizeof(count_i32));
  if (!input || count_i32 < 0) {
    throw std::runtime_error("invalid scripted IDs header: " + path);
  }

  const size_t          count = static_cast<size_t>(count_i32);
  std::vector<uint32_t> ids(count);
  input.read(reinterpret_cast<char *>(ids.data()),
             static_cast<std::streamsize>(count * sizeof(uint32_t)));
  if (!input) {
    throw std::runtime_error("short read in scripted IDs file: " + path);
  }

  char extra = 0;
  input.read(&extra, 1);
  if (!input.eof()) {
    throw std::runtime_error(
        "unexpected trailing bytes in scripted IDs file: " + path);
  }
  return ids;
}

std::vector<uint32_t> take_scripted_ids(
    const std::vector<uint32_t> &source, size_t &cursor, const uint32_t count,
    const std::string &label) {
  if (count == 0) {
    return {};
  }

  if (cursor + count > source.size()) {
    throw std::runtime_error("not enough scripted " + label +
                             " IDs for requested workload");
  }

  std::vector<uint32_t> picked;
  picked.reserve(count);
  for (size_t idx = cursor; idx < cursor + count; ++idx) {
    picked.push_back(source[idx]);
  }
  cursor += count;
  return picked;
}

template<typename T, typename TagT = uint32_t>
void seed_iter(const std::string &inserted_points_file,
               const std::string &inserted_tags_file,
               tsl::robin_set<TagT> &deleted_tags) {
  const uint32_t insert_count = params[std::string("insert_count")];
  const uint32_t delete_count = params[std::string("delete_count")];
  const uint32_t ndims = params[std::string("ndims")];

  // ID files are mandatory and pre-sized for the complete workload.
  std::vector<uint32_t> delete_vec =
      take_scripted_ids(::scripted_delete_ids, ::scripted_delete_cursor,
                        delete_count, "delete");
  for (auto iter : delete_vec)
    deleted_tags.insert(iter);
  std::cout << "ITER: DELETE - " << delete_vec.size() << " IDs\n";

  std::vector<uint32_t> insert_vec =
      take_scripted_ids(::scripted_insert_ids, ::scripted_insert_cursor,
                        insert_count, "insert");

  std::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
            << inserted_tags_file << "\n";
  std::sort(insert_vec.begin(), insert_vec.end());
  TagT *tag_data = new TagT[insert_vec.size()];
  for (size_t i = 0; i < insert_vec.size(); i++)
    tag_data[i] = insert_vec[i];
  diskann::save_bin<TagT>(inserted_tags_file, tag_data, insert_vec.size(), 1);
  delete[] tag_data;

  // use ifstream reader to load node coordinates
  std::ifstream base_reader;
  base_reader.open(::all_points_file, std::ios::binary | std::ios::ate);
  if (!base_reader.is_open()) {
    throw std::runtime_error("failed to open base data file for inserts: " +
                             ::all_points_file);
  }

  base_reader.seekg(0, std::ios::beg);
  int32_t src_npts_i32 = 0, src_ndims_i32 = 0;
  base_reader.read(reinterpret_cast<char *>(&src_npts_i32),
                   sizeof(src_npts_i32));
  base_reader.read(reinterpret_cast<char *>(&src_ndims_i32),
                   sizeof(src_ndims_i32));
  if (!base_reader || src_npts_i32 <= 0 || src_ndims_i32 <= 0) {
    throw std::runtime_error(
        "failed to read valid metadata from insert source "
        "file: " +
        ::all_points_file);
  }

  const uint32_t src_npts = static_cast<uint32_t>(src_npts_i32);
  const uint32_t src_ndims = static_cast<uint32_t>(src_ndims_i32);
  if (src_ndims != ndims) {
    throw std::runtime_error(
        "insert source dims mismatch. expected " + std::to_string(ndims) +
        ", got " + std::to_string(src_ndims) + " from " + ::all_points_file);
  }

  base_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);

  std::ofstream inserted_points_writer(inserted_points_file, std::ios::binary);
  if (!inserted_points_writer.is_open()) {
    throw std::runtime_error("failed to open inserted points output file: " +
                             inserted_points_file);
  }
  T *new_pts = new T[insert_vec.size() * (size_t) ndims];
  for (uint64_t idx = 0; idx < insert_vec.size(); idx++) {
    uint32_t actual_idx = insert_vec[idx];
    if (actual_idx >= src_npts) {
      throw std::runtime_error("insert ID out of range for source data file: " +
                               std::to_string(actual_idx) +
                               " >= " + std::to_string(src_npts));
    }
    T                   *point = new T[ndims];
    const std::streamoff point_off = static_cast<std::streamoff>(
        2 * sizeof(uint32_t) + actual_idx * (uint64_t) ndims * sizeof(T));
    base_reader.seekg(point_off, std::ios::beg);
    if (!base_reader) {
      delete[] point;
      throw std::runtime_error("seek failed while reading insert ID " +
                               std::to_string(actual_idx) + " from " +
                               ::all_points_file);
    }
    base_reader.read((char *) point, ((uint64_t) ndims) * sizeof(T));
    if (!base_reader) {
      delete[] point;
      throw std::runtime_error("read failed while loading insert ID " +
                               std::to_string(actual_idx) + " from " +
                               ::all_points_file);
    }
    T *dest_ptr = new_pts + idx * (uint64_t) ndims;
    std::memcpy(dest_ptr, point, ndims * sizeof(T));
    delete[] point;
  }
  base_reader.close();

  uint32_t npts_u32 = (uint32_t) insert_vec.size();
  uint32_t ndims_u32 = ndims;
  inserted_points_writer.write((char *) &npts_u32, sizeof(uint32_t));
  inserted_points_writer.write((char *) &ndims_u32, sizeof(uint32_t));
  inserted_points_writer.write(
      (char *) new_pts,
      (uint64_t) insert_vec.size() * (uint64_t) ndims * sizeof(T));
  inserted_points_writer.close();
  delete[] new_pts;
}

template<typename T>
void merge_kernel(diskann::MergeInsert<T> &merge_insert) {
  diskann::Timer timer;
  merge_insert.final_merge(2);
  const auto merge_us = timer.elapsed();
  ::maintenance_phase_time_s += static_cast<double>(merge_us) / 1000000.0;
  std::cout << "Merge time : " << merge_us / 1000 << " ms" << std::endl;
}

template<typename T, typename TagT = uint32_t>
void run_single_iter(const Config &cfg, size_t iter, diskann::MergeInsert<T> &merge_insert,
                     const std::string &,
                     const std::string &, const std::string &mem_prefix,
                     ann_bench::StreamingQuerySource<T> *query_pool,
                     diskann::Distance<T> *) {
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_tags_file = mem_prefix + ".tags_orig";

  std::cout << "ITER: Seeding iteration"
            << "\n";
  tsl::robin_set<uint32_t> deleted_tags;
  seed_iter<T, TagT>(mem_pts_file, mem_tags_file, deleted_tags);

  // Insert payload for this round (mem file written by seed_iter).
  T *    data_insert = nullptr;
  size_t insert_n = 0, ndim = 0, aligned_dim = 0;
  diskann::load_aligned_bin<T>(mem_pts_file, data_insert, insert_n, ndim,
                               aligned_dim);
  size_t tag_num = 0, tag_dim = 0;
  TagT * tag_data = nullptr;
  diskann::load_bin<TagT>(mem_tags_file, tag_data, tag_num, tag_dim);
  if (tag_num != insert_n) {
    throw std::runtime_error(
        "insert tags count != insert points count in seeded round");
  }

  // Delete payload for this round.
  std::vector<uint32_t> delete_ids(deleted_tags.begin(), deleted_tags.end());
  const size_t          delete_n = delete_ids.size();

  // Query payload: concurrent stress queries drawn cyclically from the query
  // pool, sharing the worker pool with inserts and deletes. These are disabled
  // by skip_update_search; per-round/final recall is measured separately.
  const uint64_t recall_at = cfg.recall_k;
  const bool     query_enabled =
      cfg.skip_update_search == 0 && query_pool != nullptr &&
      query_pool->pool_size() > 0 && cfg.query_ratio > 0 &&
      !::Lvec.empty();
  const size_t query_n =
      query_enabled ? ann_bench::mixed_update_query_count(
                          insert_n, delete_n, cfg.query_ratio)
                    : 0;
  const uint32_t query_L = query_enabled ? ::Lvec.front() : 0;
  // Per-index result/latency slots (written without contention by the pool).
  std::vector<double>              insert_latencies_us(insert_n, 0.0);
  std::vector<double>              delete_latencies_us(delete_n, 0.0);

  std::function<void(uint64_t)> tasks[ann_bench::WeightedUpdatePool::NUM_OPS];
  tasks[ann_bench::WeightedUpdatePool::INSERT] = [&](uint64_t i) {
    const auto s = std::chrono::high_resolution_clock::now();
    if (merge_insert.insert(data_insert + i * aligned_dim, tag_data[i]) != 0) {
      std::cout << "Point " << i << " could not be inserted." << std::endl;
    }
    const auto e = std::chrono::high_resolution_clock::now();
    insert_latencies_us[i] =
        std::chrono::duration<double, std::micro>(e - s).count();
    if (::phase_window_metrics != nullptr) {
      ::phase_window_metrics->record_insert(insert_latencies_us[i]);
    }
  };
  tasks[ann_bench::WeightedUpdatePool::DELETE] = [&](uint64_t i) {
    const auto s = std::chrono::high_resolution_clock::now();
    merge_insert.lazy_delete(delete_ids[i]);
    const auto e = std::chrono::high_resolution_clock::now();
    delete_latencies_us[i] =
        std::chrono::duration<double, std::micro>(e - s).count();
    if (::phase_window_metrics != nullptr) {
      ::phase_window_metrics->record_delete(delete_latencies_us[i]);
    }
  };
  tasks[ann_bench::WeightedUpdatePool::QUERY] = [&](uint64_t i) {
    thread_local typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
    qstate.result_tags.resize(recall_at);
    qstate.result_dists.resize(recall_at);
    diskann::QueryStats stats;
    const T *query = query_pool->get(i, qstate);
    const auto   s = std::chrono::high_resolution_clock::now();
    merge_insert.search_sync(
        query, recall_at, query_L, qstate.result_tags.data(),
        qstate.result_dists.data(), &stats);
    const auto e = std::chrono::high_resolution_clock::now();
    const double lat_us = std::chrono::duration<double, std::micro>(e - s).count();
    if (::phase_window_metrics != nullptr) {
      ::phase_window_metrics->record_query(lat_us, iter, "foreground");
    } else {
      ::query_latency_sampler.record(i, lat_us);
    }
  };

  const uint64_t counts[ann_bench::WeightedUpdatePool::NUM_OPS] = {
      static_cast<uint64_t>(insert_n), static_cast<uint64_t>(delete_n),
      static_cast<uint64_t>(query_n)};

  if (::phase_window_metrics != nullptr) {
    ::phase_window_metrics->resume_rss();
  } else if (::phase_rss_sampler != nullptr) {
    ::phase_rss_sampler->resume();
  }
  ann_bench::LogicalIoCounter::reset_and_enable();
  if (::phase_window_metrics != nullptr) {
    ::phase_window_metrics->reset_io_baseline();
  }
  const auto foreground_begin = std::chrono::steady_clock::now();
  if (ann_bench::env_equals("UPDATE_POOL_MODE", "round_robin")) {
    ann_bench::WeightedUpdatePool::run_rq2_round_robin_90_5_5(
        cfg.pool_threads, counts, tasks);
  } else {
    ann_bench::WeightedUpdatePool::run(cfg.pool_threads, counts, tasks);
  }
  const auto   foreground_end = std::chrono::steady_clock::now();
  const double round_wall_s =
      std::chrono::duration<double>(foreground_end - foreground_begin).count();
  ::foreground_phase_time_s += round_wall_s;
  if (::phase_window_metrics != nullptr) {
    ::phase_window_metrics->flush(iter, "foreground");
    ann_bench::emit_phase_checkpoint(
        std::cout, "Greator", "foreground", iter,
        std::chrono::duration<double>(foreground_end - ::workload_start_time).count(),
        static_cast<uint64_t>(insert_n), static_cast<uint64_t>(delete_n),
        static_cast<uint64_t>(query_n), round_wall_s);
  }

  // RQ2 interval metrics own latency samples; non-interval update metrics keep
  // the legacy workload-level distributions.
  if (::phase_window_metrics == nullptr) {
    ::insert_call_latencies_us.insert(::insert_call_latencies_us.end(),
                                      insert_latencies_us.begin(),
                                      insert_latencies_us.end());
    ::delete_call_latencies_us.insert(::delete_call_latencies_us.end(),
                                      delete_latencies_us.begin(),
                                      delete_latencies_us.end());
  }
  // Aggregate per-op busy time (sum of op latencies; >= wall under parallelism).
  ::insert_phase_time_s +=
      std::accumulate(insert_latencies_us.begin(), insert_latencies_us.end(),
                      0.0) / 1000000.0;
  ::delete_phase_time_s +=
      std::accumulate(delete_latencies_us.begin(), delete_latencies_us.end(),
                      0.0) / 1000000.0;
  if (query_enabled) {
    ::update_query_count += query_n;
  }

  delete[] data_insert;
  delete[] tag_data;

  const auto merge_begin = std::chrono::steady_clock::now();
  std::atomic<bool> merge_done{false};
  const uint64_t merge_query_budget =
      ann_bench::env_u64_or("MERGE_QUERY_BUDGET", 0);
  std::atomic<uint64_t> next_merge_query{0};
  std::thread merge_thread([&]() {
    merge_kernel<T>(merge_insert);
    merge_done.store(true, std::memory_order_release);
  });
  std::vector<std::thread> merge_query_threads;
  std::vector<uint64_t> merge_query_counts(cfg.merge_query_threads, 0);
  if (query_enabled && cfg.merge_query_threads > 0) {
    for (uint32_t t = 0; t < cfg.merge_query_threads; t++) {
      merge_query_threads.emplace_back([&, t]() {
        typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
        qstate.result_tags.resize(recall_at);
        qstate.result_dists.resize(recall_at);
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
          diskann::QueryStats stats;
          const T *query = query_pool->get(seq, qstate);
          const auto s = std::chrono::high_resolution_clock::now();
          merge_insert.search_sync(query, recall_at, query_L,
              qstate.result_tags.data(), qstate.result_dists.data(), &stats);
          const auto e = std::chrono::high_resolution_clock::now();
          const double lat_us =
              std::chrono::duration<double, std::micro>(e - s).count();
          if (::phase_window_metrics != nullptr) {
            ::phase_window_metrics->record_query(lat_us, iter, "merge");
          } else {
            ::query_latency_sampler.record(seq, lat_us);
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
  const auto merge_end = std::chrono::steady_clock::now();
  const double round_merge_s =
      std::chrono::duration<double>(merge_end - merge_begin).count();
  uint64_t merge_query_count = 0;
  for (uint64_t c : merge_query_counts) merge_query_count += c;
  uint64_t post_merge_query_count = 0;
  double post_merge_query_s = 0.0;
  if (query_enabled && merge_query_budget > merge_query_count) {
    const uint64_t remaining = merge_query_budget - merge_query_count;
    const uint32_t tail_threads = cfg.pool_threads;
    const auto tail_begin = std::chrono::steady_clock::now();
    std::atomic<uint64_t> next_tail{0};
    std::vector<std::thread> tail_query_threads;
    const uint32_t effective_tail_threads = tail_threads == 0 ? 1 : tail_threads;
    tail_query_threads.reserve(effective_tail_threads);
    for (uint32_t t = 0; t < effective_tail_threads; t++) {
      tail_query_threads.emplace_back([&]() {
        typename ann_bench::StreamingQuerySource<T>::ThreadState qstate;
        qstate.result_tags.resize(recall_at);
        qstate.result_dists.resize(recall_at);
        for (;;) {
          const uint64_t local =
              next_tail.fetch_add(1, std::memory_order_relaxed);
          if (local >= remaining) {
            break;
          }
          const uint64_t seq =
              static_cast<uint64_t>(query_n) + merge_query_count + local;
          diskann::QueryStats stats;
          const T *query = query_pool->get(seq, qstate);
          const auto s = std::chrono::high_resolution_clock::now();
          merge_insert.search_sync(query, recall_at, query_L,
              qstate.result_tags.data(), qstate.result_dists.data(), &stats);
          const auto e = std::chrono::high_resolution_clock::now();
          const double lat_us =
              std::chrono::duration<double, std::micro>(e - s).count();
          if (::phase_window_metrics != nullptr) {
            ::phase_window_metrics->record_query(lat_us, iter, "post_merge_query");
          } else {
            ::query_latency_sampler.record(seq, lat_us);
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
  ::update_query_count += merge_query_count + post_merge_query_count;
  ::post_merge_query_phase_time_s += post_merge_query_s;
  if (::phase_window_metrics != nullptr) {
    ::phase_window_metrics->flush(iter, "merge");
    ann_bench::emit_phase_checkpoint(
        std::cout, "Greator", "merge", iter,
        std::chrono::duration<double>(merge_end - ::workload_start_time).count(),
        0, 0, merge_query_count, round_merge_s);
    if (post_merge_query_count > 0) {
      ::phase_window_metrics->flush(iter, "post_merge_query");
      ann_bench::emit_phase_checkpoint(
          std::cout, "Greator", "post_merge_query", iter,
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - ::workload_start_time).count(),
          0, 0, post_merge_query_count, post_merge_query_s);
    }
  }
  const double round_total_s = round_wall_s + round_merge_s + post_merge_query_s;
  if (query_enabled) {
    ::update_query_time_s += round_total_s;
  }
  ann_bench::add_logical_io(::phase_logical_io_delta,
                            ann_bench::LogicalIoCounter::snapshot_and_disable());
  ann_bench::RssStats round_rss;
  if (::phase_window_metrics != nullptr) {
    round_rss = ::phase_window_metrics->pause_and_take_round_rss();
  } else if (::phase_rss_sampler != nullptr) {
    round_rss = ::phase_rss_sampler->stop_and_take_interval();
  }
  const double round_elapsed_before_recall =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - ::workload_start_time).count();

  double round_recall = -1.0;
  if (!cfg.gt_rounds_dir.empty() && query_pool->query_data != nullptr && query_pool->query_n > 0) {
    const std::string gt_path = cfg.gt_rounds_dir + "/round_" +
        std::to_string(iter) + "_gt" + std::to_string(cfg.recall_k) + ".bin";
    std::ifstream gt_check(gt_path);
    if (gt_check.good()) {
      uint32_t *gt_ids = nullptr, *gt_tags = nullptr;
      float *gt_dists = nullptr;
      size_t gt_num = 0, gt_dim = 0;
      diskann::load_truthset(gt_path, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
      std::vector<uint32_t> res_tags(query_pool->query_n * cfg.recall_k);
      std::vector<float> res_dists(query_pool->query_n * cfg.recall_k);
#pragma omp parallel for num_threads((int) cfg.pool_threads) schedule(dynamic, 1)
      for (int64_t qi = 0; qi < static_cast<int64_t>(query_pool->query_n); qi++) {
        diskann::QueryStats stats;
        merge_insert.search_sync(
            query_pool->query_data + static_cast<size_t>(qi) * query_pool->aligned_dim,
            cfg.recall_k, cfg.recall_search_L,
            res_tags.data() + static_cast<size_t>(qi) * cfg.recall_k,
            res_dists.data() + static_cast<size_t>(qi) * cfg.recall_k, &stats);
      }
      unsigned *gt_eval = (gt_tags != nullptr) ? gt_tags : gt_ids;
      round_recall = diskann::calculate_recall(
          static_cast<unsigned>(query_pool->query_n), gt_eval, gt_dists,
          static_cast<unsigned>(gt_dim), res_tags.data(),
          static_cast<unsigned>(cfg.recall_k), static_cast<unsigned>(cfg.recall_k));
      delete[] gt_ids;
      delete[] gt_dists;
      delete[] gt_tags;
    }
  }

  const uint64_t round_query_count =
      static_cast<uint64_t>(query_n) + merge_query_count + post_merge_query_count;
  const double round_query_qps =
      ann_bench::safe_div(static_cast<double>(round_query_count), round_total_s);
  bool rfirst = true;
  std::cout << std::fixed << std::setprecision(6) << "{";
  ann_bench::json_kv(std::cout, rfirst, "baseline", "Greator");
  ann_bench::json_kv(std::cout, rfirst, "workload", "mixed_update");
  ann_bench::json_kv(std::cout, rfirst, "phase", "round");
  ann_bench::json_kv(std::cout, rfirst, "round_index", static_cast<uint64_t>(iter));
  ann_bench::json_kv(std::cout, rfirst, "elapsed_s", round_elapsed_before_recall);
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
  ann_bench::json_kv(std::cout, rfirst, "workload_time_s", round_total_s);
  ann_bench::emit_rss_stats(std::cout, rfirst, round_rss);
  ann_bench::json_kv(std::cout, rfirst, "query_qps", round_query_qps);
  ann_bench::json_kv(std::cout, rfirst, "insert_ops_per_s",
                     ann_bench::safe_div(static_cast<double>(insert_n), round_total_s));
  ann_bench::json_kv(std::cout, rfirst, "delete_ops_per_s",
                     ann_bench::safe_div(static_cast<double>(delete_n), round_total_s));
  ann_bench::json_kv(std::cout, rfirst, "update_ops_per_s",
                     ann_bench::safe_div(static_cast<double>(insert_n + delete_n), round_total_s));
  if (round_recall >= 0.0) {
    ann_bench::json_kv(std::cout, rfirst, "recall", round_recall);
    ann_bench::json_kv(std::cout, rfirst, "recall_at", static_cast<uint64_t>(cfg.recall_k));
    ann_bench::json_kv(std::cout, rfirst, "recall_search_L", static_cast<uint64_t>(cfg.recall_search_L));
  }
  std::cout << "}" << std::endl;
}

template<typename T, typename TagT = uint32_t>
void run_all_iters(const Config &cfg, diskann::Distance<T> *dist_cmp,
                   ann_bench::StreamingQuerySource<T> *query_pool) {
  const std::string &base_prefix = cfg.base_prefix;
  const std::string &merge_prefix = cfg.merge_prefix;
  const std::string &mem_prefix = cfg.mem_prefix;
  const std::string &data_file = cfg.full_data_bin;
  ::all_points_file = data_file;
  // load all data points
  uint64_t npts = 0, ndims = 0;
  diskann::get_bin_metadata(data_file, npts, ndims);
  std::cout << "Loaded base bin" << std::endl;
  params[std::string("ndims")] = (uint32_t) ndims;

  uint32_t n_iters = params["n_iters"];

  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", params[std::string("mem_l_index")]);
  paras.Set<unsigned>("R_mem", params[std::string("range")]);
  paras.Set<float>("alpha_mem", ::mem_alpha);
  paras.Set<unsigned>("L_disk", params[std::string("merge_l_index")]);
  paras.Set<unsigned>("R_disk", params[std::string("range")]);
  paras.Set<float>("alpha_disk", ::merge_alpha);
  paras.Set<unsigned>("C", params[std::string("merge_maxc")]);
  paras.Set<unsigned>("beamwidth", params[std::string("beam_width")]);
  paras.Set<unsigned>("nodes_to_cache",
                      params[std::string("disk_search_node_cache_count")]);
  paras.Set<unsigned>("num_search_threads",
                      params[std::string("disk_search_nthreads")]);
  paras.Set<uint64_t>("merge_nthreads", params[std::string("merge_threads")]);

  const std::string             working_folder = cfg.working_folder;
  diskann::Metric               metric = diskann::Metric::L2;
  diskann::MergeInsert<T, TagT> merge_insert(
      paras, ndims, mem_prefix, base_prefix, merge_prefix, dist_cmp, metric,
      ::save_index_as_one_file, working_folder);
  const uint32_t merge_id_map = 2;
  if (merge_id_map == 2 && !should_output_merged_index_to_merge_prefix()) {
    merge_insert._disk_index_prefix_out = base_prefix;
  }
  ::workload_start_time = std::chrono::steady_clock::now();
  if (::phase_window_metrics != nullptr && query_pool != nullptr) {
    ::phase_window_metrics->start(
        "Greator", ann_bench::latency_reserve_hint(
                       static_cast<uint64_t>(query_pool->query_n)));
  }
  for (size_t i = 0; i < n_iters; i++) {
    std::cout << "ITER : " << i << std::endl;
    run_single_iter<T>(cfg, i, merge_insert, base_prefix, merge_prefix, mem_prefix,
                       query_pool, dist_cmp);
  }
}

template<typename T>
void run_workload(const Config &cfg, diskann::Distance<T> *dist_cmp) {
  params.clear();
  ::Lvec = cfg.search_ls;
  ::query_file = cfg.query_bin;
  ::update_query_count = 0;
  ::update_query_time_s = 0.0;
  ::insert_phase_time_s = 0.0;
  ::delete_phase_time_s = 0.0;
  ::maintenance_phase_time_s = 0.0;
  ::foreground_phase_time_s = 0.0;
  ::post_merge_query_phase_time_s = 0.0;
  ::query_latency_sampler.reset();
  ::insert_call_latencies_us.clear();
  ::delete_call_latencies_us.clear();
  ::phase_logical_io_delta = ann_bench::LogicalIoSnapshot();
  ::scripted_insert_cursor = 0;
  ::scripted_delete_cursor = 0;
  ::save_index_as_one_file = cfg.single_file;
  ::mem_alpha = cfg.alpha_mem;
  ::merge_alpha = cfg.alpha_disk;

  params[std::string("n_iters")] = static_cast<uint32_t>(cfg.n_iters);
  params[std::string("insert_count")] = cfg.insert_count;
  params[std::string("delete_count")] = cfg.delete_count;
  params[std::string("range")] = cfg.range;
  params[std::string("recall_k")] = cfg.recall_k;
  params[std::string("pool_threads")] = cfg.pool_threads;
  params[std::string("merge_threads")] = cfg.merge_threads;
  params[std::string("disk_search_node_cache_count")] = cfg.nodes_to_cache;
  // The unified pool calls search_sync from up to pool_threads workers, so the
  // disk index must provide at least that many beam-search scratch contexts.
  params[std::string("disk_search_nthreads")] = cfg.pool_threads;
  params[std::string("beam_width")] = cfg.beamwidth;
  params[std::string("mem_l_index")] = cfg.l_mem;
  params[std::string("merge_maxc")] = cfg.merge_maxc;
  params[std::string("merge_l_index")] = cfg.l_disk;

  ::scripted_insert_ids = load_scripted_ids_file(cfg.insert_ids_file);
  ::scripted_delete_ids = load_scripted_ids_file(cfg.delete_ids_file);
  const size_t required_insert_ids =
      cfg.n_iters * static_cast<size_t>(cfg.insert_count);
  const size_t required_delete_ids =
      cfg.n_iters * static_cast<size_t>(cfg.delete_count);
  if (::scripted_insert_ids.size() < required_insert_ids) {
    throw std::runtime_error(
        "scripted insert IDs file does not contain enough IDs");
  }
  if (::scripted_delete_ids.size() < required_delete_ids) {
    throw std::runtime_error(
        "scripted delete IDs file does not contain enough IDs");
  }

  ann_bench::StreamingQuerySource<T> query_pool;
  if (!cfg.query_bin.empty() && cfg.query_bin != "null") {
    query_pool.load(cfg.query_bin, cfg.full_data_bin,
                    cfg.tail_query_begin, cfg.tail_query_end);
  }

  ann_bench::RssSampler rss_sampler;
  ann_bench::QueryCycleWindowMetrics interval_metrics;
  const bool enable_interval_metrics =
      cfg.skip_update_search == 0 && cfg.enable_interval_metrics != 0;
  ::phase_window_metrics =
      enable_interval_metrics ? &interval_metrics : nullptr;
  ::phase_rss_sampler = enable_interval_metrics ? nullptr : &rss_sampler;
  run_all_iters<T>(cfg, dist_cmp, &query_pool);
  if (enable_interval_metrics) {
    const uint64_t last_round =
        cfg.n_iters > 0 ? static_cast<uint64_t>(cfg.n_iters - 1) : 0;
    interval_metrics.finalize(last_round, "workload");
  }
  interval_metrics.pause_rss();
  ::phase_rss_sampler = nullptr;
  ::phase_window_metrics = nullptr;
  const auto workload_end_time = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(workload_end_time - ::workload_start_time).count();

  const uint64_t insert_count =
      static_cast<uint64_t>(cfg.n_iters) * static_cast<uint64_t>(cfg.insert_count);
  const uint64_t delete_count =
      static_cast<uint64_t>(cfg.n_iters) * static_cast<uint64_t>(cfg.delete_count);
  const double maintenance_time_s = ::maintenance_phase_time_s;
  const double foreground_time_s = ::foreground_phase_time_s;
  const double workload_time_s =
      foreground_time_s + maintenance_time_s + ::post_merge_query_phase_time_s;
  const double insert_ops_per_s =
      ann_bench::safe_div(static_cast<double>(insert_count), workload_time_s);
  const double delete_ops_per_s =
      ann_bench::safe_div(static_cast<double>(delete_count), workload_time_s);
  const double update_ops_per_s =
      ann_bench::safe_div(static_cast<double>(insert_count + delete_count),
                          workload_time_s);
  const uint64_t total_ops = insert_count + delete_count + ::update_query_count;
  const double overall_ops_per_s =
      ann_bench::safe_div(static_cast<double>(total_ops), workload_time_s);
  const double query_qps =
      ann_bench::safe_div(static_cast<double>(::update_query_count),
                          ::update_query_time_s);
  ann_bench::LatencyStats query_lat =
      enable_interval_metrics ? interval_metrics.query_latency_summary()
                              : ::query_latency_sampler.summarize();
  ann_bench::LatencyStats insert_call_lat =
      enable_interval_metrics
          ? interval_metrics.insert_latency_summary()
          : ann_bench::summarize_latencies(::insert_call_latencies_us);
  ann_bench::LatencyStats delete_call_lat =
      enable_interval_metrics
          ? interval_metrics.delete_latency_summary()
          : ann_bench::summarize_latencies(::delete_call_latencies_us);
  const auto io_delta = ::phase_logical_io_delta;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "inserts=" << insert_count << std::endl;
  std::cout << "deletes=" << delete_count << std::endl;
  std::cout << "update_phase_time_sec=" << workload_time_s << std::endl;
  std::cout << "maintenance_time_s=" << maintenance_time_s << std::endl;
  std::cout << "inserts_per_sec=" << insert_ops_per_s << std::endl;
  std::cout << "deletes_per_sec=" << delete_ops_per_s << std::endl;
  std::cout << "query_count=" << ::update_query_count << std::endl;
  std::cout << "query_time_s=" << ::update_query_time_s << std::endl;
  std::cout << "queries_per_sec=" << query_qps << std::endl;

  bool first = true;
  std::cout << "{";
  ann_bench::json_kv(std::cout, first, "baseline", "Greator");
  ann_bench::json_kv(std::cout, first, "workload", "mixed_update");
  ann_bench::json_kv(std::cout, first, "phase", "workload");
  ann_bench::json_kv(std::cout, first, "elapsed_s", elapsed_s);
  ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(insert_count));
  ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(delete_count));
  ann_bench::json_kv(std::cout, first, "query_count", static_cast<uint64_t>(::update_query_count));
  ann_bench::json_kv(std::cout, first, "foreground_time_s", foreground_time_s);
  ann_bench::json_kv(std::cout, first, "insert_foreground_time_s", ::insert_phase_time_s);
  ann_bench::json_kv(std::cout, first, "delete_foreground_time_s", ::delete_phase_time_s);
  ann_bench::json_kv(std::cout, first, "query_foreground_time_s", ::update_query_time_s);
  ann_bench::json_kv(std::cout, first, "query_time_s", ::update_query_time_s);
  ann_bench::json_kv(std::cout, first, "maintenance_time_s", maintenance_time_s);
  ann_bench::json_kv(std::cout, first, "post_merge_query_time_s",
                     ::post_merge_query_phase_time_s);
  ann_bench::json_kv(std::cout, first, "workload_time_s", workload_time_s);
  ann_bench::json_kv(std::cout, first, "query_qps", query_qps);
  ann_bench::json_kv(std::cout, first, "insert_ops_per_s", insert_ops_per_s);
  ann_bench::json_kv(std::cout, first, "delete_ops_per_s", delete_ops_per_s);
  ann_bench::json_kv(std::cout, first, "update_ops_per_s", update_ops_per_s);
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
  ann_bench::emit_logical_io(std::cout, first, io_delta, workload_time_s,
                             static_cast<uint64_t>(total_ops));
  std::cout << "}" << std::endl;
}

int main(int argc, char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    Config cfg = parse_args(argc, argv);
    if (cfg.type == "float") {
      diskann::DistanceL2 dist_cmp;
      run_workload<float>(cfg, &dist_cmp);
    } else if (cfg.type == "uint8") {
      diskann::DistanceL2UInt8 dist_cmp;
      run_workload<uint8_t>(cfg, &dist_cmp);
    } else if (cfg.type == "int8") {
      diskann::DistanceL2Int8 dist_cmp;
      run_workload<int8_t>(cfg, &dist_cmp);
    } else {
      throw std::invalid_argument("unsupported type: " + cfg.type);
    }
    std::cout.flush();
    std::cerr.flush();
    // libomp + TBB atexit handlers can conflict with each other when both are
    // active.  GREATOR_BENCHMARK_QUICK_EXIT=1 bypasses normal teardown for
    // benchmark runs where only the output metrics matter.
    if (std::getenv("GREATOR_BENCHMARK_QUICK_EXIT") != nullptr) {
      std::_Exit(0);
    }
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
