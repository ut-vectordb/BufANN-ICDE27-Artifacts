// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "aux_utils.h"
#include "distance.h"
#include "math_utils.h"
#include "parameters.h"
#include "timer.h"
#include "utils.h"
#include "v2/merge_insert.h"
#include "../../src/ann_bench_metrics.h"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace {

enum class WorkloadMode { INSERT, DELETE };

struct Config {
  WorkloadMode mode;
  std::string  type;
  std::string  working_folder;
  std::string  base_prefix;
  std::string  merge_prefix;
  std::string  mem_prefix;
  unsigned     l_mem;
  float        alpha_mem;
  unsigned     l_disk;
  float        alpha_disk;
  std::string  full_data_bin;
  bool         single_file;
  size_t       n_iters;
  uint32_t     count_per_iter;
  uint32_t     range;
  uint32_t     beamwidth;
  uint32_t     nthreads;
  std::string  ids_file;
  // Optional per-round recall measurement (enabled when the trailing
  // query/gt args are supplied). Searches the live merged index after each
  // round via MergeInsert::search_sync, so no index reload is needed.
  bool         recall_enabled = false;
  std::string  query_bin;
  std::string  gt_dir;  // dir of per-round GT: round_<k>_gt<recall_at>.bin
  uint32_t     recall_at = 10;
  uint32_t     recall_search_L = 100;
};

uint64_t parse_u64(const char *value, const std::string &name) {
  errno = 0;
  char *end = nullptr;
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

WorkloadMode parse_mode(const std::string &mode) {
  if (mode == "insert") {
    return WorkloadMode::INSERT;
  }
  if (mode == "delete") {
    return WorkloadMode::DELETE;
  }
  throw std::invalid_argument("mode must be insert or delete");
}

Config parse_args(int argc, char **argv) {
  if (argc != 19 && argc != 23) {
    std::cerr
        << "Correct usage: " << argv[0]
        << " <mode[insert/delete]> <type[int8/uint8/float]> <WORKING_FOLDER>"
        << " <base_prefix> <merge_prefix> <mem_prefix>"
        << " <L_mem> <alpha_mem> <L_disk> <alpha_disk>"
        << " <full_data_bin> <single_file[0/1]>"
        << " <n_iters> <count_per_iter> <range> <beamwidth> <nthreads>"
        << " <ids_file>"
        << " [<query_bin> <gt_dir> <recall_at> <recall_search_L>]"
        << std::endl;
    throw std::invalid_argument("wrong number of arguments");
  }

  int arg_no = 1;
  Config cfg;
  cfg.mode = parse_mode(argv[arg_no++]);
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
  cfg.n_iters = static_cast<size_t>(parse_u64(argv[arg_no++], "n_iters"));
  cfg.count_per_iter = parse_u32(argv[arg_no++], "count_per_iter");
  cfg.range = parse_u32(argv[arg_no++], "range");
  cfg.beamwidth = parse_u32(argv[arg_no++], "beamwidth");
  cfg.nthreads = parse_u32(argv[arg_no++], "nthreads");
  cfg.ids_file = argv[arg_no++];
  if (argc == 23) {
    cfg.query_bin = argv[arg_no++];
    cfg.gt_dir = argv[arg_no++];
    cfg.recall_at = parse_u32(argv[arg_no++], "recall_at");
    cfg.recall_search_L = parse_u32(argv[arg_no++], "recall_search_L");
    cfg.recall_enabled = true;
  }
  if (cfg.n_iters == 0) {
    throw std::invalid_argument("n_iters must be positive");
  }
  return cfg;
}

std::vector<uint32_t> load_counted_ids(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open IDs file: " + path);
  }

  int32_t count = 0;
  in.read(reinterpret_cast<char *>(&count), sizeof(int32_t));
  if (!in) {
    throw std::runtime_error("short read from IDs file header: " + path);
  }
  if (count < 0) {
    throw std::runtime_error("negative ID count in file: " + path);
  }

  std::vector<uint32_t> ids(static_cast<size_t>(count));
  if (!ids.empty()) {
    in.read(reinterpret_cast<char *>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    if (!in) {
      throw std::runtime_error("short read from IDs file: " + path);
    }
  }
  return ids;
}

template<typename T>
T *load_insert_batch(const std::string &data_file,
                     const std::vector<uint32_t> &ids, size_t begin,
                     uint32_t count, uint32_t full_npts, uint32_t ndims,
                     uint32_t aligned_dim) {
  T *batch = nullptr;
  const size_t alloc_size =
      static_cast<size_t>(count) * static_cast<size_t>(aligned_dim) *
      sizeof(T);
  diskann::alloc_aligned(reinterpret_cast<void **>(&batch), alloc_size,
                         8 * sizeof(T));
  std::memset(batch, 0, alloc_size);

  std::ifstream reader(data_file, std::ios::binary);
  if (!reader) {
    diskann::aligned_free(batch);
    throw std::runtime_error("cannot open full-data file: " + data_file);
  }

  uint32_t file_npts = 0;
  uint32_t file_ndims = 0;
  reader.read(reinterpret_cast<char *>(&file_npts), sizeof(uint32_t));
  reader.read(reinterpret_cast<char *>(&file_ndims), sizeof(uint32_t));
  if (!reader || file_npts != full_npts || file_ndims != ndims) {
    diskann::aligned_free(batch);
    throw std::runtime_error("full-data metadata changed while reading: " +
                             data_file);
  }

  for (uint32_t i = 0; i < count; i++) {
    uint32_t id = ids[begin + i];
    if (id >= full_npts) {
      diskann::aligned_free(batch);
      throw std::runtime_error("insert ID out of range: " +
                               std::to_string(id));
    }
    const uint64_t offset =
        2 * sizeof(uint32_t) +
        static_cast<uint64_t>(id) * static_cast<uint64_t>(ndims) * sizeof(T);
    reader.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    reader.read(reinterpret_cast<char *>(batch + static_cast<size_t>(i) *
                                                   aligned_dim),
                static_cast<std::streamsize>(ndims * sizeof(T)));
    if (!reader) {
      diskann::aligned_free(batch);
      throw std::runtime_error("failed to read vector ID " +
                               std::to_string(id));
    }
  }

  return batch;
}

template<typename T>
void run_workload(const Config &cfg, diskann::Distance<T> *dist_cmp) {
  uint64_t npts_u64 = 0;
  uint64_t ndims_u64 = 0;
  diskann::get_bin_metadata(cfg.full_data_bin, npts_u64, ndims_u64);
  if (npts_u64 > std::numeric_limits<uint32_t>::max() ||
      ndims_u64 > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("full-data metadata exceeds uint32 limits");
  }
  const uint32_t full_npts = static_cast<uint32_t>(npts_u64);
  const uint32_t ndims = static_cast<uint32_t>(ndims_u64);
  const uint32_t aligned_dim = static_cast<uint32_t>(ROUND_UP(ndims, 8));

  std::vector<uint32_t> ids = load_counted_ids(cfg.ids_file);
  const size_t required_ids =
      cfg.n_iters * static_cast<size_t>(cfg.count_per_iter);
  if (ids.size() < required_ids) {
    throw std::runtime_error("IDs file does not contain enough IDs: need " +
                             std::to_string(required_ids) + ", found " +
                             std::to_string(ids.size()));
  }

  diskann::Parameters paras;
  paras.Set<unsigned>("L_mem", cfg.l_mem);
  paras.Set<unsigned>("R_mem", cfg.range);
  paras.Set<float>("alpha_mem", cfg.alpha_mem);
  paras.Set<unsigned>("L_disk", cfg.l_disk);
  paras.Set<unsigned>("R_disk", cfg.range);
  paras.Set<float>("alpha_disk", cfg.alpha_disk);
  paras.Set<unsigned>("C", static_cast<unsigned>(cfg.range * 2.5));
  paras.Set<unsigned>("beamwidth", cfg.beamwidth);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<_u32>("num_search_threads", cfg.nthreads);
  paras.Set<size_t>("merge_th", static_cast<size_t>(cfg.count_per_iter));
  paras.Set<uint64_t>("merge_nthreads", static_cast<uint64_t>(cfg.nthreads));

  diskann::MergeInsert<T, uint32_t> merge_insert(
      paras, ndims, cfg.mem_prefix, cfg.base_prefix, cfg.merge_prefix, dist_cmp,
      diskann::Metric::L2, cfg.single_file, cfg.working_folder);

  omp_set_dynamic(0);

  // Per-round recall: load queries once, then search the live merged index
  // after each round and score against that round's groundtruth
  // (<gt_dir>/round_<k>_gt<recall_at>.bin, generated by the prepare step).
  T *rq_query = nullptr;
  size_t rq_num = 0, rq_dim = 0, rq_aligned_dim = 0;
  if (cfg.recall_enabled) {
    diskann::load_aligned_bin<T>(cfg.query_bin, rq_query, rq_num, rq_dim,
                                 rq_aligned_dim);
  }
  auto measure_recall = [&](size_t round_idx) -> double {
    if (!cfg.recall_enabled || rq_num == 0) return -1.0;
    const std::string gt_path = cfg.gt_dir + "/round_" +
                                std::to_string(round_idx) + "_gt" +
                                std::to_string(cfg.recall_at) + ".bin";
    if (!file_exists(gt_path)) {
      std::cerr << "WARNING: missing per-round GT " << gt_path
                << "; skipping recall for round " << round_idx << std::endl;
      return -1.0;
    }
    uint32_t *gt_ids = nullptr, *gt_tags = nullptr;
    float *gt_dists = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    diskann::load_truthset(gt_path, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
    std::vector<uint32_t> res_tags(rq_num * cfg.recall_at);
    std::vector<float> res_dists(rq_num * cfg.recall_at);
#pragma omp parallel for num_threads((int) cfg.nthreads) schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(rq_num); i++) {
      diskann::QueryStats stats;
      merge_insert.search_sync(
          rq_query + static_cast<size_t>(i) * rq_aligned_dim, cfg.recall_at,
          cfg.recall_search_L, res_tags.data() + static_cast<size_t>(i) * cfg.recall_at,
          res_dists.data() + static_cast<size_t>(i) * cfg.recall_at, &stats);
    }
    unsigned *gt_eval = (gt_tags != nullptr) ? gt_tags : gt_ids;
    const double r = diskann::calculate_recall(
        static_cast<unsigned>(rq_num), gt_eval, gt_dists,
        static_cast<unsigned>(gt_dim), res_tags.data(),
        static_cast<unsigned>(cfg.recall_at), static_cast<unsigned>(cfg.recall_at));
    delete[] gt_ids;
    delete[] gt_dists;
    delete[] gt_tags;
    return r;
  };

  double foreground_time_s = 0.0;
  double maintenance_time_s = 0.0;
  double last_recall = -1.0;
  std::vector<double> call_latencies_us;
  call_latencies_us.reserve(required_ids);
  ann_bench::LogicalIoSnapshot logical_io_delta;
  ann_bench::RssSampler rss_sampler;
  for (size_t iter = 0; iter < cfg.n_iters; iter++) {
    const size_t batch_begin =
        iter * static_cast<size_t>(cfg.count_per_iter);
    std::cout << "ITER " << iter << ": "
              << (cfg.mode == WorkloadMode::INSERT ? "insert " : "delete ")
              << cfg.count_per_iter << " IDs" << std::endl;

    double round_fg_s = 0.0, round_maint_s = 0.0;
    std::vector<double> batch_latencies_us(cfg.count_per_iter, 0.0);
    ann_bench::LogicalIoCounter::reset_and_enable();
    rss_sampler.resume();

    if (cfg.mode == WorkloadMode::INSERT) {
      T *batch = load_insert_batch<T>(cfg.full_data_bin, ids, batch_begin,
                                      cfg.count_per_iter, full_npts, ndims,
                                      aligned_dim);
      std::atomic<uint32_t> failures(0);
      const auto foreground_begin = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads((int) cfg.nthreads)
      for (int64_t i = 0; i < static_cast<int64_t>(cfg.count_per_iter); i++) {
        const uint32_t tag = ids[batch_begin + static_cast<size_t>(i)];
        const auto op_begin = std::chrono::steady_clock::now();
        if (merge_insert.insert(batch + static_cast<size_t>(i) * aligned_dim,
                                tag) != 0) {
          failures.fetch_add(1);
        }
        const auto op_end = std::chrono::steady_clock::now();
        batch_latencies_us[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::micro>(op_end - op_begin).count();
      }
      const auto foreground_end = std::chrono::steady_clock::now();
      round_fg_s =
          std::chrono::duration<double>(foreground_end - foreground_begin).count();
      diskann::aligned_free(batch);
      if (failures.load() != 0) {
        throw std::runtime_error("failed to insert " +
                                 std::to_string(failures.load()) + " points");
      }
    } else {
      const auto foreground_begin = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads((int) cfg.nthreads)
      for (int64_t i = 0; i < static_cast<int64_t>(cfg.count_per_iter); i++) {
        const auto op_begin = std::chrono::steady_clock::now();
        merge_insert.lazy_delete(ids[batch_begin + static_cast<size_t>(i)]);
        const auto op_end = std::chrono::steady_clock::now();
        batch_latencies_us[static_cast<size_t>(i)] =
            std::chrono::duration<double, std::micro>(op_end - op_begin).count();
      }
      const auto foreground_end = std::chrono::steady_clock::now();
      round_fg_s =
          std::chrono::duration<double>(foreground_end - foreground_begin).count();
    }

    const auto maintenance_begin = std::chrono::steady_clock::now();
    merge_insert.final_merge(2);
    const auto maintenance_end = std::chrono::steady_clock::now();
    round_maint_s =
        std::chrono::duration<double>(maintenance_end - maintenance_begin).count();
    const ann_bench::RssStats round_rss =
        rss_sampler.stop_and_take_interval();
    const ann_bench::LogicalIoSnapshot round_io =
        ann_bench::LogicalIoCounter::snapshot_and_disable();

    // Roll the round into the workload-level aggregate.
    foreground_time_s += round_fg_s;
    maintenance_time_s += round_maint_s;
    ann_bench::add_logical_io(logical_io_delta, round_io);
    call_latencies_us.insert(call_latencies_us.end(),
                             batch_latencies_us.begin(),
                             batch_latencies_us.end());

    // Per-round recall on the live merged index (excluded from the timed/IO
    // window above, which is already closed).
    const double round_recall = measure_recall(iter);
    if (round_recall >= 0.0) last_recall = round_recall;

    // Emit one record per round (phase "round").
    const double round_time_s = round_fg_s + round_maint_s;
    const double round_ops_per_s = ann_bench::safe_div(
        static_cast<double>(cfg.count_per_iter), round_time_s);
    const ann_bench::LatencyStats round_lat =
        ann_bench::summarize_latencies(batch_latencies_us);
    bool rfirst = true;
    std::cout << std::fixed << std::setprecision(6) << "{";
    ann_bench::json_kv(std::cout, rfirst, "baseline", "Greator");
    ann_bench::json_kv(std::cout, rfirst, "workload",
                       cfg.mode == WorkloadMode::INSERT ? "insert" : "delete");
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
    if (cfg.recall_enabled) {
      ann_bench::json_kv(std::cout, rfirst, "recall", round_recall);
      ann_bench::json_kv(std::cout, rfirst, "recall_at",
                         static_cast<uint64_t>(cfg.recall_at));
      ann_bench::json_kv(std::cout, rfirst, "recall_search_L",
                         static_cast<uint64_t>(cfg.recall_search_L));
    }
    ann_bench::emit_logical_io(std::cout, rfirst, round_io, round_time_s,
                               static_cast<uint64_t>(cfg.count_per_iter));
    std::cout << "}" << std::endl;
  }
  const size_t total_ops = required_ids;
  const double workload_time_s = foreground_time_s + maintenance_time_s;
  const double ops_per_sec =
      ann_bench::safe_div(static_cast<double>(total_ops), workload_time_s);
  const ann_bench::LatencyStats call_lat =
      ann_bench::summarize_latencies(call_latencies_us);

  std::cout << std::fixed << std::setprecision(6);
  if (cfg.mode == WorkloadMode::INSERT) {
    std::cout << "total_inserts=" << total_ops << std::endl;
    std::cout << "insert_update_merge_time_sec=" << workload_time_s
              << std::endl;
    std::cout << "inserts_per_sec=" << ops_per_sec << std::endl;
  } else {
    std::cout << "total_deletes=" << total_ops << std::endl;
    std::cout << "delete_update_merge_time_sec=" << workload_time_s
              << std::endl;
    std::cout << "deletes_per_sec=" << ops_per_sec << std::endl;
  }
  bool first = true;
  std::cout << "{";
  ann_bench::json_kv(std::cout, first, "baseline", "Greator");
  ann_bench::json_kv(std::cout, first, "workload",
                     cfg.mode == WorkloadMode::INSERT ? "insert" : "delete");
  ann_bench::json_kv(std::cout, first, "phase", "workload");
  if (cfg.mode == WorkloadMode::INSERT) {
    ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(total_ops));
    ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(0));
    ann_bench::json_kv(std::cout, first, "insert_ops_per_s", ops_per_sec);
    ann_bench::json_kv(std::cout, first, "delete_ops_per_s", 0.0);
    ann_bench::emit_latency(std::cout, first, "insert_call", call_lat);
  } else {
    ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(0));
    ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(total_ops));
    ann_bench::json_kv(std::cout, first, "insert_ops_per_s", 0.0);
    ann_bench::json_kv(std::cout, first, "delete_ops_per_s", ops_per_sec);
    ann_bench::emit_latency(std::cout, first, "delete_call", call_lat);
  }
  ann_bench::json_kv(std::cout, first, "update_ops_per_s", ops_per_sec);
  ann_bench::json_kv(std::cout, first, "overall_ops_per_s", ops_per_sec);
  ann_bench::json_kv(std::cout, first, "foreground_time_s", foreground_time_s);
  ann_bench::json_kv(std::cout, first, "maintenance_time_s", maintenance_time_s);
  ann_bench::json_kv(std::cout, first, "workload_time_s", workload_time_s);
  ann_bench::json_kv(std::cout, first, "avg_rss_mb", rss_sampler.avg_mb());
  ann_bench::json_kv(std::cout, first, "peak_rss_mb", rss_sampler.peak_mb());
  if (cfg.recall_enabled) {
    ann_bench::json_kv(std::cout, first, "recall", last_recall);
    ann_bench::json_kv(std::cout, first, "recall_at",
                       static_cast<uint64_t>(cfg.recall_at));
    ann_bench::json_kv(std::cout, first, "recall_search_L",
                       static_cast<uint64_t>(cfg.recall_search_L));
  }
  ann_bench::emit_logical_io(std::cout, first, logical_io_delta,
                             workload_time_s, static_cast<uint64_t>(total_ops));
  std::cout << "}" << std::endl;

  if (rq_query != nullptr) diskann::aligned_free(rq_query);
}

}  // namespace

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
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
