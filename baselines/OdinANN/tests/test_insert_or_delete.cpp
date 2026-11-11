#include "dynamic_index.h"

#include <omp.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gperftools/malloc_extension.h>

#include "distance.h"
#include "utils.h"
#include "../../src/ann_bench_metrics.h"

namespace {

enum class WorkloadMode { INSERT, DELETE, MERGE };

struct Config {
  WorkloadMode mode;
  std::string type;
  std::string full_data_bin;
  unsigned l_disk;
  std::string base_prefix;
  std::string merge_prefix;
  size_t n_iters;
  uint32_t count_per_iter;
  uint32_t range;
  uint32_t beamwidth;
  uint32_t recall_beamwidth;
  uint32_t nthreads;
  std::string ids_file;
  std::string query_bin;
  std::string gt_dir;  // dir of per-round GT: round_<k>_gt<recall_at>.bin
  uint32_t query_threads;
  uint32_t recall_at;
  uint32_t search_l;
  bool skip_merge;
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
  if (mode == "merge") {
    return WorkloadMode::MERGE;
  }
  throw std::invalid_argument("mode must be insert, delete, or merge");
}

Config parse_args(int argc, char **argv) {
  if (argc != 18) {
    std::cerr << "Correct usage: " << argv[0]
              << " <mode[insert/delete/merge]> <type[int8/uint8/float]> "
                 "<full_data_bin>"
              << " <L_disk> <base_prefix> <merge_prefix>"
              << " <n_iters> <count_per_iter> <range> <beamwidth> <nthreads>"
              << " <ids_file>"
              << " <query_bin[\"null\" to skip]> <gt_dir> <query_threads>"
              << " <recall_at> <search_l>" << std::endl;
    std::cerr << "  merge mode: n_iters/count_per_iter/ids_file/query ignored "
                 "(use 1 / 1 / null / null); final_merge(nthreads) + RSS only."
              << std::endl;
    throw std::invalid_argument("wrong number of arguments");
  }

  int arg_no = 1;
  Config cfg;
  cfg.mode = parse_mode(argv[arg_no++]);
  cfg.type = argv[arg_no++];
  cfg.full_data_bin = argv[arg_no++];
  cfg.l_disk = parse_u32(argv[arg_no++], "L_disk");
  cfg.base_prefix = argv[arg_no++];
  cfg.merge_prefix = argv[arg_no++];
  cfg.n_iters = static_cast<size_t>(parse_u64(argv[arg_no++], "n_iters"));
  cfg.count_per_iter = parse_u32(argv[arg_no++], "count_per_iter");
  cfg.range = parse_u32(argv[arg_no++], "range");
  cfg.beamwidth = parse_u32(argv[arg_no++], "beamwidth");
  // RECALL_BEAMWIDTH, if set, overrides the beam width of the recall and
  // final-query searches only; inserts and deletes keep cfg.beamwidth.
  cfg.recall_beamwidth = std::getenv("RECALL_BEAMWIDTH") != nullptr
                             ? parse_u32(std::getenv("RECALL_BEAMWIDTH"), "RECALL_BEAMWIDTH")
                             : cfg.beamwidth;
  cfg.nthreads = parse_u32(argv[arg_no++], "nthreads");
  cfg.ids_file = argv[arg_no++];
  cfg.query_bin = argv[arg_no++];
  cfg.gt_dir = argv[arg_no++];
  cfg.query_threads = parse_u32(argv[arg_no++], "query_threads");
  cfg.recall_at = parse_u32(argv[arg_no++], "recall_at");
  cfg.search_l = parse_u32(argv[arg_no++], "search_l");
  if (cfg.mode != WorkloadMode::MERGE && cfg.n_iters == 0) {
    throw std::invalid_argument("n_iters must be positive");
  }
  cfg.skip_merge = (cfg.mode == WorkloadMode::INSERT);
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

  pipeann::IndexBuildParameters paras;
  paras.set(0, cfg.l_disk, 384, 1.2f, cfg.nthreads, true, cfg.beamwidth);

  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, uint32_t> sync_index(paras, cfg.base_prefix, cfg.merge_prefix, dist_cmp, metric,
                                                   BEAM_SEARCH, false, "", false);

  omp_set_dynamic(0);

  // merge: final_merge only; wall + RSS (no insert/delete/query rounds).
  if (cfg.mode == WorkloadMode::MERGE) {
    std::cout << "merge_only: nthreads=" << cfg.nthreads
              << " base=" << cfg.base_prefix
              << " out=" << cfg.merge_prefix << std::endl;

    ann_bench::RssSampler rss_sampler;
    ann_bench::LogicalIoCounter::reset_and_enable();
    rss_sampler.start();
    const auto t0 = std::chrono::steady_clock::now();
    sync_index.final_merge(cfg.nthreads);
    const auto t1 = std::chrono::steady_clock::now();
    const ann_bench::RssStats rss = rss_sampler.stop_and_take_interval();
    const double merge_s = std::chrono::duration<double>(t1 - t0).count();
    const ann_bench::LogicalIoSnapshot io =
        ann_bench::LogicalIoCounter::snapshot_and_disable();
    std::cout << std::fixed << std::setprecision(4)
              << "merge_only wall_s=" << merge_s
              << " peak_rss_mb=" << rss.peak_mb
              << " avg_rss_mb=" << rss.avg_mb << std::endl;
    bool first = true;
    std::cout << "{";
    ann_bench::json_kv(std::cout, first, "baseline", "PipeANN");
    ann_bench::json_kv(std::cout, first, "workload", "merge_only");
    ann_bench::json_kv(std::cout, first, "phase", "merge");
    ann_bench::json_kv(std::cout, first, "nthreads",
                       static_cast<uint64_t>(cfg.nthreads));
    ann_bench::json_kv(std::cout, first, "wall_s", merge_s);
    ann_bench::emit_rss_stats(std::cout, first, rss);
    ann_bench::emit_logical_io(std::cout, first, io, merge_s, 0);
    std::cout << "}" << std::endl;
    return;
  }

  const size_t required_ids = cfg.n_iters * static_cast<size_t>(cfg.count_per_iter);
  std::vector<uint32_t> ids = load_ids(cfg.ids_file, required_ids);

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
                        cfg.search_l, cfg.recall_beamwidth,
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

  double foreground_time_s = 0.0;
  double maintenance_time_s = 0.0;
  double last_recall = -1.0;
  std::vector<double> call_latencies_us;
  call_latencies_us.reserve(required_ids);
  ann_bench::LogicalIoSnapshot logical_io_delta;
  ann_bench::RssSampler rss_sampler;
  for (size_t iter = 0; iter < cfg.n_iters; iter++) {
    const size_t batch_begin = iter * static_cast<size_t>(cfg.count_per_iter);
    std::cout << "ITER " << iter << ": "
              << (cfg.mode == WorkloadMode::INSERT ? "insert " : "delete ") << cfg.count_per_iter << " IDs"
              << std::endl;

    double round_fg_s = 0.0, round_maint_s = 0.0;
    std::vector<double> batch_latencies_us(cfg.count_per_iter, 0.0);
    ann_bench::LogicalIoCounter::reset_and_enable();
    rss_sampler.resume();

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
      if (!cfg.skip_merge) {
        const auto maintenance_begin = std::chrono::steady_clock::now();
        sync_index.final_merge(cfg.nthreads);
        const auto maintenance_end = std::chrono::steady_clock::now();
        round_maint_s =
            std::chrono::duration<double>(maintenance_end - maintenance_begin).count();
      }
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
      sync_index.final_merge(cfg.nthreads);
      const auto maintenance_end = std::chrono::steady_clock::now();
      round_maint_s =
          std::chrono::duration<double>(maintenance_end - maintenance_begin).count();
    }

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

    // Per-round recall on the live index (excluded from the timed/IO window).
    // RECALL_DELAY_S (debug) waits before measuring, e.g. to let background
    // write-back drain.
    if (const char *d = std::getenv("RECALL_DELAY_S")) {
      std::this_thread::sleep_for(std::chrono::seconds(std::atoi(d)));
    }
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
    ann_bench::json_kv(std::cout, rfirst, "baseline", "PipeANN");
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
    if (rq_have_queries) {
      ann_bench::json_kv(std::cout, rfirst, "recall", round_recall);
      ann_bench::json_kv(std::cout, rfirst, "recall_at",
                         static_cast<uint64_t>(cfg.recall_at));
      ann_bench::json_kv(std::cout, rfirst, "recall_search_L",
                         static_cast<uint64_t>(cfg.search_l));
    }
    ann_bench::emit_logical_io(std::cout, rfirst, round_io, round_time_s,
                               static_cast<uint64_t>(cfg.count_per_iter));
    std::cout << "}" << std::endl;
  }

  const double workload_time_s = foreground_time_s + maintenance_time_s;
  const double ops_per_sec =
      ann_bench::safe_div(static_cast<double>(required_ids), workload_time_s);
  const ann_bench::LatencyStats call_lat =
      ann_bench::summarize_latencies(call_latencies_us);

  std::cout << std::fixed << std::setprecision(6);
  if (cfg.mode == WorkloadMode::INSERT) {
    std::cout << "total_inserts=" << required_ids << std::endl;
    std::cout << "insert_update_merge_time_sec=" << workload_time_s << std::endl;
    std::cout << "inserts_per_sec=" << ops_per_sec << std::endl;
  } else {
    std::cout << "total_deletes=" << required_ids << std::endl;
    std::cout << "delete_update_merge_time_sec=" << workload_time_s << std::endl;
    std::cout << "deletes_per_sec=" << ops_per_sec << std::endl;
  }
  bool first = true;
  std::cout << "{";
  ann_bench::json_kv(std::cout, first, "baseline", "PipeANN");
  ann_bench::json_kv(std::cout, first, "workload",
                      cfg.mode == WorkloadMode::INSERT ? "insert" : "delete");
  ann_bench::json_kv(std::cout, first, "phase", "workload");
  if (cfg.mode == WorkloadMode::INSERT) {
    ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(required_ids));
    ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(0));
    ann_bench::json_kv(std::cout, first, "insert_ops_per_s", ops_per_sec);
    ann_bench::json_kv(std::cout, first, "delete_ops_per_s", 0.0);
    ann_bench::emit_latency(std::cout, first, "insert_call", call_lat);
  } else {
    ann_bench::json_kv(std::cout, first, "insert_count", static_cast<uint64_t>(0));
    ann_bench::json_kv(std::cout, first, "delete_count", static_cast<uint64_t>(required_ids));
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
  if (rq_have_queries) {
    ann_bench::json_kv(std::cout, first, "recall", last_recall);
    ann_bench::json_kv(std::cout, first, "recall_at",
                       static_cast<uint64_t>(cfg.recall_at));
    ann_bench::json_kv(std::cout, first, "recall_search_L",
                       static_cast<uint64_t>(cfg.search_l));
  }
  ann_bench::emit_logical_io(std::cout, first, logical_io_delta,
                             workload_time_s,
                             static_cast<uint64_t>(required_ids));
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
      sync_index.search(query + i * query_dim, cfg.recall_at, 0, cfg.search_l, cfg.recall_beamwidth,
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
    std::cout << std::setw(6) << cfg.search_l << std::setw(12) << cfg.recall_beamwidth << std::setw(12)
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
    ann_bench::json_kv(std::cout, qfirst, "baseline", "PipeANN");
    ann_bench::json_kv(std::cout, qfirst, "workload",
                       cfg.mode == WorkloadMode::INSERT ? "insert" : "delete");
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
