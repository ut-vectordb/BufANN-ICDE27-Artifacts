#include <chrono>
#include <cstring>
#include <omp.h>
#include <ssd_index.h>
#include <dynamic_index.h>
#include <string.h>
#include <time.h>
#include <iomanip>

#include "utils/log.h"
#include "nbr/nbr.h"
#include "utils/timer.h"
#include "utils.h"
#include "../../src/ann_bench_metrics.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"

void print_rss_checkpoint(const std::string &label) {
  std::cout << "RSS_CHECKPOINT " << label
            << " rss_mb=" << ann_bench::current_rss_mb() << std::endl;
}

template<typename T>
int search_disk_index(int argc, char **argv) {
  // load query bin
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;
  std::vector<uint64_t> Lvec;

  bool tags_flag = true;

  int index = 2;
  std::string index_prefix_path(argv[index++]);
  uint32_t num_threads = std::atoi(argv[index++]);
  uint32_t beamwidth = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string truthset_bin(argv[index++]);
  uint64_t recall_at = std::atoi(argv[index++]);
  std::string dist_metric(argv[index++]);
  std::string nbr_type = argv[index++];
  int search_mode = std::atoi(argv[index++]);
  bool use_page_search = search_mode != 0;
  uint32_t mem_L = std::atoi(argv[index++]);
  const char *update_index_env = std::getenv("QUERY_USE_UPDATE_INDEX");
  const bool use_update_index_query =
      update_index_env != nullptr && std::atoi(update_index_env) != 0;
  std::cout << "Query path: "
            << (use_update_index_query
                    ? "update-capable DynamicSSDIndex"
                    : "legacy SSDIndex")
            << std::endl;

  pipeann::Metric m = pipeann::get_metric(dist_metric);

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  for (int ctr = index; ctr < argc; ctr++) {
    uint64_t curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at" << std::endl;
    return -1;
  }

  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    std::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    std::cout << " beamwidth: " << beamwidth << std::endl;

  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);

  // Neighbor IDs returned by search are disk tags. Type-3 truthsets store internal
  // row ids plus a tag matrix (see DiskANN compute_groundtruth); recall must use
  // tags when present, same as baselines/DiskANN/tests/search_disk_index.cpp.
  unsigned *gt_eval_ids = nullptr;
  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
    }
    calc_recall_flag = true;
    gt_eval_ids = (tags != nullptr) ? reinterpret_cast<unsigned *>(tags) : gt_ids;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
  pipeann::AbstractNeighbor<T> *nbr_handler = nullptr;
  std::unique_ptr<pipeann::SSDIndex<T>> _pFlashIndex;
  std::unique_ptr<pipeann::DynamicSSDIndex<T, uint32_t>> dynamic_index;
  std::unique_ptr<pipeann::Distance<T>> dist_cmp;

  if (use_update_index_query) {
    if (search_mode == SearchMode::CORO_SEARCH) {
      std::cout << "DynamicSSDIndex does not support coro search mode." << std::endl;
      return -1;
    }
    if (nbr_type != "pq") {
      std::cout << "DynamicSSDIndex query path requires pq neighbor type." << std::endl;
      return -1;
    }
    pipeann::IndexBuildParameters paras;
    paras.set(0, static_cast<uint32_t>(Lvec.back()), 384, 1.2f,
              num_threads, true, beamwidth);
    dist_cmp.reset(pipeann::get_distance_function<T>(m));
    dynamic_index.reset(new pipeann::DynamicSSDIndex<T, uint32_t>(
        paras, index_prefix_path, index_prefix_path + "_query_merge",
        dist_cmp.get(), m, search_mode, mem_L != 0, index_prefix_path, false));
    print_rss_checkpoint("after_index_load");
  } else {
    reader.reset(new LinuxAlignedFileReader());
    nbr_handler = pipeann::get_nbr_handler<T>(m, nbr_type);
    _pFlashIndex.reset(new pipeann::SSDIndex<T>(m, reader, nbr_handler, tags_flag));

    int res = _pFlashIndex->load(index_prefix_path.c_str(), num_threads, use_page_search);
    if (res != 0) {
      return res;
    }
    print_rss_checkpoint("after_index_load");

    if (mem_L != 0) {
      auto mem_index_path = index_prefix_path + "_mem.index";
      LOG(INFO) << "Load memory index from " << mem_index_path;
      _pFlashIndex->load_mem_index(mem_index_path);
      print_rss_checkpoint("after_load_mem_index");
    }
  }

  omp_set_num_threads(num_threads);

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) {
    std::vector<pipeann::QueryStats> stats(query_num);
    uint64_t L = Lvec[test_id];

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    ann_bench::RssSampler rss_sampler;
    ann_bench::LogicalIoCounter::reset_and_enable();
    rss_sampler.start();
    auto s = std::chrono::high_resolution_clock::now();

    if (use_update_index_query) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        dynamic_index->search(query + (i * query_dim), (uint64_t) recall_at,
                              mem_L, (uint64_t) L, (uint64_t) beamwidth,
                              query_result_tags_32.data() + (i * recall_at),
                              query_result_dists[test_id].data() + (i * recall_at),
                              stats.data() + i, false);
      }
    } else if (search_mode == SearchMode::PIPE_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->pipe_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                               query_result_tags_32.data() + (i * recall_at),
                               query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                               stats.data() + i);
      }
    } else if (search_mode == SearchMode::PAGE_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->page_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                               query_result_tags_32.data() + (i * recall_at),
                               query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                               stats.data() + i);
      }
    } else if (search_mode == SearchMode::CORO_SEARCH) {
      constexpr uint64_t kBatchSize = 8;
      T *q[kBatchSize];
      uint32_t *res_tags[kBatchSize];
      float *res_dists[kBatchSize];
      int N;
#pragma omp parallel for schedule(dynamic, 1) private(q, res_tags, res_dists, N)
      for (int64_t i = 0; i < (int64_t) query_num; i += kBatchSize) {
        N = std::min(kBatchSize, query_num - i);
        for (int v = 0; v < N; ++v) {
          q[v] = query + ((i + v) * query_dim);
          res_tags[v] = query_result_tags_32.data() + ((i + v) * recall_at);
          res_dists[v] = query_result_dists[test_id].data() + ((i + v) * recall_at);
        }

        _pFlashIndex->coro_search(q, (uint64_t) recall_at, mem_L, (uint64_t) L, res_tags, res_dists,
                               (uint64_t) beamwidth, N);
      }
    } else if (search_mode == SearchMode::BEAM_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->beam_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                               query_result_tags_32.data() + (i * recall_at),
                               query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                               stats.data() + i,
                               nullptr, false);
      }
    } else {
      std::cout << "Unknown search mode: " << search_mode << std::endl;
      exit(-1);
    }

    auto e = std::chrono::high_resolution_clock::now();
    rss_sampler.stop();
    const auto io_delta = ann_bench::LogicalIoCounter::snapshot_and_disable();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) ((1.0 * (double) query_num) / (1.0 * (double) diff.count()));

    pipeann::convert_types<uint32_t, uint32_t>(query_result_tags_32.data(), query_result_tags[test_id].data(),
                                               (size_t) query_num, (size_t) recall_at);

    float mean_latency = (float) pipeann::get_mean_stats(
        stats.data(), query_num, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float latency_50 = (float) pipeann::get_percentile_stats(
        stats.data(), query_num, 0.500f, [](const pipeann::QueryStats &stats) { return stats.total_us; });
    float latency_90 = (float) pipeann::get_percentile_stats(
        stats.data(), query_num, 0.900f, [](const pipeann::QueryStats &stats) { return stats.total_us; });
    float latency_95 = (float) pipeann::get_percentile_stats(
        stats.data(), query_num, 0.950f, [](const pipeann::QueryStats &stats) { return stats.total_us; });
    float latency_99 = (float) pipeann::get_percentile_stats(
        stats.data(), query_num, 0.990f, [](const pipeann::QueryStats &stats) { return stats.total_us; });
    float latency_999 = (float) pipeann::get_percentile_stats(
        stats.data(), query_num, 0.999f, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float mean_hops = (float) pipeann::get_mean_stats(stats.data(), query_num,
                                                      [](const pipeann::QueryStats &stats) { return stats.n_hops; });

    float mean_ios =
        (float) pipeann::get_mean_stats(stats.data(), query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

    // Per-query blocking IO wait (us), to match BufANN's query_io_* series.
    // NOTE: only BEAM_SEARCH/PAGE_SEARCH populate io_us; PIPE_SEARCH overlaps IO
    // with compute and leaves io_us=0 (its timer is commented out).
    float mean_io_us =
        (float) pipeann::get_mean_stats(stats.data(), query_num, [](const pipeann::QueryStats &stats) { return stats.io_us; });

    if (output) {
      float recall = 0;
      if (calc_recall_flag) {
        /* Attention: in SPACEV, there may be multiple vectors with the same distance,
          which may cause lower than expected recall@1 (?) */
        recall = (float) pipeann::calculate_recall((uint32_t) query_num, gt_eval_ids, gt_dists, (uint32_t) gt_dim,
                                                   query_result_tags[test_id].data(), (uint32_t) recall_at,
                                                   (uint32_t) recall_at);
      }

      std::cout << std::setw(6) << L << std::setw(12) << beamwidth << std::setw(12) << qps << std::setw(12)
                << mean_latency << std::setw(12) << latency_50 << std::setw(12) << latency_90 << std::setw(12)
                << latency_95 << std::setw(12) << latency_99 << std::setw(12) << latency_999 << std::setw(12)
                << mean_hops << std::setw(12) << mean_ios;
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall << std::endl;
      }
      bool first = true;
      const char *phase_env = std::getenv("ANN_BENCH_PHASE");
      std::cout << "{";
      ann_bench::json_kv(std::cout, first, "baseline", "PipeANN");
      ann_bench::json_kv(std::cout, first, "workload", "query");
      ann_bench::json_kv(std::cout, first, "phase",
                         phase_env != nullptr && *phase_env != '\0' ? phase_env : "query");
      ann_bench::json_kv(std::cout, first, "search_L", static_cast<uint64_t>(L));
      ann_bench::json_kv(std::cout, first, "query_count", static_cast<uint64_t>(query_num));
      ann_bench::json_kv(std::cout, first, "query_time_s", diff.count());
      ann_bench::json_kv(std::cout, first, "query_qps", static_cast<double>(qps));
      ann_bench::json_kv(std::cout, first, "query_lat_avg_us", static_cast<double>(mean_latency));
      // IO wait time on the query path. mean_io_us is per-query avg IO wait (us),
      // io_ns_total converts us->ns to match BufANN; fraction is unit-free.
      ann_bench::json_kv(std::cout, first, "query_io_ns_total",
                         static_cast<uint64_t>((double) mean_io_us * (double) query_num * 1000.0));
      ann_bench::json_kv(std::cout, first, "query_io_avg_us_per_query", static_cast<double>(mean_io_us));
      ann_bench::json_kv(std::cout, first, "query_io_fraction_of_latency",
                         mean_latency > 0 ? static_cast<double>(mean_io_us) / static_cast<double>(mean_latency) : 0.0);
      ann_bench::json_kv(std::cout, first, "query_lat_p50_us", static_cast<double>(latency_50));
      ann_bench::json_kv(std::cout, first, "query_lat_p90_us", static_cast<double>(latency_90));
      ann_bench::json_kv(std::cout, first, "query_lat_p95_us", static_cast<double>(latency_95));
      ann_bench::json_kv(std::cout, first, "query_lat_p99_us", static_cast<double>(latency_99));
      ann_bench::json_kv(std::cout, first, "recall", static_cast<double>(recall));
      ann_bench::json_kv(std::cout, first, "avg_rss_mb", rss_sampler.avg_mb());
      ann_bench::json_kv(std::cout, first, "peak_rss_mb", rss_sampler.peak_mb());
      ann_bench::emit_logical_io(std::cout, first, io_delta, diff.count(),
                                 static_cast<uint64_t>(query_num));
      std::cout << "}" << std::endl;
    }
  };

  // LOG(INFO) << "Use two ANNS for warming up...";
  // uint32_t prev_L = Lvec[0];
  // Lvec[0] = 200;
  // run_tests(0, false);
  // run_tests(0, false);
  // Lvec[0] = prev_L;
  // LOG(INFO) << "Warming up finished.";

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
            << "AvgLat(us)" << std::setw(12) << "P50(us)" << std::setw(12) << "P90(us)" << std::setw(12)
            << "P95(us)" << std::setw(12) << "P99(us)" << std::setw(12) << "P99.9(us)" << std::setw(12)
            << "Mean Hops" << std::setw(12) << "Mean IOs" << std::setw(12);
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "=============================================="
               "==========================================="
            << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    run_tests(test_id, true);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 12) {
    // tags == 1!
    std::cout << "Usage: " << argv[0]
              << " <index_type (float/int8/uint8)>  <index_prefix_path>"
                 " <num_threads>  <pipeline width> "
                 " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
                 " <K> <similarity (cosine/l2/mips)> <nbr_type (pq/rabitq)>"
                 " <search_mode(0 for beam search / 1 for page search / 2 for pipe search)> <mem_L (0 means not "
                 "using mem index)> <L1> [L2] etc."
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8" << std::endl;
}
