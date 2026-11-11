#include <sstream>
#include <cstring>
#include <omp.h>
#include <ssd_index.h>
#include <string.h>
#include <time.h>
#include <iostream>

#include "log.h"
#include "timer.h"
#include "utils.h"
#include "aux_utils.h"
#include "global_stats.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#include "ann_bench_metrics.h"

#define WARMUP false

int begin_time = 0;
pipeann::Timer globalTimer;

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results) {
  std::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(8) << percentiles[s] << "%";
  }
  std::cout << std::endl;
  std::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(9) << results[s];
  }
  std::cout << std::endl;
}

void ShowPeakMemoryStatus() {

  unsigned long peak_memory_kb = 0;
  std::ifstream status_file("/proc/self/status");
  std::string line;
  
  while (std::getline(status_file, line)) {
      if (line.find("VmHWM:") != std::string::npos) {
          sscanf(line.c_str(), "VmHWM: %lu kB", &peak_memory_kb);
          break;
      }
  }

  // 输出峰值内存和当前时间
  std::cout << "Peak Memory: " << peak_memory_kb << " KB" 
            << std::endl;
}

template<typename T>
int search_disk_index(int argc, char **argv) {
  // load query bin
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  bool tags_flag = true;

  int index = 2;
  std::string index_prefix_path(argv[index++]);
  _u32 num_threads = std::atoi(argv[index++]);
  _u32 beamwidth = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string truthset_bin(argv[index++]);
  _u64 recall_at = std::atoi(argv[index++]);
  std::string dist_metric(argv[index++]);
  int search_mode = std::atoi(argv[index++]);
  bool use_page_search = search_mode != 0;
  _u32 mem_L = std::atoi(argv[index++]);
  int strategy = std::atoi(argv[index++]);

  pipeann::Metric m = dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
  if (dist_metric != "l2" && m == pipeann::Metric::L2) {
    std::cout << "Unknown distance metric: " << dist_metric << ". Using default(L2) instead." << std::endl;
  }

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  for (int ctr = index; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
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
  // std::load_aligned_bin<T>(query_bin, query, query_num, query_dim, query_aligned_dim);

  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());
  ((LinuxAlignedFileReader *)reader.get())->strategy = strategy;

  LOG(INFO) << "START TO LOAD INDEX";
  std::unique_ptr<pipeann::SSDIndex<T>> _pFlashIndex(
      new pipeann::SSDIndex<T>(m, reader, SearchMode(search_mode), tags_flag));

  int res = _pFlashIndex->load(index_prefix_path.c_str(), num_threads, true, use_page_search);
  LOG(INFO) << "INDEX LOADED";
  if (res != 0) {
    return res;
  }

  if (mem_L != 0) {
    auto mem_index_path = index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index " << mem_index_path << " " << query_dim;
    _pFlashIndex->load_mem_index(m, query_dim, mem_index_path);
  }

  omp_set_num_threads(num_threads);

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) {
    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
    _u64 L = Lvec[test_id];

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    // Harness metrics for the JSON record below: RSS and logical I/O over the timed window only.
    ann_bench::RssSampler rss_sampler;
    ann_bench::LogicalIoCounter::reset_and_enable();
    rss_sampler.start();
    auto s = std::chrono::high_resolution_clock::now();
    // query_num = 10;
    if (search_mode == SearchMode::PIPE_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (_s64 i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->pipe_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i, nullptr);
      }
    } else if (search_mode == SearchMode::PAGE_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (_s64 i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->page_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i);
      }
    } else if (search_mode == SearchMode::CORO_SEARCH) {
      constexpr uint64_t kBatchSize = 8;
      T *q[kBatchSize];
      uint32_t *res_tags[kBatchSize];
      float *res_dists[kBatchSize];
      int N;
#pragma omp parallel for schedule(dynamic, 1) private(q, res_tags, res_dists, N)
      for (_s64 i = 0; i < (int64_t) query_num; i += kBatchSize) {
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
      for (_s64 i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->beam_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth, stats + i,
                                  nullptr, false);
      }
    } else if(search_mode == SearchMode::RERANK_SEARCH) {
#pragma omp parallel for schedule(dynamic, 1)
      for (_s64 i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->rerank_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i, nullptr, output ? gt_ids + (i * recall_at) : nullptr);
      }
    }
    else{
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
        stats, query_num, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    // The table's "P99 Lat" column has always printed the 99.9th percentile; the JSON record carries p50-p99.
    float latency_999 = (float) pipeann::get_percentile_stats(
        stats, query_num, 0.999f, [](const pipeann::QueryStats &stats) { return stats.total_us; });
    auto latency_pct = [&](float q) {
      return (float) pipeann::get_percentile_stats(stats, query_num, q,
                                                   [](const pipeann::QueryStats &stats) { return stats.total_us; });
    };
    float latency_50 = latency_pct(0.50f), latency_90 = latency_pct(0.90f);
    float latency_95 = latency_pct(0.95f), latency_99 = latency_pct(0.99f);

    float mean_hops = (float) pipeann::get_mean_stats(stats, query_num,
                                                      [](const pipeann::QueryStats &stats) { return stats.n_hops; });

    float mean_ios =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

        
    float rerank_ios =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.rerank_ios; });
    
    float rerank_us =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.rerank_us; });

    float io_us =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.io_us; });

        float t2 =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.io_us1; });
        float t3 =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.io_us2; });
        float t4 =
            (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.pq_compute_us; });
        float t5 =
            (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.accuracy_compute_us; });
        float t6 =
            (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.populate_chunk_distances_us; });
        float t7 =
            (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.rerank_us_2; });
        //     float t8 =
        //         (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.buffer_us; });
        //         float t9 =
        //             (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.buffer_us1; });
        //             float t10 =
        //                 (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.buffer_us2; });
        //                 float t11 =
        //                     (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.buffer_us3; });
        //                     float t12 =
        //                         (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.buffer_us4; });
        //                     float t13 =
        //                         (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.bk_us1; });
        //                     float t14 =
        //                         (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.bk_us2; });
        //                     float t15 =
        //                         (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.bk_us3; });
        //                         float t16 =
        //                             (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.bk_us4; });
        //                             float t17 =
        //                                 (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.bk_us5; });
        //                                 float t18 =
        //                                     (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return 1.0 * stats.hit_count / (stats.hit_count + stats.miss_count); });


    delete[] stats;

    if (output) {
      float recall = 0;
      if (calc_recall_flag) {
        /* Attention: in SPACEV, there may be multiple vectors with the same distance,
          which may cause lower than expected recall@1 (?) */
        recall =
            (float) pipeann::calculate_recall((_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim,
                                              query_result_tags[test_id].data(), (_u32) recall_at, (_u32) recall_at);
      }

      std::cout << std::setw(6) << L << std::setw(12) << beamwidth << std::setw(12) << qps << std::setw(12)
                << mean_latency << std::setw(12) << latency_999 << std::setw(12) << mean_hops << std::setw(12)
                << mean_ios << std::setw(12) << rerank_ios 
                // << std::setw(16) << rerank_us;
                << std::setw(16) << io_us
                // //  << std::setw(16) << t << std::setw(16) << t2 
                // // << std::setw(16) << t3
                // << std::setw(16) << t4
                // << std::setw(16) << t5
                // << std::setw(16) << t6
                // << std::setw(16) << t7
                // << std::setw(16) << t2
                // << std::setw(16) << t3
                // << std::setw(16) << t8
                // << std::setw(16) << t9
                // // << std::setw(16) << t10
                // // << std::setw(16) << t11
                // // << std::setw(16) << t12
                // << std::setw(16) << t13
                // << std::setw(16) << t14
                // << std::setw(16) << t15
                // << std::setw(16) << t16
                // << std::setw(16) << t17
                // << std::setw(16) << t18
                ;
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall << std::endl;
      }
      // if(recall >= 99) exit(0);

      // One ann_bench record per L, the same keys as the PipeANN harness driver, plus DGAI's trunc_len and
      // mean rerank I/Os. write_ann_bench_metrics_json collects these lines into result.json.
      // json_kv changes the stream's number formatting; restore it so the next table row keeps its columns.
      const std::ios_base::fmtflags saved_flags = std::cout.flags();
      const std::streamsize saved_precision = std::cout.precision();
      bool first = true;
      const char *phase_env = std::getenv("ANN_BENCH_PHASE");
      std::cout << "{";
      ann_bench::json_kv(std::cout, first, "baseline", "DGAI");
      ann_bench::json_kv(std::cout, first, "workload", "query");
      ann_bench::json_kv(std::cout, first, "phase", phase_env != nullptr && *phase_env != '\0' ? phase_env : "query");
      ann_bench::json_kv(std::cout, first, "search_L", static_cast<uint64_t>(L));
      ann_bench::json_kv(std::cout, first, "beamwidth", static_cast<uint64_t>(beamwidth));
      ann_bench::json_kv(std::cout, first, "trunc_len",
                         static_cast<uint64_t>(((LinuxAlignedFileReader *) reader.get())->trunc_len));
      ann_bench::json_kv(std::cout, first, "query_count", static_cast<uint64_t>(query_num));
      ann_bench::json_kv(std::cout, first, "query_time_s", diff.count());
      ann_bench::json_kv(std::cout, first, "query_qps", static_cast<double>(qps));
      ann_bench::json_kv(std::cout, first, "query_lat_avg_us", static_cast<double>(mean_latency));
      ann_bench::json_kv(std::cout, first, "query_io_ns_total",
                         static_cast<uint64_t>((double) io_us * (double) query_num * 1000.0));
      ann_bench::json_kv(std::cout, first, "query_io_avg_us_per_query", static_cast<double>(io_us));
      ann_bench::json_kv(std::cout, first, "query_io_fraction_of_latency",
                         mean_latency > 0 ? static_cast<double>(io_us) / static_cast<double>(mean_latency) : 0.0);
      ann_bench::json_kv(std::cout, first, "query_lat_p50_us", static_cast<double>(latency_50));
      ann_bench::json_kv(std::cout, first, "query_lat_p90_us", static_cast<double>(latency_90));
      ann_bench::json_kv(std::cout, first, "query_lat_p95_us", static_cast<double>(latency_95));
      ann_bench::json_kv(std::cout, first, "query_lat_p99_us", static_cast<double>(latency_99));
      ann_bench::json_kv(std::cout, first, "query_lat_p999_us", static_cast<double>(latency_999));
      ann_bench::json_kv(std::cout, first, "query_mean_ios", static_cast<double>(mean_ios));
      ann_bench::json_kv(std::cout, first, "query_mean_rerank_ios", static_cast<double>(rerank_ios));
      ann_bench::json_kv(std::cout, first, "recall", static_cast<double>(recall));
      ann_bench::json_kv(std::cout, first, "avg_rss_mb", rss_sampler.avg_mb());
      ann_bench::json_kv(std::cout, first, "peak_rss_mb", rss_sampler.peak_mb());
      ann_bench::emit_logical_io(std::cout, first, io_delta, diff.count(), static_cast<uint64_t>(query_num));
      std::cout << "}" << std::endl;
      std::cout.flags(saved_flags);
      std::cout.precision(saved_precision);
    }
  };

  // Upstream warms the cache by running the test queries themselves at L=100.
  // Setting DGAI_WARMUP_QUERY_FILE replaces that with a separate query file
  // (vectors sampled from the base set, as the benchmark harness does
  // for BufANN), run untimed at each L just before that L is measured.
  const char *warmup_env = std::getenv("DGAI_WARMUP_QUERY_FILE");
  T *warmup = nullptr;
  size_t warmup_num = 0, warmup_dim = 0;
  if (warmup_env != nullptr && *warmup_env != '\0') {
    pipeann::load_bin<T>(warmup_env, warmup, warmup_num, warmup_dim);
    if (warmup_dim != query_dim) {
      std::cout << "Warmup file dim " << warmup_dim << " does not match query dim " << query_dim << std::endl;
      exit(-1);
    }
    if (search_mode != SearchMode::RERANK_SEARCH && search_mode != SearchMode::BEAM_SEARCH) {
      std::cout << "DGAI_WARMUP_QUERY_FILE supports only beam and rerank search" << std::endl;
      exit(-1);
    }
  }
  auto run_warmup = [&](_u64 L) {
    std::vector<uint32_t> tags(recall_at * warmup_num);
    std::vector<float> dists(recall_at * warmup_num);
    std::vector<pipeann::QueryStats> warmup_stats(warmup_num);
    auto ws = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
      if (search_mode == SearchMode::RERANK_SEARCH) {
        _pFlashIndex->rerank_search(warmup + (i * warmup_dim), (uint64_t) recall_at, mem_L, L,
                                    tags.data() + (i * recall_at), dists.data() + (i * recall_at),
                                    (uint64_t) beamwidth, warmup_stats.data() + i, nullptr, nullptr);
      } else {
        _pFlashIndex->beam_search(warmup + (i * warmup_dim), (uint64_t) recall_at, mem_L, L,
                                  tags.data() + (i * recall_at), dists.data() + (i * recall_at),
                                  (uint64_t) beamwidth, warmup_stats.data() + i, nullptr, false);
      }
    }
    std::chrono::duration<double> wd = std::chrono::high_resolution_clock::now() - ws;
    std::cout << "Warmup: " << warmup_num << " queries from " << warmup_env << " at L=" << L << " took "
              << wd.count() << "s" << std::endl;
  };

  if (warmup == nullptr) {
    LOG(INFO) << "Use two ANNS for warming up...";
    uint32_t prev_L = Lvec[0];
    Lvec[0] = 100;
    run_tests(0, false);
    // run_tests(0, false);
    Lvec[0] = prev_L;
    LOG(INFO) << "Warming up finished.";
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "Ls" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
            << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12) << "Mean IOs"
            << std::setw(16) << "Rerank IOs"
            // << std::setw(16) << "Rerank Lat(us)";
            << std::setw(16) << "Mean IO(us)";
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "================================================================"
               "========================================================="
            << std::endl;
  // if (1) {
  //   Lvec.clear();
  //   // for (int i = 200; i <= 300; ) {
  //   //   Lvec.push_back(i);
  //   //   // if (i < 250) {
  //   //   //   i += 2;
  //   //   // } else if (i < 500){
  //   //   //   i += 5;
  //   //   // }else{
  //   //   //   i += 10;
  //   //   // }
  //   //   i+=10;
  //   // }
  //   for (int i = 90; i <= 100; ) {
  //     Lvec.push_back(i);
  //     // if (i < 250) {
  //     //   i += 2;
  //     // } else if (i < 500){
  //     //   i += 5;
  //     // }else{
  //     //   i += 10;
  //     // }
  //     i+=1;
  //   }
  //   query_result_dists.resize(Lvec.size());
  //   query_result_ids.resize(Lvec.size());
  //   query_result_tags.resize(Lvec.size());
  //   query_result_dists.resize(Lvec.size());
  // }

  // Lvec[0]= 100;
  // // for (uint32_t i = 10; i < 94; i+=10) {
  // for (uint32_t i = 20; i < 100; i+=1) {
  //   auto p_reader = (LinuxAlignedFileReader *)_pFlashIndex->reader.get();
  //   p_reader->trunc_len = i;
  //   std::cerr << "trunc_len: " << p_reader->trunc_len << std::endl;
  //   run_tests(0, true);
  //   std::cerr << "bad_query_count: " << gs->bad_query_count << std::endl;
  //   gs->clear();
  // }

  // DGAI_TRUNC_LENS=p1,p2,... repeats the L sweep for each rerank truncation
  // length p (the trunc_len behind the paper's tau) in one process, so a sweep
  // loads the index once. Unset, it runs the L sweep once with the loaded value.
  std::vector<int> trunc_lens;
  if (const char *tl = std::getenv("DGAI_TRUNC_LENS")) {
    std::stringstream ss(tl);
    for (std::string item; std::getline(ss, item, ',');) {
      if (!item.empty()) trunc_lens.push_back(std::stoi(item));
    }
  }
  auto *p_reader = (LinuxAlignedFileReader *) reader.get();
  if (trunc_lens.empty()) trunc_lens.push_back(p_reader->trunc_len);
  for (int p : trunc_lens) {
    p_reader->trunc_len = p;
    std::cout << "trunc_len: " << p << std::endl;
    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
      if (warmup != nullptr) {
        run_warmup(Lvec[test_id]);
      }
      run_tests(test_id, true);
    }
  }
  ShowPeakMemoryStatus();
  if (gs != nullptr) {
    delete gs;
  }
  exit(0);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 12) {
    // tags == 1!
    std::cout << "Usage: " << argv[0]
              << " <index_type (float/int8/uint8)>  <index_prefix_path>"
                 " <num_threads>  <pipeline width> "
                 " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
                 " <K> <similarity (cosine/l2)> "
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
