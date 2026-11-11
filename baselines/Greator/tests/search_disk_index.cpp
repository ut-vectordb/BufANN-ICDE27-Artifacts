// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "parameters.h"
#include "timer.h"
#include "utils.h"
#include "v2/merge_insert.h"
#include "../../src/ann_bench_metrics.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  diskann::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(8) << percentiles[s] << "%";
  }
  diskann::cout << std::endl;
  diskann::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(9) << results[s];
  }
  diskann::cout << std::endl;
}

void print_rss_checkpoint(const std::string& label) {
  std::ios::fmtflags old_flags = diskann::cout.flags();
  std::streamsize old_precision = diskann::cout.precision();
  diskann::cout << "RSS_CHECKPOINT " << label << " rss_mb=" << std::fixed
                << std::setprecision(3) << ann_bench::current_rss_mb()
                << std::endl;
  diskann::cout.flags(old_flags);
  diskann::cout.precision(old_precision);
}

template<typename T>
int search_disk_index(int argc, char** argv) {
  // load query bin
  T*                query = nullptr;
  unsigned*         gt_ids = nullptr;
  float*            gt_dists = nullptr;
  uint32_t*         tags = nullptr;
  size_t            query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  int         index = 2;
  std::string index_prefix_path(argv[index++]);
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  bool        single_file_index = std::atoi(argv[index++]) != 0;
  bool        tags_flag = std::atoi(argv[index++]) != 0;
  _u64        num_nodes_to_cache = std::atoi(argv[index++]);
  _u32        num_threads = std::atoi(argv[index++]);
  _u32        beamwidth = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string truthset_bin(argv[index++]);
  _u64        recall_at = std::atoi(argv[index++]);
  std::string result_output_prefix(argv[index++]);
  std::string dist_metric(argv[index++]);
  const char* update_index_env = std::getenv("QUERY_USE_UPDATE_INDEX");
  const bool  use_update_index_query =
      update_index_env != nullptr && std::atoi(update_index_env) != 0;
  diskann::cout << "Query path: "
                << (use_update_index_query
                        ? "update-capable MergeInsert disk-only"
                        : "legacy PQFlashIndex")
                << std::endl;

  diskann::Metric m =
      dist_metric == "cosine" ? diskann::Metric::COSINE : diskann::Metric::L2;
  if (dist_metric != "l2" && m == diskann::Metric::L2) {
    diskann::cout << "Unknown distance metric: " << dist_metric
                  << ". Using default(L2) instead." << std::endl;
  }

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  for (int ctr = index; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    diskann::cout
        << "No valid Lsearch found. Lsearch must be at least recall_at"
        << std::endl;
    return -1;
  }
  if (use_update_index_query && beamwidth <= 0) {
    diskann::cout << "Update-capable query path requires explicit beamwidth"
                  << std::endl;
    return -1;
  }
  if (use_update_index_query && !tags_flag) {
    diskann::cout << "Update-capable query path requires tags enabled"
                  << std::endl;
    return -1;
  }

  diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    diskann::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    diskann::cout << " beamwidth: " << beamwidth << std::endl;

  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  unsigned* gt_eval_ids = nullptr;
  if (file_exists(truthset_bin)) {
    diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &tags);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
    gt_eval_ids = (tags != nullptr) ? tags : gt_ids;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
  reader.reset(new WindowsAlignedFileReader());
#else
  reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
  reader.reset(new LinuxAlignedFileReader());
//  reader.reset(new diskann::MemAlignedFileReader());
#endif

  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex;
  std::unique_ptr<diskann::MergeInsert<T, uint32_t>> merge_insert;
  std::unique_ptr<diskann::Distance<T>> dist_cmp;

  if (use_update_index_query) {
    diskann::Parameters paras;
    paras.Set<unsigned>("L_mem", static_cast<unsigned>(Lvec.back()));
    paras.Set<unsigned>("R_mem", 32);
    paras.Set<float>("alpha_mem", 1.2f);
    paras.Set<unsigned>("L_disk", static_cast<unsigned>(Lvec.back()));
    paras.Set<unsigned>("R_disk", 32);
    paras.Set<float>("alpha_disk", 1.2f);
    paras.Set<unsigned>("C", 80);
    paras.Set<unsigned>("beamwidth", beamwidth);
    paras.Set<unsigned>("nodes_to_cache", num_nodes_to_cache);
    paras.Set<_u32>("num_search_threads", num_threads);
    paras.Set<size_t>("merge_th", static_cast<size_t>(10000));
    paras.Set<uint64_t>("merge_nthreads", static_cast<uint64_t>(1));

    dist_cmp.reset(diskann::get_distance_function<T>(m));
    merge_insert.reset(new diskann::MergeInsert<T, uint32_t>(
        paras, query_dim, index_prefix_path + "_query_mem", index_prefix_path,
        index_prefix_path + "_query_merge", dist_cmp.get(), m,
        single_file_index, result_output_prefix + "_query_tmp"));
    print_rss_checkpoint("after_index_load");
    merge_insert->load_disk_cache();
    print_rss_checkpoint("after_load_disk_cache");
  } else {
    _pFlashIndex.reset(new diskann::PQFlashIndex<T>(m, reader, single_file_index,
                                                    tags_flag));

    int res = _pFlashIndex->load(index_prefix_path.c_str(), num_threads);
    if (res != 0) {
      return res;
    }
    print_rss_checkpoint("after_index_load");
  }

  if (!use_update_index_query) {
    // cache bfs levels
    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache
                  << " BFS nodes around medoid(s)" << std::endl;
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    //_pFlashIndex->generate_cache_list_from_sample_queries(
    //   warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();
  }

  omp_set_num_threads(num_threads);

  uint64_t warmup_L;
  warmup_L = 20;
  uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
  T*       warmup = nullptr;
  if (WARMUP) {
    if (file_exists(warmup_query_file)) {
      diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
                                   warmup_dim, warmup_aligned_dim);
    } else {
      warmup_num = (std::min)((_u32) 150000, (_u32) 15000 * num_threads);
      warmup_dim = query_dim;
      warmup_aligned_dim = query_aligned_dim;
      diskann::alloc_aligned(((void**) &warmup),
                             warmup_num * warmup_aligned_dim * sizeof(T),
                             8 * sizeof(T));
      std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
      std::random_device              rd;
      std::mt19937                    gen(rd());
      std::uniform_int_distribution<> dis(-128, 127);
      for (uint32_t i = 0; i < warmup_num; i++) {
        for (uint32_t d = 0; d < warmup_dim; d++) {
          warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
        }
      }
    }
    diskann::cout << "Warming up index... " << std::flush;
    std::vector<uint64_t> warmup_result_tags_64(warmup_num, 0);
    std::vector<uint32_t> warmup_result_tags_32(warmup_num, 0);
    std::vector<float>    warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
      if (use_update_index_query) {
        diskann::QueryStats stats;
        merge_insert->search_disk_only(
            warmup + (i * warmup_aligned_dim), 1, warmup_L,
            warmup_result_tags_32.data() + (i * 1),
            warmup_result_dists.data() + (i * 1), &stats);
      } else {
        _pFlashIndex->cached_beam_search_ids(
            warmup + (i * warmup_aligned_dim), 1, warmup_L,
            warmup_result_tags_64.data() + (i * 1),
            warmup_result_dists.data() + (i * 1), (uint64_t) 4);
      }
    }
    diskann::cout << "..done" << std::endl;
  }

  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
                << std::setw(16) << "QPS" << std::setw(16) << "AvgLat(us)"
                << std::setw(16) << "P50(us)" << std::setw(16) << "P90(us)"
                << std::setw(16) << "P95(us)" << std::setw(16) << "P99(us)"
                << std::setw(16) << "P99.9(us)" << std::setw(16)
                << "Mean IOs" << std::setw(16) << "CPU (s)";
  if (calc_recall_flag) {
    diskann::cout << std::setw(16) << recall_string << std::endl;
  } else
    diskann::cout << std::endl;
  diskann::cout
      << "==============================================================="
         "==========================================="
      << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  uint32_t optimized_beamwidth = 2;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (!use_update_index_query && beamwidth <= 0) {
      // diskann::cout<<"Tuning beamwidth.." << std::endl;
      optimized_beamwidth =
          optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
                             warmup_aligned_dim, (_u32) L, optimized_beamwidth);
    } else
      optimized_beamwidth = beamwidth;

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    std::vector<diskann::QueryStats> stats(query_num);

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    ann_bench::RssSampler rss_sampler;
    ann_bench::LogicalIoCounter::reset_and_enable();
    rss_sampler.start();
    auto                  s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      if (use_update_index_query) {
        merge_insert->search_disk_only(
            query + (i * query_aligned_dim), (uint64_t) recall_at, (uint64_t) L,
            query_result_tags_32.data() + (i * recall_at),
            query_result_dists[test_id].data() + (i * recall_at),
            stats.data() + i);
      } else {
        _pFlashIndex->cached_beam_search(
            query + (i * query_aligned_dim), (uint64_t) recall_at, (uint64_t) L,
            query_result_tags_32.data() + (i * recall_at),
            query_result_dists[test_id].data() + (i * recall_at),
            (uint64_t) optimized_beamwidth, stats.data() + i);
      }
    }

    auto                          e = std::chrono::high_resolution_clock::now();
    rss_sampler.stop();
    const auto io_delta = ann_bench::LogicalIoCounter::snapshot_and_disable();
    std::chrono::duration<double> diff = e - s;
    float                         qps =
        (float) ((1.0 * (double) query_num) / (1.0 * (double) diff.count()));

    diskann::convert_types<uint32_t, uint32_t>(
        query_result_tags_32.data(), query_result_tags[test_id].data(),
        (size_t) query_num, (size_t) recall_at);

    float mean_latency = (float) diskann::get_mean_stats(
        stats.data(), query_num,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float latency_50 = (float) diskann::get_percentile_stats(
        stats.data(), query_num, 0.500f,
        [](const diskann::QueryStats& stats) { return stats.total_us; });
    float latency_90 = (float) diskann::get_percentile_stats(
        stats.data(), query_num, 0.900f,
        [](const diskann::QueryStats& stats) { return stats.total_us; });
    float latency_95 = (float) diskann::get_percentile_stats(
        stats.data(), query_num, 0.950f,
        [](const diskann::QueryStats& stats) { return stats.total_us; });
    float latency_99 = (float) diskann::get_percentile_stats(
        stats.data(), query_num, 0.990f,
        [](const diskann::QueryStats& stats) { return stats.total_us; });
    float latency_999 = (float) diskann::get_percentile_stats(
        stats.data(), query_num, 0.999f,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float mean_ios = (float) diskann::get_mean_stats(
        stats.data(), query_num,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });

    // Per-query blocking IO wait (us), to match BufANN's query_io_* series.
    float mean_io_us = (float) diskann::get_mean_stats(
        stats.data(), query_num,
        [](const diskann::QueryStats& stats) { return stats.io_us; });

    float mean_cpuus = (float) diskann::get_mean_stats(
        stats.data(), query_num,
        [](const diskann::QueryStats& stats) { return stats.cpu_us; });
    float recall = 0;
    if (calc_recall_flag) {
      recall = (float) diskann::calculate_recall(
          (_u32) query_num, gt_eval_ids, gt_dists, (_u32) gt_dim,
          query_result_tags[test_id].data(), (_u32) recall_at,
          (_u32) recall_at);
    }

    diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth
                  << std::setw(16) << qps << std::setw(16) << mean_latency
                  << std::setw(16) << latency_50
                  << std::setw(16) << latency_90
                  << std::setw(16) << latency_95
                  << std::setw(16) << latency_99
                  << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                  << std::setw(16) << mean_cpuus;
    if (calc_recall_flag) {
      diskann::cout << std::setw(16) << recall << std::endl;
    }
    bool first = true;
    const char *phase_env = std::getenv("ANN_BENCH_PHASE");
    std::cout << "{";
    ann_bench::json_kv(std::cout, first, "baseline", "Greator");
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
  std::this_thread::sleep_for(std::chrono::seconds(10));

  diskann::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    std::string cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                            query_num, recall_at);

    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_tags_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_tags[test_id].data(),
                            query_num, recall_at);
    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
    diskann::save_bin<float>(cur_result_path,
                             query_result_dists[test_id++].data(), query_num,
                             recall_at);
  }

  diskann::aligned_free(query);
  if (warmup != nullptr)
    diskann::aligned_free(warmup);
  delete[] gt_ids;
  delete[] gt_dists;
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 14) {
    diskann::cout
        << "Usage: " << argv[0]
        << " <index_type (float/int8/uint8)>  <index_prefix_path>"
           " <single_file_index(0/1)> <tags(0/1) "
           " <num_nodes_to_cache>  <num_threads>  <beamwidth (use 0 to "
           "optimize internally)> "
           " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
           " <K>  <result_output_prefix> <similarity (cosine/l2)> "
           " <L1>  [L2] etc.  See README for more information on parameters."
        << std::endl;
    exit(-1);
  }

  diskann::cout << "Attach  debugger and press a key" << std::endl;
  /*  char x;
    std::cin >> x;
    */

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    diskann::cout << "Unsupported index type. Use float or int8 or uint8"
                  << std::endl;
}
