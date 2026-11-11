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
#include "timer.h"
#include "utils.h"

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

#define WARMUP true

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

  diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    diskann::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    diskann::cout << " beamwidth: " << beamwidth << std::endl;

  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  if (file_exists(truthset_bin)) {
    diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &tags);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
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

  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(m, reader, single_file_index,
                                   tags_flag));  // no tags support yet.

  int res = _pFlashIndex->load(index_prefix_path.c_str(), num_threads);
  if (res != 0) {
    return res;
  }

  if (tags_flag) {
    tsl::robin_set<_u32> active_tags;
    _pFlashIndex->get_active_tags(active_tags);

    diskann::cout << "Loaded " << active_tags.size()
                  << " tags from index for recall measurement." << std::endl;
  } else {
    diskann::cout << "Not loading tags since they are disabled." << std::endl;
  }

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
    std::vector<float>    warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
      _pFlashIndex->cached_beam_search_ids(
          warmup + (i * warmup_aligned_dim), 1, warmup_L,
          warmup_result_tags_64.data() + (i * 1),
          warmup_result_dists.data() + (i * 1), (uint64_t) 4);
    }
    diskann::cout << "..done" << std::endl;
  }

  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  uint32_t optimized_beamwidth = 2;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (beamwidth <= 0) {
      //    diskann::cout<<"Tuning beamwidth.." << std::endl;
      optimized_beamwidth =
          optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
                             warmup_aligned_dim, (_u32) L, optimized_beamwidth);
    } else
      optimized_beamwidth = beamwidth;

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    diskann::QueryStats* stats = new diskann::QueryStats[query_num];

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    auto                  s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      if (tags_flag) {
        _pFlashIndex->cached_beam_search(
            query + (i * query_aligned_dim), (uint64_t) recall_at, (uint64_t) L,
            query_result_tags_32.data() + (i * recall_at),
            query_result_dists[test_id].data() + (i * recall_at),
            (uint64_t) optimized_beamwidth, stats + i);
      } else {
        _pFlashIndex->cached_beam_search_ids(
            query + (i * query_aligned_dim), (uint64_t) recall_at, (uint64_t) L,
            query_result_tags_64.data() + (i * recall_at),
            query_result_dists[test_id].data() + (i * recall_at),
            (uint64_t) optimized_beamwidth, stats + i);
      }
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float                         qps =
        (float) ((1.0 * (double) query_num) / (1.0 * (double) diff.count()));

    if (tags_flag) {
      diskann::convert_types<uint32_t, uint32_t>(
          query_result_tags_32.data(), query_result_tags[test_id].data(),
          (size_t) query_num, (size_t) recall_at);
    } else {
      diskann::convert_types<uint64_t, uint32_t>(
          query_result_tags_64.data(), query_result_tags[test_id].data(),
          (size_t) query_num, (size_t) recall_at);
    }

    float mean_latency = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& s) { return s.total_us; });
    float p50_latency = (float) diskann::get_percentile_stats(
        stats, query_num, 0.50f,
        [](const diskann::QueryStats& s) { return s.total_us; });
    float p99_latency = (float) diskann::get_percentile_stats(
        stats, query_num, 0.99f,
        [](const diskann::QueryStats& s) { return s.total_us; });

    float mean_io_us = (float) diskann::get_mean_stats(
        stats, query_num, [](const diskann::QueryStats& s) { return s.io_us; });
    float p50_io_us = (float) diskann::get_percentile_stats(
        stats, query_num, 0.50f,
        [](const diskann::QueryStats& s) { return s.io_us; });
    float p99_io_us = (float) diskann::get_percentile_stats(
        stats, query_num, 0.99f,
        [](const diskann::QueryStats& s) { return s.io_us; });

    float mean_cpu_us = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& s) { return s.cpu_us; });
    float p50_cpu_us = (float) diskann::get_percentile_stats(
        stats, query_num, 0.50f,
        [](const diskann::QueryStats& s) { return s.cpu_us; });
    float p99_cpu_us = (float) diskann::get_percentile_stats(
        stats, query_num, 0.99f,
        [](const diskann::QueryStats& s) { return s.cpu_us; });

    float mean_cmps = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& s) { return s.n_cmps; });

    float mean_hops = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& s) { return s.n_hops; });
    float p50_hops = (float) diskann::get_percentile_stats(
        stats, query_num, 0.50f,
        [](const diskann::QueryStats& s) { return s.n_hops; });
    float p99_hops = (float) diskann::get_percentile_stats(
        stats, query_num, 0.99f,
        [](const diskann::QueryStats& s) { return s.n_hops; });

    double total_cache_hits = 0, total_ios = 0, total_sectors = 0;
    for (uint64_t i = 0; i < query_num; i++) {
      total_cache_hits += stats[i].n_cache_hits;
      total_ios += stats[i].n_ios;
      total_sectors += stats[i].n_4k;
    }
    double hit_rate =
        (total_cache_hits + total_ios > 0)
            ? 100.0 * total_cache_hits / (total_cache_hits + total_ios)
            : 0.0;

    delete[] stats;

    float recall = 0;
    if (calc_recall_flag) {
      unsigned* gold_std = (tags != nullptr) ? (unsigned*) tags : gt_ids;
      recall = (float) diskann::calculate_recall(
          (_u32) query_num, gold_std, gt_dists, (_u32) gt_dim,
          query_result_tags[test_id].data(), (_u32) recall_at,
          (_u32) recall_at);
    }

    diskann::cout << "\n=== L=" << L << ", Beamwidth=" << optimized_beamwidth
                  << " ===\n";
    diskann::cout << std::setw(28) << std::left
                  << "- Time cost (us):" << std::right
                  << (uint64_t) (diff.count() * 1e6) << "\n";
    diskann::cout << std::setw(28) << std::left << "- QPS:" << std::right << qps
                  << "\n";
    float throughput_mb_s =
        (float) (total_sectors * 4096.0 / (1024.0 * 1024.0) / diff.count());
    diskann::cout << std::setw(28) << std::left
                  << "- Disk throughput (MB/s):" << std::right
                  << throughput_mb_s << "\n";
    if (calc_recall_flag)
      diskann::cout << std::setw(28) << std::left << "- " + recall_string + ":"
                    << std::right << recall << "\n";
    diskann::cout << std::setw(28) << std::left
                  << "- Avg latency (us):" << std::right << mean_latency
                  << "\n";
    diskann::cout << std::setw(28) << std::left
                  << "- P50 latency (us):" << std::right << p50_latency << "\n";
    diskann::cout << std::setw(28) << std::left
                  << "- P99 latency (us):" << std::right << p99_latency << "\n";
    diskann::cout << std::setw(28) << std::left << "- Avg io_us:" << std::right
                  << mean_io_us << "\n";
    diskann::cout << std::setw(28) << std::left << "- P50 io_us:" << std::right
                  << p50_io_us << "\n";
    diskann::cout << std::setw(28) << std::left << "- P99 io_us:" << std::right
                  << p99_io_us << "\n";
    diskann::cout << std::setw(28) << std::left << "- Avg cpu_us:" << std::right
                  << mean_cpu_us << "\n";
    diskann::cout << std::setw(28) << std::left << "- P50 cpu_us:" << std::right
                  << p50_cpu_us << "\n";
    diskann::cout << std::setw(28) << std::left << "- P99 cpu_us:" << std::right
                  << p99_cpu_us << "\n";
    diskann::cout << std::setw(28) << std::left << "- Avg n_cmps:" << std::right
                  << mean_cmps << "\n";
    diskann::cout << std::setw(28) << std::left << "- Avg n_hops:" << std::right
                  << mean_hops << "\n";
    diskann::cout << std::setw(28) << std::left << "- P50 n_hops:" << std::right
                  << p50_hops << "\n";
    diskann::cout << std::setw(28) << std::left << "- P99 n_hops:" << std::right
                  << p99_hops << "\n";
    diskann::cout << std::setw(28) << std::left << "- Cache hits:" << std::right
                  << (uint64_t) total_cache_hits << "\n";
    diskann::cout << std::setw(28) << std::left
                  << "- Cache hit rate:" << std::right << hit_rate << "%\n";
    diskann::cout << std::setw(28) << std::left
                  << "- Total sectors:" << std::right
                  << (uint64_t) total_sectors << std::endl;
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
