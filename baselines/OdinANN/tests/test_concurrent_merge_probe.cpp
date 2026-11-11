#include "dynamic_index.h"

#include <index.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "utils.h"
#include "utils/log.h"
#include "utils/index_build_utils.h"
#include "utils/timer.h"

namespace {

int NUM_UPDATE_THREADS = 1;
int NUM_MERGE_THREADS = 1;
int NUM_SEARCH_THREADS = 1;
int SEARCH_MODE = BEAM_SEARCH;

template<typename T>
void load_vector_range(const std::string &data_bin, uint64_t start_tag, uint64_t count, std::vector<T> &data,
                       size_t &dim) {
  int npts_i32 = 0;
  int dim_i32 = 0;
  std::ifstream reader(data_bin, std::ios::binary);
  if (!reader) {
    LOG(FATAL) << "failed to open data bin: " << data_bin;
  }

  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));
  if (!reader) {
    LOG(FATAL) << "failed to read header from " << data_bin;
  }

  if (start_tag + count > static_cast<uint64_t>(npts_i32)) {
    LOG(FATAL) << "requested range [" << start_tag << ", " << (start_tag + count) << ") exceeds point count "
               << npts_i32 << " in " << data_bin;
  }

  dim = static_cast<size_t>(dim_i32);
  data.resize(count * dim);
  reader.seekg(2 * sizeof(int) + static_cast<std::streamoff>(start_tag * dim * sizeof(T)), std::ios::beg);
  reader.read((char *) data.data(), static_cast<std::streamsize>(count * dim * sizeof(T)));
  if (!reader) {
    LOG(FATAL) << "failed to read vector range from " << data_bin;
  }
}

template<typename T, typename TagT>
void run_insert_batch(const std::string &data_bin, uint64_t start_tag, uint64_t count,
                      pipeann::DynamicSSDIndex<T, TagT> &sync_index) {
  if (count == 0) {
    return;
  }

  std::vector<T> data;
  size_t dim = 0;
  load_vector_range<T>(data_bin, start_tag, count, data, dim);

  pipeann::Timer timer;
#pragma omp parallel for num_threads(NUM_UPDATE_THREADS)
  for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
    sync_index.insert(data.data() + i * dim, static_cast<TagT>(start_tag + static_cast<uint64_t>(i)));
  }
  LOG(INFO) << "Inserted " << count << " points in " << timer.elapsed() / 1.0e6f << "s";
}

template<typename T, typename TagT>
void run_delete_batch(uint64_t start_tag, uint64_t count, pipeann::DynamicSSDIndex<T, TagT> &sync_index) {
  if (count == 0) {
    return;
  }

  pipeann::Timer timer;
#pragma omp parallel for num_threads(NUM_UPDATE_THREADS)
  for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
    sync_index.lazy_delete(static_cast<TagT>(start_tag + static_cast<uint64_t>(i)));
  }
  LOG(INFO) << "Deleted " << count << " points in " << timer.elapsed() / 1.0e6f << "s";
}

template<typename T, typename TagT>
void run_query_once(T *query, size_t query_num, size_t query_dim, const std::string &truthset_file, int recall_at,
                    uint32_t mem_L, uint64_t L, uint32_t beam_width, pipeann::DynamicSSDIndex<T, TagT> &sync_index) {
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  size_t gt_num = 0, gt_dim = 0;
  pipeann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

  auto *query_result_dists = new float[recall_at * query_num];
  auto *query_result_tags = new TagT[recall_at * query_num];
  auto *stats = new pipeann::QueryStats[query_num];
  std::vector<double> latency_stats(query_num, 0);

  for (uint32_t q = 0; q < query_num; q++) {
    for (uint32_t r = 0; r < static_cast<uint32_t>(recall_at); r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < static_cast<int64_t>(query_num); i++) {
    sync_index.search(query + i * query_dim, recall_at, mem_L, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);
    latency_stats[i] = stats[i].total_us / 1000.0;
    if (SEARCH_MODE == BEAM_SEARCH) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float qps = static_cast<float>(query_num / diff.count());
  float recall = pipeann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
  float mean_ios = static_cast<float>(
      pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; }));

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cout << std::setw(6) << L << std::setw(12) << qps << std::setw(12)
            << (std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<double>(query_num))
            << std::setw(12) << latency_stats[static_cast<size_t>(0.50 * static_cast<double>(query_num))]
            << std::setw(12) << latency_stats[static_cast<size_t>(0.90 * static_cast<double>(query_num))]
            << std::setw(12) << latency_stats[static_cast<size_t>(0.95 * static_cast<double>(query_num))]
            << std::setw(12) << latency_stats[static_cast<size_t>(0.99 * static_cast<double>(query_num))]
            << std::setw(12) << recall << std::setw(12) << mean_ios << std::endl;

  delete[] gt_ids;
  delete[] gt_dists;
  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] stats;
}

template<typename T, typename TagT>
void merge_kernel(pipeann::DynamicSSDIndex<T, TagT> &sync_index) {
  pipeann::Timer timer;
  sync_index.final_merge(NUM_MERGE_THREADS);
  LOG(INFO) << "Merge time : " << timer.elapsed() / 1000 << " ms";
}

template<typename T, typename TagT>
void run_probe(const std::string &probe_mode, const std::string &workload, const std::string &data_bin,
               unsigned L_disk, uint64_t base_points, uint64_t batch_size, uint64_t num_steps,
               const std::string &index_prefix, const std::string &query_file, const std::string &truthset_file,
               int recall_at, const std::vector<uint64_t> &Lsearch, unsigned beam_width, unsigned search_beam_width,
               unsigned search_mem_L, pipeann::Distance<T> *dist_cmp) {
  pipeann::IndexBuildParameters paras;
  paras.set(0, L_disk, pipeann::env_u32("C", 384), pipeann::env_f32("ALPHA", 1.2f),
            NUM_SEARCH_THREADS + NUM_UPDATE_THREADS, true, beam_width);

  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric,
                                               SEARCH_MODE, (search_mem_L > 0));

  T *query = nullptr;
  size_t query_num = 0, query_dim = 0;
  if (probe_mode == "query_merge") {
    pipeann::load_bin<T>(query_file, query, query_num, query_dim);
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    std::cout << std::setw(6) << "L" << std::setw(12) << "QPS" << std::setw(12) << "Avg(ms)" << std::setw(12)
              << "50(ms)" << std::setw(12) << "90(ms)" << std::setw(12) << "95(ms)" << std::setw(12) << "99(ms)"
              << std::setw(12) << ("Recall@" + std::to_string(recall_at)) << std::setw(12) << "MeanIOs"
              << std::endl;
    std::cout << "================================================================================================"
              << std::endl;
  }

  LOG(INFO) << "=== PipeANN concurrent merge probe ===";
  LOG(INFO) << "probe_mode: " << probe_mode;
  LOG(INFO) << "workload: " << workload;
  LOG(INFO) << "base_points: " << base_points;
  LOG(INFO) << "batch_size: " << batch_size;
  LOG(INFO) << "num_steps: " << num_steps;

  if (workload == "insert") {
    run_insert_batch<T, TagT>(data_bin, base_points, batch_size, sync_index);
  } else {
    run_delete_batch<T, TagT>(base_points, batch_size, sync_index);
  }

  auto merge_future = std::async(std::launch::async, merge_kernel<T, TagT>, std::ref(sync_index));

  if (probe_mode == "update_merge") {
    uint64_t remaining = batch_size * (num_steps > 1 ? (num_steps - 1) : 0);
    if (remaining > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (workload == "insert") {
        run_insert_batch<T, TagT>(data_bin, base_points + batch_size, remaining, sync_index);
      } else {
        run_delete_batch<T, TagT>(base_points + batch_size, remaining, sync_index);
      }
    } else {
      LOG(INFO) << "No remaining updates after pre-merge batch; set num_steps >= 2 to exercise overlap.";
    }
    merge_future.get();
    LOG(INFO) << "Concurrent update+merge probe complete.";
  } else {
    for (uint64_t L : Lsearch) {
      run_query_once<T, TagT>(query, query_num, query_dim, truthset_file, recall_at, search_mem_L, L,
                              search_beam_width, sync_index);
    }
    int query_round = 0;
    while (merge_future.wait_for(std::chrono::milliseconds(250)) != std::future_status::ready) {
      for (uint64_t L : Lsearch) {
        run_query_once<T, TagT>(query, query_num, query_dim, truthset_file, recall_at, search_mem_L, L,
                                search_beam_width, sync_index);
      }
      ++query_round;
      LOG(INFO) << "Completed query round " << query_round << " while merge was active.";
    }
    merge_future.get();
    LOG(INFO) << "Running post-merge query sweep.";
    for (uint64_t L : Lsearch) {
      run_query_once<T, TagT>(query, query_num, query_dim, truthset_file, recall_at, search_mem_L, L,
                              search_beam_width, sync_index);
    }
    delete[] query;
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 20) {
    LOG(INFO) << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <probe_mode[update_merge/query_merge]> <workload[insert/delete]>"
              << " <data_bin> <L_disk> <base_points> <batch_size> <num_steps> <update_threads> <merge_threads>"
              << " <search_threads> <search_mode> <index_prefix> <query_file_or_null> <truthset_file_or_null>"
              << " <recall@> <beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>";
    return -1;
  }

  int arg_no = 2;
  const std::string probe_mode(argv[arg_no++]);
  const std::string workload(argv[arg_no++]);
  const std::string data_bin(argv[arg_no++]);
  const unsigned L_disk = static_cast<unsigned>(atoi(argv[arg_no++]));
  const uint64_t base_points = std::stoull(argv[arg_no++]);
  const uint64_t batch_size = std::stoull(argv[arg_no++]);
  const uint64_t num_steps = std::stoull(argv[arg_no++]);
  NUM_UPDATE_THREADS = atoi(argv[arg_no++]);
  NUM_MERGE_THREADS = atoi(argv[arg_no++]);
  NUM_SEARCH_THREADS = atoi(argv[arg_no++]);
  SEARCH_MODE = atoi(argv[arg_no++]);
  const std::string index_prefix(argv[arg_no++]);
  const std::string query_file(argv[arg_no++]);
  const std::string truthset_file(argv[arg_no++]);
  const int recall_at = atoi(argv[arg_no++]);
  const unsigned beam_width = static_cast<unsigned>(atoi(argv[arg_no++]));
  const unsigned search_beam_width = static_cast<unsigned>(atoi(argv[arg_no++]));
  const unsigned search_mem_L = static_cast<unsigned>(atoi(argv[arg_no++]));

  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(static_cast<uint64_t>(atoi(argv[i])));
  }

  if (probe_mode != "update_merge" && probe_mode != "query_merge") {
    LOG(FATAL) << "Unsupported probe_mode: " << probe_mode;
  }
  if (workload != "insert" && workload != "delete") {
    LOG(FATAL) << "Unsupported workload: " << workload;
  }
  if (probe_mode == "query_merge" && (query_file == "null" || truthset_file == "null")) {
    LOG(FATAL) << "query_merge requires query and truth files";
  }

  if (std::string(argv[1]) == "float") {
    pipeann::DistanceL2Float dist_cmp;
    run_probe<float, unsigned>(probe_mode, workload, data_bin, L_disk, base_points, batch_size, num_steps,
                               index_prefix, query_file, truthset_file, recall_at, Lsearch, beam_width,
                               search_beam_width, search_mem_L, &dist_cmp);
  } else if (std::string(argv[1]) == "int8") {
    pipeann::DistanceL2Int8 dist_cmp;
    run_probe<int8_t, unsigned>(probe_mode, workload, data_bin, L_disk, base_points, batch_size, num_steps,
                                index_prefix, query_file, truthset_file, recall_at, Lsearch, beam_width,
                                search_beam_width, search_mem_L, &dist_cmp);
  } else if (std::string(argv[1]) == "uint8") {
    pipeann::DistanceL2UInt8 dist_cmp;
    run_probe<uint8_t, unsigned>(probe_mode, workload, data_bin, L_disk, base_points, batch_size, num_steps,
                                 index_prefix, query_file, truthset_file, recall_at, Lsearch, beam_width,
                                 search_beam_width, search_mem_L, &dist_cmp);
  } else {
    LOG(FATAL) << "Unsupported type. Use float/int8/uint8";
  }

  return 0;
}
