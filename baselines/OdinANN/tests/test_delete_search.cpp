#include "dynamic_index.h"

#include <index.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

#include "utils.h"
#include "utils/log.h"
#include "utils/index_build_utils.h"
#include "utils/timer.h"

int NUM_DELETE_THREADS = 1;
int NUM_MERGE_THREADS = 1;
int NUM_SEARCH_THREADS = 32;
int search_mode = BEAM_SEARCH;

static int get_delete_threads() {
  return static_cast<int>(pipeann::env_u32("DELETE_THREADS", NUM_DELETE_THREADS));
}

static int get_merge_threads() {
  return static_cast<int>(pipeann::env_u32("MERGE_THREADS", NUM_MERGE_THREADS));
}

static int get_search_threads() {
  return static_cast<int>(pipeann::env_u32("QUERY_THREADS", NUM_SEARCH_THREADS));
}

template<typename T, typename TagT>
void sync_search_kernel(T *query, size_t query_num, size_t query_dim, const int recall_at, uint32_t mem_L, uint64_t L,
                        uint32_t beam_width, pipeann::DynamicSSDIndex<T, TagT> &sync_index,
                        const std::string &truthset_file) {
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  size_t gt_num = 0, gt_dim = 0;

  if (!file_exists(truthset_file)) {
    LOG(FATAL) << "Truth file not found: " << truthset_file;
  }
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
#pragma omp parallel for num_threads(get_search_threads()) schedule(dynamic)
  for (int64_t i = 0; i < static_cast<int64_t>(query_num); i++) {
    sync_index.search(query + i * query_dim, recall_at, mem_L, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);
    latency_stats[i] = stats[i].total_us / 1000.0;
    if (search_mode == BEAM_SEARCH) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float qps = static_cast<float>(query_num / diff.count());
  float recall = pipeann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
  float mean_latency = static_cast<float>(
      pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.total_us; }));
  float mean_ios =
      static_cast<float>(pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) {
        return stats.n_ios;
      }));

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cout << std::setw(6) << L << std::setw(12) << beam_width << std::setw(12) << qps << std::setw(12)
            << mean_latency << std::setw(12) << mean_ios << std::setw(12) << recall << std::endl;

  delete[] gt_ids;
  delete[] gt_dists;
  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] stats;
}

template<typename T, typename TagT>
void deletion_kernel(pipeann::DynamicSSDIndex<T, TagT> &sync_index, const std::vector<TagT> &delete_tags) {
  LOG(INFO) << "Begin delete-only workload for " << delete_tags.size() << " tags";
  pipeann::Timer timer;
#pragma omp parallel for num_threads(get_delete_threads())
  for (int64_t i = 0; i < static_cast<int64_t>(delete_tags.size()); i++) {
    sync_index.lazy_delete(delete_tags[i]);
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  LOG(INFO) << "Deleted " << delete_tags.size() << " / " << delete_tags.size() << " points in " << time_secs << "s";
}

template<typename T, typename TagT>
void merge_kernel(pipeann::DynamicSSDIndex<T, TagT> &sync_index) {
  pipeann::Timer timer;
  sync_index.final_merge(get_merge_threads());
  LOG(INFO) << "Merge time : " << timer.elapsed() / 1000 << " ms";
}

template<typename T, typename TagT>
void run_delete_search(const unsigned L_disk, const uint64_t delete_start_tag, const uint64_t delete_count,
                       const std::string &index_prefix, const std::string &query_file, const std::string &truthset_file,
                       const int recall_at, const std::vector<uint64_t> &Lsearch, const unsigned beam_width,
                       const uint32_t search_beam_width, const uint32_t search_mem_L, pipeann::Distance<T> *dist_cmp) {
  pipeann::IndexBuildParameters paras;
  paras.set(0, L_disk, pipeann::env_u32("C", 384), pipeann::env_f32("ALPHA", 1.2f),
            static_cast<unsigned>(get_search_threads() + get_delete_threads()), true, beam_width);

  T *query = nullptr;
  size_t query_num = 0, query_dim = 0;
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);

  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric,
                                               search_mode, (search_mem_L > 0));

  std::vector<TagT> delete_tags(delete_count);
  std::iota(delete_tags.begin(), delete_tags.end(), static_cast<TagT>(delete_start_tag));
  deletion_kernel(sync_index, delete_tags);
  merge_kernel(sync_index);

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
            << "AvgLat(us)" << std::setw(12) << "Mean IOs" << std::setw(12) << ("Recall@" + std::to_string(recall_at))
            << std::endl;
  std::cout << "========================================================================" << std::endl;

  for (uint64_t L : Lsearch) {
    sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, L, search_beam_width, sync_index,
                       truthset_file);
  }
}

int main(int argc, char **argv) {
  if (argc < 17) {
    LOG(INFO) << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <L_disk> <delete_start_tag> <delete_count>"
              << " <delete_threads> <merge_threads> <search_threads> <search_mode> <index_prefix> <query_file> <truthset_file>"
              << " <recall@> <beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>";
    exit(-1);
  }

  int arg_no = 2;
  unsigned L_disk = static_cast<unsigned>(atoi(argv[arg_no++]));
  uint64_t delete_start_tag = std::stoull(argv[arg_no++]);
  uint64_t delete_count = std::stoull(argv[arg_no++]);
  NUM_DELETE_THREADS = atoi(argv[arg_no++]);
  NUM_MERGE_THREADS = atoi(argv[arg_no++]);
  NUM_SEARCH_THREADS = atoi(argv[arg_no++]);
  search_mode = atoi(argv[arg_no++]);
  std::string index_prefix(argv[arg_no++]);
  std::string query_file(argv[arg_no++]);
  std::string truthset_file(argv[arg_no++]);
  int recall_at = atoi(argv[arg_no++]);
  unsigned beam_width = static_cast<unsigned>(atoi(argv[arg_no++]));
  unsigned search_beam_width = static_cast<unsigned>(atoi(argv[arg_no++]));
  unsigned search_mem_L = static_cast<unsigned>(atoi(argv[arg_no++]));

  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(static_cast<uint64_t>(atoi(argv[i])));
  }

  if (std::string(argv[1]) == "float") {
    pipeann::DistanceL2Float dist_cmp;
    run_delete_search<float, unsigned>(L_disk, delete_start_tag, delete_count, index_prefix, query_file, truthset_file,
                                       recall_at, Lsearch, beam_width, search_beam_width, search_mem_L, &dist_cmp);
  } else if (std::string(argv[1]) == "int8") {
    pipeann::DistanceL2Int8 dist_cmp;
    run_delete_search<int8_t, unsigned>(L_disk, delete_start_tag, delete_count, index_prefix, query_file, truthset_file,
                                        recall_at, Lsearch, beam_width, search_beam_width, search_mem_L, &dist_cmp);
  } else if (std::string(argv[1]) == "uint8") {
    pipeann::DistanceL2UInt8 dist_cmp;
    run_delete_search<uint8_t, unsigned>(L_disk, delete_start_tag, delete_count, index_prefix, query_file,
                                         truthset_file, recall_at, Lsearch, beam_width, search_beam_width,
                                         search_mem_L, &dist_cmp);
  } else {
    LOG(INFO) << "Unsupported type. Use float/int8/uint8";
    exit(-1);
  }
}
