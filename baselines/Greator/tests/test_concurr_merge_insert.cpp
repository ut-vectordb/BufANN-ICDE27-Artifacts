// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "v2/index_merger.h"
#include "v2/merge_insert.h"

#include <mutex>
#include <numeric>
#include <random>
#include <omp.h>
#include <cstring>
#include <ctime>
#include <timer.h>
#include <iomanip>
#include <atomic>
#include <cstdlib>
#include <stdexcept>

#include "aux_utils.h"
#include "utils.h"
#include "math_utils.h"
#include "partition_and_pq.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <pthread.h>
#include <sched.h>

// random number generator
std::random_device dev;
std::mt19937       rng(dev());

tsl::robin_map<std::string, uint32_t> params;
float                                 mem_alpha, merge_alpha;
uint32_t              medoid_id = std::numeric_limits<uint32_t>::max();
std::atomic_bool      _insertions_done(true);
std::atomic_bool      _del_done(true);
std::vector<uint32_t> Lvec;
std::future<void>     delete_future;
std::future<void>     insert_future;
std::future<void>     merge_future;
diskann::Timer        global_timer;
std::string           all_points_file;
bool                  save_index_as_one_file;
std::string           TMP_FOLDER;
std::string           query_file = "";
std::string           truthset_file = "";
std::vector<uint32_t> scripted_insert_ids;
std::vector<uint32_t> scripted_delete_ids;
size_t                scripted_insert_cursor = 0;
size_t                scripted_delete_cursor = 0;

uint32_t env_u32(const char *name, uint32_t default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return default_value;
  }

  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    throw std::runtime_error(std::string("invalid integer env var ") + name);
  }
  return static_cast<uint32_t>(parsed);
}

int get_insert_threads() {
  return static_cast<int>(env_u32("INSERT_THREADS", 2));
}

int get_delete_threads() {
  return static_cast<int>(env_u32("DELETE_THREADS", 1));
}

int get_search_threads() {
  return static_cast<int>(env_u32("QUERY_THREADS", 16));
}

uint32_t get_merge_id_map() {
  return env_u32("GREATOR_ID_MAP", 2);
}

std::chrono::milliseconds get_update_poll_sleep() {
  return std::chrono::milliseconds(env_u32("GREATOR_UPDATE_POLL_MS", 1000));
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

  const size_t count = static_cast<size_t>(count_i32);
  std::vector<uint32_t> ids(count);
  input.read(reinterpret_cast<char *>(ids.data()),
             static_cast<std::streamsize>(count * sizeof(uint32_t)));
  if (!input) {
    throw std::runtime_error("short read in scripted IDs file: " + path);
  }

  char extra = 0;
  input.read(&extra, 1);
  if (!input.eof()) {
    throw std::runtime_error("unexpected trailing bytes in scripted IDs file: " +
                             path);
  }
  return ids;
}

void maybe_load_scripted_ids(const char             *env_name,
                             std::vector<uint32_t> &target,
                             const std::string     &label) {
  const char *path = std::getenv(env_name);
  if (path == nullptr || path[0] == '\0') {
    return;
  }

  target = load_scripted_ids_file(path);
  std::cout << "Loaded " << target.size() << " scripted " << label
            << " IDs from " << path << std::endl;
}

std::vector<uint32_t> take_scripted_ids(
    const std::vector<uint32_t>    &source, size_t &cursor,
    const uint32_t                  count,
    const tsl::robin_set<uint32_t> &membership,
    const std::string              &label) {
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
    const uint32_t id = source[idx];
    if (membership.find(id) == membership.end()) {
      throw std::runtime_error("scripted " + label +
                               " ID is not valid for the current iteration: " +
                               std::to_string(id));
    }
    picked.push_back(id);
  }
  cursor += count;
  return picked;
}

template<typename T, typename TagT = uint32_t>
void seed_iter(tsl::robin_set<uint32_t> &active_set,
               tsl::robin_set<uint32_t> &deletable_set,
               tsl::robin_set<uint32_t> &inactive_set,
               const std::string        &inserted_points_file,
               const std::string        &inserted_tags_file,
               tsl::robin_set<TagT>     &deleted_tags) {
  const uint32_t insert_count = params[std::string("insert_count")];
  const uint32_t delete_count = params[std::string("delete_count")];
  const uint32_t ndims = params[std::string("ndims")];
  std::cout << "ITER: start = " << active_set.size() << ", "
            << inactive_set.size() << "\n";

  // pick `delete_count` tags
  std::vector<uint32_t> delete_vec;
  if (!::scripted_delete_ids.empty()) {
    delete_vec =
        take_scripted_ids(::scripted_delete_ids, ::scripted_delete_cursor,
                          delete_count, deletable_set, "delete");
  } else {
    std::vector<uint32_t> active_vec(deletable_set.begin(), deletable_set.end());
    std::shuffle(active_vec.begin(), active_vec.end(), rng);
    if (active_vec.size() < delete_count)
      delete_vec.insert(delete_vec.end(), active_vec.begin(), active_vec.end());
    else
      delete_vec.insert(delete_vec.end(), active_vec.begin(),
                        active_vec.begin() + delete_count);
  }
  for (auto iter : delete_vec)
    deleted_tags.insert(iter);
  for (auto iter : delete_vec)
    active_set.erase(iter);
  for (auto iter : delete_vec)
    deletable_set.erase(iter);
  std::cout << "ITER: DELETE - " << delete_vec.size() << " IDs\n";
  // pick `insert_count` tags
  std::vector<uint32_t> insert_vec;
  if (!::scripted_insert_ids.empty()) {
    insert_vec =
        take_scripted_ids(::scripted_insert_ids, ::scripted_insert_cursor,
                          insert_count, inactive_set, "insert");
  } else {
    std::vector<uint32_t> inactive_vec(inactive_set.begin(), inactive_set.end());
    std::shuffle(inactive_vec.begin(), inactive_vec.end(), rng);
    if (inactive_vec.size() < insert_count)
      insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                        inactive_vec.end());
    else
      insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                        inactive_vec.begin() + insert_count);
  }

  std::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
            << inserted_tags_file << "\n";
  for (auto iter : insert_vec)
    inactive_set.erase(iter);
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
    throw std::runtime_error("failed to read valid metadata from insert source "
                             "file: " +
                             ::all_points_file);
  }

  const uint32_t src_npts = static_cast<uint32_t>(src_npts_i32);
  const uint32_t src_ndims = static_cast<uint32_t>(src_ndims_i32);
  if (src_ndims != ndims) {
    throw std::runtime_error("insert source dims mismatch. expected " +
                             std::to_string(ndims) + ", got " +
                             std::to_string(src_ndims) + " from " +
                             ::all_points_file);
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
                               std::to_string(actual_idx) + " >= " +
                               std::to_string(src_npts));
    }
    T       *point = new T[ndims];
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

  // balance tags
  inactive_set.insert(delete_vec.begin(), delete_vec.end());
  active_set.insert(insert_vec.begin(), insert_vec.end());

  diskann::cout << "ITER: end = " << active_set.size() << ", "
                << inactive_set.size() << "\n";
#ifndef _WINDOWS
  std::cout << "ITER: end = " << active_set.size() << ", "
            << inactive_set.size() << "\n";
  // malloc_stats();
#endif
}

float compute_active_recall(const uint32_t *result_tags,
                            const uint32_t  result_count,
                            const uint32_t *gs_tags, const uint64_t gs_count,
                            const tsl::robin_set<uint32_t> &inactive_set) {
  tsl::robin_set<uint32_t> active_gs;
  for (uint32_t i = 0; i < gs_count && active_gs.size() < result_count; i++) {
    auto iter = inactive_set.find(gs_tags[i]);
    if (iter == inactive_set.end()) {
      active_gs.insert(gs_tags[i]);
    }
  }
  uint32_t match = 0;
  for (uint32_t i = 0; i < result_count; i++) {
    match += (active_gs.find(result_tags[i]) != active_gs.end());
  }
  return ((float) match / (float) result_count) * 100;
}

float compute_truthset_recall(const uint32_t *result_tags, const uint32_t result_count,
                              const unsigned *gt_eval_ids, float *gt_dists,
                              const unsigned gt_dim, const unsigned query_num) {
  return diskann::calculate_recall(query_num, const_cast<unsigned *>(gt_eval_ids),
                                   gt_dists, gt_dim,
                                   const_cast<uint32_t *>(result_tags),
                                   result_count, result_count);
}

template<typename T, typename TagT = uint32_t>
void search_disk_index(const std::string              &index_prefix_path,
                       const tsl::robin_set<uint32_t> &inactive_tags,
                       const std::string              &query_path,
                       const std::string              &gs_path) {
  std::string pq_prefix = index_prefix_path + "_pq";
  std::string disk_index_file = index_prefix_path + "_disk.index";
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  uint32_t    beamwidth = params[std::string("beam_width")];
  uint32_t    num_threads = 60;
  std::string query_bin = query_path;
  std::string truthset_bin = gs_path;
  uint64_t    recall_at = params[std::string("recall_k")];
  uint64_t    search_L = ::Lvec[0];
  // hold data
  T        *query = nullptr;
  unsigned *gt_ids = nullptr;
  uint32_t *gt_tags = nullptr;
  float    *gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;

  // load query + truthset
  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);
  diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                         &gt_tags);
  unsigned *gt_eval_ids = (gt_tags != nullptr) ? gt_tags : gt_ids;
  if (gt_num != query_num) {
    std::cout << "Error. Mismatch in number of queries and ground truth data"
              << std::endl;
  }

  // load PQ Flash Index
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<T, uint32_t>> _pFlashIndex(
      new diskann::PQFlashIndex<T, uint32_t>(diskann::Metric::L2, reader,
                                             ::save_index_as_one_file, true));
  int res = _pFlashIndex->load(num_threads, pq_prefix.c_str(),
                               disk_index_file.c_str());
  if (res != 0) {
    std::cerr << "Failed to load index.\n";
    exit(-1);
  }

  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint32_t> query_result_tags;
  std::vector<float>    query_result_dists;
  query_result_ids.resize(recall_at * query_num);
  query_result_dists.resize(recall_at * query_num);
  query_result_tags.resize(recall_at * query_num);
  diskann::QueryStats  *stats = new diskann::QueryStats[query_num];
  std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
#pragma omp parallel for schedule(dynamic, 1)  // num_threads(1)
  for (_s64 i = 0; i < (int64_t) query_num; i++) {
    _pFlashIndex->cached_beam_search(
        query + (i * query_aligned_dim), recall_at, search_L,
        query_result_ids_64.data() + (i * recall_at),
        query_result_dists.data() + (i * recall_at), beamwidth, stats + i,
        query_result_tags.data() + (i * recall_at));
  }

  // compute mean recall, IOs
  float mean_recall = compute_truthset_recall(
      query_result_tags.data(), (unsigned) recall_at, gt_eval_ids, gt_dists,
      (unsigned) gt_dim, (unsigned) query_num);

  float mean_ios = (float) diskann::get_mean_stats(
      stats, query_num,
      [](const diskann::QueryStats &stats) { return stats.n_ios; });
  std::cout << "PQFlashIndex :: recall-" << recall_at << "@" << recall_at
            << ": " << mean_recall << ", mean IOs: " << mean_ios << "\n";
  diskann::aligned_free(query);
  delete[] stats;
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

template<typename T, typename TagT = uint32_t>
void search_kernel(diskann::MergeInsert<T>        &merge_insert,
                   const tsl::robin_set<uint32_t> &active_tags,
                   bool                            print_stats = false) {
  uint64_t recall_at = params[std::string("recall_k")];

  // hold data
  T        *query = nullptr;
  unsigned *gt_ids = nullptr;
  uint32_t *gt_tags = nullptr;
  float    *gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;

  // const std::string temp = "/mnt/t-adisin/sift_query.bin";
  std::cout << "Loading query : " << ::query_file << std::endl;
  // load query + truthset
  diskann::load_aligned_bin<T>(::query_file, query, query_num, query_dim,
                               query_aligned_dim);
  std::cout << "Loaded query : " << ::query_file << std::endl;
  diskann::load_truthset(::truthset_file, gt_ids, gt_dists, gt_num, gt_dim,
                         &gt_tags);
  std::cout << "Loaded gt" << std::endl;
  if (gt_num != query_num) {
    std::cout << "Error. Mismatch in number of queries and ground truth data"
              << std::endl;
  }

  if (print_stats) {
    std::string recall_string = "SS-Recall@" + std::to_string(recall_at);
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
              << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
              << "99 Latency" << std::setw(12) << "99.9 Latency"
              << std::setw(12) << recall_string << std::setw(12)
              << "Mean disk IOs" << std::endl;

    std::cout

        << "==============================================================="
           "==============="
        << std::endl;
  } else {
    std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
              << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
              << "99 Latency" << std::setw(12) << "99.9 Latency"
              << std::setw(12) << recall_string << std::setw(12)
              << "Mean disk IOs" << std::endl;
    std::cout
        << "==============================================================="
           "==============="
        << std::endl;
  }

  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint32_t> query_result_tags;
  std::vector<float>    query_result_dists;
  query_result_ids.resize(recall_at * query_num);
  query_result_dists.resize(recall_at * query_num);
  query_result_tags.resize(recall_at * query_num);
  std::vector<uint32_t> query_result_ids_32(recall_at * query_num);
  for (size_t test_id = 0; test_id < ::Lvec.size(); test_id++) {
    diskann::QueryStats *stats = new diskann::QueryStats[query_num];
    uint32_t             L = Lvec[test_id];
    std::vector<double>  latency_stats(query_num, 0);
    auto                 s = std::chrono::high_resolution_clock::now();
    std::cout << "test_0" << std::endl;
#pragma omp parallel for num_threads(get_search_threads())
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      merge_insert.search_sync(query + (i * query_aligned_dim), recall_at, L,
                               (query_result_tags.data() + (i * recall_at)),
                               query_result_dists.data() + (i * recall_at),
                               stats + i);
      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000;
      //      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::cout << "test_1" << std::endl;
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) (((double) query_num) / diff.count());
    // compute mean recall, IOs
    float mean_recall = 0.0f;
    unsigned *gt_eval_ids = (gt_tags != nullptr) ? gt_tags : gt_ids;
    mean_recall = compute_truthset_recall(
        query_result_tags.data(), (unsigned) recall_at, gt_eval_ids, gt_dists,
        (unsigned) gt_dim, (unsigned) query_num);
    //    mean_recall /= (float) query_num;
    float mean_ios = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.n_ios; });
    std::sort(latency_stats.begin(), latency_stats.end());
    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
              << ((float) std::accumulate(latency_stats.begin(),
                                          latency_stats.end(), 0)) /
                     (float) query_num
              << std::setw(12)
              << (float) latency_stats[(_u64) (0.90 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64) (0.95 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64) (0.99 * ((double) query_num))]
              << std::setw(12)
              << (float) latency_stats[(_u64) (0.999 * ((double) query_num))]
              << std::setw(12) << mean_recall << std::setw(12) << mean_ios
              << std::endl;
    delete[] stats;
  }
  diskann::aligned_free(query);
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

template<typename T, typename TagT = uint32_t>
void insertion_kernel(diskann::MergeInsert<T> &merge_insert,
                      std::string mem_pts_file, std::string mem_tags_file) {
  if (::_insertions_done.load()) {
    std::cout << "Insertions_done is true at the beginning of insertion kernel"
              << std::endl;
    exit(-1);
  }
  T     *data_insert = nullptr;
  size_t npts, ndim, aligned_dim;
  diskann::load_aligned_bin<T>(mem_pts_file, data_insert, npts, ndim,
                               aligned_dim);
  size_t tag_num, tag_dim;
  TagT  *tag_data;
  diskann::load_bin<TagT>(mem_tags_file, tag_data, tag_num, tag_dim);
  if (tag_num != npts) {
    std::cout << "In insertion_kernel(), number of tags loaded is not equal to "
                 "number of points loaded. Exiting....."
              << std::endl;
    exit(-1);
  }
  _s64                i;
  std::vector<double> insert_latencies(npts, 0);
  diskann::Timer      timer;
  if (npts == 0) {
    std::cout << "Mem index insertion skipped: no points to insert" << std::endl;
    ::_insertions_done.store(true);
    delete[] data_insert;
    delete[] tag_data;
    return;
  }
#pragma omp parallel for num_threads(get_insert_threads())
  for (i = 0; i < (_s64) npts; i++) {
    diskann::Timer insert_timer;
    if (merge_insert.insert(data_insert + i * aligned_dim, tag_data[i]) == 0) {
      insert_latencies[i] = ((double) insert_timer.elapsed());
    } else {
      std::cout << "Point " << i << "could not be inserted." << std::endl;
    }
    if ((i % 1000000 == 0) && (i > 0))
      std::cout << "Inserted another 1M points" << std::endl;
  }
  std::cout << "Mem index insertion time : " << timer.elapsed() / 1000 << " ms"
            << std::endl
            << "10th percentile insertion time : "
            << insert_latencies[(size_t) (0.10 * ((double) npts))]
            << " microsec" << std::endl
            << "50th percentile insertion time : "
            << insert_latencies[(size_t) (0.5 * ((double) npts))] << " microsec"
            << "90th percentile insertion time : "
            << insert_latencies[(size_t) (0.90 * ((double) npts))]
            << " microsec" << std::endl;
  ::_insertions_done.store(true);
  delete[] data_insert;
  delete[] tag_data;
}
template<typename T, typename TagT = uint32_t>
void deletion_kernel(diskann::MergeInsert<T> &merge_insert,
                     tsl::robin_set<uint32_t> del_tags) {
  if (::_del_done.load()) {
    std::cout << "_del_done is already true" << std::endl;
    exit(-1);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  diskann::Timer timer;
  for (auto iter : del_tags) {
    merge_insert.lazy_delete(iter);
  }
  std::cout << "Deletion time : " << timer.elapsed() / 1000 << " ms"
            << std::endl;
  ::_del_done.store(true);
}

template<typename T>
void merge_kernel(diskann::MergeInsert<T> &merge_insert) {
  diskann::Timer timer;
  merge_insert.final_merge(get_merge_id_map());
  std::cout << "Merge time : " << timer.elapsed() / 1000 << " ms"
            << std::endl;
}

template<typename T, typename TagT = uint32_t>
void run_iter(diskann::MergeInsert<T>  &merge_insert,
              const std::string        &mem_prefix,
              tsl::robin_set<uint32_t> &active_set,
              tsl::robin_set<uint32_t> &deletable_set,
              tsl::robin_set<uint32_t> &inactive_set) {
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_tags_file = mem_prefix + ".tags_orig";
  std::this_thread::sleep_for(std::chrono::seconds(10));

  ::merge_future =
      std::async(std::launch::async, merge_kernel<T>, std::ref(merge_insert));

  while (!(::_insertions_done.load() && ::_del_done.load())) {
    std::cout << "Search at " << ::global_timer.elapsed() / 1000000
              << " seconds " << std::endl;
    search_kernel<T>(merge_insert, active_set);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }

  if (::_insertions_done.load() && ::_del_done.load()) {
    ::_insertions_done.store(false);
    ::_del_done.store(false);

    std::cout << "Searching all indices" << std::endl;
    std::cout << "Search at " << ::global_timer.elapsed() / 1000000
              << " seconds " << std::endl;
    search_kernel<T>(merge_insert, active_set, true);

    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    tsl::robin_set<uint32_t> deleted_tags;
    seed_iter<T, TagT>(active_set, deletable_set, inactive_set, mem_pts_file,
                       mem_tags_file, deleted_tags);
    ::delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>,
                                 std::ref(merge_insert), deleted_tags);
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(merge_insert), mem_pts_file, mem_tags_file);
  }

  std::future_status merge_status;
  do {
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
    if (merge_status == std::future_status::timeout)
      std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                << " seconds : merge_status: timeout" << std::endl;
    if (merge_status == std::future_status::deferred)
      std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                << " seconds : merge_status: deferred" << std::endl;
    else
      std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                << " seconds : merge_status: ready" << std::endl;
    search_kernel<T>(merge_insert, active_set);

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  } while ((merge_status != std::future_status::ready));
  if (::merge_future.valid()) {
    ::merge_future.get();
  }
}

template<typename T, typename TagT = uint32_t>
void run_single_iter(diskann::MergeInsert<T>  &merge_insert,
                     const std::string        &base_prefix,
                     const std::string        &merge_prefix,
                     const std::string        &mem_prefix,
                     tsl::robin_set<uint32_t> &active_set,
                     tsl::robin_set<uint32_t> &deletable_set,
                     tsl::robin_set<uint32_t> &inactive_set,
                     diskann::Distance<T>     *dist_cmp) {
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".data_orig";
  std::string mem_tags_file = mem_prefix + ".tags_orig";
  if (::_insertions_done.load() && ::_del_done.load()) {
    ::_insertions_done.store(false);
    ::_del_done.store(false);

    /*    std::cout << "Searching all indices" << std::endl;
        std::cout << "Search at " << ::global_timer.elapsed() / 1000000
                  << " seconds " << std::endl;
        search_kernel<T>(merge_insert, active_set, true);
        */
    std::cout << "ITER: Seeding iteration"
              << "\n";
    // seed the iteration
    tsl::robin_set<uint32_t> deleted_tags;
    seed_iter<T, TagT>(active_set, deletable_set, inactive_set, mem_pts_file,
                       mem_tags_file, deleted_tags);
    ::delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>,
                                 std::ref(merge_insert), deleted_tags);
    ::insert_future =
        std::async(std::launch::async, insertion_kernel<T>,
                   std::ref(merge_insert), mem_pts_file, mem_tags_file);
  }
  std::future_status insert_status, delete_status;
  const auto poll_sleep = get_update_poll_sleep();
  do {
    insert_status = ::insert_future.wait_for(std::chrono::milliseconds(1));
    delete_status = ::delete_future.wait_for(std::chrono::milliseconds(1));
    std::this_thread::sleep_for(poll_sleep);
  } while ((insert_status != std::future_status::ready) ||
           (delete_status != std::future_status::ready));
  if (::insert_future.valid()) {
    ::insert_future.get();
  }
  if (::delete_future.valid()) {
    ::delete_future.get();
  }

  ::merge_future =
      std::async(std::launch::async, merge_kernel<T>, std::ref(merge_insert));

  std::future_status merge_status;
  do {
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
  } while ((merge_status != std::future_status::ready));
  if (::merge_future.valid()) {
    ::merge_future.get();
  }
}

template<typename T, typename TagT = uint32_t>
void run_all_iters(std::string base_prefix, std::string merge_prefix,
                   const std::string mem_prefix, const std::string data_file,
                   const std::string     active_tags_file,
                   diskann::Distance<T> *dist_cmp) {
  ::all_points_file = data_file;
  // load all data points
  uint64_t npts = 0, ndims = 0;
  diskann::get_bin_metadata(data_file, npts, ndims);
  std::cout << "Loaded base bin" << std::endl;
  params[std::string("ndims")] = (uint32_t) ndims;

  uint32_t n_iters = params["n_iters"];
  // load active tags
  tsl::robin_set<uint32_t> active_tags;
  TagT                    *tag_data;
  size_t                   tag_num, tag_dim;
  if (::save_index_as_one_file) {
    uint64_t *metadata;
    size_t    nr, nc;
    diskann::load_bin<uint64_t>(active_tags_file, metadata, nr, nc);
    std::cout << nr << " " << nc << std::endl;
    diskann::load_bin<TagT>(active_tags_file, tag_data, tag_num, tag_dim,
                            metadata[7]);
    std::cout << tag_num << " " << tag_dim << std::endl;
  } else {
    diskann::load_bin<TagT>(active_tags_file, tag_data, tag_num, tag_dim);
  }

  size_t tags_loaded = 0;
  size_t del_tags_found = 0;
  active_tags.reserve(tag_num);
  for (size_t i = 0; i < tag_num; i++) {
    if (tag_data[i] != std::numeric_limits<uint32_t>::max()) {
      active_tags.insert(tag_data[i]);
      tags_loaded++;
    } else {
      if (del_tags_found < 5)
        std::cout << "Driver file found invalid tag in active tag file : "
                  << tag_data[i] << std::endl;
      del_tags_found++;
    }
  }
  std::cout << "Loaded " << tags_loaded << " tags" << std::endl;
  delete[] tag_data;
  std::cout << del_tags_found
            << " deleted/invalid tags found in active tags file" << std::endl;
  // read medoid ID from base_prefix
  std::ifstream disk_reader(base_prefix + "_disk.index", std::ios::binary);
  disk_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
  disk_reader.seekg(2 * sizeof(uint64_t), std::ios::cur);
  uint64_t medoid = std::numeric_limits<uint64_t>::max();
  disk_reader.read((char *) &medoid, sizeof(uint64_t));
  std::cout << "Detected medoid = " << medoid
            << " ==> excluding from active recall bookkeeping.\n";
  ::medoid_id = (uint32_t) medoid;

  // generate inactive tags
  tsl::robin_set<uint32_t> inactive_tags;
  inactive_tags.reserve(npts - tag_num);
  for (uint32_t i = 0; i < npts; i++) {
    auto iter = active_tags.find(i);
    if (iter == active_tags.end()) {
      inactive_tags.insert(i);
    }
  }
  std::cout << "Inactive tags : " << inactive_tags.size() << std::endl;
  tsl::robin_set<uint32_t> deletable_tags = active_tags;
  // remove medoid from active_set
  active_tags.erase(::medoid_id);

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

  const std::string             working_folder = ::TMP_FOLDER;
  diskann::Metric               metric = diskann::Metric::L2;
  diskann::MergeInsert<T, TagT> merge_insert(
      paras, ndims, mem_prefix, base_prefix, merge_prefix, dist_cmp, metric,
      ::save_index_as_one_file, working_folder);
  const uint32_t merge_id_map = get_merge_id_map();
  if (merge_id_map == 2 && !should_output_merged_index_to_merge_prefix()) {
    merge_insert._disk_index_prefix_out = base_prefix;
  }
  const bool one_sided_workload =
      params["insert_count"] == 0 || params["delete_count"] == 0;
  for (size_t i = 0; i < n_iters; i++) {
    std::cout << "ITER : " << i << std::endl;
    if (one_sided_workload) {
      run_single_iter<T>(merge_insert, base_prefix, merge_prefix, mem_prefix,
                         active_tags, deletable_tags, inactive_tags, dist_cmp);
    } else {
      run_iter<T>(merge_insert, mem_prefix, active_tags, deletable_tags,
                  inactive_tags);
    }
  }
  /*
  std::cout << "Done running all iterations, now merging any leftover points."
            << std::endl;
  std::future_status merge_status, insert_status, delete_status;
  do {
    merge_status = ::merge_future.wait_for(std::chrono::milliseconds(1));
    insert_status = ::insert_future.wait_for(std::chrono::milliseconds(1));
    delete_status = ::delete_future.wait_for(std::chrono::milliseconds(1));

    //    search_kernel<T>(merge_insert, active_tags,
    //    false);
  } while ((merge_status != std::future_status::ready) ||
           (insert_status != std::future_status::ready) ||
           (delete_status != std::future_status::ready));
  merge_kernel(merge_insert);
  */
  //  search_kernel<T, TagT>(merge_insert, active_tags,
  //  true);
}

int main(int argc, char **argv) {
  std::cout << "Entering main()" << std::endl;
  if (argc < 20) {
    std::cout << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <WORKING_FOLDER> <base_prefix> "
                 "<merge_prefix> <mem_prefix> <L_mem> <alpha_mem> <L_disk> "
                 "<alpha_disk> "
              << " <full_data_bin> <single_file[0/1]> <query_bin> <truthset>"
              << " <n_iters> <total_insert_count> <total_delete_count> <range> "
                 "<recall_k> "
                 "<search_L1> <search_L2> <search_L3> ...."
              << "\n WARNING: Other parameters set inside CPP source."
              << std::endl;
    exit(-1);
  } else {
    std::cout << "This driver file only works with uint32 type tags"
              << std::endl;
  }
  std::cout.setf(std::ios::unitbuf);

  int         arg_no = 1;
  std::string index_type = argv[arg_no++];
  TMP_FOLDER = argv[arg_no++];
  std::string base_prefix(argv[arg_no++]);
  std::string merge_prefix(argv[arg_no++]);
  std::string mem_prefix(argv[arg_no++]);
  unsigned    L_mem = (unsigned) atoi(argv[arg_no++]);
  float       alpha_mem = (float) atof(argv[arg_no++]);
  unsigned    L_disk = (unsigned) atoi(argv[arg_no++]);
  float       alpha_disk = (float) atof(argv[arg_no++]);
  std::string data_bin(argv[arg_no++]);
  int         single_file = atoi(argv[arg_no++]);
  std::string query_path(argv[arg_no++]);
  std::string gt_file(argv[arg_no++]);
  int         n_iters = atoi(argv[arg_no++]);
  uint32_t    insert_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    delete_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    range = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    recall_k = (uint32_t) atoi(argv[arg_no++]);

  for (int ctr = arg_no; ctr < argc; ctr++) {
    _u32 curL = std::atoi(argv[ctr]);
    if (curL >= recall_k)
      ::Lvec.push_back(curL);
  }

  std::cout << "Assigning parameters" << std::endl;
  params[std::string("n_iters")] = n_iters;
  params[std::string("insert_count")] = insert_count;
  params[std::string("delete_count")] = delete_count;
  params[std::string("range")] = range;
  params[std::string("recall_k")] = recall_k;

  // Defaults mirror the SIFT dispatcher while keeping env overrides available.
  params[std::string("disk_search_node_cache_count")] = 100;
  params[std::string("disk_search_nthreads")] = env_u32("QUERY_THREADS", 16);
  params[std::string("beam_width")] = env_u32("BEAMWIDTH", 1);
  params[std::string("mem_l_index")] = L_mem;
  mem_alpha = alpha_mem;
  merge_alpha = alpha_disk;
  params[std::string("merge_maxc")] =
      env_u32("C", static_cast<uint32_t>(range * 2.5));
  params[std::string("merge_l_index")] = L_disk;

  ::query_file = ::query_file + query_path;
  ::truthset_file = gt_file;
  maybe_load_scripted_ids("GREATOR_INSERT_IDS_FILE", ::scripted_insert_ids,
                          "insert");
  maybe_load_scripted_ids("GREATOR_DELETE_IDS_FILE", ::scripted_delete_ids,
                          "delete");
  if (single_file == 1)
    ::save_index_as_one_file = true;
  else
    ::save_index_as_one_file = false;

  std::string active_tags_filename;
  if (single_file)
    active_tags_filename = base_prefix + "_disk.index";
  else
    active_tags_filename = base_prefix + "_disk.index.tags";

  std::cout << "Calling run_all_iters()" << std::endl;
  if (index_type == std::string("float")) {
    diskann::DistanceL2 dist_cmp;
    run_all_iters<float>(base_prefix, merge_prefix, mem_prefix, data_bin,
                         active_tags_filename, &dist_cmp);
  } else if (index_type == std::string("uint8")) {
    diskann::DistanceL2UInt8 dist_cmp;
    run_all_iters<uint8_t>(base_prefix, merge_prefix, mem_prefix, data_bin,
                           active_tags_filename, &dist_cmp);
  } else if (index_type == std::string("int8")) {
    diskann::DistanceL2Int8 dist_cmp;
    run_all_iters<int8_t>(base_prefix, merge_prefix, mem_prefix, data_bin,
                          active_tags_filename, &dist_cmp);
  } else {
    std::cout << "Unsupported type : " << index_type << "\n";
  }
  std::cout << "Exiting\n";
  return 0;
}
