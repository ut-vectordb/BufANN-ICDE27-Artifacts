#include "v2/dynamic_index.h"
#include "linux_aligned_file_reader.h"

#include <index.h>
#include <cstddef>
#include <future>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "global_stats.h"

#define MERGE_ROUND 10
#define MERGE_IO_THRESHOLD 1.2
#define NUM_INSERT_THREADS 10
#define NUM_MERGE_THREADS 20
#define NUM_DELETE_THREADS 1
#define NUM_SEARCH_THREADS 8
#define DeleteQPS 1000

extern std::atomic<long long> counter;
int begin_time = 0;
pipeann::Timer globalTimer;

// acutually also shows disk size
void ShowMemoryStatus(const std::string &filename) {
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;

  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  std::string index_file_name = filename + "_disk.index";
  stat(index_file_name.c_str(), &st);

  std::cout << " memory current time: " << current_time << " RSS : " << rss << " KB " << index_file_name
            << " Index size " << (st.st_size / (1 << 20)) << " MB" << std::endl;
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

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(std::string &truthFilePrefix, int l_start) {
  std::string fileName(truthFilePrefix);
  fileName = fileName + "/gt_" + std::to_string(l_start) + ".bin";
  LOG(INFO) << "Truth file name: " << fileName;
  return fileName;
}

template<typename T>
inline uint64_t save_bin_test(const std::string &filename, T *id, float *dist, size_t npts, size_t ndims,
                              size_t offset = 0) {
  std::ofstream writer;
  open_file_to_write(writer, filename);

  std::cout << "Writing bin: " << filename.c_str() << std::endl;
  writer.seekp(offset, writer.beg);
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  std::cout << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B" << std::endl;

  for (int i = 0; i < npts; i++) {
    for (int j = 0; j < ndims; j++) {
      writer.write((char *) (id + i * ndims + j), sizeof(T));
      writer.write((char *) (dist + i * ndims + j), sizeof(float));
    }
  }
  writer.close();
  std::cout << "Finished writing bin." << std::endl;
  return bytes_written;
}

template<typename T, typename TagT>
void sync_search_kernel(T *query, size_t query_num, size_t query_aligned_dim, const int recall_at, _u64 L,
                        uint32_t beam_width, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::string &truthset_file,
                        bool merged, bool calRecall, double &disk_io) {
  unsigned *gt_ids = NULL;
  float *gt_dists = NULL;
  size_t gt_num, gt_dim;

  if (!file_exists(truthset_file)) {
    calRecall = false;
  }

  if (calRecall) {
    std::cout << "current truthfile: " << truthset_file << std::endl;
    pipeann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
  }

  float *query_result_dists = new float[recall_at * query_num];
  TagT *query_result_tags = new TagT[recall_at * query_num];

  for (_u32 q = 0; q < query_num; q++) {
    for (_u32 r = 0; r < (_u32) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double> latency_stats(query_num, 0);
  pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    auto qs = std::chrono::high_resolution_clock::now();
    // stats[i].n_current_used = 8;
    sync_index.search(query + i * query_aligned_dim, recall_at, 0, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);

    auto qe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    latency_stats[i] = diff.count() * 1000000;
    // std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float qps = (query_num / diff.count());
  float recall = 0;

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
  if (calRecall) {
    recall = pipeann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
    delete[] gt_ids;
  }

  // std::cout << "search current time: " << current_time << std::endl;

  float mean_ios =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

  float io_us =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.io_us; });
  std::sort(latency_stats.begin(), latency_stats.end());
  
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "Beam_width" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat(us)" << std::setw(12)
            << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
            << std::setw(12) << "99.9 Lat" << std::setw(12) << "Disk IOs" << std::setw(12) << "Mean IO(us)" << std::setw(12) << recall_string
            << std::endl;
  std::cout << "==============================================================="
               "==================================================="
            << std::endl;
  std::cout << std::setw(4) << L << std::setw(12) << beam_width << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0f)) / (float) query_num
            << std::setw(12) << (float) latency_stats[(_u64) (0.50 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.90 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.95 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.99 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.999 * ((double) query_num))] << std::setw(12)
            << mean_ios << std::setw(12)
            << io_us << std::setw(12)
            << recall << std::endl;
  disk_io = mean_ios;
  std::cout << "search cost " << diff.count() << "s" << std::endl;

  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] stats;
}

template<typename T, typename TagT>
void merge_kernel(pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::string &save_path) {
#ifdef USE_TOPO_DISK
  sync_index.trigger_deletion(NUM_MERGE_THREADS, 20);
#else
  sync_index.final_merge(NUM_MERGE_THREADS, 20);
#endif
}

template<typename T, typename TagT>
void deletion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &delete_vec,
                     size_t aligned_dim) {
  pipeann::Timer timer;
  size_t npts = delete_vec.size();
  std::vector<double> delete_latencies(npts, 0);
  std::cout << "Begin Delete" << std::endl;
#pragma omp parallel for num_threads(NUM_DELETE_THREADS)
  for (_s64 i = 0; i < (_s64) delete_vec.size(); i++) {
    pipeann::Timer delete_timer;
    pipeann::QueryStats stats;
    sync_index.lazy_delete(delete_vec[i]);
    delete_latencies[i] = ((double) delete_timer.elapsed());
  }
  std::sort(delete_latencies.begin(), delete_latencies.end());
  std::cout << "10p deletion time : " << delete_latencies[(size_t) (0.10 * ((double) npts))] << " ms" << std::endl
            << "50p deletion time : " << delete_latencies[(size_t) (0.5 * ((double) npts))] << " ms" << std::endl
            << "90p deletion time : " << delete_latencies[(size_t) (0.90 * ((double) npts))] << " ms" << std::endl
            << "99p deletion time : " << delete_latencies[(size_t) (0.99 * ((double) npts))] << " ms" << std::endl
            << "99.9p deletion time : " << delete_latencies[(size_t) (0.999 * ((double) npts))] << " ms" << std::endl;
}

template<typename T, typename TagT>
void insertion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &insert_vec,
                      size_t aligned_dim) {
  pipeann::Timer timer;
  size_t npts = insert_vec.size();
  std::vector<double> insert_latencies(npts, 0);
  std::cout << "Begin Insert" << std::endl;
  std::atomic_size_t success(0);
  // insert_vec.resize(1);
  // std::cerr << "counter = " << counter << "\n";
  // sync_index._disk_index->verify_id2loc_topo();

// #ifdef USE_TOPO_DISK
//   sync_index._disk_index->verify_id2loc_topo();
//   sync_index._disk_index->verify_id2loc_coord();
// #endif
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (_s64 i = 0; i < (_s64) insert_vec.size(); i++) {

    pipeann::Timer insert_timer;
    sync_index.insert(data_load + aligned_dim * i, insert_vec[i]);
    success++;
    insert_latencies[i] = ((double) insert_timer.elapsed());

    size_t done = success.fetch_add(1, std::memory_order_relaxed) + 1;

    // if (done % (insert_vec.size() / 1000) == 0) {  // 千分之一时打印
    //     #pragma omp critical
    //     {
    //       ShowMemoryStatus(sync_index._disk_index_prefix_in);
    //       std::cout << "\rProgress: (" << done << "/" << insert_vec.size() << ")" << std::flush;
    //     }
    // }
    // LOG(ERROR) << "insert " << i << " done\n";
    // }
  }
  float time_secs = timer.elapsed() / 1.0e6f;
  std::sort(insert_latencies.begin(), insert_latencies.end());
  std::cout << "$E2E insert cost " << timer.elapsed() / 1e6 << " s.\n";

  std::cout << "Inserted " << insert_vec.size() << " points in " << time_secs << "s" << std::endl;
  std::cout << "10p insertion time : " << insert_latencies[(size_t) (0.10 * ((double) npts))] << " us" << std::endl
            << "50p insertion time : " << insert_latencies[(size_t) (0.5 * ((double) npts))] << " us" << std::endl
            << "90p insertion time : " << insert_latencies[(size_t) (0.90 * ((double) npts))] << " us" << std::endl
            << "99p insertion time : " << insert_latencies[(size_t) (0.99 * ((double) npts))] << " us" << std::endl
            << "99.9p insertion time : " << insert_latencies[(size_t) (0.999 * ((double) npts))] << " us" << std::endl;
  // std::cerr << "counter : " << counter.load() << "\n";
  // std::cerr << "loc(x) : " << sync_index._disk_index->id2loc_topo(x) << "\n";
  // std::cerr << "id(loc(x)) : " << sync_index._disk_index->loc2id_topo(sync_index._disk_index->id2loc_topo(x)) << "\n";
  // if(x != sync_index._disk_index->loc2id_topo(sync_index._disk_index->id2loc_topo(x))){
    //   crash();
    // }
}

template<typename T, typename TagT = uint32_t>
void get_trace(std::string data_bin, uint64_t l_start, uint64_t r_start, uint64_t n, std::vector<TagT> &delete_tags,
               std::vector<TagT> &insert_tags, std::vector<T> &data_load) {
  for (uint64_t i = l_start; i < l_start + n; ++i) {
    delete_tags.push_back(i);
  }

  for (uint64_t i = r_start; i < r_start + n; ++i) {
    insert_tags.push_back(i);
  }
  
  if(0){
    delete_tags.clear();
    for (uint64_t i = r_start; i < r_start + n; ++i) {
      delete_tags.push_back(i);
    }
  }

  // load data, load n vecs from r_start.
  int npts_i32, dim_i32;
  std::ifstream reader(data_bin, std::ios::binary | std::ios::ate);
  reader.seekg(0, reader.beg);
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));

  size_t data_dim = dim_i32;
  data_load.resize(n * data_dim * 2);
  reader.seekg(2 * sizeof(int) + r_start * data_dim * sizeof(T), reader.beg);
  reader.read((char *) data_load.data(), sizeof(T) * n * data_dim);
}

template <typename T, typename TagT = uint32_t>
void get_trace2(tsl::robin_set<uint32_t> &inactive_set,
               std::vector<TagT> &delete_tags, std::vector<TagT> &insert_tags,
               std::string &file_name, T *&data_load, std::string &data_path,
               size_t &dim, size_t &rounded_dim) {
  /**Loading delete_tags & insert_tags & inactive_set*/
  std::cout << "Loading " << file_name << std::endl;
  std::ifstream base_reader;
  int update_size;
  base_reader.open(file_name, std::ios::binary | std::ios::ate);
  base_reader.seekg(0, std::ios::beg);
  base_reader.read((char *)&update_size, sizeof(int));
  std::cout << "update_size:" << update_size << std::endl;
  delete_tags.clear();
  delete_tags.resize(update_size);
  insert_tags.clear();
  insert_tags.resize(update_size);
  base_reader.seekg(sizeof(int), std::ios::beg);
  base_reader.read((char *)delete_tags.data(), sizeof(int) * update_size);
  base_reader.seekg((update_size + 1) * sizeof(int), std::ios::beg);
  base_reader.read((char *)insert_tags.data(), sizeof(int) * update_size);
  // delete[] data_load;
  // if(1)
  // {
    // update_size = 1;
    // delete_tags.clear();
    // insert_tags.clear();
  // }
  std::cout << "Reading trace vectors from bin file " << data_path << "..."
                << std::flush;
  std::ifstream reader(data_path, std::ios::binary | std::ios::ate);
  int npts_i32, dim_i32;
  reader.seekg(0);
  reader.read((char *)&npts_i32, sizeof(int));
  reader.read((char *)&dim_i32, sizeof(int));
  dim = (unsigned)dim_i32;
  rounded_dim = ROUND_UP(dim, 8);
  size_t allocSize = update_size * rounded_dim * sizeof(T);

  pipeann::alloc_aligned(((void **)&data_load), allocSize, 8 * sizeof(T));

  for (int i = 0; i < update_size; i++) {
    reader.seekg(2 * sizeof(int) + dim * sizeof(T) * insert_tags[i],
                 reader.beg);
    reader.read((char *)(data_load + i * rounded_dim), dim * sizeof(T));
    memset(data_load + i * rounded_dim + dim, 0,
           (rounded_dim - dim) * sizeof(T));
  }

  std::cout << "Loaded full data for driver: (" << update_size << "," << dim
            << ") vectors." << std::endl;
}

template<typename T, typename TagT>
void update(const std::string &data_bin, const unsigned L_disk, int step, float batch_size, const unsigned nodes_to_cache,
            std::string &save_path, const std::string &query_file, std::string &truthset_file, const int recall_at,
            std::vector<_u64> Lsearch, const unsigned beam_width, pipeann::Distance<T> *dist_cmp, int search_mode, int strategy) {
  pipeann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", 32);
  paras.Set<float>("alpha_disk", 1.2);
  paras.Set<unsigned>("C", 160);//384
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("nodes_to_cache", nodes_to_cache);
  paras.Set<unsigned>("num_threads", 32);  // more for less contention of thread data.
  // std::vector<T> data_load;
  size_t dim{}, aligned_dim{};

  std::cout << std::fixed << std::setprecision(4);
  pipeann::Timer timer;

  std::cout << "Loading queries " << std::endl;
  T *query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  pipeann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

  dim = query_dim;
  aligned_dim = query_aligned_dim;
  pipeann::Metric metric = pipeann::Metric::L2;
  std::cerr << "save_path: " << save_path << std::endl;
  pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, save_path, save_path, dist_cmp, metric, search_mode, 0, strategy);

  std::cout << "Searching before inserts: " << std::endl;

  uint64_t res = 0;

  std::string currentFileName = GetTruthFileName(truthset_file, res);
  // std::string currentFileName = truthset_file;
  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus(sync_index._disk_index_prefix_in);

  query_num = 1000;
  std::vector<double> ref_diskio;
  for (uint64_t j = 0; j < Lsearch.size(); ++j) {
    // std::cerr << "L = " << Lsearch[j] << "\n";
    double diskio = 0;
    sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch[j], beam_width, sync_index,
                       currentFileName, false, true, diskio);
    ref_diskio.push_back(diskio);
  }
  
  int batch = step;
  int inMemorySize = 0;
  std::future<void> merge_future;
  uint64_t index_npts = sync_index._disk_index->num_points;
  uint64_t vecs_per_step = batch_size;
  std::vector<T> data_load;
  std::cerr << "batch num: " << batch << std::endl;
  std::cerr << "batch size: " << batch_size << std::endl;
  std::cerr << "vecs_per_step: " << vecs_per_step << std::endl;
  if(step == -1){
    vecs_per_step = index_npts * batch_size;
    batch = index_npts * 0.1 / vecs_per_step;
    std::cerr << "batch num: " << batch << std::endl;
    std::cerr << "batch size: " << batch_size << std::endl;
    std::cerr << "vecs_per_step: " << vecs_per_step << std::endl;
  }
  
  for (int i = 0; i < batch; i++) {
    ShowPeakMemoryStatus();

    std::cout << "Batch: " << i << " Total Batch : " << batch << std::endl;
    timer.reset();

    std::vector<unsigned> insert_vec;
    std::vector<unsigned> delete_vec;

    /**Prepare for update*/
    uint64_t st = vecs_per_step * i;
    get_trace<T, TagT>(data_bin, st, st + index_npts, vecs_per_step, delete_vec, insert_vec, data_load);

    if(gs != nullptr){
      gs->clear();
    }
    std::future<void> delete_future = std::async(std::launch::async, [&] {
      deletion_kernel<T, TagT>(data_load.data(), std::ref(sync_index), std::ref(delete_vec), aligned_dim);
    });
    std::future<void> insert_future = std::async(std::launch::async, [&] {
      insertion_kernel<T, TagT>(data_load.data(), std::ref(sync_index), std::ref(insert_vec), aligned_dim);
    });
  
    // insertion_kernel<T, TagT>(data_load.data(), std::ref(sync_index), std::ref(insert_vec), aligned_dim);
    
    int total_queries = 0;
    std::future_status insert_status;
    std::future_status delete_status;
    do {
      insert_status = insert_future.wait_for(std::chrono::milliseconds(200));
      delete_status = delete_future.wait_for(std::chrono::milliseconds(200));
      if (insert_status == std::future_status::deferred || delete_status == std::future_status::deferred) {
        std::cout << "deferred\n";
      } else if (insert_status == std::future_status::timeout || delete_status == std::future_status::timeout) {
        ShowMemoryStatus(sync_index._disk_index_prefix_in);
        double dummy;
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch[0], beam_width, sync_index,
                           currentFileName, false, false, dummy);
        total_queries += query_num;
        std::cout << "Queries processed: " << total_queries << std::endl;
      }
      if (insert_status == std::future_status::ready) {
        std::cout << "Insertions complete!\n";
      }
      if (delete_status == std::future_status::ready) {
        std::cout << "Deletions complete!\n";
      }
    } while (insert_status != std::future_status::ready || delete_status != std::future_status::ready);
    // deletion_kernel<T, TagT>(data_load.data(), std::ref(sync_index), std::ref(delete_vec), aligned_dim);
    // merge_kernel<T, TagT>(std::ref(sync_index), std::ref(save_path));

    double e2e_time = ((double)timer.elapsed()) / (1000000.0);
    std::cerr << "Batch #" << i << " use " << e2e_time << " s.\n";

    ShowMemoryStatus(sync_index._disk_index_prefix_in);

    res += vecs_per_step;
    currentFileName = GetTruthFileName(truthset_file, res);
    // currentFileName = truthset_file;
    if(gs != nullptr){
      std::cerr << "insert io1: " << gs->insert_io1 / NUM_INSERT_THREADS << " " << gs->insert_io1 / NUM_INSERT_THREADS / 1e6 << " s\n";
      std::cerr << "insert io2: " << gs->insert_io2 / NUM_INSERT_THREADS << " " << gs->insert_io2 / NUM_INSERT_THREADS / 1e6 << " s\n";
      std::cerr << "insert io3: " << gs->insert_io3 / NUM_INSERT_THREADS << " " << gs->insert_io3 / NUM_INSERT_THREADS / 1e6 << " s\n";
      std::cerr << "insert io4: " << gs->insert_io4 / NUM_INSERT_THREADS << " " << gs->insert_io4 / NUM_INSERT_THREADS / 1e6 << " s\n";
      std::cerr << "insert io5: " << gs->insert_io5 / NUM_INSERT_THREADS << " " << gs->insert_io5 / NUM_INSERT_THREADS / 1e6 << " s\n";
      std::cerr << "insert io6: " << gs->insert_io6 / NUM_INSERT_THREADS << " " << gs->insert_io6 / NUM_INSERT_THREADS / 1e6 << " s\n";
      std::cerr << "insert io7: " << gs->insert_io7 / NUM_INSERT_THREADS << " " << gs->insert_io7 / NUM_INSERT_THREADS / 1e6 << " s\n";
      std::cerr << "insert io8: " << gs->insert_io8 / NUM_INSERT_THREADS << " " << gs->insert_io8 / NUM_INSERT_THREADS / 1e6 << " s\n";
      
      std::cerr << "-------------------------------------------------------------\n";
      std::cerr << "insert ios1: " << gs->insert_ios1  << " " << gs->insert_ios1  / 1024 << " KB\n";
      std::cerr << "insert ios2: " << gs->insert_ios2  << " " << gs->insert_ios2  / 1024 << " KB\n";
      std::cerr << "insert ios3: " << gs->insert_ios3  << " " << gs->insert_ios3  / 1024 << " KB\n";
      std::cerr << "insert ios4: " << gs->insert_ios4  << " " << gs->insert_ios4  / 1024 << " KB\n";
      std::cerr << "insert ios5: " << gs->insert_ios5  << " " << gs->insert_ios5  / 1024 << " KB\n";
      std::cerr << "insert ios6: " << gs->insert_ios6  << " " << gs->insert_ios6  / 1024 << " KB\n";
      std::cerr << "insert ios7: " << gs->insert_ios7  << " " << gs->insert_ios7  / 1024 << " KB\n";
      std::cerr << "insert ios8: " << gs->insert_ios8  << " " << gs->insert_ios8  / 1024 << " KB\n";
      std::cerr << "-------------------------------------------------------------\n";
      std::cerr << "update_effective_topo_ios: " << gs->update_effective_topo_ios  << " B\n";
      std::cerr << "update_effective_coord_ios: " << gs->update_effective_coord_ios  << " B\n";
      
      // LOG(INFO) << "consolidate_deletes: " << gs->consolidate_deletes / 20 / 1e6 << "s";
      // LOG(INFO) << "occlude_list_pq: " << gs->occlude_list_pq / 20 / 1e6 << "s";
      // LOG(INFO) << "delete_scan1: " << gs->delete_scan1 / 20 / 1e6 << "s";
      std::cerr << "$E2E UPDATE IO cost " << (gs->insert_io / NUM_INSERT_THREADS + gs->delete_io) / 1e6 << " s\n";
      std::cerr << "$E2E UPDATE IOS count " << gs->update_ios << " B\n";
    }
    std::vector<double> disk_ios;
    // disk_ios.reserve(10);
    // for (uint64_t j = 0; j < 1; ++j) {
    //   double diskio = 0;
    //   sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch[j], beam_width, sync_index,
    //                      currentFileName, false, true, diskio);
    //   disk_ios.push_back(diskio);
    // }
    if (i == batch - 1) {
      std::cout << "Done" << std::endl;
      exit(0);
    } else if (i % MERGE_ROUND == MERGE_ROUND - 1) {
      std::cout << "Begin Merge" << std::endl;
      merge_future = std::async(std::launch::async, merge_kernel<T, TagT>, std::ref(sync_index), std::ref(save_path));
      // std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "Sending Merge" << std::endl;
      inMemorySize = 0;
      std::future_status merge_status;
      do {
        // merge_status = merge_future.wait_for(std::chrono::seconds(1));
        merge_status = merge_future.wait_for(std::chrono::milliseconds(1000));
        double dummy = 0;
        ShowMemoryStatus(sync_index._disk_index_prefix_in);
        sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch[0], beam_width, sync_index,
                           currentFileName, false, false, dummy);
      } while (merge_status != std::future_status::ready);

      // ref_diskio.clear();
      // for (uint64_t j = 0; j < Lsearch.size(); ++j) {
      //   double diskio = 0;
      //   sync_search_kernel(query, query_num, query_aligned_dim, recall_at, Lsearch[j], beam_width, sync_index,
      //                      currentFileName, false, true, diskio);
      //   disk_ios.push_back(diskio);
      // }
      std::cout << "Merge finished " << std::endl;
    }
  }
  
  ShowPeakMemoryStatus();
}

int main(int argc, char **argv) {
  if (argc < 12) {
    std::cout << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <data_bin> <L_disk> "
              << " <indice_path> <query_file> <truthset_prefix> <recall@>"
              << " <#beam_width> <step> <batch_size> <search_mode> <strategy> <Lsearch> <L2>" << std::endl;
    exit(-1);
  }

  int arg_no = 2;
  std::string data_bin = std::string(argv[arg_no++]);
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);
  std::string save_path(argv[arg_no++]);

  std::string query_file(argv[arg_no++]);
  std::string truthset(argv[arg_no++]);
  int recall_at = (int) std::atoi(argv[arg_no++]);
  unsigned beam_width = (unsigned) std::atoi(argv[arg_no++]);
  int step = (int) std::atoi(argv[arg_no++]);
  float batch_size = (float) std::atof(argv[arg_no++]);
  int search_mode = (int) std::atoi(argv[arg_no++]);
  int strategy = (int) std::atoi(argv[arg_no++]);
  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(std::atoi(argv[i]));
  }

  unsigned nodes_to_cache = 0;

  if (std::string(argv[1]) == std::string("int8")) {
    pipeann::DistanceL2Int8 dist_cmp;
    update<int8_t, unsigned>(data_bin, L_disk, step, batch_size, nodes_to_cache, save_path, query_file, truthset, recall_at,
                             Lsearch, beam_width, &dist_cmp, search_mode, strategy);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    pipeann::DistanceL2UInt8 dist_cmp;
    update<uint8_t, unsigned>(data_bin, L_disk, step, batch_size, nodes_to_cache, save_path, query_file, truthset, recall_at,
                              Lsearch, beam_width, &dist_cmp, search_mode, strategy);
  } else if (std::string(argv[1]) == std::string("float")) {
    pipeann::DistanceL2 dist_cmp;
    update<float, unsigned>(data_bin, L_disk, step, batch_size, nodes_to_cache, save_path, query_file, truthset, recall_at, Lsearch,
                            beam_width, &dist_cmp, search_mode, strategy);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
