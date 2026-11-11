#pragma once
#include <immintrin.h>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <string>
#include <set>
#include <omp.h>

#include "aligned_file_reader.h"
#include "ssd_index_defs.h"
#include "filter/selector.h"
#include "utils/concurrent_queue.h"
#include "utils/lock_table.h"
#include "utils/rw_lock.h"
#include "utils/percentile_stats.h"
#include "nbr/nbr.h"
#include "utils.h"
#include "index.h"

enum SearchMode { BEAM_SEARCH = 0, PAGE_SEARCH = 1, PIPE_SEARCH = 2, CORO_SEARCH = 3 };

namespace pipeann {
  template<typename T, typename TagT = uint32_t>
  class SSDIndex {
   public:
    static constexpr uint32_t kAllocatedID = std::numeric_limits<uint32_t>::max() - 1;

    std::unique_ptr<Index<T, uint32_t>> mem_index_;  // in-memory navigation graph

    // Index metadata (consolidated).
    SSDIndexMetadata<T> meta_;

    // For updates:
    // meta.npoints is the index's initial size (constant between two merges).
    // cur_id is the ID to be allocated (+1 for each insert, starting from meta.npoints)
    // cur_loc is the tail of the index file, which will be greater than cur_id if overprovisioned.
    // - This is because holes may exist in the index for update combining.
    std::atomic<uint64_t> cur_id, cur_loc;

    SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader,
             AbstractNeighbor<T> *nbr = new PQNeighbor<T>(), bool tags = false,
             IndexBuildParameters *parameters = nullptr);

    ~SSDIndex();

    // Use node_from_page() to create instances for DiskNode.
    // Params: location (id2loc first if using ID), the page-aligned buffer.
    // The offset is calculated in the constructor.
    inline DiskNode<T> node_from_page(char *page_buf, uint32_t loc) {
      return DiskNode<T>(page_buf, loc, meta_);
    }

    // Size of the data region in a DiskNode.
    inline uint64_t node_label_size() const {
      return meta_.label_size;
    }

    // Unaligned offset to location.
    inline uint64_t u_loc_offset(uint64_t loc) {
      return loc * meta_.max_node_len;  // compacted store.
    }

    inline uint64_t u_loc_offset_nbr(uint64_t loc) {
      return loc * meta_.max_node_len + meta_.data_dim * sizeof(T);
    }

    // Avoid integer overflow when * SECTOR_LEN.
    inline uint64_t loc_sector_no(uint64_t loc) {
      return 1 + (meta_.nnodes_per_sector > 0 ? loc / meta_.nnodes_per_sector
                                              : loc * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) {
      return meta_.nnodes_per_sector == 0
                 ? (sector_no - 1) / DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN)  // sector_off == 0.
                 : (sector_no - 1) * meta_.nnodes_per_sector + sector_off;
    }

    void init_metadata(const SSDIndexMetadata<T> &meta) {
      meta.print();
      this->meta_ = meta;
      this->cur_id = this->cur_loc = meta.npoints;
      this->aligned_dim = ROUND_UP(meta_.data_dim, 8);
      this->params.R = meta.range;
      this->size_per_io = SECTOR_LEN * (meta_.nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));
      LOG(INFO) << "Size per IO: " << size_per_io;

      // Aligned.
      if (meta_.nnodes_per_sector != 0 && meta_.npoints % meta_.nnodes_per_sector != 0) {
        cur_loc += meta_.nnodes_per_sector - (meta_.npoints % meta_.nnodes_per_sector);
      }
      LOG(INFO) << "Cur location: " << this->cur_loc;

      // Update-related metadata, if not initialized in constructor, initialize here.
      if (params.L == 0) {
        // Experience values.
        LOG(INFO) << "Automatically set the update-related parameters.";
        params.set(meta.range, meta.range + 32, 384, 1.2, 0, true, 4);
        params.print();
      }
    }

    void init_query_buf(QueryBuffer<T> &buf) {
      pipeann::alloc_aligned((void **) &buf.coord_scratch, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      pipeann::alloc_aligned((void **) &buf.sector_scratch, MAX_N_SECTOR_READS * size_per_io, SECTOR_LEN);
      pipeann::alloc_aligned((void **) &buf.nbr_vec_scratch,
                             MAX_N_EDGES * AbstractNeighbor<T>::MAX_BYTES_PER_NBR * sizeof(uint8_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_ctx_scratch, ROUND_UP(nbr_handler->query_ctx_size(), 256), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_dist_scratch, MAX_N_EDGES * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_query_T, this->aligned_dim * sizeof(T), 8 * sizeof(T));

      buf.visited = new tsl::robin_set<uint64_t>(4096);
      buf.page_visited = new tsl::robin_set<unsigned>(4096);

      memset(buf.sector_scratch, 0, MAX_N_SECTOR_READS * SECTOR_LEN);
      memset(buf.coord_scratch, 0, this->aligned_dim * sizeof(T));
      memset(buf.aligned_query_T, 0, this->aligned_dim * sizeof(T));
    }

    QueryBuffer<T> *pop_query_buf(const T *query) {
      QueryBuffer<T> *data = this->thread_data_queue.pop();
      while (data == this->thread_data_queue.null_T) {
        this->thread_data_queue.wait_for_push_notify();
        data = this->thread_data_queue.pop();
      }

      if (likely(query != nullptr)) {
        if (this->metric == Metric::COSINE) {
          // Data has been normalized. Normalize search vector too.
          pipeann::normalize_data(data->aligned_query_T, query, meta_.data_dim);
        } else {
          memcpy(data->aligned_query_T, query, meta_.data_dim * sizeof(T));
        }
      }
      return data;
    }

    void push_query_buf(QueryBuffer<T> *data) {
      this->thread_data_queue.push(data);
      this->thread_data_queue.push_notify_all();
    }

    // Load compressed data, and obtains the handle to the disk-resident index.
    int load(const char *index_prefix, uint32_t num_threads, bool use_page_search = false);

    void load_mem_index(const std::string &mem_index_path);

    // Load disk index to memory index.
    Index<T, TagT> *load_to_mem(const std::string &filename);

    // Search supporting update.
    size_t beam_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       tsl::robin_set<uint32_t> *deleted_nodes = nullptr, bool dyn_search_l = true);

    size_t coro_search(T **queries, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT **res_tags, float **res_dists, const uint64_t beam_width, int N);

    // Read-only search algorithms.
    size_t page_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       AbstractSelector *selector = nullptr, const void *filter_data = nullptr,
                       const uint64_t relaxed_monotonicity_l = 0);

    int insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set = nullptr);

    void reserve_startup_capacity(uint64_t id_capacity, uint64_t loc_capacity, uint64_t tag_capacity);

    // Merge deletes (NOTE: index read-only during merge.)
    void merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                       const std::vector<TagT> &deleted_nodes, const tsl::robin_set<TagT> &deleted_nodes_set,
                       uint32_t nthreads, const uint32_t &n_sampled_nbrs);

    // After merge, reload the index.
    void reload(const char *index_prefix, uint32_t num_threads);

    void write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                               std::vector<TagT> *new_tags = nullptr);

    void copy_index(const std::string &prefix_in, const std::string &prefix_out);

   private:
    // Background insert I/O commit.
    struct BgTask {
      QueryBuffer<T> *thread_data;
      std::vector<IORequest> writes;
      std::vector<uint64_t> pages_to_unlock;
      std::vector<uint64_t> pages_to_deref;
      bool terminate = false;
    };

    using PageArr = std::vector<uint32_t>;

    static constexpr int kBgIOThreads = 1;

    // Derived/runtime metadata not stored in SSDIndexMetadata.
    uint64_t aligned_dim = 0;
    uint64_t size_per_io = 0;

    // File reader and index file path.
    std::shared_ptr<AlignedFileReader> &reader;
    std::string disk_index_file;

    // Neighbor handler.
    AbstractNeighbor<T> *nbr_handler;

    // Distance comparator.
    pipeann::Metric metric;
    std::shared_ptr<Distance<T>> dist_cmp;

    // Update-related parameters.
    IndexBuildParameters params;

    // Thread-specific scratch buffers.
    ConcurrentQueue<QueryBuffer<T> *> thread_data_queue;
    std::vector<QueryBuffer<T> *> thread_data_bufs;  // pre-allocated thread data
    uint64_t max_nthreads;

    // Background I/O threads for insert.
    ConcurrentQueue<BgTask *> bg_tasks = ConcurrentQueue<BgTask *>(nullptr);
    std::thread *bg_io_thread_[kBgIOThreads]{nullptr};

    // Locking tables for concurrency control.
    pipeann::SparseLockTable<uint64_t> page_lock_table;
    pipeann::SparseLockTable<uint64_t> vec_lock_table;
    pipeann::SparseLockTable<uint64_t> page_idx_lock_table;
    pipeann::SparseLockTable<uint64_t> idx_lock_table;
    RWlock merge_lock;  // merge/search lock

    // Page search mode flag.
    bool use_page_search_ = true;

    // ID to location mapping.
    // Concurrency control is done in lock_idx.
    // Only resize should be protected.
    std::vector<uint32_t> id2loc_;
    pipeann::ReaderOptSharedMutex id2loc_resize_mu_;

    // Location to ID mapping.
    // If nnodes_per_sector >= 1, page_layout[i * nnodes_per_sector + j] is the id of the j-th node in the i-th page.
    // ElseIf nnodes_per_sector == 0, page_layout[i] is the id of the i-th node (starting from loc_sector_no(i)).
    std::vector<uint32_t> loc2id_;
    pipeann::ReaderOptSharedMutex loc2id_resize_mu_;
    std::mutex alloc_lock;
    ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);

    // Tag support.
    // If ID == tag, then it is not stored.
    libcuckoo::cuckoohash_map<uint32_t, TagT> tags;
    std::vector<uint32_t> entry_points_;

    // Flags.
    bool load_flag = false;    // already loaded.
    bool enable_tags = false;  // support for tags and dynamic indexing

    void init_buffers(uint64_t nthreads);
    void destroy_buffers();
    void apply_configured_startup_capacity();

    // Load id2loc and loc2id (i.e., page_layout), to support index reordering.
    void load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector = 0,
                          const uint64_t num_points = 0);

    void load_tags(const std::string &tag_file, size_t offset = 0);
    void load_entry_points(const std::string &index_prefix);

    // Direct insert related.
    void do_beam_search(const T *vec, uint32_t mem_L, uint32_t Lsize, const uint32_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
                        T *coord_buf = nullptr, QueryStats *stats = nullptr,
                        tsl::robin_set<uint32_t> *exclude_nodes = nullptr, bool dyn_search_l = true,
                        std::vector<uint64_t> *passthrough_page_ref = nullptr);

    // Background I/O thread function.
    void bg_io_thread();

    int get_vector_by_id(const uint32_t &id, T *vector);

    // ID, loc, page mapping.
    TagT id2tag(uint32_t id);

    uint32_t id2loc(uint32_t id);
    void set_id2loc(uint32_t id, uint32_t loc);
    uint64_t id2page(uint32_t id);

    uint32_t loc2id(uint32_t loc);
    void set_loc2id(uint32_t loc, uint32_t id);
    void erase_loc2id(uint32_t loc);

    PageArr get_page_layout(uint32_t page_no);

    void erase_and_set_loc(const std::vector<uint64_t> &old_locs, const std::vector<uint64_t> &new_locs,
                           const std::vector<uint32_t> &new_ids);
    // Returns <loc, need_read>.
    std::vector<uint64_t> alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                    std::set<uint64_t> &page_need_to_read);
    void verify_id2loc();

    // lock-related.
    void lock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                  const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                    const std::vector<uint32_t> &neighbors);

    std::vector<uint32_t> get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors);

    // Lock the mapping for target/page if use_page_search == false/true.
    std::vector<uint32_t> lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock);

    // Two-level lock, as id2page may change before and after grabbing the lock.
    std::vector<uint32_t> lock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                        const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);
  };
}  // namespace pipeann
