#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <cstdlib>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils.h"
#include "utils/page_cache.h"
#include "utils/prune_neighbors.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"
#include "../../../src/ann_bench_metrics.h"

namespace {
  IOStats count_io_requests(const std::vector<IORequest> &requests) {
    IOStats stats;
    stats.n_ops = requests.size();
    for (const auto &request : requests) {
      stats.n_bytes += request.len;
    }
    return stats;
  }
}  // namespace

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                        const std::vector<TagT> &deleted_nodes,
                                        const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                        const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    void *ctx = reader->get_ctx();

    while (!bg_tasks.empty()) {
      sleep(5);  // simple way to wait for background IO thread.
    }
    std::string disk_index_out = out_path_prefix + "_disk.index";
    // Note that the index is immutable currently.
    // Step 1: populate neighborhoods, allocate IDs.
    libcuckoo::cuckoohash_map<uint32_t, uint32_t> id_map, rev_id_map;           // old_id -> new_id & new_id -> old_id
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    // Size id_map and rev_id_map for cur_loc (an upper bound on the surviving points) up front. Grown from libcuckoo's
    // default instead, they double ~12 times, and tcmalloc may keep the last round of old tables (~12 GB on sift1b)
    // resident through the PQ shuffle, inflating the merge's peak RSS. PIPEANN_MERGE_PRESIZE_MAPS=0 restores the growth.
    const char *presize_env = std::getenv("PIPEANN_MERGE_PRESIZE_MAPS");
    if (presize_env == nullptr || std::string(presize_env) != "0") {
      id_map.reserve(cur_loc);
      rev_id_map.reserve(cur_loc);
      LOG(INFO) << "PIPEANN_MERGE_PRESIZE_MAPS: reserved id_map and rev_id_map for " << cur_loc.load() << " ids";
    }
    std::atomic<uint64_t> new_npoints = 0;
    Timer delete_timer;

    constexpr uint64_t LOC_PER_POPULATE = 128 * 8;  // small to avoid blocking search threads.
    // process this many locs per iteration
    const uint64_t LOC_PER_MERGE = meta_.nnodes_per_sector > 0 ? ROUND_UP(131072, meta_.nnodes_per_sector) : 131072;
    const uint64_t VEC_IN_WBUF = 2 * LOC_PER_MERGE;  // sliding window buffer holds 2 batches

    // Calculate buffer size based on whether we have large or small nodes
    uint64_t sectors_per_batch = meta_.nnodes_per_sector > 0
                                     ? LOC_PER_MERGE / meta_.nnodes_per_sector
                                     : LOC_PER_MERGE * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN);
    uint64_t buf_size_per_batch = sectors_per_batch * SECTOR_LEN;

    char *rbuf = nullptr, *wbuf = nullptr;
    alloc_aligned((void **) &rbuf, buf_size_per_batch, SECTOR_LEN);
    alloc_aligned((void **) &wbuf, 2 * buf_size_per_batch, SECTOR_LEN);  // sliding window buffer.
    LOG(INFO) << "Cur loc: " << cur_loc.load() << ", cur ID: " << cur_id
              << ", nnodes_per_sector: " << meta_.nnodes_per_sector
              << ", buf_size_per_batch: " << buf_size_per_batch / 1024 / 1024 << "MB";

    // NOTE: PipeANN deliberately *throttles* the merge batch to make bandwidth for concurrent searches.
    // But also, even if I remove this std::min() throttling, it's still not hugely fast. The reader->read() is in a single-threaded loop.
    uint32_t populate_nthreads = std::min(nthreads, 4u);  // restrict the flow.

    for (uint64_t loc_st = 0; loc_st < cur_loc; loc_st += LOC_PER_POPULATE) {
      uint64_t loc_ed = std::min(cur_loc.load(), loc_st + LOC_PER_POPULATE);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, rbuf, 0, 0));
      const IOStats seq_reads = count_io_requests(read_reqs);
      reader->read(read_reqs, ctx, false);
      ann_bench::LogicalIoCounter::account_seq_read(seq_reads.n_ops, seq_reads.n_bytes);

      std::vector<uint32_t> live_ids(loc_ed - loc_st, kInvalidID);

#pragma omp parallel for num_threads(populate_nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        // populate nhood.
        uint32_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 2. not deleted, record for ID allocation.
          live_ids[loc - loc_st] = id;
          continue;
        }

        // 3. deleted, populate nhoods.
        uint64_t loc_sector = loc_sector_no(loc);
        auto page_rbuf = rbuf + (loc_sector - st_sector) * SECTOR_LEN;
        DiskNode<T> node = node_from_page(page_rbuf, loc);
        std::vector<uint32_t> nhood;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
            nhood.push_back(node.nbrs[i]);  // filtered neighborhoods.
          }
        }
        // sample for less space consumption.
        if (nhood.size() > n_sampled_nbrs) {
          // std::shuffle(nhood.begin(), nhood.end(), std::default_random_engine());
          nhood.resize(n_sampled_nbrs);  // nearest.
        }
        deleted_nhoods.insert(id, nhood);
      }

      // Assign compacted IDs in old-location order. The writer below scans old
      // locations and flushes compacted-ID ranges, so this ordering is required.
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint32_t id = live_ids[loc - loc_st];
        if (id == kInvalidID) {
          continue;
        }
        uint64_t new_id = new_npoints.fetch_add(1);
        id_map.insert(id, new_id);
        rev_id_map.insert(new_id, id);
      }
    }
    LOG(INFO) << "Finished populating neighborhoods, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms, new npoints: " << new_npoints.load() << " " << "id_map size: " << id_map.size();
    LOG(INFO) << "Deleted nodes size: " << deleted_nodes.size() << ", deleted_nhoods size: " << deleted_nhoods.size();

    // Step 2: prune neighbors, populate PQ and tags.
    int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);
    uint64_t wb_id = 0;
    std::atomic<uint64_t> n_used_id = 0;
    std::vector<uint8_t> populated_slots(VEC_IN_WBUF, 0);
    auto write_back = [&]() {
      // write one buffer.
      uint64_t buf_id = (wb_id % VEC_IN_WBUF) / LOC_PER_MERGE;
      auto b = wbuf + buf_id * buf_size_per_batch;
      std::vector<IORequest> write_reqs;
      uint64_t current_used = n_used_id.load();
      uint64_t id_delta = std::min(LOC_PER_MERGE, current_used - wb_id);

      for (uint64_t id = wb_id; id < wb_id + id_delta; ++id) {
        if (unlikely(populated_slots[id % VEC_IN_WBUF] == 0)) {
          LOG(ERROR) << "delete_merge write invariant failed: compacted ID " << id
                     << " is being flushed before its node was populated"
                     << ", wb_id: " << wb_id << ", id_delta: " << id_delta
                     << ", n_used_id: " << current_used << ", VEC_IN_WBUF: " << VEC_IN_WBUF;
          std::abort();
        }
      }

      // Calculate write length: from wb_id to wb_id + id_delta - 1
      uint64_t st_sector = loc_sector_no(wb_id);
      uint64_t ed_sector = loc_sector_no(wb_id + id_delta - 1);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t write_len = (ed_sector - st_sector + 1) * SECTOR_LEN;

      write_reqs.push_back(IORequest(st_sector * SECTOR_LEN, write_len, b, 0, 0));
      const IOStats seq_writes = count_io_requests(write_reqs);
      reader->write_fd(fd, write_reqs, ctx);
      ann_bench::LogicalIoCounter::account_seq_write(seq_writes.n_ops, seq_writes.n_bytes);
      for (uint64_t id = wb_id; id < wb_id + id_delta; ++id) {
        populated_slots[id % VEC_IN_WBUF] = 0;
      }
      wb_id += id_delta;
      LOG(INFO) << "Write back " << wb_id << "/" << current_used << " IDs.";
    };

    std::vector<TagT> new_tags(new_npoints);

    for (uint64_t loc_st = 0; loc_st < cur_loc; loc_st += LOC_PER_MERGE) {
      uint64_t loc_ed = std::min(cur_loc.load(), loc_st + LOC_PER_MERGE);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, rbuf, 0, 0));
      const IOStats seq_reads = count_io_requests(read_reqs);
      reader->read(read_reqs, ctx, false);  // read in fd
      ann_bench::LogicalIoCounter::account_seq_read(seq_reads.n_ops, seq_reads.n_bytes);

#pragma omp parallel for num_threads(nthreads)
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc2id(loc);
        if (id == kInvalidID) {
          continue;
        }

        uint64_t tag = id2tag(id);
        if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted.
          continue;
        }

        uint64_t loc_sector = loc_sector_no(loc);
        auto page_rbuf = rbuf + (loc_sector - st_sector) * SECTOR_LEN;
        DiskNode<T> node = node_from_page(page_rbuf, loc);
        // prune neighbors.
        std::unordered_set<uint32_t> nhood_set;
        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          uint32_t nbr_tag = id2tag(node.nbrs[i]);
          if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
            // deleted, insert neighbors.
            const auto &nhoods = deleted_nhoods.find(node.nbrs[i]);
            nhood_set.insert(nhoods.begin(), nhoods.end());
          } else {
            nhood_set.insert(node.nbrs[i]);
            // LOG(INFO) << id << " insert " << node.nbrs[i];
          }
        }
        nhood_set.erase(id);  // remove self.
        std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());

        if (nhood.size() > this->params.R) {
          std::vector<float> dists(nhood.size(), 0.0f);
          std::vector<Neighbor> pool(nhood.size());
          // Use dynamic buffer instead of pre-initialized buffer to save space.
          uint8_t *pq_buf = nullptr;
          pipeann::alloc_aligned((void **) &pq_buf, nhood.size() * AbstractNeighbor<T>::MAX_BYTES_PER_NBR, 256);
          nbr_handler->compute_dists(id, nhood.data(), nhood.size(), dists.data(), pq_buf);

          for (uint32_t k = 0; k < nhood.size(); k++) {
            pool[k].id = nhood[k];
            pool[k].distance = dists[k];
          }
          auto nbr = this->nbr_handler;
          pipeann::prune_neighbors(pool, nhood, params, metric, [nbr, pq_buf](uint32_t a, uint32_t b) {
            float dist;
            nbr->compute_dists(a, &b, 1, &dist, pq_buf);
            return dist;
          });
          pipeann::aligned_free(pq_buf);
        }

        // map to new IDs.
        for (auto &nbr : nhood) {
          nbr = id_map.find(nbr);
          if (unlikely(nbr > new_npoints)) {
            LOG(ERROR) << "Invalid neighbor ID: " << nbr << ", new_npoints: " << new_npoints;
          }
        }

        // write neighbors.
        uint64_t new_id = id_map.find(id);
        uint64_t off = new_id % VEC_IN_WBUF;
        // loc_sector_no(off) - loc_sector_no(0) == loc_sector_no(new_id) - loc_sector_no(start_id)
        auto page_wbuf = wbuf + (loc_sector_no(off) - loc_sector_no(0)) * SECTOR_LEN;
        DiskNode<T> w_node = node_from_page(page_wbuf, new_id);

        memcpy(w_node.coords, node.coords, meta_.data_dim * sizeof(T));
        w_node.nnbrs = nhood.size();
        *(w_node.nbrs - 1) = w_node.nnbrs;
        memcpy(w_node.nbrs, nhood.data(), w_node.nnbrs * sizeof(uint32_t));
        populated_slots[off] = 1;
        ++n_used_id;
        // copy tags.
        new_tags[new_id] = id2tag(id);
      }

      LOG(INFO) << "Processed " << loc_ed << "/" << cur_loc << " locs, n_used_id: " << n_used_id << ".";
      if (n_used_id - wb_id >= LOC_PER_MERGE) {
        write_back();
      }
    }

    while (wb_id < n_used_id) {
      write_back();
    }
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";

    close(fd);
    aligned_free((void *) rbuf);
    aligned_free((void *) wbuf);

    // Block search during PQ shuffle and metadata swap (unlock in reload()).
    merge_lock.lock();
    while (deleted_nodes_set.find(id2tag(meta_.entry_point)) != deleted_nodes_set.end()) {
      LOG(INFO) << "Medoid deleted. Choosing another start node. Medoid ID: " << meta_.entry_point
                << " tag: " << id2tag(meta_.entry_point);
      const auto &nhoods = deleted_nhoods.find(meta_.entry_point);
      meta_.entry_point = nhoods[0];
    }

    // duplicate neighbor handler.
    auto new_nbr_handler = this->nbr_handler->shuffle(rev_id_map, new_npoints, nthreads);

    // set metadata, PQ and tags.
    // metadata.
    this->meta_.npoints = new_npoints;
    this->meta_.entry_point = id_map.find(meta_.entry_point);
    // Compact entry_points_ with the ID map (stale IDs OOBs first PQ lookup).
    {
      std::vector<uint32_t> remapped_eps;
      remapped_eps.reserve(entry_points_.size());
      for (uint32_t ep : entry_points_) {
        uint32_t new_ep = 0;
        if (id_map.find(ep, new_ep)) {
          remapped_eps.push_back(new_ep);
        }
      }
      if (remapped_eps.empty()) {
        remapped_eps.push_back(static_cast<uint32_t>(meta_.entry_point));
      }
      entry_points_.swap(remapped_eps);
      LOG(INFO) << "Remapped " << entry_points_.size() << " entry point(s) after delete merge";
    }
    // PQ.
    auto tmp = this->nbr_handler;
    this->nbr_handler = new_nbr_handler;
    delete tmp;
    // tags.
    tags.clear();
    // no need to clear id2loc & loc2id as they are arrays.
    // out-of-bound loc2id is initialized in reload().
#pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < new_tags.size(); ++i) {
      tags.insert_or_assign(i, new_tags[i]);
      set_id2loc(i, i);
      set_loc2id(i, i);
    }

    this->write_metadata_and_pq(in_path_prefix, out_path_prefix, &new_tags);
    pipeann::save_bin<uint32_t>(out_path_prefix + "_disk.index_medoids.bin", entry_points_.data(),
                                entry_points_.size(), 1, 0);
    LOG(INFO) << "Write metadata and PQ finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                                std::vector<TagT> *new_tags) {
    // Use meta_ directly since it's already been updated
    meta_.print();
    std::string disk_index_out = out_path_prefix + "_disk.index";
    meta_.save_to_disk_index(disk_index_out);

    // Step 3. Write tags and PQ.
    std::vector<TagT> tags_vec;
    if (new_tags == nullptr) {
      tags_vec.resize(meta_.npoints);
      for (uint64_t i = 0; i < meta_.npoints; ++i) {
        tags_vec[i] = id2tag(i);
      }
      new_tags = &tags_vec;
    }
    pipeann::save_bin<TagT>(out_path_prefix + "_disk.index.tags", new_tags->data(), meta_.npoints, 1, 0);
    nbr_handler->save(out_path_prefix.c_str());
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::reload(const char *index_prefix, uint32_t num_threads) {
    std::string iprefix = std::string(index_prefix);
    std::string disk_index_file = iprefix + "_disk.index";
    this->disk_index_file = disk_index_file;
    this->max_nthreads = num_threads;

    reader->close();
    reader->open(disk_index_file, true, false);

    // Capture cur_loc before init_metadata shrinks it (else clear loop is empty).
    const uint64_t old_cur_loc = this->cur_loc.load();
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // No need to reload PQ, as it is already reloaded in merge_deletes.
    for (uint64_t i = this->meta_.npoints; i < old_cur_loc; ++i) {
      set_loc2id(static_cast<uint32_t>(i), kInvalidID);  // reset loc2id.
    }
    while (!this->empty_pages.empty()) {
      this->empty_pages.pop();
    }
    this->load_entry_points(iprefix);
    merge_lock.unlock();
    LOG(INFO) << "Reload finished, cur_id: " << this->cur_id << ", cur_loc: " << this->cur_loc;
    return;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
