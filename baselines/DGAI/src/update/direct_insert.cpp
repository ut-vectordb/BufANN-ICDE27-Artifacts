#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "timer.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

#include "global_stats.h"

namespace pipeann {
  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    QueryBuffer<T> *read_data = this->pop_query_buf(point);
    void *ctx = reader->get_ctx();

    Timer timer;
    uint32_t target_id = cur_id++;
    // write PQ.

    #ifdef COLLECT_INFO_2
    timer.reset();
    #endif
    std::vector<uint8_t> pq_coords = deflate_vector(point);
    uint64_t pq_offset = target_id * n_chunks;
    {
      static std::mutex pq_mu;
      std::lock_guard<std::mutex> lock(pq_mu);
      if (this->data.size() < pq_offset + n_chunks) {
        while (this->data.size() < pq_offset + n_chunks) {
          this->data.resize(1.5 * this->data.size());
        }
      }
      memcpy(this->data.data() + pq_offset, pq_coords.data(), n_chunks);
    }

    #ifdef COLLECT_INFO_2
    if (gs != nullptr ){
      gs->insert_time1 += timer.elapsed();
    }
    timer.reset();
    #endif
    std::vector<Neighbor> exp_node_info;
    tsl::robin_map<uint32_t, T *> coord_map;
    coord_map.reserve(2 * this->l_index);

    std::vector<uint64_t> page_ref{};
    QueryStats stats;
    this->do_beam_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false,
                         &page_ref, read_data);
    if(gs != nullptr) {
      gs->insert_io1 += stats.io_us;
      gs->insert_io += stats.io_us;
      gs->update_ios += stats.n_ios * SECTOR_LEN;
      gs->update_effective_topo_ios += stats.n_ios * (this->range + 1) * sizeof(uint32_t);
      gs->update_effective_coord_ios += stats.n_ios * this->data_dim * sizeof(T);
      gs->insert_ios1 += stats.n_ios * SECTOR_LEN + stats.rerank_ios * SECTOR_LEN;
    }
    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time2 += timer.elapsed();
    }
    timer.reset();
    #endif

    std::vector<uint32_t> new_nhood;
    prune_neighbors(coord_map, exp_node_info, new_nhood);// TODO : coord_map may be wrong! add data_buf as paramter to do_beam_search
    // locs[new_nhood.size()] is the target, locs[0:new_nhood.size() - 1] are the neighbors.
    // lock the pages to write
    std::set<uint64_t> pages_need_to_read;

    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time3 += timer.elapsed();
    }
    timer.reset();
    #endif
    
#ifdef IN_PLACE_RECORD_UPDATE
    std::vector<uint64_t> locs;
    for (auto &nbr : new_nhood) {
      locs.emplace_back(id2loc(nbr));
      pages_need_to_read.insert(node_sector_no(nbr));
    }
    locs.push_back(target_id);
    pages_need_to_read.insert(loc_sector_no(target_id));
    id2loc_.insert_or_assign(target_id, target_id);

    // update loc2id, target_id <-> target_id.
    cur_loc++;  // for target ID, atomic update.
    set_loc2id(target_id, target_id);
#else
    auto locs = this->alloc_loc(new_nhood.size() + 1, page_ref, pages_need_to_read);
#endif

    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set
    for (auto &page_no : pages_to_rmw_set) {
      pages_to_rmw.push_back(IORequest(1ULL * page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }
    
    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time4 += timer.elapsed();
    }
    #endif

    // lock the target and the neighbor ids (ensure that sector_no does not change).
    auto pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);
    lock_vec(vec_lock_table, target_id, new_nhood);

    #ifdef COLLECT_INFO_2 
    timer.reset();
    #endif
    // re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint32_t, char *> page_buf_map;

    auto &update_buf = read_data->update_buf;
    std::vector<IORequest> reads, writes_4k, writes;
    assert(new_nhood.size() < MAX_N_EDGES);
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      reads.push_back(
          IORequest(node_sector_no(new_nhood[i]) * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0));
      page_buf_map[node_sector_no(new_nhood[i])] = update_buf + i * size_per_io;
    }

    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;
      writes_4k.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
        reads.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      }
      page_buf_map[off / SECTOR_LEN] = update_buf + i * size_per_io;
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
    uint64_t start_idx = 0;
    uint64_t cur_off = writes_4k[0].offset;
    for (uint32_t i = 1; i < writes_4k.size(); ++i) {
      if (writes_4k[i].offset != cur_off + size_per_io) {
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
        start_idx = i;
      }
      cur_off = writes_4k[i].offset;
    }
    writes_4k.pop_back();

    timer.reset();
#ifdef DIRECT_READ_CC
    reader->read(reads, ctx);
#else
    reader->read_alloc(reads, ctx, &page_ref);
#endif
    if(gs != nullptr) {
      gs->insert_io2 += timer.elapsed();
      gs->insert_io += timer.elapsed();
      gs->update_ios += gs->read_ios * SECTOR_LEN;
      gs->insert_ios2 += gs->read_ios * SECTOR_LEN;
      gs->read_ios = 0;
    }

    // update the target node.
    auto sector = loc_sector_no(locs[new_nhood.size()]);
    auto node_buf = offset_to_loc(page_buf_map[sector], locs[new_nhood.size()]);
    DiskNode<T> target_node(target_id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
    memcpy(target_node.coords, point, data_dim * sizeof(T));
    target_node.nnbrs = new_nhood.size();
    *(target_node.nbrs - 1) = target_node.nnbrs;
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    tags.insert_or_assign(target_id, tag);

    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time5 += timer.elapsed();
    }
    timer.reset();
    #endif
    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto r_sector = node_sector_no(new_nhood[i]);
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        LOG(ERROR) << new_nhood[i] << " " << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }
      auto r_node_buf = offset_to_node(page_buf_map[r_sector], new_nhood[i]);
      DiskNode<T> r_nbr_node(new_nhood[i], offset_to_node_coords(r_node_buf), offset_to_node_nhood(r_node_buf));
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      if (nhood.size() > this->range) {  // prune neighbors
#ifdef DELTA_PRUNING
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        compute_pq_dists(target_id, nhood.data(), tgt_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        compute_pq_dists(r_nbr_node.id, nhood.data(), nbr_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        std::vector<TriangleNeighbor> tri_pool(nhood.size());

        for (uint32_t k = 0; k < nhood.size(); k++) {
          tri_pool[k].id = nhood[k];
          tri_pool[k].tgt_dis = tgt_dists[k];
          tri_pool[k].distance = nbr_dists[k];
        }
        std::sort(tri_pool.begin(), tri_pool.end());

        int tgt_idx = -1;
        for (int k = 0; k < (int) nhood.size(); ++k) {
          if (tri_pool[k].id == target_id) {
            tgt_idx = k;
            break;
          }
        }
        if (unlikely(tgt_idx == -1)) {
          LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
          exit(-1);
        }
        this->delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
#else
        std::vector<float> dists(nhood.size(), 0.0f);
        std::vector<Neighbor> pool(nhood.size());
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        compute_pq_dists(r_nbr_node.id, nhood.data(), dists.data(), (_u32) nhood.size(), thread_pq_buf);
        for (uint32_t k = 0; k < nhood.size(); k++) {
          pool[k].id = nhood[k];
          pool[k].distance = dists[k];
        }
        nhood.clear();
        std::sort(pool.begin(), pool.end());
        this->prune_neighbors_pq(pool, nhood, thread_pq_buf);
#endif
      }

      auto w_sector = loc_sector_no(locs[i]);
      auto w_node_buf = offset_to_loc(page_buf_map[w_sector], locs[i]);
      DiskNode<T> w_nbr_node(new_nhood[i], offset_to_node_coords(w_node_buf), offset_to_node_nhood(w_node_buf));
      w_nbr_node.nnbrs = (_u32) nhood.size();
      *(w_nbr_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      memcpy(w_nbr_node.coords, r_nbr_node.coords, data_dim * sizeof(T));
      memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));
    }

    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time6 += timer.elapsed();
    }
    timer.reset();
    #endif
    std::vector<uint64_t> write_page_ref;
    reader->wbc_write(writes, ctx, &write_page_ref);

#ifndef IN_PLACE_RECORD_UPDATE
    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    id2loc_.insert_or_assign(target_id, locs[new_nhood.size()]);
    auto locked = lock_idx(idx_lock_table, target_id, new_nhood);
    auto page_locked = lock_page_idx(page_idx_lock_table, target_id, new_nhood);
    std::vector<uint64_t> orig_locs;
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      id2loc_.insert_or_assign(new_nhood[i], locs[i]);
    }

    // with lock, for simple concurrency with alloc_loc.
    // Only for convenience, note that locs[new_nhood.size()] -> target.
    new_nhood.push_back(target_id);
    erase_and_set_loc(orig_locs, locs, new_nhood);
    unlock_page_idx(page_idx_lock_table, page_locked);
    unlock_idx(idx_lock_table, locked);
#endif
    unlock_vec(vec_lock_table, target_id, new_nhood);

    // LOG(INFO) << "ID " << target_id << " Target loc " << id2loc(target_id);

    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time7 += timer.elapsed();
    }
    timer.reset();
    #endif
    // commit writes (in the background thread.)
#ifdef BG_IO_THREAD
    if (!page_ref.empty()) {
      auto bg_task = new BgTask{
          .thread_data = read_data,
          .writes = std::move(writes),
          .pages_to_unlock = std::move(pages_locked),
          .pages_to_deref = std::move(write_page_ref),
      };
      bg_tasks.push(bg_task);
      bg_tasks.push_notify_all();
    } else {
      v2::unlockReqs(this->page_lock_table, pages_locked);
    }
    reader->deref(&page_ref, ctx);
#else
    timer.reset();
    reader->write(writes, ctx);
    if(gs != nullptr) {
      gs->insert_io3 += timer.elapsed();
      gs->insert_io += timer.elapsed();
      gs->update_effective_coord_ios += this->data_dim * sizeof(T);
      for(auto &write : writes) {
        gs->update_ios += write.len;
        gs->insert_ios3 += write.len;
      }
    }
    v2::unlockReqs(this->page_lock_table, pages_locked);
    reader->deref(&write_page_ref, ctx);

    reader->deref(&page_ref, ctx);
    this->push_query_buf(read_data);
#endif
    return target_id;
  }

  template<class T, class TagT>
  void SSDIndex<T, TagT>::bg_io_thread() {
    auto ctx = reader->get_ctx();
    uint8_t *buf = nullptr;
    pipeann::alloc_aligned((void **) &buf, (MAX_N_EDGES + 1) * SECTOR_LEN, SECTOR_LEN);
    auto timer = pipeann::Timer();
    uint64_t n_tasks = 0;

    while (true) {
      auto task = bg_tasks.pop();
      while (task == nullptr) {
        this->bg_tasks.wait_for_push_notify();
        task = bg_tasks.pop();
      }

      timer.reset();
      reader->write(task->writes, ctx);
      if(gs != nullptr) {
        gs->insert_io3 += timer.elapsed();
        gs->insert_io += timer.elapsed();
        gs->update_effective_coord_ios += this->data_dim * sizeof(T);
        for(auto &write : task->writes) {
          gs->update_ios += write.len;
          gs->insert_ios3 += write.len;
        }
      }
      v2::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      reader->deref(&task->pages_to_deref, ctx);
      this->push_query_buf(task->thread_data);
      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
