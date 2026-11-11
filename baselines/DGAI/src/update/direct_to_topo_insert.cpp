#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <optional>
#include <cstring>
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
  int SSDIndex<T, TagT>::insert_to_topo_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    QueryBuffer<T> *read_data = this->pop_query_buf(point);
    // std::cerr << "read_data " << (void *)read_data << "\n";
    void *ctx = reader->get_ctx();
    Timer timer;
    // std::cerr << "counter is " << counter << "\n";
    uint32_t target_id = cur_id++;
    // write PQ.
    // std::cerr << "insert id " << target_id << "\n";

    int tid = omp_get_thread_num();

    auto p_reader = (LinuxAlignedFileReader *)this->reader.get();
    int strategy = p_reader->strategy;
    FileHandle coord_fd = p_reader->coord_file_desc;
    FileHandle topo_fd = p_reader->topo_file_desc;

    auto & block_cache = p_reader->block_cache;
    int use_topo_reorder = (strategy >> 1) & 0x1;
    int use_double_pq = (strategy >> 2) & 0x1;
    int use_coord_reorder = (strategy >> 3) & 0x1;
    int use_topo_buffer = (strategy >> 4) & 0x1;
    if (use_coord_reorder) {
      LOG(ERROR) << "use_coord_reorder is not supported in direct to topo insert";
    }
    // Upstream adds the new vector's page to the insert-wide page-lock set, so it stays locked from lockReqs
    // until the very end of the insert. With no freed slot to reuse, consecutive inserts append to the same
    // tail page of the vector file and therefore run one at a time. The only thing that lock needs to protect
    // is the read-modify-write of that page below (no other code writes vector pages, and searches read them
    // under vec_lock_table), so DGAI_NARROW_COORD_LOCK (added for the benchmark harness) locks the
    // page only around that read-modify-write, in its own table. That lock is a leaf: its holder waits on
    // nothing but its own two I/Os, so taking it while holding graph-page locks cannot deadlock.
    // Enabled when DGAI_NARROW_COORD_LOCK is set to anything but "" or "0".
    static const bool narrow_coord_lock = [] {
      const char *v = std::getenv("DGAI_NARROW_COORD_LOCK");
      return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    }();
    
    #ifdef COLLECT_INFO_2
    timer.reset();
    #endif
    if(!use_double_pq){
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
    }else{
      std::vector<float> dist(n_chunks, 0.0f);
      std::vector<float> dist_2(n_chunks, 0.0f);
      std::vector<uint8_t> pq_coords = deflate_vector_with_dist(point, dist.data());
      std::vector<uint8_t> pq_coords_2 = deflate_vector_2_with_dist(point, dist_2.data());

      std::vector<_u8> final_code(n_chunks);
      if (unlikely(target_id * n_chunks >= map_bytes.size())) {
        LOG(ERROR) << "target_id * n_chunks >= map_bytes.size()";
        exit(-1);
      }
      for (uint32_t c = 0; c < n_chunks; c++) {
        if (dist[c] <= dist_2[c]) {
          final_code[c] = pq_coords[c];
          map_bytes[target_id * n_chunks + c] = 0;
        } else {
          final_code[c] = pq_coords_2[c];
          map_bytes[target_id * n_chunks + c] = 1;
        }
      }

      uint64_t pq_offset = target_id * n_chunks;
      {
        static std::mutex pq_mu;
        std::lock_guard<std::mutex> lock(pq_mu);
        if (this->data.size() < pq_offset + n_chunks) {
          while (this->data.size() < pq_offset + n_chunks) {
            this->data.resize(1.5 * this->data.size());
          }
        }
        memcpy(this->data.data() + pq_offset, final_code.data(), n_chunks);
      }
    }

    #ifdef COLLECT_INFO_2
    if (gs != nullptr ){
      gs->insert_time1 += timer.elapsed();
    }
    timer.reset();
    #endif
    
    std::vector<Neighbor> exp_node_info;
    tsl::robin_map<uint32_t, T *> coord_map;
    tsl::robin_map<uint32_t, char *> topo_page_map;
    coord_map.reserve(2 * this->l_index);
    topo_page_map.reserve(2 * this->l_index);

    tsl::robin_map<uint32_t, int> page_ref;
    QueryStats stats;

    this->do_rerank_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, &topo_page_map, &stats, deletion_set,
                read_data, use_topo_buffer ? &page_ref : nullptr, 4 + 32);//这里可能会抛出 not found in id2loc_topo id2loc_coord
    if(gs != nullptr) {
      gs->insert_io += stats.io_us;
      gs->update_ios += stats.n_ios * SECTOR_LEN;
      gs->insert_io1 += stats.io_us;
      gs->insert_ios1 += stats.n_ios * SECTOR_LEN + stats.rerank_ios * SECTOR_LEN;
      gs->insert_io8 += stats.rerank_us;
      gs->insert_ios8 += stats.rerank_ios * SECTOR_LEN;
    }
    
    
    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time2 += timer.elapsed();
    }
    timer.reset();
    #endif

    std::vector<uint32_t> new_nhood;
    prune_neighbors(coord_map, exp_node_info, new_nhood);// (correct) : coord_map may be wrong! add data_buf as paramter to do_beam_search

    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time3 += timer.elapsed();
    }
    timer.reset();
    #endif
    
    if(new_nhood.empty()){
      LOG(ERROR) << "new_nhood is empty";
      exit(-1);
    }

    uint32_t target_loc_topo = kInvalidID;
    uint32_t target_loc_coord = kInvalidID;
    
    std::set<uint64_t> pages_to_rmw_set;//所有需要修改因此加锁的page

    uint32_t new_topo_pid = -1, new_coord_pid = kInvalidID;
    {// 选择topo和coord的插入loc
      std::lock_guard<std::mutex> lock(alloc_lock);
      if(use_topo_reorder) {
        //得到一个空盘块，之后一起加锁
        uint32_t empty = kInvalidID;
        if ((empty = empty_pages_topo.pop()) != kInvalidID) {
          new_topo_pid = empty;
        }
        if(new_topo_pid == kInvalidID){
          new_topo_pid = cur_loc_topo / ntopo_per_sector;
          cur_loc_topo = (new_topo_pid + 1) * ntopo_per_sector;
        }
        // std::cerr << "new_topo_pid " << new_topo_pid << "\n";
        // std::cerr << "cur_loc_topo " << cur_loc_topo << "\n";
        pages_to_rmw_set.insert(new_topo_pid);
      } else {
        uint32_t empty_loc = kInvalidID;//这里只能用kInvalidID，因为queu空时就是kInvalidID
        if ((empty_loc = empty_locs_topo.pop()) != kInvalidID) {// reuse loc
          target_loc_topo = empty_loc;
        }
        if(target_loc_topo == kInvalidID){
          target_loc_topo = cur_loc_topo.fetch_add(1);
        }
        uint32_t pid = loc_sector_no_topo(target_loc_topo);
        pages_to_rmw_set.insert(pid);
        // std::cerr << "insert id2loc_topo_ " << target_id << " " << target_loc_topo << "\n";
        id2loc_topo_.insert_or_assign(target_id, target_loc_topo);
        set_loc2id_topo(target_loc_topo, target_id);//直接设置就可以，因为只有当反向边加入后，target才可见
      }
      
      // if (1) 
      // {
      //   target_loc_coord = cur_loc_coord.load() + coord_buffer_idx.fatch_add(1); 
        
      //   memcpy(coord_buffer, point, data_dim * sizeof(T));
      //   set_loc2id_coord(target_loc_coord, target_id);
      //   id2loc_coord_.insert_or_assign(target_id, target_loc_coord);
      //   if (coord_buffer_idx.load() == COORD_BUFFER_SIZE) {
          
      //   }
      // }
      // else
      {
        uint32_t empty_loc = kInvalidID;
        if ((empty_loc = empty_locs_coord.pop()) != kInvalidID) {
          target_loc_coord = empty_loc;
        }
        if(target_loc_coord == kInvalidID){
          // Upstream appends at the tail, so consecutive inserts share one
          // vector page, and that page stays locked for the whole insert.
          // DGAI_STRIPE_COORD_ALLOC (added for the benchmark harness)
          // instead stripes consecutive appends across kStripePages pages
          // beyond the tail, to test whether that shared page is what
          // serializes inserts.
          static const bool stripe_coord = std::getenv("DGAI_STRIPE_COORD_ALLOC") != nullptr;
          if (!stripe_coord) {
            target_loc_coord = cur_loc_coord.fetch_add(1);
          } else {
            constexpr uint64_t kStripePages = 128;
            static const uint64_t stripe_base = cur_loc_coord.load();  // page-aligned at load
            static std::atomic<uint64_t> stripe_next{0};
            const uint64_t k = stripe_next.fetch_add(1);
            const uint64_t per_group = kStripePages * ncoord_per_sector;
            const uint64_t group = k / per_group, in_group = k % per_group;
            target_loc_coord = stripe_base + (group * kStripePages + in_group % kStripePages) * ncoord_per_sector +
                               in_group / kStripePages;
            uint64_t end = cur_loc_coord.load();
            while (end < target_loc_coord + 1 && !cur_loc_coord.compare_exchange_weak(end, target_loc_coord + 1)) {
            }
          }
        }
        uint32_t pid = loc_sector_no_coord(target_loc_coord);
        if (!narrow_coord_lock) {
          pages_to_rmw_set.insert(pid);
        }
        set_loc2id_coord(target_loc_coord, target_id);
        id2loc_coord_.insert_or_assign(target_id, target_loc_coord);
      }
      tags.insert_or_assign(target_id, tag);
    }
    
    //
    for (auto &nbr : new_nhood) {
      uint32_t loc = id2loc_topo(nbr);// TODO : 这里到lockReq之间有并发风险
      if(loc == kInvalidID){
        LOG(ERROR) << "id " << nbr << " not found in id2loc_topo";
        exit(-1);
      }
      uint32_t pid = loc_sector_no_topo(loc);
      pages_to_rmw_set.insert(pid);
    }

    std::vector<IORequest> pages_to_rmw;
    for (auto &page_no : pages_to_rmw_set) {//整理成iorequest
      pages_to_rmw.push_back(IORequest(1ULL * page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }

    
    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time4 += timer.elapsed();
    }
    #endif
    timer.reset();
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    auto lock_begin = std::chrono::steady_clock::now();
    auto pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);//加上大的page locks
    if (gs != nullptr) {
      gs->insert_cpu += timer.elapsed();
      gs->harness_lock_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - lock_begin).count();
      gs->harness_pages_locked += pages_locked.size();
      gs->harness_inserts++;
    }

    #ifdef COLLECT_INFO_2 
    timer.reset();
    #endif
    // re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint64_t, char *> topo_page_buf_map;// just for topo

    auto &update_buf = read_data->update_buf;
    
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {//设置好topo_page_buf_map，用于更新邻居点的topo
      uint32_t id = new_nhood[i];
      uint32_t loc = id2loc_topo(id);
      if(loc == kInvalidID){
        LOG(ERROR) << "id " << id << " not found in id2loc_topo";
        exit(-1);
      }
      uint32_t pid = loc_sector_no_topo(loc);
      topo_page_buf_map[pid] = update_buf + i * size_per_io;
      if(use_topo_buffer){//如果有buffer，就使用最新的缓存里的数据
        if(page_ref.find(pid) == page_ref.end()){//说明所在页面因为reorder的原因被换掉了，所以需要重新读取
          // LOG(ERROR) << "pid " << pid << " not found in page_ref";
          int cb_idx = block_cache->request_block(pid);
          if(block_cache->cache_status[cb_idx] != 1){//假如不在cache里，应该从磁盘里读，读到位置上
            // LOG(ERROR) << "Sector " << pid << " is not cached";
            IORequest req = IORequest(1ULL * pid * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0);
            timer.reset();
            reader->send_io(req, ctx, false, topo_fd);
            while(!req.finished){
              reader->poll_wait(ctx);
            }
            if (gs != nullptr) {
              gs->insert_io4 += timer.elapsed();
              gs->insert_ios4 += SECTOR_LEN;
              gs->insert_ios2 += SECTOR_LEN;
              gs->insert_io2 += timer.elapsed();
              gs->insert_io += timer.elapsed();
              gs->update_ios += SECTOR_LEN;
            }
            memcpy(&block_cache->cache_block_vec[cb_idx], update_buf + i * size_per_io, size_per_io);
            block_cache->cache_status[cb_idx] = 1;
          } else {
            memcpy(update_buf + i * size_per_io, &block_cache->cache_block_vec[cb_idx], size_per_io);
          }
          page_ref[pid] = cb_idx;
        } else {
          if(unlikely( block_cache->cache_status[page_ref[pid]] != 1)){//TODO : 这里可能有问题
            LOG(ERROR) << "Sector " << pid << " is not cached";
            // exit(-1);
            memcpy(update_buf + i * size_per_io, topo_page_map[pid], size_per_io);
          } else {
            memcpy(update_buf + i * size_per_io, &block_cache->cache_block_vec[page_ref[pid]], size_per_io);
          }
        }
      } else {//没有buffer，就用搜索时的数据
        memcpy(update_buf + i * size_per_io, topo_page_map[pid], size_per_io);
      }
    }
    
    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time5 += timer.elapsed();
    }
    timer.reset();
    #endif
    std::unordered_map<uint32_t, unsigned *> nhood_cached_ids;
    
    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      uint32_t id = new_nhood[i];
      uint32_t loc = id2loc_topo(id);
      if(unlikely( loc == kInvalidID)){
        LOG(ERROR) << "id " << id << " not found in id2loc_topo";
        exit(-1);
      }
      uint32_t pid = loc_sector_no_topo(loc);
      if (unlikely(topo_page_buf_map.find(pid) == topo_page_buf_map.end())) {
        LOG(ERROR) << id << " " << "Sector " << pid << " not found in topo_page_buf_map";
        exit(-1);
      }
      char *node_buf = offset_to_loc_topo(topo_page_buf_map[pid], loc);
      DiskNode<T> nbr_node(id, (unsigned *)(node_buf));
      std::vector<uint32_t> nhood(nbr_node.nnbrs + 1);
      nhood.assign(nbr_node.nbrs, nbr_node.nbrs + nbr_node.nnbrs);
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      // nhood.resize(this->range);
      if (nhood.size() > this->range) {  // prune neighbors
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        // timer.reset();
        compute_pq_dists(target_id, nhood.data(), tgt_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        compute_pq_dists(nbr_node.id, nhood.data(), nbr_dists.data(), (_u32) nhood.size(), thread_pq_buf);
        // if (gs != nullptr) {
        //   gs->insert_cpu += timer.elapsed();
        // }
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
        // timer.reset();
        this->delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
        // if (gs != nullptr) {
        //   gs->insert_cpu += timer.elapsed();
        // }
      }
      nbr_node.nnbrs = (_u32) nhood.size();
      *(nbr_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      memcpy(nbr_node.nbrs, nhood.data(), nbr_node.nnbrs * sizeof(uint32_t));
#ifdef USE_NHOOD_CACHE
      if(nhood_cache.find(id) != nhood_cache.end()){
        nhood_cached_ids.insert(std::pair(id, nbr_node.nbrs - 1));
      }
#endif
    }
    
    
    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time6 += timer.elapsed();
    }
    timer.reset();
    #endif
    // update target COORD
    // if(!use_coord_reorder)
    {
      char *target_coord_buf = update_buf + (new_nhood.size() + 1) * size_per_io;
      IORequest coord_req = IORequest(loc_sector_no_coord(target_loc_coord) * SECTOR_LEN, size_per_io, target_coord_buf, 0, 0);
      std::optional<v2::SparseWriteLockGuard<uint64_t>> coord_page_guard;
      if (narrow_coord_lock) {
        auto coord_lock_begin = std::chrono::steady_clock::now();
        coord_page_guard.emplace(&this->coord_page_lock_table, loc_sector_no_coord(target_loc_coord));
        if (gs != nullptr) {
          gs->harness_lock_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - coord_lock_begin).count();
          gs->harness_pages_locked += 1;
        }
      }
      timer.reset();
      reader->send_io(coord_req, ctx, false, coord_fd);
      while(!coord_req.finished){
        reader->poll_wait(ctx);
      }

      char *coord_node_buf = offset_to_loc_coord(target_coord_buf, target_loc_coord);
      DiskNode<T> target_node_coord(target_id, (T *)coord_node_buf);
      memcpy(target_node_coord.coords, point, data_dim * sizeof(T));

      reader->send_io(coord_req, ctx, true, coord_fd);
      while(!coord_req.finished){
        reader->poll_wait(ctx);
      }
      if (gs != nullptr) {
        gs->insert_io2 += timer.elapsed();
        gs->insert_io += timer.elapsed();
        gs->update_ios += 2 * SECTOR_LEN;
      }
      
    } 

    bool split = false;
    uint32_t split_pid = kInvalidID;
    std::unordered_set<uint32_t> ids;
    // std::vector<uint32_t> ids;
    std::vector<uint32_t> ids1, ids2;
    std::vector<uint32_t> to_lock;
    if(!use_topo_reorder)
    {
      char *target_topo_buf = (update_buf + new_nhood.size() * size_per_io);
      IORequest topo_req = IORequest(loc_sector_no_topo(target_loc_topo) * SECTOR_LEN, size_per_io, target_topo_buf, 0, 0);
      timer.reset();
      reader->send_io(topo_req, ctx, false, topo_fd);
      while(!topo_req.finished){
        reader->poll_wait(ctx);
      }
      uint32_t target_topo_pid = loc_sector_no_topo(target_loc_topo);
      char *topo_node_buf = offset_to_loc_topo(target_topo_buf, target_loc_topo);
      DiskNode<T> target_node_topo(target_id, (unsigned *)(topo_node_buf));

      target_node_topo.nnbrs = new_nhood.size();
      *(target_node_topo.nbrs - 1) = target_node_topo.nnbrs;
      memcpy(target_node_topo.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
      
      reader->send_io(topo_req, ctx, true, topo_fd);
      while(!topo_req.finished){
        reader->poll_wait(ctx);
      }
      if (gs != nullptr) {
        gs->insert_io2 += timer.elapsed();
        gs->insert_io += timer.elapsed();
        gs->update_ios += 2 * SECTOR_LEN;
      }

      
      if(use_topo_buffer){
        auto target_cb_idx = block_cache->request_block(target_topo_pid);
        memcpy(&block_cache->cache_block_vec[target_cb_idx], topo_req.buf, topo_req.len);
        block_cache->cache_status[target_cb_idx] = 1;
        block_cache->release_cache_block(target_cb_idx, 2);// TODO : 写回后释放
      }
      
    } else {
      //TODO : 检查new_nhood是否根据距离排好序了
      for(uint32_t i = 0; i < new_nhood.size(); ++i){
        uint32_t id = new_nhood[i];
        uint32_t loc = id2loc_topo(id);
        if(loc == kInvalidID){
          LOG(ERROR) << "id " << id << " not found in id2loc_topo";
          exit(-1);
        }
        uint32_t pid = loc_sector_no_topo(loc);
        if(!page_layout_topo.update_fn(pid, [&](PageArrTopo &v) {//设置好了loc2id
          for (uint32_t i = 0; i < ntopo_per_sector; ++i) {
            if (v[i] == kInvalidID) {
              v[i] = target_id;
              target_loc_topo = sector_to_loc_topo(pid, i);
              break;
            }
          }
        })){
          LOG(ERROR) << "pid:" << pid << " not found in page_layout_topo";
        }
        if (target_loc_topo != kInvalidID) {
          break;
        }
      }
      if(target_loc_topo != kInvalidID){//不需要split，直接将自己的topo拷贝到对应的缓冲区位置就行

        empty_pages_topo.push(new_topo_pid);
        uint32_t pid = loc_sector_no_topo(target_loc_topo);
        DiskNode<T> target_node(target_id, (unsigned *)(offset_to_loc_topo(topo_page_buf_map[pid], target_loc_topo)));
        target_node.nnbrs = new_nhood.size();
        *(target_node.nbrs - 1) = target_node.nnbrs;
        memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
        id2loc_topo_.insert_or_assign(target_id, target_loc_topo);
        set_loc2id_topo(target_loc_topo, target_id);
      } else {
        uint32_t id = new_nhood[0];
        uint32_t loc = id2loc_topo(id);
        if(unlikely( loc == kInvalidID)){
          LOG(ERROR) << "id " << id << " not found in id2loc_topo";
          exit(-1);
        }
        split_pid = loc_sector_no_topo(loc);
        // new_topo_pid = ...;//新的pid
        topo_page_buf_map[new_topo_pid] = (update_buf + new_nhood.size() * size_per_io);
        if(!page_layout_topo.find_fn(split_pid, [&](PageArrTopo &v) { 
          for (uint32_t i = 0; i < ntopo_per_sector; ++i) {
            if(unlikely(v[i] == kAllocatedID)){
              LOG(ERROR) << "topo split, but some id are allocated";
              crash();
            }
            if(unlikely( v[i] == kInvalidID)){
              LOG(ERROR) << "topo split, but some id are kInvalidID";
              crash();
            }
            // ids.push_back(v[i]);
            ids.insert(v[i]);
          }
        })){
          LOG(ERROR) << "split_pid:" << split_pid << " not found in page_layout_topo";
        }
        if(unlikely( ids.size() != ntopo_per_sector)){
          std::cerr << ids.size() << "\n";
          LOG(ERROR) << "topo split, but ids.size() != ntopo_per_sector";
          crash();
        }
        
        split = true;
        if (gs != nullptr) {
          gs->harness_topo_splits++;
        }

        to_lock.insert(to_lock.end(), ids.begin(), ids.end());
        to_lock.push_back(target_id);

        if(1)
        {
          tsl::robin_map<uint32_t, uint32_t> group_id;
          for (auto id : ids) {
            uint32_t b_target = 0;
            if (group_id.find(id) == group_id.end()) {
              if (ids2.size() < ids1.size()) {
                ids2.push_back(id);
                group_id[id] = 2;
                b_target = 2;
              } else {
                ids1.push_back(id);
                group_id[id] = 1;
                b_target = 1;
              }
            } else {
              b_target = group_id[id];
            }
            if (b_target == 0){
              LOG(ERROR) << "b_target is 0";
              crash();
            }

            uint32_t loc = id2loc_topo(id);
            if(unlikely( loc == kInvalidID)){
              LOG(ERROR) << "id " << id << " not found in id2loc_topo";
              exit(-1);
            }
            uint32_t pid = loc_sector_no_topo(loc);
            if (unlikely( topo_page_buf_map.find(pid) == topo_page_buf_map.end()) ){
              LOG(ERROR) << id << " " << "Sector " << pid << " not found in topo_page_buf_map";
              exit(-1);
            }
            char *node_buf = offset_to_loc_topo(topo_page_buf_map[pid], loc);
            DiskNode<T> nbr_node(id, (unsigned *)(node_buf));
            std::vector<uint32_t> nhood(nbr_node.nnbrs);
            nhood.assign(nbr_node.nbrs, nbr_node.nbrs + nbr_node.nnbrs);
            for (auto w : nhood){
              if (group_id.find(w) == group_id.end() && ids.find(w) != ids.end()) {

                if (b_target == 1 && ids1.size() <= ntopo_per_sector / 2) {
                  group_id[w] = 1;
                  ids1.push_back(w);
                } else if(b_target == 2 && ids2.size() <= ntopo_per_sector / 2){
                  group_id[w] = 2;
                  ids2.push_back(w);
                }
              }
            }
          }

          if(unlikely( ids1.size() + ids2.size() != ids.size())){
            LOG(ERROR) << "ids1 size " << ids1.size() << " ids2 size " << ids2.size() << " ids size " << ids.size();
            crash();
          }

          if (unlikely(group_id.find(new_nhood[0]) == group_id.end())){
            LOG(ERROR) << "new_nhood[0] " << new_nhood[0] << " not found in group_id";
            crash();
          }
          if(group_id[new_nhood[0]] == 1 && ids1.size() < ntopo_per_sector && group_id.find(target_id) == group_id.end()){
            // if(group_id[pool[0].id] == 1 && group_id.find(node) == group_id.end()){
            ids1.push_back(target_id);
          } else {
            ids2.push_back(target_id);
          }
        } 
        //  {
        //   for(uint32_t i = 0; i < ntopo_per_sector; i ++){
        //     if (i < ntopo_per_sector / 2){
        //       ids1.push_back(ids[i]);
        //     } else {
        //       ids2.push_back(ids[i]);
        //     }
        //   }
        //   ids2.push_back(target_id);
        // }

        memcpy(topo_page_buf_map[new_topo_pid], topo_page_buf_map[split_pid] + topo_len * ids1.size(), topo_len * ids2.size());// 保证前面必须检查好了没有空id
        
        DiskNode<T> target_node(target_id, (unsigned *)(topo_page_buf_map[new_topo_pid] + topo_len * ids2.size()));

        target_node.nnbrs = new_nhood.size();
        *(target_node.nbrs - 1) = target_node.nnbrs;
        memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
      }
    }
    
    // std::cerr << "split " << split << "\n";
    std::vector<IORequest> writes;
    for(auto [pid, buf] : topo_page_buf_map) {
      writes.push_back(IORequest(pid * SECTOR_LEN, size_per_io, buf, 0, 0));
    }

    if(split){
      
      sort(to_lock.begin(), to_lock.end());
      auto last = std::unique(to_lock.begin(), to_lock.end());
      if(last != to_lock.end()){
        to_lock.erase(last, to_lock.end());
        LOG(ERROR) << "Topo split, but some id are duplicated";
      }
      // std::cerr << "lock\n";
      // std::cerr << "counter is " << counter << "\n";
      for (auto &id : to_lock) {
        idx_lock_table.wrlock(id);
      }
      // std::cerr << "lock begin\n";

      if(!page_layout_topo.update_fn(split_pid, [&](PageArrTopo &v) {
        for (uint32_t i = 0; i < ntopo_per_sector; ++i) {
          if(i < ids1.size()){
            v[i] = ids1[i];
            id2loc_topo_.insert_or_assign(v[i], sector_to_loc_topo(split_pid, i));
          }else {
            v[i] = kInvalidID;
          }
        }
      })){
        LOG(ERROR) << "split pid:" << split_pid << " not found in page_layout_topo";
      }
      
      page_layout_topo.upsert(new_topo_pid, [&](PageArrTopo &v, libcuckoo::UpsertContext ctx) {
        // if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
        for (uint32_t i = 0; i < ntopo_per_sector; ++i) {
          if(i < ids2.size()){
            v[i] = ids2[i];
            id2loc_topo_.insert_or_assign(v[i], sector_to_loc_topo(new_topo_pid, i));
          }else {
            v[i] = kInvalidID;
          }
        }
        // } else {
        //   LOG(ERROR) << "This is not new insert topo\n";
        //   crash();
        // }
      });

      if(!use_topo_buffer){
        LOG(ERROR) << "Need use_topo_buffer";
        exit(-1);
      } 

      if(unlikely( block_cache->cache_status[page_ref[split_pid]] != 1)){
        LOG(ERROR) << "Topo split, but split_pid is cached";
        exit(-1);
      }
      auto cb_idx = block_cache->request_block(new_topo_pid);
      if(unlikely(page_ref.find(new_topo_pid) != page_ref.end())){
        LOG(ERROR) << "Topo split, but new_topo_pid is cached";
        exit(-1);
      }
      page_ref[new_topo_pid] = cb_idx;
      memcpy(&block_cache->cache_block_vec[page_ref[split_pid]], topo_page_buf_map[split_pid], SECTOR_LEN);
      memcpy(&block_cache->cache_block_vec[page_ref[new_topo_pid]], topo_page_buf_map[new_topo_pid], SECTOR_LEN);
      block_cache->cache_status[page_ref[split_pid]] = 1;
      block_cache->cache_status[page_ref[new_topo_pid]] = 1;
      for (auto &id : to_lock) {
        idx_lock_table.unlock(id);
      }
      // std::cerr << "lock end\n";
    }
    
    #ifdef COLLECT_INFO_2 
    if (gs != nullptr ){
      gs->insert_time7 += timer.elapsed();
    }
    timer.reset();
    #endif
    
    if(use_topo_buffer){
      for(auto [pid, buf] : topo_page_buf_map){
        int cb_idx = page_ref[pid];
        memcpy(&block_cache->cache_block_vec[cb_idx], buf, SECTOR_LEN);
      }
      reader->write_fd(topo_fd, writes, ctx);
      for(auto [pid, cb_idx] : page_ref){
        block_cache->release_cache_block(cb_idx, 3);
      }
    } else {
      reader->write_fd(topo_fd, writes, ctx);
    }
    #ifdef COLLECT_INFO_2
    if (gs!= nullptr ){
      gs->insert_time8 += timer.elapsed();
    }
    #endif
    
    if(gs != nullptr) {
      gs->insert_io3 += timer.elapsed();
      gs->insert_io += timer.elapsed();
      for(auto &write : writes) {
        gs->update_ios += write.len;
        gs->insert_ios3 += write.len;
      }
    }
#ifdef USE_NHOOD_CACHE
      for(auto [id, buf] : nhood_cached_ids){
        if(nhood_cache.find(id) == nhood_cache.end()){
          LOG(ERROR) << "Nhood cache not found id: " << id;
          exit(-1);
        }
        memcpy(nhood_cache[id], buf, (*buf + 1) * sizeof(uint32_t));
      }
#endif
    
    v2::unlockReqs(this->page_lock_table, pages_locked);
  // }
    
    this->push_query_buf(read_data);
    // LOG(DEBUG) << tid << " " << "\n";
    // exit(0);
    return target_id;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
