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
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::coro_search(T **queries, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT **res_tags, float **res_dists, const _u64 beam_width, int N) {
    // beam search with intra-thread parallelism.
    static constexpr int kMaxCoroPerThread = 8;
    static constexpr int kMaxVectorDim = 512;
    struct alignas(SECTOR_LEN) CoroDataOne {
      // buffer.
      char sectors[SECTOR_LEN * 128];
      T query[kMaxVectorDim];
      _u8 pq_coord_scratch[32768 * 32];
      float pq_dists[32768];
      T data_buf[ROUND_UP(1024 * kMaxVectorDim, 256)];
      float dist_scratch[512];
      _u64 data_buf_idx;
      _u64 sector_idx;

      // search state.
      std::vector<Neighbor> full_retset;
      std::vector<Neighbor> retset;
      tsl::robin_set<_u64> visited;

      std::vector<unsigned> frontier;
      using fnhood_t = std::tuple<unsigned, unsigned, char *>;
      std::vector<fnhood_t> frontier_nhoods;
      std::vector<IORequest> frontier_read_reqs;

      SSDIndex<T> *parent;
      unsigned cur_list_size, cmps, k;

      void compute_dists(const unsigned *ids, const _u64 n_ids, float *dists_out) {
        ::aggregate_coords(ids, n_ids, parent->data.data(), parent->n_chunks, pq_coord_scratch);
        ::pq_dist_lookup(pq_coord_scratch, n_ids, parent->n_chunks, pq_dists, dists_out);
      };

      void print() {
        LOG(INFO) << "Full retset size " << full_retset.size() << " retset size: " << retset.size()
                  << " visited size: " << visited.size() << " frontier size: " << frontier.size()
                  << " frontier nhood size: " << frontier_nhoods.size()
                  << " frontier read reqs size: " << frontier_read_reqs.size();
      }

      void reset() {
        data_buf_idx = 0;
        sector_idx = 0;
        visited.clear();  // does not deallocate memory.
        retset.resize(4096);
        retset.clear();
        full_retset.clear();
        cur_list_size = cmps = k = 0;
      }

      void compute_and_add_to_retset(const unsigned *node_ids, const _u64 n_ids) {
        compute_dists(node_ids, n_ids, dist_scratch);
        for (_u64 i = 0; i < n_ids; ++i) {
          auto &item = retset[cur_list_size];
          item.id = node_ids[i];
          item.distance = dist_scratch[i];
          item.flag = true;
          cur_list_size++;
          visited.insert(node_ids[i]);
        }
      };

      void issue_next_io_batch(const _u64 beam_width, void *ctx) {
        if (search_ends()) {
          return;
        }
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        sector_idx = 0;

        _u32 marker = k;
        _u32 num_seen = 0;
        while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
          if (retset[marker].flag) {
            num_seen++;
            frontier.push_back(retset[marker].id);
            retset[marker].flag = false;
          }
          marker++;
        }

        // read nhoods of frontier ids
        std::vector<uint32_t> locked;
        if (!frontier.empty()) {
          for (_u64 i = 0; i < frontier.size(); i++) {
            uint32_t loc = frontier[i];
            uint64_t offset = parent->loc_sector_no(loc) * SECTOR_LEN;
            auto sector_buf = sectors + sector_idx * parent->size_per_io;
            fnhood_t fnhood = std::make_tuple(loc, loc, sector_buf);
            sector_idx++;
            frontier_nhoods.push_back(fnhood);
            frontier_read_reqs.emplace_back(IORequest(offset, parent->size_per_io, sector_buf, 0, 0));
          }
          parent->reader->send_io(frontier_read_reqs, ctx, false);
        }
      }

      bool io_finished(void *ctx) {
        parent->reader->poll(ctx);
        for (auto &req : frontier_read_reqs) {
          if (!req.finished) {
            return false;
          }
        }
        return true;
      }

      void explore_frontier(uint64_t l_search) {
        auto nk = cur_list_size;

        for (auto &frontier_nhood : frontier_nhoods) {
          auto [id, loc, sector_buf] = frontier_nhood;
          char *node_disk_buf = parent->offset_to_loc(sector_buf, loc);
          unsigned *node_buf = parent->offset_to_node_nhood(node_disk_buf);
          _u64 nnbrs = (_u64) (*node_buf);
          T *node_fp_coords = parent->offset_to_node_coords(node_disk_buf);

          T *node_fp_coords_copy = data_buf + (data_buf_idx * parent->aligned_dim);
          data_buf_idx++;
          memcpy(node_fp_coords_copy, node_fp_coords, parent->data_dim * sizeof(T));
          float cur_expanded_dist =
              parent->dist_cmp->compare(query, node_fp_coords_copy, (unsigned) parent->aligned_dim);

          Neighbor n(id, cur_expanded_dist, true);
          full_retset.push_back(n);

          unsigned *node_nbrs = (node_buf + 1);
          // compute node_nbrs <-> query dist in PQ space
          compute_dists(node_nbrs, nnbrs, dist_scratch);

          // process prefetch-ed nhood
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
              float dist = dist_scratch[m];
              if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
                continue;
              Neighbor nn(id, dist, true);
              // variable search_L for deleted nodes.
              // Return position in sorted list where nn inserted.

              auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

              if (cur_list_size < l_search) {
                ++cur_list_size;
              }

              if (r < nk)
                nk = r;
            }
          }
        }

        if (nk <= k)
          k = nk;  // k is the best position in retset updated in this round.
        else
          ++k;
      }

      bool search_ends() {
        // this->print();
        return k >= cur_list_size;
      }
    };

    struct alignas(4096) CoroData {
      CoroDataOne data[kMaxCoroPerThread];
      CoroData(SSDIndex<T> *parent) {
        for (int i = 0; i < kMaxCoroPerThread; ++i) {
          data[i].parent = parent;
        }
      }
    };

    static __thread CoroData *data;
    if (unlikely(data == nullptr)) {
      data = new CoroData(this);
    }

    if (unlikely(N > kMaxCoroPerThread)) {
      LOG(ERROR) << "N > kMaxCoroPerThread";
      exit(-1);
    }
    if (unlikely(data_is_normalized)) {
      LOG(INFO) << "Unsupported yet";
      exit(-1);
    }

    // do not use the thread data's buf.
    QueryBuffer<T> *thread_data = pop_query_buf(queries[0]);
    void *ctx = reader->get_ctx();
    // lambda to batch compute query<-> node distances in PQ space

    for (int v = 0; v < N; ++v) {
      auto &coro_data = data->data[v];
      auto &query1 = queries[v];
      memcpy(coro_data.query, query1, this->data_dim * sizeof(T));

      auto &query = coro_data.query;

      // pointers to buffers for data
      T *data_buf = coro_data.data_buf;
      _mm_prefetch((char *) data_buf, _MM_HINT_T1);

      // query <-> PQ chunk centers distances
      float *pq_dists = coro_data.pq_dists;
      pq_table.populate_chunk_distances(query, pq_dists);

      coro_data.reset();

      _u32 best_medoid = medoids[0];

      if (mem_L) {
        std::vector<unsigned> mem_tags(mem_L);
        std::vector<float> mem_dists(mem_L);
        mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
        coro_data.compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
      } else {
        // Do not use optimized start point.
        coro_data.compute_and_add_to_retset(&best_medoid, 1);
      }
      std::sort(coro_data.retset.begin(), coro_data.retset.begin() + coro_data.cur_list_size);
    }

    // SEARCH!
    for (int i = 0; i < N; ++i) {
      auto &coro_data = data->data[i];
      coro_data.issue_next_io_batch(beam_width, ctx);
    }

    bool all_finished = false;
    while (!all_finished) {
      all_finished = true;
      for (int i = 0; i < N; ++i) {
        auto &coro_data = data->data[i];
        if (!coro_data.search_ends()) {
          all_finished = false;
          if (!coro_data.io_finished(ctx)) {
            continue;
          }
          // LOG(INFO) << "Full retset size: " << coro_data.full_retset.size();
          coro_data.explore_frontier(l_search);
          coro_data.issue_next_io_batch(beam_width, ctx);
        }
      }
    }

    for (int v = 0; v < N; ++v) {
      // re-sort by distance
      auto &full_retset = data->data[v].full_retset;
      std::sort(full_retset.begin(), full_retset.end(),
                [](const Neighbor &left, const Neighbor &right) { return left < right; });

      _u64 t = 0;
      for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
        if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
          continue;  // deduplicate.
        }
        res_tags[v][t] = full_retset[i].id;  // use ID to replace tags
        if (res_dists[v] != nullptr) {
          res_dists[v][t] = full_retset[i].distance;
        }
        t++;
      }
    }

    this->push_query_buf(thread_data);
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace pipeann
