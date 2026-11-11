#pragma once

#include <atomic>
extern int global_sw;
extern std::atomic<long long> counter;

#include "aligned_file_reader.h"
#include "buffer/BlockCache.hpp"
#include "v2/lock_table.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 public:
  uint64_t file_sz;
  FileHandle file_desc;
  void *bad_ctx = nullptr;

  // yuquan
  char *topo_ptr = nullptr;
  FileHandle coord_file_desc = -1;
  FileHandle topo_file_desc = -1;
  std::vector<uint32_t> phy2loc_coord;
  std::vector<uint32_t> loc2phy_coord;
  std::vector<uint32_t> phy2loc_topo;
  std::vector<uint32_t> loc2phy_topo;

  std::string disk_index_folder;
  int trunc_len = 1e9;

  int strategy;
  // topo buffer
  Serializer serializer;
  // std::vector<int> cache_cb_idxs;
  BlockCache<Block> *block_cache = nullptr;
 public:
  // Size of one vector element in bytes. SSDIndex<T>::load sets it to sizeof(T) before open(), which needs it
  // to derive the graph record length (and from that the topology buffer size) from the index header.
  size_t data_elem_size = sizeof(float);

  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  void *get_ctx(int flag = 0);

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<IORequest> &read_reqs, void *ctx, bool async = false);
  void write(std::vector<IORequest> &write_reqs, void *ctx, bool async = false);
  void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx);
  void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx);

  // read and update cache.
  void read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref = nullptr);
  // read but not update cache.
  int send_read_no_alloc(IORequest &req, void *ctx);
  int send_read_no_alloc(std::vector<IORequest> &reqs, void *ctx);

  void send_io(IORequest &reqs, void *ctx, bool write, FileHandle fd = -1);
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write);
  int poll(void *ctx);
  void poll_all(void *ctx);
  void poll_wait(void *ctx);

  // register thread-id for a context
  void register_thread(int flag = 0);

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();
};

namespace v2 {
  inline std::vector<uint64_t> lockReqs(SparseLockTable<uint64_t> &lock_table, std::vector<IORequest> &reqs) {
    std::vector<uint64_t> ret;
    for (auto &req : reqs) {
      for (uint64_t i = 0; i < req.len; i += SECTOR_LEN) {
        ret.push_back((req.offset + i) / SECTOR_LEN);
      }
    }
    std::sort(ret.begin(), ret.end());
    ret = std::vector<uint64_t>(ret.begin(), std::unique(ret.begin(), ret.end()));
    for (auto &x : ret) {
      lock_table.wrlock(x);
    }
    return ret;
  }

  inline void unlockReqs(SparseLockTable<uint64_t> &lock_table, std::vector<uint64_t> &reqs) {
    for (auto &x : reqs) {
      lock_table.unlock(x);
    }
  }
};  // namespace v2
