#include "global_stats.h"
#define USE_AIO
#ifndef USE_AIO
#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "liburing.h"

#define MAX_EVENTS 256

namespace {
  constexpr uint64_t kNoUserData = 0;
  void execute_io(void *context, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    io_uring *ring = (io_uring *) context;
    while (true) {
      for (uint64_t j = 0; j < reqs.size(); j++) {
        auto sqe = io_uring_get_sqe(ring);
        sqe->user_data = kNoUserData;
        if (write) {
          io_uring_prep_write(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        } else {
          io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        }
      }
      io_uring_submit(ring);

      io_uring_cqe *cqe = nullptr;
      bool fail = false;
      for (uint64_t j = 0; j < reqs.size(); j++) {
        int ret = 0;
        do {
          ret = io_uring_wait_cqe(ring, &cqe);
        } while (ret == -EINTR);

        if (ret < 0 || cqe->res < 0) {
          fail = true;
          LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << j << " " << reqs[j].buf << " "
                     << reqs[j].len << " " << reqs[j].offset;
          break;  // CQE broken.
        }
        io_uring_cqe_seen(ring, cqe);
      }
      if (!fail) {  // repeat until no fails.
        break;
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_uring *ring = nullptr;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ring == nullptr)) {
    register_thread(flag);
  }
  return ioctx::ring;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ring == nullptr) {
    ioctx::ring = new io_uring();
    io_uring_queue_init(MAX_EVENTS, ioctx::ring, flag);
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  io_uring_queue_exit(ioctx::ring);
  delete ioctx::ring;
  ioctx::ring = nullptr;
}

void LinuxAlignedFileReader::deregister_all_threads() {
  return;
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);

  // error checks
  assert(this->file_desc != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  auto sqe = io_uring_get_sqe(ring);
  req.finished = false;
  sqe->user_data = (uint64_t) &req;
  if (write) {
    io_uring_prep_write(sqe, this->file_desc, req.buf, req.len, req.offset);
  } else {
    io_uring_prep_read(sqe, this->file_desc, req.buf, req.len, req.offset);
  }
  io_uring_submit(ring);
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  for (uint64_t j = 0; j < reqs.size(); j++) {
    auto sqe = io_uring_get_sqe(ring);
    reqs[j].finished = false;
    sqe->user_data = (uint64_t) &reqs[j];
    if (write) {
      io_uring_prep_write(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_uring_prep_read(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
  }
  io_uring_submit(ring);
}

int LinuxAlignedFileReader::poll(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = io_uring_peek_cqe(ring, &cqe);
  if (ret < 0) {
    return ret;  // not finished yet.
  }
  if (cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *) cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
  return 0;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  static __thread io_uring_cqe *cqes[MAX_EVENTS];
  int ret = io_uring_peek_batch_cqe(ring, cqes, MAX_EVENTS);
  if (ret < 0) {
    return;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    if (cqes[i]->res < 0) {
      LOG(ERROR) << "Failed " << strerror(-cqes[i]->res);
    }
    IORequest *req = (IORequest *) cqes[i]->user_data;
    if (req != nullptr) {
      req->finished = true;
    }
    io_uring_cqe_seen(ring, cqes[i]);
  }
}

void LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = 0;
  do {
    ret = io_uring_wait_cqe(ring, &cqe);
  } while (ret == -EINTR);
  if (ret < 0 || cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *) cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
}

#else
#include "linux_aligned_file_reader.h"

#include <libaio.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "ann_bench_metrics.h"

int global_sw = 0;
std::atomic<long long> counter(0);

#define MAX_EVENTS 1024

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb iocb_t;

  // Harness logical-I/O accounting (no-op unless a driver enabled the counters). A single-page request counts
  // as random; a multi-page request (delete consolidation's batched sector reads and writes) as sequential.
  inline void account_io(uint64_t len, bool write) {
    if (len > SECTOR_LEN) {
      write ? ann_bench::LogicalIoCounter::account_seq_write(1, len)
            : ann_bench::LogicalIoCounter::account_seq_read(1, len);
    } else {
      write ? ann_bench::LogicalIoCounter::account_random_write(1, len)
            : ann_bench::LogicalIoCounter::account_random_read(1, len);
    }
  }

  void execute_io(void *ctx, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops = std::min((uint64_t) reqs.size() - (iter * MAX_EVENTS), (uint64_t) MAX_EVENTS);
      std::vector<iocb_t *> cbs(n_ops, nullptr);
      std::vector<io_event_t> evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        account_io(reqs[j + iter * MAX_EVENTS].len, write);
        if (write) {
          io_prep_pwrite(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                         reqs[j + iter * MAX_EVENTS].offset);
        } else {
          io_prep_pread(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                        reqs[j + iter * MAX_EVENTS].offset);
        }
      }

      // initialize `cbs` using `cb` array
      //

      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      uint64_t n_tries = 0;
      while (n_tries <= n_retries) {
        // issue reads
        int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno << "="
                     << ::strerror((int) -ret) << ", try #" << n_tries + 1 << " ctx: " << ctx << "\n";
          exit(-1);
        } else {
          // wait on io_getevents
          ret = io_getevents((io_context_t) ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(), nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            LOG(ERROR) << "io_getevents() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                       << "=" << ::strerror((int) -ret) << ", try #" << n_tries + 1;
            exit(-1);
          } else {
            break;
          }
        }
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  // Each thread's AIO context, destroyed when the thread exits (benchmark harness fix). Upstream never
  // destroyed it, so a driver that starts fresh threads every round leaks one context per thread per round
  // until io_setup fails with EAGAIN at fs.aio-max-nr, which the multi-round RQ4 run hit after 9 rounds.
  struct ThreadCtx {
    io_context_t ctx = nullptr;
    ~ThreadCtx() {
      if (ctx != nullptr) {
        io_destroy(ctx);
      }
    }
  };
  static thread_local ThreadCtx tctx;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::tctx.ctx == nullptr)) {
    register_thread(flag);
  }
  return (void *) ioctx::tctx.ctx;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::tctx.ctx == nullptr) {
    int ret = io_setup(MAX_EVENTS, &ioctx::tctx.ctx);
    if (ret != 0) {
      // Without a context every later io_getevents fails and the caller retries forever, so stop here.
      LOG(ERROR) << "io_setup() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
      exit(-1);
    }
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  if (ioctx::tctx.ctx != nullptr) {
    io_destroy(ioctx::tctx.ctx);
    ioctx::tctx.ctx = nullptr;
  }
}

void LinuxAlignedFileReader::deregister_all_threads() {
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->file_desc != -1);
  if(this->file_desc == -1){
    std::cerr << "open file failed: " << fname << std::endl;
    exit(-1);
  }
  //  std::cerr << "Opened file : " << fname << std::endl;
  if(1)
  {// yuquan
    // this->strategy = 9;
    std::cout << "***************************\n";
    std::cout << "STRATEGY: " << strategy << "\n";
    std::cout << "***************************\n";
    size_t pos = fname.find_last_of("/\\");
    this->disk_index_folder = (pos != std::string::npos) ? fname.substr(0, pos + 1) : ".";
    std::cerr << "Index folder is " << disk_index_folder << "\n";
    int use_rerank = (strategy >> 0) & 0x1;
    int use_topo_reorder = (strategy >> 1) & 0x1;
    int use_double_pq = (strategy >> 2) & 0x1;
    int use_truncate = (strategy >> 6) & 0x1;
    int use_triple_pq = (strategy >> 7) & 0x1;
    int use_coord_reorder = (strategy >> 3) & 0x1;
    int use_topo_buffer = (strategy >> 4) & 0x1;

    if (use_truncate || use_double_pq || use_triple_pq) {
    // if(use_double_pq){
      this->trunc_len = 388;
      LOG(INFO) << "pq trunc_len: " << this->trunc_len;
    }
    if (use_rerank) {// use_rerank
      std::ifstream reader(fname, std::ios::binary);
      if (!reader) {
          std::cerr << "Failed to open: " << fname << std::endl;
          exit(-1);
      }

      int meta_npts, meta_ndims;
      reader.read((char *)&meta_npts, sizeof(int));
      reader.read((char *)&meta_ndims, sizeof(int));

      std::vector<uint64_t> meta(meta_npts);
      for (auto &x : meta) {
          reader.read((char *)&x, sizeof(uint64_t));
      }

      uint64_t npts = meta[0];
      uint64_t ndims = meta[1];
      uint64_t max_node_len = meta[3];
      // uint64_t nnodes_per_sector = meta[4];
      // Upstream hard-coded sizeof(float) here, which for uint8/int8 data underflows and happens to size the
      // topology buffer as if vectors were 4 bytes per dimension (4x too large for uint8).
      uint64_t max_node_degree = (max_node_len - ndims * data_elem_size - sizeof(uint32_t)) / sizeof(uint32_t);
      uint64_t topo_len = (max_node_degree + 1) * sizeof(uint32_t);
      uint64_t ntopo_per_sector = SECTOR_LEN / topo_len;
      reader.close();

      std::string coord_path = disk_index_folder + "disk_index_data";

      if(use_coord_reorder){
        // coord_path = disk_index_folder + "reordered_disk_index_data";
        // std::string reorder_map_file_path = disk_index_folder + "reorder_map_data";
        coord_path = disk_index_folder + "reordered_disk_index_data_2";
        std::string reorder_map_file_path = disk_index_folder + "reorder_map_data_2";
        std::ifstream map_reader(reorder_map_file_path.c_str(), std::ios::binary);
        if (!map_reader) {
          LOG(ERROR) << "Failed to open map file for reading: " << reorder_map_file_path;
          exit(-1);
        }

        uint64_t size;
        map_reader.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
        this->loc2phy_coord.resize(size);
        this->phy2loc_coord.resize(size * 2);
        
        map_reader.read(reinterpret_cast<char*>(loc2phy_coord.data()), size * sizeof(uint32_t));

        map_reader.close();
        for(uint32_t i = 0; i < loc2phy_coord.size(); i ++){
          this->phy2loc_coord[loc2phy_coord[i]] = i;	
        }
        std::cerr << "Load Data loc2phy and phy2loc map done.\n";
      }
      this->coord_file_desc = ::open(coord_path.c_str(), O_RDWR | O_LARGEFILE | O_DIRECT);
      if (this->coord_file_desc == -1) {
        LOG(ERROR) << "Open coord file " << coord_path << " fail!";
        exit(1);
      }
      std::cout << "coord_path is " << coord_path << std::endl;

      std::string topo_path, reordered_topo_path;
      if(1) // use_topo_disk
      {
        topo_path = disk_index_folder + "disk_index_graph";
        if (use_topo_reorder) {
          // topo_path = disk_index_folder + "reordered_disk_index_graph";
          topo_path = disk_index_folder + "reordered_disk_index_graph_2";
          // std::string reorder_map_file_path = disk_index_folder + "reorder_map_graph";
          std::string reorder_map_file_path = disk_index_folder + "reorder_map_graph_2";
          std::ifstream map_reader(reorder_map_file_path.c_str(), std::ios::binary);
          if (!map_reader) {
            LOG(ERROR) << "Failed to open map file for reading: " << reorder_map_file_path;
            exit(-1);
          }

          uint64_t size;
          map_reader.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
          this->loc2phy_topo.resize(size);
          // this->phy2loc_topo.resize(size * 2);
          
          map_reader.read(reinterpret_cast<char*>(loc2phy_topo.data()), size * sizeof(uint32_t));

          map_reader.close();
          // for(uint32_t i = 0; i < loc2phy_topo.size(); i ++){
          //   this->phy2loc_topo[loc2phy_topo[i]] = i;	
          // }
          std::cerr << "Load Graph loc2phy and phy2loc map done.\n";
        }
        std::cout << "topo_path is " << topo_path << std::endl;
        this->topo_file_desc = ::open(topo_path.c_str(), O_RDWR | O_LARGEFILE | O_DIRECT);
        if (topo_file_desc == -1) {
          perror("Open topo file fail!\n");
          exit(1);
        }
        if (use_topo_buffer) {
          // this->serializer.open_file(topo_path, MODE::SYNC_READ);
          // int dyn_buffer_size = npts / ntopo_per_sector * 4 / 100; //sift
          std::cerr << "max_node_len is " << max_node_len << "\n";
          std::cerr << "topo_len is " << topo_len << "\n";
          int dyn_buffer_size = std::ceil(0.01 * npts * (max_node_len - topo_len) / SECTOR_LEN); //gist
          dyn_buffer_size = std::max(dyn_buffer_size, 800 * 10);//要大于线程数*max(L,L_disk)
          // DGAI_TOPO_BUFFER_PAGES (added for the benchmark harness) overrides the size, for experiments.
          if (const char *pages = std::getenv("DGAI_TOPO_BUFFER_PAGES"); pages != nullptr && *pages != '\0') {
            dyn_buffer_size = std::atoi(pages);
          }
          // dyn_buffer_size = std::max(dyn_buffer_size, 388140);//要大于线程数*max(L,L_disk)
          LOG(INFO) << "dyn_buffer_size: " << dyn_buffer_size << " * 4k";
          this->block_cache = new BlockCache<Block>(nullptr, dyn_buffer_size);
        }
      } 
      // if (use_dram)
      // {
      //   topo_path = disk_index_folder + "dram_index_graph";
        
      //   std::cout << "topo_path is " << topo_path << std::endl;
      //   int topo_file_desc = ::open(topo_path.c_str(), O_RDWR | O_LARGEFILE);
      //   if (topo_file_desc == -1) {
      //     perror("Open topo file fail!\n");
      //     exit(1);
      //   }

      //   size_t topo_file_size = lseek(topo_file_desc, 0, SEEK_END);
      //   if (topo_file_size == (size_t) - 1) {
      //     perror("lseek topo file fail!");
      //     ::close(topo_file_desc);
      //     exit(1);
      //   }
      //   uint32_t max_degree = (topo_file_size / npts / sizeof(uint32_t) - 1);
      //   // uint32_t ori_topo_len = (max_degree + 1) * sizeof(uint32_t);
      //   uint32_t topo_len = (max_degree + 1) * sizeof(uint32_t);
      //   // uint32_t topo_len = (std::ceil(max_degree * GRAPH_SLACK_FACTOR) + 1) * sizeof(uint32_t);

      //   std::cerr << "ori_topo_len is " << ori_topo_len << "\n";
      //   std::cerr << "topo_len is " << topo_len << "\n";
      //   // exit(0);
      //   this->topo_ptr = (char *)malloc((uint64_t)(topo_len * npts));
      //   for (uint32_t i = 0; i < npts; ++i) {
      //     pread(topo_file_desc, this->topo_ptr + i * topo_len, ori_topo_len, i * ori_topo_len);
      //   }
      //   ::close(topo_file_desc);
      // }
      
      // std::cerr << "Open topo file success!\n";
      // int block_id = req.offset / BLOCK_SIZE;
      // int offset = req.offset % BLOCK_SIZE;
      // int cb_idx = this->block_cache->request_block(block_id);
      // Block* b_ptr = this->block_cache->get_cache_block(cb_idx, block_id);

      // std::memcpy(req.buf, (char *)b_ptr + offset, req.len);

      // this->block_cache->release_cache_block(cb_idx, b_ptr);

      // exit(-1);
      
    }
  }

}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  uint64_t n_ops = std::min(reqs.size(), (uint64_t) MAX_EVENTS);
  std::vector<iocb_t *> cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (uint64_t j = 0; j < n_ops; j++) {
    account_io(reqs[j].len, write);
    if (write) {
      io_prep_pwrite(cb.data() + j, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_prep_pread(cb.data() + j, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
    reqs[j].finished = false;  // reset finished flag
  }

  for (uint64_t i = 0; i < n_ops; i++) {
    cbs[i] = cb.data() + i;
  }

  // issue reads
  int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
  // if requests didn't get accepted
  if (ret != (int64_t) n_ops) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", " << strerror(errno);
  }
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write, FileHandle fd) {
  iocb_t cb;
  req.finished = false;  // reset finished flag
  account_io(req.len, write);
  if (write) {
    io_prep_pwrite(&cb, fd != -1 ? fd : this->file_desc, req.buf, req.len, req.offset);
  } else {
    io_prep_pread(&cb, fd != -1 ? fd : this->file_desc, req.buf, req.len, req.offset);
  }
  cb.data = (void *) &req;  // set user data to point to the request

  iocb_t *cbs[1] = {&cb};  // create an array of iocb_t pointers
  int ret = io_submit((io_context_t) ctx, 1, cbs);
  if (ret != 1) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
  }
}

int LinuxAlignedFileReader::poll(void *ctx) {
  // Poll a single completed IO request in the io_uring context.
  io_event event;
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, 1, &event, nullptr);
  if (ret < 0) {
    return ret;  // not finished yet.
  }
  if (ret) {
    IORequest *req = (IORequest *) event.data;
    if (req != nullptr) {
      req->finished = true;
    }
  }
  return 0;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  // Poll all completed IO requests in the io_uring context.
  static __thread io_event_t evts[MAX_EVENTS];
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, MAX_EVENTS, evts, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    IORequest *req = (IORequest *) evts[i].data;
    if (req != nullptr) {
      // std::cerr << (void *)req->buf << "finished\n";
      req->finished = true;
    }
  }
}

void LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_event_t event;
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 1, 1, &event, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;  // not finished yet.
  }
  IORequest *req = (IORequest *) event.data;
  if (req != nullptr) {
    req->finished = true;
  }
}

#endif

int LinuxAlignedFileReader::send_read_no_alloc(IORequest &req, void *ring) {
#ifndef READ_ONLY_TESTS
  if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf)) {
    send_io(req, ring, false);
  } else {
    req.finished = true;  // mark as finished for cache miss
  }
#else
  send_io(req, ring, false);
#endif
  return 1;
}

int LinuxAlignedFileReader::send_read_no_alloc(std::vector<IORequest> &reqs, void *ring) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;
  // fetch from cache.
  for (auto &req : reqs) {
    if (req.offset % SECTOR_LEN != 0 || req.len != SECTOR_LEN) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
    }
    if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf)) {
      disk_read_reqs.push_back(req);
    }
  }
  send_io(disk_read_reqs, ring, false);
  return disk_read_reqs.size();
#else
  send_io(reqs, ring, false);
  return reqs.size();
#endif
}

void LinuxAlignedFileReader::read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;

  // TODO(gh): introduce size_per_io to cache.
  for (auto &req : read_reqs) {
    if (req.offset % SECTOR_LEN != 0) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
      crash();
    }
    if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf, true)) {
      disk_read_reqs.push_back(req);
    }
  }

  if (disk_read_reqs.size() > 0) {
    read(disk_read_reqs, ctx);
    if (gs != nullptr) {
      gs->read_ios += disk_read_reqs.size();
    }
    for (auto &req : disk_read_reqs) {
      v2::cache.put(req.offset / SECTOR_LEN, (uint8_t *) req.buf, true);
    }
  }

  // ref.
  if (page_ref != nullptr) {
    for (auto &req : read_reqs) {
      page_ref->push_back(req.offset / SECTOR_LEN);
    }
  }
#else
  read(read_reqs, ctx);
#endif
}