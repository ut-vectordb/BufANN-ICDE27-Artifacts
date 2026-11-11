#pragma once

#include "ann_bench_metrics.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ann_bench {

template<typename T>
inline size_t round_up_dim(size_t dim) {
  return ((dim + 7) / 8) * 8;
}

template<typename T>
struct StreamingQuerySource {
  T *query_data = nullptr;
  size_t query_n = 0;
  size_t dim = 0;
  size_t aligned_dim = 0;
  std::string full_data_bin;
  uint64_t full_npts = 0;
  uint64_t tail_begin = 0;
  uint64_t tail_end = 0;
  size_t tail_block_points = 4096;

  struct ThreadState {
    std::ifstream full_reader;
    std::vector<T> packed;
    std::vector<T> aligned;
    uint64_t block_begin = std::numeric_limits<uint64_t>::max();
    size_t block_len = 0;
    std::vector<uint32_t> result_tags;
    std::vector<float> result_dists;
  };

  void load(const std::string &query_path, const std::string &full_path,
            uint64_t tail_first, uint64_t tail_last) {
    full_data_bin = full_path;
    read_full_metadata();
    tail_begin = tail_first;
    tail_end = std::min<uint64_t>(tail_last, full_npts);
    if (tail_begin > tail_end) {
      throw std::runtime_error("invalid tail query range");
    }
    load_query_bin(query_path);
    if (query_n == 0 && tail_size() == 0) {
      throw std::runtime_error("empty streaming query source");
    }
  }

  void release() {
    if (query_data != nullptr) {
      free(query_data);
      query_data = nullptr;
    }
  }

  ~StreamingQuerySource() { release(); }

  size_t tail_size() const {
    return tail_end > tail_begin ? static_cast<size_t>(tail_end - tail_begin) : 0;
  }

  size_t pool_size() const { return query_n + tail_size(); }

  const T *get(uint64_t seq, ThreadState &state) const {
    const size_t pool = pool_size();
    if (pool == 0) {
      throw std::runtime_error("empty query pool");
    }
    const uint64_t pos = seq % pool;
    if (pos < query_n) {
      return query_data + static_cast<size_t>(pos) * aligned_dim;
    }
    const uint64_t tail_pos = pos - query_n;
    const uint64_t point_id = tail_begin + (tail_pos % tail_size());
    ensure_tail_block(point_id, state);
    return state.aligned.data() + static_cast<size_t>(point_id - state.block_begin) * aligned_dim;
  }

 private:
  void read_full_metadata() {
    std::ifstream in(full_data_bin, std::ios::binary);
    if (!in) {
      throw std::runtime_error("cannot open full data file: " + full_data_bin);
    }
    int32_t n = 0, d = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
    in.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
    if (!in || n <= 0 || d <= 0) {
      throw std::runtime_error("invalid full data header: " + full_data_bin);
    }
    full_npts = static_cast<uint64_t>(n);
    dim = static_cast<size_t>(d);
    aligned_dim = round_up_dim<T>(dim);
  }

  void load_query_bin(const std::string &path) {
    if (path.empty() || path == "null") {
      return;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("cannot open query file: " + path);
    }
    int32_t n = 0, d = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
    in.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
    if (!in || n < 0 || d <= 0) {
      throw std::runtime_error("invalid query header: " + path);
    }
    if (static_cast<size_t>(d) != dim) {
      throw std::runtime_error("query dim mismatch: " + path);
    }
    query_n = static_cast<size_t>(n);
    if (query_n == 0) {
      return;
    }
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, query_n * aligned_dim * sizeof(T)) != 0) {
      throw std::runtime_error("failed to allocate query buffer");
    }
    query_data = reinterpret_cast<T *>(ptr);
    std::fill(query_data, query_data + query_n * aligned_dim, T{});
    std::vector<T> packed(query_n * dim);
    in.read(reinterpret_cast<char *>(packed.data()),
            static_cast<std::streamsize>(packed.size() * sizeof(T)));
    if (!in) {
      throw std::runtime_error("short read from query file: " + path);
    }
    for (size_t i = 0; i < query_n; i++) {
      std::memcpy(query_data + i * aligned_dim, packed.data() + i * dim,
                  dim * sizeof(T));
    }
  }

  void ensure_tail_block(uint64_t point_id, ThreadState &state) const {
    if (point_id >= state.block_begin &&
        point_id < state.block_begin + state.block_len) {
      return;
    }
    if (!state.full_reader.is_open()) {
      state.full_reader.open(full_data_bin, std::ios::binary);
      if (!state.full_reader) {
        throw std::runtime_error("cannot open full data file: " + full_data_bin);
      }
    }
    const uint64_t block_begin =
        tail_begin + ((point_id - tail_begin) / tail_block_points) * tail_block_points;
    const size_t block_len = static_cast<size_t>(
        std::min<uint64_t>(tail_block_points, tail_end - block_begin));
    state.block_begin = block_begin;
    state.block_len = block_len;
    state.packed.resize(block_len * dim);
    state.aligned.assign(block_len * aligned_dim, T{});
    const uint64_t offset =
        2ULL * sizeof(int32_t) + block_begin * dim * sizeof(T);
    state.full_reader.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    state.full_reader.read(reinterpret_cast<char *>(state.packed.data()),
                           static_cast<std::streamsize>(state.packed.size() * sizeof(T)));
    if (!state.full_reader) {
      throw std::runtime_error("short read from tail query range");
    }
    for (size_t i = 0; i < block_len; i++) {
      std::memcpy(state.aligned.data() + i * aligned_dim,
                  state.packed.data() + i * dim, dim * sizeof(T));
    }
  }
};

class LatencySampler {
 public:
  void reset() {
    std::lock_guard<std::mutex> lk(mu_);
    samples_.clear();
    count_.store(0, std::memory_order_relaxed);
    sum_ns_.store(0, std::memory_order_relaxed);
  }

  void record(uint64_t seq, double latency_us) {
    count_.fetch_add(1, std::memory_order_relaxed);
    sum_ns_.fetch_add(static_cast<uint64_t>(latency_us * 1000.0),
                      std::memory_order_relaxed);
    if ((seq & 127ULL) == 0) {
      std::lock_guard<std::mutex> lk(mu_);
      samples_.push_back(latency_us);
    }
  }

  uint64_t count() const { return count_.load(std::memory_order_relaxed); }
  double sum_us() const {
    return static_cast<double>(sum_ns_.load(std::memory_order_relaxed)) / 1000.0;
  }

  LatencyStats summarize() const {
    LatencyStats s;
    const uint64_t n = count();
    s.avg_us = safe_div(sum_us(), static_cast<double>(n));
    std::vector<double> copy;
    {
      std::lock_guard<std::mutex> lk(mu_);
      copy = samples_;
    }
    if (copy.empty()) {
      return s;
    }
    std::sort(copy.begin(), copy.end());
    s.p50_us = percentile_sorted(copy, 0.50);
    s.p90_us = percentile_sorted(copy, 0.90);
    s.p95_us = percentile_sorted(copy, 0.95);
    s.p99_us = percentile_sorted(copy, 0.99);
    return s;
  }

 private:
  mutable std::mutex mu_;
  std::vector<double> samples_;
  std::atomic<uint64_t> count_{0};
  std::atomic<uint64_t> sum_ns_{0};
};

}  // namespace ann_bench
