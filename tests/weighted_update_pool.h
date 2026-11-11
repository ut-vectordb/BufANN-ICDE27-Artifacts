// Shared weighted-draw scheduler for the mixed-update (concurrent
// insert/delete/query) workload across the baseline drivers
// (DiskANN/Greator/PipeANN). BufANN vendors its own copy under
// vectordb-bench/tests/ since it lives in a separate repo.
//
// Model: one pool of `num_threads` workers drains three finite operation
// streams (INSERT, DELETE, QUERY). The default scheduler repeatedly draws a
// stream with probability proportional to that stream's remaining op count.
// RQ2 can opt into a strict global pattern via run_round_robin().
//
// Per-worker RNG (seed = base_seed + worker_id) keeps runs reproducible without
// lock contention on a shared generator. Op indices are dispatched densely in
// [0, counts[op]) via a lock-free atomic counter, so each task callback may
// write its result/latency into a pre-sized per-index slot with no contention.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ann_bench {

inline uint64_t env_u64_or(const char *name, uint64_t fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(raw, &end, 10);
  if (end == raw || *end != '\0') {
    throw std::runtime_error(std::string("invalid integer env ") + name +
                             "=" + raw);
  }
  return static_cast<uint64_t>(parsed);
}

inline bool env_equals(const char *name, const char *expected) {
  const char *raw = std::getenv(name);
  return raw != nullptr && std::strcmp(raw, expected) == 0;
}

// query_ratio_pct is the desired query share of foreground ops in [1, 99]
// (e.g. 80 -> 80% query; with equal insert/delete each gets 10%).
inline size_t mixed_update_query_count(size_t insert_n, size_t delete_n,
                                       uint32_t query_ratio_pct) {
  if (query_ratio_pct == 0) {
    return 0;
  }
  const uint64_t update_n = static_cast<uint64_t>(insert_n) + delete_n;
  if (update_n == 0) {
    return 0;
  }
  if (query_ratio_pct >= 100) {
    throw std::runtime_error(
        "query_ratio must be in [1, 99] (percent of foreground ops that are "
        "queries)");
  }
  return static_cast<size_t>((static_cast<uint64_t>(query_ratio_pct) * update_n) /
                             (100u - query_ratio_pct));
}

// Vector reserve hint for per-window latency samples (not a flush trigger).
inline uint64_t latency_reserve_hint(uint64_t query_pool_n) {
  return query_pool_n == 0 ? 1 : query_pool_n;
}

class WeightedUpdatePool {
 public:
  enum Op { INSERT = 0, DELETE = 1, QUERY = 2, NUM_OPS = 3 };

  // Fixed default RNG seed; override per run via run(..., base_seed).
  static constexpr uint64_t kDefaultSeed = 0xA5A5A5A5ull;

  // counts[op]  : number of ops to dispatch for each stream.
  // tasks[op]   : executes one op; argument is the dense op index in
  //               [0, counts[op]). A stream with counts[op]==0 is skipped.
  // num_threads : pool size (clamped to >= 1).
  static void run(uint32_t num_threads, const uint64_t counts[NUM_OPS],
                  const std::function<void(uint64_t)> tasks[NUM_OPS],
                  uint64_t base_seed = kDefaultSeed) {
    if (num_threads == 0) {
      num_threads = 1;
    }
    std::atomic<uint64_t> next[NUM_OPS];
    for (int t = 0; t < NUM_OPS; t++) {
      next[t].store(0, std::memory_order_relaxed);
    }

    auto worker = [&](uint32_t wid) {
      std::mt19937_64 rng(base_seed + wid);
      for (;;) {
        uint64_t remaining[NUM_OPS];
        uint64_t total_remaining = 0;
        for (int t = 0; t < NUM_OPS; t++) {
          const uint64_t done = next[t].load(std::memory_order_relaxed);
          remaining[t] = done < counts[t] ? counts[t] - done : 0;
          total_remaining += remaining[t];
        }
        if (total_remaining == 0) {
          return;
        }
        // Draw a stream weighted by its remaining share.
        std::uniform_int_distribution<uint64_t> dist(0, total_remaining - 1);
        uint64_t r = dist(rng);
        int chosen = NUM_OPS - 1;
        for (int t = 0; t < NUM_OPS; t++) {
          if (r < remaining[t]) {
            chosen = t;
            break;
          }
          r -= remaining[t];
        }
        // Claim a dense index; if another worker drained this stream between the
        // weight snapshot and the claim, redraw.
        const uint64_t idx =
            next[chosen].fetch_add(1, std::memory_order_relaxed);
        if (idx >= counts[chosen]) {
          continue;
        }
        tasks[chosen](idx);
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t w = 0; w < num_threads; w++) {
      threads.emplace_back(worker, w);
    }
    for (auto &th : threads) {
      th.join();
    }
  }

  // Dispatch a global cyclic pattern. For RQ2's 90/5/5 mix, callers pass
  // {QUERY x18, DELETE, INSERT}. If one stream exhausts, workers skip those
  // pattern slots and keep draining the others (same interleaving as before).
  // Exit when every stream index is claimed, like run(), so idle workers do
  // not busy-spin waiting on a stuck in-flight op.
  static void run_round_robin(
      uint32_t num_threads, const uint64_t counts[NUM_OPS],
      const std::function<void(uint64_t)> tasks[NUM_OPS],
      const std::vector<Op> &pattern) {
    if (pattern.empty()) {
      throw std::runtime_error("round-robin pattern must not be empty");
    }
    if (num_threads == 0) {
      num_threads = 1;
    }

    std::atomic<uint64_t> next_seq{0};
    std::atomic<uint64_t> next[NUM_OPS];
    for (int t = 0; t < NUM_OPS; t++) {
      next[t].store(0, std::memory_order_relaxed);
    }

    auto worker = [&]() {
      for (;;) {
        uint64_t total_remaining = 0;
        for (int t = 0; t < NUM_OPS; t++) {
          const uint64_t claimed = next[t].load(std::memory_order_relaxed);
          if (claimed < counts[t]) {
            total_remaining += counts[t] - claimed;
          }
        }
        if (total_remaining == 0) {
          return;
        }

        const uint64_t seq = next_seq.fetch_add(1, std::memory_order_relaxed);
        const Op chosen = pattern[seq % pattern.size()];
        const uint64_t idx =
            next[chosen].fetch_add(1, std::memory_order_relaxed);
        if (idx >= counts[chosen]) {
          // Exhausted pattern slot: skip and redraw (RQ2 schedule unchanged).
          continue;
        }
        tasks[chosen](idx);
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t w = 0; w < num_threads; w++) {
      threads.emplace_back(worker);
    }
    for (auto &th : threads) {
      th.join();
    }
  }


  static void run_rq2_round_robin_90_5_5(
      uint32_t num_threads, const uint64_t counts[NUM_OPS],
      const std::function<void(uint64_t)> tasks[NUM_OPS]) {
    std::vector<Op> pattern;
    pattern.reserve(20);
    for (int i = 0; i < 18; i++) {
      pattern.push_back(QUERY);
    }
    pattern.push_back(DELETE);
    pattern.push_back(INSERT);
    run_round_robin(num_threads, counts, tasks, pattern);
  }

};

}  // namespace ann_bench
