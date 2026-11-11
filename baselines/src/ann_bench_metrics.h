#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WINDOWS
#include <unistd.h>
#endif

namespace ann_bench {

inline double safe_div(double num, double den) {
  return den > 0.0 ? num / den : 0.0;
}

// RQ2 time-window JSON metrics flush every QUERY_WINDOW_INTERVAL_MS
// milliseconds (default 1000). A value of 0 disables timer-driven auto flushes;
// phase-boundary and workload-final flushes still drain the recorder.
inline uint64_t query_window_interval_ms() {
  const char *raw = std::getenv("QUERY_WINDOW_INTERVAL_MS");
  uint64_t interval_ms = 1000;
  if (raw != nullptr && *raw != '\0') {
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end != raw && *end == '\0') {
      interval_ms = static_cast<uint64_t>(parsed);
    }
  }
  return interval_ms;
}

// Vector reserve hint for per-window latency samples (not a flush trigger).
inline uint64_t latency_reserve_hint(uint64_t query_pool_n) {
  return query_pool_n == 0 ? 1 : query_pool_n;
}

struct LogicalIoSnapshot {
  uint64_t read_seq_ios = 0;
  uint64_t write_seq_ios = 0;
  uint64_t read_random_ios = 0;
  uint64_t write_random_ios = 0;
  uint64_t read_seq_bytes = 0;
  uint64_t write_seq_bytes = 0;
  uint64_t read_random_bytes = 0;
  uint64_t write_random_bytes = 0;
};

class LogicalIoCounter {
 public:
  static void reset_and_enable() {
    read_seq_ios().store(0, std::memory_order_relaxed);
    write_seq_ios().store(0, std::memory_order_relaxed);
    read_random_ios().store(0, std::memory_order_relaxed);
    write_random_ios().store(0, std::memory_order_relaxed);
    read_seq_bytes().store(0, std::memory_order_relaxed);
    write_seq_bytes().store(0, std::memory_order_relaxed);
    read_random_bytes().store(0, std::memory_order_relaxed);
    write_random_bytes().store(0, std::memory_order_relaxed);
    enabled().store(true, std::memory_order_release);
  }

  static LogicalIoSnapshot snapshot() {
    LogicalIoSnapshot s;
    s.read_seq_ios = read_seq_ios().load(std::memory_order_relaxed);
    s.write_seq_ios = write_seq_ios().load(std::memory_order_relaxed);
    s.read_random_ios = read_random_ios().load(std::memory_order_relaxed);
    s.write_random_ios = write_random_ios().load(std::memory_order_relaxed);
    s.read_seq_bytes = read_seq_bytes().load(std::memory_order_relaxed);
    s.write_seq_bytes = write_seq_bytes().load(std::memory_order_relaxed);
    s.read_random_bytes = read_random_bytes().load(std::memory_order_relaxed);
    s.write_random_bytes = write_random_bytes().load(std::memory_order_relaxed);
    return s;
  }

  static LogicalIoSnapshot snapshot_and_disable() {
    LogicalIoSnapshot s = snapshot();
    enabled().store(false, std::memory_order_release);
    return s;
  }

  static void disable() { enabled().store(false, std::memory_order_release); }

  // total_bytes is the sum across all ios in this accounting event.
  static void account_seq_read(uint64_t ios, uint64_t total_bytes) {
    if (!enabled().load(std::memory_order_acquire) || ios == 0) {
      return;
    }
    read_seq_ios().fetch_add(ios, std::memory_order_relaxed);
    read_seq_bytes().fetch_add(total_bytes, std::memory_order_relaxed);
  }

  static void account_seq_write(uint64_t ios, uint64_t total_bytes) {
    if (!enabled().load(std::memory_order_acquire) || ios == 0) {
      return;
    }
    write_seq_ios().fetch_add(ios, std::memory_order_relaxed);
    write_seq_bytes().fetch_add(total_bytes, std::memory_order_relaxed);
  }

  static void account_random_read(uint64_t ios, uint64_t total_bytes) {
    if (!enabled().load(std::memory_order_acquire) || ios == 0) {
      return;
    }
    read_random_ios().fetch_add(ios, std::memory_order_relaxed);
    read_random_bytes().fetch_add(total_bytes, std::memory_order_relaxed);
  }

  static void account_random_write(uint64_t ios, uint64_t total_bytes) {
    if (!enabled().load(std::memory_order_acquire) || ios == 0) {
      return;
    }
    write_random_ios().fetch_add(ios, std::memory_order_relaxed);
    write_random_bytes().fetch_add(total_bytes, std::memory_order_relaxed);
  }

 private:
  static std::atomic_bool &enabled() {
    static std::atomic_bool value{false};
    return value;
  }
  static std::atomic<uint64_t> &read_seq_ios() {
    static std::atomic<uint64_t> value{0};
    return value;
  }
  static std::atomic<uint64_t> &write_seq_ios() {
    static std::atomic<uint64_t> value{0};
    return value;
  }
  static std::atomic<uint64_t> &read_random_ios() {
    static std::atomic<uint64_t> value{0};
    return value;
  }
  static std::atomic<uint64_t> &write_random_ios() {
    static std::atomic<uint64_t> value{0};
    return value;
  }
  static std::atomic<uint64_t> &read_seq_bytes() {
    static std::atomic<uint64_t> value{0};
    return value;
  }
  static std::atomic<uint64_t> &write_seq_bytes() {
    static std::atomic<uint64_t> value{0};
    return value;
  }
  static std::atomic<uint64_t> &read_random_bytes() {
    static std::atomic<uint64_t> value{0};
    return value;
  }
  static std::atomic<uint64_t> &write_random_bytes() {
    static std::atomic<uint64_t> value{0};
    return value;
  }
};

inline void add_logical_io(LogicalIoSnapshot &total,
                           const LogicalIoSnapshot &delta) {
  total.read_seq_ios += delta.read_seq_ios;
  total.write_seq_ios += delta.write_seq_ios;
  total.read_random_ios += delta.read_random_ios;
  total.write_random_ios += delta.write_random_ios;
  total.read_seq_bytes += delta.read_seq_bytes;
  total.write_seq_bytes += delta.write_seq_bytes;
  total.read_random_bytes += delta.read_random_bytes;
  total.write_random_bytes += delta.write_random_bytes;
}

inline double current_rss_mb() {
#ifndef _WINDOWS
  std::ifstream in("/proc/self/statm");
  uint64_t pages = 0;
  uint64_t resident = 0;
  in >> pages >> resident;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (!in || page_size <= 0) {
    return 0.0;
  }
  return static_cast<double>(resident) * static_cast<double>(page_size) /
         (1024.0 * 1024.0);
#else
  return 0.0;
#endif
}

struct RssStats {
  double avg_mb = 0.0;
  double peak_mb = 0.0;
  uint64_t sample_count = 0;
};

class RssSampler {
 public:
  void start() {
    stop();
    samples_.clear();
    interval_begin_ = 0;
    start_worker();
  }

  void resume() {
    if (!worker_.joinable()) {
      start_worker();
    }
  }

  void stop() {
    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
      worker_.join();
    }
    if (samples_.empty()) {
      samples_.push_back(current_rss_mb());
    }
  }

  RssStats stop_and_take_interval() {
    stop();
    RssStats stats;
    if (interval_begin_ > samples_.size()) {
      interval_begin_ = samples_.size();
    }
    stats.sample_count =
        static_cast<uint64_t>(samples_.size() - interval_begin_);
    if (stats.sample_count > 0) {
      double sum_mb = 0.0;
      for (size_t i = interval_begin_; i < samples_.size(); ++i) {
        sum_mb += samples_[i];
        if (samples_[i] > stats.peak_mb) {
          stats.peak_mb = samples_[i];
        }
      }
      stats.avg_mb = sum_mb / static_cast<double>(stats.sample_count);
    }
    interval_begin_ = samples_.size();
    return stats;
  }

  ~RssSampler() {
    stop();
    if (lifecycle_started_) {
      std::clog << "[ann_bench::RssSampler] event=teardown current_rss_mb=" << current_rss_mb() << std::endl;
    }
  }

  double avg_mb() const {
    if (samples_.empty()) {
      return 0.0;
    }
    return std::accumulate(samples_.begin(), samples_.end(), 0.0) /
           static_cast<double>(samples_.size());
  }

  double peak_mb() const {
    if (samples_.empty()) {
      return 0.0;
    }
    return *std::max_element(samples_.begin(), samples_.end());
  }

 private:
  void start_worker() {
    if (!lifecycle_started_) {
      lifecycle_started_ = true;
      std::clog << "[ann_bench::RssSampler] event=startup current_rss_mb=" << current_rss_mb() << std::endl;
    }
    stop_.store(false, std::memory_order_release);
    worker_ = std::thread([this]() {
      while (!stop_.load(std::memory_order_acquire)) {
        samples_.push_back(current_rss_mb());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      samples_.push_back(current_rss_mb());
    });
  }

  std::atomic_bool stop_{true};
  std::thread worker_;
  std::vector<double> samples_;
  size_t interval_begin_ = 0;
  bool lifecycle_started_ = false;
};

struct LatencyStats {
  double avg_us = 0.0;
  double p50_us = 0.0;
  double p90_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
};

inline double percentile_sorted(const std::vector<double> &v, double p) {
  if (v.empty()) {
    return 0.0;
  }
  size_t idx = static_cast<size_t>(p * static_cast<double>(v.size()));
  if (idx >= v.size()) {
    idx = v.size() - 1;
  }
  return v[idx];
}

inline LatencyStats summarize_latencies(std::vector<double> latencies_us) {
  LatencyStats s;
  if (latencies_us.empty()) {
    return s;
  }
  s.avg_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) /
             static_cast<double>(latencies_us.size());
  std::sort(latencies_us.begin(), latencies_us.end());
  s.p50_us = percentile_sorted(latencies_us, 0.50);
  s.p90_us = percentile_sorted(latencies_us, 0.90);
  s.p95_us = percentile_sorted(latencies_us, 0.95);
  s.p99_us = percentile_sorted(latencies_us, 0.99);
  return s;
}

inline std::string json_escape(const std::string &value) {
  std::ostringstream out;
  for (char c : value) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

inline void json_kv(std::ostream &out, bool &first, const std::string &key,
                    const std::string &value) {
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << json_escape(key) << "\":\"" << json_escape(value) << "\"";
}

inline void json_kv(std::ostream &out, bool &first, const std::string &key,
                    const char *value) {
  json_kv(out, first, key, std::string(value));
}

inline void json_kv(std::ostream &out, bool &first, const std::string &key,
                    double value) {
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << json_escape(key) << "\":" << std::setprecision(12) << value;
}

inline void json_kv(std::ostream &out, bool &first, const std::string &key,
                    uint64_t value) {
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << json_escape(key) << "\":" << value;
}

inline void json_kv(std::ostream &out, bool &first, const std::string &key,
                    int64_t value) {
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << json_escape(key) << "\":" << value;
}

inline void emit_rss_stats(std::ostream &out, bool &first,
                           const RssStats &rss) {
  json_kv(out, first, "avg_rss_mb", rss.avg_mb);
  json_kv(out, first, "peak_rss_mb", rss.peak_mb);
  json_kv(out, first, "rss_sample_count", rss.sample_count);
}

inline void emit_logical_io(std::ostream &out, bool &first,
                            const LogicalIoSnapshot &io, double time_s,
                            uint64_t op_count) {
  (void)time_s;
  (void)op_count;
  json_kv(out, first, "io_read_seq_ios", io.read_seq_ios);
  json_kv(out, first, "io_write_seq_ios", io.write_seq_ios);
  json_kv(out, first, "io_read_random_ios", io.read_random_ios);
  json_kv(out, first, "io_write_random_ios", io.write_random_ios);
  json_kv(out, first, "io_read_seq_bytes", io.read_seq_bytes);
  json_kv(out, first, "io_write_seq_bytes", io.write_seq_bytes);
  json_kv(out, first, "io_read_random_bytes", io.read_random_bytes);
  json_kv(out, first, "io_write_random_bytes", io.write_random_bytes);
}

inline void emit_latency(std::ostream &out, bool &first,
                         const std::string &prefix, const LatencyStats &s) {
  json_kv(out, first, prefix + "_lat_avg_us", s.avg_us);
  json_kv(out, first, prefix + "_lat_p50_us", s.p50_us);
  json_kv(out, first, prefix + "_lat_p90_us", s.p90_us);
  json_kv(out, first, prefix + "_lat_p95_us", s.p95_us);
  json_kv(out, first, prefix + "_lat_p99_us", s.p99_us);
}

class QueryCycleWindowMetrics {
 public:
  ~QueryCycleWindowMetrics() {
    pause_rss();
    if (enabled_) std::cout << "[ann_bench::QueryCycleWindowMetrics] event=teardown current_rss_mb=" << current_rss_mb() << std::endl;
  }

  void start(const std::string &baseline, uint64_t latency_reserve_hint) {
    std::lock_guard<std::mutex> lk(flush_mu_);
    pause_rss();
    baseline_ = baseline;
    latency_reserve_hint_ = latency_reserve_hint == 0 ? 1 : latency_reserve_hint;
    window_interval_ms_ = query_window_interval_ms();
    enabled_ = true;
    std::cout << "[ann_bench::QueryCycleWindowMetrics] event=startup current_rss_mb=" << current_rss_mb() << std::endl;
    workload_start_ = std::chrono::steady_clock::now();
    window_start_ = workload_start_;
    next_flush_time_ =
        workload_start_ + std::chrono::milliseconds(window_interval_ms_);
    active_elapsed_s_ = 0.0;
    current_round_index_.store(0, std::memory_order_relaxed);
    current_stage_id_.store(0, std::memory_order_relaxed);
    window_index_ = 0;
    window_query_count_.store(0, std::memory_order_relaxed);
    window_insert_count_.store(0, std::memory_order_relaxed);
    window_delete_count_.store(0, std::memory_order_relaxed);
    query_latency_count_.store(0, std::memory_order_relaxed);
    query_latency_sum_ns_.store(0, std::memory_order_relaxed);
    insert_latency_count_.store(0, std::memory_order_relaxed);
    insert_latency_sum_ns_.store(0, std::memory_order_relaxed);
    delete_latency_count_.store(0, std::memory_order_relaxed);
    delete_latency_sum_ns_.store(0, std::memory_order_relaxed);
    in_flight_records_.store(0, std::memory_order_relaxed);
    flushing_.store(false, std::memory_order_release);
    last_io_snapshot_ = LogicalIoSnapshot();
    emit_io_ = false;
    reset_rss_window_locked();
    round_rss_sum_mb_ = 0.0;
    round_rss_sample_count_ = 0;
    round_peak_rss_mb_ = 0.0;
    workload_rss_sum_mb_ = 0.0;
    workload_rss_sample_count_ = 0;
    workload_peak_rss_mb_ = 0.0;
    workload_query_latency_count_ = 0;
    workload_query_latency_sum_ns_ = 0;
    workload_insert_latency_count_ = 0;
    workload_insert_latency_sum_ns_ = 0;
    workload_delete_latency_count_ = 0;
    workload_delete_latency_sum_ns_ = 0;
    latency_generation_++;
    std::lock_guard<std::mutex> rlk(latency_registry_mu_);
    latency_buffers_.clear();
  }

  bool enabled() const { return enabled_; }

  void resume_rss() {
    if (!enabled_) return;
    if (rss_worker_.joinable()) return;
    // Exclude per-round recall GT (and other RSS-paused gaps) from the next
    // window's window_time_s denominator.
    {
      std::lock_guard<std::mutex> lk(flush_mu_);
      window_start_ = std::chrono::steady_clock::now();
      next_flush_time_ =
          window_start_ + std::chrono::milliseconds(window_interval_ms_);
    }
    rss_stop_.store(false, std::memory_order_release);
    rss_worker_ = std::thread([this]() {
      while (!rss_stop_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        record_rss_sample(current_rss_mb());
        maybe_flush_time_window(now);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      record_rss_sample(current_rss_mb());
    });
  }

  void pause_rss() {
    rss_stop_.store(true, std::memory_order_release);
    if (rss_worker_.joinable()) {
      rss_worker_.join();
    }
  }

  RssStats pause_and_take_round_rss() {
    pause_rss();
    std::lock_guard<std::mutex> lk(rss_mu_);
    RssStats rss;
    rss.sample_count = round_rss_sample_count_;
    rss.avg_mb = safe_div(round_rss_sum_mb_,
                          static_cast<double>(round_rss_sample_count_));
    rss.peak_mb = round_peak_rss_mb_;
    round_rss_sum_mb_ = 0.0;
    round_rss_sample_count_ = 0;
    round_peak_rss_mb_ = 0.0;
    // A stage flush may leave a short RSS-only tail. It belongs to the round
    // summary above and must not leak into the next round's first window.
    window_rss_sum_mb_ = 0.0;
    window_rss_sample_count_ = 0;
    window_peak_rss_mb_ = 0.0;
    return rss;
  }

  double avg_rss_mb() const {
    std::lock_guard<std::mutex> lk(rss_mu_);
    return safe_div(workload_rss_sum_mb_,
                    static_cast<double>(workload_rss_sample_count_));
  }

  double peak_rss_mb() const {
    std::lock_guard<std::mutex> lk(rss_mu_);
    return workload_peak_rss_mb_;
  }

  LatencyStats query_latency_summary() const {
    return latency_summary_from_totals(workload_query_latency_count_,
                                       workload_query_latency_sum_ns_);
  }

  LatencyStats insert_latency_summary() const {
    return latency_summary_from_totals(workload_insert_latency_count_,
                                       workload_insert_latency_sum_ns_);
  }

  LatencyStats delete_latency_summary() const {
    return latency_summary_from_totals(workload_delete_latency_count_,
                                       workload_delete_latency_sum_ns_);
  }

  void reset_io_baseline() {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lk(flush_mu_);
    last_io_snapshot_ = LogicalIoCounter::snapshot();
    emit_io_ = true;
  }

  void record_insert(double latency_us) {
    if (!enabled_) return;
    enter_record();
    window_insert_count_.fetch_add(1, std::memory_order_relaxed);
    record_latency(latency_us, insert_latency_count_, insert_latency_sum_ns_,
                   thread_latency_buffers().insert);
    leave_record();
  }

  void record_delete(double latency_us) {
    if (!enabled_) return;
    enter_record();
    window_delete_count_.fetch_add(1, std::memory_order_relaxed);
    record_latency(latency_us, delete_latency_count_, delete_latency_sum_ns_,
                   thread_latency_buffers().del);
    leave_record();
  }

  void record_query(double latency_us, uint64_t round_index,
                    const std::string &stage) {
    if (!enabled_) return;
    enter_record();
    current_round_index_.store(round_index, std::memory_order_relaxed);
    current_stage_id_.store(stage_id(stage), std::memory_order_relaxed);
    window_query_count_.fetch_add(1, std::memory_order_relaxed);
    record_latency(latency_us, query_latency_count_, query_latency_sum_ns_,
                   thread_latency_buffers().query);
    leave_record();
  }

  void finalize(uint64_t round_index, const std::string &stage) {
    flush(round_index, stage);
  }

  void flush(uint64_t round_index, const std::string &stage) {
    if (!enabled_) return;
    std::unique_lock<std::mutex> lk(flush_mu_);
    flush_locked(round_index, stage);
  }

  // Label auto time-window flushes (record_insert alone leaves stage as
  // previous query stage / foreground).
  void set_stage(uint64_t round_index, const std::string &stage) {
    if (!enabled_) return;
    current_round_index_.store(round_index, std::memory_order_relaxed);
    current_stage_id_.store(stage_id(stage), std::memory_order_relaxed);
  }

 private:
  struct RssWindow {
    double avg_mb = 0.0;
    double peak_mb = 0.0;
    uint64_t sample_count = 0;
  };

  struct ThreadLatencyBuffers {
    std::vector<double> query;
    std::vector<double> insert;
    std::vector<double> del;
  };

  void maybe_flush_time_window(const std::chrono::steady_clock::time_point &now) {
    if (window_interval_ms_ == 0 || now < next_flush_time_) {
      return;
    }
    std::unique_lock<std::mutex> lk(flush_mu_, std::try_to_lock);
    if (!lk.owns_lock()) {
      return;
    }
    if (now < next_flush_time_) {
      return;
    }
    const uint64_t round_index =
        current_round_index_.load(std::memory_order_relaxed);
    const std::string stage =
        stage_name(current_stage_id_.load(std::memory_order_relaxed));
    flush_locked(round_index, stage);
    next_flush_time_ =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(window_interval_ms_);
  }

  void flush_locked(uint64_t round_index, const std::string &stage) {
    flushing_.store(true, std::memory_order_release);
    while (in_flight_records_.load(std::memory_order_acquire) != 0) {
      std::this_thread::yield();
    }
    const uint64_t q =
        window_query_count_.exchange(0, std::memory_order_relaxed);
    const uint64_t ins =
        window_insert_count_.exchange(0, std::memory_order_relaxed);
    const uint64_t del =
        window_delete_count_.exchange(0, std::memory_order_relaxed);
    const uint64_t query_lat_count =
        query_latency_count_.exchange(0, std::memory_order_relaxed);
    const uint64_t query_lat_sum_ns =
        query_latency_sum_ns_.exchange(0, std::memory_order_relaxed);
    const uint64_t insert_lat_count =
        insert_latency_count_.exchange(0, std::memory_order_relaxed);
    const uint64_t insert_lat_sum_ns =
        insert_latency_sum_ns_.exchange(0, std::memory_order_relaxed);
    const uint64_t delete_lat_count =
        delete_latency_count_.exchange(0, std::memory_order_relaxed);
    const uint64_t delete_lat_sum_ns =
        delete_latency_sum_ns_.exchange(0, std::memory_order_relaxed);
    std::vector<double> query_samples;
    std::vector<double> insert_samples;
    std::vector<double> delete_samples;
    drain_latency_samples(query_samples, insert_samples, delete_samples);
    if (q == 0 && ins == 0 && del == 0) {
      flushing_.store(false, std::memory_order_release);
      return;
    }

    LatencyStats query_lat =
        summarize_window_latency(query_samples, query_lat_count,
                                 query_lat_sum_ns);
    LatencyStats insert_lat =
        summarize_window_latency(insert_samples, insert_lat_count,
                                 insert_lat_sum_ns);
    LatencyStats delete_lat =
        summarize_window_latency(delete_samples, delete_lat_count,
                                 delete_lat_sum_ns);
    const RssWindow rss = snapshot_rss_window();

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(now - workload_start_).count();
    const double window_time_s =
        std::chrono::duration<double>(now - window_start_).count();
    window_start_ = now;
    active_elapsed_s_ += window_time_s;

    const uint64_t total_ops = q + ins + del;
    LogicalIoSnapshot io_delta;
    if (emit_io_) {
      const LogicalIoSnapshot current_io = LogicalIoCounter::snapshot();
      io_delta = subtract_io(current_io, last_io_snapshot_);
      last_io_snapshot_ = current_io;
    }
    bool first = true;
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "{";
    json_kv(line, first, "baseline", baseline_);
    json_kv(line, first, "workload", "mixed_update");
    json_kv(line, first, "phase", "window");
    json_kv(line, first, "round_index", round_index);
    json_kv(line, first, "window_index", window_index_++);
    json_kv(line, first, "stage", stage);
    json_kv(line, first, "elapsed_s", elapsed_s);
    json_kv(line, first, "active_elapsed_s", active_elapsed_s_);
    json_kv(line, first, "window_time_s", window_time_s);
    json_kv(line, first, "query_count", q);
    json_kv(line, first, "insert_count", ins);
    json_kv(line, first, "delete_count", del);
    json_kv(line, first, "query_qps",
            safe_div(static_cast<double>(q), window_time_s));
    json_kv(line, first, "insert_ops_per_s",
            safe_div(static_cast<double>(ins), window_time_s));
    json_kv(line, first, "delete_ops_per_s",
            safe_div(static_cast<double>(del), window_time_s));
    json_kv(line, first, "update_ops_per_s",
            safe_div(static_cast<double>(ins + del), window_time_s));
    json_kv(line, first, "overall_ops_per_s",
            safe_div(static_cast<double>(total_ops), window_time_s));
    emit_latency(line, first, "query", query_lat);
    emit_latency(line, first, "insert_call", insert_lat);
    emit_latency(line, first, "delete_call", delete_lat);
    json_kv(line, first, "avg_rss_mb", rss.avg_mb);
    json_kv(line, first, "peak_rss_mb", rss.peak_mb);
    json_kv(line, first, "rss_sample_count", rss.sample_count);
    if (emit_io_) {
      emit_logical_io(line, first, io_delta, window_time_s, total_ops);
    }
    line << "}";
    std::cout << line.str() << std::endl;
    accumulate_workload_summary(query_lat_count, query_lat_sum_ns,
                                insert_lat_count, insert_lat_sum_ns,
                                delete_lat_count, delete_lat_sum_ns);
    flushing_.store(false, std::memory_order_release);
  }

  void enter_record() {
    while (flushing_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    in_flight_records_.fetch_add(1, std::memory_order_acq_rel);
    while (flushing_.load(std::memory_order_acquire)) {
      in_flight_records_.fetch_sub(1, std::memory_order_acq_rel);
      std::this_thread::yield();
      while (flushing_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      in_flight_records_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  void leave_record() {
    in_flight_records_.fetch_sub(1, std::memory_order_acq_rel);
  }

  static uint32_t stage_id(const std::string &stage) {
    if (stage == "merge") return 1;
    if (stage == "post_merge_query") return 2;
    if (stage == "workload") return 3;
    if (stage == "query") return 4;
    if (stage == "insert") return 5;
    return 0;
  }

  static std::string stage_name(uint32_t id) {
    switch (id) {
      case 1:
        return "merge";
      case 2:
        return "post_merge_query";
      case 3:
        return "workload";
      case 4:
        return "query";
      case 5:
        return "insert";
      default:
        return "foreground";
    }
  }

  void record_latency(double latency_us, std::atomic<uint64_t> &count,
                      std::atomic<uint64_t> &sum_ns,
                      std::vector<double> &samples) {
    count.fetch_add(1, std::memory_order_relaxed);
    sum_ns.fetch_add(static_cast<uint64_t>(latency_us * 1000.0),
                     std::memory_order_relaxed);
    samples.push_back(latency_us);
  }

  ThreadLatencyBuffers &thread_latency_buffers() {
    struct Cache {
      QueryCycleWindowMetrics *owner = nullptr;
      uint64_t generation = 0;
      ThreadLatencyBuffers *buffers = nullptr;
    };
    thread_local Cache cache;
    if (cache.owner == this && cache.generation == latency_generation_ &&
        cache.buffers != nullptr) {
      return *cache.buffers;
    }

    std::lock_guard<std::mutex> lk(latency_registry_mu_);
    latency_buffers_.emplace_back(new ThreadLatencyBuffers());
    ThreadLatencyBuffers *buffers = latency_buffers_.back().get();
    const size_t per_thread_query_reserve =
        static_cast<size_t>(latency_reserve_hint_ / 64 + 16);
    buffers->query.reserve(per_thread_query_reserve);
    buffers->insert.reserve(per_thread_query_reserve / 18 + 1);
    buffers->del.reserve(per_thread_query_reserve / 18 + 1);
    cache.owner = this;
    cache.generation = latency_generation_;
    cache.buffers = buffers;
    return *buffers;
  }

  void drain_latency_samples(std::vector<double> &query_samples,
                             std::vector<double> &insert_samples,
                             std::vector<double> &delete_samples) {
    std::lock_guard<std::mutex> lk(latency_registry_mu_);
    size_t query_n = 0;
    size_t insert_n = 0;
    size_t delete_n = 0;
    for (const auto &buffers : latency_buffers_) {
      query_n += buffers->query.size();
      insert_n += buffers->insert.size();
      delete_n += buffers->del.size();
    }
    query_samples.reserve(query_n);
    insert_samples.reserve(insert_n);
    delete_samples.reserve(delete_n);
    for (auto &buffers : latency_buffers_) {
      query_samples.insert(query_samples.end(), buffers->query.begin(),
                           buffers->query.end());
      insert_samples.insert(insert_samples.end(), buffers->insert.begin(),
                            buffers->insert.end());
      delete_samples.insert(delete_samples.end(), buffers->del.begin(),
                            buffers->del.end());
      buffers->query.clear();
      buffers->insert.clear();
      buffers->del.clear();
    }
  }

  static LatencyStats summarize_window_latency(std::vector<double> samples,
                                               uint64_t count,
                                               uint64_t sum_ns) {
    LatencyStats lat;
    lat.avg_us = safe_div(static_cast<double>(sum_ns) / 1000.0,
                          static_cast<double>(count));
    std::sort(samples.begin(), samples.end());
    if (!samples.empty()) {
      lat.p50_us = percentile_sorted(samples, 0.50);
      lat.p90_us = percentile_sorted(samples, 0.90);
      lat.p95_us = percentile_sorted(samples, 0.95);
      lat.p99_us = percentile_sorted(samples, 0.99);
    }
    return lat;
  }

  static LatencyStats latency_summary_from_totals(uint64_t count,
                                                  uint64_t sum_ns) {
    LatencyStats lat;
    lat.avg_us = safe_div(static_cast<double>(sum_ns) / 1000.0,
                          static_cast<double>(count));
    return lat;
  }

  void record_rss_sample(double rss_mb) {
    std::lock_guard<std::mutex> lk(rss_mu_);
    window_rss_sum_mb_ += rss_mb;
    window_rss_sample_count_++;
    if (rss_mb > window_peak_rss_mb_) {
      window_peak_rss_mb_ = rss_mb;
    }
    round_rss_sum_mb_ += rss_mb;
    round_rss_sample_count_++;
    if (rss_mb > round_peak_rss_mb_) {
      round_peak_rss_mb_ = rss_mb;
    }
    // Accumulate workload RSS from raw samples rather than emitted windows so
    // RSS-only tails and zero-operation maintenance intervals are retained.
    workload_rss_sum_mb_ += rss_mb;
    workload_rss_sample_count_++;
    if (rss_mb > workload_peak_rss_mb_) {
      workload_peak_rss_mb_ = rss_mb;
    }
  }

  void reset_rss_window_locked() {
    std::lock_guard<std::mutex> lk(rss_mu_);
    window_rss_sum_mb_ = 0.0;
    window_rss_sample_count_ = 0;
    window_peak_rss_mb_ = 0.0;
  }

  RssWindow snapshot_rss_window() {
    std::lock_guard<std::mutex> lk(rss_mu_);
    RssWindow rss;
    rss.sample_count = window_rss_sample_count_;
    rss.avg_mb = safe_div(window_rss_sum_mb_,
                          static_cast<double>(window_rss_sample_count_));
    rss.peak_mb = window_peak_rss_mb_;
    window_rss_sum_mb_ = 0.0;
    window_rss_sample_count_ = 0;
    window_peak_rss_mb_ = 0.0;
    return rss;
  }

  void accumulate_workload_summary(uint64_t query_count, uint64_t query_sum_ns,
                                   uint64_t insert_count, uint64_t insert_sum_ns,
                                   uint64_t delete_count,
                                   uint64_t delete_sum_ns) {
    workload_query_latency_count_ += query_count;
    workload_query_latency_sum_ns_ += query_sum_ns;
    workload_insert_latency_count_ += insert_count;
    workload_insert_latency_sum_ns_ += insert_sum_ns;
    workload_delete_latency_count_ += delete_count;
    workload_delete_latency_sum_ns_ += delete_sum_ns;
  }

  static uint64_t sub_u64(uint64_t a, uint64_t b) { return a >= b ? a - b : 0; }

  static LogicalIoSnapshot subtract_io(const LogicalIoSnapshot &a,
                                       const LogicalIoSnapshot &b) {
    LogicalIoSnapshot d;
    d.read_seq_ios = sub_u64(a.read_seq_ios, b.read_seq_ios);
    d.write_seq_ios = sub_u64(a.write_seq_ios, b.write_seq_ios);
    d.read_random_ios = sub_u64(a.read_random_ios, b.read_random_ios);
    d.write_random_ios = sub_u64(a.write_random_ios, b.write_random_ios);
    d.read_seq_bytes = sub_u64(a.read_seq_bytes, b.read_seq_bytes);
    d.write_seq_bytes = sub_u64(a.write_seq_bytes, b.write_seq_bytes);
    d.read_random_bytes = sub_u64(a.read_random_bytes, b.read_random_bytes);
    d.write_random_bytes = sub_u64(a.write_random_bytes, b.write_random_bytes);
    return d;
  }

  std::string baseline_;
  uint64_t latency_reserve_hint_ = 1;
  uint64_t window_interval_ms_ = 1000;
  bool enabled_ = false;
  std::chrono::steady_clock::time_point workload_start_;
  std::chrono::steady_clock::time_point window_start_;
  std::chrono::steady_clock::time_point next_flush_time_;
  double active_elapsed_s_ = 0.0;
  std::atomic<uint64_t> current_round_index_{0};
  std::atomic<uint32_t> current_stage_id_{0};
  uint64_t window_index_ = 0;
  std::atomic<uint64_t> window_query_count_{0};
  std::atomic<uint64_t> window_insert_count_{0};
  std::atomic<uint64_t> window_delete_count_{0};
  std::atomic<uint64_t> query_latency_count_{0};
  std::atomic<uint64_t> query_latency_sum_ns_{0};
  std::atomic<uint64_t> insert_latency_count_{0};
  std::atomic<uint64_t> insert_latency_sum_ns_{0};
  std::atomic<uint64_t> delete_latency_count_{0};
  std::atomic<uint64_t> delete_latency_sum_ns_{0};
  std::atomic<uint64_t> in_flight_records_{0};
  std::atomic_bool flushing_{false};
  LogicalIoSnapshot last_io_snapshot_;
  bool emit_io_ = false;
  std::mutex latency_registry_mu_;
  std::vector<std::unique_ptr<ThreadLatencyBuffers>> latency_buffers_;
  uint64_t latency_generation_ = 0;
  std::mutex flush_mu_;
  std::atomic_bool rss_stop_{true};
  std::thread rss_worker_;
  mutable std::mutex rss_mu_;
  double window_rss_sum_mb_ = 0.0;
  uint64_t window_rss_sample_count_ = 0;
  double window_peak_rss_mb_ = 0.0;
  double round_rss_sum_mb_ = 0.0;
  uint64_t round_rss_sample_count_ = 0;
  double round_peak_rss_mb_ = 0.0;
  double workload_rss_sum_mb_ = 0.0;
  uint64_t workload_rss_sample_count_ = 0;
  double workload_peak_rss_mb_ = 0.0;
  uint64_t workload_query_latency_count_ = 0;
  uint64_t workload_query_latency_sum_ns_ = 0;
  uint64_t workload_insert_latency_count_ = 0;
  uint64_t workload_insert_latency_sum_ns_ = 0;
  uint64_t workload_delete_latency_count_ = 0;
  uint64_t workload_delete_latency_sum_ns_ = 0;
};

inline void emit_phase_checkpoint(std::ostream &out,
                                  const std::string &baseline,
                                  const std::string &phase,
                                  uint64_t round_index, double elapsed_s,
                                  uint64_t insert_count,
                                  uint64_t delete_count,
                                  uint64_t query_count, double phase_time_s) {
  bool first = true;
  std::ostringstream line;
  line << std::fixed << std::setprecision(6) << "{";
  json_kv(line, first, "baseline", baseline);
  json_kv(line, first, "workload", "mixed_update");
  json_kv(line, first, "phase", phase);
  json_kv(line, first, "round_index", round_index);
  json_kv(line, first, "elapsed_s", elapsed_s);
  json_kv(line, first, "insert_count", insert_count);
  json_kv(line, first, "delete_count", delete_count);
  json_kv(line, first, "query_count", query_count);
  json_kv(line, first, "phase_time_s", phase_time_s);
  json_kv(line, first, "query_qps",
          safe_div(static_cast<double>(query_count), phase_time_s));
  json_kv(line, first, "insert_ops_per_s",
          safe_div(static_cast<double>(insert_count), phase_time_s));
  json_kv(line, first, "delete_ops_per_s",
          safe_div(static_cast<double>(delete_count), phase_time_s));
  json_kv(line, first, "update_ops_per_s",
          safe_div(static_cast<double>(insert_count + delete_count),
                   phase_time_s));
  line << "}";
  out << line.str() << std::endl;
}

}  // namespace ann_bench
