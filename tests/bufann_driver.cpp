// bufann_driver.cpp
//
// Use an existing BufANN index for component-level search, insert, and delete
// benchmarks. Update workloads call the public bufann_api directly and then
// run one final recall search.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <limits>
#include <iostream>
#include <memory>
#include <utility>
#include <mutex>
#include <thread>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include "tcmalloc/malloc_extension.h"

#include "ann_exception.h"
#include "aux_utils.h"
#include "bufann/ann_bench_metrics.h"
#include "bufann/bufann_api.h"
#include "bufann/inplace_graph_ops.h"
#include "utils.h"
#include "weighted_update_pool.h"

using namespace diskann::inplace;

namespace {

static std::string get_arg(int argc, char **argv, const std::string &flag,
                           const std::string &def = "") {
    // Last match wins so trailing "$@" overrides earlier common.sh defaults.
    std::string value = def;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) value = argv[i + 1];
    }
    return value;
}

static bool has_flag(int argc, char **argv, const std::string &flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) return true;
    }
    return false;
}

static bool parse_bool_arg(int argc, char **argv, const std::string &flag,
                           bool def) {
    const std::string raw = get_arg(argc, argv, flag, def ? "1" : "0");
    return raw == "1" || raw == "true" || raw == "TRUE" || raw == "yes" ||
           raw == "on";
}

static bool workload_has_inserts(const std::string &workload) {
    return workload == "insert_only" || workload == "search_after_update";
}

static bool workload_has_deletes(const std::string &workload) {
    return workload == "delete_only" || workload == "search_after_update";
}

static double elapsed_s(const std::chrono::high_resolution_clock::time_point &begin,
                        const std::chrono::high_resolution_clock::time_point &end) {
    return std::chrono::duration<double>(end - begin).count();
}

static void release_tcmalloc_free_memory() {
    if (MallocExtension *ext = MallocExtension::instance()) {
        ext->ReleaseFreeMemory();
    }
}

static std::vector<uint32_t> load_ids_file(const std::string &path) {
    std::vector<uint32_t> ids;
    if (path.empty()) return ids;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open IDs file: " + path);
    int32_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
    if (n <= 0) return ids;
    ids.resize(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    if (!in) throw std::runtime_error("short read from IDs file: " + path);
    return ids;
}

template <typename T>
static std::vector<T> load_coords_by_ids(const std::string &path,
                                         const std::vector<uint32_t> &ids,
                                         size_t dim, size_t aligned_dim) {
    std::vector<T> out(ids.size() * aligned_dim, static_cast<T>(0));
    if (ids.empty()) return out;

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open data file: " + path);
    int32_t file_npts = 0, file_dim = 0;
    in.read(reinterpret_cast<char *>(&file_npts), sizeof(int32_t));
    in.read(reinterpret_cast<char *>(&file_dim), sizeof(int32_t));
    if (static_cast<size_t>(file_dim) != dim) {
        throw std::runtime_error("dimension mismatch in data file: " + path);
    }

    size_t run_start = 0;
    while (run_start < ids.size()) {
        size_t run_end = run_start + 1;
        while (run_end < ids.size() && ids[run_end] == ids[run_end - 1] + 1) {
            ++run_end;
        }
        const size_t run_len = run_end - run_start;
        const uint64_t offset =
            2ULL * sizeof(int32_t) +
            static_cast<uint64_t>(ids[run_start]) * dim * sizeof(T);
        in.seekg(static_cast<std::streamoff>(offset));
        std::vector<T> packed(run_len * dim);
        in.read(reinterpret_cast<char *>(packed.data()),
                static_cast<std::streamsize>(packed.size() * sizeof(T)));
        if (!in) throw std::runtime_error("EOF reading insert coords");
        for (size_t j = 0; j < run_len; ++j) {
            std::memcpy(out.data() + (run_start + j) * aligned_dim,
                        packed.data() + j * dim, dim * sizeof(T));
        }
        run_start = run_end;
    }
    return out;
}

static double percentile(std::vector<double> &v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * static_cast<double>(v.size()));
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

template <typename T>
struct StreamingQuerySource {
    T *query_data = nullptr;
    bool owns_query_data = false;
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
        std::vector<uint32_t> result_ids;
    };

    static size_t round_up_dim(size_t d) { return ((d + 7) / 8) * 8; }

    void load(const std::string &query_path, const std::string &full_path,
              uint64_t tail_first, uint64_t tail_last) {
        release_query();
        load_full_metadata(full_path, tail_first, tail_last);

        std::ifstream in(query_path, std::ios::binary);
        if (in) {
            int32_t qn = 0, qd = 0;
            in.read(reinterpret_cast<char *>(&qn), sizeof(int32_t));
            in.read(reinterpret_cast<char *>(&qd), sizeof(int32_t));
            if (!in || qn < 0 || qd <= 0) throw std::runtime_error("invalid query header");
            if (static_cast<size_t>(qd) != dim) throw std::runtime_error("query dim mismatch");
            query_n = static_cast<size_t>(qn);
            if (query_n > 0) {
                void *ptr = nullptr;
                if (posix_memalign(&ptr, 64, query_n * aligned_dim * sizeof(T)) != 0) {
                    throw std::runtime_error("failed to allocate query buffer");
                }
                query_data = reinterpret_cast<T *>(ptr);
                owns_query_data = true;
                std::fill(query_data, query_data + query_n * aligned_dim, T{});
                std::vector<T> packed(query_n * dim);
                in.read(reinterpret_cast<char *>(packed.data()),
                        static_cast<std::streamsize>(packed.size() * sizeof(T)));
                if (!in) throw std::runtime_error("short read from query file");
                for (size_t i = 0; i < query_n; i++) {
                    std::memcpy(query_data + i * aligned_dim, packed.data() + i * dim,
                                dim * sizeof(T));
                }
            }
        }
        if (pool_size() == 0) throw std::runtime_error("empty query source");
    }

    void load_from_aligned_query(T *queries, size_t nqueries, size_t query_dim,
                                 size_t query_aligned_dim,
                                 const std::string &full_path,
                                 uint64_t tail_first, uint64_t tail_last) {
        release_query();
        load_full_metadata(full_path, tail_first, tail_last);
        if (query_dim != dim) throw std::runtime_error("query dim mismatch");
        if (query_aligned_dim != aligned_dim) throw std::runtime_error("query aligned dim mismatch");
        query_data = queries;
        query_n = nqueries;
        owns_query_data = false;
        if (pool_size() == 0) throw std::runtime_error("empty query source");
    }

    ~StreamingQuerySource() { release_query(); }

    size_t tail_size() const {
        return tail_end > tail_begin ? static_cast<size_t>(tail_end - tail_begin) : 0;
    }
    size_t pool_size() const { return query_n + tail_size(); }

    const T *get(uint64_t seq, ThreadState &state) const {
        const uint64_t pos = seq % pool_size();
        if (pos < query_n) return query_data + static_cast<size_t>(pos) * aligned_dim;
        const uint64_t point_id = tail_begin + ((pos - query_n) % tail_size());
        if (point_id < state.block_begin || point_id >= state.block_begin + state.block_len) {
            if (!state.full_reader.is_open()) {
                state.full_reader.open(full_data_bin, std::ios::binary);
                if (!state.full_reader) throw std::runtime_error("cannot open full data file");
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
            if (!state.full_reader) throw std::runtime_error("short read from tail query range");
            for (size_t i = 0; i < block_len; i++) {
                std::memcpy(state.aligned.data() + i * aligned_dim,
                            state.packed.data() + i * dim, dim * sizeof(T));
            }
        }
        return state.aligned.data() + static_cast<size_t>(point_id - state.block_begin) * aligned_dim;
    }

  private:
    void release_query() {
        if (query_data && owns_query_data) free(query_data);
        query_data = nullptr;
        query_n = 0;
        owns_query_data = false;
    }

    void load_full_metadata(const std::string &full_path,
                            uint64_t tail_first, uint64_t tail_last) {
        full_data_bin = full_path;
        std::ifstream full(full_data_bin, std::ios::binary);
        if (!full) throw std::runtime_error("cannot open full data file: " + full_data_bin);
        int32_t n = 0, d = 0;
        full.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
        full.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
        if (!full || n <= 0 || d <= 0) throw std::runtime_error("invalid full data header");
        full_npts = static_cast<uint64_t>(n);
        dim = static_cast<size_t>(d);
        aligned_dim = round_up_dim(dim);
        tail_begin = tail_first;
        tail_end = std::min<uint64_t>(tail_last, full_npts);
        if (tail_begin > tail_end) throw std::runtime_error("invalid tail query range");
    }
};

struct SearchMetrics {
    double search_qps = 0.0;
    double recall = -1.0;
    double lat_avg_us = 0.0;
    double lat_p50_us = 0.0;
    double lat_p90_us = 0.0;
    double lat_p95_us = 0.0;
    double lat_p99_us = 0.0;
    double lat_p999_us = 0.0;
    double mean_ios = 0.0;
    size_t nqueries = 0;
    uint64_t newly_inserted_results_gt = 0;
    uint64_t newly_inserted_results_hit = 0;
};

struct QueryLatencySampler {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> sum_ns{0};
    std::mutex mu;
    std::vector<double> samples;

    void record(uint64_t seq, double latency_us) {
        count.fetch_add(1, std::memory_order_relaxed);
        sum_ns.fetch_add(static_cast<uint64_t>(latency_us * 1000.0), std::memory_order_relaxed);
        if ((seq & 127ULL) == 0) {
            std::lock_guard<std::mutex> lk(mu);
            samples.push_back(latency_us);
        }
    }

    SearchMetrics metrics(double wall_s) {
        SearchMetrics m;
        m.nqueries = count.load(std::memory_order_relaxed);
        m.search_qps = wall_s > 0.0 ? static_cast<double>(m.nqueries) / wall_s : 0.0;
        m.lat_avg_us = m.nqueries > 0
            ? static_cast<double>(sum_ns.load(std::memory_order_relaxed)) / 1000.0 /
                  static_cast<double>(m.nqueries)
            : 0.0;
        std::vector<double> copy;
        {
            std::lock_guard<std::mutex> lk(mu);
            copy = samples;
        }
        m.lat_p50_us = percentile(copy, 0.50);
        m.lat_p90_us = percentile(copy, 0.90);
        m.lat_p95_us = percentile(copy, 0.95);
        m.lat_p99_us = percentile(copy, 0.99);
        m.lat_p999_us = percentile(copy, 0.999);
        return m;
    }
};

class BufANNWindowMetrics {
  public:
    struct LatencyWindow {
        double avg_us = 0.0;
        double p50_us = 0.0;
        double p90_us = 0.0;
        double p95_us = 0.0;
        double p99_us = 0.0;
    };

    ~BufANNWindowMetrics() { pause_rss(); }

    void start(uint64_t latency_reserve_hint) {
        std::lock_guard<std::mutex> lk(flush_mu_);
        pause_rss();
        latency_reserve_hint_ =
            latency_reserve_hint == 0 ? 1 : latency_reserve_hint;
        window_interval_ms_ = query_window_interval_ms();
        enabled_ = true;
        workload_start_ = std::chrono::high_resolution_clock::now();
        window_start_ = workload_start_;
        next_flush_time_ = workload_start_ +
            std::chrono::milliseconds(window_interval_ms_);
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
        workload_query_samples_.clear();
        workload_insert_samples_.clear();
        workload_delete_samples_.clear();
        workload_query_sample_seen_ = 0;
        workload_insert_sample_seen_ = 0;
        workload_delete_sample_seen_ = 0;
        latency_generation_++;
        std::lock_guard<std::mutex> rlk(latency_registry_mu_);
        latency_buffers_.clear();
    }

    void resume_rss() {
        if (!enabled_) return;
        if (rss_worker_.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(flush_mu_);
            window_start_ = std::chrono::high_resolution_clock::now();
            next_flush_time_ =
                window_start_ + std::chrono::milliseconds(window_interval_ms_);
        }
        rss_stop_.store(false, std::memory_order_release);
        rss_worker_ = std::thread([this]() {
            while (!rss_stop_.load(std::memory_order_acquire)) {
                const auto now = std::chrono::high_resolution_clock::now();
                record_rss_sample(bufann_bench::current_rss_mb());
                maybe_flush_time_window(now);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            record_rss_sample(bufann_bench::current_rss_mb());
        });
    }

    void pause_rss() {
        rss_stop_.store(true, std::memory_order_release);
        if (rss_worker_.joinable()) {
            rss_worker_.join();
        }
    }

    bufann_bench::RssStats pause_and_take_round_rss() {
        pause_rss();
        std::lock_guard<std::mutex> lk(rss_mu_);
        bufann_bench::RssStats rss;
        rss.sample_count = round_rss_sample_count_;
        rss.avg_mb = bufann_bench::safe_div(
            round_rss_sum_mb_, static_cast<double>(round_rss_sample_count_));
        rss.peak_mb = round_peak_rss_mb_;
        round_rss_sum_mb_ = 0.0;
        round_rss_sample_count_ = 0;
        round_peak_rss_mb_ = 0.0;
        // Do not carry an RSS-only tail into the next round's first window.
        window_rss_sum_mb_ = 0.0;
        window_rss_sample_count_ = 0;
        window_peak_rss_mb_ = 0.0;
        return rss;
    }

    double avg_rss_mb() const {
        std::lock_guard<std::mutex> lk(rss_mu_);
        return bufann_bench::safe_div(
            workload_rss_sum_mb_,
            static_cast<double>(workload_rss_sample_count_));
    }

    double peak_rss_mb() const {
        std::lock_guard<std::mutex> lk(rss_mu_);
        return workload_peak_rss_mb_;
    }

    LatencyWindow query_latency_summary() const {
        return summarize_latency(workload_query_samples_,
                                 workload_query_latency_count_,
                                 workload_query_latency_sum_ns_);
    }

    LatencyWindow insert_latency_summary() const {
        return summarize_latency(workload_insert_samples_,
                                 workload_insert_latency_count_,
                                 workload_insert_latency_sum_ns_);
    }

    LatencyWindow delete_latency_summary() const {
        return summarize_latency(workload_delete_samples_,
                                 workload_delete_latency_count_,
                                 workload_delete_latency_sum_ns_);
    }

    // Online eval only: RQ keeps avg-from-totals workload pxx (zero) so the
    // normal path does not retain every latency sample in memory.
    void enable_workload_percentiles(bool enable) {
        retain_workload_percentiles_ = enable;
    }

    void set_io_emitter(std::function<void(std::ostream &, bool &, double,
                                           uint64_t)> emit_io) {
        emit_io_ = std::move(emit_io);
    }

    void set_runtime_emitter(
        std::function<void(std::ostream &, bool &)> emit_runtime) {
        emit_runtime_ = std::move(emit_runtime);
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

    static uint64_t query_window_interval_ms() {
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

    void maybe_flush_time_window(
        const std::chrono::high_resolution_clock::time_point &now) {
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
        next_flush_time_ = std::chrono::high_resolution_clock::now() +
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

        LatencyWindow query_lat = summarize_latency(query_samples, query_lat_count,
                                                    query_lat_sum_ns);
        LatencyWindow insert_lat = summarize_latency(insert_samples, insert_lat_count,
                                                     insert_lat_sum_ns);
        LatencyWindow delete_lat = summarize_latency(delete_samples, delete_lat_count,
                                                     delete_lat_sum_ns);
        const RssWindow rss = snapshot_rss_window();
        const auto now = std::chrono::high_resolution_clock::now();
        const double elapsed = elapsed_s(workload_start_, now);
        const double window_time = elapsed_s(window_start_, now);
        window_start_ = now;
        active_elapsed_s_ += window_time;
        const uint64_t total_ops = q + ins + del;
        bool first = true;
        std::ostringstream line;
        line << std::fixed << std::setprecision(6) << "{";
        bufann_bench::json_kv(line, first, "baseline", "BufANN");
        bufann_bench::json_kv(line, first, "workload", "mixed_update");
        bufann_bench::json_kv(line, first, "phase", "window");
        bufann_bench::json_kv(line, first, "round_index", round_index);
        bufann_bench::json_kv(line, first, "window_index", window_index_++);
        bufann_bench::json_kv(line, first, "stage", stage);
        bufann_bench::json_kv(line, first, "elapsed_s", elapsed);
        bufann_bench::json_kv(line, first, "active_elapsed_s", active_elapsed_s_);
        bufann_bench::json_kv(line, first, "window_time_s", window_time);
        bufann_bench::json_kv(line, first, "query_count", q);
        bufann_bench::json_kv(line, first, "insert_count", ins);
        bufann_bench::json_kv(line, first, "delete_count", del);
        bufann_bench::json_kv(line, first, "query_qps",
                              bufann_bench::safe_div(
                                  static_cast<double>(q), window_time));
        bufann_bench::json_kv(line, first, "insert_ops_per_s",
                              bufann_bench::safe_div(
                                  static_cast<double>(ins), window_time));
        bufann_bench::json_kv(line, first, "delete_ops_per_s",
                              bufann_bench::safe_div(
                                  static_cast<double>(del), window_time));
        bufann_bench::json_kv(line, first, "update_ops_per_s",
                              bufann_bench::safe_div(
                                  static_cast<double>(ins + del), window_time));
        bufann_bench::json_kv(line, first, "overall_ops_per_s",
                              bufann_bench::safe_div(
                                  static_cast<double>(total_ops), window_time));
        emit_latency(line, first, "query", query_lat);
        emit_latency(line, first, "insert_call", insert_lat);
        emit_latency(line, first, "delete_call", delete_lat);
        bufann_bench::json_kv(line, first, "avg_rss_mb", rss.avg_mb);
        bufann_bench::json_kv(line, first, "peak_rss_mb", rss.peak_mb);
        bufann_bench::json_kv(line, first, "rss_sample_count",
                              rss.sample_count);
        if (emit_io_) {
            emit_io_(line, first, window_time, total_ops);
        }
        if (emit_runtime_) {
            emit_runtime_(line, first);
        }
        line << "}";
        std::cout << line.str() << std::endl;
        accumulate_workload_summary(query_lat_count, query_lat_sum_ns,
                                    insert_lat_count, insert_lat_sum_ns,
                                    delete_lat_count, delete_lat_sum_ns);
        if (retain_workload_percentiles_) {
            append_workload_samples(workload_query_samples_, query_samples,
                                    workload_query_sample_seen_);
            append_workload_samples(workload_insert_samples_, insert_samples,
                                    workload_insert_sample_seen_);
            append_workload_samples(workload_delete_samples_, delete_samples,
                                    workload_delete_sample_seen_);
        }
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
        if (stage == "pre_cleanup") return 5;
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
            case 5:
                return "pre_cleanup";
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
            BufANNWindowMetrics *owner = nullptr;
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

    static LatencyWindow summarize_latency(std::vector<double> samples,
                                           uint64_t count,
                                           uint64_t sum_ns) {
        LatencyWindow s;
        s.avg_us = count > 0 ? static_cast<double>(sum_ns) / 1000.0 /
                                   static_cast<double>(count)
                             : 0.0;
        std::sort(samples.begin(), samples.end());
        s.p50_us = percentile(samples, 0.50);
        s.p90_us = percentile(samples, 0.90);
        s.p95_us = percentile(samples, 0.95);
        s.p99_us = percentile(samples, 0.99);
        return s;
    }

    // Bound retained samples for online workload percentiles. Exact while
    // under the cap; reservoir thereafter so long runs stay O(1) memory.
    static constexpr size_t kWorkloadPercentileCap = 200000;

    static uint64_t mix64(uint64_t x) {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    static void append_workload_samples(std::vector<double> &dst,
                                        const std::vector<double> &src,
                                        uint64_t &seen) {
        for (double v : src) {
            ++seen;
            if (dst.size() < kWorkloadPercentileCap) {
                dst.push_back(v);
                continue;
            }
            const uint64_t j = mix64(seen) % seen;
            if (j < kWorkloadPercentileCap) {
                dst[static_cast<size_t>(j)] = v;
            }
        }
    }

    static void emit_latency(std::ostream &out, bool &first,
                             const std::string &prefix,
                             const LatencyWindow &s) {
        bufann_bench::json_kv(out, first, prefix + "_lat_avg_us", s.avg_us);
        bufann_bench::json_kv(out, first, prefix + "_lat_p50_us", s.p50_us);
        bufann_bench::json_kv(out, first, prefix + "_lat_p90_us", s.p90_us);
        bufann_bench::json_kv(out, first, prefix + "_lat_p95_us", s.p95_us);
        bufann_bench::json_kv(out, first, prefix + "_lat_p99_us", s.p99_us);
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
        // Keep workload RSS independent of whether a window emitted ops.
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
        rss.avg_mb = bufann_bench::safe_div(
            window_rss_sum_mb_,
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

    uint64_t latency_reserve_hint_ = 1;
    uint64_t window_interval_ms_ = 1000;
    bool enabled_ = false;
    std::chrono::high_resolution_clock::time_point workload_start_;
    std::chrono::high_resolution_clock::time_point window_start_;
    std::chrono::high_resolution_clock::time_point next_flush_time_;
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
    std::mutex latency_registry_mu_;
    std::vector<std::unique_ptr<ThreadLatencyBuffers>> latency_buffers_;
    uint64_t latency_generation_ = 0;
    std::function<void(std::ostream &, bool &, double, uint64_t)> emit_io_;
    std::function<void(std::ostream &, bool &)> emit_runtime_;
    bool retain_workload_percentiles_ = false;
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
    // Window samples are drained continuously. Online mode may retain a
    // bounded reservoir for workload-level percentiles; RQ leaves these empty
    // and keeps the prior avg-from-totals workload pxx fields.
    std::vector<double> workload_query_samples_;
    std::vector<double> workload_insert_samples_;
    std::vector<double> workload_delete_samples_;
    uint64_t workload_query_sample_seen_ = 0;
    uint64_t workload_insert_sample_seen_ = 0;
    uint64_t workload_delete_sample_seen_ = 0;
};

static void emit_bufann_phase_checkpoint(std::ostream &out,
                                         const std::string &phase,
                                         uint64_t round_index,
                                         double elapsed,
                                         uint64_t insert_count,
                                         uint64_t delete_count,
                                         uint64_t query_count,
                                         double phase_time_s) {
    bool first = true;
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "{";
    bufann_bench::json_kv(line, first, "baseline", "BufANN");
    bufann_bench::json_kv(line, first, "workload", "mixed_update");
    bufann_bench::json_kv(line, first, "phase", phase);
    bufann_bench::json_kv(line, first, "round_index", round_index);
    bufann_bench::json_kv(line, first, "elapsed_s", elapsed);
    bufann_bench::json_kv(line, first, "insert_count", insert_count);
    bufann_bench::json_kv(line, first, "delete_count", delete_count);
    bufann_bench::json_kv(line, first, "query_count", query_count);
    bufann_bench::json_kv(line, first, "phase_time_s", phase_time_s);
    bufann_bench::json_kv(line, first, "query_qps",
                          bufann_bench::safe_div(
                              static_cast<double>(query_count), phase_time_s));
    bufann_bench::json_kv(line, first, "insert_ops_per_s",
                          bufann_bench::safe_div(
                              static_cast<double>(insert_count), phase_time_s));
    bufann_bench::json_kv(line, first, "delete_ops_per_s",
                          bufann_bench::safe_div(
                              static_cast<double>(delete_count), phase_time_s));
    bufann_bench::json_kv(line, first, "update_ops_per_s",
                          bufann_bench::safe_div(
                              static_cast<double>(insert_count + delete_count),
                              phase_time_s));
    line << "}";
    out << line.str() << std::endl;
}

struct UpdateMetrics {
    size_t n_inserts = 0;
    size_t n_deletes = 0;
    size_t query_count = 0;
    double foreground_update_wall_time_s = 0.0;
    double maintenance_wall_time_s = 0.0;
    double post_merge_query_time_s = 0.0;
    double flush_wall_time_s = 0.0;
    double total_update_wall_time_s = 0.0;
    double insert_time_s = 0.0;
    double delete_time_s = 0.0;
    double insert_ops_per_s = 0.0;
    double delete_ops_per_s = 0.0;
    double query_time_s = 0.0;
    double query_qps = 0.0;
    double query_lat_avg_us = 0.0;
    double query_lat_p50_us = 0.0;
    double query_lat_p90_us = 0.0;
    double query_lat_p95_us = 0.0;
    double query_lat_p99_us = 0.0;
    double query_lat_p999_us = 0.0;
    double update_ops_per_s = 0.0;
    double overall_ops_per_s = 0.0;
    double elapsed_s = 0.0;
    double insert_lat_avg_us = 0.0;
    double insert_lat_p50_us = 0.0;
    double insert_lat_p90_us = 0.0;
    double insert_lat_p95_us = 0.0;
    double insert_lat_p99_us = 0.0;
    double delete_lat_avg_us = 0.0;
    double delete_lat_p50_us = 0.0;
    double delete_lat_p90_us = 0.0;
    double delete_lat_p95_us = 0.0;
    double delete_lat_p99_us = 0.0;
};

struct MetricConfig {
    std::string data_type;
    uint32_t pq_chunks = 0;
    uint32_t recall_at = 0;
    uint32_t search_L = 0;
    uint32_t insert_search_L = 0;
    uint32_t graph_L = 0;
    std::string delete_algorithm = "freshdiskann";
    bool search_pq = true;
    uint32_t beamwidth = 0;
    uint32_t buffer_pool_frames = 0;
    uint32_t meta_growth_copy_threads = 0;
    uint32_t query_threads = 0;
    uint32_t insert_threads = 0;
    uint32_t delete_threads = 0;
    uint32_t maintenance_threads = 0;
    uint32_t maintenance_query_threads = 0;
    bool skip_update_search = false;
    bool enable_interval_metrics = false;
    bool run_maintenance = false;
    bool flush_after_maintenance = false;
    bool flush_background = false;
    double flush_dirty_high_ratio = 0.0;
    double flush_dirty_low_ratio = 0.0;
    uint32_t flush_budget_pages = 0;
};

struct OpLatencyStats {
    double avg_us = 0.0;
    double p50_us = 0.0;
    double p90_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
};

static OpLatencyStats compute_op_latency_stats(std::vector<double> &latencies_us) {
    OpLatencyStats s;
    if (latencies_us.empty()) return s;
    double sum = 0.0;
    for (double v : latencies_us) sum += v;
    s.avg_us = sum / static_cast<double>(latencies_us.size());
    s.p50_us = percentile(latencies_us, 0.50);
    s.p90_us = percentile(latencies_us, 0.90);
    s.p95_us = percentile(latencies_us, 0.95);
    s.p99_us = percentile(latencies_us, 0.99);
    return s;
}

struct InsertRunMetrics {
    double elapsed_s = 0.0;
    double maintenance_s = 0.0;
    bool background_repair = false;
};

static uint64_t elapsed_ns(const std::chrono::high_resolution_clock::time_point &begin,
                           const std::chrono::high_resolution_clock::time_point &end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

struct StatsSnapshot {
    uint64_t page_size = 4096;
    uint64_t sequential_read_ios = 0;
    uint64_t sequential_read_bytes = 0;
    uint64_t buffer_pool_write_ios = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t page_fault_total = 0;
    uint64_t pages_flushed = 0;
    uint64_t query_io_ns = 0;
    uint64_t meta_growth_count = 0;
    uint64_t meta_growth_drain_ns = 0;
    uint64_t meta_growth_alloc_ns = 0;
    uint64_t meta_growth_copy_ns = 0;
    uint64_t meta_growth_total_ns = 0;
    uint64_t meta_growth_last_from = 0;
    uint64_t meta_growth_last_to = 0;
    uint64_t meta_growth_last_workers = 0;
    uint64_t meta_growth_last_copy_bytes = 0;
};

static StatsSnapshot snapshot_stats(const InPlaceIOStats &stats,
                                    uint64_t page_size = 4096) {
    StatsSnapshot s;
    s.page_size = page_size;
    s.sequential_read_ios =
        stats.sequential_read_ios.load(std::memory_order_relaxed);
    s.sequential_read_bytes =
        stats.sequential_read_bytes.load(std::memory_order_relaxed);
    s.buffer_pool_write_ios =
        stats.buffer_pool_write_ios.load(std::memory_order_relaxed);
    s.cache_hits = stats.cache_hits.load(std::memory_order_relaxed);
    s.cache_misses = stats.cache_misses.load(std::memory_order_relaxed);
    s.page_fault_total = stats.page_fault_total.load(std::memory_order_relaxed);
    s.pages_flushed = stats.pages_flushed.load(std::memory_order_relaxed);
    s.query_io_ns = stats.query_io_ns.load(std::memory_order_relaxed);
    s.meta_growth_count =
        stats.meta_growth_count.load(std::memory_order_relaxed);
    s.meta_growth_drain_ns =
        stats.meta_growth_drain_ns.load(std::memory_order_relaxed);
    s.meta_growth_alloc_ns =
        stats.meta_growth_alloc_ns.load(std::memory_order_relaxed);
    s.meta_growth_copy_ns =
        stats.meta_growth_copy_ns.load(std::memory_order_relaxed);
    s.meta_growth_total_ns =
        stats.meta_growth_total_ns.load(std::memory_order_relaxed);
    s.meta_growth_last_from =
        stats.meta_growth_last_from.load(std::memory_order_relaxed);
    s.meta_growth_last_to =
        stats.meta_growth_last_to.load(std::memory_order_relaxed);
    s.meta_growth_last_workers =
        stats.meta_growth_last_workers.load(std::memory_order_relaxed);
    s.meta_growth_last_copy_bytes =
        stats.meta_growth_last_copy_bytes.load(std::memory_order_relaxed);
    // Fold pin_batch (libaio) numbers into the simple counters so reported
    // miss-ratio and avg-I/O-ns formulas remain agnostic to batching.
    s.page_fault_total +=
        stats.batch_misses_total.load(std::memory_order_relaxed);
    s.query_io_ns +=
        stats.batch_io_ns_total.load(std::memory_order_relaxed);
    return s;
}

static uint64_t saturated_sub(uint64_t lhs, uint64_t rhs) {
    return lhs >= rhs ? lhs - rhs : 0;
}

static StatsSnapshot stats_delta(const StatsSnapshot &after,
                                 const StatsSnapshot &before) {
    StatsSnapshot s;
    s.page_size = after.page_size;
    s.sequential_read_ios =
        saturated_sub(after.sequential_read_ios, before.sequential_read_ios);
    s.sequential_read_bytes =
        saturated_sub(after.sequential_read_bytes, before.sequential_read_bytes);
    s.buffer_pool_write_ios =
        saturated_sub(after.buffer_pool_write_ios, before.buffer_pool_write_ios);
    s.cache_hits = saturated_sub(after.cache_hits, before.cache_hits);
    s.cache_misses = saturated_sub(after.cache_misses, before.cache_misses);
    s.page_fault_total =
        saturated_sub(after.page_fault_total, before.page_fault_total);
    s.pages_flushed = saturated_sub(after.pages_flushed, before.pages_flushed);
    s.query_io_ns = saturated_sub(after.query_io_ns, before.query_io_ns);
    s.meta_growth_count =
        saturated_sub(after.meta_growth_count, before.meta_growth_count);
    s.meta_growth_drain_ns =
        saturated_sub(after.meta_growth_drain_ns, before.meta_growth_drain_ns);
    s.meta_growth_alloc_ns =
        saturated_sub(after.meta_growth_alloc_ns, before.meta_growth_alloc_ns);
    s.meta_growth_copy_ns =
        saturated_sub(after.meta_growth_copy_ns, before.meta_growth_copy_ns);
    s.meta_growth_total_ns =
        saturated_sub(after.meta_growth_total_ns, before.meta_growth_total_ns);
    if (s.meta_growth_count != 0) {
        s.meta_growth_last_from = after.meta_growth_last_from;
        s.meta_growth_last_to = after.meta_growth_last_to;
        s.meta_growth_last_workers = after.meta_growth_last_workers;
        s.meta_growth_last_copy_bytes = after.meta_growth_last_copy_bytes;
    }
    return s;
}

static void add_stats(StatsSnapshot &total, const StatsSnapshot &delta) {
    total.page_size = delta.page_size;
    total.sequential_read_ios += delta.sequential_read_ios;
    total.sequential_read_bytes += delta.sequential_read_bytes;
    total.buffer_pool_write_ios += delta.buffer_pool_write_ios;
    total.cache_hits += delta.cache_hits;
    total.cache_misses += delta.cache_misses;
    total.page_fault_total += delta.page_fault_total;
    total.pages_flushed += delta.pages_flushed;
    total.query_io_ns += delta.query_io_ns;
    total.meta_growth_count += delta.meta_growth_count;
    total.meta_growth_drain_ns += delta.meta_growth_drain_ns;
    total.meta_growth_alloc_ns += delta.meta_growth_alloc_ns;
    total.meta_growth_copy_ns += delta.meta_growth_copy_ns;
    total.meta_growth_total_ns += delta.meta_growth_total_ns;
    if (delta.meta_growth_count != 0) {
        total.meta_growth_last_from = delta.meta_growth_last_from;
        total.meta_growth_last_to = delta.meta_growth_last_to;
        total.meta_growth_last_workers = delta.meta_growth_last_workers;
        total.meta_growth_last_copy_bytes = delta.meta_growth_last_copy_bytes;
    }
}

template <typename T>
class UpdateStatsTracker {
  public:
    explicit UpdateStatsTracker(BufANNIndex<T> &idx) {
        store_ = bufann_store_snapshot(idx);
        base_ = snapshot_stats(store_->stats(), store_->page_size());
    }

    StatsSnapshot snapshot(BufANNIndex<T> &idx) {
        std::lock_guard<std::mutex> lock(mu_);
        (void)idx;

        StatsSnapshot cumulative = completed_;
        add_stats(cumulative,
                  stats_delta(snapshot_stats(store_->stats(),
                                             store_->page_size()),
                              base_));
        return cumulative;
    }

  private:
    std::mutex mu_;
    std::shared_ptr<InPlaceGraphStore> store_;
    StatsSnapshot base_;
    StatsSnapshot completed_;
};

static void emit_bufann_logical_io(std::ostream &out, bool &first,
                                    const StatsSnapshot &stats,
                                    double time_s, uint64_t op_count) {
    (void)time_s;
    (void)op_count;
    const uint64_t read_ios = stats.page_fault_total;
    const uint64_t write_ios = stats.pages_flushed;
    const uint64_t read_bytes = read_ios * stats.page_size;
    const uint64_t write_bytes = stats.pages_flushed * stats.page_size;
    const uint64_t read_seq_ios =
        stats.sequential_read_ios <= read_ios ? stats.sequential_read_ios : read_ios;
    const uint64_t read_random_ios = saturated_sub(read_ios, read_seq_ios);
    const uint64_t read_seq_bytes =
        stats.sequential_read_bytes <= read_bytes
            ? stats.sequential_read_bytes
            : read_bytes;
    const uint64_t read_random_bytes = saturated_sub(read_bytes, read_seq_bytes);
    bufann_bench::json_kv(out, first, "io_read_seq_ios",        read_seq_ios);
    bufann_bench::json_kv(out, first, "io_read_seq_bytes",      read_seq_bytes);
    bufann_bench::json_kv(out, first, "io_read_random_ios",     read_random_ios);
    bufann_bench::json_kv(out, first, "io_read_random_bytes",   read_random_bytes);
    bufann_bench::json_kv(out, first, "io_write_seq_ios",       0);
    bufann_bench::json_kv(out, first, "io_write_seq_bytes",     0);
    bufann_bench::json_kv(out, first, "io_write_random_ios",    write_ios);
    bufann_bench::json_kv(out, first, "io_write_random_bytes",  write_bytes);
    bufann_bench::json_kv(out, first, "buffer_pool_write_ios",  stats.buffer_pool_write_ios);
}

static void emit_bufann_storage_stats(std::ostream &out, bool &first,
                                      const StatsSnapshot &stats) {
    const uint64_t cache_misses = stats.cache_misses;
    const uint64_t cache_lookups = stats.cache_hits + cache_misses;
    bufann_bench::json_kv(out, first, "cache_hits", stats.cache_hits);
    bufann_bench::json_kv(out, first, "cache_misses", cache_misses);
    // Omit when unused so RQ result.json matches the prior key set.
    if (stats.meta_growth_count != 0 || stats.meta_growth_total_ns != 0) {
        bufann_bench::json_kv(
            out, first, "meta_growth_count", stats.meta_growth_count);
        bufann_bench::json_kv(out, first, "meta_growth_drain_us",
                               stats.meta_growth_drain_ns / 1000);
        bufann_bench::json_kv(out, first, "meta_growth_alloc_us",
                               stats.meta_growth_alloc_ns / 1000);
        bufann_bench::json_kv(out, first, "meta_growth_copy_us",
                               stats.meta_growth_copy_ns / 1000);
        bufann_bench::json_kv(out, first, "meta_growth_total_us",
                               stats.meta_growth_total_ns / 1000);
        bufann_bench::json_kv(out, first, "meta_growth_last_from",
                               stats.meta_growth_last_from);
        bufann_bench::json_kv(out, first, "meta_growth_last_to",
                               stats.meta_growth_last_to);
        bufann_bench::json_kv(out, first, "meta_growth_last_workers",
                               stats.meta_growth_last_workers);
        bufann_bench::json_kv(out, first, "meta_growth_last_copy_bytes",
                               stats.meta_growth_last_copy_bytes);
    }
    bufann_bench::json_kv(out, first, "cache_hit_rate",
                           bufann_bench::safe_div(
                               static_cast<double>(stats.cache_hits),
                               static_cast<double>(cache_lookups)));
    bufann_bench::json_kv(out, first, "cache_miss_rate",
                           bufann_bench::safe_div(
                               static_cast<double>(cache_misses),
                               static_cast<double>(cache_lookups)));
    bufann_bench::json_kv(out, first, "page_fault_total",
                           stats.page_fault_total);
    bufann_bench::json_kv(out, first, "pages_flushed",
                           stats.pages_flushed);
}

template <typename T>
static void preload_buffer_pool_pages(BufANNIndex<T> &idx,
                                      uint32_t max_pages,
                                      uint32_t preload_threads) {
    const uint32_t total_pages = bufann_store_snapshot(idx)->total_pages();
    const uint32_t target_pages =
        max_pages == 0 ? total_pages : std::min(total_pages, max_pages);
    if (target_pages == 0) return;

    std::cout << "Preloading heap pages into buffer pool: target_pages="
              << target_pages << " total_pages=" << total_pages
              << " threads=" << std::max<uint32_t>(1, preload_threads) << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    bufann_store_snapshot(idx)->preload_pages(target_pages, preload_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Preload complete in " << elapsed_s(t0, t1) << "s" << std::endl;
}

template <typename T>
static SearchMetrics run_search(BufANNIndex<T> &idx, T *queries_raw,
                                size_t nqueries, size_t qaligned_dim,
                                const std::string &gt_file,
                                uint32_t recall_at, uint32_t search_L,
                                uint32_t query_threads,
                                size_t source_nqueries = 0,
                                size_t source_start = 0,
                                const std::vector<uint32_t>& newly_inserted_tags = {}) {
    SearchMetrics m;
    m.nqueries = nqueries;
    std::vector<double> latency_us(nqueries, 0.0);
    std::vector<uint32_t> result_ids(nqueries * static_cast<size_t>(recall_at),
                                     UINT32_MAX);

    std::atomic<bool> failed{false};
    std::mutex mu;
    std::mutex progress_mu;
    std::string err;

    auto t0 = std::chrono::high_resolution_clock::now();
    const uint64_t total = static_cast<uint64_t>(nqueries);
    const uint64_t progress_step = std::max<uint64_t>(1, total / 5);
    std::atomic<uint64_t> completed_count{0};
#pragma omp parallel for num_threads(static_cast<int>(query_threads)) schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(nqueries); ++i) {
        if (failed.load(std::memory_order_acquire)) continue;
        try {
            auto q0 = std::chrono::high_resolution_clock::now();
            uint32_t *out_ptr =
                result_ids.data() + static_cast<size_t>(i) * recall_at;
            const size_t source_i =
                source_nqueries == 0
                    ? static_cast<size_t>(i)
                    : (source_start + static_cast<size_t>(i)) % source_nqueries;
            bufann_query_into(idx,
                               queries_raw + source_i * qaligned_dim,
                               recall_at, out_ptr, search_L);
            auto q1 = std::chrono::high_resolution_clock::now();
            latency_us[static_cast<size_t>(i)] =
                std::chrono::duration<double, std::micro>(q1 - q0).count();
            const uint64_t done = completed_count.fetch_add(
                                      1, std::memory_order_relaxed) +
                                  1;
            if ((done % progress_step) == 0 || done == total) {
                const auto now = std::chrono::high_resolution_clock::now();
                const double elapsed = elapsed_s(t0, now);
                const double pct = (100.0 * static_cast<double>(done)) /
                                   static_cast<double>(total);
                std::lock_guard<std::mutex> lk(progress_mu);
                std::cout << "Query progress: " << done << "/" << total
                          << " (" << std::fixed << std::setprecision(1)
                          << pct << "%, elapsed " << std::setprecision(2)
                          << elapsed << "s)" << std::endl;
            }
        } catch (const std::exception &e) {
            if (!failed.exchange(true)) {
                std::lock_guard<std::mutex> lk(mu);
                err = e.what();
            }
        }
    }
    if (failed.load()) throw std::runtime_error("search failed: " + err);
    auto t1 = std::chrono::high_resolution_clock::now();

    const double wall_s = elapsed_s(t0, t1);
    m.search_qps =
        (wall_s > 0.0 && nqueries > 0) ? static_cast<double>(nqueries) / wall_s
                                       : 0.0;
    const double total_lat =
        std::accumulate(latency_us.begin(), latency_us.end(), 0.0);
    m.lat_avg_us =
        nqueries > 0 ? total_lat / static_cast<double>(nqueries) : 0.0;
    m.lat_p50_us = percentile(latency_us, 0.50);
    m.lat_p90_us = percentile(latency_us, 0.90);
    m.lat_p95_us = percentile(latency_us, 0.95);
    m.lat_p99_us = percentile(latency_us, 0.99);
    m.lat_p999_us = percentile(latency_us, 0.999);

    if (!gt_file.empty()) {
        uint32_t *gt_ids = nullptr;
        float *gt_dists = nullptr;
        uint32_t *gt_tags = nullptr;
        size_t gt_npts = 0;
        size_t gt_dim = 0;
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_npts, gt_dim,
                               &gt_tags);
        uint32_t* gt_eval = gt_tags != nullptr ? gt_tags : gt_ids;
        const size_t eval_queries = std::min(nqueries, gt_npts);
        m.recall = static_cast<double>(diskann::calculate_recall(
            static_cast<unsigned>(eval_queries), gt_eval, gt_dists,
            static_cast<unsigned>(gt_dim), result_ids.data(), recall_at,
            recall_at));
        if (!newly_inserted_tags.empty()) {
            std::unordered_set<uint32_t> inserted;
            inserted.reserve(newly_inserted_tags.size() * 2 + 1);
            inserted.insert(newly_inserted_tags.begin(), newly_inserted_tags.end());
            const size_t gt_limit =
                std::min<size_t>(static_cast<size_t>(recall_at), gt_dim);
            for (size_t qi = 0; qi < eval_queries; ++qi) {
                const uint32_t* gt_vec = gt_eval + qi * gt_dim;
                const uint32_t* res_vec =
                    result_ids.data() + qi * static_cast<size_t>(recall_at);
                std::unordered_set<uint32_t> retrieved;
                retrieved.reserve(static_cast<size_t>(recall_at) * 2 + 1);
                retrieved.insert(res_vec, res_vec + recall_at);
                for (size_t j = 0; j < gt_limit; ++j) {
                    const uint32_t tag = gt_vec[j];
                    if (inserted.find(tag) == inserted.end()) continue;
                    ++m.newly_inserted_results_gt;
                    if (retrieved.find(tag) != retrieved.end()) {
                        ++m.newly_inserted_results_hit;
                    }
                }
            }
        }
        delete[] gt_ids;
        delete[] gt_dists;
        delete[] gt_tags;
    }
    return m;
}

template <typename T>
static double measure_round_recall(BufANNIndex<T> &idx, T *queries_raw,
                                 size_t nqueries, size_t qaligned_dim,
                                 const std::string &gt_path, uint32_t recall_at,
                                 uint32_t recall_search_L, uint32_t query_threads,
                                 bufann_bench::RssSampler &rss_sampler,
                                 BufANNWindowMetrics *window_metrics,
                                 StatsSnapshot &per_round_recall_stats,
                                 bool resume_rss_after) {
    if (window_metrics != nullptr) {
        window_metrics->pause_rss();
    } else {
        rss_sampler.stop();
    }
    const StatsSnapshot before =
        snapshot_stats(bufann_store_snapshot(idx)->stats(), bufann_store_snapshot(idx)->page_size());
    const SearchMetrics rm =
        run_search(idx, queries_raw, nqueries, qaligned_dim, gt_path, recall_at,
                   recall_search_L, query_threads);
    add_stats(per_round_recall_stats,
              stats_delta(snapshot_stats(bufann_store_snapshot(idx)->stats(), bufann_store_snapshot(idx)->page_size()),
                          before));
    if (resume_rss_after && window_metrics != nullptr) {
        window_metrics->resume_rss();
    } else if (resume_rss_after) {
        rss_sampler.resume();
    }
    return rm.recall;
}

template <typename T>
static double run_deletes(BufANNIndex<T> &idx,
                          const std::vector<uint32_t> &delete_ids,
                          uint32_t delete_threads,
                          std::vector<double> &op_latencies_us,
                          const std::function<void()> &progress_notify = {}) {
    op_latencies_us.clear();
    if (delete_ids.empty()) return 0.0;
    op_latencies_us.assign(delete_ids.size(), 0.0);
    std::atomic<bool> failed{false};
    std::mutex mu;
    std::mutex progress_mu;
    std::string err;
    std::atomic<uint64_t> deleted_count{0};

    auto t0 = std::chrono::high_resolution_clock::now();
    const uint64_t total = static_cast<uint64_t>(delete_ids.size());
    const uint64_t progress_step = std::max<uint64_t>(1, total / 20);
#pragma omp parallel for num_threads(static_cast<int>(delete_threads)) schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(delete_ids.size()); ++i) {
        if (failed.load(std::memory_order_acquire)) continue;
        try {
            auto op_t0 = std::chrono::high_resolution_clock::now();
            bufann_delete(idx, delete_ids[static_cast<size_t>(i)]);
            auto op_t1 = std::chrono::high_resolution_clock::now();
            op_latencies_us[static_cast<size_t>(i)] =
                static_cast<double>(elapsed_ns(op_t0, op_t1)) / 1000.0;
            const uint64_t done =
                deleted_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if ((done % progress_step) == 0 || done == total) {
                const auto now = std::chrono::high_resolution_clock::now();
                const double elapsed = elapsed_s(t0, now);
                const double pct = (100.0 * static_cast<double>(done)) /
                                   static_cast<double>(total);
                std::lock_guard<std::mutex> lk(progress_mu);
                std::cout << "Delete progress: " << done << "/" << total
                          << " (" << std::fixed << std::setprecision(1)
                          << pct << "%, elapsed " << std::setprecision(2)
                          << elapsed << "s)" << std::endl;
            }
            if (progress_notify && ((done & 255ULL) == 0)) {
                progress_notify();
            }
        } catch (const std::exception &e) {
            if (!failed.exchange(true)) {
                std::lock_guard<std::mutex> lk(mu);
                err = e.what();
            }
        }
    }
    if (failed.load()) throw std::runtime_error("delete failed: " + err);
    auto t1 = std::chrono::high_resolution_clock::now();
    return elapsed_s(t0, t1);
}

template <typename T>
static InsertRunMetrics run_inserts(BufANNIndex<T> &idx,
                                    const std::string &full_data_file,
                                    const std::vector<uint32_t> &insert_ids,
                                    uint32_t insert_search_L,
                                    uint32_t insert_threads,
                                    std::vector<double> &op_latencies_us,
                                    const std::function<void()> &progress_notify = {}) {
    InsertRunMetrics metrics;
    op_latencies_us.clear();
    if (insert_ids.empty()) return metrics;
    op_latencies_us.assign(insert_ids.size(), 0.0);
    if (full_data_file.empty()) {
        throw std::runtime_error("--full_data_file is required for inserts");
    }

    std::vector<uint32_t> sorted_ids = insert_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    const size_t required_internal_ids =
        static_cast<size_t>(bufann_store_snapshot(idx)->num_active()) + sorted_ids.size() + 1;
    const size_t reserve_internal_ids = std::max<size_t>(
        idx.config.max_dataset_size, required_internal_ids);
    if (reserve_internal_ids > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("insert metadata reservation exceeds uint32_t node ID range");
    }
    std::cout << "reserve_meta_capacity: " << reserve_internal_ids << "..." << std::endl;
    bufann_store_snapshot(idx)->reserve_meta_capacity(reserve_internal_ids);
    std::cout << "reserve_meta_capacity done" << std::endl;
    auto sorted_coords = load_coords_by_ids<T>(
        full_data_file, sorted_ids, idx.config.dim, idx.aligned_dim);
    std::cout << "load_coords_by_ids done. Starting insertion timing" << std::endl;

    std::atomic<bool> failed{false};
    std::atomic<uint64_t> inserted_count{0};
    std::mutex mu;
    std::mutex progress_mu;
    std::string err;
    metrics.background_repair = false;

    auto t0 = std::chrono::high_resolution_clock::now();
    const uint64_t total = static_cast<uint64_t>(sorted_ids.size());
    const uint64_t progress_step = std::max<uint64_t>(1, total / 20);
#pragma omp parallel for num_threads(static_cast<int>(insert_threads)) schedule(dynamic, 1)
    for (int64_t si = 0; si < static_cast<int64_t>(sorted_ids.size()); ++si) {
        if (failed.load(std::memory_order_acquire)) continue;
        try {
            auto op_t0 = std::chrono::high_resolution_clock::now();
            bufann_insert(idx, sorted_ids[static_cast<size_t>(si)],
                           sorted_coords.data() + static_cast<size_t>(si) *
                                                      idx.aligned_dim,
                           insert_search_L);
            auto op_t1 = std::chrono::high_resolution_clock::now();
            op_latencies_us[static_cast<size_t>(si)] =
                static_cast<double>(elapsed_ns(op_t0, op_t1)) / 1000.0;
            const uint64_t done = inserted_count.fetch_add(
                                      1, std::memory_order_relaxed) +
                                  1;
            if ((done % progress_step) == 0 || done == total) {
                const auto now = std::chrono::high_resolution_clock::now();
                const double elapsed = elapsed_s(t0, now);
                const double pct = (100.0 * static_cast<double>(done)) /
                                   static_cast<double>(total);
                std::lock_guard<std::mutex> lk(progress_mu);
                std::cout << "Insert progress: " << done << "/" << total
                          << " (" << std::fixed << std::setprecision(1)
                          << pct << "%, elapsed " << std::setprecision(2)
                          << elapsed << "s)" << std::endl;
            }
            if (progress_notify && ((done & 255ULL) == 0)) {
                progress_notify();
            }
        } catch (const std::exception &e) {
            if (!failed.exchange(true)) {
                std::lock_guard<std::mutex> lk(mu);
                err = e.what();
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    metrics.elapsed_s = elapsed_s(t0, t1);
    if (failed.load(std::memory_order_acquire)) {
        throw std::runtime_error("insert failed: " + err);
    }
    metrics.maintenance_s = 0.0;
    return metrics;
}

// Result of one concurrent (insert+delete+query) update phase driven by the
// shared weighted-draw worker pool. Used only by search_after_update.
struct ConcurrentUpdateResult {
    double insert_busy_s = 0.0;   // sum of per-op insert latencies (seconds)
    double delete_busy_s = 0.0;   // sum of per-op delete latencies (seconds)
    double foreground_wall_s = 0.0;
    std::vector<double> insert_op_latencies_us;
    std::vector<double> delete_op_latencies_us;
    SearchMetrics query;
    bool have_query = false;
};

// Drive inserts, deletes, and concurrent (stress) queries through one unified
// worker pool, each worker drawing an op type with probability proportional to
// that op's remaining count (see weighted_update_pool.h). Queries are drawn
// cyclically from queries_raw[0, source_nqueries) and their recall is NOT
// computed -- accuracy is the standalone final query after maintenance. The
// delete path uses single-op bufann_delete.
template <typename T>
static ConcurrentUpdateResult run_concurrent_update(
    BufANNIndex<T> &idx, const std::string &full_data_file,
    const std::vector<uint32_t> &insert_ids, uint32_t insert_search_L,
    const std::vector<uint32_t> &delete_ids, StreamingQuerySource<T> *query_source,
    size_t query_ratio,
    uint32_t recall_at, uint32_t search_L, uint32_t pool_threads,
    BufANNWindowMetrics *window, uint64_t round_index,
    bool skip_meta_pre_reserve = false) {
    using clock = std::chrono::high_resolution_clock;
    ConcurrentUpdateResult r;

    // Insert prep mirrors run_inserts: sort ids, optionally reserve metadata,
    // load coords. skip_meta_pre_reserve forces growth under concurrent load.
    std::vector<uint32_t> sorted_ids;
    std::vector<T> sorted_coords;
    if (!insert_ids.empty()) {
        if (full_data_file.empty()) {
            throw std::runtime_error("--full_data_file is required for inserts");
        }
        sorted_ids = insert_ids;
        std::sort(sorted_ids.begin(), sorted_ids.end());
        if (!skip_meta_pre_reserve) {
            const size_t required_internal_ids =
                static_cast<size_t>(bufann_store_snapshot(idx)->num_active()) +
                sorted_ids.size() + 1;
            const size_t reserve_internal_ids = std::max<size_t>(
                idx.config.max_dataset_size, required_internal_ids);
            if (reserve_internal_ids > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    "insert metadata reservation exceeds uint32_t node ID range");
            }
            bufann_store_snapshot(idx)->reserve_meta_capacity(reserve_internal_ids);
        }
        sorted_coords = load_coords_by_ids<T>(full_data_file, sorted_ids,
                                              idx.config.dim, idx.aligned_dim);
    }

    const size_t insert_n = sorted_ids.size();
    const size_t delete_n = delete_ids.size();
    const bool query_enabled =
        query_source != nullptr && query_source->pool_size() > 0 && query_ratio > 0;
    const size_t query_n =
        query_enabled
            ? ann_bench::mixed_update_query_count(insert_n, delete_n,
                                                  query_ratio)
            : 0;

    r.insert_op_latencies_us.assign(insert_n, 0.0);
    r.delete_op_latencies_us.assign(delete_n, 0.0);
    QueryLatencySampler query_lat;

    std::atomic<bool> failed{false};
    std::mutex mu;
    std::string err;
    auto record_failure = [&](const std::exception &e) {
        if (!failed.exchange(true)) {
            std::lock_guard<std::mutex> lk(mu);
            err = e.what();
        }
    };

    std::function<void(uint64_t)> tasks[ann_bench::WeightedUpdatePool::NUM_OPS];
    auto record_any_failure = [&](const std::string &msg) {
        if (!failed.exchange(true)) {
            std::lock_guard<std::mutex> lk(mu);
            err = msg;
        }
    };
    tasks[ann_bench::WeightedUpdatePool::INSERT] = [&](uint64_t i) {
        if (failed.load(std::memory_order_acquire)) return;
        try {
            auto t0 = clock::now();
            bufann_insert(idx, sorted_ids[i],
                           sorted_coords.data() + i * idx.aligned_dim,
                           insert_search_L);
            auto t1 = clock::now();
            r.insert_op_latencies_us[i] =
                static_cast<double>(elapsed_ns(t0, t1)) / 1000.0;
            if (window != nullptr) {
                window->record_insert(r.insert_op_latencies_us[i]);
            }
        } catch (const diskann::ANNException &e) {
            // ANNException does not inherit std::exception; uncaught would
            // terminate the whole process from a pool thread.
            record_any_failure(std::string("insert ANNException: ") +
                               e.message());
        } catch (const std::exception &e) {
            record_failure(e);
        }
    };
    tasks[ann_bench::WeightedUpdatePool::DELETE] = [&](uint64_t i) {
        if (failed.load(std::memory_order_acquire)) return;
        try {
            auto t0 = clock::now();
            bufann_delete(idx, delete_ids[i]);
            auto t1 = clock::now();
            r.delete_op_latencies_us[i] =
                static_cast<double>(elapsed_ns(t0, t1)) / 1000.0;
            if (window != nullptr) {
                window->record_delete(r.delete_op_latencies_us[i]);
            }
        } catch (const diskann::ANNException &e) {
            record_any_failure(std::string("delete ANNException: ") +
                               e.message());
        } catch (const std::exception &e) {
            record_failure(e);
        }
    };
    tasks[ann_bench::WeightedUpdatePool::QUERY] = [&](uint64_t i) {
        if (failed.load(std::memory_order_acquire)) return;
        try {
            thread_local typename StreamingQuerySource<T>::ThreadState qstate;
            qstate.result_ids.resize(recall_at);
            const T *query = query_source->get(i, qstate);
            auto t0 = clock::now();
            bufann_query_into(
                idx, query, recall_at, qstate.result_ids.data(), search_L);
            auto t1 = clock::now();
            const double lat_us =
                static_cast<double>(elapsed_ns(t0, t1)) / 1000.0;
            if (window != nullptr) {
                window->record_query(lat_us, round_index, "foreground");
            } else {
                query_lat.record(i, lat_us);
            }
        } catch (const diskann::ANNException &e) {
            record_any_failure(std::string("query ANNException: ") +
                               e.message());
        } catch (const std::exception &e) {
            record_failure(e);
        }
    };

    const uint64_t counts[ann_bench::WeightedUpdatePool::NUM_OPS] = {
        static_cast<uint64_t>(insert_n), static_cast<uint64_t>(delete_n),
        static_cast<uint64_t>(query_n)};

    auto fg0 = clock::now();
    if (ann_bench::env_equals("UPDATE_POOL_MODE", "round_robin")) {
        ann_bench::WeightedUpdatePool::run_rq2_round_robin_90_5_5(
            pool_threads, counts, tasks);
    } else {
        ann_bench::WeightedUpdatePool::run(pool_threads, counts, tasks);
    }
    auto fg1 = clock::now();
    if (failed.load()) {
        throw std::runtime_error("concurrent update failed: " + err);
    }
    r.foreground_wall_s = elapsed_s(fg0, fg1);

    // Inserts and deletes overlap, so per-op-type wall time isn't separable;
    // report aggregate busy time (sum of op latencies) like the baselines.
    r.insert_busy_s = std::accumulate(r.insert_op_latencies_us.begin(),
                                      r.insert_op_latencies_us.end(), 0.0) /
                      1e6;
    r.delete_busy_s = std::accumulate(r.delete_op_latencies_us.begin(),
                                      r.delete_op_latencies_us.end(), 0.0) /
                      1e6;

    if (query_enabled) {
        r.have_query = true;
        r.query = query_lat.metrics(r.foreground_wall_s);
        if (window != nullptr) {
            r.query.nqueries = query_n;
            r.query.search_qps =
                r.foreground_wall_s > 0.0
                    ? static_cast<double>(query_n) / r.foreground_wall_s
                    : 0.0;
        }
    }
    return r;
}

template <typename T>
static SearchMetrics run_queries_until_done(BufANNIndex<T> &idx,
                                            StreamingQuerySource<T> *query_source,
                                            std::atomic<bool> &done,
                                            uint32_t query_threads,
                                            uint32_t recall_at,
                                            uint32_t search_L,
                                            uint64_t start_seq,
                                            BufANNWindowMetrics *window,
                                            uint64_t round_index,
                                            uint64_t query_budget = 0,
                                            const char *stage = "merge") {
    QueryLatencySampler lat;
    std::vector<std::thread> workers;
    if (query_source == nullptr || query_source->pool_size() == 0 ||
        query_threads == 0) {
        return lat.metrics(0.0);
    }
    std::atomic<uint64_t> next_budget{0};
    std::atomic<uint64_t> completed{0};
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t tid = 0; tid < query_threads; tid++) {
        workers.emplace_back([&, tid]() {
            typename StreamingQuerySource<T>::ThreadState qstate;
            qstate.result_ids.resize(recall_at);
            uint64_t seq = start_seq + tid;
            while (!done.load(std::memory_order_acquire)) {
                if (query_budget > 0) {
                    const uint64_t local =
                        next_budget.fetch_add(1, std::memory_order_relaxed);
                    if (local >= query_budget) {
                        break;
                    }
                    seq = start_seq + local;
                }
                const T *query = query_source->get(seq, qstate);
                const auto qs = std::chrono::high_resolution_clock::now();
                bufann_query_into(idx, query, recall_at, qstate.result_ids.data(), search_L);
                const auto qe = std::chrono::high_resolution_clock::now();
                const double lat_us =
                    static_cast<double>(elapsed_ns(qs, qe)) / 1000.0;
                if (window != nullptr) {
                    window->record_query(lat_us, round_index, stage);
                } else {
                    lat.record(seq, lat_us);
                }
                completed.fetch_add(1, std::memory_order_relaxed);
                if (query_budget == 0) {
                    seq += query_threads;
                }
            }
        });
    }
    for (auto &th : workers) th.join();
    const auto t1 = std::chrono::high_resolution_clock::now();
    SearchMetrics m = lat.metrics(elapsed_s(t0, t1));
    if (window != nullptr) {
        m.nqueries = completed.load(std::memory_order_relaxed);
        m.search_qps = m.nqueries > 0
                           ? static_cast<double>(m.nqueries) / elapsed_s(t0, t1)
                           : 0.0;
    }
    return m;
}

template <typename T>
static SearchMetrics run_queries_fixed_count(BufANNIndex<T> &idx,
                                             StreamingQuerySource<T> *query_source,
                                             uint64_t query_count,
                                             uint32_t query_threads,
                                             uint32_t recall_at,
                                             uint32_t search_L,
                                             uint64_t start_seq,
                                             BufANNWindowMetrics *window,
                                             uint64_t round_index,
                                             const char *stage) {
    QueryLatencySampler lat;
    if (query_source == nullptr || query_source->pool_size() == 0 ||
        query_threads == 0 || query_count == 0) {
        return lat.metrics(0.0);
    }
    std::atomic<uint64_t> next{0};
    std::vector<std::thread> workers;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t tid = 0; tid < query_threads; tid++) {
        workers.emplace_back([&]() {
            typename StreamingQuerySource<T>::ThreadState qstate;
            qstate.result_ids.resize(recall_at);
            for (;;) {
                const uint64_t local = next.fetch_add(1, std::memory_order_relaxed);
                if (local >= query_count) {
                    break;
                }
                const uint64_t seq = start_seq + local;
                const T *query = query_source->get(seq, qstate);
                const auto qs = std::chrono::high_resolution_clock::now();
                bufann_query_into(idx, query, recall_at, qstate.result_ids.data(), search_L);
                const auto qe = std::chrono::high_resolution_clock::now();
                const double lat_us =
                    static_cast<double>(elapsed_ns(qs, qe)) / 1000.0;
                if (window != nullptr) {
                    window->record_query(lat_us, round_index, stage);
                } else {
                    lat.record(seq, lat_us);
                }
            }
        });
    }
    for (auto &th : workers) th.join();
    const auto t1 = std::chrono::high_resolution_clock::now();
    SearchMetrics m = lat.metrics(elapsed_s(t0, t1));
    if (window != nullptr) {
        m.nqueries = query_count;
        m.search_qps = m.nqueries > 0
                           ? static_cast<double>(m.nqueries) / elapsed_s(t0, t1)
                           : 0.0;
    }
    return m;
}

static void emit_canonical_search_json(std::ostream &out,
                                       const std::string &phase,
                                       const SearchMetrics &s,
                                       const StatsSnapshot &stats,
                                       const MetricConfig &config,
                                       double avg_rss_mb,
                                       double peak_rss_mb) {
    const double query_time_s =
        s.search_qps > 0.0 ? static_cast<double>(s.nqueries) / s.search_qps : 0.0;
    bool first = true;
    out << std::fixed << std::setprecision(6) << "{";
    bufann_bench::json_kv(out, first, "baseline", "BufANN");
    bufann_bench::json_kv(out, first, "workload", "query");
    bufann_bench::json_kv(out, first, "phase", phase);
    bufann_bench::json_kv(out, first, "data_type", config.data_type);
    bufann_bench::json_kv(out, first, "page_table", diskann::inplace::kPageTableName);
    bufann_bench::json_kv(out, first, "search_pq", config.search_pq ? 1 : 0);
    bufann_bench::json_kv(out, first, "pq_chunks", config.pq_chunks);
    bufann_bench::json_kv(out, first, "recall_at", config.recall_at);
    bufann_bench::json_kv(out, first, "search_L", config.search_L);
    bufann_bench::json_kv(out, first, "beamwidth", config.beamwidth);
    bufann_bench::json_kv(out, first, "buffer_pool_frames",
                           config.buffer_pool_frames);

    bufann_bench::json_kv(out, first, "query_threads",
                           config.query_threads);
    bufann_bench::json_kv(out, first, "query_count", s.nqueries);
    bufann_bench::json_kv(out, first, "query_time_s", query_time_s);
    bufann_bench::json_kv(out, first, "query_qps", s.search_qps);
    bufann_bench::json_kv(out, first, "query_lat_avg_us", s.lat_avg_us);
    bufann_bench::json_kv(out, first, "query_lat_p50_us", s.lat_p50_us);
    bufann_bench::json_kv(out, first, "query_lat_p90_us", s.lat_p90_us);
    bufann_bench::json_kv(out, first, "query_lat_p95_us", s.lat_p95_us);
    bufann_bench::json_kv(out, first, "query_lat_p99_us", s.lat_p99_us);
    bufann_bench::json_kv(out, first, "query_lat_p999_us", s.lat_p999_us);
    bufann_bench::json_kv(out, first, "recall", s.recall);
    bufann_bench::json_kv(out, first, "newly_inserted_results_gt",
                           s.newly_inserted_results_gt);
    bufann_bench::json_kv(out, first, "newly_inserted_results_hit",
                           s.newly_inserted_results_hit);
    emit_bufann_storage_stats(out, first, stats);
    // IO wait time on the query path. query_io_ns is summed across all
    // worker threads (each OMP worker bumps the same atomic around the
    // blocking pread). Dividing by nqueries gives average per-query IO
    // wait, which is comparable to per-query wall latency regardless of
    // thread count.
    const double avg_query_io_us =
        bufann_bench::safe_div(static_cast<double>(stats.query_io_ns) / 1000.0,
                                static_cast<double>(s.nqueries));
    bufann_bench::json_kv(out, first, "query_io_ns_total", stats.query_io_ns);
    bufann_bench::json_kv(out, first, "query_io_avg_us_per_query",
                           avg_query_io_us);
    bufann_bench::json_kv(out, first, "query_io_fraction_of_latency",
                           bufann_bench::safe_div(avg_query_io_us,
                                                   s.lat_avg_us));
    bufann_bench::json_kv(out, first, "avg_rss_mb", avg_rss_mb);
    bufann_bench::json_kv(out, first, "peak_rss_mb", peak_rss_mb);
    emit_bufann_logical_io(out, first, stats, query_time_s,
                            static_cast<uint64_t>(s.nqueries));
    out << "}" << std::endl;
}

static void emit_canonical_update_json(std::ostream &out,
                                       const std::string &canonical_workload,
                                       const UpdateMetrics &update,
                                       const StatsSnapshot &stats,
                                       const MetricConfig &config,
                                       double avg_rss_mb,
                                       double peak_rss_mb) {
    const bool is_mixed = canonical_workload == "mixed_update";
    const double workload_time_s =
        update.total_update_wall_time_s > 0.0
            ? update.total_update_wall_time_s
            : update.foreground_update_wall_time_s;
    const size_t n_updates = update.n_inserts + update.n_deletes;
    const uint64_t total_ops =
        static_cast<uint64_t>(n_updates + update.query_count);
    bool first = true;
    out << std::fixed << std::setprecision(6) << "{";
    bufann_bench::json_kv(out, first, "baseline", "BufANN");
    bufann_bench::json_kv(out, first, "workload", canonical_workload);
    bufann_bench::json_kv(out, first, "phase", "workload");
    bufann_bench::json_kv(out, first, "elapsed_s", update.elapsed_s);
    bufann_bench::json_kv(out, first, "data_type", config.data_type);
    bufann_bench::json_kv(out, first, "pq_chunks", config.pq_chunks);
    bufann_bench::json_kv(out, first, "recall_at", config.recall_at);
    bufann_bench::json_kv(out, first, "search_L", config.search_L);
    bufann_bench::json_kv(out, first, "insert_search_L",
                           config.insert_search_L);
    bufann_bench::json_kv(out, first, "graph_L", config.graph_L);
    bufann_bench::json_kv(out, first, "delete_algorithm", config.delete_algorithm);
    bufann_bench::json_kv(out, first, "page_table", diskann::inplace::kPageTableName);
    bufann_bench::json_kv(out, first, "search_pq", config.search_pq ? 1 : 0);
    bufann_bench::json_kv(out, first, "beamwidth", config.beamwidth);
    bufann_bench::json_kv(out, first, "buffer_pool_frames",
                           config.buffer_pool_frames);

    bufann_bench::json_kv(out, first, "query_threads",
                           config.query_threads);
    bufann_bench::json_kv(out, first, "insert_threads",
                           config.insert_threads);
    bufann_bench::json_kv(out, first, "delete_threads",
                           config.delete_threads);
    bufann_bench::json_kv(out, first, "maintenance_threads",
                           config.maintenance_threads);
    bufann_bench::json_kv(out, first, "maintenance_query_threads",
                           config.maintenance_query_threads);
    bufann_bench::json_kv(out, first, "skip_update_search",
                           config.skip_update_search ? 1 : 0);
    bufann_bench::json_kv(out, first, "flush_background",
                           config.flush_background ? 1 : 0);
    bufann_bench::json_kv(out, first, "run_maintenance",
                           config.run_maintenance ? 1 : 0);
    bufann_bench::json_kv(out, first, "flush_after_maintenance",
                           config.flush_after_maintenance ? 1 : 0);
    bufann_bench::json_kv(out, first, "insert_count", update.n_inserts);
    bufann_bench::json_kv(out, first, "delete_count", update.n_deletes);
    if (is_mixed) {
        bufann_bench::json_kv(out, first, "query_count", update.query_count);
    }
    bufann_bench::json_kv(out, first, "foreground_time_s",
                           update.foreground_update_wall_time_s);
    bufann_bench::json_kv(out, first, "insert_foreground_time_s",
                           update.insert_time_s);
    bufann_bench::json_kv(out, first, "delete_foreground_time_s",
                           update.delete_time_s);
    if (is_mixed) {
        bufann_bench::json_kv(out, first, "query_foreground_time_s",
                               update.query_time_s);
        bufann_bench::json_kv(out, first, "query_time_s",
                               update.query_time_s);
    }
    bufann_bench::json_kv(out, first, "maintenance_time_s",
                           update.maintenance_wall_time_s);
    if (is_mixed) {
        bufann_bench::json_kv(out, first, "post_merge_query_time_s",
                               update.post_merge_query_time_s);
    }
    bufann_bench::json_kv(out, first, "background_repair_time_s",
                           update.maintenance_wall_time_s);
    bufann_bench::json_kv(out, first, "flush_time_s",
                           update.flush_wall_time_s);
    bufann_bench::json_kv(out, first, "workload_time_s", workload_time_s);
    if (is_mixed) {
        bufann_bench::json_kv(out, first, "query_qps", update.query_qps);
    }
    bufann_bench::json_kv(out, first, "insert_ops_per_s",
                           update.insert_ops_per_s);
    bufann_bench::json_kv(out, first, "delete_ops_per_s",
                           update.delete_ops_per_s);
    bufann_bench::json_kv(out, first, "update_ops_per_s",
                           update.update_ops_per_s);
    bufann_bench::json_kv(out, first, "overall_ops_per_s",
                           update.overall_ops_per_s);
    if (is_mixed) {
        bufann_bench::json_kv(out, first, "query_lat_avg_us",
                               update.query_lat_avg_us);
        bufann_bench::json_kv(out, first, "query_lat_p50_us",
                               update.query_lat_p50_us);
        bufann_bench::json_kv(out, first, "query_lat_p90_us",
                               update.query_lat_p90_us);
        bufann_bench::json_kv(out, first, "query_lat_p95_us",
                               update.query_lat_p95_us);
        bufann_bench::json_kv(out, first, "query_lat_p99_us",
                               update.query_lat_p99_us);
        bufann_bench::json_kv(out, first, "query_lat_p999_us",
                               update.query_lat_p999_us);
    }
    bufann_bench::json_kv(out, first, "insert_call_lat_avg_us",
                           update.insert_lat_avg_us);
    bufann_bench::json_kv(out, first, "insert_call_lat_p50_us",
                           update.insert_lat_p50_us);
    bufann_bench::json_kv(out, first, "insert_call_lat_p90_us",
                           update.insert_lat_p90_us);
    bufann_bench::json_kv(out, first, "insert_call_lat_p95_us",
                           update.insert_lat_p95_us);
    bufann_bench::json_kv(out, first, "insert_call_lat_p99_us",
                           update.insert_lat_p99_us);
    bufann_bench::json_kv(out, first, "delete_call_lat_avg_us",
                           update.delete_lat_avg_us);
    bufann_bench::json_kv(out, first, "delete_call_lat_p50_us",
                           update.delete_lat_p50_us);
    bufann_bench::json_kv(out, first, "delete_call_lat_p90_us",
                           update.delete_lat_p90_us);
    bufann_bench::json_kv(out, first, "delete_call_lat_p95_us",
                           update.delete_lat_p95_us);
    bufann_bench::json_kv(out, first, "delete_call_lat_p99_us",
                           update.delete_lat_p99_us);
    bufann_bench::json_kv(out, first, "avg_rss_mb", avg_rss_mb);
    bufann_bench::json_kv(out, first, "peak_rss_mb", peak_rss_mb);
    emit_bufann_storage_stats(out, first, stats);
    emit_bufann_logical_io(out, first, stats, workload_time_s, total_ops);
    out << "}" << std::endl;
}

template <typename T>
static int run_workload(int argc, char **argv) {
    const std::string index_prefix = get_arg(argc, argv, "--index_prefix");
    const std::string query_file = get_arg(argc, argv, "--query_file");
    const std::string workload = get_arg(argc, argv, "--workload");
    const std::string data_type = get_arg(argc, argv, "--data_type", "float");
    const std::string gt_file = get_arg(argc, argv, "--gt_file", "");
    const std::string result_file = get_arg(argc, argv, "--result_file", "");

    if (index_prefix.empty() || workload.empty()) {
        std::cerr << "ERROR: --index_prefix and --workload are required"
                  << std::endl;
        return 1;
    }
    if (workload != "search_only" && workload != "search_after_update" &&
        workload != "insert_only" && workload != "delete_only") {
        std::cerr << "ERROR: --workload must be search_only, search_after_update, "
                     "insert_only, delete_only\n";
        return 1;
    }
    if (query_file.empty()) {
        std::cerr << "ERROR: --query_file is required" << std::endl;
        return 1;
    }

    BufANNConfig cfg;
    cfg.dim =
        static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--dim", "0")));
    if (cfg.dim == 0) {
        std::cerr << "ERROR: --dim must be non-zero" << std::endl;
        return 1;
    }
    cfg.R = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--R", "64")));
    cfg.L = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--L", "100")));
    cfg.C = static_cast<uint32_t>(std::stoul(get_arg(argc, argv, "--C", "160")));
    cfg.alpha = std::stof(get_arg(argc, argv, "--alpha", "1.2"));
    cfg.beamwidth = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--beamwidth", "4")));
    cfg.search_pq = get_arg(argc, argv, "--search_pq", "1") != "0";
    cfg.buffer_pool_frames = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--buffer_pool_frames", "16384")));
    cfg.pq_chunks = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--pq_chunks", "0")));
    cfg.max_dataset_size = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--max_dataset_size", "0")));
    cfg.meta_growth_copy_threads = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--meta_growth_copy_threads", "16")));
    if (cfg.meta_growth_copy_threads == 0) {
        throw std::runtime_error(
            "meta_growth_copy_threads must be non-zero");
    }

    const uint32_t recall_at = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--recall_at", "10")));
    // --search_L accepts either a single value or a whitespace/comma-separated
    // list. For search_only we sweep the whole list (one measured run + JSON
    // record per L, re-warming between Ls) so a QPS/Recall curve comes out of a
    // single driver invocation. Update workloads use only the first value.
    const std::string search_L_arg = get_arg(argc, argv, "--search_L", "100");
    std::vector<uint32_t> search_L_list;
    {
        std::string tok;
        std::stringstream ss(search_L_arg);
        while (ss >> tok) {
            size_t pos = 0;
            while (pos < tok.size()) {
                size_t comma = tok.find(',', pos);
                std::string piece = tok.substr(
                    pos, comma == std::string::npos ? std::string::npos
                                                    : comma - pos);
                if (!piece.empty())
                    search_L_list.push_back(
                        static_cast<uint32_t>(std::stoul(piece)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
    }
    if (search_L_list.empty()) search_L_list.push_back(100);
    const uint32_t search_L = search_L_list.front();
    const uint32_t query_threads = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--query_threads", "1")));
    const uint32_t query_ratio = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--query_ratio", "80")));
    const bool skip_update_search =
        parse_bool_arg(argc, argv, "--skip_update_search", false);
    if (!skip_update_search) {
        if (query_ratio == 0) {
            throw std::runtime_error(
                "query_ratio must be > 0 when skip_update_search=0");
        }
        if (query_ratio >= 100) {
            throw std::runtime_error(
                "query_ratio must be in [1, 99] (percent of foreground ops "
                "that are queries)");
        }
    }
    const bool enable_interval_metrics = parse_bool_arg(
        argc, argv, "--enable_interval_metrics", false);
    const uint64_t tail_query_begin = static_cast<uint64_t>(
        std::stoull(get_arg(argc, argv, "--tail_query_begin", "0")));
    const uint64_t tail_query_end = static_cast<uint64_t>(
        std::stoull(get_arg(argc, argv, "--tail_query_end", "0")));
    const uint32_t warmup_threads = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--warmup_threads",
                           get_arg(argc, argv, "--preload_threads",
                                   std::to_string(query_threads)))));
    const uint32_t warmup_queries = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--warmup_queries", "0")));
    // Optional: separate warmup query file (e.g. rows sampled from the base
    // vectors) so cache fill doesn't overlap the timed query set. Takes
    // precedence over --warmup_queries when both are set.
    const std::string warmup_query_file =
        get_arg(argc, argv, "--warmup_query_file", "");
    const uint32_t insert_search_L = static_cast<uint32_t>(std::stoul(
        get_arg(argc, argv, "--insert_search_L", std::to_string(search_L))));
    uint32_t insert_threads = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--insert_threads", "1")));
    uint32_t delete_threads = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--delete_threads", "1")));
    uint32_t maintenance_threads = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--maintenance_threads",
                           std::to_string(std::max<uint32_t>(1u, delete_threads)))));
    uint32_t maintenance_query_threads = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--maintenance_query_threads",
                           std::to_string(query_threads))));
    const uint32_t effective_query_ratio =
        skip_update_search ? 0u : query_ratio;
    const uint32_t effective_maintenance_query_threads =
        skip_update_search ? 0u : maintenance_query_threads;
    // search_after_update drives inserts/deletes/queries through one unified
    // worker pool of this size (the per-op --insert/--delete/--query_threads
    // are ignored for that workload). Defaults to query_threads.
    const uint32_t pool_threads = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--pool_threads",
                           std::to_string(query_threads))));
    // Per-round structure for insert_only/delete_only (mirrors the DiskANN
    // family's n_iters/count_per_iter). cap==0 means a single round over all
    // ids. Each round applies its id chunk, runs bufann_cleanup_deleted_edges
    // as maintenance (when it deleted), then measures recall against that
    // round's GT at <gt_dir>/round_<k>_gt<recall_at>.bin and emits a
    // phase:"round" record. recall_search_L is the L for that recall search.
    const uint32_t insert_cap = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--insert_cap", "0")));
    const uint32_t delete_cap = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--delete_cap", "0")));
    // Opt-in only. Default false keeps RQ insert prep (bulk meta reserve).
    bool skip_meta_pre_reserve =
        parse_bool_arg(argc, argv, "--skip_meta_pre_reserve", false);
    const std::string gt_dir = get_arg(argc, argv, "--gt_dir", "");
    const uint32_t recall_search_L = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--recall_search_L", "100")));
    const bool run_maintenance =
        parse_bool_arg(argc, argv, "--run_maintenance", false);
    const bool flush_after_maintenance =
        parse_bool_arg(argc, argv, "--flush_after_maintenance", false);
    const std::string delete_algorithm =
        get_arg(argc, argv, "--delete_algorithm", "freshdiskann");
    if (delete_algorithm == "freshdiskann") {
        cfg.delete_algorithm = DeleteAlgorithm::FreshDiskANN;
    } else if (delete_algorithm == "freshdiskann-nosnapshot") {
        cfg.delete_algorithm = DeleteAlgorithm::FreshDiskANNNoSnapshot;
    } else if (delete_algorithm == "lazy") {
        cfg.delete_algorithm = DeleteAlgorithm::Lazy;
    } else {
        throw std::runtime_error("--delete_algorithm must be freshdiskann, freshdiskann-nosnapshot or lazy");
    }
    const bool flush_background =
        parse_bool_arg(argc, argv, "--flush_background", false);
    const double flush_dirty_high_ratio =
        std::stod(get_arg(argc, argv, "--flush_dirty_high_ratio", "0.20"));
    const double flush_dirty_low_ratio =
        std::stod(get_arg(argc, argv, "--flush_dirty_low_ratio", "0.05"));
    const uint32_t flush_budget_pages = static_cast<uint32_t>(
        std::stoul(get_arg(argc, argv, "--flush_budget_pages", "1024")));
    // Buffer-pool preload (workload-agnostic). New names take precedence over
    // the deprecated --update_preload* aliases when both are set.
    const uint32_t preload_max_pages = static_cast<uint32_t>(std::stoul(
        get_arg(argc, argv, "--preload_max_pages",
                get_arg(argc, argv, "--update_preload_max_pages", "0"))));
    const uint32_t preload_threads = static_cast<uint32_t>(std::stoul(
        get_arg(argc, argv, "--preload_threads",
                get_arg(argc, argv, "--update_preload_threads", "1"))));
    // Legacy boolean gate for the update path. If true, preload is invoked
    // before the update phase even when --preload_max_pages is 0 (= all pages).
    const bool update_preload =
        parse_bool_arg(argc, argv, "--update_preload", false);
    // BFS warmup: pin the first N pages reachable from the entry point. Goes
    // through the buffer pool's normal pin path so visited pages stay cached.
    // Mutually exclusive with the sequential preload (combining them would
    // just evict the BFS-warmed pages once the sequential prefill kicks in).
    const uint32_t bfs_warmup_pages = static_cast<uint32_t>(std::stoul(
        get_arg(argc, argv, "--bfs_warmup_pages", "0")));
    if (bfs_warmup_pages > 0 && (preload_max_pages > 0 || update_preload)) {
        throw std::runtime_error(
            "--bfs_warmup_pages is mutually exclusive with "
            "--preload_max_pages / --update_preload");
    }
    cfg.L = std::max(cfg.L, search_L);

    const std::string full_data_file =
        get_arg(argc, argv, "--full_data_file", "");
    std::vector<uint32_t> insert_ids;
    std::vector<uint32_t> delete_ids;
    if (workload != "search_only") {
        if (workload_has_inserts(workload)) {
            insert_ids = load_ids_file(get_arg(argc, argv, "--insert_ids_file", ""));
            if (cfg.max_dataset_size == 0 && !insert_ids.empty()) {
                const uint32_t max_insert_id =
                    *std::max_element(insert_ids.begin(), insert_ids.end());
                if (max_insert_id == std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error(
                        "insert id exceeds supported max_dataset_size");
                }
                const uint32_t active_cap =
                    bufann_snapshot_active_cap(index_prefix);
                const uint64_t by_count =
                    static_cast<uint64_t>(active_cap) + insert_ids.size() + 1;
                const uint64_t by_tag = static_cast<uint64_t>(max_insert_id) + 1;
                const uint64_t inferred = std::max(by_count, by_tag);
                if (inferred > std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error(
                        "inferred max_dataset_size exceeds uint32_t range");
                }
                cfg.max_dataset_size = static_cast<uint32_t>(inferred);
            }
        }
        if (workload_has_deletes(workload)) {
            delete_ids = load_ids_file(get_arg(argc, argv, "--delete_ids_file", ""));
        }
    }

    MetricConfig metric_config;
    metric_config.data_type = data_type;
    metric_config.pq_chunks = cfg.pq_chunks;
    metric_config.recall_at = recall_at;
    metric_config.search_L = search_L;
    metric_config.insert_search_L = insert_search_L;
    metric_config.graph_L = cfg.L;
    metric_config.delete_algorithm = delete_algorithm;
    metric_config.search_pq = cfg.search_pq;
    metric_config.beamwidth = cfg.beamwidth;
    metric_config.buffer_pool_frames = cfg.buffer_pool_frames;
    metric_config.meta_growth_copy_threads = cfg.meta_growth_copy_threads;
    metric_config.query_threads = query_threads;
    metric_config.insert_threads =
        workload == "search_after_update" ? pool_threads : insert_threads;
    metric_config.delete_threads =
        workload == "search_after_update" ? pool_threads : delete_threads;
    metric_config.maintenance_threads = maintenance_threads;
    metric_config.maintenance_query_threads = effective_maintenance_query_threads;
    metric_config.skip_update_search = skip_update_search;
    metric_config.enable_interval_metrics = enable_interval_metrics;
    metric_config.run_maintenance = run_maintenance;
    metric_config.flush_after_maintenance = flush_after_maintenance;
    metric_config.flush_background = flush_background;
    metric_config.flush_dirty_high_ratio = flush_dirty_high_ratio;
    metric_config.flush_dirty_low_ratio = flush_dirty_low_ratio;
    metric_config.flush_budget_pages = flush_budget_pages;


    std::cout << "Loading index from: " << index_prefix << std::endl;
    BufANNIndex<T> *idx = bufann_load<T>(index_prefix, cfg);
    std::cout << "Index loaded. aligned_dim=" << idx->aligned_dim << std::endl;
    release_tcmalloc_free_memory();

    T *queries_raw = nullptr;
    size_t nqueries = 0, qdim = 0, qaligned_dim = 0;
    diskann::load_aligned_bin<T>(query_file, queries_raw, nqueries, qdim,
                                 qaligned_dim);
    if (qdim != cfg.dim) {
        throw std::runtime_error("query dim mismatch: expected " +
                                 std::to_string(cfg.dim) + " got " +
                                 std::to_string(qdim));
    }
    StreamingQuerySource<T> streaming_queries;
    if (workload == "search_after_update" && !skip_update_search) {
        streaming_queries.load_from_aligned_query(
            queries_raw, nqueries, qdim, qaligned_dim, full_data_file,
            tail_query_begin,
            tail_query_end == 0 ? cfg.max_dataset_size : tail_query_end);
    }

    UpdateMetrics update;
    StatsSnapshot update_stats;
    StatsSnapshot per_round_recall_stats;
    double update_avg_rss_mb = 0.0;
    double update_peak_rss_mb = 0.0;
    if (workload != "search_only") {
        update.n_inserts = insert_ids.size();
        update.n_deletes = delete_ids.size();

        if (bfs_warmup_pages > 0) {
            std::cout << "BFS warmup: " << bfs_warmup_pages
                      << " pages from entry_point=" << bufann_store_snapshot(*idx)->entry_point()
                      << std::endl;
            bufann_store_snapshot(*idx)->warmup_bfs_pages(bufann_store_snapshot(*idx)->entry_point(),
                                        bfs_warmup_pages, preload_threads);
        } else if (update_preload || preload_max_pages > 0) {
            preload_buffer_pool_pages(*idx, preload_max_pages, preload_threads);
        }

        bufann_store_snapshot(*idx)->reset_stats();
        bufann_bench::RssSampler update_rss_sampler;
        auto update_begin = std::chrono::high_resolution_clock::now();

        auto update_stats_tracker =
            std::make_shared<UpdateStatsTracker<T>>(*idx);
        BufANNWindowMetrics interval_metrics;
        BufANNWindowMetrics *window_metrics_ptr =
            (!skip_update_search && enable_interval_metrics)
                ? &interval_metrics
                : nullptr;
        auto begin_update_round_rss = [&]() {
            if (window_metrics_ptr != nullptr) {
                window_metrics_ptr->resume_rss();
            } else {
                update_rss_sampler.resume();
            }
        };
        auto finish_update_round_rss = [&]() -> bufann_bench::RssStats {
            return window_metrics_ptr != nullptr
                ? window_metrics_ptr->pause_and_take_round_rss()
                : update_rss_sampler.stop_and_take_interval();
        };
        auto window_io_base = std::make_shared<StatsSnapshot>(
            update_stats_tracker->snapshot(*idx));
        if (window_metrics_ptr != nullptr) {

            window_metrics_ptr->set_io_emitter(
                [update_stats_tracker, window_io_base,
                 idx](std::ostream &out, bool &first,
                      double window_time_s, uint64_t total_ops) {
                    const StatsSnapshot current =
                        update_stats_tracker->snapshot(*idx);
                    const StatsSnapshot delta =
                        stats_delta(current, *window_io_base);
                    *window_io_base = current;
                    emit_bufann_logical_io(out, first, delta, window_time_s,
                                           total_ops);
                });

            window_metrics_ptr->start(ann_bench::latency_reserve_hint(
                static_cast<uint64_t>(nqueries)));
        }

        std::cout << "Running update workload: " << workload << std::endl;
        const bool use_flush_background = flush_background;
        const double flush_high_ratio = std::max(0.0, flush_dirty_high_ratio);
        const double flush_low_ratio =
            std::min(std::max(0.0, flush_dirty_low_ratio), flush_high_ratio);
        const uint32_t flush_budget =
            flush_budget_pages == 0 ? 1024 : flush_budget_pages;

        std::atomic<bool> flush_stop{false};
        std::atomic<bool> bg_failed{false};
        std::atomic<uint64_t> flush_ns{0};
        std::mutex bg_mu;
        std::condition_variable bg_cv;
        std::string bg_err;
        std::thread flush_thread;

        auto record_bg_error = [&](const std::string &prefix,
                                   const std::exception &e) {
            if (!bg_failed.exchange(true)) {
                std::lock_guard<std::mutex> lk(bg_mu);
                bg_err = prefix + e.what();
            }
            flush_stop.store(true, std::memory_order_release);
            bg_cv.notify_all();
        };

        if (use_flush_background) {
            flush_thread = std::thread([&]() {
                try {
                    while (!flush_stop.load(std::memory_order_acquire)) {
                        if (bufann_store_snapshot(*idx)->dirty_page_count() == 0 ||
                            bufann_store_snapshot(*idx)->dirty_ratio() < flush_high_ratio) {
                            std::unique_lock<std::mutex> lk(bg_mu);
                            bg_cv.wait(lk, [&]() {
                                return flush_stop.load(std::memory_order_acquire) ||
                                       (bufann_store_snapshot(*idx)->dirty_page_count() > 0 &&
                                        bufann_store_snapshot(*idx)->dirty_ratio() >= flush_high_ratio);
                            });
                            continue;
                        }
                        while (!flush_stop.load(std::memory_order_acquire) &&
                               bufann_store_snapshot(*idx)->dirty_ratio() > flush_low_ratio) {
                            auto fb = std::chrono::high_resolution_clock::now();
                            const uint32_t flushed =
                                bufann_flush_dirty_budget(*idx, flush_budget);
                            auto fe = std::chrono::high_resolution_clock::now();
                            flush_ns.fetch_add(elapsed_ns(fb, fe),
                                               std::memory_order_relaxed);
                            if (flushed == 0) break;
                        }
                    }
                    if (flush_after_maintenance) {
                        auto fb = std::chrono::high_resolution_clock::now();
                        bufann_flush_dirty(*idx);
                        auto fe = std::chrono::high_resolution_clock::now();
                        flush_ns.fetch_add(elapsed_ns(fb, fe),
                                           std::memory_order_relaxed);
                    }
                } catch (const std::exception &e) {
                    record_bg_error("background dirty flush failed: ", e);
                }
            });
        }

        auto notify_background_if_threshold = [&]() {
            bool should_notify = false;
            if (use_flush_background &&
                bufann_store_snapshot(*idx)->dirty_page_count() > 0 &&
                bufann_store_snapshot(*idx)->dirty_ratio() >= flush_high_ratio) {
                should_notify = true;
            }
            if (should_notify) {
                bg_cv.notify_all();
            }
        };

        std::vector<double> delete_op_latencies_us;
        std::vector<double> insert_op_latencies_us;
        double delete_s = 0.0;
        double insert_s = 0.0;
        double update_foreground_wall_s = 0.0;
        try {
            if (workload == "search_after_update") {
                const size_t ic = insert_cap > 0 ? insert_cap
                                  : std::max<size_t>(insert_ids.size(), 1);
                const size_t dc = delete_cap > 0 ? delete_cap
                                  : std::max<size_t>(delete_ids.size(), 1);
                const size_t n_ins_rounds =
                    insert_ids.empty() ? 0 : (insert_ids.size() + ic - 1) / ic;
                const size_t n_del_rounds =
                    delete_ids.empty() ? 0 : (delete_ids.size() + dc - 1) / dc;
                const size_t n_rounds = std::max(n_ins_rounds, n_del_rounds);

                auto gt_exists = [](const std::string &p) {
                    std::ifstream f(p);
                    return f.good();
                };
                for (size_t k = 0; k < n_rounds; ++k) {
                    const size_t ib = k * ic;
                    const size_t ie = std::min(insert_ids.size(), ib + ic);
                    const size_t db = k * dc;
                    const size_t de = std::min(delete_ids.size(), db + dc);
                    std::vector<uint32_t> insert_chunk;
                    std::vector<uint32_t> delete_chunk;
                    if (ib < insert_ids.size()) {
                        insert_chunk.assign(insert_ids.begin() + ib, insert_ids.begin() + ie);
                    }
                    if (db < delete_ids.size()) {
                        delete_chunk.assign(delete_ids.begin() + db, delete_ids.begin() + de);
                    }
                    std::cout << "ROUND " << k << ": mixed insert "
                              << insert_chunk.size() << ", delete "
                              << delete_chunk.size() << ", query_ratio "
                              << effective_query_ratio << std::endl;
                    begin_update_round_rss();
                    const StatsSnapshot round_io_base =
                        update_stats_tracker->snapshot(*idx);

                    ConcurrentUpdateResult cu = run_concurrent_update<T>(
                        *idx, full_data_file, insert_chunk, insert_search_L,
                        delete_chunk,
                        skip_update_search ? nullptr : &streaming_queries,
                        effective_query_ratio, recall_at, search_L,
                        pool_threads, window_metrics_ptr, k,
                        skip_meta_pre_reserve);
                    insert_s += cu.insert_busy_s;
                    delete_s += cu.delete_busy_s;
                    update_foreground_wall_s += cu.foreground_wall_s;
                    if (window_metrics_ptr == nullptr) {
                        insert_op_latencies_us.insert(
                            insert_op_latencies_us.end(),
                            cu.insert_op_latencies_us.begin(),
                            cu.insert_op_latencies_us.end());
                        delete_op_latencies_us.insert(
                            delete_op_latencies_us.end(),
                            cu.delete_op_latencies_us.begin(),
                            cu.delete_op_latencies_us.end());
                    }
                    const auto fg_done = std::chrono::high_resolution_clock::now();
                    if (window_metrics_ptr != nullptr) {
                        window_metrics_ptr->flush(k, "foreground");
                        emit_bufann_phase_checkpoint(
                            std::cout, "foreground", k,
                            elapsed_s(update_begin, fg_done),
                            static_cast<uint64_t>(insert_chunk.size()),
                            static_cast<uint64_t>(delete_chunk.size()),
                            static_cast<uint64_t>(cu.query.nqueries),
                            cu.foreground_wall_s);
                    }
                    bg_cv.notify_all();

                    double r_maint_s = 0.0;
                    double r_post_merge_query_s = 0.0;
                    SearchMetrics maint_q;
                    SearchMetrics post_maint_q;

                    // Cleanup runs concurrently with merge queries.
                    // queries. MERGE_QUERY_BUDGET=0 => queries run until
                    // cleanup finishes, then stop (no post_merge fill).
                    if (run_maintenance && !delete_chunk.empty()) {
                        const uint64_t merge_query_budget =
                            ann_bench::env_u64_or("MERGE_QUERY_BUDGET", 0);
                        std::atomic<bool> maint_done{false};
                        auto mb = std::chrono::high_resolution_clock::now();
                        std::thread maint_thread([&]() {
                            bufann_cleanup_deleted_edges(*idx, maintenance_threads);
                            maint_done.store(true, std::memory_order_release);
                        });
                        // A lazy deletion episode may rely on these queries to finish, so they run until cleanup
                        // returns, and the budget only limits the queries after it.
                        const bool lazy = cfg.delete_algorithm == DeleteAlgorithm::Lazy;
                        if (!skip_update_search &&
                            effective_maintenance_query_threads > 0) {
                            maint_q = run_queries_until_done<T>(
                                *idx, &streaming_queries, maint_done,
                                effective_maintenance_query_threads,
                                recall_at, search_L, cu.query.nqueries,
                                window_metrics_ptr, k, lazy ? 0 : merge_query_budget,
                                "merge");
                        }
                        maint_thread.join();
                        auto me = std::chrono::high_resolution_clock::now();
                        r_maint_s = elapsed_s(mb, me);
                        update.maintenance_wall_time_s += r_maint_s;
                        if (window_metrics_ptr != nullptr) {
                            window_metrics_ptr->flush(k, "merge");
                            emit_bufann_phase_checkpoint(
                                std::cout, "merge", k,
                                elapsed_s(update_begin, me),
                                0, 0, static_cast<uint64_t>(maint_q.nqueries),
                                r_maint_s);
                        }
                        // POST_MERGE_QUERY_S > 0 runs the queries after the merge for that many seconds instead of
                        // up to the budget; with MERGE_QUERY_BUDGET=0 the merge's queries are then uncapped too.
                        const uint64_t post_merge_query_s = ann_bench::env_u64_or("POST_MERGE_QUERY_S", 0);
                        const uint32_t tail_threads = pool_threads == 0 ? 1 : pool_threads;
                        bool ran_tail = false;
                        if (!skip_update_search && post_merge_query_s > 0) {
                            ran_tail = true;
                            std::atomic<bool> tail_done{false};
                            std::thread timer([&]() {
                                std::this_thread::sleep_for(std::chrono::seconds(post_merge_query_s));
                                tail_done.store(true, std::memory_order_release);
                            });
                            post_maint_q = run_queries_until_done<T>(
                                *idx, &streaming_queries, tail_done, tail_threads, recall_at, search_L,
                                cu.query.nqueries + maint_q.nqueries, window_metrics_ptr, k, 0,
                                "post_merge_query");
                            timer.join();
                        } else if (!skip_update_search && merge_query_budget > maint_q.nqueries) {
                            ran_tail = true;
                            const uint64_t remaining =
                                merge_query_budget - maint_q.nqueries;
                            post_maint_q = run_queries_fixed_count<T>(
                                *idx, &streaming_queries, remaining,
                                tail_threads,
                                recall_at, search_L,
                                cu.query.nqueries + maint_q.nqueries,
                                window_metrics_ptr, k, "post_merge_query");
                        }
                        if (ran_tail) {
                            r_post_merge_query_s =
                                post_maint_q.search_qps > 0.0
                                    ? static_cast<double>(post_maint_q.nqueries) /
                                          post_maint_q.search_qps
                                    : 0.0;
                            update.post_merge_query_time_s += r_post_merge_query_s;
                            if (window_metrics_ptr != nullptr) {
                                window_metrics_ptr->flush(k, "post_merge_query");
                                emit_bufann_phase_checkpoint(
                                    std::cout, "post_merge_query", k,
                                    elapsed_s(update_begin,
                                              std::chrono::high_resolution_clock::now()),
                                    0, 0,
                                    static_cast<uint64_t>(post_maint_q.nqueries),
                                    r_post_merge_query_s);
                            }
                        }
                    }

                    const bufann_bench::RssStats round_rss =
                        finish_update_round_rss();
                    const double r_time =
                        cu.foreground_wall_s + r_maint_s + r_post_merge_query_s;
                    const uint64_t r_query_count =
                        static_cast<uint64_t>(cu.query.nqueries + maint_q.nqueries +
                                              post_maint_q.nqueries);
                    if (r_query_count > 0) {
                        update.query_count += r_query_count;
                        update.query_time_s += r_time;
                    }
                    auto fold_query = [&](const SearchMetrics &qm) {
                        if (qm.nqueries == 0) return;
                        update.query_lat_avg_us += qm.lat_avg_us * static_cast<double>(qm.nqueries);
                        update.query_lat_p50_us = qm.lat_p50_us;
                        update.query_lat_p90_us = qm.lat_p90_us;
                        update.query_lat_p95_us = qm.lat_p95_us;
                        update.query_lat_p99_us = qm.lat_p99_us;
                        update.query_lat_p999_us = qm.lat_p999_us;
                    };
                    fold_query(cu.query);
                    fold_query(maint_q);
                    fold_query(post_maint_q);

                    const StatsSnapshot round_stats = stats_delta(
                        update_stats_tracker->snapshot(*idx), round_io_base);

                    const double round_elapsed_before_recall =
                        elapsed_s(update_begin,
                                  std::chrono::high_resolution_clock::now());
                    double r_recall = -1.0;
                    if (!gt_dir.empty()) {
                        const std::string gtp = gt_dir + "/round_" +
                            std::to_string(k) + "_gt" +
                            std::to_string(recall_at) + ".bin";
                        if (gt_exists(gtp)) {
                            r_recall = measure_round_recall<T>(
                                *idx, queries_raw, nqueries, qaligned_dim, gtp,
                                recall_at, recall_search_L, query_threads,
                                update_rss_sampler, window_metrics_ptr,
                                per_round_recall_stats, false);
                        }
                    }

                    bool rf = true;
                    std::cout << std::fixed << std::setprecision(6) << "{";
                    bufann_bench::json_kv(std::cout, rf, "baseline", "BufANN");
                    bufann_bench::json_kv(std::cout, rf, "workload", "mixed_update");
                    bufann_bench::json_kv(std::cout, rf, "phase", "round");
                    bufann_bench::json_kv(std::cout, rf, "round_index",
                                          static_cast<uint64_t>(k));
                    bufann_bench::json_kv(std::cout, rf, "elapsed_s",
                                          round_elapsed_before_recall);
                    bufann_bench::json_kv(std::cout, rf, "insert_count",
                                          static_cast<uint64_t>(insert_chunk.size()));
                    bufann_bench::json_kv(std::cout, rf, "delete_count",
                                          static_cast<uint64_t>(delete_chunk.size()));
                    bufann_bench::json_kv(std::cout, rf, "query_count", r_query_count);
                    bufann_bench::json_kv(std::cout, rf, "foreground_query_count",
                                          static_cast<uint64_t>(cu.query.nqueries));
                    bufann_bench::json_kv(std::cout, rf, "merge_query_count",
                                          static_cast<uint64_t>(maint_q.nqueries));
                    bufann_bench::json_kv(std::cout, rf, "post_merge_query_count",
                                          static_cast<uint64_t>(post_maint_q.nqueries));
                    bufann_bench::json_kv(std::cout, rf, "foreground_time_s",
                                          cu.foreground_wall_s);
                    bufann_bench::json_kv(std::cout, rf, "maintenance_time_s", r_maint_s);
                    bufann_bench::json_kv(std::cout, rf, "post_merge_query_time_s",
                                          r_post_merge_query_s);
                    bufann_bench::json_kv(std::cout, rf, "workload_time_s", r_time);
                    bufann_bench::emit_rss_stats(std::cout, rf, round_rss);
                    bufann_bench::json_kv(std::cout, rf, "query_qps",
                                          bufann_bench::safe_div(
                                              static_cast<double>(r_query_count),
                                              r_time));
                    bufann_bench::json_kv(std::cout, rf, "insert_ops_per_s",
                                          bufann_bench::safe_div(
                                              static_cast<double>(insert_chunk.size()),
                                              r_time));
                    bufann_bench::json_kv(std::cout, rf, "delete_ops_per_s",
                                          bufann_bench::safe_div(
                                              static_cast<double>(delete_chunk.size()),
                                              r_time));
                    bufann_bench::json_kv(std::cout, rf, "update_ops_per_s",
                                          bufann_bench::safe_div(
                                              static_cast<double>(insert_chunk.size() +
                                                                  delete_chunk.size()),
                                              r_time));
                    emit_bufann_storage_stats(std::cout, rf, round_stats);
                    emit_bufann_logical_io(
                        std::cout, rf, round_stats, r_time,
                        static_cast<uint64_t>(insert_chunk.size() +
                                              delete_chunk.size() +
                                              r_query_count));
                    if (r_recall >= 0.0) {
                        bufann_bench::json_kv(std::cout, rf, "recall", r_recall);
                        bufann_bench::json_kv(std::cout, rf, "recall_at",
                                              static_cast<uint64_t>(recall_at));
                        bufann_bench::json_kv(std::cout, rf, "recall_search_L",
                                              static_cast<uint64_t>(recall_search_L));
                    }

                    std::cout << "}" << std::endl;
                }

                if (window_metrics_ptr != nullptr && n_rounds > 0) {
                    window_metrics_ptr->finalize(n_rounds - 1, "workload");
                }
                if (update.query_count > 0) {
                    update.query_lat_avg_us /=
                        static_cast<double>(update.query_count);
                }
            } else {
                // insert_only / delete_only: per-round structure mirroring the
                // DiskANN family. Each round applies its contiguous id chunk
                // (insert_cap / delete_cap), runs bufann_cleanup_deleted_edges
                // as that round's maintenance (when it deleted), then measures
                // recall against that round's GT and emits a phase:"round"
                // record. cap==0 collapses to a single round over all ids.
                // Exactly one of insert_ids / delete_ids is populated here.
                const size_t ic = insert_cap > 0 ? insert_cap
                                  : std::max<size_t>(insert_ids.size(), 1);
                const size_t dc = delete_cap > 0 ? delete_cap
                                  : std::max<size_t>(delete_ids.size(), 1);
                const size_t n_ins_rounds =
                    insert_ids.empty() ? 0 : (insert_ids.size() + ic - 1) / ic;
                const size_t n_del_rounds =
                    delete_ids.empty() ? 0 : (delete_ids.size() + dc - 1) / dc;
                const size_t n_rounds = std::max(n_ins_rounds, n_del_rounds);
                auto gt_exists = [](const std::string &p) {
                    std::ifstream f(p);
                    return f.good();
                };

                for (size_t k = 0; k < n_rounds; ++k) {
                    std::vector<double> r_ins_lat, r_del_lat;
                    double r_insert_s = 0.0, r_delete_s = 0.0, r_maint_s = 0.0;
                    bool r_has_delete = false, r_has_insert = false;
                    begin_update_round_rss();
                    const StatsSnapshot round_io_base =
                        update_stats_tracker->snapshot(*idx);

                    if (!delete_ids.empty() && k * dc < delete_ids.size()) {
                        const size_t b = k * dc;
                        const size_t e = std::min(delete_ids.size(), b + dc);
                        std::vector<uint32_t> chunk(delete_ids.begin() + b,
                                                    delete_ids.begin() + e);
                        std::cout << "ROUND " << k << ": delete " << chunk.size()
                                  << " points..." << std::endl;
                        r_delete_s = run_deletes(*idx, chunk, delete_threads,
                                                 r_del_lat,
                                                 notify_background_if_threshold);
                        delete_s += r_delete_s;
                        update_foreground_wall_s += r_delete_s;
                        r_has_delete = true;
                        bg_cv.notify_all();
                    }
                    if (!insert_ids.empty() && k * ic < insert_ids.size()) {
                        const size_t b = k * ic;
                        const size_t e = std::min(insert_ids.size(), b + ic);
                        std::vector<uint32_t> chunk(insert_ids.begin() + b,
                                                    insert_ids.begin() + e);
                        std::cout << "ROUND " << k << ": insert " << chunk.size()
                                  << " points..." << std::endl;
                        InsertRunMetrics im = run_inserts(
                            *idx, full_data_file, chunk, insert_search_L,
                            insert_threads, r_ins_lat,
                            notify_background_if_threshold);
                        r_insert_s = im.elapsed_s;
                        insert_s += r_insert_s;
                        update_foreground_wall_s += r_insert_s;
                        update.maintenance_wall_time_s += im.maintenance_s;
                        r_has_insert = true;
                        bg_cv.notify_all();
                    }

                    // Per-round maintenance: clean up edges to deleted nodes.
                    if (run_maintenance && r_has_delete) {
                        auto mb = std::chrono::high_resolution_clock::now();
                        bufann_cleanup_deleted_edges(*idx, maintenance_threads);
                        auto me = std::chrono::high_resolution_clock::now();
                        r_maint_s = elapsed_s(mb, me);
                        update.maintenance_wall_time_s += r_maint_s;
                    }

                    const bufann_bench::RssStats round_rss =
                        finish_update_round_rss();
                    // Fold round op-latencies into the workload aggregate.
                    insert_op_latencies_us.insert(insert_op_latencies_us.end(),
                                                  r_ins_lat.begin(), r_ins_lat.end());
                    delete_op_latencies_us.insert(delete_op_latencies_us.end(),
                                                  r_del_lat.begin(), r_del_lat.end());

                    const StatsSnapshot round_stats = stats_delta(
                        update_stats_tracker->snapshot(*idx), round_io_base);

                    const double round_elapsed_before_recall =
                        elapsed_s(update_begin,
                                  std::chrono::high_resolution_clock::now());
                    // Per-round recall against this round's GT (no reload; the
                    // live index is searched). run_search loads the GT, scores,
                    // and returns recall.
                    double r_recall = -1.0;
                    if (!gt_dir.empty()) {
                        const std::string gtp = gt_dir + "/round_" +
                            std::to_string(k) + "_gt" +
                            std::to_string(recall_at) + ".bin";
                        if (gt_exists(gtp)) {
                            r_recall = measure_round_recall<T>(
                                *idx, queries_raw, nqueries, qaligned_dim, gtp,
                                recall_at, recall_search_L, query_threads,
                                update_rss_sampler, window_metrics_ptr,
                                per_round_recall_stats, false);
                        } else {
                            std::cerr << "WARNING: missing per-round GT " << gtp
                                      << "; skipping recall for round " << k
                                      << std::endl;
                        }
                    }

                    // Emit the per-round record.
                    const uint64_t r_count =
                        r_has_insert ? static_cast<uint64_t>(r_ins_lat.size())
                                     : static_cast<uint64_t>(r_del_lat.size());
                    const double r_time = (r_has_insert ? r_insert_s : r_delete_s) + r_maint_s;
                    const double r_ops = bufann_bench::safe_div(
                        static_cast<double>(r_count), r_time);
                    OpLatencyStats r_lat = compute_op_latency_stats(
                        r_has_insert ? r_ins_lat : r_del_lat);
                    bool rf = true;
                    std::cout << std::fixed << std::setprecision(6) << "{";
                    bufann_bench::json_kv(std::cout, rf, "baseline", "BufANN");
                    bufann_bench::json_kv(std::cout, rf, "workload",
                                          r_has_insert ? "insert" : "delete");
                    bufann_bench::json_kv(std::cout, rf, "phase", "round");
                    bufann_bench::json_kv(std::cout, rf, "round_index",
                                          static_cast<uint64_t>(k));
                    bufann_bench::json_kv(std::cout, rf, "elapsed_s",
                                          round_elapsed_before_recall);
                    if (r_has_insert) {
                        bufann_bench::json_kv(std::cout, rf, "insert_count", r_count);
                        bufann_bench::json_kv(std::cout, rf, "delete_count", static_cast<uint64_t>(0));
                        bufann_bench::json_kv(std::cout, rf, "insert_ops_per_s", r_ops);
                        bufann_bench::json_kv(std::cout, rf, "delete_ops_per_s", 0.0);
                    } else {
                        bufann_bench::json_kv(std::cout, rf, "insert_count", static_cast<uint64_t>(0));
                        bufann_bench::json_kv(std::cout, rf, "delete_count", r_count);
                        bufann_bench::json_kv(std::cout, rf, "insert_ops_per_s", 0.0);
                        bufann_bench::json_kv(std::cout, rf, "delete_ops_per_s", r_ops);
                    }
                    const std::string pfx = r_has_insert ? "insert_call" : "delete_call";
                    bufann_bench::json_kv(std::cout, rf, "update_ops_per_s", r_ops);
                    bufann_bench::json_kv(std::cout, rf, pfx + "_lat_avg_us", r_lat.avg_us);
                    bufann_bench::json_kv(std::cout, rf, pfx + "_lat_p50_us", r_lat.p50_us);
                    bufann_bench::json_kv(std::cout, rf, pfx + "_lat_p90_us", r_lat.p90_us);
                    bufann_bench::json_kv(std::cout, rf, pfx + "_lat_p95_us", r_lat.p95_us);
                    bufann_bench::json_kv(std::cout, rf, pfx + "_lat_p99_us", r_lat.p99_us);
                    bufann_bench::json_kv(std::cout, rf, "foreground_time_s",
                                          r_has_insert ? r_insert_s : r_delete_s);
                    bufann_bench::json_kv(std::cout, rf, "maintenance_time_s", r_maint_s);
                    bufann_bench::json_kv(std::cout, rf, "workload_time_s", r_time);
                    bufann_bench::emit_rss_stats(std::cout, rf, round_rss);
                    emit_bufann_storage_stats(std::cout, rf, round_stats);
                    emit_bufann_logical_io(std::cout, rf, round_stats, r_time,
                                           r_count);
                    if (r_recall >= 0.0) {
                        bufann_bench::json_kv(std::cout, rf, "recall", r_recall);
                        bufann_bench::json_kv(std::cout, rf, "recall_at",
                                              static_cast<uint64_t>(recall_at));
                        bufann_bench::json_kv(std::cout, rf, "recall_search_L",
                                              static_cast<uint64_t>(recall_search_L));
                    }
                    std::cout << "}" << std::endl;
                }
                if (window_metrics_ptr != nullptr && n_rounds > 0) {
                    window_metrics_ptr->finalize(n_rounds - 1, "workload");
                }
            }
        } catch (...) {
            flush_stop.store(true, std::memory_order_release);
            bg_cv.notify_all();
            if (flush_thread.joinable()) flush_thread.join();
            throw;
        }
        update.foreground_update_wall_time_s = update_foreground_wall_s;
        if (!delete_op_latencies_us.empty()) {
            OpLatencyStats s = compute_op_latency_stats(delete_op_latencies_us);
            update.delete_lat_avg_us = s.avg_us;
            update.delete_lat_p50_us = s.p50_us;
            update.delete_lat_p90_us = s.p90_us;
            update.delete_lat_p95_us = s.p95_us;
            update.delete_lat_p99_us = s.p99_us;
        }
        if (!insert_op_latencies_us.empty()) {
            OpLatencyStats s = compute_op_latency_stats(insert_op_latencies_us);
            update.insert_lat_avg_us = s.avg_us;
            update.insert_lat_p50_us = s.p50_us;
            update.insert_lat_p90_us = s.p90_us;
            update.insert_lat_p95_us = s.p95_us;
            update.insert_lat_p99_us = s.p99_us;
        }
        if (window_metrics_ptr != nullptr) {
            const BufANNWindowMetrics::LatencyWindow q =
                window_metrics_ptr->query_latency_summary();
            const BufANNWindowMetrics::LatencyWindow ins =
                window_metrics_ptr->insert_latency_summary();
            const BufANNWindowMetrics::LatencyWindow del =
                window_metrics_ptr->delete_latency_summary();
            update.query_lat_avg_us = q.avg_us;
            update.query_lat_p50_us = q.p50_us;
            update.query_lat_p90_us = q.p90_us;
            update.query_lat_p95_us = q.p95_us;
            update.query_lat_p99_us = q.p99_us;
            update.query_lat_p999_us = 0.0;
            update.insert_lat_avg_us = ins.avg_us;
            update.insert_lat_p50_us = ins.p50_us;
            update.insert_lat_p90_us = ins.p90_us;
            update.insert_lat_p95_us = ins.p95_us;
            update.insert_lat_p99_us = ins.p99_us;
            update.delete_lat_avg_us = del.avg_us;
            update.delete_lat_p50_us = del.p50_us;
            update.delete_lat_p90_us = del.p90_us;
            update.delete_lat_p95_us = del.p95_us;
            update.delete_lat_p99_us = del.p99_us;
        }

        if (bg_failed.load(std::memory_order_acquire)) {
            flush_stop.store(true, std::memory_order_release);
            bg_cv.notify_all();
            if (flush_thread.joinable()) flush_thread.join();
            throw std::runtime_error(bg_err);
        }

        diskann::inplace::dump_and_reset_insert_phase_stats();

        flush_stop.store(true, std::memory_order_release);
        bg_cv.notify_all();
        if (flush_thread.joinable()) flush_thread.join();
        if (bg_failed.load(std::memory_order_acquire)) {
            throw std::runtime_error(bg_err);
        }
        update.flush_wall_time_s +=
            static_cast<double>(flush_ns.load(std::memory_order_relaxed)) / 1e9;

        if (flush_after_maintenance && !use_flush_background) {
            auto flush_begin = std::chrono::high_resolution_clock::now();
            bufann_flush_dirty(*idx);
            auto flush_end = std::chrono::high_resolution_clock::now();
            update.flush_wall_time_s += elapsed_s(flush_begin, flush_end);
        }

        update.total_update_wall_time_s =
            update.foreground_update_wall_time_s +
            update.maintenance_wall_time_s +
            update.post_merge_query_time_s;
        update.elapsed_s = update.total_update_wall_time_s;
        if (window_metrics_ptr != nullptr) {
            window_metrics_ptr->pause_rss();
            update_avg_rss_mb = window_metrics_ptr->avg_rss_mb();
            update_peak_rss_mb = window_metrics_ptr->peak_rss_mb();
        } else {
            update_rss_sampler.stop();
            update_avg_rss_mb = update_rss_sampler.avg_mb();
            update_peak_rss_mb = update_rss_sampler.peak_mb();
        }
        update.insert_time_s = insert_s;
        update.delete_time_s = delete_s;
        const size_t n_updates = update.n_inserts + update.n_deletes;
        const double workload_time_s =
            update.total_update_wall_time_s > 0.0
                ? update.total_update_wall_time_s
                : update.foreground_update_wall_time_s;
        if (update.query_count > 0 && update.query_time_s > 0.0) {
            update.query_qps =
                static_cast<double>(update.query_count) / update.query_time_s;
        }
        update.insert_ops_per_s =
            workload_time_s > 0.0 ? static_cast<double>(update.n_inserts) / workload_time_s : 0.0;
        update.delete_ops_per_s =
            workload_time_s > 0.0 ? static_cast<double>(update.n_deletes) / workload_time_s : 0.0;
        update.update_ops_per_s = workload_time_s > 0.0
                                ? static_cast<double>(n_updates) /
                                      workload_time_s
                                : 0.0;
        update.overall_ops_per_s =
            workload_time_s > 0.0
                ? static_cast<double>(n_updates + update.query_count) / workload_time_s
                : 0.0;
        update_stats = update_stats_tracker->snapshot(*idx);
        if (workload == "insert_only" || workload == "delete_only" ||
            workload == "search_after_update") {
            // Per-round recall searches are validation work, not update-phase IO.
            update_stats = stats_delta(update_stats, per_round_recall_stats);
        }
    }

    // search_only sweeps the full --search_L list in one invocation, emitting
    // one measured run + JSON record per L. Each L re-warms the buffer pool
    // (preload + warmup queries at that L) so every measurement starts from the
    // same warm state, rather than inheriting the previous L's timed query pass
    // (which would leave the exact query working set cached and make later Ls
    // finish in a split second). Update workloads fall through to the single-L
    // path below; their preload already fired before the update phase.
    if (workload == "search_only") {
        // Load the optional warmup query file once; reused across all Ls.
        T *warmup_raw = nullptr;
        size_t wnq = 0, wdim = 0, waligned_dim = 0;
        if (!warmup_query_file.empty()) {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup_raw, wnq, wdim,
                                         waligned_dim);
            if (wdim != cfg.dim) {
                diskann::aligned_free(warmup_raw);
                throw std::runtime_error("warmup query dim mismatch: expected " +
                                         std::to_string(cfg.dim) + " got " +
                                         std::to_string(wdim));
            }
        }

        std::ofstream rf;
        if (!result_file.empty()) {
            rf.open(result_file, std::ios::out | std::ios::trunc);
            if (!rf) {
                std::cerr << "WARNING: cannot write result file: " << result_file
                          << std::endl;
            }
        }

        for (size_t li = 0; li < search_L_list.size(); ++li) {
            const uint32_t cur_L = search_L_list[li];
            std::cout << "\n=== search_L sweep " << (li + 1) << "/"
                      << search_L_list.size() << ": L=" << cur_L << " ===\n";

            // Re-warm for this L (see block comment above).
            if (bfs_warmup_pages > 0) {
                std::cout << "BFS warmup: " << bfs_warmup_pages
                          << " pages from entry_point=" << bufann_store_snapshot(*idx)->entry_point()
                          << std::endl;
                bufann_store_snapshot(*idx)->warmup_bfs_pages(bufann_store_snapshot(*idx)->entry_point(),
                                            bfs_warmup_pages, preload_threads);
            } else if (preload_max_pages > 0) {
                preload_buffer_pool_pages(*idx, preload_max_pages, preload_threads);
            }
            if (warmup_raw != nullptr) {
                std::cout << "Warmup: running " << wnq << " queries from "
                          << warmup_query_file << "  threads=" << warmup_threads
                          << "  search_L=" << cur_L << std::endl;
#pragma omp parallel for num_threads(static_cast<int>(warmup_threads)) schedule(dynamic, 1)
                for (int64_t i = 0; i < static_cast<int64_t>(wnq); ++i) {
                    bufann_query_into(*idx,
                                       warmup_raw + static_cast<size_t>(i) * waligned_dim,
                                       recall_at, nullptr, cur_L);
                }
            } else if (warmup_queries > 0) {
                const uint32_t wq =
                    std::min(warmup_queries, static_cast<uint32_t>(nqueries));
                std::cout << "Warmup: running " << wq << " queries"
                          << "  threads=" << warmup_threads
                          << "  search_L=" << cur_L << std::endl;
#pragma omp parallel for num_threads(static_cast<int>(warmup_threads)) schedule(dynamic, 1)
                for (int64_t i = 0; i < static_cast<int64_t>(wq); ++i) {
                    bufann_query_into(*idx,
                                       queries_raw + static_cast<size_t>(i) * qaligned_dim,
                                       recall_at, nullptr, cur_L);
                }
            }

            bufann_store_snapshot(*idx)->reset_stats();
            std::cout << "Running search: " << nqueries
                      << " queries  search_L=" << cur_L
                      << "  recall_at=" << recall_at
                      << "  threads=" << query_threads << std::endl;
            bufann_bench::RssSampler search_rss_sampler;
            search_rss_sampler.start();
            SearchMetrics search =
                run_search(*idx, queries_raw, nqueries, qaligned_dim, gt_file,
                           recall_at, cur_L, query_threads, 0, 0, insert_ids);
            search_rss_sampler.stop();
            const double search_avg_rss_mb = search_rss_sampler.avg_mb();
            const double search_peak_rss_mb = search_rss_sampler.peak_mb();
            StatsSnapshot search_stats =
                snapshot_stats(bufann_store_snapshot(*idx)->stats(), bufann_store_snapshot(*idx)->page_size());
            if (search.nqueries > 0) {
                search.mean_ios =
                    static_cast<double>(search_stats.page_fault_total) /
                    static_cast<double>(search.nqueries);
            }

            std::cout << std::fixed << std::setprecision(4)
                      << "  search_qps                    = " << search.search_qps << "\n"
                      << "  lat_p50_ms                    = " << search.lat_p50_us / 1000.0 << "\n"
                      << "  lat_p90_ms                    = " << search.lat_p90_us / 1000.0 << "\n"
                      << "  lat_p95_ms                    = " << search.lat_p95_us / 1000.0 << "\n"
                      << "  lat_p99_ms                    = " << search.lat_p99_us / 1000.0 << "\n"
                      << "  read_ios_per_op               = " << search.mean_ios << "\n"
                      << "  final_recall@" << recall_at << "               = "
                      << (search.recall >= 0.0 ? std::to_string(search.recall)
                                                : "N/A (no GT)")
                      << std::endl;

            metric_config.search_L = cur_L;
            emit_canonical_search_json(std::cout, "query", search, search_stats,
                                       metric_config, search_avg_rss_mb,
                                       search_peak_rss_mb);
            if (rf) {
                emit_canonical_search_json(rf, "query", search, search_stats,
                                           metric_config, search_avg_rss_mb,
                                           search_peak_rss_mb);
            }
        }

        if (warmup_raw != nullptr) diskann::aligned_free(warmup_raw);
        diskann::aligned_free(queries_raw);
        bufann_free(idx);
        return 0;
    }

    if (!warmup_query_file.empty()) {
        T *warmup_raw = nullptr;
        size_t wnq = 0, wdim = 0, waligned_dim = 0;
        diskann::load_aligned_bin<T>(warmup_query_file, warmup_raw, wnq, wdim,
                                     waligned_dim);
        if (wdim != cfg.dim) {
            diskann::aligned_free(warmup_raw);
            throw std::runtime_error("warmup query dim mismatch: expected " +
                                     std::to_string(cfg.dim) + " got " +
                                     std::to_string(wdim));
        }
        std::cout << "Warmup: running " << wnq << " queries from "
                  << warmup_query_file
                  << "  threads=" << warmup_threads << std::endl;
#pragma omp parallel for num_threads(static_cast<int>(warmup_threads)) schedule(dynamic, 1)
        for (int64_t i = 0; i < static_cast<int64_t>(wnq); ++i) {
            bufann_query_into(*idx,
                               warmup_raw + static_cast<size_t>(i) * waligned_dim,
                               recall_at, nullptr, search_L);
        }
        diskann::aligned_free(warmup_raw);
    } else if (warmup_queries > 0) {
        const uint32_t wq =
            std::min(warmup_queries, static_cast<uint32_t>(nqueries));
        std::cout << "Warmup: running " << wq << " queries"
                  << "  threads=" << warmup_threads << std::endl;
#pragma omp parallel for num_threads(static_cast<int>(warmup_threads)) schedule(dynamic, 1)
        for (int64_t i = 0; i < static_cast<int64_t>(wq); ++i) {
            bufann_query_into(*idx,
                               queries_raw + static_cast<size_t>(i) * qaligned_dim,
                               recall_at, nullptr, search_L);
        }
    }

    bufann_store_snapshot(*idx)->reset_stats();
    std::cout << "Running final search: " << nqueries
              << " queries  search_L=" << search_L
              << "  recall_at=" << recall_at
              << "  threads=" << query_threads << std::endl;
    bufann_bench::RssSampler search_rss_sampler;
    search_rss_sampler.start();
    SearchMetrics search =
        run_search(*idx, queries_raw, nqueries, qaligned_dim, gt_file, recall_at,
                   search_L, query_threads, 0, 0, insert_ids);
    search_rss_sampler.stop();
    const double search_avg_rss_mb = search_rss_sampler.avg_mb();
    const double search_peak_rss_mb = search_rss_sampler.peak_mb();
    StatsSnapshot search_stats =
        snapshot_stats(bufann_store_snapshot(*idx)->stats(), bufann_store_snapshot(*idx)->page_size());
    if (search.nqueries > 0) {
        search.mean_ios = static_cast<double>(search_stats.page_fault_total) /
                          static_cast<double>(search.nqueries);
    }

    std::cout << "\n=== Results ===\n";
    std::cout << std::fixed << std::setprecision(4);
    if (workload != "search_only") {
        std::cout << "  foreground_update_wall_time_s = "
                  << update.foreground_update_wall_time_s << "\n";
        std::cout << "  maintenance_wall_time_s       = "
                  << update.maintenance_wall_time_s << "\n";
        std::cout << "  flush_wall_time_s             = "
                  << update.flush_wall_time_s << "\n";
        std::cout << "  total_update_wall_time_s      = "
                  << update.total_update_wall_time_s << "\n";
        if (workload_has_inserts(workload)) {
            std::cout << "  insert_ops_per_s              = "
                      << update.insert_ops_per_s << "  (n=" << update.n_inserts << ")\n";
            std::cout << "  insert_call_lat_avg_us        = "
                      << update.insert_lat_avg_us << "\n";
            std::cout << "  insert_call_lat_p50_us        = "
                      << update.insert_lat_p50_us << "\n";
            std::cout << "  insert_call_lat_p90_us        = "
                      << update.insert_lat_p90_us << "\n";
            std::cout << "  insert_call_lat_p99_us        = "
                      << update.insert_lat_p99_us << "\n";
        }
        if (workload_has_deletes(workload)) {
            std::cout << "  delete_ops_per_s              = "
                      << update.delete_ops_per_s << "  (n=" << update.n_deletes << ")\n";
            std::cout << "  delete_call_lat_avg_us        = "
                      << update.delete_lat_avg_us << "\n";
            std::cout << "  delete_call_lat_p50_us        = "
                      << update.delete_lat_p50_us << "\n";
            std::cout << "  delete_call_lat_p90_us        = "
                      << update.delete_lat_p90_us << "\n";
            std::cout << "  delete_call_lat_p99_us        = "
                      << update.delete_lat_p99_us << "\n";
        }
        std::cout << "  update_ops_per_s              = "
                  << update.update_ops_per_s << "  (n="
                  << (update.n_inserts + update.n_deletes) << ")\n";
        if (update.query_count > 0) {
            std::cout << "  query_count                   = " << update.query_count << "\n";
            std::cout << "  query_time_s                  = " << update.query_time_s << "\n";
            std::cout << "  query_qps                     = " << update.query_qps << "\n";
        }
    }
    if (workload == "search_only") {
        std::cout << "  search_qps                    = " << search.search_qps
                  << "\n";
        std::cout << "  lat_p50_ms                    = "
                  << search.lat_p50_us / 1000.0 << "\n";
        std::cout << "  lat_p90_ms                    = "
                  << search.lat_p90_us / 1000.0 << "\n";
        std::cout << "  lat_p95_ms                    = "
                  << search.lat_p95_us / 1000.0 << "\n";
        std::cout << "  lat_p99_ms                    = "
                  << search.lat_p99_us / 1000.0 << "\n";
        std::cout << "  lat_p999_ms                   = "
                  << search.lat_p999_us / 1000.0 << "\n";
        std::cout << "  read_ios_per_op               = "
                  << search.mean_ios << "\n";
    }
    std::cout << "  final_recall@" << recall_at << "               = "
              << (search.recall >= 0.0 ? std::to_string(search.recall)
                                        : "N/A (no GT)")
              << std::endl;
    std::cout << "  newly_inserted_results_gt     = "
              << search.newly_inserted_results_gt << "\n";
    std::cout << "  newly_inserted_results_hit    = "
              << search.newly_inserted_results_hit << "\n";

    if (workload != "search_only") {
        const char *canonical_workload =
            workload == "insert_only"
                ? "insert"
                : (workload == "delete_only" ? "delete" : "mixed_update");
        emit_canonical_update_json(std::cout, canonical_workload, update,
                                   update_stats, metric_config, update_avg_rss_mb,
                                   update_peak_rss_mb);
    }
    emit_canonical_search_json(
        std::cout, workload == "search_only" ? "query" : "final_query",
        search, search_stats, metric_config, search_avg_rss_mb,
        search_peak_rss_mb);
    if (!result_file.empty()) {
        std::ofstream rf(result_file, std::ios::out | std::ios::trunc);
        if (!rf) {
            std::cerr << "WARNING: cannot write result file: " << result_file
                      << std::endl;
        } else {
            if (workload != "search_only") {
                const char *canonical_workload =
                    workload == "insert_only"
                        ? "insert"
                        : (workload == "delete_only" ? "delete" : "mixed_update");
                emit_canonical_update_json(rf, canonical_workload, update,
                                           update_stats, metric_config,
                                           update_avg_rss_mb,
                                           update_peak_rss_mb);
            }
            emit_canonical_search_json(
                rf, workload == "search_only" ? "query" : "final_query",
                search, search_stats, metric_config, search_avg_rss_mb,
                search_peak_rss_mb);
        }
    }

    diskann::aligned_free(queries_raw);
    bufann_free(idx);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    std::set_terminate([]() {
        if (auto ep = std::current_exception()) {
            try { std::rethrow_exception(ep); }
            catch (const diskann::ANNException &e) {
                std::cerr << "TERMINATE: ANNException: " << e.message()
                          << " (code=" << e.errorCode() << ")" << std::endl;
            }
            catch (const std::exception &e) {
                std::cerr << "TERMINATE: unhandled exception: " << e.what() << std::endl;
            }
            catch (...) { std::cerr << "TERMINATE: unknown exception" << std::endl; }
        } else {
            std::cerr << "TERMINATE: called without active exception" << std::endl;
        }
        std::abort();
    });
    if (has_flag(argc, argv, "--help") || argc < 2) {
        std::cout
            << "Usage: bufann_driver --data_type float|int8|uint8\n"
            << "         --index_prefix <path> --dim <N>\n"
            << "         --workload search_only|insert_only|delete_only|"
               "search_after_update\n"
            << "         --query_file <path>\n"
            << "        [--gt_file <path>] [--recall_at N] [--search_L N]\n"
            << "        [--beamwidth N] [--query_threads N] [--warmup_threads N]\n"
            << "        [--search_pq 0|1]  # default 1; 0 scores neighbors with full vectors\n"
            << "        [--R N] [--L N] [--C N] [--buffer_pool_frames N]\n"
            << "        [--pq_chunks N] [--max_dataset_size N] [--result_file <path>]\n"
            << "        [--meta_growth_copy_threads N]  # default: 16\n"
            << "        [--skip_meta_pre_reserve 0|1]  # default 0; RQ unchanged\n"
            << "  update workloads:\n"
            << "        [--full_data_file <path>]\n"
            << "        [--insert_ids_file <path>] [--delete_ids_file <path>]\n"
            << "        [--delete_algorithm freshdiskann|freshdiskann-nosnapshot|lazy]\n"
            << "        [--flush_background 0|1]\n"
            << "        [--flush_dirty_high_ratio F] [--flush_dirty_low_ratio F]\n"
            << "        [--flush_budget_pages N]\n"
            << "        [--preload_max_pages N] [--preload_threads N]\n"
            << "        [--update_preload 0|1]  # legacy; fires preload on update path\n"
            << "        [--bfs_warmup_pages N]  # mutually exclusive with --preload_max_pages\n"
            << "        [--run_maintenance 0|1] [--flush_after_maintenance 0|1]\n"
            << "        [--insert_search_L N] [--insert_threads N] [--delete_threads N]\n"
            << "        [--maintenance_threads N] [--maintenance_query_threads N]\n"
            << "        [--query_ratio N]  # [1,99] percent of foreground ops that are queries\n"
            << "        [--skip_update_search 0|1] [--enable_interval_metrics 0|1]\n"
            << "        [--insert_cap N] [--delete_cap N]" << std::endl;
        return 0;
    }

    const std::string dtype = get_arg(argc, argv, "--data_type", "float");
    try {
        if (dtype == "float") return run_workload<float>(argc, argv);
        if (dtype == "int8") return run_workload<int8_t>(argc, argv);
        if (dtype == "uint8") return run_workload<uint8_t>(argc, argv);
        std::cerr << "ERROR: unknown --data_type '" << dtype << "'" << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
