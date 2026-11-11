#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WINDOWS
#include <unistd.h>
#endif

namespace bufann_bench {

inline double safe_div(double num, double den) {
    return den > 0.0 ? num / den : 0.0;
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

    ~RssSampler() { stop(); }

    double avg_mb() const {
        return samples_.empty()
                   ? 0.0
                   : std::accumulate(samples_.begin(), samples_.end(), 0.0) /
                         static_cast<double>(samples_.size());
    }

    double peak_mb() const {
        return samples_.empty()
                   ? 0.0
                   : *std::max_element(samples_.begin(), samples_.end());
    }

  private:
    void start_worker() {
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
};

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
    if (!first) out << ",";
    first = false;
    out << "\"" << json_escape(key) << "\":\"" << json_escape(value) << "\"";
}

inline void json_kv(std::ostream &out, bool &first, const std::string &key,
                    const char *value) {
    json_kv(out, first, key, std::string(value));
}

template <typename T>
inline void json_kv(std::ostream &out, bool &first, const std::string &key,
                    const T &value) {
    if (!first) out << ",";
    first = false;
    out << "\"" << json_escape(key) << "\":" << value;
}

inline void emit_rss_stats(std::ostream &out, bool &first,
                           const RssStats &rss) {
    json_kv(out, first, "avg_rss_mb", rss.avg_mb);
    json_kv(out, first, "peak_rss_mb", rss.peak_mb);
    json_kv(out, first, "rss_sample_count", rss.sample_count);
}

}  // namespace bufann_bench
