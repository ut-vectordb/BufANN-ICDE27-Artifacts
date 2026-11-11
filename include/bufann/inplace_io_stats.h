#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace diskann {
namespace inplace {

struct InPlaceIOStats {
    std::atomic<uint64_t> sequential_read_ios{0};
    std::atomic<uint64_t> sequential_read_bytes{0};
    std::atomic<uint64_t> buffer_pool_write_ios{0};

    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> dirty_evictions{0};
    std::atomic<uint64_t> page_fault_total{0};
    std::atomic<uint64_t> page_fault_with_evict{0};
    std::atomic<uint64_t> page_fault_no_evict{0};
    std::atomic<uint64_t> page_fault_total_ns{0};
    std::atomic<uint64_t> page_in_ns{0};
    std::atomic<uint64_t> eviction_total_ns{0};
    std::atomic<uint64_t> eviction_scan_ns{0};
    std::atomic<uint64_t> eviction_wait_pins_ns{0};
    std::atomic<uint64_t> eviction_flush_ns{0};
    std::atomic<uint64_t> page_fault_page_table_ns{0};
    std::atomic<uint64_t> page_fault_frame_setup_ns{0};

    // Batched pin via libaio. Tracked separately to keep batch I/O distinguishable
    // in-process; the driver folds these into the simple page_fault_* / page_in_ns
    // counters before emitting JSON.
    std::atomic<uint64_t> batch_pin_calls{0};
    std::atomic<uint64_t> batch_misses_total{0};
    std::atomic<uint64_t> batch_with_evict{0};
    std::atomic<uint64_t> batch_no_evict{0};
    std::atomic<uint64_t> batch_io_ns_total{0};

    std::atomic<uint64_t> pages_flushed{0};
    std::atomic<uint64_t> dirty_page_bytes_flushed{0};

    std::atomic<uint64_t> total_inserts{0};
    std::atomic<uint64_t> total_deletes{0};

    std::atomic<uint64_t> repair_queue_length{0};

    std::atomic<uint64_t> query_comparable_cpu_ns{0};
    std::atomic<uint64_t> query_comparable_cmps{0};
    std::atomic<uint64_t> query_io_ns{0};
    std::atomic<uint64_t> update_io_ns{0};
    std::atomic<uint64_t> total_io_ns{0};

    std::atomic<uint64_t> meta_growth_count{0};
    std::atomic<uint64_t> meta_growth_drain_ns{0};
    std::atomic<uint64_t> meta_growth_alloc_ns{0};
    std::atomic<uint64_t> meta_growth_copy_ns{0};
    std::atomic<uint64_t> meta_growth_total_ns{0};
    std::atomic<uint64_t> meta_growth_last_from{0};
    std::atomic<uint64_t> meta_growth_last_to{0};
    std::atomic<uint64_t> meta_growth_last_workers{0};
    std::atomic<uint64_t> meta_growth_last_copy_bytes{0};
    // Steady-clock time_since_epoch microseconds for last completed growth.
    std::atomic<uint64_t> meta_growth_last_started_us{0};
    std::atomic<uint64_t> meta_growth_last_finished_us{0};
    // Accumulated writer wait time while meta growth holds admission closed.
    std::atomic<uint64_t> meta_growth_update_stall_ns{0};
    // Cumulative generations retired/reclaimed asynchronously.
    std::atomic<uint64_t> meta_retired_generations{0};
    std::atomic<uint64_t> meta_reclaimed_generations{0};

    void   reset();
    std::string to_json() const;
    void record_io_ns(bool is_query, uint64_t ns) {
        total_io_ns.fetch_add(ns, std::memory_order_relaxed);
        if (is_query) {
            query_io_ns.fetch_add(ns, std::memory_order_relaxed);
        } else {
            update_io_ns.fetch_add(ns, std::memory_order_relaxed);
        }
    }
};


uint64_t thread_page_access_count();
uint64_t thread_physical_reads_count();
void record_thread_page_accesses(uint64_t n = 1);
void record_thread_physical_reads(uint64_t n = 1);
void record_query_cache_hit(uint64_t n = 1);
void flush_query_hot_stats(InPlaceIOStats& stats);
void drain_query_hot_stats(InPlaceIOStats& stats);

}  // namespace inplace
}  // namespace diskann
