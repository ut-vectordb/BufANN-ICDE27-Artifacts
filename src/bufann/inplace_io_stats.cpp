#include "bufann/inplace_io_stats.h"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <vector>

namespace diskann {
namespace inplace {

namespace {
// Per-thread batched query-hit counters. Relaxed atomics so a foreign
// thread (TlsRegistry::drain_into, called from reset_stats) can safely
// exchange-to-zero while the owner thread is mid-increment.
struct QueryHotStatsBuffer {
    std::atomic<uint64_t> cache_hits{0};
    QueryHotStatsBuffer();
    ~QueryHotStatsBuffer();
};

// Global registry of every live TLS buffer. Each thread self-registers on
// first access (TLS ctor) and deregisters on thread death (TLS dtor), so
// drain_into reaches every thread that has ever touched the counters —
// OMP workers, std::thread, anything — with no thread-count plumbing.
class TlsRegistry {
public:
    void add(QueryHotStatsBuffer* b) {
        std::lock_guard<std::mutex> lk(_mtx);
        _bufs.push_back(b);
    }
    void remove(QueryHotStatsBuffer* b) {
        std::lock_guard<std::mutex> lk(_mtx);
        _bufs.erase(std::remove(_bufs.begin(), _bufs.end(), b), _bufs.end());
    }
    void drain_into(InPlaceIOStats& s) {
        std::lock_guard<std::mutex> lk(_mtx);
        for (auto* b : _bufs) {
            uint64_t h = b->cache_hits.exchange(0, std::memory_order_relaxed);
            if (h) s.cache_hits.fetch_add(h, std::memory_order_relaxed);
        }
    }
private:
    std::mutex _mtx;
    std::vector<QueryHotStatsBuffer*> _bufs;
};

static TlsRegistry& tls_registry() {
    static TlsRegistry r;
    return r;
}

QueryHotStatsBuffer::QueryHotStatsBuffer()  { tls_registry().add(this); }
QueryHotStatsBuffer::~QueryHotStatsBuffer() { tls_registry().remove(this); }

thread_local QueryHotStatsBuffer tls_query_hot_stats;
thread_local uint64_t t_page_accesses = 0;
thread_local uint64_t t_phys_reads = 0;

}  // namespace

void record_query_cache_hit(uint64_t n) {
    tls_query_hot_stats.cache_hits.fetch_add(n, std::memory_order_relaxed);
}

void flush_query_hot_stats(InPlaceIOStats& stats) {
    uint64_t h = tls_query_hot_stats.cache_hits.exchange(0, std::memory_order_relaxed);
    if (h) stats.cache_hits.fetch_add(h, std::memory_order_relaxed);
}


void drain_query_hot_stats(InPlaceIOStats& stats) {
    tls_registry().drain_into(stats);
}

uint64_t thread_page_access_count() { return t_page_accesses; }
uint64_t thread_physical_reads_count() { return t_phys_reads; }

void record_thread_page_accesses(uint64_t n) { t_page_accesses += n; }
void record_thread_physical_reads(uint64_t n) { t_phys_reads += n; }

// ===========================================================================
// InPlaceIOStats
// ===========================================================================
void InPlaceIOStats::reset() {
    sequential_read_ios.store(0);
    sequential_read_bytes.store(0);
    buffer_pool_write_ios.store(0);
    cache_hits.store(0);
    cache_misses.store(0);
    evictions.store(0);
    dirty_evictions.store(0);
    page_fault_total.store(0);
    page_fault_with_evict.store(0);
    page_fault_no_evict.store(0);
    page_fault_total_ns.store(0);
    page_in_ns.store(0);
    eviction_total_ns.store(0);
    eviction_scan_ns.store(0);
    eviction_wait_pins_ns.store(0);
    eviction_flush_ns.store(0);
    page_fault_page_table_ns.store(0);
    page_fault_frame_setup_ns.store(0);
    batch_pin_calls.store(0);
    batch_misses_total.store(0);
    batch_with_evict.store(0);
    batch_no_evict.store(0);
    batch_io_ns_total.store(0);
    pages_flushed.store(0);
    dirty_page_bytes_flushed.store(0);
    total_inserts.store(0);
    total_deletes.store(0);
    repair_queue_length.store(0);
    query_comparable_cpu_ns.store(0);
    query_comparable_cmps.store(0);
    query_io_ns.store(0);
    update_io_ns.store(0);
    total_io_ns.store(0);
    meta_growth_count.store(0);
    meta_growth_drain_ns.store(0);
    meta_growth_alloc_ns.store(0);
    meta_growth_copy_ns.store(0);
    meta_growth_total_ns.store(0);
    meta_growth_last_from.store(0);
    meta_growth_last_to.store(0);
    meta_growth_last_workers.store(0);
    meta_growth_last_copy_bytes.store(0);
    meta_growth_last_started_us.store(0);
    meta_growth_last_finished_us.store(0);
    meta_growth_update_stall_ns.store(0);
    meta_retired_generations.store(0);
    meta_reclaimed_generations.store(0);
}

std::string InPlaceIOStats::to_json() const {
    std::ostringstream ss;
    ss << "{"
       << "\"sequential_read_ios\":" << sequential_read_ios.load() << ","
       << "\"sequential_read_bytes\":" << sequential_read_bytes.load() << ","
       << "\"buffer_pool_write_ios\":" << buffer_pool_write_ios.load() << ","
       << "\"cache_hits\":" << cache_hits.load() << ","
       << "\"cache_misses\":" << cache_misses.load() << ","
       << "\"evictions\":" << evictions.load() << ","
       << "\"dirty_evictions\":" << dirty_evictions.load() << ","
       << "\"page_fault_total\":" << page_fault_total.load() << ","
       << "\"page_fault_with_evict\":" << page_fault_with_evict.load() << ","
       << "\"page_fault_no_evict\":" << page_fault_no_evict.load() << ","
       << "\"page_fault_total_ns\":" << page_fault_total_ns.load() << ","
       << "\"page_in_ns\":" << page_in_ns.load() << ","
       << "\"eviction_total_ns\":" << eviction_total_ns.load() << ","
       << "\"eviction_scan_ns\":" << eviction_scan_ns.load() << ","
       << "\"eviction_wait_pins_ns\":" << eviction_wait_pins_ns.load() << ","
       << "\"eviction_flush_ns\":" << eviction_flush_ns.load() << ","
       << "\"page_fault_page_table_ns\":" << page_fault_page_table_ns.load() << ","
       << "\"page_fault_frame_setup_ns\":" << page_fault_frame_setup_ns.load() << ","
       << "\"pages_flushed\":" << pages_flushed.load() << ","
       << "\"dirty_page_bytes_flushed\":" << dirty_page_bytes_flushed.load()
       << ","
       << "\"total_inserts\":" << total_inserts.load() << ","
       << "\"total_deletes\":" << total_deletes.load() << ","
       << "\"query_comparable_cpu_ns\":" << query_comparable_cpu_ns.load() << ","
       << "\"query_comparable_cmps\":" << query_comparable_cmps.load() << ","
       << "\"query_io_ns\":" << query_io_ns.load() << ","
       << "\"update_io_ns\":" << update_io_ns.load() << ","
       << "\"total_io_ns\":" << total_io_ns.load() << ","
       << "\"meta_growth_count\":" << meta_growth_count.load() << ","
       << "\"meta_growth_drain_ns\":" << meta_growth_drain_ns.load() << ","
       << "\"meta_growth_alloc_ns\":" << meta_growth_alloc_ns.load() << ","
       << "\"meta_growth_copy_ns\":" << meta_growth_copy_ns.load() << ","
       << "\"meta_growth_total_ns\":" << meta_growth_total_ns.load() << ","
       << "\"meta_growth_last_from\":" << meta_growth_last_from.load() << ","
       << "\"meta_growth_last_to\":" << meta_growth_last_to.load() << ","
       << "\"meta_growth_last_workers\":" << meta_growth_last_workers.load()
       << ","
       << "\"meta_growth_last_copy_bytes\":"
       << meta_growth_last_copy_bytes.load() << ","
       << "\"meta_growth_last_started_us\":"
       << meta_growth_last_started_us.load() << ","
       << "\"meta_growth_last_finished_us\":"
       << meta_growth_last_finished_us.load() << ","
       << "\"meta_growth_update_stall_ns\":"
       << meta_growth_update_stall_ns.load() << ","
       << "\"meta_retired_generations\":"
       << meta_retired_generations.load() << ","
       << "\"meta_reclaimed_generations\":"
       << meta_reclaimed_generations.load()
       << "}";
    return ss.str();
}


}  // namespace inplace
}  // namespace diskann
