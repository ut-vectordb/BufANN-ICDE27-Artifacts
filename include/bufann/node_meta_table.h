#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "bufann/inplace_defs.h"
#include "bufann/inplace_io_stats.h"

class ThreadPool;

namespace diskann {
namespace inplace {

// Return tcmalloc free spans to the OS (no-op if unavailable). Used after large metadata teardown (store destructor, meta
// reclaim worker).
void release_allocator_free_memory() noexcept;

// A node ID is its slot address, so the state array also records which slots are free. States are listed in
// lifecycle order. Allocating means an insert owns the slot but has not published the node yet.
enum class NodeState : uint8_t {
    Free = 0,
    Allocating = 1,
    Active = 2,
    DeletePending = 3,
};

// ---------------------------------------------------------------------------
// NodeMetaTable -- per-node metadata (state, tag, PQ codes) indexed by internal node ID, with online growth.
//
// The arrays are published RCU-style as one generation at a time. Growth builds a larger generation, copies the old
// one in, and publishes it; the old generation is retired and freed by a background reclaim worker once no reader
// still holds it. Every operation holds an epoch ticket (Guard) for its duration: readers (queries) may keep using a
// retired generation while growth copies, but writers (insert/delete/cleanup) are drained before the copy so updates
// cannot land on the source generation mid-grow.
// ---------------------------------------------------------------------------
class NodeMetaTable {
 public:
    static constexpr size_t kDefaultCapacity = 1u << 20;

    struct Arrays {
        explicit Arrays(size_t count, uint32_t pq_chunks);
        size_t capacity = 0;
        uint32_t n_chunks = 0;
        std::unique_ptr<TagType[]> tags;
        std::unique_ptr<std::atomic<NodeState>[]> states;
        std::unique_ptr<uint8_t[]> pq_codes;
    };

    // RCU epoch ticket, released on destruction.
    class Guard {
     public:
        Guard() = default;
        Guard(Guard&& other) noexcept;
        Guard& operator=(Guard&& other) noexcept;
        ~Guard();
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
     private:
        friend class NodeMetaTable;
        Guard(NodeMetaTable* table, uint8_t epoch, bool writer) : _table(table), _epoch(epoch), _writer(writer) {}
        NodeMetaTable* _table = nullptr;
        uint8_t _epoch = 0;
        bool _writer = false;
    };

    explicit NodeMetaTable(InPlaceIOStats& stats);
    ~NodeMetaTable();
    NodeMetaTable(const NodeMetaTable&) = delete;
    NodeMetaTable& operator=(const NodeMetaTable&) = delete;

    // Publish a fresh generation and drop any retired ones. Not safe against concurrent operations; called from store init only.
    void init(size_t capacity, uint32_t pq_chunks, uint32_t copy_threads);
    // Stop the reclaim worker and free every generation.
    void shutdown() noexcept;

    Guard enter_read();
    Guard enter_write();

    // _current is atomic so that growth can swap in a larger array safely. On x86 this load is a plain mov.
    Arrays* current() const { return _current.load(std::memory_order_acquire); }
    size_t capacity() const;
    bool growth_active() const { return _growth_active.load(std::memory_order_acquire); }
    size_t retired_count() const { return _retired_count.load(std::memory_order_acquire); }

    // Grow to at least `needed` entries, by 1.2x of the current capacity. Must not be called while holding a write
    // Guard: growth waits for all writers to drain and would deadlock on the caller's own ticket.
    void reserve(size_t needed, uint32_t pq_chunks);
    // Republish the current capacity with PQ storage of `pq_chunks` width if the current generation lacks it. Same
    // no-write-Guard rule as reserve().
    void ensure_pq_chunks(uint32_t pq_chunks);

 private:
    struct Retired {
        uint8_t reader_epoch = 0;
        std::unique_ptr<Arrays> arrays;
    };

    Guard enter_epoch(bool writer);
    void release_epoch(uint8_t epoch, bool writer) noexcept;
    void begin_growth_locked(std::unique_lock<std::mutex>& lock);
    void end_growth_locked();
    class GrowthGuard {
     public:
        GrowthGuard(NodeMetaTable& table, std::unique_lock<std::mutex>& lock);
        ~GrowthGuard();
        GrowthGuard(const GrowthGuard&) = delete;
        GrowthGuard& operator=(const GrowthGuard&) = delete;
     private:
        NodeMetaTable& _table;
    };
    void copy_prefix_parallel(const Arrays& source, Arrays& destination, size_t count);
    // Swap `next` in as the current generation and retire the old one. Caller holds _growth_mtx and a GrowthGuard.
    void publish_locked(std::unique_ptr<Arrays> next);

    void start_reclaim_worker_locked();
    void notify_retirement();
    void request_reclaim(uint8_t epoch) noexcept;
    std::unique_ptr<Arrays> take_reclaimable_locked();
    void reclaim_loop();
    void stop_reclaim_worker() noexcept;

    InPlaceIOStats& _stats;
    std::atomic<Arrays*> _current{nullptr};
    std::unique_ptr<Arrays> _owned;
    std::vector<Retired> _retired;
    mutable std::mutex _growth_mtx;
    std::condition_variable _growth_cv;
    std::atomic<bool> _growth_active{false};
    uint32_t _copy_threads = 16;
    std::unique_ptr<::ThreadPool> _copy_pool;
    std::atomic<uint8_t> _reader_epoch{0};
    std::array<std::atomic<uint64_t>, 2> _readers;
    std::array<std::atomic<uint64_t>, 2> _writers;
    std::array<std::atomic<bool>, 2> _retired_pending;
    std::mutex _reclaim_wait_mtx;
    std::condition_variable _reclaim_cv;
    std::thread _reclaim_thread;
    std::atomic<bool> _reclaim_requested{false};
    std::atomic<bool> _reclaim_stop{false};
    std::atomic<size_t> _retired_count{0};
    bool _reclaim_started = false;  // Protected by _growth_mtx.
};

}  // namespace inplace
}  // namespace diskann
