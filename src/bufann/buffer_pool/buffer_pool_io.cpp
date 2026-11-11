#include "bufann/buffer_pool/buffer_pool.h"

#include "ann_exception.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace diskann {
namespace inplace {

namespace {

void* checked_aligned_alloc(size_t alignment, size_t size, const char* what) {
    void* ptr = nullptr;
    int rc = posix_memalign(&ptr, alignment, size);
    if (rc != 0) {
        throw std::runtime_error(std::string("posix_memalign failed for ") +
                                 what + " rc=" + std::to_string(rc));
    }
    return ptr;
}

}  // namespace


void BufferPool::init(uint32_t page_size, uint32_t num_frames,
                           std::shared_ptr<FileManager> file_manager,
                           FileNo default_file_no,
                           InPlaceIOStats* stats,
                           uint32_t flush_budget_pages_per_cycle,
                           uint32_t flush_wakeup_ms) {
    stop_bg_flush();
    _page_size = page_size;
    _num_frames = num_frames;
    _file_manager = std::move(file_manager);
    _default_file_no = default_file_no;
    _stats = stats;
    _flush_budget_pages_per_cycle = std::max<uint32_t>(1, flush_budget_pages_per_cycle);
    _flush_wakeup_ms = std::max<uint32_t>(1, flush_wakeup_ms);
    _dirty_frame_count.store(0, std::memory_order_relaxed);

    _frames.clear();
    for (uint32_t i = 0; i < num_frames; ++i) {
        _frames.emplace_back();
        auto& f = _frames.back();
        f.frame_index = i;
        f.data = static_cast<char*>(checked_aligned_alloc(4096, page_size, "buffer frame"));
        memset(f.data, 0, page_size);
    }

    uint32_t desired_shards =
        std::min<uint32_t>(16, std::max<uint32_t>(4, num_frames / 64));
    _num_shards = std::max<uint32_t>(
        1, std::min<uint32_t>(num_frames == 0 ? 1 : num_frames, desired_shards));
    _shards.clear();
    _shards.reserve(_num_shards);
#ifdef BUFANN_ARRAY_PAGE_TABLE
    auto arrays = std::make_shared<ArrayPageTable::Arrays>();
#endif
    uint32_t frame_begin = 0;
    for (uint32_t shard_idx = 0; shard_idx < _num_shards; ++shard_idx) {
        auto shard = std::make_unique<ShardState>();
#ifdef BUFANN_ARRAY_PAGE_TABLE
        shard->page_table.attach(arrays);
#endif
        uint32_t frame_end = ((shard_idx + 1) * num_frames) / _num_shards;
        shard->frame_begin = frame_begin;
        shard->frame_end = frame_end;
        shard->clock_hand = frame_begin;
        shard->flush_cursor = frame_begin;
        _shards.emplace_back(std::move(shard));
        frame_begin = frame_end;
    }
}


uint32_t BufferPool::flush_dirty_budget(uint32_t max_pages) {
    if (_frames.empty()) return 0;
    uint32_t flushed = 0;
    constexpr uint32_t kMaxCollectBatch = 4096;
    uint32_t remaining =
        max_pages == 0 ? kMaxCollectBatch
                       : std::min(max_pages, kMaxCollectBatch);
    // Start at a different shard each call, so one shard that is clean is not rescanned every time.
    const uint32_t first = _flush_shard_next.fetch_add(1, std::memory_order_relaxed);
    for (uint32_t s = 0; s < _num_shards; ++s) {
        if (remaining == 0) break;
        auto frames = collect_dirty_frames(*_shards[(first + s) % _num_shards], remaining);
        for (size_t i = 0; i < frames.size(); i += kPinBatchMax) {
            write_frames(frames.data() + i, std::min<size_t>(kPinBatchMax, frames.size() - i));
        }
        flushed += static_cast<uint32_t>(frames.size());
        remaining -= static_cast<uint32_t>(frames.size());
    }
    return flushed;
}

void BufferPool::flush_all_dirty() {
    constexpr uint32_t kFlushBatchPages = 4096;
    for (;;) {
        if (flush_dirty_budget(kFlushBatchPages) == 0) break;
    }
}

uint32_t BufferPool::dirty_page_count() const {
    return _dirty_frame_count.load(std::memory_order_relaxed);
}

double BufferPool::dirty_ratio() const {
    if (_num_frames == 0) return 0.0;
    return static_cast<double>(dirty_page_count()) / static_cast<double>(_num_frames);
}

std::vector<PageFrame*> BufferPool::collect_dirty_frames(ShardState& shard, uint32_t max_frames) {
    std::vector<PageFrame*> frames;
    if (_frames.empty() || shard.frame_begin >= shard.frame_end) return frames;
    const uint32_t shard_frames = shard.frame_end - shard.frame_begin;
    const uint32_t budget = std::max<uint32_t>(1, std::min<uint32_t>(max_frames, shard_frames));
    // The cursor lock is also the clock's, so take it once per run of frames rather than once per frame.
    constexpr uint32_t kRun = 1024;
    for (uint32_t scanned = 0; scanned < shard_frames && frames.size() < budget;) {
        uint32_t begin, n;
        {
            std::lock_guard<SpinLock> clk(shard.cursor_lock);
            begin = shard.flush_cursor;
            n = std::min({kRun, shard.frame_end - begin, shard_frames - scanned});
            shard.flush_cursor = begin + n >= shard.frame_end ? shard.frame_begin : begin + n;
        }
        scanned += n;
        for (uint32_t idx = begin; idx < begin + n && frames.size() < budget; ++idx) {
            // A shared latch blocks writers and eviction but not readers; write_frames keeps it until the write is on
            // disk, then marks the frame clean.
            PageFrame& f = _frames[idx];
            if (!f.dirty.load(std::memory_order_relaxed) || !f.latch.try_lock_shared()) continue;
            const PageID page_id = f.page_id();
            if (page_id.first == INVALID_FILE || page_id.second == INVALID_PAGE ||
                !f.dirty.load(std::memory_order_acquire)) {
                f.latch.unlock_shared();
                continue;
            }
            frames.push_back(&f);
        }
    }
    return frames;
}



size_t BufferPool::configured_bytes() const noexcept {
    return static_cast<size_t>(_num_frames) * _page_size;
}


}  // namespace inplace
}  // namespace diskann
