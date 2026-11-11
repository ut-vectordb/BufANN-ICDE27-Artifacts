#pragma once

// Lazy deletion (DeleteAlgorithm::Lazy), run in episodes.
// While no episode runs, a delete only records its tag in the store's deleted-tag set, and searches do nothing more.
// An episode starts when bufann_cleanup_deleted_edges is called with its maintenance threads, which become the
// episode's repair workers. It resolves the deleted tags to nodes and snapshots their neighbors. Each search thread
// then hands the pages it reads to one worker, always the same one, and each worker repairs the pages it is handed
// while they are still cached; whenever its queue is empty it sweeps the next pages in order instead. Once every page
// is done, no edge points to a node resolved at the start, so the caller frees their slots and the episode ends.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "bufann/buffer_pool/buffer_pool.h"
#include "bufann/inplace_defs.h"

namespace diskann {

template <typename T>
class FixedChunkPQTable;

namespace inplace {

class InPlaceGraphStore;

struct LazyDeleteStats {
    std::atomic<uint64_t> episodes{0};
    std::atomic<uint64_t> resolved{0};        // deleted nodes resolved at the start of an episode
    std::atomic<uint64_t> handed{0};          // pages searches handed to the workers
    std::atomic<uint64_t> pages_handed{0};    // pages finished from the workers' queues
    std::atomic<uint64_t> pages_swept{0};     // pages finished by the workers' sweep
    std::atomic<uint64_t> evicted_handed{0};  // handed pages evicted before their worker reached them
    std::atomic<uint64_t> repaired{0};        // nodes whose neighbor list was rewritten
    std::atomic<uint64_t> freed{0};           // slots freed at the end of an episode
};

struct LazyShared;  // episode state shared by every thread (lazy_delete.cpp)

struct LazyDeletes {
    LazyDeletes(unsigned R, unsigned C, float alpha);
    ~LazyDeletes();

    unsigned R = 0;
    unsigned C = 0;
    float alpha = 1.2f;
    LazyDeleteStats stats;
    std::shared_ptr<LazyShared> shared;
};

// Runs one episode with num_threads repair workers and returns how many slots it freed; called by
// bufann_cleanup_deleted_edges. Blocks until every graph page is done; a call made during an episode waits for it and
// then runs its own.
template <typename T>
size_t lazy_run_episode(InPlaceGraphStore* store, FixedChunkPQTable<T>* pq_table, uint32_t num_threads);

// Called by a search after each batch of page pins: hands the batch's pages to this thread's worker while an episode
// runs, and does nothing otherwise.
void lazy_hand_pages(InPlaceGraphStore* store, const PageGuard* pages, size_t n_pages);

}  // namespace inplace
}  // namespace diskann
