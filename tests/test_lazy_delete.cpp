#include "bufann/bufann_api.h"

#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

using namespace diskann::inplace;

namespace {

constexpr uint32_t kN = 20000;
constexpr uint32_t kInserts = 500;  // inserted, and as many deleted, while the first episode runs

// Runs an episode and returns how many slots it freed, read from the stats since cleanup returns nothing.
// Two repair workers; each setup thread pins up to 256 pages at once, so the pool must hold more than 512.
size_t bufann_cleanup_deleted_edges_freed(BufANNIndex<float>& idx) {
    const uint64_t before = idx.store->lazy_deletes()->stats.freed;
    bufann_cleanup_deleted_edges(idx, 2);
    return idx.store->lazy_deletes()->stats.freed - before;
}
constexpr uint32_t kDim = 8;
constexpr uint32_t kQueries = 2000;
constexpr uint32_t kK = 10;

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<float> random_vectors(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-10, 10);
    std::vector<float> v(n * kDim);
    for (float& x : v) x = dist(rng);
    return v;
}

// Vectors inserted after the build, tagged kN onwards; recall's brute force covers them too.
std::vector<float> g_inserted;

// Recall@kK of searches over all queries against brute force on the tags not in `deleted`.
double recall(BufANNIndex<float>& idx, const std::vector<float>& base, const std::vector<float>& queries,
              const std::set<uint32_t>& deleted, bool check_no_deleted_results) {
    size_t hits = 0;
#pragma omp parallel for num_threads(8) reduction(+ : hits) schedule(dynamic, 16)
    for (uint32_t q = 0; q < kQueries; ++q) {
        const float* qv = queries.data() + static_cast<size_t>(q) * kDim;
        std::vector<std::pair<float, uint32_t>> all;
        const uint32_t n_all = kN + static_cast<uint32_t>(g_inserted.size() / kDim);
        for (uint32_t i = 0; i < n_all; ++i) {
            if (deleted.count(i)) continue;
            const float* v = i < kN ? base.data() + static_cast<size_t>(i) * kDim
                                    : g_inserted.data() + static_cast<size_t>(i - kN) * kDim;
            float d = 0;
            for (uint32_t c = 0; c < kDim; ++c) d += (qv[c] - v[c]) * (qv[c] - v[c]);
            all.emplace_back(d, i);
        }
        std::partial_sort(all.begin(), all.begin() + kK, all.end());
        std::set<uint32_t> truth;
        for (uint32_t k = 0; k < kK; ++k) truth.insert(all[k].second);
        TagType out[kK];
        const uint32_t n = bufann_query_into(idx, qv, kK, out, 50);
        for (uint32_t k = 0; k < n; ++k) {
            if (check_no_deleted_results && deleted.count(out[k])) throw std::runtime_error("deleted tag returned");
            hits += truth.count(out[k]);
        }
    }
    return static_cast<double>(hits) / (kQueries * kK);
}

// Every edge must point to a node whose slot is still in use, since a slot is freed only once nothing points to it.
// After an episode, no edge may point to a deleted node either.
void check_edges(InPlaceGraphStore& store, uint32_t R, bool after_episode) {
    for (uint32_t id = 0; id < kN; ++id) {
        if (!store.is_active(id)) continue;
        auto frame = store.pin_page_for_node(id, READ);
        auto node = store.read_guard(frame).read_node(id);
        require(node.valid() && node.node_id() == id, "invalid node");
        std::set<uint32_t> unique;
        for (uint16_t j = 0; j < node.degree(); ++j) {
            require(node.neighbors()[j] != id, "self edge");
            require(store.node_present(node.neighbors()[j]), "edge to a freed slot");
            require(!after_episode || store.is_active(node.neighbors()[j]), "edge to a deleted node");
            require(unique.insert(node.neighbors()[j]).second, "duplicate edge");
        }
        require(node.degree() <= R, "degree overflow");
    }
}

}  // namespace

int main() {
    char temp[] = "/tmp/bufann-lazy-delete-XXXXXX";
    const char* dir = mkdtemp(temp);
    if (!dir) return 1;
    try {
        const std::string root(dir);
        const std::string data = root + "/data.bin";
        const std::vector<float> base = random_vectors(kN, 42);
        const std::vector<float> queries = random_vectors(kQueries, 7);
        {
            std::ofstream out(data, std::ios::binary);
            const uint32_t header[] = {kN, kDim};
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            out.write(reinterpret_cast<const char*>(base.data()), base.size() * sizeof(float));
        }

        BufANNConfig cfg;
        cfg.dim = kDim;
        cfg.R = 16;
        cfg.L = 32;
        cfg.C = 64;
        cfg.pq_chunks = 4;
        cfg.build_threads = 8;
        cfg.buffer_pool_frames = 4096;
        const std::string prefix = root + "/idx";
        {
            std::unique_ptr<BufANNIndex<float>> built(bufann_build<float>(data, prefix, cfg));
            built->pq_table.load_pq_centroid_bin((prefix + "_pq_pivots.bin").c_str(), cfg.pq_chunks);
            built->store->load_pq_codes_from_disk_index(prefix, cfg.pq_chunks);
            for (uint32_t id = 0; id < kN; ++id) {
                auto frame = built->store->pin_page_for_node(id, READ);
                auto guard = built->store->read_guard(frame);
                built->store->encode_pq(id, guard.read_node(id).coords<float>(), built->pq_table);
            }
            bufann_flush_dirty(*built);
            built->store->save_snapshot(prefix + ".meta");
        }

        // A pool smaller than the index, so searches miss and handed pages can be evicted before their worker comes.
        cfg.buffer_pool_frames = 640;
        cfg.delete_algorithm = DeleteAlgorithm::Lazy;
        std::unique_ptr<BufANNIndex<float>> idx(bufann_load<float>(prefix, cfg));
        InPlaceGraphStore& store = *idx->store;
        std::cout << "pages=" << store.total_pages() << " frames=" << cfg.buffer_pool_frames << "\n";

        const double recall_before = recall(*idx, base, queries, {}, false);

        std::set<uint32_t> deleted;
        std::mt19937 rng(1);
        while (deleted.size() < kN / 10) deleted.insert(rng() % kN);
        for (uint32_t tag : deleted) bufann_delete(*idx, tag);
        require(store.deleted_tag_count() == deleted.size(), "wrong deleted tag count");
        require(store.num_active() == kN, "a delete changed node state before any search");

        // An episode while 8 threads keep searching and two more insert and delete: the searches hand the pages they
        // read to the two repair workers. Deletes made during the episode wait for the next one.
        const LazyDeleteStats& st = store.lazy_deletes()->stats;
        std::atomic<bool> episode_done{false};
        size_t freed = 0;
        std::thread episode([&] {
            freed = bufann_cleanup_deleted_edges_freed(*idx);
            episode_done = true;
        });
        const std::vector<float> inserted = random_vectors(kInserts, 99);
        std::vector<uint32_t> deleted_during;
        for (uint32_t tag = 0; deleted_during.size() < kInserts; ++tag) {
            if (!deleted.count(tag)) deleted_during.push_back(tag);
        }
        std::thread updater([&] {
            for (uint32_t i = 0; i < kInserts; ++i) {
                bufann_insert(*idx, kN + i, inserted.data() + static_cast<size_t>(i) * kDim);
                bufann_delete(*idx, deleted_during[i]);
            }
        });
        double recall_mid = 0;
        while (!episode_done) recall_mid = recall(*idx, base, queries, deleted, true);
        episode.join();
        updater.join();
        g_inserted = inserted;
        const size_t resolved_first = deleted.size();
        deleted.insert(deleted_during.begin(), deleted_during.end());
        std::cout << "searching: freed=" << freed << " handed=" << st.handed << " pages_handed=" << st.pages_handed
                  << " pages_swept=" << st.pages_swept << " evicted_handed=" << st.evicted_handed
                  << " repaired=" << st.repaired << "\n";
        require(st.pages_handed > 0, "no handed page was repaired while searches ran");
        require(freed == resolved_first, "episode did not free every deleted node");
        // Nodes deleted during the episode stay Active until the next one resolves them.
        require(store.num_active() == kN + kInserts - resolved_first && store.deleted_tag_count() == kInserts,
                "wrong state after episode");
        check_edges(store, cfg.R, true);
        const double recall_after = recall(*idx, base, queries, deleted, true);

        // An episode with no search running, so the workers only sweep. It also resolves the deletes made during
        // the first one.
        std::set<uint32_t> more;
        while (more.size() < kN / 20) {
            const uint32_t tag = rng() % kN;
            if (!deleted.count(tag)) more.insert(tag);
        }
        for (uint32_t tag : more) bufann_delete(*idx, tag);
        const uint64_t swept_before = st.pages_swept;
        freed = bufann_cleanup_deleted_edges_freed(*idx);
        std::cout << "sweep only: freed=" << freed << " pages_swept=" << st.pages_swept - swept_before << "\n";
        require(freed == more.size() + deleted_during.size() && st.pages_swept > swept_before,
                "sweep-only episode did not finish");
        require(store.deleted_tag_count() == 0, "deleted tags left after the sweep-only episode");
        check_edges(store, cfg.R, true);
        deleted.insert(more.begin(), more.end());

        std::cout << "recall@" << kK << " before=" << recall_before << " during=" << recall_mid
                  << " after_episode=" << recall_after << " after_sweep_episode=" << recall(*idx, base, queries, deleted, true)
                  << "\n";

        idx.reset();
        std::filesystem::remove_all(root);
        std::cout << "Lazy deletion tests passed\n";
        return 0;
    } catch (const diskann::ANNException& e) {
        std::cerr << e.message() << " (artifacts: " << dir << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << " (artifacts: " << dir << ")\n";
        return 1;
    }
}
