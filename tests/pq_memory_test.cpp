#include "bufann/bufann_api.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace diskann;
using namespace diskann::inplace;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// An exact, deterministic codebook avoids training noise in lifecycle checks.
static void write_pivots(const std::string& path) {
    std::vector<float> pivots(256 * 8);
    for (size_t i = 0; i < 256; ++i)
        for (size_t d = 0; d < 8; ++d) pivots[i * 8 + d] = static_cast<float>(i) - 128;
    float mean[8] = {};
    uint32_t permutation[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint32_t offsets[3] = {0, 4, 8};
    save_bin<float>(path, pivots.data(), 256, 8);
    save_bin<float>(path + "_centroid.bin", mean, 8, 1);
    save_bin<uint32_t>(path + "_rearrangement_perm.bin", permutation, 8, 1);
    save_bin<uint32_t>(path + "_chunk_offsets.bin", offsets, 3, 1);
}

template<typename T>
static void run_case(const std::string& prefix) {
    const std::string pivots = prefix + "_pq_pivots.bin";
    write_pivots(pivots);
    BufANNConfig config;
    config.dim = 8;
    config.R = 4;
    config.L = 8;
    config.C = 16;
    config.pq_chunks = 2;
    config.buffer_pool_frames = 512;
    config.max_dataset_size = 4;
    config.meta_growth_copy_threads = 2;
    auto make_index = [&](bool fresh) {
        auto idx = std::make_unique<BufANNIndex<T>>();
        idx->config = config;
        idx->aligned_dim = 8;
        idx->pq_prefix = prefix;
        idx->pq_table.load_pq_centroid_bin(pivots.c_str(), 2);
        // PQ traversal does not require a raw-distance comparator for insertion.
        if constexpr (std::is_same_v<T, float>) idx->dist_cmp = new DistanceL2();
        else if constexpr (std::is_same_v<T, uint8_t>) idx->dist_cmp = new DistanceL2UInt8();
        else idx->dist_cmp = new DistanceL2Int8();
        idx->store->init(8, 4, sizeof(T), 4096, 512, prefix + ".heap",
                         32, 100, fresh, 4, 2, 2);
        if (!fresh) {
            idx->store->load_pq_codes_from_disk_index(prefix, 2);
            idx->store->load_snapshot(prefix + ".meta");
        }
        return idx;
    };
    auto idx = make_index(true);
    std::vector<std::array<T, 8>> vectors(24);
    std::vector<std::array<uint8_t, 2>> expected(24);
    // Node IDs are slot addresses chosen by claim_page, so they are not 0..N-1.
    std::vector<unsigned> node_ids(vectors.size(), INVALID_NODE);
    for (uint32_t i = 0; i < vectors.size(); ++i) {
        vectors[i].fill(static_cast<T>(i * 3));
        idx->pq_table.deflate_vec(vectors[i].data(), expected[i].data());
        bufann_insert(*idx, 1000 + i, vectors[i].data(), 8, &node_ids[i]);
        require(node_ids[i] != INVALID_NODE &&
                    std::count(node_ids.begin(), node_ids.begin() + i, node_ids[i]) == 0,
                "insertion ID missing or duplicated");
        if (i == 0) idx->medoids = {node_ids[0]};
    }
    auto store = idx->store;
    require(store->meta_capacity() >= vectors.size(), "insertion did not grow metadata");
    require(store->pq_data_bytes() == store->meta_capacity() * 2, "PQ is not accounted as a dense memory array");
    require(store->pq_codes_for_node(node_ids[0]) != store->pq_codes_for_node(node_ids[1]), "single-code access still uses shared scratch");
    {
        auto guard = store->enter_meta_read();
        const std::vector<unsigned>& ids = node_ids;
        std::vector<uint8_t> gathered(ids.size() * 2);
        const auto reads = store->stats().page_fault_total.load();
        store->gather_pq_codes(ids.data(), ids.size(), gathered.data());
        require(store->stats().page_fault_total.load() == reads, "PQ gather performed heap IO");
        for (size_t i = 0; i < ids.size(); ++i)
            require(std::equal(expected[i].begin(), expected[i].end(), gathered.begin() + i * 2), "PQ code changed during insertion growth");
        // Grow from another thread while this reader holds its guard; the codes must survive the copy.
        std::thread grower([&] { store->reserve_meta_capacity(store->meta_capacity() * 2); });
        grower.join();
        store->gather_pq_codes(ids.data(), ids.size(), gathered.data());
        for (size_t i = 0; i < ids.size(); ++i)
            require(std::equal(expected[i].begin(), expected[i].end(), gathered.begin() + i * 2), "retired metadata lost PQ bytes");
    }
    {
        auto guard = store->enter_meta_read();
        const unsigned ids[] = {node_ids[0], node_ids[7], INVALID_NODE};
        uint8_t scratch[6] = {255, 255, 255, 255, 255, 255};
        uint8_t reference_codes[6] = {expected[0][0], expected[0][1],
                                      expected[7][0], expected[7][1], 0, 0};
        float actual[3] = {}, reference[3] = {};
        idx->pq_table.compute_distances(expected[2].data(), reference_codes, reference, 3);
        store->compute_pq_dists_src(node_ids[2], ids, actual, 3, scratch, idx->pq_table);
        require(std::equal(scratch, scratch + 6, reference_codes), "PQ distance gather mismatch");
        require(std::equal(actual, actual + 3, reference), "PQ distances changed");
        actual[0] = -1;
        store->compute_pq_dists_src(INVALID_NODE, ids, actual, 3, scratch, idx->pq_table);
        require(actual[0] == -1, "missing source should leave distances untouched");
    }
    auto hits = bufann_query(*idx, vectors[3].data(), 4, 16);
    require(std::find(hits.begin(), hits.end(), 1003) != hits.end(), "inserted vector not searchable");
    bufann_delete(*idx, 1003);
    hits = bufann_query(*idx, vectors[3].data(), 4, 16);
    require(std::find(hits.begin(), hits.end(), 1003) == hits.end(), "deleted tag remains visible");
    bufann_cleanup_deleted_edges(*idx, 2);
    require(store->pq_codes_for_node(node_ids[3]) == nullptr, "reclaimed PQ slot is still ready");
    std::array<T, 8> replacement;
    replacement.fill(static_cast<T>(91));
    idx->pq_table.deflate_vec(replacement.data(), expected[3].data());
    uint32_t reused = INVALID_NODE;
    bufann_insert(*idx, 9000, replacement.data(), 8, &reused);
    // Whether the freed slot is reused depends on claim_page's policy, so only check that the insert succeeded.
    require(reused != INVALID_NODE, "replacement insert failed");
    node_ids[3] = reused;
    require(std::equal(expected[3].begin(), expected[3].end(), store->pq_codes_for_node(reused)), "reused slot has stale PQ code");
    store->flush();
    store->save_snapshot(prefix + ".meta");
    store->save_dense_pq_codes(prefix);
    store.reset();
    idx.reset();
    idx = make_index(false);
    idx->medoids = {node_ids[0]};
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto* code = idx->store->pq_codes_for_node(node_ids[i]);
        require(code && std::equal(expected[i].begin(), expected[i].end(), code), "PQ reload mismatch");
    }
    hits = bufann_query(*idx, replacement.data(), 4, 16);
    require(std::find(hits.begin(), hits.end(), 9000) != hits.end(), "replacement lost after reload");
    idx.reset();
    std::filesystem::remove(prefix + "_pq_compressed.bin");
    bool rejected = false;
    try {
        std::unique_ptr<BufANNIndex<T>> missing(bufann_load<T>(prefix, config));
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("PQ requested") != std::string::npos;
    }
    require(rejected, "missing PQ files silently disabled PQ");

}

static void check_buffer_pool_reinitialization(const std::string& path) {
    auto files = std::make_shared<FileManager>(4096);
    const FileNo file = files->register_file(path, true);
    BufferPool pool;
    pool.init(4096, 8, files, file, nullptr);
    pool.grow_heap(file, 1);
    {
        auto guard = pool.pin({file, 0}, AccessMode::WRITE);
        guard.data()[0] = 42;
        guard.mark_dirty();
    }
    pool.flush_all_dirty();
    // Reinitialization releases old frames and discards their lookup entries.
    pool.init(4096, 4, files, file, nullptr);
    auto guard = pool.pin({file, 0});
    require(guard.data()[0] == 42, "buffer pool reinitialization lost persisted data");
}

int main() {
    char directory[] = "/tmp/bufann-pq-memory-XXXXXX";
    require(mkdtemp(directory) != nullptr, "mkdtemp failed");
    try {
        check_buffer_pool_reinitialization(std::string(directory) + "/pool.heap");
        run_case<float>(std::string(directory) + "/float");
        run_case<int8_t>(std::string(directory) + "/int8");
        run_case<uint8_t>(std::string(directory) + "/uint8");
        std::filesystem::remove_all(directory);
        std::cout << "PQ memory lifecycle checks passed for float, int8, uint8\n";
    } catch (const ANNException& error) {
        std::cerr << error.message() << " (fixture: " << directory << ")\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << " (fixture: " << directory << ")\n";
        return 1;
    }
}
