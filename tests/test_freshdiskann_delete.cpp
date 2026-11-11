#include "bufann/bufann_api.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <unistd.h>

using namespace diskann::inplace;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static std::vector<uint32_t> neighbors(InPlaceGraphStore& store, uint32_t id) {
    auto frame = store.pin_page_for_node(id, READ);
    auto guard = store.read_guard(frame);
    auto node = guard.read_node(id);
    require(node.valid() && node.node_id() == id, "invalid test node");
    return {node.neighbors(), node.neighbors() + node.degree()};
}

static std::vector<std::vector<uint32_t>> run(const std::string& data,
                                             const std::string& prefix,
                                             uint32_t threads) {
    BufANNConfig cfg;
    cfg.dim = 8;
    cfg.R = 8;
    cfg.L = 16;
    cfg.C = 32;
    cfg.pq_chunks = 1;
    cfg.build_threads = 1;
    cfg.buffer_pool_frames = 128;
    std::unique_ptr<BufANNIndex<float>> idx(threads == 1
        ? bufann_build<float>(data, prefix, cfg)
        : bufann_load<float>(prefix, cfg));
    if (threads == 1) {
        // The builder initializes the store before generating new PQ sidecars.
        idx->pq_table.load_pq_centroid_bin((prefix + "_pq_pivots.bin").c_str(), 1);
        idx->store->load_pq_codes_from_disk_index(prefix, 1);
        for (uint32_t id = 0; id < 512; ++id) {
            auto frame = idx->store->pin_page_for_node(id, READ);
            auto guard = idx->store->read_guard(frame);
            idx->store->encode_pq(id, guard.read_node(id).coords<float>(), idx->pq_table);
        }
        bufann_flush_dirty(*idx);
        idx->store->save_snapshot(prefix + ".meta");
        // Both worker counts start with identical graph and PQ artifacts.
        const auto root = std::filesystem::path(prefix).parent_path();
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("serial", 0) == 0) {
                std::filesystem::copy_file(entry.path(), root / ("parallel" + name.substr(6)));
                std::filesystem::copy_file(entry.path(), root / ("normal" + name.substr(6)));
            }
        }
    }
    // Deliberately include deleted chains and self/duplicate bypass candidates.
    std::vector<std::vector<uint32_t>> graph(512);
    graph[0] = {1};
    graph[1] = {2, 4, 0, 4};
    graph[2] = {5};
    graph[3] = {2};
    graph[4] = {5, 6};
    graph[6] = {9};
    graph[7] = {1, 2, 4};
    graph[9] = {2};
    graph[64] = {1};
    graph[128] = {2};
    graph[192] = {1, 2};
    graph[256] = {9};
    for (uint32_t id = 0; id < graph.size(); ++id) {
        auto frame = idx->store->pin_page_for_node(id, WRITE);
        auto guard = idx->store->write_guard(frame);
        guard.write_node(id).set_neighbors(graph[id].data(), graph[id].size());
    }
    for (uint32_t id : {1u, 2u, 9u, 1u, 9999u}) bufann_delete(*idx, id);
    bufann_cleanup_deleted_edges(*idx, threads);
    require(idx->store->num_active() == 509, "wrong active count");
    require(neighbors(*idx->store, 0) == std::vector<uint32_t>{4}, "wrong bypass");
    require(neighbors(*idx->store, 3) == std::vector<uint32_t>{5}, "wrong shared bypass");
    require(neighbors(*idx->store, 6).empty(), "deleted chain recursively expanded");
    require(neighbors(*idx->store, 4) == graph[4], "unaffected list changed");
    std::vector<std::vector<uint32_t>> result(512);
    for (uint32_t id = 0; id < 512; ++id) {
        if (!idx->store->is_active(id)) continue;
        result[id] = neighbors(*idx->store, id);
        std::set<uint32_t> unique;
        for (uint32_t n : result[id]) {
            require(n != id && idx->store->is_active(n), "dead or self edge");
            require(unique.insert(n).second, "duplicate edge");
        }
        require(result[id].size() <= cfg.R, "degree overflow");
    }
    bufann_cleanup_deleted_edges(*idx, threads);
    require(idx->store->num_active() == 509, "repeat cleanup changed count");
    const uint32_t reused = idx->store->reserve_node_id();
    require(reused == 1 || reused == 2 || reused == 9, "deleted ID not reclaimed");
    idx->store->allocate_node(reused);
    // Delete through a tag: cleanup only drains the tag set.
    constexpr TagType kReusedTag = 100000;
    require(idx->store->set_node_tag(reused, kReusedTag), "could not tag reused ID");
    idx->store->publish_node(reused);
    bufann_delete(*idx, kReusedTag);
    bufann_cleanup_deleted_edges(*idx, threads);
    bufann_flush_dirty(*idx);
    idx->store->save_snapshot(prefix + ".meta");
    idx.reset();
    idx.reset(bufann_load<float>(prefix, cfg));
    require(neighbors(*idx->store, 0) == result[0], "repair did not persist");
    for (uint32_t id = 0; id < 512; ++id) {
        if (idx->store->is_active(id)) idx->store->mark_tag_deleted(idx->store->node_tag(id));
    }
    bufann_cleanup_deleted_edges(*idx, threads);
    require(idx->store->num_active() == 0, "all-node deletion failed");
    return result;
}

int main() {
    char temp[] = "/tmp/bufann-fresh-delete-XXXXXX";
    const char* dir = mkdtemp(temp);
    if (!dir) return 1;
    try {
        const std::string root(dir);
        const std::string data = root + "/data.bin";
        {
            std::ofstream out(data, std::ios::binary);
            const uint32_t header[] = {512, 8};
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-10, 10);
            for (size_t i = 0; i < 512 * 8; ++i) {
                const float v = dist(rng);
                out.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }
        const auto serial = run(data, root + "/serial", 1);
        const auto parallel = run(data, root + "/parallel", 4);
        require(serial == parallel, "serial/parallel results differ");
        std::filesystem::remove_all(root);
        std::cout << "FreshDiskANN deletion tests passed\n";
        return 0;
    } catch (const diskann::ANNException& e) {
        std::cerr << e.message() << " (artifacts: " << dir << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << " (artifacts: " << dir << ")\n";
        return 1;
    }
}
