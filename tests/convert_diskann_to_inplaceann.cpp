#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bufann/inplace_backend.h"
#include "utils.h"

namespace {

constexpr std::size_t kSectorLen = 4096;

template <typename T>
T read_scalar(std::ifstream &in) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed to read scalar");
    }
    return value;
}

struct DiskannMetadata {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint64_t npoints = 0;
    uint64_t dim = 0;
    uint64_t medoid = 0;
    uint64_t max_node_len = 0;
    uint64_t nnodes_per_sector = 0;
};

template <typename T>
void save_medoids(const std::string &path, const std::vector<T> &medoids) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write medoids: " + path);
    }
    int32_t rows = static_cast<int32_t>(medoids.size());
    int32_t cols = 1;
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char *>(medoids.data()),
              static_cast<std::streamsize>(medoids.size() * sizeof(T)));
}

template <typename T>
void collect_and_save_medoid_coords(const std::string &path,
                                    diskann::inplace::InPlaceGraphStore &store,
                                    const std::vector<uint32_t> &medoids,
                                    uint32_t aligned_dim) {
    if (medoids.empty()) {
        return;
    }

    std::vector<T> coords(medoids.size() * static_cast<std::size_t>(aligned_dim), static_cast<T>(0));
    std::vector<uint8_t> found;
    store.batch_fetch_coords(medoids, reinterpret_cast<char *>(coords.data()), found);
    for (std::size_t i = 0; i < medoids.size(); ++i) {
        if (!found[i]) {
            throw std::runtime_error("failed to read medoid coords from imported store");
        }
    }
    diskann::save_bin<T>(path, coords.data(), medoids.size(), aligned_dim);
}

DiskannMetadata load_diskann_metadata(const std::string &disk_index_path) {
    std::ifstream in(disk_index_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open DiskANN index: " + disk_index_path);
    }

    DiskannMetadata meta;
    meta.rows = read_scalar<uint32_t>(in);
    meta.cols = read_scalar<uint32_t>(in);
    meta.npoints = read_scalar<uint64_t>(in);
    meta.dim = read_scalar<uint64_t>(in);
    meta.medoid = read_scalar<uint64_t>(in);
    meta.max_node_len = read_scalar<uint64_t>(in);
    meta.nnodes_per_sector = read_scalar<uint64_t>(in);
    return meta;
}

std::vector<uint32_t> load_medoids(const std::string &prefix,
                                   const DiskannMetadata &meta) {
    const std::string medoids_path = prefix + "_disk.index_medoids.bin";
    if (!std::filesystem::is_regular_file(medoids_path)) {
        return {static_cast<uint32_t>(meta.medoid)};
    }

    uint32_t *raw = nullptr;
    std::size_t nr = 0, nc = 0;
    diskann::load_bin<uint32_t>(medoids_path, raw, nr, nc);
    std::vector<uint32_t> medoids(raw, raw + nr * nc);
    delete[] raw;
    if (medoids.empty()) {
        medoids.push_back(static_cast<uint32_t>(meta.medoid));
    }
    return medoids;
}

void copy_if_exists(const std::string &src, const std::string &dst) {
    if (std::filesystem::exists(src)) {
        std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
    }
}

template <typename T>
void import_nodes(const std::string &disk_index_path,
                  const DiskannMetadata &meta,
                  diskann::inplace::InPlaceGraphStore &store,
                  uint32_t aligned_dim) {
    using diskann::inplace::PageGuard;
    using diskann::inplace::WRITE;

    std::ifstream in(disk_index_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot reopen DiskANN index: " + disk_index_path);
    }
    in.seekg(static_cast<std::streamoff>(kSectorLen), std::ios::beg);

    std::vector<char> sector(kSectorLen, 0);
    std::vector<T> padded(aligned_dim, static_cast<T>(0));

    uint64_t node_id = 0;
    while (node_id < meta.npoints) {
        in.read(sector.data(), static_cast<std::streamsize>(sector.size()));
        if (!in) {
            throw std::runtime_error("unexpected EOF while importing DiskANN sectors");
        }

        for (uint64_t slot = 0; slot < meta.nnodes_per_sector && node_id < meta.npoints; ++slot, ++node_id) {
            const char *node_buf = sector.data() + slot * meta.max_node_len;
            std::fill(padded.begin(), padded.end(), static_cast<T>(0));
            std::memcpy(padded.data(), node_buf, meta.dim * sizeof(T));

            const auto *degree_ptr =
                reinterpret_cast<const uint32_t *>(node_buf + meta.dim * sizeof(T));
            uint32_t degree = *degree_ptr;
            const auto *nbrs =
                reinterpret_cast<const uint32_t *>(node_buf + meta.dim * sizeof(T) + sizeof(uint32_t));

            store.allocate_node(static_cast<uint32_t>(node_id));
            PageGuard frame = store.pin_page_for_node(static_cast<uint32_t>(node_id), WRITE);
            if (!frame.valid()) {
                throw std::runtime_error("pin_page_for_node failed during import");
            }
            auto guard = store.write_guard(frame);
            auto node = guard.write_node(static_cast<uint32_t>(node_id));
            if (!node.valid()) {
                throw std::runtime_error("write_node failed during import");
            }
            node.set_coords<T>(padded.data());
            node.set_neighbors(nbrs, degree);
            store.publish_node(static_cast<uint32_t>(node_id));
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_type> <diskann_index_prefix> <output_index_prefix>"
                  << " <dim> <R> <pq_chunks> <buffer_pool_frames>\n";
        return 1;
    }

    try {
        const std::string data_type = argv[1];
        const std::string src_prefix = argv[2];
        const std::string out_prefix = argv[3];
        const uint32_t dim = static_cast<uint32_t>(std::stoul(argv[4]));
        const uint32_t degree_cap = static_cast<uint32_t>(std::stoul(argv[5]));
        const uint32_t pq_chunks = static_cast<uint32_t>(std::stoul(argv[6]));
        const uint32_t buffer_pool_frames = static_cast<uint32_t>(std::stoul(argv[7]));

        const std::string disk_index_path = src_prefix + "_disk.index";
        if (!std::filesystem::is_regular_file(disk_index_path)) {
            throw std::runtime_error("missing DiskANN disk index: " + disk_index_path);
        }

        const DiskannMetadata meta = load_diskann_metadata(disk_index_path);
        if (meta.dim != dim) {
            throw std::runtime_error("dimension mismatch between CLI dim and DiskANN index");
        }

        std::filesystem::create_directories(std::filesystem::path(out_prefix).parent_path());
        const std::string heap_path = out_prefix + ".heap";
        const std::string meta_path = out_prefix + ".meta";
        const std::string medoids_path = out_prefix + ".medoids";
        const std::string centroids_path = out_prefix + ".centroids";

        if (pq_chunks > 0) {
            copy_if_exists(src_prefix + "_pq_pivots.bin", out_prefix + "_pq_pivots.bin");
            copy_if_exists(src_prefix + "_pq_compressed.bin", out_prefix + "_pq_compressed.bin");
        }

        diskann::inplace::InPlaceGraphStore store;
        store.init(dim, degree_cap, data_type == "float" ? sizeof(float) :
                                 data_type == "uint8" ? sizeof(uint8_t) :
                                 sizeof(int8_t),
                   4096, buffer_pool_frames, heap_path, 32, 100, true);
        if (pq_chunks > 0)
            store.load_pq_codes_from_disk_index(out_prefix, pq_chunks);

        const auto medoids = load_medoids(src_prefix, meta);
        const uint32_t aligned_dim = store.aligned_dim();

        if (data_type == "float") {
            import_nodes<float>(disk_index_path, meta, store, aligned_dim);
        } else if (data_type == "uint8") {
            import_nodes<uint8_t>(disk_index_path, meta, store, aligned_dim);
        } else if (data_type == "int8") {
            import_nodes<int8_t>(disk_index_path, meta, store, aligned_dim);
        } else {
            throw std::runtime_error("unsupported data type: " + data_type);
        }

        // Copy DiskANN's float `_centroids.bin` verbatim instead of dumping the
        // medoids' raw coords. DiskANN centroids are cluster means and give a
        // better seed-medoid pick than the medoids' own vectors.
        copy_if_exists(src_prefix + "_disk.index_centroids.bin", centroids_path);

        store.set_entry_point(medoids.front());
        store.seed_entry_pool_from_reservoir();
        store.flush();
        store.save_snapshot(meta_path);
        save_medoids(medoids_path, medoids);

        std::cout << "Converted DiskANN prefix " << src_prefix
                  << " -> inplaceann prefix " << out_prefix << "\n";
        std::cout << "npoints=" << meta.npoints
                  << " dim=" << meta.dim
                  << " medoid=" << medoids.front()
                  << " degree_cap=" << degree_cap
                  << " pq_chunks=" << pq_chunks << "\n";
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
