// Fast DiskANN -> InplaceANN converter. Bypasses InPlaceGraphStore's buffer
// pool and writes heap pages + snapshot directly using O_DIRECT parallel IO.
//
// Usage: same as convert_diskann_to_inplaceann
//   <data_type> <diskann_prefix> <output_prefix> <dim> <R> <pq_chunks> <buffer_pool_frames>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>


static constexpr uint32_t SECTOR_LEN = 4096;
static constexpr uint32_t PAGE_SIZE  = 4096;
static constexpr uint32_t INVALID_NODE = 0xFFFFFFFFu;
static constexpr uint32_t INVALID_PAGE = 0xFFFFFFFFu;

struct DiskannMeta {
    uint64_t npoints;
    uint64_t dim;
    uint64_t medoid;
    uint64_t max_node_len;
    uint64_t nnodes_per_sector;
};

#pragma pack(push, 1)
struct PackedSlotHeader {
    uint16_t degree;
    uint8_t  flags;
    uint8_t  _pad;
    uint32_t node_id;
};
static_assert(sizeof(PackedSlotHeader) == 8, "");

struct NodeRID {
    uint32_t page_id;
    uint16_t slot_idx;
    uint8_t  active;
    uint8_t  pq_ready;
};
static_assert(sizeof(NodeRID) == 8, "");

struct PageDir {
    uint16_t slots_per_page;
    uint16_t num_occupied;
};
#pragma pack(pop)

// SnapshotHeader must match the runtime struct layout (natural alignment)
struct SnapshotHeader {
    char     magic[8];
    uint32_t version;
    uint32_t dim;
    uint32_t aligned_dim;
    uint32_t Mmax;
    uint32_t elem_size;
    uint32_t page_size;
    uint32_t slot_size;
    uint32_t slots_per_page;
    uint32_t _reserved;
    uint32_t max_nodes;
    uint32_t n_chunks;
    uint32_t num_active;
    uint32_t total_pages;
    uint64_t rid_size;
    uint64_t active_cap;
    uint64_t page_dir_size;
    uint64_t pages_with_space_size;
    uint64_t reservoir_cursor;
    uint64_t _reserved2;
};

static uint32_t round_up8(uint32_t v) { return (v + 7) & ~7u; }

static DiskannMeta load_diskann_meta(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    uint32_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), 4);
    in.read(reinterpret_cast<char*>(&cols), 4);
    DiskannMeta m{};
    in.read(reinterpret_cast<char*>(&m.npoints), 8);
    in.read(reinterpret_cast<char*>(&m.dim), 8);
    in.read(reinterpret_cast<char*>(&m.medoid), 8);
    in.read(reinterpret_cast<char*>(&m.max_node_len), 8);
    in.read(reinterpret_cast<char*>(&m.nnodes_per_sector), 8);
    return m;
}

static std::vector<uint32_t> load_medoids(const std::string& prefix, uint64_t fallback) {
    std::string path = prefix + "_disk.index_medoids.bin";
    std::ifstream in(path, std::ios::binary);
    if (!in) return {static_cast<uint32_t>(fallback)};
    int32_t nr, nc;
    in.read(reinterpret_cast<char*>(&nr), 4);
    in.read(reinterpret_cast<char*>(&nc), 4);
    if (nr <= 0) return {static_cast<uint32_t>(fallback)};
    std::vector<uint32_t> med(nr);
    in.read(reinterpret_cast<char*>(med.data()), nr * 4);
    return med;
}

static void copy_if_exists(const std::string& src, const std::string& dst) {
    if (std::filesystem::exists(src))
        std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
}

static void write_bin_u32(const std::string& path, const uint32_t* data, size_t npts, size_t ndim) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    int32_t n = static_cast<int32_t>(npts);
    int32_t d = static_cast<int32_t>(ndim);
    out.write(reinterpret_cast<const char*>(&n), 4);
    out.write(reinterpret_cast<const char*>(&d), 4);
    out.write(reinterpret_cast<const char*>(data), npts * ndim * 4);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_type> <diskann_prefix> <output_prefix>"
                  << " <dim> <R> <pq_chunks> <buffer_pool_frames>" << std::endl;
        return 1;
    }

    const std::string data_type  = argv[1];
    const std::string src_prefix = argv[2];
    const std::string out_prefix = argv[3];
    const uint32_t dim           = std::stoul(argv[4]);
    const uint32_t Mmax          = std::stoul(argv[5]);
    const uint32_t pq_chunks     = std::stoul(argv[6]);
    const uint32_t buffer_pool_frames = std::stoul(argv[7]);

    const uint32_t elem_size = (data_type == "float") ? 4 : 1;
    const uint32_t aligned_dim = round_up8(dim);
    const uint32_t slot_size = sizeof(PackedSlotHeader) +
                               aligned_dim * elem_size +
                               Mmax * sizeof(uint32_t);

    // Compute slots_per_page (same algorithm as InPlaceGraphStore::init)
    uint32_t slots_per_page = 0;
    for (uint32_t s = 1; s <= (PAGE_SIZE - 8) / slot_size; s++) {
        uint32_t bmap = (s + 7) / 8;
        if (8 + bmap + s * slot_size <= PAGE_SIZE) slots_per_page = s;
    }
    if (slots_per_page == 0) {
        std::cerr << "slot_size " << slot_size << " exceeds page" << std::endl;
        return 1;
    }

    const uint32_t bitmap_bytes = (slots_per_page + 7) / 8;
    const uint32_t header_bytes = 8 + bitmap_bytes;

    std::string disk_index = src_prefix + "_disk.index";
    DiskannMeta meta = load_diskann_meta(disk_index);
    if (meta.dim != dim) {
        std::cerr << "dim mismatch" << std::endl;
        return 1;
    }

    const uint64_t npoints = meta.npoints;
    const uint64_t total_pages = (npoints + slots_per_page - 1) / slots_per_page;
    const uint64_t heap_bytes = total_pages * PAGE_SIZE;

    std::cout << "npoints=" << npoints << " dim=" << dim
              << " aligned_dim=" << aligned_dim
              << " slot_size=" << slot_size
              << " slots_per_page=" << slots_per_page
              << " total_pages=" << total_pages
              << " heap=" << heap_bytes / (1024*1024*1024) << "G" << std::endl;

    std::filesystem::create_directories(std::filesystem::path(out_prefix).parent_path());

    // Copy PQ files
    if (pq_chunks > 0) {
        std::cout << "Converting PQ files..." << std::endl;
        copy_if_exists(src_prefix + "_pq_pivots.bin", out_prefix + "_pq_pivots.bin");
        copy_if_exists(src_prefix + "_pq_compressed.bin", out_prefix + "_pq_compressed.bin");
    }

    // Allocate page buffer (aligned for O_DIRECT)
    // Process in large batches: read many DiskANN sectors, fill pages, write pages
    const uint32_t BATCH_PAGES = 8192; // 32MB of output pages per batch
    const uint64_t diskann_sector_size = SECTOR_LEN;
    const uint64_t nodes_per_batch = static_cast<uint64_t>(BATCH_PAGES) * slots_per_page;

    // Open DiskANN index for reading
    int src_fd = open(disk_index.c_str(), O_RDONLY | O_DIRECT);
    if (src_fd < 0) {
        // Fallback without O_DIRECT
        src_fd = open(disk_index.c_str(), O_RDONLY);
        if (src_fd < 0) { perror("open src"); return 1; }
        std::cout << "Warning: O_DIRECT not supported for read, using buffered" << std::endl;
    }

    // Open heap for writing
    std::string heap_path = out_prefix + ".heap";
    int dst_fd = open(heap_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0600);
    bool direct_write = true;
    if (dst_fd < 0) {
        dst_fd = open(heap_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (dst_fd < 0) { perror("open dst"); return 1; }
        direct_write = false;
        std::cout << "Warning: O_DIRECT not supported for write, using buffered" << std::endl;
    }

    // Pre-allocate heap
    if (ftruncate(dst_fd, static_cast<off_t>(heap_bytes)) != 0) {
        perror("ftruncate");
        return 1;
    }

    // Aligned buffers for O_DIRECT
    // +3: ceil(batch/nnps) can grow by 1 when batch starts mid-sector.
    const size_t read_buf_size = static_cast<size_t>(nodes_per_batch / meta.nnodes_per_sector + 3) * SECTOR_LEN;
    void* read_buf_raw = nullptr;
    posix_memalign(&read_buf_raw, 4096, read_buf_size);
    char* read_buf = static_cast<char*>(read_buf_raw);

    const size_t write_buf_size = static_cast<size_t>(BATCH_PAGES) * PAGE_SIZE;
    void* write_buf_raw = nullptr;
    posix_memalign(&write_buf_raw, 4096, write_buf_size);
    char* write_buf = static_cast<char*>(write_buf_raw);

    // RID map for snapshot
    std::vector<NodeRID> rid_map(npoints);

    auto t0 = std::chrono::steady_clock::now();
    uint64_t node_id = 0;
    uint64_t page_id = 0;
    uint64_t src_sector = 1; // skip first sector (header)
    uint64_t bytes_written = 0;
    size_t last_progress_len = 0;
    auto next_progress_report = t0 + std::chrono::seconds(5);

    while (node_id < npoints) {
        // How many nodes in this batch?
        uint64_t batch_nodes = std::min(nodes_per_batch, npoints - node_id);
        uint64_t batch_pages_count = (batch_nodes + slots_per_page - 1) / slots_per_page;

        // Absolute DiskANN addressing (BufANN node i == DiskANN node i).
        // Old code advanced src_sector by ceil(batch/nnps) after consuming
        // only `batch` nodes; when nodes_per_batch % nnps != 0 that skipped
        // trailing slots (turing nnps=5, nbatch=32768 → recall 0).
        // When node_id % nnps == 0 and batch % nnps == 0 (sift1b/spacev/deep
        // at default BATCH_PAGES), this matches the old sector math exactly.
        const uint64_t nnps = meta.nnodes_per_sector;
        src_sector = 1 + node_id / nnps;
        const uint64_t last_node = node_id + batch_nodes - 1;
        const uint64_t last_sector = 1 + last_node / nnps;
        const uint64_t sectors_needed = last_sector - src_sector + 1;
        uint64_t read_bytes = sectors_needed * SECTOR_LEN;

        // Read DiskANN sectors
        uint64_t src_offset = src_sector * SECTOR_LEN;
        ssize_t rd = pread(src_fd, read_buf, read_bytes, static_cast<off_t>(src_offset));
        if (rd < static_cast<ssize_t>(read_bytes)) {
            // Partial read near end of file is OK
            if (rd < 0) { perror("pread"); return 1; }
        }

        // Zero output pages
        memset(write_buf, 0, batch_pages_count * PAGE_SIZE);

        // Fill output pages
        uint64_t local_node = 0;
        for (uint64_t p = 0; p < batch_pages_count; p++) {
            char* page = write_buf + p * PAGE_SIZE;
            uint16_t spp = static_cast<uint16_t>(slots_per_page);
            memcpy(page, &spp, 2);

            uint16_t occupied = 0;
            for (uint16_t s = 0; s < slots_per_page && (node_id + local_node) < npoints; s++, local_node++) {
                uint64_t cur_node = node_id + local_node;

                // Map cur_node into the read buffer (may start mid-sector).
                const uint64_t abs_sector = 1 + cur_node / nnps;
                const uint64_t sector_idx = abs_sector - src_sector;
                const uint64_t slot_in_sector = cur_node % nnps;
                const char* diskann_node = read_buf + sector_idx * SECTOR_LEN +
                                           slot_in_sector * meta.max_node_len;

                // Parse DiskANN node: [vector (dim*elem_size)] [degree (u32)] [neighbors (degree*u32)]
                const char* vec_data = diskann_node;
                uint32_t degree = 0;
                memcpy(&degree, diskann_node + dim * elem_size, 4);
                if (degree > Mmax) degree = Mmax;
                const uint32_t* nbrs = reinterpret_cast<const uint32_t*>(
                    diskann_node + dim * elem_size + 4);

                // Set bitmap bit
                uint8_t* bmap = reinterpret_cast<uint8_t*>(page + 8);
                bmap[s / 8] |= (1u << (s % 8));

                // Write slot
                char* slot = page + header_bytes + static_cast<size_t>(s) * slot_size;
                PackedSlotHeader hdr{};
                hdr.degree = static_cast<uint16_t>(degree);
                hdr.flags = 0;
                hdr._pad = 0;
                hdr.node_id = static_cast<uint32_t>(cur_node);
                memcpy(slot, &hdr, sizeof(hdr));

                // Copy vector (zero-pad to aligned_dim)
                memcpy(slot + sizeof(PackedSlotHeader), vec_data, dim * elem_size);
                // (rest is already zeroed from memset)

                // Copy neighbors
                memcpy(slot + sizeof(PackedSlotHeader) + aligned_dim * elem_size,
                       nbrs, degree * sizeof(uint32_t));

                // Record RID
                rid_map[cur_node] = {static_cast<uint32_t>(page_id + p), s, 1, 0};
                occupied++;
            }
        }

        // Write pages
        uint64_t dst_offset = page_id * PAGE_SIZE;
        uint64_t write_bytes = batch_pages_count * PAGE_SIZE;
        ssize_t wr = pwrite(dst_fd, write_buf, write_bytes, static_cast<off_t>(dst_offset));
        if (wr != static_cast<ssize_t>(write_bytes)) {
            perror("pwrite");
            return 1;
        }

        node_id += local_node;
        page_id += batch_pages_count;
        bytes_written += write_bytes;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (now >= next_progress_report || node_id >= npoints) {
            double gb = bytes_written / (1024.0 * 1024.0 * 1024.0);
            std::ostringstream progress;
            progress << "progress: " << node_id << "/" << npoints
                     << " (" << (100.0 * node_id / npoints) << "%)"
                     << "  " << gb << " GB written"
                     << "  " << (gb / elapsed) << " GB/s";

            std::string progress_line = progress.str();
            std::cout << '\r' << progress_line;
            if (last_progress_len > progress_line.size()) {
                std::cout << std::string(last_progress_len - progress_line.size(), ' ');
            }
            std::cout.flush();
            last_progress_len = progress_line.size();
            while (next_progress_report <= now) {
                next_progress_report += std::chrono::seconds(5);
            }
        }
    }

    if (last_progress_len > 0) {
        std::cout << std::endl;
    }

    close(src_fd);
    close(dst_fd);

    auto t1 = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();
    double total_gb = bytes_written / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Heap written: " << total_gb << " GB in "
              << total_sec << "s = " << (total_gb / total_sec) << " GB/s" << std::endl;

    // Write medoids
    auto medoids = load_medoids(src_prefix, meta.medoid);
    std::string medoids_path = out_prefix + ".medoids";
    write_bin_u32(medoids_path, medoids.data(), medoids.size(), 1);

    // Copy centroids file verbatim from the DiskANN canonical
    // (`${src_prefix}_disk.index_centroids.bin`). DiskANN's centroids are
    // float-precision cluster centers (averages of each medoid's neighborhood),
    // not the medoids' own coords — using them for seed selection gives a
    // better starting medoid than reading the raw vectors from the heap.
    {
        std::string centroids_src = src_prefix + "_disk.index_centroids.bin";
        std::string centroids_dst = out_prefix + ".centroids";
        std::ifstream in(centroids_src, std::ios::binary);
        if (!in) {
            std::cerr << "WARNING: missing canonical centroids " << centroids_src
                      << " — BufANN seed selection will be degraded." << std::endl;
        } else {
            std::ofstream out(centroids_dst, std::ios::binary | std::ios::trunc);
            out << in.rdbuf();
        }
    }

    // Write snapshot (.meta)
    std::cout << "Writing snapshot..." << std::endl;
    {
        std::string meta_path = out_prefix + ".meta";
        std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);

        SnapshotHeader hdr{};
        memcpy(hdr.magic, "IPGSNP3", 8);
        hdr.version = 3;
        hdr.dim = dim;
        hdr.aligned_dim = aligned_dim;
        hdr.Mmax = Mmax;
        hdr.elem_size = elem_size;
        hdr.page_size = PAGE_SIZE;
        hdr.slot_size = slot_size;
        hdr.slots_per_page = slots_per_page;
        hdr.n_chunks = pq_chunks;
        hdr.num_active = static_cast<uint32_t>(npoints);
        hdr.total_pages = static_cast<uint32_t>(total_pages);
        hdr.rid_size = npoints;
        hdr.active_cap = npoints;
        hdr.page_dir_size = total_pages;
        hdr.pages_with_space_size = 0;
        hdr.reservoir_cursor = 0;
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        // RID map: (node_id, NodeRID) pairs
        for (uint64_t i = 0; i < npoints; i++) {
            uint32_t nid = static_cast<uint32_t>(i);
            out.write(reinterpret_cast<const char*>(&nid), 4);
            out.write(reinterpret_cast<const char*>(&rid_map[i]), sizeof(NodeRID));
        }

        // Active flags
        std::vector<uint8_t> active_flags(npoints, 1);
        out.write(reinterpret_cast<const char*>(active_flags.data()),
                  static_cast<std::streamsize>(active_flags.size()));

        // Page directory
        std::vector<PageDir> page_dir(total_pages);
        for (uint64_t p = 0; p < total_pages; p++) {
            uint64_t first_node = p * slots_per_page;
            uint64_t last_node = std::min(first_node + slots_per_page, npoints);
            page_dir[p].slots_per_page = static_cast<uint16_t>(slots_per_page);
            page_dir[p].num_occupied = static_cast<uint16_t>(last_node - first_node);
        }
        out.write(reinterpret_cast<const char*>(page_dir.data()),
                  static_cast<std::streamsize>(page_dir.size() * sizeof(PageDir)));

        // No pages_with_space (all full or last page)

        // Entry pool (10 entries, first = medoid, rest = INVALID)
        std::vector<uint32_t> entry_pool(10, INVALID_NODE);
        entry_pool[0] = medoids[0];
        out.write(reinterpret_cast<const char*>(entry_pool.data()), 10 * 4);

        // Reservoir (64 entries, fill with some nodes for search diversity)
        std::vector<uint32_t> reservoir(64, INVALID_NODE);
        for (size_t i = 0; i < std::min<size_t>(64, medoids.size()); i++)
            reservoir[i] = medoids[i];
        out.write(reinterpret_cast<const char*>(reservoir.data()), 64 * 4);
    }

    std::cout << "Converted DiskANN prefix " << src_prefix
              << " -> InplaceANN prefix " << out_prefix << std::endl;
    std::cout << "npoints=" << npoints << " dim=" << dim
              << " medoid=" << medoids[0]
              << " degree_cap=" << Mmax
              << " pq_chunks=" << pq_chunks << std::endl;
    return 0;
}
