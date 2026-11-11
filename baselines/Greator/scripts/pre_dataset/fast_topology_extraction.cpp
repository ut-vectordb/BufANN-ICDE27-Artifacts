// Fast Greator topology sidecar extractor.
//
// Drop-in replacement for topology_extraction.cpp with the same CLI:
//   <indir> <outdir> <SECTOR_LEN> <range> [elem_size=4]
//
// Strategy: O_DIRECT + posix_memalign'd buffers + large pread/pwrite batches.
// No per-node heap allocations, no per-sector seek. The output layout is
// identical to the old binary (sector 0 = [npts u64][range u64] header,
// sectors 1.. = packed [nnbrs u32][nbrs[range] u32] slots, zero-padded).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

static constexpr uint64_t TARGET_BATCH_BYTES = 32ULL * 1024ULL * 1024ULL;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <indir> <outdir> <SECTOR_LEN> <range> [elem_size=4]\n";
        return 1;
    }
    const std::string indir   = argv[1];
    const std::string outdir  = argv[2];
    const uint64_t SECTOR_LEN = std::strtoull(argv[3], nullptr, 10);
    const uint32_t range      = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
    const uint64_t elem_size  = (argc >= 6) ? std::strtoull(argv[5], nullptr, 10)
                                            : sizeof(float);

    if (SECTOR_LEN == 0 || (SECTOR_LEN % 4096) != 0) {
        std::cerr << "SECTOR_LEN must be a multiple of 4096\n";
        return 1;
    }

    // Open input. Try O_DIRECT; fall back to buffered.
    int src_fd = open(indir.c_str(), O_RDONLY | O_DIRECT);
    if (src_fd < 0) {
        src_fd = open(indir.c_str(), O_RDONLY);
        if (src_fd < 0) { perror(indir.c_str()); return 1; }
        std::cout << "note: O_DIRECT read not available, using buffered\n";
    }

    // Read sector 0 (header) to learn npts/dim.
    void* hdr_raw = nullptr;
    if (posix_memalign(&hdr_raw, 4096, SECTOR_LEN) != 0) {
        std::cerr << "posix_memalign failed\n"; return 1;
    }
    char* hdr_buf = static_cast<char*>(hdr_raw);
    if (pread(src_fd, hdr_buf, SECTOR_LEN, 0) != static_cast<ssize_t>(SECTOR_LEN)) {
        perror("pread header"); return 1;
    }
    uint64_t npts = 0, dim = 0;
    std::memcpy(&npts, hdr_buf + sizeof(uint64_t), sizeof(uint64_t));
    std::memcpy(&dim,  hdr_buf + 2 * sizeof(uint64_t), sizeof(uint64_t));

    const uint64_t size_per_node_read  = static_cast<uint64_t>(1 + range) * sizeof(uint32_t)
                                       + dim * elem_size;
    const uint64_t nnodes_per_sector_r = SECTOR_LEN / size_per_node_read;
    const uint64_t size_per_node_write = static_cast<uint64_t>(1 + range) * sizeof(uint32_t);
    const uint64_t nnodes_per_sector_w = SECTOR_LEN / size_per_node_write;
    if (nnodes_per_sector_r == 0 || nnodes_per_sector_w == 0) {
        std::cerr << "node does not fit in sector (range/dim/elem too large)\n";
        return 1;
    }

    const uint64_t in_data_sectors = (npts + nnodes_per_sector_r - 1) / nnodes_per_sector_r;
    const uint64_t out_data_sectors = (npts + nnodes_per_sector_w - 1) / nnodes_per_sector_w;
    const uint64_t out_total_bytes = (out_data_sectors + 1) * SECTOR_LEN;

    std::cout << "npts=" << npts << " dim=" << dim
              << " range=" << range << " elem=" << elem_size << "\n"
              << "in:  size_per_node=" << size_per_node_read
              <<     " nodes_per_sec=" << nnodes_per_sector_r
              <<     " sectors=" << in_data_sectors << "\n"
              << "out: size_per_node=" << size_per_node_write
              <<     " nodes_per_sec=" << nnodes_per_sector_w
              <<     " sectors=" << out_data_sectors << "\n";

    // Open output and preallocate.
    int dst_fd = open(outdir.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    bool direct_write = true;
    if (dst_fd < 0) {
        dst_fd = open(outdir.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (dst_fd < 0) { perror(outdir.c_str()); return 1; }
        direct_write = false;
        std::cout << "note: O_DIRECT write not available, using buffered\n";
    }
    if (ftruncate(dst_fd, static_cast<off_t>(out_total_bytes)) != 0) {
        perror("ftruncate"); return 1;
    }

    // Write output header.
    std::memset(hdr_buf, 0, SECTOR_LEN);
    std::memcpy(hdr_buf, &npts, sizeof(uint64_t));
    uint64_t range64 = range;
    std::memcpy(hdr_buf + sizeof(uint64_t), &range64, sizeof(uint64_t));
    if (pwrite(dst_fd, hdr_buf, SECTOR_LEN, 0) != static_cast<ssize_t>(SECTOR_LEN)) {
        perror("pwrite header"); return 1;
    }
    free(hdr_raw);

    // Buffers: large input batch, large output batch (must be sector-aligned).
    uint64_t in_sectors_per_batch  = std::max<uint64_t>(1, TARGET_BATCH_BYTES / SECTOR_LEN);
    if (in_sectors_per_batch > in_data_sectors) in_sectors_per_batch = in_data_sectors;
    const uint64_t in_buf_bytes  = in_sectors_per_batch * SECTOR_LEN;

    // Output buffer sized so we can pack nodes from one input batch before
    // having to flush, with comfortable slack (~one extra page).
    const uint64_t max_nodes_per_input_batch = in_sectors_per_batch * nnodes_per_sector_r;
    const uint64_t out_pages_capacity =
        max_nodes_per_input_batch / nnodes_per_sector_w + 2;
    const uint64_t out_buf_bytes = out_pages_capacity * SECTOR_LEN;

    void* in_raw  = nullptr;
    void* out_raw = nullptr;
    if (posix_memalign(&in_raw,  4096, in_buf_bytes)  != 0 ||
        posix_memalign(&out_raw, 4096, out_buf_bytes) != 0) {
        std::cerr << "posix_memalign failed\n"; return 1;
    }
    char* in_buf  = static_cast<char*>(in_raw);
    char* out_buf = static_cast<char*>(out_raw);

    // Pack state: nodes accumulated in out_buf since the last flush.
    uint64_t nodes_in_out_buf = 0;
    // Pages already pwritten beyond the header.
    uint64_t out_pages_written = 0;
    uint64_t bytes_out = 0;

    auto pack_node = [&](const char* node_hdr_ptr) {
        const uint64_t page_idx = nodes_in_out_buf / nnodes_per_sector_w;
        const uint64_t slot_idx = nodes_in_out_buf % nnodes_per_sector_w;
        char* page = out_buf + page_idx * SECTOR_LEN;
        if (slot_idx == 0) {
            // First slot in a fresh page -> zero whole page so trailing slack
            // and any unused tail slots are guaranteed zero.
            std::memset(page, 0, SECTOR_LEN);
        }
        char* slot = page + slot_idx * size_per_node_write;
        uint32_t nnbrs = 0;
        std::memcpy(&nnbrs, node_hdr_ptr, sizeof(uint32_t));
        if (nnbrs > range) nnbrs = range;
        std::memcpy(slot, &nnbrs, sizeof(uint32_t));
        std::memcpy(slot + sizeof(uint32_t),
                    node_hdr_ptr + sizeof(uint32_t),
                    static_cast<size_t>(nnbrs) * sizeof(uint32_t));
        // Bytes [4 + nnbrs*4 .. 4 + range*4) already zero from memset above.
        ++nodes_in_out_buf;
    };

    auto flush = [&](bool final_flush) -> bool {
        const uint64_t full_pages = nodes_in_out_buf / nnodes_per_sector_w;
        const uint64_t leftover   = nodes_in_out_buf % nnodes_per_sector_w;
        const uint64_t pages_to_write = full_pages + (final_flush && leftover > 0 ? 1 : 0);
        if (pages_to_write == 0) return true;
        const off_t off = static_cast<off_t>((1 + out_pages_written) * SECTOR_LEN);
        const ssize_t want = static_cast<ssize_t>(pages_to_write * SECTOR_LEN);
        const ssize_t wr = pwrite(dst_fd, out_buf, want, off);
        if (wr != want) { perror("pwrite"); return false; }
        out_pages_written += pages_to_write;
        bytes_out += static_cast<uint64_t>(wr);

        if (final_flush) {
            nodes_in_out_buf = 0;
        } else if (leftover > 0) {
            // Move the partial trailing page (already populated + zero-padded)
            // to the front so the next batch keeps filling it.
            std::memcpy(out_buf, out_buf + full_pages * SECTOR_LEN, SECTOR_LEN);
            nodes_in_out_buf = leftover;
        } else {
            nodes_in_out_buf = 0;
        }
        return true;
    };

    auto t0 = std::chrono::steady_clock::now();
    auto next_progress = t0 + std::chrono::seconds(5);
    uint64_t node_id = 0;
    uint64_t input_sector = 1; // sector 0 was the header

    while (node_id < npts) {
        const uint64_t remaining_in_sectors = in_data_sectors - (input_sector - 1);
        const uint64_t sectors_this = std::min(in_sectors_per_batch, remaining_in_sectors);
        if (sectors_this == 0) break;

        const off_t off = static_cast<off_t>(input_sector * SECTOR_LEN);
        const ssize_t want = static_cast<ssize_t>(sectors_this * SECTOR_LEN);
        const ssize_t rd = pread(src_fd, in_buf, want, off);
        if (rd < 0) { perror("pread"); return 1; }
        const uint64_t got_sectors = static_cast<uint64_t>(rd) / SECTOR_LEN;
        if (got_sectors == 0) break;

        for (uint64_t s = 0; s < got_sectors && node_id < npts; ++s) {
            const char* sec = in_buf + s * SECTOR_LEN;
            for (uint64_t j = 0; j < nnodes_per_sector_r && node_id < npts; ++j, ++node_id) {
                // Per-node layout in DiskANN: [vector (dim*elem)] [nnbrs u32]
                //                             [nbrs[nnbrs] u32]
                const char* node_hdr_ptr =
                    sec + j * size_per_node_read + dim * elem_size;
                pack_node(node_hdr_ptr);
            }
        }
        input_sector += got_sectors;
        if (!flush(false)) return 1;

        auto now = std::chrono::steady_clock::now();
        if (now >= next_progress || node_id >= npts) {
            const double sec = std::chrono::duration<double>(now - t0).count();
            const double gb = static_cast<double>(bytes_out) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "\rprogress: " << node_id << "/" << npts
                      << " (" << (100.0 * static_cast<double>(node_id)
                                  / static_cast<double>(npts)) << "%)  "
                      << gb << " GB written  "
                      << (gb / std::max(sec, 1e-9)) << " GB/s    " << std::flush;
            while (next_progress <= now) next_progress += std::chrono::seconds(5);
        }
    }

    if (!flush(true)) return 1;
    std::cout << "\n";

    if (out_pages_written != out_data_sectors) {
        std::cerr << "WARNING: page count mismatch (wrote " << out_pages_written
                  << " expected " << out_data_sectors << ")\n";
    }

    close(src_fd);
    close(dst_fd);
    free(in_raw);
    free(out_raw);

    auto t1 = std::chrono::steady_clock::now();
    const double total_sec = std::chrono::duration<double>(t1 - t0).count();
    const double total_gb = static_cast<double>(bytes_out) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "fast_topology_extraction: wrote " << total_gb << " GB in "
              << total_sec << "s = " << (total_gb / total_sec) << " GB/s"
              << " (direct_write=" << (direct_write ? "yes" : "no") << ")\n";
    return 0;
}
