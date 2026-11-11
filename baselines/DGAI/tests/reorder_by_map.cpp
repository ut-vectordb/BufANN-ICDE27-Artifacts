#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

constexpr uint32_t SECTOR_LEN = 4096;

void reorder(uint64_t npts,
             uint32_t  ndims,
             const std::string& disk_index_path,
             const std::string& reorder_map_path,
             const std::string& reordered_index_path)
{
    const uint32_t nbytes_per_vector = ndims;
    const uint32_t nvectors_per_sector = SECTOR_LEN / nbytes_per_vector;
    const uint64_t n_sectors = (npts + nvectors_per_sector - 1) / nvectors_per_sector;
    const uint64_t total_bytes = n_sectors * SECTOR_LEN;

    /* ---------- 1. 读映射文件 ---------- */
    std::ifstream map_in(reorder_map_path, std::ios::binary);
    if (!map_in) throw std::runtime_error("cannot open reorder map");

    uint64_t map_n = 0;
    map_in.read(reinterpret_cast<char*>(&map_n), sizeof(map_n));
    if (map_n != npts) {
        throw std::runtime_error("map_n != npts");
    }

    std::vector<uint32_t> map(npts);
    map_in.read(reinterpret_cast<char*>(map.data()), npts * sizeof(uint32_t));
    map_in.close();

    /* ---------- 2. 打开原始 / 目标文件 ---------- */
    int src_fd = open(disk_index_path.c_str(), O_RDONLY);
    int dst_fd = open(reordered_index_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (src_fd < 0 || dst_fd < 0) throw std::runtime_error("open file failed");
    // if (ftruncate(dst_fd, total_bytes) < 0) throw std::runtime_error("ftruncate failed");

    /* ---------- 3. 按映射重排 ---------- */
    alignas(SECTOR_LEN) uint8_t sec_buf[SECTOR_LEN];

    for (uint64_t id = 0; id < npts; ++id) {
        uint64_t loc = map[id];              // 新下标
        uint64_t old_offset = id / nvectors_per_sector * SECTOR_LEN + id % nvectors_per_sector * nbytes_per_vector;
        uint64_t new_offset = loc / nvectors_per_sector * SECTOR_LEN + loc % nvectors_per_sector * nbytes_per_vector;

        ssize_t r = pread(src_fd, sec_buf, nbytes_per_vector, old_offset);
        if (r != nbytes_per_vector) throw std::runtime_error("pread failed");

        ssize_t w = pwrite(dst_fd, sec_buf, nbytes_per_vector, new_offset);
        if (w != nbytes_per_vector) throw std::runtime_error("pwrite failed");

        if (id % (npts / 100 + 1) == 0)
            std::cerr << "\rReordering: " << (id * 100 / npts) << "%" << std::flush;
    }
    std::cerr << "\rReordering: 100%\n";

    // DGAI reads full sectors with O_DIRECT, including the last occupied page.
    struct stat output_stat;
    if (fstat(dst_fd, &output_stat) != 0 ||
        ftruncate(dst_fd, ((output_stat.st_size + SECTOR_LEN - 1) / SECTOR_LEN) * SECTOR_LEN) != 0)
        throw std::runtime_error("pad reordered output failed");
    close(src_fd);
    close(dst_fd);
    std::cout << "Reorder completed successfully!\n";
}


int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << "<npts> <dim> <disk_index_data> <reorder_map_file> <reorder_disk_index_data>" << std::endl;
        return 1;
    }

    uint64_t npts = atoi(argv[1]);
    uint32_t dim = atoi(argv[2]);
    if (dim * 2 >= 4096) return 0;
    const std::string disk_index_path   = argv[3];   // 原始磁盘索引文件
    const std::string reorder_map_path  = argv[4];   // 重排映射文件
    const std::string output_index_path = argv[5];   // 重排后磁盘索引文件

    reorder(npts, dim, disk_index_path, reorder_map_path, output_index_path);
    return 0;
}