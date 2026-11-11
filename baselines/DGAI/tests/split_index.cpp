#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define SECTOR_LEN 4096ULL
#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))
#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

template <typename T>
int split_index(const char* disk_index_file_path, const char* topo_file_path, const char* topo_file_aligned_path, const char* coord_file_aligned_path) {
    std::ifstream reader(disk_index_file_path, std::ios::binary);
    if (!reader) {
        std::cerr << "Failed to open: " << disk_index_file_path << std::endl;
        return 1;
    }

    // TODO : 把npts ndim 写道meta
    int meta_npts, meta_ndims;
    reader.read((char *)&meta_npts, sizeof(int));
    reader.read((char *)&meta_ndims, sizeof(int));

    std::vector<uint64_t> meta(meta_npts);
    for (auto &x : meta) {
        reader.read((char *)&x, sizeof(uint64_t));
    }

    uint64_t npts = meta[0];
    uint64_t ndims = meta[1];
    uint64_t max_node_len = meta[3];
    uint64_t nnodes_per_sector = meta[4];
    uint64_t nsector_per_node = DIV_ROUND_UP(max_node_len, SECTOR_LEN);

    uint64_t max_node_degree = (max_node_len - ndims * sizeof(T) - sizeof(uint32_t)) / sizeof(uint32_t);
    uint64_t topo_len = (max_node_degree + 1) * sizeof(uint32_t);
    uint64_t ssd_node_len = ndims * sizeof(T);

    uint64_t n_sectors = nnodes_per_sector > 0 ? ROUND_UP(npts, nnodes_per_sector) / nnodes_per_sector : nsector_per_node * npts;

    uint64_t ssd_nodes_per_sector = SECTOR_LEN / ssd_node_len;
    uint64_t ssd_nsector_per_node = SECTOR_LEN / ssd_node_len;
    uint64_t ssd_n_sectors = DIV_ROUND_UP(npts, ssd_nodes_per_sector);
    uint64_t ssd_size = ssd_n_sectors * SECTOR_LEN;
    size_t topo_size = npts * topo_len;
    auto simulate_topo_size_aligned = [&](size_t npts, size_t topo_len) {
        size_t offset = 0;
        for (size_t i = 0; i < npts; ++i) {
            size_t sector_offset = offset % SECTOR_LEN;
            if (sector_offset + topo_len > SECTOR_LEN) {
                // 跨sector，跳到下一个sector起点
                offset += SECTOR_LEN - sector_offset;
                sector_offset = 0;
            }
            offset += topo_len;
        }
        return offset;
    };

    size_t topo_size_aligned = ROUND_UP(simulate_topo_size_aligned(npts, topo_len), SECTOR_LEN);


    std::cerr << "npts: " << npts << " ndims: " << ndims << " max_node_len: " <<
     max_node_len << " nnodes_per_sector: " << nnodes_per_sector << " nsector_per_node: " 
     << nsector_per_node << " max_node_degree: " << max_node_degree << " topo_len: "
      << topo_len << " ssd_node_len: " << ssd_node_len << " n_sectors: " << n_sectors 
      << " ssd_nodes_per_sector: " << ssd_nodes_per_sector << " ssd_nsector_per_node: " 
      << ssd_nsector_per_node << " ssd_n_sectors: " << ssd_n_sectors << " ssd_size: " 
      << ssd_size << " topo_size: " << topo_size << " topo_size_aligned: " << topo_size_aligned << std::endl;

    int topo_fd = -1;
    // open(topo_file_path, O_RDWR | O_CREAT | O_LARGEFILE, 0666);
    // if (topo_fd == -1 || ftruncate(topo_fd, topo_size) == -1) {
    //     perror("topo file open/truncate failed");
    //     if (topo_fd != -1) close(topo_fd);
    //     return 1;
    // }

    // char* topo_addr = (char*)mmap(NULL, topo_size, PROT_READ | PROT_WRITE, MAP_SHARED, topo_fd, 0);
    // close(topo_fd);
    // if (topo_addr == MAP_FAILED) {
    //     perror("mmap failed");
    //     return 1;
    // }

    int coord_fd = open(coord_file_aligned_path, O_CREAT | O_RDWR | O_LARGEFILE, 0666);
    if (coord_fd == -1 || ftruncate(coord_fd, ssd_size) == -1) {
        perror("coord file open/truncate failed");
        if (coord_fd != -1) close(coord_fd);
        return 1;
    }

    topo_fd = open(topo_file_aligned_path, O_RDWR | O_CREAT | O_LARGEFILE, 0666);
    if (topo_fd == -1 || ftruncate(topo_fd, topo_size_aligned) == -1) {
        perror("topo file open/truncate failed");
        if (topo_fd != -1) close(topo_fd);
        return 1;
    }

    reader.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> sector_buf(new char[SECTOR_LEN * nsector_per_node]);
    reader.read(sector_buf.get(), SECTOR_LEN);  // skip first sector

    auto get_sector_offset = [&](uint64_t node_id) {
        return (node_id / ssd_nodes_per_sector) * SECTOR_LEN +
               (node_id % ssd_nodes_per_sector) * ssd_node_len;
    };

    auto get_sector_offset2 = [&](uint64_t node_id) {
        return (node_id / ssd_nsector_per_node) * SECTOR_LEN;
    };

    size_t topo_offset = 0;
    size_t topo_offset_aligned = 0;
    uint64_t node_id = 0;
    double total_deg = 0;

    if(nsector_per_node == 1){
        for (uint64_t s = 0; s < n_sectors; ++s) {
            reader.read(sector_buf.get(), SECTOR_LEN);
            for (uint32_t i = 0; i < nnodes_per_sector && node_id < npts; ++i, ++node_id) {
                char* node = sector_buf.get() + i * max_node_len;
                T* coords = reinterpret_cast<T*>(node);
                uint32_t nnbrs = *reinterpret_cast<uint32_t*>(node + ndims * sizeof(T));
                uint32_t* nbrs = reinterpret_cast<uint32_t*>(node + ndims * sizeof(T) + sizeof(uint32_t));

                if (pwrite(coord_fd, coords, ndims * sizeof(T), get_sector_offset(node_id)) == -1) {
                    perror("pwrite failed");
                    return 1;
                }

                // std::memcpy(topo_addr + topo_offset, &nnbrs, sizeof(uint32_t));
                // std::memcpy(topo_addr + topo_offset + sizeof(uint32_t), nbrs, nnbrs * sizeof(uint32_t));
                
                // 原来是 memcpy 到 topo_addr，现在改成：
                if (pwrite(topo_fd, &nnbrs, sizeof(uint32_t), topo_offset_aligned) == -1) {
                    perror("pwrite topo nnbrs failed");
                    return 1;
                }

                if (pwrite(topo_fd, nbrs, nnbrs * sizeof(uint32_t), topo_offset_aligned + sizeof(uint32_t)) == -1) {
                    perror("pwrite topo nbrs failed");
                    return 1;
                }
                // 写完当前 topo
                topo_offset_aligned += topo_len;

                // 检查下一个 topo 是否会跨 sector
                uint64_t sector_offset = topo_offset_aligned % SECTOR_LEN;
                if (sector_offset + topo_len > SECTOR_LEN) {
                    // 跳到下一个 sector
                    topo_offset_aligned = (topo_offset_aligned / SECTOR_LEN + 1) * SECTOR_LEN;
                }

                topo_offset += topo_len;
                total_deg += nnbrs;
            }
            if (s % (n_sectors / 100 + 1) == 0)
                std::cout << "\rProgress: " << (s * 100 / n_sectors) << "%" << std::flush;
        } 
    }else {
        for (uint64_t s = 0; s < n_sectors; s += nsector_per_node) {
            // Read all sectors needed for the current set of nodes
            reader.read(sector_buf.get(), SECTOR_LEN * nsector_per_node);
            
            char* node = sector_buf.get();
            T* coords = reinterpret_cast<T*>(node);
            uint32_t nnbrs = *reinterpret_cast<uint32_t*>(node + ndims * sizeof(T));
            uint32_t* nbrs = reinterpret_cast<uint32_t*>(node + ndims * sizeof(T) + sizeof(uint32_t));
    
            if (pwrite(coord_fd, coords, ndims * sizeof(T), get_sector_offset2(node_id)) == -1) {
                perror("pwrite failed");
                return 1;
            }
            
            // std::memcpy(topo_addr + topo_offset, &nnbrs, sizeof(uint32_t));
            // std::memcpy(topo_addr + topo_offset + sizeof(uint32_t), nbrs, nnbrs * sizeof(uint32_t));

            // 原来是 memcpy 到 topo_addr，现在改成：
            if (pwrite(topo_fd, &nnbrs, sizeof(uint32_t), topo_offset_aligned) == -1) {
                perror("pwrite topo nnbrs failed");
                return 1;
            }

            if (pwrite(topo_fd, nbrs, nnbrs * sizeof(uint32_t), topo_offset_aligned + sizeof(uint32_t)) == -1) {
                perror("pwrite topo nbrs failed");
                return 1;
            }
            
            // 写完当前 topo
            topo_offset_aligned += topo_len;

            // 检查下一个 topo 是否会跨 sector
            uint64_t sector_offset = topo_offset_aligned % SECTOR_LEN;
            if (sector_offset + topo_len > SECTOR_LEN) {
                // 跳到下一个 sector
                topo_offset_aligned = (topo_offset_aligned / SECTOR_LEN + 1) * SECTOR_LEN;
            }

            total_deg += nnbrs;
            node_id ++;
        
            if (s % (n_sectors / 100 + 1) == 0)
                std::cout << "\rProgress: " << (s * 100 / n_sectors) << "%" << std::flush;
        }
    }

    std::cout << "\rProgress: 100%\n";
    std::cout << "avg_degree = " << (total_deg / npts) << "\n";

    close(coord_fd);
    close(topo_fd);
    // munmap(topo_addr, topo_size);
    reader.close();
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <topo_file> <topo_file_aligned> <coord_file_aligned> <data_type>\n";
        return 1;
    }

    const char* input = argv[1];
    const char* topo = argv[2];
    const char* topo_aligned = argv[3];
    const char* coord_aligned = argv[4];
    const char* dtype = argv[5];

    int ret = 1;
    if (strcmp(dtype, "float") == 0)
        ret = split_index<float>(input, topo, topo_aligned, coord_aligned);
    else if (strcmp(dtype, "uint8") == 0)
        ret = split_index<uint8_t>(input, topo, topo_aligned, coord_aligned);
    else 
        std::cout << "Wrong data type!\n";

    std::cout << (ret ? "Split failed.\n" : "Index split successfully!\n");
    return ret;
}
