#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
// #include <libpmemobj.h>
#include <unordered_map>
#include <unordered_set>
#include <sys/mman.h>  // for mmap, msync, munmap, PROT_READ, PROT_WRITE, MAP_SHARED, MS_SYNC
#include <assert.h>
#include <algorithm>
#include <random>
#include <chrono> // 这里是关键

#define SECTOR_LEN 4096ULL

#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))
#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

int main(int argc, char** argv) {
    
    std::string reroder_topo_path = "/gauss_yusong/ljh_data/freshdiskann_output/deep100m/reordered_disk_index_graph";
    // std::string reroder_topo_path = "/gauss_yusong/ljh_data/freshdiskann_output/deep100m/disk_init_pq32/disk_index_graph";
    std::string reorder_map_file_path = "/gauss_yusong/ljh_data/freshdiskann_output/deep100m/disk_init_pq32/reorder_map_graph";

    std::ifstream map_reader(reorder_map_file_path.c_str(), std::ios::binary);
    std::vector<uint32_t> loc2phy_topo;
    if (!map_reader) {
        std::cerr << "Failed to open map file for reading: " << reorder_map_file_path;
        exit(-1);
    }

    uint64_t size;
    map_reader.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
    loc2phy_topo.resize(size);
    // this->phy2loc_topo.resize(size * 2);
    
    map_reader.read(reinterpret_cast<char*>(loc2phy_topo.data()), size * sizeof(uint32_t));

    map_reader.close();
    std::cerr << "size is " << size << std::endl;
    int topo_fd = open(reroder_topo_path.c_str(), O_RDONLY);
    if (topo_fd < 0) {
        std::cerr << "Failed to open topo file for reading: " << reroder_topo_path;
        exit(-1);
    }
    char * buf = (char *)malloc(SECTOR_LEN);
    for (uint64_t i = 0; i < 2580646; i ++){
        uint64_t offset = i * SECTOR_LEN;
        pread(topo_fd, (void *)buf, SECTOR_LEN, offset);
        for(int  k = 0; k < 4096; k += 132){
            unsigned nnbrs = *(unsigned *)(buf + k);
            if(nnbrs > 32){
                std::cout << nnbrs << " " << i << " " << k << std::endl;
                exit(-1);
            }
        }
        // for(uint32_t j = 0; j < 31; j ++){
        //     unsigned nnbrs = *(unsigned *)(buf + j * 132);
        // }
    }
    return 0;
}