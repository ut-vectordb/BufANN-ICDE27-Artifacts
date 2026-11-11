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

// template <typename T>
// void reorder(int data_len, const char * data_file_path, const char * reordered_data_file_path, const char * reorder_map_file_path, const char * ranked_edge_list_path) {
// void reorder(std::string work_folder, uint32_t npts, uint32_t ndims){
void reorder(uint64_t npts, uint64_t ndims, std::string work_folder){

    if(work_folder.back() != '/'){
        work_folder += '/';
    }
    // uint32_t npts = 5e5;
    // uint32_t ndims = 32;
    uint32_t nbytes_per_vector = (ndims + 1) * sizeof(uint32_t);
    // uint32_t nbytes_per_vector = 28 * sizeof(float);
    // std::string work_folder = "/gauss_yusong/ljh_data/freshdiskann_output/sift100w/disk_init_pq64/";
    // std::string work_folder = "/gauss_yusong/ljh_data/freshdiskann_output/gist/disk_init_pq64/";
    std::string ranked_edges_file_path = work_folder + "ranked_edges";
    // std::string coord_path = work_folder + "disk_index_data";
    std::string disk_topo_path = work_folder + "disk_index_graph";
    std::string reorder_map_file_path = work_folder + "reorder_map_graph";
    // std::string reordered_data_file_path = work_folder + "reordered_disk_index_data";
    std::string reordered_topo_file_path = work_folder + "reordered_disk_index_graph";

    alignas(SECTOR_LEN) uint8_t single_sector_buffer[SECTOR_LEN];
    
    uint64_t nvectors_per_sector, sector_offset, n_sectors;
    nvectors_per_sector = SECTOR_LEN / nbytes_per_vector;
    
    n_sectors = (npts + nvectors_per_sector - 1) / nvectors_per_sector;
    
	auto get_sector_id = [&](uint64_t node_id) -> uint64_t { 
        // return node_id >> sector_offset; 
        return node_id / nvectors_per_sector;
	};
    
	auto get_sector_offset = [&](uint64_t node_id) -> uint64_t {
        // return node_id & ((1u << sector_offset) - 1);
        return node_id % nvectors_per_sector;
	};
    
    std::cerr << "npts " << npts << "\nn_sectors " << n_sectors << "\n";
    std::cerr << "nbytes_per_vector " << nbytes_per_vector << "\nnvectors_per_sector " << nvectors_per_sector << "\nsector_offset " << sector_offset << "\n";
    // n_sectors ++;
    //根据排序好的边进行reorder，处理gist数据集可能有问题
    // TODO ：map的格式还没有修改，如果sector能够对齐盘块就是没问题，不然会出问题
    
    std::cout << "Open file:" << ranked_edges_file_path << std::endl;
    std::ifstream edge_reader(ranked_edges_file_path, std::ios::binary);
    if (!edge_reader) {
        std::cerr << "Failed open: " << ranked_edges_file_path << std::endl;
        exit(0) ;
    }
    size_t edge_count;
    edge_reader.read(reinterpret_cast<char*>(&edge_count), sizeof(size_t));  // 先读入边数量
    std::cout << "Ranked edges count is " << edge_count << "\n";

    std::vector<int64_t> sector_id(npts, -1);
    std::vector<std::vector<uint32_t>> sector_node_id(n_sectors);
    std::unordered_set<uint64_t> unused_sector;

    // exit(0);
    uint64_t used_sector_idx = 0;
    uint64_t used_node_idx = 0;
    uint64_t cnt[5] = {};
    double sum_w = 0;
    uint64_t cnt_w = 0;
    for(uint64_t i = 0 ; i < n_sectors; i ++){
        unused_sector.insert(i);
    }
    for(uint64_t i = 0; i < edge_count; i++) {
        float w;
        uint32_t u, v;
        edge_reader.read(reinterpret_cast<char*>(&w), sizeof(float));  // 写入数据
        edge_reader.read(reinterpret_cast<char*>(&u), sizeof(uint32_t));  // 写入数据
        edge_reader.read(reinterpret_cast<char*>(&v), sizeof(uint32_t));  // 写入数据
        // std::cout << "[Edge " << i << "] Read: w = " << w << ", u = " << u << ", v = " << v << "\n";

        if(u == v){
            continue;
        }
        // break;
        if(sector_id[u] == -1 && sector_id[v] == -1) {//u和v都没有被分配，尝试开一个新sector
            // std::cout << "  -> Both unassigned. Assigning new sector " << used_sector_idx << " to u and v.\n";
            if(unused_sector.size() > 0 && nvectors_per_sector > 1){//有空的sector
                
                uint64_t unused_sector_id = *unused_sector.begin();
                unused_sector.erase(unused_sector.begin());
                sector_id[u] = unused_sector_id;
                sector_id[v] = unused_sector_id;
                sector_node_id[unused_sector_id].push_back(u);
                sector_node_id[unused_sector_id].push_back(v);
                
                used_node_idx += 2;
                sum_w += w; cnt_w ++;
                cnt[0] ++;
            }else{
                cnt[2] ++;

            }
        }else if(sector_id[u] == -1 && sector_id[v] != -1) {//u没有被分配，v被分配
            uint64_t vid = sector_id[v];
            if(sector_node_id[vid].size() < nvectors_per_sector){
                cnt[1] ++;
                
                sum_w += w; cnt_w ++;

                sector_id[u] = vid;
                sector_node_id[vid].push_back(u);
                used_node_idx ++;
            }else {
                cnt[2] ++;

            }
        }else if(sector_id[u] != -1 && sector_id[v] == -1) {//u被分配，v没有被分配
            uint64_t uid = sector_id[u];
            if(sector_node_id[uid].size() < nvectors_per_sector){
                cnt[1] ++;
                sum_w += w; cnt_w ++;
                sector_id[v] = uid;
                sector_node_id[uid].push_back(v);
                used_node_idx ++;
            }else{
                cnt[2] ++;
            }
        }else if(sector_id[u] != -1 && sector_id[v] != -1) {//u和v都被分配
            // std::cout << "  -> Both assigned. u in sector " << sector_id[u] << ", v in sector " << sector_id[v] << ". Skipping.\n";
            
            uint64_t uid = sector_id[u];
            uint64_t vid = sector_id[v];
            if(uid == vid) continue;
            size_t usz = sector_node_id[uid].size();
            size_t vsz = sector_node_id[vid].size();
            if(usz + vsz <= nvectors_per_sector){
                sector_node_id[uid].insert(
                    sector_node_id[uid].end(),
                    sector_node_id[vid].begin(),
                    sector_node_id[vid].end()
                );
                for (uint32_t node : sector_node_id[vid]) {
                    sector_id[node] = uid;
                }
                sector_node_id[vid].clear();
                unused_sector.insert(vid);
            
                sum_w += w; 
                cnt_w++;
                cnt[3]++;
            
            }else{
                cnt[4] ++;
                continue;
            }
        }
        
        if(i % (edge_count/1000) == 0){
            std::cout << "\rProgress: " << (i*1000/edge_count) << "%" << std::flush;
        }
        if(used_node_idx > npts)
            break;
        // std::cout << "w = " << w << ", u = " << u << ", v = " << v << "\n";
        // if(i > 20 ) break;
    }
    std::vector<uint32_t> unassigned_node_id;
    for(uint64_t i = 0; i < npts; i ++ ){
        if(sector_id[i] == -1){
            unassigned_node_id.push_back(i);
        }
    }
    
    for(uint64_t i = 0; i < 5; i ++ ){
        std::cout << "cnt[" << i << "] = " << cnt[i] << "\n";
    }
    // std::cout << "used_sector_idx = " << used_sector_idx << "\n";
    std::cout << "used_node_idx = " << used_node_idx << "\n";
    std::cout << "avg weight is " << sum_w / cnt_w << "\n";
    // uint64_t cnt1 = 0, cnt2 = 0;
    // for(uint64_t i = 0; i < n_sectors; i ++){
    //     cnt1 += sector_node_id[i].size();
    //     cnt2 += 31 - sector_node_id[i].size();
    // }
    // std::cerr << "cnt1 = " << cnt1 << "\n";
    // std::cerr << "cnt2 = " << cnt2 << "\n";
    // return ;
    uint64_t empty_cnt = 0;
    for(uint64_t i = 0; i < n_sectors; i ++ ){
        while(unassigned_node_id.size() && sector_node_id[i].size() < nvectors_per_sector){
            sector_node_id[i].push_back(unassigned_node_id.back());
            sector_id[unassigned_node_id.back()] = i;
            unassigned_node_id.pop_back();
            used_node_idx ++;
        }
        empty_cnt += nvectors_per_sector - sector_node_id[i].size();
        // if(sector_node_id[i].size() != nvectors_per_sector && i != n_sectors - 1){
        //     std::cout << i << " sector's size is " << sector_node_id[i].size() << ", but is not " << nvectors_per_sector << ".\n";
        //     exit(1);
        // }
    }

    if(empty_cnt > 0){
        std::cout << "empty_cnt is " << empty_cnt << "\n";
        if(empty_cnt != nvectors_per_sector * n_sectors - npts){
            std::cout << "empty_cnt is not equal to nvectors_per_sector * n_sectors - npts.\n";
            exit(1);
        }
    }
    if(unassigned_node_id.size() > 0){
        std::cout << "unassigned_node_id.size() is " << unassigned_node_id.size() << "\n";
        exit(1);
    }
    if(used_node_idx != npts){
        std::cout << "used_node_idx is " << used_node_idx << ", but is not " << npts << ".\n";
        exit(1);
    }

    edge_reader.close();

    
    // 2. 生成索引数组
    std::vector<uint32_t> indices, inv_indices;
    indices.resize(npts);
    inv_indices.resize(npts);
    std::vector<size_t> sector_indices(n_sectors);
    for (size_t i = 0; i < npts; i++) {
        indices[i] = i;
    }
    for (size_t i = 0; i < n_sectors; i++) {
        sector_indices[i] = i;
    }

    // 3. 按 cloest_center 排序索引
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return sector_id[a] < sector_id[b];
    });

    std::random_device rd;
    std::mt19937 g(rd());
    // // 打乱 indices
    // std::shuffle(indices.begin(), indices.end(), g);// 打乱 indices，用随机分布
    // std::shuffle(sector_indices.begin(), sector_indices.end(), g);
    
    for (size_t i = 0; i < npts; i++) {
        inv_indices[indices[i]] = i;
    }

    for (size_t i = 0; i < npts; i++) {
        size_t old_id = i;
        size_t new_id = inv_indices[old_id];
        size_t sector_id = new_id / nvectors_per_sector;
        size_t sector_offset = new_id % nvectors_per_sector;
        size_t new_new_id = sector_indices[sector_id] * nvectors_per_sector + sector_offset;
        inv_indices[old_id] = new_new_id;
        // std::cout << old_id << " " << new_new_id << " " <<sector_indices[sector_id] << " " << sector_offset << "\n";
        // if(i > 10) exit(0);
    }

    // std::unique_ptr<char[]> sector_buf(new char[SECTOR_LEN]);
    uint64_t cur_node_id = 0;
    uint64_t ssd_size = n_sectors * SECTOR_LEN;

    std::cerr << "n_sectors is " << n_sectors << "\n";
    // 打开原始数据文件（只读）
    int orig_fd = open(disk_topo_path.c_str(), O_RDONLY);
    if (orig_fd == -1) {
        throw std::runtime_error("Failed to open original data file.");
    }

    // 创建/打开目标文件（重排后）
    int data_fd = open(reordered_topo_file_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (data_fd == -1) {
        close(orig_fd);
        throw std::runtime_error("Failed to open reordered data file.");
    }

    if (ftruncate(data_fd, ssd_size) == -1) {
        close(orig_fd);
        close(data_fd);
        throw std::runtime_error("Failed to truncate reordered data file.");
    }

    for (uint64_t sector = 0; sector < n_sectors; ++sector) {
        off_t read_offset = sector * SECTOR_LEN;
        ssize_t r = pread(orig_fd, single_sector_buffer, SECTOR_LEN, read_offset);
        if (r != (ssize_t)SECTOR_LEN) {
            close(orig_fd);
            close(data_fd);
            throw std::runtime_error("Failed to read sector from original data.");
        }

        for (uint32_t sector_node_id = 0; sector_node_id < nvectors_per_sector && cur_node_id < npts; ++sector_node_id, ++cur_node_id) {
            char* node_start = (char *)single_sector_buffer + (sector_node_id * nbytes_per_vector);
            uint64_t new_node_id = inv_indices[cur_node_id];
            uint64_t new_offset = (new_node_id / nvectors_per_sector) * SECTOR_LEN +
                                (new_node_id % nvectors_per_sector) * nbytes_per_vector;

            ssize_t w = pwrite(data_fd, node_start, nbytes_per_vector, new_offset);
            if (w != (ssize_t)nbytes_per_vector) {
                close(orig_fd);
                close(data_fd);
                throw std::runtime_error("Failed to write reordered vector to file.");
            }
        }
        
        if(sector % (n_sectors/1000) == 0){
            std::cout << "\rProgress: " << (sector*1000/n_sectors) << "%" << std::flush;
        }
    }

    close(orig_fd);
    close(data_fd);
    
    //把reorder
    std::ofstream reorder_map_writer(reorder_map_file_path.c_str(), std::ios::binary);
    
    if (!reorder_map_writer) {
        std::cerr << "Failed to open file for writing: " << reorder_map_file_path << std::endl;
        return;
    }
    
    for(uint32_t i = 0; i < inv_indices.size(); i ++){
        uint32_t sector_id = get_sector_id(inv_indices[i]);
        uint32_t offset = get_sector_offset(inv_indices[i]);

        // inv_indices[i] = (sector_id << sector_offset) | offset;
        inv_indices[i] = sector_id * nvectors_per_sector + offset;
    }
    size_t size = inv_indices.size();
    reorder_map_writer.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));
    reorder_map_writer.write(reinterpret_cast<const char*>(inv_indices.data()), size * sizeof(uint32_t));

    reorder_map_writer.close();

    std::cout << "Reorder done!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <npts> <ndims> <work_folder>" << std::endl;
        return 1;
    }

    // 解析命令行参数
    uint64_t npts = std::stoul(argv[1]);  // 字符串转无符号整数
    uint64_t ndims = std::stoul(argv[2]);
    std::string work_folder = argv[3];

    // 调用 reorder 函数
    reorder(npts, ndims, work_folder);
    std::cout << "Reorder successfully!" << std::endl;

    return 0;
}