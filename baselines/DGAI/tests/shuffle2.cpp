#include <cstring>
#include <ctime>
#include <random>
#include <vector>
#include <fstream>
#include "utils.h"  // pipeann::save_bin

template<class T>
void shuffle_dimensions(const char *path) {
    int num_points, dim;
    std::ifstream reader(path, std::ios::binary);
    reader.read((char*)&num_points, sizeof(int));
    reader.read((char*)&dim, sizeof(int));

    // 读取所有数据
    std::vector<T> data(num_points * dim);
    reader.read((char*)data.data(), data.size() * sizeof(T));

    // 为维度打乱创建随机顺序（每个向量使用相同的打乱模式）
    std::vector<int> dim_order(dim);
    for (int d = 0; d < dim; d++) {
        dim_order[d] = d;
    }
    std::cout << "num_points: " << num_points << std::endl;
    std::cout << "dim: " << dim << std::endl;

    // 使用固定种子确保可重复性
    std::mt19937_64 rng(42);
    std::shuffle(dim_order.begin(), dim_order.end(), rng);

    // 为每个向量打乱维度
    std::vector<T> shuffled(data.size());
    for (int i = 0; i < num_points; i++) {
        // 当前向量在原始数据中的起始位置
        int original_start = i * dim;
        // 当前向量在打乱后数据中的起始位置
        int shuffled_start = i * dim;
        
        // 按照打乱的维度顺序重新排列
        for (int d = 0; d < dim; d++) {
            shuffled[shuffled_start + d] = data[original_start + dim_order[d]];
        }
    }

    // 保存结果
    pipeann::save_bin<T>((std::string(path) + ".dim_shuffled").c_str(),
                         shuffled.data(), num_points, dim);

    // 保存维度映射到单独的文件
    std::string map_output_path = std::string(path) + ".shuffled_map";
    std::ofstream map_writer(map_output_path, std::ios::binary);
    if (map_writer.is_open()) {
        // 先写入维度大小，再写入映射数组
        map_writer.write((char*)&dim, sizeof(int));
        map_writer.write((char*)dim_order.data(), dim * sizeof(int));
        std::cout << "维度映射已保存到: " << map_output_path << std::endl;
    } else {
        std::cerr << "无法创建映射文件: " << map_output_path << std::endl;
    } 
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("Usage: %s <uint8/int8/float> <file>\n", argv[0]); return -1; }
    if (!strcmp(argv[1], "uint8")) shuffle_dimensions<uint8_t>(argv[2]);
    else if (!strcmp(argv[1], "int8")) shuffle_dimensions<int8_t>(argv[2]);
    else if (!strcmp(argv[1], "float")) shuffle_dimensions<float>(argv[2]);
    else { printf("Unknown type\n"); return -1; }
}
