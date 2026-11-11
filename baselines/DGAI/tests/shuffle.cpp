#include <cstring>
#include <ctime>
#include <random>
#include <vector>
#include <fstream>
#include "utils.h"  // pipeann::save_bin

template<class T>
void shuffle_data(const char *path) {
    int num_points, dim;
    std::ifstream reader(path, std::ios::binary);
    reader.read((char*)&num_points, sizeof(int));
    reader.read((char*)&dim, sizeof(int));

    std::vector<T> data(num_points * dim);
    reader.read((char*)data.data(), data.size() * sizeof(T));

    std::vector<int> idx(num_points);
    for (int i = 0; i < num_points; i++) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), std::mt19937_64(42));

    std::vector<T> shuffled(data.size());
    for (int i = 0; i < num_points; i++)
        std::memcpy(&shuffled[i * dim], &data[idx[i] * dim], dim * sizeof(T));

    pipeann::save_bin<T>((std::string(path) + ".shuffled").c_str(),
                         shuffled.data(), num_points, dim);
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("Usage: %s <uint8/int8/float> <file>\n", argv[0]); return -1; }
    if (!strcmp(argv[1], "uint8")) shuffle_data<uint8_t>(argv[2]);
    else if (!strcmp(argv[1], "int8")) shuffle_data<int8_t>(argv[2]);
    else if (!strcmp(argv[1], "float")) shuffle_data<float>(argv[2]);
    else { printf("Unknown type\n"); return -1; }
}
