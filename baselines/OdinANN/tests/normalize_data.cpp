#include <omp.h>
#include <cstring>
#include <ctime>
#include "utils/timer.h"
#include "utils/log.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "distance.h"

template<class T>
void convert(char *path) {
  int num_points, dim;
  std::ifstream reader(path, std::ios::binary | std::ios::ate);
  reader.seekg(0, std::ios::beg);
  reader.read((char *) &num_points, sizeof(int));
  reader.read((char *) &dim, sizeof(int));
  T *data = new T[num_points * dim];
  reader.read((char *) data, num_points * dim * sizeof(T));
  for (int i = 0; i < num_points; ++i) {
    pipeann::normalize_data(data + (i * dim), data + (i * dim), dim);
  }
  pipeann::save_bin<T>((std::string(path) + "_normalized").c_str(), data, num_points, dim);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Correct usage: " << argv[0] << " <type[uint8/float]> <file>" << std::endl;
    exit(-1);
  }

  int arg_no = 1;
  char *type = argv[arg_no++];
  char *base_data_file = argv[arg_no++];

  if (strcmp(type, "uint8") == 0) {
    convert<uint8_t>(base_data_file);
  } else if (strcmp(type, "int8") == 0) {
    convert<int8_t>(base_data_file);
  } else if (strcmp(type, "float") == 0) {
    convert<float>(base_data_file);
  } else {
    std::cout << "Unknown type: " << type << std::endl;
    exit(-1);
  }
}