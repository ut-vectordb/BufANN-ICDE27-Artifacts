#include <omp.h>
#include <cstring>
#include <ctime>
#include <timer.h>
#include "log.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

template<class T>
void convert(char *path) {
  T *data;
  size_t num_points, dim;
  pipeann::load_bin<T>(path, data, num_points, dim);
  pipeann::save_bin_sector_aligned<T>(std::string(path) + ".aligned", data, num_points, dim);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Correct usage: " << argv[0] << " <type[uint8/uint32]> <file> " << std::endl;
    exit(-1);
  }

  int arg_no = 1;
  char *type = argv[arg_no++];
  char *base_data_file = argv[arg_no++];

  if (strcmp(type, "uint8") == 0) {
    convert<uint8_t>(base_data_file);
  } else if (strcmp(type, "uint32") == 0) {
    convert<uint32_t>(base_data_file);
  } else {
    std::cout << "Unknown type: " << type << std::endl;
    exit(-1);
  }
}