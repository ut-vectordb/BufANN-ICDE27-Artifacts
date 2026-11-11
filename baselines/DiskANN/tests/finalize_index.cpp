// Pack a merged mem-index (graph) plus the raw base file into a sectorized
// _disk.index. Wraps diskann::create_disk_layout (src/aux_utils.cpp:675).
//
// Usage:
//   finalize_index <type=float|int8|uint8>
//                  <merged_mem_index> <base.bin> <out_disk_index>
//                  [npts_override]
//
// If npts_override > 0 and < the file's actual npts, only the first
// npts_override points of <base.bin> are sectorized into the _disk.index.
// This lets a 0.9B build run directly off the full 1B base file without
// materializing a slice copy.
//
// Note: this driver assumes single_file_index=false (the default). PQ files
// are not embedded in the disk index in that mode, so they are unused here
// and we pass empty strings.

#include <cstddef>
#include <string>

#include "aux_utils.h"
#include "utils.h"

template<typename T>
void run(const std::string &mem_index, const std::string &base,
         const std::string &out, size_t npts_override) {
  diskann::create_disk_layout<T>(mem_index, base, /*tag_file=*/"",
                                 /*pq_pivots_file=*/"",
                                 /*pq_compressed_vectors_file=*/"",
                                 /*single_file_index=*/false, out,
                                 npts_override);
}

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    diskann::cout << "Usage: " << argv[0]
                  << " <type=float|int8|uint8>"
                     " <merged_mem_index> <base.bin> <out_disk_index>"
                     " [npts_override]"
                  << std::endl;
    return 1;
  }
  std::string type = argv[1];
  std::string mem_index = argv[2];
  std::string base = argv[3];
  std::string out = argv[4];
  size_t      npts_override = (argc == 6) ? (size_t) std::stoull(argv[5]) : 0;

  diskann::cout << "Finalizing: mem_index=" << mem_index << " base=" << base
                << " -> " << out << " npts_override=" << npts_override
                << std::endl;

  if (type == "float")
    run<float>(mem_index, base, out, npts_override);
  else if (type == "int8")
    run<int8_t>(mem_index, base, out, npts_override);
  else if (type == "uint8")
    run<uint8_t>(mem_index, base, out, npts_override);
  else {
    diskann::cerr << "Unknown type: " << type << std::endl;
    return 1;
  }
  return 0;
}
