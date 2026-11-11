// Standalone PQ builder: produces <prefix>_pq_pivots.bin and
// <prefix>_pq_compressed.bin for an index that was built with SKIP_PQ=1.
// Mirrors the PQ block in diskann::build_disk_index (src/aux_utils.cpp
// ~line 1048) so search-time files match what the original builder would
// have written.
//
// Usage:
//   build_pq_standalone <type=float|int8|uint8>
//                       <base.bin> <index_prefix> <num_pq_chunks>
//
// Outputs:
//   <index_prefix>_pq_pivots.bin
//   <index_prefix>_pq_compressed.bin
//
// Assumes single_file_index=false (default for SKIP_PQ builds). num_pq_chunks
// is clamped to MAX_PQ_CHUNKS, matching build_disk_index's behavior.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "aux_utils.h"
#include "partition_and_pq.h"
#include "pq_flash_index.h"  // MAX_PQ_CHUNKS
#include "utils.h"

// NUM_KMEANS is a translation-unit-local define in src/aux_utils.cpp; mirror
// the same value here so pivot training matches the in-tree builder.
#define NUM_KMEANS 15

template<typename T>
void run(const std::string &base_file, const std::string &prefix,
         uint32_t num_pq_chunks) {
  const std::string pq_pivots_path     = prefix + "_pq_pivots.bin";
  const std::string pq_compressed_path = prefix + "_pq_compressed.bin";

  size_t npts, dim;
  diskann::get_bin_metadata(base_file, npts, dim);

  num_pq_chunks =
      std::min(num_pq_chunks, static_cast<uint32_t>(MAX_PQ_CHUNKS));

  // Match build_disk_index's training-set sizing.
  const size_t training_set_size =
      diskann::PQ_TRAINING_SET_FRACTION * npts > diskann::MAX_PQ_TRAINING_SET_SIZE
          ? diskann::MAX_PQ_TRAINING_SET_SIZE
          : static_cast<size_t>(diskann::PQ_TRAINING_SET_FRACTION * npts);
  const double p_val = static_cast<double>(training_set_size) / npts;

  diskann::cout << "build_pq_standalone: base=" << base_file
                << " prefix=" << prefix << " npts=" << npts << " dim=" << dim
                << " num_pq_chunks=" << num_pq_chunks
                << " train_frac=" << p_val << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();

  float *train_data = nullptr;
  size_t train_n, train_dim;
  ::gen_random_slice<T>(base_file, p_val, train_data, train_n, train_dim);

  diskann::cout << "Generating PQ pivots: train_size=" << train_n
                << " train_dim=" << train_dim << std::endl;
  ::generate_pq_pivots(train_data, train_n, static_cast<uint32_t>(dim), 256,
                       num_pq_chunks, NUM_KMEANS, pq_pivots_path);

  auto t1 = std::chrono::high_resolution_clock::now();
  diskann::cout << "Pivots generated in "
                << std::chrono::duration<double>(t1 - t0).count() << "s."
                << std::endl;

  ::generate_pq_data_from_pivots<T>(base_file, 256, num_pq_chunks,
                                    pq_pivots_path, pq_compressed_path);
  delete[] train_data;

  auto t2 = std::chrono::high_resolution_clock::now();
  diskann::cout << "Compressed data written in "
                << std::chrono::duration<double>(t2 - t1).count() << "s."
                << std::endl;
  diskann::cout << "Wrote " << pq_pivots_path << " and " << pq_compressed_path
                << std::endl;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    diskann::cout << "Usage: " << argv[0]
                  << " <type=float|int8|uint8>"
                     " <base.bin> <index_prefix> <num_pq_chunks>"
                  << std::endl;
    return 1;
  }
  std::string type           = argv[1];
  std::string base_file      = argv[2];
  std::string prefix         = argv[3];
  uint32_t    num_pq_chunks  = static_cast<uint32_t>(std::stoul(argv[4]));

  if (type == "float")
    run<float>(base_file, prefix, num_pq_chunks);
  else if (type == "int8")
    run<int8_t>(base_file, prefix, num_pq_chunks);
  else if (type == "uint8")
    run<uint8_t>(base_file, prefix, num_pq_chunks);
  else {
    diskann::cerr << "Unknown type: " << type << std::endl;
    return 1;
  }
  return 0;
}
