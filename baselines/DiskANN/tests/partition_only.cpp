// Standalone wrapper for the k-means partitioning + retry-on-RAM-overflow
// step. Mirrors what build_merged_vamana_index does at its head
// (src/aux_utils.cpp:559-565) before the per-shard build loop.
//
// Usage:
//   partition_only <type=float|int8|uint8> <base.bin>
//                  <index_prefix> <R_global> <M_GB> [npts_override]
//
// If npts_override > 0 and < the file's actual npts, only the first
// npts_override points of <base.bin> are treated as the dataset (k-means
// training, cluster-size estimation, and shard assignment all clamp to it).
// This lets a 0.9B build run directly off the full 1B base file without
// materializing a slice copy.
//
// Outputs (multi-file mode, single_file_index=false):
//   <index_prefix>_mem.index_tempFiles_subshard-N.bin               (N=0..K-1)
//   <index_prefix>_mem.index_tempFiles_subshard-N_ids_uint32.bin    (N=0..K-1)
//   <index_prefix>_disk.index_centroids.bin                          (renamed)
//
// num_shards K is determined by the retry loop inside
// partition_with_ram_budget — starts at 3 and increments until the largest
// shard's estimated RAM footprint fits within M_GB.
//
// The sampling rate for k-means training is computed the same way as
// build_disk_index: p_val = min(PQ_TRAINING_SET_FRACTION, MAX_PQ_TRAINING_SET_SIZE/npts).
// Reusing this matches the original code path exactly so subsequent stages
// stay consistent.

#include <cstdio>
#include <string>

#include "aux_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

template<typename T>
int run(const std::string &base_file, const std::string &index_prefix,
        unsigned R_global, double M_GB, size_t npts_override) {
  std::string mem_index_path = index_prefix + "_mem.index";
  std::string merged_index_prefix = mem_index_path + "_tempFiles";
  std::string centroids_file = index_prefix + "_disk.index_centroids.bin";

  size_t npts, dim;
  diskann::get_bin_metadata(base_file, npts, dim);
  size_t effective_npts = npts;
  if (npts_override > 0 && npts_override < npts) {
    effective_npts = npts_override;
  }

  size_t training_set_size =
      diskann::PQ_TRAINING_SET_FRACTION * effective_npts >
              diskann::MAX_PQ_TRAINING_SET_SIZE
          ? diskann::MAX_PQ_TRAINING_SET_SIZE
          : (size_t) std::round(diskann::PQ_TRAINING_SET_FRACTION *
                                effective_npts);
  if (training_set_size == 0)
    training_set_size = 1;
  double p_val = (double) training_set_size / (double) effective_npts;

  diskann::cout << "Partitioning " << base_file << " (file_npts=" << npts
                << ", effective_npts=" << effective_npts << ", dim=" << dim
                << ") with R_global=" << R_global << ", M_GB=" << M_GB
                << ", p_val=" << p_val
                << ", k_base=2, R_per_shard=" << (2 * R_global / 3) << std::endl;

  int num_parts = partition_with_ram_budget<T>(
      base_file, p_val, M_GB, 2 * R_global / 3, merged_index_prefix, 2,
      npts_override);

  if (num_parts < 0) {
    diskann::cerr << "partition_with_ram_budget failed (returned " << num_parts
                  << ")" << std::endl;
    return 1;
  }

  // Match build_merged_vamana_index: rename centroid file to its final path.
  std::string cur_centroid = merged_index_prefix + "_centroids.bin";
  if (std::rename(cur_centroid.c_str(), centroids_file.c_str()) != 0) {
    diskann::cerr << "Warning: rename of " << cur_centroid << " -> "
                  << centroids_file << " failed (errno=" << errno << ")"
                  << std::endl;
  }

  diskann::cout << "Done. num_shards=" << num_parts
                << ". Centroids at: " << centroids_file << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 6 && argc != 7) {
    diskann::cout << "Usage: " << argv[0]
                  << " <type=float|int8|uint8> <base.bin>"
                     " <index_prefix> <R_global> <M_GB> [npts_override]"
                  << std::endl;
    return 1;
  }
  std::string type = argv[1];
  std::string base = argv[2];
  std::string index_prefix = argv[3];
  unsigned    R = (unsigned) std::stoul(argv[4]);
  double      M = std::stod(argv[5]);
  size_t      npts_override = (argc == 7) ? (size_t) std::stoull(argv[6]) : 0;

  if (type == "float")
    return run<float>(base, index_prefix, R, M, npts_override);
  if (type == "int8")
    return run<int8_t>(base, index_prefix, R, M, npts_override);
  if (type == "uint8")
    return run<uint8_t>(base, index_prefix, R, M, npts_override);
  diskann::cerr << "Unknown type: " << type << std::endl;
  return 1;
}
