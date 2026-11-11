// Build the in-memory Vamana graph for a single pre-partitioned shard.
// Mirrors the per-shard loop body in build_merged_vamana_index
// (src/aux_utils.cpp:567-590).
//
// Usage:
//   build_shard <type=float|int8|uint8>
//               <shard_base.bin> <shard_index.out>
//               <R_global> <L> <T> <metric=l2|cosine>
//
// The per-shard graph degree is computed as 2*R_global/3 internally,
// matching what build_disk_index does for partitioned builds.
//
// If <shard_index.out> already exists, exits early with success — making
// this idempotent and safe to invoke from a shell loop.

#include <sys/stat.h>
#include <memory>
#include <string>

#include "omp.h"

#include "aux_utils.h"
#include "index.h"
#include "parameters.h"
#include "utils.h"

static bool shard_output_exists(const std::string &path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

template<typename T>
int build_one_shard(const std::string &shard_base_file,
                    const std::string &shard_index_file, unsigned R_global,
                    unsigned L, unsigned num_threads, diskann::Metric metric) {
  if (shard_output_exists(shard_index_file)) {
    diskann::cout << "Shard index already exists at " << shard_index_file
                  << ", skipping." << std::endl;
    return 0;
  }

  if (num_threads != 0)
    omp_set_num_threads(num_threads);

  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", (2 * (R_global / 3)));
  paras.Set<unsigned>("C", diskann::env_u32("C", 750));
  paras.Set<float>("alpha",
                   diskann::env_f32("ALPHA_DISK", diskann::env_f32("ALPHA",
                                                                   1.2f)));
  paras.Set<unsigned>("num_rnds", 2);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<std::string>("save_path", shard_index_file);

  size_t shard_base_pts, shard_base_dim;
  diskann::get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);

  diskann::cout << "Building shard graph: " << shard_base_file
                << " -> " << shard_index_file << " (npts=" << shard_base_pts
                << ", dim=" << shard_base_dim << ", R'=" << (2 * (R_global / 3))
                << ", L=" << L << ", T=" << num_threads << ")" << std::endl;

  std::unique_ptr<diskann::Index<T>> idx(new diskann::Index<T>(
      metric, shard_base_dim, shard_base_pts, /*dynamic_index=*/false,
      /*single_file_index=*/false));
  idx->build(shard_base_file.c_str(), shard_base_pts, paras);
  idx->save(shard_index_file.c_str());
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 8) {
    diskann::cout
        << "Usage: " << argv[0]
        << " <type=float|int8|uint8> <shard_base.bin> <shard_index.out>"
           " <R_global> <L> <T> <metric=l2|cosine>"
        << std::endl;
    return 1;
  }
  std::string type = argv[1];
  std::string shard_base = argv[2];
  std::string shard_out = argv[3];
  unsigned    R = (unsigned) std::stoul(argv[4]);
  unsigned    L = (unsigned) std::stoul(argv[5]);
  unsigned    T = (unsigned) std::stoul(argv[6]);
  std::string metric_s = argv[7];

  diskann::Metric metric =
      (metric_s == "cosine") ? diskann::Metric::COSINE : diskann::Metric::L2;
  if (metric_s != "l2" && metric_s != "cosine") {
    diskann::cerr << "Unknown metric: " << metric_s << " (use l2 or cosine)"
                  << std::endl;
    return 1;
  }

  if (type == "float")
    return build_one_shard<float>(shard_base, shard_out, R, L, T, metric);
  if (type == "int8")
    return build_one_shard<int8_t>(shard_base, shard_out, R, L, T, metric);
  if (type == "uint8")
    return build_one_shard<uint8_t>(shard_base, shard_out, R, L, T, metric);
  diskann::cerr << "Unknown type: " << type << std::endl;
  return 1;
}
