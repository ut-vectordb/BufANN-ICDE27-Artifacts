// Merge per-shard Vamana graphs into a single mem index.
// Wraps diskann::merge_shards (src/aux_utils.cpp:341).
//
// Usage:
//   merge_shards_only <subshard_prefix> <num_parts> <R_global>
//                     <out_mem_index> <out_medoids>
//
// <subshard_prefix> is the part before the shard number. Files are read as
//   <prefix><N>_mem.index   for N in 0..num_parts-1
//   <prefix><N>_ids_uint32.bin
//
// Example: if your shards are at
//   /tmp/idx/foo_mem.index_tempFiles_subshard-0_mem.index
//   /tmp/idx/foo_mem.index_tempFiles_subshard-0_ids_uint32.bin
// then pass "/tmp/idx/foo_mem.index_tempFiles_subshard-" as the prefix.

#include <string>

#include "aux_utils.h"

int main(int argc, char **argv) {
  if (argc != 6) {
    diskann::cout << "Usage: " << argv[0]
                  << " <subshard_prefix> <num_parts> <R_global>"
                     " <out_mem_index> <out_medoids>"
                  << std::endl;
    return 1;
  }
  std::string  prefix = argv[1];
  unsigned     num_parts = (unsigned) std::stoul(argv[2]);
  unsigned     R = (unsigned) std::stoul(argv[3]);
  std::string  out_mem = argv[4];
  std::string  out_medoids = argv[5];

  diskann::cout << "Merging " << num_parts << " shards from prefix '" << prefix
                << "' (R=" << R << ") -> " << out_mem << " (+ " << out_medoids
                << ")" << std::endl;

  return diskann::merge_shards(prefix, "_mem.index", prefix, "_ids_uint32.bin",
                               num_parts, R, out_mem, out_medoids);
}
