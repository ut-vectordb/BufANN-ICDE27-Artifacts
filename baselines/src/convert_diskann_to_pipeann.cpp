#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSectorLen = 4096;

template <typename T>
T read_scalar(std::ifstream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error("failed to read scalar");
  }
  return value;
}

struct DiskannMetadata {
  uint32_t rows = 0;
  uint32_t cols = 0;
  uint64_t npoints = 0;
  uint64_t dim = 0;
  uint64_t medoid = 0;
  uint64_t max_node_len = 0;
  uint64_t nnodes_per_sector = 0;
};

DiskannMetadata load_diskann_metadata(const std::string &disk_index_path) {
  std::ifstream in(disk_index_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open DiskANN index: " + disk_index_path);
  }

  DiskannMetadata meta;
  meta.rows = read_scalar<uint32_t>(in);
  meta.cols = read_scalar<uint32_t>(in);
  meta.npoints = read_scalar<uint64_t>(in);
  meta.dim = read_scalar<uint64_t>(in);
  meta.medoid = read_scalar<uint64_t>(in);
  meta.max_node_len = read_scalar<uint64_t>(in);
  meta.nnodes_per_sector = read_scalar<uint64_t>(in);
  return meta;
}

void require_file(const std::string &path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("required file missing: " + path);
  }
}

// std::filesystem::copy_file replicates the source's permission bits. The
// canonical index is chmod 0400 (read-only fail-fast guard), so a plain copy
// yields a 0400 destination and the later in|out reopen for the header write
// fails with EACCES. Force owner-write on everything we create.
void make_owner_writable(const std::string &path) {
  if (std::filesystem::exists(path)) {
    std::filesystem::permissions(path, std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
  }
}

void copy_file_writable(const std::string &src, const std::string &dst) {
  std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
  make_owner_writable(dst);
}

void copy_if_exists(const std::string &src, const std::string &dst) {
  if (std::filesystem::exists(src)) {
    copy_file_writable(src, dst);
  }
}

void write_pipeann_metadata(const std::string &disk_index_path,
                            const DiskannMetadata &src_meta) {
  std::fstream out(disk_index_path, std::ios::binary | std::ios::in | std::ios::out);
  if (!out) {
    throw std::runtime_error("cannot reopen output disk index: " + disk_index_path);
  }

  std::vector<char> sector(kSectorLen, 0);
  std::size_t offset = 0;
  auto write_u32 = [&](uint32_t value) {
    std::memcpy(sector.data() + offset, &value, sizeof(value));
    offset += sizeof(value);
  };
  auto write_u64 = [&](uint64_t value) {
    std::memcpy(sector.data() + offset, &value, sizeof(value));
    offset += sizeof(value);
  };

  write_u32(7);
  write_u32(1);
  write_u64(src_meta.npoints);
  write_u64(src_meta.dim);
  write_u64(src_meta.medoid);
  write_u64(src_meta.max_node_len);
  write_u64(src_meta.nnodes_per_sector);
  write_u64(src_meta.npoints);
  write_u64(0);

  out.seekp(0, std::ios::beg);
  out.write(sector.data(), static_cast<std::streamsize>(sector.size()));
  if (!out) {
    throw std::runtime_error("failed to write PipeANN metadata sector");
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<std::string> positional;
  bool already_copied_dst = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--already-copied-dst") {
      already_copied_dst = true;
    } else {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 2) {
    std::cerr << "Usage: " << argv[0]
              << " <diskann_index_prefix> <pipeann_index_prefix>"
              << " [--already-copied-dst]\n"
              << "  --already-copied-dst: dst family was already copied"
              << " (e.g. via fastcopy); skip copying, just rewrite the"
              << " PipeANN header in-place.\n";
    return 1;
  }

  try {
    const std::string src_prefix = positional[0];
    const std::string dst_prefix = positional[1];

    const std::string src_disk = src_prefix + "_disk.index";
    const std::string src_tags = src_prefix + "_disk.index.tags";
    const std::string src_pq_pivots = src_prefix + "_pq_pivots.bin";
    const std::string src_pq_codes = src_prefix + "_pq_compressed.bin";

    const std::string dst_disk = dst_prefix + "_disk.index";
    const std::string dst_tags = dst_prefix + "_disk.index.tags";
    const std::string dst_pq_pivots = dst_prefix + "_pq_pivots.bin";
    const std::string dst_pq_codes = dst_prefix + "_pq_compressed.bin";

    // src_disk is always needed: even in --already-copied-dst mode we read
    // the DiskANN header from the source to build the PipeANN header.
    require_file(src_disk);

    if (already_copied_dst) {
      // Trust that the dst family was already duplicated (e.g. by fastcopy).
      // Verify the mandatory files are present; die loudly if not. Then fall
      // through to the in-place header rewrite only.
      require_file(dst_disk);
      require_file(dst_tags);
      require_file(dst_pq_pivots);
      require_file(dst_pq_codes);
      make_owner_writable(dst_disk);  // header write needs in|out on dst_disk
    } else {
      require_file(src_tags);
      require_file(src_pq_pivots);
      require_file(src_pq_codes);

      std::filesystem::create_directories(std::filesystem::path(dst_prefix).parent_path());
      copy_file_writable(src_disk, dst_disk);
      copy_file_writable(src_tags, dst_tags);
      copy_file_writable(src_pq_pivots, dst_pq_pivots);
      copy_file_writable(src_pq_codes, dst_pq_codes);

      copy_if_exists(src_disk + "_medoids.bin", dst_disk + "_medoids.bin");
      copy_if_exists(src_disk + "_centroids.bin", dst_disk + "_centroids.bin");
    }

    const DiskannMetadata meta = load_diskann_metadata(src_disk);
    write_pipeann_metadata(dst_disk, meta);

    std::cout << "Converted DiskANN prefix " << src_prefix
              << " -> PipeANN prefix " << dst_prefix << "\n";
    std::cout << "npoints=" << meta.npoints << " dim=" << meta.dim
              << " medoid=" << meta.medoid
              << " max_node_len=" << meta.max_node_len
              << " nnodes_per_sector=" << meta.nnodes_per_sector << "\n";
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
