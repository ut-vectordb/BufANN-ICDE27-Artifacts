// Incremental groundtruth: given a cached K_cached top-K file computed over
// the FULL base (e.g. 1B), plus optional --inserts / --deletes / --base_trunc,
// emit a K_out groundtruth in the same on-disk format as compute_groundtruth
// (ids, dists, tags=ids) without re-running the gemm pass.
//
// Validity: if id i would be in the post-mutation top-K_out for query q, then
// i must also be in the full-base top-K_cached for q (top-K of a subset is
// contained in top-K_cached of the superset when K_out <= K_cached). So we
// filter the cached list and pick the first K_out survivors. An underflow
// (|survivors| < K_out) means the cache was too shallow; the tool aborts with
// the offending query indices.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

void usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " <cached_gt> <K_out> <gt_out>"
         "  [--inserts FILE] [--deletes FILE] [--base_trunc N]"
         " [--delete_range BEGIN COUNT]\n"
      << "  cached_gt: output of compute_groundtruth (with tags column)\n"
      << "  K_out:     output truthset width (must be <= K_cached)\n"
      << "  --inserts: counted uint32 id file (int32 count + N uint32 ids).\n"
      << "             Inserted base IDs (typically >= base_trunc).\n"
      << "  --deletes: counted uint32 id file. Base IDs to drop.\n"
      << "  --base_trunc N: simulate a canonical of size N by treating any id\n"
      << "             >= N as removed UNLESS it appears in --inserts.\n"
      << "  --delete_range BEGIN COUNT: drop base IDs in [BEGIN, BEGIN+COUNT).\n"
      << "             Numeric form of --deletes for a contiguous round range;\n"
      << "             composes with --base_trunc (insert/delete/mixed rounds).\n"
      << std::endl;
}

// Read a counted uint32 id file: int32 count, then count uint32 ids.
bool load_counted_ids(const std::string &path,
                      std::vector<uint32_t> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "ERROR: cannot open " << path << std::endl;
    return false;
  }
  int32_t count = 0;
  f.read(reinterpret_cast<char *>(&count), sizeof(int32_t));
  if (!f || count < 0) {
    std::cerr << "ERROR: bad count header in " << path << std::endl;
    return false;
  }
  out.resize(static_cast<size_t>(count));
  if (count > 0)
    f.read(reinterpret_cast<char *>(out.data()),
           static_cast<std::streamsize>(count) * sizeof(uint32_t));
  if (!f) {
    std::cerr << "ERROR: short read from " << path << std::endl;
    return false;
  }
  return true;
}

// Read a compute_groundtruth output file. Supports the "with tags" layout this
// repo writes (ids + dists + tags). Returns (npts, dim) via out params.
bool load_truthset_with_dists(const std::string &path,
                              std::vector<uint32_t> &ids,
                              std::vector<float> &dists, size_t &npts,
                              size_t &dim) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "ERROR: cannot open " << path << std::endl;
    return false;
  }
  int32_t npts_i32 = 0, dim_i32 = 0;
  f.read(reinterpret_cast<char *>(&npts_i32), sizeof(int32_t));
  f.read(reinterpret_cast<char *>(&dim_i32), sizeof(int32_t));
  if (!f || npts_i32 <= 0 || dim_i32 <= 0) {
    std::cerr << "ERROR: bad header in " << path << std::endl;
    return false;
  }
  npts = static_cast<size_t>(npts_i32);
  dim = static_cast<size_t>(dim_i32);
  const size_t n = npts * dim;
  ids.resize(n);
  dists.resize(n);
  f.read(reinterpret_cast<char *>(ids.data()),
         static_cast<std::streamsize>(n) * sizeof(uint32_t));
  f.read(reinterpret_cast<char *>(dists.data()),
         static_cast<std::streamsize>(n) * sizeof(float));
  if (!f) {
    std::cerr << "ERROR: short read from " << path
              << " (expected ids+dists; tags column ignored)" << std::endl;
    return false;
  }
  return true;
}

// Write same format compute_groundtruth uses with no --tags: header, ids,
// dists, and a duplicate of ids in the tags slot. Recall consumers accept
// either column; duplicating keeps the file shape identical.
bool save_truthset(const std::string &path,
                   const std::vector<uint32_t> &ids,
                   const std::vector<float> &dists, size_t npts, size_t dim) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "ERROR: cannot open " << path << " for writing" << std::endl;
    return false;
  }
  int32_t npts_i32 = static_cast<int32_t>(npts);
  int32_t dim_i32 = static_cast<int32_t>(dim);
  f.write(reinterpret_cast<const char *>(&npts_i32), sizeof(int32_t));
  f.write(reinterpret_cast<const char *>(&dim_i32), sizeof(int32_t));
  const size_t n = npts * dim;
  f.write(reinterpret_cast<const char *>(ids.data()),
          static_cast<std::streamsize>(n) * sizeof(uint32_t));
  f.write(reinterpret_cast<const char *>(dists.data()),
          static_cast<std::streamsize>(n) * sizeof(float));
  f.write(reinterpret_cast<const char *>(ids.data()),
          static_cast<std::streamsize>(n) * sizeof(uint32_t));
  return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    usage(argv[0]);
    return 2;
  }
  const std::string cached_gt = argv[1];
  const size_t      K_out = std::stoul(argv[2]);
  const std::string gt_out = argv[3];
  std::string inserts_path, deletes_path;
  uint64_t    base_trunc = UINT64_MAX;  // sentinel: no truncation
  // Contiguous deleted ID range [del_range_begin, del_range_end): a numeric
  // alternative to --deletes for the common case where a round removes a
  // contiguous block of IDs (insert/delete/mixed all do). Avoids writing a
  // per-round delete-id file. Sentinel begin==end means "no range".
  uint64_t    del_range_begin = 0, del_range_end = 0;

  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--inserts" && i + 1 < argc) {
      inserts_path = argv[++i];
    } else if (a == "--deletes" && i + 1 < argc) {
      deletes_path = argv[++i];
    } else if (a == "--base_trunc" && i + 1 < argc) {
      base_trunc = std::stoull(argv[++i]);
    } else if (a == "--delete_range" && i + 2 < argc) {
      del_range_begin = std::stoull(argv[++i]);
      const uint64_t cnt = std::stoull(argv[++i]);
      del_range_end = del_range_begin + cnt;
    } else {
      std::cerr << "ERROR: unknown arg " << a << std::endl;
      usage(argv[0]);
      return 2;
    }
  }

  std::vector<uint32_t> cached_ids;
  std::vector<float>    cached_dists;
  size_t                Q = 0, K_cached = 0;
  if (!load_truthset_with_dists(cached_gt, cached_ids, cached_dists, Q,
                                K_cached))
    return 1;
  std::cout << "cached_gt: Q=" << Q << " K_cached=" << K_cached << std::endl;
  if (K_out > K_cached) {
    std::cerr << "ERROR: K_out (" << K_out << ") > K_cached (" << K_cached
              << "); rerun compute_groundtruth with larger K." << std::endl;
    return 1;
  }

  std::vector<uint32_t> insert_ids_v, delete_ids_v;
  if (!inserts_path.empty() && !load_counted_ids(inserts_path, insert_ids_v))
    return 1;
  if (!deletes_path.empty() && !load_counted_ids(deletes_path, delete_ids_v))
    return 1;

  std::unordered_set<uint32_t> insert_set(insert_ids_v.begin(),
                                          insert_ids_v.end());
  std::unordered_set<uint32_t> delete_set(delete_ids_v.begin(),
                                          delete_ids_v.end());
  std::cout << "inserts=" << insert_set.size()
            << " deletes=" << delete_set.size()
            << " base_trunc=" << (base_trunc == UINT64_MAX
                                      ? std::string("none")
                                      : std::to_string(base_trunc))
            << " K_out=" << K_out << std::endl;

  // An id survives iff: NOT in delete_set AND (id < base_trunc OR id in
  // insert_set). Equivalently, suppressed iff in delete_set OR (id >=
  // base_trunc AND id NOT in insert_set).
  auto suppressed = [&](uint32_t id) -> bool {
    if (delete_set.count(id))
      return true;
    if (static_cast<uint64_t>(id) >= base_trunc && !insert_set.count(id))
      return true;
    return false;
  };

  std::vector<uint32_t> out_ids(Q * K_out);
  std::vector<float>    out_dists(Q * K_out);
  size_t                deletes_seen_in_cache = 0;
  size_t                truncations_seen_in_cache = 0;
  size_t                inserts_kept_from_cache = 0;
  std::vector<size_t>   underflow_queries;

  for (size_t q = 0; q < Q; ++q) {
    size_t kept = 0;
    for (size_t k = 0; k < K_cached && kept < K_out; ++k) {
      uint32_t id = cached_ids[q * K_cached + k];
      if (delete_set.count(id)) {
        ++deletes_seen_in_cache;
        continue;
      }
      if (del_range_end > del_range_begin &&
          static_cast<uint64_t>(id) >= del_range_begin &&
          static_cast<uint64_t>(id) < del_range_end) {
        ++deletes_seen_in_cache;
        continue;
      }
      if (static_cast<uint64_t>(id) >= base_trunc) {
        if (insert_set.count(id)) {
          ++inserts_kept_from_cache;
        } else {
          ++truncations_seen_in_cache;
          continue;
        }
      }
      (void) suppressed;  // logic above is the inlined form
      out_ids[q * K_out + kept] = id;
      out_dists[q * K_out + kept] = cached_dists[q * K_cached + k];
      ++kept;
    }
    if (kept < K_out) underflow_queries.push_back(q);
  }

  std::cout << "deletes_seen_in_cache=" << deletes_seen_in_cache
            << " truncations_seen_in_cache=" << truncations_seen_in_cache
            << " inserts_kept_from_cache=" << inserts_kept_from_cache
            << std::endl;
  if (!underflow_queries.empty()) {
    std::cerr << "ERROR: " << underflow_queries.size()
              << " queries underflowed (cached top-" << K_cached
              << " ran out before " << K_out << " survivors). First few: ";
    for (size_t i = 0; i < std::min<size_t>(10, underflow_queries.size()); ++i)
      std::cerr << underflow_queries[i] << " ";
    std::cerr << "\nRerun compute_groundtruth with K_cached larger than "
              << K_cached << ", or shrink mutation sets." << std::endl;
    return 3;
  }

  if (!save_truthset(gt_out, out_ids, out_dists, Q, K_out)) return 1;
  std::cout << "wrote " << gt_out << " (Q=" << Q << " K=" << K_out << ")"
            << std::endl;
  return 0;
}
