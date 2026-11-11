// bufann_gt_computer.cpp
//
// Driver 2: Compute exact ground-truth nearest neighbours for a search workload.
//
// Given a base dataset and query set this tool computes the exact K nearest
// neighbours of each query over the active base points.  "Active" means the
// full base file minus any point IDs listed in the optional --exclude_file
// (used to model points that have been deleted before the search).
//
// Usage:
//   bufann_gt_computer --data_type float|int8|uint8
//                       --base_file  <path>    (.bin format)
//                       --query_file <path>    (.bin format)
//                       --gt_file    <path>    output truthset (.bin)
//                       --K          <N>       number of neighbours
//                      [--exclude_file <path>] binary: [int32 count][uint32*count IDs]
//                      [--num_threads <N>]     default: all available
//
// Output format (compatible with diskann::load_truthset):
//   [int32 nqueries][int32 K]
//   [uint32 nqueries*K  nearest-neighbour IDs  (row-major)]
//   [float  nqueries*K  L2^2 distances         (row-major)]
//   [uint32 nqueries*K  tags = same as IDs     (row-major)]
//
// The IDs in the output are the original 0-based positions of the points in
// the base file (so they can be compared against a BufANN index whose node
// IDs match those positions).
//
// Distance metric: L2 (squared), computed in float space regardless of the
// on-disk storage type (int8/uint8 are cast to float before distance
// computation — the rank order is preserved for L2).

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>
#include "mkl.h"

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

static std::string get_arg(int argc, char **argv, const std::string &flag,
                            const std::string &def = "") {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == flag) return argv[i + 1];
    return def;
}
static bool has_flag(int argc, char **argv, const std::string &flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

// Load exclude IDs from a binary file: [int32 count][uint32 * count].
static std::unordered_set<uint32_t> load_exclude_set(const std::string &path) {
    std::unordered_set<uint32_t> s;
    if (path.empty()) return s;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open exclude file: " + path);
    int32_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
    if (n <= 0) return s;
    std::vector<uint32_t> ids(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    s.insert(ids.begin(), ids.end());
    std::cout << "Loaded " << s.size() << " excluded IDs" << std::endl;
    return s;
}

// Read the header of a .bin file: [int32 npts][int32 dim].
static void read_bin_header(const std::string &path, size_t &npts, size_t &dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open bin file: " + path);
    int32_t n = 0, d = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
    in.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
    npts = static_cast<size_t>(n);
    dim  = static_cast<size_t>(d);
}

// Load a partition of the base file as float, converting from storage type T.
// part_idx is 0-based; PARTSIZE controls partition size.
static constexpr size_t PARTSIZE = 10000000;
static constexpr size_t ALIGNMENT = 512;

template <typename T>
static float *load_part_as_float(const std::string &path, size_t total_npts,
                                  size_t ndims, size_t part_idx,
                                  size_t &out_npts) {
    size_t part_start = part_idx * PARTSIZE;
    size_t part_end   = std::min(part_start + PARTSIZE, total_npts);
    out_npts          = part_end - part_start;
    if (out_npts == 0) return nullptr;

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open base file: " + path);
    in.seekg(2 * sizeof(int32_t) + part_start * ndims * sizeof(T));

    std::vector<T> buf(out_npts * ndims);
    in.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(out_npts * ndims * sizeof(T)));
    if (!in) throw std::runtime_error("unexpected EOF reading base partition");

    // Aligned allocation for BLAS (ALIGNMENT bytes).
    float *data = static_cast<float *>(
        aligned_alloc(ALIGNMENT, out_npts * ndims * sizeof(float)));
    if (!data) throw std::runtime_error("aligned_alloc failed");

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(out_npts * ndims); ++i)
        data[i] = static_cast<float>(buf[static_cast<size_t>(i)]);

    return data;
}

// Load the full query file as float.
template <typename T>
static float *load_queries_as_float(const std::string &path, size_t &nqueries,
                                     size_t &ndims) {
    read_bin_header(path, nqueries, ndims);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open query file: " + path);
    int32_t tmp1, tmp2;
    in.read(reinterpret_cast<char *>(&tmp1), sizeof(int32_t));
    in.read(reinterpret_cast<char *>(&tmp2), sizeof(int32_t));
    std::vector<T> buf(nqueries * ndims);
    in.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(nqueries * ndims * sizeof(T)));

    float *data = static_cast<float *>(
        aligned_alloc(ALIGNMENT, nqueries * ndims * sizeof(float)));
    if (!data) throw std::runtime_error("aligned_alloc for queries failed");
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(nqueries * ndims); ++i)
        data[i] = static_cast<float>(buf[static_cast<size_t>(i)]);
    return data;
}

// ---------------------------------------------------------------------------
// Distance computation (BLAS)
// ---------------------------------------------------------------------------

// Compute per-point L2 squared norms.
// Build uses -DMKL_ILP64, so cblas takes MKL_INT (= long long), not int.
static void compute_l2sq(float *norms, const float *matrix,
                          size_t npts, size_t dim) {
#pragma omp parallel for schedule(static)
    for (int64_t d = 0; d < static_cast<int64_t>(npts); ++d)
        norms[d] = cblas_sdot(static_cast<MKL_INT>(dim),
                              matrix + d * static_cast<ptrdiff_t>(dim), 1,
                              matrix + d * static_cast<ptrdiff_t>(dim), 1);
}

// Fill dist_matrix[npoints x nqueries] (col-major) with squared L2 distances
// using the identity  ||a-b||^2 = ||a||^2 - 2<a,b> + ||b||^2.
static void compute_distances(size_t dim, float *dist_matrix,
                               size_t npoints, const float *points,
                               const float *points_l2sq,
                               size_t nqueries, const float *queries,
                               const float *queries_l2sq) {
    std::vector<float> ones(std::max(npoints, nqueries), 1.0f);
    MKL_INT m  = static_cast<MKL_INT>(npoints);
    MKL_INT n  = static_cast<MKL_INT>(nqueries);
    MKL_INT k  = static_cast<MKL_INT>(dim);
    MKL_INT one = 1;

    // dist = -2 * P^T * Q
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, k,
                -2.0f, points, k, queries, k, 0.0f, dist_matrix, m);
    // dist += ||p_i||^2
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, one,
                1.0f, points_l2sq, m, ones.data(), n, 1.0f, dist_matrix, m);
    // dist += ||q_j||^2
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, one,
                1.0f, ones.data(), m, queries_l2sq, n, 1.0f, dist_matrix, m);
}

// ---------------------------------------------------------------------------
// Ground-truth writer
// ---------------------------------------------------------------------------

static void save_truthset(const std::string &path,
                           const std::vector<std::vector<std::pair<float, uint32_t>>> &results,
                           size_t K) {
    size_t nqueries = results.size();
    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write gt file: " + path);
    int32_t npts_i32 = static_cast<int32_t>(nqueries);
    int32_t k_i32    = static_cast<int32_t>(K);
    out.write(reinterpret_cast<const char *>(&npts_i32), sizeof(int32_t));
    out.write(reinterpret_cast<const char *>(&k_i32),    sizeof(int32_t));
    // IDs
    for (size_t q = 0; q < nqueries; ++q) {
        for (size_t k = 0; k < K; ++k) {
            uint32_t id = (k < results[q].size()) ? results[q][k].second : UINT32_MAX;
            out.write(reinterpret_cast<const char *>(&id), sizeof(uint32_t));
        }
    }
    // Distances
    for (size_t q = 0; q < nqueries; ++q) {
        for (size_t k = 0; k < K; ++k) {
            float d = (k < results[q].size()) ? results[q][k].first : FLT_MAX;
            out.write(reinterpret_cast<const char *>(&d), sizeof(float));
        }
    }
    std::cout << "Wrote truthset to " << path
              << "  (" << nqueries << " queries, K=" << K << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

template <typename T>
static int run(int argc, char **argv) {
    const std::string base_file    = get_arg(argc, argv, "--base_file");
    const std::string query_file   = get_arg(argc, argv, "--query_file");
    const std::string gt_file      = get_arg(argc, argv, "--gt_file");
    const std::string exclude_file = get_arg(argc, argv, "--exclude_file", "");
    size_t K = static_cast<size_t>(std::stoul(get_arg(argc, argv, "--K", "10")));
    int num_threads = std::stoi(get_arg(argc, argv, "--num_threads", "0"));

    if (base_file.empty() || query_file.empty() || gt_file.empty()) {
        std::cerr << "ERROR: --base_file, --query_file, --gt_file are required" << std::endl;
        return 1;
    }
    if (K == 0) { std::cerr << "ERROR: K must be > 0" << std::endl; return 1; }

    if (num_threads > 0) omp_set_num_threads(num_threads);

    // Load exclude set.
    const auto exclude_set = load_exclude_set(exclude_file);

    // Read headers.
    size_t base_npts = 0, base_dim = 0;
    read_bin_header(base_file, base_npts, base_dim);
    std::cout << "Base: " << base_npts << " points, dim=" << base_dim << std::endl;

    // Load queries.
    size_t nqueries = 0, query_dim = 0;
    float *queries = load_queries_as_float<T>(query_file, nqueries, query_dim);
    if (query_dim != base_dim)
        throw std::runtime_error("query dim != base dim");
    std::cout << "Queries: " << nqueries << std::endl;

    float *queries_l2sq = new float[nqueries];
    compute_l2sq(queries_l2sq, queries, nqueries, query_dim);

    // Per-query results: sorted (dist, global_id) ascending.
    // Maintained as a max-heap during accumulation.
    using Candidate = std::pair<float, uint32_t>;
    std::vector<std::vector<Candidate>> results(nqueries);

    // Batch size for dist_matrix memory: process this many queries at once.
    const size_t Q_BATCH = 512;

    size_t num_parts = (base_npts + PARTSIZE - 1) / PARTSIZE;
    size_t total_active = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (size_t p = 0; p < num_parts; ++p) {
        size_t part_npts = 0;
        float *base_part = load_part_as_float<T>(base_file, base_npts, base_dim, p, part_npts);
        size_t part_start = p * PARTSIZE;
        std::cout << "  Partition " << p << ": " << part_npts
                  << " points [" << part_start << ", " << part_start + part_npts << ")" << std::endl;

        float *part_l2sq = new float[part_npts];
        compute_l2sq(part_l2sq, base_part, part_npts, base_dim);

        // Count active (non-excluded) points.
        for (size_t i = 0; i < part_npts; ++i)
            if (!exclude_set.count(static_cast<uint32_t>(part_start + i))) ++total_active;

        // Process queries in batches to bound memory.
        float *dist_matrix = new float[Q_BATCH * part_npts];

        for (size_t q_b = 0; q_b < nqueries; q_b += Q_BATCH) {
            size_t q_e = std::min(q_b + Q_BATCH, nqueries);
            size_t q_count = q_e - q_b;

            compute_distances(base_dim, dist_matrix,
                               part_npts,   base_part,   part_l2sq,
                               q_count,     queries + q_b * base_dim,
                               queries_l2sq + q_b);

            // dist_matrix layout: col-major [part_npts x q_count]
            // => dist_matrix[i + q * part_npts] = dist^2(point i, query q_b+q)

            // Mask excluded points: set their distance to FLT_MAX.
            for (size_t i = 0; i < part_npts; ++i) {
                if (exclude_set.count(static_cast<uint32_t>(part_start + i))) {
                    for (size_t q = 0; q < q_count; ++q)
                        dist_matrix[i + q * part_npts] = FLT_MAX;
                }
            }

            // Merge partition results into global top-K per query.
#pragma omp parallel for schedule(dynamic, 4)
            for (int64_t q = 0; q < static_cast<int64_t>(q_count); ++q) {
                size_t gq = static_cast<size_t>(q) + q_b;
                auto &heap = results[gq];

                for (size_t i = 0; i < part_npts; ++i) {
                    float d = dist_matrix[i + static_cast<size_t>(q) * part_npts];
                    if (d >= FLT_MAX) continue;
                    uint32_t gid = static_cast<uint32_t>(part_start + i);

                    if (heap.size() < K) {
                        heap.push_back({d, gid});
                        if (heap.size() == K)
                            std::make_heap(heap.begin(), heap.end()); // max-heap
                    } else if (d < heap.front().first) {
                        std::pop_heap(heap.begin(), heap.end());
                        heap.back() = {d, gid};
                        std::push_heap(heap.begin(), heap.end());
                    }
                }
            }
        }

        delete[] dist_matrix;
        delete[] part_l2sq;
        free(base_part);
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Distance computation: " << elapsed << "s"
              << "  active base points: " << total_active << std::endl;

    // Sort each query's results ascending by distance.
    for (size_t q = 0; q < nqueries; ++q)
        std::sort(results[q].begin(), results[q].end());

    delete[] queries_l2sq;
    free(queries);

    save_truthset(gt_file, results, K);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (has_flag(argc, argv, "--help") || argc < 2) {
        std::cout << "Usage: bufann_gt_computer --data_type float|int8|uint8\n"
                     "         --base_file <path> --query_file <path>\n"
                     "         --gt_file <path> --K <N>\n"
                     "        [--exclude_file <path>] [--num_threads N]" << std::endl;
        return 0;
    }

    const std::string dtype = get_arg(argc, argv, "--data_type", "float");
    try {
        if (dtype == "float")  return run<float>(argc, argv);
        if (dtype == "int8")   return run<int8_t>(argc, argv);
        if (dtype == "uint8")  return run<uint8_t>(argc, argv);
        std::cerr << "ERROR: unknown --data_type '" << dtype << "'" << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
