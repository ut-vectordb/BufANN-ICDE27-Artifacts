// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <future>
#include <chrono>

#ifdef _WINDOWS
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#include "mkl.h"
#include "omp.h"
#include "utils.h"

// WORKS FOR UPTO 2 BILLION POINTS (as we use INT INSTEAD OF UNSIGNED)

#define PARTSIZE 25000000
#define ALIGNMENT 512

void command_line_help() {
  std::cerr << "<exact-kann> <int8/uint8/float>   <base bin file> <query bin "
               "file>  <K: # nearest neighbors to compute> "
               "<output-truthset-file> optional:<tag_file>"
            << std::endl;
}

template<class T>
T div_round_up(const T numerator, const T denominator) {
  return (numerator % denominator == 0) ? (numerator / denominator)
                                        : 1 + (numerator / denominator);
}

using pairIF = std::pair<int, float>;
// Order by (dist asc, id asc). Used as std::priority_queue comparator, which
// is a max-heap under operator<: top is the (largest dist, largest id) — i.e.
// the worst candidate to evict first. Matches the convention of the published
// sift1b GT (ties broken by ascending id).
struct cmpmaxstruct {
  bool operator()(const pairIF &l, const pairIF &r) {
    if (l.second != r.second) return l.second < r.second;
    return l.first < r.first;
  };
};

using maxPQIFCS =
    std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

template<class T>
T *aligned_malloc(const size_t n, const size_t alignment) {
#ifdef _WINDOWS
  return (T *) _aligned_malloc(sizeof(T) * n, alignment);
#else
  return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
}

inline bool custom_dist(const std::pair<uint32_t, float> &a,
                        const std::pair<uint32_t, float> &b) {
  if (a.second != b.second) return a.second < b.second;
  return a.first < b.first;
}

void compute_l2sq(float *const points_l2sq, const float *const matrix,
                  const int64_t num_points, const int dim) {
  assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
  for (int64_t d = 0; d < num_points; ++d)
    points_l2sq[d] = cblas_sdot(dim, matrix + (ptrdiff_t) d * (ptrdiff_t) dim,
                                1, matrix + (ptrdiff_t) d * (ptrdiff_t) dim, 1);
}

void distsq_to_points(
    const size_t dim,
    float *      dist_matrix,  // Col Major, cols are queries, rows are points
    size_t npoints, const float *const points,
    const float *const points_l2sq,  // points in Col major
    size_t nqueries, const float *const queries,
    const float *const queries_l2sq,  // queries in Col major
    float *ones_vec = NULL)  // unused; kept for ABI
{
  (void) ones_vec;
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim,
              (float) -2.0, points, dim, queries, dim, (float) 0.0, dist_matrix,
              npoints);
  // Fused rank-1 adds: dist[i,j] += points_l2sq[i] + queries_l2sq[j].
  // Replaces two K=1 sgemms that dispatched to a single-threaded ger kernel
  // on AMD (MKL def-ISA fallback) and stalled minutes per chunk.
#pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < (int64_t) nqueries; ++j) {
    const float qsq = queries_l2sq[j];
    float *col = dist_matrix + j * (ptrdiff_t) npoints;
    for (size_t i = 0; i < npoints; ++i)
      col[i] += points_l2sq[i] + qsq;
  }
}

void exact_knn(const size_t dim, const size_t k,
               int *const closest_points,  // k * num_queries preallocated, col
                                           // major, queries columns
               float *const dist_closest_points,  // k * num_queries
                                                  // preallocated, Dist to
                                                  // corresponding closes_points
               size_t             npoints,
               const float *const points,  // points in Col major
               size_t             nqueries,
               const float *const queries,  // queries in Col major
               int part_idx = -1, int num_parts = -1)
{
  float *points_l2sq = new float[npoints];
  float *queries_l2sq = new float[nqueries];
  compute_l2sq(points_l2sq, points, npoints, dim);
  compute_l2sq(queries_l2sq, queries, nqueries, dim);

  // Was (1 << 9) = 512 — too small to keep all 96 threads busy in the heap
  // merge phase below. 2048 spreads the per-query top-K work across enough
  // OMP chunks (2048/16 = 128) to saturate a 96-core box, at the cost of 4x
  // dist_matrix (now ~80GB for PARTSIZE=10M).
  size_t q_batch_size = (1 << 11);
  float *dist_matrix = new float[(size_t) q_batch_size * (size_t) npoints];

  const _u64 nbatches = div_round_up(nqueries, q_batch_size);
  for (_u64 b = 0; b < nbatches; ++b) {
    int64_t q_b = b * q_batch_size;
    int64_t q_e =
        ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

    auto t0 = std::chrono::steady_clock::now();
    distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, q_e - q_b,
                     queries + (ptrdiff_t) q_b * (ptrdiff_t) dim,
                     queries_l2sq + q_b);
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (long long q = q_b; q < q_e; q++) {
      maxPQIFCS point_dist;
      for (_u64 p = 0; p < k; p++)
        point_dist.emplace(
            p, dist_matrix[(ptrdiff_t) p +
                           (ptrdiff_t)(q - q_b) * (ptrdiff_t) npoints]);
      for (_u64 p = k; p < npoints; p++) {
        float cand_dist = dist_matrix[(ptrdiff_t) p +
                                      (ptrdiff_t)(q - q_b) * (ptrdiff_t) npoints];
        const auto &top = point_dist.top();
        // Admit if (cand_dist, p) < (top.dist, top.id) under (dist asc, id asc).
        if (cand_dist < top.second ||
            (cand_dist == top.second && (int) p < top.first)) {
          point_dist.emplace(p, cand_dist);
          point_dist.pop();
        }
      }
      for (ptrdiff_t l = 0; l < (ptrdiff_t) k; ++l) {
        closest_points[(ptrdiff_t)(k - 1 - l) + (ptrdiff_t) q * (ptrdiff_t) k] =
            point_dist.top().first;
        dist_closest_points[(ptrdiff_t)(k - 1 - l) +
                            (ptrdiff_t) q * (ptrdiff_t) k] =
            point_dist.top().second;
        point_dist.pop();
      }
      assert(std::is_sorted(
          dist_closest_points + (ptrdiff_t) q * (ptrdiff_t) k,
          dist_closest_points + (ptrdiff_t)(q + 1) * (ptrdiff_t) k));
    }
    auto t2 = std::chrono::steady_clock::now();
    double sgemm_s =
        std::chrono::duration<double>(t1 - t0).count();
    double knn_s = std::chrono::duration<double>(t2 - t1).count();
    if (part_idx >= 0) {
      diskann::cout << "[part " << (part_idx + 1) << "/" << num_parts
                    << " batch " << (b + 1) << "/" << nbatches << "] queries ["
                    << q_b << "," << q_e << ") sgemm=" << sgemm_s
                    << "s knn=" << knn_s << "s" << std::endl;
    }
  }

  delete[] dist_matrix;

  delete[] points_l2sq;
  delete[] queries_l2sq;
}

template<typename T>
inline int get_num_parts(const char *filename) {
  std::ifstream reader(filename, std::ios::binary);
  diskann::cout << "Reading bin file " << filename << " ...\n";
  int npts_i32, ndims_i32;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &ndims_i32, sizeof(int));
  diskann::cout << "#pts = " << npts_i32 << ", #dims = " << ndims_i32
                << std::endl;
  reader.close();
  uint32_t num_parts =
      (npts_i32 % PARTSIZE) == 0
          ? (_u32)(npts_i32 / PARTSIZE)
          : (_u32) std::floor((double) npts_i32 / (double) PARTSIZE) + 1;
  diskann::cout << "Number of parts: " << num_parts << std::endl;
  return num_parts;
}

template<typename T>
inline void load_bin_as_float(const char *filename, float *&data, size_t &npts,
                              size_t &ndims, int part_num) {
  std::ifstream reader(filename, std::ios::binary);
  diskann::cout << "Reading bin file " << filename << " ...\n";
  int npts_i32, ndims_i32;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &ndims_i32, sizeof(int));
  uint64_t start_id = part_num * PARTSIZE;
  uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t) npts_i32);
  npts = end_id - start_id;
  ndims = (unsigned) ndims_i32;
  uint64_t nptsuint64_t = (uint64_t) npts;
  uint64_t ndimsuint64_t = (uint64_t) ndims;
  diskann::cout << "#pts in part = " << npts << ", #dims = " << ndims
                << ", size = " << nptsuint64_t * ndimsuint64_t * sizeof(T)
                << "B" << std::endl;

  reader.seekg(start_id * ndims * sizeof(T) + 2 * sizeof(uint32_t),
               std::ios::beg);
  //    data = new T[nptsuint64_t * ndimsuint64_t];
  T *data_T = new T[nptsuint64_t * ndimsuint64_t];
  reader.read((char *) data_T, sizeof(T) * nptsuint64_t * ndimsuint64_t);
  diskann::cout << "Finished reading part of the bin file." << std::endl;
  reader.close();
  //  data =  (nptsuint64_t*ndimsuint64_t, ALIGNMENT);
  data = aligned_malloc<float>(nptsuint64_t * ndimsuint64_t, ALIGNMENT);
#pragma omp parallel for schedule(dynamic, 32768)
  for (int64_t i = 0; i < (int64_t) nptsuint64_t; i++) {
    for (int64_t j = 0; j < (int64_t) ndimsuint64_t; j++) {
      float cur_val_float = (float) data_T[i * ndimsuint64_t + j];
      std::memcpy((char *) (data + i * ndimsuint64_t + j),
                  (char *) &cur_val_float, sizeof(float));
    }
  }
  delete[] data_T;
  diskann::cout << "Finished converting part data to float." << std::endl;
}

template<typename T>
inline void save_bin(const std::string filename, T *data, size_t npts,
                     size_t ndims) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  diskann::cout << "Writing bin: " << filename << "\n";
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  diskann::cout << "bin: #pts = " << npts << ", #dims = " << ndims
                << ", size = " << npts * ndims * sizeof(T) + 2 * sizeof(int)
                << "B" << std::endl;

  //    data = new T[npts_u64 * ndims_u64];
  writer.write((char *) data, npts * ndims * sizeof(T));
  writer.close();
  diskann::cout << "Finished writing bin" << std::endl;
}

inline void save_groundtruth_as_one_file(const std::string filename,
                                         int32_t *data, float *distances,
                                         size_t npts, size_t ndims,
                                         uint32_t *tags = nullptr) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  int           npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  diskann::cout
      << "Saving truthset in one file (npts, dim, npts*dim id-matrix, "
         "npts*dim dist-matrix) with npts = "
      << npts << ", dim = " << ndims
      << ", size = " << 2 * npts * ndims * sizeof(unsigned) + 2 * sizeof(int)
      << "B" << std::endl;

  //    data = new T[npts_u64 * ndims_u64];
  writer.write((char *) data, npts * ndims * sizeof(uint32_t));
  writer.write((char *) distances, npts * ndims * sizeof(float));
  if (tags != nullptr) {
    writer.write((char *) tags, npts * ndims * sizeof(uint32_t));
  } else {
    writer.write((char *) data, npts * ndims * sizeof(uint32_t));
  }

  writer.close();
  diskann::cout << "Finished writing truthset" << std::endl;
}

template<typename T>
int aux_main(int argc, char **argv) {
  size_t      npoints, nqueries, dim;
  std::string base_file(argv[2]);
  std::string query_file(argv[3]);
  size_t      k = atoi(argv[4]);
  std::string gt_file(argv[5]);

  float *base_data;
  float *query_data;

  int num_parts = get_num_parts<T>(base_file.c_str());
  load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, 0);

  std::vector<std::vector<std::pair<uint32_t, float>>> results(nqueries);

  int *  closest_points = new int[nqueries * k];
  float *dist_closest_points = new float[nqueries * k];

  // Double-buffer the base-file reads: while chunk p is being scored against
  // queries, kick off the disk read + uint8->float cast of chunk p+1 in a
  // background std::async. The compute phase dominates (~tens of seconds per
  // chunk on 96 cores), so the ~2s read + ~0.1s cast hides cleanly under it.
  float *next_base = nullptr;
  size_t next_npts = 0, next_dim = 0;
  std::future<void> next_load;
  // Prime chunk 0 synchronously.
  load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, 0);
  for (int p = 0; p < num_parts; p++) {
    size_t start_id = p * PARTSIZE;
    if (p + 1 < num_parts) {
      next_base = nullptr;
      next_npts = 0;
      next_dim = 0;
      next_load = std::async(std::launch::async, [base_file, p, &next_base,
                                                  &next_npts, &next_dim]() {
        load_bin_as_float<T>(base_file.c_str(), next_base, next_npts, next_dim,
                             p + 1);
      });
    }

    int *  closest_points_part = new int[nqueries * k];
    float *dist_closest_points_part = new float[nqueries * k];
    auto part_t0 = std::chrono::steady_clock::now();
    exact_knn(dim, k, closest_points_part, dist_closest_points_part, npoints,
              base_data, nqueries, query_data, p, num_parts);
    auto part_t1 = std::chrono::steady_clock::now();
    diskann::cout << "[part " << (p + 1) << "/" << num_parts
                  << "] done in "
                  << std::chrono::duration<double>(part_t1 - part_t0).count()
                  << "s" << std::endl;

    for (_u64 i = 0; i < nqueries; i++) {
      for (_u64 j = 0; j < k; j++) {
        results[i].push_back(std::make_pair(
            (uint32_t)(closest_points_part[i * k + j] + start_id),
            dist_closest_points_part[i * k + j]));
      }
    }

    delete[] closest_points_part;
    delete[] dist_closest_points_part;
    diskann::aligned_free(base_data);

    if (p + 1 < num_parts) {
      next_load.get();  // blocks if read isn't done; usually a no-op.
      base_data = next_base;
      npoints = next_npts;
      // dim doesn't change across chunks; sanity check.
      assert(next_dim == dim);
    }
  }

  for (_u64 i = 0; i < nqueries; i++) {
    std::vector<std::pair<uint32_t, float>> &cur_res = results[i];
    std::sort(cur_res.begin(), cur_res.end(), custom_dist);
    for (_u64 j = 0; j < k; j++) {
      closest_points[i * k + j] = (int32_t) cur_res[j].first;
      dist_closest_points[i * k + j] = cur_res[j].second;
    }
  }
  uint32_t *tags = nullptr;
  if (argc == 7) {
    std::cout << "Loading tags from " << argv[6] << "\n";
    tags = new uint32_t[nqueries * k];
    uint32_t *  all_tags;
    std::string tag_file = std::string(argv[6]);
    size_t      tag_pts, tag_dim;
    diskann::load_bin(tag_file, all_tags, tag_pts, tag_dim);

    diskann::cout << "Loaded tags for " << tag_pts << " points.\n";
    for (uint64_t i = 0; i < nqueries * k; i++) {
      tags[i] = all_tags[closest_points[i]];
    }
  }

  save_groundtruth_as_one_file(gt_file, closest_points, dist_closest_points,
                               nqueries, k, tags);
  diskann::aligned_free(query_data);
  delete[] closest_points;
  delete[] dist_closest_points;
  if (tags != nullptr) {
    delete[] tags;
  }

  return 0;
}

int main(int argc, char **argv) {
  if (argc != 6 && argc != 7) {
    command_line_help();
    return -1;
  }
  if (std::string(argv[1]) == std::string("float"))
    aux_main<float>(argc, argv);
  if (std::string(argv[1]) == std::string("int8"))
    aux_main<int8_t>(argc, argv);
  if (std::string(argv[1]) == std::string("uint8"))
    aux_main<uint8_t>(argc, argv);
}
