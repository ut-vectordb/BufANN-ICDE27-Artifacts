// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

struct GlobalStats {
  // Insert-contention counters added for the benchmark harness. They
  // are atomics because every insert thread updates them; the double fields
  // below are updated without synchronization and are only approximate.
  std::atomic<uint64_t> harness_inserts{0};
  std::atomic<uint64_t> harness_topo_splits{0};
  std::atomic<uint64_t> harness_pages_locked{0};
  std::atomic<uint64_t> harness_lock_wait_ns{0};
  void clear_harness_counters() {
    harness_inserts = 0;
    harness_topo_splits = 0;
    harness_pages_locked = 0;
    harness_lock_wait_ns = 0;
  }

  double search_io = 0;
  double search_stage_1 = 0;
  double search_stage_2 = 0;
  double search_stage_3 = 0;
  double search_read_neighbors_io = 0;
  double search_read_coord_io = 0;
  double search_all = 0;

  double compute_deleted_ids = 0;
  double delete_scan1 = 0;
  int delete_scan_count = 0;
  int delete_scan_hit = 0;
  int insert_scan_hit = 0;
  double delete_io = 0;
  double update_io = 0;
  uint64_t update_ios = 0;
  uint64_t insert_ios1 = 0;
  uint64_t insert_ios2 = 0;
  uint64_t insert_ios3 = 0;
  uint64_t insert_ios4 = 0;
  uint64_t insert_ios5 = 0;
  uint64_t insert_ios6 = 0;
  uint64_t insert_ios7 = 0;
  uint64_t insert_ios8 = 0;
  uint64_t read_ios = 0;
  uint64_t write_ios = 0;
  uint64_t update_ios_topo = 0;
  uint64_t update_ios_coord = 0;
  // uint64_t update_ios_redundant = 0;
  double delete_io1 = 0;
  double populate_deleted_nhoods= 0;
  double process_deletes = 0;
  double delete_io2 = 0;
  double delete_io3 = 0;
  double delete_cpu = 0;
  double compute_pq_dists = 0;
  double consolidate_deletes = 0;
  double consolidate_deletes_actually = 0;
  double occlude_list_pq = 0;
  double delete_all = 0;

  double insert_io1 = 0;
  double insert_io2 = 0;
  double insert_io3 = 0;
  double insert_io4 = 0;
  double insert_io5 = 0;
  double insert_io6 = 0;
  double insert_io7 = 0;
  double insert_io8 = 0;
  double insert_io = 0;
  double insert_cpu = 0;
  double insert_all = 0;

  double insert_time1 = 0;
  double insert_time2 = 0;
  double insert_time3 = 0;
  double insert_time4 = 0;
  double insert_time5 = 0;
  double insert_time6 = 0;
  double insert_time7 = 0;
  double insert_time8 = 0;

  double pq_tag_io = 0;
  double pq_cpu = 0;

  double write_tag_file_io = 0;

  uint64_t update_effective_coord_ios = 0;
  uint64_t update_effective_topo_ios = 0;
  double merge_io = 0;
  double merge_cpu = 0;
  double merge_all = 0;

  uint64_t insert_pq = 0;
  uint64_t insert_search = 0;
  uint64_t insert_prune = 0;
  uint64_t insert_buffer_io = 0;
  std::vector<double> recall;
  std::vector<double> qps;
  std::vector<double> rerank_ios;
  std::vector<double> search_latency;
  std::vector<double> update_latency;
  std::vector<double> insert_latency;
  std::vector<double> delete_latency;
  std::vector<double> merge_latency;
  std::vector<double> hits;
  std::map<int, int> hist;
  int bad_query_count = 0;

  std::vector<std::vector<int>> cache_hit_vec;
  int cache_count = 0;
  int cache_miss = 0;
  int cache_hit = 0;

  int cnt = 0;
  std::string log_path;
  GlobalStats(){
    // std::cout << "hehe\n";
  }
  ~GlobalStats(){
    
    if (hist.size()){
      double sum = 0;
      std::cerr << "hist:";
      for(auto x : hist){
        sum += x.second;
      }
      double tmp = 0;
      bool flag = 0;
      std::cerr << "sum : " << sum << "\n";
      for(auto x: hist){
        // std::cerr << x.second << "\n";
        tmp += x.second;
        std::cerr << std::fixed << std::setprecision(6) << x.first << " " << x.second / sum << " " << x.second << "\n";
        if (tmp > sum * 0.99 && flag == 0){
          std::cerr << "99% : " << x.first + 1 << " " << tmp / sum << " " << "\n";
          flag = 1;
        }
      }
    }
    if (bad_query_count) {
      std::cerr << "bad query count: " << bad_query_count << "\n";
    }
  }
  void clear() {
    // 数值类型
    search_io = search_stage_1 = search_stage_2 = search_stage_3 = 0;
    search_read_neighbors_io = search_read_coord_io = search_all = 0;
    compute_deleted_ids = delete_scan1 = 0;
    delete_scan_count = delete_scan_hit = insert_scan_hit = 0;
    delete_io = update_io = update_ios = update_ios_topo = update_ios_coord = 0;
    delete_io1 = populate_deleted_nhoods = process_deletes = delete_io2 = delete_io3 = delete_cpu = 0;
    compute_pq_dists = consolidate_deletes = consolidate_deletes_actually = occlude_list_pq = delete_all = 0;
    insert_io = insert_cpu = insert_all = pq_tag_io = pq_cpu = write_tag_file_io = 0;
    merge_io = merge_cpu = merge_all = 0;
    insert_pq = insert_search = insert_prune = insert_buffer_io = 0;
    cache_count = cache_miss = cache_hit = cnt = 0;
    read_ios = write_ios = 0;
    insert_io1 = insert_io2 = insert_io3 = insert_io4 = insert_io5 = insert_io6 = insert_io7 = insert_io8 = 0;
    insert_ios1 = insert_ios2 = insert_ios3 = insert_ios4 = insert_ios5 = insert_ios6 = insert_ios7 = insert_ios8 = 0;
    update_effective_coord_ios = update_effective_topo_ios = 0;
    insert_time1 = insert_time2 = insert_time3 = insert_time4 = insert_time5 = insert_time6 = insert_time7 = insert_time8 = 0;
    bad_query_count = 0;
    // 容器
    recall.clear();
    qps.clear();
    rerank_ios.clear();
    search_latency.clear();
    update_latency.clear();
    insert_latency.clear();
    delete_latency.clear();
    merge_latency.clear();
    hits.clear();
    cache_hit_vec.clear();
    hist.clear();

    // 字符串
    log_path.clear();
  }

};

extern GlobalStats * gs;