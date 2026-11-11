set(CMAKE_CXX_STANDARD 17)

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME})

add_executable(search_memory_index search_memory_index.cpp)
target_link_libraries(search_memory_index ${PROJECT_NAME})

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME})

add_executable(gen_tags gen_tags.cpp)
target_link_libraries(gen_tags ${PROJECT_NAME})

add_executable(gt_update gt_update.cpp)
target_link_libraries(gt_update ${PROJECT_NAME} )

add_executable(convert_bin convert_bin.cpp)
target_link_libraries(convert_bin ${PROJECT_NAME} )

add_executable(change_pts change_pts.cpp)
target_link_libraries(change_pts ${PROJECT_NAME})

add_executable(overall_performance overall_performance.cpp)
target_link_libraries(overall_performance ${PROJECT_NAME})

add_executable(mixed_workload mixed_workload.cpp)
target_link_libraries(mixed_workload ${PROJECT_NAME})

add_executable(test_insert_search test_insert_search.cpp)
target_link_libraries(test_insert_search ${PROJECT_NAME})

add_executable(overall_perf_mem overall_perf_mem.cpp)
target_link_libraries(overall_perf_mem ${PROJECT_NAME})

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME})

add_executable(search_disk_index_mem search_disk_index_mem.cpp)
target_link_libraries(search_disk_index_mem ${PROJECT_NAME})

add_executable(pad_partition pad_partition.cpp)
target_link_libraries(pad_partition ${PROJECT_NAME} )

add_executable(split_index split_index.cpp)
target_link_libraries(split_index ${PROJECT_NAME} )

add_executable(reorder_topo reorder_topo.cpp)
target_link_libraries(reorder_topo ${PROJECT_NAME} )

add_executable(check_reorder_topo check_reorder_topo.cpp)
target_link_libraries(check_reorder_topo ${PROJECT_NAME} )

add_executable(reorder_by_map reorder_by_map.cpp)
target_link_libraries(reorder_by_map ${PROJECT_NAME} )

add_executable(shuffle_bin shuffle.cpp)
target_link_libraries(shuffle_bin ${PROJECT_NAME} )

add_executable(shuffle_dim shuffle2.cpp)
target_link_libraries(shuffle_dim ${PROJECT_NAME} )

# Harness driver for the benchmark harness's insert/delete/update workloads. It shares the harness's JSON metrics
# header (ANN_BENCH_INCLUDE_DIR, set in the top-level CMakeLists.txt).
add_executable(bench_insert_or_delete bench_insert_or_delete.cpp)
target_link_libraries(bench_insert_or_delete ${PROJECT_NAME})
