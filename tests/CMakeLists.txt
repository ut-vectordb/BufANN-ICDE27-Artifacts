# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

add_executable(build_disk_index build_disk_index.cpp)
if(MSVC)
	target_link_options(build_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL "/INCLUDE:_tcmalloc")
	target_link_libraries(build_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
	target_link_libraries(build_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib ${PROJECT_SOURCE_DIR}/dependencies/windows/tcmalloc/libtcmalloc_minimal.lib)
else()
	target_link_libraries(build_disk_index ${PROJECT_NAME} -ltcmalloc aio)
endif()

add_executable(search_disk_index search_disk_index.cpp
    ${PROJECT_SOURCE_DIR}/src/aux_utils.cpp)
if(MSVC)
	target_link_options(search_disk_index PRIVATE /MACHINE:x64 /DEBUG:FULL)
	target_link_libraries(search_disk_index debug ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/diskann_dll.lib)
	target_link_libraries(search_disk_index optimized ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/diskann_dll.lib)
else()
	target_link_libraries(search_disk_index ${PROJECT_NAME} aio -ltcmalloc)
endif()

# Driver 1: build an index from a data file.
add_executable(bufann_index_builder bufann_index_builder.cpp)
target_link_libraries(bufann_index_builder ${PROJECT_NAME} aio -ltcmalloc)


# Driver 1b: import a canonical DiskANN disk index into inplaceann artifacts.
add_executable(convert_diskann_to_inplaceann convert_diskann_to_inplaceann.cpp)
set_target_properties(convert_diskann_to_inplaceann PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_link_libraries(convert_diskann_to_inplaceann ${PROJECT_NAME} aio -ltcmalloc)

# Driver 1c: fast (file-level streaming) variant of the converter; preferred by
# benchmark/BufANN/common.sh when present.
add_executable(fast_convert_diskann_to_inplaceann fast_convert_diskann_to_inplaceann.cpp)
set_target_properties(fast_convert_diskann_to_inplaceann PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
target_link_libraries(fast_convert_diskann_to_inplaceann ${PROJECT_NAME} aio -ltcmalloc)


# Driver 2: compute exact ground-truth nearest neighbours (with optional exclusion).
add_executable(bufann_gt_computer bufann_gt_computer.cpp)
target_link_libraries(bufann_gt_computer ${PROJECT_NAME} aio -ltcmalloc)

# Driver 3: use an index — search_only or search_after_update workloads.
add_executable(bufann_driver bufann_driver.cpp)
target_link_libraries(bufann_driver ${PROJECT_NAME} aio -ltcmalloc)


add_executable(tcmalloc_release_smoke tcmalloc_release_smoke.cpp)
target_link_libraries(tcmalloc_release_smoke ${PROJECT_NAME} aio -ltcmalloc)

add_executable(pq_memory_test pq_memory_test.cpp)
target_link_libraries(pq_memory_test ${PROJECT_NAME} aio -ltcmalloc)

add_executable(test_freshdiskann_delete test_freshdiskann_delete.cpp)
target_link_libraries(test_freshdiskann_delete ${PROJECT_NAME} aio -ltcmalloc)
add_executable(test_lazy_delete test_lazy_delete.cpp)
target_link_libraries(test_lazy_delete ${PROJECT_NAME} aio -ltcmalloc)
