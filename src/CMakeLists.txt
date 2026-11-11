# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
        partition_and_pq.cpp pq_flash_index.cpp logger.cpp distance.cpp
		 utils.cpp
        bufann/inplace_io_stats.cpp bufann/buffer_pool/file_manager.cpp
        bufann/buffer_pool/buffer_pool.cpp bufann/buffer_pool/buffer_pool_io.cpp
        bufann/node_meta_table.cpp bufann/inplace_backend.cpp bufann/inplace_graph_ops.cpp bufann/lazy_delete.cpp
        bufann/bufann_api.cpp)
	add_library(${PROJECT_NAME} ${CPP_SOURCES})
	add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()
install()
