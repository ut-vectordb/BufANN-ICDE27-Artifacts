#!/bin/bash

source "$(dirname "$0")/config.sh"

log_path=$LOG_PATH/fig5

mkdir -p $log_path

num_threads=8
for num_threads in 2 4 8 16 32
do
linux_aligned_file_reader=$PROJECT_PATH/src/utils/linux_aligned_file_reader.cpp
trunc_len=10000000
for sys in odinann
do
    K=10
    # 如果不用双pq，要把宏关掉，策略置零，文件拷贝对，同时rerank_search 用之前的版本
    # 跑不用双pq的策略都要替换rerank文件
    if [ "$sys" = "dgai" ]; then
        flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_DOUBLE_PQ -DUSE_NHOOD_CACHE" 
        # flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_NHOOD_CACHE" 
        # flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK " 
        search_mode=3 
        pipeline_width=32
        # strategy=7
        strategy=7 # 1 3 5 7
        L_mem=0
    else 
        flag=" -DBG_IO_THREAD"
        search_mode=0
        pipeline_width=4
        strategy=0
        L_mem=10
    fi

    for data_name in gist
    do

        L='100'
        batch_size=1000
    
        if [ "$data_name" = "sift100w" ]  || [ "$data_name" = "sift100w_shuffled" ] ; then
            L="50"
            L="150"
            L="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160"
        elif [ "$data_name" = "msong" ]; then
            L="100"
            L="30 50 70 90 110 130 160 190 220 250 280 310 340 370 400 450 500"
        elif [ "$data_name" = "text" ]; then
            L="100"
            L="300"
            L="30 50 70 90 110 130 150 170 190 210 230 250 270 290 310 350 400 450 500 550 600 650 700 750 800"
            # L="700 750 800"
        elif [ "$data_name" = "gist" ]; then
            L="400"
            L="1000"
            L="100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500"
        elif [ "$data_name" = "deep100m" ]; then
            L="800"
            L="50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000"
        elif [ "$data_name" = "deep10m" ]; then
            L="400"
            L="30 40 50 60 70 80 90 100 110 130 160 190 220 250 280 310 340 370 400 450 500"
        elif [ "$data_name" = "deep1m" ]; then
            L="150"
            L="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
        elif [ "$data_name" = "deep1b" ]; then
            L="400"
        fi

        trunc_len=$(get_trunc_len "$data_name")
        echo "trunc_len = "$trunc_len
        sed -i 's/\(trunc_len *= *\)[0-9]\+/\1'"$trunc_len"'/' $linux_aligned_file_reader 
        cd $PROJECT_PATH/build
        export ADDITIONAL_DEFINITIONS="$flag"
        cmake ..
        make -j
        cd "$PROJECT_PATH"

        data_type=float
        data="${DATA_PATH}/${data_name}/${data_name}_base.fbin"
        query="${DATA_PATH}/${data_name}/${data_name}_query.fbin"
        truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_gt_0.00125
        gt=${DATA_PATH}/${data_name}/${data_name}_gt_0.00125/gt_0.bin 

        # original_index_path=${DISKANN_INDEX_PATH}/${data_name}/original_index
        original_index_dgai_path=${DISKANN_INDEX_PATH}/${data_name}/original_index
        index_path_prefix=${OUTPUT_PATH}/${data_name}/${data_name}
        l_disk=128

        rm $OUTPUT_PATH/${data_name}/*.tags

        if [ "$sys" = "dgai" ]; then
            # cp $original_index_dgai_path/${data_name}_pq_compressed.bin "$OUTPUT_PATH"/${data_name}/
            # cp $original_index_dgai_path/${data_name}_pq_pivots.bin "$OUTPUT_PATH"/${data_name}/

            cp $original_index_dgai_path/${data_name}_pq_compressed_refined.bin "$OUTPUT_PATH"/${data_name}/${data_name}_pq_compressed.bin
            cp $original_index_dgai_path/${data_name}_pq_pivots.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_pq_pivots_refined.bin "$OUTPUT_PATH"/${data_name}/${data_name}_pq_pivots_2.bin
            cp $original_index_dgai_path/${data_name}_map.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/reordered_disk_index_graph_2 "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/reorder_map_graph_2 "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/disk_index_data "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/disk_index_graph "$OUTPUT_PATH"/${data_name}/
            # dd if=$original_index_dgai_path/${data_name}_disk.index of=$OUTPUT_PATH/${data_name}/${data_name}_disk.index bs=1 count=4096
            cp $original_index_dgai_path/${data_name}_disk.index "$OUTPUT_PATH"/${data_name}/ # 需要bfs_cache所以需要原始索引
        else 
            cp $original_index_dgai_path/${data_name}_disk.index "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_pq_compressed.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_pq_pivots.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_mem.index "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_mem.index.data "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_mem.index.tags "$OUTPUT_PATH"/${data_name}/
        fi

        # log_file_path=${log_path}/_.log
        
        sys_name=$(get_standard_sys_name "$sys")
        dataset_name=$(get_standard_dataset_name "$data_name")
        log_file_path=${log_path}/${sys_name}_${dataset_name}_$num_threads.log

        cmd="${PROJECT_PATH}/build/tests/search_disk_index  \
            ${data_type} \
            ${index_path_prefix} \
            ${num_threads} \
            ${pipeline_width} \
            ${query} \
            ${gt} \
            ${K} \
            l2 \
            ${search_mode} \
            ${L_mem} \
            ${strategy} \
            ${L} \
            2>&1 | tee ${log_file_path}"
            
        echo ${cmd}
        eval ${cmd} 2>&1 | tee ${log_file_path}

    done
done
done