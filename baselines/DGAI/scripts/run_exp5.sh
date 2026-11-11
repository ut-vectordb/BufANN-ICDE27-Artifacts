#!/bin/bash

source "$(dirname "$0")/config.sh"

log_path=$LOG_PATH/fig7
batch_num=100
mkdir -p $log_path

linux_aligned_file_reader=$PROJECT_PATH/src/utils/linux_aligned_file_reader.cpp
trunc_len=10000000
for sys in dgai odinann
do
    K=10
    # 如果不用双pq，要把宏关掉，策略置零，文件拷贝对，同时rerank_search 用之前的版本
    if [ "$sys" = "dgai" ]; then
        flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_DOUBLE_PQ -DCOLLECT_INFO_2" 
        # flag="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK " 
        search_mode=3 
        pipeline_width=32
        # strategy=7
        strategy=23
    else 
        flag=" -DBG_IO_THREAD -DCOLLECT_INFO_2"
        search_mode=0
        pipeline_width=4
        strategy=0
    fi

    for data_name in sift100w_shuffled #gist
    do

        L='100'
        batch_size=1000
    
        trunc_len=$(get_trunc_len "$data_name")
        if [ "$data_name" = "sift100w" ]  || [ "$data_name" = "sift100w_shuffled" ] ; then
            L="50"
        elif [ "$data_name" = "msong" ]; then
            L="100"
        elif [ "$data_name" = "text" ]; then
            L="100"
            L="300"
        elif [ "$data_name" = "gist" ]; then
            L="400"
        elif [ "$data_name" = "deep100m" ]; then
            L="200"
            batch_size=100000
        elif [ "$data_name" = "deep10m" ]; then
            L="150"
            batch_size=10000
        elif [ "$data_name" = "deep1m" ]; then
            L="50"
        elif [ "$data_name" = "deep1b" ]; then
            L="400"
            batch_size=1000000
        fi

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

        # original_index_path=${DISKANN_INDEX_PATH}/${data_name}/original_index
        original_index_dgai_path=${DISKANN_INDEX_PATH}/${data_name}/original_index
        index_path_prefix=${OUTPUT_PATH}/${data_name}/${data_name}
        l_disk=128

        rm $OUTPUT_PATH/${data_name}/*.tags

        if [ "$sys" = "dgai" ]; then
            # cp $original_index_path/${data_name}_pq_compressed.bin "$OUTPUT_PATH"/${data_name}/
            # cp $original_index_path/${data_name}_pq_pivots.bin "$OUTPUT_PATH"/${data_name}/

            cp $original_index_dgai_path/${data_name}_pq_compressed_refined.bin "$OUTPUT_PATH"/${data_name}/${data_name}_pq_compressed.bin
            cp $original_index_dgai_path/${data_name}_pq_pivots.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_pq_pivots_refined.bin "$OUTPUT_PATH"/${data_name}/${data_name}_pq_pivots_2.bin
            cp $original_index_dgai_path/${data_name}_map.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/reordered_disk_index_graph_2 "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/reorder_map_graph_2 "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/disk_index_data "$OUTPUT_PATH"/${data_name}/
            dd if=$original_index_dgai_path/${data_name}_disk.index of=$OUTPUT_PATH/${data_name}/${data_name}_disk.index bs=1 count=4096
        else 
            cp $original_index_dgai_path/${data_name}_disk.index "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_pq_compressed.bin "$OUTPUT_PATH"/${data_name}/
            cp $original_index_dgai_path/${data_name}_pq_pivots.bin "$OUTPUT_PATH"/${data_name}/

            
        fi

        # log_file_path=${log_path}/_.log
        
        sys_name=$(get_standard_sys_name "$sys")
        dataset_name=$(get_standard_dataset_name "$data_name")
        log_file_path=${log_path}/${sys_name}_${dataset_name}.log

        cmd="${PROJECT_PATH}/build/tests/mixed_workload  \
            ${data_type} \
            ${data} \
            ${l_disk} \
            ${index_path_prefix} \
            ${query} \
            ${truthset_prefix} \
            ${K} \
            ${pipeline_width} \
            ${batch_num} \
            ${batch_size} \
            ${search_mode} \
            ${strategy} \
            ${L}
            "
        echo ${cmd}
        eval ${cmd} 2>&1 | tee ${log_file_path}

    done
done