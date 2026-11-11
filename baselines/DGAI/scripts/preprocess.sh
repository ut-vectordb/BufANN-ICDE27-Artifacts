#!/bin/bash

source "$(dirname "$0")/config.sh"

for type in doublepq #512 #error_refine 512  original
do
    pq_dim=64
    aux_utils_cpp_path=$PROJECT_PATH/src/utils/aux_utils.cpp
    
    if [ "$type" = "doublepq" ]; then
        sed -i 's/int type = [0-9];/int type = 1;/' $aux_utils_cpp_path
    elif [ "$type" = "error_refine" ]; then
        sed -i 's/int type = [0-9];/int type = 2;/' $aux_utils_cpp_path
    elif [ "$type" = "512" ]; then
        sed -i 's/int type = [0-9];/int type = 3;/' $aux_utils_cpp_path
    else
        sed -i 's/int type = [0-9];/int type = 0;/' $aux_utils_cpp_path
    fi
    
    sed -i "s/size_t num_pq_chunks\s*=\s*[0-9]*;/size_t num_pq_chunks = ${pq_dim};/g" "$aux_utils_cpp_path"

    for data_name in deep100m
    do
        npts=100000000

        batch_npts=80000000
        full_data=${DATA_PATH}/${data_name}/${data_name}_base.fbin
        query=${DATA_PATH}/${data_name}/${data_name}_query.fbin
        topk=10000
        gt=${DATA_PATH}/${data_name}/${data_name}_gt_K${topk}
        target_topk=10
        data_type=float

        # cd $PROJECT_PATH/build
        # cmake  ..
        # make -j
        # cd $PROJECT_PATH

        # ${PROJECT_PATH}/build/tests/utils/compute_groundtruth $data_type $full_data $query $topk $gt
        # for update_ratio in 0.001 0.002 0.004 0.008 0.01
        for update_ratio in 0.00125
        do
            truthset_prefix=${DATA_PATH}/${data_name}/${data_name}_gt_${update_ratio}
            # ${PROJECT_PATH}/build/tests/gt_update $gt $npts $batch_npts $target_topk $truthset_prefix $update_ratio 0
        done
        # exit 0
        # ${PROJECT_PATH}/build/tests/change_pts $data_type $full_data $batch_npts
        
        original_index_path=${DISKANN_INDEX_PATH}/$data_name/original_index
        mkdir -p ${original_index_path}
        data=$full_data$batch_npts
        index_path_prefix=${original_index_path}/${data_name}

        # $PROJECT_PATH/build/tests/utils/gen_random_slice $data_type $data ${index_path_prefix}_SAMPLE_RATE_0.01 0.01

        # $PROJECT_PATH/build/tests/build_memory_index $data_type ${index_path_prefix}_SAMPLE_RATE_0.01_data.bin ${index_path_prefix}_SAMPLE_RATE_0.01_ids.bin ${index_path_prefix}_mem.index 0 0 32 64 1.2 24 l2
        # continue
        R=32
        B=64
        M=64
        Lbuild=75
        metric=l2
        single_file_index=0
        
        cmd="$PROJECT_PATH/build/tests/build_disk_index \
            ${data_type}  \
            ${data} \
            ${index_path_prefix} \
            ${R} \
            ${Lbuild} \
            ${B} \
            ${M} \
            32 \
            ${metric} \
            ${single_file_index} \
            "
        echo "BUILD cmd:"
        echo $cmd
        # eval $cmd


        index_path=$original_index_path/$data_name"_disk.index"

        index_path=$original_index_path/$data_name"_disk.index"
        topo_path=$original_index_path/dram_index_graph
        aligned_topo_path=$original_index_path/disk_index_graph
        coord_path=$original_index_path/disk_index_data

        $PROJECT_PATH/build/tests/split_index $index_path $topo_path $aligned_topo_path $coord_path $data_type
        
        # $PROJECT_PATH/build/tests/reorder_by_map $batch_npts 132 $aligned_topo_path $original_index_path/reorder_map_graph_2 $original_index_path/reordered_disk_index_graph_2
        # $PROJECT_PATH/build/tests/create_reverse_graph $index_path
        # $PROJECT_PATH/build/tests/reorder_by_map $batch_npts $ndim $coord_path $original_index_path/reorder_map_data_2 $original_index_path/reordered_disk_index_data_2
        
    done
done