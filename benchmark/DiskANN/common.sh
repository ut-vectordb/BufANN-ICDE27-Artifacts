#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/utils.sh"

BASELINE_REPO="$REPO_DIR/baselines/DiskANN"

choose_diskann_compiler() {
    local cxx cc
    cxx="$1"
    cc="$2"
    export CXX="$cxx"
    export CC="$cc"
}

configure_diskann_repo() {
    local repo cxx cc
    repo="$1"
    cxx="$2"
    cc="$3"
    shift 3
    choose_diskann_compiler "$cxx" "$cc"
    configure_cmake_repo "$repo" "$@"
}

# DiskANN consumes the canonical index directly; no sliced base dataset is
# materialized for index construction.
provision_index_into() {
    local canon index_dir dataset index_prefix
    index_dir="$1"
    dataset="$2"
    canon="$(resolve_canonical_prefix "${WORKLOAD:-}")"
    [[ -n "$canon" ]] || error "no canonical index configured for $dataset/${WORKLOAD:-unknown}"
    index_prefix="$index_dir/$dataset"
    require_canonical "$canon"
    if canonical_family_present "$index_prefix"; then
        note "Reusing staged DiskANN canonical index $index_prefix"
        return 0
    fi
    note "Materializing DiskANN canonical (fastcopy) $canon -> $index_prefix"
    materialize_canonical_family "$canon" "$index_dir" "$dataset"
    canonical_family_present "$index_prefix" || \
        error "DiskANN canonical materialization incomplete: $index_prefix"
}

run_query_on_index() {
    local repo index_dir dataset data_type npts query_bin gt_file output_dir query_threads beamwidth recall_at metric search_l nodes_to_cache cxx cc index_prefix
    repo="$1"
    index_dir="$2"
    dataset="$3"
    data_type="$4"
    npts="$5"
    query_bin="$6"
    gt_file="$7"
    output_dir="$8"
    query_threads="$9"
    beamwidth="${10}"
    recall_at="${11}"
    metric="${12}"
    search_l="${13}"
    nodes_to_cache="${14}"
    cxx="${15}"
    cc="${16}"
    index_prefix="$index_dir/$dataset"
    mkdir -p "$output_dir"
    configure_diskann_repo "$repo" "$cxx" "$cc" search_disk_index
    ensure_identity_tags "$npts" "$index_prefix"
    local ann_bench_phase="${ANN_BENCH_PHASE:-query}"
    [[ "$output_dir" == */final_query ]] && ann_bench_phase="${ANN_BENCH_PHASE:-final_query}"
    ANN_BENCH_PHASE="$ann_bench_phase" "$repo/build/tests/search_disk_index" \
        "$data_type" "$index_prefix" \
        0 1 \
        "$nodes_to_cache" "$query_threads" "$beamwidth" \
        "$query_bin" "$gt_file" \
        "$recall_at" "$output_dir/${dataset}_result" \
        "$metric" \
        $search_l 2>&1 | tee "$output_dir/run.log"
}
