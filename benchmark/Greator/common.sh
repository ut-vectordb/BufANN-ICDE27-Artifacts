#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/utils.sh"

BASELINE_REPO="$REPO_DIR/baselines/Greator"

choose_greator_compiler() {
    local cxx cc
    cxx="$1"
    cc="$2"
    export CXX="$cxx"
    export CC="$cc"
}

configure_greator_repo() {
    local repo cxx cc
    repo="$1"
    cxx="$2"
    cc="$3"
    shift 3
    choose_greator_compiler "$cxx" "$cc"
    configure_cmake_repo "$repo" "$@"
}

# Materialize the canonical index and produce Greator's topology sidecar.
# No sliced base dataset is materialized for index construction.
provision_index_into() {
    local canon repo index_dir dataset data_type index_prefix sidecar src bin idx range elem sector cache npts
    repo="$1"
    index_dir="$2"
    dataset="$3"
    data_type="$4"
    canon="$(resolve_canonical_prefix "${WORKLOAD:-}")"
    [[ -n "$canon" ]] || error "no canonical index configured for $dataset/${WORKLOAD:-unknown}"
    index_prefix="$index_dir/$dataset"
    sidecar="${index_prefix}_disk.index_with_only_nbrs"
    require_canonical "$canon"
    if canonical_family_present "$index_prefix" && [[ -s "$sidecar" ]]; then
        note "Reusing materialized Greator canonical index $index_prefix"
        return 0
    fi
    note "Materializing Greator canonical (fastcopy) $canon -> $index_prefix"
    materialize_canonical_family "$canon" "$index_dir" "$dataset"
    canonical_family_present "$index_prefix" || \
        error "Greator canonical materialization incomplete: $index_prefix"
    rm -f "$sidecar"
    # topology_extraction argv: <indir> <outdir> <SECTOR_LEN> <range=R> <elem_size>
    #  - range MUST be the canonical's real max degree (R in the prefix), not a
    #    leaf build tunable; an R mismatch reads the wrong neighbor stride.
    #  - elem_size MUST match the dataset element size (uint8/int8=1, float=4);
    #    it defaults to sizeof(float) in the binary, so it must be passed for
    #    non-float datasets or the per-node stride is wrong.
    : "${CANONICAL_R:?CANONICAL_R must be set by the dataset config to consume a canonical with Greator}"
    range="$CANONICAL_R"
    sector="${CANONICAL_SECTOR_LEN:-4096}"
    case "$data_type" in
        uint8|int8) elem=1 ;;
        float)      elem=4 ;;
        *) error "Greator topology extraction: unsupported data_type '$data_type' (need uint8|int8|float)" ;;
    esac
    idx="${index_prefix}_disk.index"
    # The topology sidecar depends on (canonical, R, sector, elem). The 1B
    # (query/delete) and 0.9B (insert/update) canonicals share dataset/R/sector/
    # elem, so the cache key MUST also include npts to discriminate them — same
    # convention as BufANN (pool${pool_npts}).
    # npts comes from the materialized canonical's .tags (npts == row count).
    npts="$(bin_count "${index_prefix}_disk.index.tags")"
    cache="$EVAL_CACHED/$DATASET/Greator/topology/${dataset}_n${npts}_R${range}_s${sector}_e${elem}_with_only_nbrs"
    if eval_cache_try_restore "$cache" "$sidecar"; then
        :
    else
        if [[ -f "$repo/scripts/pre_dataset/fast_topology_extraction.cpp" ]]; then
            src="$repo/scripts/pre_dataset/fast_topology_extraction.cpp"
            bin="$repo/build/fast_topology_extraction"
        else
            src="$repo/scripts/pre_dataset/topology_extraction.cpp"
            bin="$repo/build/topology_extraction"
        fi
        [[ -f "$src" ]] || error "Greator topology extractor source missing: $src"
        if [[ ! -x "$bin" || "$src" -nt "$bin" ]]; then
            mkdir -p "$(dirname "$bin")"
            note "Building Greator topology extractor ($(basename "$src"))"
            g++ "$src" -O3 -std=c++17 -o "$bin"
        fi
        [[ -x "$bin" ]] || error "Greator topology extractor missing after build: $bin"
        note "Extracting Greator topology sidecar (R=$range sector=$sector elem=$elem) -> $sidecar"
        "$bin" "$idx" "$sidecar" "$sector" "$range" "$elem"
        [[ -s "$sidecar" ]] || \
            error "Greator topology extraction produced no sidecar: $sidecar"
        eval_cache_store "$sidecar" "$cache"
    fi
    [[ -s "$sidecar" ]] || \
        error "Greator topology sidecar missing after provision: $sidecar"
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
    configure_greator_repo "$repo" "$cxx" "$cc" search_disk_index
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
