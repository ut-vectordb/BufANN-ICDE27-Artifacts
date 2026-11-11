#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/utils.sh"

# DGAI's C++ is vendored under baselines/DGAI. Env-overridable for
# relocations.
BASELINE_REPO="${DGAI_REPO:-$REPO_DIR/baselines/DGAI}"
[[ -n "$BASELINE_REPO" && -d "$BASELINE_REPO" ]] || error \
"DGAI repo not found. Expected $REPO_DIR/baselines/DGAI, or set DGAI_REPO."

# DGAI selects its search path with compile-time flags, so the query binaries
# get their own build directory. The repo's build/ holds the index-build tools,
# which must not be relinked against a library compiled with these flags.
# The flags match the authors' DGAI configuration in scripts/run_exp1.sh.
DGAI_QUERY_BUILD_SUBDIR="build-query"
DGAI_QUERY_DEFINITIONS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK -DUSE_DOUBLE_PQ -DCOLLECT_INFO_2"

configure_dgai_repo() {
    local repo build_dir definitions target compiler_libstdcpp
    repo="$1"
    build_dir="$2"
    definitions="$3"
    shift 3
    choose_compiler
    setup_build_env
    export CC="${CC:-$(command -v gcc)}" CXX="${CXX:-$(command -v g++)}"
    compiler_libstdcpp="$($CXX -print-file-name=libstdc++.so)"
    if [[ "$compiler_libstdcpp" != "libstdc++.so" && -e "$compiler_libstdcpp" ]]; then
        export LD_LIBRARY_PATH="$(dirname "$(readlink -f "$compiler_libstdcpp")")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # CMakeLists.txt reads the flags from this variable at configure time, so
    # every call reconfigures with it set.
    ADDITIONAL_DEFINITIONS="$definitions" cmake -S "$repo" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE:STRING="$BENCHMARK_CMAKE_BUILD_TYPE" \
        -DUSE_AIO=ON \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
    for target in "$@"; do
        cmake --build "$build_dir" --target "$target" -j"$(nproc)"
    done
}

# DGAI_STRATEGY comes from benchmark/datasets/<DATASET>.sh. Its bit 1 (topology reorder) decides which graph
# file DGAI opens: strategy 23 reads reordered_disk_index_graph_2 through reorder_map_graph_2, strategy 21 reads
# the plain disk_index_graph (linux_aligned_file_reader.cpp). Both run the authors' rerank search with double
# PQ and the topology buffer.
: "${DGAI_STRATEGY:?DGAI_STRATEGY unset in benchmark/datasets/$DATASET.sh}"

# Inserts lock the new vector's page only around its read-modify-write instead of for the whole insert
# (DGAI_NARROW_COORD_LOCK in src/update/direct_to_topo_insert.cpp). Upstream holds that lock for the whole
# insert, which serializes consecutive inserts appending to the same tail page of the vector file.
# Set DGAI_NARROW_COORD_LOCK=0 to run upstream's locking.
export DGAI_NARROW_COORD_LOCK="${DGAI_NARROW_COORD_LOCK:-1}"

# SSDIndex::load() sizes the PQ codes and the id and page maps for NPTS_FULL points up front, as OdinANN does, so no
# insert grows them inside the timed loop. Without it, sift1b's first insert grew the 28.8 GB PQ array (~25 s).
export NPTS_FULL

# The canonical files the graph needs, as "<canonical path> <name DGAI expects>" lines.
dgai_graph_files() {
    local name="$1"
    if (( (DGAI_STRATEGY >> 1) & 1 )); then
        echo "$name.layout/reordered_disk_index_graph_2 reordered_disk_index_graph_2"
        echo "$name.layout/reorder_map_graph_2 reorder_map_graph_2"
    else
        echo "$name.layout/disk_index_graph disk_index_graph"
    fi
}

# Every canonical file DGAI reads, as "<canonical path> <name DGAI expects>" lines. DGAI finds everything from
# one prefix, $index_dir/$dataset: the PQ files and the header sit at the prefix, and the split graph and vectors
# must sit in the same directory under fixed names. The renames follow the authors' run scripts. DGAI's primary
# PQ code file is _pq_compressed_refined.bin, the hybrid encoding, and the refined codebook is its second
# codebook. The loader also reads a second code file. Its only use in search is compiled out (an if(0) block in
# rerank_search.cpp), but the loader takes the index's point count from it, so it must be a full-size code file.
# It gets the original (non-refined) codes, which is what that slot held upstream.
dgai_canonical_files() {
    local name="$1" dataset="$2"
    echo "${name}_disk.index_header ${dataset}_disk.index"
    echo "PQ/${name}_pq_compressed_refined.bin ${dataset}_pq_compressed.bin"
    echo "PQ/${name}_pq_pivots.bin ${dataset}_pq_pivots.bin"
    echo "PQ/${name}_pq_pivots_refined.bin ${dataset}_pq_pivots_2.bin"
    echo "PQ/${name}_pq_compressed.bin ${dataset}_pq_compressed_2.bin"
    echo "PQ/${name}_map.bin ${dataset}_map.bin"
    echo "$name.layout/disk_index_data disk_index_data"
    dgai_graph_files "$name"
}

# Fail unless the canonical has every file DGAI reads under this dataset's strategy.
require_dgai_canonical() {
    local dir="$1" name="$2" src dst
    while read -r src dst; do
        [[ -f "$dir/$src" ]] || error \
"DGAI canonical incomplete: missing $dir/$src
       The benchmark consumes a prebuilt DGAI index and never builds it (see benchmark/datasets/$DATASET.sh
       and README-datasets.md)."
    done < <(dgai_canonical_files "$name" "$DATASET")
}

# Materialize DGAI's native layout under $index_dir as real, owner-writable files, one straight copy per file
# (dgai_canonical_files). Nothing goes through eval_cached, since there is no slow conversion worth caching.
provision_index_into() {
    local index_dir dataset canon_dir name fcp src dst
    index_dir="$1"
    dataset="$2"
    canon_dir="$3"
    name="$4"
    require_dgai_canonical "$canon_dir" "$name"
    safe_mutable_path "$index_dir" >/dev/null
    mkdir -p "$index_dir"
    fcp="${FASTCOPY:-$BENCHMARK_DIR/fastcopy.sh}"
    while read -r src dst; do
        "$fcp" "$canon_dir/$src" "$index_dir/$dst" || error "fastcopy failed: $canon_dir/$src -> $index_dir/$dst"
    done < <(dgai_canonical_files "$name" "$dataset")
    chmod -R u+w "$index_dir"
}

# Sample DGAI_WARMUP_N base vectors (default 100000, seed 0) from the canonical base [0, NPTS_BASE) into
# $work_dir/warmup_queries.bin, with the same sampler BufANN uses. query.sh hands the file to search_disk_index
# as DGAI_WARMUP_QUERY_FILE, which runs it before each L instead of upstream's warmup on the timed queries.
# BufANN also re-warms before each L; the Jun22 batch passes BUFANN_WARMUP_N through so both use the same count.
ensure_dgai_warmup_queries() {
    local work_dir="$1" n="${DGAI_WARMUP_N:-100000}" dst
    dst="$work_dir/warmup_queries.bin"
    rm -f "$dst"
    if (( n == 0 )); then
        note "DGAI warmup queries disabled (DGAI_WARMUP_N=0)"
        return 0
    fi
    python3 "$BENCHMARK_DIR/sample_base_warmup.py" \
        "$DATA_BIN" "$DATA_TYPE" "$n" "$dst" --seed 0 --npts-limit "$NPTS_BASE" || error "warmup sampling failed"
    note "DGAI warmup queries ($n base vectors) -> $dst"
}

run_query_on_index() {
    local repo index_dir dataset data_type query_bin gt_file output_dir query_threads beamwidth recall_at metric strategy search_l build_dir
    repo="$1"
    index_dir="$2"
    dataset="$3"
    data_type="$4"
    query_bin="$5"
    gt_file="$6"
    output_dir="$7"
    query_threads="$8"
    beamwidth="$9"
    recall_at="${10}"
    metric="${11}"
    strategy="${12}"
    search_l="${13}"
    build_dir="$repo/$DGAI_QUERY_BUILD_SUBDIR"
    mkdir -p "$output_dir"
    configure_dgai_repo "$repo" "$build_dir" "$DGAI_QUERY_DEFINITIONS" search_disk_index
    # argv: <type> <prefix> <threads> <beamwidth> <query> <gt> <K> <metric>
    #       <search_mode> <mem_L> <strategy> <L...>
    # search_mode 3 is RERANK_SEARCH; mem_L 0 disables the in-memory index.
    "$build_dir/tests/search_disk_index" \
        "$data_type" "$index_dir/$dataset" \
        "$query_threads" "$beamwidth" \
        "$query_bin" "$gt_file" \
        "$recall_at" "$metric" \
        3 0 "$strategy" \
        $search_l 2>&1 | tee "$output_dir/run.log"
    # search_disk_index exits 0 even when the index fails to load, so check
    # that the results table was printed.
    grep -q 'Recall@' "$output_dir/run.log" || error "DGAI query produced no results; see $output_dir/run.log"
}

# Run one insert or delete workload with the harness driver
# (tests/bench_insert_or_delete.cpp in the DGAI repo). The driver mutates the
# index in place, measures per-round recall on the live index, and ends with a
# post-update query pass over the full query set. It emits the same ann_bench
# JSON records as the OdinANN driver.
run_update_on_index() {
    local repo mode data_type data_bin l_disk index_dir dataset n_iters count range beamwidth nthreads ids_file query_bin gt_dir query_threads recall_at search_l strategy output_dir build_dir
    repo="$1"
    mode="$2"
    data_type="$3"
    data_bin="$4"
    l_disk="$5"
    index_dir="$6"
    dataset="$7"
    n_iters="$8"
    count="$9"
    range="${10}"
    beamwidth="${11}"
    nthreads="${12}"
    ids_file="${13}"
    query_bin="${14}"
    gt_dir="${15}"
    query_threads="${16}"
    recall_at="${17}"
    search_l="${18}"
    strategy="${19}"
    output_dir="${20}"
    build_dir="$repo/$DGAI_QUERY_BUILD_SUBDIR"
    configure_dgai_repo "$repo" "$build_dir" "$DGAI_QUERY_DEFINITIONS" bench_insert_or_delete
    # search_mode 3 is RERANK_SEARCH, matching the query workload.
    "$build_dir/tests/bench_insert_or_delete" \
        "$mode" "$data_type" "$data_bin" \
        "$l_disk" "$index_dir/$dataset" \
        "$n_iters" "$count" "$range" "$beamwidth" "$nthreads" \
        "$ids_file" \
        "$query_bin" "$gt_dir" "$query_threads" "$recall_at" "$search_l" \
        3 "$strategy" 2>&1 | tee "$output_dir/run.log"
    grep -q '"phase":"workload"' "$output_dir/run.log" || error "DGAI $mode produced no workload record; see $output_dir/run.log"
}
