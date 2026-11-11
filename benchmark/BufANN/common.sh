#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/utils.sh"

# BufANN's C++ side is this repository itself (the repository root); the
# baselines are vendored under baselines/. Env-overridable for relocations.
BASELINE_REPO="${BUFANN_REPO:-$REPO_DIR}"
[[ -n "$BASELINE_REPO" && -d "$BASELINE_REPO" ]] || error \
"BufANN repo not found. Expected the repo root $REPO_DIR, or set BUFANN_REPO."

# Buffer pool size shared by the converter and the driver. The conversion
# routine uses this only to size scratch frames; the driver uses it as the
# runtime buffer pool, so query/insert/delete results are sensitive to it.
BUFANN_BUFFER_POOL_FRAMES="${BUFANN_BUFFER_POOL_FRAMES:-262144}"  # 1 GB at 4KB/page

bufann_detect_pq_chunks() {
    local prefix pivots chunk_offset rows
    prefix="$1"
    pivots="${prefix}_pq_pivots.bin"
    [[ -f "$pivots" ]] || return 1
    # New-format DiskANN PQ pivots start with a bin header followed by five
    # uint64 offsets. Offset[3] points at the chunk-offsets bin block, whose
    # row count is n_chunks + 1.
    chunk_offset="$(od -An -tu8 -j 32 -N 8 "$pivots" 2>/dev/null | awk '{print $1}')"
    [[ "$chunk_offset" =~ ^[0-9]+$ && "$chunk_offset" -gt 0 ]] || return 1
    rows="$(od -An -tu4 -j "$chunk_offset" -N 4 "$pivots" 2>/dev/null | awk '{print $1}')"
    [[ "$rows" =~ ^[0-9]+$ && "$rows" -gt 1 ]] || return 1
    echo "$(( rows - 1 ))"
}

bufann_resolve_pq_chunks() {
    local prefix detected configured
    prefix="$1"
    configured="${PQ_BYTES:-32}"
    if detected="$(bufann_detect_pq_chunks "$prefix")"; then
        if [[ "$detected" != "$configured" ]]; then
            note "BufANN PQ chunks from $prefix: $detected (overrides PQ_BYTES=$configured)"
        fi
        echo "$detected"
    else
        echo "$configured"
    fi
}

bufann_cache_dir_for_workload() {
    local workload canon dim degree_cap pq_chunks canon_id
    workload="${1:-${WORKLOAD:-}}"
    canon="$(resolve_canonical_prefix "$workload")"
    [[ -n "$canon" ]] || error "BufANN requires a canonical DiskANN index (WORKLOAD=${workload:-unset})"
    require_canonical "$canon"
    dim="${DIM:?DIM must be set by the dataset profile}"
    degree_cap="${CANONICAL_R:-${R:-64}}"
    pq_chunks="$(bufann_resolve_pq_chunks "$canon")"
    canon_id="$(basename "$canon")"
    printf '%s\n' "$EVAL_CACHED/$DATASET/BufANN/${canon_id}_${DATA_TYPE}_dim${dim}_R${degree_cap}_pq${pq_chunks}"
}

configure_bufann_repo() {
    local repo build_type target selected_cc selected_cxx compiler_libstdcpp compiler_lib_dir
    repo="$1"
    build_type="$2"
    shift 2
    choose_compiler
    setup_build_env
    selected_cc="${CC:-$(command -v gcc)}"
    selected_cxx="${CXX:-$(command -v g++)}"
    # As in configure_odinann_repo: keep the build's libstdc++ on the runtime
    # path, else binaries pick the system libstdc++ and fail GLIBCXX checks.
    compiler_libstdcpp="$($selected_cxx -print-file-name=libstdc++.so)"
    if [[ -n "$compiler_libstdcpp" && "$compiler_libstdcpp" != "libstdc++.so" && -e "$compiler_libstdcpp" ]]; then
        compiler_lib_dir="$(dirname "$(readlink -f "$compiler_libstdcpp")")"
        export LD_LIBRARY_PATH="$compiler_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # BUFANN_CMAKE_ARGS adds configure flags, e.g. -DBUFANN_PAGE_TABLE=array (word-split).
    # shellcheck disable=SC2086
    cmake -S "$repo" -B "$repo/build" \
        -DCMAKE_BUILD_TYPE:STRING="$build_type" \
        -DCMAKE_C_COMPILER="$selected_cc" \
        -DCMAKE_CXX_COMPILER="$selected_cxx" \
        ${BUFANN_CMAKE_ARGS:-}
    for target in "$@"; do
        cmake --build "$repo/build" --config "$build_type" --target "$target" -j"$(nproc)"
    done
}

# Convert the canonical DiskANN index for this workload into BufANN's
# native heap/meta/medoids/centroids family under $index_dir. Prefers the
# fast converter (file-level streaming) over the buffered one.
#
# Conversion is slow (full scan of the ~370 GB canonical), but its output is
# deterministic in (canonical identity, dtype, dim, R, pq_chunks) and shared
# across all four workloads that consume the same canonical. So it earns an
# eval_cached entry, restored via fastcopy of the whole index dir into
# eval_tempfiles (the family is opened O_RDWR by insert/delete - mutable
# files, not symlinks).
provision_index_into() {
    local index_dir prefix canon converter converter_src dim degree_cap pq_chunks frames cache_dir
    index_dir="$1"
    prefix="$index_dir/$DATASET"

    # Opt-in smoke/dev path: copy a prebuilt BufANN family. RQ scripts leave
    # BUFANN_PREBUILT_INDEX_DIR unset and keep the DiskANN convert path.
    if [[ -n "${BUFANN_PREBUILT_INDEX_DIR:-}" ]]; then
        local src="${BUFANN_PREBUILT_INDEX_DIR%/}"
        [[ -f "${src}/${DATASET}.heap" && -f "${src}/${DATASET}.meta" ]] || \
            error "BUFANN_PREBUILT_INDEX_DIR missing ${DATASET}.heap/.meta: $src"
        safe_mutable_path "$index_dir" >/dev/null
        mkdir -p "$index_dir"
        note "Using prebuilt BufANN index from $src -> $index_dir"
        # Mutable workload needs private copies (O_RDWR).
        # Heap family uses dots (sift1k.heap); PQ sidecars use underscores
        # (sift1k_pq_compressed.bin / sift1k_pq_pivots.bin*).
        for f in "$src"/"$DATASET".* "$src"/"$DATASET"_pq*; do
            [[ -e "$f" ]] || continue
            cp -f "$f" "$index_dir/"
        done
        [[ -f "${prefix}.heap" && -f "${prefix}.meta" ]] || \
            error "BufANN prebuilt copy incomplete: $prefix"
        return 0
    fi

    canon="$(resolve_canonical_prefix "${WORKLOAD:-}")"
    [[ -n "$canon" ]] || error "BufANN requires a canonical DiskANN index (WORKLOAD=${WORKLOAD:-unset})"
    require_canonical "$canon"
    dim="${DIM:?DIM must be set by the dataset profile}"
    degree_cap="${CANONICAL_R:-${R:-64}}"
    pq_chunks="$(bufann_resolve_pq_chunks "$canon")"
    frames="$BUFANN_BUFFER_POOL_FRAMES"
    cache_dir="$(bufann_cache_dir_for_workload "${WORKLOAD:-}")"

    safe_mutable_path "$index_dir" >/dev/null
    if eval_cache_try_restore "$cache_dir" "$index_dir"; then
        note "Reusing cached BufANN converted index -> $index_dir"
        return 0
    fi
    if [[ -n "${BUFANN_CONVERTER:-}" ]]; then
        converter="$BUFANN_CONVERTER"
        [[ -x "$converter" ]] || error "BufANN converter missing: $converter"
    else
        converter_src="$BASELINE_REPO/tests/fast_convert_diskann_to_inplaceann.cpp"
        converter="$BASELINE_REPO/build/tests/fast_convert_diskann_to_inplaceann"
        [[ -f "$converter_src" ]] || error "BufANN converter source missing: $converter_src"
        if [[ ! -x "$converter" || "$converter_src" -nt "$converter" ]]; then
            mkdir -p "$(dirname "$converter")"
            note "Building BufANN converter ($(basename "$converter_src"))"
            g++ "$converter_src" -O3 -std=c++17 -pthread -o "$converter"
        fi
        [[ -x "$converter" ]] || error "BufANN converter missing after build: $converter"
    fi
    mkdir -p "$index_dir"
    note "Converting canonical -> BufANN at $prefix (R=$degree_cap pq=$pq_chunks frames=$frames)"

    choose_compiler
    setup_build_env
    selected_cc="${CC:-$(command -v gcc)}"
    selected_cxx="${CXX:-$(command -v g++)}"
    # As in configure_odinann_repo: keep the build's libstdc++ on the runtime
    # path, else binaries pick the system libstdc++ and fail GLIBCXX checks.
    compiler_libstdcpp="$($selected_cxx -print-file-name=libstdc++.so)"
    if [[ -n "$compiler_libstdcpp" && "$compiler_libstdcpp" != "libstdc++.so" && -e "$compiler_libstdcpp" ]]; then
        compiler_lib_dir="$(dirname "$(readlink -f "$compiler_libstdcpp")")"
        export LD_LIBRARY_PATH="$compiler_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    "$converter" "$DATA_TYPE" "$canon" "$prefix" "$dim" "$degree_cap" "$pq_chunks" "$frames"
    [[ -f "${prefix}.heap" && -f "${prefix}.meta" ]] || \
        error "BufANN conversion incomplete: $prefix"
    eval_cache_store "$index_dir" "$cache_dir"
}

# Sample N rows from the canonical base prefix [0, pool_npts) into a warmup
# query file that bufann_driver consumes via --warmup_query_file. The pool is
# the 0.9B canonical prefix shared by all four workloads (post-workload active
# set always contains [0, BASE_POINTS)), so a single cache file covers
# query/insert/delete/update. Tunable via BUFANN_WARMUP_N / BUFANN_WARMUP_SEED.
ensure_bufann_warmup_queries() {
    local work_dir pool_npts n seed cache dst
    work_dir="$1"
    pool_npts="${2:?ensure_bufann_warmup_queries: pool_npts required}"
    n="${BUFANN_WARMUP_N:-1000000}"
    seed="${BUFANN_WARMUP_SEED:-0}"
    dst="$work_dir/warmup_queries.bin"
    # BUFANN_WARMUP_N=0 disables the warmup-query phase; clear any stale
    # symlink from a previous run so run_bufann_driver doesn't pick it up.
    if [[ "$n" == "0" ]]; then
        rm -f "$dst"
        note "BufANN warmup queries disabled (BUFANN_WARMUP_N=0)"
        return 0
    fi
    cache="$EVAL_CACHED/$DATASET/BufANN/warmup/pool${pool_npts}_n${n}_seed${seed}_${DATA_TYPE}.bin"
    mkdir -p "$(dirname "$dst")"
    if [[ ! -e "$cache" ]]; then
        safe_mutable_path "$cache" >/dev/null
        mkdir -p "$(dirname "$cache")"
        note "Sampling $n BufANN warmup queries from [0,$pool_npts) of $DATA_BIN (seed=$seed)"
        python3 "$BENCHMARK_DIR/sample_base_warmup.py" \
            "$DATA_BIN" "$DATA_TYPE" "$n" "$cache" \
            --seed "$seed" --npts-limit "$pool_npts"
    fi
    ln -sfn "$(readlink -f "$cache")" "$dst"
    note "BufANN warmup queries -> $dst"
}

# Run bufann_driver. Workload-specific args are appended by the caller via $@.
# Common args (data_type/dim/index_prefix/query/gt/...) live here so leaves
# stay short and consistent.
run_bufann_driver() {
    local index_prefix workload query_file gt_file output_dir search_l beamwidth query_threads recall_at
    index_prefix="$1"
    workload="$2"
    query_file="$3"
    gt_file="$4"
    output_dir="$5"
    search_l="$6"
    beamwidth="$7"
    query_threads="$8"
    recall_at="$9"
    shift 9
    mkdir -p "$output_dir"
    configure_bufann_repo "$BASELINE_REPO" "$BENCHMARK_CMAKE_BUILD_TYPE" bufann_driver
    # Optional cache warmup using a separate query file (typically rows sampled
    # from the base set so warmup doesn't overlap the timed queries). Resolved
    # in priority order: explicit env override, then the per-workload file
    # planted by ensure_bufann_warmup_queries at $WORK_DIR/warmup_queries.bin.
    local warmup_args=() warmup_file="" work_dir_guess
    if [[ -n "${BUFANN_WARMUP_QUERY_FILE:-}" && -f "$BUFANN_WARMUP_QUERY_FILE" ]]; then
        warmup_file="$BUFANN_WARMUP_QUERY_FILE"
    else
        work_dir_guess="$(dirname "$(dirname "$index_prefix")")"
        [[ -e "$work_dir_guess/warmup_queries.bin" ]] && \
            warmup_file="$work_dir_guess/warmup_queries.bin"
    fi
    if [[ -n "$warmup_file" ]]; then
        warmup_args=(
            --warmup_query_file "$warmup_file"
            --warmup_threads "${BUFANN_PRELOAD_THREADS:-$(nproc)}"
        )
        note "BufANN warmup: $warmup_file, ${BUFANN_PRELOAD_THREADS:-$(nproc)} threads"
    fi
    # Optional page-level preload (skips query work; just pins heap pages
    # into the buffer pool). BUFANN_PRELOAD_PAGES is the page budget in both
    # modes; BUFANN_PRELOAD_MODE picks which one:
    #   - sequential (default): pin pages [0, BUFANN_PRELOAD_PAGES). Cheap
    #     and parallelizable but ignores graph topology.
    #   - bfs: BFS from the entry-point medoid, pinning the first
    #     BUFANN_PRELOAD_PAGES distinct pages reached. Topology-aware;
    #     single-threaded. Driver refuses to combine bfs with sequential.
    local preload_args=() preload_mode="${BUFANN_PRELOAD_MODE:-sequential}"
    if [[ -n "${BUFANN_PRELOAD_PAGES:-}" && "$BUFANN_PRELOAD_PAGES" != "0" ]]; then
        case "$preload_mode" in
            bfs)
                preload_args+=(--bfs_warmup_pages "$BUFANN_PRELOAD_PAGES")
                preload_args+=(--preload_threads "${BUFANN_PRELOAD_THREADS:-$(nproc)}")
                note "BufANN preload: BFS warmup, $BUFANN_PRELOAD_PAGES pages, ${BUFANN_PRELOAD_THREADS:-$(nproc)} threads"
                ;;
            sequential)
                preload_args+=(--preload_max_pages "$BUFANN_PRELOAD_PAGES")
                preload_args+=(--preload_threads "${BUFANN_PRELOAD_THREADS:-$(nproc)}")
                note "BufANN preload: sequential, $BUFANN_PRELOAD_PAGES pages, ${BUFANN_PRELOAD_THREADS:-$(nproc)} threads"
                ;;
            *)
                error "BUFANN_PRELOAD_MODE must be sequential|bfs (got: $preload_mode)"
                ;;
        esac
    fi
    # Opt-in driver wrapper (e.g. a profiler). When BUFANN_DRIVER_WRAPPER is
    # set, its whitespace-split tokens are prefixed before the driver binary so
    # the wrapper scopes to the driver process alone (not cmake/build/tee).
    # Unset -> no behavior change. Example:
    #   BUFANN_DRIVER_WRAPPER="perf record -F 999 -o /tmp/p.data --"
    local driver_wrapper=()
    if [[ -n "${BUFANN_DRIVER_WRAPPER:-}" ]]; then
        read -ra driver_wrapper <<< "$BUFANN_DRIVER_WRAPPER"
    fi
    : "${NPTS_FULL:?NPTS_FULL must be set by the dataset profile for BufANN max_dataset_size}"
    local pq_chunks
    pq_chunks="$(bufann_resolve_pq_chunks "$index_prefix")"
    pq_chunks="${BUFANN_PQ_CHUNKS:-$pq_chunks}"
    # Compaction knobs (driver defaults: enabled=0, auto_trigger=1, ratio=0.50,
    # threads=1). Online eval sets AUTO_TRIGGER=0 plus dedicated flags.
    # One thread budget for Pass1 densify, Pass2 write shards, and D0/D1 merges.
    local compaction_args=(
        --compaction_enabled "${BUFANN_COMPACTION_ENABLED:-0}"
        --compaction_auto_trigger "${BUFANN_COMPACTION_AUTO_TRIGGER:-1}"
        --compaction_fragmentation_ratio "${BUFANN_COMPACTION_FRAGMENTATION_RATIO:-0.50}"
        --compaction_threads "${BUFANN_COMPACTION_THREADS:-1}"
    )
    "${driver_wrapper[@]}" "$BASELINE_REPO/build/tests/bufann_driver" \
        --data_type "$DATA_TYPE" \
        --index_prefix "$index_prefix" \
        --dim "$DIM" \
        --workload "$workload" \
        --query_file "$query_file" \
        --gt_file "$gt_file" \
        --recall_at "$recall_at" \
        --search_L "$search_l" \
        --beamwidth "$beamwidth" \
        --query_threads "$query_threads" \
        --maintenance_threads "${MAINTENANCE_THREADS:-${NTHREADS:-$(nproc)}}" \
        --buffer_pool_frames "$BUFANN_BUFFER_POOL_FRAMES" \
        --pq_chunks "$pq_chunks" \
        ${BUFANN_SEARCH_PQ:+--search_pq "$BUFANN_SEARCH_PQ"} \
        --R "${CANONICAL_R:-${R:-64}}" \
        --L "${L:-100}" \
        --result_file "$output_dir/result.json" \
        --max_dataset_size "${BUFANN_MAX_DATASET_SIZE:-$NPTS_FULL}" \
        "${compaction_args[@]}" \
        "${warmup_args[@]}" \
        "${preload_args[@]}" \
        "$@" 2>&1 | tee "$output_dir/run.log"
}
