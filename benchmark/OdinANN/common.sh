#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/utils.sh"

BASELINE_REPO="$REPO_DIR/baselines/OdinANN"

export NPTS_FULL

configure_odinann_repo() {
    local repo use_aio build_type target selected_cc selected_cxx uring_dir
    local compiler_libstdcpp compiler_lib_dir
    repo="$1"
    use_aio="$2"
    build_type="$3"
    shift 3
    choose_compiler
    setup_build_env
    selected_cc="${CC:-$(command -v gcc)}"
    selected_cxx="${CXX:-$(command -v g++)}"
    # Mirror configure_cmake_repo: when built with a newer gcc (gcc-15/14/11
    # via choose_compiler), the binary needs that toolchain's libstdc++ at
    # runtime. Without this the system libstdc++ (GCC 9.4) is picked and
    # search_disk_index fails: "GLIBCXX_3.4.32 not found".
    compiler_libstdcpp="$($selected_cxx -print-file-name=libstdc++.so)"
    if [[ -n "$compiler_libstdcpp" && "$compiler_libstdcpp" != "libstdc++.so" && -e "$compiler_libstdcpp" ]]; then
        compiler_lib_dir="$(dirname "$(readlink -f "$compiler_libstdcpp")")"
        export LD_LIBRARY_PATH="$compiler_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    uring_dir="$repo/third_party/liburing"
    if [[ "$use_aio" != "ON" && -d "$uring_dir" && ! -f "$uring_dir/src/liburing.a" ]]; then
        (cd "$uring_dir" && ./configure && make -j"$(nproc)")
    fi
    cmake -S "$repo" -B "$repo/build" \
        -DCMAKE_BUILD_TYPE:STRING="$build_type" \
        -DUSE_AIO="$use_aio" \
        -DCMAKE_C_COMPILER="$selected_cc" \
        -DCMAKE_CXX_COMPILER="$selected_cxx"
    for target in "$@"; do
        cmake --build "$repo/build" --config "$build_type" --target "$target" -j"$(nproc)"
    done
}

# Consume the per-workload canonical via
# convert_diskann_to_pipeann (OdinANN rewrites the disk.index header, so
# the converted family must be real files - this copy is necessary, not
# avoidable by symlink). No sliced base dataset is materialized.
#
# The bulk duplication is done by fastcopy.sh (parallel O_DIRECT, several x
# faster than the converter's std::filesystem::copy_file), then the converter
# runs with --already-copied-dst to only rewrite the 4 KB OdinANN header in
# place. fastcopy yields owner-writable (0600) dst files, so the canonical's
# 0400 mode is not propagated.
provision_index_into() {
    local canon dataset index_dir index_prefix converter converter_src fcp s src dst
    index_dir="$1"
    dataset="$2"
    canon="$(resolve_canonical_prefix "${WORKLOAD:-}")"
    [[ -n "$canon" ]] || error "no canonical index configured for $dataset/${WORKLOAD:-unknown}"
    index_prefix="$index_dir/$dataset"
    require_canonical "$canon"
    if canonical_family_present "$index_prefix"; then
        note "Reusing converted OdinANN index $index_prefix"
        return 0
    fi
    if [[ -n "${ODINANN_CONVERTER:-}" ]]; then
        converter="$ODINANN_CONVERTER"
        [[ -x "$converter" ]] || error "OdinANN converter missing: $converter"
    else
        converter_src="$REPO_DIR/baselines/src/convert_diskann_to_pipeann.cpp"
        converter="$REPO_DIR/baselines/DiskANN/build/tests/convert_diskann_to_pipeann"
        [[ -f "$converter_src" ]] || error "OdinANN converter source missing: $converter_src"
        if [[ ! -x "$converter" || "$converter_src" -nt "$converter" ]]; then
            mkdir -p "$(dirname "$converter")"
            note "Building OdinANN converter ($(basename "$converter_src"))"
            g++ "$converter_src" -O3 -std=c++17 -o "$converter"
        fi
        [[ -x "$converter" ]] || error "OdinANN converter missing after build: $converter"
    fi
    safe_mutable_path "$index_dir" >/dev/null
    mkdir -p "$index_dir"
    fcp="${FASTCOPY:-$BENCHMARK_DIR/fastcopy.sh}"
    note "fastcopy canonical family -> $index_prefix"
    for s in "${CANONICAL_FAMILY_SUFFIXES[@]}"; do
        src="${canon}${s}"
        dst="${index_prefix}${s}"
        [[ -e "$src" ]] || continue   # centroids/medoids are optional
        "$fcp" "$src" "$dst" || error "fastcopy failed: $src -> $dst"
    done
    note "Rewriting OdinANN header in place: $index_prefix"
    "$converter" "$canon" "$index_prefix" --already-copied-dst
    canonical_family_present "$index_prefix" || \
        error "OdinANN conversion incomplete: $index_prefix"
}

run_query_on_index() {
    local repo use_aio build_type dataset data_type index_dir npts query_bin gt_file output_dir query_threads beamwidth recall_at metric nbr_type search_mode mem_l search_l index_prefix
    repo="$1"
    use_aio="$2"
    build_type="$3"
    dataset="$4"
    data_type="$5"
    index_dir="$6"
    npts="$7"
    query_bin="$8"
    gt_file="$9"
    output_dir="${10}"
    query_threads="${11}"
    beamwidth="${12}"
    recall_at="${13}"
    metric="${14}"
    nbr_type="${15}"
    search_mode="${16}"
    mem_l="${17}"
    search_l="${18}"
    index_prefix="$index_dir/$dataset"
    mkdir -p "$output_dir"
    configure_odinann_repo "$repo" "$use_aio" "$build_type" search_disk_index
    ensure_identity_tags "$npts" "$index_prefix"
    local ann_bench_phase="${ANN_BENCH_PHASE:-query}"
    [[ "$output_dir" == */final_query ]] && ann_bench_phase="${ANN_BENCH_PHASE:-final_query}"
    ANN_BENCH_PHASE="$ann_bench_phase" "$repo/build/tests/search_disk_index" \
        "$data_type" "$index_prefix" \
        "$query_threads" "$beamwidth" \
        "$query_bin" "$gt_file" \
        "$recall_at" "$metric" "$nbr_type" \
        "$search_mode" "$mem_l" $search_l 2>&1 | tee "$output_dir/run.log"
}

run_query_on_index_preserve_tags() {
    local repo use_aio build_type dataset data_type index_dir npts query_bin gt_file output_dir query_threads beamwidth recall_at metric nbr_type search_mode mem_l search_l index_prefix
    repo="$1"
    use_aio="$2"
    build_type="$3"
    dataset="$4"
    data_type="$5"
    index_dir="$6"
    npts="$7"
    query_bin="$8"
    gt_file="$9"
    output_dir="${10}"
    query_threads="${11}"
    beamwidth="${12}"
    recall_at="${13}"
    metric="${14}"
    nbr_type="${15}"
    search_mode="${16}"
    mem_l="${17}"
    search_l="${18}"
    index_prefix="$index_dir/$dataset"
    mkdir -p "$output_dir"
    configure_odinann_repo "$repo" "$use_aio" "$build_type" search_disk_index
    require_index_tags "$index_prefix" "$npts"
    local ann_bench_phase="${ANN_BENCH_PHASE:-query}"
    [[ "$output_dir" == */final_query ]] && ann_bench_phase="${ANN_BENCH_PHASE:-final_query}"
    ANN_BENCH_PHASE="$ann_bench_phase" "$repo/build/tests/search_disk_index" \
        "$data_type" "$index_prefix" \
        "$query_threads" "$beamwidth" \
        "$query_bin" "$gt_file" \
        "$recall_at" "$metric" "$nbr_type" \
        "$search_mode" "$mem_l" $search_l 2>&1 | tee "$output_dir/run.log"
}
