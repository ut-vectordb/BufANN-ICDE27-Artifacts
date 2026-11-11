#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DiskANN"
WORKLOAD="update"
# eval_tempfiles contract: consume the tree update_prepare.sh produced, in
# place; test_mix_update mutates WORK_DIR/index. Re-running requires
# re-running update_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
UPDATE_BATCH="${UPDATE_BATCH:-}"
if [[ -n "$UPDATE_BATCH" ]]; then
    INSERT_CAP="${INSERT_CAP:-$UPDATE_BATCH}"
    DELETE_CAP="${DELETE_CAP:-$UPDATE_BATCH}"
fi
SEARCH_L="${SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
R="${R:-32}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
ALPHA="${ALPHA:-1.2}"
# Mixed update runs insert/delete/query through one unified worker pool of
# NTHREADS workers; the former per-op INSERT/DELETE_THREADS are ignored.
# QUERY_THREADS now only sizes the standalone final recall query.
NTHREADS="${NTHREADS:-$(nproc)}"
QUERY_THREADS="${QUERY_THREADS:-$NTHREADS}"
NODES_TO_CACHE="${NODES_TO_CACHE:-0}"
MERGE_MAXC="${MERGE_MAXC:-$((R * 5 / 2))}"
SKIP_UPDATE_SEARCH="${SKIP_UPDATE_SEARCH:-0}"
ENABLE_INTERVAL_METRICS="${ENABLE_INTERVAL_METRICS:-0}"
DISKANN_BUILD_ONLY_UPDATE="${DISKANN_BUILD_ONLY_UPDATE:-0}"
DISKANN_CXX="${DISKANN_CXX:-$(command -v g++)}"
DISKANN_CC="${DISKANN_CC:-$(command -v gcc)}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_update_layout
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing update index; run update_prepare.sh first"
[[ -f "$WORK_DIR/ids/insert_ids.bin" && -f "$WORK_DIR/ids/delete_ids.bin" ]] || error "missing update ids; run update_prepare.sh first"
if ! [[ "$NODES_TO_CACHE" =~ ^[0-9]+$ ]]; then
    error "NODES_TO_CACHE must be a non-negative integer (got: $NODES_TO_CACHE)"
fi
MERGE_THREADS="${MERGE_THREADS:-$NTHREADS}"
if (( SKIP_UPDATE_SEARCH != 0 )); then
    QUERY_RATIO=0
    MERGE_QUERY_THREADS=0
    ENABLE_INTERVAL_METRICS=0
else
    QUERY_RATIO="${QUERY_RATIO:-80}"
    MERGE_QUERY_THREADS="$((NTHREADS - MERGE_THREADS))"
    (( MERGE_QUERY_THREADS > 0 )) || error "NTHREADS ($NTHREADS) must be > MERGE_THREADS ($MERGE_THREADS)"
fi
note "Mixed update: pool_threads=$NTHREADS, query ratio=${QUERY_RATIO}, merge_threads=${MERGE_THREADS}, merge_query_threads=${MERGE_QUERY_THREADS}"

# set mem_index size to round size.
MERGE_TH="${MERGE_TH:-$INSERT_CAP}"
note "Mixed update: MERGE_TH=$MERGE_TH (mem index capacity $((MERGE_TH * 2)) >= INSERT_CAP $INSERT_CAP)"

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_update_i${INSERT_POINTS}_d${DELETE_POINTS}_gt${RECALL_AT}.bin"

mkdir -p "$WORK_DIR/tmp" "$WORK_DIR/index_merge" "$WORK_DIR/index_mem"

configure_diskann_repo "$BASELINE_REPO" "$DISKANN_CXX" "$DISKANN_CC" test_mix_update search_disk_index
MERGE_TH="$MERGE_TH" \
"$BASELINE_REPO/build/tests/test_mix_update" \
    "$DATA_TYPE" "$WORK_DIR/tmp/" \
    "$WORK_DIR/index/$DATASET" "$WORK_DIR/index_merge/$DATASET" "$WORK_DIR/index_mem/$DATASET" \
    "$INSERT_SEARCH_L" "$ALPHA" \
    "$INSERT_SEARCH_L" "$ALPHA" \
    "$DATA_BIN" 0 \
    "$QUERY_BIN" \
    "$UPDATE_ROUNDS" "$INSERT_CAP" "$DELETE_CAP" "$R" "$RECALL_AT" \
    "$NTHREADS" "$MERGE_THREADS" \
    "$WORK_DIR/ids/insert_ids.bin" "$WORK_DIR/ids/delete_ids.bin" \
    "$NODES_TO_CACHE" "$BEAMWIDTH" "$MERGE_MAXC" \
    "$SKIP_UPDATE_SEARCH" "$DISKANN_BUILD_ONLY_UPDATE" \
    "$QUERY_RATIO" "$MERGE_QUERY_THREADS" \
    "$BASE_POINTS" "$TOTAL_POINTS" "$WORK_DIR/groundtruth/rounds" "$RECALL_SEARCH_L" \
    "$ENABLE_INTERVAL_METRICS" \
    "$SEARCH_L" 2>&1 | tee "$WORK_DIR/run.log"

if (( UPDATE_ROUNDS % 2 == 1 )); then
    FINAL_INDEX_DIR="$WORK_DIR/index_merge"
else
    FINAL_INDEX_DIR="$WORK_DIR/index"
fi
[[ -f "$FINAL_INDEX_DIR/${DATASET}_disk.index" ]] || error "missing final index: $FINAL_INDEX_DIR/${DATASET}_disk.index"

INDEX_TAG_FILE="$FINAL_INDEX_DIR/${DATASET}_disk.index.tags"
write_update_final_tags "$INDEX_TAG_FILE" "$BASE_POINTS" "$INSERT_POINTS" "$DELETE_POINTS"

echo "=== Running final standalone recall query ==="
run_query_on_index "$BASELINE_REPO" "$FINAL_INDEX_DIR" "$DATASET" "$DATA_TYPE" \
    "$FINAL_POINTS" "$QUERY_BIN" "$GT_FILE" "$WORK_DIR/final_query" \
    "$QUERY_THREADS" "$BEAMWIDTH" "$RECALL_AT" "$METRIC" "$SEARCH_L" \
    "$NODES_TO_CACHE" "$DISKANN_CXX" "$DISKANN_CC" | tee -a "$WORK_DIR/run.log"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
