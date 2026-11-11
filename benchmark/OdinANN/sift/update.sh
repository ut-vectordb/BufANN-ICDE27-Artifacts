#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="OdinANN"
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
RECALL_SEARCH_L="${RECALL_SEARCH_L:-$SEARCH_L}"
BEAMWIDTH="${BEAMWIDTH:-4}"
R="${R:-32}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
# Mixed update runs insert/delete/query through one unified worker pool of
# NTHREADS workers; the former per-op INSERT/DELETE_THREADS are ignored.
# QUERY_THREADS now only sizes the standalone final recall query.
NTHREADS="${NTHREADS:-$(nproc)}"
QUERY_THREADS="${QUERY_THREADS:-$NTHREADS}"
SKIP_UPDATE_SEARCH="${SKIP_UPDATE_SEARCH:-0}"
ENABLE_INTERVAL_METRICS="${ENABLE_INTERVAL_METRICS:-0}"
USE_AIO="${USE_AIO:-ON}"
MERGE_MAXC="${MERGE_MAXC:-384}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_update_layout
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing update index; run update_prepare.sh first"
[[ -f "$WORK_DIR/ids/insert_ids.bin" && -f "$WORK_DIR/ids/delete_ids.bin" ]] || error "missing update ids; run update_prepare.sh first"
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

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_update_i${INSERT_POINTS}_d${DELETE_POINTS}_gt${RECALL_AT}.bin"

mkdir -p "$WORK_DIR/index_merge"

configure_odinann_repo "$BASELINE_REPO" "$USE_AIO" "$BENCHMARK_CMAKE_BUILD_TYPE" test_mix_update search_disk_index
"$BASELINE_REPO/build/tests/test_mix_update" \
    "$DATA_TYPE" "$DATA_BIN" \
    "$INSERT_SEARCH_L" "$WORK_DIR/index/$DATASET" "$WORK_DIR/index_merge/$DATASET" \
    "$UPDATE_ROUNDS" "$INSERT_CAP" "$DELETE_CAP" "$R" "$BEAMWIDTH" "$NTHREADS" \
    "$MERGE_THREADS" \
    "$MERGE_MAXC" \
    "$WORK_DIR/ids/insert_ids.bin" "$WORK_DIR/ids/delete_ids.bin" \
    "$QUERY_BIN" "$RECALL_AT" "$SEARCH_L" \
    "$SKIP_UPDATE_SEARCH" "$QUERY_RATIO" "$MERGE_QUERY_THREADS" \
    "$BASE_POINTS" "$TOTAL_POINTS" "$WORK_DIR/groundtruth/rounds" "$RECALL_SEARCH_L" \
    "$ENABLE_INTERVAL_METRICS" \
    2>&1 | tee "$WORK_DIR/run.log"

if (( UPDATE_ROUNDS % 2 == 1 )); then
    DYNAMIC_FINAL_DIR="$WORK_DIR/index_merge"
else
    DYNAMIC_FINAL_DIR="$WORK_DIR/index"
fi
[[ -f "$DYNAMIC_FINAL_DIR/${DATASET}_disk.index" ]] || error "missing final index: $DYNAMIC_FINAL_DIR/${DATASET}_disk.index"

# Merge writes semantic tags keyed by dense internal id; preserve them.
FINAL_INDEX_DIR="$DYNAMIC_FINAL_DIR"
echo "=== Running final standalone recall query ==="
run_query_on_index_preserve_tags "$BASELINE_REPO" "$USE_AIO" "$BENCHMARK_CMAKE_BUILD_TYPE" \
    "$DATASET" "$DATA_TYPE" "$FINAL_INDEX_DIR" "$FINAL_POINTS" \
    "$QUERY_BIN" "$GT_FILE" "$WORK_DIR/final_query" "$QUERY_THREADS" "$BEAMWIDTH" \
    "$RECALL_AT" "$METRIC" pq 0 0 "$SEARCH_L" | tee -a "$WORK_DIR/run.log"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
