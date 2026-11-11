#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="BufANN"
WORKLOAD="update"
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
SEARCH_L="${SEARCH_L:-100}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
# Mixed update (search_after_update): foreground insert/delete/query share
# pool_threads. Per-round maintenance (edge cleanup) uses MERGE_THREADS;
# concurrent queries during maintenance use MERGE_QUERY_THREADS. QUERY_THREADS
# sizes the standalone final recall query.
NTHREADS="${NTHREADS:-$(nproc)}"
QUERY_THREADS="${QUERY_THREADS:-$NTHREADS}"
DELETE_ALGORITHM="${DELETE_ALGORITHM:-freshdiskann}"   # freshdiskann | freshdiskann-nosnapshot | lazy
SKIP_UPDATE_SEARCH="${SKIP_UPDATE_SEARCH:-0}"
ENABLE_INTERVAL_METRICS="${ENABLE_INTERVAL_METRICS:-0}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_update_layout
[[ -f "$WORK_DIR/index/${DATASET}.heap" ]] || error "missing update index; run update_prepare.sh first"
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
# Both delete algorithms use MERGE_THREADS for maintenance: FreshDiskANN's merge threads, lazy deletion's repair workers.
export MAINTENANCE_THREADS="$MERGE_THREADS"
export MAINTENANCE_QUERY_THREADS="$MERGE_QUERY_THREADS"
RECALL_SEARCH_L="${RECALL_SEARCH_L:-$SEARCH_L}"
note "Mixed update: pool_threads=$NTHREADS, query ratio=$QUERY_RATIO, maintenance_threads=$MERGE_THREADS, maintenance_query_threads=$MERGE_QUERY_THREADS"

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_update_i${INSERT_POINTS}_d${DELETE_POINTS}_gt${RECALL_AT}.bin"

# search_after_update: per-round insert_cap/delete_cap chunks (same as insert/delete
# drivers), concurrent foreground ops, maintenance with concurrent queries, then
# per-round recall vs groundtruth/rounds/round_<k>_gt*.bin (full query.bin).
run_bufann_driver "$WORK_DIR/index/$DATASET" search_after_update \
    "$QUERY_BIN" "$GT_FILE" "$WORK_DIR" \
    "$SEARCH_L" "$BEAMWIDTH" "$QUERY_THREADS" "$RECALL_AT" \
    --full_data_file "$WORK_DIR/data/full.bin" \
    --insert_ids_file "$WORK_DIR/ids/insert_ids.bin" \
    --delete_ids_file "$WORK_DIR/ids/delete_ids.bin" \
    --delete_algorithm "$DELETE_ALGORITHM" \
    --insert_search_L "$INSERT_SEARCH_L" \
    --pool_threads "$NTHREADS" \
    --skip_update_search "$SKIP_UPDATE_SEARCH" \
    --enable_interval_metrics "$ENABLE_INTERVAL_METRICS" \
    --query_ratio "$QUERY_RATIO" \
    --tail_query_begin "$BASE_POINTS" \
    --tail_query_end "$TOTAL_POINTS" \
    --insert_cap "$INSERT_CAP" \
    --delete_cap "$DELETE_CAP" \
    --gt_dir "$WORK_DIR/groundtruth/rounds" \
    --recall_search_L "$RECALL_SEARCH_L" \
    --maintenance_query_threads "$MERGE_QUERY_THREADS" \
    --run_maintenance 1 \
    --flush_after_maintenance 1 \
    "$@"

write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
