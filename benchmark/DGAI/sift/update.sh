#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DGAI"
WORKLOAD="update"
# eval_tempfiles contract: consume the tree update_prepare.sh produced, in place. The driver mutates
# WORK_DIR/index, so re-running requires re-running update_prepare.sh.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

SEARCH_L="${SEARCH_L:-100}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-32}"
NTHREADS="${NTHREADS:-$(nproc)}"
QUERY_THREADS="${QUERY_THREADS:-$NTHREADS}"
# RQ1 runs without concurrent queries (SKIP_UPDATE_SEARCH=1, the default here). RQ2 sets SKIP_UPDATE_SEARCH=0,
# and the driver then runs OdinANN's concurrent queries: QUERY_RATIO in the pool, MERGE_QUERY_THREADS beside a
# consolidation on MERGE_THREADS, and MERGE_QUERY_BUDGET. DGAI_STRATEGY comes from benchmark/datasets/$DATASET.sh.
SKIP_UPDATE_SEARCH="${SKIP_UPDATE_SEARCH:-1}"
ENABLE_INTERVAL_METRICS="${ENABLE_INTERVAL_METRICS:-0}"

source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_update_layout
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
note "Mixed update: pool_threads=$NTHREADS, query ratio=${QUERY_RATIO}, merge_threads=${MERGE_THREADS}, merge_query_threads=${MERGE_QUERY_THREADS}, merge_query_budget=${MERGE_QUERY_BUDGET:-0}"
# The driver reads these from the environment. The tail range adds the not-yet-inserted base vectors to the
# cyclic query pool, as OdinANN's driver does with the same bounds.
export SKIP_UPDATE_SEARCH QUERY_RATIO MERGE_THREADS MERGE_QUERY_THREADS ENABLE_INTERVAL_METRICS
export TAIL_QUERY_BEGIN="$BASE_POINTS" TAIL_QUERY_END="$TOTAL_POINTS"
(( INSERT_CAP == DELETE_CAP )) \
    || error "DGAI update needs INSERT_CAP == DELETE_CAP (got $INSERT_CAP and $DELETE_CAP)"
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing update index; run update_prepare.sh first"
[[ -f "$WORK_DIR/ids/insert_ids.bin" && -f "$WORK_DIR/ids/delete_ids.bin" ]] \
    || error "missing update ids; run update_prepare.sh first"

run_update_on_index "$BASELINE_REPO" update "$DATA_TYPE" "$DATA_BIN" \
    "$INSERT_SEARCH_L" "$WORK_DIR/index" "$DATASET" \
    "$UPDATE_ROUNDS" "$INSERT_CAP" "$R" "$BEAMWIDTH" "$NTHREADS" \
    "$WORK_DIR/ids/insert_ids.bin:$WORK_DIR/ids/delete_ids.bin" \
    "$QUERY_BIN" "$WORK_DIR/groundtruth/rounds" "$QUERY_THREADS" "$RECALL_AT" "$SEARCH_L" \
    "$DGAI_STRATEGY" "$WORK_DIR"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
