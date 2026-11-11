#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DGAI"
WORKLOAD="insert"
# eval_tempfiles contract: consume the tree insert_prepare.sh produced, in
# place. The driver mutates WORK_DIR/index, so re-running requires re-running
# insert_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
SEARCH_L="${SEARCH_L:-100}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-32}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
NTHREADS="${NTHREADS:-$(nproc)}"
# DGAI_STRATEGY comes from benchmark/datasets/$DATASET.sh (see benchmark/DGAI/common.sh).

source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
if [[ -n "${INSERT_POINTS:-}" ]]; then
    [[ "$INSERT_POINTS" =~ ^[0-9]+$ && "$INSERT_POINTS" -gt 0 ]] \
        || error "INSERT_POINTS must be a positive integer (got: $INSERT_POINTS)"
    UPDATE_POINTS="$INSERT_POINTS"
fi
INSERT_CAP="${INSERT_CAP:-$UPDATE_POINTS}"
[[ "$INSERT_CAP" =~ ^[0-9]+$ ]] && (( INSERT_CAP > 0 )) || error "INSERT_CAP must be a positive integer (got: $INSERT_CAP)"
(( UPDATE_POINTS % INSERT_CAP == 0 )) \
    || error "UPDATE_POINTS ($UPDATE_POINTS) must be divisible by INSERT_CAP ($INSERT_CAP)"
N_ITERS=$((UPDATE_POINTS / INSERT_CAP))
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing insert index; run insert_prepare.sh first"
[[ -f "$WORK_DIR/ids/insert_ids.bin" ]] || error "missing insert ids; run insert_prepare.sh first"

run_update_on_index "$BASELINE_REPO" insert "$DATA_TYPE" "$DATA_BIN" \
    "$INSERT_SEARCH_L" "$WORK_DIR/index" "$DATASET" \
    "$N_ITERS" "$INSERT_CAP" "$R" "$BEAMWIDTH" "$NTHREADS" \
    "$WORK_DIR/ids/insert_ids.bin" \
    "$QUERY_BIN" "$WORK_DIR/groundtruth/rounds" "$QUERY_THREADS" "$RECALL_AT" "$SEARCH_L" \
    "$DGAI_STRATEGY" "$WORK_DIR"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
