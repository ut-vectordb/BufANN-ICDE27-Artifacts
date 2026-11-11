#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DGAI"
WORKLOAD="delete"
# eval_tempfiles contract: consume the tree delete_prepare.sh produced, in
# place. The driver mutates WORK_DIR/index, so re-running requires re-running
# delete_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
SEARCH_L="${SEARCH_L:-100}"
DELETE_SEARCH_L="${DELETE_SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-32}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
NTHREADS="${NTHREADS:-$(nproc)}"
# DGAI_STRATEGY comes from benchmark/datasets/$DATASET.sh (see benchmark/DGAI/common.sh).

source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
if [[ -n "${DELETE_POINTS:-}" ]]; then
    [[ "$DELETE_POINTS" =~ ^[0-9]+$ && "$DELETE_POINTS" -gt 0 ]] \
        || error "DELETE_POINTS must be a positive integer (got: $DELETE_POINTS)"
    UPDATE_POINTS="$DELETE_POINTS"
fi
DELETE_CAP="${DELETE_CAP:-$UPDATE_POINTS}"
[[ "$DELETE_CAP" =~ ^[0-9]+$ ]] && (( DELETE_CAP > 0 )) || error "DELETE_CAP must be a positive integer (got: $DELETE_CAP)"
(( UPDATE_POINTS % DELETE_CAP == 0 )) \
    || error "UPDATE_POINTS ($UPDATE_POINTS) must be divisible by DELETE_CAP ($DELETE_CAP)"
N_ITERS=$((UPDATE_POINTS / DELETE_CAP))
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing delete index; run delete_prepare.sh first"
[[ -f "$WORK_DIR/ids/delete_ids.bin" ]] || error "missing delete ids; run delete_prepare.sh first"

run_update_on_index "$BASELINE_REPO" delete "$DATA_TYPE" "$DATA_BIN" \
    "$DELETE_SEARCH_L" "$WORK_DIR/index" "$DATASET" \
    "$N_ITERS" "$DELETE_CAP" "$R" "$BEAMWIDTH" "$NTHREADS" \
    "$WORK_DIR/ids/delete_ids.bin" \
    "$QUERY_BIN" "$WORK_DIR/groundtruth/rounds" "$QUERY_THREADS" "$RECALL_AT" "$SEARCH_L" \
    "$DGAI_STRATEGY" "$WORK_DIR"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
