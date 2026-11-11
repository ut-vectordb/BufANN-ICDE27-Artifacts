#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="BufANN"
WORKLOAD="delete"
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
SEARCH_L="${SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
NTHREADS="${NTHREADS:-$(nproc)}"
DELETE_THREADS="${DELETE_THREADS:-$NTHREADS}"
DELETE_ALGORITHM="${DELETE_ALGORITHM:-freshdiskann}"   # freshdiskann | freshdiskann-nosnapshot | lazy


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
if [[ -n "${DELETE_POINTS:-}" ]]; then
    [[ "$DELETE_POINTS" =~ ^[0-9]+$ && "$DELETE_POINTS" -gt 0 ]] \
        || error "DELETE_POINTS must be a positive integer (got: $DELETE_POINTS)"
    UPDATE_POINTS="$DELETE_POINTS"
fi
FINAL_POINTS=$((BASE_POINTS - UPDATE_POINTS))
[[ -f "$WORK_DIR/index/${DATASET}.heap" ]] || error "missing delete index; run delete_prepare.sh first"
[[ -f "$WORK_DIR/ids/delete_ids.bin" ]] || error "missing delete ids; run delete_prepare.sh first"

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${FINAL_POINTS}_gt${RECALL_AT}.bin"
GT_ROUNDS_DIR="$WORK_DIR/groundtruth/rounds"   # per-round GT (round_<k>_gt<recall>.bin)
DELETE_CAP="${DELETE_CAP:-$UPDATE_POINTS}"     # per-round delete count (n_iters = UPDATE_POINTS/DELETE_CAP)
RECALL_SEARCH_L="${RECALL_SEARCH_L:-100}"      # L for the per-round recall search

run_bufann_driver "$WORK_DIR/index/$DATASET" delete_only \
    "$QUERY_BIN" "$GT_FILE" "$WORK_DIR" \
    "$SEARCH_L" "$BEAMWIDTH" "$QUERY_THREADS" "$RECALL_AT" \
    --delete_ids_file "$WORK_DIR/ids/delete_ids.bin" \
    --delete_threads "$DELETE_THREADS" \
    --delete_algorithm "$DELETE_ALGORITHM" \
    --run_maintenance 1 \
    --flush_after_maintenance 1 \
    --delete_cap "$DELETE_CAP" \
    --gt_dir "$GT_ROUNDS_DIR" \
    --recall_search_L "$RECALL_SEARCH_L"
