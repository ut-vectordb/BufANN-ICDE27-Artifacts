#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="BufANN"
WORKLOAD="insert"
# eval_tempfiles contract: consume the tree insert_prepare.sh produced, in
# place; bufann_driver mutates WORK_DIR/index (.heap is rewritten).
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
SEARCH_L="${SEARCH_L:-100}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
NTHREADS="${NTHREADS:-$(nproc)}"
INSERT_THREADS="${INSERT_THREADS:-$NTHREADS}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
if [[ -n "${INSERT_POINTS:-}" ]]; then
    [[ "$INSERT_POINTS" =~ ^[0-9]+$ && "$INSERT_POINTS" -gt 0 ]] \
        || error "INSERT_POINTS must be a positive integer (got: $INSERT_POINTS)"
    UPDATE_POINTS="$INSERT_POINTS"
fi
[[ -f "$WORK_DIR/index/${DATASET}.heap" ]] || error "missing insert index; run insert_prepare.sh first"
[[ -f "$WORK_DIR/ids/insert_ids.bin" ]] || error "missing insert ids; run insert_prepare.sh first"

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${TOTAL_POINTS}_gt${RECALL_AT}.bin"
GT_ROUNDS_DIR="$WORK_DIR/groundtruth/rounds"   # per-round GT (round_<k>_gt<recall>.bin)
INSERT_CAP="${INSERT_CAP:-$UPDATE_POINTS}"     # per-round insert count (n_iters = UPDATE_POINTS/INSERT_CAP)
RECALL_SEARCH_L="${RECALL_SEARCH_L:-100}"      # L for the per-round recall search

run_bufann_driver "$WORK_DIR/index/$DATASET" insert_only \
    "$QUERY_BIN" "$GT_FILE" "$WORK_DIR" \
    "$SEARCH_L" "$BEAMWIDTH" "$QUERY_THREADS" "$RECALL_AT" \
    --full_data_file "$WORK_DIR/data/full.bin" \
    --insert_ids_file "$WORK_DIR/ids/insert_ids.bin" \
    --insert_search_L "$INSERT_SEARCH_L" \
    --insert_threads "$INSERT_THREADS" \
    --insert_cap "$INSERT_CAP" \
    --gt_dir "$GT_ROUNDS_DIR" \
    --recall_search_L "$RECALL_SEARCH_L"
