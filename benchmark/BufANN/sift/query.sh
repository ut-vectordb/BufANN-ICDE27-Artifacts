#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="BufANN"
WORKLOAD="query"
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

SEARCH_L="${SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
# compute_split sizes the warmup pool to the 0.9B canonical prefix (shared
# with insert/delete/update). Cheap to call here; lets us tune
# BUFANN_WARMUP_N / BUFANN_WARMUP_SEED without re-running query_prepare.sh.
compute_split
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
[[ -f "$WORK_DIR/index/${DATASET}.heap" ]] || error "missing query index; run query_prepare.sh first"
[[ -f "$GT_FILE" ]] || error "missing query groundtruth; run query_prepare.sh first"
ensure_bufann_warmup_queries "$WORK_DIR" "$BASE_POINTS"

run_bufann_driver "$WORK_DIR/index/$DATASET" search_only \
    "$QUERY_BIN" "$GT_FILE" "$WORK_DIR" \
    "$SEARCH_L" "$BEAMWIDTH" "$QUERY_THREADS" "$RECALL_AT"
