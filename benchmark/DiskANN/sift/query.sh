#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DiskANN"
WORKLOAD="query"
# eval_tempfiles contract: consume the tree query_prepare.sh produced, in
# place. No re-materialization. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

SEARCH_L="${SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
NODES_TO_CACHE="${NODES_TO_CACHE:-0}"
export QUERY_USE_UPDATE_INDEX="${QUERY_USE_UPDATE_INDEX:-1}"
# Leave empty by default so utils.sh choose_compiler() can prefer gcc-15/14/11.
# Hard-coding bare g++/gcc here would pin the system GCC (9.4) and bypass that
# preference. Still env-overridable for a controlled compiler.
DISKANN_CXX="${DISKANN_CXX:-}"
DISKANN_CC="${DISKANN_CC:-}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
BASE_POINTS="$(canonical_base_points)"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing query index; run query_prepare.sh first"
[[ -f "$GT_FILE" ]] || error "missing query groundtruth; run query_prepare.sh first"
if ! [[ "$NODES_TO_CACHE" =~ ^[0-9]+$ ]]; then
    error "NODES_TO_CACHE must be a non-negative integer (got: $NODES_TO_CACHE)"
fi
run_query_on_index "$BASELINE_REPO" "$WORK_DIR/index" "$DATASET" "$DATA_TYPE" \
    "$BASE_POINTS" "$QUERY_BIN" "$GT_FILE" "$WORK_DIR" \
    "$QUERY_THREADS" "$BEAMWIDTH" "$RECALL_AT" "$METRIC" "$SEARCH_L" \
    "$NODES_TO_CACHE" "$DISKANN_CXX" "$DISKANN_CC"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
