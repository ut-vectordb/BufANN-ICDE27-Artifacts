#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="OdinANN"
WORKLOAD="query"
# eval_tempfiles contract: consume the tree query_prepare.sh produced, in
# place. No re-materialization. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

SEARCH_L="${SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
USE_AIO="${USE_AIO:-ON}"
export QUERY_USE_UPDATE_INDEX="${QUERY_USE_UPDATE_INDEX:-1}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
BASE_POINTS="$(canonical_base_points)"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing query index; run query_prepare.sh first"
[[ -f "$GT_FILE" ]] || error "missing query groundtruth; run query_prepare.sh first"
run_query_on_index "$BASELINE_REPO" "$USE_AIO" "$BENCHMARK_CMAKE_BUILD_TYPE" \
    "$DATASET" "$DATA_TYPE" "$WORK_DIR/index" "$BASE_POINTS" \
    "$QUERY_BIN" "$GT_FILE" "$WORK_DIR" "$QUERY_THREADS" "$BEAMWIDTH" \
    "$RECALL_AT" "$METRIC" pq 0 0 "$SEARCH_L"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
