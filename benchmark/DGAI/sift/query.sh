#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DGAI"
WORKLOAD="query"
# eval_tempfiles contract: consume the tree query_prepare.sh produced, in
# place. No re-materialization. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

SEARCH_L="${SEARCH_L:-100}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
# The authors run DGAI with pipeline width 32. DGAI_STRATEGY comes from benchmark/datasets/$DATASET.sh (see
# benchmark/DGAI/common.sh).
BEAMWIDTH="${BEAMWIDTH:-32}"

source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
BASE_POINTS="$NPTS_BASE"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing query index; run query_prepare.sh first"
[[ -f "$GT_FILE" ]] || error "missing query groundtruth; run query_prepare.sh first"
# Warm up with the base vectors query_prepare.sh sampled (see ensure_dgai_warmup_queries).
if [[ -f "$WORK_DIR/warmup_queries.bin" ]]; then
    export DGAI_WARMUP_QUERY_FILE="$WORK_DIR/warmup_queries.bin"
fi
run_query_on_index "$BASELINE_REPO" "$WORK_DIR/index" "$DATASET" "$DATA_TYPE" \
    "$QUERY_BIN" "$GT_FILE" "$WORK_DIR" \
    "$QUERY_THREADS" "$BEAMWIDTH" "$RECALL_AT" "$METRIC" "$DGAI_STRATEGY" "$SEARCH_L"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
