#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="Greator"
WORKLOAD="insert"
# eval_tempfiles contract: consume the tree insert_prepare.sh produced, in
# place; the test binary mutates WORK_DIR/index. Re-running requires
# re-running insert_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# Percentage of the full dataset used for insert/delete updates.
# INSERT_POINTS, if set, overrides the per-percent count without shifting
# BASE_POINTS (mirrors insert_prepare.sh). INSERT_CAP defaults to
# UPDATE_POINTS, so a single-batch run falls out naturally.
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
SEARCH_L="${SEARCH_L:-100}"
RECALL_SEARCH_L="${RECALL_SEARCH_L:-100}"   # L for the per-round in-binary recall search
BEAMWIDTH="${BEAMWIDTH:-4}"
R="${R:-32}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
ALPHA="${ALPHA:-1.2}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
NTHREADS="${NTHREADS:-$(nproc)}"
NODES_TO_CACHE="${NODES_TO_CACHE:-0}"
GREATOR_CXX="${GREATOR_CXX:-$(command -v g++)}"
GREATOR_CC="${GREATOR_CC:-$(command -v gcc)}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
if [[ -n "${INSERT_POINTS:-}" ]]; then
    [[ "$INSERT_POINTS" =~ ^[0-9]+$ && "$INSERT_POINTS" -gt 0 ]] \
        || error "INSERT_POINTS must be a positive integer (got: $INSERT_POINTS)"
    UPDATE_POINTS="$INSERT_POINTS"
fi
INSERT_CAP="${INSERT_CAP:-$UPDATE_POINTS}"
if ! [[ "$INSERT_CAP" =~ ^[0-9]+$ ]] || (( INSERT_CAP <= 0 )); then
    error "INSERT_CAP must be a positive integer (got: $INSERT_CAP)"
fi
if (( UPDATE_POINTS % INSERT_CAP != 0 )); then
    error "UPDATE_POINTS ($UPDATE_POINTS) must be divisible by INSERT_CAP ($INSERT_CAP)"
fi
if ! [[ "$NODES_TO_CACHE" =~ ^[0-9]+$ ]]; then
    error "NODES_TO_CACHE must be a non-negative integer (got: $NODES_TO_CACHE)"
fi
N_ITERS=$((UPDATE_POINTS / INSERT_CAP))
INSERT_COUNT="$INSERT_CAP"
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing insert index; run insert_prepare.sh first"
[[ -f "$WORK_DIR/ids/insert_ids.bin" ]] || error "missing insert ids; run insert_prepare.sh first"
mkdir -p "$WORK_DIR/tmp" "$WORK_DIR/index_mem"

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${TOTAL_POINTS}_gt${RECALL_AT}.bin"
GT_ROUNDS_DIR="$WORK_DIR/groundtruth/rounds"   # per-round GT (round_<k>_gt<recall>.bin)

configure_greator_repo "$BASELINE_REPO" "$GREATOR_CXX" "$GREATOR_CC" test_insert_or_delete search_disk_index
INDEX_PREFIX="$WORK_DIR/index/$DATASET"
mkdir -p "${INDEX_PREFIX}_temp"
"$BASELINE_REPO/build/tests/test_insert_or_delete" \
    insert "$DATA_TYPE" "$INDEX_PREFIX" \
    "$INDEX_PREFIX" "$INDEX_PREFIX" "$WORK_DIR/index_mem/$DATASET" \
    "$INSERT_SEARCH_L" "$ALPHA" \
    "$INSERT_SEARCH_L" "$ALPHA" \
    "$DATA_BIN" 0 \
    "$N_ITERS" "$INSERT_COUNT" "$R" "$BEAMWIDTH" "$NTHREADS" \
    "$WORK_DIR/ids/insert_ids.bin" \
    "$QUERY_BIN" "$GT_ROUNDS_DIR" "$RECALL_AT" "$RECALL_SEARCH_L" 2>&1 | tee "$WORK_DIR/run.log"
FINAL_INDEX_DIR="$WORK_DIR/index"
[[ -f "$FINAL_INDEX_DIR/${DATASET}_disk.index" ]] || error "missing final index: $FINAL_INDEX_DIR/${DATASET}_disk.index"
run_query_on_index "$BASELINE_REPO" "$FINAL_INDEX_DIR" "$DATASET" "$DATA_TYPE" \
    "$((BASE_POINTS + UPDATE_POINTS))" "$QUERY_BIN" "$GT_FILE" "$WORK_DIR/final_query" \
    "$QUERY_THREADS" "$BEAMWIDTH" "$RECALL_AT" "$METRIC" "$SEARCH_L" \
    "$NODES_TO_CACHE" "$GREATOR_CXX" "$GREATOR_CC" | tee -a "$WORK_DIR/run.log"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
