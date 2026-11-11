#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DiskANN"
WORKLOAD="delete"
# eval_tempfiles contract: consume the tree delete_prepare.sh produced, in
# place; the test binary mutates WORK_DIR/index. Re-running requires
# re-running delete_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# Percentage of the full dataset used for insert/delete updates.
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
SEARCH_L="${SEARCH_L:-100}"
RECALL_SEARCH_L="${RECALL_SEARCH_L:-100}"   # L for the per-round in-binary recall search
DELETE_SEARCH_L="${DELETE_SEARCH_L:-100}"   # L for delete merge/search construction
BEAMWIDTH="${BEAMWIDTH:-4}"
R="${R:-32}"
ALPHA="${ALPHA:-1.2}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
NTHREADS="${NTHREADS:-$(nproc)}"
NODES_TO_CACHE="${NODES_TO_CACHE:-0}"
DISKANN_CXX="${DISKANN_CXX:-$(command -v g++)}"
DISKANN_CC="${DISKANN_CC:-$(command -v gcc)}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
if [[ -n "${DELETE_POINTS:-}" ]]; then
    [[ "$DELETE_POINTS" =~ ^[0-9]+$ && "$DELETE_POINTS" -gt 0 ]] \
        || error "DELETE_POINTS must be a positive integer (got: $DELETE_POINTS)"
    (( DELETE_POINTS < BASE_POINTS )) \
        || error "DELETE_POINTS ($DELETE_POINTS) must be < BASE_POINTS ($BASE_POINTS)"
    UPDATE_POINTS="$DELETE_POINTS"
fi
DELETE_CAP="${DELETE_CAP:-$UPDATE_POINTS}"
if ! [[ "$DELETE_CAP" =~ ^[0-9]+$ ]] || (( DELETE_CAP <= 0 )); then
    error "DELETE_CAP must be a positive integer (got: $DELETE_CAP)"
fi
if (( UPDATE_POINTS % DELETE_CAP != 0 )); then
    error "UPDATE_POINTS ($UPDATE_POINTS) must be divisible by DELETE_CAP ($DELETE_CAP)"
fi
if ! [[ "$NODES_TO_CACHE" =~ ^[0-9]+$ ]]; then
    error "NODES_TO_CACHE must be a non-negative integer (got: $NODES_TO_CACHE)"
fi
N_ITERS=$((UPDATE_POINTS / DELETE_CAP))
DELETE_COUNT="$DELETE_CAP"
FINAL_POINTS=$((BASE_POINTS - UPDATE_POINTS))
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing delete index; run delete_prepare.sh first"
[[ -f "$WORK_DIR/ids/delete_ids.bin" ]] || error "missing delete ids; run delete_prepare.sh first"
mkdir -p "$WORK_DIR/tmp" "$WORK_DIR/index_merge" "$WORK_DIR/index_mem"

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${FINAL_POINTS}_gt${RECALL_AT}.bin"
GT_ROUNDS_DIR="$WORK_DIR/groundtruth/rounds"   # per-round GT (round_<k>_gt<recall>.bin)

configure_diskann_repo "$BASELINE_REPO" "$DISKANN_CXX" "$DISKANN_CC" test_insert_or_delete search_disk_index
"$BASELINE_REPO/build/tests/test_insert_or_delete" \
    delete "$DATA_TYPE" "$WORK_DIR/tmp/" \
    "$WORK_DIR/index/$DATASET" "$WORK_DIR/index_merge/$DATASET" "$WORK_DIR/index_mem/$DATASET" \
    "$DELETE_SEARCH_L" "$ALPHA" \
    "$DELETE_SEARCH_L" "$ALPHA" \
    "$DATA_BIN" 0 \
    "$N_ITERS" "$DELETE_COUNT" "$R" "$BEAMWIDTH" "$NTHREADS" \
    "$WORK_DIR/ids/delete_ids.bin" \
    "$QUERY_BIN" "$GT_ROUNDS_DIR" "$RECALL_AT" "$RECALL_SEARCH_L" 2>&1 | tee "$WORK_DIR/run.log"
if (( N_ITERS % 2 == 1 )); then
    FINAL_INDEX_DIR="$WORK_DIR/index_merge"
else
    FINAL_INDEX_DIR="$WORK_DIR/index"
fi
[[ -f "$FINAL_INDEX_DIR/${DATASET}_disk.index" ]] || error "missing final index: $FINAL_INDEX_DIR/${DATASET}_disk.index"
run_query_on_index "$BASELINE_REPO" "$FINAL_INDEX_DIR" "$DATASET" "$DATA_TYPE" \
    "$FINAL_POINTS" "$QUERY_BIN" "$GT_FILE" "$WORK_DIR/final_query" \
    "$QUERY_THREADS" "$BEAMWIDTH" "$RECALL_AT" "$METRIC" "$SEARCH_L" \
    "$NODES_TO_CACHE" "$DISKANN_CXX" "$DISKANN_CC" | tee -a "$WORK_DIR/run.log"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
