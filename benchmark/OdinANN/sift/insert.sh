#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="OdinANN"
WORKLOAD="insert"
# eval_tempfiles contract: consume the tree insert_prepare.sh produced, in
# place; the test binary mutates WORK_DIR/index. Re-running requires
# re-running insert_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# Percentage of the full dataset used for insert/delete updates.
# INSERT_POINTS, if set, overrides the per-percent count without shifting
# BASE_POINTS (mirrors insert_prepare.sh). OdinANN insert-only skips the
# per-batch merge (no maintenance needed for inserts), but still runs N_ITERS =
# UPDATE_POINTS/INSERT_CAP rounds: each round inserts INSERT_CAP IDs into the
# in-memory delta and per-round recall is measured against that round's GT.
# This mirrors the delete path's round split (same unified loop in the binary).
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
SEARCH_L="${SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
R="${R:-32}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"
NTHREADS="${NTHREADS:-$(nproc)}"
USE_AIO="${USE_AIO:-ON}"


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
N_ITERS=$((UPDATE_POINTS / INSERT_CAP))
INSERT_COUNT="$INSERT_CAP"
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing insert index; run insert_prepare.sh first"
[[ -f "$WORK_DIR/ids/insert_ids.bin" ]] || error "missing insert ids; run insert_prepare.sh first"
mkdir -p "$WORK_DIR/index_merge"

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${TOTAL_POINTS}_gt${RECALL_AT}.bin"
GT_ROUNDS_DIR="$WORK_DIR/groundtruth/rounds"   # per-round GT (round_<k>_gt<recall>.bin)

configure_odinann_repo "$BASELINE_REPO" "$USE_AIO" "$BENCHMARK_CMAKE_BUILD_TYPE" test_insert_or_delete
"$BASELINE_REPO/build/tests/test_insert_or_delete" \
    insert "$DATA_TYPE" "$DATA_BIN" \
    "$INSERT_SEARCH_L" "$WORK_DIR/index/$DATASET" "$WORK_DIR/index_merge/$DATASET" \
    "$N_ITERS" "$INSERT_COUNT" "$R" "$BEAMWIDTH" "$NTHREADS" \
    "$WORK_DIR/ids/insert_ids.bin" \
    "$QUERY_BIN" "$GT_ROUNDS_DIR" "$QUERY_THREADS" "$RECALL_AT" "$SEARCH_L" \
    2>&1 | tee "$WORK_DIR/run.log"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
