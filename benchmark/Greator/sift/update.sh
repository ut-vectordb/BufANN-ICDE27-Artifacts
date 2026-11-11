#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="Greator"
WORKLOAD="update"
# eval_tempfiles contract: consume the tree update_prepare.sh produced, in
# place; test_mix_update mutates WORK_DIR/index. Re-running requires
# re-running update_prepare.sh. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
UPDATE_BATCH="${UPDATE_BATCH:-}"
if [[ -n "$UPDATE_BATCH" ]]; then
    INSERT_CAP="${INSERT_CAP:-$UPDATE_BATCH}"
    DELETE_CAP="${DELETE_CAP:-$UPDATE_BATCH}"
fi
SEARCH_L="${SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
R="${R:-32}"
INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
ALPHA="${ALPHA:-1.2}"
# Mixed update runs insert/delete/query through one unified worker pool of
# NTHREADS workers; the former per-op INSERT/DELETE_THREADS are ignored.
# QUERY_THREADS now only sizes the standalone final recall query.
NTHREADS="${NTHREADS:-$(nproc)}"
QUERY_THREADS="${QUERY_THREADS:-$NTHREADS}"
SKIP_UPDATE_SEARCH="${SKIP_UPDATE_SEARCH:-0}"
ENABLE_INTERVAL_METRICS="${ENABLE_INTERVAL_METRICS:-0}"
# Like benchmark/Greator/sift/insert.sh: merged disk index stays under INDEX_PREFIX (no separate index_merge/).
# Set GREATOR_OUTPUT_TO_MERGE_PREFIX=1 to write merged output to a different merge prefix passed as the 5th CLI arg.
GREATOR_OUTPUT_TO_MERGE_PREFIX="${GREATOR_OUTPUT_TO_MERGE_PREFIX:-0}"
NODES_TO_CACHE="${NODES_TO_CACHE:-0}"
# Matches former default merge_maxc (approx. range * 2.5); override with MERGE_MAXC=...
MERGE_MAXC="${MERGE_MAXC:-$((R * 5 / 2))}"
GREATOR_CXX="${GREATOR_CXX:-$(command -v g++)}"
GREATOR_CC="${GREATOR_CC:-$(command -v gcc)}"


source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_update_layout
[[ -f "$WORK_DIR/index/${DATASET}_disk.index" ]] || error "missing update index; run update_prepare.sh first"
[[ -f "$WORK_DIR/ids/insert_ids.bin" && -f "$WORK_DIR/ids/delete_ids.bin" ]] || error "missing update ids; run update_prepare.sh first"
if ! [[ "$NODES_TO_CACHE" =~ ^[0-9]+$ ]]; then
    error "NODES_TO_CACHE must be a non-negative integer (got: $NODES_TO_CACHE)"
fi
MERGE_THREADS="${MERGE_THREADS:-$NTHREADS}"
if (( SKIP_UPDATE_SEARCH != 0 )); then
    QUERY_RATIO=0
    MERGE_QUERY_THREADS=0
    ENABLE_INTERVAL_METRICS=0
else
    QUERY_RATIO="${QUERY_RATIO:-80}"
    MERGE_QUERY_THREADS="$((NTHREADS - MERGE_THREADS))"
    (( MERGE_QUERY_THREADS > 0 )) || error "NTHREADS ($NTHREADS) must be > MERGE_THREADS ($MERGE_THREADS)"
fi
note "Mixed update: pool_threads=$NTHREADS, query ratio=${QUERY_RATIO}, merge_threads=${MERGE_THREADS}, merge_query_threads=${MERGE_QUERY_THREADS}"

# The mixed driver loads an entire round (INSERT_CAP points) into Greator's
# in-memory index, then does a single final_merge -- there is no mid-stream
# trigger_merge. That index holds MERGE_TH*2 points; the compiled default
# (MERGE_TH=100000 -> cap 200k) overflows large rounds and every insert past
# the cap fails with "Capacity exceeded". Size the threshold to the round so
# the whole batch fits (cap = 2*INSERT_CAP), matching DiskANN's large MERGE_TH.
MERGE_TH="${MERGE_TH:-$INSERT_CAP}"
note "Mixed update: MERGE_TH=$MERGE_TH (mem index capacity $((MERGE_TH * 2)) >= INSERT_CAP $INSERT_CAP)"

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_update_i${INSERT_POINTS}_d${DELETE_POINTS}_gt${RECALL_AT}.bin"

mkdir -p "$WORK_DIR/index_mem"

configure_greator_repo "$BASELINE_REPO" "$GREATOR_CXX" "$GREATOR_CC" test_mix_update search_disk_index
INDEX_PREFIX="$WORK_DIR/index/$DATASET"
WORKING_PREFIX="$INDEX_PREFIX"
mkdir -p "${INDEX_PREFIX}_temp"

GREATOR_OUTPUT_TO_MERGE_PREFIX="$GREATOR_OUTPUT_TO_MERGE_PREFIX" \
GREATOR_BENCHMARK_QUICK_EXIT=1 \
MERGE_TH="$MERGE_TH" \
"$BASELINE_REPO/build/tests/test_mix_update" \
    "$DATA_TYPE" "$WORKING_PREFIX" \
    "$INDEX_PREFIX" "$INDEX_PREFIX" "$WORK_DIR/index_mem/$DATASET" \
    "$INSERT_SEARCH_L" "$ALPHA" \
    "$INSERT_SEARCH_L" "$ALPHA" \
    "$DATA_BIN" 0 \
    "$QUERY_BIN" \
    "$UPDATE_ROUNDS" "$INSERT_CAP" "$DELETE_CAP" "$R" "$RECALL_AT" \
    "$NTHREADS" "$MERGE_THREADS" \
    "$WORK_DIR/ids/insert_ids.bin" "$WORK_DIR/ids/delete_ids.bin" \
    "$NODES_TO_CACHE" "$BEAMWIDTH" "$MERGE_MAXC" \
    "$SKIP_UPDATE_SEARCH" \
    "$QUERY_RATIO" "$MERGE_QUERY_THREADS" \
    "$BASE_POINTS" "$TOTAL_POINTS" "$WORK_DIR/groundtruth/rounds" "$RECALL_SEARCH_L" \
    "$ENABLE_INTERVAL_METRICS" \
    "$SEARCH_L" 2>&1 | tee "$WORK_DIR/run.log"

FINAL_INDEX_DIR="$WORK_DIR/index"
[[ -f "$FINAL_INDEX_DIR/${DATASET}_disk.index" ]] || error "missing final index: $FINAL_INDEX_DIR/${DATASET}_disk.index"

INDEX_TAG_FILE="${FINAL_INDEX_DIR}/${DATASET}_disk.index.tags"
write_update_final_tags "$INDEX_TAG_FILE" "$BASE_POINTS" "$INSERT_POINTS" "$DELETE_POINTS"

echo "=== Running final standalone recall query ==="
run_query_on_index "$BASELINE_REPO" "$FINAL_INDEX_DIR" "$DATASET" "$DATA_TYPE" \
    "$FINAL_POINTS" "$QUERY_BIN" "$GT_FILE" "$WORK_DIR/final_query" \
    "$QUERY_THREADS" "$BEAMWIDTH" "$RECALL_AT" "$METRIC" "$SEARCH_L" \
    "$NODES_TO_CACHE" "$GREATOR_CXX" "$GREATOR_CC" | tee -a "$WORK_DIR/run.log"
write_ann_bench_metrics_json "$WORK_DIR/run.log" "$WORK_DIR/result.json"
