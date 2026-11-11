#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="BufANN"
WORKLOAD="query"
# Query-only direct mode: keep small run artifacts in eval_tempfiles, but read
# the immutable converted BufANN index directly from eval_cached.
WORK_DIR="${BUFANN_DIRECT_WORK_DIR:-${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/query_direct}"

SEARCH_L="${SEARCH_L:-100}"
BEAMWIDTH="${BEAMWIDTH:-4}"
QUERY_THREADS="${QUERY_THREADS:-$(nproc)}"

source "$REPO_DIR/benchmark/$BASELINE/common.sh"

ensure_dataset
compute_split
mkdir -p "$WORK_DIR"
link_dataset_files "$WORK_DIR"

if [[ -n "${BUFANN_DIRECT_INDEX_DIR:-}" ]]; then
    INDEX_DIR="${BUFANN_DIRECT_INDEX_DIR%/}"
elif [[ -n "${BUFANN_PREBUILT_INDEX_DIR:-}" ]]; then
    INDEX_DIR="${BUFANN_PREBUILT_INDEX_DIR%/}"
else
    INDEX_DIR="$(bufann_cache_dir_for_workload "$WORKLOAD")"
fi
INDEX_PREFIX="$INDEX_DIR/$DATASET"

if [[ -z "${BUFANN_DIRECT_ALLOW_RUNNING_PREPARE:-}" &&
      -z "${BUFANN_DIRECT_INDEX_DIR:-}" &&
      -z "${BUFANN_PREBUILT_INDEX_DIR:-}" ]] &&
   pgrep -f 'benchmark/BufANN/sift/query_prepare.sh' >/dev/null; then
    error "BufANN query_prepare.sh is still running; wait for eval_cached store to finish before direct query.
       Set BUFANN_DIRECT_ALLOW_RUNNING_PREPARE=1 to bypass this guard."
fi

missing=()
for f in \
    "${INDEX_PREFIX}.heap" \
    "${INDEX_PREFIX}.meta" \
    "${INDEX_PREFIX}.medoids" \
    "${INDEX_PREFIX}_pq_pivots.bin" \
    "${INDEX_PREFIX}_pq_compressed.bin"; do
    [[ -f "$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} > 0 )); then
    printf 'missing direct BufANN index family at %s\n' "$INDEX_PREFIX" >&2
    printf '  missing: %s\n' "${missing[@]}" >&2
    error \
"Run query_prepare.sh to complete and store the conversion in eval_cached,
       or set BUFANN_DIRECT_INDEX_DIR / BUFANN_PREBUILT_INDEX_DIR."
fi

GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
resolve_incremental_gt "$GT_FILE" "$RECALL_AT" --base_trunc "$BASE_POINTS"
[[ -f "$GT_FILE" ]] || error "missing query groundtruth: $GT_FILE"

ensure_bufann_warmup_queries "$WORK_DIR" "$BASE_POINTS"
if [[ -z "${BUFANN_WARMUP_QUERY_FILE:-}" && -e "$WORK_DIR/warmup_queries.bin" ]]; then
    export BUFANN_WARMUP_QUERY_FILE="$WORK_DIR/warmup_queries.bin"
fi

note "Running direct BufANN query against $INDEX_PREFIX"
run_bufann_driver "$INDEX_PREFIX" search_only \
    "$QUERY_BIN" "$GT_FILE" "$WORK_DIR" \
    "$SEARCH_L" "$BEAMWIDTH" "$QUERY_THREADS" "$RECALL_AT"
