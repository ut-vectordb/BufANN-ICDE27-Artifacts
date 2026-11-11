#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DGAI"
WORKLOAD="query"
# eval_tempfiles contract: prepare materializes the full consumable here;
# query.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

source "$REPO_DIR/benchmark/$BASELINE/common.sh"

: "${DGAI_CANONICAL_DIR:?no DGAI canonical declared in benchmark/datasets/$DATASET.sh}"
: "${DGAI_CANONICAL_NAME:?no DGAI canonical declared in benchmark/datasets/$DATASET.sh}"

run_optional_fstrim

ensure_dataset
BASE_POINTS="$NPTS_BASE"
reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
provision_index_into "$WORK_DIR/index" "$DATASET" "$DGAI_CANONICAL_DIR" "$DGAI_CANONICAL_NAME"
resolve_incremental_gt "$GT_FILE" "$RECALL_AT" --base_trunc "$BASE_POINTS"
ensure_dgai_warmup_queries "$WORK_DIR"
note "Prepared $BASELINE query work tree at $WORK_DIR"
