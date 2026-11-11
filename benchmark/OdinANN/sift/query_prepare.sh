#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="OdinANN"
WORKLOAD="query"
# eval_tempfiles contract: prepare materializes the full consumable here;
# query.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

source "$REPO_DIR/benchmark/$BASELINE/common.sh"

run_optional_fstrim

ensure_dataset
BASE_POINTS="$(canonical_base_points)"
reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${BASE_POINTS}_gt${RECALL_AT}.bin"
provision_index_into "$WORK_DIR/index" "$DATASET"
resolve_incremental_gt "$GT_FILE" "$RECALL_AT" --base_trunc "$BASE_POINTS"
note "Prepared $BASELINE query work tree at $WORK_DIR"
