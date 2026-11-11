#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DGAI"
WORKLOAD="update"
# eval_tempfiles contract: build the work tree update.sh consumes in place. The layout (insert rows from the
# tail, delete rows from the end of the base, per-round groundtruth) is the same as OdinANN's update_prepare.sh.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

source "$REPO_DIR/benchmark/$BASELINE/common.sh"

: "${DGAI_CANONICAL_DIR:?no DGAI canonical declared in benchmark/datasets/$DATASET.sh}"
: "${DGAI_CANONICAL_NAME:?no DGAI canonical declared in benchmark/datasets/$DATASET.sh}"

run_optional_fstrim

ensure_dataset
compute_update_layout
(( INSERT_CAP == DELETE_CAP )) \
    || error "DGAI update needs INSERT_CAP == DELETE_CAP (got $INSERT_CAP and $DELETE_CAP)"
reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
write_counted_ids_range "$WORK_DIR/ids/insert_ids.bin" "$BASE_POINTS" "$INSERT_POINTS"
write_counted_ids_range "$WORK_DIR/ids/delete_ids.bin" "$((BASE_POINTS - DELETE_POINTS))" "$DELETE_POINTS"
provision_index_into "$WORK_DIR/index" "$DATASET" "$DGAI_CANONICAL_DIR" "$DGAI_CANONICAL_NAME"
generate_round_groundtruths "$WORK_DIR/groundtruth/rounds" "$RECALL_AT" \
    "$UPDATE_ROUNDS" "$BASE_POINTS" "$INSERT_CAP" \
    "$((BASE_POINTS - DELETE_POINTS))" "$DELETE_CAP"
note "Prepared $BASELINE update work tree at $WORK_DIR"
