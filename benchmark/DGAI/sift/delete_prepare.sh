#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DGAI"
WORKLOAD="delete"
# eval_tempfiles contract: prepare materializes the full consumable here;
# delete.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

# DELETE_POINTS, if set, overrides the per-percent count without moving BASE_POINTS
# off the canonical. Mirrors the OdinANN scripts.
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
source "$REPO_DIR/benchmark/$BASELINE/common.sh"

: "${DGAI_CANONICAL_DIR:?no DGAI canonical declared in benchmark/datasets/$DATASET.sh}"
: "${DGAI_CANONICAL_NAME:?no DGAI canonical declared in benchmark/datasets/$DATASET.sh}"

run_optional_fstrim

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
(( UPDATE_POINTS % DELETE_CAP == 0 )) \
    || error "UPDATE_POINTS ($UPDATE_POINTS) must be divisible by DELETE_CAP ($DELETE_CAP)"

reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
IDS_FILE="$WORK_DIR/ids/delete_ids.bin"
# Deletes remove the tail of the base, [BASE_POINTS - UPDATE_POINTS, BASE_POINTS).
write_counted_ids_range "$IDS_FILE" "$((BASE_POINTS - UPDATE_POINTS))" "$UPDATE_POINTS"

provision_index_into "$WORK_DIR/index" "$DATASET" "$DGAI_CANONICAL_DIR" "$DGAI_CANONICAL_NAME"

# Per-round GT: round k deletes DELETE_CAP IDs from the start of the
# deleted range, so the active set after round k is [0, BASE_POINTS) minus
# [BASE_POINTS - UPDATE_POINTS, BASE_POINTS - UPDATE_POINTS + (k+1)*DELETE_CAP).
generate_round_groundtruths "$WORK_DIR/groundtruth/rounds" "$RECALL_AT" \
    "$(( UPDATE_POINTS / DELETE_CAP ))" "$BASE_POINTS" 0 "$((BASE_POINTS - UPDATE_POINTS))" "$DELETE_CAP"

note "Prepared $BASELINE delete work tree at $WORK_DIR"
