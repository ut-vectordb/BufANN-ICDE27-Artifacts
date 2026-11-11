#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DGAI"
WORKLOAD="insert"
# eval_tempfiles contract: prepare materializes the full consumable here;
# insert.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"

# INSERT_POINTS, if set, overrides the per-percent count without moving BASE_POINTS
# off the canonical. Mirrors the OdinANN scripts.
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
source "$REPO_DIR/benchmark/$BASELINE/common.sh"

: "${DGAI_CANONICAL_DIR:?no DGAI canonical declared in benchmark/datasets/$DATASET.sh}"
: "${DGAI_CANONICAL_NAME:?no DGAI canonical declared in benchmark/datasets/$DATASET.sh}"

run_optional_fstrim

ensure_dataset
compute_split
if [[ -n "${INSERT_POINTS:-}" ]]; then
    [[ "$INSERT_POINTS" =~ ^[0-9]+$ && "$INSERT_POINTS" -gt 0 ]] \
        || error "INSERT_POINTS must be a positive integer (got: $INSERT_POINTS)"
    (( INSERT_POINTS < TOTAL_POINTS - BASE_POINTS + 1 )) \
        || error "INSERT_POINTS ($INSERT_POINTS) exceeds available tail rows ($((TOTAL_POINTS - BASE_POINTS)))"
    UPDATE_POINTS="$INSERT_POINTS"
fi
INSERT_CAP="${INSERT_CAP:-$UPDATE_POINTS}"
(( UPDATE_POINTS % INSERT_CAP == 0 )) \
    || error "UPDATE_POINTS ($UPDATE_POINTS) must be divisible by INSERT_CAP ($INSERT_CAP)"

reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
IDS_FILE="$WORK_DIR/ids/insert_ids.bin"
write_counted_ids_range "$IDS_FILE" "$BASE_POINTS" "$UPDATE_POINTS"

provision_index_into "$WORK_DIR/index" "$DATASET" "$DGAI_CANONICAL_DIR" "$DGAI_CANONICAL_NAME"

# Per-round GT: round k inserts INSERT_CAP IDs, so the active set after
# round k is the prefix [0, BASE_POINTS + (k+1)*INSERT_CAP).
generate_round_groundtruths "$WORK_DIR/groundtruth/rounds" "$RECALL_AT" \
    "$(( UPDATE_POINTS / INSERT_CAP ))" "$BASE_POINTS" "$INSERT_CAP" 0 0

note "Prepared $BASELINE insert work tree at $WORK_DIR"
