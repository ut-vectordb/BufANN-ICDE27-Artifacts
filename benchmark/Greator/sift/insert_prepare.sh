#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="Greator"
WORKLOAD="insert"
# eval_tempfiles contract: prepare materializes the full consumable here;
# insert.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# Percentage of the full dataset used for insert/delete updates.
# INSERT_POINTS, if set, overrides the per-percent count *without* shifting
# BASE_POINTS off the canonical (compute_split derives BASE from
# UPDATE_PERCENT, then we re-pin UPDATE_POINTS below).
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
source "$REPO_DIR/benchmark/$BASELINE/common.sh"

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
reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
INSERT_IDS="$WORK_DIR/ids/insert_ids.bin"
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${TOTAL_POINTS}_gt${RECALL_AT}.bin"
write_counted_ids_range "$INSERT_IDS" "$BASE_POINTS" "$UPDATE_POINTS"

provision_index_into "$BASELINE_REPO" "$WORK_DIR/index" "$DATASET" "$DATA_TYPE"

# Exact GT for the post-insert universe [0, BASE+INSERT) derived from the
# cached deep-K full GT.
resolve_incremental_gt "$GT_FILE" "$RECALL_AT" \
    --base_trunc "$((BASE_POINTS + UPDATE_POINTS))"

# Per-round GT for the in-binary per-round recall. Round k inserts INSERT_CAP
# IDs, so the active set after round k is the prefix
# [0, BASE_POINTS + (k+1)*INSERT_CAP). Must match insert.sh's round split.
INSERT_CAP="${INSERT_CAP:-$UPDATE_POINTS}"
(( UPDATE_POINTS % INSERT_CAP == 0 )) \
    || error "UPDATE_POINTS ($UPDATE_POINTS) must be divisible by INSERT_CAP ($INSERT_CAP)"
generate_round_groundtruths "$WORK_DIR/groundtruth/rounds" "$RECALL_AT" \
    "$(( UPDATE_POINTS / INSERT_CAP ))" "$BASE_POINTS" "$INSERT_CAP" 0 0

note "Prepared $BASELINE insert work tree at $WORK_DIR"
