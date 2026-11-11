#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET="${DATASET:-sift}"
source "$REPO_DIR/benchmark/datasets/_load.sh"
BASELINE="DiskANN"
WORKLOAD="delete"
# eval_tempfiles contract: prepare materializes the full consumable here;
# delete.sh consumes it in place. See benchmark/docs/eval-tempfiles-contract.md.
WORK_DIR="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}/$DATASET/$BASELINE/$WORKLOAD"


# Percentage of the full dataset used for insert/delete updates.
UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
source "$REPO_DIR/benchmark/$BASELINE/common.sh"

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
reset_work_dir "$WORK_DIR"
link_dataset_files "$WORK_DIR"
DELETE_IDS="$WORK_DIR/ids/delete_ids.bin"
FINAL_POINTS=$((BASE_POINTS - UPDATE_POINTS))
GT_FILE="$WORK_DIR/groundtruth/${DATASET}_${FINAL_POINTS}_gt${RECALL_AT}.bin"
write_counted_ids_range "$DELETE_IDS" "$FINAL_POINTS" "$UPDATE_POINTS"

# delete starts from the shared 0.9B canonical and removes tail IDs in-place.
provision_index_into "$WORK_DIR/index" "$DATASET"

# Exact GT for the post-delete universe ([0, BASE_POINTS) minus DELETE_IDS)
# derived from the cached deep-K full GT.
resolve_incremental_gt "$GT_FILE" "$RECALL_AT" \
    --base_trunc "$BASE_POINTS" --deletes "$DELETE_IDS"

# Per-round GT for the in-binary per-round recall. Round k deletes DELETE_CAP
# IDs ending at BASE_POINTS; active set after round k is
# [0, BASE_POINTS) minus [FINAL_POINTS, FINAL_POINTS+(k+1)*DELETE_CAP). Must match
# delete.sh's round split (n_iters = UPDATE_POINTS / DELETE_CAP).
DELETE_CAP="${DELETE_CAP:-$UPDATE_POINTS}"
(( UPDATE_POINTS % DELETE_CAP == 0 )) \
    || error "UPDATE_POINTS ($UPDATE_POINTS) must be divisible by DELETE_CAP ($DELETE_CAP)"
generate_round_groundtruths "$WORK_DIR/groundtruth/rounds" "$RECALL_AT" \
    "$(( UPDATE_POINTS / DELETE_CAP ))" "$BASE_POINTS" 0 "$FINAL_POINTS" "$DELETE_CAP"

MEDOID_ID="$(python3 - "$WORK_DIR/index/${DATASET}_disk.index" <<'PYEOF'
import struct, sys
with open(sys.argv[1], 'rb') as f:
    f.seek(2 * 4 + 2 * 8)
    raw = f.read(8)
print(struct.unpack('<Q', raw)[0] if len(raw) == 8 else -1)
PYEOF
)"
if [[ "$MEDOID_ID" =~ ^[0-9]+$ ]] && (( MEDOID_ID >= FINAL_POINTS && MEDOID_ID < BASE_POINTS )); then
    error "delete workload would delete DiskANN medoid $MEDOID_ID; refusing to silently change delete IDs"
fi

note "Prepared $BASELINE delete work tree at $WORK_DIR"
