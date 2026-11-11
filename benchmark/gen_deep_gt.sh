#!/usr/bin/env bash
# Generate the K=100 "deep" groundtruth cache for a dataset, written to the
# persistent path its config declares (GT_BIN_KCACHED). This one file is the
# single source of truth for GT: resolve_incremental_gt derives every
# per-workload / per-round / sub-N groundtruth from it by filtering. The
# benchmark never regenerates it — resolve_incremental_gt fails fast and points
# here — so run this once per dataset (the output is durable: it lives in
# shared-datasets, not the disposable eval_cached).
#
# Expensive: a full-base exact kNN. Hours on 1B-point datasets.
#
# Usage: benchmark/gen_deep_gt.sh <dataset> [--force]
#   --force   recompute even if the cache already exists
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?usage: gen_deep_gt.sh <dataset> [--force]}"; shift || true
FORCE=0; [[ "${1:-}" == "--force" ]] && FORCE=1
DEEP_K="${GT_DEEP_K:-100}"

# Dataset identity + paths (DATA_BIN, QUERY_BIN, DATA_TYPE, GT_BIN_KCACHED, ...).
source "$REPO_DIR/benchmark/datasets/_load.sh"
# note / error / bin_count / configure_cmake_repo.
source "$REPO_DIR/benchmark/utils.sh"

TOTAL_POINTS="$(bin_count "$DATA_BIN")"
# The dataset cfg must declare GT_BIN_KCACHED (the persistent deep-cache path).
[[ -n "${GT_BIN_KCACHED:-}" ]] || error "GT_BIN_KCACHED unset for $DATASET — declare it in benchmark/datasets/$DATASET.sh"
out="$GT_BIN_KCACHED"

if [[ -e "$out" && "$FORCE" -ne 1 ]]; then
    note "deep GT cache already present: $out (use --force to recompute)"
    exit 0
fi
[[ -e "$DATA_BIN" ]]  || error "DATA_BIN missing: $DATA_BIN"
[[ -e "$QUERY_BIN" ]] || error "QUERY_BIN missing: $QUERY_BIN"

mkdir -p "$(dirname "$out")"
configure_cmake_repo "$REPO_DIR/baselines/DiskANN" compute_groundtruth
note "Computing K=$DEEP_K deep GT for '$DATASET' ($TOTAL_POINTS pts, $DATA_TYPE) -> $out"
note "  (full-base exact kNN; this can take hours on large datasets)"
"$REPO_DIR/baselines/DiskANN/build/tests/utils/compute_groundtruth" \
    "$DATA_TYPE" "$DATA_BIN" "$QUERY_BIN" "$DEEP_K" "$out"
note "deep GT cache written: $out"
note "  (back it up to durable storage if desired)"
