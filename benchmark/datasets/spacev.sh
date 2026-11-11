#!/usr/bin/env bash

DATASET="spacev"
DATA_TYPE="int8"
DIM=100
METRIC="l2"
RECALL_AT="${RECALL_AT:-10}"

NPTS_FULL=1000000000
NPTS_BASE=900000000
UPDATE_POINTS=100000000

# base.1.0B.i8bin is the first 1B vectors of upstream's 1.4B base. The 30K queries' groundtruth is computed
# against it.
__spacev_root="${SPACEV_ROOT:-$REPO_DIR/datasets/spacev}"
DATA_BIN="${DATA_BIN:-$__spacev_root/base.1.0B.i8bin}"
QUERY_BIN="${QUERY_BIN:-$__spacev_root/query.30K.i8bin}"
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__spacev_root/spacev_1000000000_gt100_30Kqueries_20260716.bin}"

__spacev_idx="${SPACEV_INDEX_DIR:-$REPO_DIR/indexes/spacev}"
CANONICAL_INDEX_0P9B="${CANONICAL_INDEX_0P9B:-$__spacev_idx/spacev0.9b_R64_L100_B32_M200_noPQ}"
# CANONICAL_R must equal the canonical's R: Greator's topology extraction uses it, and the StreamingMerger in
# insert/update uses it as the target degree of rebuilt neighborhoods.
CANONICAL_R="${CANONICAL_R:-64}"
R="${R:-$CANONICAL_R}"

__spacev_dgai="${SPACEV_DGAI_DIR:-$REPO_DIR/indexes-DGAI/spacev}"
DGAI_CANONICAL_DIR="${DGAI_CANONICAL_DIR:-$__spacev_dgai}"
DGAI_CANONICAL_NAME="${DGAI_CANONICAL_NAME:-spacev_900000000_R64_L100_PQ32_M64}"
DGAI_STRATEGY="${DGAI_STRATEGY:-21}"
