#!/usr/bin/env bash

DATASET="sift1b"
DATA_TYPE="uint8"
DIM=128
METRIC="l2"
RECALL_AT="${RECALL_AT:-10}"

NPTS_FULL=1000000000
NPTS_BASE=900000000
UPDATE_POINTS=100000000

__sift1b_root="${SIFT1B_ROOT:-$REPO_DIR/datasets/sift1b}"
DATA_BIN="${DATA_BIN:-$__sift1b_root/base.1B.u8bin}"
QUERY_BIN="${QUERY_BIN:-$__sift1b_root/query.public.10K.u8bin}"
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__sift1b_root/sift1b_1000000000_gt100.bin}"

__sift1b_idx="${SIFT1B_INDEX_DIR:-$REPO_DIR/indexes/sift1b}"
CANONICAL_INDEX_0P9B="${CANONICAL_INDEX_0P9B:-$__sift1b_idx/sift0.9b_R64_L100_B32_M200_noPQ}"
# CANONICAL_R must equal the canonical's R: Greator's topology extraction uses it, and the StreamingMerger in
# insert/update uses it as the target degree of rebuilt neighborhoods.
CANONICAL_R="${CANONICAL_R:-64}"
R="${R:-$CANONICAL_R}"

__sift1b_dgai="${SIFT1B_DGAI_DIR:-$REPO_DIR/indexes-DGAI/sift1b}"
DGAI_CANONICAL_DIR="${DGAI_CANONICAL_DIR:-$__sift1b_dgai}"
DGAI_CANONICAL_NAME="${DGAI_CANONICAL_NAME:-sift1b_900000000_R64_L100_PQ32_M64}"
DGAI_STRATEGY="${DGAI_STRATEGY:-21}"

