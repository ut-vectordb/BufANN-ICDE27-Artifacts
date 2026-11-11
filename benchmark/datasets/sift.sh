#!/usr/bin/env bash

DATASET="sift"
DATA_TYPE="float"
DIM=128
METRIC="l2"
RECALL_AT="${RECALL_AT:-10}"

NPTS_FULL=1000000
NPTS_BASE=900000

__sift_root="${SIFT_ROOT:-$REPO_DIR/datasets/sift}"

DATA_BIN="${DATA_BIN:-$__sift_root/sift_base.bin}"
QUERY_BIN="${QUERY_BIN:-$__sift_root/sift_query.bin}"
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__sift_root/sift_1000000_gt100.bin}"

DATA_FVECS="${DATA_FVECS:-$__sift_root/sift_base.fvecs}"
QUERY_FVECS="${QUERY_FVECS:-$__sift_root/sift_query.fvecs}"

__sift_idx="${SIFT_INDEX_DIR:-$REPO_DIR/indexes/sift}"
CANONICAL_INDEX_0P9B="${CANONICAL_INDEX_0P9B:-$__sift_idx/sift0.9M_R32_L100_B0.3_M2}"
# CANONICAL_R must equal the canonical's R: Greator's topology extraction uses it, and the StreamingMerger in
# insert/update uses it as the target degree of rebuilt neighborhoods.
CANONICAL_R="${CANONICAL_R:-32}"
R="${R:-$CANONICAL_R}"

