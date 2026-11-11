#!/usr/bin/env bash

DATASET="turing"
DATA_TYPE="float"
DIM=100
METRIC="l2"
RECALL_AT="${RECALL_AT:-10}"

NPTS_FULL=1000000000
NPTS_BASE=900000000
UPDATE_POINTS=100000000

__turing_root="${TURING_ROOT:-$REPO_DIR/datasets/turing}"
DATA_BIN="${DATA_BIN:-$__turing_root/base1b.fbin}"
QUERY_BIN="${QUERY_BIN:-$__turing_root/query100K.fbin}"
GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__turing_root/query_gt100.official.bin}"

__turing_idx="${TURING_INDEX_DIR:-$REPO_DIR/indexes/turing}"
CANONICAL_INDEX_0P9B="${CANONICAL_INDEX_0P9B:-$__turing_idx/turing0.9b_R100_L100_B32_M80_noPQ}"
CANONICAL_R="${CANONICAL_R:-100}"
R="${R:-$CANONICAL_R}"

__turing_dgai="${TURING_DGAI_DIR:-$REPO_DIR/indexes-DGAI/turing}"
DGAI_CANONICAL_DIR="${DGAI_CANONICAL_DIR:-$__turing_dgai}"
DGAI_CANONICAL_NAME="${DGAI_CANONICAL_NAME:-turing_900000000_R100_L100_PQ32_M64}"
DGAI_STRATEGY="${DGAI_STRATEGY:-21}"
