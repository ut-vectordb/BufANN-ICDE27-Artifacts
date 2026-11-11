#!/usr/bin/env bash

DATASET="sift100m"
DATA_TYPE="uint8"
DIM=128
METRIC="l2"
RECALL_AT="${RECALL_AT:-10}"

NPTS_FULL=100000000
NPTS_BASE=90000000
UPDATE_POINTS=10000000

__sift100m_root="${SIFT100M_ROOT:-$REPO_DIR/datasets/sift100m}"
DATA_BIN="${DATA_BIN:-$__sift100m_root/sift100m_base.bin}"
QUERY_BIN="${QUERY_BIN:-$__sift100m_root/sift100m_query.bin}"

GT_BIN_KCACHED="${GT_BIN_KCACHED:-$__sift100m_root/sift100m_100000000_gt100.bin}"

__sift100m_idx="${SIFT100M_INDEX_DIR:-$REPO_DIR/indexes/sift100m}"
CANONICAL_INDEX_0P9B="${CANONICAL_INDEX_0P9B:-$__sift100m_idx/sift100m_90M_R64_L100_M8_noPQ}"
CANONICAL_R="${CANONICAL_R:-64}"
R="${R:-$CANONICAL_R}"

# DGAI canonical layout: the graph and vectors are split into <name>.layout/, PQ/ holds DGAI's PQ files with
# _pq_compressed_refined.bin the hybrid code file, and
# <name>_disk.index_header is the first 4 KiB of the coupled index, which is all DGAI reads from that file.
__sift100m_dgai="${SIFT100M_DGAI_DIR:-$REPO_DIR/indexes-DGAI/sift100m}"
DGAI_CANONICAL_DIR="${DGAI_CANONICAL_DIR:-$__sift100m_dgai}"
DGAI_CANONICAL_NAME="${DGAI_CANONICAL_NAME:-sift100m_90000000_R64_L100_PQ32_M64}"
DGAI_STRATEGY="${DGAI_STRATEGY:-23}"
