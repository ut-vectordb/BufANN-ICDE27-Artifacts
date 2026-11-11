#!/usr/bin/env bash
# Dataset dispatcher. Sourced by every leaf workload script AFTER it sets
# REPO_DIR and DATASET, and BEFORE it sources the baseline common.sh.
#
#   DATASET="${DATASET:-sift}"
#   source "$REPO_DIR/benchmark/datasets/_load.sh"
#
# It validates DATASET and sources the single source of truth for that dataset
# (benchmark/datasets/<DATASET>.sh), which defines identity + absolute source
# paths. Every value in those files is env-overridable (${VAR:-default}).
#
# Depends only on REPO_DIR + DATASET (no utils.sh / common.sh), so there is no
# sourcing-order trap.

: "${REPO_DIR:?REPO_DIR must be set before sourcing datasets/_load.sh}"
: "${DATASET:?DATASET must be set before sourcing datasets/_load.sh}"

__bench_datasets_dir="${BENCH_DATASETS_DIR:-$REPO_DIR/benchmark/datasets}"
__dataset_file="$__bench_datasets_dir/$DATASET.sh"

if [[ ! -f "$__dataset_file" ]]; then
    __avail="$(cd "$__bench_datasets_dir" 2>/dev/null && \
        ls *.sh 2>/dev/null | grep -v '^_' | sed 's/\.sh$//' | tr '\n' ' ')"
    echo "ERROR: unknown dataset '$DATASET' (no such file: $__dataset_file)" >&2
    echo "       available datasets: ${__avail:-<none>}" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$__dataset_file"
