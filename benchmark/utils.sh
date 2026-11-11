#!/usr/bin/env bash

: "${REPO_DIR:?REPO_DIR must be set before sourcing benchmark/utils.sh}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$REPO_DIR/benchmark}"
BENCH_DEPS_PREFIX="${BENCH_DEPS_PREFIX:-$(cd "$REPO_DIR/.." && pwd)/baselines_env}"
BENCHMARK_CMAKE_BUILD_TYPE="Release"

# Two explicit, non-hidden, overridable roots (see
# benchmark/docs/eval-tempfiles-contract.md):
#   EVAL_TEMPFILES - the live working tree. *_prepare.sh materializes the full
#                    consumable here; *.sh reads it in place. Safe to wipe
#                    (re-create by re-running prepare).
#   EVAL_CACHED    - prepare-internal cache for slow-to-produce artifacts only
#                    (Greator topology, derived GT).
#                    Touched solely by *_prepare.sh via eval_cache_* helpers.
#                    Set EVAL_CACHE_NO_STORE=1 to skip writes into EVAL_CACHED
#                    (restores still happen); useful for one-off runs where the
#                    artifact should land in EVAL_TEMPFILES but not pollute the
#                    persistent cache.
EVAL_CACHED="${EVAL_CACHED:-$REPO_DIR/eval_cached}"
EVAL_TEMPFILES="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}"

error() {
    echo "ERROR: $*" >&2
    exit 1
}

note() {
    echo "=== $* ==="
}

run_optional_fstrim() {
    local helper="${FSTRIM_HELPER:-/usr/local/bin/fstrim-helper}"
    if [[ ! -e "$helper" ]]; then
        note "Skipping fstrim: $helper not found"
        return 0
    fi
    note "Running fstrim via $helper"
    "$helper" || error "fstrim failed via $helper"
}

write_ann_bench_metrics_json() {
    local run_log out_file
    run_log="${1:-${WORK_DIR:-}/run.log}"
    out_file="${2:-${WORK_DIR:-}/result.json}"
    [[ -n "$run_log" && -n "$out_file" ]] || return 0
    mkdir -p "$(dirname "$out_file")"
    grep -h '^{"baseline":' "$run_log" > "$out_file" || true
}

# Permit destructive ops only under the two eval roots. Note this intentionally
# allows rm -rf under EVAL_CACHED too: *_prepare.sh legitimately resets its own
# cached input dir, so EVAL_CACHED is reusable but not append-only.
safe_mutable_path() {
    local path abs cached temp
    path="$1"
    abs="$(readlink -m "$path")"
    cached="$(readlink -m "$EVAL_CACHED")"
    temp="$(readlink -m "$EVAL_TEMPFILES")"
    case "$abs" in
        "$cached"|"$cached"/*|"$temp"|"$temp"/*) printf '%s\n' "$abs" ;;
        *) error "refusing to mutate path outside eval roots: $path" ;;
    esac
}

# Back-compat alias for any out-of-tree script still calling the old name.
safe_work_path() { safe_mutable_path "$@"; }

reset_work_dir() {
    local dir
    dir="$(safe_mutable_path "$1")"
    rm -rf "$dir"
    mkdir -p "$dir"
}

# Generic primitive: materialize a real, owner-writable copy of a directory
# tree into an eval root. Symlinks are dereferenced (-L). Under the new
# contract this is called from *_prepare.sh (to restore/store eval_cached
# dir artifacts), NOT from workload *.sh. Large files go through fastcopy.sh
# (parallel O_DIRECT); nested dirs via cp -rL.
copy_dir_contents() {
    local src dst fcp f d
    src="$1"
    dst="$2"
    [[ -d "$src" ]] || error "missing source directory: $src"
    reset_work_dir "$dst"
    fcp="${FASTCOPY:-$BENCHMARK_DIR/fastcopy.sh}"
    for f in "$src"/*; do
        [[ -e "$f" ]] || continue          # empty dir / no glob match
        d="$dst/$(basename "$f")"
        if [[ -d "$f" ]]; then
            cp -rL "$f" "$d"               # nested dir (rare; index dirs are flat)
        else
            "$fcp" "$f" "$d" || error "fastcopy failed: $f -> $d"
        fi
    done
    # The shared canonical is 0400 (fail-fast guard); workloads open the
    # execution copy O_RDWR, so it must be owner-writable here.
    chmod -R u+w "$dst"
}

# ── eval_tempfiles contract ───────────────────────────────────────────
# See benchmark/docs/eval-tempfiles-contract.md. *_prepare.sh materializes a
# real, owner-writable working tree under eval_tempfiles; *.sh consumes it in
# place. eval_cached is a prepare-internal cache for slow-to-produce artifacts
# only (Greator topology sidecar, derived-data GT),
# accessed exclusively via the two helpers below.

# Restore a cached artifact into $dest. Returns 0 (restored) or 1 (cache miss;
# caller must then produce $dest itself and call eval_cache_store). $cache_path
# may be a file or a directory.
eval_cache_try_restore() {
    local cache_path dest
    cache_path="$1"
    dest="$2"
    [[ -e "$cache_path" ]] || return 1
    if [[ -d "$cache_path" ]]; then
        copy_dir_contents "$cache_path" "$dest"
    else
        mkdir -p "$(dirname "$dest")"
        local fcp="${FASTCOPY:-$BENCHMARK_DIR/fastcopy.sh}"
        "$fcp" "$cache_path" "$dest" || error "cache restore failed: $cache_path -> $dest"
        chmod u+w "$dest" 2>/dev/null || true
    fi
    note "Restored from eval_cached: $cache_path -> $dest"
    return 0
}

# Persist a freshly produced artifact into the prepare-internal cache. $src may
# be a file or a directory. Only ever called from *_prepare.sh.
eval_cache_store() {
    local src cache_path
    src="$1"
    cache_path="$2"
    [[ -e "$src" ]] || error "eval_cache_store: missing src $src"
    if [[ -n "${EVAL_CACHE_NO_STORE:-}" ]]; then
        note "Skipped eval_cached store (EVAL_CACHE_NO_STORE set): $src -> $cache_path"
        return 0
    fi
    safe_mutable_path "$cache_path" >/dev/null   # refuse outside eval roots
    if [[ -d "$src" ]]; then
        copy_dir_contents "$src" "$cache_path"
    else
        mkdir -p "$(dirname "$cache_path")"
        local fcp="${FASTCOPY:-$BENCHMARK_DIR/fastcopy.sh}"
        "$fcp" "$src" "$cache_path" || error "cache store failed: $src -> $cache_path"
    fi
    note "Stored to eval_cached: $src -> $cache_path"
}

# Materialize the canonical DiskANN family as REAL owner-writable files under
# $index_dir as <dataset><suffix> (fastcopy, parallel O_DIRECT). Replaces the
# old symlink-then-copy-at-runtime path: under the new contract *_prepare.sh
# produces the real index directly in eval_tempfiles. The canonical source is
# chmod -w'd as a fail-fast guard (a deliberate out-of-band rebuild rm's it,
# needing dir write not file write).
materialize_canonical_family() {
    local canon="$1" index_dir="$2" dataset="$3" s src dst fcp copied=0
    safe_mutable_path "$index_dir" >/dev/null   # refuse outside eval roots
    mkdir -p "$index_dir"
    fcp="${FASTCOPY:-$BENCHMARK_DIR/fastcopy.sh}"
    for s in "${CANONICAL_FAMILY_SUFFIXES[@]}"; do
        src="${canon}${s}"
        [[ -e "$src" ]] || continue            # centroids/medoids optional
        src="$(readlink -f "$src")"
        chmod -w "$src" 2>/dev/null || true
        dst="${index_dir}/${dataset}${s}"
        "$fcp" "$src" "$dst" || error "fastcopy failed: $src -> $dst"
    done
    for s in "${CANONICAL_REQUIRED_SUFFIXES[@]}"; do
        [[ -e "${index_dir}/${dataset}${s}" ]] && copied=$((copied + 1))
    done
    (( copied == ${#CANONICAL_REQUIRED_SUFFIXES[@]} )) || \
        error "canonical materialization incomplete under $index_dir"
    chmod -R u+w "$index_dir"
}

# Dataset paths come from the per-dataset file of truth (benchmark/datasets/
# <name>.sh, loaded via datasets/_load.sh). ensure_dataset only materializes a
# .bin when the dataset declares an optional <X>_FVECS source and the .bin is
# absent. Datasets whose .bin already exists (e.g. large prebuilt sets on shared
# storage) declare no *_FVECS and this is a no-op beyond path canonicalization.
__ensure_bin() {
    local kind="$1" bin="$2" fvecs="$3"
    if [[ ! -f "$bin" ]]; then
        [[ -n "$fvecs" && -f "$fvecs" ]] || \
            error "missing $kind vectors: $bin not found and no convertible source (${fvecs:-<none>})"
        python3 "$BENCHMARK_DIR/convert_fvec_to_bin.py" "$fvecs" "$bin"
    fi
}

ensure_dataset() {
    : "${DATA_BIN:?DATA_BIN unset - dataset config not loaded?}"
    : "${QUERY_BIN:?QUERY_BIN unset - dataset config not loaded?}"
    __ensure_bin "dataset" "$DATA_BIN" "${DATA_FVECS:-}"
    __ensure_bin "query"   "$QUERY_BIN" "${QUERY_FVECS:-}"
    DATA_BIN="$(readlink -f "$DATA_BIN")"
    QUERY_BIN="$(readlink -f "$QUERY_BIN")"
}

# Exact post-mutation GT derived from a deep-K cached full-base GT
# ($GT_BIN_KCACHED, e.g. K=100). Filters the cached top-K_cached list by the
# workload's delete / insert / base_trunc semantics; output matches the
# compute_groundtruth file format. Builds incremental_groundtruth on demand,
# then fails fast if the deep cached GT is missing.
#
# Args: <out_gt> <K_out> [--inserts FILE] [--deletes FILE] [--base_trunc N]
#       [--delete_range BEGIN COUNT]
resolve_incremental_gt() {
    local out_gt="$1" k_out="$2"
    shift 2
    local src="$REPO_DIR/baselines/DiskANN/tests/utils/incremental_groundtruth.cpp"
    local bin="${INCREMENTAL_GT_BIN:-$REPO_DIR/baselines/DiskANN/build/tests/utils/incremental_groundtruth}"
    # K=100 deep cache: the dataset cfg must declare GT_BIN_KCACHED (persistent path).
    [[ -n "${GT_BIN_KCACHED:-}" ]] || error "GT_BIN_KCACHED unset for $DATASET — declare it in benchmark/datasets/$DATASET.sh"
    [[ -f "$src" ]] || error "incremental_groundtruth source missing: $src"
    if [[ ! -x "$bin" || "$src" -nt "$bin" ]]; then
        mkdir -p "$(dirname "$bin")"
        note "Building incremental_groundtruth ($(basename "$src"))"
        g++ "$src" -O3 -std=c++17 -o "$bin"
    fi
    [[ -x "$bin" ]] || error "missing incremental_groundtruth binary after build: $bin"
    [[ -e "$GT_BIN_KCACHED" ]] || error "missing deep GT cache: $GT_BIN_KCACHED
       Generate it once with: benchmark/gen_deep_gt.sh $DATASET"
    mkdir -p "$(dirname "$out_gt")"
    note "Deriving exact GT (K=$k_out) from $GT_BIN_KCACHED"
    "$bin" "$GT_BIN_KCACHED" "$k_out" "$out_gt" "$@"
}

# Generate one groundtruth file per round into $gt_dir as
# round_<k>_gt<k_out>.bin, for the per-round recall measured inside the
# insert/delete/mixed binaries. The active set after round k (0-indexed) is a
# numeric function of contiguous ID ranges, so each round is a single
# incremental_groundtruth call with no per-round ID files:
#
#   active_k = [0, base_trunc_start + (k+1)*ic)  minus  [del_begin, del_begin + (k+1)*dc)
#
# Per workload (see workload-semantics.md):
#   insert-only: base_trunc_start=BASE_POINTS,  ic=INSERT_CAP, dc=0
#   delete-only: base_trunc_start=BASE_POINTS,  ic=0,          del_begin=BASE_POINTS-DELETE_PTS, dc=DELETE_CAP
#   mixed:       base_trunc_start=BASE_POINTS,  ic=INSERT_CAP, del_begin=BASE_POINTS-DELETE_PTS, dc=DELETE_CAP
generate_round_groundtruths() {
    local gt_dir="$1" k_out="$2" n_iters="$3" base_trunc_start="$4" ic="$5" del_begin="$6" dc="$7"
    mkdir -p "$gt_dir"
    local k bt out
    for (( k=0; k<n_iters; k++ )); do
        bt=$(( base_trunc_start + (k + 1) * ic ))
        out="$gt_dir/round_${k}_gt${k_out}.bin"
        if (( dc > 0 )); then
            resolve_incremental_gt "$out" "$k_out" \
                --base_trunc "$bt" --delete_range "$del_begin" "$(( (k + 1) * dc ))"
        else
            resolve_incremental_gt "$out" "$k_out" --base_trunc "$bt"
        fi
    done
    note "Generated $n_iters per-round GT files in $gt_dir"
}


# Cached wrappers around the slow Python bin_slice / write_update_final_*
# helpers below. bin_slice.py is single-threaded; on a 1B-point dataset a full
# 115 GB prefix slice is multi-minute. The slices are deterministic in
# (source dataset id, start, count, dtype), so prepare can cache them in
# eval_cached and instantly fastcopy back on re-prepare. Cross-workload reuse
# is real: at default 10% updates insert's base.bin and delete's final.bin are
# the same prefix slice; insert.bin and delete.bin are the same range slice.
# Source identity is keyed via $DATASET (the cache lives under
# $EVAL_CACHED/$DATASET/data/); overriding DATA_BIN= to a different file under
# the same DATASET would lie - same caveat as the canonical materialization.

# Install a read-only data-bin slice. On cache hit, symlink eval_tempfiles
# at the cache file (no fastcopy); on miss, run the producer (which writes
# $dst as a real file in eval_tempfiles), then mv the real file into the
# cache (instant rename - same filesystem) and symlink $dst back at it.
# Net: ONE copy of the data on disk (in cache), eval_tempfiles is a symlink.
# Only safe for read-only artifacts. Do NOT use for the index family, the
# topology sidecar, or anything a workload opens O_RDWR - those would write
# through the symlink and corrupt the cache.
_cache_install_data_bin() {
    local cache_path dst
    cache_path="$1"
    dst="$2"
    shift 2
    mkdir -p "$(dirname "$dst")"
    if [[ -e "$cache_path" ]]; then
        ln -sfn "$(readlink -f "$cache_path")" "$dst"
        note "Symlinked from eval_cached: $cache_path -> $dst"
        return 0
    fi
    safe_mutable_path "$cache_path" >/dev/null   # refuse outside eval roots
    mkdir -p "$(dirname "$cache_path")"
    "$@"
    [[ -e "$dst" && ! -L "$dst" ]] || \
        error "_cache_install_data_bin: producer left no real file at $dst"
    mv "$dst" "$cache_path"
    ln -sfn "$(readlink -f "$cache_path")" "$dst"
    note "Stored to eval_cached and symlinked: $cache_path <- $dst"
}

cached_write_bin_prefix() {
    local src dst count dtype cache
    src="$1"
    dst="$2"
    count="$3"
    dtype="${4:-${DATA_TYPE:-float}}"
    cache="$EVAL_CACHED/$DATASET/data/prefix_count${count}_${dtype}.bin"
    _cache_install_data_bin "$cache" "$dst" \
        write_bin_prefix "$src" "$dst" "$count"
}

cached_write_bin_range() {
    local src dst start count dtype cache
    src="$1"
    dst="$2"
    start="$3"
    count="$4"
    dtype="${5:-${DATA_TYPE:-float}}"
    cache="$EVAL_CACHED/$DATASET/data/range_start${start}_count${count}_${dtype}.bin"
    _cache_install_data_bin "$cache" "$dst" \
        write_bin_range "$src" "$dst" "$start" "$count"
}

cached_write_update_final_bin() {
    # Final-state bin = rows [0, base-delete) ++ rows [base, base+insert).
    # Cache key is parameterized on insert (not total) so a smaller workload
    # gets its own slice instead of fabricating absent tail rows.
    local src dst base insert delete dtype cache
    src="$1"
    dst="$2"
    base="$3"
    insert="$4"
    delete="$5"
    dtype="${6:-${DATA_TYPE:-float}}"
    cache="$EVAL_CACHED/$DATASET/data/update_final_base${base}_insert${insert}_delete${delete}_${dtype}.bin"
    _cache_install_data_bin "$cache" "$dst" \
        write_update_final_bin "$src" "$dst" "$base" "$insert" "$delete" "$dtype"
}

cached_write_update_final_tags() {
    local dst base insert delete cache
    dst="$1"
    base="$2"
    insert="$3"
    delete="$4"
    cache="$EVAL_CACHED/$DATASET/data/update_final_tags_base${base}_insert${insert}_delete${delete}.bin"
    _cache_install_data_bin "$cache" "$dst" \
        write_update_final_tags "$dst" "$base" "$insert" "$delete"
}

# ── Canonical index consumption ───────────────────────────────────────
# The benchmark consumes a prebuilt canonical DiskANN-format index and
# converts it to the per-baseline format; it never builds the canonical.
# All workloads consume the 0.9B base canonical. Workload mutations are
# represented by GT filters and ID ranges, not by switching to the 1B index.
# Canonical-index baselines fail fast when no canonical is configured.
CANONICAL_FAMILY_SUFFIXES=(
    _disk.index _disk.index.tags _pq_pivots.bin _pq_compressed.bin
    _disk.index_centroids.bin _disk.index_medoids.bin
)
CANONICAL_REQUIRED_SUFFIXES=(
    _disk.index _disk.index.tags _pq_pivots.bin _pq_compressed.bin
)

resolve_canonical_prefix() {
    case "${1:-}" in
        # query_then_insert: same 0.9B base as insert (RSS check; no merge).
        query|delete|insert|update|query_then_insert) printf '%s' "${CANONICAL_INDEX_0P9B:-}" ;;
        *)             printf '%s' "" ;;
    esac
}

# Materialize an identity `_disk.index.tags` file (tag[i] = i) at the canonical
# prefix if it's missing. The whole benchmark assumes tag space == base ID
# space (see write_update_final_tags + survey in `.tags` consumers), so an
# identity file is the right default for any dataset whose canonical wasn't
# built with tag mode. npts is read from `_pq_compressed.bin`, which is a
# standard bin file (int32 npts, int32 pq_chunks, ...). No-op if the .tags
# already exists.
ensure_canonical_tags() {
    local prefix="$1" tag_file pq_file npts
    tag_file="${prefix}_disk.index.tags"
    [[ -e "$tag_file" ]] && return 0
    pq_file="${prefix}_pq_compressed.bin"
    [[ -e "$pq_file" ]] || return 0  # let require_canonical report the missing PQ
    npts="$(python3 - "$pq_file" <<'PYEOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    n, _ = struct.unpack("<ii", f.read(8))
print(n)
PYEOF
)" || error "ensure_canonical_tags: failed to read npts from $pq_file"
    note "Materializing identity .tags at $tag_file (npts=$npts)"
    python3 - "$tag_file" "$npts" <<'PYEOF'
import struct, sys, numpy as np
path, n = sys.argv[1], int(sys.argv[2])
with open(path, "wb") as f:
    f.write(struct.pack("<ii", n, 1))
    np.arange(n, dtype=np.uint32).tofile(f)
PYEOF
}

# Hard-fail if the canonical is absent/incomplete (consume-only: never built
# here). Auto-creates the identity `.tags` first if other family files exist.
require_canonical() {
    local prefix="$1" s
    ensure_canonical_tags "$prefix"
    for s in "${CANONICAL_REQUIRED_SUFFIXES[@]}"; do
        [[ -e "${prefix}${s}" ]] || error \
"canonical index incomplete: missing ${prefix}${s}
       The benchmark consumes a prebuilt canonical and never builds it.
       Build/stage the canonical out-of-band, then re-run prepare."
    done
}

# True iff the staged DiskANN family is already present for this dest prefix.
canonical_family_present() {
    local p="$1" s
    for s in "${CANONICAL_REQUIRED_SUFFIXES[@]}"; do
        [[ -e "${p}${s}" ]] || return 1
    done
    return 0
}

# DEPRECATED under the eval_tempfiles contract: prepare now fastcopies the
# canonical via materialize_canonical_family (real files straight into
# eval_tempfiles) instead of symlinking into eval_cached. Retained only for
# any out-of-tree caller; not used by the in-tree scripts.
# Symlink the canonical family into index_dir as <dataset><suffix>.
stage_canonical_family() {
    local canon="$1" index_dir="$2" dataset="$3" s src linked=0
    safe_mutable_path "$index_dir" >/dev/null   # refuse outside eval roots
    mkdir -p "$index_dir"
    for s in "${CANONICAL_FAMILY_SUFFIXES[@]}"; do
        src="${canon}${s}"
        [[ -e "$src" ]] || continue
        src="$(readlink -f "$src")"
        chmod -w "$src" 2>/dev/null || true
        ln -sfn "$src" "${index_dir}/${dataset}${s}"
        linked=$((linked + 1))
    done
    (( linked > 0 )) || error "no canonical files matched ${canon}*"
}

bin_header() {
    python3 - "$1" <<'PYEOF'
import struct, sys
with open(sys.argv[1], 'rb') as f:
    raw = f.read(8)
if len(raw) != 8:
    raise SystemExit(f'invalid bin header: {sys.argv[1]}')
print(*struct.unpack('<ii', raw))
PYEOF
}

bin_count() {
    local npts dim
    read -r npts dim <<<"$(bin_header "$1")"
    echo "$npts"
}

canonical_base_points() {
    if [[ -n "${NPTS_BASE:-}" ]]; then
        printf '%s\n' "$NPTS_BASE"
        return 0
    fi

    local canon="${CANONICAL_INDEX_0P9B:-}"
    if [[ -n "$canon" ]]; then
        if [[ -e "${canon}_disk.index.tags" ]]; then
            bin_count "${canon}_disk.index.tags"
            return 0
        fi
        if [[ -e "${canon}_pq_compressed.bin" ]]; then
            bin_count "${canon}_pq_compressed.bin"
            return 0
        fi
    fi

    : "${DATA_BIN:?DATA_BIN unset - dataset config not loaded?}"
    python3 - "$DATA_BIN" <<'PYEOF'
import struct, sys

with open(sys.argv[1], "rb") as f:
    raw = f.read(8)
if len(raw) != 8:
    raise SystemExit(f"invalid bin header: {sys.argv[1]}")
npts, _dim = struct.unpack("<ii", raw)
print((npts * 9) // 10)
PYEOF
}

write_bin_prefix() {
    local src dst count
    src="$1"
    dst="$2"
    count="$3"
    mkdir -p "$(dirname "$dst")"
    python3 "$BENCHMARK_DIR/bin_slice.py" "$src" "$dst" 0 "$count" "${DATA_TYPE:-float}"
}

write_bin_range() {
    local src dst start count
    src="$1"
    dst="$2"
    start="$3"
    count="$4"
    mkdir -p "$(dirname "$dst")"
    python3 "$BENCHMARK_DIR/bin_slice.py" "$src" "$dst" "$start" "$count" "${DATA_TYPE:-float}"
}

write_ids_range() {
    local dst start count
    dst="$1"
    start="$2"
    count="$3"
    mkdir -p "$(dirname "$dst")"
    python3 - "$dst" "$start" "$count" <<'PYEOF'
import array, sys
path, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids = array.array('I', range(start, start + count))
with open(path, 'wb') as f:
    ids.tofile(f)
PYEOF
}

write_counted_ids_range() {
    local dst start count
    dst="$1"
    start="$2"
    count="$3"
    mkdir -p "$(dirname "$dst")"
    python3 - "$dst" "$start" "$count" <<'PYEOF'
import array, struct, sys
path, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids = array.array('I', range(start, start + count))
with open(path, 'wb') as f:
    f.write(struct.pack('<i', count))
    ids.tofile(f)
PYEOF
}

ensure_identity_tags() {
    local npts prefix tag_file existing_npts tmp
    npts="$1"
    prefix="$2"
    tag_file="${prefix}_disk.index.tags"
    [[ "$npts" =~ ^[0-9]+$ ]] && (( npts > 0 )) || \
        error "ensure_identity_tags: npts must be a positive integer (got: $npts)"
    if [[ -f "$tag_file" ]]; then
        existing_npts="$(bin_count "$tag_file" || echo 0)"
        if [[ "$existing_npts" == "$npts" ]]; then
            return 0
        fi
        note "Replacing tag file with wrong point count ($existing_npts != $npts): $tag_file"
    fi
    mkdir -p "$(dirname "$tag_file")"
    tmp="${tag_file}.tmp.$$"
    if ! python3 - "$tmp" "$npts" <<'PYEOF'
import array, struct, sys
from pathlib import Path
path = Path(sys.argv[1])
npts = int(sys.argv[2])
with path.open('wb') as out:
    out.write(struct.pack('<ii', npts, 1))
    chunk = 1_000_000
    for start in range(0, npts, chunk):
        end = min(start + chunk, npts)
        vals = array.array('I', range(start, end))
        vals.tofile(out)
PYEOF
    then
        rm -f "$tmp"
        return 1
    fi
    mv -f "$tmp" "$tag_file"
}

require_index_tags() {
    local prefix expected_npts tag_file actual_npts
    prefix="$1"
    expected_npts="$2"
    tag_file="${prefix}_disk.index.tags"
    [[ -f "$tag_file" ]] || error "missing index tags: $tag_file"
    actual_npts="$(bin_count "$tag_file")"
    [[ "$actual_npts" == "$expected_npts" ]] || \
        error "index tag count mismatch: $tag_file has $actual_npts, expected $expected_npts"
}

link_dataset_files() {
    local input_dir
    input_dir="$1"
    mkdir -p "$input_dir/data" "$input_dir/groundtruth" "$input_dir/ids"
    ln -sfn "$DATA_BIN" "$input_dir/data/full.bin"
    ln -sfn "$QUERY_BIN" "$input_dir/data/query.bin"
}

compute_split() {
    local split
    UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
    TOTAL_POINTS="$(bin_count "$DATA_BIN")"
    BASE_POINTS="$(canonical_base_points)"
    split="$(python3 - "$TOTAL_POINTS" "$BASE_POINTS" "$UPDATE_PERCENT" <<'PYEOF'
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import sys

total = int(sys.argv[1])
base = int(sys.argv[2])
raw_percent = sys.argv[3]
try:
    percent = Decimal(raw_percent)
except InvalidOperation:
    raise SystemExit(f'invalid UPDATE_PERCENT: {raw_percent}')
if percent <= 0 or percent >= 100:
    raise SystemExit(f'UPDATE_PERCENT must be > 0 and < 100, got {raw_percent}')
update = int((Decimal(total) * percent / Decimal(100)).to_integral_value(rounding=ROUND_FLOOR))
if update <= 0:
    raise SystemExit(f'UPDATE_PERCENT {raw_percent} selects zero points from {total}')
if base <= 0 or base >= total:
    raise SystemExit(f'canonical base size must be > 0 and < total, got base={base} total={total}')
print(base, update)
PYEOF
)" || error "failed to compute dataset split for UPDATE_PERCENT=$UPDATE_PERCENT"
    read -r BASE_POINTS UPDATE_POINTS <<<"$split"
}

compute_update_layout() {
    # BASE_POINTS is pinned to the 0.9B canonical, independent of workload
    # size. INSERT_POINTS / DELETE_POINTS, when set, override only how many
    # rows the workload inserts / deletes. Default (neither override set):
    # INSERT = DELETE = TOTAL - BASE, which preserves prior behavior.
    INSERT_POINTS="${INSERT_POINTS:-}"
    DELETE_POINTS="${DELETE_POINTS:-}"
    UPDATE_PERCENT="${UPDATE_PERCENT:-10}"
    TOTAL_POINTS="$(bin_count "$DATA_BIN")"
    BASE_POINTS="$(canonical_base_points)"

    local layout
    layout="$(python3 - "$TOTAL_POINTS" "$BASE_POINTS" "${INSERT_POINTS:-}" "${DELETE_POINTS:-}" "$UPDATE_PERCENT" "${INSERT_CAP:-}" "${DELETE_CAP:-}" <<'PYEOF'
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import sys

total = int(sys.argv[1])
base_points = int(sys.argv[2])
insert_raw = sys.argv[3].strip()
delete_raw = sys.argv[4].strip()
percent_raw = sys.argv[5].strip()
insert_cap_raw = sys.argv[6].strip()
delete_cap_raw = sys.argv[7].strip()

def parse_opt_int(raw, name):
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw}")
    if value <= 0:
        raise SystemExit(f"{name} must be > 0, got {raw}")
    return value

try:
    percent = Decimal(percent_raw)
except InvalidOperation:
    raise SystemExit(f"invalid UPDATE_PERCENT: {percent_raw}")
if percent <= 0 or percent >= 100:
    raise SystemExit(f"UPDATE_PERCENT must be > 0 and < 100, got {percent_raw}")

default_workload = int((Decimal(total) * percent / Decimal(100)).to_integral_value(rounding=ROUND_FLOOR))
if default_workload <= 0:
    raise SystemExit(f"UPDATE_PERCENT {percent_raw} selects zero points from {total}")
if base_points <= 0 or base_points >= total:
    raise SystemExit(f"canonical base size must be > 0 and < total, got base={base_points} total={total}")
tail_points = total - base_points  # rows available to insert: [base, total)

insert_points = parse_opt_int(insert_raw, "INSERT_POINTS")
delete_points = parse_opt_int(delete_raw, "DELETE_POINTS")
if insert_points is None:
    insert_points = default_workload
if delete_points is None:
    delete_points = default_workload

if insert_points > tail_points:
    raise SystemExit(f"INSERT_POINTS ({insert_points}) exceeds available tail rows ({tail_points})")
if delete_points >= base_points:
    raise SystemExit(f"DELETE_POINTS must be < BASE_POINTS ({base_points}), got {delete_points}")

final_points = base_points - delete_points + insert_points

insert_cap = parse_opt_int(insert_cap_raw, "INSERT_CAP") if insert_cap_raw else insert_points
delete_cap = parse_opt_int(delete_cap_raw, "DELETE_CAP") if delete_cap_raw else delete_points

if insert_points % insert_cap != 0:
    raise SystemExit(f"INSERT_POINTS ({insert_points}) must be divisible by INSERT_CAP ({insert_cap})")
if delete_points % delete_cap != 0:
    raise SystemExit(f"DELETE_POINTS ({delete_points}) must be divisible by DELETE_CAP ({delete_cap})")

insert_rounds = insert_points // insert_cap
delete_rounds = delete_points // delete_cap
if insert_rounds != delete_rounds:
    raise SystemExit(
        f"mixed update requires equal round count, got insert rounds {insert_rounds} and delete rounds {delete_rounds}"
    )

print(base_points, insert_points, delete_points, final_points, insert_cap, delete_cap, insert_rounds)
PYEOF
)" || error "failed to compute update layout"

    read -r BASE_POINTS INSERT_POINTS DELETE_POINTS FINAL_POINTS INSERT_CAP DELETE_CAP UPDATE_ROUNDS <<<"$layout"
    UPDATE_POINTS="$INSERT_POINTS"

    # Concurrent query load for the mixed-update workload. QUERY_POINTS is a
    # TOTAL across the whole workload (mirrors INSERT_POINTS / DELETE_POINTS).
    # Round-based drivers (DiskANN/Greator/OdinANN) consume a per-round batch =
    # QUERY_POINTS / UPDATE_ROUNDS (nearest int, >=1); single-shot drivers
    # (BufANN) use QUERY_POINTS directly. Legacy: when only UPDATE_QUERY_BATCH
    # is set it is interpreted as the per-round batch.
    if [[ -z "${QUERY_POINTS:-}" && -n "${UPDATE_QUERY_BATCH:-}" ]]; then
        QUERY_POINTS=$(( UPDATE_QUERY_BATCH * UPDATE_ROUNDS ))
    fi
    if [[ -n "${QUERY_POINTS:-}" ]]; then
        if ! [[ "$QUERY_POINTS" =~ ^[0-9]+$ ]]; then
            error "QUERY_POINTS must be a non-negative integer (got: $QUERY_POINTS)"
        fi
        QUERY_BATCH_PER_ROUND=$(( (QUERY_POINTS + UPDATE_ROUNDS / 2) / UPDATE_ROUNDS ))
        if (( QUERY_POINTS > 0 && QUERY_BATCH_PER_ROUND < 1 )); then
            QUERY_BATCH_PER_ROUND=1
        fi
        QUERY_POINTS_EFFECTIVE=$(( QUERY_BATCH_PER_ROUND * UPDATE_ROUNDS ))
    fi
}

prepend_path() {
    local var_name path current
    var_name="$1"
    path="$2"
    [[ -d "$path" ]] || return 0
    current="${!var_name:-}"
    case ":$current:" in
        *":$path:"*) ;;
        *) export "$var_name=$path${current:+:$current}" ;;
    esac
}

setup_build_env() {
    prepend_path CPATH "$BENCH_DEPS_PREFIX/include"
    prepend_path CPATH "$BENCH_DEPS_PREFIX/include/mkl"
    prepend_path CPLUS_INCLUDE_PATH "$BENCH_DEPS_PREFIX/include"
    prepend_path CPLUS_INCLUDE_PATH "$BENCH_DEPS_PREFIX/include/mkl"
    prepend_path CMAKE_INCLUDE_PATH "$BENCH_DEPS_PREFIX/include"
    prepend_path CMAKE_INCLUDE_PATH "$BENCH_DEPS_PREFIX/include/mkl"
    prepend_path CMAKE_PREFIX_PATH "$BENCH_DEPS_PREFIX"
    prepend_path CMAKE_PREFIX_PATH "$BENCH_DEPS_PREFIX/lib/cmake"
    prepend_path LIBRARY_PATH "$BENCH_DEPS_PREFIX/lib"
    prepend_path LD_LIBRARY_PATH "$BENCH_DEPS_PREFIX/lib"
    prepend_path CMAKE_LIBRARY_PATH "$BENCH_DEPS_PREFIX/lib"
    prepend_path PKG_CONFIG_PATH "$BENCH_DEPS_PREFIX/lib/pkgconfig"
    prepend_path CPATH "/usr/include/mkl"
    prepend_path CPLUS_INCLUDE_PATH "/usr/include/mkl"
    prepend_path CMAKE_INCLUDE_PATH "/usr/include/mkl"
    prepend_path LIBRARY_PATH "/usr/lib/x86_64-linux-gnu"
    prepend_path CMAKE_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu"
}

choose_compiler() {
    if [[ -z "${CXX:-}" ]]; then
        if [[ -x /lusr/bin/g++-15 && -x /lusr/bin/gcc-15 ]]; then
            export CXX=/lusr/bin/g++-15 CC=/lusr/bin/gcc-15
        elif [[ -x /lusr/bin/g++-14 && -x /lusr/bin/gcc-14 ]]; then
            export CXX=/lusr/bin/g++-14 CC=/lusr/bin/gcc-14
        elif [[ -x /lusr/bin/g++-11 && -x /lusr/bin/gcc-11 ]]; then
            export CXX=/lusr/bin/g++-11 CC=/lusr/bin/gcc-11
        fi
    fi
}

configure_cmake_repo() {
    local repo target selected_cc selected_cxx compiler_libstdcpp compiler_lib_dir cache_file build_type config_types
    repo="$1"
    shift
    choose_compiler
    setup_build_env
    selected_cc="${CC:-$(command -v gcc)}"
    selected_cxx="${CXX:-$(command -v g++)}"
    compiler_libstdcpp="$($selected_cxx -print-file-name=libstdc++.so)"
    if [[ -n "$compiler_libstdcpp" && "$compiler_libstdcpp" != "libstdc++.so" && -e "$compiler_libstdcpp" ]]; then
        compiler_lib_dir="$(dirname "$(readlink -f "$compiler_libstdcpp")")"
        export LD_LIBRARY_PATH="$compiler_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    cmake -S "$repo" -B "$repo/build" \
        -DCMAKE_BUILD_TYPE:STRING="$BENCHMARK_CMAKE_BUILD_TYPE" \
        -DCMAKE_C_COMPILER="$selected_cc" \
        -DCMAKE_CXX_COMPILER="$selected_cxx" \
        -DINTEL_ROOT="$BENCH_DEPS_PREFIX" \
        -DMKL_ROOT="$BENCH_DEPS_PREFIX"
    cache_file="$repo/build/CMakeCache.txt"
    if [[ -f "$cache_file" ]]; then
        build_type="$(sed -n 's/^CMAKE_BUILD_TYPE:[^=]*=//p' "$cache_file" | tail -1)"
        config_types="$(sed -n 's/^CMAKE_CONFIGURATION_TYPES:[^=]*=//p' "$cache_file" | tail -1)"
        if [[ -z "$config_types" && "$build_type" != "$BENCHMARK_CMAKE_BUILD_TYPE" ]]; then
            cmake -S "$repo" -B "$repo/build" \
                -DCMAKE_BUILD_TYPE:STRING="$BENCHMARK_CMAKE_BUILD_TYPE" \
                -DINTEL_ROOT="$BENCH_DEPS_PREFIX" \
                -DMKL_ROOT="$BENCH_DEPS_PREFIX"
            build_type="$(sed -n 's/^CMAKE_BUILD_TYPE:[^=]*=//p' "$cache_file" | tail -1)"
            [[ "$build_type" == "$BENCHMARK_CMAKE_BUILD_TYPE" ]] || error "failed to configure $BENCHMARK_CMAKE_BUILD_TYPE build for $repo; CMAKE_BUILD_TYPE=$build_type"
        fi
    fi
    for target in "$@"; do
        cmake --build "$repo/build" --config "$BENCHMARK_CMAKE_BUILD_TYPE" --target "$target" -j"$(nproc)"
    done
}


write_update_final_bin() {
    # Final-state bin = rows [0, base-delete) ++ rows [base, base+insert).
    # Streams via bin_slice.py (64MiB chunks); same bytes as the old one-shot read.
    local src dst base_points insert_points delete_points data_type
    src="$1"
    dst="$2"
    base_points="$3"
    insert_points="$4"
    delete_points="$5"
    data_type="${6:-${DATA_TYPE:-float}}"
    mkdir -p "$(dirname "$dst")"
    python3 "$BENCHMARK_DIR/bin_slice.py" update-final \
        "$src" "$dst" "$base_points" "$insert_points" "$delete_points" "$data_type"
}

write_update_final_tags() {
    # Tags for FINAL_BIN row order: [0, base-delete) ++ [base, base+insert).
    # Chunked uint32 writes (1M ids); same tag sequence as the old in-memory build.
    local dst base_points insert_points delete_points
    dst="$1"
    base_points="$2"
    insert_points="$3"
    delete_points="$4"
    mkdir -p "$(dirname "$dst")"
    python3 "$BENCHMARK_DIR/bin_slice.py" update-final-tags \
        "$dst" "$base_points" "$insert_points" "$delete_points"
}
