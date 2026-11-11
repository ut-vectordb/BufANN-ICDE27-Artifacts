#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- run the five systems across the workloads and sweeps in the paper.
#
#   Systems:   BufANN (ours), DiskANN, OdinANN, Greator, DGAI
#   Workloads: query, insert, delete, update (mixed insert+delete)
#   Datasets:  sift1b (default) | sift | sift100m | spacev | turing | deep
#
# Every cell is two steps:
#
#     benchmark/<system>/sift/<workload>_prepare.sh   # stage the index + groundtruth
#     benchmark/<system>/sift/<workload>.sh           # run it, write result.json
#
# Prerequisites (a prebuilt canonical index + groundtruth per dataset) are in
# README.md and README-datasets.md. Results are copied to results/<dataset>/<phase>/.
#
# Usage:
#     bash run_all.sh                      # core matrix: 5 systems x 4 workloads
#     DATASET=spacev bash run_all.sh       # same matrix on another dataset
#     bash run_all.sh mixed threshold      # run specific phases instead
#     SYSTEMS="BufANN DiskANN" bash run_all.sh
#
# Phases:
#     query insert delete update   the core matrix (default)
#     mixed                        queries served while a merge runs
#     threshold                    sweep the merge threshold (per-round batch size)
#     memory                       sweep BufANN's buffer-pool budget (BufANN only)
#     threads                      sweep the thread count
#     snapshot                     BufANN deletion with/without caching deleted nodes' neighbor lists
#     repair                       BufANN mixed workload, varying the repair workers (collaborative repairing)
#     calico                       BufANN queries with Calico's buffer pool, and without PQ codes
# Named phases run only those; with no arguments the core matrix runs.
# =============================================================================
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# -----------------------------------------------------------------------------
# Configuration (all env-overridable).
# -----------------------------------------------------------------------------
DATASET="${DATASET:-sift1b}"
SYSTEMS="${SYSTEMS:-DiskANN OdinANN Greator DGAI BufANN}"

# Fixed graph/search parameters (paper "Systems configuration").
export CONSTRUCTION_L="${CONSTRUCTION_L:-100}"
export ALPHA="${ALPHA:-1.2}"
export BEAMWIDTH="${BEAMWIDTH:-4}"
export RECALL_AT="${RECALL_AT:-10}"
export NTHREADS="${NTHREADS:-96}"
export QUERY_THREADS="${QUERY_THREADS:-96}"
export INSERT_THREADS="${INSERT_THREADS:-96}"
export DELETE_THREADS="${DELETE_THREADS:-96}"

# Per-dataset scaling. SCALE is the workload unit (one round); PAGES_BASE is the
# page count (bytes/4096) of the shared 0.9x base canonical index that every
# workload consumes -- used to size BufANN's buffer pool.
case "$DATASET" in
  sift1b) SCALE=1000000  ; PAGES_BASE=90000000  ;;   # BIGANN 1B
  spacev) SCALE=1000000  ; PAGES_BASE=81818183  ;;
  turing) SCALE=1000000  ; PAGES_BASE=180000001 ;;
  deep)   SCALE=1000000  ; PAGES_BASE=150000001 ;;
  sift100m) SCALE=100000 ; PAGES_BASE=9000000  ;;   # 100M trial
  sift)   SCALE=1000      ; PAGES_BASE=150001    ;;   # 1M trial
  *) echo "run_all: unknown DATASET '$DATASET' (add a scaling block)" >&2; exit 1 ;;
esac

# Workload sizes (one round each, derived from SCALE).
SWEEP_ROUND="${SWEEP_ROUND:-$(( 1 * SCALE ))}"          # insert/delete: 1 round
UPDATE_ROUND="${UPDATE_ROUND:-$(( SCALE / 2 ))}"         # update: 0.5 in + 0.5 del

# Search-L grid for the query QPS/recall + latency/recall curve (RQ1a).
QUERY_L_LIST="${QUERY_L_LIST:-10 50 100 200 300 400 500 1000}"
# Per-round recall / build search L for the mutating workloads.
export RECALL_SEARCH_L="${RECALL_SEARCH_L:-100}"
export INSERT_SEARCH_L="${INSERT_SEARCH_L:-100}"
export DELETE_SEARCH_L="${DELETE_SEARCH_L:-100}"
export NODES_TO_CACHE="${NODES_TO_CACHE:-0}"

# BufANN buffer pool = BUFANN_BUDGET_PCT% of the base index pages (paper: 1%).
BUFANN_BUDGET_PCT="${BUFANN_BUDGET_PCT:-1}"
BUFANN_FRAMES_BASE=$(( PAGES_BASE * BUFANN_BUDGET_PCT / 100 ))
export BUFANN_PRELOAD_THREADS="${BUFANN_PRELOAD_THREADS:-48}"
export BUFANN_WARMUP_N="${BUFANN_WARMUP_N:-100000}"
# BufANN's deletion algorithm: lazy (collaborative repairing, the default),
# freshdiskann, or freshdiskann-nosnapshot (no cache of deleted nodes' neighbor lists).
export DELETE_ALGORITHM="${DELETE_ALGORITHM:-lazy}"
# BufANN's page table, pinned on every build (CMake caches it): hash (BufANN's
# own) or array (Calico's array translation; the calico phase only).
export BUFANN_CMAKE_ARGS="${BUFANN_CMAKE_ARGS:--DBUFANN_PAGE_TABLE=hash}"

# Sweep grids.
THRESHOLDS="${THRESHOLDS:-10000 100000 1000000 10000000}"   # RQ3 merge thresholds
THRESHOLDS_Greator="${THRESHOLDS_Greator:-10000 100000 1000000}"  # Greator OOMs at 10M
MEMORY_BUDGETS_PCT="${MEMORY_BUDGETS_PCT:-1 5 10 20 30 40 50}"    # RQ5 BufANN budgets
THREAD_COUNTS="${THREAD_COUNTS:-1 24 48 96}"                       # thread sweep
REPAIR_WORKERS="${REPAIR_WORKERS:-6 12 24}"                        # collaborative repairing ablation
CALICO_POOLS_PCT="${CALICO_POOLS_PCT:-1 10}"                       # Calico ablation buffer pools

EVAL_TEMPFILES="${EVAL_TEMPFILES:-$REPO_DIR/eval_tempfiles}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results/$DATASET}"

# -----------------------------------------------------------------------------
# Helpers.
# -----------------------------------------------------------------------------
banner() { echo; echo "=== $* ==="; }

# Wipe the per-dataset scratch tree so the next cell starts from a clean base.
reset_scratch() { rm -rf "${EVAL_TEMPFILES:?}/$DATASET"/* 2>/dev/null || true; mkdir -p "$EVAL_TEMPFILES/$DATASET"; }

# Export BufANN's buffer-pool + preload frame budget in one place.
bufann_frames() { export BUFANN_BUFFER_POOL_FRAMES="$1" BUFANN_PRELOAD_PAGES="$1"; }

# Run one (system, workload) cell and harvest its result.json + run.log.
#   run_cell <system> <workload> <phase-label> [dst-suffix]
run_cell() {
  local sys="$1" workload="$2" phase="$3" suffix="${4:-$sys}"
  local work="$EVAL_TEMPFILES/$DATASET/$sys/$workload"
  local out="$RESULTS_ROOT/$phase"; mkdir -p "$out"
  [[ "$sys" == "BufANN" ]] && bufann_frames "${BUFANN_BUDGET_FRAMES:-$BUFANN_FRAMES_BASE}"
  bash "benchmark/$sys/sift/${workload}_prepare.sh" || { echo "FAILED prepare: $sys/$workload"; return 1; }
  bash "benchmark/$sys/sift/${workload}.sh"         || { echo "FAILED run: $sys/$workload"; return 1; }
  cp "$work/result.json" "$out/${suffix}.result.json" 2>/dev/null || true
  cp "$work/run.log"     "$out/${suffix}.log"         2>/dev/null || true
}

FAIL=0

# -----------------------------------------------------------------------------
# RQ1a -- query-only QPS/Recall + Latency/Recall curve (sweeps the full L list
# in one run; binaries loop the L-vector internally).
# -----------------------------------------------------------------------------
phase_query() {
  export SEARCH_L="$QUERY_L_LIST"
  for sys in $SYSTEMS; do
    banner "[$DATASET/$sys] query  L={$SEARCH_L}"
    reset_scratch
    run_cell "$sys" query query || FAIL=1
  done
}

# -----------------------------------------------------------------------------
# RQ1b -- insert-only / delete-only / update, one round each. Per round:
# throughput + p50/p90/p95/p99 latency + recall.
# -----------------------------------------------------------------------------
phase_insert() {
  export INSERT_POINTS="$SWEEP_ROUND" INSERT_CAP="$SWEEP_ROUND"
  for sys in $SYSTEMS; do
    banner "[$DATASET/$sys] insert  total=$INSERT_POINTS"
    reset_scratch
    run_cell "$sys" insert insert || FAIL=1
  done
}

phase_delete() {
  export DELETE_POINTS="$SWEEP_ROUND" DELETE_CAP="$SWEEP_ROUND"
  for sys in $SYSTEMS; do
    banner "[$DATASET/$sys] delete  total=$DELETE_POINTS"
    reset_scratch
    run_cell "$sys" delete delete || FAIL=1
  done
}

phase_update() {
  export INSERT_POINTS="$UPDATE_ROUND" DELETE_POINTS="$UPDATE_ROUND"
  export INSERT_CAP="$UPDATE_ROUND"    DELETE_CAP="$UPDATE_ROUND"
  export SKIP_UPDATE_SEARCH="${SKIP_UPDATE_SEARCH:-1}"   # no query during merge
  for sys in $SYSTEMS; do
    banner "[$DATASET/$sys] update  insert=$INSERT_POINTS delete=$DELETE_POINTS"
    reset_scratch
    run_cell "$sys" update update || FAIL=1
  done
}

# -----------------------------------------------------------------------------
# RQ2 -- mixed workload: queries served while a merge runs. MERGE_THREADS of
# the 96 threads repair the graph; the rest serve queries until MIXED_BUDGET
# queries have run. BufANN runs with both deletion algorithms.
# -----------------------------------------------------------------------------
mixed_env() {
  local total="${MIXED_TOTAL:-$(( SCALE / 2 ))}" pool_threads="${MIXED_POOL_THREADS:-96}"
  export INSERT_POINTS="$total" DELETE_POINTS="$total"
  export INSERT_CAP="$total"    DELETE_CAP="$total"
  export NTHREADS="$pool_threads" QUERY_THREADS="$pool_threads"
  export MERGE_THREADS="$1" MERGE_QUERY_THREADS="$(( pool_threads - $1 ))"
  export QUERY_RATIO="${QUERY_RATIO:-90}" UPDATE_POOL_MODE="${UPDATE_POOL_MODE:-round_robin}"
  export QUERY_WINDOW_INTERVAL_MS="${QUERY_WINDOW_INTERVAL_MS:-10000}"
  export SEARCH_L="${MIXED_L:-100}" RECALL_SEARCH_L="${MIXED_L:-100}"
  export SKIP_UPDATE_SEARCH=0 ENABLE_INTERVAL_METRICS=1
}

phase_mixed() {
  mixed_env "${MERGE_THREADS_MIXED:-12}"
  export MERGE_QUERY_BUDGET="${MIXED_BUDGET:-80000000}"
  for sys in $SYSTEMS; do
    if [[ "$sys" == "BufANN" ]]; then
      local alg
      for alg in lazy freshdiskann; do
        banner "[$DATASET/BufANN-$alg] mixed  in=$INSERT_POINTS del=$DELETE_POINTS merge_threads=$MERGE_THREADS"
        reset_scratch
        DELETE_ALGORITHM="$alg" run_cell BufANN update mixed "BufANN_$alg" || FAIL=1
      done
    else
      banner "[$DATASET/$sys] mixed  in=$INSERT_POINTS del=$DELETE_POINTS merge_threads=$MERGE_THREADS"
      reset_scratch
      run_cell "$sys" update mixed || FAIL=1
    fi
  done
}

# -----------------------------------------------------------------------------
# RQ3 -- merge-threshold sweep. Threshold == per-round cap. One round each of
# insert / delete / update at every threshold.
# -----------------------------------------------------------------------------
phase_threshold() {
  export SEARCH_L="${THRESHOLD_L:-100}" RECALL_SEARCH_L="${THRESHOLD_L:-100}"
  for sys in $SYSTEMS; do
    local grid="$THRESHOLDS"
    [[ "$sys" == "Greator" ]] && grid="$THRESHOLDS_Greator"
    for T in $grid; do
      banner "[$DATASET/$sys] threshold=$T  insert"
      reset_scratch; export INSERT_POINTS="$T" INSERT_CAP="$T"
      run_cell "$sys" insert "threshold/insert" "${sys}_thr${T}" || FAIL=1

      banner "[$DATASET/$sys] threshold=$T  delete"
      reset_scratch; export DELETE_POINTS="$T" DELETE_CAP="$T"
      run_cell "$sys" delete "threshold/delete" "${sys}_thr${T}" || FAIL=1

      banner "[$DATASET/$sys] threshold=$T  update (in=$(( T/2 )) del=$(( T/2 )))"
      reset_scratch
      export INSERT_POINTS="$(( T/2 ))" DELETE_POINTS="$(( T/2 ))"
      export INSERT_CAP="$INSERT_POINTS" DELETE_CAP="$DELETE_POINTS"
      run_cell "$sys" update "threshold/update" "${sys}_thr${T}" || FAIL=1
    done
  done
}

# -----------------------------------------------------------------------------
# RQ5 -- BufANN performance vs memory budget (BufANN only). One round each of
# query / insert / delete / update, sweeping the buffer pool as a % of pages.
# -----------------------------------------------------------------------------
phase_memory() {
  export SEARCH_L="${MEMORY_L:-100}" RECALL_SEARCH_L="${MEMORY_L:-100}"
  local out="$RESULTS_ROOT/memory"; mkdir -p "$out"

  # query: prepare once, sweep budgets in place.
  banner "[$DATASET/BufANN] memory: prepare query once"
  reset_scratch; bufann_frames "$BUFANN_FRAMES_BASE"
  if bash "benchmark/BufANN/sift/query_prepare.sh"; then
    for pct in $MEMORY_BUDGETS_PCT; do
      banner "[$DATASET/BufANN] memory query budget=${pct}%"
      bufann_frames "$(( PAGES_BASE * pct / 100 ))"
      bash "benchmark/BufANN/sift/query.sh" || FAIL=1
      cp "$EVAL_TEMPFILES/$DATASET/BufANN/query/result.json" "$out/query_pct${pct}.result.json" 2>/dev/null || true
    done
  else echo "FAILED memory query prepare"; FAIL=1; fi

  # insert / delete / update: fresh index per budget.
  local wl
  for wl in insert delete update; do
    for pct in $MEMORY_BUDGETS_PCT; do
      banner "[$DATASET/BufANN] memory $wl budget=${pct}%"
      reset_scratch; bufann_frames "$(( PAGES_BASE * pct / 100 ))"
      case "$wl" in
        insert) export INSERT_POINTS="$SWEEP_ROUND" INSERT_CAP="$SWEEP_ROUND" ;;
        delete) export DELETE_POINTS="$SWEEP_ROUND" DELETE_CAP="$SWEEP_ROUND" ;;
        update) export INSERT_POINTS="$UPDATE_ROUND" DELETE_POINTS="$UPDATE_ROUND"
                export INSERT_CAP="$UPDATE_ROUND" DELETE_CAP="$UPDATE_ROUND" ;;
      esac
      bash "benchmark/BufANN/sift/${wl}_prepare.sh" || { echo "FAILED memory $wl prepare pct=$pct"; FAIL=1; continue; }
      bash "benchmark/BufANN/sift/${wl}.sh"         || { echo "FAILED memory $wl pct=$pct"; FAIL=1; continue; }
      cp "$EVAL_TEMPFILES/$DATASET/BufANN/$wl/result.json" "$out/${wl}_pct${pct}.result.json" 2>/dev/null || true
    done
  done
}

# -----------------------------------------------------------------------------
# Thread-count scaling sweep (all systems). Cores are pinned to half the thread
# count via taskset. One round each of query / insert / delete / update.
# -----------------------------------------------------------------------------
phase_threads() {
  export SEARCH_L="${THREADS_L:-100}"
  local out="$RESULTS_ROOT/threads"; mkdir -p "$out"
  run_tasked() { local t="$1"; shift; local c=$(( t/2 )); (( c<1 )) && c=1; taskset -c "0-$(( c-1 ))" "$@"; }
  for sys in $SYSTEMS; do
    # query: prepare once, sweep threads in place.
    banner "[$DATASET/$sys] threads query: prepare once"
    reset_scratch
    [[ "$sys" == "BufANN" ]] && bufann_frames "$BUFANN_FRAMES_BASE"
    if bash "benchmark/$sys/sift/query_prepare.sh"; then
      for t in $THREAD_COUNTS; do
        banner "[$DATASET/$sys] threads query t=$t"
        export QUERY_THREADS="$t" NTHREADS="$t"
        run_tasked "$t" bash "benchmark/$sys/sift/query.sh" || FAIL=1
        cp "$EVAL_TEMPFILES/$DATASET/$sys/query/result.json" "$out/query_${sys}_t${t}.result.json" 2>/dev/null || true
      done
    else echo "FAILED threads query prepare $sys"; FAIL=1; fi

    # insert / delete / update: fresh index per thread count.
    local wl
    for wl in insert delete update; do
      for t in $THREAD_COUNTS; do
        banner "[$DATASET/$sys] threads $wl t=$t"
        reset_scratch
        export NTHREADS="$t" QUERY_THREADS="$t" INSERT_THREADS="$t" DELETE_THREADS="$t"
        case "$wl" in
          insert) export INSERT_POINTS="$SWEEP_ROUND" INSERT_CAP="$SWEEP_ROUND" ;;
          delete) export DELETE_POINTS="$SWEEP_ROUND" DELETE_CAP="$SWEEP_ROUND" ;;
          update) export INSERT_POINTS="$UPDATE_ROUND" DELETE_POINTS="$UPDATE_ROUND"
                  export INSERT_CAP="$UPDATE_ROUND" DELETE_CAP="$UPDATE_ROUND" ;;
        esac
        [[ "$sys" == "BufANN" ]] && bufann_frames "$BUFANN_FRAMES_BASE"
        bash "benchmark/$sys/sift/${wl}_prepare.sh" || { echo "FAILED threads $wl prepare $sys t=$t"; FAIL=1; continue; }
        run_tasked "$t" bash "benchmark/$sys/sift/${wl}.sh" || { echo "FAILED threads $wl $sys t=$t"; FAIL=1; continue; }
        cp "$EVAL_TEMPFILES/$DATASET/$sys/$wl/result.json" "$out/${wl}_${sys}_t${t}.result.json" 2>/dev/null || true
      done
    done
  done
}

# -----------------------------------------------------------------------------
# Ablation -- caching deleted nodes' neighbor lists (BufANN only). Deletion at
# every merge threshold, with and without the cache.
# -----------------------------------------------------------------------------
phase_snapshot() {
  export SEARCH_L="${THRESHOLD_L:-100}" RECALL_SEARCH_L="${THRESHOLD_L:-100}"
  local T alg
  for T in $THRESHOLDS; do
    for alg in freshdiskann freshdiskann-nosnapshot; do
      banner "[$DATASET/BufANN-$alg] snapshot  delete threshold=$T"
      reset_scratch; export DELETE_POINTS="$T" DELETE_CAP="$T"
      DELETE_ALGORITHM="$alg" run_cell BufANN delete snapshot "delete_thr${T}_${alg}" || FAIL=1
    done
  done
}

# -----------------------------------------------------------------------------
# Ablation -- collaborative repairing (BufANN only). The mixed workload with
# REPAIR_WORKERS repair threads, lazy deletion against freshdiskann. Queries
# run through the merge and for 600 s after it.
# -----------------------------------------------------------------------------
phase_repair() {
  local x alg
  for x in $REPAIR_WORKERS; do
    mixed_env "$x"
    export MERGE_QUERY_BUDGET=0 POST_MERGE_QUERY_S="${POST_MERGE_QUERY_S:-600}"
    for alg in lazy freshdiskann; do
      banner "[$DATASET/BufANN-$alg] repair  workers=$x"
      reset_scratch
      DELETE_ALGORITHM="$alg" run_cell BufANN update repair "m${x}_${alg}" || FAIL=1
    done
  done
  unset POST_MERGE_QUERY_S
}

# -----------------------------------------------------------------------------
# Ablation -- buffer pool design and PQ codes (BufANN only). Queries at L=100
# with BufANN's buffer pool, with Calico's array page translation, and with
# BufANN's buffer pool but no PQ codes (every neighbor scored with its full
# vector through the buffer pool).
# -----------------------------------------------------------------------------
phase_calico() {
  export SEARCH_L="${CALICO_L:-100}"
  local pct variant table pq
  for pct in $CALICO_POOLS_PCT; do
    for variant in bufann calico nopq; do
      case "$variant" in
        bufann) table=hash  pq=1 ;;
        calico) table=array pq=1 ;;
        nopq)   table=hash  pq=0 ;;
      esac
      banner "[$DATASET/BufANN] calico  $variant pool=${pct}%"
      reset_scratch
      BUFANN_BUDGET_FRAMES="$(( PAGES_BASE * pct / 100 ))" \
      BUFANN_CMAKE_ARGS="-DBUFANN_PAGE_TABLE=$table" BUFANN_SEARCH_PQ="$pq" \
        run_cell BufANN query calico "${variant}_pool${pct}" || FAIL=1
    done
  done
}

# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
PHASES=("$@")
[[ ${#PHASES[@]} -eq 0 ]] && PHASES=(query insert delete update)   # default: core matrix

echo "run_all: DATASET=$DATASET SYSTEMS=[$SYSTEMS] phases=[${PHASES[*]}]"
echo "         L_build=$CONSTRUCTION_L alpha=$ALPHA beam=$BEAMWIDTH threads=$NTHREADS"
echo "         SCALE=$SCALE round=$SWEEP_ROUND update_round=$UPDATE_ROUND bufann_frames=$BUFANN_FRAMES_BASE (${BUFANN_BUDGET_PCT}%)"
mkdir -p "$RESULTS_ROOT"

for p in "${PHASES[@]}"; do
  case "$p" in
    query|insert|delete|update|mixed|threshold|memory|threads|snapshot|repair|calico) "phase_$p" ;;
    *) echo "run_all: unknown phase '$p'" >&2; FAIL=1 ;;
  esac
done

reset_scratch
banner "run_all done (DATASET=$DATASET, FAIL=$FAIL); results in $RESULTS_ROOT"
exit $FAIL
