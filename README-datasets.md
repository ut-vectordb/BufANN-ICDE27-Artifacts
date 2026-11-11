# Building a reference index

The benchmark consumes a prebuilt **canonical index** for each dataset and converts it into each system's native format at run time. This document builds that canonical index from the raw base vectors. `sift1b` (BIGANN, `uint8` × 128, 1B vectors) is the running example; the other datasets differ only in data type, dimension, and file paths.

## What the canonical index is

A canonical index is a family of files that share one prefix:

```text
<prefix>_disk.index          the on-disk Vamana graph
<prefix>_pq_pivots.bin       the PQ codebook
<prefix>_pq_compressed.bin   the per-point PQ codes
```

plus the `medoids`, `centroids`, and `tags` files the build emits alongside them. The prefix encodes the build parameters, for example `sift0.9b_R64_L100_B32_M200_noPQ`.

Every workload runs against a single index built over the first 90% of the base vectors, the **0.9× index**. Query and delete run against it directly; insert and update add the held-out 10%. The `0.9b` in the prefix marks it.

## Parameters

All datasets use the same graph parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `R`       | 64    | maximum graph degree |
| `L`       | 100   | build beam width |
| `M`       | 200   | per-shard build RAM budget, in GB |
| `T`       | 96    | build threads |
| metric    | `l2`  | distance |
| PQ chunks | 32    | bytes per PQ code |

The graph is built without PQ (`SKIP_PQ`); the PQ codes are built separately in step 2. The graph covers the first `NPTS_OVERRIDE` vectors: 90% of the base, `900000000` for `sift1b`.

## Compile the build tools

The build tools are DiskANN targets. Compile them once through the benchmark build path, which wires in the Release and MKL flags:

```bash
export REPO_DIR=/path/to/vectordb-bench
source "$REPO_DIR/benchmark/utils.sh"
source "$REPO_DIR/benchmark/DiskANN/common.sh"
configure_diskann_repo "$BASELINE_REPO" "" "" \
  partition_only build_shard merge_shards_only finalize_index build_pq_standalone
```

The binaries land in `$REPO_DIR/baselines/DiskANN/build/tests/`.

## Step 1 — build the graph

The graph is built in shards so a single out-of-memory shard never costs the whole run. Set the paths once:

```bash
BIN=$REPO_DIR/baselines/DiskANN/build/tests
BASE=/path/to/sift1b/base.1B.u8bin
PREFIX=/path/to/indexes/sift1b/sift0.9b_R64_L100_B32_M200_noPQ
NPTS_OVERRIDE=900000000
```

Partition the first 90% of the base into shards. Each point is replicated to its two nearest centroids, and `M` sets how many shards the data is split into:

```bash
$BIN/partition_only uint8 "$BASE" "$PREFIX" 64 200 "$NPTS_OVERRIDE"
```

Build a graph for each shard:

```bash
NSHARDS=$(ls ${PREFIX}_mem.index_tempFiles_subshard-[0-9]*_ids_uint32.bin | wc -l)
for p in $(seq 0 $((NSHARDS - 1))); do
  $BIN/build_shard uint8 \
    ${PREFIX}_mem.index_tempFiles_subshard-${p}.bin \
    ${PREFIX}_mem.index_tempFiles_subshard-${p}_mem.index \
    64 100 96 l2
done
```

Merge the per-shard graphs into one, then finalize it into the on-disk index:

```bash
$BIN/merge_shards_only \
  ${PREFIX}_mem.index_tempFiles_subshard- "$NSHARDS" 64 \
  ${PREFIX}_mem.index ${PREFIX}_disk.index_medoids.bin

$BIN/finalize_index uint8 ${PREFIX}_mem.index "$BASE" ${PREFIX}_disk.index "$NPTS_OVERRIDE"
```

The trailing `NPTS_OVERRIDE` argument to `partition_only` and `finalize_index` limits the graph to the first 90% of the base. This produces `${PREFIX}_disk.index` and its companion files.

## Step 2 — build the PQ codes

Build the PQ codebook and codes at the same prefix:

```bash
$BIN/build_pq_standalone uint8 "$BASE" "$PREFIX" 32
```

This writes `${PREFIX}_pq_pivots.bin` and `${PREFIX}_pq_compressed.bin`. PQ covers the full base, so the vectors added by the insert and update workloads already have codes.

## Step 3 — build the groundtruth

Each dataset has one exact groundtruth file, from which the benchmark derives every per-workload groundtruth. Build it once:

```bash
bash "$REPO_DIR/benchmark/gen_deep_gt.sh" sift1b
```

It writes to the path the dataset config declares (`GT_BIN_KCACHED`).

The ground-truth file stores each query's exact top 100 neighbors, although the benchmark scores only recall@10. Because inserts, deletes, and truncation change the active dataset, the benchmark filters this cached list for each workload and takes the first 10 valid neighbors. A top-10 cache could leave too few results after filtering, hence we use top-100.

## DGAI's index

DGAI reads its own index format, built over the same first 90% of the base with the same parameters (`R`, `L`, PQ32) using DGAI's build tools in `baselines/DGAI`. Each dataset's DGAI index lives in one directory:

```text
<prefix>_disk.index_header   the first 4 KB of DGAI's coupled index
<prefix>.layout/             the graph and the vectors, stored separately
PQ/                          DGAI's PQ codebooks and codes
```

`benchmark/datasets/<DATASET>.sh` names the directory (`DGAI_CANONICAL_DIR`), the prefix (`DGAI_CANONICAL_NAME`), and the layout options (`DGAI_STRATEGY`).

## Point the dataset config at the index

The dataset config `benchmark/datasets/sift1b.sh` names the canonical prefix it expects (`CANONICAL_INDEX_0P9B`, resolved under `SIFT1B_INDEX_DIR`) and the groundtruth path (`GT_BIN_KCACHED`). Build the files under those paths, or override the variables to point at where they live. Once the config resolves, `bash run_all.sh` runs.
