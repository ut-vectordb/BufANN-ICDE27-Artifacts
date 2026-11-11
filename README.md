# BufANN

This repository contains **BufANN** plus a self-contained benchmark suite that evaluates it against four baselines.

This repo contains:

1. **BufANN** (`src/`, `include/`, `tests/`) and the four baseline systems it is compared against, vendored under `baselines/`: DiskANN, OdinANN, Greator, and DGAI.
2. **`benchmark/`**, the scripts that build and run each system on a shared set of datasets and workloads, with a uniform input and output layout.
3. **`run_all.sh`**, which runs the five systems across the workloads and sweeps used in the paper.

## Systems evaluated

| System   | Role            | Source tree          |
|----------|-----------------|----------------------|
| **BufANN**  | ours         | `src/`, `include/`, `tests/` |
| DiskANN  | baseline        | `baselines/DiskANN`  |
| OdinANN  | baseline        | `baselines/OdinANN`  |
| Greator  | baseline        | `baselines/Greator`  |
| DGAI     | baseline        | `baselines/DGAI`     |

## Repository layout

```text
.
├── src/  include/  tests/   BufANN core library + drivers (bufann_driver, …)
├── baselines/
│   ├── DiskANN/  Greator/  OdinANN/  DGAI/   vendored baseline source trees.
│   └── src/                shared headers + converters used by the baselines
├── benchmark/              per-system build/run scripts + dataset configs
│   ├── <system>/sift/      per-(system, workload) prepare + run scripts
│   ├── datasets/           one config file per dataset (paths, dim, dtype)
│   └── utils.sh, ...        shared helpers
├── run_all.sh              run the paper's experiments
└── results/                collected result.json / run.log (gitignored)
```

## How the benchmark works

Each system exposes the same four workloads over a shared dataset config. Running one cell is two steps:

```bash
benchmark/<system>/sift/<workload>_prepare.sh   # stage the index + groundtruth
benchmark/<system>/sift/<workload>.sh           # run it, write result.json
```

- **Workloads:** `query` (query-only), `insert` (insert-only), `delete` (delete-only), and `update` (mixed insert+delete, optionally with queries served during the merge).
- **Datasets:** selected with `DATASET=` (default `sift1b`), resolving `benchmark/datasets/<DATASET>.sh`. Billion-scale sets are `sift1b`, `spacev`, `turing`, and `deep` (from the [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html) NeurIPS'21 track); `sift` (1M) and `sift100m` (100M) are smaller sets for trial runs.
- **Index building is kept out of the experiment path.** `prepare` takes a prebuilt canonical DiskANN-format index for the dataset and converts it into the target system's native format. See prerequisites below.

Scratch files go under `eval_tempfiles/`, and a reuse cache for slow-to-produce artifacts goes under `eval_cached/`. Both are gitignored and safe to delete.

## Prerequisites

System packages (Ubuntu; the union across all four systems):

```bash
sudo apt install cmake g++ make libaio-dev liburing-dev libgoogle-perftools-dev libboost-all-dev libmkl-full-dev
```

DiskANN, Greator, and OdinANN link Intel MKL and `tcmalloc`; OdinANN also uses `io_uring` (`liburing-dev`). If MKL or the other dependencies are not installed system-wide, point the build at a prefix that provides them:

```bash
export BENCH_DEPS_PREFIX=/path/to/deps   # provides include/, include/mkl, lib/
```

Datasets need to be supplied externally:

1. A canonical DiskANN-format index for the dataset (graph + PQ sidecar).
2. A DGAI index for the dataset, in DGAI's own format.
3. A groundtruth file for the query set.
4. The base and query vectors.

Their locations are env-overridable defaults at the top of each `benchmark/datasets/<DATASET>.sh`. Edit that one file, or export the corresponding variables, to point at where the dataset lives.

To build the index, PQ codes, and groundtruth from the base vectors, see [README-datasets.md](README-datasets.md).

## Build

Each system is built on demand before a cell runs (the scripts invoke `cmake` with the right flags), so you normally do not build anything by hand. To build BufANN's driver directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bufann_driver -j
```

## Running

Run the core matrix (5 systems × 4 workloads) on the default dataset:

```bash
bash run_all.sh
```

Select a dataset, a subset of systems, or specific phases:

```bash
DATASET=spacev bash run_all.sh
SYSTEMS="BufANN DiskANN" bash run_all.sh query insert
bash run_all.sh mixed threshold memory threads
```

With no arguments, `run_all.sh` runs the core matrix (`query insert delete update`). The phases map to the paper's experiments:

| Phase | Experiment |
|-------|------------|
| `query insert delete update` | performance under diverse workloads |
| `mixed` | runtime behavior under the mixed workload |
| `threshold` | impact of merge batch size |
| `threads` | scalability |
| `memory` | impact of memory budget |
| `snapshot repair calico` | ablation study |

The main knobs (`DATASET`, `SYSTEMS`, thread counts, sweep grids, and BufANN's buffer-pool budget, 1% of the graph's pages by default) are environment variables documented at the top of the script.

## Results

Each run writes a `result.json` (throughput, recall, p50/p90/p95/p99 latency, peak RSS, and IO counters) and a `run.log`. `run_all.sh` copies these into `results/<dataset>/<phase>/`.
