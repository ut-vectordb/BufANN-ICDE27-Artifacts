#!/usr/bin/env python3
"""Sample N random rows from a DiskANN-style base bin (header: i32 npts, i32 dim)
and emit a warmup-query file with the same header layout. Used to fill PageANN's
buffer pool without overlapping the timed query set.

Usage:
  sample_base_warmup.py <base.bin> <dtype> <n_samples> <out.bin> [--seed 0]

dtype: uint8 | int8 | float (matches DATA_TYPE in the benchmark scripts).
"""
import argparse, struct, sys
import numpy as np

DTYPE = {"uint8": np.uint8, "int8": np.int8, "float": np.float32}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("dtype", choices=DTYPE)
    ap.add_argument("n", type=int); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npts-limit", type=int, default=0,
                    help="cap the sampling pool to the first L rows (0=use all)")
    a = ap.parse_args()
    dt = DTYPE[a.dtype]; itemsize = np.dtype(dt).itemsize

    with open(a.base, "rb") as f:
        npts, dim = struct.unpack("<ii", f.read(8))
        row_bytes = dim * itemsize
        pool = npts if a.npts_limit <= 0 else min(npts, a.npts_limit)
        if a.n <= 0 or a.n > pool:
            sys.exit(f"bad n={a.n} for sampling pool={pool} (base npts={npts})")
        rng = np.random.default_rng(a.seed)
        # Sorted indices => one forward sweep over the file (sequential reads).
        ids = np.sort(rng.choice(pool, size=a.n, replace=False).astype(np.int64))
        buf = np.empty((a.n, dim), dtype=dt)
        for i, rid in enumerate(ids):
            f.seek(8 + rid * row_bytes)
            buf[i] = np.frombuffer(f.read(row_bytes), dtype=dt)

    with open(a.out, "wb") as f:
        f.write(struct.pack("<ii", a.n, dim))
        buf.tofile(f)
    print(f"wrote {a.out}: {a.n} x {dim} ({a.dtype}), seed={a.seed}, "
          f"from npts={npts}")

if __name__ == "__main__":
    main()
