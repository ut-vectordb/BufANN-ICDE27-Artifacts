#!/usr/bin/env python3
"""Convert little-endian .fvec/.fvecs files to DiskANN-style float .bin files."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input .fvec or .fvecs file")
    parser.add_argument("output", type=Path, help="output .bin file")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return parser.parse_args()


def read_dim(fh, path: Path) -> int:
    raw = fh.read(4)
    if len(raw) != 4:
        raise SystemExit(f"ERROR: failed to read vector dimension from {path}")
    (dim,) = struct.unpack("<i", raw)
    if dim <= 0:
        raise SystemExit(f"ERROR: invalid vector dimension {dim} in {path}")
    return dim


def main() -> int:
    args = parse_args()
    src = args.input
    dst = args.output

    if not src.is_file():
        raise SystemExit(f"ERROR: input fvec file does not exist: {src}")
    if dst.exists() and not args.force:
        print(f"Reusing existing bin file: {dst}")
        return 0

    file_size = src.stat().st_size
    if file_size < 4:
        raise SystemExit(f"ERROR: input fvec file is too small: {src}")

    with src.open("rb") as fh:
        dim = read_dim(fh, src)

    record_size = 4 + dim * 4
    if file_size % record_size != 0:
        raise SystemExit(
            f"ERROR: {src} size {file_size} is not divisible by fvec record size {record_size}"
        )
    npts = file_size // record_size
    if npts <= 0:
        raise SystemExit(f"ERROR: no vectors found in {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        with src.open("rb") as inp, tmp.open("wb") as out:
            out.write(struct.pack("<ii", npts, dim))
            for idx in range(npts):
                cur_dim = read_dim(inp, src)
                if cur_dim != dim:
                    raise SystemExit(
                        f"ERROR: inconsistent dim at vector {idx}: expected {dim}, found {cur_dim}"
                    )
                payload = inp.read(dim * 4)
                if len(payload) != dim * 4:
                    raise SystemExit(f"ERROR: truncated vector {idx} in {src}")
                out.write(payload)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"Converted {src} -> {dst} ({npts} x {dim}, float32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
