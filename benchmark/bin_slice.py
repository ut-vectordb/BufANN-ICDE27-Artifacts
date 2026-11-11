#!/usr/bin/env python3
"""DiskANN-style .bin helpers: contiguous slice + update final bin/tags.

All copies stream in CHUNK_BYTES blocks so billion-scale float bases fit in
modest RAM (tens of MB peak, not hundreds of GB).
"""

from __future__ import annotations

import argparse
import array
import os
import struct
import sys
from pathlib import Path

CHUNK_BYTES = 64 * 1024 * 1024
TAG_CHUNK = 1_000_000

ITEM_SIZES = {
    "float": 4,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
}


def normalized_dtype(name: str) -> str:
    lowered = name.lower()
    if lowered == "float":
        return "float"
    if lowered in ITEM_SIZES:
        return lowered
    raise SystemExit(f"ERROR: unsupported data_type: {name}")


def copy_exact_range(inp, out, offset: int, byte_count: int) -> None:
    """Seek to offset and copy byte_count bytes in CHUNK_BYTES blocks."""
    if byte_count < 0:
        raise SystemExit(f"ERROR: negative byte_count {byte_count}")
    if byte_count == 0:
        return
    inp.seek(offset)
    remaining = byte_count
    buf = bytearray(min(CHUNK_BYTES, remaining))
    view = memoryview(buf)
    while remaining:
        want = min(len(buf), remaining)
        got = inp.readinto(view[:want])
        if not got:
            copied = byte_count - remaining
            raise SystemExit(
                f"ERROR: short read: expected {byte_count} bytes, got {copied}"
            )
        out.write(view[:got])
        remaining -= got


def write_uint32_id_ranges(path: Path, ranges: list[tuple[int, int]]) -> int:
    """Write DiskANN tags bin: header (npts, 1) then uint32 ids from half-open ranges."""
    npts = 0
    for start, end in ranges:
        if start < 0 or end < start:
            raise SystemExit(f"ERROR: bad id range [{start}, {end})")
        npts += end - start

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as out:
            out.write(struct.pack("<ii", npts, 1))
            for start, end in ranges:
                for chunk_start in range(start, end, TAG_CHUNK):
                    chunk_end = min(chunk_start + TAG_CHUNK, end)
                    array.array("I", range(chunk_start, chunk_end)).tofile(out)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return npts


def write_update_final_bin(
    src: Path,
    dst: Path,
    base_points: int,
    insert_points: int,
    delete_points: int,
    data_type: str,
) -> None:
    """final.bin = rows [0, base-delete) ++ rows [base, base+insert)."""
    if min(base_points, insert_points, delete_points) < 0:
        raise SystemExit("ERROR: base/insert/delete must be non-negative")
    if delete_points > base_points:
        raise SystemExit(
            f"ERROR: delete_points {delete_points} > base_points {base_points}"
        )

    dtype = normalized_dtype(data_type)
    item_size = ITEM_SIZES[dtype]
    prefix_points = base_points - delete_points
    insert_end = base_points + insert_points
    out_npts = prefix_points + insert_points

    with src.open("rb") as inp:
        header = inp.read(8)
        if len(header) != 8:
            raise SystemExit(f"ERROR: invalid bin header: {src}")
        npts, dim = struct.unpack("<ii", header)
        if npts < 0 or dim <= 0:
            raise SystemExit(f"ERROR: invalid bin metadata in {src}: {npts} x {dim}")
        if insert_end > npts:
            raise SystemExit(
                f"ERROR: base+insert {insert_end} > dataset points {npts}"
            )

        row_bytes = dim * item_size
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        try:
            with tmp.open("wb") as out:
                out.write(struct.pack("<ii", out_npts, dim))
                # Surviving prefix [0, base-delete).
                copy_exact_range(inp, out, 8, prefix_points * row_bytes)
                # Inserted tail [base, base+insert).
                copy_exact_range(
                    inp, out, 8 + base_points * row_bytes, insert_points * row_bytes
                )
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink()

    print(
        f"Wrote {dst}: update_final prefix={prefix_points} "
        f"insert={insert_points} dim={dim} dtype={dtype}"
    )


def write_update_final_tags(
    dst: Path, base_points: int, insert_points: int, delete_points: int
) -> None:
    """Tags for FINAL_BIN row order: [0, base-delete) ++ [base, base+insert)."""
    if min(base_points, insert_points, delete_points) < 0:
        raise SystemExit("ERROR: base/insert/delete must be non-negative")
    if delete_points > base_points:
        raise SystemExit(
            f"ERROR: delete_points {delete_points} > base_points {base_points}"
        )

    prefix_end = base_points - delete_points
    insert_end = base_points + insert_points
    ranges: list[tuple[int, int]] = []
    if prefix_end > 0:
        ranges.append((0, prefix_end))
    if insert_points > 0:
        ranges.append((base_points, insert_end))

    npts = write_uint32_id_ranges(dst, ranges)
    print(f"Wrote {dst}: update_final_tags npts={npts}")


def parse_slice_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("start", type=int)
    parser.add_argument("count", type=int)
    parser.add_argument("data_type", help="float, int8, or uint8")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main_slice(argv: list[str] | None = None) -> int:
    args = parse_slice_args(argv)
    dtype = normalized_dtype(args.data_type)
    item_size = ITEM_SIZES[dtype]

    if args.start < 0 or args.count < 0:
        raise SystemExit("ERROR: start and count must be non-negative")
    if args.output.exists() and not args.force:
        print(f"Reusing existing bin slice: {args.output}")
        return 0

    with args.input.open("rb") as inp:
        header = inp.read(8)
        if len(header) != 8:
            raise SystemExit(f"ERROR: invalid bin header: {args.input}")
        npts, dim = struct.unpack("<ii", header)
        if npts < 0 or dim <= 0:
            raise SystemExit(
                f"ERROR: invalid bin metadata in {args.input}: {npts} x {dim}"
            )
        if args.start + args.count > npts:
            raise SystemExit(
                f"ERROR: requested range [{args.start}, {args.start + args.count}) "
                f"from {npts} vectors"
            )

        row_bytes = dim * item_size
        payload_bytes = args.count * row_bytes
        payload_offset = 8 + args.start * row_bytes

        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(args.output.name + ".tmp")
        try:
            with tmp.open("wb") as out:
                out.write(struct.pack("<ii", args.count, dim))
                copy_exact_range(inp, out, payload_offset, payload_bytes)
            os.replace(tmp, args.output)
        finally:
            if tmp.exists():
                tmp.unlink()

    print(f"Wrote {args.output}: start={args.start} count={args.count} dim={dim}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        parse_slice_args([])  # prints usage / exits
        return 2

    cmd = argv[0]
    if cmd == "update-final":
        if len(argv) != 7:
            raise SystemExit(
                "usage: bin_slice.py update-final <src> <dst> <base> <insert> "
                "<delete> <dtype>"
            )
        _, src, dst, base, insert, delete, dtype = argv
        write_update_final_bin(
            Path(src), Path(dst), int(base), int(insert), int(delete), dtype
        )
        return 0

    if cmd == "update-final-tags":
        if len(argv) != 5:
            raise SystemExit(
                "usage: bin_slice.py update-final-tags <dst> <base> <insert> <delete>"
            )
        _, dst, base, insert, delete = argv
        write_update_final_tags(Path(dst), int(base), int(insert), int(delete))
        return 0

    return main_slice(argv)


if __name__ == "__main__":
    raise SystemExit(main())
