"""Stage 1 — normalize every source and shard it by country.

Writes ``work/norm_<split>_s<source>_<country>.parquet`` with one row per record
in a fixed order; later stages address records by their row index in these
files, so the files must not be regenerated between stages.

    python -m src.prepare --split train
    python -m src.prepare --split test
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from .normalize import addr_fields, name_fields
from .paths import WORK, ensure_dirs, norm_path, parquet_path

CHUNK = 400_000


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    names = [name_fields(s) for s in df.business_name]
    addrs = [addr_fields(s) for s in df.business_address]
    return pd.DataFrame({
        "entity_id": df.entity_id.to_numpy(),
        "name_norm": [n[0] for n in names],
        "name_core": [n[1] for n in names],
        "name_flags": pd.array([n[2] for n in names], dtype="int8"),
        "addr_tok": [a[0] for a in addrs],
        "addr_nums": [a[1] for a in addrs],
        "postal": [a[2] for a in addrs],
    })


def prepare(split: str) -> None:
    for source in (1, 2, 3):
        t0 = time.time()
        raw = pd.read_parquet(parquet_path(split, source))
        parts: dict[str, list[pd.DataFrame]] = {}
        for start in range(0, len(raw), CHUNK):
            chunk = raw.iloc[start:start + CHUNK]
            for country, sub in chunk.groupby("country", sort=False):
                parts.setdefault(country, []).append(normalize_frame(sub))
        del raw
        for country, frames in parts.items():
            out = pd.concat(frames, ignore_index=True)
            out.to_parquet(norm_path(split, source, country), compression="zstd",
                           index=False)
            print(f"  {split} S{source} {country:<7} {len(out):>10,} rows")
            del out, frames
        print(f"  ... S{source} done in {time.time() - t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    ensure_dirs()
    prepare(args.split)
    print(f"normalized files in {WORK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
