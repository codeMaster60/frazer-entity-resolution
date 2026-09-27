"""Convert the large TSVs to Parquet once, streaming in chunks.

Parquet with dictionary-encoded strings is ~4x smaller and ~20x faster to load
than re-parsing a 500 MB TSV, and lets later stages read single columns.

    python -m src.to_parquet
    python -m src.to_parquet --split test     # just one split
"""
from __future__ import annotations

import argparse
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .paths import WORK, ensure_dirs, ground_truth_tsv, iter_tsv, parquet_path, source_tsv

CHUNK = 500_000


def convert(tsv: str, out: str, schema: pa.Schema) -> int:
    t0, rows = time.time(), 0
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    try:
        for chunk in iter_tsv(tsv, chunksize=CHUNK):
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            writer.write_table(table)
            rows += len(chunk)
            print(f"  {rows:>10,} rows", end="\r", flush=True)
    finally:
        writer.close()
    print(f"  {rows:>10,} rows in {time.time() - t0:.0f}s -> {out}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "test", "all"])
    args = ap.parse_args()
    ensure_dirs()
    src_schema = pa.schema(
        [("entity_id", pa.string()), ("business_name", pa.string()),
         ("business_address", pa.string()), ("country", pa.string())]
    )
    splits = ("train", "test") if args.split == "all" else (args.split,)
    for split in splits:
        for source in (1, 2, 3):
            print(f"{split} source{source}")
            convert(source_tsv(split, source), parquet_path(split, source), src_schema)

    if "train" not in splits:
        return 0

    print("train ground truth")
    gt_schema = pa.schema([("source1_entity_id", pa.string()),
                           ("matched_entity_ids", pa.string())])
    convert(ground_truth_tsv(), f"{WORK}/train_ground_truth.parquet", gt_schema)
    return 0


if __name__ == "__main__":
    sys.exit(main())
