"""Streaming profile of the challenge data.

Reads one column at a time from the parquet copies so peak RAM stays low.

    python -m src.profile_data
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from .paths import WORK, parquet_path

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
NON_ASCII = re.compile(r"[^\x00-\x7F]")
URLISH = re.compile(r"(?:\.com|\.in\b|\.org|\.net|\.co\.|www\.)", re.I)
IN_PIN = re.compile(r"\b\d{6}\b")
FIVE_DIGIT = re.compile(r"\b\d{5}\b")


def col(split: str, source: int, name: str) -> pd.Series:
    return pd.read_parquet(parquet_path(split, source), columns=[name])[name]


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "    n/a"


def profile_sources() -> None:
    for split in ("train", "test"):
        print(f"\n{'=' * 70}\n{split.upper()} SOURCES\n{'=' * 70}")
        for source in (1, 2, 3):
            country = col(split, source, "country")
            n = len(country)
            print(f"\nS{source}: {n:,} rows")
            for c, k in country.value_counts().items():
                print(f"   country {c!r:12} {k:>10,}  {pct(k, n)}")
            del country

            name = col(split, source, "business_name")
            addr = col(split, source, "business_address")
            stats = {
                "empty name": (name.str.len() == 0).sum(),
                "empty address": (addr.str.len() == 0).sum(),
                "name has devanagari": name.str.contains(DEVANAGARI, regex=True).sum(),
                "name has non-ascii": name.str.contains(NON_ASCII, regex=True).sum(),
                "name looks like a url": name.str.contains(URLISH, regex=True).sum(),
                "name starts '--'": name.str.startswith("--").sum(),
                "name is ALLCAPS": (name.str.upper() == name).sum(),
                "addr is ALLCAPS": (addr.str.upper() == addr).sum(),
                "addr has 5-digit code": addr.str.contains(FIVE_DIGIT, regex=True).sum(),
                "addr has 6-digit PIN": addr.str.contains(IN_PIN, regex=True).sum(),
            }
            for k, v in stats.items():
                print(f"   {k:<24} {int(v):>10,}  {pct(int(v), n)}")
            print(f"   {'distinct names':<24} {name.nunique():>10,}")
            print(f"   {'mean name len':<24} {name.str.len().mean():>10.1f}")
            print(f"   {'mean addr len':<24} {addr.str.len().mean():>10.1f}")
            del name, addr


def profile_ground_truth() -> None:
    print(f"\n{'=' * 70}\nTRAIN GROUND TRUTH\n{'=' * 70}")
    gt = pd.read_parquet(f"{WORK}/train_ground_truth.parquet")
    n = len(gt)
    print(f"rows: {n:,}   distinct S1: {gt.source1_entity_id.nunique():,}")

    ids = gt.matched_entity_ids
    empty = ids.str.len() == 0
    print(f"singletons (empty match list): {empty.sum():,}  {pct(int(empty.sum()), n)}")

    nonempty = ids[~empty]
    n_match = nonempty.str.count(",") + 1
    n_s2 = nonempty.str.count("S2-")
    n_s3 = nonempty.str.count("S3-")
    print(f"\nmatches per non-singleton entity ({len(nonempty):,} entities):")
    print(f"   mean {n_match.mean():.2f}  median {n_match.median():.0f} max {n_match.max()}")
    print("   size distribution:")
    for k, v in n_match.value_counts().sort_index().head(12).items():
        print(f"      {k:>3} matches {v:>10,}  {pct(v, n)}")
    print(f"   from S2: mean {n_s2.mean():.2f}  zero-S2 {pct(int((n_s2 == 0).sum()), len(nonempty))}")
    print(f"   from S3: mean {n_s3.mean():.2f}  zero-S3 {pct(int((n_s3 == 0).sum()), len(nonempty))}")

    exploded = nonempty.str.split(",").explode()
    total, distinct = len(exploded), exploded.nunique()
    print(f"\nmatched IDs: {total:,} total, {distinct:,} distinct "
          f"(reused by >1 S1: {total - distinct:,})")
    s2_total = len(col("train", 2, "entity_id"))
    s3_total = len(col("train", 3, "entity_id"))
    used_s2 = int(exploded.str.startswith("S2-").sum())
    used_s3 = total - used_s2
    print(f"   S2 records used {used_s2:,} / {s2_total:,}  {pct(used_s2, s2_total)}")
    print(f"   S3 records used {used_s3:,} / {s3_total:,}  {pct(used_s3, s3_total)}")
    print(f"\nbaseline: predicting empty for everything scores "
          f"macro F0.5 = {empty.mean():.4f}")


def profile_country_consistency(sample: int = 200_000) -> None:
    print(f"\n{'=' * 70}\nCOUNTRY CONSISTENCY (sample {sample:,} matched entities)\n{'=' * 70}")
    gt = pd.read_parquet(f"{WORK}/train_ground_truth.parquet")
    gt = gt[gt.matched_entity_ids.str.len() > 0]
    rng = np.random.default_rng(0)
    gt = gt.iloc[rng.choice(len(gt), size=min(sample, len(gt)), replace=False)]

    s1 = pd.read_parquet(parquet_path("train", 1), columns=["entity_id", "country"])
    s1map = dict(zip(s1.entity_id, s1.country))
    del s1
    pairs = gt.assign(mid=gt.matched_entity_ids.str.split(",")).explode("mid")
    cmap = {}
    for source in (2, 3):
        t = pd.read_parquet(parquet_path("train", source), columns=["entity_id", "country"])
        cmap.update(zip(t.entity_id, t.country))
        del t
    left = pairs.source1_entity_id.map(s1map)
    right = pairs.mid.map(cmap)
    same = left == right
    print(f"pairs checked: {len(pairs):,}")
    print(f"same country:  {int(same.sum()):,}  {pct(int(same.sum()), len(pairs))}")
    if (~same).any():
        print("cross-country combinations:")
        print(pd.DataFrame({"s1_country": left[~same], "other_country": right[~same]})
              .value_counts().head(10).to_string())


def main() -> int:
    pd.set_option("display.width", 200)
    profile_sources()
    profile_ground_truth()
    profile_country_consistency()
    return 0


if __name__ == "__main__":
    sys.exit(main())
