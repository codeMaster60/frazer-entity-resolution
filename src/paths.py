"""Shared paths and low-memory TSV reading helpers."""
from __future__ import annotations

import csv
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET = os.path.join(ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET, "train")
TEST_DIR = os.path.join(DATASET, "test")
WORK = os.path.join(ROOT, "work")          # intermediate parquet / models (gitignored)
OUTPUT = os.path.join(ROOT, "output")

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]

# Every TSV in this challenge must be read as raw text: no NaN coercion (empty
# addresses are meaningful), no quote handling (names contain stray quotes).
READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def source_tsv(split: str, source: int) -> str:
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return os.path.join(d, f"{split}_source{source}.tsv")


def ground_truth_tsv() -> str:
    return os.path.join(TRAIN_DIR, "train_ground_truth.tsv")


def parquet_path(split: str, source: int) -> str:
    return os.path.join(WORK, f"{split}_source{source}.parquet")


def read_tsv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **{**READ_KW, **kw})


def iter_tsv(path: str, chunksize: int = 500_000, **kw):
    return pd.read_csv(path, chunksize=chunksize, **{**READ_KW, **kw})


def read_source(split: str, source: int, columns=None) -> pd.DataFrame:
    """Read a source table, preferring the parquet copy when it exists."""
    pq = parquet_path(split, source)
    if os.path.exists(pq):
        return pd.read_parquet(pq, columns=columns)
    df = read_tsv(source_tsv(split, source))
    return df[columns] if columns else df


def ensure_dirs() -> None:
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)


def norm_path(split: str, source: int, country: str) -> str:
    """Normalized, country-sharded records written by ``src.prepare``."""
    safe = country.replace(" ", "_")
    return os.path.join(WORK, f"norm_{split}_s{source}_{safe}.parquet")


def cand_path(split: str, country: str, source: int) -> str:
    """Blocking candidates written by ``src.blocking``."""
    safe = country.replace(" ", "_")
    return os.path.join(WORK, f"cand_{split}_{safe}_s{source}.parquet")


def countries(split: str) -> list[str]:
    """Country labels present in a split's Source 1, most frequent first."""
    c = pd.read_parquet(parquet_path(split, 1), columns=["country"])["country"]
    return list(c.value_counts().index)
