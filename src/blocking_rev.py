"""Stage 2 (v2) — reverse candidate generation: noisy record -> clean S1.

The forward direction asks each clean Source-1 record to find its ~4 noisy copies
among 5M candidates. The reverse direction asks each noisy record to find the one
clean record it is a copy of, which is a far easier retrieval problem:

* the ground truth is 1-to-many from S1, so every S2/S3 record has **at most one**
  correct S1. A top-3 retrieval is therefore enough, where forward blocking had to
  keep ~40 to catch a whole match set.
* the index side is the clean, deduplicated source, so index tokens are not noisy.
* candidates per S1 fall out as (|S2|+|S3|)/|S1| * K — about 11 at K=2 and 17 at
  K=3 — instead of 80+, which is what the reduction-ratio audit rewards.
* every noisy record competes over the *whole* S1 table, so the assignment
  competition during validation is identical to the one at test time. Forward
  blocking on a sampled slice of S1 could not model that, which made its
  validation optimistic.

One index holds every kind of evidence as a *typed* token, so a single sparse
product combines address and name agreement with IDF weighting rather than
scoring each signal separately:

    a:<address token>   n:<name token>   s:<name consonant skeleton>

The skeleton is what makes Indic-script names reachable. `unidecode` renders
a Devanagari name as a long-vowel transliteration that shares no token with its English form,
but dropping vowels and collapsing doubled letters maps both to `rm mrktng`. The
same trick absorbs doubled-letter typos and injected accents.

    python -m src.blocking_rev --split train
    python -m src.blocking_rev --split test --country France
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .blocking import TokenIndex, _topk
from .paths import WORK, cand_path, countries, ensure_dirs, norm_path
from .sampling import sampled_mask

VOWELS = frozenset("aeiou")
N_PROCS = max(1, (os.cpu_count() or 2))
QUERY_CHUNK = 40_000      # queries per parallel task; small enough that every core fills


@dataclass(frozen=True)
class RevParams:
    maxdf: int = 60_000      # index side is S1, ~3x smaller than S2/S3
    r_query: int = 28        # rarest typed tokens used per noisy record
    k: int = 3               # S1 candidates kept per noisy record
    # Kept on, against expectation. Miss diagnosis showed 99% of misses are
    # "outranked" rather than unreachable, and the misses skew hard to
    # empty-address records, which suggested that dividing by sqrt(index-side
    # token count) was penalising the long, well-populated S1 records most likely
    # to be right. Measured, turning it off was *worse* (94.59% vs 95.09% at k=3),
    # so the hypothesis was wrong and the measurement wins.
    length_norm: bool = True
    # Records with no usable address have only their name to go on, so the right
    # S1 often sits just outside the top 3: miss diagnosis found empty-address
    # records were 28.5% of misses against a 2.6% base rate, and "outranked"
    # rather than unreachable. They get a much deeper list; everyone else keeps
    # k=3, so the cost to the average candidate count is about one per entity.
    k_short_addr: int = 18
    short_addr_tokens: int = 2
    max_rows: int = 20_000
    # Per-worker budget: each of N_PROCS workers holds its own score matrix, so
    # the cap is divided rather than shared.
    posting_budget: int = 6_000_000

    def label(self) -> str:
        return (f"maxdf={self.maxdf} r={self.r_query} k={self.k}"
                f"/{self.k_short_addr}short lennorm={int(self.length_norm)}")


def skeleton(tok: str) -> str:
    """Consonant skeleton: drop vowels, collapse runs of the same letter."""
    out: list[str] = []
    for ch in tok:
        if ch in VOWELS:
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


def signatures(name_core: np.ndarray, addr_tok: np.ndarray,
               rich: bool = False) -> np.ndarray:
    """Typed-token signature per record, order-unique.

    Token types, each carrying its own IDF weight in one shared index:

    ``a:`` address token, ``n:`` name token, ``s:`` name consonant skeleton,
    and, with ``rich``, ``p:`` 4-character address prefix and ``b:`` address
    consonant skeleton.

    ``rich`` defaults to **off because it measured worse**: 93.35% recall at k=3
    against 94.43% for the plain signature. The address prefix and skeleton tokens
    are individually sensible, but they crowd genuinely rare exact tokens out of
    the query's fixed token budget and add score noise, since a wrong S1 sharing a
    common prefix can outrank the right one. Kept behind a flag as a measured
    negative result rather than deleted.
    """
    out = np.empty(len(name_core), dtype=object)
    for i in range(len(name_core)):
        parts: dict[str, None] = {}
        a = addr_tok[i]
        if a:
            for t in a.split():
                parts["a:" + t] = None
                if rich and len(t) > 4:
                    parts["p:" + t[:4]] = None
                if rich:
                    sk = skeleton(t)
                    if len(sk) > 2 and sk != t:
                        parts["b:" + sk] = None
        n = name_core[i]
        if n:
            for t in n.split():
                parts["n:" + t] = None
                sk = skeleton(t)
                if len(sk) > 2 and sk != t:
                    parts["s:" + sk] = None
        out[i] = " ".join(parts)
    return out


def load_sig(split: str, source: int, country: str,
             rich: bool = False) -> np.ndarray:
    df = pd.read_parquet(norm_path(split, source, country),
                         columns=["name_core", "addr_tok"])
    return signatures(df.name_core.to_numpy(), df.addr_tok.to_numpy(), rich)


# Set in the parent before the pool forks; children inherit them copy-on-write,
# so the S1 index is built once rather than per worker. macOS defaults to "spawn",
# which would pickle the index to every worker, so the fork context is explicit.
_G: dict = {}


def _init_worker() -> None:
    # Each worker is single-threaded by design; the parallelism is across
    # workers, and letting each spawn BLAS threads only oversubscribes the CPU.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _run_chunk(bounds: tuple[int, int, int]):
    lo, hi, k = bounds
    index, params = _G["index"], _G["params"]
    q_sig, q_rows, keep = _G["q_sig"], _G["q_rows"], _G["keep"]
    s1_out, cand_out, score_out = [], [], []
    for offset, rows, cols, vals in index.query_topk(
            q_sig[lo:hi], k, params.r_query, params.max_rows,
            params.posting_budget, params.length_norm):
        q_idx = q_rows[lo + rows + offset]
        s1_idx = cols.astype(np.int32)
        if keep is not None:
            m = keep[s1_idx]
            q_idx, s1_idx, vals = q_idx[m], s1_idx[m], vals[m]
        s1_out.append(s1_idx)
        cand_out.append(q_idx.astype(np.int32))
        score_out.append(vals)
    if not s1_out:
        e32, ef = np.empty(0, np.int32), np.empty(0, np.float32)
        return e32, e32, ef
    return (np.concatenate(s1_out), np.concatenate(cand_out),
            np.concatenate(score_out))


def retrieve(split: str, country: str, source: int, params: RevParams,
             query_frac: float = 1.0, procs: int = N_PROCS,
             keep_s1: np.ndarray | None = None, rich: bool = False):
    """Top-k S1 candidates for each noisy record, across ``procs`` processes.

    Returns ``(s1_idx, cand_idx, score, n_s1, n_queried, index)``. The sparse
    products that dominate the runtime are single-threaded in scipy, so a single
    process uses one core however many the machine has; every caller goes through
    here so no tool is accidentally left serial.

    ``query_frac`` < 1 queries a deterministic stride of the noisy records. Pair
    recall over a random subset is an unbiased estimate of pair recall over all of
    them, so tuning runs in a fraction of the time.
    """
    s1_sig = load_sig(split, 1, country, rich)
    n_s1 = len(s1_sig)
    index = TokenIndex(s1_sig, params.maxdf)
    del s1_sig

    q_sig = load_sig(split, source, country, rich)
    q_rows = np.arange(len(q_sig), dtype=np.int32)
    if query_frac < 1.0:
        step = max(int(round(1.0 / query_frac)), 1)
        q_rows, q_sig = q_rows[::step], q_sig[::step]
    n_q = len(q_sig)

    # Split the queries by how much address they have, so each group can be
    # retrieved at its own depth. Grouping first keeps every chunk homogeneous,
    # which is what lets one k apply per chunk.
    addr_len = np.fromiter(
        (s.count("a:") for s in q_sig), dtype=np.int32, count=n_q)
    short = addr_len <= params.short_addr_tokens
    order = np.concatenate([np.flatnonzero(~short), np.flatnonzero(short)])
    n_wide = int((~short).sum())
    q_sig, q_rows = q_sig[order], q_rows[order]

    _G.clear()
    _G.update(index=index, params=params, q_sig=q_sig, q_rows=q_rows,
              keep=keep_s1)
    chunks = []
    for lo in range(0, n_wide, QUERY_CHUNK):
        chunks.append((lo, min(lo + QUERY_CHUNK, n_wide), params.k))
    for lo in range(n_wide, n_q, QUERY_CHUNK):
        chunks.append((lo, min(lo + QUERY_CHUNK, n_q), params.k_short_addr))
    if params.k_short_addr == params.k:
        chunks = [(a, b, params.k) for a, b, _ in chunks]
    parts = []
    if procs > 1 and len(chunks) > 1:
        ctx = mp.get_context("fork")   # inherit the index copy-on-write
        with ctx.Pool(procs, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_run_chunk, chunks, chunksize=1):
                parts.append(res)
    else:
        parts = [_run_chunk(c) for c in chunks]
    _G.clear()
    del q_sig

    s1_idx = np.concatenate([p[0] for p in parts])
    cand_idx = np.concatenate([p[1] for p in parts])
    score = np.concatenate([p[2] for p in parts]).astype(np.float32)
    parts.clear()
    return s1_idx, cand_idx, score, n_s1, n_q, index


def block_country(split: str, country: str, source: int, params: RevParams,
                  keep_sampled_s1: bool, query_frac: float = 1.0,
                  procs: int = N_PROCS):
    """Candidate pairs for one (country, noisy source), as a DataFrame."""
    t0 = time.time()
    keep = None
    if keep_sampled_s1:
        n_s1 = len(pd.read_parquet(norm_path(split, 1, country),
                                   columns=["entity_id"]))
        keep = sampled_mask(np.arange(n_s1))
    s1_idx, cand_idx, score, n_s1, n_q, index = retrieve(
        split, country, source, params, query_frac, procs, keep)
    del index

    out = pd.DataFrame({"s1_idx": s1_idx, "cand_idx": cand_idx,
                        "block_score": score, "pass_id": np.int8(1)})
    out = out.sort_values("block_score", ascending=False)
    out = out.drop_duplicates(["s1_idx", "cand_idx"], keep="first")
    out = out.sort_values("s1_idx", kind="stable", ignore_index=True)
    n_ent = int(keep.sum()) if keep is not None else n_s1
    print(f"    S{source}: {len(out):>10,} pairs from {n_q:,} records "
          f"({len(out) / n_ent:5.2f}/S1 entity)  {time.time() - t0:.0f}s "
          f"on {procs} procs", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--country", default=None)
    ap.add_argument("--k", type=int, default=RevParams.k)
    ap.add_argument("--maxdf", type=int, default=RevParams.maxdf)
    ap.add_argument("--r", type=int, default=RevParams.r_query)
    ap.add_argument("--all-entities", action="store_true")
    ap.add_argument("--query-frac", type=float, default=1.0,
                    help="query a stride of the noisy records (for fast tuning)")
    ap.add_argument("--procs", type=int, default=N_PROCS)
    ap.add_argument("--k-short", type=int, default=RevParams.k_short_addr)
    args = ap.parse_args()
    ensure_dirs()

    params = RevParams(maxdf=args.maxdf, r_query=args.r, k=args.k,
                       k_short_addr=args.k_short)
    keep_sampled = (args.split == "train") and not args.all_entities
    print(f"  reverse blocking {params.label()}"
          + ("  [keeping train-sample S1 only]" if keep_sampled else ""), flush=True)
    for country in ([args.country] if args.country else countries(args.split)):
        print(f"  {args.split.upper()} {country}", flush=True)
        for source in (2, 3):
            block_country(args.split, country, source, params, keep_sampled,
                          args.query_frac, args.procs).to_parquet(
                cand_path(args.split, country, source), compression="zstd",
                index=False)
    print(f"candidates -> {WORK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
