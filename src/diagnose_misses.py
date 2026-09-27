"""Why does reverse blocking miss the true pairs it misses?

For every sampled noisy record whose true S1 is not retrieved, compares the two
signatures and assigns one cause, so effort goes to the largest group rather than
to a guess:

* **no shared token** — the two signatures intersect nowhere. Only a new kind of
  token can reach these.
* **shared but unindexed** — they share tokens, but every shared token is above
  the document-frequency cap, so none is in the index. Raising the cap reaches
  these, at a cost in time and candidates.
* **outranked** — shared indexed tokens exist, so the pair was reachable and lost
  on score. Better scoring or a larger k reaches these.

Each miss is also tagged with the noise type of the record.

    python -m src.diagnose_misses --country India --source 2
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .blocking_rev import N_PROCS, RevParams, load_sig, retrieve
from .paths import norm_path
from .truth import id_positions, truth_pairs

INDIC = "ऀ-෿"


def noise_of(name: str, addr: str) -> str:
    import re
    if not addr:
        return "empty address"
    if re.search(f"[{INDIC}]", name):
        return "indic script"
    if re.search(r"\.com|\.in\b|\.net|\.org|www\.|@", name, re.I):
        return "url-like name"
    if name.isupper():
        return "allcaps"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--source", type=int, default=2)
    ap.add_argument("--query-frac", type=float, default=0.08)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--maxdf", type=int, default=RevParams.maxdf)
    ap.add_argument("--r", type=int, default=RevParams.r_query)
    ap.add_argument("--examples", type=int, default=30)
    ap.add_argument("--procs", type=int, default=N_PROCS)
    args = ap.parse_args()

    s1_pos = id_positions("train", 1, args.country)
    cand_pos = id_positions("train", args.source, args.country)
    n_s1, n_cand = len(s1_pos), len(cand_pos)
    t_s1, t_cd = truth_pairs(args.country, args.source, s1_pos, cand_pos)
    del s1_pos, cand_pos

    s1_raw = pd.read_parquet(norm_path("train", 1, args.country),
                             columns=["entity_id", "name_core", "addr_tok"])
    q_raw = pd.read_parquet(norm_path("train", args.source, args.country),
                            columns=["entity_id", "name_norm", "name_core", "addr_tok"])
    s1_sig = load_sig("train", 1, args.country)
    q_sig_all = load_sig("train", args.source, args.country)
    step = max(int(round(1.0 / args.query_frac)), 1)
    q_rows = np.arange(n_cand, dtype=np.int64)[::step]
    q_sig = q_sig_all[::step]
    del q_sig_all

    truth_of = np.full(n_cand, -1, dtype=np.int64)
    truth_of[t_cd] = t_s1

    params = RevParams(maxdf=args.maxdf, r_query=args.r, k=args.k)
    s1_idx, cand_idx, _, _, _, index = retrieve(
        "train", args.country, args.source, params, args.query_frac, args.procs)
    hit_pairs = set(zip(cand_idx.tolist(), s1_idx.tolist()))

    want = truth_of[q_rows]
    has_truth = want >= 0
    found = np.fromiter(
        ((int(q_rows[i]), int(want[i])) in hit_pairs for i in range(len(q_rows))),
        dtype=bool, count=len(q_rows))

    missed = np.flatnonzero(has_truth & ~found)
    n_true = int(has_truth.sum())
    print(f"{args.country} S{args.source}: {n_true:,} records with a true S1, "
          f"{len(missed):,} missed at k={args.k} "
          f"({len(missed)/n_true:.2%})\n", flush=True)

    df, vocab = index.df, index.vocab
    causes, noises, shared_all = [], [], []
    for qi in missed:
        srow, qrow = int(want[qi]), int(q_rows[qi])
        a = set(s1_sig[srow].split())
        b = set(q_sig[qi].split())
        shared = a & b
        indexed = [t for t in shared if t in vocab]
        if not shared:
            cause = "no shared token"
        elif not indexed:
            cause = "shared but unindexed"
        else:
            cause = "outranked"
        causes.append(cause)
        noises.append(noise_of(q_raw.name_norm.iloc[qrow], q_raw.addr_tok.iloc[qrow]))
        shared_all.append((qi, srow, qrow, cause, sorted(shared), sorted(indexed)))

    res = pd.DataFrame({"cause": causes, "noise": noises})
    print("=== misses by cause ===")
    c = res.cause.value_counts()
    print(pd.DataFrame({"misses": c, "share_of_misses": c / len(res),
                        "share_of_all_true": c / n_true}).to_string(
        float_format=lambda v: f"{v:.2%}"))
    print("\n=== misses by cause x noise type ===")
    print(pd.crosstab(res.noise, res.cause, margins=True).to_string())

    print(f"\n=== {args.examples} random missed pairs ===")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(shared_all), min(args.examples, len(shared_all)),
                      replace=False)
    for j in sorted(pick):
        qi, srow, qrow, cause, shared, indexed = shared_all[j]
        print(f"\n[{cause}] {q_raw.entity_id.iloc[qrow]} -> "
              f"{s1_raw.entity_id.iloc[srow]}")
        print(f"  S1   name: {s1_raw.name_core.iloc[srow]!r}")
        print(f"       addr: {s1_raw.addr_tok.iloc[srow]!r}")
        print(f"  Sx   name: {q_raw.name_core.iloc[qrow]!r}  "
              f"(raw {q_raw.name_norm.iloc[qrow]!r})")
        print(f"       addr: {q_raw.addr_tok.iloc[qrow]!r}")
        print(f"  shared tokens: {shared if shared else 'NONE'}")
        if shared and not indexed:
            print(f"  all shared tokens above df cap: "
                  f"{[(t, df[t]) for t in shared][:6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
