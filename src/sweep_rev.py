"""Tune reverse blocking: the recall@k curve and its cost, from one pass.

Retrieving with a large k and deriving each hit's rank gives recall at every
smaller k from a single run, which separates the two failure modes: a curve that
plateaus below 100% is a *retrieval* failure (the right S1 is never returned, so
only new tokens help), while a rising curve is a *ranking* failure (a larger k or
better scoring recovers it).

Runs through the same parallel retrieval core as the real blocking stage.

    python -m src.sweep_rev --country India --source 2
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .blocking_rev import N_PROCS, RevParams, retrieve
from .truth import id_positions, truth_pairs

KS = (1, 2, 3, 4, 5, 8, 12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--source", type=int, default=2)
    ap.add_argument("--query-frac", type=float, default=0.08)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--configs", default="60000:28:0,60000:28:1",
                    help="maxdf:r_query:length_norm triples")
    ap.add_argument("--procs", type=int, default=N_PROCS)
    ap.add_argument("--rich", action="store_true",
                    help="add the address prefix/skeleton tokens (measured worse)")
    args = ap.parse_args()

    s1_pos = id_positions("train", 1, args.country)
    cand_pos = id_positions("train", args.source, args.country)
    n_s1, n_cand = len(s1_pos), len(cand_pos)
    t_s1, t_cd = truth_pairs(args.country, args.source, s1_pos, cand_pos)
    del s1_pos, cand_pos
    truth_of = np.full(n_cand, -1, dtype=np.int64)
    truth_of[t_cd] = t_s1

    print(f"{args.country} S{args.source}: {n_cand:,} records, {n_s1:,} S1, "
          f"query_frac={args.query_frac}, rich={args.rich}\n", flush=True)
    print(f"{'config':<20} " + " ".join(f"r@{k:<4}" for k in KS)
          + f" {'cand/S1@3':>10} {'secs':>6}", flush=True)

    for cfg in args.configs.split(","):
        bits = cfg.split(":")
        maxdf, r = int(bits[0]), int(bits[1])
        ln = bool(int(bits[2])) if len(bits) > 2 else False
        params = RevParams(maxdf=maxdf, r_query=r, k=args.kmax, length_norm=ln)
        t0 = time.time()
        s1_idx, cand_idx, score, _, n_q, _ = retrieve(
            "train", args.country, args.source, params, args.query_frac,
            args.procs, None, args.rich)
        # rank per queried record, best score first
        order = np.lexsort((-score, cand_idx))
        cand_s, s1_s = cand_idx[order], s1_idx[order]
        starts = np.flatnonzero(np.diff(cand_s, prepend=np.int32(-1)))
        rank = np.arange(len(cand_s)) - np.repeat(
            starts, np.diff(np.append(starts, len(cand_s))))
        want = truth_of[cand_s.astype(np.int64)]
        hit = (want >= 0) & (s1_s.astype(np.int64) == want)

        best = {}
        for c, rk in zip(cand_s[hit], rank[hit]):
            if c not in best or rk < best[c]:
                best[c] = rk
        queried = np.unique(cand_idx)
        n_with_truth = int((truth_of[queried.astype(np.int64)] >= 0).sum())
        ranks = np.array(list(best.values())) if best else np.empty(0, int)
        cells = [f"{(ranks < k).sum() / max(n_with_truth, 1):5.2%}" for k in KS]
        per_s1 = len(cand_idx) / max(len(queried), 1) * min(3, args.kmax) \
            / args.kmax * (n_cand / n_s1)
        print(f"maxdf={maxdf},r={r},ln={int(ln):<3} " + " ".join(cells)
              + f" {per_s1:10.2f} {time.time() - t0:6.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
