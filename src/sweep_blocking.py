"""Stage 2b — tune blocking against measured ground-truth recall.

Blocking recall is the hard ceiling on the final score, so the parameters are
chosen by measurement rather than by guess. One country's tables and its truth
pairs are loaded once, then every configuration is evaluated in-process.

Reports, per setting, both numbers that matter: recall (micro over true pairs,
and macro per S1 entity, which is what F0.5 averages) and candidates per entity.

    python -m src.sweep_blocking --country India --source 2
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from .blocking import BlockParams, TokenIndex, collect, prefix_docs
from .paths import WORK, norm_path

# (maxdf_addr, k_addr, maxdf_name, k_name, maxdf_prefix, k_prefix)
# Each pass is reported alone and as the union, which is what blocking emits.
GRID = [
    (2_000, 20, 2_000, 10, 0, 0),        # token-exact union, the current best
    (2_000, 20, 20_000, 10, 0, 0),
    (2_000, 20, 2_000, 10, 2_000, 10),   # + typo-tolerant prefix pass
    (2_000, 20, 20_000, 10, 5_000, 10),
    (5_000, 24, 20_000, 12, 5_000, 12),
]


def load(country: str, source: int):
    s1 = pd.read_parquet(norm_path("train", 1, country),
                         columns=["entity_id", "addr_tok", "name_core"])
    cand = pd.read_parquet(norm_path("train", source, country),
                           columns=["entity_id", "addr_tok", "name_core"])
    s1_pos = {e: i for i, e in enumerate(s1.entity_id)}
    cand_pos = {e: i for i, e in enumerate(cand.entity_id)}

    gt = pd.read_parquet(f"{WORK}/train_ground_truth.parquet")
    gt = gt[(gt.matched_entity_ids.str.len() > 0)
            & gt.source1_entity_id.isin(s1_pos.keys())]
    ex = gt.assign(mid=gt.matched_entity_ids.str.split(",")).explode("mid")
    ex = ex[ex.mid.str.startswith(f"S{source}-")]
    t_s1 = ex.source1_entity_id.map(s1_pos).to_numpy(dtype=np.int64)
    t_cd = ex.mid.map(cand_pos).to_numpy(dtype=np.int64)
    return s1, cand, len(cand_pos), t_s1, t_cd


def recall_of(pair_s1, pair_cd, t_s1, t_cd, n_cand, n_s1):
    """Micro (per true pair) and macro (per S1 entity) recall."""
    got = np.sort(pair_s1.astype(np.int64) * n_cand + pair_cd.astype(np.int64))
    want = t_s1 * n_cand + t_cd
    pos = np.searchsorted(got, want)
    hit = (pos < len(got)) & (got[np.minimum(pos, len(got) - 1)] == want)
    per_true = np.bincount(t_s1, minlength=n_s1)
    per_hit = np.bincount(t_s1, weights=hit.astype(np.float64), minlength=n_s1)
    live = per_true > 0
    return hit.mean(), (per_hit[live] / per_true[live]).mean()


def dedup(parts):
    """Union of pass outputs, duplicate (s1, cand) pairs removed."""
    s1 = np.concatenate([p[0] for p in parts])
    cd = np.concatenate([p[1] for p in parts])
    key = np.unique(s1.astype(np.int64) << 32 | cd.astype(np.int64))
    return (key >> 32).astype(np.int64), (key & 0xFFFFFFFF).astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--source", type=int, default=2)
    args = ap.parse_args()

    s1, cand, n_cand, t_s1, t_cd = load(args.country, args.source)
    n_s1 = len(s1)
    print(f"{args.country} S{args.source}: {n_s1:,} S1 x {n_cand:,} candidates, "
          f"{len(t_s1):,} true pairs\n", flush=True)
    addr_len = cand.addr_tok.str.len().to_numpy()
    has_addr = np.flatnonzero(addr_len > 0)
    cand_addr = cand.addr_tok.to_numpy()[has_addr]
    cand_name = cand.name_core.to_numpy()
    all_rows = np.arange(len(cand), dtype=np.int64)
    s1_addr, s1_name = s1.addr_tok.to_numpy(), s1.name_core.to_numpy()
    del cand, s1

    print(f"{'setting':<74} {'micro':>7} {'macro':>7} {'cand/ent':>9} {'secs':>6}",
          flush=True)
    pre_cand = pre_s1 = None
    for maxdf_a, k_a, maxdf_n, k_n, maxdf_p, k_p in GRID:
        params = BlockParams(maxdf_addr=maxdf_a, k_addr=k_a, maxdf_name=maxdf_n,
                             k_name=k_n, maxdf_prefix=maxdf_p, k_prefix=k_p)
        t0 = time.time()
        parts, report = [], []

        idx = TokenIndex(cand_addr, maxdf_a)
        a = collect(idx, s1_addr, k_a, params.r_query, has_addr, params)
        del idx
        parts.append(a)
        report.append(("addr only", a))

        idx = TokenIndex(cand_name, maxdf_n)
        nm = collect(idx, s1_name, k_n, params.r_query, all_rows, params)
        del idx
        parts.append(nm)
        report.append(("name only", nm))

        if k_p:
            if pre_cand is None or pre_cand[0] != params.prefix_len:
                pre_cand = (params.prefix_len, prefix_docs(cand_addr, params.prefix_len))
                pre_s1 = prefix_docs(s1_addr, params.prefix_len)
            idx = TokenIndex(pre_cand[1], maxdf_p)
            pr = collect(idx, pre_s1, k_p, params.r_query, has_addr, params)
            del idx
            parts.append(pr)
            report.append(("prefix only", pr))

        u_s1, u_cd = dedup(parts)
        report.append(("UNION", (u_s1, u_cd, None)))
        secs = time.time() - t0
        for label, (p1, p2, _) in report:
            micro, macro = recall_of(p1, p2, t_s1, t_cd, n_cand, n_s1)
            tag = params.label() if label == "UNION" else f"  {label}"
            print(f"{tag:<74} {micro:6.2%} {macro:6.2%} {len(p1) / n_s1:9.2f}"
                  f" {secs if label == 'UNION' else 0:6.0f}", flush=True)
        print(flush=True)
        del parts, report
    return 0


if __name__ == "__main__":
    sys.exit(main())
