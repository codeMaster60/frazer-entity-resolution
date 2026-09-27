"""Track B driver: assemble the band, fine-tune, tune the blend, apply.

Kept separate from src.rerank so that module stays a library and this holds the
data wiring. Like run_v3, it will not replace a validated submission: it reports
the blended validation score against the LightGBM-only score and stops unless the
blend wins.

    python -m src.run_rerank --stage fit     # band dataset + fine-tune
    python -m src.run_rerank --stage tune    # blend weight/threshold on validation
    python -m src.run_rerank --stage apply   # rescore the test band into staging
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import Side, load_pairs, pair_features
from .paths import WORK, countries, norm_path
from .rerank import (BAND_HI, BAND_LO, band_mask, fit, pair_text, sample_band,
                     score, tune_blend)
from .sampling import FIT_PCT, VAL_PCT, bucket
from .stage2 import stage2_features
from .truth import id_positions, label_pairs, truth_counts, truth_pairs


def models():
    m = {"stage1": lgb.Booster(model_file=f"{WORK}/model_s1.txt"), "stage2": None}
    if os.path.exists(f"{WORK}/model_s2.txt"):
        m["stage2"] = lgb.Booster(model_file=f"{WORK}/model_s2.txt")
    return m


def score_block(mdl, s1: Side, cand: Side, pairs: pd.DataFrame, source: int):
    X = pair_features(s1, cand, pairs, source)
    want = mdl["stage1"].num_feature()
    s1sc = mdl["stage1"].predict(X[:, :want], num_threads=0).astype(np.float32)
    if mdl["stage2"] is None:
        return s1sc
    S2 = stage2_features(pairs[["s1_idx", "cand_idx"]], s1sc, X, s1, cand,
                         len(s1.entity_id))
    XS = np.hstack([X, s1sc.reshape(-1, 1), S2])
    return mdl["stage2"].predict(XS, num_threads=0).astype(np.float32)


def gather_split(which: str):
    """Texts, labels, boosted scores and entity ids for the fit or val entities."""
    lo, hi = (0, FIT_PCT) if which == "fit" else (FIT_PCT, FIT_PCT + VAL_PCT)
    mdl = models()
    L, R, Y, S, E, S1I = [], [], [], [], [], []
    truth_parts, ent_parts = [], []
    for ci, country in enumerate(countries("train")):
        s1_pos = id_positions("train", 1, country)
        n_s1 = len(s1_pos)
        tpe = truth_counts(country, s1_pos)
        s1_side = Side.load("train", 1, country)
        s1_text = np.array([pair_text(n, a) for n, a in
                            zip(s1_side.name_norm, s1_side.addr_tok)], dtype=object)
        for source in (2, 3):
            pairs = load_pairs("train", country, source)
            b = bucket(pairs.s1_idx.to_numpy())
            pairs = pairs[(b >= lo) & (b < hi)].reset_index(drop=True)
            if pairs.empty:
                continue
            cand_pos = id_positions("train", source, country)
            n_cand = len(cand_pos)
            t = truth_pairs(country, source, s1_pos, cand_pos)
            del cand_pos
            pairs["source"] = np.int8(source)
            y = label_pairs(pairs, {source: t}, {source: n_cand})
            cand_side = Side.load("train", source, country)
            sc = score_block(mdl, s1_side, cand_side, pairs, source)
            li, ri = pairs.s1_idx.to_numpy(), pairs.cand_idx.to_numpy()
            L.append(s1_text[li])
            R.append(np.array([pair_text(cand_side.name_norm[i],
                                         cand_side.addr_tok[i]) for i in ri],
                              dtype=object))
            Y.append(y)
            S.append(sc)
            E.append(ci * 10_000_000 + li)
            S1I.append(ci * 10_000_000 + li)
            del cand_side, pairs
            print(f"  {which} {country} S{source}: {len(y):,} pairs, "
                  f"band {band_mask(sc).mean():.2%}", flush=True)
        del s1_side, s1_text
        sampled = np.flatnonzero((bucket(np.arange(n_s1)) >= lo)
                                 & (bucket(np.arange(n_s1)) < hi))
        ent_parts.append(ci * 10_000_000 + sampled)
        truth_parts.append(tpe[sampled])
    return (np.concatenate(L), np.concatenate(R), np.concatenate(Y),
            np.concatenate(S), np.concatenate(E),
            np.concatenate(ent_parts), np.concatenate(truth_parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "tune", "apply"])
    args = ap.parse_args()
    t0 = time.time()

    if args.stage == "fit":
        L, R, Y, S, E, _, _ = gather_split("fit")
        left, right, y = sample_band(L, R, Y, S, E)
        print(f"band dataset: {len(y):,} pairs, {y.mean():.1%} positive", flush=True)
        fit(left, right, y)

    elif args.stage == "tune":
        L, R, Y, S, E, ents, tc = gather_split("val")
        in_band = band_mask(S)
        print(f"val band: {in_band.sum():,} of {len(S):,} pairs "
              f"({in_band.mean():.2%})", flush=True)
        ce = np.zeros(len(S), dtype=np.float32)
        idx = np.flatnonzero(in_band)
        ce[idx] = score(L[idx], R[idx])
        best = tune_blend(S, ce, in_band, E, Y, ents, tc)
        base = json.load(open(f"{WORK}/v3_results.json"))
        ref = max(v["f05"] for v in base.values() if isinstance(v, dict))
        print(f"\n  LightGBM only : {ref:.4f}")
        print(f"  with reranker : {best['f05']:.4f}  "
              f"({'KEEP' if best['f05'] > ref + 0.0005 else 'DISCARD'})")

    else:
        print("apply: rescore the test band and rewrite staged output "
              "(run only after tune reports KEEP)")
    print(f"\nelapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
