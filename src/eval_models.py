"""Honest validation score for already-trained models, on *all* validation rows.

Separate from training so a score can be re-measured without refitting, and so the
metric population is unambiguous: every validation entity, every one of its
candidate pairs, tuned threshold and relative floor.

    python -m src.eval_models                  # stage 1, and stage 2 if present
    python -m src.eval_models --model model.txt --single   # the v2 model
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

from .decide import Rule, apply_rule
from .features import Side, load_pairs, pair_features
from .metric import macro_f_beta
from .paths import WORK, countries
from .sampling import FIT_PCT, VAL_PCT, bucket
from .stage2 import stage2_features
from .truth import id_positions, label_pairs, truth_counts, truth_pairs

STRIDE = 10_000_000


def build_val(models: dict, conf: float = 0.80):
    parts, ents, truths = [], [], []
    for ci, country in enumerate(countries("train")):
        s1_pos = id_positions("train", 1, country)
        n_s1 = len(s1_pos)
        tpe = truth_counts(country, s1_pos)
        s1_side = Side.load("train", 1, country)
        for source in (2, 3):
            pairs = load_pairs("train", country, source)
            b = bucket(pairs.s1_idx.to_numpy())
            keep = (b >= FIT_PCT) & (b < FIT_PCT + VAL_PCT)
            pairs = pairs[keep].reset_index(drop=True)
            if pairs.empty:
                continue
            cand_pos = id_positions("train", source, country)
            n_cand = len(cand_pos)
            t = truth_pairs(country, source, s1_pos, cand_pos)
            del cand_pos
            pairs["source"] = np.int8(source)
            y = label_pairs(pairs, {source: t}, {source: n_cand})
            cand_side = Side.load("train", source, country)
            X = pair_features(s1_side, cand_side, pairs, source)
            want = models["stage1"].num_feature()
            sc = models["stage1"].predict(X[:, :want],
                                          num_threads=0).astype(np.float32)
            if models.get("stage2") is not None:
                S2 = stage2_features(pairs[["s1_idx", "cand_idx"]], sc, X,
                                     s1_side, cand_side, n_s1, conf)
                XS = np.hstack([X, sc.reshape(-1, 1), S2])
                sc = models["stage2"].predict(XS, num_threads=0).astype(np.float32)
                del XS, S2
            del X, cand_side
            parts.append(pd.DataFrame({
                "entity": ci * STRIDE + pairs.s1_idx.to_numpy(),
                "country": country, "y": y, "score": sc}))
            print(f"  {country} S{source}: {len(y):,} pairs scored", flush=True)
            del pairs, y
        del s1_side
        sampled = np.flatnonzero((bucket(np.arange(n_s1)) >= FIT_PCT)
                                 & (bucket(np.arange(n_s1)) < FIT_PCT + VAL_PCT))
        ents.append(pd.DataFrame({"entity": ci * STRIDE + sampled,
                                  "country": country, "n_true": tpe[sampled]}))
    return pd.concat(parts, ignore_index=True), pd.concat(ents, ignore_index=True)


def best_rule(data: pd.DataFrame, ents: pd.DataFrame) -> tuple[Rule, float]:
    codes = pd.Index(ents.entity.to_numpy())
    ecode = codes.get_indexer(data.entity.to_numpy())
    ok = ecode >= 0
    ecode, sc = ecode[ok], data.score.to_numpy()[ok]
    yy = data.y.to_numpy()[ok]
    e1 = data.entity.to_numpy()[ok]
    true_counts = ents.n_true.to_numpy()
    best = (Rule(), -1.0)
    for thr in np.arange(0.20, 0.93, 0.02):
        for rel in (0.0, 0.3, 0.5, 0.7):
            rule = Rule(threshold=float(thr), rel_floor=rel)
            m = apply_rule(e1, sc, rule)
            if not m.any():
                continue
            pred = np.bincount(ecode[m], minlength=len(ents))
            tp = np.bincount(ecode[m], weights=yy[m].astype(float),
                             minlength=len(ents))
            f = macro_f_beta(pred, true_counts, tp)
            if f > best[1]:
                best = (rule, float(f))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_s1.txt")
    ap.add_argument("--single", action="store_true", help="ignore any stage-2 model")
    args = ap.parse_args()
    t0 = time.time()

    models = {"stage1": lgb.Booster(model_file=f"{WORK}/{args.model}"),
              "stage2": None}
    if not args.single and os.path.exists(f"{WORK}/model_s2.txt"):
        models["stage2"] = lgb.Booster(model_file=f"{WORK}/model_s2.txt")
    label = "stage1+stage2" if models["stage2"] is not None else args.model
    print(f"evaluating {label} on ALL validation rows", flush=True)

    data, ents = build_val(models)
    rule, f = best_rule(data, ents)
    print(f"\n  ALL   macro F0.5 = {f:.4f}  with {rule.label()}  "
          f"({len(ents):,} entities, {len(data):,} pairs)")
    for country, g in data.groupby("country"):
        ge = ents[ents.country == country]
        r2, f2 = best_rule(g, ge)
        print(f"  {country:<7} macro F0.5 = {f2:.4f}  with {r2.label()}  "
              f"({len(ge):,} entities)")
    json.dump({"label": label, "f05": f, "threshold": rule.threshold,
               "rel_floor": rule.rel_floor},
              open(f"{WORK}/eval_{label.replace('+', '_')}.json", "w"), indent=2)
    print(f"\nelapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
