"""Stage-1 + stage-2 training with out-of-fold stage-1 scores, and ablations.

Stage 2 consumes stage-1 scores as features, so those scores must be **out of
fold**: a stage-1 model is overconfident on the rows it was fitted on, and stage 2
trained on such scores learns "high stage-1 score implies correct" far more
strongly than holds at inference, then under-uses its own sibling evidence. Fit
rows therefore get scores from a 5-fold rotation, while validation (and test) use
the single full stage-1 model, matching how inference works.

Reports F0.5 after each addition so only changes that measure better are kept:

    python -m src.train_v3                 # everything
    python -m src.train_v3 --no-stage2     # ablate

Folds use a cheaper fixed-round configuration than the final stage-1 model: their
only job is to produce honest inputs for stage 2, and five early-stopped 3000-round
fits would cost hours for no gain in that role.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decide import Rule, apply_rule
from .features import FEATURE_NAMES, Side, load_pairs, pair_features
from .metric import macro_f_beta
from .paths import WORK, countries, norm_path
from .sampling import FIT_PCT, VAL_PCT, bucket
from .stage2 import STAGE2_NAMES, stage2_features
from .truth import id_positions, label_pairs, truth_counts, truth_pairs

COUNTRY_STRIDE = 10_000_000
N_FOLDS = 5
FOLD_PARAMS = dict(objective="binary", learning_rate=0.10, num_leaves=127,
                   min_data_in_leaf=100, feature_fraction=0.85,
                   bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                   num_threads=0)
FOLD_ROUNDS = 250


def build_all():
    """Base features for every sampled pair, with block bookkeeping."""
    mats, metas = [], []
    truth_parts, ent_parts, bkt_parts = [], [], []
    for ci, country in enumerate(countries("train")):
        s1_pos = id_positions("train", 1, country)
        n_s1 = len(s1_pos)
        tpe = truth_counts(country, s1_pos)
        s1_side = Side.load("train", 1, country)
        for source in (2, 3):
            t0 = time.time()
            pairs = load_pairs("train", country, source)
            b = bucket(pairs.s1_idx.to_numpy())
            pairs = pairs[b < FIT_PCT + VAL_PCT].reset_index(drop=True)
            bkt = b[b < FIT_PCT + VAL_PCT]
            if pairs.empty:
                continue
            cand_pos = id_positions("train", source, country)
            n_cand = len(cand_pos)
            t_s1, t_cd = truth_pairs(country, source, s1_pos, cand_pos)
            del cand_pos
            pairs["source"] = np.int8(source)
            y = label_pairs(pairs, {source: (t_s1, t_cd)}, {source: n_cand})
            del t_s1, t_cd

            cand_side = Side.load("train", source, country)
            mats.append(pair_features(s1_side, cand_side, pairs, source))
            del cand_side
            metas.append(pd.DataFrame({
                "entity": ci * COUNTRY_STRIDE + pairs.s1_idx.to_numpy(),
                "country": np.int8(ci), "source": np.int8(source),
                "s1_idx": pairs.s1_idx.to_numpy(),
                "cand_idx": pairs.cand_idx.to_numpy(),
                "bucket": bkt, "y": y,
            }))
            print(f"  {country} S{source}: {len(pairs):,} pairs, "
                  f"{y.mean():.2%} positive, {time.time() - t0:.0f}s", flush=True)
            del pairs, y
        del s1_side
        sampled = np.flatnonzero(bucket(np.arange(n_s1)) < FIT_PCT + VAL_PCT)
        ent_parts.append(ci * COUNTRY_STRIDE + sampled)
        bkt_parts.append(bucket(sampled))
        truth_parts.append(pd.Series(tpe[sampled],
                                     index=ci * COUNTRY_STRIDE + sampled))
    X = np.concatenate(mats)
    mats.clear()
    meta = pd.concat(metas, ignore_index=True)
    metas.clear()
    return (X, meta, np.concatenate(ent_parts), np.concatenate(bkt_parts),
            pd.concat(truth_parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--conf", type=float, default=0.80)
    ap.add_argument("--reuse-stage1", action="store_true",
                    help="load work/model_s1.txt instead of retraining it, "
                         "for stage-2-only feature changes")
    args = ap.parse_args()

    t0 = time.time()
    X, meta, entities, ent_bucket, truth_of = build_all()
    y = meta.y.to_numpy()
    is_fit = meta.bucket.to_numpy() < FIT_PCT
    print(f"  base features {X.shape}, fit {int(is_fit.sum()):,} / "
          f"val {int((~is_fit).sum()):,}, {time.time() - t0:.0f}s", flush=True)

    # ---- stage 1 -----------------------------------------------------------
    # The early-stopping slice must be split by ENTITY, not by row. Slicing rows
    # leaves each entity partially represented in the tuning set while the metric
    # still counts all of its true matches, so recall is mechanically depressed by
    # the slice fraction — which cost a full 4.8 F0.5 points of apparent score
    # before it was caught.
    val_ent = np.unique(meta.entity.to_numpy()[~is_fit])
    stop_ent = set(val_ent[::5].tolist())
    is_stop_ent = np.fromiter((e in stop_ent for e in meta.entity.to_numpy()),
                              dtype=bool, count=len(meta))
    val_rows = np.flatnonzero(~is_fit)
    stop_rows = np.flatnonzero((~is_fit) & is_stop_ent)
    tune_rows = np.flatnonzero((~is_fit) & ~is_stop_ent)

    if args.reuse_stage1:
        stage1 = lgb.Booster(model_file=f"{WORK}/model_s1.txt")
        print("  stage1 reused from work/model_s1.txt (not retrained)", flush=True)
    else:
        stage1 = lgb.train(
            dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves,
                 min_data_in_leaf=100, feature_fraction=0.85, bagging_fraction=0.8,
                 bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0),
            lgb.Dataset(X[is_fit], label=y[is_fit], feature_name=FEATURE_NAMES),
            num_boost_round=args.rounds,
            valid_sets=[lgb.Dataset(X[stop_rows], label=y[stop_rows],
                                    feature_name=FEATURE_NAMES)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(500)],
        )
        stage1.save_model(f"{WORK}/model_s1.txt")
        print(f"  stage1 best iteration {stage1.best_iteration}", flush=True)

    scores = np.zeros(len(X), dtype=np.float32)
    scores[tune_rows] = stage1.predict(X[tune_rows], num_threads=0)
    scores[stop_rows] = stage1.predict(X[stop_rows], num_threads=0)

    res = {}
    res["stage1"] = report("stage1", scores, meta, entities, ent_bucket,
                           truth_of, tune_rows)

    if args.no_stage2:
        json.dump(res, open(f"{WORK}/v3_results.json", "w"), indent=2)
        return 0

    # ---- out-of-fold stage-1 scores for the fit rows -----------------------
    ent_fold = {e: int((e * 2654435761 >> 7) % N_FOLDS) for e in entities.tolist()}
    row_fold = np.array([ent_fold[e] for e in meta.entity.to_numpy()],
                        dtype=np.int8)
    for f in range(N_FOLDS):
        tr = is_fit & (row_fold != f)
        te = is_fit & (row_fold == f)
        if not te.any():
            continue
        t1 = time.time()
        m = lgb.train(FOLD_PARAMS,
                      lgb.Dataset(X[tr], label=y[tr], feature_name=FEATURE_NAMES),
                      num_boost_round=FOLD_ROUNDS)
        scores[te] = m.predict(X[te], num_threads=0)
        print(f"  oof fold {f}: {int(te.sum()):,} rows, "
              f"{time.time() - t1:.0f}s", flush=True)
        del m

    # ---- stage 2 -----------------------------------------------------------
    S2 = np.zeros((len(X), len(STAGE2_NAMES)), dtype=np.float32)
    for (ci, source), blk in meta.groupby(["country", "source"], sort=False):
        country = countries("train")[int(ci)]
        rows = blk.index.to_numpy()
        s1_side = Side.load("train", 1, country)
        cand_side = Side.load("train", int(source), country)
        n_s1 = len(s1_side.entity_id)
        S2[rows] = stage2_features(
            blk[["s1_idx", "cand_idx"]], scores[rows], X[rows],
            s1_side, cand_side, n_s1, args.conf)
        del s1_side, cand_side
        print(f"  stage2 features {country} S{source}: {len(rows):,}", flush=True)

    # hstack allocates a second full copy, so the base matrix is released
    # immediately: at 38% of entities and ~20 candidates each this is several
    # gigabytes and holding both would risk the overnight run.
    XS = np.hstack([X, scores.reshape(-1, 1), S2])
    del S2, X
    names2 = FEATURE_NAMES + ["stage1_score"] + STAGE2_NAMES
    stage2m = lgb.train(
        dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves,
             min_data_in_leaf=100, feature_fraction=0.85, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0),
        lgb.Dataset(XS[is_fit], label=y[is_fit], feature_name=names2),
        num_boost_round=args.rounds,
        valid_sets=[lgb.Dataset(XS[stop_rows], label=y[stop_rows],
                                feature_name=names2)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(500)],
    )
    stage2m.save_model(f"{WORK}/model_s2.txt")
    print(f"  stage2 best iteration {stage2m.best_iteration}", flush=True)

    s2scores = np.zeros(len(XS), dtype=np.float32)
    s2scores[tune_rows] = stage2m.predict(XS[tune_rows], num_threads=0)
    res["stage2"] = report("stage2", s2scores, meta, entities, ent_bucket,
                           truth_of, tune_rows)

    top = sorted(zip(names2, stage2m.feature_importance("gain")),
                 key=lambda kv: -kv[1])[:12]
    print("  stage2 top features: " + ", ".join(k for k, _ in top))
    json.dump(res, open(f"{WORK}/v3_results.json", "w"), indent=2)
    print(f"\n  results -> {WORK}/v3_results.json")
    return 0


def report(label: str, scores: np.ndarray, meta: pd.DataFrame,
           entities: np.ndarray, ent_bucket: np.ndarray, truth_of: pd.Series,
           rows: np.ndarray) -> dict:
    """Tune the decision rule on the given rows and print the best F0.5."""
    ent = meta.entity.to_numpy()[rows]
    s1 = meta.s1_idx.to_numpy()[rows]
    yy = meta.y.to_numpy()[rows]
    sc = scores[rows]
    # Only entities present in the scored rows, so predicted and true counts refer
    # to the same population.
    val_ents = np.intersect1d(entities[ent_bucket >= FIT_PCT], np.unique(ent))
    codes = pd.Index(val_ents)
    ecode = codes.get_indexer(ent)
    keep_rows = ecode >= 0
    ent, s1, yy, sc, ecode = (ent[keep_rows], s1[keep_rows], yy[keep_rows],
                              sc[keep_rows], ecode[keep_rows])
    true_counts = truth_of.reindex(val_ents).to_numpy()

    best = (None, -1.0)
    for thr in np.arange(0.20, 0.91, 0.02):
        for rel in (0.0, 0.3, 0.5, 0.7):
            rule = Rule(threshold=float(thr), rel_floor=rel)
            m = apply_rule(ent, sc, rule)
            if not m.any():
                continue
            pred = np.bincount(ecode[m], minlength=len(val_ents))
            tp = np.bincount(ecode[m], weights=yy[m].astype(float),
                             minlength=len(val_ents))
            f = macro_f_beta(pred, true_counts, tp)
            if f > best[1]:
                best = (rule, f)
    rule, f = best
    print(f"\n  [{label}] macro F0.5 = {f:.4f}  with {rule.label()}  "
          f"({len(val_ents):,} val entities)", flush=True)
    return {"f05": f, "threshold": rule.threshold, "rel_floor": rule.rel_floor}


if __name__ == "__main__":
    sys.exit(main())
