"""Stage 4 — train the pair classifier and tune the F0.5 threshold.

Entities (not pairs) are split into a fit and a validation sample by a
deterministic hash of the Source-1 row index, so every candidate of a sampled
entity stays together — the per-entity rank features and the macro metric are
both only meaningful over a complete candidate set.

    python -m src.train
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, Side, load_pairs, pair_features
from .metric import macro_f_beta
from .paths import WORK, countries, norm_path
from .sampling import FIT_PCT, VAL_PCT, bucket
from .truth import id_positions, label_pairs, truth_counts, truth_pairs

COUNTRY_STRIDE = 10_000_000


def gather(split: str, source: int, country: str, wanted: np.ndarray):
    """Load only the rows in ``wanted`` and return (Side, remap array)."""
    df = pd.read_parquet(norm_path(split, source, country))
    side = Side.from_frame(df.iloc[wanted].reset_index(drop=True))
    remap = np.full(len(df), -1, dtype=np.int64)
    remap[wanted] = np.arange(len(wanted))
    del df
    return side, remap


def build_country(country: str, ci: int):
    """Features, labels and metadata for one country's sampled entities."""
    s1_pos = id_positions("train", 1, country)
    n_s1 = len(s1_pos)
    true_per_entity = truth_counts(country, s1_pos)

    mats, metas = [], []
    for source in (2, 3):
        pairs = load_pairs("train", country, source)
        b = bucket(pairs.s1_idx.to_numpy())
        keep = b < (FIT_PCT + VAL_PCT)
        pairs = pairs[keep].reset_index(drop=True)
        bkt = b[keep]
        del b
        if pairs.empty:
            continue

        cand_pos = id_positions("train", source, country)
        n_cand = len(cand_pos)
        t_s1, t_cd = truth_pairs(country, source, s1_pos, cand_pos)
        del cand_pos
        pairs["source"] = np.int8(source)
        y = label_pairs(pairs, {source: (t_s1, t_cd)}, {source: n_cand})
        del t_s1, t_cd

        s1_rows = np.unique(pairs.s1_idx.to_numpy())
        cand_rows = np.unique(pairs.cand_idx.to_numpy())
        s1_side, s1_remap = gather("train", 1, country, s1_rows)
        cand_side, cand_remap = gather("train", source, country, cand_rows)

        local = pairs.copy()
        local["s1_idx"] = s1_remap[pairs.s1_idx.to_numpy()]
        local["cand_idx"] = cand_remap[pairs.cand_idx.to_numpy()]
        local = local.sort_values("s1_idx", kind="stable")
        order = local.index.to_numpy()
        mats.append(pair_features(s1_side, cand_side,
                                 local.reset_index(drop=True), source))
        del s1_side, cand_side, local

        metas.append(pd.DataFrame({
            "entity": ci * COUNTRY_STRIDE + pairs.s1_idx.to_numpy()[order],
            "cand_key": (np.int64(source) << 40) + pairs.cand_idx.to_numpy()[order],
            "bucket": bkt[order],
            "y": y[order],
        }))
        del pairs, y, bkt

    X = np.concatenate(mats) if len(mats) > 1 else mats[0]
    mats.clear()
    meta = pd.concat(metas, ignore_index=True)
    sampled = np.flatnonzero(bucket(np.arange(n_s1)) < FIT_PCT + VAL_PCT)
    entities = ci * COUNTRY_STRIDE + sampled
    return X, meta, entities, bucket(sampled), pd.Series(
        true_per_entity[sampled], index=entities)


def sweep_threshold(scores: np.ndarray, data: pd.DataFrame, entities: np.ndarray,
                    truth_of: pd.Series, exclusive_pass: bool) -> tuple[float, float]:
    """Best threshold and its macro F0.5 over the validation entities."""
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    ent = data.entity.to_numpy()[order]
    y = data.y.to_numpy()[order]
    # A threshold selects a prefix of the score-sorted pairs, so the global
    # "first claim wins" duplicate mask restricted to that prefix is exactly the
    # exclusivity outcome at that threshold — computed once, reused for all.
    if exclusive_pass:
        key = ent // COUNTRY_STRIDE * COUNTRY_STRIDE + data.cand_key.to_numpy()[order]
        alive = ~pd.Index(key).duplicated(keep="first")
    else:
        alive = np.ones(len(s), dtype=bool)

    codes = pd.Index(entities)
    ent_code = codes.get_indexer(ent)
    true_counts = truth_of.reindex(entities).to_numpy()
    n_ent = len(entities)

    best = (0.0, -1.0)
    for t in np.arange(0.05, 0.96, 0.01):
        sel = alive & (s >= t)
        if not sel.any():
            continue
        pred = np.bincount(ent_code[sel], minlength=n_ent)
        tp = np.bincount(ent_code[sel], weights=y[sel].astype(np.float64),
                         minlength=n_ent)
        f = macro_f_beta(pred, true_counts, tp)
        if f > best[1]:
            best = (float(t), f)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    mats, metas, ents, bkts, truths = [], [], [], [], []
    for ci, country in enumerate(countries("train")):
        t0 = time.time()
        X, meta, entities, ent_bucket, truth_of = build_country(country, ci)
        print(f"  {country}: {len(meta):,} sampled pairs, "
              f"{meta.y.mean():.3%} positive, {time.time() - t0:.0f}s", flush=True)
        mats.append(X)
        metas.append(meta)
        ents.append(entities)
        bkts.append(ent_bucket)
        truths.append(truth_of)

    X = np.concatenate(mats)
    mats.clear()
    meta = pd.concat(metas, ignore_index=True)
    metas.clear()
    entities = np.concatenate(ents)
    ent_bucket = np.concatenate(bkts)
    truth_of = pd.concat(truths)

    is_fit = meta.bucket.to_numpy() < FIT_PCT
    y = meta.y.to_numpy()
    print(f"  fit pairs {int(is_fit.sum()):,}   val pairs {int((~is_fit).sum()):,}"
          f"   features {X.shape[1]}", flush=True)

    # Early stopping on a slice of the validation entities, so tree count is
    # chosen by measurement rather than guessed. The slice is held out of the
    # threshold tuning below to keep that estimate honest.
    val_all = np.flatnonzero(~is_fit)
    stop_slice = val_all[::5]
    tune_mask = ~is_fit
    tune_mask[stop_slice] = False

    model = lgb.train(
        dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves,
             min_data_in_leaf=100, feature_fraction=0.85, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0),
        lgb.Dataset(X[is_fit], label=y[is_fit], feature_name=FEATURE_NAMES),
        num_boost_round=args.rounds,
        valid_sets=[lgb.Dataset(X[stop_slice], label=y[stop_slice],
                                feature_name=FEATURE_NAMES)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)],
    )
    print(f"  best iteration {model.best_iteration} of {args.rounds}", flush=True)
    model.save_model(f"{WORK}/model.txt")

    val_meta = meta.loc[tune_mask].reset_index(drop=True)
    val_entities = entities[ent_bucket >= FIT_PCT]
    scores = model.predict(X[tune_mask], num_threads=0).astype(np.float32)
    del X

    plain_t, plain_f = sweep_threshold(scores, val_meta, val_entities, truth_of, False)
    excl_t, excl_f = sweep_threshold(scores, val_meta, val_entities, truth_of, True)
    print(f"\n  validation entities: {len(val_entities):,}")
    print(f"  macro F0.5  no exclusivity : {plain_f:.4f} at threshold {plain_t:.2f}")
    print(f"  macro F0.5  + exclusivity  : {excl_f:.4f} at threshold {excl_t:.2f}")

    top = sorted(zip(FEATURE_NAMES, model.feature_importance("gain")),
                 key=lambda kv: -kv[1])[:10]
    print("  top features by gain: " + ", ".join(k for k, _ in top))

    json.dump({"threshold": excl_t, "val_f05": excl_f,
               "threshold_no_exclusivity": plain_t,
               "val_f05_no_exclusivity": plain_f,
               "fit_pct": FIT_PCT, "val_pct": VAL_PCT,
               "val_entities": int(len(val_entities))},
              open(f"{WORK}/threshold.json", "w"), indent=2)
    print(f"  model -> {WORK}/model.txt   threshold -> {WORK}/threshold.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
