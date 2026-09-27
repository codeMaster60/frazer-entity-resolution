"""Error attribution on a realistic validation split.

Splits the lost F0.5 into the three causes that need different fixes:

* **blocking miss** — the true pair never reached the model, so no threshold can
  recover it. This is the recall ceiling.
* **false negative** — the pair was scored but fell below the threshold.
* **false positive** — a wrong pair was predicted, which F0.5 punishes hardest.

and then breaks the blocking misses and false negatives down by the noise type of
the candidate record, so effort goes where the loss actually is.

    python -m src.analyze
"""
from __future__ import annotations

import argparse
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, Side, load_pairs, pair_features
from .metric import exclusive, macro_f_beta
from .paths import WORK, countries, norm_path
from .sampling import FIT_PCT, VAL_PCT, bucket
from .truth import id_positions, label_pairs, truth_counts, truth_pairs

NOISE_ORDER = ["indic script", "url-like name", "empty address", "alias name",
               "allcaps", "clean-ish"]


def noise_labels(side: Side, idx: np.ndarray, s1: Side, s1_idx: np.ndarray
                 ) -> np.ndarray:
    """One coarse noise class per candidate record, most specific first."""
    flags = side.flags[idx]
    out = np.full(len(idx), "clean-ish", dtype=object)
    name1 = s1.name_core[s1_idx]
    name2 = side.name_core[idx]
    # "alias": the noisy name shares no token with the clean one
    alias = np.fromiter(
        (len(set(a.split()) & set(b.split())) == 0 for a, b in zip(name1, name2)),
        dtype=bool, count=len(idx))
    out[alias] = "alias name"
    upper = np.fromiter((n.isupper() for n in side.name_norm[idx]),
                        dtype=bool, count=len(idx))
    out[upper & (out == "clean-ish")] = "allcaps"
    out[side.addr_tok[idx] == ""] = "empty address"
    out[(flags & 4) > 0] = "url-like name"
    out[(flags & 1) > 0] = "indic script"
    return out


def analyze_country(model, threshold: float, country: str):
    s1_pos = id_positions("train", 1, country)
    n_s1 = len(s1_pos)
    true_per_entity = truth_counts(country, s1_pos)
    s1_side = Side.load("train", 1, country)

    rows = []            # scored pairs
    miss_noise: list[np.ndarray] = []
    n_true_total = 0
    for source in (2, 3):
        pairs = load_pairs("train", country, source)
        cand_pos = id_positions("train", source, country)
        n_cand = len(cand_pos)
        t_s1, t_cd = truth_pairs(country, source, s1_pos, cand_pos)
        del cand_pos
        # restrict truth to validation entities
        b = bucket(t_s1.astype(np.int64))
        vm = (b >= FIT_PCT) & (b < FIT_PCT + VAL_PCT)
        t_s1, t_cd = t_s1[vm], t_cd[vm]
        n_true_total += len(t_s1)

        pb = bucket(pairs.s1_idx.to_numpy())
        pairs = pairs[(pb >= FIT_PCT) & (pb < FIT_PCT + VAL_PCT)].reset_index(drop=True)
        pairs["source"] = np.int8(source)
        y = label_pairs(pairs, {source: (t_s1, t_cd)}, {source: n_cand})

        cand_side = Side.load("train", source, country)
        X = pair_features(s1_side, cand_side, pairs, source)
        # New features are appended to FEATURE_NAMES, so the leading columns still
        # match an older model's feature set exactly and can simply be sliced.
        want = model.num_feature()
        score = model.predict(X[:, :want] if X.shape[1] > want else X,
                              num_threads=0).astype(np.float32)
        del X

        # blocking misses: true pairs absent from the candidate set
        have = np.sort(pairs.s1_idx.to_numpy().astype(np.int64) * n_cand
                       + pairs.cand_idx.to_numpy().astype(np.int64))
        want = t_s1 * n_cand + t_cd
        pos = np.searchsorted(have, want)
        found = (pos < len(have)) & (have[np.minimum(pos, len(have) - 1)] == want)
        if (~found).any():
            miss_noise.append(noise_labels(cand_side, t_cd[~found],
                                          s1_side, t_s1[~found]))

        rows.append(pd.DataFrame({
            "s1_idx": pairs.s1_idx.to_numpy(), "source": pairs.source.to_numpy(),
            "cand_idx": pairs.cand_idx.to_numpy(), "y": y, "score": score,
            "noise": noise_labels(cand_side, pairs.cand_idx.to_numpy(),
                                 s1_side, pairs.s1_idx.to_numpy()),
        }))
        del cand_side, pairs
    del s1_side

    data = pd.concat(rows, ignore_index=True)
    ents = np.flatnonzero((bucket(np.arange(n_s1)) >= FIT_PCT)
                          & (bucket(np.arange(n_s1)) < FIT_PCT + VAL_PCT))
    return data, ents, true_per_entity[ents], n_true_total, miss_noise


def score_of(data: pd.DataFrame, ents: np.ndarray, true_counts: np.ndarray,
             threshold: float, use_exclusive: bool) -> tuple[float, dict]:
    sel = data.score.to_numpy() >= threshold
    sub = data[sel]
    if use_exclusive and len(sub):
        sub = sub[exclusive(sub, sub.score.to_numpy())]
    code = pd.Index(ents).get_indexer(sub.s1_idx.to_numpy())
    pred = np.bincount(code, minlength=len(ents))
    tp = np.bincount(code, weights=sub.y.to_numpy().astype(float),
                     minlength=len(ents))
    f = macro_f_beta(pred, true_counts, tp)
    return f, {"predicted": int(len(sub)), "tp": int(tp.sum()),
               "fp": int(len(sub) - tp.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--model", default="model.txt")
    args = ap.parse_args()
    model = lgb.Booster(model_file=f"{WORK}/{args.model}")
    tuned = json.load(open(f"{WORK}/threshold.json"))
    thr = args.threshold if args.threshold is not None else tuned["threshold"]

    all_parts, summary = [], []
    for country in countries("train"):
        data, ents, tc, n_true, miss = analyze_country(model, thr, country)
        f_excl, st = score_of(data, ents, tc, thr, True)
        f_plain, _ = score_of(data, ents, tc, thr, False)
        n_found = int(data.y.sum())
        summary.append({
            "country": country, "val_entities": len(ents), "true_pairs": n_true,
            "blocked": n_found, "block_recall": n_found / max(n_true, 1),
            "F0.5": f_excl, "F0.5_no_excl": f_plain,
            "predicted": st["predicted"], "tp": st["tp"], "fp": st["fp"],
            "precision": st["tp"] / max(st["predicted"], 1),
            "recall_of_true": st["tp"] / max(n_true, 1),
            "cand_per_entity": len(data) / len(ents),
        })
        all_parts.append((data, ents, tc, miss, n_true))
        print(f"  {country}: F0.5={f_excl:.4f}  blocking recall="
              f"{n_found / max(n_true, 1):.2%}  cand/entity={len(data)/len(ents):.1f}",
              flush=True)

    print("\n=== per country ===")
    print(pd.DataFrame(summary).to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    # pooled, weighted the way the test set is weighted
    print("\n=== loss attribution (pooled validation) ===")
    tot_true = sum(p[4] for p in all_parts)
    tot_blocked = sum(int(p[0].y.sum()) for p in all_parts)
    print(f"true pairs {tot_true:,}")
    print(f"  reached the model      {tot_blocked:,} ({tot_blocked/tot_true:.2%})")
    print(f"  lost to blocking       {tot_true-tot_blocked:,} "
          f"({1-tot_blocked/tot_true:.2%})  <- no threshold can recover these")

    data = pd.concat([p[0] for p in all_parts], ignore_index=True)
    above = data.score.to_numpy() >= thr
    tp = int((above & (data.y.to_numpy() == 1)).sum())
    fp = int((above & (data.y.to_numpy() == 0)).sum())
    fn = int((~above & (data.y.to_numpy() == 1)).sum())
    print(f"  of those reaching it: tp {tp:,}  fn {fn:,} (scored below "
          f"{thr:.2f})  fp {fp:,}")

    print("\n=== blocking misses by candidate noise type ===")
    miss = np.concatenate([m for p in all_parts for m in p[3]]) \
        if any(p[3] for p in all_parts) else np.array([], dtype=object)
    if len(miss):
        mc = pd.Series(miss).value_counts()
        base = pd.Series(data.noise).value_counts()
        tab = pd.DataFrame({"missed_by_blocking": mc}).join(
            pd.DataFrame({"scored_pairs": base}), how="outer").fillna(0)
        tab["share_of_misses"] = tab.missed_by_blocking / max(len(miss), 1)
        print(tab.sort_values("missed_by_blocking", ascending=False).to_string(
            float_format=lambda v: f"{v:,.3f}"))

    print("\n=== false negatives by candidate noise type ===")
    fn_mask = (~above) & (data.y.to_numpy() == 1)
    if fn_mask.any():
        print(pd.Series(data.noise[fn_mask]).value_counts().to_string())
    print("\n=== false positives by candidate noise type ===")
    fp_mask = above & (data.y.to_numpy() == 0)
    if fp_mask.any():
        print(pd.Series(data.noise[fp_mask]).value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
