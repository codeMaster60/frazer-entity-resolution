"""Dump v3 validation errors (false negatives/positives) for manual review.

Reuses the exact VAL_PCT-bucket validation entities, models, and decision rule
as ``src.eval_models``, but at pair level with raw + normalized text attached,
restricted to "clean-ish" records (no Indic/non-ASCII/URL name flags, non-empty
address on both sides) so examples are readable by eye.

    python -m src.error_analysis
"""
from __future__ import annotations

import json
import random

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decide import Rule, apply_rule
from .features import Side, load_pairs, pair_features
from .metric import exclusive
from .paths import WORK
from .sampling import FIT_PCT, VAL_PCT, bucket
from .stage2 import stage2_features
from .truth import id_positions, label_pairs, truth_pairs

SEED = 0
N_SAMPLE = 40
COUNTRIES = ("US", "India")


def load_raw_by_country(source: int) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(f"{WORK}/train_source{source}.parquet",
                         columns=["entity_id", "business_name",
                                  "business_address", "country"])
    out = {}
    for country in COUNTRIES:
        sub = df[df.country == country][
            ["entity_id", "business_name", "business_address"]]
        out[country] = sub.set_index("entity_id")
    del df
    return out


def main() -> int:
    cfg = json.load(open(f"{WORK}/v3_results.json"))
    best = cfg["stage2"]
    rule = Rule(threshold=best["threshold"], rel_floor=best["rel_floor"])
    models = {"stage1": lgb.Booster(model_file=f"{WORK}/model_s1.txt"),
              "stage2": lgb.Booster(model_file=f"{WORK}/model_s2.txt")}
    print(f"rule: {rule.label()}")

    raw2 = load_raw_by_country(2)
    raw3 = load_raw_by_country(3)
    raw_by_source = {2: raw2, 3: raw3}

    rng = random.Random(SEED)
    fn_never = fn_below = fn_taken = 0
    fn_pool, fp_pool = [], []

    for country in COUNTRIES:
        print(f"  {country}", flush=True)
        s1_pos = id_positions("train", 1, country)
        n_s1 = len(s1_pos)
        s1_side = Side.load("train", 1, country)
        s1_norm = pd.read_parquet(f"{WORK}/norm_train_s1_{country}.parquet")
        s1_raw = load_raw_by_country(1)[country]
        val_ent = (bucket(np.arange(n_s1)) >= FIT_PCT) & \
            (bucket(np.arange(n_s1)) < FIT_PCT + VAL_PCT)

        country_pairs = []
        cand_ctx = {}
        for source in (2, 3):
            pairs = load_pairs("train", country, source)
            b = bucket(pairs.s1_idx.to_numpy())
            in_val = (b >= FIT_PCT) & (b < FIT_PCT + VAL_PCT)
            pairs = pairs[in_val].reset_index(drop=True)
            cand_pos = id_positions("train", source, country)
            n_cand = len(cand_pos)
            t_s1_all, t_cd_all = truth_pairs(country, source, s1_pos, cand_pos)
            pairs["source"] = np.int8(source)
            y = label_pairs(pairs, {source: (t_s1_all, t_cd_all)}, {source: n_cand})
            cand_side = Side.load("train", source, country)
            X = pair_features(s1_side, cand_side, pairs, source)
            want = models["stage1"].num_feature()
            s1_score = models["stage1"].predict(X[:, :want]).astype(np.float32)
            S2 = stage2_features(pairs[["s1_idx", "cand_idx"]], s1_score, X,
                                 s1_side, cand_side, n_s1, 0.80)
            XS = np.hstack([X, s1_score.reshape(-1, 1), S2])
            score = models["stage2"].predict(XS).astype(np.float32)
            pairs["y"] = y
            pairs["score"] = score
            country_pairs.append(pairs)

            # true pairs restricted to VAL entities, for "never a candidate" count
            keep_t = val_ent[t_s1_all]
            t_s1v, t_cdv = t_s1_all[keep_t], t_cd_all[keep_t]
            cand_set = set(zip(pairs.s1_idx.tolist(), pairs.cand_idx.tolist()))
            never = [(s1i, ci) for s1i, ci in zip(t_s1v, t_cdv)
                     if (s1i, ci) not in cand_set]
            fn_never += len(never)

            cand_norm = pd.read_parquet(f"{WORK}/norm_train_s{source}_{country}.parquet")
            cand_raw = raw_by_source[source][country]
            true_owner = {ci: s1i for s1i, ci in zip(t_s1_all, t_cd_all)}
            cand_pos_inv = {i: e for e, i in cand_pos.items()}
            cand_ctx[source] = dict(norm=cand_norm, raw=cand_raw,
                                    true_owner=true_owner, inv=cand_pos_inv,
                                    never=never)
            del X, S2, XS, s1_score, cand_side

        allp = pd.concat(country_pairs, ignore_index=True)
        allp["rank"] = allp.groupby("s1_idx")["score"] \
            .rank(ascending=False, method="first").astype(np.int32)

        keep_mask = apply_rule(allp.s1_idx.to_numpy(), allp.score.to_numpy(), rule)
        predicted = np.zeros(len(allp), dtype=bool)
        if keep_mask.any():
            kept = allp[keep_mask]
            ex_mask = exclusive(kept, kept.score.to_numpy())
            predicted[np.flatnonzero(keep_mask)[ex_mask]] = True
        allp["predicted"] = predicted

        # winner: (source, cand_idx) -> (s1_idx, score) among predicted
        winner = {}
        for row in allp[allp.predicted].itertuples():
            winner[(row.source, row.cand_idx)] = (row.s1_idx, row.score)

        s1_flags = s1_norm.name_flags.to_numpy()
        s1_addr = s1_norm.addr_tok.to_numpy()
        s1_pos_inv = {i: e for e, i in s1_pos.items()}

        def clean_s1(s1i):
            return s1_flags[s1i] == 0 and bool(s1_addr[s1i])

        def clean_cand(source, ci):
            ctx = cand_ctx[source]
            row = ctx["norm"].iloc[ci]
            return row.name_flags == 0 and bool(row.addr_tok)

        # Resolved immediately (not deferred): a closure defined once per
        # country inside a shared loop shares one cell per free variable
        # across iterations, so a later call would silently use the last
        # country's data instead of the one it was built for.
        def s1_line(s1i):
            eid = s1_pos_inv[s1i]
            r = s1_raw.loc[eid] if eid in s1_raw.index else None
            nr = s1_norm.iloc[s1i]
            return (f"{eid} | raw: {r.business_name!r} / {r.business_address!r} | "
                    f"norm: {nr.name_norm!r} / {nr.addr_tok!r}") if r is not None \
                else f"{eid} | (raw not found)"

        def cand_line(source, ci):
            ctx = cand_ctx[source]
            eid = ctx["inv"][ci]
            raw = ctx["raw"]
            r = raw.loc[eid] if eid in raw.index else None
            nr = ctx["norm"].iloc[ci]
            return (f"{eid} | raw: {r.business_name!r} / {r.business_address!r} | "
                    f"norm: {nr.name_norm!r} / {nr.addr_tok!r}") if r is not None \
                else f"{eid} | (raw not found)"

        # False negatives present in the candidate set
        fn_mask = (allp.y == 1) & (~allp.predicted)
        for row in allp[fn_mask].itertuples():
            s1i, source, ci = row.s1_idx, row.source, row.cand_idx
            if row.score < rule.threshold:
                reason = "below_threshold"
                fn_below += 1
            else:
                reason = "taken_by_another_s1"
                fn_taken += 1
            if clean_s1(s1i) and clean_cand(source, ci):
                w = winner.get((source, ci))
                went = (f"{s1_line(w[0])} (score={w[1]:.4f})"
                       if w is not None else None)
                fn_pool.append((country, source, reason, row.score,
                               int(row.rank), s1_line(s1i),
                               cand_line(source, ci), went))

        # "Never a candidate" false negatives, clean-ish only
        for source in (2, 3):
            ctx = cand_ctx[source]
            for s1i, ci in ctx["never"]:
                if clean_s1(s1i) and clean_cand(source, ci):
                    fn_pool.append((country, source, "never_a_candidate",
                                   None, None, s1_line(s1i),
                                   cand_line(source, ci), None))

        # False positives
        fp_mask = allp.predicted & (allp.y == 0)
        for row in allp[fp_mask].itertuples():
            s1i, source, ci = row.s1_idx, row.source, row.cand_idx
            if clean_s1(s1i) and clean_cand(source, ci):
                true_s1 = cand_ctx[source]["true_owner"].get(ci)
                true_line = s1_line(true_s1) if true_s1 is not None else None
                fp_pool.append((country, source, row.score, int(row.rank),
                               s1_line(s1i), cand_line(source, ci), true_line))

        del allp, country_pairs, s1_side

    rng.shuffle(fn_pool)
    rng.shuffle(fp_pool)

    lines = []
    lines.append(f"rule: {rule.label()}")
    lines.append("")
    lines.append("=== FALSE NEGATIVE COUNTS (all validation pairs, not just sample) ===")
    lines.append(f"never_a_candidate: {fn_never}")
    lines.append(f"below_threshold:   {fn_below}")
    lines.append(f"taken_by_another_s1: {fn_taken}")
    lines.append(f"total FN: {fn_never + fn_below + fn_taken}")
    lines.append("")
    lines.append(f"=== {N_SAMPLE} SAMPLE FALSE NEGATIVES (clean-ish) ===")

    picked_fn = fn_pool[:N_SAMPLE]
    for country, source, reason, score, rank, s1_txt, cand_txt, went in picked_fn:
        score_str = f"{score:.4f}" if score is not None else "n/a"
        rank_str = rank if rank is not None else "n/a"
        lines.append(f"-- {country} S{source} reason={reason} score={score_str} "
                     f"rank={rank_str}")
        lines.append(f"   S1:   {s1_txt}")
        lines.append(f"   miss: {cand_txt}")
        if went is not None:
            lines.append(f"   went instead to: {went}")
        lines.append("")

    lines.append(f"=== {N_SAMPLE} SAMPLE FALSE POSITIVES (clean-ish) ===")
    picked_fp = fp_pool[:N_SAMPLE]
    for country, source, score, rank, s1_txt, cand_txt, true_line in picked_fp:
        lines.append(f"-- {country} S{source} score={score:.4f} rank={rank}")
        lines.append(f"   S1 (predicted): {s1_txt}")
        lines.append(f"   candidate:      {cand_txt}")
        if true_line is not None:
            lines.append(f"   true S1 instead: {true_line}")
        else:
            lines.append("   true S1: none (genuine non-match / decoy)")
        lines.append("")

    out_path = f"{WORK}/errors.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {len(picked_fn)} FN + {len(picked_fp)} FP samples -> {out_path}")
    print(f"FN pool sizes: never={fn_never} below={fn_below} taken={fn_taken}  "
          f"(clean-ish pool: {len(fn_pool)} FN, {len(fp_pool)} FP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
