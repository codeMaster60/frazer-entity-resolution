"""Rebuild a matching_results/candidate_pairs variant from cached test scores.

No rescoring: reads ``work/test_scores_<country>.parquet`` (written by
``src.save_test_scores``) and only re-applies the per-country decision rule
and the global exclusivity pass. Writes into ``work/variants/<name>/`` and
runs the official validator against it.

    python -m src.rethreshold --name france_thr60 \\
        --US thr=0.80 --India thr=0.74 --France thr=0.60

Rule spec is ``thr=<float>[,rel=<float>][,cap=<int>]``. Countries not passed
keep the current submission's values (US 0.80, India 0.74, France 0.74).
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

from .decide import Rule, apply_rule
from .features import Side
from .paths import ROOT, WORK, countries, ensure_dirs

DEFAULT = {"US": Rule(threshold=0.80), "India": Rule(threshold=0.74),
          "France": Rule(threshold=0.74)}


def parse_rule(spec: str) -> Rule:
    key_map = {"thr": "threshold", "rel": "rel_floor", "cap": "max_per_entity"}
    kw = {}
    for part in spec.split(","):
        k, v = part.split("=")
        kw[key_map[k]] = float(v)
    if "max_per_entity" in kw:
        kw["max_per_entity"] = int(kw["max_per_entity"])
    return Rule(**kw)


def per_entity_strings(s1_idx: np.ndarray, ids: np.ndarray, n_s1: int) -> np.ndarray:
    out = np.full(n_s1, "", dtype=object)
    if len(s1_idx) == 0:
        return out
    order = np.argsort(s1_idx, kind="stable")
    s1_sorted, ids_sorted = s1_idx[order], ids[order]
    edges = np.flatnonzero(np.diff(s1_sorted)) + 1
    keys = s1_sorted[np.concatenate([[0], edges])]
    joined = [",".join(g) for g in np.split(ids_sorted, edges)]
    out[keys] = joined
    return out


def exclusive_by_id(source: np.ndarray, cand_id: np.ndarray,
                    score: np.ndarray) -> np.ndarray:
    """Same rule as ``src.metric.exclusive``, keyed by (source, cand_id) string
    instead of a candidate-table row index, since only ids are cached here."""
    order = np.argsort(-score, kind="stable")
    key = (pd.Series(source, copy=False).astype(str) + "|"
           + pd.Series(cand_id, copy=False)).to_numpy()
    seen = pd.Index(key[order]).duplicated(keep="first")
    mask = np.zeros(len(score), dtype=bool)
    mask[order[~seen]] = True
    return mask


def build_country(country: str, rule: Rule):
    path = f"{WORK}/test_scores_{country.replace(' ', '_')}.parquet"
    pairs = pd.read_parquet(path)
    s1_ids = Side.load("test", 1, country).entity_id
    n_s1 = len(s1_ids)

    candidates = per_entity_strings(pairs.s1_idx.to_numpy(),
                                    pairs.cand_id.to_numpy(), n_s1)

    keep = apply_rule(pairs.s1_idx.to_numpy(), pairs.score.to_numpy(), rule)
    surv = pairs[keep]
    if len(surv):
        mask = exclusive_by_id(surv.source.to_numpy(), surv.cand_id.to_numpy(),
                               surv.score.to_numpy())
        surv = surv[mask]
        matched = per_entity_strings(surv.s1_idx.to_numpy(),
                                     surv.cand_id.to_numpy(), n_s1)
    else:
        matched = np.full(n_s1, "", dtype=object)
    return s1_ids, candidates, matched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="variant folder name under work/variants/")
    for c in ("US", "India", "France"):
        ap.add_argument(f"--{c}", default=None,
                        help=f"rule spec for {c}, e.g. thr=0.65,rel=0.3,cap=6")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    rules = dict(DEFAULT)
    for c in ("US", "India", "France"):
        spec = getattr(args, c)
        if spec:
            rules[c] = parse_rule(spec)

    out_dir = f"{WORK}/variants/{args.name}"
    import os
    os.makedirs(out_dir, exist_ok=True)
    m_path, c_path = f"{out_dir}/matching_results.tsv", f"{out_dir}/candidate_pairs.tsv"
    with open(m_path, "w", encoding="utf-8") as mf, \
         open(c_path, "w", encoding="utf-8") as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        total = nonempty = 0
        for country in countries("test"):
            rule = rules[country]
            ids, candidates, matched = build_country(country, rule)
            mf.writelines(f"{i}\t{v}\n" for i, v in zip(ids, matched))
            cf.writelines(f"{i}\t{v}\n" for i, v in zip(ids, candidates))
            total += len(ids)
            nonempty += int((matched != "").sum())
            print(f"  {country}: {rule.label()} -> "
                  f"{int((matched != '').sum()):,}/{len(ids):,} with matches",
                  flush=True)

    print(f"\n  wrote {total:,} rows ({nonempty:,} with matches) -> {out_dir}")

    if not args.no_validate:
        rc = subprocess.call([
            sys.executable, f"{ROOT}/utils/validate_submission.py",
            "--matching", m_path, "--candidate", c_path,
            "--test-dir", f"{ROOT}/dataset/test", "--check-ids",
        ])
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
