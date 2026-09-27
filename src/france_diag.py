"""Per-country calibration diagnostics on a finished submission, plus the
normalization gaps behind France's under-matching.

France has no training labels, so nothing in validation can reveal that the model
is systematically more conservative there. The only available signal is whether
France's *output distribution* looks like the countries we can measure, and where
its unmatched records differ textually from their best candidate.

Uses the cached blocking scores rather than model inference, so it is cheap enough
to run beside a training job.

    python -m src.france_diag --examples 30
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

from .paths import (OUTPUT, READ_KW, cand_path, norm_path, parquet_path,
                    source_tsv)

TRAIN_SINGLETON_RATE = 0.0558
TRAIN_MEAN_MATCHES = 3.67

# French forms that normalization must handle; presence here is not a bug, absence
# of handling is.
FR_PATTERNS = {
    "rue": r"\brue\b", "avenue": r"\bavenue\b|\bav\b", "boulevard": r"\bboulevard\b|\bbd\b|\bbld\b",
    "chemin": r"\bchemin\b", "allee": r"\ball[eé]e\b", "impasse": r"\bimpasse\b",
    "place": r"\bplace\b", "quai": r"\bquai\b", "route": r"\broute\b",
    "bis/ter": r"\b(bis|ter|quater)\b", "cedex": r"\bcedex\b",
    "postal5": r"\b\d{5}\b", "apostrophe": r"[’']",
    "accents": r"[àâäçéèêëîïôöùûüÿœæ]",
    "SARL": r"\bsarl\b", "SAS": r"\bsasu?\b", "EURL": r"\beurl\b",
    "SA": r"\bsa\b", "SNC": r"\bsnc\b", "SCI": r"\bsci\b",
}


def country_of(split: str, source: int) -> pd.Series:
    return pd.read_parquet(parquet_path(split, source), columns=["country"])["country"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default=f"{OUTPUT}/matching_results.tsv")
    ap.add_argument("--examples", type=int, default=30)
    args = ap.parse_args()

    m = pd.read_csv(args.matching, **READ_KW)
    s1 = pd.read_parquet(parquet_path("test", 1), columns=["entity_id", "country"])
    m = m.merge(s1, left_on="source1_entity_id", right_on="entity_id", how="left")
    m["n"] = np.where(m.matched_entity_ids.str.len() > 0,
                      m.matched_entity_ids.str.count(",") + 1, 0)

    print("=== per-country output calibration ===")
    rows = []
    for country, g in m.groupby("country"):
        rows.append({
            "country": country, "entities": len(g),
            "singleton_rate": (g.n == 0).mean(),
            "mean_per_nonsingleton": g.n[g.n > 0].mean(),
            "matches_per_entity": g.n.mean(),
        })
    tab = pd.DataFrame(rows).sort_values("country")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ntrain truth: singleton_rate {TRAIN_SINGLETON_RATE:.4f}, "
          f"mean_per_nonsingleton {TRAIN_MEAN_MATCHES:.2f}")

    # share of S2/S3 records that received an assignment, per country
    print("\n=== share of Source-2/3 records assigned ===")
    used = set()
    for s in m.matched_entity_ids:
        if s:
            used.update(s.split(","))
    for source in (2, 3):
        ids = pd.read_parquet(parquet_path("test", source),
                              columns=["entity_id", "country"])
        ids["used"] = ids.entity_id.isin(used)
        out = ids.groupby("country").used.agg(["size", "mean"])
        out.columns = ["records", "share_assigned"]
        print(f"S{source}:")
        print(out.to_string(float_format=lambda v: f"{v:.4f}"))
        del ids

    # France: unmatched records vs their best blocking candidate
    print(f"\n=== {args.examples} unmatched France records vs best S1 candidate ===")
    matched_fr = {i for s in m.loc[m.country == "France", "matched_entity_ids"]
                  if s for i in s.split(",")}
    s1n = pd.read_parquet(norm_path("test", 1, "France"))
    raw1 = pd.read_csv(source_tsv("test", 1), **READ_KW)
    raw1 = raw1[raw1.country == "France"].reset_index(drop=True)

    gap_counts: Counter = Counter()
    shown = 0
    rng = np.random.default_rng(0)
    for source in (2, 3):
        cand = pd.read_parquet(cand_path("test", "France", source))
        nrm = pd.read_parquet(norm_path("test", source, "France"))
        raw = pd.read_csv(source_tsv("test", source), **READ_KW)
        raw = raw[raw.country == "France"].reset_index(drop=True)
        # best blocking candidate per noisy record
        best = cand.sort_values("block_score", ascending=False) \
                   .drop_duplicates("cand_idx", keep="first")
        best_ids = nrm.entity_id.to_numpy()[best.cand_idx.to_numpy()]
        unmatched = best[~pd.Index(best_ids).isin(matched_fr)]
        print(f"\n-- S{source}: {len(unmatched):,} of {len(nrm):,} records unmatched "
              f"({len(unmatched)/len(nrm):.1%})")
        pick = rng.choice(len(unmatched), min(args.examples // 2, len(unmatched)),
                          replace=False)
        for j in pick:
            r = unmatched.iloc[int(j)]
            ci, si = int(r.cand_idx), int(r.s1_idx)
            nx, n1 = raw.business_name.iloc[ci], raw1.business_name.iloc[si]
            ax, a1 = raw.business_address.iloc[ci], raw1.business_address.iloc[si]
            if shown < args.examples:
                print(f"\n  [{nrm.entity_id.iloc[ci]} -> {s1n.entity_id.iloc[si]}  "
                      f"block={r.block_score:.2f}]")
                print(f"    S1 raw : {n1!r} | {a1!r}")
                print(f"    Sx raw : {nx!r} | {ax!r}")
                print(f"    S1 norm: {s1n.name_core.iloc[si]!r} | "
                      f"{s1n.addr_tok.iloc[si]!r}")
                print(f"    Sx norm: {nrm.name_core.iloc[ci]!r} | "
                      f"{nrm.addr_tok.iloc[ci]!r}")
                shown += 1
        # pattern frequency in unmatched raw text
        blob = " ".join(raw.business_address.iloc[
            unmatched.cand_idx.to_numpy()[:20000]].tolist()).lower()
        nblob = " ".join(raw.business_name.iloc[
            unmatched.cand_idx.to_numpy()[:20000]].tolist()).lower()
        for label, pat in FR_PATTERNS.items():
            gap_counts[label] += len(re.findall(pat, blob + " " + nblob))
        del cand, nrm, raw, best

    print("\n=== French pattern frequency in unmatched France text (20k sample) ===")
    for label, c in gap_counts.most_common():
        print(f"  {label:<12} {c:>9,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
