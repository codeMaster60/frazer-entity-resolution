"""Stage 2b — blocking recall report (train only; needs ground truth).

Blocking recall is the hard ceiling on the final score, so it is measured
before any model is trained.

    python -m src.block_eval
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .paths import WORK, cand_path, countries, norm_path
from .sampling import sampled_mask


def pair_keys(s1_idx: np.ndarray, cand_idx: np.ndarray, n_cand: int) -> np.ndarray:
    return s1_idx.astype(np.int64) * n_cand + cand_idx.astype(np.int64)


def truth_for_country(country: str, source: int, s1_pos: dict, cand_pos: dict,
                      keep_s1: np.ndarray | None = None):
    """True (s1_idx, cand_idx) pairs restricted to one country and source.

    ``keep_s1`` restricts the truth to the entities that were actually blocked.
    Without it, scoring a sample-only candidate file against every true pair in
    the country reports the sample fraction, not the recall.
    """
    gt = pd.read_parquet(f"{WORK}/train_ground_truth.parquet")
    gt = gt[gt.matched_entity_ids.str.len() > 0]
    gt = gt[gt.source1_entity_id.isin(s1_pos.keys())]
    ex = gt.assign(mid=gt.matched_entity_ids.str.split(",")).explode("mid")
    ex = ex[ex.mid.str.startswith(f"S{source}-")]
    s1 = ex.source1_entity_id.map(s1_pos).to_numpy()
    cd = ex.mid.map(cand_pos).to_numpy()
    ok = ~pd.isna(cd)
    s1, cd = s1[ok].astype(np.int64), cd[ok].astype(np.int64)
    if keep_s1 is not None:
        m = keep_s1[s1]
        s1, cd = s1[m], cd[m]
    return s1, cd


def evaluate(country: str, sample_only: bool) -> None:
    s1_ids = pd.read_parquet(norm_path("train", 1, country), columns=["entity_id"])
    s1_pos = {e: i for i, e in enumerate(s1_ids.entity_id)}
    n_s1 = len(s1_pos)
    keep_s1 = sampled_mask(np.arange(n_s1)) if sample_only else None
    n_queried = int(keep_s1.sum()) if keep_s1 is not None else n_s1
    found_any = np.zeros(n_s1, dtype=bool)
    true_any = np.zeros(n_s1, dtype=bool)
    tot_true = tot_found = tot_pairs = 0

    for source in (2, 3):
        cand_ids = pd.read_parquet(norm_path("train", source, country),
                                   columns=["entity_id"])
        cand_pos = {e: i for i, e in enumerate(cand_ids.entity_id)}
        n_cand = len(cand_pos)
        del cand_ids

        pairs = pd.read_parquet(cand_path("train", country, source),
                                columns=["s1_idx", "cand_idx", "pass_id"])
        keys = np.sort(pair_keys(pairs.s1_idx.to_numpy(), pairs.cand_idx.to_numpy(),
                                n_cand))
        t_s1, t_cd = truth_for_country(country, source, s1_pos, cand_pos, keep_s1)
        t_keys = pair_keys(t_s1, t_cd, n_cand)
        hit = np.isin(t_keys, keys, assume_unique=False)

        print(f"    S{source}: recall {hit.mean():6.2%}  "
              f"({int(hit.sum()):,}/{len(hit):,} true pairs)  "
              f"{len(pairs) / n_queried:5.2f} cand/entity")
        tot_true += len(hit)
        tot_found += int(hit.sum())
        tot_pairs += len(pairs)
        np.logical_or.at(true_any, t_s1, True)
        np.logical_or.at(found_any, t_s1[hit], True)
        del pairs, keys, cand_pos

    print(f"    ALL: pair recall {tot_found / max(tot_true, 1):6.2%}   "
          f"{tot_pairs / n_queried:5.2f} cand/entity   "
          f"entities with >=1 match found {found_any.sum() / max(true_any.sum(), 1):6.2%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default=None)
    ap.add_argument("--all-entities", action="store_true",
                    help="candidate files cover every entity, not just the sample")
    args = ap.parse_args()
    for country in ([args.country] if args.country else countries("train")):
        print(f"  TRAIN {country}"
              + ("" if args.all_entities else "  [scoring the blocked sample]"))
        evaluate(country, sample_only=not args.all_entities)
    return 0


if __name__ == "__main__":
    sys.exit(main())
