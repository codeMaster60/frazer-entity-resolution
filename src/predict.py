"""Stage 5 — score the test candidates and write the submission files.

Runs one (country, source) at a time and keeps only per-entity joined strings
between stages, so peak memory never holds both candidate tables at once. The
global exclusivity pass runs per country over the surviving pairs of both
sources together, since a candidate record may only belong to one entity.

By default candidates are generated in-process and never written to disk: at ~82
candidates per entity the test candidate files are over a gigabyte of pure
intermediate. Pass --from-disk to read candidate files written by src.blocking
instead. Output rows are appended per country as each one finishes, so the whole
submission is never held in memory at once.

    python -m src.predict
    python -m src.predict --from-disk
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

from .blocking_rev import RevParams, block_country
from .decide import Rule, apply_rule
from .features import FEATURE_NAMES, Side, iter_shards, load_pairs, pair_features
from .metric import exclusive
from .stage2 import stage2_features
from .paths import (OUTPUT, WORK, cand_path, countries, ensure_dirs,
                    norm_path)

SHARD_ENTITIES = 60_000


def join_groups(s1_sorted: np.ndarray, ids: np.ndarray):
    """Comma-join ``ids`` per run of equal ``s1_sorted`` values."""
    if len(s1_sorted) == 0:
        return np.empty(0, np.int64), []
    edges = np.flatnonzero(np.diff(s1_sorted)) + 1
    keys = s1_sorted[np.concatenate([[0], edges])]
    return keys, [",".join(g) for g in np.split(ids, edges)]


def per_entity_strings(s1_idx: np.ndarray, ids: np.ndarray, n_s1: int) -> np.ndarray:
    """Dense array of joined id lists, empty string where an entity has none."""
    out = np.full(n_s1, "", dtype=object)
    order = np.argsort(s1_idx, kind="stable")
    keys, joined = join_groups(s1_idx[order], ids[order])
    if len(keys):
        out[keys] = joined
    return out


def score_pairs(models: dict, s1_side: Side, cand_side: Side,
                pairs: pd.DataFrame, source: int, conf: float) -> np.ndarray:
    """Stage-1 score, refined by stage 2 when a stage-2 model is present.

    Safe to call per shard: shards are cut on Source-1 entity boundaries, so an
    entity's whole candidate set — its stage-2 "family" — always lies inside one
    shard and is never split.
    """
    X = pair_features(s1_side, cand_side, pairs, source)
    # Features are only ever appended to FEATURE_NAMES, so the leading columns
    # still match an older model's training set exactly and can be sliced. Without
    # this, an older model raises "number of features in data (45) is not the same
    # as it was in training data (38)".
    want = models["stage1"].num_feature()
    Xs = X[:, :want] if X.shape[1] > want else X
    s1_score = models["stage1"].predict(Xs, num_threads=0).astype(np.float32)
    if models.get("stage2") is None:
        return s1_score
    n_s1 = len(s1_side.entity_id)
    S2 = stage2_features(pairs[["s1_idx", "cand_idx"]], s1_score, X,
                         s1_side, cand_side, n_s1, conf)
    XS = np.hstack([X, s1_score.reshape(-1, 1), S2])
    del X, S2, Xs
    return models["stage2"].predict(XS, num_threads=0).astype(np.float32)


def score_country(models: dict, rule: Rule, split: str, country: str,
                  from_disk: bool, conf: float = 0.80,
                  save_candidates: bool = False):
    """Return (candidate strings, matched strings, s1 entity ids) for a country."""
    s1_side = Side.load(split, 1, country)
    n_s1 = len(s1_side.entity_id)
    cand_strings, survivors = [], []

    for source in (2, 3):
        t0 = time.time()
        cand_side = Side.load(split, source, country)
        cached = os.path.exists(cand_path(split, country, source))
        if from_disk or (save_candidates and cached):
            # Retrieval is the expensive stage, so an existing candidate file is
            # always reused; this makes an interrupted run resumable per source.
            pairs = load_pairs(split, country, source)
            print(f"    S{source}: reusing {len(pairs):,} cached candidates",
                  flush=True)
        else:
            pairs = block_country(split, country, source, RevParams(),
                                  keep_sampled_s1=False)
            if save_candidates:
                # ~300 MB for the whole test set, and it saves re-running the
                # 2.5-hour retrieval for every later model that reuses this
                # candidate set.
                pairs.to_parquet(cand_path(split, country, source),
                                 compression="zstd", index=False)
        cand_ids = cand_side.entity_id

        kept_s1, kept_idx, kept_score = [], [], []
        for shard in iter_shards(pairs, n_s1, SHARD_ENTITIES):
            p = score_pairs(models, s1_side, cand_side, shard, source, conf)
            keep = apply_rule(shard.s1_idx.to_numpy(), p, rule)
            if keep.any():
                kept_s1.append(shard.s1_idx.to_numpy()[keep])
                kept_idx.append(shard.cand_idx.to_numpy()[keep])
                kept_score.append(p[keep])
            del p

        cand_strings.append(per_entity_strings(
            pairs.s1_idx.to_numpy(), cand_ids[pairs.cand_idx.to_numpy()], n_s1))
        n_pairs = len(pairs)
        del pairs

        if kept_s1:
            k_s1 = np.concatenate(kept_s1)
            k_idx = np.concatenate(kept_idx)
            survivors.append(pd.DataFrame({
                "s1_idx": k_s1,
                "source": np.int8(source),
                "cand_idx": k_idx,
                "cand_id": cand_ids[k_idx],
                "score": np.concatenate(kept_score),
            }))
            kept = len(k_s1)
        else:
            kept = 0
        print(f"    S{source}: {n_pairs:,} candidates -> {kept:,} above threshold "
              f"({time.time() - t0:.0f}s)", flush=True)
        del cand_side, cand_ids, kept_s1, kept_idx, kept_score

    # Joined per element, not with np.char: casting an object array of long
    # comma-joined lists to a fixed-width unicode dtype would allocate
    # 4 bytes * longest_list * n_entities, gigabytes for no reason.
    left, right = cand_strings
    candidates = np.array(
        [a if not b else b if not a else f"{a},{b}" for a, b in zip(left, right)],
        dtype=object)

    if survivors:
        surv = pd.concat(survivors, ignore_index=True)
        mask = exclusive(surv, surv.score.to_numpy())
        dropped = int((~mask).sum())
        surv = surv[mask]
        matched = per_entity_strings(surv.s1_idx.to_numpy(),
                                    surv.cand_id.to_numpy(), n_s1)
        print(f"    exclusivity pass dropped {dropped:,} contested pairs; "
              f"{len(surv):,} matches kept", flush=True)
    else:
        matched = np.full(n_s1, "", dtype=object)

    return candidates, matched, s1_side.entity_id


class RowWriter:
    """Appends id/value rows to a TSV, header first.

    With ``append`` an existing file is extended and no header is written, so a
    run interrupted part-way through the countries can be resumed without
    rescoring the countries already written.
    """

    def __init__(self, path: str, header: str, append: bool = False):
        exists = append and os.path.exists(path) and os.path.getsize(path) > 0
        self.fh = open(path, "a" if exists else "w", encoding="utf-8")
        if not exists:
            self.fh.write(header + "\n")

    def write(self, ids, values) -> None:
        self.fh.writelines(f"{i}\t{v}\n" for i, v in zip(ids, values))
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--country", default=None,
                    help="comma-separated countries to score instead of all "
                         "(partial output unless combined with --append)")
    ap.add_argument("--v2-model", action="store_true",
                    help="force the single-stage v2 model")
    ap.add_argument("--no-stage2", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="write the two TSVs here instead of output/, so a run "
                         "cannot damage an already-validated submission")
    ap.add_argument("--append", action="store_true",
                    help="extend existing output files instead of rewriting "
                         "them, to resume an interrupted run")
    ap.add_argument("--save-candidates", action="store_true",
                    help="persist the generated candidates so a later run can "
                         "reuse them with --from-disk")
    ap.add_argument("--from-disk", action="store_true",
                    help="read candidate files from src.blocking instead of "
                         "generating candidates in-process")
    args = ap.parse_args()
    ensure_dirs()

    # Prefer the two-stage v3 models when they exist, else the single v2 model.
    models: dict = {"stage2": None}
    if os.path.exists(f"{WORK}/model_s1.txt") and not args.v2_model:
        models["stage1"] = lgb.Booster(model_file=f"{WORK}/model_s1.txt")
        if os.path.exists(f"{WORK}/model_s2.txt") and not args.no_stage2:
            models["stage2"] = lgb.Booster(model_file=f"{WORK}/model_s2.txt")
        cfg = json.load(open(f"{WORK}/v3_results.json"))
        best = cfg.get("stage2" if models["stage2"] is not None else "stage1")
        rule = Rule(threshold=args.threshold if args.threshold is not None
                    else best["threshold"], rel_floor=best["rel_floor"])
        print(f"  v3 models, {rule.label()} (val macro F0.5 {best['f05']:.4f})",
              flush=True)
    else:
        models["stage1"] = lgb.Booster(model_file=f"{WORK}/model.txt")
        tuned = json.load(open(f"{WORK}/threshold.json"))
        rule = Rule(threshold=args.threshold if args.threshold is not None
                    else tuned["threshold"])
        print(f"  v2 model, {rule.label()} "
              f"(val macro F0.5 {tuned['val_f05']:.4f})", flush=True)

    out_dir = args.out_dir or OUTPUT
    os.makedirs(out_dir, exist_ok=True)
    matches = RowWriter(f"{out_dir}/matching_results.tsv",
                        "source1_entity_id\tmatched_entity_ids", args.append)
    cands = RowWriter(f"{out_dir}/candidate_pairs.tsv",
                      "source1_entity_id\tcandidate_entity_ids", args.append)
    total = nonempty = 0
    try:
        todo = (args.country.split(",") if args.country
                else countries(args.split))
        for country in todo:
            t0 = time.time()
            print(f"  {args.split.upper()} {country}", flush=True)
            candidates, matched, ids = score_country(
                models, rule, args.split, country, args.from_disk,
                save_candidates=args.save_candidates)
            matches.write(ids, matched)
            cands.write(ids, candidates)
            total += len(ids)
            nonempty += int((matched != "").sum())
            print(f"    {len(ids):,} rows written ({time.time() - t0:.0f}s total)",
                  flush=True)
            del candidates, matched, ids
    finally:
        matches.close()
        cands.close()

    print(f"\n  wrote {total:,} rows  ({nonempty:,} with matches, "
          f"{total - nonempty:,} predicted singletons)")
    print(f"  {out_dir}/matching_results.tsv\n  {out_dir}/candidate_pairs.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
