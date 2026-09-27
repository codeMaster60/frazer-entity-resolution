"""Cache stage-1(+stage-2) scores for every test candidate pair.

Runs the scoring half of ``src.predict`` (blocking is already cached in
``work/cand_test_*.parquet``) but keeps every pair, unfiltered, so
``src.rethreshold`` can build threshold/rel_floor/cap variants afterwards with
no rescoring. One parquet per country: ``s1_idx``, ``source``, ``cand_id``,
``score``.

    python -m src.save_test_scores
    python -m src.save_test_scores --country France
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import Side, iter_shards, load_pairs, pair_features
from .paths import WORK, countries, ensure_dirs
from .stage2 import stage2_features

SHARD_ENTITIES = 60_000


def score_country(models: dict, country: str, conf: float = 0.80) -> pd.DataFrame:
    s1_side = Side.load("test", 1, country)
    n_s1 = len(s1_side.entity_id)
    parts = []
    for source in (2, 3):
        t0 = time.time()
        cand_side = Side.load("test", source, country)
        pairs = load_pairs("test", country, source)
        cand_ids = cand_side.entity_id
        scores = np.empty(len(pairs), dtype=np.float32)
        for shard in iter_shards(pairs, n_s1, SHARD_ENTITIES):
            X = pair_features(s1_side, cand_side, shard, source)
            want = models["stage1"].num_feature()
            Xs = X[:, :want] if X.shape[1] > want else X
            s1_score = models["stage1"].predict(Xs, num_threads=0).astype(np.float32)
            if models.get("stage2") is not None:
                S2 = stage2_features(shard[["s1_idx", "cand_idx"]], s1_score, X,
                                     s1_side, cand_side, n_s1, conf)
                XS = np.hstack([X, s1_score.reshape(-1, 1), S2])
                sc = models["stage2"].predict(XS, num_threads=0).astype(np.float32)
                del XS, S2
            else:
                sc = s1_score
            scores[shard.index.to_numpy()] = sc
            del X, Xs, sc
        parts.append(pd.DataFrame({
            "s1_idx": pairs.s1_idx.to_numpy(),
            "source": np.int8(source),
            "cand_id": cand_ids[pairs.cand_idx.to_numpy()],
            "score": scores,
        }))
        print(f"    S{source}: {len(pairs):,} pairs scored "
              f"({time.time() - t0:.0f}s)", flush=True)
        del cand_side, pairs, scores
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default=None,
                    help="comma-separated countries instead of all")
    ap.add_argument("--conf", type=float, default=0.80)
    args = ap.parse_args()
    ensure_dirs()

    models: dict = {"stage1": lgb.Booster(model_file=f"{WORK}/model_s1.txt"),
                    "stage2": None}
    if os.path.exists(f"{WORK}/model_s2.txt"):
        models["stage2"] = lgb.Booster(model_file=f"{WORK}/model_s2.txt")
    print(f"  stage1{'+stage2' if models['stage2'] is not None else ''} models",
          flush=True)

    todo = args.country.split(",") if args.country else countries("test")
    for country in todo:
        out_path = f"{WORK}/test_scores_{country.replace(' ', '_')}.parquet"
        if os.path.exists(out_path):
            print(f"  {country}: already cached at {out_path}, skipping",
                  flush=True)
            continue
        t0 = time.time()
        print(f"  TEST {country}", flush=True)
        df = score_country(models, country, args.conf)
        df.to_parquet(out_path, compression="zstd", index=False)
        print(f"    wrote {len(df):,} scored pairs -> {out_path} "
              f"({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
