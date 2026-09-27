"""Ground-truth lookup helpers (training split only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .paths import WORK, norm_path


def id_positions(split: str, source: int, country: str) -> dict[str, int]:
    ids = pd.read_parquet(norm_path(split, source, country), columns=["entity_id"])
    return {e: i for i, e in enumerate(ids.entity_id)}


def truth_pairs(country: str, source: int, s1_pos: dict, cand_pos: dict):
    """True ``(s1_idx, cand_idx)`` for one country and candidate source."""
    gt = pd.read_parquet(f"{WORK}/train_ground_truth.parquet")
    gt = gt[(gt.matched_entity_ids.str.len() > 0)
            & gt.source1_entity_id.isin(s1_pos.keys())]
    ex = gt.assign(mid=gt.matched_entity_ids.str.split(",")).explode("mid")
    ex = ex[ex.mid.str.startswith(f"S{source}-")]
    s1 = ex.source1_entity_id.map(s1_pos).to_numpy()
    cd = ex.mid.map(cand_pos).to_numpy()
    ok = ~pd.isna(cd)
    return s1[ok].astype(np.int64), cd[ok].astype(np.int64)


def truth_counts(country: str, s1_pos: dict) -> np.ndarray:
    """Number of true matches per S1 row index, including zeros for singletons."""
    gt = pd.read_parquet(f"{WORK}/train_ground_truth.parquet")
    gt = gt[gt.source1_entity_id.isin(s1_pos.keys())]
    idx = gt.source1_entity_id.map(s1_pos).to_numpy().astype(np.int64)
    n = np.where(gt.matched_entity_ids.str.len() > 0,
                 gt.matched_entity_ids.str.count(",") + 1, 0).astype(np.int64)
    out = np.zeros(len(s1_pos), dtype=np.int64)
    out[idx] = n
    return out


def label_pairs(pairs: pd.DataFrame, truth: dict[int, tuple], n_cand: dict[int, int]
                ) -> np.ndarray:
    """1 for pairs present in the ground truth, 0 otherwise."""
    y = np.zeros(len(pairs), dtype=np.int8)
    src = pairs.source.to_numpy()
    s1 = pairs.s1_idx.to_numpy().astype(np.int64)
    cd = pairs.cand_idx.to_numpy().astype(np.int64)
    for source, (t_s1, t_cd) in truth.items():
        m = src == source
        if not m.any():
            continue
        want = s1[m] * n_cand[source] + cd[m]
        have = np.sort(t_s1 * n_cand[source] + t_cd)
        pos = np.searchsorted(have, want)
        hit = (pos < len(have)) & (have[np.minimum(pos, len(have) - 1)] == want)
        y[np.flatnonzero(m)] = hit.astype(np.int8)
    return y
