"""Macro F0.5, the challenge metric, plus the global exclusivity pass."""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25   # beta = 0.5


def macro_f_beta(pred_counts: np.ndarray, true_counts: np.ndarray,
                 tp_counts: np.ndarray) -> float:
    """F0.5 per Source-1 entity, averaged over every entity.

    An entity with no true matches scores 1.0 when nothing is predicted for it
    and 0.0 otherwise, which is why singletons are worth chasing.
    """
    pred = pred_counts.astype(np.float64)
    true = true_counts.astype(np.float64)
    tp = tp_counts.astype(np.float64)

    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    denom = BETA2 * precision + recall
    f = np.divide((1 + BETA2) * precision * recall, denom,
                  out=np.zeros_like(tp), where=denom > 0)
    # Correctly predicted singleton: nothing predicted, nothing true.
    f[(pred == 0) & (true == 0)] = 1.0
    return float(f.mean())


def exclusive(pairs: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Keep each candidate record for its single best Source-1 entity.

    Training ground truth shows matched IDs are globally disjoint — 7,638,365
    matched IDs, all distinct — so a candidate claimed by two entities is
    necessarily one false merge, exactly what F0.5 punishes hardest.

    Returns a boolean mask over ``pairs``.
    """
    order = np.argsort(-score, kind="stable")
    key = (pairs.source.to_numpy().astype(np.int64) << 40) | \
        pairs.cand_idx.to_numpy().astype(np.int64)
    seen = pd.Index(key[order]).duplicated(keep="first")
    mask = np.zeros(len(pairs), dtype=bool)
    mask[order[~seen]] = True
    return mask
