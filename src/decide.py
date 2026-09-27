"""Entity-level decision rules, tuned on validation.

A single global threshold ignores two things the metric cares about. First, F0.5
is macro-averaged per entity, so a weak extra match costs much more on an entity
that already has a strong match than on one that has none. Second, the countries
differ enough in blocking recall (US 97.6%, India 94.8%) that one threshold is
unlikely to be optimal for both.

Rules, each tunable and each measurable on its own:

* ``threshold`` — absolute score floor.
* ``rel_floor`` — drop a candidate scoring below this fraction of the entity's
  best candidate. Suppresses the weak tail on entities that already have a
  confident match, without touching entities whose best is itself marginal.
* ``max_per_entity`` — cap the list length; the truth never exceeds 11 and
  averages 3.7, so a long list is usually a precision leak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    threshold: float = 0.5
    rel_floor: float = 0.0        # 0 disables
    max_per_entity: int = 0       # 0 disables

    def label(self) -> str:
        return (f"thr={self.threshold:.2f} rel={self.rel_floor:.2f} "
                f"cap={self.max_per_entity or '-'}")


def apply_rule(s1_idx: np.ndarray, score: np.ndarray, rule: Rule) -> np.ndarray:
    """Boolean mask of kept pairs."""
    keep = score >= rule.threshold
    if not keep.any():
        return keep
    if rule.rel_floor > 0:
        best = pd.Series(score).groupby(s1_idx).transform("max").to_numpy()
        keep &= score >= rule.rel_floor * best
    if rule.max_per_entity > 0:
        order = np.lexsort((-score, s1_idx))
        ranked = np.empty(len(score), dtype=np.int64)
        starts = np.flatnonzero(np.diff(s1_idx[order], prepend=s1_idx[order][0] - 1))
        ranked[order] = np.arange(len(order)) - np.repeat(
            starts, np.diff(np.append(starts, len(order))))
        keep &= ranked < rule.max_per_entity
    return keep
