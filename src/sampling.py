"""Deterministic entity sampling, shared by blocking and training.

Training samples Source-1 *entities*, never pairs: an entity's whole candidate
set has to stay together for the per-entity rank features and the macro metric to
mean anything. Blocking uses the same function to skip unsampled entities
entirely on the training split, which is pure waste otherwise.
"""
from __future__ import annotations

import numpy as np

# Reverse blocking produces ~14 candidates per entity instead of ~84, so a much
# larger entity sample fits in the same memory: 30% of entities is ~9M pairs at
# 45 float32 features, about 1.6 GB.
FIT_PCT = 30       # percent of entities used to fit the classifier
VAL_PCT = 8        # percent used to tune the threshold and measure
SAMPLE_PCT = FIT_PCT + VAL_PCT


def bucket(s1_idx: np.ndarray) -> np.ndarray:
    """Stable 0-99 bucket per Source-1 row index (Knuth multiplicative hash)."""
    h = (np.asarray(s1_idx).astype(np.uint32) * np.uint32(2654435761)) >> np.uint32(13)
    return (h % np.uint32(100)).astype(np.int32)


def sampled_mask(s1_idx: np.ndarray) -> np.ndarray:
    return bucket(s1_idx) < SAMPLE_PCT
