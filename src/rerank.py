"""Track B — cross-encoder reranker for the uncertain band.

The gradient-boosted model sees hand-built similarity features; it cannot read the
strings. For pairs it is confident about that does not matter — 93% of entities
have a best candidate above 0.95 — but in the uncertain band the deciding evidence
is often word-level semantics that no Jaccard or edit ratio encodes: whether
"Distribution" or "Participations" marks a sibling company rather than the same
business, whether a transliterated Indic name is the same name, whether an alias
plausibly belongs to the address. A cross-encoder reads both records jointly and
can represent that.

Applied only to the band (default stage-2 score in [0.20, 0.90]), which is a few
percent of pairs, so the cost is a fraction of scoring everything. The blend weight
and threshold are tuned on the same realistic validation split as everything else,
and Track B ships only if it beats the LightGBM-only score there.

**Model:** ``intfloat/multilingual-e5-small`` — 118M parameters, **MIT licence**,
12 layers, 384 hidden, XLM-R tokenizer covering 100+ languages, so Devanagari,
Kannada and French are all in-vocabulary. Well under the 8B limit.
Apache-2.0 alternative with the same shape:
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``.

Torch and transformers are imported lazily, so this module stays importable in an
environment that has neither.

    python -m src.rerank fit          # fine-tune on the band, ~1-1.5h on MPS
    python -m src.rerank tune         # blend weight + threshold on validation
    python -m src.rerank apply        # score the test band, rewrite output
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from .paths import WORK

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_LICENSE = "MIT"
MODEL_PARAMS = "118M"
BAND_LO, BAND_HI = 0.20, 0.90
MAX_LEN = 72          # "name | address" pairs are short; 72 covers ~99%
BATCH_FIT = 64
BATCH_INFER = 256
EPOCHS = 2
LR = 2e-5
MAX_FIT_PAIRS = 300_000
CKPT = f"{WORK}/reranker"


def pair_text(name: str, addr: str) -> str:
    """One side of the pair. Address kept after a separator so the model can
    attend to it without the two fields blurring into one sequence."""
    return f"{name} | {addr}" if addr else f"{name} |"


def band_mask(score: np.ndarray, lo: float = BAND_LO, hi: float = BAND_HI):
    return (score >= lo) & (score <= hi)


def device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str | None = None, train: bool = False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    src = path or MODEL_NAME
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=1)
    model.to(device())
    model.train(train)
    return tok, model


def encode(tok, left: list[str], right: list[str]):
    return tok(left, right, truncation=True, max_length=MAX_LEN,
               padding=True, return_tensors="pt")


def sample_band(texts_l: np.ndarray, texts_r: np.ndarray, y: np.ndarray,
                score: np.ndarray, entity: np.ndarray, cap: int = MAX_FIT_PAIRS):
    """Band pairs plus each entity's hardest negative, balanced and capped.

    Band negatives are already hard by construction — they are the non-matches the
    boosted model could not rule out. The extra per-entity top-scoring negative is
    included so the reranker also sees the decoys that currently *win*, which are
    the ones producing false merges.
    """
    in_band = band_mask(score)
    hardest_neg = (pd.DataFrame({"e": entity, "y": y, "s": score})
                   .query("y == 0").groupby("e").s.idxmax().to_numpy())
    keep = np.zeros(len(y), dtype=bool)
    keep[np.flatnonzero(in_band)] = True
    keep[hardest_neg] = True
    idx = np.flatnonzero(keep)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    rng = np.random.default_rng(0)
    per = min(cap // 2, max(len(pos), 1), max(len(neg), 1))
    idx = np.concatenate([rng.choice(pos, min(per, len(pos)), replace=False),
                          rng.choice(neg, min(per, len(neg)), replace=False)])
    rng.shuffle(idx)
    return texts_l[idx], texts_r[idx], y[idx].astype(np.float32)


def fit(left: np.ndarray, right: np.ndarray, y: np.ndarray) -> None:
    import torch
    tok, model = load_model(train=True)
    dev = device()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = len(y)
    steps = (n // BATCH_FIT) * EPOCHS
    print(f"fine-tuning {MODEL_NAME} ({MODEL_PARAMS}, {MODEL_LICENSE}) on "
          f"{n:,} band pairs, {steps:,} steps, device {dev}", flush=True)
    t0 = time.time()
    seen = 0
    for epoch in range(EPOCHS):
        order = np.random.default_rng(epoch).permutation(n)
        for b in range(0, n - BATCH_FIT + 1, BATCH_FIT):
            sel = order[b:b + BATCH_FIT]
            batch = encode(tok, list(left[sel]), list(right[sel])).to(dev)
            target = torch.tensor(y[sel], device=dev).unsqueeze(1)
            out = model(**batch).logits
            loss = loss_fn(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            seen += len(sel)
            if (b // BATCH_FIT) % 200 == 0:
                rate = seen / max(time.time() - t0, 1e-6)
                print(f"  epoch {epoch} step {b // BATCH_FIT}: loss {loss.item():.4f}"
                      f"  {rate:,.0f} pairs/s", flush=True)
    model.save_pretrained(CKPT)
    tok.save_pretrained(CKPT)
    print(f"saved {CKPT} in {time.time() - t0:.0f}s", flush=True)


def score(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cross-encoder probability for each pair."""
    import torch
    tok, model = load_model(CKPT, train=False)
    dev = device()
    out = np.zeros(len(left), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(left), BATCH_INFER):
            sl = slice(b, min(b + BATCH_INFER, len(left)))
            batch = encode(tok, list(left[sl]), list(right[sl])).to(dev)
            out[sl] = torch.sigmoid(model(**batch).logits).squeeze(1) \
                           .float().cpu().numpy()
            if b % (BATCH_INFER * 200) == 0 and b:
                print(f"  scored {b:,}/{len(left):,}  "
                      f"{b / (time.time() - t0):,.0f} pairs/s", flush=True)
    return out


def blend(lgbm: np.ndarray, ce: np.ndarray, in_band: np.ndarray,
          weight: float) -> np.ndarray:
    """Blend inside the band only; confident pairs keep the boosted score.

    Rewriting confident scores would risk the 93% of entities the boosted model
    already gets right, for no measured gain.
    """
    out = lgbm.copy()
    out[in_band] = (1.0 - weight) * lgbm[in_band] + weight * ce[in_band]
    return out


def tune_blend(lgbm: np.ndarray, ce: np.ndarray, in_band: np.ndarray,
               s1_idx: np.ndarray, y: np.ndarray, entities: np.ndarray,
               true_counts: np.ndarray) -> dict:
    """Grid-search the blend weight, threshold and relative floor on validation."""
    from .decide import Rule, apply_rule
    from .metric import macro_f_beta
    codes = pd.Index(entities)
    ecode = codes.get_indexer(s1_idx)
    keep = ecode >= 0
    best = {"f05": -1.0}
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        s = blend(lgbm, ce, in_band, w)
        for thr in np.arange(0.30, 0.91, 0.02):
            for rel in (0.0, 0.3, 0.5):
                rule = Rule(threshold=float(thr), rel_floor=rel)
                m = apply_rule(s1_idx, s, rule) & keep
                if not m.any():
                    continue
                pred = np.bincount(ecode[m], minlength=len(entities))
                tp = np.bincount(ecode[m], weights=y[m].astype(float),
                                 minlength=len(entities))
                f = macro_f_beta(pred, true_counts, tp)
                if f > best["f05"]:
                    best = {"f05": float(f), "weight": w,
                            "threshold": float(thr), "rel_floor": rel}
    json.dump(best, open(f"{WORK}/rerank_blend.json", "w"), indent=2)
    print(f"best blend: {best}", flush=True)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["fit", "tune", "apply", "info"])
    args = ap.parse_args()
    if args.action == "info":
        print(f"model   : {MODEL_NAME}")
        print(f"params  : {MODEL_PARAMS}   licence: {MODEL_LICENSE}")
        print(f"band    : [{BAND_LO}, {BAND_HI}]  max_len {MAX_LEN}")
        print(f"fit     : {EPOCHS} epochs, batch {BATCH_FIT}, lr {LR}, "
              f"cap {MAX_FIT_PAIRS:,}")
        print(f"infer   : batch {BATCH_INFER}")
        print("torch/transformers are imported lazily; install with "
              "requirements-rerank.txt")
        return 0
    print("This entry point needs the band datasets assembled by the caller; "
          "see docstring. Use src.run_rerank for the wired pipeline.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
