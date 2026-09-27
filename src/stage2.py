"""Stage-2 sibling features: use an entity's confident matches to judge its weak ones.

A Source-1 entity has ~3-4 noisy copies, and those copies resemble *each other*,
not just the clean original. That is leverage stage 1 cannot use, because it scores
each pair in isolation.

So: run stage 1, take each entity's high-confidence candidates as a provisional
"family", and describe every candidate by how well it fits that family — token
overlap with the family's pooled address and name, how many confident siblings
there are, and how the candidate's own best-vs-second-best S1 margin compares.

This targets exactly the two groups the miss diagnosis flagged. An alias-named
record (``Orbivio``) shares nothing with the clean name but sits at the same
address as its siblings; an empty-address record shares no address at all but its
name matches the family's names.

Family features are built from a *pooled* family vector rather than pairwise
sibling comparisons: an entity with c candidates would need c^2 comparisons, and
at ~14 candidates over millions of entities that is hundreds of millions of
string pairs, while the pooled form is one sparse product.
"""
from __future__ import annotations

import re
from zlib import crc32

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse import csr_matrix

from .features import COL, HASH_MASK, Side

STAGE2_NAMES = [
    "fam_size",          # confident siblings backing this entity
    "fam_addr_share",    # overlap of this candidate's address with the family's
    "fam_name_share",
    "fam_addr_idf",      # same, IDF-weighted, so rare agreement counts more
    "fam_is_member",     # was this candidate itself in the family
    "fam_score_max",     # strongest stage-1 score for this entity
    "fam_score_mean",
    "fam_score_gap",     # this candidate's score minus the entity's best
    "own_rev_margin",    # best-vs-second-best S1 margin for this record
    # Decoy detection: the error log showed near-miss false positives sharing an
    # S1's address but with exactly one specific name word swapped for a
    # different specific word (Delp/Doralynne's), vs. genuine noise where the
    # extra token is filler or a typo of the same word.
    "novel_s1_tok",       # S1 name-core tokens absent from the candidate
    "novel_cand_tok",     # candidate name-core tokens absent from S1
    "one_word_swap",      # exactly one novel token on each side, rest overlaps
    "novel_edit_sim",     # char-ratio between the two novel tokens (typo vs swap)
    "novel_is_common",    # the novel candidate token is frequent in S1's corpus
    "house_norm_eq",      # leading house number, zero/letter-stripped, equal
    "house_num_diff",     # |numeric difference| of the leading house numbers
    "house_trunc",        # one leading number is a digit prefix/suffix of the other
    "house_fam_match_s1", # share of siblings whose house number equals S1's own
    "house_fam_majority", # this candidate's number is the family's most common
    "house_fam_share",    # share of siblings (self excluded) sharing this number
    "novel_fam_decoy",    # this candidate's novel token recurs in a sibling too
]

_LEAD_DIGITS = re.compile(r"^0*(\d+)")


def _house_norm(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-record (leading number as int, or -1; its zero-stripped digit string)."""
    num = np.full(len(vals), -1, dtype=np.int64)
    dig = np.empty(len(vals), dtype=object)
    for i, s in enumerate(vals):
        dig[i] = ""
        if not s:
            continue
        m = _LEAD_DIGITS.match(s)
        if m:
            num[i] = int(m.group(1))
            dig[i] = m.group(1)
    return num, dig


def _name_diff_features(out: np.ndarray, c2: dict, base: np.ndarray,
                        s1: Side, cand: Side, li: np.ndarray,
                        ri: np.ndarray) -> np.ndarray:
    """Novel-token / decoy-word features; returns the per-pair novel cand token."""
    n1c, n2c = base[:, COL["name_n1"]], base[:, COL["name_n2"]]
    inter_c = base[:, COL["name_inter"]]
    novel_s1 = np.maximum(n1c - inter_c, 0.0)
    novel_cand = np.maximum(n2c - inter_c, 0.0)
    out[:, c2["novel_s1_tok"]] = novel_s1
    out[:, c2["novel_cand_tok"]] = novel_cand
    one_swap = (novel_s1 == 1) & (novel_cand == 1) & (inter_c >= 1)
    out[:, c2["one_word_swap"]] = one_swap.astype(np.float32)

    cand_novel_tok = np.full(len(li), "", dtype=object)
    idxs = np.flatnonzero(one_swap)
    if len(idxs):
        s1_tok, cd_tok = [], []
        for a, b in zip(s1.name_core[li[idxs]], cand.name_core[ri[idxs]]):
            sa, sb = set(a.split()), set(b.split())
            da, db = sa - sb, sb - sa
            s1_tok.append(next(iter(da)) if da else "")
            cd_tok.append(next(iter(db)) if db else "")
        sims = process.cpdist(np.array(s1_tok), np.array(cd_tok),
                              scorer=fuzz.ratio, workers=-1, dtype=np.float32)
        out[idxs, c2["novel_edit_sim"]] = sims

        df = np.asarray(s1.sets["name_core"].sum(axis=0)).ravel()
        common = np.zeros(len(idxs), dtype=np.float32)
        for k, tok in enumerate(cd_tok):
            cand_novel_tok[idxs[k]] = tok
            if tok:
                h = crc32(tok.encode()) & HASH_MASK
                common[k] = 1.0 if df[h] >= 5 else 0.0
        out[idxs, c2["novel_is_common"]] = common
    return cand_novel_tok


def _house_features(out: np.ndarray, c2: dict, s1: Side, cand: Side,
                    li: np.ndarray, ri: np.ndarray) -> None:
    s1_num, s1_dig = _house_norm(s1.first_num)
    cd_num, cd_dig = _house_norm(cand.first_num)
    h1n, h1s = s1_num[li], s1_dig[li]
    h2n, h2s = cd_num[ri], cd_dig[ri]
    has1, has2 = h1n >= 0, h2n >= 0
    both = has1 & has2
    out[:, c2["house_norm_eq"]] = np.where(both, (h1n == h2n).astype(np.float32), -1.0)
    out[:, c2["house_num_diff"]] = np.where(
        both, np.abs(h1n - h2n).astype(np.float32), -1.0)
    out[:, c2["house_trunc"]] = np.array([
        1.0 if (bn and a and b and a != b and (a in b or b in a)) else 0.0
        for bn, a, b in zip(both, h1s, h2s)], dtype=np.float32)

    # family consensus over the house number, within this (s1, source) shard
    eq_s1 = (h2n == h1n) & both
    fr = pd.DataFrame({"s1": li, "eq": eq_s1.astype(np.float32)})
    g = fr.groupby("s1", sort=False)["eq"]
    tot_eq, grp_n = g.transform("sum").to_numpy(), g.transform("size").to_numpy()
    others_n = np.maximum(grp_n - 1, 1)
    out[:, c2["house_fam_match_s1"]] = np.where(
        has1, (tot_eq - eq_s1.astype(np.float32)) / others_n, 0.0)

    sentinel = -1_000_000 - np.arange(len(li), dtype=np.int64)
    key = pd.DataFrame({"s1": li, "hnum": np.where(has2, h2n, sentinel)})
    cnt = key.groupby(["s1", "hnum"], sort=False)["hnum"].transform("size").to_numpy()
    grp_max = pd.DataFrame({"s1": li, "cnt": cnt}) \
        .groupby("s1")["cnt"].transform("max").to_numpy()
    out[:, c2["house_fam_majority"]] = ((cnt == grp_max) & has2).astype(np.float32)
    out[:, c2["house_fam_share"]] = np.where(
        has2, (cnt.astype(np.float32) - 1.0) / others_n, 0.0)


def _decoy_group_feature(out: np.ndarray, c2: dict, li: np.ndarray,
                         novel_tok: np.ndarray) -> None:
    has_novel = novel_tok != ""
    key = pd.DataFrame({
        "s1": li,
        "tok": np.where(has_novel, novel_tok, pd.NA),
    })
    cnt = key.groupby(["s1", "tok"], dropna=True, sort=False)["tok"] \
        .transform("size").reindex(key.index).fillna(0.0).to_numpy()
    out[:, c2["novel_fam_decoy"]] = np.where(has_novel, (cnt >= 2).astype(np.float32), 0.0)


def _family_matrix(n_s1: int, s1_idx: np.ndarray, cand_rows: csr_matrix,
                   weight: np.ndarray) -> csr_matrix:
    """Pool the confident candidates' token vectors per Source-1 entity."""
    sel = csr_matrix(
        (weight, (s1_idx, np.arange(len(s1_idx)))),
        shape=(n_s1, cand_rows.shape[0]), dtype=np.float32)
    return sel @ cand_rows


def stage2_features(pairs: pd.DataFrame, stage1: np.ndarray, base: np.ndarray,
                    s1: Side, cand: Side, n_s1: int,
                    conf: float = 0.80) -> np.ndarray:
    """Family features for every pair, given stage-1 scores for the same pairs."""
    li = pairs.s1_idx.to_numpy()
    ri = pairs.cand_idx.to_numpy()
    out = np.zeros((len(pairs), len(STAGE2_NAMES)), dtype=np.float32)
    c2 = {n: i for i, n in enumerate(STAGE2_NAMES)}

    if len(li):
        novel_tok = _name_diff_features(out, c2, base, s1, cand, li, ri)
        _house_features(out, c2, s1, cand, li, ri)
        _decoy_group_feature(out, c2, li, novel_tok)

    confident = stage1 >= conf
    fam_size = np.bincount(li[confident], minlength=n_s1).astype(np.float32)
    out[:, c2["fam_size"]] = fam_size[li]
    out[:, c2["fam_is_member"]] = confident.astype(np.float32)

    frame = pd.DataFrame({"s1": li, "sc": stage1})
    g = frame.groupby("s1", sort=False)["sc"]
    best = g.transform("max").to_numpy()
    out[:, c2["fam_score_max"]] = best
    out[:, c2["fam_score_mean"]] = g.transform("mean").to_numpy()
    out[:, c2["fam_score_gap"]] = stage1 - best
    out[:, c2["own_rev_margin"]] = base[:, COL["rev_margin_2nd"]]

    if not confident.any():
        return out

    # Pooled family vectors, built from confident candidates only. A candidate is
    # compared against its family with itself removed, or it would always match.
    for field, share_col, idf_col in (("addr_tok", "fam_addr_share", "fam_addr_idf"),
                                      ("name_core", "fam_name_share", None)):
        rows = cand.sets[field]
        fam = _family_matrix(n_s1, li[confident], rows[ri[confident]],
                             np.ones(int(confident.sum()), dtype=np.float32))
        mine = rows[ri]
        fam_of_pair = fam[li]
        overlap = np.asarray(mine.multiply(fam_of_pair).sum(axis=1)).ravel()
        # remove self-contribution for members
        self_tokens = cand.sizes[field][ri]
        overlap = overlap - np.where(confident, self_tokens, 0.0)
        denom = np.maximum(self_tokens, 1.0) * np.maximum(
            fam_size[li] - confident.astype(np.float32), 1.0)
        out[:, c2[share_col]] = (overlap / denom).astype(np.float32)

        if idf_col is not None:
            m = s1.sets[field]
            df = np.asarray(m.sum(axis=0)).ravel().astype(np.float32)
            idf = np.log((m.shape[0] + 1.0) / (df + 1.0)).astype(np.float32)
            w = np.asarray(mine.multiply(fam_of_pair) @ idf).ravel()
            tot = np.asarray(mine @ idf).ravel()
            out[:, c2[idf_col]] = (w / np.maximum(tot, 1e-6)).astype(np.float32)
    return out
