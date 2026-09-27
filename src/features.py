"""Stage 3 — pair features.

Candidate volume is in the tens of millions, so nothing here loops over pairs in
Python. Token-set features come from hashed binary CSR matrices built once per
table (intersection = elementwise sparse product), and the fuzzy string ratios
come from ``rapidfuzz.process.cpdist``, which scores two aligned lists in C
across threads.

Work is organised per (country, candidate source) and then per shard of Source-1
indices, which keeps peak memory to one candidate table plus one shard of
features. Rank features therefore compare a candidate against its rivals from
the same source — the meaningful comparison, since an entity typically has
matches in both sources.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from zlib import crc32

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse import csr_matrix

from .paths import cand_path, norm_path

HASH_BITS = 22
HASH_MASK = (1 << HASH_BITS) - 1
SET_FIELDS = ("name_core", "addr_tok", "addr_nums")
NGRAM = 3

FEATURE_NAMES = [
    "block_score", "pass_id", "source",
    "name_inter", "name_jac", "name_cont", "name_n1", "name_n2",
    "addr_inter", "addr_jac", "addr_cont", "addr_n1", "addr_n2",
    "num_inter", "num_jac", "postal_cmp",
    "fuzz_name_set", "fuzz_name_core", "fuzz_addr_sort",
    "cand_indic", "cand_nonascii", "cand_url", "cand_addr_empty",
    "name_len1", "name_len2",
    "n_cand", "rank_addr", "rank_name", "rank_total",
    "addr_gap_best", "name_gap_best", "total_gap_best",
    # Reverse-direction competition: a noisy record is a copy of at most one S1,
    # so how much better its best S1 is than its runner-up is the single most
    # direct evidence that the assignment is right.
    "rev_n", "rev_rank", "rev_gap_best", "rev_gap_addr", "rev_gap_name",
    "rev_margin_2nd",
    # Stronger similarity signals, aimed at the cases token overlap cannot see.
    "name_tri_cos",      # char 3-gram cosine: survives typos that break tokens
    "name_idf_share",    # fraction of S1's name IDF mass that is shared
    "addr_idf_share",
    "skel_ratio",        # similarity of consonant skeletons (Indic, doubled letters)
    "nospace_ratio",     # S1 name vs candidate with separators stripped (.com, @handle)
    "acronym_match",     # candidate is an acronym of the S1 name, or vice versa
    "house_eq",          # first house number identical
]
COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _ngram_csr(docs: np.ndarray, n: int = NGRAM) -> csr_matrix:
    """Binary CSR over hashed character n-grams of each string.

    Token overlap is blind to a typo inside a token; n-grams are not. Built for
    names only — address n-grams would triple the nonzeros for a signal the
    address token bag already carries.
    """
    indices, indptr = array("i"), array("i", [0])
    for s in docs:
        if s:
            seen = set()
            for j in range(max(len(s) - n + 1, 1)):
                g = s[j:j + n]
                if g not in seen:
                    seen.add(g)
                    indices.append(crc32(g.encode()) & HASH_MASK)
        indptr.append(len(indices))
    idx = np.frombuffer(indices, dtype=np.int32)
    ptr = np.frombuffer(indptr, dtype=np.int32)
    return csr_matrix((np.ones(len(idx), dtype=np.float32), idx, ptr),
                      shape=(len(docs), HASH_MASK + 1))


def _nospace(docs: np.ndarray) -> np.ndarray:
    out = np.empty(len(docs), dtype=object)
    for i, s in enumerate(docs):
        out[i] = s.replace(" ", "") if s else ""
    return out


def _skeleton_str(docs: np.ndarray) -> np.ndarray:
    """Consonant skeleton of a whole string, space-separated per token."""
    from .blocking_rev import skeleton
    out = np.empty(len(docs), dtype=object)
    for i, s in enumerate(docs):
        out[i] = " ".join(skeleton(t) for t in s.split()) if s else ""
    return out


def _acronym(docs: np.ndarray) -> np.ndarray:
    out = np.empty(len(docs), dtype=object)
    for i, s in enumerate(docs):
        out[i] = "".join(t[0] for t in s.split() if t) if s else ""
    return out


def _first_num(docs: np.ndarray) -> np.ndarray:
    out = np.empty(len(docs), dtype=object)
    for i, s in enumerate(docs):
        out[i] = s.split()[0] if s else ""
    return out


def _csr(docs: np.ndarray) -> csr_matrix:
    """Binary CSR over hashed tokens; tokens are already unique per record.

    crc32 rather than the builtin hash(): Python randomizes string hashing per
    process, so hash() would map tokens to different columns in the training run
    than in inference. Set intersections are almost invariant to the mapping, but
    collisions are not, which made runs differ by a handful of pairs.
    """
    indices, indptr = array("i"), array("i", [0])
    for s in docs:
        if s:
            for t in s.split():
                indices.append(crc32(t.encode()) & HASH_MASK)
        indptr.append(len(indices))
    idx = np.frombuffer(indices, dtype=np.int32)
    ptr = np.frombuffer(indptr, dtype=np.int32)
    return csr_matrix((np.ones(len(idx), dtype=np.float32), idx, ptr),
                      shape=(len(docs), HASH_MASK + 1))


@dataclass
class Side:
    """One normalized table as arrays, with prebuilt token matrices."""

    entity_id: np.ndarray
    name_norm: np.ndarray
    name_core: np.ndarray
    addr_tok: np.ndarray
    postal: np.ndarray
    flags: np.ndarray
    sets: dict
    sizes: dict
    nospace: np.ndarray
    skel: np.ndarray
    acronym: np.ndarray
    first_num: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Side":
        sets, sizes = {}, {}
        for field in SET_FIELDS:
            m = _csr(df[field].to_numpy())
            sets[field] = m
            sizes[field] = np.diff(m.indptr).astype(np.float32)
        name_norm = df.name_norm.to_numpy()
        name_core = df.name_core.to_numpy()
        tri = _ngram_csr(name_norm)
        sets["name_tri"] = tri
        sizes["name_tri"] = np.diff(tri.indptr).astype(np.float32)
        return cls(
            entity_id=df.entity_id.to_numpy(),
            name_norm=name_norm,
            name_core=name_core,
            addr_tok=df.addr_tok.to_numpy(),
            postal=df.postal.to_numpy(),
            flags=df.name_flags.to_numpy().astype(np.int8),
            sets=sets, sizes=sizes,
            nospace=_nospace(name_norm),
            skel=_skeleton_str(name_core),
            acronym=_acronym(name_core),
            first_num=_first_num(df.addr_nums.to_numpy()),
        )

    @classmethod
    def load(cls, split: str, source: int, country: str) -> "Side":
        return cls.from_frame(pd.read_parquet(norm_path(split, source, country)))


def _overlap(left: Side, right: Side, field: str, li: np.ndarray, ri: np.ndarray):
    """Intersection size, Jaccard and containment for one token field."""
    inter = np.asarray(
        left.sets[field][li].multiply(right.sets[field][ri]).sum(axis=1)
    ).ravel().astype(np.float32)
    n1, n2 = left.sizes[field][li], right.sizes[field][ri]
    union = np.maximum(n1 + n2 - inter, 1.0)
    return inter, inter / union, inter / np.maximum(np.minimum(n1, n2), 1.0), n1, n2


def pair_features(s1: Side, cand: Side, pairs: pd.DataFrame,
                  source: int) -> np.ndarray:
    """Feature matrix for one shard of pairs from a single candidate source."""
    li = pairs.s1_idx.to_numpy()
    ri = pairs.cand_idx.to_numpy()
    out = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)

    out[:, COL["block_score"]] = pairs.block_score.to_numpy()
    out[:, COL["pass_id"]] = pairs.pass_id.to_numpy()
    out[:, COL["source"]] = source

    inter, jac, cont, n1, n2 = _overlap(s1, cand, "name_core", li, ri)
    out[:, COL["name_inter"]], out[:, COL["name_jac"]] = inter, jac
    out[:, COL["name_cont"]], out[:, COL["name_n1"]], out[:, COL["name_n2"]] = \
        cont, n1, n2

    inter, jac, cont, n1, n2 = _overlap(s1, cand, "addr_tok", li, ri)
    out[:, COL["addr_inter"]], out[:, COL["addr_jac"]] = inter, jac
    out[:, COL["addr_cont"]], out[:, COL["addr_n1"]], out[:, COL["addr_n2"]] = \
        cont, n1, n2

    inter, jac, _, _, _ = _overlap(s1, cand, "addr_nums", li, ri)
    out[:, COL["num_inter"]], out[:, COL["num_jac"]] = inter, jac

    # postal: 1 = both present and equal, 0 = both present and different,
    # -1 = at least one missing (a real third state, not a missing value)
    p1, p2 = s1.postal[li], cand.postal[ri]
    both = (p1 != "") & (p2 != "")
    out[:, COL["postal_cmp"]] = np.where(both, (p1 == p2).astype(np.float32), -1.0)

    kw = dict(workers=-1, dtype=np.float32)
    out[:, COL["fuzz_name_set"]] = process.cpdist(
        s1.name_norm[li], cand.name_norm[ri], scorer=fuzz.token_set_ratio, **kw)
    # name_core is a sorted token set, so a plain ratio is already order-invariant.
    out[:, COL["fuzz_name_core"]] = process.cpdist(
        s1.name_core[li], cand.name_core[ri], scorer=fuzz.ratio, **kw)
    out[:, COL["fuzz_addr_sort"]] = process.cpdist(
        s1.addr_tok[li], cand.addr_tok[ri], scorer=fuzz.token_sort_ratio, **kw)

    flags = cand.flags[ri]
    out[:, COL["cand_indic"]] = flags & 1
    out[:, COL["cand_nonascii"]] = (flags >> 1) & 1
    out[:, COL["cand_url"]] = (flags >> 2) & 1
    out[:, COL["cand_addr_empty"]] = (cand.addr_tok[ri] == "")
    out[:, COL["name_len1"]] = [len(x) for x in s1.name_norm[li]]
    out[:, COL["name_len2"]] = [len(x) for x in cand.name_norm[ri]]

    _add_extra_features(out, s1, cand, li, ri)
    _add_rank_features(out, li)
    _add_reverse_features(out, ri)
    return out


def _idf_vector(side: Side, field: str) -> np.ndarray:
    """Per-hash-column IDF from the clean Source-1 distribution."""
    m = side.sets[field]
    df = np.asarray(m.sum(axis=0)).ravel().astype(np.float32)
    return np.log((m.shape[0] + 1.0) / (df + 1.0)).astype(np.float32)


def _add_extra_features(out: np.ndarray, s1: Side, cand: Side,
                        li: np.ndarray, ri: np.ndarray) -> None:
    # char 3-gram cosine on the full normalized name
    a, b = s1.sets["name_tri"][li], cand.sets["name_tri"][ri]
    inter = np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)
    denom = np.sqrt(np.maximum(s1.sizes["name_tri"][li], 1.0)
                    * np.maximum(cand.sizes["name_tri"][ri], 1.0))
    out[:, COL["name_tri_cos"]] = inter / np.maximum(denom, 1e-6)

    # IDF-weighted share: how much of the S1 record's rare-token mass is shared.
    # A shared rare token is far stronger evidence than a shared common one, which
    # a plain Jaccard cannot express.
    for field, col in (("name_core", "name_idf_share"),
                       ("addr_tok", "addr_idf_share")):
        idf = _idf_vector(s1, field)
        av, bv = s1.sets[field][li], cand.sets[field][ri]
        shared = np.asarray(av.multiply(bv) @ idf).ravel().astype(np.float32)
        total = np.asarray(av @ idf).ravel().astype(np.float32)
        out[:, COL[col]] = shared / np.maximum(total, 1e-6)

    kw = dict(workers=-1, dtype=np.float32)
    out[:, COL["skel_ratio"]] = process.cpdist(
        s1.skel[li], cand.skel[ri], scorer=fuzz.token_sort_ratio, **kw)
    # A ".com" or "@handle" alias keeps the letters but loses the spaces.
    out[:, COL["nospace_ratio"]] = process.cpdist(
        s1.nospace[li], cand.nospace[ri], scorer=fuzz.ratio, **kw)

    ac1, ac2 = s1.acronym[li], cand.acronym[ri]
    n1, n2 = s1.nospace[li], cand.nospace[ri]
    out[:, COL["acronym_match"]] = [
        2.0 if a and a == b else 1.0 if (a and a == y) or (b and b == x) else 0.0
        for a, b, x, y in zip(ac1, ac2, n1, n2)]

    f1, f2 = s1.first_num[li], cand.first_num[ri]
    out[:, COL["house_eq"]] = [
        1.0 if a and a == b else -1.0 if not a or not b else 0.0
        for a, b in zip(f1, f2)]


def _add_reverse_features(out: np.ndarray, cand_idx: np.ndarray) -> None:
    if len(cand_idx) == 0:      # a block can be empty once pairs are filtered
        return
    """Competition among the S1 records that one noisy record could belong to.

    Ground truth is 1-to-many from Source 1, so each Source-2/3 record has at
    most one correct Source-1 record. Grouping by the *candidate* therefore turns
    matching into a choice between mutually exclusive options, and the margin
    between the best and second-best option says far more than either score alone.
    """
    total = out[:, COL["addr_jac"]] + out[:, COL["name_jac"]]
    frame = pd.DataFrame({
        "cand": cand_idx,
        "tot": total,
        "addr": out[:, COL["addr_jac"]],
        "name": out[:, COL["name_jac"]],
    })
    g = frame.groupby("cand", sort=False)
    out[:, COL["rev_n"]] = g["cand"].transform("size").to_numpy()
    out[:, COL["rev_rank"]] = g["tot"].rank(ascending=False,
                                            method="min").to_numpy()
    best = g["tot"].transform("max").to_numpy()
    out[:, COL["rev_gap_best"]] = total - best
    out[:, COL["rev_gap_addr"]] = (out[:, COL["addr_jac"]]
                                   - g["addr"].transform("max").to_numpy())
    out[:, COL["rev_gap_name"]] = (out[:, COL["name_jac"]]
                                   - g["name"].transform("max").to_numpy())
    # Margin of this record's best option over its runner-up: 0 when there is no
    # runner-up, so an uncontested candidate is distinguishable from a close call.
    # Computed by sorting rather than a per-group lambda: the lambda ran Python
    # per candidate group, and with millions of groups it dominated the whole
    # feature build.
    order = np.lexsort((-total, cand_idx))
    c_sorted = cand_idx[order]
    t_sorted = total[order]
    starts = np.flatnonzero(np.diff(c_sorted, prepend=c_sorted[0] - 1))
    within = np.arange(len(c_sorted)) - np.repeat(
        starts, np.diff(np.append(starts, len(c_sorted))))
    second_sorted = np.where(within == 0, t_sorted, np.nan)
    # the runner-up of each group is the row at within == 1
    runner = np.full(len(c_sorted), np.nan, dtype=np.float32)
    has_second = starts + 1 < np.append(starts[1:], len(c_sorted))
    runner[starts[has_second]] = t_sorted[starts[has_second] + 1]
    margin_sorted = np.where(np.isnan(runner), 0.0, second_sorted - runner)
    margin = np.zeros(len(total), dtype=np.float32)
    margin[order] = np.nan_to_num(margin_sorted, nan=0.0)
    out[:, COL["rev_margin_2nd"]] = np.where(total >= best, margin, 0.0)


def _add_rank_features(out: np.ndarray, s1_idx: np.ndarray) -> None:
    if len(s1_idx) == 0:
        return
    """Per-entity competition features: how a pair ranks among its rivals.

    F0.5 is macro-averaged per entity, so whether a candidate is the *best*
    available for its entity matters as much as its absolute similarity.
    """
    frame = pd.DataFrame({
        "s1": s1_idx,
        "addr": out[:, COL["addr_jac"]],
        "name": out[:, COL["name_jac"]],
        "tot": out[:, COL["addr_jac"]] + out[:, COL["name_jac"]],
    })
    g = frame.groupby("s1", sort=False)
    out[:, COL["n_cand"]] = g["s1"].transform("size").to_numpy()
    for field, rank_col, gap_col in (("addr", "rank_addr", "addr_gap_best"),
                                     ("name", "rank_name", "name_gap_best"),
                                     ("tot", "rank_total", "total_gap_best")):
        out[:, COL[rank_col]] = g[field].rank(ascending=False,
                                              method="min").to_numpy()
        out[:, COL[gap_col]] = (frame[field].to_numpy()
                                - g[field].transform("max").to_numpy())


def load_pairs(split: str, country: str, source: int) -> pd.DataFrame:
    """Candidate pairs for one (country, source), sorted by Source-1 index."""
    df = pd.read_parquet(cand_path(split, country, source))
    return df.sort_values("s1_idx", kind="stable", ignore_index=True)


def iter_shards(pairs: pd.DataFrame, n_s1: int, shard: int):
    """Yield contiguous slices of ``pairs`` covering ``shard`` S1 entities each."""
    edges = np.searchsorted(pairs.s1_idx.to_numpy(),
                            np.arange(0, n_s1 + shard, shard))
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            yield pairs.iloc[a:b]
