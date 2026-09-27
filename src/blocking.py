"""Stage 2 — address-first candidate generation.

Two passes, both run strictly inside one ``country`` (true matches never cross
country) and against one candidate source at a time, so peak memory stays at
roughly one source shard:

* **address pass** — an inverted index over normalized address tokens, keeping
  only tokens whose document frequency is <= ``maxdf_addr``. Very common tokens
  (city, state, "rd") carry almost no discriminating signal and would explode
  the postings list, so the cap both sharpens and cheapens the index. Each S1
  record is queried with its ``r_query`` rarest indexed tokens, candidates are
  scored by the summed IDF of shared tokens, and the top ``k_addr`` are kept.
  This is the primary pass: 94.95% of true pairs share >=2 address tokens.
* **prefix pass** — the address index again, over tokens truncated to their
  first few characters, which is what makes blocking tolerant of the character
  typos that are everywhere in Sources 2 and 3.
* **name pass** — the same machinery over name-core tokens, over *every*
  candidate. It is not merely a fallback for the 3.3% of records with an empty
  address: the two passes fail on different records, so their union recovers far
  more than either does alone. Raising the address DF cap to chase those pairs
  instead costs recall per second and floods the candidate set, because the
  tokens it admits are the non-discriminating ones.

Parameters were chosen by ``src.sweep_blocking``, which measures recall and
candidates-per-entity directly against the training ground truth.

    python -m src.blocking --split train
    python -m src.blocking --split test --country France
"""
from __future__ import annotations

import argparse
import sys
import time
from array import array
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .paths import WORK, cand_path, countries, ensure_dirs, norm_path
from .sampling import SAMPLE_PCT, sampled_mask


@dataclass(frozen=True)
class BlockParams:
    """Blocking configuration. Defaults are the tuned setting."""

    # Tuned by src.sweep_blocking against ground-truth recall: this setting
    # reaches 90.97% recall over true pairs at 40 candidates per entity per
    # source. See CLAUDE.md for the full sweep.
    maxdf_addr: int = 5_000
    maxdf_name: int = 20_000
    maxdf_prefix: int = 5_000
    r_query: int = 24
    k_addr: int = 24
    k_name: int = 12
    k_prefix: int = 12      # 0 disables the typo-tolerant prefix pass
    prefix_len: int = 4
    length_norm: bool = True   # divide score by sqrt(candidate token count)
    max_rows: int = 20_000     # query rows per sparse product
    # Hard cap on the nonzeros of one query block's score matrix. A block's cost
    # is the summed document frequency of its query tokens, so with a loose DF
    # cap a single 20k-row block can reach billions of nonzeros and be OOM-killed.
    # Blocks are therefore cut on this budget, not on a fixed row count.
    posting_budget: int = 20_000_000

    def label(self) -> str:
        return (f"addr(maxdf={self.maxdf_addr},k={self.k_addr}) "
                f"name(maxdf={self.maxdf_name},k={self.k_name}) "
                f"pre(maxdf={self.maxdf_prefix},k={self.k_prefix},"
                f"n={self.prefix_len}) r={self.r_query}")


def prefix_docs(docs: np.ndarray, n: int = 4) -> np.ndarray:
    """Truncate every token to its first ``n`` characters, keeping order-unique.

    Token-exact indexing cannot see a single-character typo: "Beulaaville" and
    "Beulaville" share no token, and typos are pervasive in Sources 2 and 3.
    Indexing truncated tokens recovers those pairs at a fraction of the cost of
    character n-grams, since any typo after the prefix becomes invisible.
    """
    out = np.empty(len(docs), dtype=object)
    for i, s in enumerate(docs):
        out[i] = " ".join(dict.fromkeys(t[:n] for t in s.split())) if s else ""
    return out


class TokenIndex:
    """DF-capped inverted index over whitespace-joined token strings."""

    def __init__(self, docs: np.ndarray, maxdf: int):
        df: Counter[str] = Counter()
        for s in docs:
            if s:
                df.update(s.split())
        self.df = df
        # Ids must be dense over the *kept* tokens only, so they cannot come
        # from enumerate() over the unfiltered counter.
        self.vocab: dict[str, int] = {}
        for t, c in df.items():
            if c <= maxdf:
                self.vocab[t] = len(self.vocab)

        n_doc, n_vocab = len(docs), len(self.vocab)
        indices, indptr = array("i"), array("i", [0])
        get = self.vocab.get
        for s in docs:
            if s:
                for t in s.split():
                    j = get(t)
                    if j is not None:
                        indices.append(j)
            indptr.append(len(indices))
        idx = np.frombuffer(indices, dtype=np.int32)
        ptr = np.frombuffer(indptr, dtype=np.int32)
        # Rows are candidate records, columns tokens.
        matrix = csr_matrix(
            (np.ones(len(idx), dtype=np.float32), idx, ptr), shape=(n_doc, n_vocab)
        )
        self.n_doc = n_doc
        self.n_vocab = n_vocab
        self.idf = np.log(n_doc / np.maximum(
            np.asarray([df[t] for t in self.vocab], dtype=np.float32), 1.0))
        # Candidates with more indexed tokens have more chances to collide;
        # dividing by sqrt(count) keeps a long address from crowding out a
        # short one that shares the same rare street token.
        counts = np.diff(ptr).astype(np.float32)
        self.cand_norm = np.sqrt(np.maximum(counts, 1.0))
        self.cand_t = matrix.T.tocsc()

    def query_topk(self, docs: np.ndarray, k: int, r: int, max_rows: int,
                   budget: int, length_norm: bool):
        """Yield ``(offset, rows, cols, scores)`` of the top-k hits per query.

        Query blocks are cut whenever the accumulated posting cost reaches
        ``budget``, which bounds the size of the sparse product independently of
        how loose the document-frequency cap is.
        """
        vocab, dfc, idf = self.vocab, self.df, self.idf
        q_ind, q_ptr, q_dat = array("i"), array("i", [0]), array("f")
        offset, cost = 0, 0

        def flush(offset):
            q = csr_matrix(
                (np.frombuffer(q_dat, dtype=np.float32),
                 np.frombuffer(q_ind, dtype=np.int32),
                 np.frombuffer(q_ptr, dtype=np.int32)),
                shape=(len(q_ptr) - 1, self.n_vocab),
            )
            scores = (q @ self.cand_t).tocsr()
            if length_norm and scores.nnz:
                scores.data /= self.cand_norm[scores.indices]
            return (offset, *_topk(scores, k))

        for i, s in enumerate(docs):
            if s:
                hits = [(dfc[t], vocab[t]) for t in s.split() if t in vocab]
                if hits:
                    hits.sort()                        # rarest tokens first
                    for dfv, j in hits[:r]:
                        q_ind.append(j)
                        q_dat.append(float(idf[j]))
                        cost += dfv
            q_ptr.append(len(q_ind))
            if cost >= budget or len(q_ptr) - 1 >= max_rows:
                yield flush(offset)
                offset = i + 1
                q_ind, q_ptr, q_dat = array("i"), array("i", [0]), array("f")
                cost = 0
        if len(q_ptr) > 1:
            yield flush(offset)

def _topk(scores: csr_matrix, k: int):
    ptr, idx, dat = scores.indptr, scores.indices, scores.data
    rows, cols, vals = [], [], []
    for r in range(scores.shape[0]):
        a, b = ptr[r], ptr[r + 1]
        if b <= a:
            continue
        d = dat[a:b]
        sel = np.argpartition(d, -k)[-k:] if b - a > k else np.arange(b - a)
        rows.append(np.full(len(sel), r, dtype=np.int32))
        cols.append(idx[a:b][sel].astype(np.int32))
        vals.append(d[sel].astype(np.float32))
    if not rows:
        i, f = np.empty(0, np.int32), np.empty(0, np.float32)
        return i, i, f
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def collect(index: TokenIndex, queries: np.ndarray, k: int, r: int,
            cand_map: np.ndarray, params: BlockParams):
    """Run every query shard and map local candidate ids back to source rows."""
    s1, cand, score = [], [], []
    for offset, rows, cols, vals in index.query_topk(
            queries, k, r, params.max_rows, params.posting_budget,
            params.length_norm):
        s1.append(rows + offset)
        cand.append(cand_map[cols])
        score.append(vals)
    if not s1:
        i, f = np.empty(0, np.int32), np.empty(0, np.float32)
        return i, i, f
    return np.concatenate(s1), np.concatenate(cand), np.concatenate(score)


def block_country(split: str, country: str, source: int, params: BlockParams,
                  sample_only: bool = False) -> pd.DataFrame:
    """Candidate pairs for one (country, candidate source).

    With ``sample_only`` the query side is restricted to the entities training
    samples. Blocking every training entity and then discarding 92% of the
    output wastes both the query time and, at ~42 candidates per entity, about a
    gigabyte of disk.
    """
    s1 = pd.read_parquet(norm_path(split, 1, country),
                         columns=["addr_tok", "name_core"])
    cand = pd.read_parquet(norm_path(split, source, country),
                           columns=["addr_tok", "name_core"])
    n_s1 = len(s1)
    if sample_only:
        # Keep the original row indices: every downstream stage addresses
        # records by their row number in the normalized files.
        wanted = np.flatnonzero(sampled_mask(np.arange(n_s1)))
        s1 = s1.iloc[wanted].reset_index(drop=True)
    else:
        wanted = None
    addr_len = cand.addr_tok.str.len().to_numpy()
    has_addr = np.flatnonzero(addr_len > 0)
    all_rows = np.arange(len(cand), dtype=np.int64)

    t0 = time.time()
    idx = TokenIndex(cand.addr_tok.to_numpy()[has_addr], params.maxdf_addr)
    a_s1, a_cand, a_score = collect(
        idx, s1.addr_tok.to_numpy(), params.k_addr, params.r_query, has_addr, params)
    del idx
    t_addr = time.time() - t0

    t0 = time.time()
    idx = TokenIndex(cand.name_core.to_numpy(), params.maxdf_name)
    n_s1_, n_cand_, n_score = collect(
        idx, s1.name_core.to_numpy(), params.k_name, params.r_query, all_rows, params)
    del idx
    t_name = time.time() - t0

    t0 = time.time()
    if params.k_prefix:
        idx = TokenIndex(prefix_docs(cand.addr_tok.to_numpy()[has_addr],
                                     params.prefix_len), params.maxdf_prefix)
        p_s1, p_cand, p_score = collect(
            idx, prefix_docs(s1.addr_tok.to_numpy(), params.prefix_len),
            params.k_prefix, params.r_query, has_addr, params)
        del idx
    else:
        p_s1 = p_cand = np.empty(0, np.int32)
        p_score = np.empty(0, np.float32)
    t_pre = time.time() - t0

    out = pd.DataFrame({
        "s1_idx": np.concatenate([a_s1, n_s1_, p_s1]).astype(np.int32),
        "cand_idx": np.concatenate([a_cand, n_cand_, p_cand]).astype(np.int32),
        "block_score": np.concatenate([a_score, n_score, p_score]).astype(np.float32),
        "pass_id": np.concatenate([
            np.ones(len(a_s1), np.int8), np.full(len(n_s1_), 2, np.int8),
            np.full(len(p_s1), 3, np.int8)]),
    })
    out = out.sort_values("block_score", ascending=False)
    out = out.drop_duplicates(["s1_idx", "cand_idx"], keep="first")
    if wanted is not None:
        out["s1_idx"] = wanted[out.s1_idx.to_numpy()].astype(np.int32)
    queried = len(s1)
    print(f"    S{source}: {len(out):>11,} pairs ({len(out) / queried:5.2f}/entity)  "
          f"addr {t_addr:.0f}s name {t_name:.0f}s prefix {t_pre:.0f}s")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--country", default=None, help="default: every country")
    ap.add_argument("--maxdf", type=int, default=BlockParams.maxdf_addr)
    ap.add_argument("--k", type=int, default=BlockParams.k_addr)
    ap.add_argument("--r", type=int, default=BlockParams.r_query)
    ap.add_argument("--sources", default="2,3")
    ap.add_argument("--all-entities", action="store_true",
                    help="train split: block every entity, not just the sample")
    args = ap.parse_args()
    ensure_dirs()

    params = BlockParams(maxdf_addr=args.maxdf, k_addr=args.k, r_query=args.r)
    # Only the sampled entities are ever used for training, so the train split
    # blocks just those unless asked otherwise. Test always blocks everything.
    sample_only = (args.split == "train") and not args.all_entities
    print(f"  blocking with {params.label()}"
          + (f"  [train sample: {SAMPLE_PCT}% of entities]" if sample_only else ""))
    for country in ([args.country] if args.country else countries(args.split)):
        print(f"  {args.split.upper()} {country}")
        for source in (int(x) for x in args.sources.split(",")):
            block_country(args.split, country, source, params,
                          sample_only=sample_only).to_parquet(
                cand_path(args.split, country, source), compression="zstd",
                index=False)
    print(f"candidates written to {WORK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
