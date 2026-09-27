# Business Entity Resolution — pipeline

Matches every Source-1 record to its Source-2 and Source-3 counterparts and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Uses **only the provided training data** — no external databases, APIs, geocoders,
or pretrained models of any kind.

## Approach in one paragraph

Source 1 is clean and Sources 2/3 carry all the noise, so candidate generation runs
**in reverse**: each noisy record asks which clean Source-1 record it is a copy of.
Ground truth is 1-to-many from Source 1, so a noisy record has **at most one**
correct answer, and a top-3 retrieval suffices where matching forwards needed ~40
candidates. A DF-capped inverted index over typed tokens (address, name, and name
consonant skeletons) retrieves candidates inside each country; a LightGBM pair
classifier scores them, a second stage re-scores using each entity's other
confident matches, and a per-entity decision rule emits the final lists.

## Environment

Python 3.11. On macOS LightGBM also needs OpenMP (`brew install libomp`), or it
fails at import with `Library not loaded: @rpath/libomp.dylib`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce end to end

Run from `code/business_entity_resolution/`; paths are derived in `src/paths.py`, so
there is nothing to configure.

```bash
cd code/business_entity_resolution
PY=../../.venv/bin/python

$PY -m src.to_parquet                      # TSV -> Parquet                    ~25s
$PY -m src.prepare --split train           # normalize, shard by country       ~3m
$PY -m src.prepare --split test            #                                   ~3m
$PY -m src.blocking_rev --split train      # reverse candidates (train sample) ~3h
$PY -m src.block_eval                      # blocking recall vs ground truth   ~5m
$PY -m src.train_v3                        # stage 1 + OOF + stage 2 + tuning  ~1.5h
$PY -m src.predict --save-candidates       # test candidates + both TSVs       ~3h
```

Or in one command, which also validates and only replaces an existing submission
if the new one scores better:

```bash
$PY -m src.run_v3
```

Validate explicitly (from the repository root):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

Total end-to-end runtime is about **8 hours on an 8-core laptop**, dominated by
retrieval. Every stage is resumable: an existing candidate file is reused rather
than regenerated, `--split` / `--country` restrict any stage, and
`src.predict --append` extends partial output.

## Reproducibility notes

- Token hashing uses `crc32`, not Python's `hash()`, which is randomized per
  process and would map tokens to different feature columns between training and
  inference.
- Entity sampling and cross-validation folds come from a fixed multiplicative hash
  of the row index, so the fit/validation split is identical across runs.
- All stages read the Parquet copies in `work/`; the provided TSVs are never
  modified.

## Source layout

| file | role |
| --- | --- |
| `src/paths.py` | paths, and the low-memory TSV read settings every stage uses |
| `src/to_parquet.py` | stage 0 — chunked TSV to Parquet |
| `src/normalize.py` | name and address normalization |
| `src/prepare.py` | stage 1 — normalize all sources, shard by country |
| `src/blocking_rev.py` | stage 2 — reverse candidate generation (**the one used**) |
| `src/blocking.py` | forward candidate generation (v1) + the shared token index |
| `src/block_eval.py` | blocking recall against ground truth |
| `src/sweep_rev.py`, `src/sweep_blocking.py` | blocking parameter sweeps |
| `src/diagnose_misses.py` | why blocking misses what it misses, by cause |
| `src/features.py` | stage 3 — vectorized pair features |
| `src/stage2.py` | stage-2 sibling/family features |
| `src/truth.py`, `src/metric.py` | ground truth; macro F0.5 and the exclusivity pass |
| `src/decide.py` | per-entity decision rules |
| `src/train_v3.py` | stage-1 + stage-2 training with out-of-fold scores |
| `src/train.py` | single-stage trainer (v2) |
| `src/predict.py` | stage 5 — scoring and submission output |
| `src/analyze.py` | loss attribution by cause and noise type |
| `src/run_v3.py` | the whole pipeline, with a validation gate before promotion |
| `src/sampling.py` | the shared entity sampling hash |
| `src/profile_data.py` | exploratory data profile (not needed to reproduce) |

`src/blocking.py` is retained because `src/blocking_rev.py` imports its token index
and top-k helpers, and because the forward/reverse comparison is part of the
methodology.

## Scale and memory

Inputs total ~2.5 GB of TSV and ~25M records; the pipeline is written for a 16 GB
laptop. TSVs are read in chunks; everything downstream reads Parquet one column at
a time; blocking, features and scoring each work on one country and one candidate
source at a time, sharded by Source-1 index. Retrieval blocks are cut on a
posting budget rather than a row count, so the sparse products stay bounded
whatever the document-frequency cap. Retrieval runs across processes with a `fork`
context, so the index is built once and inherited copy-on-write.

Keep **3 GB of free disk**: `candidate_pairs.tsv` is ~460 MB and intermediates add
a few hundred MB more.
