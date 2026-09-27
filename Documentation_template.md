# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** FRAZER
**Team Members:** A. Sai Jayanth, Saiteja, Vivek, Nelson
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We invert the obvious framing. Source 1 is perfectly clean and Sources 2/3 carry all the noise, so
instead of asking each clean record to find its ~4 noisy copies among 5M candidates, we ask each
**noisy record which clean record it is a copy of**. Ground truth is 1-to-many from Source 1, so a
noisy record has **at most one** correct answer: a top-3 retrieval suffices where matching forwards
needed ~40 candidates per entity. That single change raised blocking recall from 89.0% to 94.8% on
the harder country while cutting candidates per entity from ~84 to ~14, and it removed a bias that
had made our forward validation optimistic. A LightGBM pair classifier scores the candidates, a
second stage re-scores each candidate using its entity's other confident matches, and a per-entity
decision rule emits the lists.

---

## 2. Methodology

### 2.1 Problem Analysis

We profiled all 25M records before modelling. Five findings shaped every later decision.

**1. Source 1 is completely clean; all noise lives in Sources 2 and 3.**

| | S1 | S2 | S3 |
|---|---|---|---|
| Indic-script name | 0% | 5.4% | 3.0% |
| non-ASCII name | 0% | 15.2% | 11.5% |
| URL-like name | 0% | 4.0% | 4.0% |
| ALLCAPS name | 0% | 28.0% | 7.6% |
| ALLCAPS address | 0.0% | 66.7% | 3.3% |
| empty address | 0% | 3.4% | 3.3% |

Normalization is therefore a one-directional cleanup, and — the key consequence — the *clean* side
is the better thing to index.

**2. Singletons are rare.** 5.6% of Source-1 entities have no match; the rest average 3.67 (median
4, max 11). Predicting empty everywhere scores 0.0558. Despite F0.5's precision weighting this is a
recall problem; being uniformly conservative is not a strategy.

**3. Matched IDs are globally disjoint.** 7,638,365 matched IDs, all distinct — no Source-2/3 record
is claimed by two Source-1 entities. ~26% of Source-2/3 records match nothing, so the constraint is
exclusive-*or-nothing*. This is the structural fact the whole design rests on.

**4. Country never crosses.** 733,931 true pairs checked, zero cross-country matches. All work
happens inside one country label, which also makes France — 15% of test, absent from training —
self-contained rather than a domain-shift problem.

**5. Address is a stronger signal than name.** Over 219,898 true pairs: 94.95% share >=2 address
tokens and only 0.02% of non-empty-address pairs share none, while **14.35% share no name token at
all** (alias records: coined brand words, `@handles`, `website.com`-style names). 4.39% have an empty candidate
address. Only 0.0018% have neither signal.

**Noise catalogue.** Names: legal suffixes added, removed or **moved to the front**
(e.g. `llc acme & sons marine`); bracketed suffixes; `M/s`, `Mr`, `DBA:`, `trading as`, `--`
prefixes; `(ID: 12345)` tails; character typos (e.g. `Acm3 Solutoins`, doubled vowels); injected accents;
injected boilerplate (`Center`, `Service`); word transposition; ALLCAPS. Indic scripts are **not
only Hindi** — Kannada appears too. Addresses: free **component
reordering**; `NULL`/`<NULL>` literals; `##`, `Door No`, `H.no`, `PO Box` prefixes; house numbers
dropped or perturbed (`5721`->`5720`, `141`->`141-143`); city suffixes `Township`/`City`/`County`;
state abbreviation vs full name vs Indic script. ZIP/PIN appears in only 6.5% of addresses, so
postal blocking keys are nearly useless here.

### 2.2 Solution Strategy

**Approach Type:** Reverse candidate generation + two-stage classifier + per-entity decision rule
**Core Innovation:** Retrieving in the noisy->clean direction, which turns "find all copies of this
record" into "pick the one record this is a copy of" — a fundamentally easier problem that
simultaneously improves recall, shrinks the candidate set ~6x, and removes a validation bias.

Normalization (`src/normalize.py`) transliterates with `unidecode`, strips junk affixes and a union
of US/Indian/French legal forms (`inc`, `llc`, `pvt`, `limited`, `sarl`, `sas`, `sasu`, `eurl`,
`sci`, `gmbh`, ...), canonicalizes street types across all three countries (`street`/`st`,
`road`/`rd`, `boulevard`/`bd`/`blvd`, `rue`/`r`, `impasse`/`imp`, `chemin`/`ch`), drops municipal
and unit noise plus French articles and `bis`/`ter`/`quater`, and emits addresses as an
order-independent token bag with digit tokens and any 5/6-digit postal code kept separately.
Everything is country-agnostic, so French records get identical treatment despite having no labels.

One measurement worth recording: `unidecode` drops the inherent vowel of Indic scripts, so
`महाराष्ट्र` renders as `mhaaraassttr`, not `maharashtra`. Our state alias table uses the *measured*
transliterations; every form we first guessed by ear was wrong.

---

## 3. Candidate Generation (Blocking)

**Direction.** For each Source-2/3 record we retrieve its best Source-1 records. Because ground
truth is 1-to-many from Source 1, each noisy record has at most one correct answer. Per-entity
candidate count falls out as `(|S2| + |S3|) / |S1| * k` rather than being something we must choose.

**Blocking keys.** One index over **typed tokens**, so a single sparse product combines every
signal under IDF weighting instead of scoring each separately:

| type | content | purpose |
|---|---|---|
| `a:` | normalized address token | primary signal |
| `n:` | name-core token (legal forms stripped) | names, reordered or suffix-shifted |
| `s:` | **name consonant skeleton** | Indic transliterations, doubled-letter typos, accents |

The skeleton is what makes Indic names reachable at all: `unidecode` renders a Devanagari name like `कृष्णा ट्रेडर्स` as
`krssnnaa ttrerddrs`, sharing no token with `Krishna Traders`, but dropping vowels and collapsing
doubled letters maps both to a near-identical consonant skeleton. The same mapping absorbs
doubled-vowel typos and transliterated legal suffixes such as `praaivett`/`private`.

Scoring is summed IDF of shared tokens divided by sqrt(candidate token count); tokens above a
document-frequency cap are not indexed at all. Query blocks are cut on a **posting budget** rather
than a fixed row count, so the sparse products stay bounded whatever the cap.

**Adaptive-k.** Records with at most two address tokens have only their name to go on, and miss
diagnosis found empty-address records were **28.5% of all misses against a 2.6% base rate** — and
were *outranked* rather than unreachable. Those records get k=18; everyone else keeps k=3.

### Adaptive-k verdict: keep it (measured 2026-09-27)

The open question — whether adaptive-k earns its extra candidates — is settled:

| | k=3 uniform | adaptive-k | cand/entity | entities reachable |
|---|---|---|---|---|
| US | 97.56% | **98.22%** | 13.99 -> 16.54 | 99.88% |
| India | 94.83% | **95.33%** | 13.66 -> 15.89 | 99.43% |

+0.66pp (US) and +0.50pp (India) of pair recall for about 2.4 more candidates per entity. At perfect
precision that moves the F0.5 ceiling from 0.9950 to 0.9964, so it is a small but genuine gain, and
16.5 candidates per entity remains in the same order as the 10-15 target. Kept.

**Candidate pairs generated:** per Source-1 entity, across both sources — US **16.5**, India
**~19.5**, France **~19.4**; ~30M pairs over 1,732,544 test entities. That is the exact set the
model scores, and it is what `candidate_pairs.tsv` contains.

**How we ensured true matches were not lost: by measuring.** `src/sweep_rev.py` retrieves at k=12
and derives recall at every smaller k from one pass, which distinguishes *retrieval* failures (the
right record never returned) from *ranking* failures. On train India S2:

| config | r@1 | r@3 | r@5 | r@12 |
|---|---|---|---|---|
| maxdf 10000, r 14 | 89.70% | 92.67% | 93.61% | 94.88% |
| maxdf 30000, r 20 | 92.04% | 94.43% | 95.18% | 96.23% |
| **maxdf 60000, r 28** | **92.87%** | **95.08%** | 95.84% | 96.84% |

r@1 = 92.9% is the reason a per-record assignment works so well. Recall flattens after k=3, so we
kept k=3 and put the remaining effort into tokens and scoring.

**Final blocking recall** (held-out entities, retrieved against *all* Source-2/3 records of the
country, exactly as at test time): **US 97.56%**, **India 94.83%**; 99.84% / 99.40% of entities with
matches have at least one match reachable.

**Forward vs reverse, measured on train India:**

| | pair recall | cand / entity |
|---|---|---|
| forward (clean -> noisy, 3 unioned passes) | 89.04% | 80.71 |
| **reverse (noisy -> clean)** | **94.83%** | **13.66** |

**Negative results, kept behind flags rather than deleted.** Address prefix and skeleton tokens in
the signature: 93.35% at k=3 vs 94.43% without — they crowd genuinely rare exact tokens out of the
query's fixed token budget. Disabling length normalization: 94.59% vs 95.09% — it looked wrong in
the reverse direction, since it divides by index-side length and so penalises the long,
well-populated Source-1 records most likely to be correct, but it measured worse.

---

## 4. Matching Model

**Stage 1 — 45 pair features** (`src/features.py`), computed with no Python loop over pairs: token
set overlaps come from hashed binary CSR matrices (intersection as an elementwise sparse product)
and fuzzy ratios from `rapidfuzz.process.cpdist`, which scores aligned lists in C across threads.

- *Name:* token-set intersection, Jaccard and containment over suffix-stripped cores; `token_set_ratio`
  on the normalized name; `ratio` on the sorted core (order-invariant by construction); **char 3-gram
  cosine** (survives typos that break tokens); **consonant-skeleton similarity**; a
  **separator-stripped ratio** that catches `.com`/`@handle` aliases; **bidirectional acronym match**.
- *Address:* intersection, Jaccard, containment over the token bag; `token_sort_ratio`; separate
  overlap over **digit tokens only** (house numbers are the rarest, most discriminative component);
  **house-number equality**; a three-state postal comparison (equal / different / at least one absent).
- *IDF-weighted share:* what fraction of the Source-1 record's rare-token mass is shared. A shared
  rare token is not worth the same as a shared common one, which a Jaccard cannot express.
- *Noise flags on the candidate:* Indic script, non-ASCII, URL-like name, empty address — letting the
  model learn how far to trust the name for exactly the records where it is unreliable.
- *Competition, both directions:* the candidate's rank and gap-to-best among its entity's rivals, and
  — because a noisy record belongs to at most one entity — its **margin over its own second-best
  Source-1 option**, which is the most direct evidence that an assignment is right.

**Stage 2 — sibling/family features** (`src/stage2.py`). An entity's ~3-4 noisy copies resemble *each
other*, not only the clean original, and stage 1 cannot use that because it scores pairs in
isolation. After stage 1, each entity's high-confidence candidates form a provisional family, and
every candidate is described by its fit to that family (token overlap with the pooled family
address and name, IDF-weighted, family size, and its own score gap to the family's best). This
targets precisely the two groups diagnosis flagged: an alias-named record shares nothing with the
clean name but sits at its siblings' address, and an empty-address record shares no address but
matches its siblings' names.

Two implementation points that matter:

- **Out-of-fold stage-1 scores.** Stage 2 consumes stage-1 scores as a feature, and a stage-1 model
  is overconfident on its own training rows. Trained naively, stage 2 learns "high stage-1 score
  implies correct" far more strongly than holds at inference and under-uses its sibling evidence.
  Fit rows therefore get scores from a **5-fold rotation split by entity** (splitting by row would
  leak sibling information across the fold boundary — exactly what stage 2 exploits); validation and
  test use the single full model, matching inference.
- **Pooled, not pairwise.** Comparing each candidate against each sibling is c^2 per entity — at ~14
  candidates over millions of entities, hundreds of millions of string comparisons. Pooling the
  family into one vector makes it a single sparse product, with each member's self-contribution
  removed so a candidate is never compared against itself.

**Model type:** LightGBM, binary objective, 255 leaves, learning rate 0.05, up to 3000 rounds with
early stopping on a validation slice that is **excluded from threshold tuning**, so that estimate
stays honest. Trained on 30% of Source-1 entities (~38% sampled in total) — affordable precisely
because reverse blocking yields ~14-19 candidates per entity instead of ~84.

**Threshold and decision rule** (`src/decide.py`). Entities, not pairs, are split by a fixed hash of
the row index, so an entity's whole candidate set stays on one side — required for the per-entity
features and for the macro metric to mean anything. We then grid-search the threshold and a
**relative floor** (drop a candidate scoring below a fraction of its entity's best) on held-out
entities, maximising macro F0.5 computed exactly as the challenge defines it, singletons included,
verified against the README's worked example. Because a threshold selects a prefix of the
score-sorted pairs, the exclusivity duplicate mask is computed once and reused across the sweep.

**Constrained assignment.** Finding 3 says a Source-2/3 record belongs to at most one entity, so a
contested record is necessarily a false merge. Notably, with reverse retrieval at k=3 this pass
became nearly redundant — it moved F0.5 by 0.0003, because the model rarely puts two Source-1
records above threshold for the same noisy record. The structure of the candidate generation now
enforces what the post-hoc pass used to repair.

---

## 5. Results & Error Analysis

| version | approach | validation macro F0.5 | blocking recall | cand/entity | leaderboard |
|---|---|---|---|---|---|
| baseline | predict empty everywhere | 0.0558 | — | 0 | — |
| v1 | forward blocking, 1 stage | 0.9236 | 94.3% US / 89.0% India | ~84 | 0.904 |
| **v2** | **reverse blocking, 1 stage** | **0.9455** | **97.6% US / 94.8% India** | **~13.8** | *pending* |
| v3 | + adaptive-k, 45 features, stage 2 | *see note* | *see note* | ~16.5-19.5 | *pending* |

*(v3 figures are filled in from `work/v3_results.json` once its run completes; v3 is only shipped if
it beats v2's 0.9455 on validation, and `output/` otherwise retains v2.)*

**v1's validation overstated its leaderboard score (0.9236 vs 0.904), and we know why.** Forward
blocking was validated on an 8% slice of Source 1, so each entity competed for candidates against
only 8% of its rivals; at test time all of them compete and matches are taken by higher-scoring
entities. Reverse retrieval removes this by construction — every noisy record competes over the
whole Source-1 table — so the v2 figure should track the leaderboard far more closely. This is the
single most useful thing we learned about our own evaluation.

**Where the remaining loss sits.** At precision 1.0, recall 0.95 already gives F0.5 = 0.9896, so
blocking now costs roughly one point and the rest is the matching model. Loss attribution
(`src/analyze.py`) splits the residual into blocking misses (unrecoverable by any threshold), model
false negatives, and false positives, by country and by candidate noise type.

**Miss diagnosis** (`src/diagnose_misses.py`, train India S2: 123,577 records with a true match,
6,060 missed at k=3):

| cause | share of misses | share of all true pairs |
|---|---|---|
| outranked (reachable, scored too low) | **99.04%** | 4.86% |
| shared tokens but all above the DF cap | 0.83% | 0.04% |
| no shared token at all | 0.13% | 0.01% |

Empty-address records are 28.5% of misses against a 2.6% base rate. The misses are a *ranking*
problem, not token coverage — which is why we invested in scoring and adaptive-k rather than
character n-grams, and this measurement is what stopped us doing the latter.

**Common false positives (wrong merges).** Concentrated where an address is genuinely shared by
different businesses — office buildings and multi-tenant municipal addresses, common in the Indian
data. When several businesses share a
building and one has an alias-style name, address features agree strongly while name features carry
no signal. The per-record assignment is the main defence: at most one of those entities can own the
record.

**Common false negatives.** Dominated by blocking, not the classifier: ~2.4% (US) to ~5.2% (India)
of true pairs never reach the model. India trails US because Indic-script names transliterate to
tokens that match nothing on the clean side, leaving the address as the only route; a record with
both an Indic name and a sparse address is close to unreachable.

**France-specific investigation (and why we applied no France-specific correction).** v2 scored
0.930 on the leaderboard against a 0.9455 validation. Since validation can only cover US and India,
a natural reading is that France — 15% of test, no labels — dragged the score down, implying ~0.84
there. We tested that directly (`src/france_diag.py`) and found no support for it:

| | entities | singleton rate | mean/non-singleton | matches/entity | S2/S3 assigned |
|---|---|---|---|---|---|
| France | 259,452 | **4.40%** | 3.62 | 3.46 | 60.6% / 64.5% |
| India | 809,986 | 6.19% | 3.34 | 3.14 | 53.7% / 53.9% |
| US | 663,106 | 5.51% | 3.46 | 3.27 | 56.3% / 57.4% |
| train truth | | 5.58% | 3.67 | | |

France has the *lowest* singleton rate, the *highest* matches per entity and assigns the *largest*
share of noisy records — it is the least conservative country, not the most. (v1 did under-match
France at 9.6% singletons; reverse blocking fixed that.)

Three further checks:

- **Normalization.** Every French pattern is already handled: `rue`/`RUE` dropped, `Avenue`/`AVE` ->
  `ave`, `BD`/`Bd` -> `blvd`, `Cours` -> `crs`, `RTE` -> `rte`, `Imp` -> `imp`, accents stripped
  (`Mérignac` -> `merignac`), apostrophes handled (`l'Yser` -> `yser`, `d'Artois` -> `artois`),
  `SARL`/`SAS`/`SA`/`SCI`/`SNC` stripped, `No` dropped.
- **Unmatched France records are mostly true non-matches.** France's noise recipe includes
  near-duplicate decoys on the same street: the same business name with a different legal
  suffix at a nearby house number, or a shared name prefix with a different second word. Matching those would be false merges.
- **House numbers.** Among predicted matches, the share where house numbers differ is France 8.95%,
  US 16.67%, India 25.75%, against 22.22% in train truth. France is the strictest.

**There is also no threshold lever.** 93.3% of France entities have their best candidate scoring
above 0.95, so sweeping the threshold from 0.50 to 0.75 moves the singleton rate only 3.52% -> 4.70%
and matches per entity 3.70 -> 3.42. Reaching train's 5.58% singleton rate would require a threshold
far above 0.75 and would strip genuine matches on the way.

We therefore left France alone. The more parsimonious explanation of 0.9455 -> 0.930 is a uniform
train-to-test gap: v1 lost 1.9pp on the same comparison (0.9236 -> 0.904) for a reason that was
demonstrably *not* France-specific — the forward-blocking validation bias, which applied to every
country. Attributing the whole residual to the one country we cannot measure, and then tuning it,
would have been fitting to an unverifiable assumption.

**Calibration.** v2 predicts 5.66% singletons against a true training rate of 5.58%, and 3.15
matches per non-singleton against 3.67 — close, and v1's 7.8% singleton rate was visibly worse.
France, which contributes no training labels, profiles like the trained-on countries (9.6%
singletons, 3.10 matches per non-singleton, against US 6.0% / 3.37), which is our evidence that the
country-agnostic normalization holds up on an unseen country.

All 5,158,567 matched IDs in the v2 submission are distinct, so the disjointness property observed
in training holds exactly in our output.

---

## 6. Conclusion

The decisive move was reframing rather than tuning: because matches are 1-to-many from a clean
source, retrieving noisy->clean turns an "find all the copies" problem into a "pick the original"
problem, which improved recall, cut the candidate set six-fold, and repaired a validation bias all
at once. Everything else followed from measuring instead of guessing — a miss diagnosis that
redirected us from character n-grams to ranking, two plausible ideas that measured worse and were
dropped, and a 123x performance bug found only because a timing looked wrong.

---

## Appendix

### A. Code Artefacts

Complete pipeline in `code/business_entity_resolution/`, all source under `src/`, with a `README.md`
giving exact reproduction commands and a pinned `requirements.txt`. Entry point:
`python -m src.run_v3` runs everything and will not replace an existing submission unless the new
one validates and scores better; any single stage is independently runnable and resumable.

Runtime is about 8 hours on an 8-core 16 GB laptop, dominated by retrieval. Design points that make
that possible: TSVs read in chunks; Parquet read one column at a time; one country and one candidate
source in memory at a time, sharded by Source-1 index; retrieval blocks cut on a posting budget;
retrieval parallelised across processes with a `fork` context so the index is built once and
inherited copy-on-write (~7.3x on 8 cores). Token hashing uses `crc32` rather than Python's
`hash()`, which is randomized per process and would otherwise map tokens to different feature
columns between training and inference.

| file | role |
|---|---|
| `src/paths.py` | paths and the low-memory TSV read settings |
| `src/to_parquet.py`, `src/prepare.py` | TSV -> Parquet; normalize and shard by country |
| `src/normalize.py` | name and address normalization |
| `src/blocking_rev.py` | **reverse candidate generation (the one used)** |
| `src/blocking.py` | forward generation (v1); supplies the shared token index |
| `src/block_eval.py`, `src/sweep_rev.py`, `src/diagnose_misses.py` | recall, tuning, miss diagnosis |
| `src/features.py`, `src/stage2.py` | pair features; family features |
| `src/train_v3.py`, `src/train.py` | two-stage training with out-of-fold scores; v2 trainer |
| `src/decide.py`, `src/metric.py`, `src/truth.py` | decision rules; macro F0.5; ground truth |
| `src/predict.py`, `src/run_v3.py` | scoring and output; the gated end-to-end driver |
| `src/analyze.py`, `src/profile_data.py` | loss attribution; data profile |
| `src/make_submission.py` | builds the submission archive, validator-gated |

**No external data.** No APIs, geocoders, business registries, pretrained models or internet
augmentation of any kind — only the provided training data.

### B. Additional Results

Sweep tables in section 3, miss diagnosis and calibration in section 5. Measured negative results
are retained in the code behind flags, with their numbers, so a reviewer can re-run them:
address prefix/skeleton tokens (93.35% vs 94.43%), length normalization off (94.59% vs 95.09%), and
the forward blocking DF-cap sweep where recall moved 52.2% -> 78.1% purely by raising the cap.
