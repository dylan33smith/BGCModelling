# Phase 3 — PRE-REGISTRATION (RIPP)

**Written 2026-08-14, BEFORE any Phase-3 model is trained.** Everything below is fixed in advance.
Changing any of it after seeing a result makes the result exploratory, and it must be reported as
exploratory. This file exists because this project has, twice in two days, read a null from a test
that could not have produced anything else.

---

## 0. Why this document is worded this way

Each clause below traces to a specific error already made in this project. They are named so the
clause cannot be quietly dropped later.

| clause | the error it prevents | where it happened |
|---|---|---|
| primary endpoint is a RATE | the mean of a zero-inflated metric tracks the luckiest draw | "frame −10.201" was one record of 24 |
| n fixed by power analysis | an underpowered null read as a closure | n=24, power 0.15 for a doubling |
| intervention must be verified to land | a null with no treatment delivered | weighted arm: 3× and 10× gave identical in-domain loss |
| identical generation length, fixed scoring window | length inflates detection mechanically | 6 kb cap censored the frame arm only |
| post-processing applied to controls too | comparing processed generations to raw reality | flagged during the extract-genes discussion |
| novelty is a gate, not a metric | every ladder rung is maximised by copying | ECTOINE: 85% of held-out is near-duplicate |
| no pooling across modes | batched vs sequential is different conditioning | left-padding makes a padded prompt a different prompt |

---

## 1. Target and substrate

- **Class: RIPP.** 8,129 train / 579 test, 6,848 distinct genomes, median 1,931 nt, 89% under 8 kb,
  **43% near-duplicate loss** (the lowest of the viable classes, i.e. the most internally diverse).
  Chosen over TERPENE despite terpene being shorter: terpene's de novo detection is 0.079 against
  RIPP's 0.158, and diversity is comparable (46% vs 43%).
- **Substrate: Evo2 1B.** All testing. 7B confirms anything publishable. GenomeOcean held.
- **Data: strict antiSMASH cores, `--flank 0`.** NO regulatory context. Any claim must say
  "biosynthetic core", never "cluster ready to express".

## 2. Primary endpoint — ONE number, fixed now

> **`on_class_rate` = the fraction of generated sequences containing at least one RIPP-defining
> biosynthetic Pfam domain**, measured on a fixed 2,000-nt scoring window (§5).
>
> **THE ACCESSION SET IS `OBLIGATE_DOMAINS['RIPP']` — 8 Pfams: PF00881, PF02624, PF03070,
> PF04055, PF05114, PF05402, PF13353, PF14028.** Named explicitly because the first
> implementation did not use them.

⚠️ **IMPLEMENTATION DEFECT, FOUND 2026-08-14 AFTER THE FIRST ARM AND CORRECTED.**
`ladder_audit.one()` accepts a `cls` argument but uses it ONLY for the NRPS/PKS architecture rung;
its `bio` score is computed against a **fixed global ~91-model biosynthetic set**. Proof: rescoring
one arm under RIPP / NRPS / PKS / TERPENE returns **4/50 every time**. So every number first
reported as "RIPP on-class" actually meant *"contains ANY biosynthetic domain"*.
⇒ The endpoint text above was correct; the code was not. Scoring MUST subset the HMM to the RIPP
accessions (`_subset_hmm(pfam, out, set(OBLIGATE_DOMAINS['RIPP']))`). *Rule: a pre-registered
endpoint is only as fixed as the code that computes it — verify the implementation measures the
quantity the document names, on a case where a wrong implementation would give a different answer.*

**Why a rate and not `best_bio_bits` itself:** `best_bio_bits` is zero-inflated (median 0.00 in
every arm ever measured) and heavy-tailed, so its mean is an outlier detector. A rate has an exact
binomial CI and an exact test, and cannot be moved by one lucky sequence.

**Why this rung and not antiSMASH detect+class:** the strict gate reads ~0.012–0.15 de novo, which
cannot be powered at any n we can afford. It is reported as a SECONDARY outcome on every arm, always,
including when it is zero.

## 3. Secondary outcomes — reported always, decisive never

`antismash_detect`, `antismash_correct_class`, `n_bio_domains`, `bio_span_frac`, `best_bio_bits`
(rank test only), `max_orf_aa` (structural diagnostic), `hit_eos` rate and generated length.
None of these may be promoted to primary after the fact.

## 4. Hard gate — novelty

**`max k=21 containment` against the RIPP TRAIN set must be < 0.80** for an arm to be eligible at
all. An arm that improves the primary endpoint while failing this has learned to recite and is
reported as a failure, not a result. Containment is reported for every arm regardless of outcome.

⚠️ The phage paper (Hie et al., *Science* 2026) found seed length trades directly against
memorisation — 4–8 nt optimal, longer seeds caused the model to reproduce known sequences. Our seed
arms are therefore the highest-risk arms for this gate, and it is why the gate is absolute.

## 5. Generation and scoring protocol — frozen

1. **All arms use identical `max_new_tokens`.** No arm gets more room than another.
2. **Scoring uses a FIXED 2,000-nt window** (positions 1–2,000 of the generated sequence, excluding
   any seed) for every arm and every control. Length therefore cannot bias detection: it is the same
   for everything scored. Termination (`hit_eos`) and full generated length are reported separately
   and NEVER change the scored span.
3. **Any post-processing (gene extraction, trimming) applied to generations is applied identically
   to the real-core positive control.** No exceptions.
4. **Identical prompts, decoding parameters and RNG seed across arms.** Batched throughout; batched
   and sequential outputs are never pooled.
5. **The seed is excluded from the scored span**, and this is verified per record, not assumed
   (`tests/test_scored_span.py` pins the analogous property for the existing generator).

## 6. Controls — mandatory, they bracket the range

| control | role |
|---|---|
| **base 1B, no adapter, no seed** | absolute floor |
| **general Phase-2 adapter, no seed** | does specialising beat generalising? |
| **real held-out RIPP cores** | ceiling, scored through the identical pipeline and window |
| **shuffled/negative** | false-positive rate of the instrument itself |

## 7. Arms

| arm | seed | note |
|---|---|---|
| A0 | none | tests whether seeding is needed AT ALL — run first |
| A1 | real exemplar prefix | performance ceiling, novelty floor (the current 0.283 mode) |
| A2 | **mosaic** — fragments from k different real clusters, new k-subset per sample | no new machinery |
| A3 | **consensus/centroid prefix**, per-sample bootstrap over exemplar subsets | the phage-paper approach |

**Seed length is itself a variable, swept at 4, 8, 20, 100 nt** — following the phage paper's finding
that 4–8 nt was optimal and longer caused memorisation. Our historical seeds were ~500 nt, which is
~100× longer than their optimum, and that is a plausible reason our seeded results carry a novelty
risk that theirs did not.

## 8. Sample size — set by a pilot, before the real run

`n` is NOT chosen now, because it depends on the floor. Procedure, fixed:

1. **PILOT** (n=50, control arms only) to estimate the floor rate `p0` for RIPP under this protocol.
2. **Compute n** for 80% power at α=0.05 two-sided to detect `p0 → 2·p0`.
3. **Record that n in this file**, then run. The pilot data is NOT reused in the confirmatory
   analysis.

For reference, at p0=0.15 a doubling needs ~120/arm; at p0=0.05 it needs ~430/arm. If the required n
is unaffordable, the honest response is to say the experiment cannot be run as designed — NOT to run
it underpowered and interpret the null.

### 8.1 PILOT RESULT (2026-08-14) and the fixed n

| arm | n | on-class | rate | 95% CI |
|---|---|---|---|---|
| **real cores (CEILING)** | 50 | 29/50 | **0.580** | [0.442, 0.706] |
| shuffled (instrument FPR) | 50 | 0/50 | **0.000** | [0.000, 0.071] |
| **base 1B (FLOOR)** | 50 | 0/50 | **0.000** | [0.000, 0.071] |
| general Phase-2 adapter | 50 | 4/50 | 0.080 | [0.032, 0.188] |

**THE FLOOR IS EXACTLY ZERO.** The pilot script's power calculation substituted `p0 = 1/n = 0.02`
as a divide-by-zero guard and printed 376–1,140 per arm. That is wrong for a *true* zero floor:
against 0/n, Fisher's exact reaches p<0.05 at **6 successes regardless of n**.

**⇒ CONFIRMATORY n = 150 PER ARM.** At n=150 vs a zero control: k≥6 is significant, giving power
**0.89** for a true rate of 0.06 and **0.98** at 0.08. n=100 gives only 0.56 at 0.06, which is the
same underpowered regime as Phase 2. n=150 × 4 arms × 4 seed lengths is affordable at ~2 kb/sample.

**⚠️ THE CEILING IS 0.580, NOT 1.0, AND MY PRE-REGISTERED THRESHOLD WAS ≥0.60 — A MARGINAL MISS.**
Recorded rather than adjusted: the 95% CI [0.442, 0.706] contains 0.60, so this is not a significant
failure, but the threshold was set without data and RIPP sits right at it. The substantive point is
that **only 58% of REAL RIPP cores carry a detectable RIPP-specific biosynthetic Pfam domain** —
RiPP precursor peptides are short and poorly covered by Pfam. Every generated rate must therefore be
read against **0.58**, not against 1.0. The threshold is NOT being moved to accommodate the result.

**Instrument FPR is 0.000** — the negative control is clean, so a non-zero generated rate cannot be
explained by the scorer.

### 8.2 ⚠️ THE PILOT NUMBERS ABOVE ARE THE *GENERIC* METRIC — SUPERSEDED BY §8.3

Every rate in the 8.1 table was computed with the defective scorer, so it means "contains any
biosynthetic domain". They are kept as the record of what was run, not as the endpoint.

### 8.3 THE ENDPOINT, SCORED CORRECTLY (RIPP-specific, 2026-08-14)

| arm | RIPP-specific | rate | 95% CI |
|---|---|---|---|
| base 1B (floor) | 0/50 | **0.000** | [0.000, 0.071] |
| **general all-class adapter** | **0/50** | **0.000** | [0.000, 0.071] |
| **A0 — RIPP-only, no seed** | **4/150** | **0.027** | [0.010, 0.067] |
| real RIPP cores (CEILING) | 22/50 | **0.440** | [0.312, 0.577] |

**THE CONCLUSION INVERTS.** Under the generic metric the general adapter scored 0.080 against A0's
0.040 and the fine-tune looked like a failure. Under the pre-registered RIPP-specific metric the
general adapter scores **exactly zero** — its 0.080 was other classes' biosynthetic content, which
is what an all-22-class model should produce — and **A0 is the only non-real arm producing RIPP
machinery at all.**

Not yet significant: 4/150 vs the two controls pooled (0/100) gives **Fisher p = 0.152**. The
direction is what changed, not the significance. The ceiling is **0.440**, so A0 reaches ~6% of what
real RIPP clusters score.

⇒ **The `n=150` in §8.1 was sized against a floor measured with the wrong scorer.** The correct
floor is 0/50 for BOTH controls, which is the same zero, so n=150 still applies — but the effect
size to detect is now against a true zero rather than against 0.080.

## 9. Decision rules — fixed in advance

- **An arm SUCCEEDS** iff its `on_class_rate` exceeds the A0 floor by Fisher's exact p < 0.05 **and**
  it passes the novelty gate (§4).
- **An arm's NULL is interpretable** only if (a) the analysis was powered per §8, and (b) the
  intervention was independently verified to have changed the model. For seed arms, verification =
  the seeded generations differ from unseeded generations on some measured axis; a seed that changes
  nothing is not a treatment.
- **If no arm succeeds at adequate power**, the conclusion is that the approach fails for RIPP on
  this substrate. The response is to change substrate or target — **not** to re-cut the metric.
- **Multiple comparisons:** with 4 arms × 4 seed lengths, report Holm-corrected p alongside raw p.
  The primary claim rests on the corrected value.

## 10. What would make this exploratory rather than confirmatory

Any of: changing the primary endpoint; changing the scoring window; adding arms after seeing
results; pooling across generation lengths or batching modes; reporting an uncorrected p as the
headline; dropping the novelty gate. If any occurs, the result is labelled exploratory in every
document that reports it.
