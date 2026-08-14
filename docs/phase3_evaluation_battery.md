# Phase 3 — the evaluation battery, test by test

Companion to [`phase3_preregistration.md`](phase3_preregistration.md), which fixes the *primary
endpoint and decision rules*. This file specifies **every test that will be run**, what each one
catches, and — for each — the threshold, the statistic, and whether it gates.

**The organising principle:** no single test is trusted. Each one has a known way to be fooled, and
the battery is built so that the way to fool one test is caught by another. §7 states the failure
modes explicitly and names the test that catches each.

Legend: **GATE** = failure disqualifies the arm regardless of anything else. **PRIMARY** = the
pre-registered decision endpoint. **REPORTED** = always shown, never decisive.

---

## Tier 0 — instrument validation (run FIRST, every time)

If these fail, no other number in the run means anything. This project has twice reported a result
from a dead instrument; these run before the arms, not after.

### T0.1 Positive control — real held-out RIPP cores · **GATE**
Real cores from `splits_class/RIPP/test.jsonl`, pushed through the **identical** scoring window,
gene caller and Pfam scan as the generations. Establishes the ceiling.
**Threshold:** on-class rate must be ≥0.60. If real BGCs don't score, the scorer is broken.
**Why it matters:** without a ceiling, a generated rate of 0.15 is uninterpretable — it could be
near-perfect or near-useless.

### T0.2 Negative control — dinucleotide-shuffled real cores · **GATE**
Same cores, shuffled. Measures the **instrument's own false-positive rate**.
**Threshold:** on-class rate ≤0.05. Measured historically: antiSMASH `is_bgc` FPR **0.000**, but the
retired any-Pfam proxy was **0.960** — which is exactly why this control is mandatory.
**Why shuffled and not random:** random DNA has the wrong composition and is too easy to reject.

### T0.3 Scorer dynamic range · **GATE**
`containment(x, x) == 1.0` and a 5%-mutated copy lands intermediate (measured 0.47).
**Why:** the novelty floor is *legitimately zero* at k=21, so "everything reads 0" cannot be
distinguished from "the function is broken" by looking at the output. Only a positive control can.
This exact confusion cost a day on the GenomeOcean leakage test.

---

## Tier 1 — the primary endpoint

### T1.1 `on_class_rate` · **PRIMARY**
Fraction of generations containing ≥1 RIPP-defining biosynthetic Pfam domain (`best_bio_bits > 0`,
class-aware), on the fixed 2,000-nt scoring window.
**Statistic:** proportion + Wilson 95% CI; Fisher's exact vs the A0 floor; Holm-corrected across arms.
**Why a rate, not a mean:** `best_bio_bits` has median 0.00 in every arm ever measured and a heavy
tail, so its mean is an outlier detector. The headline "frame −10.201" was one record out of 24.

---

## Tier 2 — the validated ladder (REPORTED, never decisive)

All AUROC values are against the independent antiSMASH outcome, n=120 with 44 detections.

| test | what it measures | AUROC | role |
|---|---|---|---|
| **T2.1** `best_bio_bits` | strength of the best biosynthetic domain match, bits | **0.950** | rank test (Mann-Whitney) only |
| **T2.2** `n_bio_domains` | how many biosynthetic domains at all | 0.919 | mean + rank test |
| **T2.3** `bio_span_frac` | how far apart they sit — is it a CLUSTER or one lucky gene | 0.896 | mean + rank test |
| **T2.4** `antismash_detect` | the gold-standard binary gate | — | rate, always reported even at 0 |
| **T2.5** `antismash_correct_class` | detected AND assigned to RIPP | — | rate, the true goal |

T2.4/T2.5 are reported at every arm **including when zero**, because their floor (~0.012–0.15 de
novo) is too low to power. Reporting them only when non-zero would be selective.

---

## Tier 3 — novelty and diversity · **ALL GATES**

Every ladder rung above is maximised by copying training data. These are the tests that stop that,
and two of them are **new in Phase 3** because the existing battery had holes.

### T3.1 Nucleotide novelty — k=21 containment vs the RIPP training set · **GATE**
Cut each generation into overlapping 21-mers; report the fraction also present in the single most
similar training record.
**Thresholds:** ≥0.95 `FAIL_memorized`; ≥0.80 `WARN`; below `PASS_novel`.
**Known blind spot:** synonymous codon changes make a sequence novel at the nucleotide level while
the encoded protein is unchanged. That is what T3.2 is for.

### T3.3 Intra-set diversity — pairwise containment AMONG generations · **GATE** · ⚠️ NEW
All-vs-all k=21 containment **within** an arm's own output, plus a distinct-cluster count at 0.80.
**Threshold:** ≥90% of generations must be distinct from each other; median pairwise containment
<0.30.
**Why this is a real hole in the old battery:** every novelty check we have compares generations to
**training data**. A model that emits one good sequence a thousand times passes all of them. Mode
collapse would have been invisible, and it becomes *more* likely with a shared seed — which is
exactly what the seeding arms introduce.

### T3.2 Protein novelty — AAI of translated ORFs vs training proteins · **GATE** · ⚠️ NEW
MMseqs2 search of predicted ORFs against a database built from the RIPP **training** set's proteins
(`check_protein_homology` already accepts `db_path`; the DB is the new part).
**Report:** best average amino-acid identity per generation, and the distribution.
**Threshold:** median best-AAI < 0.95. Flag anything ≥0.98 as a probable paraphrase.
**Reference:** the phage paper reported AAI **as low as 63%** to natural proteins as its novelty
evidence — protein-level identity is the standard a reviewer in this field will expect, and
nucleotide-level containment alone does not supply it.

### T3.4 Seed attribution — containment vs the specific seed source · **GATE**, seeded arms only
For every seeded generation, containment against (a) the exact seed, (b) the record the seed came
from, (c) all other training records.
**Threshold:** (b) must not exceed (c) by more than 0.10.
**Why:** the whole reviewer objection is "your result is the seed." This measures it directly per
record rather than arguing about it. The seed itself is excluded from the scored span, verified per
record — `tests/test_scored_span.py` pins the analogous property for the existing generator.

---

## Tier 4 — sanity and structure (REPORTED)

| test | measures | reference values | catches |
|---|---|---|---|
| **T4.1** coding density | fraction of bases in predicted ORFs | real 0.97, de novo 0.74–0.82 | output that is not gene-like at all |
| **T4.2** `max_orf_aa` | longest reading frame | real 729, de novo 448 | structural only — **DEMOTED**, AUROC 0.709 and r=0.051/−0.120 within de novo |
| **T4.3** GC content + dinucleotide composition | vs the RIPP training distribution | — | a model matching composition without content, or drifting off it |
| **T4.4** `hit_eos` rate + generated length distribution | does the model TERMINATE | **currently 0/204 — has never worked** | whether length is learned or imposed |
| **T4.5** `n_orfs` | ORF count | real 2.12, de novo 4.02 (INVERTED — more is worse) | fragmentation |

T4.4 is the one to watch on the first RIPP run: whole-record training gives `|END|` its first clean
signal, and if it fires, generation length becomes an output rather than a hyperparameter.

---

## Tier 5 — specificity (REPORTED)

### T5.1 `bio_fraction` — biosynthetic share of ALL Pfam signal
Real cores **0.836**, de novo **0.100**. This is the number that identified the central finding:
the model writes real protein of the **wrong kind**. A rise here without a rise in T1.1 means better
targeting; a rise in T1.1 without a rise here means it got luckier, not better.

### T5.2 `best_any_bits` — best match to ANY Pfam, biosynthetic or not
The contrast with T2.1 is the whole diagnosis. At n=152 the frame arm was **21.94 vs 35.18**
(p=0.004) — significantly *less* recognisable protein, not merely less biosynthetic. Reporting
T2.1 without T5.2 would have hidden that.

### T5.3 Off-class detection rate
Of generations antiSMASH detects, the fraction assigned to a class **other** than RIPP.
**Reference:** on REAL cores at 3 kb, **31.4%** of detections are off-class — so the class call
genuinely discriminates, and a high concordance in our generations is a fact about the generations,
not about a lax ruler.

---

## Tier 6 — the composite that actually matters

### T6.1 Joint pass rate · **the headline deliverable**
Per record, the fraction of generations that **simultaneously** satisfy: on-class (T1.1) **AND**
nucleotide-novel (T3.1) **AND** protein-novel (T3.2) **AND** distinct from other generations (T3.3).

**Why this is separate from everything above, and why it is the number to report:** an arm can post
30% on-class and 100% novel while the on-class ones are exactly the non-novel ones. Marginal rates
cannot detect that; only the per-record intersection can. This is the analogue of the phage paper's
**302 candidates from hundreds of thousands** — the count of sequences that survive *every* filter
at once, which is the only number that describes what you could actually take forward.

---

## 7. Coverage — how the battery is fooled, and what catches it

| a model could… | caught by |
|---|---|
| recite training DNA | T3.1 nucleotide novelty |
| paraphrase it with synonymous codons | **T3.2 protein novelty** (T3.1 alone would pass it) |
| emit one good sequence many times | **T3.3 intra-set diversity** (all training-comparison checks pass it) |
| succeed only because of the seed | T3.4 seed attribution + the A0 no-seed arm |
| produce junk that hits a domain by chance | T0.2 negative control, T4.1 coding density |
| produce real protein of the wrong kind | T5.1 / T5.2 — the central Phase-2 finding |
| produce right domains, not clustered | T2.3 `bio_span_frac` |
| win by generating more sequence | fixed 2,000-nt scoring window (pre-reg §5) |
| win on one lucky sequence | T1.1 is a rate; per-record distributions reported |
| pass each gate separately but never together | **T6.1 joint pass rate** |
| be right but underpowered | pilot-set `n` (pre-reg §8), Wilson CIs, Holm correction |
| look good because the scorer is broken | T0.1 / T0.2 / T0.3, run first |

**Two of these — T3.2 and T3.3 — did not exist before Phase 3.** The old battery could be passed by
a model that paraphrased training data at the codon level, or by one that produced a single sequence
repeatedly. Both are live risks specifically because Phase 3 introduces shared seeds.

## 8. Implementation status

| test | status |
|---|---|
| T0.1, T0.2 | `scripts/make_positive_control.py`, `make_negative_control.py` — exist |
| T0.3 | in `quantify_smc_leakage.py`; **lift into the shared scorer** |
| T1.1, T2.1–T2.3, T4.1–T4.5, T5.1–T5.2 | `ladder_audit.one()` / `score_ladder.py` — exist |
| T2.4, T2.5, T5.3 | `check_antismash` in the eval suite — exists |
| T3.1 | `check_kmer_novelty` / `memorization_check.py` — exists |
| **T3.2** | `check_protein_homology` exists but needs a **RIPP-training protein DB** — TO BUILD |
| **T3.3** | **TO BUILD** — all-vs-all containment within an arm |
| **T3.4** | **TO BUILD** — per-record seed attribution |
| **T6.1** | **TO BUILD** — the per-record intersection |
