# RESULTS — the 48 arms, 2026-09-16

Produced by the code at tag `results-frozen-2026-09-16` (commit `27e13aa`). Every number below
is read from `/data2/ds85/bgcbench/runs/` by `bgcbench/run/results_table.py`.

⚠ **This file did not exist until 2026-09-16, and `FINDINGS.md` is not a substitute for it.**
FINDINGS describes the generation of arms frozen on 2026-09-08/09/10 — before the retrain — and
contains no mention of `_fx`. Its instrument and limitation sections transfer; its *rates* do
not. See `DEFECTS.md` for what invalidated them.

## What makes these rows comparable

Configuration is deliberately **not** matched across substrates (SPEC §14A): rank, injection
sites, α, decoding and prefix are chosen per substrate, and per class, by measurement. What is
held identical is the *measurement*, and `results_table.py` asserts all four on every run:

| invariant | value |
|---|---|
| `n_per_class` | 200, every arm |
| `budget_nt` | 8,192 nt, every arm, none exceeding it |
| scoring config | one antiSMASH hash, `ee8c025c1593` (v8.0.4, `--minimal --genefinding-tool prodigal --minlength 1`) |
| novelty gate | passed by every generation of all 48 arms |

Substrate treatments, for the record — these differ *by design*:

| | Evo2-1B | GenomeOcean-4B |
|---|---|---|
| prefix | GTDB lineage (its native pretraining format) | bare sequence (never pretrained on lineages) |
| tokenizer | byte-level, 1 nt/token | BPE, ~4.8 nt/token |
| attention sites | 4 of 25 blocks | 24 of 24 layers |
| decoding | `top_k=4` (retains 0.9999 of mass over 4 letters) | `top_k=0`, full 4,096 vocabulary |
| LoRA rank | 16 | 16 (inherited, not gated — see DEFECTS) |

## 1. Calibration — what licenses reading a zero

Most arms read near zero. These three measurements are what separate "the model cannot do it"
from "the instrument cannot see it".

```
QUESTION  what are the ends of the measurement scale, and is the harness wired correctly?
MODEL     none — instrument and harness only
ARM       G5 = real held-out BGC cores; G1 = real non-BGC bacterial DNA;
          ORACLE = real cores pushed through the COMPLETE arm path
METRIC    on-target detection rate. G5/oracle should be ~1.0, G1 ~0.0
UNIT      fraction
```

| class | G5 ceiling | G1 false positive | oracle (full arm path) |
|---|---|---|---|
| TERPENE | 82/82 = **1.000** | 0/300 = **0.000** | 60/60 = **1.000** |
| RIPP | 82/82 = **1.000** | 0/300 = **0.000** | 60/60 = **1.000** |
| ARYLPOLYENE | 82/82 = **1.000** | 0/300 = **0.000** | 60/60 = **1.000** |
| REDOX_COFACTOR | 81/81 = **1.000** | 0/300 = **0.000** | 60/60 = **1.000** |

antiSMASH detects every real core and labels its class correctly; it invents nothing in 1,200
ordinary coding intervals; and the arm path — novelty gate, record builder, rates, confusion,
lift — preserves that end to end. **So a 0/200 arm is a statement about the model.**

⚠ These were measured on the pre-§12.A3 split (2026-09-07), i.e. the same instrument and the
same `ee8c025c1593` config as the 48 arms, but an earlier snapshot of the held-out records.
Both are saturated at 1.000 and 0.000, so re-deriving them was judged not worth the compute.

## 2. De novo — the two models are indistinguishable

```
QUESTION  unaided, how often does each weight state produce its own BGC class?
MODEL     rows labelled per substrate
ARM       de novo: no seed, no steering. W0 = base weights; W1n = pooled class-balanced
          adapter; W2_<CLASS> = per-class adapter
METRIC    TWO endpoints, both reported. DETECTED = antiSMASH called ANY BGC ("is it a
          cluster at all?"). ON-TARGET = it called the row's OWN class ("is it the right
          cluster?"). on-target <= detected always; the gap is off-class production.
          p = Fisher exact, two-sided, Evo2 vs GenomeOcean on the same class
UNIT      counts out of n=200
```

| class | Evo2 detected | Evo2 on-target | GO detected | GO on-target | p (det) | p (on-tgt) |
|---|---|---|---|---|---|---|
| ARYLPOLYENE | 16 | 16 | 21 | 21 | 0.49 | 0.49 |
| TERPENE | 6 | 6 | 4 | 4 | 0.75 | 0.75 |
| REDOX_COFACTOR | **6** | 3 | **3** | 1 | 0.50 | 0.62 |
| RIPP | **1** | 0 | 1 | 1 | 1.00 | 1.00 |
| W1n pooled | 3 | 3 | 1 | 1 | 0.62 | 0.62 |
| W0 base | 0 | 0 | 0 | 0 | — | — |

**Class distribution of the W1n de novo positives.** The pooled arm has no class channel, so
"on-target in row C" simply means antiSMASH called that generation class C:

* **Evo2 — 3 detected of 200, all 3 on-target for some class** — RIPP 2 · ARYLPOLYENE 1 ·
  TERPENE 0 · REDOX_COFACTOR 0
* **GenomeOcean — 1 detected of 200, on-target** — ARYLPOLYENE 1 · TERPENE 0 · RIPP 0 ·
  REDOX_COFACTOR 0

Here detected and on-target agree: the pooled arm has no class channel, so every detection is
"on-target" for whichever class antiSMASH assigned it. The two columns only diverge for a
CLASS-BEARING arm, which can be conditioned on one class and produce another.

⚠ Both arms produced **12 and 4 raw detections** respectively (Evo2 3 generations × 4 rows,
GO 1 × 4): the same handful of sequences counted once per class row. Only the on-target column
is a count of distinct generations, which is why the pooled row reads 3 and 1 rather than 12
and 4. At these counts the distribution is not interpretable — it is 3 and 1 sequences.

**No de novo contrast is significant.** Both models sit near the floor unaided, and the
apparent GenomeOcean lead on ARYLPOLYENE (21 vs 16) is noise at this n.

Their *failure modes* differ, and that is the readable de novo finding. Evo2's base model never
terminates — hit-EOS 0.01, every generation running to the full 8,192 nt, 0 detections in 1,600
draws. GenomeOcean's base terminates 99% of the time but at a 684 nt median, too short to carry
a cluster. Same zero, opposite causes.

## 3. Seeding — the one large effect, and Evo2 wins it

```
QUESTION  how much does continuing a real 64 nt core prefix add, and does it differ by model?
MODEL     rows labelled per substrate
ARM       seeded (_S1) vs de novo, same weight state. seed_len_nt = 64 is shared TASK
METRIC    on-target and DETECTED, both. p(within) = seeded vs de novo, same model.
          p(across) = Evo2 vs GenomeOcean, both seeded
UNIT      counts out of 200, except the pooled rows (n as marked)
```

**On-target** — the right class:

| class | Evo2 de novo → seeded | p (within) | GO de novo → seeded | p (within) | **p (across, seeded)** |
|---|---|---|---|---|---|
| ARYLPOLYENE | 16 → **73** | 4.0e-12 | 21 → **50** | 2.1e-04 | **0.017** |
| TERPENE | 6 → **41** | 3.2e-08 | 4 → **19** | 2.0e-03 | **0.0030** |
| RIPP | 0 → 5 | — | 1 → 2 | — | 0.45 |
| REDOX_COFACTOR | 3 → 3 | — | 1 → 1 | — | 0.62 |
| W1n pooled | 3/200 → **94/800** | — | 1/200 → **60/800** | — | **0.0050** |

**Detected** — any BGC at all. Same conclusions, so they do not depend on which endpoint you read:

| class | Evo2 de novo → seeded | p (within) | GO de novo → seeded | p (within) | **p (across, seeded)** |
|---|---|---|---|---|---|
| ARYLPOLYENE | 16 → **73** | 4.0e-12 | 21 → **50** | 2.1e-04 | **0.017** |
| TERPENE | 6 → **41** | 3.2e-08 | 4 → **19** | 2.0e-03 | **0.0030** |
| RIPP | 1 → 5 | 0.22 | 1 → 2 | 1.00 | 0.45 |
| REDOX_COFACTOR | **6 → 11** | 0.32 | **3 → 6** | 0.50 | 0.32 |
| W1n pooled | 3/200 → **101/800** | — | 1/200 → **66/800** | — | **0.0053** |
| W0 base | 0/200 → 0/800 | — | 0/200 → **9/800** | — | — |

**Class distribution of the W1n seeded positives.** Each class contributes its own 200 seeded
generations here, so these are four independent 200-draw cells, not one pooled 800:

* **Evo2 — 94 on-target of 800 (101 detected)**
  * ARYLPOLYENE **62** on-target of 200 · 62 detected
  * TERPENE **27** of 200 · 27 detected
  * REDOX_COFACTOR **3** of 200 · **10 detected** — 7 off-class
  * RIPP **2** of 200 · 2 detected
* **GenomeOcean — 60 on-target of 800 (66 detected)**
  * ARYLPOLYENE **50** on-target of 200 · 51 detected
  * TERPENE **8** of 200 · 8 detected
  * RIPP **2** of 200 · **4 detected** — 2 off-class
  * REDOX_COFACTOR **0** of 200 · **3 detected** — all 3 off-class

⚠ **The pooled totals are dominated by one class.** ARYLPOLYENE alone is 62 of Evo2's 94 (66%)
and 50 of GenomeOcean's 60 (83%); with TERPENE it is 95% and 97%. RIPP and REDOX_COFACTOR
contribute 5 and 2. So the pooled `94 vs 60` contrast — and its p=0.0050 — is very largely an
ARYLPOLYENE and TERPENE result restated, not independent evidence from four classes. Read it
alongside the per-class rows above rather than as a separate finding.

⚠ **Precision differs by class and only the pooled row hides it.** Evo2's REDOX_COFACTOR cell
detected 10 but was on-target 3 (precision 0.30) — seven generations were called as some other
class. GenomeOcean's RIPP cell detected 4 for 2 on-target (0.50) and its REDOX_COFACTOR
detected 3 for **0** on-target (0.00). Every other cell is at or near 1.00. That is why the
detected and on-target totals differ (Evo2 101 vs 94, GO 66 vs 60).

Two findings, both significant:

1. **Seeding is a large, real effect** on the two classes where anything happens at all —
   4.6× on Evo2 ARYLPOLYENE, 6.8× on Evo2 TERPENE.
2. **Evo2 converts a seed better than GenomeOcean does** — ARYLPOLYENE p=0.017, TERPENE
   p=0.0030. This *reverses* the de novo ordering, where GenomeOcean nominally led
   ARYLPOLYENE, and it is the clearest cross-model result in the benchmark.
   ⚠ The pooled arm's p=0.0050 is **not a third independent confirmation**: 95% of Evo2's 94
   and 97% of GenomeOcean's 60 are ARYLPOLYENE and TERPENE, so the pooled contrast largely
   restates the two above. Count this as two findings, not three.

⚠ Seeded pooled arms are n=800 (200 per class, each seeded from its own class) because a seed
carries the class. De novo pooled arms are n=200 generations scored against all four class
rows — summing their `per_class` gives a false n=800 and 4× the detections.

## 4. Activation steering — nothing, on either model, anywhere

```
QUESTION  does injecting a derived class direction change yield, and is it the direction's
          CONTENT rather than the perturbation?
MODEL     rows labelled per substrate
ARM       I1 = derived direction at that class's own chosen α and site set;
          I1rand = random unit vectors, SAME norm, SAME sites, SAME α (SPEC §6.3);
          I1xS1 = steering composed with seeding
METRIC    on-target, with DETECTED in parentheses where the two differ; p = Fisher exact,
          two-sided, on the on-target endpoint
UNIT      counts out of 200
```

| model | class | α / sites | unsteered | I1 | p | I1rand | p (I1 vs rand) | seeded | I1xS1 | p |
|---|---|---|---|---|---|---|---|---|---|---|
| Evo2 | ARYLPOLYENE | 0.1 / 4 of 4 | 16 | 23 (det 24) | 0.31 | 16 | 0.31 | 73 | 70 | 0.84 |
| Evo2 | TERPENE | 0.4 / 1 of 4 | 6 | 2 | 0.28 | 0 | 0.50 | 41 | **3** | **<0.001** |
| Evo2 | RIPP | 0.4 / 4 of 4 | 0 (det 1) | 0 | 1.00 | 0 | 1.00 | 5 | 0 | 0.061 |
| Evo2 | REDOX_COFACTOR | 0.075 / 4 of 4 | 3 (det 6) | 0 (**det 4**) | 0.25 | 0 (det 1) | 1.00 | 3 (det 11) | 3 (**det 12**) | 1.00 |
| GO | ARYLPOLYENE | 0.4 / 1 of 24 | 21 | 25 | 0.64 | 16 | 0.19 | 50 | 65 | 0.12 |
| GO | TERPENE | 0.05 / 8 of 24 | 4 | 1 | 0.37 | 4 | 0.37 | 19 | 26 | 0.34 |
| GO | RIPP | 0.1 / 1 of 24 | 1 | 0 | 1.00 | 0 | 1.00 | 2 | 3 | 1.00 |
| GO | REDOX_COFACTOR | 0.3 / 24 of 24 | 1 (det 3) | 0 | 1.00 | 0 | 1.00 | 1 (det 6) | 0 | 1.00 |

⚠ **Evo2 REDOX_COFACTOR is the one cell where steering moves the DETECTION endpoint while
leaving on-target at zero.** I1 produces 4 BGCs where the random control produces 1, and
I1xS1 produces 12 where seeding alone produces 11 — and *none of them* is a REDOX_COFACTOR.
Whatever the direction is doing there, it is not making the target class.

**Of 24 steering contrasts, exactly one reaches significance, and it is negative:** steering
Evo2 TERPENE at α=0.4 collapses the seeded arm from 41/200 to 3/200. Every I1-vs-unsteered and
every I1-vs-random comparison is null.

⚠ **The random control is norm-matched, and FINDINGS §20 established that is the wrong
invariant.** §6.3 specifies matching ‖d‖; §20 measured that the derived direction tolerates 3×
the magnitude before generation degrades, because it is by construction a direction the model
already moves along. A norm-matched random vector is therefore a more destructive
perturbation — not a control but a second, broken arm. §20's fix (give the control its own α
sweep, run it at its own health ceiling) was **not** carried into this grid. Measured
consequence:

| cell | I1 coding density | I1rand coding density | control usable? |
|---|---|---|---|
| Evo2 TERPENE | 0.9667 | **0.4854** (hit-EOS 0.96 → 0.03) | no |
| Evo2 RIPP | 0.9916 | **0.6306** (0.91 → 0.06) | no |
| GO REDOX_COFACTOR | 0.9992 | **0.7092** | no |
| the other five cells | 0.943–0.988 | 0.949–0.985 | yes |

Three of eight controls broke the model rather than testing it. **The confound lands where the
results are already null** — all three are 0-or-2-of-200 for the real arm too — and the two
cells carrying any signal (both ARYLPOLYENE) have healthy controls. So it is a disclosable
limitation, not a reason to re-run.

## 4b. Off-class production — where detected and on-target diverge

In most cells the two endpoints are identical: the model either makes the right cluster or makes
nothing. **The exception is REDOX_COFACTOR, in every arm of both models**, and it changes what a
zero there means.

```
QUESTION  when a model produces a recognisable BGC, is it the class it was conditioned on?
MODEL     rows labelled per substrate
ARM       every cell where detected > on-target
METRIC    detected / on-target, and precision = on-target / detected
UNIT      counts out of 200 (W1n seeded out of 800)
```

| model | arm | detected | on-target | precision |
|---|---|---|---|---|
| Evo2 | W2 REDOX de novo | 6 | 3 | 0.50 |
| Evo2 | W2 REDOX seeded | 11 | 3 | 0.27 |
| Evo2 | I1 REDOX | 4 | **0** | **0.00** |
| Evo2 | I1xS1 REDOX | 12 | 3 | 0.25 |
| Evo2 | W2 RIPP de novo | 1 | **0** | **0.00** |
| Evo2 | I1 ARYLPOLYENE | 24 | 23 | 0.96 |
| Evo2 | W1n seeded | 101 | 94 | 0.93 |
| GO | W2 REDOX de novo | 3 | 1 | 0.33 |
| GO | W2 REDOX seeded | 6 | 1 | 0.17 |
| GO | W1n seeded | 66 | 60 | 0.91 |

Every other cell in the benchmark is at precision 1.00.

⚠ **This means REDOX_COFACTOR is not "unreachable" — it is MIS-HIT.** Evo2's seeded REDOX arm
produces 11 recognisable clusters per 200 and 8 of them are some other class. The model is
building BGC-like sequence and antiSMASH is confidently calling it something else. That is a
different failure from RIPP, where almost nothing is produced at all (1-5 detections per 200),
and a different failure again from a model that produces nothing (base W0, 0/200).

⚠ **It also means the on-target rate alone understates BGC production.** Reading only the
on-target column, Evo2's REDOX arms look like a flat 3/200 across de novo, seeded and steered.
Reading detection, seeding takes it 6 → 11 and steering-plus-seeding 11 → 12. Neither movement
is significant (p=0.32, p=1.00), but the arm is not inert the way the on-target column suggests.

## 4c. Why the classes differ — label granularity, not gene count

The class set was designed to span a **multi-gene ladder** monotonically. It does. It does not
predict performance.

```
QUESTION  which property of a class predicts how well either model generates it?
MODEL     "best arm" = the highest on-target cell across all 48 arms
ARM       class design properties, measured on the 8,044 train records per class
METRIC    on-target /200 at the best arm; detected in parentheses where it differs
UNIT      nt, gene counts, product-string counts, counts out of 200
```

| class | median core nt | avg core genes | ≥2 core genes | **antiSMASH product strings** | best arm on-target |
|---|---|---|---|---|---|
| TERPENE | 1,157 | 1.24 | 16.9% | 3 | 41/200 |
| RIPP | 2,234 | 1.48 | 34.7% | **32** | **5/200** |
| ARYLPOLYENE | 3,599 | 1.62 | **55.9%** | **1** | **73/200** |
| REDOX_COFACTOR | 3,096 | 2.99 | **100%** | **1** | 3/200 (det 6) |

*Best arm is Evo2 seeded in all four cases except REDOX_COFACTOR, where it is Evo2 de novo.*

⚠ **The multi-gene ladder does not predict performance.** ARYLPOLYENE is 55.9% multi-gene and
the best class in the benchmark; TERPENE is 16.9% multi-gene and second. REDOX_COFACTOR is 100%
multi-gene with an average of 3 core genes, and its on-target rate is 3/200 — but that is a
*scoring* failure (§4b), not a length or gene-count one. Nothing in the length or gene columns
orders the results.

**What does order them is how many things the class label collapses.** Of antiSMASH's 103
product strings, ARYLPOLYENE and REDOX_COFACTOR are one each, TERPENE is three, and RIPP is
**thirty-two**. The two single-product classes are the two the models either nail (ARYLPOLYENE)
or produce-and-mis-score (REDOX_COFACTOR); the 32-product class is the one that stays at zero.

This matches the independent evidence in FINDINGS §12, which reached the same conclusion from
held-out loss and subtype entropy: RIPP has ~15 *effective* subtypes and the worst val loss of
the four (Evo2 1.060 nats/nt against ARYLPOLYENE's 0.715), while REDOX_COFACTOR has ~1.9 and
the second best (0.721). RIPP is a genuine generation failure; REDOX_COFACTOR is not a learning
failure at all.

## 4d. What subtype do the correct generations actually produce?

Every on-target generation carries the antiSMASH product string that earned it. Pooled over all
48 arms, against the real corpus for comparison:

```
QUESTION  within a class the models get right, WHICH subtype do they make?
MODEL     both, pooled over all 48 arms (the pattern is identical per arm)
ARM       every on-target generation
METRIC    share of on-target generations by antiSMASH product string
UNIT      counts and % of that class's on-target total
```

| class | generated subtypes | real corpus (n=82 held-out cores) |
|---|---|---|
| **ARYLPOLYENE** (498 on-target) | `arylpolyene` 498 = **100%** | `arylpolyene` 100% — 1 subtype |
| **REDOX_COFACTOR** (14) | `redox-cofactor` 14 = **100%** | `redox-cofactor` 100% — 1 subtype |
| **TERPENE** (141) | `terpene-precursor` 135 = **95.7%**<br>`terpene` 6 = 4.3% | `terpene-precursor` 61%<br>`terpene` 39% |
| **RIPP** (17) | `RiPP-like` 10 = **58.8%**<br>`RRE-containing` 7 = 41.2% | 16 distinct, modal share 42.7%:<br>`RiPP-like` 35, `RRE-containing` 11, `lassopeptide` 7, `ranthipeptide` 7, `cyclic-lactone-autoinducer` 7, `azole-containing-RiPP` 3, `proteusin` 2, `triceptide` 2, + 8 singletons |

⚠ **Both models collapse onto the head of the subtype distribution.** The two single-subtype
classes are reproduced faithfully because there is nothing to collapse. TERPENE's split is
61/39 in real cores and **96/4** in generations — the models overwhelmingly produce the easier
`terpene-precursor`. And RIPP's 16 real subtypes come out as **exactly 2**, which are its two
commonest; `lassopeptide`, all four `lanthipeptide` classes, `proteusin`, `linaridin`,
`thioamitides` and the rest are **never produced once** across 9,600 generations.

⇒ So RIPP's near-zero rate is not only "rarely produces anything". When it does produce
something, it produces the two most frequent subtypes and nothing else. A class label that
collapses 32 product strings is not one target — it is a distribution, and these models cover
its head only.

## 5. What this benchmark says

1. **Unaided, neither model generates BGCs at a useful rate**, and they are statistically
   indistinguishable from each other. Evo2 fails by never stopping; GenomeOcean by stopping too
   soon.
2. **Seeding is the only intervention that works**, and it works well (up to 6.8×).
3. **Evo2 uses a seed better than GenomeOcean**, significantly, on both classes where there is
   signal — the benchmark's clearest cross-model claim.
4. **Activation steering does not move the endpoint** on either architecture, at magnitudes
   chosen per class against generation health, with or without seeding.
5. **Two classes are out of reach, but for DIFFERENT reasons** — a distinction visible only
   once detection is read alongside on-target (§4b).
   * **RIPP is not produced.** It never exceeds 5 detections per 200 in any of the 48 arms, on
     either model. "No effect" there is uninformative rather than negative: at 0/200 the rule of
     three puts the 95% upper bound at 0.015, and no arm has the power to resolve that.
   * **REDOX_COFACTOR is produced and MIS-CLASSED.** Evo2's seeded arm detects 11 per 200 and is
     on-target for 3 (precision 0.27); its steered arm detects 4 and is on-target for **none**.
     The model is building BGC-like sequence that antiSMASH assigns to another class. Calling
     this "unreachable" would misdescribe it, and calling the on-target zero a generation
     failure would too.

## Reporting status

**EXPLORATORY.** No pre-registration; α and injection sites were selected per class against
generation health (never against the endpoint — `endpoint_fields_read` is `[]` in every gate
artifact), and the endpoint was read once per arm afterwards. The seeding and cross-model
seeded results survive multiplicity correction comfortably; the steering nulls are reported as
nulls and the three broken controls are disclosed above.
