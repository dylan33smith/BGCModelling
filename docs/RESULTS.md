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
METRIC    on-target = antiSMASH called the row's own class. Higher better. p = Fisher exact,
          two-sided, Evo2 vs GenomeOcean on the same class
UNIT      counts out of n=200
```

| class | Evo2 | GenomeOcean | p |
|---|---|---|---|
| ARYLPOLYENE | 16 | 21 | 0.49 |
| TERPENE | 6 | 4 | 0.75 |
| REDOX_COFACTOR | 3 | 1 | 0.62 |
| RIPP | 0 | 1 | 1.00 |
| W1n pooled | 3 | 1 | 0.62 |
| W0 base | 0 | 0 | — |

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
METRIC    on-target /200. p(within) = seeded vs de novo, same model.
          p(across) = Evo2 vs GenomeOcean, both seeded
UNIT      counts
```

| class | Evo2 de novo → seeded | p (within) | GO de novo → seeded | p (within) | **p (across, seeded)** |
|---|---|---|---|---|---|
| ARYLPOLYENE | 16 → **73** | 4.0e-12 | 21 → **50** | 2.1e-04 | **0.017** |
| TERPENE | 6 → **41** | 3.2e-08 | 4 → **19** | 2.0e-03 | **0.0030** |
| RIPP | 0 → 5 | — | 1 → 2 | — | 0.45 |
| REDOX_COFACTOR | 3 → 3 | — | 1 → 1 | — | 0.62 |
| W1n pooled | 3/200 → **94/800** | — | 1/200 → **60/800** | — | **0.0050** |

Two findings, both significant:

1. **Seeding is a large, real effect** on the two classes where anything happens at all —
   4.6× on Evo2 ARYLPOLYENE, 6.8× on Evo2 TERPENE.
2. **Evo2 converts a seed better than GenomeOcean does**, on both of those classes and on the
   pooled arm. This *reverses* the de novo ordering, where GenomeOcean nominally led
   ARYLPOLYENE. It is the clearest cross-model result in the benchmark.

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
METRIC    on-target /200; p = Fisher exact, two-sided
UNIT      counts
```

| model | class | α / sites | unsteered | I1 | p | I1rand | p (I1 vs rand) | seeded | I1xS1 | p |
|---|---|---|---|---|---|---|---|---|---|---|
| Evo2 | ARYLPOLYENE | 0.1 / 4 of 4 | 16 | 23 | 0.31 | 16 | 0.31 | 73 | 70 | 0.84 |
| Evo2 | TERPENE | 0.4 / 1 of 4 | 6 | 2 | 0.28 | 0 | 0.50 | 41 | **3** | **<0.001** |
| Evo2 | RIPP | 0.4 / 4 of 4 | 0 | 0 | 1.00 | 0 | 1.00 | 5 | 0 | 0.061 |
| Evo2 | REDOX_COFACTOR | 0.075 / 4 of 4 | 3 | 0 | 0.25 | 0 | 1.00 | 3 | 3 | 1.00 |
| GO | ARYLPOLYENE | 0.4 / 1 of 24 | 21 | 25 | 0.64 | 16 | 0.19 | 50 | 65 | 0.12 |
| GO | TERPENE | 0.05 / 8 of 24 | 4 | 1 | 0.37 | 4 | 0.37 | 19 | 26 | 0.34 |
| GO | RIPP | 0.1 / 1 of 24 | 0 | 0 | 1.00 | 0 | 1.00 | 2 | 3 | 1.00 |
| GO | REDOX_COFACTOR | 0.3 / 24 of 24 | 1 | 0 | 1.00 | 0 | 1.00 | 1 | 0 | 1.00 |

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

## 5. What this benchmark says

1. **Unaided, neither model generates BGCs at a useful rate**, and they are statistically
   indistinguishable from each other. Evo2 fails by never stopping; GenomeOcean by stopping too
   soon.
2. **Seeding is the only intervention that works**, and it works well (up to 6.8×).
3. **Evo2 uses a seed better than GenomeOcean**, significantly, on both classes where there is
   signal — the benchmark's clearest cross-model claim.
4. **Activation steering does not move the endpoint** on either architecture, at magnitudes
   chosen per class against generation health, with or without seeding.
5. **Two classes are unreachable for both models.** RIPP and REDOX_COFACTOR never exceed 5/200
   in any of the 48 arms, so "no effect" there is uninformative rather than negative: at 0/200
   the rule of three puts the 95% upper bound at 0.015, and no arm has the power to resolve
   differences that small.

## Reporting status

**EXPLORATORY.** No pre-registration; α and injection sites were selected per class against
generation health (never against the endpoint — `endpoint_fields_read` is `[]` in every gate
artifact), and the endpoint was read once per arm afterwards. The seeding and cross-model
seeded results survive multiplicity correction comfortably; the steering nulls are reported as
nulls and the three broken controls are disclosed above.
