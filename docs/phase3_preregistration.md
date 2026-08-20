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

### 4.1 AMENDMENT 2026-08-18 — §4 above is NECESSARY BUT NOT SUFFICIENT

The original §4 text stands unedited (Standing Constraint 4). This amendment records that the gate
it specifies is **insufficient on its own**, discovered in Stage 1.

§4 names `containment` as the absolute gate and cites the phage paper's memorisation warning as the
reason. **The gate it specifies cannot see the memorisation it is guarding against.** In the Stage-1
sweep the L=500 cell held max containment **0.021** — two orders of magnitude inside the 0.80
threshold, i.e. maximally clean — while **12/12 of its on-class generations reproduced a marker
domain their own source cluster carries.** A model that rebuilds the *protein* with different
synonymous codons shares almost no exact 21-mers.

⇒ **Both gates are hard gates from 2026-08-18: `containment` < 0.80 AND `protein_aai` < 0.95.**
Pre-registered for Stage 2 in §8.5 before that stage generated. Any arm reported on containment
alone is reported incompletely.

⇒ **`protein_aai` must be read among ON-CLASS records against the real-core reference** (real
held-out cores 0.641). A pooled median is not interpretable — it is dominated by how many records
have any hit at all. See `memory.md` 2026-08-18.

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

### 8.4 CONTROL EXPANSION [P3-B3] — pre-registered 2026-08-17, BEFORE generating

**This does not change the endpoint** (§2) or any decision rule. It fixes the *control* sample size,
which §8.1–8.3 left at the pilot's n=50/arm.

**Why.** A0 stands at 4/150 vs 0/100 pooled controls, p=0.128. Generating more *A0* does not close
this — against a fixed 0/100 control, Fisher's exact plateaus at p≈0.09 and never reaches 0.05
(150→0.128, 300→0.098, 500→0.091). The **control arm is the binding constraint**: 100 controls with
zero hits is still consistent with a true control rate near A0's own.

**Fixed now, before any sequence is generated:**
- **n = 150 additional generations per control arm** (base 1B, general adapter) → 400 pooled.
- **Seed = 1** (the pilot used seed 0; reusing it would reproduce the same sequences).
- **Written to new files.** `phase3_pilot.py:generate()` returns early if the output exists, so the
  pilot's 50 are untouched and are **pooled**, not replaced.
- **A0 is NOT regenerated.** Its 4/150 is fixed and was collected before this was planned.
- **Everything else identical to §5:** 4,000 nt generated, fixed 2,000 nt scoring window,
  `OBLIGATE_DOMAINS[RIPP]`, same prompts file.

**Decision rule, fixed in advance** (Fisher's exact, one-sided, A0 4/150 vs pooled controls):

| control hits (of 400) | p | verdict |
|---|---|---|
| 0 | 0.0054 | ✅ A0 significant |
| 1 | 0.0211 | ✅ A0 significant |
| 2 | 0.0499 | ✅ marginal — report as marginal, not as a win |
| ≥3 | >0.05 | ❌ A0 not significant; the de novo line closes |

n=150/arm was chosen precisely because it survives two control hits. **Read once at n=400. Do not
extend the sample after seeing the result** — that converts this to exploratory under §10.

### 8.5 STAGE 2 — seeded arms at L\*=8 nt. Pre-registered 2026-08-17, BEFORE generating.

**Endpoint unchanged** (§2). This fixes the arms, n, and both novelty gates.

⚠️ **§7's A2 (mosaic) and A3 (consensus) are DEGENERATE at L\*=8 and are NOT run.** A mosaic
splits a seed across k clusters — at 8 nt that is ~2–3 nt each. A consensus over RIPP starts is
noise past position 3 (measured entropy ≈2.0 bits). Both were designed when seeds were ~500 nt.
**At 8 nt the meaningful contrast is the MODEL and the seed's INFORMATION CONTENT, not its
provenance.** Recorded rather than silently dropped.

**Arms — n=200 each, L=8 nt, `--no-boundary-orf`, TEST seeds (`eval_prompts.jsonl`), seed 11:**

| id | model | seed | question |
|---|---|---|---|
| S2-1 | RIPP LoRA | real RIPP 8-mer | the main arm |
| S2-2 | general all-class adapter | real RIPP 8-mer | **is the lift class-specific?** (the §7 comparison) |
| S2-3 | base 1B | real RIPP 8-mer | floor |
| S2-4 | RIPP LoRA | **codon-shuffled** 8-mer | does seed *content* matter, or just having a prefix? |
| S2-5 | RIPP LoRA | real 8-mer, **mismatched class tag** | does the continuation track the SEED or the TAG? |

**Powered contrasts, fixed now** (Fisher exact, one-sided, from the Stage-1 rate of 0.160):
- **S2-1 vs S2-2** — powered to p<0.001 at n=200 if general ≈0.02.
- **S2-1 vs S2-4** — powered to **p=0.0101** at n=200 if shuffling halves the rate (0.160→0.08).
  ⚠️ A 4-point difference (0.160 vs 0.12) needs n≈600 and is **out of reach**; if the observed gap
  is that small the result is **"underpowered, read descriptively"**, NOT "no effect".
- S2-3 is a floor, not a contrast.

**BOTH novelty gates are gates** (Stage 1 showed containment alone passes a memorising arm):
`containment` FAIL ≥0.95 · **`protein_aai` FAIL ≥0.95** · intra-set distinctness · JOINT_PASS.

**Reported for every arm:** THE PHASE-3 REPORTING SET in full, plus the seed→generation domain
match (Stage 1's decisive readout: 0/8 at L=8 vs 12/12 at L=500).

**Read once at n=200. Do not extend after seeing results** — that makes it exploratory under §10.

### 8.6 [P4-WIDE-SEEDED] — pre-registered 2026-08-18, BEFORE generating

**Why this supersedes the de novo WIDE arm.** That arm ran at n≈80 in the 0.024 de novo regime and
is **uninformative** — detecting a doubling there needs n≈800. This runs in the **seeded L\*=8
regime** where the base rate is 0.176 and n=188 is already demonstrated adequate.

**Arms — n=188, L=8 nt, `--no-boundary-orf`, TEST seeds (`eval_prompts.jsonl`), seed 21, 2,200 nt:**

| id | adapter | trained on | question |
|---|---|---|---|
| W-1 | **WIDE** | 3,723 wide spans (mean 4.41 genes pre-filter) | does a wider substrate produce multi-domain output? |
| W-2 | **STRICT size+cluster matched** | 3,723 **strict** spans, **the same accessions** | isolates span width from dataset size |
| S2-1 | STRICT (existing) | 7,250 strict spans | the published 0.176 reference |

W-2 is the control that matters: same clusters, same count, **only the span width differs**. A
WIDE−S2-1 difference alone would confound width with the 7,250→3,723 size drop.

**PRIMARY endpoint unchanged** (§2): `best_bio_bits > 0` @ `OBLIGATE_DOMAINS[RIPP]`, 2,000 nt.
**CO-PRIMARY for this arm, declared now:** **`n_class_domains ≥ 2`** — the metric the whole
WIDE hypothesis targets. It has read **0–2/188 in every arm ever run**; real cores read 29%.

**Powered contrasts, fixed now:**
- **W-1 vs W-2** — the clean span-width test. At 0.176 baseline, n=188 detects a lift to ~0.28.
- **W-1 vs S2-1 (0.176)** — reported, but confounded with training-set size; secondary.
- **`n_class_domains ≥ 2`:** from 2/188, n=188 detects a rise to ≥10/188 (Fisher p<0.05). A smaller
  rise is **underpowered, read descriptively** — not "no effect".

**Both novelty gates** (§4.1): `containment` < 0.80 **and** `protein_aai` < 0.95. Report Stage A and
Stage B separately (§ THE TWO MEASUREMENT STAGES) and Holm-correct across the contrasts (§9.1).

**Read once at n=188.** Do not extend after seeing results (§10).

### 8.7 AMENDMENT 2026-08-18 — 8 kb generation added to §8.6, BEFORE those sequences exist

§8.6 specified 2,200-nt generation. **That may structurally cap the co-primary endpoint.** The
window sweep showed real cores carry **1.69** biosynthetic domains at 2 kb and **2.67** at 8 kb, so
multi-domain content largely lives beyond 2 kb. An arm whose entire hypothesis is multi-gene
structure cannot demonstrate it in 2.2 kb of output.

⇒ **Every §8.6 arm is additionally generated at 8,000 nt and scored at 2 kb / 4 kb / 8 kb.**
- The **2 kb score remains the PRIMARY** and stays comparable to S2-1's 0.176 (§8.5 unchanged).
- 4 kb and 8 kb are **declared secondaries**, each with its own real-core ceiling
  (0.515 / 0.559 / 0.618) — never cross-compared.
- The 2.2-kb arms already generated are **retained and reported**, not discarded.
- A third arm is added: **STRICT-full (the S2-1 adapter) regenerated at 8 kb**, so all three
  adapters are read at matched generation length and training-span width is not confounded with
  output length.

**Headline rates are antiSMASH-corrected** (§ THE TWO-PASS DETECTION ARCHITECTURE): antiSMASH on
**all** Pfam-positives **plus a random sample of Pfam-negatives** per arm, then
`rate = [P·conf(pos) + N·conf(neg)] / (P+N)`. Pfam-only rates inflate ~1.8× and are reported as
their own row, never as the headline.

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

### 9.1 AMENDMENT 2026-08-18 — two ambiguities in §9, resolved

§9 text stands unedited. Two clarifications, both forced by A0 landing positive:

1. **"exceeds the A0 floor" is ambiguous now that A0 is not a floor.** A0 was designed as an
   unseeded floor presumed ~0; it returned 4/150 = 0.027, significant at p=0.0054. The comparator is
   therefore stated explicitly: **an arm is compared against (a) the base-model floor 0/N, (b) A0,
   and (c) the same-seed general adapter.** Stage 2 reported all three, so no result rests on the
   ambiguity.
2. **Holm correction was required by §9 and was NOT reported with the Stage-1/Stage-2 results.**
   Declared here rather than quietly fixed. Computed 2026-08-18 across the four Stage-2 contrasts:

   | contrast | raw p | **Holm p** |
   |---|---|---|
   | S2-1 vs S2-2 (class-specific) | 2.50e-11 | **9.99e-11** |
   | S2-1 vs S2-3 (vs base) | 2.50e-11 | **9.99e-11** |
   | S2-1 vs A0 de novo | 4.27e-06 | **8.55e-06** |
   | S2-1 vs S2-4 (seed content) | 0.656 | 0.656 (n.s.) |

   **Every claim significant raw remains significant corrected.** No conclusion changes — but the
   omission was a real deviation from the pre-registered analysis plan and is recorded as one.

## 9.2 AMENDMENT 2026-08-19 — `subclass_specificity` is adopted as a REPORTED SECONDARY, not a new primary

**The primary endpoint does not change** (Standing Constraint 4; §2 stands unedited). `best_bio_bits
> 0 @ OBLIGATE_DOMAINS[RIPP]`, 2,000 nt, remains the pre-registered primary for every Phase-3 arm,
so every arm stays comparable to A0 and Stage 2. This amendment adds a **declared secondary** and
records why, before the arm it will judge has been run.

**What it is.** `subclass_specificity` (`terms.md`) = of the generations antiSMASH **detects**, the
fraction assigned a **specific** RiPP subclass (lassopeptide, lanthipeptide class i–v, thiopeptide,
sactipeptide, thioamitides, azole-containing-RiPP …) rather than the generic catch-all `RiPP-like`.
Denominator = detected regions, so it is a **Stage-B** metric and needs a real-core reference in
every table it appears in.

**Why it is being added now.** Three metrics were used in turn as proxies for "is this a cluster,
not a lone gene" — `n_class_domains >= 2`, `bio_span_frac`, and the precursor panels. **All three
failed validation** (`memory.md` 2026-08-19): only ~16% of *real* cores reach `n_class_domains >= 2`
in a 2 kb window, `bio_span_frac` inverts within positives, and the precursor detector tops out at
8–50% sensitivity on our own real data. `subclass_specificity` measures the same underlying question
using **antiSMASH's own rule hierarchy** — tight subclass rules each require a specific *combination*
of domains, the loose `RiPP-like` rule fires when none match — so producing a subclass call is
strictly harder than producing a detection, and it needs no detector we built ourselves.

**Scoring config, frozen here.** Full-mode antiSMASH (⚠️ **never `--minimal`**, which disables the
analysis modules), region `product` qualifiers, **counted per detected sequence, not per product
string** — a region may carry several products, and counting strings is what produced the incorrect
"~70%" real-core figure now corrected to **0.909**. Output directories retained, never a
`TemporaryDirectory`. Deduplicate the generation set first (`bugs.md`, fan-out shard collision).

**Reference values, measured 2026-08-19 and stated before any new arm runs:**

| set | detected | specific | `subclass_specificity` |
|---|---|---|---|
| real held-out cores (ceiling) | 33/40 | 30 | **0.909** |
| STRICT-full 8k, Pfam-positive (best arm) | 3/4 unique | 0 | **0.000** |
| STRICT-matched, Pfam-positive | 4/4 unique | 0 | **0.000** |
| base 1B (floor) | 0/39 | 0 | n/a — no detections |

⚠️ **Power, declared honestly:** the generated denominators are **3 and 4 unique detections**. Fisher
on the pooled 0/7 vs 30/33 gives p≈1e-5, so the *direction* is established, but **any claim about a
change in `subclass_specificity` needs a denominator of >=15 detections** before it is believed. That
threshold is pre-registered here so a future arm cannot be read as "moved the metric" on n=4.

**Kill criterion for the metric itself:** if an arm reaches >=15 detections and `subclass_specificity`
remains 0.000 while the detection rate rises, the honest reading is that the model produces
RiPP-like signal without subclass chemistry, and the limitation is reported as permanent for this
substrate rather than re-proxied a fourth time.

## 10. What would make this exploratory rather than confirmatory

Any of: changing the primary endpoint; changing the scoring window; adding arms after seeing
results; pooling across generation lengths or batching modes; reporting an uncorrected p as the
headline; dropping the novelty gate. If any occurs, the result is labelled exploratory in every
document that reports it.
