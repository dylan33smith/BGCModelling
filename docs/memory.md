# memory.md — the linear ledger

**DO NOT read this file on startup.** It is the permanent laboratory notebook and it only grows.
`grep` it when you need historical context.

```bash
grep -n "2026-08" docs/memory.md      # by date
grep -n "^## \[" docs/memory.md       # list all entries
grep -ni "best_bio_bits" docs/memory.md
grep -n "INCORRECT\|CORRECTION" docs/memory.md   # what we got wrong
```

## Rules

1. **Append only.** Newest at the bottom. Never delete, never overwrite.
2. **In-place correction.** When something here is proven wrong: prepend `[INCORRECT] - ` to the
   original line, then insert `[CORRECTION - YYYY-MM-DD]: ` directly below it. The wrong version
   stays — it is what makes the reasoning legible later.
3. **Provenance or it didn't happen.** Every result carries checkpoint · generation set · n ·
   scoring config · window.
4. **Metric names come from `terms.md`.** No synonyms.

Entry types: `Intervention` (an experiment) · `Decision` (why the project is built this way) ·
`Correction` (a retraction).

---

## PRE-FRAMEWORK ARCHIVE (imported 2026-08-14)

Everything before 2026-08-12 lives, unedited, in the pre-framework files. It was **not** rewritten
into this format — rewriting ~3,000 lines of history is where errors get introduced. Search there
for anything older:

| Source | Lines | Covers |
|---|---|---|
| `docs/archive/pre-framework/progress.md` | 1,851 | Phases 0–3, chronological, 2026-05 → 2026-08-14 |
| `docs/archive/pre-framework/decisions.md` | 1,166 | architecture + approach decisions with rationale |
| `docs/archive/gputee/`, `docs/archive/trojai/` | ~5,000 | deep runbooks, superseded |
| `docs/archive/AUDIT_FINDINGS.md` | 465 | the 2026-06-02 leakage audit |
| `docs/archive/REDESIGN_PLAN.md`, `EVAL_RUNBOOK.md` | 620 | the 2026-06-17 eval rewrite |
| `docs/archive/steering_program.md`, `_technical.md` | 738 | the closed steering programme, in full |
| `docs/archive/conditioning_next_steps.md` | 398 | ranked conditioning list — **ranking SUPERSEDED**, citations accurate |
| `docs/phase3_preregistration.md` | 203 | **LIVE** — the Phase-3 pre-registration, do not edit mid-phase |
| `docs/model_comparison_evo2_vs_genomeocean.md` | 325 | measured head-to-head |

**Load-bearing history in one paragraph.** Class conditioning was the wrong target: Evo2 already
*represents* compound class (linear probe 0.911 vs 0.091 chance — and 0.911 in **base** Evo2, so
the LoRA installed nothing), but every inference-time lever that edits input or activations is
closed (prefix labels, CFG, activation steering in all variants, affine concept editing,
cross-class transplants, soft prefixes). Guided decoding is the sole exception and is
underpowered, not null. Conditioning was never the binding constraint: the class tag is worth
**−0.0006 nats** to the training loss, so gradient descent had no reason to build a pathway that
reads it. The real constraint is **capability** — but not reading-frame length. De novo output is
real protein of the wrong kind.

---

## 2026-08-12 — Decision: the ladder replaces the binary gate

**What changed.** `max_orf_aa` was **DEMOTED** and `best_bio_bits` adopted as PRIMARY.

**Why.** `max_orf_aa` had been adopted on *between-group* evidence (generated is shorter than real)
plus a mechanistic story ("cannot sustain a reading frame"). It failed the *within-group* test:
inside de novo generations it does not track domain content at all — r = **0.051** at 2 kb and
**−0.120** at 6 kb. AUROC 0.709. `biosynthetic_fraction` rested on exactly the same kind of
evidence (a clean between-group ladder 0.015 → 0.100 → 0.464 → 0.836), so adopting *it* as primary
would have repeated the error one metric later.

**The validated ladder** (`evo2/scripts/ladder_audit.py`, AUROC against seeded-arm outcomes — the
only regime with real variance in detection, 0.367 rather than a floor):

| rung | metric | AUROC |
|---|---|---|
| 1 | `best_bio_bits` | **0.950** |
| 2 | `n_bio_domains` | 0.919 |
| 3 | `bio_span_frac` | 0.896 |

**Novelty is a GUARD on the ladder, not a rung of it.** Every rung is maximised by copying training
data. Any run optimising these metrics must report containment alongside or the improvement is
uninterpretable.

**What this settled about the failure mode.** De novo output is not junk — coding density 0.74–0.82
vs 0.97 real, and **100% of 6 kb generations hit some Pfam family**. The model writes real protein.
It writes the **wrong kind**: `biosynthetic_fraction` 0.100 vs 0.836, `bio_span_frac` 0.051 vs
0.876.

---

## 2026-08-12 — Decision: conditioning was aimed at the smaller of two problems

Two measurements closed the conditioning programme as a *target*:

1. **The class tag is worth −0.0006 nats** to the training loss (−0.0000 with the tag 200 nt away),
   against 0.149 nats for all long-range context and 1.386 for a uniform guess.
2. **`correct_class` decomposes as `P(detect) × P(right | detect)`.** On the same adapter:
   de novo `P(detect)` = **0.012 (1/81)**; seeded = **0.367 (44/120)**; and
   `P(right | detect)` when seeded is already **0.932**.

⇒ In the seeded regime there is ~7% left for conditioning to win. De novo there is nothing to
install class *into*. Conditioning was never the binding constraint.

---

## 2026-08-13 — Intervention: Phase 2, objective change (frame-aware × domain-weighted)

- **Hypothesis:** a frame-aware or domain-weighted training objective raises biosynthetic domain
  content de novo.
- **Technical:** Evo2 1B, L=8192, chunked, bit-identical baseline, 2×2 arms.
- **Provenance:** `phase2_1b/`, `phase2_long/` · n=152/arm · `best_bio_bits` · novelty guarded.
- **Result — FRAME ARM:** the frame objective **worked** — it moved its own variable ~8× — and
  **domain content did not follow**. Length is not the bottleneck. Test WAS powered; the kill
  criterion applies on its own terms. **CLOSED.**
- **Result — WEIGHTED ARM:** gene length p=0.23, any-Pfam p=0.25, `best_bio_bits` p=0.81,
  `n_bio_domains` p=0.88, `bio_span_frac` p=0.89, stop-completion mass 0.1228 vs 0.1227.

[INCORRECT] - Track B (objective change) is CLOSED on a powered test.
[CORRECTION - 2026-08-13]: **Track B is HALF closed.** The closure applies to the FRAME arm only.
The weighted arm's **treatment never landed** — the dose-response curve was almost flat across a
10× sweep — so its null is **uninterpretable**, not negative. A null requires both power *and* a
verified manipulation. This is why the manipulation check is now a required field in `plan.md`.

---

## 2026-08-14 — Decision: Phase 3 opens — one small class at a time

**The reframe.** Two problems had been tangled throughout: **long-context coherence** (2% of real
BGC genes exceed the 1B's entire context; Evo2 fits 0% of whole BGC *regions*) and **BGC
specificity** (real protein of the wrong kind). Restricting to one small class does not work
around the first — it **deletes** it. What remains is specificity, alone, with thousands of
examples. And because a per-class LoRA means the model never reads a class label, every Phase-1
conditioning closure stops applying.

**Target selection — and the finding that drove it.** Short classes are short *because they are
conserved*; length and diversity are anti-correlated here.

| class | held-out near-dup loss | verdict |
|---|---|---|
| MELANIN | 95% | disqualified |
| **ECTOINE** | **85%** | **disqualified — was the obvious target** |
| BUTYROLACTONE | 81% | disqualified |
| HSERLACTONE | 69% | disqualified |
| TERPENE | 46% | viable |
| **RIPP** | **43%** | ✅ **TARGET** — most diverse; 2× TERPENE's de novo detection (0.158 vs 0.079) |

**Substrate policy.** The **1B is the testing substrate for all of Phase 3**; the 7B confirms
anything publishable; GenomeOcean is live-but-held (leakage gate PASSED — 0.0000 containment,
greedy, positive control demonstrated first). A final paper may compare all three. **Testing does
not fan out — that confounds method with model.**

**Pre-registered** in `docs/phase3_preregistration.md`, fixed before any Phase-3 model was trained:
primary endpoint is a RATE; n from a pilot power analysis; identical generation length with a fixed
2,000-nt scoring window; novelty an absolute gate; a null interpretable only if powered AND the
intervention verified to have landed.

**What Hie et al. (Science 2026) changes.** Evo fine-tuned on ~15,000 Microviridae genomes, seeded
with a **consensus sequence in SEQUENCE space** (Evo2 takes token ids only — no `inputs_embeds`),
**seed length 4–8 nt optimal because longer seeds caused memorisation** (ours have been ~500 nt),
and ~1000:1 overgeneration-and-filtering.

---

## 2026-08-14 — Intervention: P3-A0, RIPP-only adapter, de novo

- **Hypothesis:** a RIPP-only LoRA produces RIPP biosynthetic machinery de novo where an all-class
  adapter does not.
- **Technical:** Evo2 1B + RIPP-only LoRA. 7,250 whole records (89.2% of RIPP fits at L=8192; 879
  **dropped, not truncated**, so `|END|` never lands on a cut sequence). 3 epochs / 1,350 steps,
  113 min, `loss_ce` 0.790 → **0.410**.
- **Provenance:** `phase3_RIPP/adapter_run` · `A0_8k.jsonl`, `A0_noseed.jsonl`,
  `phase3_ripp/pilot_*.jsonl` · scoring `OBLIGATE_DOMAINS[RIPP]` · window 2,000 nt.

[INCORRECT] - A0 failed: the general all-class adapter (0.080) beat the RIPP-only adapter (0.040).
[CORRECTION - 2026-08-14]: **The conclusion inverts on the pre-registered endpoint.** See the bug
below. On `OBLIGATE_DOMAINS[RIPP]` the generalist scores **exactly zero** — its 0.080 was other
classes' domains, which is correct behaviour for an all-22-class model — and **A0 is the only
non-real arm producing RIPP machinery at all.**

| arm | generic set (WRONG) | **RIPP-specific (pre-registered)** |
|---|---|---|
| base 1B | 0.000 | **0/50 = 0.000** |
| general all-class adapter | 0.080 | **0/50 = 0.000** |
| **A0 — RIPP-only** | 0.040 | **4/150 = 0.027** |
| real RIPP cores (ceiling) | 0.580 | **22/50 = 0.440** |

- **Verdict:** positive direction, **not significant** — 4/150 vs 0/100 pooled, **p = 0.152**. The
  direction changed, the significance did not. A0 reaches ~6% of ceiling.
- **Novelty guard — PASS, but read it carefully.** DNA containment max **0.003**; protein AAI
  median 0.000 / max 0.470; **150/150 intra-set distinct**. No memorisation, no codon paraphrase,
  no mode collapse. **But near-perfect novelty at a 2.7% hit rate is also exactly what a model
  producing plausible non-RIPP DNA would show.** Novelty is cheap when you are not hitting target.
- **Where the signal sits.** Per-block detection falls with position: block 0 **0.040**, then
  0.024 / 0.028 / 0.022. 4× the tokens bought **2.2×** the hits (independence predicts 0.151).
  Per-sequence indices show the six block-0 hits appear in **no** later block — different sequences
  produce RIPP content at different positions, sparsely. It is not one long gene continuing.
  ⇒ **Generate short, sample many, filter hard.**
- **`|END|`:** `hit_eos` **0/150**, as it was 0/204 before; whole-record training did not change it.
  **Stop trying to fix it.** The phage paper did not rely on a stop token either — they used a
  length filter (4–6 kb). Adopt the filter shape.
- **Next:** the comparison that matters is **seeded vs seeded**. No seeding has been run at all.

### Bug found by this intervention → `bugs.md`

**[Symptom]** Class-specific arm scores identically (4/50) under RIPP, NRPS, PKS *and* TERPENE
scoring — the class argument appears to do nothing.
**[Cause]** `evo2/scripts/ladder_audit.py:one()` takes a `cls` argument but uses it **only** for
the NRPS/PKS module-architecture rung. `bio` is scored against a fixed **global ~91-model**
biosynthetic set (`BIO = /data2/ds85/pfam/biosynthetic_subset.hmm`, the union of all
`OBLIGATE_DOMAINS` values). So "on-class" silently meant *any* biosynthetic domain.
**[Proven fix]** Always subset the HMM to `OBLIGATE_DOMAINS[<CLASS>]` for class-specific work.
Never quote a `best_bio_bits` number without stating which Pfam set produced it.
**[Severity]** This inverted a headline conclusion. It is the reason `terms.md` requires a
`CHANGES MEANING WITH` field on every metric.

---

## 2026-08-14 — Decision: length-bucket the batches, every future training run

Carried forward to **every** future training round: bucket batches by length before training.
Applies to all arms in any multi-arm comparison so throughput differences cannot confound the arms.

---

## 2026-08-10 — Decision: the probe/steering leakage debt is CLEARED

Recorded here because it was nearly re-opened by mistake on 2026-08-14: the standing debt from
2026-07-30 ("steering directions are fit on val+test") **was cleared on 2026-08-10** and the older
entry is the one that surfaces first on a grep.

- Probe and directions **both refit train-only**: `class_probe_sweep/acts_v2_train500.npz`
  (provenance-verified, `.provenance.json` on disk), `trainonly.steerdirs.npz` at 9 layers, probe
  cached at `acts_v2_train500.probe_L16_s0.joblib`.
- `probe_score_generations.py:_fit_probe` now **REFUSES** a non-train fit set;
  `--allow-leaky-probe` exists only to reproduce a historical number.
- ⚠️ `splits_core/valtest_fit.jsonl` still exists and its name still reads innocent. It is the
  **old leaky fit set**. See `data.md`.

**Standing methodological bar** that three weakened-or-retracted findings paid for: a paired design
with the control built in, a continuous readout alongside the binary gates, and an instrument whose
sensitivity *and* false-positive rate are measured BEFORE a result is read off it.

---

## 2026-08-14 — Intervention: documentation framework cutover

Six-file framework live; `CLAUDE.md` 362 → 130 lines. Filesystem fixes: `phase3_ripp/` merged into
`phase3_RIPP/`; orphaned HSERLACTONE + BUTYROLACTONE splits deleted (~6,600 records, disqualified on
diversity and un-reconstructable provenance); three empty run dirs removed; archived TERPENE claim
corrected in place. `tests/test_docs_contract.py` (26 checks) now gates the docs against code+disk.

**[Symptom] → [Fix] recorded in `bugs.md`:** `build_single_class_splits.py` initialises
`manifest = {}` and writes the whole file, so rebuilding one class silently drops every other
entry. Rebuild all classes or hand-merge the manifest.

---

## 2026-08-17 — Intervention: P3-B3, control expansion. **A0 IS SIGNIFICANT.**

- **Hypothesis:** A0's p=0.128 was limited by the *control* arm, not the treatment arm. Adding
  controls — not more A0 — closes it.
- **Method:** pre-registered in `phase3_preregistration.md` §8.4 **before generating**: n=150 per
  control arm, seed 1 (pilot used seed 0), new files so the pilot's 50 pool rather than being
  replaced. A0 not regenerated. Decision rule fixed in advance for 0/1/2/≥3 control hits.
- **Provenance:** `phase3_RIPP/ctrl_base_n150_s1.jsonl`, `ctrl_general_n150_s1.jsonl` ·
  substrate `evo2_1b_base.pt` (guard-verified) · scoring `OBLIGATE_DOMAINS[RIPP]`, 8 accessions ·
  window 2,000 nt · `novelty_battery.py` post-B0 fix.

| arm | n | RIPP-specific | generic |
|---|---|---|---|
| **A0 — RIPP-only LoRA** | 150 | **4/150 = 0.0267** | 6/150 = 0.040 |
| base 1B | 200 | 0/200 = 0.000 | 1/150 = 0.007 |
| general all-class adapter | 200 | 0/200 = 0.000 | 10/150 = 0.067 |
| pooled controls | **400** | **0/400 = 0.000** | — |

**Fisher exact, one-sided: p = 0.0054.** Exactly the pre-registered value for 0 control hits.
95% CI on A0 [0.0073, 0.0669]. **A class-specific LoRA produces RIPP machinery de novo where
neither the base model nor an all-class adapter does at all.**

**The inversion, reproduced at n=150/arm.** On the generic metric the *generalist* wins
(0.067 vs 0.040). On the pre-registered endpoint it scores **exactly zero**. Same sequences, same
window — only the Pfam subset differs. This is why `best_bio_bits` carries a `CHANGES MEANING
WITH` field in `terms.md`.

**What this does NOT show.** A0 reaches ~6% of the 0.440 ceiling. All four hits carry exactly
**one** RIPP domain (PF05114 ×2, PF04055, PF05402); real cores carry 1.45 on average with 9/31
carrying ≥2. One of the four is PF04055 (radical SAM), a near-universal family. None hit PF02624
(YcaO), the most RiPP-specific marker. **One domain is not a cluster.** The right claim is "the
adapter puts RIPP-associated machinery into de novo output at a low but real rate", not "it
generates RiPP clusters".

**Novelty guard: PASS.** Controls max containment 0.003 / 0.025, median 0.000.

---

## 2026-08-17 — Finding: RIPP core starts carry no class information past the start codon

Measured while designing leg 2, because the seed-content question (exemplar vs consensus vs mosaic)
only has an answer if the seed region is conserved.

**Method.** Position-wise base entropy over the first 20 nt of all 8,129 RIPP training cores.
2.00 bits = uniform = no conservation.

```
pos  1    2    3    4    5    6    7    8   ...  20
    1.61 0.81 1.05 1.94 1.99 1.97 1.97 1.95 ... 1.97      mean 1.85
```

Distinct prefixes among 8,129 records: **75 at 4 nt, 2,651 at 8 nt, 6,920 at 20 nt.** Top 4-mer
`ATGA` = 21.0% of cores; top 8-mer = 0.9%.

**Result.** Only positions 1–3 are informative, and they are the **start codon** (or a
reverse-strand stop — `TCAG`/`TTAT` appear at 8.5%/7.3%). **From position 4 the 5′ end of a RIPP
core is indistinguishable from random sequence.**

**Consequences for the seed ladder.**
1. **A3 (consensus/centroid prefix) is near-vacuous for RIPP.** Hie et al.'s consensus worked
   because ~15,000 Microviridae genomes are homologous and alignable. **RIPP was selected FOR
   diversity** (43% held-out near-dup loss — it beat TERPENE on exactly this axis). A consensus
   over non-alignable sequence is noise. Keep A3 as a cheap negative control only.
2. **At 4–8 nt the "it is just completing a memorised cluster" objection largely evaporates** —
   and so does the distinction between arms. 8 nt is ~16 bits and `ATGA` is a start codon, not
   RIPP information. Seed *content* only matters at long seeds.
3. **Class information must live later in the sequence**, which motivates **domain-anchored
   seeding** (plan.md [P3-B1d]): seed with the nt span of a RIPP marker domain instead of the
   arbitrary 5′ boundary. Within a Pfam domain sequences ARE alignable, so a representative seed is
   well-defined *there* even though it is not at the start. `scripts/build_domain_spans.py` already
   emits the spans; it has only ever been run on `splits_core`.

**Note on A0.** A0 was deliberately run first because "if a per-class adapter works de novo,
seeding is unnecessary and the reviewer objection evaporates." A0 came back **significant**
(p=0.0054), so the objection is already partly answered without any seed at all.

---

## 2026-08-17 — Finding: RIPP marker domains sit at the START of the core, not the middle

Measured to test a design objection to domain-anchored seeding: *if domains sit mid-sequence, then
seeding with a domain span means generating everything before it, which is not seeding.* The
objection is sound in general. **For RIPP it does not apply.**

**Method.** First RIPP marker domain position in 400 training cores, via ORF calling + the
8-accession `OBLIGATE_DOMAINS[RIPP]` subset.

| statistic | value |
|---|---|
| first domain starts at **nt 0** | **86.3%** of cores |
| p50 / p75 offset | **0 nt** / **0 nt** |
| p90 offset | 433 nt |
| within 500 / 1,000 / 2,000 nt | 91.1% / 93.7% / 96.7% |

**Cause.** antiSMASH **strict-core trimming already begins the region at the biosynthetic gene** —
`strict_core_genes` is a field on every record. The core does not contain a long non-biosynthetic
runway.

**Consequence.** Domain-anchored seeding **collapses into exemplar seeding** for RIPP: the
exemplar prefix already *is* the domain start. The idea was proposed the same day and is largely
withdrawn for this class. It retains force for **NRPS/PKS**, whose cores are long multi-modular
assembly lines. What survives the intent is **A2 mosaic** — spans from k *different* clusters,
a combination present nowhere in training.

**Consistent with the entropy finding above:** 86% of cores start at a marker gene, yet
position-wise entropy from nt 4 is ~1.95–2.00. Different RIPP subtypes use different marker
domains (PF05114 / PF04055 / PF13353 / PF05402), and DNA is degenerate even where protein is
conserved. Both results point the same way: **there is no meaningful consensus prefix for RIPP.**

**Also confirmed while designing this:** `eval_prompts.jsonl` is 100% TEST (199/199 accessions,
0% genome overlap with train), so tuning seed length on it would be selecting on the test set.
Stage 1 of the sweep therefore uses a **val**-derived prompt file.

---

## 2026-08-17 — Audit: strict-core trimming, and what the RIPP marker set actually measures

**Q: does strict-core trimming cut off start codons?** **No.**
`build_core_records.py:_core_span` takes min-start/max-end over CDS with `gene_kind ==
"biosynthetic"` (`STRICT_KINDS = {"biosynthetic"}`), and `_materialize` applies `--flank`, whose
**default is 0**. So the core begins exactly at the first biosynthetic CDS boundary — which
*includes* the start codon.

This explains the position-1–3 entropy directly: forward-strand genes start `ATG` (`ATGA` 21.0%,
`ATGG` 8.2%); reverse-strand genes present the **reverse-complement of their STOP codon** at the
left edge (`TCAG` 8.5% = revcomp TGA, `TTAT` 7.3% = revcomp TAA). Nothing is truncated.
⚠️ **One exception:** cores longer than `--max-len` (262,144) are **center-truncated**, so those do
have arbitrary ends. Rare — but RIPP's max core length is exactly 262,144, so it does occur.

**Q: could a non-coding regulatory element define the boundary?** **No.** Only `f.type == "CDS"`
features are read, so promoters, RBS, terminators and riboswitches can never set the span.
Regulatory *proteins* are CDS with `gene_kind == "regulatory"`, which is not in `STRICT_KINDS`
either. Every boundary-defining gene is protein-coding with a real start and stop.

**What the RIPP marker set is.** `OBLIGATE_DOMAINS` is **data-driven, not textbook**: Pfams kept at
`freq>=0.3 & enr>=4`, OR `freq>=0.08 & enr>=8` to retain rare-but-specific subtype markers.
Semantics are **ANY-of**, not all-of.

Frequency across 50 real RIPP cores (31/50 carry any; mean **1.45** distinct markers per hit):

| accession | cores | what it is |
|---|---|---|
| PF05114 | 12 | DUF692 maturase |
| PF04055 | 11 | Radical SAM — very broad family |
| PF13353 | 9 | Fer4_12 (4Fe-4S) — very broad |
| PF05402 | 8 | PqqD, precursor-peptide *binding* |
| PF02624 | 2 | YcaO cyclodehydratase — the most RiPP-specific |
| PF03070 | 2 | TENA_THI-4 |
| PF14028 | 1 | SkfB-like radical SAM |

⇒ **No marker is ubiquitous.** Any given RiPP carries *some subset*, usually exactly one. The set
detects the **modifying machinery**, and **none of the eight is a precursor peptide** — so the
metric never tests whether the model produced the actual RiPP product.
Related: Prodigal called **0 ORFs under 30 aa** even at `min_aa=10` (its own floor), so typical
short RiPP precursors are uncallable regardless of our setting. The `min_aa=50` default costs only
3.9% of ORFs and no core loses its last ORF, so the cutoff itself is not the problem — the caller's
floor and the marker set's composition are.

**Q: does domain ORDER matter for RIPP?** **No, and it should not become a metric here.**
`MODULE_PATTERNS` covers only NRPS (`C-A-T`) and PKS (`KS-AT-ACP`), and `ladder_audit.one()`
computes `modules`/`in_order` only for `NRPS/PKS/PKS_NRPS_HYBRID`. That is correct biology: those
are **collinear assembly lines** where domain order determines the product. RiPP gene order is not
collinear — the precursor is a separate gene from its modifying enzymes and order varies across
families. Empirically it is also undefined: at **1.45 markers per core**, most cores have one
domain and one element has no order.

⇒ **The right additional metrics are the CLUSTERING rungs, not an ordering rung** — `n_bio_domains`
(AUROC 0.919) and `bio_span_frac` (0.896), both already validated in `terms.md` and both currently
absent from the Phase-3 endpoint. "One domain is not a cluster" is exactly what they measure.

---

## 2026-08-17 — ⚠️ Finding: the "cores" we train on are mostly ONE OR TWO GENES

Measured while auditing strict-core trimming. This reframes what the whole project is modelling.

| | RIPP train, n=8,129 |
|---|---|
| `strict_core_genes` = **1** | **48.8%** |
| = 2 | 29.4% |
| = 3 | 15.7% |
| median / mean | **2 / 1.86** |
| median `region_len` (full antiSMASH region) | **21,279 nt** |
| median core length (what we train on) | **1,931 nt** |
| **core as a share of the region** | **median 9.1%** |
| records hitting the 262,144 center-truncation cap | 17 |

**`STRICT_KINDS = {"biosynthetic"}` only.** Excluded from the core: `biosynthetic-additional`,
`transport`, `regulatory`, `other`. For RiPPs that means the **exporter/protease** (usually a
bifunctional ABC transporter, annotated `transport`) is outside the core, and so is any
immunity/regulatory gene.

**Why this matters for interpretation.** We describe the mission as generating *biosynthetic gene
clusters*. What the model is actually trained on is **one or two biosynthetic enzyme genes in
genomic context** — a median 1.9 kb slice of a 21 kb cluster. Three downstream facts follow
directly and stop being puzzling:

- A0's hits carry exactly **one** domain — the training data is mostly one gene.
- **0/150 A0 records carry ≥2 distinct RIPP markers**, vs 9/31 = 29% of real cores.
- Real cores average only **1.45** markers — because the "real cores" are the same trimmed slices.

⇒ **The 0.440 ceiling is not "a real BGC", it is "a real trimmed core".** The comparison is
internally consistent and remains valid; the *claim* it supports is narrower than "generates BGCs".

**The original rationale was deliberate and still partly holds** (`decisions.md`): focus the model
on the biosynthetic machinery, make ~88% of clusters single-window, avoid diluting class signal
with flanking DNA. And whole-core-only training was tried — `mega_whole_32k` — and **failed**,
"starves the data". So this is a known trade, not an oversight. What was not stated is how much a
"core" actually is: **9% of the cluster, half the time a single gene.**

**Recommended, not yet actioned:** state this explicitly wherever a rate is quoted, and consider a
`WIDE_KINDS` arm (adds `biosynthetic-additional`) as a cheap test of whether the model can hold a
2-3 gene neighbourhood. Do NOT quietly redefine the endpoint mid-phase (Standing Constraint 4).

**Strand.** 88.7% of real RIPP cores are **fully single-strand** (`co_orient` median 1.000). Which
strand is arbitrary — it is whichever the assembly stored. A core whose genes sit on the minus
strand shows the reverse-complement of its STOP codon at nt 0, which is the `TCAG`/`TTAT` signal in
the entropy profile. The stored sequence is the real genomic locus either way; nothing is inverted
or lost.

---

## 2026-08-17 — Method: fan out sequential processes when the batched path is gated

**Problem.** vortex batching is gated (left-pad perturbs StripedHyena, failed an on-GPU equivalence
gate), so generation is one sequence at a time. Measured cost: the H100 sat at **41% utilisation
and 4 GB of 80 GB** — the Phase-3 seed sweep was projecting **~4.9 h** for 10 cells of n=50.

**Method.** Run N independent *sequential* processes on disjoint work units. Generation semantics
are untouched — each process still emits one sequence at a time, so every output is identical to
the serial run. Only the idle GPU capacity is recovered.

| | sequential | 3 workers |
|---|---|---|
| GPU utilisation | 41% | **100%** |
| GPU memory | 4 GB | 22 GB (of 80) |
| per-sequence | 29 s | 25 s |
| **aggregate** | 1× | **~3.5×** |
| 10 cells, n=50 | ~4.9 h | **~1.1 h** |

**Mechanics that made it safe:**
- **Atomic claim** — `mkdir "$CLAIMS/$unit"` succeeds for exactly one worker. No shared-state race.
- **Publish on success only** — write `<out>.partial`, `mv` to `<out>` when complete, so an
  interrupted unit never looks finished to a later run's `[ -s "$out" ]` skip check.
- **One tmux session + status sentinel per worker**, per the standing execution rule.

Generalised as `scripts/fanout.sh`; rule added to `CLAUDE.md`.

⚠️ **Not valid for throughput or memory benchmarks** — contention is precisely what invalidates
those, which is why the standing rule sends training through the queue wrapper. This applies to
*generation and scoring*, where the unit outputs are deterministic given (seed, prompt, model).

Two traps hit while implementing it, both in `bugs.md`: `pkill -f` matching its own command line
and killing the issuing shell; and losing the scoring stage that had lived at the bottom of the
script being parallelised.

---

## 2026-08-17 — Intervention: P3-B1 Stage 1, seed-length sweep. **L\* = 8 nt.**

- **Hypothesis:** seeding lifts the RIPP-specific rate above the de novo floor, and there is a seed
  length beyond which the model memorises (phage paper: 4–8 nt optimal, longer memorises).
- **Method:** exemplar seeds from **val** (`val_prompts.jsonl`, 60 records, 0% genome overlap with
  train), seed length ∈ {4, 8, 20, 100, 500} nt × {RIPP LoRA, base 1B}, n=50/cell. Tuning stage —
  not confirmatory.
- **Provenance:** `phase3_RIPP_seedsweep` · `s1_<model>_L<len>.jsonl` · scoring
  `OBLIGATE_DOMAINS[RIPP]` (8 accessions) · window 2,000 nt · substrate `evo2_1b_base` ·
  generation 2,200 nt · THE PHASE-3 REPORTING SET on every cell.

| model | seed | on_class | rate | JOINT | ≥2 markers | max containment | **max AAI** | median AAI |
|---|---|---|---|---|---|---|---|---|
| base | 4–500 nt | **0/50 every length** | 0.000 | 0 | 0 | ≤0.001 | 0.000 | 0.000 |
| lora | 4 nt | 7/50 | 0.140 | 7 | 0 | 0.011 | 0.617 | 0.000 |
| **lora** | **8 nt** | **8/50** | **0.160** | **8** | 1 | 0.015 | **0.620** | **0.000** |
| lora | 20 nt | 5/50 | 0.100 | 3 | 0 | 0.016 | 0.801 | 0.000 |
| lora | 100 nt | 5/50 | 0.100 | 5 | 3 | 0.014 | 0.793 | 0.000 |
| lora | 500 nt | 12/50 | 0.240 | 12 | 4 | 0.021 | **0.914** | **0.291** |

**★ THE BASE MODEL SCORES ZERO AT EVERY SEED LENGTH, INCLUDING 500 nt.** Handed a real 500-nt RIPP
prefix, base Evo2-1B produces **no** RIPP domain in 50 tries. This is the direct answer to "the
model is just finishing a cluster that already exists": the seed alone does nothing. Every LoRA
cell beats base at the same length (p=0.0062 / 0.0029 / 0.0281 / 0.0281 / 0.0001).

**Seeding works, and it is the adapter doing it.** Every LoRA cell also beats the A0 de novo rate
of 4/150 = 0.027 (p = 0.0061 / 0.0020 / 0.0450 / 0.0450 / 0.0000). L=8 is a **~6× lift** over
de novo; L=500 reaches 0.240 = **55% of the 0.440 ceiling**.

**★ THE MEMORISATION SIGNAL IS REAL, AND `containment` IS BLIND TO IT.** Nucleotide containment
never exceeds **0.021** at any length — nowhere near the 0.80 WARN gate — so the DNA novelty gate
sees nothing at all. **Protein AAI sees it clearly and monotonically:** max 0.617 → 0.620 → 0.801 →
0.793 → **0.914**, and the *median* jumps from **0.000 at every other length to 0.291 at L=500**.
At 500 nt half the outputs have a protein resembling a training protein; at ≤100 nt most have no
protein match at all. This is exactly the phage paper's warning — visible only because T3.2
(protein novelty) exists. A DNA-only novelty gate would have passed L=500 as clean.

**L\* = 8 nt.** L=500's higher rate is **not statistically distinguishable** from L=8 (12/50 vs
8/50, p=0.227), so its only real difference is the memorisation signal. L=8 has the best rate among
the clean lengths, median AAI 0.000, max AAI 0.620, and independently reproduces the phage paper's
4–8 nt optimum.

⚠️ **Selection-rule note.** The pre-registered Stage-1 rule read "longest length whose max
containment stays under 0.80, then best on-class rate" — which taken literally selects **L=500**,
because containment never fires. The rule's *intent* was "longest length before memorisation
starts", and the memorisation appeared on an axis the rule did not name. Stage 1 is explicitly a
**tuning** stage, not confirmatory, so refining the criterion here is legitimate — but it is
recorded rather than silently applied. **Stage 2 must pre-register both gates: containment AND
protein AAI.**

**Also:** `n_class_domains ≥ 2` rises with seed length (0/1/0/3/4) but is confounded with seed
informativeness; still 4/50 at best vs 9/31 = 29% for real cores. The cluster gap persists.

---

## 2026-08-17 — ★ Finding: at L=500 the model RECONSTRUCTS the seed's own cluster; at L=8 it does not

Asked whether the generated domain relates to the cluster the seed came from. Each of the 50
generations per cell uses a **different** seed — the first `seed_nt` of a different val record —
and every record stores `seed_accession`, so this is directly checkable.

| seed length | on-class hits | hits whose generated domain matches **their own source cluster** |
|---|---|---|
| **8 nt** | 8/50 | **0 / 8** |
| **500 nt** | 12/50 | **12 / 12** |

**Perfect separation.** At 500 nt every single on-class generation produced a domain the source
cluster actually has — PF04055→PF04055, PF02624→PF02624, PF03070→PF03070. That is **recall of the
seeded instance, not generation.** At 8 nt not one hit matched its source: 6 of 8 produced PF05114
regardless of what the source carried, and PF05114 is simply the **most common** RIPP marker in
real cores (12/50). The model is emitting a class prior, not a memory.

**Mechanism.** 86% of RIPP cores begin *at* the marker gene, and the median core is 1.9 kb, so a
500-nt seed hands over a large fraction of the first marker gene; the continuation finishes it and
the domain is detected in the continuation. `seed_generate.py` already has **`--no-boundary-orf`**
for exactly this — it truncates the seed at the last in-frame stop so no ORF spans seed→
continuation, forcing any class-defining domain to be de novo. **The sweep did not use it.**

⇒ This is the same phenomenon as the AAI rise (max 0.914, median 0.000→0.291) and the higher rate
(0.240) at L=500. All three are one thing: **at 500 nt we are handing the model most of a gene and
it completes it.** It is precisely the "the model is just filling in something that already exists"
objection, now measured rather than argued.

⇒ **Strongly confirms L\* = 8 nt**, and on a much more interpretable axis than AAI. Also makes
`--no-boundary-orf` **mandatory for Stage 2**, and worth reporting as an adversary control.

---

## 2026-08-18 — Intervention: P3-B1 Stage 2, confirmatory seeded arms. **The lift is class-specific.**

- **Pre-registered** §8.5 before generating. **Deviation:** n = **188**, not the registered 200 —
  only 188 of 200 test prompts are ≥ `seed_nt+500` nt. Uniform across all five arms, fixed by the
  data not by the results, and reported rather than quietly absorbed.
- **Provenance:** `phase3_RIPP_stage2` · L=8 nt · `--no-boundary-orf` · TEST seeds
  (`eval_prompts.jsonl`) · `--seed 11` · 2,200 nt generated · `OBLIGATE_DOMAINS[RIPP]` · window
  2,000 nt · substrate `evo2_1b_base` · THE PHASE-3 REPORTING SET on every arm.

| arm | | on_class | rate | 95% CI | generic | ≥2 markers | max AAI |
|---|---|---|---|---|---|---|---|
| **S2-1** | RIPP LoRA + real seed | **33/188** | **0.176** | [0.124, 0.238] | 0.197 | 2 | 0.697 |
| **S2-2** | general adapter + real seed | **0/188** | **0.000** | [0.000, 0.019] | **0.181** | 0 | 0.714 |
| S2-3 | base 1B + real seed | 0/188 | 0.000 | [0.000, 0.019] | 0.000 | 0 | 0.000 |
| S2-4 | RIPP LoRA + **shuffled** seed | 35/188 | 0.186 | [0.133, 0.249] | 0.218 | 1 | 0.781 |
| ~~S2-5~~ | ~~mismatch tag~~ | — | — | — | — | — | — |

**★ RESULT 1 — the lift is CLASS-SPECIFIC, decisively.** S2-1 **0.176 vs S2-2 0.000**,
**p = 2.5 × 10⁻¹¹**. The general all-class adapter, given the identical RIPP seed, produces
**0/188** RIPP domains while producing **0.181 generic** biosynthetic domains — it writes plenty of
biosynthetic protein, just never RIPP. Base 1B writes neither (0/188 on both). This is the §7
comparison the whole phase was built around, and it is unambiguous.

**★ RESULT 2 — at 8 nt the seed's CONTENT is irrelevant.** Codon-shuffling the seed changed
nothing: **0.186 vs 0.176, p = 0.656**, shuffle verified to have landed (157/188 seeds differ;
31 unchanged because shuffling an 8-mer often returns itself). ⇒ **The seed is not supplying RIPP
information — it is supplying a prefix.** What it buys is the 0.027 → 0.176 lift over de novo
(**p < 10⁻⁴**), i.e. "start writing a gene here", and the *class* comes entirely from the adapter.
This is the cleanest possible answer to "the model is just completing a real cluster": the cluster
identity of the seed does not matter, and destroying it costs nothing.

**Consistency:** S2-1 0.176 (test seeds, n=188) vs Stage-1 L=8 0.160 (val seeds, n=50), p=0.49 —
the tuning estimate replicated on held-out prompts.

### ⚠️ S2-5 IS UNINFORMATIVE — the treatment did not land

`--mismatch-tag` was accepted, recorded `mismatch_tag: True`, and **did nothing**: `tag_class` was
`RIPP` for all 188 records, and the generations are **byte-identical to S2-1 (188/188)**.
**Cause:** `seed_generate.py:290` — `others = [c for c in classes_present if c != cls]; tag_cls =
others[...] if others else cls`. With `--classes RIPP` there is no other class, so the flag
silently degrades to a no-op. **It needs ≥2 classes in `--classes`.**
Per Standing Constraint 5 this is **"uninformative", not a negative result.** Recorded in
`bugs.md`. The manipulation check caught it — which is precisely the Phase-2 weighted-arm failure
not repeating.

**The cluster gap is unchanged and now very well powered:** ≥2 distinct RIPP markers in
**2/188** (S2-1) and **1/188** (S2-4), against **9/31 = 29%** for real cores. Novelty is clean
everywhere — max containment ≤0.018, max AAI ≤0.781, no record ≥0.95 on either gate.

---

## 2026-08-18 — ★ antiSMASH run on Phase-3 output for the FIRST TIME

The Phase-3 battery is Pfam-only, so ladder rungs 4–5 (`antiSMASH detection`, `correct_class`) had
**no Phase-3 numbers at all** — not a negative result, simply never computed. Run now on 218
sequences (user's suggestion). `--minimal`, prodigal gene-finding, DBs at
`/data2/ds85/antismash_db`, class map applied, 2,000-nt window to match the endpoint.

| set | n | `is_bgc` | `correct_class` |
|---|---|---|---|
| **S2-1 on-class** (RIPP adapter) | 33 | **0.485** | **0.485** |
| **S2-4 on-class** (shuffled seed) | 35 | 0.429 | 0.429 |
| S2-1 **off**-class | 50 | 0.040 | 0.040 |
| S2-4 **off**-class | 50 | 0.000 | 0.000 |
| **real held-out cores** (ceiling) | 50 | **0.760** | **0.740** |

**Readings.** `is_bgc` — antiSMASH detects a cluster; higher better; ceiling 0.760.
`correct_class` — the detected cluster is RIPP; higher better; ceiling 0.740.

**Synthesis.**
1. **Of our Pfam-on-class generations, antiSMASH confirms 48.5% as real BGCs — and every single
   detection is the CORRECT class** (`is_bgc` == `correct_class` exactly, all five rows). The model
   is not producing off-target clusters; when it produces a cluster, it is a RIPP.
2. **Arm-level, gold-standard:** 33/188 on-class × 0.485 ⇒ ~**16/188 = 0.085** antiSMASH-confirmed
   RIPP clusters, against a **0.740** real-core ceiling — **~11.5% of ceiling** on the strictest
   metric we own. That is the number to quote to a reviewer.
3. **Our Pfam gate is well-calibrated, and inflates ~2×** — 0.176 Pfam vs 0.085 antiSMASH-confirmed,
   consistent with the documented ~2.6× proxy inflation. Off-class records are 0.000–0.040, so the
   gate is not hiding real clusters.
4. ⇒ **Standing Constraint 8 is now testable and looks obsolete.** It says `correct_class` is not a
   de novo optimisation target because it reads ~0. Under a class-specific adapter it reads 0.485
   among on-class records. Rewrite rather than delete — see the audit.

---

## 2026-08-18 — ⚠️ CORRECTION: the Stage-1 AAI trend was an artifact of POOLING

[INCORRECT] - protein AAI rises monotonically 0.617 → 0.620 → 0.801 → 0.793 → 0.914, and its median jumps from 0.000 at every other length to 0.291 at L=500
[CORRECTION - 2026-08-18]: Those were **pooled over all 50 records per cell**, and are dominated by
how many records have *any* protein hit rather than how similar the hits are. Recomputed **among
on-class records only**, the trend does not hold:

| seed length | on-class n | median AAI **among on-class** | pooled median (as reported) |
|---|---|---|---|
| 8 nt | 8 | **0.499** | 0.000 |
| 100 nt | 5 | **0.635** | 0.000 |
| 500 nt | 12 | **0.450** | 0.291 |

L=500's on-class AAI (0.450) is **lower** than L=8's (0.499). The 0.914 maximum is a single outlier.
**The L=500 memorisation conclusion still stands, but on the domain-match evidence alone** — 12/12
on-class hits reproducing their own source cluster's domain, which is independent of AAI and much
stronger. The AAI framing was over-claimed.

**And the answer to "is AAI = 0 also a bad signal?"** (user, 2026-08-18) — a fair worry, and the
reference we had never computed settles it. Best AAI vs the RIPP training proteins:

| set | records with a hit | median AAI |
|---|---|---|
| **REAL held-out RIPP cores** | 98.3% | **0.641** |
| **S2-1 on-class generations** | **97.0%** | **0.496** |
| S2-1 off-class generations | 10.0% | 0.000 |

⇒ **On-class generations are homologous but more divergent than nature** — 0.496 against a real
held-out core's 0.641, with the same ~97% hit rate. That is the ideal novelty profile: recognisably
a family member, further from training than a real held-out cluster is. The 0.000 medians were
**entirely** off-class records, which fail the endpoint anyway. Pooled AAI is not interpretable;
**report AAI among on-class records, against the real-core reference.**

---

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->
