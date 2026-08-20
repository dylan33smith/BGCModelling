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

Generalised as `scripts/fanout.sh`; the *principle* is in `CLAUDE.md` and **the measured curve
lives here** (it is a finding, not a rule): **N=1 → 124 seq/h at 41% util · N=3 → 432 seq/h at 100%
util · N=5 → ~300 seq/h (regression).** N=3 was the optimum for 1B generation at 2.2 kb on an idle
H100; utilisation, not memory, is the signal to stop.

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

## 2026-08-18 — ⚠️ THE LADDER AUROCs DO NOT TRANSFER to the class-specific regime

Re-derived at the user's request, using the new antiSMASH labels as the independent outcome — the
same outcome the original derivation used, so this is a like-for-like re-test in a new regime.

**Method and the trap avoided.** The on-class pool was *selected* on `best_bio_bits > 0`, so pooling
on-class with off-class and computing AUROC for a bio-derived metric is **circular**. The honest
test is **within the on-class pool**: among sequences that pass our Pfam gate, do the rungs predict
whether antiSMASH calls a cluster? n=68 (S2-1 + S2-4 on-class), 31 antiSMASH positives (0.456).

| metric | **AUROC now** | original | verdict |
|---|---|---|---|
| `best_bio_bits` | **0.575** | 0.950 | **does not transfer** |
| `best_any_bits` | 0.575 | — | no information |
| `n_bio_orfs` | 0.532 | — | no information |
| `max_orf_aa` | 0.520 | 0.709 | does not transfer |
| `n_bio_domains` | **0.519** | 0.919 | **does not transfer** |
| `co_orient` | 0.511 | — | no information |
| `biosynthetic_fraction` | 0.500 | — | no information |
| `bio_span_frac` | **0.173** | 0.896 | **INVERTED — anti-predictive** |

Pooled on+off-class for contrast (circular, do not quote): 0.851 / 0.827 / 0.740.

**Why, and it is not that the metrics are wrong.** The original AUROCs were measured on **long-seed
(≈500 nt) arms scored against the GLOBAL Pfam set**, a regime with real spread in domain content.
In the current regime there is **almost no variance left to rank with**: `n_class_domains` among
on-class records is **1.000**, and real cores truncated to the 2 kb window average only **1.04**
biosynthetic domains. A metric that is effectively constant cannot discriminate, and AUROC ≈ 0.5 is
exactly what a constant returns. `bio_span_frac` inverting is the same degeneracy — with one domain
it reduces to "how much of the window does the single gene span", which is not a cluster measure.

**Consequences, and they are real.**
1. **Stop quoting 0.950 / 0.919 / 0.896 as validation of these metrics in Phase 3.** They validated
   adopting the metrics in the Phase-2 regime. They say nothing about Phase-3 rankings.
2. **`bio_span_frac` must not be used as a cluster-structure rung at a 2 kb window.** It is
   degenerate-to-inverted there. `n_class_domains` (a count) is the honest cluster measure.
3. **Among on-class records, nothing we measure predicts antiSMASH confirmation** (best 0.575). The
   Pfam gate is a good *filter* — off-class records are confirmed 0–4% vs on-class 45.6% — but it
   cannot rank *within* its own positives. **Any pruning or RL that ranks on-class candidates by a
   ladder metric is ranking on noise.** That directly constrains [P3-B2a] and [P4-RL].
4. ⇒ **The 2 kb window is itself implicated** — real cores show 1.04 domains in it. See the
   window question in `plan.md`.

---

## 2026-08-18 — ★ [P3-WIN] The cluster gap is NOT a window artefact. It is real, and now localised.

Tested the possibility that "one enzyme, not a cluster" was an artefact of scoring in a 2 kb window
where even real cores show ~1 domain. **Design: the sequence set is held FIXED** — 85 A0 de novo
generations ≥8 kb and 68 real held-out cores ≥8 kb — and only the scoring window varies. Every row
below is the same sequences.

| set | window | on_class | rate | ≥2 markers | mean markers \| on-class | mean bio domains \| on-class | mean n_orfs |
|---|---|---|---|---|---|---|---|
| A0 generated | 2,000 | 2/85 | 0.024 | **0** | 1.00 | 1.00 | 2.1 |
| A0 generated | 4,000 | 2/85 | 0.024 | **0** | 1.00 | 1.00 | 3.6 |
| A0 generated | 8,000 | 2/85 | 0.024 | **0** | **1.00** | **1.00** | 7.0 |
| REAL cores | 2,000 | 35/68 | 0.515 | 14 | 1.60 | 1.69 | 2.0 |
| REAL cores | 4,000 | 38/68 | 0.559 | 16 | 1.63 | 2.16 | 4.1 |
| REAL cores | 8,000 | **42/68** | **0.618** | **19** | 1.69 | **2.67** | 7.8 |

**Per metric.** `on_class` / `rate` — carries ≥1 RIPP marker; higher better. `≥2 markers` — the
cluster test; higher better. `mean markers | on-class` — Stage B, positives only. `mean bio domains
| on-class` — Stage B, all biosynthetic families not just RIPP. `mean n_orfs` — genes called;
confirms the window is actually doing something.

**Synthesis — three findings, and the third is the important one.**
1. **The window works.** `n_orfs` rises 2.1 → 7.0 (generated) and 2.0 → 7.8 (real). We really are
   looking at ~4× more sequence.
2. **Real cores GAIN structure further out**, exactly as the artefact hypothesis predicted they
   would: rate 0.515 → 0.618, ≥2 markers 14 → 19, and **mean biosynthetic domains 1.69 → 2.67
   (+58%)**. So a wider window genuinely recovers cluster content — when there is cluster content.
3. **Generations gain NOTHING. Identical at all three windows** — 2/85, zero records with ≥2
   markers, exactly 1.00 markers and 1.00 biosynthetic domains per positive, at 2 kb, 4 kb and 8 kb
   alike. ⇒ **The gap is not where we were looking. It is real.**

**What this localises.** At 8 kb the model writes **7.0 ORFs against real cores' 7.8** — it is not
failing to produce genes, and not failing to produce *length*. Of those ~7 genes, **1.00 carries a
biosynthetic domain versus 2.67 in real cores.** The deficit is specifically **biosynthetic content
per gene**, not gene count, not sequence length, and not scoring window. That is a much sharper
statement of the limitation than "one enzyme not a cluster", and it is the number [P4-WIDE] has to
move.

⚠️ **Scope:** this is the STRICT (A0) adapter, **de novo**. The seeded arm and the WIDE arm may
differ and must be run through the identical fixed-set/varying-window design before being compared.
Each window keeps its own real-core ceiling (0.515 / 0.559 / 0.618) — never cross-compare windows.

---

## 2026-08-18 — ⛔ LEG 3 HAS NO INSTRUMENT. The class probe anti-correlates with truth.

The prerequisite for [P3-B2a]: can any scorer rank *within* our on-class positives? Ran the cheap
screen — the existing 7B probe as an external oracle over the 68 on-class records that carry
antiSMASH labels.

⚠️ **First, a structural finding:** the only probe on disk, `acts_v2_train500.probe_L16_s0.joblib`,
has `coef_` shape **(22, 4096)** — fit on **evo2_7b** hidden states. Phase-3 generations come from
**`evo2_1b_base`, hidden 1920.** It **cannot** drive guided decoding on the 1B at all, and
`steer_probe_score.log` confirms guided decoding's Q1 +5.71 positive was itself established on the
**7B**. Leg 3 needs a probe that does not exist *and* its positive re-established on the substrate.

**The screen (7B probe reading finished 1B sequences — an oracle, not a decoder hook):**

| | n | median P(RIPP) | argmax==RIPP |
|---|---|---|---|
| S2-1 on-class | 33 | **0.9997** | 27/33 |
| S2-4 on-class | 35 | **0.9965** | 30/35 |

| | value |
|---|---|
| antiSMASH-confirmed / not | 31 / 37 |
| median P(RIPP), **confirmed** | **0.9963** |
| median P(RIPP), **not confirmed** | **1.0000** |
| **AUROC** | **0.337** |
| Mann-Whitney U (greater) | p = 0.9894 |

**Reading.** AUROC **below 0.5** means P(RIPP) is *anti*-correlated with being a real cluster — the
records the probe is most certain about are **less** likely to be antiSMASH-confirmed. The mechanism
is **saturation**: median P(RIPP) ≈ 0.997–0.9997, so the probe says "RIPP, ~100%" to essentially
everything that passes our Pfam gate and has no dynamic range left to rank with. This is not a weak
classifier failing — its held-out balanced accuracy on the 22-class task is **0.933**.

**Synthesis, and it closes a leg.** Two independent instrument families have now been tested for
*within-positives* discrimination and both fail: **ladder metrics** best 0.575 (`bio_span_frac`
inverted at 0.173), **class probe** 0.337. Nothing we own can tell a real cluster from a
Pfam-passing near-miss. ⇒ **Pruning and RL-by-ranking have no instrument. [P3-B2a] stays closed,
and this is now a measured closure rather than an unpowered one.**
⇒ **Do NOT fit a 1B probe.** The screen was designed so its negative would be decisive, and it is:
the failure is saturation against a target the probe cannot see, which a 1B refit would inherit.
⇒ **Rewards must be built from verified gate passes** (antiSMASH confirmation, marker identity),
never from a continuous score. [P4-RL] is unaffected; [P4-RL-1] already specified `JOINT_PASS`.

---

## 2026-08-18 — [P4-WIDE] de novo: UNINFORMATIVE, not negative. The powered test is the seeded one.

WIDE adapter trained cleanly (3,723 records, 3 epochs, 675 steps, 87 min, `loss_ce` 1.309 → 0.844)
and generated 150 de novo at 8 kb, matching A0_8k's regime so the window sweep is comparable.
Fixed-set window sweep, 79 WIDE generations ≥8 kb vs the same 68 real cores:

| set | window | on_class | rate | ≥2 markers | mean bio domains \| on-class | mean n_orfs |
|---|---|---|---|---|---|---|
| **WIDE generated** | 2,000 | 0/79 | 0.000 | 0 | — | 2.2 |
| **WIDE generated** | 4,000 | 0/79 | 0.000 | 0 | — | 3.9 |
| **WIDE generated** | 8,000 | 1/79 | 0.013 | 0 | **3.00** | 7.3 |
| A0 (strict) | any | 2/85 | 0.024 | 0 | 1.00 | 2.1→7.0 |
| REAL cores | 8,000 | 42/68 | 0.618 | 19 | 2.67 | 7.8 |

**WIDE vs A0 is n.s. at every window** (p = 0.498 / 0.498 / 1.000). **And the test is not powered:**
at a de novo base rate of ~0.024, detecting even a *doubling* needs **n ≈ 800/arm**; we ran ~80.
Per Standing Constraint 5 this is **UNINFORMATIVE, not a negative result.** WIDE is not shown to
fail; it is untested.

**My design error, recorded.** I generated de novo to match A0_8k's regime for window comparability.
That was right for comparability and **wrong for power** — de novo is the 0.024 regime. The powered
test is **seeded at L\*=8**, where the base rate is 0.176 and n=188 is already demonstrated adequate.
⇒ **Next: WIDE seeded at L=8, n=188, against S2-1 (0.176) — plus the [P4-WIDE-CTRL] size-matched
STRICT arm.**

⚠️ One anecdote worth keeping: the single WIDE hit at 8 kb carried **3 biosynthetic domains**, the
most any generation has produced (every A0/Stage-2 positive carried exactly 1). **n=1, not a
result** — but it is the first generated sequence with multi-domain content, and it is what the
seeded arm should be watched for.

---

## 2026-08-19 — ★ [P4-WIDE-SEEDED] WIDE IS REFUTED on a powered test. And the cause is DILUTION.

Pre-registered §8.6/§8.7. Overnight pipeline, `PIPELINE_OK`, 5 arms × n=188, 11 scorings, 615
antiSMASH calls. **Headline rates are antiSMASH-corrected by stratified sampling** (all
Pfam-positives + 100 sampled Pfam-negatives per arm).

| metric | W-1 WIDE 2.2k | W-2 STRICT 2.2k | W-1 WIDE 8k | W-2 STRICT 8k | STRICT-full 8k | real cores |
|---|---|---|---|---|---|---|
[INCORRECT] - | `best_bio_bits`>0 * (Pfam) | 11/188 = 0.059 | 36/188 = 0.191 | 8/188 = 0.043 | 36/188 = 0.191 | 24/188 = 0.128 | 0.515 |
[CORRECTION - 2026-08-19]: **every denominator in this row is 4x inflated** — the fan-out wrote
four byte-identical copies of the same units. On UNIQUE records: W-1 2.2k **7/141 = 0.050** ·
W-2 2.2k **9/47 = 0.191** · W-1 8k **2/47 = 0.043** · W-2 8k **9/47 = 0.191** · SF 8k
**6/47 = 0.128**. Rates are essentially unchanged (uniform duplication); **n is not**.
[INCORRECT] - | **antiSMASH CORRECTED** * | 0.027 | 0.043 | **0.000** | 0.085 | **0.116** | **0.760** |
[CORRECTION - 2026-08-19]: recomputed on unique records — 0.028 · 0.043 · 0.000 · 0.085 ·
**0.128** · 0.760. Point estimates hold; the effective n behind them is **47–141, not 188**,
and the confirmation rates `rp` rest on **2–9 unique** Pfam-positives per arm, not 8–36.
| conf(Pfam-positive) | 0.455 | 0.222 | **0.000** | 0.444 | 0.500 | — |
| `containment` * max | 0.004 | 0.013 | 0.001 | 0.025 | 0.014 | — |
| `protein_aai` * max | 0.531 | 0.627 | 0.558 | 0.660 | 0.678 | 0.641 |
[INCORRECT] - | `JOINT_PASS` | 3 | 0 | 0 | 0 | 0 | — |
[CORRECTION - 2026-08-19]: the zeros are an **artefact of the duplication**, not a model
property — `JOINT_PASS` requires intra-set distinctness, and every record in a 4x-duplicated
set has an exact twin, so the gate could not be passed by construction. `JOINT_PASS` for
these five arms is **UNMEASURED**, not zero.
| **`n_class_domains`≥2** | **0/188** | **0/188** | **0/188** | **0/188** | **0/188** | **14/68** |
| `n_bio_domains` \| on-class | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.69 |

[INCORRECT] - **Contrasts, Holm-corrected (§9.1):** W-1 vs W-2 at 2.2 kb **p=4.1e-04**; at 8 kb **p=3.2e-05** —
[CORRECTION - 2026-08-19]: recomputed on unique records, same Fisher/Holm family of 4 —
**2.2 kb p=0.0053, Holm p=0.021 (STILL SIGNIFICANT)**; **8 kb p=0.050, Holm p=0.15 (NOW n.s.)**.
⇒ **WIDE is still refuted, but on ONE window, not two, and at p=0.02 rather than 4e-04.**
W-2 vs STRICT-full (dataset size) stays n.s. (p=0.57). See the 2026-08-19 fan-out entry.
**WIDE significantly WORSE both times.** W-2 vs STRICT-full **p=0.79** — the 7,250→3,723 training-set
drop costs nothing, so **span width is isolated as the cause**. Generation length 8 kb vs 2.2 kb
**p=0.50**, n.s.

**★ CAUSE: DILUTION — measured, and it refines the hypothesis.** Paired STRICT vs WIDE on the same
250 clusters:

| | STRICT | WIDE | ratio |
|---|---|---|---|
| median length | 1,146 nt | 4,052 nt | 3.53× |
| mean ORFs | 1.52 | 3.76 | 2.48× |
| **coding density** | **0.976** | **0.938** | **0.96×** |
| **biosynthetic fraction of span** | **0.683** | **0.477** | **0.70×** |

⇒ Per 1,000 training nt: **STRICT 683 nt biosynthetic, WIDE 477 nt — 1.43× less signal per token.**
⚠️ **It is NOT intergenic space.** Coding density is essentially unchanged (0.976 → 0.938). The
extra sequence is **other genes**, not gaps. WIDE did not add context around the biosynthetic
machinery; it added *non-biosynthetic protein* the model must also learn to write.

**★ THE ANOMALY IS INTERPRETABLE.** W-1 at 8 kb: 8 Pfam-positives, **0/8 antiSMASH-confirmed**.
Under every confirmation rate the other arms show (0.444–0.500), P(0 of 8) = **0.004–0.009** — small
n, but a real signal. Reading: **the WIDE adapter emits isolated biosynthetic-looking genes without
cluster context.** Our Pfam gate scores a lone domain as a hit; antiSMASH requires co-location and
rejects all of them. That is precisely what dilution predicts — trained on sparse biosynthetic
content, it learned to produce sparse biosynthetic content.

**The co-primary never moved: `n_class_domains ≥ 2` is 0/188 in ALL FIVE ARMS** — 940 sequences,
five adapters, three windows, not one generation with two distinct RIPP markers. Real cores 14/68.
WIDE was the intervention aimed at this number and it went 2/188 → 0/188.

[INCORRECT] - **Novelty clean everywhere** — max containment ≤0.025, max AAI ≤0.678, both far under 0.95.
[CORRECTION - 2026-08-19]: the containment/AAI gates were clean, but the statement swept past
the diversity gate the scorer had already flagged — `frac_distinct` 0.25 and
`frac_with_a_near_duplicate` 1.00 on three of five arms. **That was the bug announcing itself
and being read as a model finding.** Novelty vs training data: clean. Intra-set: unmeasurable.

⇒ **[P4-WIDE] CLOSES NEGATIVE, powered.** But it closes *with a mechanism*, which is the useful
part: more sequence per record ≠ more biosynthetic signal per record. The next intervention should
add span width **while holding biosynthetic density constant** — i.e. domain-weighted loss on WIDE
spans. ⚠️ Phase-2's weighted arm never landed, so any such run needs a manipulation check first.

---

## 2026-08-19 — ⚠️ CORRECTION: `n_class_domains ≥ 2` WAS THE WRONG TARGET

Prompted by the user asking whether we are measuring the right thing. **We were not**, and I had
been amplifying the error across several reports.

**What I had been claiming.** That "0/188 records with ≥2 distinct RIPP markers, against 29% of real
cores" was a damning structural failure — "one enzyme, not a cluster" — and that moving it was the
central objective of Phase 4/5.

**What the data says.** antiSMASH — the field-standard detector, and our own gold standard —
**confirms RIPP clusters that carry exactly ONE of our markers**:

| arm | records with exactly 1 marker | antiSMASH confirmed |
|---|---|---|
| STRICT-full 8 kb | 24 | **12/24** |
| W-2 STRICT-matched 8 kb | 36 | **16/36** |

Every confirmation was classed **RIPP**. And the marker-count distribution of **real held-out cores**
in the same 2 kb window is: **0 markers 28/50 · 1 marker 14/50 · 2 markers 6/50 · 4 markers 2/50.**

⇒ **Only ~16% of REAL RIPP cores carry ≥2 of our markers in a 2 kb window.** The metric I was
treating as the definition of a cluster is a *minority property of genuine clusters*.

[INCORRECT] - The co-primary never moved: `n_class_domains ≥ 2` is 0/188 in ALL FIVE ARMS ... Real cores 14/68. WIDE was the intervention aimed at this number and it went 2/188 → 0/188.
[CORRECTION - 2026-08-19]: The 0/188 finding is **real but was over-weighted**. `n_class_domains ≥ 2`
is satisfied by only ~16–21% of real cores in the scoring window, and antiSMASH confirms clusters
with a single marker. It is a **stricter-than-field-standard proxy**, not the definition of success.
**The cluster verdict we should quote is antiSMASH confirmation**, where the best arm reads
**0.116 against a 0.760 real-core ceiling — about 15% of ceiling**, not zero.

**Why the proxy misleads.** Our 8 markers are *modifying enzymes*; antiSMASH's RiPP rules also use
**precursor-peptide evidence and gene context**, which our Pfam-only scan cannot see. A real RiPP
cluster is a precursor + modifying enzyme(s) + protease/transporter; requiring two *modifying
enzyme families* is neither necessary nor sufficient.

⇒ **Demote `n_class_domains ≥ 2` from co-primary to diagnostic.** Do not change the pre-registered
PRIMARY mid-phase (Constraint 4) — but stop reporting the ≥2 metric as the headline failure, and
quote **antiSMASH-confirmed rate against the real-core ceiling** as the cluster claim.

---

## 2026-08-19 — ★★★ THE DISCRIMINATOR IS THE PRECURSOR PEPTIDE, NOT MORE ENZYMES

Ran the analysis the whole project should have run months ago: among generations that pass our Pfam
gate with **exactly one** RIPP marker, what distinguishes the ones antiSMASH **confirms** from the
ones it **rejects**? Pooled from STRICT-full 8 kb and W-2 8 kb: **28 confirmed, 32 rejected.**

| | CONFIRMED (n=28) | REJECTED (n=32) |
|---|---|---|
| mean ORFs | **2.43** | 1.38 |
| **mean short ORFs (20–80 aa)** | **0.43** | **0.00** |
| mean distinct Pfam domains | 3.57 | 3.50 |

**★ It is not domain content — that is identical (3.57 vs 3.50).** What separates them is
**an extra gene, and specifically a SHORT one in the 20–80 aa range. Rejected records have ZERO.**

That size range is the **RiPP precursor peptide** — the gene that actually *encodes the product*.
antiSMASH's RiPP rules look for precursor evidence; our Pfam scan cannot, because **none of the 8
`OBLIGATE_DOMAINS[RIPP]` markers is a precursor** (all are modifying enzymes or binding proteins)
and **`find_orfs` defaults to `min_aa=50`** while Prodigal's own floor is ~30 aa.

⇒ **The model's real deficit is not "too few enzymes". It is "no precursor".** Every metric we have
optimised — `best_bio_bits`, `n_class_domains`, `n_bio_domains` — measures the machinery and is
structurally blind to the substrate it acts on. This reframes Phase 5 entirely.

⇒ Marker enrichment is a weak secondary signal: PF05114 appears in 28/28 confirmed vs 24/32
rejected. Nothing else separates at n=60.

## 2026-08-19 — Hypothesis REFUTED: WIDE is not secretly producing "additional" machinery

User's hypothesis: WIDE may be generating `biosynthetic-additional` content that our RIPP-specific
markers cannot see, so its apparent failure is a metric artefact. **Tested and refuted.**

Built the "additional vocabulary" empirically — Pfam domains present in the WIDE span but **absent**
from the STRICT span of the **same** real cluster (60 paired records, full Pfam-A). Top members:
PF09836, PF00106 (short-chain dehydrogenase), PF00561 (α/β hydrolase), PF00753 (metallo-β-lactamase),
PF01370 (epimerase) — i.e. genuine tailoring/accessory enzymes. Took the top 40 as the vocabulary.

| arm | generations carrying ≥1 additional-vocabulary domain |
|---|---|
| **W-1 WIDE 8 kb** | **2/60** |
| **W-2 STRICT 8 kb** | **8/60** |

⇒ The WIDE model produces **less** accessory machinery than the strict control, not more. It is not
succeeding invisibly; it is simply worse, consistent with the measured 1.43× dilution of
biosynthetic signal per token. **[P4-WIDE] stays closed negative.**

---

## 2026-08-19 — ⚠️ LEVEL 3 IS UNMEASURABLE ON THE CURRENT SUBSTRATE. Even real cores score zero.

Re-scored every arm at `min_aa=20` and added a **component panel** — precursor (20–80 aa ORF),
modifying enzyme (`OBLIGATE_DOMAINS[RIPP]`), transporter/protease (PF03412 Peptidase_C39, PF00005
ABC_tran, PF00664 ABC_membrane, PF07690 MFS_1), regulator (PF00196, PF00440, PF13411, PF01381).
n=120/arm, 2,000 nt window, panel = 16 HMMs.

| arm | n | ORF@50 | ORF@20 | short | ENZ | TRANS | REG | E+P | **E+P+T** |
|---|---|---|---|---|---|---|---|---|---|
| **real cores (CEILING)** | 120 | 1.59 | 1.65 | **17** | 68 | **3** | 2 | **8** | **0** |
| base 1B (FLOOR) | 120 | 1.48 | 1.82 | **80** | 0 | 0 | 0 | 0 | **0** |
| A0 de novo | 120 | 1.99 | 2.08 | 26 | 3 | 1 | 1 | 0 | 0 |
| S2-1 seeded 2.2k | 120 | 2.12 | 2.21 | 29 | 21 | 1 | 0 | 4 | 0 |
| W-1 WIDE 2.2k | 120 | 2.23 | 2.39 | 42 | 5 | 5 | 2 | 1 | 0 |
| W-2 STRICT 2.2k | 120 | 1.38 | 1.44 | 18 | 26 | 0 | 3 | 3 | 0 |
| W-1 WIDE 8k | 120 | 2.20 | 2.25 | 25 | 5 | 3 | 3 | 0 | 0 |
| W-2 STRICT 8k | 120 | 1.80 | 1.86 | 19 | 24 | 6 | 2 | 4 | 0 |
| STRICT-full 8k | 120 | 2.25 | 2.39 | 38 | 17 | 6 | 5 | 3 | 0 |

**Three findings, and they redirect Phase 5.**

**1. `min_aa=20` recovers almost nothing.** ORF counts barely move (real 1.59→1.65, S2-1 2.12→2.21).
The precursors were **not** hidden by the `min_aa=50` default. That hypothesis is dead — cheaply,
which is what it was for.

**2. A short ORF alone is a WORTHLESS signal.** The **base 1B floor has the MOST** precursor-sized
ORFs of any arm — **80/120**, against real cores' **17/120** — while producing **zero** RIPP markers.
Random-ish DNA is full of 20–80 aa open frames. ⇒ The confirmed-vs-rejected discriminator found on
2026-08-19 (0.43 vs 0.00 short ORFs) is real but must be read **jointly with an enzyme**, never
alone. `n_short_orfs` is a **diagnostic, not a gate.**

**3. ★ THE BLOCKER: Level 3 cannot be measured here — real cores score 0/120 too.** Transporters
appear in only **3/120** real cores and the full complement (E+P+T) in **0/120**. The cause is
structural: `STRICT_KINDS = {"biosynthetic"}` **excludes transport and regulatory genes by
construction**, so they are absent from the training data *and* from the real-core reference. A
metric no real cluster can satisfy cannot score a generation.

⇒ **Pursuing Level 3 requires changing the evaluation substrate to whole antiSMASH regions**
(median 21,262 nt), not strict cores (median 1,854 nt). That immediately collides with
`evo2_1b_base`'s **8,192-token hard cap** — a full RiPP region does not fit in the model's context.
**The substrate question ([P5-SUBSTRATE]) is therefore a prerequisite for Level 3, not an
alternative to it.**

⇒ Best current arm on components: **STRICT-full 8k** — 17 ENZ, 6 TRANS, 5 REG, 3 E+P of 120. It
leads or ties on transporter and regulator content. **WIDE remains worse on every component column**
(W-1 8k: 5 ENZ vs W-2's 24), independently confirming [P4-WIDE]'s closure.

---

## 2026-08-19 — ORF definition audit, and the raw data DOES carry component annotation

**Q: are we counting start codons, or real start→stop genes?** `find_orfs` uses **Prodigal
(pyrodigal)**, which calls **complete genes with real starts and stops** — not bare start codons,
and not six-frame fragments (the six-frame scanner is RETIRED and gated behind `BGC_EVAL_STRICT`).
**But `ORF.partial` is recorded and NEVER FILTERED**, so genes truncated by the scoring-window edge
are counted as ORFs.

Measured on the 20–80 aa "precursor-sized" ORFs, 2,000 nt window, n=120/arm:

| arm | short ORFs | COMPLETE | partial | % complete | records with a COMPLETE short ORF |
|---|---|---|---|---|---|
| real cores | 17 | 7 | 10 | 41.2% | 7/120 |
| **base 1B (FLOOR)** | 105 | **53** | 52 | 50.5% | **45/120** |
| S2-1 seeded | 36 | 10 | 26 | 27.8% | 6/120 |
| STRICT-full 8k | 48 | 20 | 28 | 41.7% | 12/120 |

⇒ **28–50% of "short ORFs" are window-edge truncation artefacts.** Filtering them changes the
counts but **not the conclusion**: the base-model floor still leads at **45/120** against real cores'
**7/120**. **Short-ORF presence remains a worthless standalone signal** — it must be read jointly
with an enzyme, and what is actually needed is a **precursor detector** (leader/core structure),
not a length filter.
⇒ **Action:** report `n_short_orfs` as complete-only; add `partial` filtering wherever ORF counts
feed a metric.

**Q: is transport/regulatory information in the raw antiSMASH data?** **YES.** Sampled 43 regions
(14 RiPP-like) from `asdb5_gbks/asdb5_gbks.tar` (185 GB): every CDS carries a `gene_kind`
qualifier — `biosynthetic` 166, `biosynthetic-additional` 492, **`regulatory` 124**,
**`transport` 107**, `other` 52 (`<none>` dominates only because the count spans whole genome
records, not region interiors).

⇒ `build_core_records.py` **reads `gene_kind` per CDS** and then stores **only the derived spans** —
the per-gene component annotation was computed and thrown away. Rebuilding a component-annotated
dataset needs a streaming pass over the tar; the information itself is not lost and no re-annotation
is required.

---

## 2026-08-19 — [P5-DATA] + [P5-STEP2]: the 1B CAN host Level-3-minimal. No model change required.

Streamed `asdb5_gbks.tar` (185 GB) keeping **per-CDS `gene_kind` + coordinates** for every RIPP
region in our splits — the annotation `build_core_records.py` computes and discards. Output:
`/data2/ds85/bgcmodel_data/ripp_components.jsonl`. Analysis below at n=4,969 regions (interim; the
pass continues, numbers stable).

**gene_kind census, all CDS inside RIPP regions:** none 66.4% · biosynthetic-additional 14.9% ·
**biosynthetic 6.8%** · **transport 5.0%** · **regulatory 4.7%** · other 2.1% · resistance 0.03%.

### THE DECIDING MEASUREMENT — span of each component set vs the 1B's usable ~7,900 nt

| span definition | median nt | p75 | p90 | mean genes | **fits 1B (≤7,900)** |
|---|---|---|---|---|---|
| STRICT (biosynthetic only) | 2,172 | 5,755 | 15,050 | 1.89 | **81.0%** |
| **COMPLEMENT (bio + transport)** | **6,653** | 14,095 | 27,489 | 3.28 | **55.5%** |
| COMPLEMENT + regulatory | 9,779 | 18,738 | 35,042 | 4.58 | 41.8% |
| WIDE (+ additional) | 12,271 | 21,623 | 37,759 | 6.03 | 32.3% |
| FULL (all annotated) | 17,017 | 26,175 | 42,248 | 9.31 | 18.3% |
| WHOLE REGION | 21,896 | 31,273 | 47,401 | 27.77 | **1.9%** |

⇒ **DECISION: stay on `evo2_1b_base`.** The **enzyme+transport complement fits in 55.5%** of real
RIPP regions — workable, and ~3,000+ training records at our scale. My earlier claim that Level 3
"requires whole regions (21 kb) and therefore a bigger model" was **wrong**: it conflated the whole
antiSMASH region with the *minimal functional complement*. Whole regions are indeed impossible on
the 1B (1.9%), but they are not what Level 3 needs.
⇒ GenomeOcean/7B ([P5-SUB]) is **deferred, not required** — reserve it for whole-region work and the
paper's model comparison.

### Level 3 must be CONDITIONED — 42% of real RIPP clusters have no transporter

| component | regions having it |
|---|---|
| enzyme (biosynthetic) | 4,969/4,969 = 100% |
| **transport** | **2,903/4,969 = 58.4%** |
| regulatory | 2,945/4,969 = 59.3% |
| enzyme + transport | **58.4%** |
| all three | **1,889/4,969 = 38.0%** |

⇒ A "produce enzyme+transporter" endpoint has a **natural ceiling of 0.584**, not 1.0, and
"all three" only 0.380. Any Level-3 rate must be quoted against the *conditioned* ceiling.

### ⚠️ The precursor is gene_kind="none" — which is why STRICT cores exclude it

72.2% of regions carry ≥1 short (20–80 aa) CDS, and **9,313 of them are `gene_kind="none"`** vs 502
`biosynthetic`. So the RiPP precursor is typically **unannotated**, and a biosynthetic-only span
**excludes the gene encoding the product by construction.** That is the mechanism behind the
2026-08-19 confirmed-vs-rejected finding.

**BUT short+unannotated is NOT a clean precursor signal.** Products of those 9,313: *hypothetical
protein* 4,606, exodeoxyribonuclease VII 225, HTH regulator 118, **PQQ precursor peptide 113**,
transposase 92, DUF397 75, ribosomal L32 44, IS-family transposases. And median distance to the
nearest biosynthetic gene is **6,654 nt** with only **41.4% within 5 kb** — most are scattered
through the region, not adjacent to the core.
⇒ **A length+position heuristic will not work. [P5-COMPONENT] must build a real precursor HMM
detector** (leader/core structure), with proximity as a secondary filter.

---

## 2026-08-19 — [P5-COMPONENT] + [P5-STEP4]: THE PRECURSOR IS AT ZERO. Validated detector.

**Built the panels empirically** from 27,481 Pfam-A descriptions rather than from memory: **81
precursor** families (Antimicrobial18 lantibiotic, Bacteriocin_II/IIc double-glycine leader, PQQ
precursor, SkfB…), **302 transport**, 750 regulator, 389 protease. Saved to
`/data2/ds85/bgcmodel_data/component_panels.json`.

**Validation, and the first attempt was wrong.** Scored on real STRICT cores the precursor panel
fired **1/120** — apparently useless. Cause: **strict cores exclude precursors by construction**
(the precursor is `gene_kind="none"`, outside the biosynthetic span). *A component detector cannot
be validated on a substrate that excludes the component.* Re-validated on **WIDE** spans:

| panel | real WIDE | base 1B (FP control) | verdict |
|---|---|---|---|
| **precursor** | **25/120 = 0.208** | **1/120 = 0.008** | **USABLE — 25× discrimination** |
| transport | 52/120 = 0.433 | 1/120 = 0.008 | USABLE |

### FINAL PANEL — length-matched, complete ORFs only, `min_aa=20`

| arm | window | n | PREC | ENZ | TRANS | P+E | E+T | **P+E+T** |
|---|---|---|---|---|---|---|---|---|
| **REAL WIDE (CEILING)** | 8,000 | 120 | **21** | 46 | 42 | **11** | 14 | **3** |
| REAL WIDE | 2,000 | 120 | 2 | 15 | 5 | 0 | 1 | 0 |
| base 1B (FLOOR) | 2,000 | 120 | 0 | 0 | 0 | 0 | 0 | 0 |
| **STRICT-full 8k (BEST ARM)** | 8,000 | 120 | **0** | 3 | **17** | 0 | 0 | **0** |
| W-2 STRICT 8k | 8,000 | 120 | 0 | 0 | 11 | 0 | 0 | 0 |
| W-1 WIDE 8k | 8,000 | 120 | 0 | 3 | 24 | 0 | 3 | 0 |
| S2-1 seeded 2.2k | 2,000 | 120 | 0 | 1 | 2 | 0 | 0 | 0 |
| A0 de novo 8k | 8,000 | 120 | **2** | 4 | 2 | 0 | 0 | 0 |

⚠️ ENZ counts here are far below the earlier component panel (STRICT-full 3 vs 17) because this scan
**filters `partial` ORFs** and uses an 8,000 nt window. Compare only *within* this table.

**★ FINDING 1 — precursor generation is at ZERO.** Every seeded arm reads **0/120** against a real
ceiling of 21/120. The one exception is A0 de novo at 2/120. **The missing component is confirmed
with a validated detector, not inferred.**
**★ FINDING 2 — transporters ARE being generated.** Best arm 17/120, W-1 WIDE 24/120, against real
42/120. So the models produce ~40–57% of the real transporter rate. **Transport is not the problem.**
**★ FINDING 3 — the full complement is rare even in REAL data at this window: 3/120 = 2.5%.** Taking
the first 8,000 nt of a real WIDE span shows precursor+enzyme+transport only 2.5% of the time,
because the complement's p75 span is 14,095 nt. **A P+E+T endpoint measured at 8 kb has a ~2.5%
ceiling — too low to power any experiment.**

⇒ **Level 3 is blocked by the SUBSTRATE, not the model.** Training data built from
`STRICT_KINDS={"biosynthetic"}` **cannot teach precursor generation** because it contains no
precursors. No amount of training, weighting or composition on that substrate will fix it.

---

## 2026-08-19 — ⚠️ CORRECTION: my precursor "detector" was ~half ENZYME. Use RODEO/antiSMASH instead.

User challenged the rigour of the component panels. **The challenge was correct.**

**How the three panels were actually built — three different methodologies, silently mixed:**
- **ENZ** = `OBLIGATE_DOMAINS[RIPP]`, 8 accessions, **data-derived** from our own corpus
  (`derive_class_markers.py`: keep Pfams with freq≥0.3 & enr≥4, OR freq≥0.08 & enr≥8). Defensible.
- **PREC / TRANS** = **regex keyword match over Pfam-A `NAME`+`DESC` text.** Not curated, not
  validated per-family, not derived from BGC data. 81 and 302 families respectively.

**The flaw, measured.** Of the six "precursor" families that actually fired on real WIDE spans:

| accession | name | what it really is |
|---|---|---|
| PF14028 | Lant_dehydr_C | ⚠️ **lantibiotic DEHYDRATASE — an ENZYME**, and already in `OBLIGATE_DOMAINS[RIPP]` |
| PF04738 | Lant_dehydr_N | ⚠️ **lantibiotic DEHYDRATASE — an ENZYME** |
| PF03515 | Cloacin | ⚠️ colicin tRNase — a **toxin**, not a RiPP precursor |
| PF10439 | Bacteriocin_IIc | ✅ genuine (double-glycine leader peptide) |
| PF09683 | Lactococcin_972 | ✅ genuine |
| PF28317 | Lant_leader_dom | ✅ genuine (Class I lanthipeptide leader) |

**Half the panel is enzymes or toxins.** The regex matched `lantibiotic` and caught the *dehydratase
that modifies* the precursor rather than the precursor. **PREC and ENZ were not disjoint**, so the
"P+E" combination was partly tautological — one PF14028 hit lit up both columns.

[INCORRECT] - | panel | real WIDE | base 1B (FP control) | verdict | ... | **precursor** | **25/120 = 0.208** | **1/120 = 0.008** | **USABLE — 25× discrimination** |
[CORRECTION - 2026-08-19]: The whole-panel validation (25/120 vs 1/120) was real but **uninformative
about precursors specifically** — it validated a mixed enzyme/precursor panel against a floor that
has neither. Recomputed with the enzyme contaminant removed: precursor **17/120** (from 19), and
**P+E drops 9/120 → 7/120**. Removing PF04738 and PF03515 as well would lower it further. **The
Level-3 P+E numbers in the 2026-08-19 final panel are not trustworthy and must be re-derived.**

**What the literature says we should have done** (searched 2026-08-19):
- RiPP precursors are **typically <150 aa and frequently unannotated** — our observation confirmed.
- **Standard gene callers are not optimised for short ORFs**; the field uses **Prodigal-short** /
  **Prodigal-shorter** (down to ~5 aa), not a `min_aa` tweak on stock Prodigal.
- **Pfam alone is explicitly insufficient** — the field pairs it with **RiPP-specific HMMs**.
- The standard tool is **RODEO** (HMM + heuristic scoring + supervised ML), and
  **antiSMASH's RiPP modules already run RODEO for precursor validation.**
- Alternatives: **NeuRiPP** (neural net), **DeepRiPP**, **RiPPMiner**, **RiPPER**.

⇒ **We already own the right detector and have been discarding its output.** antiSMASH has been run
on 833 of our sequences and we kept only `is_bgc` / `class_match`. **Parse antiSMASH's precursor
calls instead of rolling our own panel.** This is the same error as `build_core_records.py`
computing `gene_kind` and storing only spans.
⇒ **TRANS panel (302 keyword families) is equally unvalidated per-family** and should be treated as
provisional until checked the same way.

---

## 2026-08-19 — ★★★ FIDELITY BASELINE: precursor detection tops out at ~8% on OUR RIPP data

User asked the right question: *did real sequences produce zero precursors under the same method, and
how are we validating fidelity at all?* We had never established a sensitivity baseline. Measured now.

**First, a discovery that invalidates an assumption.** Our antiSMASH calls run with **`--minimal`**
(`evaluation.py:603`), which **disables every analysis module** — so **we have never run RODEO**, and
antiSMASH output went to a `TemporaryDirectory` that is deleted after `is_bgc`/`class_match` are read.
All 833 antiSMASH results to date are **detection-only**.

**Ran FULL-mode antiSMASH** (no `--minimal`) on 12 real RIPP wide spans, output retained at
`phase5_detect/as_full/`. RiPP modules ran: `lanthipeptides`, `lassopeptides`, `sactipeptides`,
`thiopeptides`. Precursor predictions live in `modules.<mod>.motifs` and `new_cds_features`.

| | value |
|---|---|
| regions analysed | 12 |
| antiSMASH detected a cluster | **12/12** |
| **regions with ≥1 precursor motif** | **1/12 = 8%** |

**★ WHY, and it reframes the target.** Precursor prediction only exists for RiPP **subclasses with a
dedicated module**. Our RIPP class is dominated by antiSMASH's **catch-all**:

| product label | share |
|---|---|
| **RiPP-like** (catch-all, NO precursor module) | **35.5%** |
| azole-containing-RiPP | 8.8% |
| cyclic-lactone-autoinducer | 7.5% |
| RRE-containing | 6.7% |
| ranthipeptide | 6.2% |
| lassopeptide | 5.8% |
| lanthipeptide-class-i…v | 11.9% combined |

**Only 20.3% of our RIPP labels have a dedicated antiSMASH precursor module. 79.7% do not.**

⇒ **The precursor endpoint has a hard ceiling near 20% on this class as defined** — and empirically
8% in a 12-region sample. **We would be asking the model to produce something we cannot detect in
four out of five real clusters.** That is not a measurable endpoint.

⇒ **This is the fidelity standard the project lacked.** Rule going forward: **before adopting any
component metric, measure its sensitivity on REAL positives.** A detector that fires on 8% of real
clusters cannot score a generation, no matter how clean its false-positive rate. The earlier
keyword panel failed the same test in the opposite direction — high aggregate discrimination,
wrong per-family content.

⇒ **Three viable responses, in order of cost:**
1. **Narrow the class** — restrict to RiPP subclasses that HAVE precursor modules (lanthipeptides,
   lassopeptides, thiopeptides, sactipeptides ≈ 20% of current data, ~2,000 records). The endpoint
   becomes measurable, the dataset shrinks, and the paper claim narrows to "lanthipeptide-class RiPPs".
2. **Keep Level 2 as the headline** (antiSMASH cluster call, 0.116 vs 0.760) and report component
   content descriptively without a precursor gate.
3. Adopt an external precursor predictor (NeuRiPP/DeepRiPP) and **validate its sensitivity on our
   real regions first** — same gate, no exceptions.

---

## 2026-08-19 — Precursor absence is a DETECTION gap, not biology. But the detector caps at ~50%.

Direct test of "are precursors missing, or just untagged?" Full-mode antiSMASH on two matched sets
of 12 real RIPP wide spans.

| region set | n | with precursor motif | rate |
|---|---|---|---|
| mixed subclasses (**as our RIPP class actually is**) | 12 | 1 | **8%** |
| **only subclasses with a dedicated precursor module** | 12 | 6 | **50%** |

⇒ **Both effects are real.** Restricting to module-covered subclasses raises detection **6×**, so the
8% is largely a **coverage gap** — the precursors are there, antiSMASH just has no model for 79.7%
of our subclasses. **But even with a dedicated module, detection is only 50%**, so the detector
itself is also limiting.

⇒ **A precursor-based endpoint would have a ~50% ceiling on ~20% of the data.** Not viable as a
gate. Usable at best as a descriptive secondary on a narrowed class.

## 2026-08-19 — ⚠️ CORRECTION: "precursor" is RiPP-specific, NOT a general BGC component

[INCORRECT] - Level 3 (precursor + enzyme + transporter) ... the minimal functional complement
[CORRECTION - 2026-08-19]: **That is the RiPP complement, not the BGC complement.** I generalised a
class-specific fact into a project-wide target. **NRPS and PKS clusters have no precursor peptide
at all** — megasynthases assemble the product from amino-acyl/malonyl building blocks. Terpenes,
alkaloids, saccharides likewise have no precursor peptide. The concept is definitional *only* for
RiPPs, where the product **is** the post-translationally modified precursor.

⇒ Any "functional complement" definition must be **per class**. A project-wide Level-3 metric built
on precursors would be meaningless for 21 of our 22 classes.
⇒ Practical consequence: the Level-3 framing needs re-scoping to something class-general
(e.g. "antiSMASH calls it a cluster of the right class", which already works) or explicitly
scoped to RiPP.

---

## 2026-08-19 — Dropped `--minimal`. Detection unchanged; but generations are ONLY ever "RiPP-like".

**Q: what did `--minimal` cost us?** A/B on **identical** sequences, both modes, n=10:
`--minimal` detected 8/10, full mode 8/10, **100% agreement on `is_bgc`.**
⇒ **No prior number is retracted.** `is_bgc` and `correct_class` come from core detection, which
runs in minimal mode. What we lost was the *analysis modules* — precursor prediction, domain
analysis, CompaRiPPson — never the detection verdict.

**Full-mode results, 2,000 nt window, identical scoring path:**

| arm | n | detected | rate | precursor motif |
|---|---|---|---|---|
| real cores | 40 | 33 | 0.825 | 0 |
[INCORRECT] - | SF Pfam-positive (best arm) | 16 | 12 | 0.750 | 0 |
[CORRECTION - 2026-08-19]: duplicate copies. **6 unique sequences, 4 produced antiSMASH output,
3 detected = 0.750.** The rate holds; the n behind it is **3 detections, not 12.**
[INCORRECT] - | W-2 Pfam-positive | 16 | 16 | 1.000 | 0 |
[CORRECTION - 2026-08-19]: duplicate copies. **9 unique sequences, 4 ran, 4 detected = 1.000.**
The n behind it is **4 detections, not 16.** `real` (40) and `base` (40) were NOT duplicated.
| SF Pfam-negative | 38 | 4 | 0.105 | 0 |
| base 1B | 39 | 0 | **0.000** | 0 |

**★ THE FINDING — product specificity, which `--minimal` output never surfaced in our pipeline:**

| arm | antiSMASH products called |
|---|---|
| **real cores** | lassopeptide 7 · RiPP-like 4 · lanthipeptide-class-iv 4 · lanthipeptide-class-i 3 · lanthipeptide-class-iii 3 · redox-cofactor 2 |
[INCORRECT] - | **SF Pfam+ (best arm)** | **RiPP-like 12 — nothing else** |
[CORRECTION - 2026-08-19]: on unique sequences — **`RiPP-like` 3, nothing else.**
[INCORRECT] - | **W-2 Pfam+** | **RiPP-like 16 — nothing else** |
[CORRECTION - 2026-08-19]: on unique sequences — **`RiPP-like` 4, nothing else.** The direction
survives (0/7 unique generated detections carry a subclass vs 30/33 real, Fisher p≈1e-5) but the
honest n is **7, not 28**.

⇒ **Every generated detection is antiSMASH's generic catch-all.** Real clusters are assigned a
*specific chemistry* — lassopeptide, lanthipeptide class I/III/IV. **Ours never are.** The model
trips the generic RiPP rules and never a subclass rule.

⇒ This is a far more precise statement of the limitation than any domain-count metric, and it is
**directly reportable**: *"generates sequence antiSMASH calls a RiPP-like cluster, but not yet any
specific RiPP subclass."* It also explains the precursor result — precursor prediction only exists
for the specific subclasses, and we never produce one.

⇒ **`n_class_domains`, `bio_span_frac` and the precursor panels were all indirect proxies for this.**
**Product specificity is the honest, field-standard readout of how far short we fall.**

---

## 2026-08-19 — ⚠️ CORRECTION: the "1.43× dilution" was partly an artefact of my own measure

User challenged whether WIDE actually failed or whether "BIO fraction" merely reflects counting one
tag. **The challenge is partly correct.**

`BIO-only fraction` counts bases inside genes tagged **`biosynthetic`** and divides by the span. As
the span widens the numerator is **held fixed** while the denominator grows, so it falls
**mechanically**. Recomputed with two better denominators (n=27,171 deduped regions):

| span definition | median nt | **BIO-only** (my original) | **DEFINING-genes** | any-CDS |
|---|---|---|---|---|
| STRICT | 2,191 | **0.869** | **0.869** | 0.980 |
| bio + transport | 6,578 | 0.551 | **0.687** | 0.950 |
| **WIDE (bio + additional)** | 12,208 | **0.310** | **0.576** | 0.919 |
| everything except `none` | 16,893 | 0.208 | 0.595 | 0.906 |

- **DEFINING-genes** = share of the span inside a gene of the kind that *defines* that span.
- **any-CDS** = share inside any annotated CDS — tests for genuine intergenic filler.

⇒ **`any-CDS` stays at 0.92 for WIDE, so the wider span is NOT full of empty space.** And
`DEFINING-genes` falls only 0.869 → 0.576, not 0.869 → 0.310. **Real dilution exists — roughly a
third of a WIDE span is CDS that is neither biosynthetic nor biosynthetic-additional — but it is
about half the magnitude I reported.**

[INCORRECT] - ⇒ Per 1,000 training nt: **STRICT 683 nt biosynthetic, WIDE 477 nt — 1.43× less signal per token.**
[CORRECTION - 2026-08-19]: That ratio was inflated by holding the numerator to the single
`biosynthetic` tag while widening the denominator. On a self-consistent denominator the gap is
0.869 → 0.576. **Dilution is real but milder, and is therefore a weaker explanation than stated.**

**WHAT DOES NOT CHANGE: WIDE'S FAILURE IS AN EXPERIMENTAL RESULT, NOT AN INFERENCE.**
W-1 vs W-2 used **the same 3,723 clusters, the same seeds, the same n** — only span width differed —
and WIDE was worse at **Holm p = 4.1e-04** (2.2 kb) and **3.2e-05** (8 kb), with the training-set
size drop separately shown to cost nothing (p=0.79). That stands independent of any explanation.
⇒ **The finding survives; my mechanism for it is now uncertain.** Dilution, span length, or
something else — we do not know which, and the docs should stop asserting dilution as the cause.

---


## 2026-08-19 — ⛔ DATA-INTEGRITY BUG: the fan-out wrote FOUR IDENTICAL COPIES. Effective n was 47.

**Found while assembling the [P5-REPORT] table.** Two arms reported `n_distinct_clusters` = **47 of
188** — exactly 188/4. That is not a biological number.

**Root cause.** `evo2/scripts/seed_generate.py` has **no shard/offset argument**. It draws seeds with
`rng = random.Random(args.seed)` → `rng.shuffle(sel)`, then sets `torch.manual_seed(args.seed)` per
generation. Four workers launched with the same `--seed` therefore select the **same** seed records
and sample the **same** continuations — byte-identical output. The fan-out contract in `CLAUDE.md`
("N *sequential* processes on **disjoint units**") was satisfied in form (4 shard files, 4 tmux
sessions, 4 sentinels) and violated in substance: nothing made the units disjoint.

**Blast radius — measured over every generation set on disk, not assumed:**

| generation set | records | unique | verdict |
|---|---|---|---|
| `SF_seeded8k.jsonl` | 188 | **47** | 4x identical shards |
| `W2_seeded.jsonl` | 188 | **47** | 4x identical shards |
| `W2_seeded8k.jsonl` | 188 | **47** | 4x identical shards |
| `W1_seeded8k.jsonl` | 188 | **47** | 4x identical shards |
| `W1_seeded.jsonl` | 188 | **141** | shards b and d collided; a/c distinct |
| `A0_8k`, `A0_noseed`, `ctrl_base`, `ctrl_general`, pilots | 150/150/150/150/50 | all | ✅ CLEAN |
| `S2-1` … `S2-5` | 188 each | 186–188 | ✅ CLEAN |
| seed sweep `s1_*` | 50 each | 49–50 | ✅ CLEAN |

⇒ **PHASE 3 IS UNAFFECTED.** A0 significance (p=0.0054), S2-1 class-specificity (p=2.5e-11), the
shuffle control (p=0.66), P3-AS antiSMASH (33 unique on-class), P3-AAI and the seed sweep all rest
on clean, unduplicated sets. **The damage is confined to Phase 4/5** — the WIDE comparison, the
"best arm" headline, and the subclass table.

**What uniform duplication does and does not do.** A rate over 4 copies of 47 equals the rate over
47, so **point estimates survive almost unchanged**. **n, confidence intervals and p-values do not.**

**Recomputed on unique records** (`scripts/novelty_battery.py` now refuses to score a duplicated
set; audit script + numbers below):

| arm | Pfam, as published | Pfam, unique | antiSMASH-corrected, unique | `rp` rests on |
|---|---|---|---|---|
| W-1 WIDE 2.2k | 11/188 = 0.059 | **7/141 = 0.050** | 0.028 | 7 unique |
| W-2 STRICT 2.2k | 36/188 = 0.191 | **9/47 = 0.191** | 0.043 | 9 unique |
| W-1 WIDE 8k | 8/188 = 0.043 | **2/47 = 0.043** | 0.000 | 2 unique |
| W-2 STRICT 8k | 36/188 = 0.191 | **9/47 = 0.191** | 0.085 | 9 unique |
| **SF STRICT-full 8k** | 24/188 = 0.128 | **6/47 = 0.128** | **0.128** | 6 unique |

**★ THE CONSEQUENCE THAT MATTERS — the WIDE refutation is HALVED, not destroyed.**
Fisher exact on Pfam counts, Holm over the same family of 4:

| contrast | as published | **recomputed on unique** |
|---|---|---|
| W-1 vs W-2 @ 2.2 kb | Holm p=4.1e-04 | **Holm p=0.021 — still significant** |
| W-1 vs W-2 @ 8 kb | Holm p=3.2e-05 | **Holm p=0.15 — NOW n.s.** |
| W-2 vs SF (dataset size) | p=0.79 | p=0.57 — still n.s. |

⇒ **WIDE is still refuted, on one window instead of two, at p=0.02 instead of 4e-04.** The
direction never reverses in any cut. The `[P4-WIDE]` verdict stands; its *strength* was overstated.

**★ AND THE SUBCLASS FINDING SHRINKS BUT SURVIVES, while the real-core ceiling moves UP.**
Recounted per *detected sequence* on unique records: real cores **30/33 = 0.909** carry a specific
subclass (the earlier "~70%" counted product *strings*, and regions carry several). Generated:
**0/3 (SF) and 0/4 (W-2)**. Fisher 0/7 vs 30/33 → **p≈1e-5**. The gap is real and in fact wider than
reported, but it now rests on **7 unique generated detections, not 28**.

**`JOINT_PASS` = 0 on those arms was an ARTEFACT, not a result.** The gate requires intra-set
distinctness, which a 4x-duplicated set cannot pass by construction. It is UNMEASURED there.

**★ THE PROCESS FAILURE, which is the more useful lesson.** The scorer **already computed and
printed** `frac_distinct` 0.25 and `frac_with_a_near_duplicate` 1.00, and already warned that
on-class records were failing a gate. The number was carried into a results table as
`JOINT_PASS` 0 and written up as *"novelty clean everywhere"*. **The instrument worked; the reading
did not.** An exact-duplicate rate is a *pipeline* assertion and must fail the run, not appear as a
diversity statistic competing for attention with a p-value.

**Fixes applied.**
1. `scripts/novelty_battery.py` — new `exact_duplicate_audit()`; under `BGC_EVAL_STRICT` (default
   on) it **raises before scoring** and refuses to emit a file carrying a false n. `effective_n` is
   now stamped into the `scoring` block and an `integrity` block of every scored file.
2. `seed_generate.py` needs a shard argument before any future fan-out — recorded in `bugs.md`.
   Until then, **vary `--seed` per shard**; it drives both seed selection and sampling.
3. Affected generation sets flagged in `data.md` with their effective n.

**Provenance:** audit over all 68 generation sets under `phase3_RIPP*/` and `phase5_detect/`;
recomputation aligned the antiSMASH TSV to `as_*.jsonl` by row order with a per-row length assertion.

---

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->
