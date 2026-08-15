# memory.md — the linear ledger

**DO NOT read this file on startup.** It is the permanent laboratory notebook and it only grows.
`grep` it when you need historical context.

```bash
grep -n "2026-08" docs/project_memory/memory.md      # by date
grep -n "^## \[" docs/project_memory/memory.md       # list all entries
grep -ni "best_bio_bits" docs/project_memory/memory.md
grep -n "INCORRECT\|CORRECTION" docs/project_memory/memory.md   # what we got wrong
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
| `docs/steering_program.md`, `_technical.md` | 738 | the closed steering programme, in full |
| `docs/conditioning_next_steps.md` | 398 | ranked conditioning list — **ranking SUPERSEDED**, citations accurate |
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

<!-- APPEND NEW ENTRIES BELOW THIS LINE -->
