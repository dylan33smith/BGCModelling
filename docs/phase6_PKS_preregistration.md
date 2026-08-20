# Phase 6 — PKS. Pre-registration.

**Opened 2026-08-19.** Written BEFORE any Phase-6 arm generates (Standing Constraint 4). Endpoints
in this file do not change mid-phase; deviations are recorded as dated amendments below, never by
editing the original text.

Companion: `docs/phase7_TERPENE_preregistration.md`. The two phases run the same machinery on
different classes and are **not** pooled — every rate is class-specific and read against its own
ceiling and floor.

---

## 1. Why PKS, and what this phase settles

Everything Phase 3 established rests on **one class**. A0's significance, the ~6× seeding lift,
class-specificity, WIDE's refutation and the subclass gap are all RIPP. This phase asks whether any
of it is a property of **the method** rather than of RiPP biology.

PKS was chosen over TERPENE on a measurement, not a preference (`memory.md` 2026-08-19,
[P5-CLASSPROBE]). Real held-out cores, n=50/class, 4,000 nt window:

| metric | RIPP | **PKS** | TERPENE |
|---|---|---|---|
| `n_class_domains` >= 2 | 0.200 | **0.740** | 0.220 |
| `n_class_domains` \| on-class | 1.47 | **2.58** | 1.40 |
| `n_bio_domains` \| on-class | 1.76 | **4.05** | 1.52 |
| `n_orfs` | 2.34 | **1.36** | 1.80 |
| `max_orf_aa` | 456 | **744** | 341 |

⇒ `n_class_domains >= 2` was demoted to a diagnostic for RIPP **because the real-core ceiling was
only 0.20** — a metric whose ceiling is 0.20 cannot grade a generator. **In PKS it is 0.740.** The
endpoint that read exactly 0/188 in every RIPP arm becomes measurable against a real reference.
PKS reaches it in **fewer ORFs with a longer max ORF** — the megasynthase signature, where modules
are *intra-genic*.

## 2. ⚠️ THE STRUCTURAL FACT THAT SHAPES EVERY DESIGN CHOICE — PKS IS BIMODAL

Full-mode antiSMASH over the same 50 real test cores (`phase5_classprobe/as_real_PKS.tsv`):

| product | n | **median nt** | Pfam `on_class` |
|---|---|---|---|
| **T3PKS** | 25 (50%) | **1,083** | 0.960 |
| **T1PKS** | 20 (40%) | **7,665** | 0.750 |
| other / none | 5 (10%) | 14,162 | — |

**T3PKS is a single ~350-aa chalcone-synthase-like enzyme. T1PKS is a giant modular megasynthase.**
They are not degrees of the same thing; they are different architectures that happen to share a
class label. The `n_class_domains >= 2` result above is carried by the T1PKS half.

**And the 1B's context limit selects against exactly that half.** `train_class_adapter.sh` keeps
records <= 7,992 nt (L=8192 minus prefix), which drops the long tail:

| population | n | T3PKS | T1PKS | other |
|---|---|---|---|---|
| all real test cores | 50 | 50% | **40%** | 10% |
| **fits the 1B (<= 7,992 nt)** | 39 | **64%** | **31%** | 5% |

Train split: **3,906/5,195 = 75.2% fit**, and the median of the *fitting* subset is **1,170 nt**,
not the 2,103 nt median of the whole split.

⇒ **The Phase-6 adapter is trained predominantly on short T3PKS-type cores.** Two consequences,
both binding:
1. **The ceiling is re-derived on the fitting population** (§4), never on all real cores. Quoting
   the 0.740 all-cores figure against a model trained on the fitting subset would overstate the gap.
2. **Every Phase-6 table reports the product breakdown, not just a detection rate** (§5). A model
   that only ever emits T3PKS-like single genes would post a high detection rate while reproducing
   RIPP's "one gene is not a cluster" limitation in a new costume. Detection alone cannot see that.

## 2.1 AMENDMENT 2026-08-19 (same day) — ⛔ §1's `n_class_domains >= 2` ARGUMENT IS LARGELY AN ARTIFACT OF THE MARKER SET

§1 above stands unedited. It argued that PKS is worth this phase because
`n_class_domains >= 2` reaches **0.740–0.840** in real PKS cores against RIPP's 0.200, so the
endpoint that read 0/188 in every RIPP arm "becomes measurable against a real reference".
**Measured properly, that argument mostly dissolves.**

**`OBLIGATE_DOMAINS[PKS]` contains several Pfam models that cover ONE catalytic domain:**

| catalytic unit | Pfam models in the marker set |
|---|---|
| **ketosynthase (KS)** | `PF00109` ketoacyl-synt · `PF02801` Ketoacyl-synt_C · `PF16197` KAsynt_C_assoc |
| **chalcone synthase (CHS, = T3PKS)** | `PF00195` Chal_sti_synt_N · `PF08392` FAE1_CUT1_RppA (Chal_sti_synt_C) |
| acyltransferase (AT) | `PF00698` |
| dehydratase (DH) | `PF14765` |

⇒ **A single T3PKS gene trips `PF00195` + `PF08392` and scores `n_class_domains` = 2 on its own.**
That pair is the most common co-occurrence in real cores (14/30). The KS trio does the same for
T1PKS.

**Recomputed on the 50 real cores that fit the 1B, window 4,000:**

| definition | count | rate |
|---|---|---|
| `n_class_domains >= 2` — distinct Pfam **accessions**, as reported in §1 | 40/50 | **0.800** |
| >= 2 distinct **catalytic units** (N/C models collapsed) | 15/50 | **0.300** |
| >= 2 distinct **ORFs** carrying a marker — a real multi-**gene** core | **3/50** | **0.060** |

**28 of 50 records are the single unit-set `('CHS',)`** — one chalcone synthase gene, counted as two
"class domains".

⇒ **PKS is NOT more multi-gene than RIPP.** At 0.060 it is level with RIPP, not 4x it. The strict
core is one gene in every class, because that is what strict-core trimming produces. **The class
pivot does not solve the cluster-structure problem; only a span-width change could — which is the
parked `bio + transport` line, not this phase.**

**What survives, and is the corrected rationale for Phase 6:**
1. **Intra-genic modular content is real and is ~2x RIPP's** — 0.300 of PKS cores carry >= 2
   distinct *catalytic units*, and positives average **4.33** biosynthetic domains against RIPP's
   1.91. A megasynthase genuinely is a multi-domain modular enzyme; that content exists in the
   training span even though it sits inside one gene.
2. **Domain ORDERING becomes the right structural endpoint, and it is the one PKS uniquely supports.**
   `MODULE_PATTERNS["PKS"]` = KS-AT-ACP. Ordering is *designed* for intra-genic modularity, which is
   exactly the kind of structure PKS actually has. §5.3 anticipated this; it is now promoted from
   "available" to **the structural endpoint of the phase**.
3. **Architectural bimodality (§2) is untouched** — it is measured from antiSMASH products, not from
   the marker set.
4. Method transfer — does the Phase-3 pipeline reproduce at all on another class — never depended on
   this argument.

**Binding consequences for every Phase-6 table:**
- ⛔ **`n_class_domains` (Pfam-accession count) MUST NOT be quoted as a cluster-structure metric for
  PKS.** Report it only beside the collapsed count and `n_bio_orfs`.
- **Report all three**: distinct accessions, distinct catalytic units, and distinct marker-bearing
  ORFs. The three answer different questions and the first one flatters.
- The ceilings for the latter two are **0.300** and **0.060** — quote those, not 0.840.

**Provenance:** `phase5_classprobe/PKS_collapsed_markers.json`, per-record Pfam scan
(`check_class_markers`, full Pfam-A) over `real_PKS_fit50.jsonl`, window 4,000, n=50.

## 2.2 AMENDMENT 2026-08-20 — ⚠️ WHAT THE PHASE-6 ADAPTER IS ACTUALLY TRAINED ON: ~60% T3PKS

§2 predicted the context filter would skew the training set toward T3PKS. **Measured directly on
`phase6_PKS/train.whole.jsonl` — the 3,906 records the trainer kept — sample n=150, per-record Pfam
scan, classified by catalytic unit:**

| what the record carries | n | share |
|---|---|---|
| **CHS** — chalcone synthase (**T3PKS**, a single ~350-aa gene) | 89 | **0.593** |
| **KS** — ketosynthase (**T1PKS-type modular**) | 47 | **0.313** |
| both | 0 | 0.000 |
| neither (no PKS marker at all) | 14 | 0.093 |

Median record length in the sample: **1,167 nt**. Of the 47 KS records, **37 carry KS *plus* AT**
(19 of those also DH), i.e. genuine multi-module content.

⇒ **STATE THIS WITH EVERY PHASE-6 RESULT. The adapter is T3PKS-dominated.** It is *not* "essentially
a T3PKS model" — nearly a third of its data is modular ketosynthase — but any unqualified claim about
"PKS" from this run is disproportionately a claim about **type III PKS**, which is a single-gene
aromatic-polyketide enzyme, not the modular assembly line the word "PKS" usually evokes to a reader.

**The honest phrasing for any write-up:**
> *"a PKS-class adapter trained predominantly (~60%) on type-III (chalcone-synthase) cores, with
> ~31% type-I modular ketosynthase cores; records above 7,992 nt were dropped to fit the 1B's
> context, which removes the longest modular assembly lines."*

⛔ **Not** *"a model that generates polyketide synthase gene clusters."*

⇒ This is also the strongest argument for the **T1PKS-only arm** (§1 discussion): it is the only way
to make a claim about modular PKS from this substrate, and at ~31% of 3,906 it would have roughly
1,200 records — small but not unworkable. The counter-argument stands: T1PKS median 7,665 nt sits at
the 7,992 budget, so that arm is truncation-biased by construction and must be declared a lower
bound.

**Provenance:** `scripts` scratch classification over `phase6_PKS/train.whole.jsonl`, n=150 sampled
`random.Random(0)`, `check_class_markers` against full Pfam-A, catalytic units per §2.1.

## 3. Primary endpoint — frozen

**`best_bio_bits` > 0 @ `OBLIGATE_DOMAINS[PKS]` (8 accessions), scoring window 4,000 nt**, substrate
`evo2_1b_base`, on the generated continuation only.

⚠️ **The window is 4,000 nt, NOT the RIPP 2,000.** PKS held-out cores have a median of 3,707 nt, so
a 2,000-nt window truncates most of a real core. This is a per-class regime choice made before any
arm ran (Standing Constraint 9), and it means **no Phase-6 number may be compared against any
Phase-3 number.** Cross-class comparison is of *shape* — does the same intervention move the same
direction — never of magnitude.

Declared secondaries, reported always, decisive never: `n_class_domains >= 2` (§5),
`subclass_specificity` (§5), and the full PHASE-3 REPORTING SET (`terms.md`) which is class-agnostic.

## 4. Ceiling and floor — measured before the arms, quoted with every rate

- **Ceiling:** 50 real held-out PKS cores **restricted to <= 7,992 nt**, the population the adapter
  is trained on (`phase5_classprobe/real_PKS_fit50_w4000.json`). The all-cores ceiling is retained
  as a second reference so the context cost stays visible.
- **Floor:** unadapted `evo2_1b_base` and the general all-class adapter, same seeds, same scoring.
- **antiSMASH ceiling:** 49/50 = **0.980** detection on real cores, full mode, default minlength.

## 5. Class-specific evaluation that RIPP did not need

1. **Product-type stratification is mandatory.** Report T1PKS / T3PKS / `PKS-like` / other counts
   for every arm. `PKS-like` is antiSMASH's generic catch-all for this class — the analogue of
   `RiPP-like` — and it is **rare in real cores (3/49 ≈ 6%)**, so `subclass_specificity` has a
   ceiling of ~0.94 here, higher than RIPP's 0.909.
2. **`bio_span_frac` is uninformative for PKS and must not be used as a gate.** Real cores read
   **0.982** — it saturates, because a megasynthase core is one long biosynthetic ORF. It stays in
   the reporting set as context, and any threshold derived on RIPP is void here.
3. **Domain ORDERING is available for the first time.** `MODULE_PATTERNS["PKS"]` =
   `PF00109` (KS) - `PF00698` (AT) - `PF00550` (ACP). There is **no RIPP entry, correctly** — RiPP
   gene order is not collinear. PKS modules are, so "are the domains in the right order" is a real,
   already-implemented question that the RIPP track could never ask.
4. **Long-ORF integrity.** Real PKS `max_orf_aa` is 744 against RIPP's 456, and T1PKS needs a single
   long correct reading frame over thousands of nt. A truncated or frameshifted megasynthase is a
   PKS-specific failure mode with no RIPP analogue; report `max_orf_aa` against the real-core
   reference and the complete-gene fraction alongside it.
5. **Length-stratified reporting.** Because the context filter is confounded with product type
   (§2), any arm-level difference must be checked against a length-matched comparison before it is
   attributed to the intervention.

## 6. Arms — the strict arm first

**[P6-A0] STRICT-span PKS adapter, de novo.** Same recipe as Phase-3 A0, no objective change, no
span widening: `train_class_adapter.sh` with `CLASS=PKS`, `DATA=splits_class/PKS`, L=8192, LoRA,
bs=1 ga=16, 3 epochs, whole-record (no chunking). Run dir `phase6_PKS/`. Launched 2026-08-19.

Nothing else is pre-registered yet. The seeding ladder, L\* and any span-width arm are registered
only after [P6-A0] reads out, because **every regime-specific value must be re-derived, not
inherited** (Standing Constraint 9): RIPP's L\*=8 came from RIPP start-codon entropy, and the
two-pass Pfam -> antiSMASH calibration is stamped RIPP-specific in `terms.md`.

## 6.1 AMENDMENT 2026-08-20 — n AND ARMS FOR [P6-A0], REGISTERED BEFORE GENERATION

**Three arms, n=200 each, on the same 200 prompts** from `splits_class/PKS/eval_prompts.jsonl`
(all of them — that file holds exactly 200), de novo (prefix only, no sequence seed):

| arm | model | role |
|---|---|---|
| **P6-A0** | `phase6_PKS/adapter_run/final_adapter` | the treatment |
| **P6-A0-C1** | unadapted `evo2_1b_base` | floor |
| **P6-A0-C2** | general all-class adapter (`phase2_long/baseline_long/final_adapter`) | floor — isolates "a class-specific adapter" from "any adapter" |

**Power.** Pooled control n=400 against treatment n=200. Phase 3 measured the binding constraint
directly ([P3-B3]): against a control that stays at exactly 0, Fisher's exact reaches **p=0.004 at
5 treatment hits** in this configuration, and **p=0.012 at 4**. Generating more *treatment* does not
help — the control arm is what moves the p-value. All 200 prompts are used by every arm, so the
arms are prompt-matched, not merely size-matched.

**Generation length 8,000 nt, scored at the registered 4,000 nt window.** Length may differ from
other phases safely: the scored span is a fixed prefix and an autoregressive model writes the same
first 4,000 tokens regardless of the total requested.

**Substrate:** `evo2_1b_base`, now enforced in code — `generate_bgc.py` refuses to run without an
explicit substrate rather than defaulting to the 7B (`bugs.md` [P3-B7]).

**Manipulation check, read BEFORE the endpoint:** the adapter arm's output must differ from
P6-A0-C1 on some measured axis. An adapter that changes nothing is not a treatment and its null is
uninformative, not negative.

**Novelty gates:** `containment` AND `protein_aai`, plus intra-set distinctness — the scorer now
refuses to score a set containing exact duplicates, and any fan-out varies `--seed` per shard
(`bugs.md`, shard collision).

**Kill criterion:** treatment indistinguishable from the pooled control at this n, with the
manipulation check passing, closes [P6-A0] as a powered negative for this class.

## 7. Novelty gates — unchanged, both hard

`containment` < 0.80 AND `protein_aai` < 0.95, reported per arm, never co-reported with the
capability metric. Plus intra-set distinctness: `scripts/novelty_battery.py` now **refuses to score
a set containing exact duplicates** (`bugs.md`, fan-out shard collision, 2026-08-19).

⚠️ **Any fan-out MUST vary `--seed` per shard.** `seed_generate.py` still has no `--shard i --of N`,
and four workers on one seed produce four byte-identical copies.

## 8. Kill criterion

If [P6-A0] reaches a de novo rate indistinguishable from the base-model floor at n powered to
detect the RIPP-equivalent effect, **and** the manipulation check confirms the adapter trained
(`loss_ce` fell, adapter loads, output differs from base), then the honest reading is that the
Phase-3 result does not transfer to PKS at this substrate size, and the phase reports a negative
rather than escalating to a larger model.

## 9. What would make this exploratory rather than confirmatory

Changing the 4,000 nt window; changing the primary endpoint; pooling PKS and TERPENE rates; pooling
across product types without reporting the breakdown; adding arms after seeing results; quoting a
Phase-6 rate against a Phase-3 number. If any occurs, the result is labelled exploratory in every
document that reports it.
