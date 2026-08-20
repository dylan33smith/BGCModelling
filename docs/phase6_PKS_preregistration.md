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
