# Phase 7 — TERPENE. Pre-registration.

**Opened 2026-08-19.** Written BEFORE any Phase-7 arm generates (Standing Constraint 4). Endpoints
in this file do not change mid-phase; deviations are recorded as dated amendments below, never by
editing the original text.

Companion: `docs/phase6_PKS_preregistration.md`. The two phases run the same machinery on different
classes and are **not** pooled — every rate is class-specific and read against its own ceiling and
floor.

---

## 1. Why TERPENE, and what this phase settles

TERPENE is the **method-transfer** phase, and it is deliberately the easier of the two. It asks
whether the Phase-3 machinery — per-class LoRA, seeded generation, the novelty battery, antiSMASH
confirmation — reproduces on a class chosen for tractability rather than for structural interest.

It is explicitly **not** the class that tests cluster structure. Measured 2026-08-19
([P5-CLASSPROBE]), TERPENE's real cores look like RIPP's, only shorter: `n_class_domains >= 2` in
**0.220** of real cores against PKS's 0.740, 1.40 markers per positive against PKS's 2.58. That
question belongs to Phase 6.

What TERPENE has that neither other class does:

| | RIPP | PKS | **TERPENE** |
|---|---|---|---|
| train records | 8,129 | 5,195 | **11,297** |
| fits the 1B (<= 7,992 nt) | 89.4% | 75.2% | **94.0%** |
| Pfam `on_class` ceiling (real cores) | 0.680 | 0.860 | **0.960** |
| **antiSMASH detection, real cores** | 0.825 | 0.980 | **1.000** |

⇒ **The largest dataset, the least context loss, and the highest ceiling on both instruments.** If
the method fails here it fails cleanly, with no "the substrate starved it" escape.

## 2. ⚠️ THE INSTRUMENT LIMIT THAT IS SPECIFIC TO THIS CLASS

**antiSMASH refuses any input record under 1,000 nt** — `ERROR: all input records smaller than
minimum length (1000)`. TERPENE's median strict core is **960 nt**, so at the default setting
**23/50 real test cores could not be scored at all** and the tool reported nothing rather than
failing loudly. Read naively that is a detection rate on a biased 54% subsample of the longest
cores.

**Resolution, validated before use:** `--minlength 200`. A/B on the 27 records that ran under both
settings — **27/27 agreement on `is_bgc` and 27/27 on the product call**. It rescues 23 records and
changes no verdict. Full 50 then detect **50/50 = 1.000**.

⇒ **`--minlength 200` is part of the frozen Phase-7 scoring config** (§3). It is a scoring-config
axis exactly like `--minimal` and the window: **a TERPENE number scored at minlength 200 may never
be compared against a Phase-3 or Phase-6 number scored at the default.** `scripts/antismash_full.py`
takes `--minlength` and warns whenever any sequence fails to run.

## 3. Primary endpoint — frozen

**`best_bio_bits` > 0 @ `OBLIGATE_DOMAINS[TERPENE]` (7 accessions), scoring window 2,000 nt**,
substrate `evo2_1b_base`, on the generated continuation only. antiSMASH confirmation in full mode
at **`--minlength 200`**, output dirs retained.

⚠️ **The 2,000 nt window is NOT inherited from RIPP — it is re-derived and happens to coincide.**
TERPENE held-out cores have a median of 975–1,041 nt, so 2,000 nt covers a whole core plus margin.
No Phase-7 number may be compared against a Phase-3 number: the class, marker set and antiSMASH
config all differ.

Declared secondaries, reported always, decisive never: the full PHASE-3 REPORTING SET (`terms.md`,
class-agnostic) plus the product breakdown of §5.

## 4. Ceiling and floor — measured before the arms, quoted with every rate

- **Ceiling:** 50 real held-out TERPENE cores restricted to <= 7,992 nt
  (`phase5_classprobe/real_TERPENE_fit50_w2000.json`). Because 94.0% of the split fits, this
  population is nearly the whole class — unlike PKS, the context filter is close to a no-op here.
- **Floor:** unadapted `evo2_1b_base` and the general all-class adapter, same seeds, same scoring.
- **antiSMASH ceiling:** **50/50 = 1.000**, full mode, `--minlength 200`.

## 5. Class-specific evaluation that RIPP did not need

1. **`subclass_specificity` HAS NO ANALOGUE HERE, and must not be forced into one.** The full
   product vocabulary over 50 real cores is `terpene-precursor` (28), `terpene` (22), plus three
   off-class singletons. **There is no generic catch-all** — no `terpene-like` corresponding to
   `RiPP-like` or `PKS-like`. The metric that is the honest measure of the remaining gap for RIPP
   simply does not exist for TERPENE, and inventing a substitute would repeat the mistake that
   produced three failed proxies for RIPP.
2. **The candidate substitute is `terpene` (cyclase rule) vs `terpene-precursor` only — and it is
   LENGTH-CONFOUNDED.** Median 2,009 nt for cyclase-rule records against 928 nt for
   precursor-only. Any use of this split as an endpoint must first show the effect survives
   length matching. **Registered here as a hypothesis to test, NOT as a secondary endpoint.**
3. **Memorisation is the headline risk for this class, not capability.** ECTOINE and MELANIN were
   disqualified because 85% / 95% of their held-out clusters are near-duplicates of training
   clusters, and the recorded reason is that **"the short classes are short *because* they are
   conserved."** TERPENE passed the diversity gate (cross-split near-dups removed at a rate
   comparable to RIPP and PKS) but it is the shortest surviving class, so it sits closest to that
   failure. Both novelty gates are therefore read as **primary safety criteria** here, not as
   routine reporting.
4. **The novelty gates need a length-matched null before they are trusted.** `containment` is a
   k=21 k-mer statistic and its null distribution depends on sequence length; the 0.80 WARN / 0.95
   FAIL thresholds were set on RIPP cores roughly twice as long. **Derive the containment null on
   shuffled TERPENE-length sequences before quoting a TERPENE containment as clean.** This is the
   one piece of new machinery Phase 7 requires.
5. **Generation length is a live variable, not a default.** At a ~975 nt median core, generating
   8,000 nt means ~87% of the output lies beyond anything a real core contains. Report the scored
   2,000 nt window as primary and state the generated length with it.

## 6. Arms — the strict arm first

**[P7-A0] STRICT-span TERPENE adapter, de novo.** Same recipe as Phase-3 A0 and Phase-6 A0:
`train_class_adapter.sh` with `CLASS=TERPENE`, `DATA=splits_class/TERPENE`, L=8192, LoRA, bs=1
ga=16, 3 epochs, whole-record. Run dir `phase7_TERPENE/`. Queued behind [P6-A0] on the shared H100.

Nothing else is pre-registered yet. The seeding ladder and L\* are registered only after [P7-A0]
reads out — RIPP's L\*=8 came from RIPP start-codon entropy and does not transfer (Standing
Constraint 9).

## 7. Novelty gates — unchanged in form, elevated in status

`containment` < 0.80 AND `protein_aai` < 0.95, reported per arm, never co-reported with the
capability metric, and read against the length-matched null of §5.4. Plus intra-set distinctness:
`scripts/novelty_battery.py` now **refuses to score a set containing exact duplicates** (`bugs.md`,
fan-out shard collision, 2026-08-19).

⚠️ **Any fan-out MUST vary `--seed` per shard.** `seed_generate.py` still has no `--shard i --of N`.

## 8. Kill criterion

If [P7-A0] reaches a de novo rate indistinguishable from the base-model floor at powered n, with
the adapter verified to have trained, then the Phase-3 result does not transfer to the easiest
available class and the method — not the substrate — is the limiting factor. That is a stronger
negative than a PKS failure would be, and it is reported as such.

Conversely, an arm that clears the capability endpoint **while failing either novelty gate against
the length-matched null** is reported as a memorisation result, not a capability result.

## 9. What would make this exploratory rather than confirmatory

Changing the 2,000 nt window or the `--minlength 200` setting; changing the primary endpoint;
pooling TERPENE and PKS rates; adopting the cyclase-vs-precursor split as an endpoint without the
length-matched check; adding arms after seeing results; quoting a Phase-7 rate against a Phase-3 or
Phase-6 number. If any occurs, the result is labelled exploratory in every document that reports it.
