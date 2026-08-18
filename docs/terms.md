# terms.md — the glossary

**Purpose.** One name, one meaning, for the life of the project. Search this before defining a
metric, writing a pipeline, or labelling a table column. Use the identifier exactly as written.

**Entry schema.** Every metric entry carries all six fields. The two that do the real work are
`CHANGES MEANING WITH` and `Status` — a metric whose scoring config is unstated is not a result,
and a demoted metric must not quietly reappear.

```
### <identifier>  [tag] [tag]
Is:                   one sentence, unambiguous.
Computed by:          file:symbol — and the call that produces it.
CHANGES MEANING WITH: config that silently alters the number under the same name.
Valid vs:             what it may be compared against. Everything else is invalid.
Status:               PRIMARY | SECONDARY | DIAGNOSTIC | DEMOTED | RETIRED
Aliases:              names seen in old docs. Do not use them.
```

**Tags:** `[evaluation]` `[model]` `[method]` `[dataset]` `[training]` `[gate]` `[diagnostic]`

---

## ⚠️ The two drifts this file exists to stop

1. **Code key ≠ doc name.** `ladder_audit.py` writes the dict keys `bio`, `any`, `frac`. The docs
   call the same numbers `best_bio_bits`, `best_any_bits`, `biosynthetic_fraction`. Both names are
   in live use. **The doc name is canonical; the code key is an implementation detail.**
2. **Same name, different scoring set.** `best_bio_bits` scored against the global biosynthetic
   Pfam set and against `OBLIGATE_DOMAINS[RIPP]` are *different numbers under one name*. This
   inverted the Phase-3 A0 conclusion on 2026-08-14. Always state the set.

---

## A

### antiSMASH detection  [evaluation] [gate]
- **Is:** Whether antiSMASH 8.0.4 annotates any biosynthetic cluster in the sequence.
- **Computed by:** `src/bgc_pipeline/evaluation.py:check_antismash` → `detected`; feeds `is_bgc`.
- **CHANGES MEANING WITH:** nothing internal, but it is *skippable*, and when skipped `is_bgc`
  silently falls back to the `class_markers` proxy. Check `_verdict_source`.
- **Valid vs:** other antiSMASH-derived rates only.
- **Status:** PRIMARY — the gold-standard `is_bgc` / `correct_class` gate. Measured FPR on real
  non-BGC DNA: **0.000**. **First Phase-3 numbers 2026-08-18:** among Pfam-on-class generations
  `is_bgc` = `correct_class` = **0.485**; real held-out cores **0.760 / 0.740**; off-class
  generations 0.000–0.040. `is_bgc` and `correct_class` were identical on every set — when this
  model produces a cluster, it is a RIPP.
- **Aliases:** "de novo detection", "P(detect)", "antiSMASH detect", "detection rate".

### `any` (code key)  →  see **`best_any_bits`**

---

## B

### `best_any_bits`  [evaluation] [diagnostic]
- **Is:** Best HMM bitscore over the **full Pfam-A** database across all called ORFs in a record.
- **Computed by:** `evo2/scripts/ladder_audit.py:one` → dict key **`any`**, via `_hits(dig, PFAM)`
  where `PFAM = /data2/ds85/pfam/Pfam-A.hmm`.
- **CHANGES MEANING WITH:** the E-value cutoff (`E=1e-3`) and the gene caller.
- **Valid vs:** same window, same caller. Its role is the denominator of `biosynthetic_fraction`.
- **Status:** DIAGNOSTIC. "Is this protein at all?" — 100% of 6 kb generations hit something.
- **Aliases:** `any`, "any-Pfam".

### `best_bio_bits`  [evaluation] [gate]
- **Is:** Best HMM bitscore over the **biosynthetic** Pfam subset across all called ORFs.
- **Computed by:** `evo2/scripts/ladder_audit.py:one` → dict key **`bio`**, via `_hits(dig, BIO)`.
- **CHANGES MEANING WITH:** ⚠️ **THE PFAM SUBSET — THE #1 DRIFT RISK IN THIS PROJECT.**
  - **GLOBAL (default):** `BIO = /data2/ds85/pfam/biosynthetic_subset.hmm`, ~91 models, the union
    of all `OBLIGATE_DOMAINS` values. `ladder_audit.one()` takes a `cls` argument but **uses it
    only for the NRPS/PKS module rung — `bio` ignores it entirely.** Rescoring one arm under
    RIPP / NRPS / PKS / TERPENE returns 4/50 every time.
  - **CLASS-SPECIFIC:** the HMM subset to `OBLIGATE_DOMAINS[<CLASS>]`. **This is what the Phase-3
    pre-registration names.** Under it the general adapter scores 0.000, not 0.080.
  - Also changes with the scoring window (2,000 nt fixed in Phase 3) and the gene caller.
- **Valid vs:** same Pfam subset, same window, same class, same caller. Nothing else.
- **Status:** **PRIMARY** (AUROC 0.950 on the validated ladder). Heavily zero-inflated — report as
  a **rate with an exact binomial CI**, never as an arm mean.
- **Aliases:** `bio`, "bio bits", "biosynthetic bits", "best_bio".

### `bio_accs`  [evaluation]
- **Is:** The sorted set of **biosynthetic Pfam accessions actually hit** in a record.
- **Computed by:** `evo2/scripts/ladder_audit.py:one` → `sorted({acc for acc,_,_ in biohits})`.
- **CHANGES MEANING WITH:** the E-value cutoff and the gene caller — but **not** the class: it is
  always the full global set that was hit, deliberately.
- **Valid vs:** anything, since it is a set of identifiers rather than a score. **This is the field
  that makes a class-specific rate possible**: intersect it with `OBLIGATE_DOMAINS[cls]`. Added
  2026-08-17 to fix the scorer that read the class-agnostic `best_bio_bits` as an on-class rate.
- **Status:** SECONDARY — the substrate for `on_class`, not a rate itself.

### `protein_aai`  [evaluation] [gate]
- **Is:** Best amino-acid identity between any ORF of a generation and any protein of the TRAINING
  set — one value per record, in [0,1].
- ⚠️ **REPORT IT AMONG ON-CLASS RECORDS, against the real-core reference.** A pooled median is
  **not interpretable**: it is dominated by how many records have *any* hit, not by how similar the
  hits are. Measured 2026-08-18 — real held-out cores **0.641** (98.3% with a hit); on-class
  generations **0.496** (97.0%); off-class generations **0.000** (10.0%). A pooled median of 0.000
  therefore means "mostly off-class", not "nothing resembles training". This artifact produced a
  retracted claim — see `memory.md` 2026-08-18.
- **Computed by:** `scripts/novelty_battery.py:protein_novelty` → MMseqs2 `easy-search` (`-e 1e-3`,
  `-s 5.7`) of translated ORFs against `<class>/train_proteins.fa`; `fident` column.
- **CHANGES MEANING WITH:** the training protein DB (per class), the MMseqs sensitivity, and the
  ORF caller's `min_aa`. A record with **no** hit scores 0.000 — so a median of 0.000 means most
  records match nothing, not that they match poorly.
- **Valid vs:** same DB, same sensitivity. Complementary to `containment`, never a substitute:
  ⚠️ **DNA containment cannot see protein-level reconstruction.** In the seed sweep containment
  stayed ≤0.021 while `protein_aai` rose to 0.914 and the model was demonstrably rebuilding the
  seeded cluster. See `bugs.md` 2026-08-17.
- **Status:** **PRIMARY GATE** alongside `containment`. `FAIL_paraphrase` at ≥0.95. Both gates must
  be pre-registered for any seeded arm.
- **Aliases:** "AAI", "protein novelty", "T3.2".

### `seed_nt`  [method]
- **Is:** Length in nucleotides of the exemplar prefix handed to the model. Stored per record.
- **Computed by:** `evo2/scripts/seed_generate.py --seed-nt N`; the seed is `src[:N]` of a **single
  distinct source record per generation** (50 generations = 50 different seeds), recorded as
  `seed_accession` and `seed_prefix_64`.
- **CHANGES MEANING WITH:** nothing internal — but it is the dominant experimental variable in
  leg 2, and the **seed is never scored** (`scored_span: continuation_only`).
- **Valid vs:** same seed source split. Seeds for tuning come from **val**, confirmatory from test.
- **Status:** **L\* = 8 nt** (chosen 2026-08-17). Beyond ~100 nt the model reconstructs the seeded
  cluster rather than generating: at 500 nt, 12/12 on-class hits reproduced their own source
  cluster's domain, vs 0/8 at 8 nt.

### `no_boundary_orf`  [method]
- **Is:** Adversary control that truncates the seed at its **last in-frame stop codon**, so no open
  reading frame spans seed→continuation. Any class-defining domain found in the continuation must
  therefore be written de novo rather than being the tail of a handed-over gene.
- **Computed by:** `evo2/scripts/seed_generate.py --no-boundary-orf` → `_truncate_at_last_stop`.
- **CHANGES MEANING WITH:** nothing; it either ran or it did not. Record it per arm.
- **Valid vs:** an arm run **without** the flag, which is the informative pairing.
- **Status:** **MANDATORY for Phase-3 seeded arms** (decided 2026-08-17). 86% of RIPP cores begin
  at the marker gene, so a long seed hands over most of that gene; without this flag a "hit" can be
  the model finishing what it was given. Stage 1 did **not** use it.

### `bio_span_frac`  [evaluation]
- **Is:** Distance from the first to the last biosynthetic-domain-carrying ORF, as a fraction of
  sequence length. Asks "are the domains *clustered*?" — one domain is not a cluster.
- **Computed by:** `evo2/scripts/ladder_audit.py:one` → `(hi - lo) / len(seq)` over `bio_orfs`.
- **CHANGES MEANING WITH:** the Pfam subset (as `best_bio_bits`); sequence length (it is a ratio,
  so it is not comparable across generation lengths).
- **Valid vs:** same length, same subset.
- **Status:** SECONDARY, rung 3 (AUROC 0.896). Real cores **0.876** vs de novo **0.051**.

### `biosynthetic_fraction`  [evaluation] [diagnostic]
- **Is:** `best_bio_bits / best_any_bits`. "Of the protein it writes, how much is biosynthetic?"
- **Computed by:** `evo2/scripts/ladder_audit.py:one` → dict key **`frac`**.
- **CHANGES MEANING WITH:** the Pfam subset; undefined (set to 0.0) when `best_any_bits == 0`.
- **Valid vs:** same subset. **Never use as a primary endpoint** — it rests on between-group
  evidence only, which is exactly how `max_orf_aa` went wrong.
- **Status:** DIAGNOSTIC (specificity). Real cores **0.836** vs de novo **0.100**.
- **Aliases:** `frac`, "bio fraction", "biosynthetic fraction".

---

## C

### `class_markers`  [evaluation]
- **Is:** Pfam-based fast proxy for `is_bgc` / `correct_class`, used when antiSMASH is skipped.
  Passes on ≥2 biosynthetic domains (ANY-of semantics over `OBLIGATE_DOMAINS[class]`).
- **Computed by:** `src/bgc_pipeline/evaluation.py:check_class_markers`.
- **CHANGES MEANING WITH:** the ≥2 threshold; whether it counts *any* Pfam (old, wrong) or
  *biosynthetic* Pfam (current). Old: sens 1.000 / spec 0.598 / PPV 0.330. Current: sens 0.882 /
  spec 0.878 / PPV 0.589.
- **Valid vs:** ⚠️ **NOTHING antiSMASH-derived.** On 768 paired records it inflates `correct_class`
  ~2.6× (precision 0.366, recall 0.972; proxy 0.249 vs antiSMASH 0.094). Standing Constraint 7.
- **Status:** SECONDARY, quick-eval only.

### `class_probe` / `class_probe_agrees`  [evaluation] [diagnostic]
- **Is:** Linear probe on model hidden states; argmax over a per-class probability dict compared to
  the conditioned class. The **only continuous class readout**.
- **Computed by:** `evo2/scripts/probe_score_generations.py --emit-sidecar` →
  `eval_suite_driver.py --probe-scores` → `evaluation.py:check_class_probe`. Model-specific by
  design, which keeps `evaluation.py` model-agnostic.
- **CHANGES MEANING WITH:** the probe checkpoint and the layer it reads.
- **Valid vs:** **PAIRED comparisons only.** It has no negative class and cannot abstain: 0.900
  mean confidence on real *non-BGC* DNA vs 0.986 on real cores. It measures resemblance, not
  validity.
- **Status:** DIAGNOSTIC — **NEVER a gate.** Three tests pin this. TPR 0.900 on real cores at 3 kb.

### `coding_density`  [evaluation] [diagnostic]
- **Is:** Fraction of sequence covered by called ORFs.
- **Computed by:** `src/bgc_pipeline/evaluation.py:check_coding_sanity` (pyrodigal).
- **Valid vs:** same caller, same length.
- **Status:** DIAGNOSTIC. De novo **0.74–0.82** vs real **0.97** — output is not junk.

### `containment` (k=21)  [evaluation] [gate]
- **Is:** Max canonical 21-mer containment against the nearest reference/training BGC.
- **Computed by:** `src/bgc_pipeline/evaluation.py:check_kmer_novelty` → `max_containment`.
- **CHANGES MEANING WITH:** the reference set it is computed against (training split vs all BGCs).
- **Valid vs:** same reference set, same k.
- **Status:** PRIMARY GATE. `FAIL_memorized` at **≥0.95**, `WARN` at **≥0.80**. Note: the gate
  raises rather than defaulting to 0.0 when the key is absent — a gate must not default to passing.
- **Aliases:** `kmer_novelty`, "k-mer novelty", "novelty gate", "k=21 containment".

### `correct_class`  [evaluation] [gate]
- **Is:** Cluster type annotated by antiSMASH matches the conditioned class.
- **Computed by:** `evaluation.py:derive_questions` from `check_antismash.class_match`.
- **CHANGES MEANING WITH:** ⚠️ its **source** — antiSMASH or the `class_markers` proxy. Recorded in
  `_verdict_source`. Always report which.
- **Valid vs:** same source, same regime (seeded vs de novo).
- **Status:** GATE, but **not an optimisation target de novo** (Standing Constraint 8): ~0 de novo
  since project start, 0.283–0.40 seeded. Decomposes as `P(detect) × P(right | detect)`; de novo
  `P(detect) = 0.012`, seeded `0.367` with `P(right|detect) = 0.932`.

---

### `co_orient`  [evaluation]
- **Is:** Fraction of called ORFs on the majority strand — `max(n_fwd, n_rev) / n_orfs`. Real BGC
  genes are largely co-oriented.
- **Computed by:** `ladder_audit.py:one` → over `ORF.strand`.
- **CHANGES MEANING WITH:** the gene caller.
- **Valid vs:** same caller. Undefined when `n_orfs == 0`.
- **Status:** DIAGNOSTIC — a candidate rung that did **not** separate real from generated.

---

## D–F

### de novo  [method]
- **Is:** Generation from the conditioning prefix alone, **no seed sequence**.
- **Computed by:** n/a — a regime label, not a metric.
- **CHANGES MEANING WITH:** nothing. **Contrast: seeded** — a real core prefix is supplied.
- **Valid vs:** ⚠️ **never pooled with seeded results.** Different regimes, different ceilings:
  `P(detect)` 0.012 de novo vs 0.367 seeded.
- **Status:** PRIMARY regime for Phase-3 capability claims.

### `frac` (code key)  →  see **`biosynthetic_fraction`**

---

## I

### `is_bgc`  [evaluation] [gate]
- **Is:** Real coding DNA containing a biosynthetic cluster.
- **Computed by:** `evaluation.py:derive_questions`. Precedence: antiSMASH (authoritative when it
  ran) → `class_markers` proxy → coding floor only.
- **CHANGES MEANING WITH:** which branch fired. Read `_verdict_source["is_bgc"]`.
- **Status:** GATE.

---

## M

### `modules` / `in_order`  [evaluation]
- **Is:** `modules` = count of complete assembly-line modules (NRPS `C-A-T` = PF00668-PF00501-
  PF00550; PKS `KS-AT-ACP` = PF00109-PF00698-PF00550). `in_order` = 1 if they appear collinear.
- **Computed by:** `ladder_audit.py:one` → `evaluation.py:check_module_architecture`; surfaces as
  the `complete` diagnostic question.
- **CHANGES MEANING WITH:** ⚠️ **computed ONLY for `cls in (NRPS, PKS, PKS_NRPS_HYBRID)`.** For
  every other class — including RIPP — both are hard-coded **0**, which is *not* a measurement.
  Never read a 0 here as evidence about a non-assembly-line class.
- **Valid vs:** assembly-line classes only.
- **Status:** DIAGNOSTIC.

### `max_orf_aa`  [evaluation] [diagnostic]
- **Is:** Length in amino acids of the longest called ORF.
- **Computed by:** `evo2/scripts/ladder_audit.py:one`.
- **Status:** ⚠️ **DEMOTED 2026-08-12.** Adopted on between-group evidence plus a mechanistic story
  ("cannot sustain a reading frame"); failed the within-group test — inside de novo generations
  r = **0.051** at 2 kb and **−0.120** at 6 kb against domain content. AUROC 0.709.
  **Do not report as evidence of capability.** Retained as a structural diagnostic only.

---

## N

### `n_bio_domains`  [evaluation]
- **Is:** **Total count of biosynthetic domain hits** (not unique domains, not unique ORFs).
- **Computed by:** `ladder_audit.py:one` → `len(biohits)`.
- **CHANGES MEANING WITH:** the Pfam subset. Distinct from `n_bio_orfs` = number of *distinct ORFs*
  carrying ≥1 hit. These are routinely confused.
- **Status:** SECONDARY, rung 2 (AUROC 0.919).

### `n_bio_orfs`  [evaluation]
- **Is:** Number of **distinct ORFs** carrying ≥1 biosynthetic domain hit.
- **Computed by:** `ladder_audit.py:one` → `len({orf_index for hits})`.
- **CHANGES MEANING WITH:** the Pfam subset; the gene caller.
- **Valid vs:** same subset, same caller.
- **Status:** SECONDARY. ⚠️ **Not the same as `n_bio_domains`** (total hits). Routinely confused.

### `n_orfs`  [evaluation] [diagnostic]
- **Is:** Total ORFs called in the record.
- **Computed by:** `ladder_audit.py:one` → `len(find_orfs(seq))` (pyrodigal).
- **CHANGES MEANING WITH:** the caller and its `min_aa` (default 50).
- **Valid vs:** same caller, same length.
- **Status:** DIAGNOSTIC.

### `novel`  [evaluation] [gate]
- **Is:** The question-level verdict derived from `containment`.
- **Computed by:** `evaluation.py:derive_questions` → `_verdict_from_pass(check_kmer_novelty)`.
- **CHANGES MEANING WITH:** the reference set and the 0.95 / 0.80 thresholds. See `containment`.
- **Valid vs:** same reference set.
- **Status:** GATE. Novelty is a **hard constraint on every rung**, never a co-reported metric.

---

## T

### THE PHASE-3 REPORTING SET  [evaluation] [method]
- **Is:** The fixed block of numbers that **every** Phase-3 intervention MUST report, so that any
  two interventions are directly comparable. Emitted automatically by
  `scripts/novelty_battery.py`; do not hand-assemble it.
- **Computed by:** `scripts/novelty_battery.py --cls <CLASS> --window <nt>` → the `scoring`,
  `ladder`, `joint` and novelty blocks of `<arm>_w<window>_<CLASS>.json`.

  **PRIMARY (the endpoint — fixed by pre-registration §2, does not change mid-phase)**

  | metric | what it answers |
  |---|---|
  | `best_bio_bits` > 0 @ `OBLIGATE_DOMAINS[cls]` | did it make machinery of THIS class |

  **NOVELTY GATES (hard; a rate without these is uninterpretable)**

  | metric | what it answers |
  |---|---|
  | `containment` | did it copy training DNA (FAIL ≥0.95, WARN ≥0.80) |
  | protein AAI | did it copy the protein via synonymous codons |
  | intra-set distinctness | did it collapse to one output repeated |
  | `JOINT_PASS` | how many records pass **all of the above at once** |

  **CLUSTER STRUCTURE (secondary — reported always, decisive never)**

  | metric | what it answers |
  |---|---|
  | `n_class_domains` | **distinct markers of the class — >1 means a cluster, not one enzyme** |
  | `n_bio_domains` | total biosynthetic domain hits (AUROC 0.919) |
  | `n_bio_orfs` | how many distinct genes carry one |
  | `bio_span_frac` | how far apart they sit = is it a CLUSTER (AUROC 0.896) |

  **CONTEXT (secondary)**

  | metric | what it answers |
  |---|---|
  | `biosynthetic_fraction` | of the protein written, how much is biosynthetic |
  | `co_orient` | share on the majority strand (real cores median 1.000) |
  | `n_orfs` | did it write genes at all |
  | `max_orf_aa` | ⚠️ DEMOTED — structural only, never quoted as capability |

- **CHANGES MEANING WITH:** ⚠️ **the config, not just the metric list.** Two arms are comparable
  only if ALL of these match, and every one of them has already caused a real error in this project:
  1. **Pfam subset** — `OBLIGATE_DOMAINS[cls]` vs global. Inverted A0 (2026-08-14).
  2. **scoring window** — fixed 2,000 nt. `_w2000` vs `_w8000` gave 0.027 vs 0.087 on one arm.
  3. **substrate** — `evo2_1b_base`. An unset `EVO2_BASE_MODEL` silently uses the 7B (2026-08-17).
  4. **generation path** — batched vs sequential; all Phase-3 arms are batched. See [P3-B8].
  5. **regime** — de novo vs seeded are never pooled.
  Generation *length* may differ safely (A0 8,000 vs controls 4,000) **because the scored span is
  a fixed 2,000-nt prefix** and an autoregressive model writes the same first 2,000 tokens whatever
  total it is asked for. That is what the fixed window is for.
- **Valid vs:** any other arm whose `scoring` stamp matches on all five axes above.
- **Status:** **MANDATORY for every Phase-3 arm.** Enforced by `tests/test_docs_contract.py`.

### THE LADDER  [evaluation] [method]
- **Is:** The validated ordering of capability metrics, replacing the single binary gate. Each rung
  is maximised by copying training data, so **novelty guards all of them**.
- ⚠️ **AUROC PROVENANCE (flagged 2026-08-18).** All three AUROCs below were measured by
  `ladder_audit.py` on **seeded** arms scored against the **GLOBAL** biosynthetic Pfam set, before
  the class-specific scorer existed. Phase 3 scores against `OBLIGATE_DOMAINS[<CLASS>]` in a
  **de novo and short-seed** regime. **Whether the AUROCs transfer to that configuration is
  unverified.** They justified adopting these metrics; they are not evidence about Phase-3 rankings.
  Re-deriving them under the class-specific scorer is cheap and unqueued.

  | rung | metric | AUROC |
  |---|---|---|
  | 1 | `best_bio_bits` | **0.950** (PRIMARY) |
  | 2 | `n_bio_domains` | 0.919 |
  | 3 | `bio_span_frac` | 0.896 |
  | 4 | antiSMASH detection | — |
  | 5 | `correct_class` | — |

  Off-ladder: `biosynthetic_fraction` (specificity diagnostic), `max_orf_aa` (structural, DEMOTED).
- **Computed by:** `evo2/scripts/score_ladder.py`, `evo2/scripts/ladder_audit.py`.

---

## Retired

### `taxon_faithfulness` / `conditioning_faithful`  [evaluation]
- **Status:** **RETIRED 2026-08-10.** Returned `no_verdict` on 870/870 records; its dinucleotide
  sub-check rejected ~63% of *real* held-out cores, so a perfect model could not pass it. It also
  graded taxon conditioning, which is not what this project tests. The function survives **only**
  for `evo2/scripts/conditioning_experiment.py`. Do not re-add it to the suite.

**Also retired** — no entry, do not reintroduce: synthesis feasibility · Evo2 perplexity ·
BiG-SCAPE · E. coli expressibility as a gate · the six-frame ORF finder (replaced by pyrodigal) ·
the flat `metric_1..metric_11` numbering.
