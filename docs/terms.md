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
- **Status:** **PRIMARY.** ⚠️ **The 0.950 ladder AUROC DOES NOT TRANSFER to Phase 3** — re-derived
  within the on-class pool against antiSMASH labels it is **0.575** (see THE LADDER). It justified
  ADOPTING this metric; it is not evidence it ranks Phase-3 candidates. Heavily zero-inflated —
  report as a **rate with an exact binomial CI**, never as an arm mean.
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
  stayed ≤0.021 while the model was demonstrably rebuilding the seeded cluster at L=500 — 12/12
  on-class hits reproduced their own source cluster's domain, vs 0/8 at L=8. ⚠️ **Corrected
  2026-08-24:** this used to be argued from "`protein_aai` rose to 0.914". **That trend was a POOLING
  artifact** — 0.914 is a single pooled maximum, and among on-class records AAI does **not** rise with
  seed length (L8 **0.499** · L100 0.635 · L500 **0.450**). The containment-is-blind conclusion stands,
  on the domain-match evidence, which is independent and stronger. See `memory.md` 2026-08-18.
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
- **Status:** SECONDARY, rung 3. ⚠️ **The 0.896 ladder AUROC DOES NOT TRANSFER** — re-derived in the
  class-specific regime it **INVERTS to 0.173**, and real cores average only 1.04 biosynthetic
  domains inside a 2 kb window, so there is no span to measure. **Must not be used as a cluster rung
  at 2 kb** (see THE LADDER). ⛔ **VOID for PKS** per `config/class_eval_policy.yaml`. Between-group
  reference, 8 kb: real cores **0.876** vs de novo **0.051**.

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
- **Status:** GATE. ⚠️ **CORRECTED 2026-08-24 — this entry previously read "~0 de novo since project
  start."** That was true of **label/prefix conditioning** and is **no longer true under a
  class-specific adapter**, where it is now MEASURED and non-zero. antiSMASH, de novo, Stage B:
  **PKS `[P6-AS]` 0.040** (ceiling 0.980) · **TERPENE `[P7-AS]` 0.065** (ceiling 1.000) · **all four
  base/general control arms 0.000**. Seeded RIPP `[P3-AS]` reads **0.485** against a **0.740** real-core
  ceiling. ⚠️ On every set measured so far `is_bgc` and `correct_class` are **identical** — when this
  model produces a cluster, it is on-class; the gap is detection, not class. Decomposes as
  `P(detect) × P(right | detect)`. The pre-Phase-3 form of this claim is superseded in `CLAUDE.md`
  Standing Constraint 8.

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

## H

### `hit_eos`  [evaluation] [diagnostic]
- **Is:** Whether a generated record terminated on the model's own stop signal, rather than running
  out of `--max-new-tokens`. ⚠️ **This identifier has meant TWO DIFFERENT THINGS**, and the change is
  the entire reason it read zero for months. **The doc-canonical meaning is: the model sampled
  TOKEN ID 0.**
- **Computed by:**
  - ⛔ **STRING PATH (legacy, still live and WRONG):** `evo2/scripts/generate_bgc.py:extract_sequence`
    → `EOS_MARKER in generated`, where `EOS_MARKER = "|END|"` is a **5-byte string**
    (124,69,78,68,124). **This is what produced every `hit_eos` on disk.**
  - ✅ **TOKEN-ID PATH (correct):** `evo2/scripts/constrained_generation.py:TokenRecorder.sequences`
    → `ids[-1] == EOS_ID` where `EOS_ID = 0`. ⚠️ **Not yet wired into `generate_bgc.py` /
    `seed_generate.py`** — `plan.md` [X1e]/[X1g].
  - GenomeOcean: `genomeocean/scripts/generate_class_go.py` tests `eos_token_id` (id 2) directly and
    is correct by construction — its BPE tokenizer auto-wraps `BOS=1 … EOS=2`.
- **CHANGES MEANING WITH:** ⚠️ **WHICH PATH COMPUTED IT — the #2 drift risk after the Pfam subset.**
  The string path is a **structural zero**: `|END|` has read **0/150, 0/188, 0/200, 0/200** across
  every arm ever generated, and it is now RETIRED from training. Also changes with `--junk-policy`
  and with constrained decoding (which raises the observed rate by renormalising junk mass onto
  `{ACGTN,EOS}` — 9/24 unconstrained vs 11/24 and 21/32 constrained, ⚠️ different n and prompt
  subsets, direction suggestive only).
- **Valid vs:** same path only. ⛔ **A string-path `hit_eos` may NEVER be compared with a token-id
  `hit_eos`, and a string-path zero may NEVER be quoted as evidence about termination.**
- **Status:** **DIAGNOSTIC.** ⚠️ **DO NOT READ A ZERO HERE AS "the model cannot stop."** That
  inference was made and is **RETRACTED (2026-08-20)**. The model terminates on **token id 0**, which
  **Evo2 pretrained with** (`P(EOS)` end/mid = 40.9x base, 2,100x adapter) and which our fine-tuning
  sharpened ~51x; at generation, **13/13 coherent stop positions are id 0 at 16x–159x uniform**; and
  masking id 0 **causally** restores median length **4,583 → 8,000**. ⚠️ ids 0 (EOS), 1 (PAD) and 32
  (space) **all detokenize to `' '`**, so no string-level test can ever see it.
- **Aliases:** "EOS rate", "termination rate", `|END|` rate, T4.4. Do not use.

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
- **Status:** SECONDARY, rung 2. ⚠️ **The 0.919 ladder AUROC DOES NOT TRANSFER** — re-derived in the
  class-specific regime it is **0.519**, i.e. chance (see THE LADDER).

### `n_bio_orfs`  [evaluation]
- ⛔ **QUOTE THE CEILING FROM THE ARM'S OWN SUBCLASS, NEVER THE CLASS-LEVEL SAMPLE (2026-09-01).**
  The widely-quoted **1.454** is the RIPP **class-level 50-core** figure. Per subclass, real held-out
  cores carry **1.79–3.38** — azole **3.378**, lassopeptide **2.833**, PKS T1PKS **2.767**. Using the
  class figure for a subclass arm **halves the apparent multi-gene gap** and did so in the `[P13]`
  report before correction. Each run dir already carries `realtest_full_<PARENT>.json` with the right
  number; use it.
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

### `biosynthetic_density`  [dataset] [training]
- **Is:** Share of a training record's nucleotides inside a gene carrying biosynthetic content.
  ⚠️ **Denominator-sensitive — state which numerator you used.**
- **Computed by:** annotated CDS coordinates from `ripp_components.jsonl` (or ORF calls + Pfam).
- **CHANGES MEANING WITH:** ⚠️ **the numerator, and this caused a retraction.** Counting only the
  `biosynthetic` tag while widening the span makes the value fall **mechanically** — the numerator
  is fixed while the denominator grows. Three variants, n=27,171 regions:

  | span | BIO-only | DEFINING-genes | any-CDS |
  |---|---|---|---|
  | STRICT | 0.869 | 0.869 | 0.980 |
  | bio+transport | 0.551 | 0.687 | 0.950 |
  | WIDE | 0.310 | **0.576** | 0.919 |
  | everything except `none` | 0.208 | 0.595 | 0.906 |

- **Valid vs:** the same numerator definition only. **Use DEFINING-genes** for comparing span
  definitions; BIO-only is only meaningful within a fixed span.
- **Status:** **DIAGNOSTIC.** ⚠️ It is **NOT established** as the cause of [P4-WIDE]'s failure — that
  failure is a direct experimental result (**Holm p=0.021 at 2.2 kb; the 8 kb contrast is n.s. at
  p=0.15**); the dilution explanation is a hypothesis whose magnitude was overstated and was
  **RETRACTED as partly circular**. ⚠️ **Corrected 2026-08-24** — this line previously quoted
  **p=4.1e-04 / 3.2e-05**, which are the **pre-deduplication** values ([P5-DEDUP]: effective n was
  47–141, not 188). See `memory.md` 2026-08-19.

### `subclass_specificity`  [evaluation] [method]
- **Is:** Of the generations antiSMASH **detects**, the fraction assigned a **specific RiPP subclass**
  (lanthipeptide-class-i…v, lassopeptide, thiopeptide, sactipeptide, thioamitides…) rather than the
  generic catch-all **`RiPP-like`**.
- **Computed by:** full-mode antiSMASH (⚠️ **not `--minimal`**) → `region` feature `product`
  qualifiers. Denominator = detected regions, so it is a **Stage-B** metric.
- ⚠️ **COUNT PER DETECTED SEQUENCE, NOT PER PRODUCT STRING.** A region can carry several products,
  so counting strings inflates the denominator and deflates the rate. Counting strings is what
  produced the withdrawn "~70%" real-core figure; per sequence it is **0.909** (30/33).
- **Why it matters:** antiSMASH uses a rule hierarchy — tight subclass rules each require a specific
  **combination** of domains; the loose `RiPP-like` rule fires on weaker evidence when none match.
  Producing a subclass call is therefore strictly harder than producing a detection.
- **CHANGES MEANING WITH:** antiSMASH version and rule set; `--minimal` (which still emits products,
  but the analysis modules that refine them do not run).
- **Valid vs:** real cores through the identical path — **including the same `--minlength`**.
- **TWO READINGS, BOTH REPORTED** (added 2026-08-24, `[P9]`): antiSMASH emits `RRE-containing`, which
  is a **domain rule** (the RiPP recognition element) and not a chemistry assignment. **strict** =
  a named chemistry only (lanthipeptide-class-i…v, lassopeptide, ranthipeptide, thiopeptide,
  sactipeptide, microviridin, azole-containing-RiPP, thioamitides, triceptide, redox-cofactor,
  cyclic-lactone-autoinducer); **lenient** = anything that is not bare `RiPP-like`, so
  `RRE-containing` counts. Off-class products on a multi-product region (`terpene-precursor`,
  `arylpolyene`, `nucleoside`, `T2PKS`, `oligosaccharide`) are excluded from both before judging.
  Quote both or neither — strict alone understates, lenient alone overstates.
- ⚠️ **THE CEILING IS REGIME-DEPENDENT AND WAS RE-DERIVED 2026-08-24.** The **0.909 (30/33)** figure
  was measured at antiSMASH's default 1,000 nt floor, which rejects **14/50 real RIPP cores**. At
  `--minlength 200` all 50 run and all 50 detect, and the ceiling is **0.500 strict / 0.740 lenient
  (n=50)**. `phase5_classprobe/as_real_RIPP_ml200.tsv`. **Never read an `ml200` arm against the
  0.909 reference.**
- **Status:** **The honest measure of the remaining gap** (adopted 2026-08-19; pre-registered as a
  declared secondary, §9.2 — the primary endpoint is unchanged).
  [INCORRECT] -
  Real cores **0.909** (30/33 detected sequences carry a specific subclass); **our best arm 0.000** —
  every detection is `RiPP-like`.
  [CORRECTION - 2026-08-24]: at a matched `--minlength 200` floor the ceiling is **0.500 strict /
  0.740 lenient (25 and 37 of 50)**, and **our best arm is NO LONGER ZERO** — GenomeOcean de novo
  scores **1/13 = 0.077 strict, 3/13 = 0.231 lenient**, including a genuine `ranthipeptide`, the
  first specific RiPP chemistry this project has generated. ⛔ **But 1/13 vs Evo2's 0/7 is p=1.0** —
  the contrast is UNDERPOWERED, so "GenomeOcean does better on subclass" is NOT established. Evo2's
  RIPP detections were measured in the pre-`ml200` regime and must be re-run before any model-vs-model
  subclass claim. Supersedes `n_class_domains ≥ 2`, `bio_span_frac` and the precursor panels, all of
  which were indirect proxies for this and each failed validation.
- ⚠️ **POWER: report the TEST, not a fixed denominator.** [WITHDRAWN 2026-08-20] An earlier version
  of this entry set an arbitrary ">=15 detections" floor. That was the wrong instrument: generation
  is the cheap step here, so the honest requirement is simply that **the contrast is significant
  against its own control at whatever n was generated**, with the exact test and denominator quoted.
  A significant 0/7-vs-30/33 (p≈1e-5) is a result; an underpowered null is not. **If a contrast is
  n.s., generate more rather than arguing from a threshold.**
- ⛔ **AND A ZERO AT A SMALL DENOMINATOR IS NOT AN ABSENCE — worked example, 2026-08-24.** The
  analogous TERPENE claim ("the model produces the precursor and **never** the cyclase", **0/13**,
  p=0.0024) was **RETRACTED** when Evo2 de novo was pooled to **n=800**: detections rose 13 → 48 and
  **3 cyclase calls appeared = 0.062**. The limitation is real — 0.062 against a real-core **0.440**,
  p=1.65e-05 — but it is **"rarely", not "never"**. ⇒ **Never write "never" off a single-digit
  detection count.** The RIPP **0/7** and PKS **0/8** zeros are **untested at power** and must be
  described as such until they are pooled the same way. See `memory.md` 2026-08-24 and
  `feedback-sample-size-reliability` — this is the same small-n trap that has now fired twice.
- ⚠️ **A significant DIRECTION is not an estimated RATE.** At these denominators "the model never
  makes the harder subclass" is established; "it makes it x% of the time" is not.

### `aai_to_ref`  [dataset] [method]
- **Is:** Coverage-weighted proteome identity of a record to a single frozen REFERENCE record —
  "how much of the reference's proteome does this record reproduce, weighted by identity", in [0,1].
  The axis the `[ID_*]` conditioning buckets are cut on (`docs/phase13_IDENTITY_BUCKET_preregistration.md`).
  ```
  aai_to_ref = Σ_refprot ( best_fident × alnlen / tlen ) × tlen  /  Σ_refprot tlen
  ```
- **Computed by:** `scripts/build_identity_buckets.py --metric protein` — `novelty_battery.translate_orfs`
  (`min_aa=30`) then MMseqs2 `easy-search -e 1e-3 -s 5.7`, matching the `protein_aai` search config.
- ⚠️ **DELIBERATELY NOT `protein_aai`.** `protein_aai` is the **max over a record's ORFs** — one
  shared enzyme scores high. Template fidelity has to be a **proteome-coverage** statement, so this
  sums over the reference's proteins instead. **Never substitute one for the other**; `protein_aai`
  stays the novelty gate, `aai_to_ref` is a conditioning axis and is **not a gate**.
- **CHANGES MEANING WITH:** the reference record (re-derive the medoid whenever the split changes),
  `min_aa`, and the MMseqs sensitivity. ⛔ **Not comparable to `containment` (k=21) or to a published
  ANI** — different instrument, different denominator.
- **Valid vs:** the same reference and the same ORF/search settings only.
- **Status:** **ACTIVE, Phase 13.** ⚠️ **The nucleotide analogue `ani_to_ref` is RETIRED for BGC use
  and the reason matters:** on AZOLE_CONTAINING_RIPP its median was **0.0000** — 76.5% of records had
  zero alignable nucleotide identity to the DNA medoid — failing Gate T0 (top bucket n=3 vs ≥30) with
  a degenerate quintile fallback (cuts `[0.0, 0.0, 0.0, 0.2021]`). Dynamic range was demonstrated, so
  that zero is real. ⇒ ★ ***Microviridae* is a taxonomic FAMILY with a canonical member; an
  "azole-containing RiPP" is a CHEMICAL annotation over genomically unrelated clusters.** The phage
  paper's conditioning axis does not exist in our data at the nucleotide level. On protein it does,
  but **bimodally**: 40 records ≥0.95, 115 at 0.80–0.95, **16 across the whole 0.50–0.80 middle**,
  628 below 0.50. See `memory.md` 2026-08-27.

### `adapter_advantage`  [evaluation] [diagnostic]
- **Is:** Teacher-forced per-nucleotide likelihood difference on the SAME real held-out sequences,
  in bits/nt. Two forms, both reported: **vs BASE** (what fine-tuning bought) and **vs the WRONG
  adapter** (what is target-SPECIFIC rather than generic corpus adaptation). Positive = the named
  adapter finds those sequences more likely. Answers *"does the model KNOW what a real target looks
  like?"* — a question no generation metric can reach, because generation confounds knowing with
  sampling.
- **Computed by:** `genomeocean/scripts/score_likelihood_go.py` — one forward pass, NLL over the
  SEQUENCE tokens only (BOS + class-token prefix masked, so the adapter is not credited for a prompt
  the base never sees). Paired per record; report paired t and the win count, never a pooled point
  estimate alone.
- ⚠️ **QUOTE IT PAIRED AND ON MATCHED SEQUENCES.** Comparing bits/nt across DIFFERENT sequence sets
  is not interpretable — one corpus can be more predictable than another for reasons unrelated to
  the model. **The "adapter prefers its own output over real biology" reading is NOT supported by
  this instrument** and must not be quoted; only same-sequence contrasts are valid.
- **CHANGES MEANING WITH:** the reference model, the class token supplied, and the target length —
  bits/nt averages architecture signal over every nucleotide, so a long diffuse target dilutes it.
  ⚠️ **Dilution cancels in the right-vs-wrong-adapter form** (both adapters score identical
  sequences) but NOT in the vs-base form.
- **Valid vs:** the same sequence set, same prefix convention.
- **Status:** **ACTIVE, Phase 13.** Measured 2026-08-27: azole **+0.0352 vs base** (t(44)=2.49,
  28/45 records) and **+0.0553 vs the wrong adapter** (t(44)=4.24, 35/45); cyclactone **+0.1341 vs
  base** (t(33)=5.95) and **+0.1559 vs the wrong adapter** (t(33)=7.83, **34/34**). ⇒ **Both adapters
  learned real target-specific structure, but azole's is ~3x weaker and far patchier.** Combined
  with a 1.43-nat train/eval gap, that is **memorisation without generalisation** — a data problem,
  not an adapter-capacity one. See `memory.md` 2026-08-27.

### `train_frac_distinct`  [dataset] [diagnostic]
- **Is:** Fraction of a training split's records that survive MMseqs2 clustering at a stated
  nucleotide identity — `n_clusters / n_seqs`. The companion count `n_clusters` is the split's
  **effective number of distinct architectures** at that threshold. Measures how much of a split is
  near-duplicate, i.e. **how many genuinely different examples the model actually sees.**
- **Computed by:** `mmseqs easy-cluster <split>.fna out tmp --min-seq-id <ID> -c 0.5 --cov-mode 0`
  on the `sequence` field of `train.jsonl`; `n_clusters` = records in `*_rep_seq.fasta`.
- **CHANGES MEANING WITH:** ⚠️ **`--min-seq-id`, and the difference is large.** The same split reads
  0.877 at 0.99 and 0.610 at 0.80 (AZOLE_CONTAINING_RIPP). **Always quote the threshold** — an
  unlabelled `frac_distinct` is not a number. Also changes with `-c` / `--cov-mode`; this project
  uses `-c 0.5 --cov-mode 0`, matching the `splits_class` near-dup criterion (`id>=0.8 cov>=0.5`).
- **Valid vs:** the same threshold and coverage settings only. ⛔ **NOT comparable to a published
  dedup retention rate unless that paper's threshold and coverage match** — the phage paper's 94.9%
  (15,246 → 14,466) is a 99%-identity retention and may only be compared to our 0.99 row.
- **Status:** **DIAGNOSTIC — measured 2026-08-27, and it explains nothing.** Across the five [P11]
  RIPP subclasses, **r(`train_frac_distinct`@0.80, own-subclass rate) = +0.228** and
  **r(`n_clusters`@0.80, own-subclass rate) = +0.007**, against **r(log10 target nt, rate) = −0.933**
  on the same five points. ⇒ **Within-target training redundancy does NOT explain the length
  gradient**, and AZOLE (0.610) is very slightly *more* redundant than CYCLIC_LACTONE_AUTOINDUCER
  (0.645) — the opposite of the "our long targets are more heterogeneous" hypothesis. Recorded so
  the confound is not re-proposed. See `memory.md` 2026-08-27.

### THE TWO-PASS DETECTION ARCHITECTURE  [evaluation] [method]
- **Is:** Pfam gate first (cheap, Stage A), antiSMASH second (gold standard, Stage B). **Calibrated
  2026-08-18** on 218 Stage-2 sequences with both run:

  | gate status | antiSMASH confirms | n |
  |---|---|---|
  | **Pfam POSITIVE** (`best_bio_bits > 0` @ RIPP) | **0.456** | 68 |
  | **Pfam NEGATIVE** | **0.020** | 100 |
  | real held-out cores | 0.760 | 50 |

- **Verdict: the Pfam gate is a good first pass and a bad final answer.**
  - **23× enrichment** — Pfam-positives are 23× more likely to be real clusters.
  - **Sensitivity ≈ 83%** — it finds ~5 of every 6 real clusters. It **misses ~17%**, because 2.0%
    of the much larger negative pool is still real.
  - **Inflation ≈ 1.8×** — S2-1's `on_class` 0.176 corresponds to an estimated true antiSMASH rate
    of **≈0.097** (15.0 real among Pfam-positives + 3.1 missed among negatives, over 188).
- **CHANGES MEANING WITH:** the class marker set and the antiSMASH version/DB. Re-calibrate per
  class — these numbers are RIPP-specific.
- **Valid vs:** ⚠️ **never mix a Pfam rate with an antiSMASH rate in one comparison** (Standing
  Constraint 7). Report them as separate rows.
- **HOW TO GET AN UNBIASED RATE:** running antiSMASH only on Pfam-positives **systematically
  under-counts by ~17%**. For a headline rate use **stratified sampling** — antiSMASH on *all*
  Pfam-positives plus a random sample of Pfam-negatives, then
  `rate = [P·conf(pos) + N·conf(neg)] / (P+N)`. That is how the 0.097 above was obtained. For
  *selection* (which sequences to carry forward) Pfam-first alone is fine and saves ~95% of the
  compute.
- **Status:** **STANDARD for Phase 3 onward.** No alternative detector exists — see below.
- **⛔ WHY THERE IS NO SUBSTITUTE FOR antiSMASH.** Two independent instrument families were tested
  for *within-positives* discrimination and both failed: **ladder metrics** best AUROC 0.575
  (`bio_span_frac` inverted at 0.173) and the **class probe** 0.337 (anti-correlated, saturated at
  P(RIPP) ≈ 0.997 despite 0.933 balanced accuracy). Nothing we own separates a real cluster from a
  Pfam-passing near-miss. antiSMASH is the only arbiter we have.

### EXPERIMENT ID CONVENTION  [method]
- **Is:** The single naming scheme for every experiment, task and intervention referenced in the
  docs. Adopted 2026-08-24 (user) to replace **five** competing schemes that had accumulated:
  `P3-B7` (a bug), `P5-REGEN` (a task), `P6-A0` (a generation arm), `P8-T4` (a phase subtask) and
  `X1a`/`X2c` (an orphan series with no phase at all) all occupied the same slot and meant
  different kinds of thing.

  **FORM: `P<phase>-<KIND>-<slug>`**, or **`INF-<KIND>-<slug>`** for cross-cutting infrastructure
  that belongs to no phase.

  | field | rule |
  |---|---|
  | `P<phase>` | the phase number the work RUNS in. `INF` if it is tooling that spans phases. |
  | `<KIND>` | **exactly one of six**, uppercase — see the table below. This is the field that was missing and it is what makes an ID self-describing. |
  | `<slug>` | lowercase, hyphenated, names the thing. Not a letter. `azole`, not `A0`. |

  | KIND | means | example |
  |---|---|---|
  | `DAT` | build or filter a dataset / substrate | `P10-DAT-drop-generic` |
  | `TRN` | train an adapter | `P10-TRN-azole` |
  | `GEN` | produce generations | `P10-GEN-azole` |
  | `EVL` | score, antiSMASH, novelty battery | `P10-EVL-azole` |
  | `ANL` | analysis over data we already have — **no new compute on the model** | `P10-ANL-subclass-dist` |
  | `FIX` | a bug or an infrastructure change | `INF-FIX-hit-eos` |

- **Why the KIND field:** it makes the ID answer "what would this cost me?" without a lookup. `ANL`
  is free, `GEN` is GPU-hours, `TRN` is a training run. Three of this project's planning errors were
  mis-estimates of exactly that.
- ⚠️ **HISTORICAL IDs ARE FROZEN.** `memory.md` is a permanent ledger and its entries keep the IDs
  they were written with — renaming them would break every cross-reference and violate the in-place
  correction rule. **The new scheme applies to all NEW and QUEUED work only.** The table below is
  the bridge; use it to read old entries, never to rewrite them.

  | old ID | new ID | state |
  |---|---|---|
  | `[X1a]` constrained decoding | `INF-FIX-constrained-decoding` | done |
  | `[X1b]` train the real EOS | `INF-FIX-eos-token` | done |
  | `[X1g]` token-id-aware generation | `INF-FIX-token-ids` | done |
  | `[X1e]` `hit_eos` must test the token id | `INF-FIX-hit-eos` | **open** |
  | `[X1f]` per-row early stopping | `INF-FIX-per-row-stop` | **open** |
  | `[X1i]` snip-and-replace | `INF-GEN-snip-replace` | **open** |
  | `[X1c]` filter prematurely-ended sequences | `INF-GEN-filter-short` | **open** |
  | `[X1d]` / `[X1h]` degeneracy needs its own gate | `INF-EVL-degeneracy-gate` | **open** |
  | `[X2a]` bigger denominators | — | **done twice**, `P9-POOL` / `P9-EVO2POOL`; do not re-run as a rescue |
  | `[X2b]` seeded hard-subclass control | `P10-GEN-seeded-subclass` | **open** |
  | `[X2c]` inverse-frequency upweighting | `P10-TRN-invfreq` | queued |
  | `[X2d]` subclass-conditioned adapters | `P10-TRN-azole`, `P10-TRN-cla` | **running** |
  | `[X3]` GenomeOcean on TERPENE | — | done; it became **Phase 8** |
  | `[X4]` is the taxonomy prefix doing anything | `INF-ANL-taxonomy-prefix` | backlog |
  | `[P10-DATA]` | `P10-ANL-subclass-dist` | done |
  | `[P10-A]` | `P10-TRN-azole` + `P10-TRN-cla` | running |
  | `[P10-B]` | `P10-TRN-multitoken` | queued |
  | `[P10-C]` | `P10-DAT-drop-generic` | queued |
  | `[P10-D]` | `P10-TRN-invfreq` | queued |

- ⚠️ **RUN DIRECTORY AND FILE NAMES ARE A SEPARATE CONVENTION** (`CLAUDE.md`, Filesystem Naming) and
  are **NOT** renamed by this. Existing artefacts keep their paths — renaming them would orphan every
  provenance line already written. New run dirs stay `<phase>_<TARGET>[_<arm>]`.
- **Enforced by:** `tests/test_docs_contract.py` — any NEW-style ID in `plan.md` must match
  `P\d+-(DAT|TRN|GEN|EVL|ANL|FIX)-[a-z0-9-]+` or `INF-(...)-[a-z0-9-]+`; legacy IDs are allowed only
  in the Phase Ledger and in the bridge table above.
- **Status:** **MANDATORY for all new work from 2026-08-24.**

### THE TWO MEASUREMENT STAGES  [evaluation] [method]
- **Is:** Every Phase-3 arm is measured **twice, on two different denominators**, because a metric
  that is meaningful over one is meaningless over the other. Conflating them produced a retracted
  claim (`memory.md` 2026-08-18, the pooled-AAI artifact).

  **STAGE A — SELECTION. Denominator: ALL generated sequences.** Asks *"is this generation a
  positive or a negative?"* Every metric here must be defined for a sequence that contains nothing.

  | metric | why it belongs here |
  |---|---|
  | `best_bio_bits`* @ `OBLIGATE_DOMAINS[cls]` | the endpoint; 0 is a valid, meaningful value |
  | `containment`* | defined for any sequence |
  | intra-set distinctness* | a property of the whole set, not of one record |
  | `JOINT_PASS` | the per-record intersection of all gates |
  | `n_orfs`, `coding_density`, `co_orient` | structural, defined everywhere |

  **STAGE B — CHARACTERISATION. Denominator: the POSITIVES only** (records passing the endpoint).
  Asks *"given that we made something on-class, what did we actually make?"* These are
  **undefined-or-misleading over all sequences**, because an off-class record contributes a
  degenerate value that swamps the statistic.

  | metric | why it must be Stage B |
  |---|---|
  | `protein_aai`* | off-class records score **0.000 = no hit**, not "poor match". Pooling makes the median report *hit rate*, not similarity. **This is the exact error that was retracted.** |
  | antiSMASH `is_bgc`*, `correct_class`* | expensive; and the question is whether our *positives* are real clusters |
  | `n_class_domains`, `n_bio_domains`, `n_bio_orfs` | 0 by construction for a non-hit; the mean over all records is just the hit rate rescaled |
  | `bio_span_frac` | undefined with no domains |
  | `biosynthetic_fraction` | undefined when `best_any_bits` = 0 |

- **Computed by:** `scripts/novelty_battery.py` emits both — Stage A as the top-level arrays, Stage B
  as `mean_among_on_class` in the `ladder` block. antiSMASH is run separately on the positives.
- **CHANGES MEANING WITH:** the denominator, and nothing else matters more. **A Stage-B metric
  quoted over all sequences is not a weaker result — it is a different and usually wrong quantity.**
- **Valid vs:** Stage A against Stage A, Stage B against Stage B. **Every Stage-B number needs a
  real-core reference on the same denominator** (e.g. `protein_aai` 0.496 generated vs 0.641 real).
- ★ **REPORTING RULE, 2026-08-24 (user decision).** **Report the Stage-A RATE of positives, and then
  report EVERY OTHER METRIC AT STAGE B ONLY.** The project will apply **post-generation filtering**
  (`plan.md`, `[P5-FILTER]` — selection, never ranking), so negatives are discarded before anything
  downstream sees them. A Stage-A ladder value therefore answers a question nobody asks, and it
  answers it badly: because a negative contributes 0 to `n_bio_orfs`, `n_bio_domains`, `frac` and
  `bio_span_frac` **by construction**, the Stage-A column is the hit rate wearing a different name.
  Measured 2026-08-24: GenomeOcean TERPENE `bio_span_frac` **0.641 over all records vs 0.934 among
  positives**; RIPP `n_bio_orfs` **0.160 vs 1.031**.
- ⚠️ **AND STAGE B CHANGES WHAT SOME METRICS MEAN — three now SATURATE and must not be read as
  capability.** Among positives, `frac` **1.20**, `co_orient` **1.10** and `bio_span_frac` **1.07**
  of the real-core value: generated positives are *more* purely biosynthetic than real cores,
  because a real core carries non-biosynthetic content and a generation is essentially one clean
  gene. **Above 1.00 here is a symptom of single-gene output, not of quality.** The Stage-B metrics
  that still discriminate are the ones with real headroom — `n_bio_orfs`, `n_bio_domains`,
  `n_class_domains`, `n_orfs`, `max_orf_aa`.
- ⚠️ **The real-core reference must ALSO be Stage B on its own positives** (22/50 for RIPP, 49/50 for
  TERPENE), never all 50. Quote the ratio to that reference (**B/ceiling**) alongside the raw value.
- **Status:** **MANDATORY for every Phase-3 arm from 2026-08-18**; **Stage-B-only reporting for
  non-primary metrics from 2026-08-24.** Label every reported number with its stage and its n.

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
  | `n_class_domains` | distinct markers of the class. ⛔ **COUNTS PFAM ACCESSIONS, NOT CATALYTIC DOMAINS — and for PKS that inflates it ~2.7x.** `OBLIGATE_DOMAINS[PKS]` holds `PF00195`+`PF08392` (one chalcone synthase) and `PF00109`+`PF02801`+`PF16197` (one ketosynthase), so a single T3PKS gene scores 2. Real PKS cores: 0.800 by accession, **0.300** by catalytic unit, **0.060** by distinct marker-bearing ORF. RIPP and TERPENE marker sets have no such pairs and are unaffected. Never quote the accession count as cluster structure for PKS (2026-08-19). ⚠️ **NOT the definition of a cluster** — only ~16% of real cores reach ≥2 in a 2 kb window, and antiSMASH confirms single-marker records (12/24, 16/36). Diagnostic, demoted 2026-08-19. |
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
- ⚠️ **THE AUROCs BELOW DO NOT TRANSFER TO PHASE 3 — re-derived and refuted 2026-08-18.** They were
  measured on **long-seed arms against the GLOBAL Pfam set**. Re-tested within the on-class pool
  against antiSMASH labels (n=68): `best_bio_bits` **0.575** (was 0.950), `n_bio_domains` **0.519**
  (was 0.919), `bio_span_frac` **0.173 — inverted** (was 0.896). Cause: no variance left to rank
  with — `n_class_domains` among on-class is 1.000 and real cores average 1.04 domains in the 2 kb
  window. **The AUROCs justified ADOPTING these metrics in Phase 2; they are not evidence about
  Phase-3 rankings, and `bio_span_frac` must not be used as a cluster rung at 2 kb.** Nothing we
  measure ranks within our own positives (best 0.575) — see `memory.md` 2026-08-18.

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
