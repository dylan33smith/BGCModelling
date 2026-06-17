# Phase-1 Reassessment & Conditioning Redesign Plan

_Created 2026-06-16. Owner: Dylan. Status: ACTIVE — executing Step 0 (diagnostic)._

This plan supersedes "just keep training the current run." It captures what we
learned, why the current setup likely can't reach the goal as-is, and the
diagnostic-first sequence to fix it. Don't lose track of this.

## The goal (unchanged)
Fine-tune Evo2 7B (LoRA) to generate **class + taxonomy-conditioned** biosynthetic
gene clusters that are biologically plausible and class-correct (right obligate
biosynthetic domains), novel, and eventually synthesizable.

## What we observed
- At ~1.1 epochs (step 250, full 32k generation, so NOT a truncation artifact):
  `obligate_fraction = 0.0`, `domain_recovery = 0.0`, but `any_domain_rate = 1.0`.
  → The model makes **generic** protein-coding DNA with recognizable Pfam domains
  but **none of the class-defining biosynthetic machinery** (NRPS C–A–T, PKS
  KS–AT–ACP, …).
- Val loss is nearly flat and is a poor proxy (pretrained base + loss-masked
  prefix), so it tells us little. `obligate_fraction` is the real signal.

## Root-cause hypothesis: TWO axes, not "train longer" and not "LoRA capacity"
1. **Conditioning axis** — the class signal is weak / may not be read:
   - Our taxonomy tag is **UPPERCASE** (`|D__BACTERIA;P__PSEUDOMONADOTA;…|`) but
     Evo2 was pretrained on **lowercase GTDB** (`|d__Bacteria;p__Pseudomonadota;…|`).
     On a byte-level tokenizer these are different tokens → our tag is
     **out-of-distribution**, so we're likely NOT engaging the pretrained
     taxonomy-conditioning pathway and are wasting LoRA capacity relearning it.
   - The class is injected as a **non-native, loss-masked** `|COMPOUND_CLASS:{cls}|`
     block (with a `||` double-pipe seam) that the model never saw in pretraining;
     LoRA must learn this channel from scratch.
   - LIMA "less is more" does NOT transfer here: LIMA *elicits* latent capability;
     we're trying to *install* a new conditional distribution, which is data-hungry
     in the conditioning dimension. At ~1000 seqs/class, prefix conditioning is
     prone to being "amortized away" (ignoring the tag costs ~no loss).
2. **Signal/objective axis** — even if conditioned, the objective barely rewards
   the class-defining domains:
   - Next-token NLL is dominated by generic sequence; the rare obligate domains are
     a small fraction of supervised tokens and the model isn't penalized for
     substituting a generic protein.
   - Training sequences include large non-core regions. antiSMASH gene-`kind`
     shows the **core biosynthetic span ≈ 50%** of a stored region; the rest is
     regulatory/transport/additional/flanking → dilutes the class signal.
   - Target classes are almost all multi-window (NRPS 95%, PKS 93%, hybrid 99%,
     siderophore 99% exceed 32k), so the de-novo class→complete-cluster signal is
     split across windows.

## Things we RULED OUT (so we don't chase them)
- **Dead adapters** — grad_norm is healthy (0.12 → ~0.015, never collapsed). LoRA
  is learning.
- **Untrained new-token embedding** — N/A. Byte-level tokenizer; the class is
  spelled in existing bytes, so there is no new random-embedding token.
- **EOS design** — custom `|END|` is correct; Evo2's native eos is byte 0 (null),
  never appears in data, and generation is fixed-length (`stop_at_eos=False`). We
  trim at `|END|`. Low priority; leave as-is.
- **Batched generation speedup** — empirically fails for this model (left-pad
  perturbs StripedHyena); gate validated and fell back to sequential. Done.

## THE PLAN (diagnostic-first)

### Step 0 — Class-discrimination diagnostic (DECISIVE, cheap, no retraining)  ← STARTING NOW
Run on the current `step_250` checkpoint. Hold taxonomy FIXED, vary only the class
tag, **greedy decoding (top_k=1)** so any difference is attributable to the class:
- If outputs are ~identical across classes → **conditioning is dead** (Axis 1
  dominates) → per-class adapters and/or native-tag + stronger conditioning.
- If outputs differ by class but lack obligate domains → **conditioning partially
  works, objective/dilution dominates** (Axis 2) → core-region trimming + objective.
- Also tabulate a class × obligate-domain matrix (does each class show ITS domains?).
Script: `scripts/diagnose_conditioning.sh` (+ parser). Output:
`/data2/ds85/bgcmodel_runs/conditioning_diag_step250/`.

### Step 1 — Free fixes (do regardless of Step 0)
- **Normalize taxonomy to native lowercase GTDB** (`d__Bacteria;p__…`), rebuild
  splits + re-fingerprint.
- **Encode the class inside the native tag** (append as a separate field, e.g.
  after species, in native style) instead of the bespoke `|COMPOUND_CLASS:|` block —
  ride the pretrained, loss-masked taxonomy channel.

### Step 2 — Architecture fork (decided by Step 0)
- **Per-class adapters** (one LoRA per class, no tag, no cross-class contrast) as a
  de-risking v1 — removes the "learn to read the tag" burden; literature-backed at
  ~1000/class (scGPT per-celltype, TaxaDiffusion per-species). OR
- **Keep one conditional model** with native tags (+ possible unmask curriculum) if
  Step 0 shows conditioning already partly works.

### Step 3 — Core-region trimming (do regardless)  [design locked 2026-06-16]
Source GBKs are NOT on disk (only a 33-genome truncated beta); user is re-acquiring
the full `asdb5_gbks.tar` (172 GiB, antiSMASH 8.1, has per-CDS `gene_kind`) →
`/data2/ds85/asdb5_gbks/`. gene_kind ∈ {biosynthetic, biosynthetic-additional,
transport, regulatory, other}; core = min-start..max-end of qualifying CDS.

**Approach — FRESH REBUILD on cores (re-split from scratch; do NOT inherit
`splits_dedup`).** Rationale: the old split kept *train frozen* only because we
were resuming the step-200 model — that constraint is gone now (fresh run).
Near-dup leakage must be recomputed on the CORE sequences (different from
full-region dedup). Re-splitting recovers the ~41% of val/test the old pass
dropped, and shorter cores let us raise the per-class cap (helps the
conditioning data-hunger problem). Pipeline:
1. `build_core_records.py` — stream the 172 GiB tar ONCE (parse all regions). Per
   region compute BOTH core spans (strict {biosynthetic}; wide {+additional}) from
   CDS `gene_kind`, and **re-extract core nucleotides directly from the GBK contig**
   `contig.seq[core_start:core_end]` — NOT from the stored record sequence (the
   original ingestion CENTER-TRUNCATES regions > context window). Store both core
   seqs + coords + lengths + class + lowercase-GTDB tag → `asdb5_core_records.jsonl`.
   (Storing both core variants is cheap since cores ≪ full regions; downstream
   needs no further tar passes.)
2. Pick strict vs +additional from per-class length stats (median core len, %
   single-window <32k, core fraction) on the REAL NRPS/PKS/hybrid data.
3. `curate` — quality filters; reconsider per-class cap UPWARD (cores are smaller
   → cheaper → afford more examples/class); diversity-stratify.
4. `split_dataset_grouped.py` — genome-disjoint train/val/test (reuse).
5. cross-split near-dup removal ON CORES (MMseqs2, ALL splits — not train-frozen)
   → `splits_core/{train,val,test}.jsonl`.

**Core definition (DECISION, deferred to real stats):** contiguous span over
`gene_kind` ∈ {biosynthetic} (strict; max signal concentration; aligns with the
obligate domains M2 grades) vs {biosynthetic, biosynthetic-additional} (full
biosynthetic locus). Strict snaps to gene boundaries; optional ~1 kb flank for
promoter/edge context. BOTH spans emitted in one pass; pick after seeing stats.

**Edge cases:** region with no qualifying CDS → fallback keep full region (flag,
report count); core > context window → center-truncate the core; contig_edge
already filtered in curate. Validate: len(core) == core_end-core_start; spot-check
M2 on a few cores (obligate domains should surface far more than on full regions).

**Cost:** one sequential ~172 GiB tar pass (~1–3 h I/O), parse only needed
genomes. Pure offline/CPU; no GPU; clean splits untouched.

- Deprioritize confidence-filtering (MIBiG already curated; detection-kind not in
  the GBKs anyway).

### Step 4 — Eval cleanup
- **Demote M4 (synthesizability)** from gates/headline → diagnostic only (real
  MIBiG BGCs fail it 85%; it's a manufacturing axis, not biology).
- **Fix or discount M1 (antiSMASH class-ID)** — it passes only 15% on real BGCs,
  so it's not a trustworthy gate yet. Trust M2 (obligate domains) as the signal.
- Add the class-discrimination check as a standing diagnostic.

## Eval calibration finding (2026-06-17) — OBLIGATE_DOMAINS too narrow
Integration-testing M10/M11 on REAL splits_core cores revealed M2's per-class
OBLIGATE_DOMAINS only cover the TEXTBOOK form of each class, missing real subtype
diversity. Example: a "TERPENE" core is a CAROTENOID cluster (SQS_PSY PF00494,
Lycopene_cyclase PF05834) — valid terpenoid biosynthesis, but our terpene obligate
list is the classic cyclases (PF03936/19086/01397), so M2 scores it "missing".
Likewise asdb5 "NRPS" includes NRPS-like single-A clusters (no C-A-T module).
=> Core extraction is CORRECT (cores DO contain the biosynthetic enzymes; v2 data
   is valid). The gap is the EVAL's obligate definitions (same class of problem as
   M1 being uncalibrated). M2 / M11 / quick_eval obligate_fraction UNDER-report on
   non-textbook members; the trend signal still holds for modular NRPS/PKS.
FOLLOW-UP (recommended): data-driven recalibration — scan splits_core cores per
   class, find the Pfams actually enriched/characteristic per class, and broaden
   OBLIGATE_DOMAINS accordingly (analogous to M1 recalibration).
RESOLVED 2026-06-17: scripts/derive_class_markers.py scanned 40 cores/class (one
   batched Pfam-A search) → data/.../class_markers.json. OBLIGATE_DOMAINS replaced
   with data-derived markers for all 22 classes (rule: freq>=0.3&enr>=4 OR
   freq>=0.08&enr>=8 for rare-but-specific subtypes, e.g. type-III PKS chalcone
   synthase, lanthipeptide RiPP, carotenoid terpene). M2 semantics changed ALL->ANY
   (pass = contains ANY class marker = "has class-defining machinery"; module
   COMPLETENESS stays with M11). Re-validated: the 3 cores that previously FAILED
   (NRPS-like, type-III PKS, carotenoid) now PASS. Held-out pass-rate check:
   scripts/validate_m2_calibration.py.

## v2 eval additions (2026-06-17): M10 sequence-quality + M11 module-architecture
- M10 (DIAGNOSTIC): coding density (union ORF coverage), ORF count/sizes,
  complete-ORF fraction — catches degenerate/low-complexity output the gates miss.
- M11 (DIAGNOSTIC): from M2's positioned obligate domains, counts assembly-line
  MODULES and whether they are IN ORDER (collinearity) — upgrades M2's binary
  presence to "complete, correctly-ordered modules". NRPS/PKS/hybrid only.
- Both wired into evaluate_bgc + DIAGNOSTIC_METRICS (10,11) + quick_eval track row
  (coding_density, module_count, ordered_modules, in_order_fraction). Tested.

## Eval suite rewrite from first principles (2026-06-17) — DONE
Rebuilt the suite around the FIVE questions that actually matter (dropping wet-lab
axes), instead of accreting metrics. One consistent gene caller feeds everything.

GENE CALLER: replaced the legacy six-frame ORF scanner with **Prodigal (pyrodigal)**
across M2/M8/M10/M11. Six-frame was ATG-only and fragmented megasynthases (the PKS
0.60 in the M2 calibration was largely this). Chose Prodigal over FragGeneScan:
standard prokaryotic caller, the one antiSMASH itself uses (internal consistency),
strict (a frameshift → partial genes, which is honest — a broken gene IS broken),
and it flags partial/edge-truncated genes (free completeness signal). Benchmarked
on real cores: 3–4× fewer/cleaner genes (NRPS 232 vs 978 ORFs), ≥ domain detection,
recovers full megasynthases (e.g. a 2110 aa hybrid PKS-NRPS gene → M11 finds a
complete ordered module). `find_orfs` is now a pyrodigal wrapper (same ORF interface
+ `.partial`); six-frame kept only as a fallback if pyrodigal is unavailable.

THE SUITE (keyed to the five questions; see GATE/DIAGNOSTIC in evaluation.py):
- Q1 is it a BGC at all?       → **M10** sequence quality (Prodigal coding density,
  gene count, COMPLETE-gene fraction). **GATE.** Note: Prodigal calls one long
  *partial* ORF even in GC-repeat junk, so the discriminator is "≥1 complete gene",
  not raw coding density.
- Q2 correct class?            → **M2** class-marker domains (ANY marker = right
  machinery; data-driven 22-class markers). **GATE.** Plus **M1** antiSMASH class
  (gold standard, **DIAGNOSTIC until recalibrated** — ~15% on real BGCs).
- Q3 plausible proteins/genes? → **M8** protein homology to known enzymes. DIAG.
- Q4 novel?                    → **M9** anti-memorization k-mer novelty. **GATE.**
- Q5 complete / correct?       → **M11** module architecture (ordered modules). DIAG.
- conditioning faithfulness    → **M7** taxon faithfulness (codon/GC vs taxon). DIAG
  (E. coli expressibility sub-score kept for the conditioning experiment but no
  longer gates anything).

GATES = (M2, M9, M10); DIAGNOSTICS = (M1, M7, M8, M11). Headline rekeyed:
generates_bgc (M10) → correct_class (M2) → biological_valid (M10∧M2) → ACCEPT
(∧ novel M9). M3 ESMFold = OPTIONAL opt-in (`EvalConfig.run_optional_esmfold`).

RETIRED (deleted, not just skipped — 209 lines removed): **M4** synthesizability
(wet-lab manufacturing axis), **M5** Evo2 base perplexity (near-circular, told us
nothing), **M6** BiG-SCAPE (unimplemented stub). E. coli expressibility pruned from
gating.

Consumers updated: eval_suite_driver headline + roles + print labels; quick_eval
(skip 1 3 8 9; data path → splits_core; track row drops synth, adds taxon_faithful;
M2 ANY-semantics note); diagnose_conditioning{,_stochastic}.sh (skip list + data
path). Tests rewritten (gates {2,9,10}, M10 completeness discriminator) — all 8 test
files pass. STILL OPEN: M1 recalibration (re-gate Q2 once antiSMASH passes real BGCs);
optional FragGeneScan supplement only if generations turn out frameshift-heavy.

## Eval v2: named checks/questions + antiSMASH as the is_bgc/correct_class gate (2026-06-17) — DONE
Two follow-ups after the first-principles rewrite, both user-approved:

(1) DROP the metric_N numbering → two named layers (evaluation.py):
- CHECKS (compute units): coding_sanity, antismash, class_markers, kmer_novelty,
  protein_homology, module_architecture, taxon_faithfulness (+ optional
  protein_foldability). Functions renamed metric_N_* → check_*.
- QUESTIONS (derived verdicts via derive_questions): is_bgc, correct_class, novel
  (GATES) + proteins_plausible, complete, conditioning_faithful (diagnostics).
- evaluate_bgc now returns per-CHECK + per-QUESTION (`questions`, aliased `summary`)
  verdicts. EvalConfig.skip_metrics→skip_checks (names); driver --skip-metrics→
  --skip-checks; quick_eval / diagnose / evaluate_bgc.py / tests all migrated. All
  tests pass; end-to-end validated on real cores.

(2) antiSMASH is the GOLD-STANDARD is_bgc + correct_class gate (recalibrated):
- WHY it was 15%: not parsing (areas[].products is correct) and not core-trimming —
  it was MAP COVERAGE. antiSMASH 8 emits 103 product types; the old map covered a
  fraction, so real clusters (NRP-metallophore, PKS-like, NI-siderophore, ...) fell
  to OTHER and failed. antiSMASH runs in ~3 s/core, detects ~97% of real cores.
- FIX: scripts/build_class_map.py regenerates config/compound_class_map.yaml from
  antiSMASH's own product→category grouping + overrides (CDPS/PUFA/arylpolyene/
  siderophores + the single-product "other" classes) → 78/103 products map to a
  specific class. check_antismash now emits `detected` (is_bgc) + `class_match`
  (correct_class, ANY mapped product = class; hybrid = both PKS & NRPS present).
- derive_questions: is_bgc trusts antiSMASH detection when it ran (authoritative);
  coding_sanity is the junk FLOOR (now a dinucleotide-ENTROPY complexity guard, NOT
  gene-completeness — a legit core can be one edge-truncated megasynthase). When
  antiSMASH is skipped (rare), class_markers (domains) is the proxy.
- VALIDATION (scripts/validate_antismash_calibration.py, 237 real held-out cores):
  is_bgc detection 0.97, correct_class 0.97 (was ~0.15). Per-class ~1.0 except
  SACCHARIDE 0.80 / HSERLACTONE 0.85 (co-located other clusters) — acceptable.
- quick_eval now RUNS antiSMASH (cheap) for real is_bgc/correct_class; skips only
  protein_homology + kmer_novelty (DB-bound / corpus scan). Calibration data:
  /data2/ds85/bgcmodel_data/as_calib.jsonl.
- Gene caller is pyrodigal everywhere (replaces six-frame); pyrodigal>=3 in requirements.

## Notes
- Current training run STOPPED 2026-06-16 (user OK'd — setup changes either way).
  Latest checkpoint kept: `step_250`.
- Run dir (v1): `/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260604_151651_L32768/`.
- v1 active data: `/data2/ds85/bgcmodel_data/splits_dedup/`.

## v2 BUILD COMPLETE — ready to launch (2026-06-17)
Data: `/data2/ds85/bgcmodel_data/splits_core/{train,val,test}.jsonl`
  train 47,524 / val 8,048 / test 18,871; 22 classes; strict cores (median ~3.1 kb,
  ~88% single-window); native lowercase GTDB tags; `training_text` present;
  leakage-clean (genome-disjoint + exact + cross-split MMseqs2 near-dup removed).
MiBIG HELD OUT: removed 8,784 train (15.6%) / 26 val / 238 test cores that were
  near-dups of the 2,636 MiBIG BGCs (so v2 never sees MiBIG, the positive-control
  eval is genuinely held-out, and MiBIG is RESERVED for a possible Phase-2
  COMPOUND-conditioned fine-tune — the holy grail: condition on compound name, not
  just class). Pre-exclusion backup: `splits_core_premibig/`. Reserve: 2,636 MiBIG
  GBKs at data/mibig/mibig_gbk_4.0/ (Phase-2: run the same strict-core+native-tag
  pipeline keyed on compound, IF v2 works). Tool: scripts/exclude_mibig_from_core.py.
Pipeline scripts: build_core_records.py → materialize (drop OTHER+rare) →
  split_dataset_grouped.py → curate_dataset.py (--train-cap 4000 --min-len 300) →
  dedup_core_splits.py.
Trainer: UNCHANGED. Native tag flows via canonical_phase1_prefix. class-in-tag
  (;b__) DEFERRED to a post-v2 A/B (real ripple, uncertain benefit).
Eval: first-principles rewrite (see section above). Gates = M2/M9/M10; diagnostics
  = M1/M7/M8/M11; M4/M5/M6 retired; Prodigal replaces six-frame (evaluation.py + driver).
Diagnostic: greedy/k-mer pitfalls fixed (3/5-mer cosine + domain Jaccard) for the
  v2 re-run.

**v2 launch command (v1-style: tmux + idle-GPU gate + checkpoints + auto-resume).
NOT yet launched — awaiting explicit go:**
```
cd /home/ds85/projects/BCGModelling
tmux new-session -d -s bgc_v2 \
  "scripts/queue_h100_production.sh \
     --train /data2/ds85/bgcmodel_data/splits_core/train.jsonl \
     --val   /data2/ds85/bgcmodel_data/splits_core/val.jsonl \
     --wandb-project bcg-evo2-phase1-v2"
# watch: tmux attach -t bgc_v2   |   run dir: /data2/ds85/bgcmodel_runs/phase1_lora_prod_<TS>_L32768
```
After v2 trains a bit: run quick_eval + re-run diagnose_conditioning_stochastic.sh
on a checkpoint → decide Step 2 (per-class vs conditional).
