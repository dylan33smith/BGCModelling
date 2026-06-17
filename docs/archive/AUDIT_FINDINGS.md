# BCGModelling Evo2 Fine-Tuning — Adversarial Deep-Dive Audit

> **SUPERSEDED (2026-06-17) — point-in-time audit record; do not read as current.**
> The eval suite has since been rewritten from the `metric_1..metric_11` numbering to
> named **CHECKS → QUESTIONS** (antiSMASH is now the recalibrated `is_bgc`/`correct_class`
> gate at ~0.97 on real cores; pyrodigal replaced the six-frame ORF finder;
> synthesizability / Evo2-perplexity / BiG-SCAPE were retired; E. coli expressibility
> pruned from gating). The active dataset is `splits_core` (strict antiSMASH cores, MiBIG
> held out). For the current state see [`REDESIGN_PLAN.md`](REDESIGN_PLAN.md),
> [`EVAL_RUNBOOK.md`](EVAL_RUNBOOK.md), and `src/bgc_pipeline/evaluation.py`.

## Deferred / Future Work (decisions)

- **2026-06-08 — DEFERRED: near-duplicate leakage (FABLE5 C2/C7).** Splits are
  genome- and exact-md5-disjoint but NOT near-duplicate-disjoint (a val record
  with first-2048nt containment = 1.0 to a train record; ~31% species overlap).
  **Decision:** do NOT block the current first production run on this. It does not
  corrupt *training* — only makes the first-window val loss slightly optimistic and
  makes rigorous generalization/novelty CLAIMS unsupportable. We want a directional
  full-epoch first pass. **Must fix before any rigorous evaluation or published
  claim**, and it also unblocks the novelty-threshold calibration (C3). Fix:
  identity-cluster all records (MIBiG + antiSMASH) at ~90–95% over the BGC interval
  (mmseqs2 / minimap2 / CD-HIT-EST), assign whole clusters to one split; rebuild the
  positive control with a containment-based disjointness guard; add a pre-train
  assertion failing on any cross-split pair > 0.9 containment; then re-validate any
  checkpoint decisions.
- **2026-06-08 — DEFERRED: `find_latest_checkpoint` sorts by mtime not step (FABLE5
  m1).** Low impact (checkpoints are written in step order, so newest-mtime ≈
  highest-step) and only fires on resume. Editing `queue_h100_production.sh` while
  its bash loop is running risks corrupting the auto-resume wrapper, so fix AFTER
  the current run's bash loop exits (or bundle with the C2/C7 de-leak work).
- **2026-06-08 — TO TEST (not assume): E. coli chassis generalization (FABLE5
  C4/C5).** Whether conditioning on an E. coli tag steers output toward E. coli
  statistics for the unseen (E.coli, NRPS/OTHER) cells is an empirical question, not
  a certain failure — Evo2's pretraining knows E. coli. Test via a taxon-swap
  controlled generation experiment (generate class C under E.coli vs source tag,
  compare M7 GC/CAI). Keep a deterministic E. coli recoding step (DNA Chisel) as the
  fallback if conditioning does not generalize.

## Resolution Log

- **2026-06-08 — FABLE5 audit Priority-1 (the novelty gate): C1/M11 + C6/M13 RESOLVED; C3/M8 PARTIAL.**
  (See FABLE5_AUDIT.md for the findings.)
  - **C1/M11** (memorization check missed memorized fragments — ranked by Jaccard,
    reported containment): rewrote `scripts/memorization_check.py` to rank candidates
    by a **containment** estimate (each query's MinHash sketch vs every reference's
    FULL k-mer-hash set, in one streaming pass) and exact-verify the top-m. A
    generated sequence that is a fragment of a longer training BGC now scores
    containment ~1.0 instead of being dropped. Regression test in
    `tests/test_memorization.py` (memorized fragment of a long ref WITH higher-Jaccard
    distractors at top_m=3 → 1.0).
  - **C6/M13** (novelty absent from the scored suite): added `metric_9_novelty` to
    `evaluation.py` and wired it into `evaluate_bgc`'s summary (loop now 1..9) — a
    memorized sequence now **FAILS** the scored suite; a missing scan is `skipped`
    (novelty UNVERIFIED, never a pass). `metric_6` (BiG-SCAPE) marked explicitly as a
    non-functional stub so it cannot masquerade as a novelty metric.
  - **M8** (reference too small): the scan now defaults to the **full grouped corpus**
    (`splits_combined_grouped/train.jsonl`), not the 18K curated subset. Evo2's
    pretraining corpus remains inaccessible — a documented limitation.
  - **C3** (threshold uncalibrated): added `--positive-control` calibration reporting +
    FAIL/WARN/PASS tiers; final threshold calibration still depends on de-leaking the
    splits (C2/C7).
  - Remaining FABLE5 Criticals **C2/C7** (near-duplicate leakage) and **C4/C5** (E. coli
    chassis) are separate priorities, not yet addressed.

- **2026-06-08 — Memorization/novelty check + real-BGC positive control: IMPLEMENTED.**
  - `scripts/memorization_check.py` (audit M4/M6): for each query sequence
    (generated, or the positive control) finds its nearest training BGC by
    **canonical-k-mer MinHash** (strand-agnostic) and reports `max_containment` —
    the exact fraction of the query's k-mers present in that training BGC (the
    interpretable "memorization %"). Training index (per-record sketch + offset)
    is built once and cached. Pure stdlib, no GPU. Core similarity functions
    unit-tested in `tests/test_memorization.py`; demoed end-to-end on the
    positive control (correctly flagged one held-out MiBIG with a near-twin in
    training at 0.955 containment).
  - `scripts/make_positive_control.py`: selects 20 real MiBIG BGCs from the TEST
    split (4/class × NRPS/PKS/RIPP/OTHER/SACCHARIDE, 4–80 kb, diverse phyla),
    verified **0 exact-sequence overlap with train/val**, written to the tracked
    `eval/positive_control_mibig.{fasta,jsonl}`. This calibrates the eval metrics
    ("what real held-out BGCs score") and the memorization baseline. Partially
    addresses audit M5 (a real positive control alongside the base-model baseline).

- **2026-06-04 — C3 (no generation/inference script): IMPLEMENTED.**
  `scripts/generate_bgc.py` loads base Evo2 + the trained LoRA adapter (merged
  into the base weights via `load_evo2_wrapper_for_inference` in
  `evo2_inference.py`), builds the Phase-1 `|COMPOUND_CLASS:{cls}|{tax}` prefix,
  samples with Evo2's efficient cached generation, and trims at the trained
  `|END|` marker (vortex hardcodes `stop_at_eos=False`, so we trim ourselves).
  Supports explicit `--class/--taxon` prompts or sampling held-out prompts via
  `--from-jsonl`; `--max-windows>1` enables chained long-seq generation using the
  `|CONTINUATION:{cls}|{tax}` prefix + carried overlap (audit M11). With
  `--adapter` omitted it generates from the base model — the **M5 generation
  baseline**. Writes FASTA (for the eval suite) + JSONL metadata, with N-content
  reporting (M8 generation side). Post-processing (EOS trim, nucleotide
  sanitation, FASTA, prompt sampling) is unit-tested in `tests/test_generation.py`
  and guards EOS/prefix consistency with training. End-to-end generation needs a
  GPU + a trained checkpoint (still pending the first run). This unblocks the
  generation-dependent evaluation (M4 memorization, M6 novelty, the eval suite,
  and the generation-based validation tier).

- **2026-06-04 — Operational/integrity hardening batch (Groups A + B): RESOLVED.**
  All in `scripts/finetune_evo2_lora.py`; backed by `tests/test_hardening.py` +
  a B1 negative test in `test_chunk_eos_windows.py`.
  - **m3** (emergency checkpoints grew unbounded): `cleanup_old_checkpoints` now
    rotates `step_N_{oom,interrupted,final}` too — keeps newest `--keep-special-ckpts`
    (default 2); `best/` always preserved. Closes the disk-fill loop that C6
    auto-resume opened.
  - **m1** (fingerprint hashed only first 100 lines, never compared): now
    full-file sha256 + size + lines; on resume it COMPARES against the run-start
    fingerprint and warns loudly on mismatch (regenerated split → no longer silent).
  - **m2** (over-claimed determinism): added `use_deterministic_algorithms(True,
    warn_only=True)` — deterministic where kernels allow, one-time warning where
    not (FFT/flash-attn); docs already softened.
  - **B1** (seam invariant unguarded): `__getitem__` now asserts the full
    tokenisation starts with exactly the prefix tokens, raising loudly if a
    tokenizer ever merges across the prefix↔sequence seam (the new EOS/continuation
    prefixes lean on this).
  - **A2** (WandB online, no fallback; unguarded per-step log): online init now
    falls back to offline; all per-step logging goes through `wandb_log_safe`
    (never stalls/kills training).
  - **A3** (DataLoader workers unseeded): added `worker_init_fn` + seeded
    generator for reproducible worker RNG.
  - **A4** (LR horizon could silently shift on resume): persists
    `schedule.json` at run start and warns loudly if a resume recomputes a
    different `total_num_steps` (different data / --max-epochs / --grad-accum).

- **2026-06-04 — Test suite added (partially addresses B8 "no automated test").**
  `tests/run_all.py` runs four GPU-free test files:
  - `test_chunk_spans.py` — `build_nt_chunk_spans` at L=32768: full coverage,
    overlap/stride, exact counts (max 262144 → 9 windows), EOS-budget reserve,
    error guards.
  - `test_chunk_eos_windows.py` — dataset chunking via a byte-level mock
    tokenizer: first→class / interior→continuation prefix, EOS only on the final
    window (supervised), masking, no overflow, first-window-only val, legacy parity.
  - `test_data_pipeline.py` — runs `split_dataset_grouped` + `curate_dataset`
    end-to-end on a 60-record fixture: asserts genome- and exact-sequence-disjoint
    splits, dedup, per-class TRAIN cap, and N/contig-edge/short dropped.
  - `test_eval_metrics.py` — M9 adherence aggregation + balanced sampling, M2
    length buckets + first-window filter, M1 resume-pointer alignment, M7
    early-stop state machine.
  Also fixed `curate_dataset.py` to report the **chunk-window** step estimate
  (long BGCs tile into several windows → ~225 steps/epoch for the curated set,
  not the 143 that records/128 implied). Remaining end-to-end coverage (real
  tokenizer at the 32k boundary; forward/generate) is the GPU smoke run.

- **2026-06-04 — M11 (interior windows mislabeled) + EOS / long-sequence
  groundwork: RESOLVED (training side).** Chunk-mode windowing is now
  window-aware in `BGCTextDataset.__getitem__`:
  - **EOS:** a supervised `EOS_MARKER` (`|END|`) is appended to the *final*
    window of each BGC (nt_end == seq_len) so the model learns to terminate;
    budget reserves room for it (`eos_reserve` threaded through
    `build_nt_chunk_spans`/`build_all_chunk_indices`). It sits after the prefix,
    so it is supervised, not masked.
  - **M11 continuation conditioning:** interior windows (nt_start > 0) now get a
    distinct `|CONTINUATION:{cls}|{tax}` prefix instead of the class-start
    prefix, so `|COMPOUND_CLASS:…|` only ever marks a real cluster start. This
    both resolves M11's conditioning dilution and gives chained long-sequence
    generation a clean continuation mode (the inference/chaining side is C3).
  - Controlled by `--eos-token` / `--continuation-prefix` (default on; legacy
    behaviour via `--no-*`). Production script passes both explicitly.
  - Backed by `tests/test_chunk_eos_windows.py` (real dataset + byte-level mock
    tokenizer): verifies prefix selection, EOS-on-last-window only, supervised
    EOS / masked prefix, no max_seq_len overflow, first-window-only val
    behaviour, and legacy parity. Regression-checked against the M1/M2 tests.
    GPU smoke run still pending (shared GPU busy).

- **2026-06-04 — M9 (no conditioning-adherence validation): IMPLEMENTED.**
  Added `scripts/eval_conditioning_adherence.py` — a generative (likelihood)
  classifier: for each held-out BGC it holds taxonomy+sequence fixed and scores
  the sequence's per-token log-likelihood under every candidate COMPOUND_CLASS
  prefix; if conditioning "took", the true class scores highest. Reports
  top-1/3/5 accuracy, MRR, mean per-token margin, per-class recall, and a
  top-1 confusion summary; `--compare-base` also scores untouched Evo2 (doubles
  as an M5 baseline) and emits the fine-tune delta. Built on a new reusable
  `scripts/evo2_inference.py` (base+adapter loader with the peft-0.19 shims, and
  `sequence_loglik`) that also seeds C3. Pure-logic aggregation unit-tested;
  dry-run validated against the curated val set (25 classes, balanced sampling).
  GPU run deferred until a trained checkpoint + free GPU exist.

- **2026-06-04 — Phase-1 hardening: C5, C6, M1, M10 RESOLVED.**
  - **C5 (unpinned deps):** added a Training-stack section to `requirements.txt`
    pinning `peft==0.19.0`, `deepspeed==0.18.9`, `wandb==0.26.0`, `numpy==2.2.5`
    (previously absent), and committed a full `requirements.lock.txt`
    (`pip freeze`, 178 pkgs) for a reproducible rebuild.
  - **C6 (no auto-resume):** `queue_h100_production.sh` now wraps the launch in a
    retry loop — on a non-zero, non-interrupt exit it locates the newest
    checkpoint (incl. `step_N_oom/_interrupted/_final`), re-waits for GPU idle,
    and relaunches with `--resume-from`, up to `--max-retries` (default 10) with
    `--retry-backoff-sec` backoff. Exit 130 (Ctrl-C) is respected, not retried.
  - **M1 (mid-accumulation resume pointer):** `make_client_state` now snaps
    `micro_step_in_epoch` DOWN to the last grad-accum boundary (passed
    `grad_accum` at all 5 save sites), so OOM/interrupted/final saves resume
    consistently with the optimizer-step counter instead of skipping the
    partially-accumulated micro-batches. Unit-tested.
  - **M10 (false center-crop docs):** corrected STATE_AND_AUDIT §4.1 and
    assumptions A1/A2 — the code does head-truncation (truncate) or forward
    tiling (chunk), never centre-crop; A2 updated for first-window validation.

- **2026-06-02 — M2 (val loss on chunked interior windows) + M7 (no early
  stopping): RESOLVED.** Validation now uses a *first-window-only* dataset
  (`BGCTextDataset(first_window_only=True)`): exactly the prefix-aligned start
  (nt_start==0) of each held-out BGC — the same regime as inference — instead of
  mislabelled interior chunk windows. `run_validation` reports loss stratified by
  full BGC length (`VAL_LENGTH_BOUNDS`), so a length-correlated regression is
  visible (`val_by_length` in val_log.jsonl). Added early stopping
  (`--early-stopping-patience`, `--early-stopping-min-delta`) on the first-window
  val loss; `best/` already holds the best checkpoint and the `finally` block
  exports the final adapter. Production script tuned for the curated set
  (warmup 50, val/save every 50, max-epochs 6 ceiling, patience 4). Verified via
  unit tests (length bucketing, first-window filter = one window/record,
  early-stop state machine) and CLI registration; a GPU smoke run is the
  remaining end-to-end check (deferred — GPU in use). NOTE: this delivers the
  *teacher-forced* half of the two-tier validation; the generation-based tier
  (generate from held-out prompts, score with the eval suite) depends on C3 and
  runs offline on checkpoints, not in-loop.

- **2026-06-02 — C4 (M7 hardcodes E. coli, auto-fails a faithful model): RESOLVED.**
  Rewrote Metric 7 in `src/bgc_pipeline/evaluation.py` to separate two questions
  that were conflated: (a) *faithfulness* to the organism the sequence was
  conditioned on (drives the verdict) and (b) *chassis expressibility* in E. coli
  (reported as an informational sub-score, never gating the verdict).
  Introduced `ReferenceProfile`, `build_profile_from_sequences`,
  `load_taxon_profiles`, and taxon resolution by phylum token; wired
  `EvalConfig.taxon_profiles` / `chassis_profile` and an `expected_taxon` arg to
  `evaluate_bgc`. Crucially, when no taxon profile is supplied the verdict is
  now `no_verdict` (None) instead of `FAIL` — a correct non-E.coli model is no
  longer auto-failed. Added `scripts/build_taxon_profiles.py` to derive empirical
  per-phylum profiles from the clean training data. Verified on a real
  Actinomycetota BGC (GC 0.724): old path FAIL → new default no_verdict → new
  with taxon profile PASS, with E. coli expressibility reported separately.
  *Note: this fixes the metric; the broader "encode the E. coli expressibility
  objective in training (recoding step / E.coli-conditioned generation)" half of
  the recommendation remains future work.*

- **2026-06-02 — C1/C2 (train/val/test leakage): RESOLVED.** Independently
  reproduced from the raw split files (94.6% genome overlap, 453 byte-identical
  sequences leaked). Wrote `scripts/split_dataset_grouped.py`: group-aware split
  keyed on `genome_accession` with global exact-sequence dedup and built-in
  disjointness assertions. New split written to
  `/data2/ds85/bgcmodel_data/splits_combined_grouped/` (train 280,448 / val
  32,317 / test 32,308). Clean-room re-verification: genome / exact-sequence /
  accession cross-split overlap all **0**. Residual species-level overlap ~48%
  is documented and accepted (genome is the chosen grouping floor). The
  production queue script now defaults to the clean split. The leaky split's
  `step_400` checkpoint and `best_val_loss` are obsolete → restart fresh on the
  clean split when the GPU frees up. *Remaining confirmed findings below are
  still open.*

## Summary

This audit raised **24 findings** against the BCGModelling Evo2 BGC fine-tuning project, each cross-examined by two independent adversarial verifiers. Outcome: **17 confirmed** (both lenses agree the issue is real), **6 disputed** (lenses disagree on severity or impact), and **1 refuted** (rejected on verification). Among confirmed findings there are **6 Critical** and **8 Major** (plus 3 Minor). The dominant themes are scientific-validity failures — pervasive train/val/test leakage with no grouped splitting, a completely missing generation/inference stage, an evaluation suite that grades against the wrong chassis and never measures novelty/conditioning — alongside operational reproducibility gaps (unpinned core dependencies, no auto-resume on a contended GPU). Several of these invalidate the project's headline deliverable and should be resolved before trusting any train/val/test comparison or committing wet-lab synthesis budget.

---

## Confirmed Findings

### Critical

#### C1 — Train/val/test leakage: 23–24% of val/test records are overlapping fragments of the same antiSMASH region present in train [CRITICAL / HIGH]
- **Category:** Train/test leakage
- **Location:** `scripts/split_dataset.py:64-89`; `data/processed/splits_combined/{train,val,test}.jsonl`
- **What is wrong:** The split groups records only by `compound_class` and shuffles per class — no grouping by `accession` or `genome_accession`. Verified on real data: 8,226/34,655 val (23.7%) and 8,094/34,655 test (23.4%) share an `accession` with train; 4,226 shared-accession val records physically overlap a train genomic window; ~95% of val/test share a `genome_accession` with train.
- **Why it matters:** Val/test loss is computed partly on nucleotide windows the model has effectively trained on. This deflates val loss, corrupts best-checkpoint/early-stopping selection, and invalidates any generalization claim — the core scientific deliverable.
- **How to verify:** Build the train `accession`/`genome_accession` sets and check val/test membership; for shared accessions, test interval overlap of `region_start`/`region_end`.
- **Suggested fix:** Re-split with group-aware assignment keyed at minimum on `genome_accession` (ideally ANI/species clusters) so all fragments/regions of a genome land in one split; assert cross-split disjointness; regenerate `splits_combined` and re-run prior val-based decisions.
- **Verifier note:** One lens corrected the stated mechanism (`accession` is reused across distinct sequences, and the coordinate frames are inconsistent, so the coordinate-overlap evidence is weaker than claimed), but both agree the leakage is real and Critical via exact md5 collisions and ~95% genome overlap.

#### C2 — Train/val/test split has no leakage control; exact sequences appear in both train and val [CRITICAL / HIGH]
- **Category:** Data integrity / evaluation validity
- **Location:** `scripts/split_dataset.py:74-94`; heldout consumer only at `scripts/antismash_db_to_jsonl.py:403`
- **What is wrong:** Plain per-class `rng.shuffle` then slice; no grouping by accession/genome/sequence hash and no near-duplicate clustering. The `heldout_accessions.txt` mechanism only protects the historical MIBiG-only split, not the active `splits_combined/`. Full scans found **544 byte-identical train↔(val/test) sequences**, 15.3% shared accession, 75.2% shared genome; val and test also overlap each other.
- **Why it matters:** `best_val_loss` checkpoint selection (`finetune_evo2_lora.py:1910-1933`) and reported `val_ppl` ride on a leaked signal; downstream memorisation/novelty flags are contaminated; the headline result is not reproducible.
- **How to verify:** md5 the val/test sequences and stream train counting collisions; count shared accession/genome across splits; `grep -rn heldout scripts/` to confirm the combined pipeline never reads it.
- **Suggested fix:** Group-aware splitting (assign whole `genome_accession`/sequence-cluster to one split), drop exact md5 duplicates pre-split, and assert zero cross-split sequence/genome overlap before training.
- **Verifier note:** Both lenses confirmed empirically; this is the same root defect as C1 viewed through the leakage-control lens.

#### C3 — No generation/inference script exists — the pipeline cannot produce the sequences it is built to produce [CRITICAL / HIGH]
- **Category:** Missing pipeline stage
- **Location:** `scripts/` (no `generate_*.py`/`sample_*.py`/`infer_*.py`); only doc-level example at `STATE_AND_AUDIT.md:718-735`
- **What is wrong:** Training saves a LoRA adapter but never samples (`grep '.generate('` finds only `generate_plots()` and validation). `evaluate_bgc.py` only consumes pre-existing FASTA/JSONL. Nothing loads base Evo2 + adapter, applies the `|COMPOUND_CLASS:..|{tax}` prefix, and autoregressively generates a BGC. Decoding choices (temperature, top-p, max_length, EOS/stop, N emission) are undefined.
- **Why it matters:** Every downstream goal (8-metric eval, wet-lab validation) depends on generated sequences. This is a whole missing stage between training and evaluation.
- **How to verify:** `ls scripts/ | grep -iE 'gen|sample|infer'` returns nothing; confirm `evaluate_bgc.py` never produces sequences.
- **Suggested fix:** Add `scripts/generate_bgc.py` loading Evo2 + `PeftModel.from_pretrained(best/adapter)`, building the byte-identical Phase-1 prefix, exposing decoding flags, handling stopping, and writing FASTA consumable by `evaluate_bgc.py`; pin a seed and log decoding params.

#### C4 — Chassis-compatibility metric (M7) hardcodes E. coli, but the model reproduces the source organism's codon/GC statistics [CRITICAL / HIGH]
- **Category:** Risky assumption / evaluation validity
- **Location:** `src/bgc_pipeline/evaluation.py:625,637,643,652-667`; conditioning at `scripts/finetune_evo2_lora.py:386-390`
- **What is wrong:** Conditioning uses the source organism's lineage (97.6% bacteria, Actinomycetota GC median 0.712, overall 0.604). M7 scores every sequence against E. coli K-12 (`target_gc=0.508`, `gc_pass` deviation <0.10, CAI vs E. coli >0.7, E. coli dinucleotide RMSD) with no taxon parameterization. A correctly-trained Streptomyces-conditioned model emits ~0.71 GC and fails `gc_pass`/`cai_pass`. No chassis-rewrite/codon-optimization objective exists in training.
- **Why it matters:** M7 reports near-universal failure for a genuinely good model (risking rejection of a working model), and the "synthesis-ready E. coli BGC" objective is encoded nowhere in the loss — only assumed at eval.
- **How to verify:** Compute training GC by taxon vs `target_gc=0.508`; read `evaluation.py:620-705` (all E. coli-referenced); grep finetune for any codon/chassis rewrite (none).
- **Suggested fix:** Parameterize M7's references by the conditioned taxon, OR split the objective into "faithful generation" (graded vs source taxon) and "E. coli expressibility" (explicit recoding/codon-optimization step or E. coli-conditioned generation with sufficient examples).
- **Verifier note:** One lens argued for downgrade to Major (M7 is Tier-2 with no aggregate gate, applied symmetrically to real/generated sequences); the other confirmed Critical. Net status is **confirmed**, but note the no-gate/secondary-tier caveat when prioritizing.

#### C5 — Load-bearing packages (peft, deepspeed, wandb, torch, flash-attn) are not pinned in any installable manifest [CRITICAL / HIGH]
- **Category:** Version fragility / environment reproducibility
- **Location:** `requirements.txt` (peft/deepspeed/wandb absent; torch & flash-attn commented out); `environment.yml` (no peft/deepspeed/wandb; torch/flash-attn pinned at lines 219,279)
- **What is wrong:** `grep -in 'peft|deepspeed|wandb' requirements.txt environment.yml` returns nothing. The documented build (`conda env create` + `pip install -r requirements.txt`) installs none of the three. The code depends on `peft==0.19.0` with three manual compatibility shims (`finetune_evo2_lora.py:1010, 1029, 1525-1551`) and `deepspeed==0.18.9`; exact versions exist only as prose.
- **Why it matters:** A clean rebuild resolves arbitrary versions → LoRA application or DeepSpeed init breaks, or runs with subtly different optimizer/adapter behavior invalidating the comparison. Central reproducibility hazard for a multi-day single-shot run.
- **How to verify:** Run the grep above (empty); in a clean env `pip show peft deepspeed` (missing or wrong version).
- **Suggested fix:** Add hard pins (`peft==0.19.0`, `deepspeed==0.18.9`, `wandb==0.26.0`) and uncomment `torch`/`flash-attn` into an enforced manifest; commit a `pip freeze` lockfile; have queue scripts assert versions before launch.
- **Verifier note:** One lens argued Major (the dominant failure is a loud ImportError on rebuild, not silent corruption; the current in-flight env is intact). Both confirm the gap is real and unmitigated.

#### C6 — Production run has no auto-resume on mid-run preemption/OOM [CRITICAL / HIGH]
- **Category:** Run survival / GPU-contention recovery
- **Location:** `scripts/queue_h100_production.sh:316-337`
- **What is wrong:** `wait_for_gpu_idle` runs once; the launch is a single invocation with no retry/resume loop; on non-zero exit the script logs and `exit 1`. The OOM handler (`finetune_evo2_lora.py:1979-2014`) writes `step_N_oom` then re-raises, so the orchestrator gives up. Working resume machinery exists but is only invoked by a fresh human re-invocation.
- **Why it matters:** On a ~2.7-day shared-GPU run, at least one preemption is likely; without auto-resume the run sits dead until a human notices, wasting GPU-days and requiring 24/7 babysitting.
- **How to verify:** Read `queue_h100_production.sh:321-337` (no `while`/`until`; terminal `exit 1`); compare with the OOM handler that writes `step_N_oom` and re-raises.
- **Suggested fix:** Wrap the launch in a bounded retry loop: on non-zero exit re-run `wait_for_gpu_idle`, locate the newest checkpoint (incl. `step_N_oom`), relaunch with `--resume-from`; add backoff and a max-retry/deadline cap.
- **Verifier note:** One lens argued Major (operational only — emergency checkpoint + faithful resume preserve correctness; lost work bounded to ~one save interval; the "2.7-day high-preemption" premise leans on a possibly-stale doc). Both confirm the mechanism.

### Major

#### M1 — OOM/final/interrupted checkpoints persist a mid-accumulation micro_step that misaligns with the saved optimizer step on resume [MAJOR / HIGH]
- **Category:** Mid-epoch resume faithfulness
- **Location:** `scripts/finetune_evo2_lora.py:1990-1999` (OOM), `2016-2028` (finally), resume skip-ahead `1788-1805`
- **What is wrong:** Periodic/best/interrupted saves occur inside the accumulation-boundary guard, so their `micro_step_in_epoch` is boundary-aligned. The OOM handler and `finally` block save outside that guard, persisting a raw mid-window `micro_step` while the saved optimizer `step` is the last completed boundary. On resume, skip-ahead advances `micro_step+1` micro-batches but the optimizer resumes from the prior boundary — the in-flight partial window's micro-batches are skipped in the data stream yet never contributed an optimizer update.
- **Why it matters:** Each OOM/exception-triggered restart silently drops up to `grad_accum-1` (=127) unique BGC windows, non-randomly (always those preceding an OOM), breaking the claimed resume==continuous (H1) invariant.
- **How to verify:** Force OOM/SIGTERM at a non-boundary `micro_step`, inspect saved client_state (`micro_step_in_epoch % grad_accum != 0` while `step` is the last boundary), resume and observe non-boundary skip count.
- **Suggested fix:** In OOM/finally paths, clamp persisted `micro_step` down to the last completed boundary (`((micro_step+1)//grad_accum)*grad_accum`), or reuse the last boundary's saved micro count; add an assertion that the saved value is a `grad_accum` multiple.
- **Verifier note:** Scope is narrower than stated — graceful/signal (`_interrupted`) restarts go through the guarded save and are faithful; only the OOM handler and exception-driven `finally` are affected.

#### M2 — Validation loss is computed on chunked interior windows, not full BGCs — it does not measure generation quality [MAJOR / HIGH]
- **Category:** Validation representativeness
- **Location:** `scripts/finetune_evo2_lora.py:744-822, 600-641, ~1665-1680`
- **What is wrong:** Val uses the same chunk strategy as train; long records are sliced into overlapping interior windows with the prefix re-prepended. Val CE is next-token accuracy on arbitrary mid-gene/mid-codon fragments. Loss is also window-weighted, over-weighting long classes (~1.48x; PKS/NRPS/hybrid up to 2.13x). No generation-based held-out metric exists.
- **Why it matters:** Val loss can look excellent while the model fails to start a BGC from its 5' end, build coherent architecture, or terminate — so model selection/early-stopping ride on a proxy that does not reflect the end goal.
- **How to verify:** Print `nt_start` of val windows (most long-class windows >0); confirm `run_validation` just averages CE; note no generation eval is invoked.
- **Suggested fix:** Add a generation-oriented held-out eval (teacher-forced full-sequence loss from position 0, and/or generate + M1/M2). At minimum restrict val to first-window-only (`nt_start==0`) or weight by record not window.

#### M3 — Train-vs-inference conditioning mismatch: model learns prefix → arbitrary interior fragment, but inference asks prefix → BGC from position 0 [MAJOR / HIGH]
- **Category:** Train/inference mismatch
- **Location:** `scripts/finetune_evo2_lora.py:764-774, 631-640`
- **What is wrong:** Chunk mode prepends the identical conditioning prefix to every interior window. For long classes (NRPS/PKS/hybrid/siderophore/arylpolyene), >50% of training windows begin mid-sequence, teaching the model that the prefix can precede interior DNA. Inference prompts the bare prefix expecting a BGC from its 5' start.
- **Why it matters:** The conditioning prefix no longer reliably means "emit a BGC from its start," threatening coherent full-BGC generation precisely for the drug-relevant long classes.
- **How to verify:** Tally `nt_start>0` fraction per class (long classes dominated by interior windows); generate from a long-class prefix and check for mid-gene vs cluster-start output.
- **Suggested fix:** Add a window-position/continuation token to interior windows, or restrict the class-conditioning prefix to `nt_start==0` windows and treat interiors as continuation-conditioned.
- **Verifier note:** Signal is diluted, not inverted — every record also yields one `nt_start=0` window and short classes are almost all `nt_start=0` — which keeps it Major rather than Critical.

#### M4 — No implemented metric distinguishes genuine generation from regurgitation of a training BGC [MAJOR / HIGH]
- **Category:** Memorization / novelty not measured
- **Location:** `src/bgc_pipeline/evaluation.py:563-616` (M6 stub), `738-812` (M8 vs UniRef50)
- **What is wrong:** M6 (BiG-SCAPE) runs the tool but never parses distances (stub note at 613-614) → no novelty verdict. M8's `memorisation_flag` is protein identity >95% vs UniRef50 (generic DB, default `None` → skipped) — expected for real enzymes, says nothing about copying a training BGC. There is no comparison of generated sequences against the project's own training corpus anywhere.
- **Why it matters:** With substantial intra-corpus leakage and 2-epoch training, a model that regurgitates a (possibly leaked) training BGC passes M1/M2/M5/M7 and is reported as success — the "novel BGC" claim is unfalsifiable as instrumented.
- **How to verify:** Read `evaluation.py:563-616` (no distance) and `800-812` (UniRef50 only); grep for any comparison to `train.jsonl` (none).
- **Suggested fix:** Implement a generated-vs-train nucleotide-identity check (minimap2/BLASTn/MMseqs2 over the BGC interval, flag >95% and a softer >80% band); finish M6 distance parsing; do this on a leak-free split.

#### M5 — No model-level baseline (untuned Evo2 generation, shuffled-label, or class-mismatch) — only a shuffled-nucleotide input control [MAJOR / MEDIUM→HIGH]
- **Category:** Missing baseline / control
- **Location:** `scripts/evaluate_bgc.py:67-70,160-164`; `scripts/eval_smoke.py:46-48,188-192`; `src/bgc_pipeline/evaluation.py:517-556`
- **What is wrong:** The only controls are nucleotide-shuffles of the sequence under test and base-Evo2 perplexity scoring (not generation). No untuned-base generation baseline, no shuffled-label/class-mismatch control. M1 passes when `expected_class in mapped` — nothing checks a different label yields a different class.
- **Why it matters:** Without a base-model baseline you cannot attribute M1/M2/M7 success to fine-tuning rather than Evo2's pretrained grammar; without a class-mismatch control you cannot show COMPOUND_CLASS is causal. Undermines the "conditioned generation" claim.
- **How to verify:** Grep for base-Evo2 generation or label permutation (none); read M1 pass condition.
- **Suggested fix:** Add three arms — base-Evo2 generations from identical prefixes; class-swap control (condition A, measure antiSMASH-called class); conditioning-dropped/shuffled-label negative — and report fine-tuned-vs-base deltas as the primary result. Note: a coarse class-frequency baseline exists in docs (random ~25%) but does not isolate fine-tuning or conditioning causality.

#### M6 — No nucleotide-level memorization/novelty check, despite being the plan's defining safety criterion [MAJOR / HIGH]
- **Category:** Missing evaluation / novelty characterization
- **Location:** `src/bgc_pipeline/evaluation.py:738-812` (M8 protein-level), `563-616` (M6 stub); criterion at `docs/gputee/BGC_Research_Plan.md:119`
- **What is wrong:** The plan requires flagging >95% nucleotide identity to a training example. No code implements NT-identity-vs-training-set. M8 is protein pident vs UniRef50 (default `None`); M6 is a stub. The evaluate summary only PASS/FAILs metrics with a `pass` key (M1/M2/M7), so M6/M8 never gate anything.
- **Why it matters:** A memorized/regurgitated training BGC passes the entire implemented suite and is never flagged — the central safety gate before committing synthesis budget is absent (this is the M4 gap stated as the explicit plan criterion).
- **How to verify:** `grep -n 'nucleotide' src/bgc_pipeline/evaluation.py` (none); M8 `db_path` default `None`/UniRef50; M6 stub note.
- **Suggested fix:** Implement nucleotide novelty (MMseqs2/minimap2/megablast vs `train.jsonl`, report max %identity + aligned fraction, flag >95%); finish or de-PASS-eligible M6.

#### M7 — No early-stopping or cross-epoch overfitting guard; training always runs the full epoch budget [MAJOR / HIGH]
- **Category:** Missing training safeguard
- **Location:** `scripts/finetune_evo2_lora.py:1777, 1893-1933, 1953`
- **What is wrong:** The epoch loop runs `max_epochs` unconditionally. Val improvement saves `best/` but there is no patience counter or stop-on-non-improvement. Only `--max-steps` (smoke) and SIGTERM exit early. `final_adapter/` (the documented inference artifact, exported in the `finally` block) is the **last-step** adapter, not `best/`.
- **Why it matters:** On a ~2.7-day run, training past the val minimum wastes shared-GPU days and the documented load path yields a likely-overfit model (rare classes have 2–14 records). Only manual use of `best/` mitigates.
- **How to verify:** `grep -n 'patience\|early'` (nothing); trace the epoch loop (no break on non-improving val); confirm `final_adapter/` copies the final step.
- **Suggested fix:** Add `--early-stop-patience`/`--early-stop-min-delta`, break on exceeded patience, log train-vs-val gap to WandB, and either point `final_adapter/` at `best/` or document `best/` as the inference target.

#### M8 — No N-character (ambiguous base) handling between training and synthesis-ready output [MAJOR / HIGH]
- **Category:** Missing data/output handling
- **Location:** `scripts/finetune_evo2_lora.py` (no ACGT/N normalization); `src/bgc_pipeline/evaluation.py:430-433` (M4 hard-fails non-ACGT)
- **What is wrong:** ~4–5% of training records contain N; the training path passes text straight to the tokenizer (N is an in-vocab token), so the model learns to emit N. `metric_4_synthesis_feasibility` immediately returns `pass=False` for any non-`^[ACGT]+$`. No N-repair, rejection, or resampling exists between generation and synthesis.
- **Why it matters:** Synthesis vendors reject ambiguous bases; the model is primed to produce N and the synthesis metric hard-fails those sequences with no automated remediation — an unhandled end-to-end blocker (reduced usable yield).
- **How to verify:** Grep the data path for N/ACGT normalization (none); read `evaluation.py:430`; confirm no resample-on-N logic.
- **Suggested fix:** Decide a policy: filter/mask N in training (distinguish scattered single-N from long assembly-gap runs), and/or add a post-generation N-rejection-and-resample loop before the synthesis metric; add an N-count field to eval output.
- **Verifier note:** One lens flagged that the "GC math silently treats N" sub-claim is inaccurate (M4 returns before GC is computed) and that M4 does reliably catch N — so this is a yield/handling gap, not silent shipment of bad sequences.

#### M9 — No conditioning-adherence validation — nothing checks the model responds to the class/taxon prompt [MAJOR / MEDIUM→HIGH]
- **Category:** Missing evaluation
- **Location:** `src/bgc_pipeline/evaluation.py` / `scripts/evaluate_bgc.py` (no controlled-conditioning experiment)
- **What is wrong:** Each sequence is scored against its own expected class; nothing holds taxon fixed and varies COMPOUND_CLASS, and there is no scrambled/empty-prefix negative control. The only "negative control" is a nucleotide shuffle (orthogonal — never touches the prefix). No generation harness exists to run such a test.
- **Why it matters:** M1 class-match can be high purely from the data-majority prior (TERPENE+RIPP ≈ 49%), giving a false signal that conditioned generation works — the core claim gating wet-lab spend.
- **How to verify:** Read `evaluate_bgc.py` (per-record independent loop, no cross-condition comparison); grep for adherence/negative-conditioning tests (none).
- **Suggested fix:** Add a conditioning-adherence harness: fixed taxon, N candidates per class with matched seeds, M1/M7, report a requested-vs-recovered confusion matrix plus a scrambled-prefix control; success requires the diagonal to beat the majority-class prior.

#### M10 — Documented "center-crop preserves core biosynthetic genes" assumption is false — code does head-truncation or forward tiling [MAJOR / HIGH]
- **Category:** Risky assumption / data integrity
- **Location:** `scripts/finetune_evo2_lora.py:800-801` (truncate head-clip), `631-641` (chunk tiles from 0); `STATE_AND_AUDIT.md:511-512` and A1
- **What is wrong:** No centering exists. Truncate mode keeps the head and drops the tail (`ids[:max_seq_len]`); chunk mode tiles forward from position 0 (covers all nucleotides but is not centered). The pilot (and the script default) is truncate, so long drug-relevant classes (NRPS/PKS, 90–99% exceed 32k) are trained on 5' ends only — and val is cropped identically, masking the bias.
- **Why it matters:** The false "centering" belief hides that the pilot/default head-crops long BGCs; any "real" run forgetting `--long-seq-strategy chunk` silently head-clips ~34% of records while val loss looks fine.
- **How to verify:** `grep -n 'center\|centre\|crop\|mid'` (nothing); read line 801 and `build_nt_chunk_spans`; confirm pilot lacks the chunk flag while production passes it.
- **Suggested fix:** Correct the docs (truncate=head-clip, chunk=forward-tile; neither centers); make `truncate` error or actually center; run the pilot with production flags so its numbers match.
- **Verifier note:** One lens stressed that production uses chunk mode (loses no genes), so the live production path is unaffected — residual severity is a documentation defect plus a truncate-default footgun.

#### M11 — Chunk mode labels gene-internal windows with the full class+taxon prefix [MAJOR / HIGH]
- **Category:** Risky assumption / label noise
- **Location:** `scripts/finetune_evo2_lora.py:768-774`, `644-658`
- **What is wrong:** Every window of a long record gets the same `|COMPOUND_CLASS:..|{tax}` prefix. Interior windows of long megasynthases (>50% of NRPS/PKS/hybrid windows) carry a class label their local sequence does not justify (~1.48x window multiplier, driven by long classes).
- **Why it matters:** Dilutes/miscalibrates the class conditioning exactly for the most important long classes; conditioning on NRPS may produce arbitrary interior coding DNA rather than a complete cluster. Val is labeled identically so it won't catch it.
- **How to verify:** Reproduce the per-class internal-window fractions; inspect `train_log` `first_nt_start` (many >0 with the same class prefix).
- **Suggested fix:** Drop/generically-tag the class+taxon prefix on non-first windows, restrict class conditioning to `nt_start==0` windows, or down-weight interior-window loss.
- **Verifier note:** Prefix tokens are loss-masked and interior windows hold class-correct coding DNA, so this is calibration dilution (Major), not a learned false mapping.

### Minor

#### m1 — `data_fingerprint.json` hashes only the first 100 lines of the 18 GB / 277K-line training file [MINOR / HIGH]
- **Category:** Data fingerprint / corruption detection
- **Location:** `scripts/finetune_evo2_lora.py:308-319`
- **What is wrong:** The "tamper detection" fingerprint digests only the first 100 lines (`sha256_first_100`); any change past line 100 (99.96% of the file) yields an identical fingerprint. It is written at run start and never read back/compared on resume.
- **Why it matters:** False provenance confidence — if splits are regenerated between an initial run and resume, the model silently trains on a different distribution with no warning.
- **How to verify:** Read lines 312-316 (counts all, hashes `i<100`); `grep -rn data_fingerprint scripts/` (only writer); modify a line >100 and rerun (fingerprint unchanged).
- **Suggested fix:** Hash the full file (streamed) or a deterministic stripe plus size+mtime as `sha256_full`; recompute and compare on resume, failing loudly on mismatch.
- **Verifier note:** One lens argued the practical impact is below Major (partially caught by the actively-checked `.lengths.meta.json` size/mtime sidecar) and that the doc overclaims "tamper detection."

#### m2 — "Faithful"/bit-exact resume is not achievable: deterministic algorithms not fully enabled and FFT/attention kernels are nondeterministic [MINOR / MEDIUM]
- **Category:** Determinism / reproducibility claims
- **Location:** `scripts/finetune_evo2_lora.py:272-278`
- **What is wrong:** `seed_everything` sets `cudnn.deterministic`/`benchmark=False` but never calls `torch.use_deterministic_algorithms(True)`; StripedHyena FFT long-convs + flash-attention use nondeterministic CUDA kernels regardless. Docs imply bit-identical continuation; floating-point results will diverge run-to-run.
- **Why it matters:** The achievable property (identical data stream + RNG on resume) is valuable, but "identical to an uninterrupted run" oversells it; a reviewer cannot bit-reproduce a checkpoint.
- **How to verify:** Read 277-278 (no `use_deterministic_algorithms`); run the same config twice with seed 42 and diff `train_log.jsonl` losses.
- **Suggested fix:** Soften the docs to "reproducible data stream + optimizer state," and/or attempt `use_deterministic_algorithms(True, warn_only=True)` documenting which ops remain nondeterministic.

#### m3 — OOM/interrupted checkpoints are exempt from rotation and grow disk unbounded [MINOR / MEDIUM]
- **Category:** Disk-space growth from checkpoints
- **Location:** `scripts/finetune_evo2_lora.py:1386-1413` (cleanup skips non-`step_<digits>` dirs), `1948` (cleanup only on periodic saves)
- **What is wrong:** `cleanup_old_checkpoints` keeps `keep_last_ckpts=5` numeric step dirs but preserves forever `step_N_oom`/`step_N_interrupted`/`step_N_final`. On a contended GPU each OOM leaves a permanent checkpoint (DeepSpeed optimizer state + adapter), accumulating with no cap.
- **Why it matters:** If `/data2` fills, subsequent saves (including `final_adapter`) fail. Conditional on many preemptions; ample headroom today.
- **How to verify:** Read 1406-1410 (only pure-digit dirs deletable); cleanup only called at 1948; simulate repeated OOM-resume cycles.
- **Suggested fix:** Cap retained oom/interrupted checkpoints (keep newest 1–2); log cumulative checkpoint disk usage and warn on low `/data2` free space; store oom checkpoints adapter-only after a successful resume consumes them.
- **Verifier note:** Checkpoints are LoRA-only (`exclude_frozen_parameters=True`) — ~220 MB per oom, ~439 MB per final; current total ~13.7 GB vs 1.5 TB free, so urgency is low.

---

## Disputed Findings (verifiers disagreed)

- **Prefix token count recomputed independently of the concatenated tokenization** — `finetune_evo2_lora.py:776-784,843/880` — *Major* — One lens: real latent off-by-k risk, one cheap assert from Critical-prevention. Other: refuted — with the byte-level CharLevelTokenizer a seam merge is mathematically impossible, so current impact is zero (hardening only).
- **Smoke/pilot memory headroom does not generalize to production (truncate vs chunk path)** — `queue_h100_pilot.sh:223-237` vs `queue_h100_production.sh:296-297` — *Major* — One lens: real gate gap, OOM risk somewhat lower than stated (bs=1 already hits 32k). Other: refuted — full-budget L=32k memory (~44 GB peak) already measured by padded/production-like sweeps independent of the pilot; downgrade to Minor.
- **WandB defaults to online with no offline fallback; unguarded per-step `.log`** — `queue_h100_production.sh:72`, `finetune_evo2_lora.py:1876-1904` — *Major* — One lens: confirmed (Major-leaning-Minor); a blocking flush could stall the loop but `final_adapter` is protected by the `finally` ordering. Other: refuted — wandb 0.26 `log()` is async/non-blocking and `finish()` runs after the export; effective severity Minor.
- **Gradient-accumulation averaging is mean-of-means, not token-weighted** — `finetune_evo2_lora.py:1821,1823,898-906` — *Minor* — One lens: confirmed Minor (real persistent skew toward short tail-chunks). Other: refuted — it's the universal framework default and the "chunk-tail bias" framing is unsupported (up-weighted population is short complete BGCs); impact effectively nil.
- **Cosine LR scheduler `total_num_steps` differs between pilot and production resume** — `finetune_evo2_lora.py:969-978,1710-1711` — *Minor* — One lens: confirmed Minor latent inconsistency (but the `--max-steps` root cause is mis-stated; real driver is truncate-vs-chunk dataset length; negligible today inside warmup). Other: refuted — `--max-steps` does not feed the horizon and both runs share `max_epochs=2`, so no discontinuity.
- **DataLoader `num_workers=2` with no worker seeding; worker RNG outside restored state** — `finetune_evo2_lora.py:1701-1708,1124-1182` — *Minor* — One lens: confirmed Minor latent robustness gap (and a slightly overstated H1 docstring). Other: refuted — data stream is deterministic via seeded `DistributedSampler` + explicit skip-ahead; harm is contingent on a speculative future augmentation.

---

## Refuted Findings (rejected on verification)

- **`micro_step` stale/undefined when resume skip-ahead exhausts the epoch iterator, corrupting the next resume pointer** (`finetune_evo2_lora.py:1766,1788-1807`) — *Rejected.* The stale value exists but is provably never persisted: the skip loop `break`s leaving ≥1 item (drop_last-aware `len`), so the inner loop always runs ≥1 body; and the `finally` write is gated by `if step > start_step`, which can only become true after a loop body advances `global_steps`. The two preconditions are mutually exclusive.

---

## Recommended Priority Actions

Fix before resuming training or trusting any results:

1. **Re-split with group-aware partitioning (C1, C2).** Assign whole `genome_accession` (ideally sequence-identity clusters) to a single split, drop exact md5 duplicates, and assert zero cross-split sequence/genome/accession overlap. Regenerate `splits_combined` and discard any val-based checkpoint/early-stopping decisions made on the leaked split.
2. **Pin the environment (C5).** Add `peft==0.19.0`, `deepspeed==0.18.9`, `wandb==0.26.0` and `torch`/`flash-attn` to an installable manifest; commit a lockfile; assert versions before launch.
3. **Add an auto-resume retry loop to production (C6)** and fix the OOM/`finally` resume-pointer misalignment (M1) so restarts neither abandon the run nor silently drop up to 127 training windows.
4. **Decide and encode the chassis contract (C4).** Either parameterize M7 by the conditioned taxon or add an explicit E. coli recoding step / E. coli-conditioned generation; do not let a faithful model be auto-failed.
5. **Build the missing generation stage (C3)** with documented decoding params and byte-identical prefix formatting — nothing downstream is testable without it.
6. **Implement memorization/novelty and conditioning-adherence evaluation (M4, M6, M9, M5)** before committing wet-lab budget: nucleotide identity vs `train.jsonl` (flag >95%), finished M6 distances, a class-swap/scrambled-prefix control, and a base-Evo2 generation baseline reported as deltas.
7. **Add generation-representative validation + early stopping (M2, M7)** and point the documented inference artifact at `best/`; handle interior-window conditioning (M3, M11) and N emission (M8); correct the center-crop docs and align the pilot to production flags (M10).
8. **Harden provenance/ops (m1, m3, m2):** full-file fingerprint with resume-time comparison, checkpoint rotation cap for oom/interrupted dirs, and softened determinism claims.
