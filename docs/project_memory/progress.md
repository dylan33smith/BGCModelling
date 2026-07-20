# Progress — current state of the research

**Read this first when resuming work.** It records the exact state at the last
checkpoint of activity. Update it at the end of a session or after a major change.
See [decisions.md](decisions.md) (the why) and [bugs.md](bugs.md) (quirks/fixes).

_Last updated: 2026-07-03._

---

## What is running right now

- **Nothing computing.** All probes complete; every single-GPU-cheap lever tested and negative
  (see "What just finished"). **Current thread (2026-07-13+): stage the multi-GPU LONG-CONTEXT
  run on IU Quartz** — the one untested version of the whole-cluster hypothesis (train all mega
  cores whole at L up to 262144 on a 4× H100 node). Setup + execution guide:
  **`docs/quartz_setup.md`**; helper scripts in `experiments/quartz/`. Blocked on an RT Project
  allocation (PI-granted) for the Slurm `-A` account; env/data prep can proceed on the login node.
  The strategic fork is real: long-context (this) **vs** repositioning Evo2 as an evaluator/scorer.

### (completed 2026-07-07) Fast capability-probe chain (launched 2026-07-06)

- **Fast capability-probe chain (launched 2026-07-06).** Tests the diagnosis fixes at reduced
  cost (L=16384, bs=1 ga=16, ~350 steps fresh-from-base, `lora_dropout=0`), each isolating one
  variable vs a shared P0 control, then quick_eval. Runner:
  `scratchpad/run_probe_chain.sh`; outputs under `/data2/ds85/bgcmodel_runs/probes_20260706/`
  (`probe_summary.tsv`). Probes: **P0** control · **B** +`projections.weight` (adapts the
  frozen Hyena input projection — 28.7M→35.8M trainable, validated) · **C** whole-core data
  (no chunking) · **D** mega-upweighted data. Read on the SENSITIVE proxies
  (class_markers / obligate_fraction / any_domain_rate) since 350 steps is deliberately
  undertrained. ~1 day sequential on the one GPU.
- The main run stays **STOPPED at step_1200** (23 checkpoints retained); resumable via
  `queue_h100_production.sh --resume-from <run-dir>` if a probe warrants scaling up.

### Code/data added for the probes (2026-07-06)
- `finetune_evo2_lora.py`: new `--lora-target-parameters` (peft 0.19 `target_parameters`,
  needs `lora_dropout=0` — peft's ParamWrapper forbids dropout).
- `quick_eval.sh`: `TEMPERATURE`/`TOP_K`/`TOP_P` + `MAX_WINDOWS`/`CHUNK_OVERLAP` env passthroughs.
- `scripts/build_probe_subsets.py` → `/data2/ds85/bgcmodel_data/probe_subsets/`
  (`subset_c_wholecore` 5,821 recs; `subset_d_megaup` 18,235 recs, 53% mega) + sidecars.

### Superseded plan (continuous-resume 2026-06-24 — did NOT deliver functional gains; kept for history)

- **v2 LoRA training — CONTINUOUS RESUME (started 2026-06-24 14:30 UTC).** tmux `bgc_v2`.
  - **Resumed from `checkpoints/step_400`** (best_val_loss 0.8179) into the same run dir
    `phase1_lora_prod_20260617_095202_L32768`; faithful H1 resume (RNG + data order restored).
  - **Early stopping DISABLED** (`--early-stopping-patience 0`) and **all checkpoints kept**
    (`--keep-last-ckpts 0`) — per the under-training hypothesis from the 2026-06-24 eval.
  - Target **6 epochs = 2,478 steps** (413 steps/epoch); ~52 h/epoch → ~11–13 days from
    step 400. Shape: `L=32768`, `bs=1 ga=128`, splits_core.
  - **LR schedule flattened in the later epochs:** `--lr-min-ratio 0.5` (was 0.1), so the
    cosine floor is 2.5e-5 instead of 5e-6 — later epochs keep a meaningful LR
    (~2.5–3e-5 by epoch 5) instead of decaying to near-zero. Horizon unchanged
    (`total_num_steps=2478`), so the restored schedule stays aligned. Rationale: the first
    pass plateaued in val loss at step 400 *while LR was still ~peak*, so we want later-epoch
    learning to actually have step size. **Watch `val_by_length` for overfitting** (flat/high
    LR can keep train loss dropping while val stalls); optional short cosine cooldown from the
    best long-run checkpoint can produce the final model.
  - Launched via `scripts/queue_h100_production.sh` (idle-GPU gated + auto-resume ×10). The
    launcher gained `--keep-last-ckpts` and `--lr-min-ratio` passthroughs (2026-06-24).

- **Milestone quick-eval watcher — running (tmux `bgc_eval`).** `scripts/eval_milestones_watch.sh`
  (new, 2026-06-24) watches `checkpoints/` and runs `quick_eval` on each `step_N` milestone
  (default stride 200 + newest), appending one row per checkpoint to
  `<run-dir>/quick_eval_milestones/eval_track.jsonl` (step → is_bgc / correct_class /
  class_markers / any_domain_rate / coding_density). **Single-GPU-safe / post-hoc:** it is
  idle-gated (proc=0, free≥70 GB, 300 s hold), so it never competes with training — the sweep
  runs once the GPU frees (training end or a long gap). View sorted:
  `jq -s 'sort_by(.step)[] | {step,is_bgc,correct_class,class_markers}' <eval-root>/eval_track.jsonl`.
  - Checkpoint cadence `save-every 50` (≈439 MB each incl. optimizer state); /data2 has 1.4 TB
    free so retaining the full trajectory (~50 ckpts/6 epochs ≈ 22 GB) is a non-issue.

## What just finished

- **Rank sweep (r=16/64/128, mega_all, n=15) — capacity is NOT the limiter (2026-07-13).**
  correct_class: r=16 0.067 · r=64 0.067 · r=128 **0.0** — flat/worse; no rank lifts the gate. r=64
  bumped DOMAIN markers (class_markers 0.133→**0.267**, obligate 0.072→0.108) but same modules
  (0.133) and floor correct_class; r=128 collapsed (correct_class 0, modules 0 — over-rank / α–r=2
  over-shrink, rsLoRA regime). Same signature: capacity nudges DOMAINS, never the correct-class
  CLUSTER. **LoRA capacity now fully closed** (coverage via probe B + rank via this sweep). Runner:
  `scratchpad/run_ranksweep.sh`; rows `rank64_n15`/`rank128_n15` in probes_20260706/probe_summary.tsv.

- **Option A (real mega-only whole-core run, L=32768) — AUTO-KILLED at epoch 4; whole-core does NOT
  lift correct_class (2026-07-12).** Milestone n=15: step 120 (~ep2) is_bgc 0.267 / correct_class
  0.133 / modules 0.200 — a flicker; step 240 (~ep4) **is_bgc 0.133 / correct_class 0.0 / modules
  0.0** — everything DECLINED with more training. Self-gate fired (correct_class 0 < 0.15). So
  whole-core mega training at feasible single-GPU L does not convert to functional correct-class
  BGCs; likely overfitting the small whole-core set (80 Mbp — whole-core@L=32k drops 62% of mega nt
  / the long cores). Runner: `scratchpad/run_optA.sh`; run dir `bgcmodel_runs/mega_whole_32k_run`.

- **Concentration probe + n=15 re-eval — C's correct_class win was n=6 NOISE (2026-07-10).** At
  n=15 (PER_CLASS=5): **correct_class = 0.067 (1/15) for ALL of P0, mega_all, and C** — tied at the
  floor; C's earlier 0.33 (2/6) did not survive. What DOES survive is a domain-level gradient
  **C > mega_all > P0**: class_markers 0.33 / 0.13 / 0.07, obligate_frac 0.147 / 0.072 / 0.044,
  module_count 0.27 / 0.13 / 0.07. So whole megasynthase cores (C) and, to a lesser degree,
  mega-only concentration (mega_all > P0) make the model place ~3–5× more class-appropriate
  obligate domains / partial modules — but **none converts into an antiSMASH-valid correct-class
  cluster.** The fast 350-step fresh-from-base probes are exhausted; no config lifts the functional
  gate at reliable n. Next real test = a multi-epoch mega-only whole-core run (mind the
  whole-core∩feasible-L tension — see decisions.md).

- **Gene-aware chunking A/B — REFUTED the "de-chunking is the lever" reading (2026-07-09).**
  Two 200-step arms on the same 17,450 long-mega strict-core dataset (`ga_blind` arithmetic vs
  `ga_geneaware` snap-to-gene): **gene-aware did NOT help — it did worse.** `ga_blind`
  class_markers 0.333 / obligate 0.104 / module 0.167; `ga_geneaware` **all 0** (coding density
  also dropped 0.85→0.76). Neither reached correct_class. So snapping cuts to gene boundaries
  does not recover C's benefit. **Reinterpretation:** C's advantage is seeing the complete
  *cluster* (fits whole), not complete *genes* — points at **longer context (larger L)**, not
  smarter chunking. **Big caveat (see decisions.md):** at the production L=32768, ~79% of mega
  cores ALREADY fit whole and the full run still failed → long-context alone may not be the fix;
  C also confounds mega-only × whole × short. n=6/arm, 200 steps — weak screen. Implementation
  (`--gene-aware-chunking`, gene-bounds, build scripts) is retained and correct; the *hypothesis*
  is what's refuted.

- **Fast capability-probe sweep — DE-CHUNKING is the lever (2026-07-07).** Four 350-step
  fresh-from-base probes at L=16384, ga=16 (runner `scratchpad/run_probe_chain.sh`; results
  `/data2/ds85/bgcmodel_runs/probes_20260706/probe_summary.tsv`), each vs a shared P0 control:
  - **P0** control: correct_class 0, module_count 0.
  - **B** (+`projections.weight`, unfreezes the frozen Hyena long-range input projection):
    **identical to P0 (all 0)** → LoRA capacity/coverage is NOT the bottleneck at this scale;
    **overturns the diagnosis's #1 "leading suspect".**
  - **C** (whole-core / de-chunked megasynthase data): **correct_class 0.33, class_markers 0.50,
    obligate_fraction 0.18, module_count 0.17** — the ONLY probe to lift the functional gates,
    and the first thing in the project to produce correct-class BGCs with ordered modules +
    real obligate domains.
  - **D** (megasynthase upweighted to 53%, still full-length/chunked): **correct_class 0,
    module_count 0** (only a 0.06 obligate flicker) → more mega data does NOT help if chunked.
  - **Verdict:** the lever is the **training signal (whole-core / de-chunking)**, NOT model
    capacity (B flat) and NOT class concentration (D flat). The model must see the **complete
    assembly line under its class label**. Reorders the diagnosis: chunking (Lane 2, rated
    "contributing") is primary; LoRA capacity (Lane 5, "leading suspect") is unsupported.
  - Caveats: n=6/probe, 350 undertrained steps; C confounds whole-core × mega-only × shorter
    (≤16k). Clean isolation = gene-aware chunking (long mega cores chunked-well vs whole).
  - **Follow-up P-tag (2026-07-08):** re-ran D's data with `--no-continuation-prefix` (constant
    `|COMPOUND_CLASS|` on every chunk + `|END|`, no `|CONTINUATION|`) → **near-identical to D**
    (class_markers 0.167 & obligate 0.056 *literally the same*; correct_class/module still 0).
    So the continuation **TAG is NOT the culprit — it's the FRAGMENTATION.** Relabeling is a dead
    end; gene-aware chunking (chunks that contain complete genes) is the fix. Diluted test (only
    ~28% of subset_d chunks); a 100%-chunked long-mega subset would fully confirm.

- **step_1200 functional eval — the decisive negative (2026-07-03).** Pooled n=21 across two
  decoding temps (artifacts under the run dir: `quick_eval_step1200/`,
  `quick_eval_step1200_confirm_baseline/` [temp 1.0, n=9], `quick_eval_step1200_confirm_lowtemp/`
  [temp 0.7, n=6]):
  - `is_bgc` ≈ **3/21 (14%)** — NOT exactly 0 (the first n=6 caught 0/6 = small-sample noise);
    `correct_class` = **0/21**; `module_count` = **0/21**; obligate core domains
    (PF00501 NRPS-A / PF00668 C / PKS KS/AT) ≈ absent.
  - **Smoking gun:** every antiSMASH-positive hit was a SIMPLE class — requested NRPS→**ectoine**,
    requested HYBRID→**terpene** — never the conditioned megasynthase. Class-conditioning fails
    functionally: the model writes generic gene-dense DNA that occasionally forms an easy cluster
    but never builds the requested class's core assembly-line machinery.
  - **Robust:** more n didn't lift `correct_class`; conservative decoding (temp 0.7) gave cleaner
    coding density (0.98) but STILL 0 modules / 0 correct_class → not a sampling artifact. With flat
    val loss and no step-400→1200 gain, the gap is **structural, not under-training** — this
    challenges the "surface results = low training, not LoRA capacity" note in [decisions.md](decisions.md).
  - **Decision: STOP the 6-epoch run; diagnose the root cause** (workflow running).
  - Training was paused cleanly for this (wrapper SIGTERM'd, trainer SIGINT'd; step_1200 intact).

- **v2 LoRA training (first pass) — COMPLETE** (finished 2026-06-19 12:09 UTC; ~50 h wall).
  - Run dir: `/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768`
  - Config: Evo2 7B + LoRA, `L=32768`, `--batch-size 1 --grad-accum 128`, bf16, DeepSpeed.
  - **Early stop at step 400** (epoch ~0.97/6): no val improvement for 4 validations
    (patience=4, min-delta 0.001). Train loss 0.98 → 0.71; **best val_loss 0.8179** (ppl 2.27).
  - Checkpoints: `checkpoints/best/adapter`, `final_adapter/` (= step_400_final), plus
    step_{200,250,300,350,400}. GPU now idle; no tmux session.
  - Data: `splits_core` (train 47,524 / val 8,048 / test 18,871; 22 classes; strict
    antiSMASH cores; native GTDB tags; MiBIG held out).

- **Post-training eval on the v2 `best` checkpoint — done 2026-06-24** (GPU was free).
  Artifacts under the run dir: `quick_eval_best/`, `conditioning_diag_stoch_best/`,
  `post_train_eval.log`.
  - **quick_eval (n=3, 32k, antiSMASH gates):** `is_bgc=0.0`, `correct_class=0.0`,
    `class_markers=0.0`, `obligate_fraction=0.0`, `any_domain_rate=0.333`,
    `coding_density=0.913`, `module_count=0`. → produces coding-dense DNA with occasional
    Pfam domains, but **antiSMASH does not call any of the 3 as a BGC** and none are
    correct-class. (Tiny n; directional, not definitive.)
  - **conditioning diag (stochastic top_k=4, 24 seqs @16k):** composition 5-mer
    within 0.229 / cross 0.247 / **ratio 1.08**; domain-set ratio 1.02; own-obligate by
    class NRPS 0.056, PKS 0.0, HYBRID 0.021, TERPENE 0.0; any-domain 0.67–0.83;
    GC 0.62–0.71 (healthy, not degenerate). Script **VERDICT: "CONDITIONING WORKS"**
    (class-differentiated + NRPS shows some of its obligate domains).
  - **Honest read:** a real improvement over the 2026-06-04 pilot (which scored
    "CONDITIONING DEAD", ratio ≈1.0). v2 shows a **measurable, class-appropriate but WEAK**
    conditioning effect — yet it is **not yet producing antiSMASH-recognizable,
    correctly-classified BGCs** (functional gates at 0). The class tag is being read; the
    model is not yet building complete class machinery/modules.

## What is done and validated

- **Dataset v2 (`splits_core`) built & leakage-clean** — strict cores from re-acquired
  antiSMASH-DB GBKs, native lowercase GTDB tags, genome-disjoint + exact + MMseqs2-dedup,
  MiBIG excluded. Pre-MiBIG backup at `splits_core_premibig/`.
- **Eval suite rewritten** to named CHECKS → QUESTIONS (`src/bgc_pipeline/evaluation.py`).
  Gene caller is **pyrodigal** everywhere; synthesis/perplexity/BiG-SCAPE retired; E. coli
  expressibility no longer gates. All `tests/run_all.py` pass (8 files).
- **antiSMASH recalibrated** to the `is_bgc`/`correct_class` gate: **0.97 / 0.97** on 237
  real held-out cores (was ~0.15). Full product→class map in
  `config/compound_class_map.yaml` (via `build_class_map.py`). Calibration data:
  `/data2/ds85/bgcmodel_data/as_calib.jsonl`.
- **`class_markers` recalibrated** (data-driven, ANY-of) — ≈0.87 on real cores.
- **Eval consumers migrated** (driver, quick_eval, run_eval, diagnose scripts, standalone
  evaluate_bgc, tests). Adversarial review workflow run; its 1 high bug (antiSMASH
  parse-error proxy) + comment/coverage findings fixed.
- **Documentation consolidated** into this `README.md` + `docs/project_memory/`; detailed
  runbooks/plans/audits archived under `docs/archive/`.

## Next actions (in order)

0. **[2026-07-07] Build gene-aware chunking / whole-core training — the validated lever.** The
   probe sweep showed de-chunking (whole cores) is what lifts the functional gates, not LoRA
   capacity (B) or class concentration (D). Plan:
   - **(i)** Persist per-gene coordinates: `build_core_records.py` already parses per-CDS
     coords (`cds_coords`) but only stores the count — add a `core_gene_bounds` field (rel
     offsets within the stored sequence). GBKs on disk at `/data2/ds85/asdb5_gbks`. Either
     re-run the core→split→dedup pipeline or emit an `accession→bounds` sidecar (non-destructive).
   - **(ii)** Snap chunk cuts to gene gaps in `build_nt_chunk_spans` (+ the duplicate in
     `build_chunk_index.py`) so no cut falls inside a gene; fall back to arithmetic when a single
     gene exceeds the budget (those need larger `L` / 7B long context).
   - **(iii)** Probe it: long-mega chunked-well vs whole-core (isolate de-chunking from the
     mega-only/short confound), then a real (not 350-step) run on 7B for publishable numbers.
   - Optional cleanup to test alongside: constant class tag every chunk + explicit START/END
     tokens (vs the current `|CONTINUATION|`) — see the 2026-07-07 chunking discussion.
   The earlier diagnosis-driven experiment list below is now **superseded by these results**.

1. **[SUPERSEDED — see item 0] Act on the diagnosis of the class-conditioning failure.**
   - **(a) [cheap control] Re-eval step_1200 with chained windows** (`generate_bgc.py
     --max-windows 3-4 --chunk-overlap 2048`) to remove the 32k-truncation confound. Expected:
     still 0 correct_class (deficit is upstream) — decisive and cheap.
   - **(b) [high-value ablation] LoRA coverage + rank.** Add the Hyena `projections` (TELinear)
     to LoRA targets (relax the isinstance check ~`finetune_evo2_lora.py:1200`; verify PEFT can
     wrap TELinear), bump rank 16→32/64 with a `rank_pattern` favouring mixer/attention over
     MLPs, optionally unfreeze prefix/class tokens (`trainable_token_indices`). Few-hundred-step
     controlled ablation watching `module_count`/`correct_class`.
   - **(c) [chunking ablation]** fine-tune on only whole-core-in-one-window NRPS/PKS/HYBRID; if
     modules emerge, chunking is the dominant lever. Structural fix: gene/module-aware splitting
     or `L=65536` for megasynthase classes; and/or label interior windows `|COMPOUND_CLASS|`.
   - **(d) [amplifier]** class-balanced sampling.
   The "train longer / raise LR" item below is **superseded**.

1. **Decide what the weak-conditioning + zero-functional-gate result means.** The class tag
   is read (ratio 1.08, class-appropriate domains) but no antiSMASH-valid BGC is produced.
   Likely levers, roughly in order of expected payoff:
   - **Train longer / harder.** Early stop fired at epoch ~0.97 (well under 1 full epoch);
     the prefix-masked loss was still drifting down. Consider loosening early stopping
     (higher patience / smaller min-delta) or raising LR, then re-eval. The conditioning
     interface may simply be under-trained (consistent with the LoRA "low-training ⇒
     surface results" note in [decisions.md](decisions.md)).
   - **Larger / less-tiny eval.** Re-run quick_eval with more sequences per class
     (`PER_CLASS`>1) so `is_bgc`/`correct_class` aren't estimated from n=3.
   - Only after the above: reconsider the **per-class-adapters vs one-conditional-model**
     fork (see [decisions.md](decisions.md) "Open architectural fork").
2. **If/when v2 conditions well** (functional gates lift off 0): start **Phase-2** — build a
   MiBIG core + compound-conditioned dataset and fine-tune for compound-level (named-product)
   generation.

_Done 2026-06-24: quick_eval + stochastic conditioning diagnostic on the v2 `best`
checkpoint (results above). Earlier "step 50" eval action is moot — the run already
completed at step 400._

## Known not-yet-done / deferred

- `protein_homology` (MMseqs2) DB is **not wired** for full-val — diagnostic-only; skipped
  in quick_eval. Wire a UniRef50 DB when running a full milestone eval.
- Generation-based offline eval depends on `generate_bgc.py` (built; sequential path).
- All work through 2026-06-17 is **committed and pushed to `main`** (commit `d337184`); the
  working branch `claude/laughing-hamilton-fdacc5` is synced to `main`. (Commit only when
  explicitly asked.)

## Pointers

- Eval suite + how to run: `README.md` → Evaluation; archived deep version
  `docs/archive/EVAL_RUNBOOK.md` and `docs/archive/REDESIGN_PLAN.md`.
- Training runbook: `docs/archive/gputee/FINETUNE_GUIDE.md`.
- Evo2 + LoRA + Hyena-block architecture explainer (why the long-range pathway is un-trained;
  what Probe B fixes): `docs/evo2_lora_and_hyena.md`.
- Auto-memory (cross-session, outside the repo):
  `~/.claude/projects/-home-ds85-projects-BCGModelling/memory/`.
