# Progress — current state of the research

**Read this first when resuming work.** It records the exact state at the last
checkpoint of activity. Update it at the end of a session or after a major change.
See [decisions.md](decisions.md) (the why) and [bugs.md](bugs.md) (quirks/fixes).

_Last updated: 2026-06-24._

---

## What is running right now

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
- Auto-memory (cross-session, outside the repo):
  `~/.claude/projects/-home-ds85-projects-BCGModelling/memory/`.
