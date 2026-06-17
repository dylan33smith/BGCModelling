# Progress — current state of the research

**Read this first when resuming work.** It records the exact state at the last
checkpoint of activity. Update it at the end of a session or after a major change.
See [decisions.md](decisions.md) (the why) and [bugs.md](bugs.md) (quirks/fixes).

_Last updated: 2026-06-17._

---

## What is running right now

- **v2 LoRA training — IN PROGRESS.**
  - Run dir: `/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768`
  - tmux session: `bgc_v2` (idle-GPU-gated, checkpoints, auto-resume up to 10 retries)
  - Config: Evo2 7B + LoRA, `L=32768`, `--batch-size 1 --grad-accum 128`, bf16, DeepSpeed.
  - Health (last seen): ~step 30, train_loss ~0.95 (down from 0.98), GPU 100%, single
    worker, no restarts. Log every 10 steps; **first checkpoint at step 50**.
  - Data: `splits_core` (train 47,524 / val 8,048 / test 18,871; 22 classes; strict
    antiSMASH cores; native GTDB tags; MiBIG held out).

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

1. **At v2 step 50 (first checkpoint):** run `scripts/quick_eval.sh <run-dir>` — it now
   reports the real `is_bgc`/`correct_class` (antiSMASH) plus the graded
   `obligate_fraction`, coding/module signals. Watch whether `correct_class` lifts off the
   floor (the conditioning is being learned).
2. **Settle the Step 2 architecture fork** (task #9, the one open task): re-run
   `scripts/diagnose_conditioning_stochastic.sh` + quick_eval on a v2 checkpoint to decide
   **per-class adapters vs one conditional model** (see [decisions.md](decisions.md)).
3. **If v2 conditions well:** start **Phase-2** — build a MiBIG core + compound-conditioned
   dataset and fine-tune for compound-level (named-product) generation.

## Known not-yet-done / deferred

- `protein_homology` (MMseqs2) DB is **not wired** for full-val — diagnostic-only; skipped
  in quick_eval. Wire a UniRef50 DB when running a full milestone eval.
- Generation-based offline eval depends on `generate_bgc.py` (built; sequential path).
- Nothing is committed to git — commit only when explicitly asked.

## Pointers

- Eval suite + how to run: `README.md` → Evaluation; archived deep version
  `docs/archive/EVAL_RUNBOOK.md` and `docs/archive/REDESIGN_PLAN.md`.
- Training runbook: `docs/archive/gputee/FINETUNE_GUIDE.md`.
- Auto-memory (cross-session, outside the repo):
  `~/.claude/projects/-home-ds85-projects-BCGModelling/memory/`.
