# BCGModelling — CLAUDE.md

This file gives AI coding agents project-specific context so they can work
correctly without repeated setup explanations.

## Purpose

- Keep edits aligned with this repo's actual workflow and constraints.
- Prioritize reproducible, documented decisions over speculative refactors.
- Treat `README.md` (consolidated current state) and `docs/project_memory/`
  (decisions / bugs / progress) as the primary project-memory sources. Deep
  runbooks and dated audits are archived under `docs/archive/`.

## Memory Protocol (READ / WRITE — required)

Working memory lives in `docs/project_memory/`:

- `progress.md` — exact current state of the research + next actions
- `decisions.md` — architecture/approach decisions and **why**
- `bugs.md` — quirks, recurring errors, and the proven fixes

**Before starting a task:** read `docs/project_memory/progress.md` first (and skim
`decisions.md` / `bugs.md` when the task touches modelling, data, or evaluation) so you
resume from the real state instead of re-deriving it.

**After solving a major bug, making a structural/architecture decision, or at the end of a
work session:** you MUST update the relevant file(s) in `docs/project_memory/` to reflect
the new state — append the bug+fix to `bugs.md`, the decision+rationale to `decisions.md`,
and refresh `progress.md` (state + next actions + the "Last updated" date). Keep entries
concise and dated. `README.md` is the consolidated current-state overview — keep it
consistent whenever behavior, flags, data, or decisions change.

## Repository Layout (reorganized 2026-07-27)

Two model tracks, one shared instrument:

- **Root = SHARED, model-agnostic.** `scripts/` (dataset pipeline + eval drivers),
  `src/bgc_pipeline/` (eval suite), `config/`, `tests/`, `data/`, `eval/`,
  `docs/project_memory/`.
- **`evo2/`** — Evo2-specific: `scripts/` (trainer, generation, queue wrappers,
  `quick_eval.sh`/`run_eval.sh`), `experiments/{probes,quartz}/`, `docs/`.
- **`genomeocean/`** — GenomeOcean-specific: `scripts/`, `experiments/`,
  `external/` (upstream clone, gitignored).

Shell wrappers under `evo2/scripts/` are still run **from the repo root**
(`evo2/scripts/queue_h100_smoke.sh`). Python scripts under `evo2/scripts/` anchor the
repo root at `Path(__file__).resolve().parents[2]`; shared ones under `scripts/` use
`parents[1]`. Tests at `tests/` add **both** `scripts/` and `evo2/scripts/` to
`sys.path`.

## Project Snapshot

- Goal: fine-tune a genome foundation model for BGC sequence generation/evaluation.
- Current production host focus: `gputee` (single H100 80 GB).
- **Evo2 track:** LoRA adapters on Evo2 7B (not full-parameter FT); DeepSpeed + PEFT +
  PyTorch (bf16); env `bgcmodel` (torch 2.5.1+cu124, transformers 4.46.3). This env also
  carries **antiSMASH 8.0.4 + Pfam**, so both tracks run the eval suite here.
- **GenomeOcean track (opened 2026-07-27):** GenomeOcean-4B / `bgcFM`, a stock
  `MistralForCausalLM`; env at `/data2/ds85/envs/genomeocean` (torch 2.11.0**+cu128**,
  transformers 5.14.1, peft 0.19.1). Invoke with
  `micromamba run -p /data2/ds85/envs/genomeocean python ...`. See
  `docs/model_comparison_evo2_vs_genomeocean.md`.
- Datasets (on `/data2`, see Current Decisions for why):
  - ACTIVE training/eval (v2): `/data2/ds85/bgcmodel_data/splits_core/{train,val,test}.jsonl`
    — strict antiSMASH **core** regions, native lowercase GTDB tags, leakage-clean
    (genome-disjoint + exact + cross-split MMseqs2); train 47,524 / val 8,048 / test 18,871;
    22 classes; **MiBIG held out** (reserved for a Phase-2 compound-conditioned FT).
  - DEPRECATED (superseded — do not use): `splits_curated/` (~18K), `splits_combined_grouped/`,
    `splits_dedup/`, and `data/processed/splits_combined/` (leaky — 94.6% genome overlap).

## Source of Truth

- Consolidated current state:
  `README.md`
- Live status + next actions:
  `docs/project_memory/progress.md`
- Evo2 vs GenomeOcean head-to-head (measured, not quoted):
  `docs/model_comparison_evo2_vs_genomeocean.md`
- Per-track entry points:
  `evo2/README.md`, `genomeocean/README.md`
- Multi-GPU long-context run on IU Quartz (setup + execution guide):
  `evo2/docs/quartz_setup.md`
- Eval suite implementation:
  `src/bgc_pipeline/evaluation.py`
- Training implementation:
  `evo2/scripts/finetune_evo2_lora.py`
- Smoke queue wrapper:
  `evo2/scripts/queue_h100_smoke.sh`
- Archived deep runbook / status (not maintained as current):
  `docs/archive/gputee/FINETUNE_GUIDE.md`, `docs/archive/gputee/PROJECT_GUIDE.md`

## Current Decisions (as of latest smoke sweeps)

- Block-level activation checkpointing is implemented and default-on in
  `evo2/scripts/finetune_evo2_lora.py` (opt out with
  `--no-activation-checkpointing`).
- Queue smoke runs default to padded train collation via
  `--smoke-pad-to-max-seq-len` so measured memory reflects requested `L`.
- No-checkpoint path is not viable above short contexts on H100.
- With checkpointing + padded sweeps:
  - `L=32768` passes with large margin **at smoke batch=1 grad-accum=1**.
  - `L=65536` passes but is near limit **at smoke batch=1 grad-accum=1**.
  - `L=98304` OOMs.
- Conservative default remains `L=32768`; `L=65536` is stretch/conditional.
- **Micro-batch shape at L=32k (audit 2026-05-14):** `--batch-size 4
  --grad-accum 32` (the original "match-trojai-128-effective-batch" plan)
  **OOMs** on this 80 GB H100 — `bs=4` fails on forward, `bs=2` fails on
  backward. The only configuration that fits at L=32,768 is
  `--batch-size 1 --grad-accum 128`. Effective batch is still 128
  sequences; the LoRA hyperparams remain valid. Use this shape for every
  L=32k pilot/production run. The wall-clock estimate in
  `FINETUNE_GUIDE.md` §4 is being re-derived from the running pilot's
  `tokens_per_sec`; the prior `3,275 tok/s` figure was measured at the
  no-longer-feasible bs=4 ga=32 shape and is treated as an
  optimistic lower bound only.
- **Long sequences:** production training uses `--long-seq-strategy chunk
  --chunk-overlap 2048` (deterministic tiling; full nucleotide coverage; canonical
  prefix from JSON fields; default `--auto-prefix-budget` scans `max_prefix_tokens`
  into sidecar meta). Sidecars: `<dataset-dir>/<split>.lengths.npy` + `.meta.json`
  (e.g. under `splits_core/`); pre-build lengths with `python evo2/scripts/build_chunk_index.py`.
  The L=32k **pilot** keeps default `--long-seq-strategy truncate` for continuity
  with earlier smoke metrics (`FINETUNE_GUIDE.md` §3, `PROJECT_GUIDE.md` §13).
- **Loss masking (H3, audit 2026-05-14):** the CE loss is masked over
  the prefix tokens — `labels[:, :prefix_token_count] = IGNORE_INDEX`.
  Only the BGC sequence half of each example contributes to the loss,
  which matches the project's "generate sequences conditioned on a
  fixed prefix" intent. Absolute train/val loss values are *not*
  comparable to pre-H3 runs.

### Data & validation decisions (2026-06-02, post-audit — see docs/archive/AUDIT_FINDINGS.md)

- The original `splits_combined/` split was record-level and leaked badly
  (94.6% genome overlap, 453 byte-identical seqs across splits). Fixed with
  group-aware (genome-keyed) splitting: `scripts/split_dataset_grouped.py`.
- Training set curated down to ~18K via `scripts/curate_dataset.py`:
  quality-filtered (no N / no contig-edge), per-class capped at 1000,
  diversity-stratified (phylum × length, distinct genomes); val/test kept full.
  Rationale: Evo2 is pretrained; Phase-1 mostly teaches the conditioning
  interface (LIMA-style), and the full set was prohibitively large (~20 d/epoch).
- Validation uses first-window-only (prefix-aligned) loss, length-stratified
  (`val_by_length`), with early stopping (`--early-stopping-patience`). The old
  interior-window val loss did not reflect generation. Generation-based eval is
  offline (depends on the not-yet-built generation script).
- Metric 7 (organism compatibility) no longer hardcodes E. coli: it grades
  faithfulness vs the conditioned taxon (`scripts/build_taxon_profiles.py`,
  `data/processed/taxon_profiles.json`) and reports E. coli expressibility
  separately. (Now the `taxon_faithfulness` check — see the 2026-06-17 note.)

### Eval suite rewrite + v2 data (2026-06-17 — see docs/archive/REDESIGN_PLAN.md / docs/archive/EVAL_RUNBOOK.md)

- The eval suite was rewritten from the flat `metric_1..metric_11` numbering to two
  named layers: **CHECKS** (`coding_sanity`, `antismash`, `class_markers`,
  `kmer_novelty`, `protein_homology`, `module_architecture`, `taxon_faithfulness`;
  optional `protein_foldability`) combined into **QUESTIONS** via `derive_questions()`.
  GATES = `is_bgc`, `correct_class`, `novel`; diagnostics = `proteins_plausible`,
  `complete`, `conditioning_faithful`. (`src/bgc_pipeline/evaluation.py`.)
- **antiSMASH is the gold-standard `is_bgc`/`correct_class` gate**, recalibrated
  ~0.15 → ~0.97 on real cores by completing the product→class map
  (`scripts/build_class_map.py` → `config/compound_class_map.yaml`). `class_markers`
  (Pfam) is the fast proxy when antiSMASH is skipped (quick-eval).
- **Gene caller: pyrodigal (Prodigal)** everywhere; replaced the six-frame ORF finder.
  RETIRED: synthesis feasibility, Evo2 perplexity, BiG-SCAPE; E. coli expressibility
  no longer gates.
- Active data is `splits_core` (above); the ~18K `splits_curated` curation is superseded
  by the strict-core rebuild (56K → 47.5K after MiBIG exclusion).
- Per-checkpoint tracking: `evo2/scripts/quick_eval.sh` (runs the cheap checks incl. antiSMASH;
  skips `protein_homology` + `kmer_novelty`). Full eval: `evo2/scripts/run_eval.sh`.
- **Adaptive seq-budget slack (H6, audit 2026-05-14):** chunk windows
  reserve `prefix_token_cap + prefix_slack_tokens` tokens before
  filling with nucleotides. Slack is empirically scanned by rank 0 and
  persisted in `<split>.lengths.meta.json` (0 under the current
  CharLevelTokenizer). On overflow the trainer now raises a
  `ValueError` instead of silently clipping the tail.
- **Faithful mid-epoch resume (H1, audit 2026-05-14):** checkpoints
  now record `epoch`, `micro_step_in_epoch`, and the Python / NumPy /
  torch-CPU / torch-CUDA RNG snapshots in `client_state`. On resume
  the trainer skips forward `micro_step_in_epoch` items in the
  shuffled `DistributedSampler` and restores RNG, so dropout / data
  order match an uninterrupted run from the same checkpoint. Legacy
  pre-H1 checkpoints still resume on `step` + `best_val_loss` only.

## Common Commands

Environment:

```bash
cd ~/projects/BCGModelling
micromamba activate bgcmodel
export HF_HOME=/data2/ds85/hf_cache
```

Shared-GPU-safe smoke matrix (default lengths):

```bash
evo2/scripts/queue_h100_smoke.sh
```

Long-context probe:

```bash
evo2/scripts/queue_h100_smoke.sh --lengths "49152 65536 98304"
```

Disable padded smoke collation (diagnostic only):

```bash
evo2/scripts/queue_h100_smoke.sh --no-smoke-pad-to-max-seq-len
```

Single run sanity:

```bash
deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \
  --train /data2/ds85/bgcmodel_data/splits_core/val.jsonl \
  --val   /data2/ds85/bgcmodel_data/splits_core/val.jsonl \
  --output-dir /data2/ds85/bgcmodel_runs/ac_sanity \
  --max-seq-len 1024 --batch-size 1 --grad-accum 1 \
  --warmup-steps 2 --max-epochs 1 --max-steps 3 \
  --log-every 1 --val-every 99 --save-every 99 \
  --wandb-mode offline
```

## Working Conventions for Agents

- Make targeted edits; avoid broad cleanup unless requested.
- Update docs in the same change when behavior/flags/decisions change.
- Do not claim memory conclusions from long-L sweeps unless
  `train_log.jsonl` confirms `collated_seq_len == L`.
- For benchmark interpretation, check both:
  - `summary.tsv` (status/peaks)
  - per-length `smoke_L*.log` (actual failure signatures).
- Prefer explicit CLI flags over hidden behavior changes.

## Validation Expectations

- After code edits:
  - run syntax checks for touched scripts;
  - run lint checks where available;
  - verify new CLI flags appear in `--help`.
- After smoke runs:
  - verify `summary.tsv`;
  - inspect `train_log.jsonl` for `gpu_mem_gb`, `collated_seq_len`,
    `content_max_len`, and (post-H3) `first_chunk_idx` /
    `first_prefix_token_count` to confirm chunk windowing + prefix
    masking are doing what they should;
  - inspect failing `smoke_L*.log` for exact OOM site.

## Known Gotchas

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reports unsupported on
  this platform; do not rely on it as an OOM fix.
- NCCL "process group not destroyed" warning on shutdown is expected in these
  short smoke runs.
- Shared host contention can invalidate measurements; use queued runs with
  idle-gpu gating.
