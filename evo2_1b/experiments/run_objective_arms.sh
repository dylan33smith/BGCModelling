#!/usr/bin/env bash
# PHASE-2 OBJECTIVE ARMS on the 1B. Run from the repo root.
#
# Three arms, identical in every respect except the objective:
#   baseline   --domain-weight 1.0 --frame-lambda 0.0   (bit-identical to causal_lm_loss)
#   frame      --frame-lambda 0.5                       (in-gene stop-completion penalty)
#   weighted   --domain-weight 3.0                      (per-record-normalised domain weights)
#
# CHUNKED, NOT TRUNCATED. The default `truncate` shows the model only the first ~4 kb of every
# core: 25.2% of the training DNA, and — worse for this experiment — a slice that is 49.0%
# class-domain against 33.7% for whole cores. Truncation therefore hands the domain-weighted arm
# LESS linker to down-weight than the real distribution has, biasing the test against the very
# intervention it is testing. It also pins nt_start at 0, so the window-offset path in
# annotations.py is never exercised by a real run.
#
# WHY L=8192 AND NOT 4096. 8192 is the 1B's native context. At 4096 the sequence budget is ~3,897
# nt after the prefix, and ONE NRPS module is 3,000-4,500 nt — the window could barely hold the
# thing the frame-aware arm is trying to teach the model to sustain. "The window could not hold a
# module" would be a poor reason for a null. Measured cost: ~3.3x the 7B's tokens/sec at L=4096, so
# the extra length is affordable.
#
# WHY BLIND CHUNKING, GIVEN ITS HISTORY. The probe sweep found gene-aware boundaries no better than
# blind (n=6, on metrics that read ~0), while confirming blind chunking never shows 36% of large
# megasynthase genes complete. At L=8192 with stride 7,168 a blind boundary cuts 13.1% of genes.
# The alternative — training only on records that fit whole — is cleaner but uses 14.7% of the DNA
# and drops the long cores, i.e. exactly the assembly-line classes the ORF question is about.
# Chunking is common-mode across all three arms, so it can add noise but cannot manufacture a
# difference between them. Overlap is 1024 (not 512) to give each window more run-up before the
# frame penalty applies, which softens the one real concern: a window starting mid-gene penalises
# the model for frame it cannot yet infer from its visible context.
#
# NOT the full 2x2 yet: the arms want different sequence lengths (frame-aware is
# length-agnostic; domain weighting is least meaningful at short context because short cores are
# already 78.6% domain), so the single arms run first and the interaction cell only if one of them
# moves. Scored on best_bio_bits with novelty as a hard constraint — NOT on max_orf_aa, which does
# not track domain content de novo (r = 0.051 / -0.120).
set -euo pipefail
export HF_HOME=/data2/ds85/hf_cache
export EVO2_BASE_MODEL=evo2_1b_base

DATA=/data2/ds85/bgcmodel_data/splits_core
ANN=$DATA/train.domain_spans.jsonl
ROOT=${ROOT:-/data2/ds85/bgcmodel_runs/phase2_1b}
STEPS=${STEPS:-400}
L=${L:-8192}   # the 1B's NATIVE context; 4096 could not hold one module

run_arm () {
  local name="$1"; shift
  local out="$ROOT/$name"
  if [[ -f "$out/final_adapter/adapter_config.json" ]]; then
    echo "[arms] $name already finished — skipping"; return 0
  fi
  echo "[arms] $(date) START $name  ($*)"
  micromamba run -n bgcmodel deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \
    --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" \
    --output-dir "$out" \
    --max-seq-len "$L" --batch-size 1 --grad-accum 16 \
    --long-seq-strategy chunk --chunk-overlap 1024 \
    --warmup-steps 20 --max-epochs 1 --max-steps "$STEPS" \
    --log-every 10 --val-every 200 --save-every "$STEPS" \
    --wandb-mode offline --annotations "$ANN" "$@" \
    > "$ROOT/$name.log" 2>&1
  echo "[arms] $(date) DONE $name"
}

mkdir -p "$ROOT"
run_arm baseline --domain-weight 1.0 --frame-lambda 0.0
run_arm frame    --domain-weight 1.0 --frame-lambda 0.5
run_arm weighted --domain-weight 3.0 --frame-lambda 0.0
echo "[arms] ALL ARMS DONE"
