#!/usr/bin/env bash
# PHASE-2 OBJECTIVE ARMS on the 1B. Run from the repo root.
#
# Three arms, identical in every respect except the objective:
#   baseline   --domain-weight 1.0 --frame-lambda 0.0   (bit-identical to causal_lm_loss)
#   frame      --frame-lambda 0.5                       (in-gene stop-completion penalty)
#   weighted   --domain-weight 3.0                      (per-record-normalised domain weights)
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
L=${L:-4096}

run_arm () {
  local name="$1"; shift
  local out="$ROOT/$name"
  if [[ -f "$out/final_adapter/adapter_config.json" ]]; then
    echo "[arms] $name already finished — skipping"; return 0
  fi
  echo "[arms] $(date) START $name  ($*)"
  deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \
    --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" \
    --output-dir "$out" \
    --max-seq-len "$L" --batch-size 1 --grad-accum 16 \
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
