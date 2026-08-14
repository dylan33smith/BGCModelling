#!/usr/bin/env bash
# THE TWO-ARM LONG RUN: baseline and weighted(3x) to 2,000 steps, checkpointed every 500.
# Run from the repo root.
#
# WHY TWO ARMS AND NOT ONE. Every Phase-2 result carries the same asterisk: each arm saw 6,400
# windows = 6.7% of ONE epoch. Baseline alone would answer "is the 1B budget-limited or
# capacity-limited?" but NOT the more interesting question — does an intervention start working once
# the model is good enough. And the treatment alone answers nothing either: with no baseline at
# matched steps, any improvement is unattributable. So both, and the GAP between the curves at each
# checkpoint is the intervention effect as a function of training budget.
#
# WHY FRESH DIRS RATHER THAN RESUMING THE 400-STEP RUNS. The LR schedule is defined over
# `max_steps`: a 400-step run has already decayed to ~0 by its end, so resuming it for 1,600 more
# would train on a schedule no native 2,000-step run has, and the curve would be uninterpretable.
# The existing 400-step arms stay on disk untouched as the low-budget point.
#
# ⚠️ THE PRIOR ON `weighted` IS WEAK, ON OUR OWN EVIDENCE. Its dose-response was FLAT — 10x did no
# more than 3x on in-domain loss (identical 0.8763), and at 10x the significant effect was DAMAGE to
# non-domain positions (p=0.0001) while the domain gain was not significant (p=0.15). If the
# mechanism worked but merely needed time, dose should matter MORE, not less. The budget objection is
# still legitimate and worth one run; the expectation should not be high, and that is recorded here
# BEFORE the result so it cannot be adjusted afterwards.
set -euo pipefail
export HF_HOME=/data2/ds85/hf_cache
export EVO2_BASE_MODEL=evo2_1b_base

DATA=/data2/ds85/bgcmodel_data/splits_core
ANN=$DATA/train.domain_spans.jsonl
ROOT=${ROOT:-/data2/ds85/bgcmodel_runs/phase2_long}
STEPS=${STEPS:-2000}
SAVE=${SAVE:-500}
L=8192
mkdir -p "$ROOT"

run () {
  local name="$1"; shift
  local out="$ROOT/$name"
  if [[ -f "$out/final_adapter/adapter_config.json" ]]; then
    echo "[long] $name already finished — skipping"; return 0
  fi
  echo "[long] $(date) START $name ($*)"
  micromamba run -n bgcmodel deepspeed --num_gpus=1 --master_port 29510 \
    evo2/scripts/finetune_evo2_lora.py \
    --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" --output-dir "$out" \
    --max-seq-len "$L" --batch-size 1 --grad-accum 16 \
    --long-seq-strategy chunk --chunk-overlap 1024 \
    --warmup-steps 50 --max-epochs 1 --max-steps "$STEPS" \
    --log-every 25 --val-every 500 --save-every "$SAVE" \
    --wandb-mode offline --annotations "$ANN" "$@" \
    > "$ROOT/$name.log" 2>&1
  echo "[long] $(date) DONE $name"
}

run baseline_long --domain-weight 1.0  --frame-lambda 0.0
run weighted_long --domain-weight 3.0  --frame-lambda 0.0
echo "[long] $(date) BOTH ARMS DONE — next: evo2_1b/experiments/score_checkpoint_curve.sh"
