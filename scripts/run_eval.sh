#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline for a trained checkpoint. Run when training finishes.
#
#   scripts/run_eval.sh <run-dir-or-checkpoint-dir> [out-dir] [per-class]
#
# Sequences (each step writes into OUT):
#   1. generate_bgc.py        — generate conditioned BGCs from held-out val prompts
#   2. memorization_check.py  — novelty vs full corpus (+ positive-control calibration)
#   3. eval_conditioning_adherence.py — M9 likelihood adherence (+ base baseline)
#   4. conditioning_experiment.py     — causal class control + E. coli taxon control
#   5. eval_suite_driver.py   — 8-metric suite + novelty gate, generated vs positive control
#
# GPU steps: 1, 3, 4, and (optionally) the GPU metrics in 5. They need the GPU
# free — run AFTER training has finished/stopped. Metrics whose external tool is
# not installed (antiSMASH/ESMFold/MMseqs2/BiG-SCAPE/Pfam) self-skip.
# ─────────────────────────────────────────────────────────────────────────────

CKPT_IN="${1:?usage: run_eval.sh <run-dir-or-checkpoint-dir> [out-dir] [per-class]}"
OUT="${2:-eval_out_$(basename "$CKPT_IN")}"
PER_CLASS="${3:-3}"

ENV_NAME="${ENV_NAME:-bgcmodel}"
export HF_HOME="${HF_HOME:-/data2/ds85/hf_cache}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_curated/val.jsonl}"
REF="${REF:-/data2/ds85/bgcmodel_data/splits_combined_grouped/train.jsonl}"
POS="${POS:-eval/positive_control_mibig.jsonl}"
MAX_NEW="${MAX_NEW:-16384}"

# Resolve checkpoint: accept a checkpoint dir (has adapter/) or a run dir (use best/).
if [[ -d "$CKPT_IN/adapter" ]]; then
  CKPT="$CKPT_IN"
elif [[ -d "$CKPT_IN/checkpoints/best/adapter" ]]; then
  CKPT="$CKPT_IN/checkpoints/best"
else
  echo "Could not find an adapter/ under $CKPT_IN (checkpoint dir or run dir with checkpoints/best/)." >&2
  exit 2
fi

mkdir -p "$OUT"
run() { echo "[$(date '+%F %T')] >> $*" | tee -a "$OUT/run_eval.log"; "$@"; }
PY() { micromamba run -n "$ENV_NAME" python "$@"; }

echo "Checkpoint: $CKPT" | tee "$OUT/run_eval.log"
echo "Output:     $OUT"  | tee -a "$OUT/run_eval.log"

# 1. Generate from held-out conditioning prompts.
run PY scripts/generate_bgc.py --adapter "$CKPT" --from-jsonl "$VAL" \
  --per-class "$PER_CLASS" --max-new-tokens "$MAX_NEW" \
  --out-fasta "$OUT/generated.fasta" --out-jsonl "$OUT/generated.jsonl"

# 2. Novelty / anti-memorization vs the full corpus, with positive-control calibration.
run PY scripts/memorization_check.py --query "$OUT/generated.jsonl" --ref "$REF" \
  --positive-control "$POS" --output "$OUT/memorization.jsonl"

# 3. Conditioning-adherence (likelihood classifier) + base-model baseline.
run PY scripts/eval_conditioning_adherence.py --adapter "$CKPT" --from-jsonl "$VAL" \
  --compare-base --output "$OUT/adherence.json"

# 4. Controlled conditioning experiment: causal class control + E. coli taxon control.
run PY scripts/conditioning_experiment.py --adapter "$CKPT" --experiment both \
  --output "$OUT/conditioning.json"

# 5. 8-metric suite + novelty gate, generated vs positive control.
run PY scripts/eval_suite_driver.py --gen "$OUT/generated.jsonl" --positive "$POS" \
  --novelty "$OUT/memorization.jsonl" --output "$OUT/eval_suite.json"

echo "[$(date '+%F %T')] EVAL COMPLETE -> $OUT" | tee -a "$OUT/run_eval.log"
echo "  generated.jsonl, memorization.jsonl, adherence.json, conditioning.json, eval_suite.json"
