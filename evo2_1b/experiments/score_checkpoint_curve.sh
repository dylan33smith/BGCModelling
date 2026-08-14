#!/usr/bin/env bash
# THE CHECKPOINT CURVE: detection rate vs training budget, for baseline AND weighted(3x).
# Run from the repo root, after run_long_2arm.sh.
#
# WHAT THIS ANSWERS. Every Phase-2 result carried one asterisk: each arm saw 6.7% of ONE epoch.
# Two questions fall out of the same run, and they need different readings:
#
#   Q1 (baseline curve alone): is the 1B BUDGET-limited or CAPACITY-limited?
#       climbing  -> Phase 2 was measured at the floor and its comparisons must be redone
#       flat      -> the 1B cannot do this at any budget; change substrate, not objective
#
#   Q2 (the GAP between curves): does the intervention start working once the model is good enough?
#       This is why both arms were trained. Baseline alone cannot answer it, and the treatment
#       alone is unattributable without a matched-step comparator.
#
# n = 52 AT EVERY CHECKPOINT, INCLUDING THE LAST. 152 was sized for ONE powered comparison
# (detection 0.112, 80% power for a doubling). This run's deliverable is a TREND, read from monotone
# movement across four points, and a uniform n keeps the confidence intervals the same width at
# every point — a curve whose last point is 3x more precise than the others is harder to read, not
# easier. 52 x 4 x 2 = 416 generations, ~50 min batched.
# CONSEQUENCE, STATED UP FRONT: the endpoint contrast is UNDERPOWERED alone (n=52 gives ~40% power
# for a doubling). If the curve shows something worth a single powered test, regenerate the
# 2000-step points at 152 THEN — do not read a lone n=52 contrast as a powered result.
#
# ⚠️ BATCHED, SO NOT POOLABLE WITH SEQUENTIAL OUTPUT — left-padding makes a padded prompt a
# different prompt. Every point on this curve is batched, so the curve is internally consistent.
set -euo pipefail
export HF_HOME=/data2/ds85/hf_cache
export EVO2_BASE_MODEL=evo2_1b_base

ROOT=${ROOT:-/data2/ds85/bgcmodel_runs/phase2_long}
DATA=/data2/ds85/bgcmodel_data/splits_core
STEPS_LIST=${STEPS_LIST:-"500 1000 1500 2000"}
N_PER_CLASS=${N_PER_CLASS:-13}     # x4 classes = 52 per checkpoint
FINAL_PER_CLASS=${FINAL_PER_CLASS:-13}   # uniform with the rest; see header
MAX_NEW=${MAX_NEW:-8000}
BATCH=${BATCH:-32}
PY() { micromamba run -n bgcmodel python "$@"; }

for arm in baseline_long weighted_long; do
  for st in $STEPS_LIST; do
    ADP="$ROOT/$arm/checkpoints/step_${st}/adapter"
    [[ -d "$ADP" ]] || { echo "[curve] $arm step_$st: no adapter — skipped"; continue; }
    pc=$N_PER_CLASS
    [[ "$st" == "2000" ]] && pc=$FINAL_PER_CLASS
    GEN="$ROOT/$arm/gen_step${st}.jsonl"
    if [[ ! -s "$GEN" ]]; then
      echo "[curve] $(date +%H:%M) generating $arm @ step $st  ($((pc*4)) records)"
      PY evo2/scripts/generate_bgc.py --adapter "$ADP" \
         --from-jsonl "$DATA/valtest_eval_4class.jsonl" \
         --per-class "$pc" --n 1 --max-new-tokens "$MAX_NEW" --seed 0 \
         --batch-size "$BATCH" --out-jsonl "$GEN" \
         > "$ROOT/$arm.gen_step${st}.log" 2>&1 || { echo "[curve] FAILED $arm@$st"; continue; }
    fi
    LAD="$ROOT/$arm/ladder_step${st}.json"
    [[ -s "$LAD" ]] && { echo "[curve] $arm@$st ladder already scored — skipping"; continue; }
    PY evo2/scripts/score_ladder.py --gen "$GEN" \
       --out-json "$LAD" \
       > "$ROOT/$arm.ladder_step${st}.log" 2>&1 || echo "[curve] ladder FAILED $arm@$st"
  done
done

echo
PY evo2_1b/experiments/plot_checkpoint_curve.py --root "$ROOT"
