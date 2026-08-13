#!/usr/bin/env bash
# PHASE-2 ARMS, RE-SCORED AT ADEQUATE POWER. Run from the repo root. No retraining — the three
# adapters already exist; this only regenerates and re-scores them.
#
# WHY. The first pass generated 24 per arm and found 3/24 records with any biosynthetic signal in
# EVERY arm (Fisher p = 1.000), with `best_bio_bits` means dominated by single draws — baseline's
# top record was 64% of its arm total. A power calculation on that baseline rate (p0 = 0.125,
# alpha .05, 80% power) needs **152 per arm to detect a doubling** and 46 to detect a tripling. The
# pre-registered kill criterion assumed a sensitive test; at n=24 it was not one, so the null it
# produced could not have rejected the hypothesis. n = 152 fixes exactly that.
#
# WHY 8,000 AND NOT 6,000 NT. Probe 1 showed the 6 kb cap CENSORED ONE ARM ONLY: baseline never
# approached it (max 1,157 aa) while the frame arm had 6/24 above the 2,000 aa ceiling. A ceiling
# that binds on one arm and not the other is a confound. 8,000 is the 1B's full context (8,192
# minus the prefix), so nothing is clipped by us.
#
# WHY BATCHED, AND THE ONE RULE THAT COMES WITH IT. --batch-size is 12.05x faster and is now
# validated (scripts/validate_batched_generation.py). vortex batches only equal-length prompts, so
# generate_bgc.py left-pads them, and a byte-level model CONDITIONS ON THE PAD BYTES — a padded
# prompt is a different prompt. Chunking here is deterministic (prompt order, fixed seed), so every
# arm sees identical padding and it cannot manufacture a between-arm difference.
# ⚠️ The output goes to gen_n150.jsonl, NOT gen.jsonl: the existing sequential 24-per-arm files are
# a DIFFERENT experiment and must never be pooled with these.
#
# WHAT WOULD COUNT. Read `best_bio_bits` and the detection RATE, with novelty as a hard constraint.
# At n=152 a doubling of detection is detectable; report the test, not the mean, because the mean is
# what misled the first pass.
set -euo pipefail
export HF_HOME=/data2/ds85/hf_cache
export EVO2_BASE_MODEL=evo2_1b_base

ROOT=${ROOT:-/data2/ds85/bgcmodel_runs/phase2_1b}
DATA=/data2/ds85/bgcmodel_data/splits_core
PER_CLASS=${PER_CLASS:-38}      # x4 classes = 152 per arm, the power-analysis target
MAX_NEW=${MAX_NEW:-8000}
BATCH=${BATCH:-32}
PY() { micromamba run -n bgcmodel python "$@"; }

for arm in baseline frame weighted; do
  ADP="$ROOT/$arm/final_adapter"
  GEN="$ROOT/$arm/gen_n150.jsonl"
  [[ -f "$ADP/adapter_model.safetensors" ]] || { echo "[highn] $arm: no adapter — skipped"; continue; }
  if [[ ! -s "$GEN" ]]; then
    echo "[highn] $(date) generating $arm ($((PER_CLASS*4)) records, ${MAX_NEW} nt, batch ${BATCH})"
    PY evo2/scripts/generate_bgc.py --adapter "$ADP" \
       --from-jsonl "$DATA/valtest_eval_4class.jsonl" \
       --per-class "$PER_CLASS" --n 1 --max-new-tokens "$MAX_NEW" --seed 0 \
       --batch-size "$BATCH" \
       --out-jsonl "$GEN" > "$ROOT/$arm.gen_n150.log" 2>&1
  fi
  echo "[highn] $(date) ladder on $arm ($(wc -l < "$GEN") records)"
  PY evo2/scripts/score_ladder.py --gen "$GEN" --out-json "$ROOT/$arm/ladder_n150.json" \
     > "$ROOT/$arm.ladder_n150.log" 2>&1 || echo "[highn] $arm ladder FAILED"
done

echo
PY evo2_1b/experiments/compare_arms_highn.py --root "$ROOT"
