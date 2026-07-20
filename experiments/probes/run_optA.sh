#!/usr/bin/env bash
# Option A: real mega-only WHOLE-CORE run. Evo2 7B + LoRA, L=32768, ga=128 (validated
# effective batch), fresh-from-base. Only change vs the failed production run is the
# DATA (mega whole cores, not all-classes chunked). Self-gating milestone discipline:
# train in ~2-epoch segments with a FIXED --max-epochs 6 (so the cosine horizon stays
# constant across resumes), eval n=15 after each, and AUTO-KILL if correct_class is
# still at the ~0.067 floor by epoch 4 — so we never repeat the 1,200-step mistake.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
DATA=/data2/ds85/bgcmodel_data/splits_core
TRAIN=/data2/ds85/bgcmodel_data/mega_whole_32k/subset_c_wholecore.jsonl
ROOT=/data2/ds85/bgcmodel_runs/mega_whole_32k_run
mkdir -p "$ROOT"; TRACK="$ROOT/milestone_track.jsonl"
L=32768; GA=128; LR=5e-5; WARM=30; EPOCHS=6; PC=5     # PER_CLASS=5 -> n=15
SEGMENTS=(120 240 354)          # ~2 / ~4 / ~6 epochs (59 steps/epoch)
GATE_AFTER=240; GATE_MIN=0.15   # after epoch ~4: correct_class must clear the floor to continue

wait_for_idle(){ local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then hold=$((hold+1)); [ $hold -ge 3 ] && return 0
    else hold=0; fi; sleep 10
  done; }
resolve_ckpt(){ local c=""; for d in $(ls -dt "$1"/checkpoints/step_* 2>/dev/null); do [ -d "$d/adapter" ] && { c="$d"; break; }; done; echo "$c"; }

prev=""
for target in "${SEGMENTS[@]}"; do
  wait_for_idle
  RESUME=(); [ -n "$prev" ] && RESUME=(--resume-from "$prev")
  echo "=========================================================="
  echo "[optA] $(date) TRAIN -> step $target   (resume: ${prev:-base})"
  micromamba run -n bgcmodel deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \
    --train "$TRAIN" --val "$DATA/val.jsonl" --output-dir "$ROOT" \
    --max-seq-len $L --batch-size 1 --grad-accum $GA \
    --lr $LR --warmup-steps $WARM --max-epochs $EPOCHS --max-steps $target \
    --val-every 60 --save-every 60 --log-every 10 \
    --long-seq-strategy chunk --chunk-overlap 2048 --eos-token --continuation-prefix \
    --keep-last-ckpts 0 --early-stopping-patience 0 \
    --wandb-mode offline --seed 42 "${RESUME[@]}" > "$ROOT/train_to_$target.log" 2>&1
  rc=$?; echo "[optA] $(date) train->$target exit=$rc"
  [ $rc -ne 0 ] && { echo "[optA] TRAIN FAILED at $target — see $ROOT/train_to_$target.log"; break; }
  CKPT=$(resolve_ckpt "$ROOT"); prev="$CKPT"
  echo "[optA] eval (n=15) ckpt: $CKPT"
  wait_for_idle
  MAX_NEW=$L PER_CLASS=$PC TEMPERATURE=1.0 TOP_K=4 \
    scripts/quick_eval.sh "$CKPT" "$ROOT/eval_step$target" > "$ROOT/eval_step$target.log" 2>&1
  row=$(tail -1 "$ROOT/eval_step$target/eval_track.jsonl" 2>/dev/null)
  [ -n "$row" ] && echo "$row" >> "$TRACK"
  cc=$(echo "$row" | python3 -c "import json,sys;print(json.loads(sys.stdin.read()).get('correct_class',0))" 2>/dev/null || echo 0)
  cm=$(echo "$row" | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print(d.get('class_markers'),d.get('module_count'))" 2>/dev/null || echo "?")
  echo "[optA] $(date) STEP $target  correct_class=$cc  (class_markers/module: $cm)"
  if [ "$target" -ge "$GATE_AFTER" ]; then
    kill_it=$(python3 -c "print(1 if float('${cc:-0}') < $GATE_MIN else 0)" 2>/dev/null || echo 0)
    if [ "$kill_it" = "1" ]; then
      echo "[optA] $(date) AUTO-KILL: correct_class $cc < $GATE_MIN after step $target (epoch ~4). Whole-core lever insufficient at this scale."
      break
    fi
    echo "[optA] correct_class $cc >= $GATE_MIN — gate PASSED, continuing."
  fi
done
echo "[optA] $(date) DONE"; echo "=== milestone_track (step -> gates) ==="
[ -f "$TRACK" ] && python3 -c "
import json
for l in open('$TRACK'):
    d=json.loads(l); print(f\"step {d['step']:>4}  is_bgc {d['is_bgc']:.3f}  correct_class {d['correct_class']:.3f}  class_markers {d['class_markers']:.3f}  modules {d['module_count']:.3f}\")
"
