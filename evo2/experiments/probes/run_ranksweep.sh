#!/usr/bin/env bash
# Rank sweep: does more LoRA expressiveness lift the gate? Same mega_all setup as the
# n=15 concentration probe (L=16384, ga=16, 350 steps, fresh-from-base), only lora_r
# changes. alpha=2r holds alpha/r=2.0 constant so we isolate rank from update strength.
# Baseline already on record: mega_all r=16 -> correct_class 0.067 / class_markers 0.133.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
DATA=/data2/ds85/bgcmodel_data/splits_core
PROBE=/data2/ds85/bgcmodel_data/probe_subsets
ROOT=/data2/ds85/bgcmodel_runs/probes_20260706
TRAIN=$PROBE/subset_mega_all.jsonl
L=16384; GA=16; STEPS=350; WARM=15; LR=1.5e-4; PC=5

wait_for_idle(){ local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then hold=$((hold+1)); [ $hold -ge 3 ] && return 0
    else hold=0; fi; sleep 10
  done; }
resolve_ckpt(){ local c=""; for d in $(ls -dt "$1"/checkpoints/step_* 2>/dev/null); do [ -d "$d/adapter" ] && { c="$d"; break; }; done; echo "$c"; }

run_rank(){  # $1=name $2=r $3=alpha
  local name="$1" r="$2" alpha="$3" OUT="$ROOT/$1"
  echo "=========================================================="
  echo "[rank] $(date) ARM $name  r=$r alpha=$alpha (alpha/r=$(python3 -c "print($alpha/$r)"))"
  wait_for_idle
  micromamba run -n bgcmodel deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \
    --train "$TRAIN" --val "$DATA/val.jsonl" --output-dir "$OUT" \
    --max-seq-len $L --batch-size 1 --grad-accum $GA \
    --lora-r $r --lora-alpha $alpha --lora-dropout 0 \
    --lr $LR --warmup-steps $WARM --max-epochs 5 --max-steps $STEPS \
    --val-every 175 --save-every $STEPS --log-every 10 \
    --long-seq-strategy chunk --chunk-overlap 2048 --eos-token --continuation-prefix \
    --keep-last-ckpts 0 --early-stopping-patience 0 \
    --wandb-mode offline --seed 42 > "$ROOT/train_$name.log" 2>&1
  local rc=$?; echo "[rank] $(date) $name train exit=$rc"
  [ $rc -ne 0 ] && { echo "[rank] TRAIN FAILED $name — $ROOT/train_$name.log"; return 1; }
  local CKPT; CKPT=$(resolve_ckpt "$OUT"); echo "[rank] eval (n=15): $CKPT"
  wait_for_idle
  MAX_NEW=$L PER_CLASS=$PC TEMPERATURE=1.0 TOP_K=4 \
    evo2/scripts/quick_eval.sh "$CKPT" "$OUT/quick_eval_n15" > "$ROOT/eval_$name.log" 2>&1
  local row; row=$(tail -1 "$OUT/quick_eval_n15/eval_track.jsonl" 2>/dev/null)
  if [ -n "$row" ]; then
    python3 - "$name" "$row" >> "$ROOT/probe_summary.tsv" <<'PY'
import json,sys
n,d=sys.argv[1],json.loads(sys.argv[2]); g=lambda k:d.get(k)
print("\t".join(str(x) for x in [n,g("is_bgc"),g("correct_class"),g("class_markers"),g("obligate_fraction"),g("any_domain_rate"),g("module_count"),g("coding_density")]))
PY
  fi
  echo "[rank] $(date) $name DONE  ($(echo "$row" | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print('correct_class',d['correct_class'],'class_markers',d['class_markers'],'modules',d['module_count'])" 2>/dev/null))"
}

run_rank rank64_n15  64  128
run_rank rank128_n15 128 256
echo "[rank] $(date) ALL DONE"; echo "=== rank sweep vs r=16 baseline ==="
grep -E "mega_all_n15|rank64_n15|rank128_n15" "$ROOT/probe_summary.tsv"
