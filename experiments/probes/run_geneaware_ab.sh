#!/usr/bin/env bash
# Gene-aware chunking A/B: same long-mega dataset, blind vs gene-aware chunk
# boundaries, 200 steps each. Isolates whether snapping cuts to gene gaps (so
# complete genes/modules aren't split) recovers C's whole-core benefit.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
DATA=/data2/ds85/bgcmodel_data/splits_core
PROBE=/data2/ds85/bgcmodel_data/probe_subsets
ROOT=/data2/ds85/bgcmodel_runs/probes_20260706
TRAIN=$PROBE/subset_longmega_geneaware.jsonl
L=16384; GA=16; STEPS=200; WARM=15; LR=1.5e-4; VALEVERY=100

wait_for_idle() { local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then hold=$((hold+1)); [ $hold -ge 3 ] && return 0
    else hold=0; fi; sleep 10
  done; }

run_arm() {  # $1=name  $2=extra-flags
  local name="$1" extra="$2" OUT="$ROOT/$1"
  echo "=========================================================="
  echo "[ab] $(date) ARM $name  extra=[$extra]"
  wait_for_idle
  # shellcheck disable=SC2086
  micromamba run -n bgcmodel deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \
    --train "$TRAIN" --val "$DATA/val.jsonl" --output-dir "$OUT" \
    --max-seq-len $L --batch-size 1 --grad-accum $GA \
    --lr $LR --lora-dropout 0 --warmup-steps $WARM --max-epochs 5 --max-steps $STEPS \
    --val-every $VALEVERY --save-every $STEPS --log-every 10 \
    --long-seq-strategy chunk --chunk-overlap 2048 --eos-token --continuation-prefix \
    --keep-last-ckpts 0 --early-stopping-patience 0 \
    --wandb-mode offline --seed 42 $extra > "$ROOT/train_$name.log" 2>&1
  local rc=$?; echo "[ab] $(date) $name train exit=$rc"
  [ $rc -ne 0 ] && { echo "[ab] TRAIN FAILED $name — see $ROOT/train_$name.log"; return 1; }
  local CKPT=""; for d in $(ls -dt "$OUT"/checkpoints/step_* 2>/dev/null); do [ -d "$d/adapter" ] && { CKPT="$d"; break; }; done
  echo "[ab] eval ckpt: $CKPT"; wait_for_idle
  MAX_NEW=$L PER_CLASS=2 TEMPERATURE=1.0 TOP_K=4 \
    scripts/quick_eval.sh "$CKPT" "$OUT/quick_eval" > "$ROOT/eval_$name.log" 2>&1
  local row; row=$(tail -1 "$OUT/quick_eval/eval_track.jsonl" 2>/dev/null)
  if [ -n "$row" ]; then
    python3 - "$name" "$row" >> "$ROOT/probe_summary.tsv" <<'PY'
import json,sys
n,d=sys.argv[1],json.loads(sys.argv[2]); g=lambda k:d.get(k)
print("\t".join(str(x) for x in [n,g("is_bgc"),g("correct_class"),g("class_markers"),g("obligate_fraction"),g("any_domain_rate"),g("module_count"),g("coding_density")]))
PY
  fi
  echo "[ab] $(date) $name DONE"
}

run_arm ga_blind ""
run_arm ga_geneaware "--gene-aware-chunking"
echo "[ab] $(date) ALL DONE"; echo "=== summary ==="; cat "$ROOT/probe_summary.tsv"
