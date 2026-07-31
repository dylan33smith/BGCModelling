#!/usr/bin/env bash
# Fast capability-probe chain for the 2026-07-03 conditioning diagnosis.
# Runs P0(control) / B(+projections) / C(whole-core) / D(mega-up) sequentially
# on one GPU: each = a short fresh-from-base LoRA probe then a quick_eval.
# Interpretation is on SENSITIVE proxies (class_markers, obligate_fraction,
# any_domain_rate) + module_count/correct_class, each vs the P0 control.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache

DATA=/data2/ds85/bgcmodel_data/splits_core
SUB=/data2/ds85/bgcmodel_data/probe_subsets
ROOT=/data2/ds85/bgcmodel_runs/probes_20260706
mkdir -p "$ROOT"
SUMMARY="$ROOT/probe_summary.tsv"
[ -f "$SUMMARY" ] || printf "probe\tis_bgc\tcorrect_class\tclass_markers\tobligate_fraction\tany_domain_rate\tmodule_count\tcoding_density\n" > "$SUMMARY"

# Shared fast-probe hyperparams
L=16384; GA=16; STEPS=350; WARM=20; LR=1.5e-4; VALEVERY=175

# probe | train-jsonl | extra finetune args
CONFIGS=(
  "p0_control|$DATA/train.jsonl|"
  "b_projections|$DATA/train.jsonl|--lora-target-parameters projections.weight"
  "c_wholecore|$SUB/subset_c_wholecore.jsonl|"
  "d_megaup|$SUB/subset_d_megaup.jsonl|"
)

wait_for_idle() {  # proc==0 & free>=70GB, 30s hold
  local hold=0
  while true; do
    local proc free
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then
      hold=$((hold+1)); [ "$hold" -ge 3 ] && return 0
    else hold=0; fi
    sleep 10
  done
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name train extra <<< "$cfg"
  OUT="$ROOT/$name"
  echo "=========================================================="
  echo "[chain] $(date) PROBE $name  train=$train  extra=[$extra]"
  echo "=========================================================="
  wait_for_idle
  echo "[chain] GPU idle — training $name"
  # shellcheck disable=SC2086
  micromamba run -n bgcmodel deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \
    --train "$train" --val "$DATA/val.jsonl" \
    --output-dir "$OUT" \
    --max-seq-len $L --batch-size 1 --grad-accum $GA \
    --lr $LR --lora-dropout 0 --warmup-steps $WARM --max-epochs 5 --max-steps $STEPS \
    --val-every $VALEVERY --save-every $STEPS --log-every 10 \
    --long-seq-strategy chunk --chunk-overlap 2048 --eos-token --continuation-prefix \
    --keep-last-ckpts 0 --early-stopping-patience 0 \
    --wandb-mode offline --seed 42 $extra \
    > "$ROOT/train_$name.log" 2>&1
  rc=$?
  echo "[chain] $(date) $name training exit=$rc"
  if [ $rc -ne 0 ]; then echo "[chain] TRAIN FAILED $name — see $ROOT/train_$name.log"; continue; fi

  # eval the final adapter (generate at the probe's L)
  CKPT=""
  for d in $(ls -dt "$OUT"/checkpoints/step_* 2>/dev/null); do
    [ -d "$d/adapter" ] && { CKPT="$d"; break; }
  done
  echo "[chain] eval ckpt: $CKPT"
  wait_for_idle
  MAX_NEW=$L PER_CLASS=2 TEMPERATURE=1.0 TOP_K=4 \
    evo2/scripts/quick_eval.sh "$CKPT" "$OUT/quick_eval" > "$ROOT/eval_$name.log" 2>&1
  # pull the eval_track row into the summary
  row=$(tail -1 "$OUT/quick_eval/eval_track.jsonl" 2>/dev/null)
  if [ -n "$row" ]; then
    python3 - "$name" "$row" >> "$SUMMARY" <<'PY'
import json,sys
name,row=sys.argv[1],sys.argv[2]
d=json.loads(row)
g=lambda k: d.get(k)
print("\t".join(str(x) for x in [name,g("is_bgc"),g("correct_class"),g("class_markers"),g("obligate_fraction"),g("any_domain_rate"),g("module_count"),g("coding_density")]))
PY
  fi
  echo "[chain] $(date) $name DONE"
done
echo "[chain] $(date) ALL PROBES DONE"
echo "=== SUMMARY ==="; cat "$SUMMARY"
