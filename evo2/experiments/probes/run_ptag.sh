#!/usr/bin/env bash
# P-tag probe: does removing the |CONTINUATION| tag (constant class tag on every
# chunk + |END|) recover C's benefit on CHUNKED data? Matched pair to the D probe:
# same data (subset_d_megaup), same fast config; only --no-continuation-prefix differs.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
DATA=/data2/ds85/bgcmodel_data/splits_core
SUB=/data2/ds85/bgcmodel_data/probe_subsets
ROOT=/data2/ds85/bgcmodel_runs/probes_20260706
OUT=$ROOT/p_tag_nocontinuation
L=16384; GA=16; STEPS=350; WARM=20; LR=1.5e-4; VALEVERY=175

wait_for_idle() { local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then hold=$((hold+1)); [ $hold -ge 3 ] && return 0
    else hold=0; fi; sleep 10
  done; }

echo "[p-tag] $(date) waiting for idle GPU"; wait_for_idle
echo "[p-tag] $(date) training: subset_d_megaup + --no-continuation-prefix (constant |COMPOUND_CLASS| every chunk + |END|)"
micromamba run -n bgcmodel deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \
  --train "$SUB/subset_d_megaup.jsonl" --val "$DATA/val.jsonl" --output-dir "$OUT" \
  --max-seq-len $L --batch-size 1 --grad-accum $GA \
  --lr $LR --lora-dropout 0 --warmup-steps $WARM --max-epochs 5 --max-steps $STEPS \
  --val-every $VALEVERY --save-every $STEPS --log-every 10 \
  --long-seq-strategy chunk --chunk-overlap 2048 --eos-token --no-continuation-prefix \
  --keep-last-ckpts 0 --early-stopping-patience 0 \
  --wandb-mode offline --seed 42 > "$ROOT/train_p_tag.log" 2>&1
rc=$?; echo "[p-tag] $(date) train exit=$rc"
[ $rc -ne 0 ] && { echo "[p-tag] TRAIN FAILED — see $ROOT/train_p_tag.log"; exit 1; }

CKPT=""; for d in $(ls -dt "$OUT"/checkpoints/step_* 2>/dev/null); do [ -d "$d/adapter" ] && { CKPT="$d"; break; }; done
echo "[p-tag] eval ckpt: $CKPT"; wait_for_idle
MAX_NEW=$L PER_CLASS=2 TEMPERATURE=1.0 TOP_K=4 \
  evo2/scripts/quick_eval.sh "$CKPT" "$OUT/quick_eval" > "$ROOT/eval_p_tag.log" 2>&1
row=$(tail -1 "$OUT/quick_eval/eval_track.jsonl" 2>/dev/null)
if [ -n "$row" ]; then
  python3 - "$row" >> "$ROOT/probe_summary.tsv" <<'PY'
import json,sys
d=json.loads(sys.argv[1]); g=lambda k: d.get(k)
print("\t".join(str(x) for x in ["p_tag_nocont",g("is_bgc"),g("correct_class"),g("class_markers"),g("obligate_fraction"),g("any_domain_rate"),g("module_count"),g("coding_density")]))
PY
fi
echo "[p-tag] $(date) DONE"; echo "=== updated summary ==="; cat "$ROOT/probe_summary.tsv"
