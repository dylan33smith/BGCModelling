#!/usr/bin/env bash
# Concentration probe: does training ONLY on megasynthase classes lift the gates?
# mega_all (mega-only, all lengths, blind-chunked, L=16384, 350 steps) vs the
# existing P0 (all-classes) and C (mega-only whole) controls — all re-evaluated at
# n=15 (PER_CLASS=5) so the comparison is at reliable n.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
DATA=/data2/ds85/bgcmodel_data/splits_core
PROBE=/data2/ds85/bgcmodel_data/probe_subsets
ROOT=/data2/ds85/bgcmodel_runs/probes_20260706
L=16384; GA=16; STEPS=350; WARM=15; LR=1.5e-4; VALEVERY=175; PC=5   # PER_CLASS=5 -> n=15

wait_for_idle(){ local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then hold=$((hold+1)); [ $hold -ge 3 ] && return 0
    else hold=0; fi; sleep 10
  done; }

resolve_ckpt(){ local out="$1" c=""
  for d in $(ls -dt "$out"/checkpoints/step_* 2>/dev/null); do [ -d "$d/adapter" ] && { c="$d"; break; }; done
  echo "$c"; }

append_row(){ local name="$1" ckpt="$2" outdir="$3"
  wait_for_idle
  MAX_NEW=$L PER_CLASS=$PC TEMPERATURE=1.0 TOP_K=4 \
    evo2/scripts/quick_eval.sh "$ckpt" "$outdir" > "$ROOT/eval_${name}.log" 2>&1
  local row; row=$(tail -1 "$outdir/eval_track.jsonl" 2>/dev/null)
  if [ -n "$row" ]; then
    python3 - "$name" "$row" >> "$ROOT/probe_summary.tsv" <<'PY'
import json,sys
n,d=sys.argv[1],json.loads(sys.argv[2]); g=lambda k:d.get(k)
print("\t".join(str(x) for x in [n,g("is_bgc"),g("correct_class"),g("class_markers"),g("obligate_fraction"),g("any_domain_rate"),g("module_count"),g("coding_density")]))
PY
  fi
  echo "[conc] $(date) $name done  (n=$(python3 -c "import json;print(json.loads(open('$outdir/eval_track.jsonl').read().strip().splitlines()[-1])['n'])" 2>/dev/null))"; }

# ---- Arm 1: NEW — mega-only all-lengths, 350 steps, then eval n=15 ----
OUT=$ROOT/mega_all
echo "[conc] $(date) TRAIN mega_all (mega-only, all lengths, blind chunk, 350 steps)"
wait_for_idle
micromamba run -n bgcmodel deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \
  --train "$PROBE/subset_mega_all.jsonl" --val "$DATA/val.jsonl" --output-dir "$OUT" \
  --max-seq-len $L --batch-size 1 --grad-accum $GA \
  --lr $LR --lora-dropout 0 --warmup-steps $WARM --max-epochs 5 --max-steps $STEPS \
  --val-every $VALEVERY --save-every $STEPS --log-every 10 \
  --long-seq-strategy chunk --chunk-overlap 2048 --eos-token --continuation-prefix \
  --keep-last-ckpts 0 --early-stopping-patience 0 \
  --wandb-mode offline --seed 42 > "$ROOT/train_mega_all.log" 2>&1
echo "[conc] $(date) mega_all train exit=$?"
append_row mega_all_n15 "$(resolve_ckpt "$OUT")" "$OUT/quick_eval_n15"

# ---- Arm 2 & 3: re-eval existing controls at n=15 ----
append_row p0_control_n15  "$(resolve_ckpt "$ROOT/p0_control")"  "$ROOT/p0_control/quick_eval_n15"
append_row c_wholecore_n15 "$(resolve_ckpt "$ROOT/c_wholecore")" "$ROOT/c_wholecore/quick_eval_n15"

echo "[conc] $(date) ALL DONE"; echo "=== n=15 rows ==="; grep -E "mega_all_n15|_n15" "$ROOT/probe_summary.tsv"
