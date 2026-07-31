#!/usr/bin/env bash
# STEP 1 of steering: CHEAP COHERENCE TITRATION — find the beta band where the model still
# writes coherent DNA, BEFORE spending hours on antiSMASH evaluation.
#
# Why this exists: the first sweep parameterized strength as a fraction of the mean activation
# norm (alpha). That was (a) far too strong — coherence collapsed at the SMALLEST value tested
# (coding_density 0.74 -> 0.19 at alpha=1) so every cell measured a wrecked model, and (b) not
# comparable across classes, since ||v_class||/mean||h|| ranges ~19x (TERPENE 0.11, ECTOINE 1.78).
# Here strength is BETA * the raw difference-of-means vector: beta=1 shifts the representation by
# exactly one class-mean offset, which is the natural, per-class-comparable unit.
#
# Short generations + coding_density only (no antiSMASH) => minutes per cell, not hours.
# Output: the beta range where coding_density stays near the unsteered control. Only that band
# is worth a full evaluation.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
# /home is a SHARED filesystem that has hit 100% (which breaks micromamba's process lock,
# git, __pycache__, matplotlib/wandb caches and antiSMASH temp files). Keep every scratch
# write on /data2 so disk pressure on /home cannot take a run down mid-flight.
export TMPDIR="${TMPDIR:-/data2/ds85/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data2/ds85/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data2/ds85/cache/mpl}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
ENV=bgcmodel
PY(){ micromamba run -n "$ENV" python "$@"; }

V2="${V2:-/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
SWEEP="${SWEEP:-/data2/ds85/bgcmodel_runs/class_probe_sweep}"
CLASSES="${CLASSES:-NRPS PKS TERPENE ECTOINE RIPP}"
BETAS="${BETAS:-0 0.1 0.25 0.5 1 2}"
LAYER="${LAYER:-16}"
PER_CLASS="${PER_CLASS:-3}"
MAX_NEW="${MAX_NEW:-2048}"          # short: coding_density is measurable well before a full cluster
SEED="${SEED:-42}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/steer_titration}"
mkdir -p "$ROOT"

# GPU gate. EXCLUSIVE=1 (default) waits for an empty device — the project convention, since
# contention invalidates memory/throughput benchmarks. EXCLUSIVE=0 shares the device and checks
# free memory only: valid for THIS experiment because coherence/class metrics are not
# timing-sensitive, so a co-tenant slows the run without changing the numbers.
EXCLUSIVE="${EXCLUSIVE:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"
wait_for_idle(){ local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if { [ "$EXCLUSIVE" -eq 0 ] || [ "${proc:-1}" -eq 0 ]; } && [ "${free:-0}" -ge "$MIN_FREE_MB" ]; then
      hold=$((hold+1)); [ "$hold" -ge 2 ] && return 0
    else hold=0; fi; sleep 10
  done; }

SUM="$ROOT/titration.tsv"
printf "layer\tbeta\tn\tcoding_density\tmean_len\tmax_orf_aa\tn_orfs\n" > "$SUM"
for B in $BETAS; do
  OUT="$ROOT/L${LAYER}_b${B}.jsonl"
  if [ ! -s "$OUT" ]; then
    echo "[titr] $(date) GEN beta=$B"; wait_for_idle
    PY evo2/scripts/steer_generate.py --adapter "$V2" --from-jsonl "$VAL" \
       --dirs-npz "$SWEEP/acts_v2.dirs.npz" --acts-npz "$SWEEP/acts_v2.npz" \
       --classes $CLASSES --layer "$LAYER" --beta "$B" \
       --per-class "$PER_CLASS" --max-new-tokens "$MAX_NEW" \
       --top-k 4 --top-p 1.0 --temperature 1.0 --seed "$SEED" \
       --out-jsonl "$OUT" > "$ROOT/gen_b$B.log" 2>&1
  fi
  [ -s "$OUT" ] || { echo "[titr] !! beta=$B failed"; continue; }
  # coding_sanity only — cheap, CPU, no antiSMASH
  PY - "$LAYER" "$B" "$OUT" "$SUM" <<'PYEOF'
import json,sys
sys.path.insert(0,"/home/ds85/projects/BCGModelling/src")
from bgc_pipeline.evaluation import check_coding_sanity
layer,beta,gen_p,sum_p=sys.argv[1:5]
recs=[json.loads(l) for l in open(gen_p)]
cd=[];ln=[];mo=[];no=[]
for r in recs:
    s=r.get("sequence","") or ""
    ln.append(len(s))
    if len(s)<100: cd.append(0.0); mo.append(0); no.append(0); continue
    c=check_coding_sanity(s)
    cd.append(c.get("coding_density",0.0)); mo.append(c.get("max_orf_aa",0)); no.append(c.get("n_orfs",0))
m=lambda x: round(sum(x)/len(x),3) if x else None
open(sum_p,"a").write(f"{layer}\t{beta}\t{len(recs)}\t{m(cd)}\t{int(m(ln) or 0)}\t{int(m(mo) or 0)}\t{m(no)}\n")
print(f"[titr] beta={beta}: coding_density={m(cd)} mean_len={int(m(ln) or 0)} max_orf_aa={int(m(mo) or 0)}")
PYEOF
done

echo "==================== COHERENCE TITRATION (L$LAYER) ===================="
column -t "$SUM" 2>/dev/null || cat "$SUM"
echo "Pick the beta band where coding_density stays near the beta=0 control; only that band"
echo "is worth a full antiSMASH evaluation (run_steer_sweep.sh with --beta)."
echo "[titr] ALL DONE $(date)"
