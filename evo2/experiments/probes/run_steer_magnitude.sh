#!/usr/bin/env bash
# STEP 1 (REDO) of steering: COHERENCE TITRATION ON THE MAGNITUDE AXIS.
#
# WHY THIS REPLACES run_steer_titration.sh (beta sweep, 2026-07-29):
# the beta sweep was confounded. beta scales the RAW difference-of-means vector, and ||v_class||
# at layer 16 spans 17x (TERPENE 1.05, PKS 1.16, RIPP 1.69, NRPS 5.80, ECTOINE 17.75). So a
# single global beta delivered a 17x-different PHYSICAL perturbation per class: at beta=0.5
# ECTOINE collapsed (coding_density 0.850 -> 0.191) while PKS was untouched at beta=2. Pooling
# those made it look like noise (paired t, df=4: nothing significant, |t|<=1.72).
#
# THE HYPOTHESIS THIS TESTS: coherence damage is governed by ||delta|| (physical magnitude),
# not by beta (semantic offsets). If true, the per-class coding-density curves should COLLAPSE
# ONTO ONE CURVE when plotted against ||delta||, giving a single class-independent coherence
# ceiling. If they do NOT collapse, coherence is class-specific too and each class needs its
# own ceiling — either way this is the measurement that decides it.
#
# Grid: ||delta|| in {0, 0.5, 1, 2, 4, 8}. Reference points at layer 16 — mean||h|| ~= 10, so
# these span 0.05-0.8 of a typical activation. In class-mean-offset (beta_equiv) terms the SAME
# magnitude means very different things, which is exactly the confound being removed:
#     ||delta||=2  ->  TERPENE 1.90 offsets | NRPS 0.34 | ECTOINE 0.11
#     ||delta||=8  ->  TERPENE 7.62 offsets | NRPS 1.38 | ECTOINE 0.45
# Note the corollary: a magnitude low enough to keep ECTOINE coherent may deliver too little
# semantic push to move it at all. That tension is a real finding, not a bug in the sweep.
#
# Short generations + coding_density only (no antiSMASH) => minutes per cell, not hours.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
# /home is a SHARED filesystem that has hit 100% (which breaks micromamba's process lock, git,
# __pycache__, matplotlib/wandb caches and antiSMASH temp files). Keep scratch on /data2 so
# disk pressure on /home cannot take a run down mid-flight.
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
DELTAS="${DELTAS:-0 0.5 1 2 4 8}"
LAYER="${LAYER:-16}"
PER_CLASS="${PER_CLASS:-5}"         # 3 was too thin to read per-class; 5 x 5 classes = 25/cell
MAX_NEW="${MAX_NEW:-2048}"          # coding_density is measurable well before a full cluster
SEED="${SEED:-42}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/steer_magnitude}"
mkdir -p "$ROOT"

# GPU gate. EXCLUSIVE=1 (default) waits for an empty device — the project convention, since
# contention invalidates memory/throughput benchmarks. EXCLUSIVE=0 shares the device and checks
# free memory only: valid here because coherence metrics are not timing-sensitive, so a
# co-tenant slows the run without changing the numbers.
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

# PER-SEQUENCE output (the beta titration persisted aggregates only, which forced a full
# pyrodigal recompute to get error bars — don't repeat that).
SEQTSV="$ROOT/per_sequence.tsv"
printf "layer\tdelta_norm\tcompound_class\tseq_idx\tv_norm\tapplied_norm\tbeta_equiv\tlength\tcoding_density\tmax_orf_aa\tn_orfs\n" > "$SEQTSV"

for D in $DELTAS; do
  OUT="$ROOT/L${LAYER}_d${D}.jsonl"
  if [ ! -s "$OUT" ]; then
    echo "[mag] $(date) GEN delta_norm=$D"; wait_for_idle
    PY evo2/scripts/steer_generate.py --adapter "$V2" --from-jsonl "$VAL" \
       --dirs-npz "$SWEEP/acts_v2.dirs.npz" --acts-npz "$SWEEP/acts_v2.npz" \
       --classes $CLASSES --layer "$LAYER" --delta-norm "$D" \
       --per-class "$PER_CLASS" --max-new-tokens "$MAX_NEW" \
       --top-k 4 --top-p 1.0 --temperature 1.0 --seed "$SEED" \
       --out-jsonl "$OUT" > "$ROOT/gen_d$D.log" 2>&1
  fi
  [ -s "$OUT" ] || { echo "[mag] !! delta_norm=$D failed (see $ROOT/gen_d$D.log)"; continue; }
  PY - "$LAYER" "$D" "$OUT" "$SEQTSV" <<'PYEOF'
import json, sys, collections
sys.path.insert(0, "/home/ds85/projects/BCGModelling/src")
from bgc_pipeline.evaluation import check_coding_sanity
layer, dn, gen_p, seq_p = sys.argv[1:5]
per = collections.defaultdict(list)
idx = collections.Counter()
with open(seq_p, "a") as fh:
    for line in open(gen_p):
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("sequence", "") or ""
        c = r.get("compound_class", "?")
        i = idx[c]; idx[c] += 1
        if len(s) < 100:
            cd, mo, no = 0.0, 0, 0
        else:
            m = check_coding_sanity(s)
            cd = m.get("coding_density", 0.0); mo = m.get("max_orf_aa", 0); no = m.get("n_orfs", 0)
        per[c].append(cd)
        fh.write(f"{layer}\t{dn}\t{c}\t{i}\t{r.get('steer_v_norm',0)}\t{r.get('steer_applied_norm',0)}\t"
                 f"{r.get('steer_beta_equiv',0)}\t{len(s)}\t{cd:.4f}\t{mo}\t{no}\n")
mean = lambda x: sum(x) / len(x) if x else 0.0
parts = " ".join(f"{c}={mean(v):.3f}" for c, v in sorted(per.items()))
allv = [x for v in per.values() for x in v]
print(f"[mag] delta_norm={dn}: pooled={mean(allv):.3f} | {parts}")
PYEOF
done

echo "==================== MAGNITUDE TITRATION (L$LAYER) ===================="
# Per-class matrix + the collapse test: if coherence is magnitude-governed, every class should
# fall off at a similar ||delta||. Spread of the per-class 50%-damage point is the readout.
PY - "$SEQTSV" <<'PYEOF'
import sys, csv, collections, statistics as st
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
if not rows:
    print("no rows"); raise SystemExit(0)
deltas = sorted({float(r["delta_norm"]) for r in rows})
classes = sorted({r["compound_class"] for r in rows})
cell = collections.defaultdict(list)
for r in rows:
    cell[(float(r["delta_norm"]), r["compound_class"])].append(float(r["coding_density"]))
print(f"{'||delta||':>9} " + " ".join(f"{c:>9}" for c in classes) + f" {'pooled':>8}")
for d in deltas:
    vals = [st.mean(cell[(d, c)]) if cell[(d, c)] else float('nan') for c in classes]
    pooled = st.mean([x for c in classes for x in cell[(d, c)]])
    print(f"{d:>9} " + " ".join(f"{v:>9.3f}" for v in vals) + f" {pooled:>8.3f}")
base = {c: st.mean(cell[(deltas[0], c)]) for c in classes}
print("\nRelative to unsteered control (fraction of baseline coding density retained):")
print(f"{'||delta||':>9} " + " ".join(f"{c:>9}" for c in classes))
for d in deltas:
    print(f"{d:>9} " + " ".join(
        f"{(st.mean(cell[(d,c)])/base[c] if base[c] else float('nan')):>9.2f}" for c in classes))
print("\nCOLLAPSE TEST — smallest ||delta|| where a class drops below 70% of its own baseline:")
pts = {}
for c in classes:
    hit = [d for d in deltas if base[c] and st.mean(cell[(d, c)]) < 0.70 * base[c]]
    pts[c] = hit[0] if hit else None
for c in classes:
    print(f"  {c:>9}: {pts[c] if pts[c] is not None else '> max tested'}")
got = [v for v in pts.values() if v is not None]
if len(got) >= 2 and len(set(got)) == 1:
    print("  => SAME breakpoint across classes: coherence IS magnitude-governed. Single ceiling.")
elif len(got) >= 2:
    print(f"  => breakpoints SPREAD {min(got)}-{max(got)}: coherence only partly magnitude-governed;"
          f" use per-class ceilings.")
else:
    print("  => too few classes broke; ceiling is above the tested range (widen DELTAS).")
PYEOF
echo "[mag] per-sequence values: $SEQTSV"
echo "[mag] ALL DONE $(date)"
