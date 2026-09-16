#!/bin/bash
# PHASE D — I1 steering arms, each at ITS OWN class's alpha, plus the health-matched random
# control (SPEC 20.1), plus the composition cell I1 x S1.
#   usage: phaseD_steer.sh <substrate-id> <adapter-suffix> <gpu-mib>
#
# ⚠ --row-class IS MANDATORY on every steered arm. A direction is keyed to one class; without
# it every confusion row is the same steered distribution relabelled and lift reads 1.000 by
# construction. arm.py refuses, but pass it explicitly rather than relying on the refusal.
set -u
SUB=$1; SUF=$2; MIB=$3
S=/data2/ds85/bgcbench/work/pipeline/phaseD_${SUB}.status
L=/data2/ds85/bgcbench/work/pipeline/phaseD_${SUB}.log
source /data2/ds85/bgcbench/work/pipeline/_lib.sh
PFX=$(cfgval $SUB prefix)
echo "PHASE-D START $SUB $(date -Is)" > $S
wait_gpu $MIB "phaseD $SUB"

for C in $CLASSES; do
  DIR=$D/${SUB}_I1_${C}_W2${SUF}.pt
  ALPHA=$($PY -c "
import json,glob
f=glob.glob('$G9/*${SUB}_${C}_W2${SUF}*.json')
d=json.load(open(f[0])) if f else {}
a=d.get('chosen_alpha'); print(a if isinstance(a,(int,float)) else (a or {}).get('alpha','') if isinstance(a,dict) else '')" 2>/dev/null)
  if [ ! -f "$DIR" ] || [ -z "$ALPHA" ]; then
    echo "  SKIP $C — direction or alpha missing (dir=$([ -f "$DIR" ] && echo ok || echo MISSING) alpha='$ALPHA')" >> $S
    continue
  fi
  SITES=$($PY -c "
import json,glob
f=sorted(glob.glob('/data2/ds85/bgcbench/g9sites/${SUB}_${C}_W2${SUF}_${SUB}_${C}.json'))
d=json.load(open(f[0])) if f else {}
print(' '.join(str(i) for i in (d.get('chosen_sites') or [])))" 2>/dev/null)
  if [ -z "$SITES" ]; then
    echo "  ⛔ $C — no chosen_sites; the arm would steer every site, not the chosen subset" >> $S
    continue
  fi
  echo "  $C at alpha=$ALPHA, sites $SITES" >> $S
  # ⚠ AN ARRAY, NOT A STRING. An unquoted multi-line string relies on word splitting and
  # breaks the moment any value contains a space; an array passes each element intact.
  base=(--substrate "$SUB" --prefix "$PFX" --batch-size 50 --off-frozen
        --stage stage1 --cpus 16
        --adapter "$A/${SUB}_W2_${C}${SUF}" --row-class "$C" --train-classes "$C"
        --direction "$DIR" --alpha "$ALPHA" --sites $SITES)
  # ⚠ WAIT PER ARM, NOT ONCE PER PHASE. The wait_gpu at the top of this script is checked
  # before the FIRST arm and never again, so a neighbour returning mid-phase OOMs everything
  # after it. Measured 2026-09-15: Evo2 phase D lost 10 of 12 arms to CUDA OOM when the other
  # user's job came back to 38.9 GB while this phase and a GenomeOcean phase were both live.
  # g9_alpha already learned this (its guard is per alpha rung, for exactly the same reason);
  # phase D had the once-per-phase version. Each arm reloads the model in a new process, so a
  # per-arm check costs nothing and turns an OOM into a wait.
  wait_gpu $MIB "phaseD $SUB I1:$C"
  step "I1:${SUB}:$C"        $PY -m bgcbench.run.arm --arm ${SUB}_I1_${C}${SUF} "${base[@]}"
  # the control: SAME magnitude, random direction. Without it a positive is uninterpretable —
  # any vector at this alpha perturbs generation, so the question is whether the DIRECTION'S
  # CONTENT does the work (FINDINGS 20).
  wait_gpu $MIB "phaseD $SUB I1rand:$C"
  step "I1rand:${SUB}:$C"    $PY -m bgcbench.run.arm --arm ${SUB}_I1rand_${C}${SUF} "${base[@]}" --random-direction 0
  # composition: does steering add anything ON TOP of seeding, or only substitute for it?
  wait_gpu $MIB "phaseD $SUB I1xS1:$C"
  step "I1xS1:${SUB}:$C"     $PY -m bgcbench.run.arm --arm ${SUB}_I1xS1_${C}${SUF} "${base[@]}" --seeded
done
echo "PHASE-D DONE $SUB $(date -Is)" >> $S
