#!/bin/bash
# PHASE A0 — DE NOVO Stage 1. The chain had NO phase that regenerated this; the frozen
# bundles are historical records of runs made under the defects, not current results.
#   usage: phaseA0_denovo.sh <substrate-id> <adapter-suffix> <gpu-mib>
set -u
SUB=$1; SUF=$2; MIB=$3
S=/data2/ds85/bgcbench/work/pipeline/phaseA0_${SUB}.status
L=/data2/ds85/bgcbench/work/pipeline/phaseA0_${SUB}.log
source /data2/ds85/bgcbench/work/pipeline/_lib.sh
PFX=$(cfgval $SUB prefix)
echo "PHASE-A0 START $SUB prefix=$PFX $(date -Is)" > $S
require_adapters "$A/${SUB}_W1n${SUF}" $(for C in $CLASSES; do echo "$A/${SUB}_W2_${C}${SUF}"; done)
wait_gpu $MIB "phaseA0 $SUB"

# ⚠ PER ARM, NOT ONCE PER PHASE. The wait_gpu above is checked before the first arm
# and never again. Measured 2026-09-15: this phase lost 8 of 12 arms to CUDA OOM when
# the shared card's other user returned to 38.8 GB mid-phase while another substrate's
# job was live. Each arm reloads the model in a new process, so checking per arm costs
# nothing and turns an OOM into a wait. Same fix as phaseD_steer.sh and g9_alpha.
gen () { local N=$1; shift; wait_gpu $MIB "phaseA0 $SUB $N"; step "${SUB}:$N" $PY -m bgcbench.run.arm \
  --substrate $SUB --arm $N --prefix $PFX \
  --batch-size 50 --off-frozen --stage stage1 --cpus 16 "$@"; }

# ⚠ W1 AND W3 ARE NOT GENERATED (2026-09-14, time). W1 was the pooled record-balanced arm;
# W1n is the pooled arm the benchmark reports and it IS here. W3 (learned per-attention-site
# conditioner) was never trained in the _fx generation, so there is no checkpoint to read.
# Both can be added later without redoing anything else.
gen ${SUB}_W0${SUF}                                                   # base, the floor
gen ${SUB}_W1n${SUF} --adapter $A/${SUB}_W1n${SUF} --train-classes $CLASSES
for C in $CLASSES; do
  gen ${SUB}_W2_${C}${SUF} --adapter $A/${SUB}_W2_${C}${SUF} --row-class $C --train-classes $C
done
echo "PHASE-A0 DONE $SUB $(date -Is)" >> $S
