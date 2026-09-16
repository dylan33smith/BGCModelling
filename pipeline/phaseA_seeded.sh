#!/bin/bash
# PHASE A — SEEDED ARMS. Depends only on trained weights, not on directions.
#   usage: phaseA_seeded.sh <substrate-id> <adapter-suffix> <gpu-mib>
#   e.g.   phaseA_seeded.sh evo2-1b _e1 28000
#
# ⚠ SEED LENGTH IS NOT PASSED. It comes from FROZEN["seed_len_nt"] = 64 and is SHARED across
# substrates because it is the TASK (SPEC 14A / 14A.1). Passing --seed-len here per substrate
# is exactly the mistake 15.6 step 5 used to mandate.
set -u
SUB=$1; SUF=$2; MIB=$3
S=/data2/ds85/bgcbench/work/pipeline/phaseA_${SUB}.status
L=/data2/ds85/bgcbench/work/pipeline/phaseA_${SUB}.log
source /data2/ds85/bgcbench/work/pipeline/_lib.sh
PFX=$(cfgval $SUB prefix)
echo "PHASE-A START $SUB prefix=$PFX suffix=$SUF $(date -Is)" > $S
require_adapters "$A/${SUB}_W1n${SUF}" $(for C in $CLASSES; do echo "$A/${SUB}_W2_${C}${SUF}"; done)
wait_gpu $MIB "phaseA $SUB"

# ⚠ PER ARM, NOT ONCE PER PHASE. The wait_gpu above is checked before the first arm
# and never again. Measured 2026-09-15: this phase lost 8 of 12 arms to CUDA OOM when
# the shared card's other user returned to 38.8 GB mid-phase while another substrate's
# job was live. Each arm reloads the model in a new process, so checking per arm costs
# nothing and turns an OOM into a wait. Same fix as phaseD_steer.sh and g9_alpha.
gen () { local N=$1; shift; wait_gpu $MIB "phaseA $SUB $N"; step "${SUB}:$N" $PY -m bgcbench.run.arm \
  --substrate $SUB --arm $N --prefix $PFX --seeded \
  --batch-size 50 --off-frozen --stage stage1 --cpus 16 "$@"; }

# W0 seeded — the control that says how much the SEED alone is worth, before any weights
gen ${SUB}_W0_S1${SUF} --train-classes $CLASSES
# pooled, nucleotide-balanced
gen ${SUB}_W1n_S1${SUF} --adapter $A/${SUB}_W1n${SUF} --train-classes $CLASSES
# per-class: the diagonal the paper reports
for C in $CLASSES; do
  gen ${SUB}_W2_${C}_S1${SUF} --adapter $A/${SUB}_W2_${C}${SUF} --row-class $C --train-classes $C
done
echo "PHASE-A DONE $SUB $(date -Is)" >> $S
