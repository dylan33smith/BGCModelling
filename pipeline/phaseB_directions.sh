#!/bin/bash
# PHASE B — I1 DIRECTIONS, derived on the arm's OWN new weights.
#   usage: phaseB_directions.sh <substrate-id> <adapter-suffix> <gpu-mib>
#
# ⚠ A DIRECTION IS KEYED TO A WEIGHT STATE. Every direction on disk was derived on the
# pre-2026-09-14 adapters and is stale; SPEC 6.4 requires the direction to come from the model
# the arm actually runs. Nothing here reuses them, and nothing is shared across substrates:
# GenomeOcean's directions cannot be Evo2's (different hidden size, different residual stream).
set -u
SUB=$1; SUF=$2; MIB=$3
S=/data2/ds85/bgcbench/work/pipeline/phaseB_${SUB}.status
L=/data2/ds85/bgcbench/work/pipeline/phaseB_${SUB}.log
source /data2/ds85/bgcbench/work/pipeline/_lib.sh
PFX=$(cfgval $SUB prefix)
echo "PHASE-B START $SUB prefix=$PFX $(date -Is)" > $S
require_adapters $(for C in $CLASSES; do echo "$A/${SUB}_W2_${C}${SUF}"; done)
wait_gpu $MIB "phaseB $SUB"

for C in $CLASSES; do
  step "dir:${SUB}:$C" $PY -m bgcbench.run.derive_directions \
    --substrate $SUB --target $C --prefix $PFX \
    --adapter $A/${SUB}_W2_${C}${SUF} --name ${SUB}_I1_${C}_W2${SUF}
done
# ⚠ THIS READ IS PRELIMINARY AND IS **NOT** THE GATE. It describes the ALL-SITES
# configuration; phase B2 chooses the injection site subset and re-reads the check there, and
# that read is what `attach_direction` gates on. A FAIL here on a substrate with many
# attention sites is expected and uninformative -- the criteria are order statistics over site
# count, so a 24-layer stack fails them at direction quality a 4-site one passes.
echo "--- manipulation checks (PRELIMINARY, all sites -- phase B2 re-reads at the chosen subset) ---" >> $S
for C in $CLASSES; do
  $PY -c "
import json,sys
try:
    d=json.load(open('$D/${SUB}_I1_${C}_W2${SUF}.json'))
    print(f'  $C  check_all_pass={d.get(\"check_all_pass\")}')
except Exception as e: print(f'  $C  NO ARTIFACT ({e})')" >> $S
done
echo "PHASE-B DONE $SUB $(date -Is)" >> $S
