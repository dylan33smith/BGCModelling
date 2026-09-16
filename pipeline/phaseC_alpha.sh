#!/bin/bash
# PHASE C — G9: the steering magnitude, PER CLASS and PER WEIGHT STATE (SPEC 12.A5).
#   usage: phaseC_alpha.sh <substrate-id> <adapter-suffix> <gpu-mib>
#
# ⚠ THE CRITERION IS GENERATION HEALTH, NEVER THE ENDPOINT (SPEC 2.4). g9_alpha reads
# per-token NLL and coding density and refuses to look at `detected`/`on_target`.
# ⚠ ALPHA DOES NOT TRANSFER -- not across classes (12.A5) and not across substrates (14A).
set -u
SUB=$1; SUF=$2; MIB=$3
S=/data2/ds85/bgcbench/work/pipeline/phaseC_${SUB}.status
L=/data2/ds85/bgcbench/work/pipeline/phaseC_${SUB}.log
source /data2/ds85/bgcbench/work/pipeline/_lib.sh
PFX=$(cfgval $SUB prefix)
echo "PHASE-C START $SUB $(date -Is)" > $S
wait_gpu $MIB "phaseC $SUB"

for C in $CLASSES; do
  DIR=$D/${SUB}_I1_${C}_W2${SUF}.pt
  if [ ! -f "$DIR" ]; then echo "  SKIP $C — no direction (phase B failed?)" >> $S; continue; fi
  # ⚠ STEER AT THE SITES PHASE B2 CHOSE. Without this the alpha sweep runs at the default
  # every-site subset, which is a DIFFERENT intervention from the one phase D will run --
  # and phase B2's entire justification is that the two must match.
  SITES=$($PY -c "
import json,glob
f=sorted(glob.glob('/data2/ds85/bgcbench/g9sites/${SUB}_${C}_W2${SUF}_${SUB}_${C}.json'))
d=json.load(open(f[0])) if f else {}
s=d.get('chosen_sites') or []
print(' '.join(str(i) for i in s))" 2>/dev/null)
  if [ -z "$SITES" ]; then
    echo "  ⛔ $C — no chosen_sites from phase B2; refusing to sweep alpha at the default" >> $S
    echo "     site subset, which would measure a different intervention than phase D runs." >> $S
    continue
  fi
  echo "  $C sites: $SITES" >> $S
  step "g9:${SUB}:$C" $PY -m bgcbench.run.g9_alpha \
    --substrate $SUB --direction $DIR --adapter $A/${SUB}_W2_${C}${SUF} \
    --row-class $C --prefix $PFX --n 50 --tag ${SUB}_${C}_W2${SUF} --sites $SITES
done
echo "--- selected alphas ---" >> $S
for C in $CLASSES; do
  $PY -c "
import json,glob
f=glob.glob('$G9/*${SUB}_${C}_W2${SUF}*.json')
print(f'  $C  ' + (str(json.load(open(f[0])).get('chosen_alpha')) if f else 'NO ARTIFACT'))" >> $S
done
echo "PHASE-C DONE $SUB $(date -Is)" >> $S
