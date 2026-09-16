#!/bin/bash
# Re-sweep ONE class over an EXTENDED alpha grid, for an arm whose whole grid failed.
#   usage: extend_alpha.sh <substrate-id> <suffix> <CLASS> <alpha> [<alpha> ...]
#
# ⚠ THIS IS NOT ENDPOINT TUNING (§2.4). g9_alpha reads coding density, self-NLL and hit-EOS
# and refuses to look at `detected`/`on_target`; `endpoint_fields_read` stays []. Extending
# the grid downward when EVERY rung fails is locating the arm's working range, not searching
# for a better benchmark number -- nothing here can see the benchmark number.
#
# ⚠ WHY SOME ARMS NEED IT, MEASURED. The directions are unit-norm PER SITE and the sites act
# in series, so the total push is roughly alpha x (number of steered sites). On GenomeOcean at
# the same grid: ARYLPOLYENE steers 1 site and its coding density is untouched at every rung
# (drops 0.010/0.008/-0.001/0.003); TERPENE steers 8 and fails every rung (0.349/0.629/0.672/
# 0.661). Same grid, same substrate, opposite verdicts -- the grid was never calibrated for an
# 8- or 24-site arm. SPEC §14A makes alpha a per-ARM treatment parameter, so the fix is the
# arm's own grid, not a shared one.
#
# ⚠ THE TOP RUNG IS LEFT ALONE ON PURPOSE. Rungs are only ADDED below the existing grid, so
# max_alpha is unchanged and the §6.4 checks already written into the direction artifact --
# which are read at max_alpha -- stay valid. Lowering the ceiling would invalidate them and
# force phases B and B2 to re-run.
#
# Existing rungs are reused from disk (g9_alpha skips a run dir that already exists), so only
# the new alphas generate.
set -u
SUB=${1:?substrate}; SUF=${2:?suffix}; C=${3:?class}; shift 3
ALPHAS="$@"
P=/data2/ds85/bgcbench/work/pipeline
S=$P/extend_${SUB}_${C}.status
# ⚠ `step` (from _lib.sh) appends command output to $L. This script never defined it, so
# under `set -u` every step died instantly with an unbound-variable error and reported
# rc=1 without running anything -- both alpha extensions "failed" in 0 seconds.
L=$P/extend_${SUB}_${C}.log
source $P/_lib.sh
PFX=$(cfgval $SUB prefix)
DIR=$D/${SUB}_I1_${C}_W2${SUF}.pt
echo "EXTEND $SUB $C alphas=[$ALPHAS] $(date -Is)" > $S
[ -f "$DIR" ] || { echo "  ⛔ no direction $DIR" >> $S; exit 2; }
SITES=$($PY -c "
import json,glob
f=sorted(glob.glob('/data2/ds85/bgcbench/g9sites/${SUB}_${C}_W2${SUF}_${SUB}_${C}.json'))
print(' '.join(str(i) for i in (json.load(open(f[0])).get('chosen_sites') or [])) if f else '')")
[ -n "$SITES" ] || { echo "  ⛔ no chosen_sites" >> $S; exit 2; }
echo "  sites=$SITES $(date -Is)" >> $S
NEED=$($PY -c "
import sys; sys.path.insert(0,'$REPO')
from bgcbench.model.substrate_config import gpu_mib
print(gpu_mib('$(fam $SUB)'))")
wait_gpu $NEED "extend $SUB $C"
step "extend:${SUB}:$C" $PY -m bgcbench.run.g9_alpha \
  --substrate $SUB --direction $DIR --adapter $A/${SUB}_W2_${C}${SUF} \
  --row-class $C --prefix $PFX --n 50 --tag ${SUB}_${C}_W2${SUF} \
  --sites $SITES --alphas $ALPHAS
$PY -c "
import json
d=json.load(open('$G9/${SUB}_${C}_W2${SUF}_${C}.json'))
print('  CHOSEN ALPHA:', d['chosen_alpha'], ' grid:', d['alphas'])
for r in d.get('rows') or []:
    if r.get('rc')==0:
        print(f\"    a={r['alpha']:<7} coding={r.get('median_coding_density')} drop={r.get('rel_coding_drop')} adm={r.get('admissible')}\")" >> $S 2>&1
echo "EXTEND DONE $SUB $C $(date -Is)" >> $S
