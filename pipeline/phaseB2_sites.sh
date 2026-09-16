#!/bin/bash
# PHASE B2 — I1 INJECTION SITES, between directions (B) and alpha (C).
#   usage: phaseB2_sites.sh <substrate-id> <adapter-suffix> <gpu-mib>
#
# ⚠ ORDER MATTERS AND I HAD IT MISSING. The alpha sweep steers at whatever site subset is in
# force, so an alpha measured at the wrong sites is an alpha for a different intervention.
# Sites are per substrate AND per class, and every existing selection was measured on the
# pre-re-train weights.
set -u
SUB=$1; SUF=$2; MIB=$3
S=/data2/ds85/bgcbench/work/pipeline/phaseB2_${SUB}.status
L=/data2/ds85/bgcbench/work/pipeline/phaseB2_${SUB}.log
source /data2/ds85/bgcbench/work/pipeline/_lib.sh
PFX=$(cfgval $SUB prefix)
echo "PHASE-B2 START $SUB $(date -Is)" > $S
wait_gpu $MIB "phaseB2 $SUB"

for C in $CLASSES; do
  DIR=$D/${SUB}_I1_${C}_W2${SUF}.pt
  if [ ! -f "$DIR" ]; then echo "  SKIP $C — no direction" >> $S; continue; fi
  # ⚠ --finalize IS THE GATE STEP, NOT AN EXTRA. Without it the direction artifact's
  # check_all_pass still describes the all-sites configuration measured in phase B, while
  # phases C and D generate at the subset chosen here. That mismatch refused all four
  # GenomeOcean directions on 2026-09-15: TERPENE's 8-site set was admissible and sitting in
  # this script's own output while the 24-site check in the artifact said FAIL.
  step "sites:${SUB}:$C" $PY -m bgcbench.run.g9_sites \
    --substrate $SUB --direction $DIR --adapter $A/${SUB}_W2_${C}${SUF} \
    --target $C --prefix $PFX --tag ${SUB}_${C}_W2${SUF} --finalize
done
# ⚠ THIS is the manipulation check that gates generation -- read at the CHOSEN sites, on
# val_B, at this substrate's own generation alpha. Phase B's report is the preliminary
# all-sites read and is not what any arm runs at.
echo "--- manipulation checks AT THE CHOSEN SITES (the gate) ---" >> $S
for C in $CLASSES; do
  $PY -c "
import json
try:
    d=json.load(open('$D/${SUB}_I1_${C}_W2${SUF}.json'))
    cfg=d.get('check_configuration') or {}
    mc=d.get('manipulation_check') or {}
    a=d.get('check_a_monotone') or {}; b=d.get('check_b_kl') or {}
    print(f'  $C  check_all_pass={d.get(\"check_all_pass\")}  sites={cfg.get(\"site_set\")} {cfg.get(\"sites\")} alpha={cfg.get(\"alpha\")}')
    print(f'       (a) first-site monotone={a.get(\"passes\")} ({a.get(\"n_sites_monotone\")}/{a.get(\"n_sites_steered\")} steered monotone)')
    print(f'       (b) KL={b.get(\"mean_kl_nats_per_nt\")} nats/NT pass={b.get(\"passes\")}')
    print(f'       (c) mean cos={mc.get(\"mean_cosine\")} min={mc.get(\"min_active_cosine\")} pass={mc.get(\"passes\")} fold={mc.get(\"fold\")}')
except Exception as e: print(f'  $C  NO ARTIFACT ({e})')" >> $S
done
echo "PHASE-B2 DONE $SUB $(date -Is)" >> $S
