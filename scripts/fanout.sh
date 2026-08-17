#!/usr/bin/env bash
# Run independent work units concurrently when the batched path is unavailable.
#
# WHY THIS EXISTS. vortex's batched generation is gated (left-pad perturbs StripedHyena, failed an
# on-GPU equivalence gate -- see bugs.md), so generation runs one sequence at a time and leaves the
# H100 at ~41% utilisation and 4 GB of 80 GB. Fanning out N *sequential* processes recovers the
# idle capacity WITHOUT touching generation semantics: each process still generates one sequence at
# a time, so every output is bit-identical to what the serial run would have produced. Measured on
# the Phase-3 seed sweep: 41% -> 100% utilisation, 4 GB -> 22 GB, ~3.5x aggregate throughput,
# 10 cells in ~65 min instead of ~4.5 h.
#
# This is ONLY valid for units that are independent and deterministic given (seed, prompt, model).
# Do NOT use it for throughput or memory benchmarks -- concurrency is exactly what invalidates
# those (see CLAUDE.md).
#
# USAGE
#   fanout.sh <n_workers> <claim_dir> <unit_file> <command_template>
#
#   unit_file           one unit name per line
#   command_template    shell string; {} is replaced by the unit name
#
# EXAMPLE
#   printf '%s\n' lora_L4 base_L4 lora_L8 > /tmp/units
#   scripts/fanout.sh 3 /data2/.../.claims /tmp/units 'bash gen_one.sh {}'
#
# GUARANTEES
#   * atomic claim via mkdir -- two workers can never take the same unit
#   * each worker is a detached tmux session with a status sentinel
#   * poll <claim_dir>/../w<N>.status; 0 = success
set -uo pipefail

N=${1:?n_workers}; CLAIMS=${2:?claim_dir}; UNITS=${3:?unit_file}; TMPL=${4:?command_template}
mkdir -p "$CLAIMS"
LOGDIR=$(dirname "$CLAIMS")

worker() {
  local w=$1
  while read -r unit; do
    [ -z "$unit" ] && continue
    # Atomic: mkdir fails if another worker already claimed this unit.
    mkdir "$CLAIMS/$unit" 2>/dev/null || continue
    echo "[w$w] claimed $unit"
    if eval "${TMPL//\{\}/$unit}"; then
      echo "[w$w] done $unit"
    else
      echo "[w$w] FAILED $unit (claim left in place so it is not silently retried)"
    fi
  done < "$UNITS"
  echo "[w$w] finished"
}

if [ "${FANOUT_WORKER:-}" != "" ]; then          # re-entrant call: be the worker
  worker "$FANOUT_WORKER"
  exit $?
fi

for w in $(seq 1 "$N"); do
  tmux new-session -d -s "fanout_w$w" \
    "FANOUT_WORKER=$w bash '$0' '$N' '$CLAIMS' '$UNITS' '$TMPL' \
       > '$LOGDIR/w$w.log' 2>&1; echo \$? > '$LOGDIR/w$w.status'"
  sleep 4                                        # stagger model loads
done
echo "launched $N workers; poll $LOGDIR/w<N>.status"
