#!/usr/bin/env bash
# Chain N hopper jobs (each <=2 days) so the long-context run survives the walltime
# cap. Each job resumes the latest checkpoint; --dependency=afterany starts the next
# when the previous ends (timeout OR finish). Between segments, run a milestone eval
# (n>=15) and cancel the chain (scancel) if correct_class stays at the floor.
#   usage:  bash experiments/quartz/resubmit_chain.sh <N-segments>
set -euo pipefail
N=${1:-4}
JOB=experiments/quartz/longcontext.sbatch
[ -f "$JOB" ] || { echo "run from repo root ($JOB not found)"; exit 1; }
prev=""
for i in $(seq 1 "$N"); do
  if [ -z "$prev" ]; then prev=$(sbatch --parsable "$JOB")
  else prev=$(sbatch --parsable --dependency=afterany:"$prev" "$JOB"); fi
  echo "segment $i -> job $prev"
done
echo "Chained $N segments. Watch: squeue -u \$USER   Stop early: scancel <jobid>"
echo "Milestone eval on the latest checkpoint (separate short job):"
echo "  MAX_NEW=32768 PER_CLASS=5 evo2/scripts/quick_eval.sh \$RUN/checkpoints/step_<N> eval_out"
