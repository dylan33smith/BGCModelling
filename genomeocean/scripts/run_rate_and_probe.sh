#!/usr/bin/env bash
# Queue: (1) n>=200 zero-shot BGC rate estimate, (2) the class linear probe on both
# GenomeOcean checkpoints. Run from the repo root.
#
# Idle-gated: gputee is shared (and a second agent session also uses it), so we wait for
# real GPU headroom before starting rather than racing another job into an OOM. Same
# convention as evo2/scripts/queue_h100_*.sh.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export MAMBA_ROOT_PREFIX=/home/ds85/.local/share/mamba
export HF_HOME=/data2/ds85/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GO="micromamba run -p /data2/ds85/envs/genomeocean python"

NUM="${NUM:-216}"                 # 9 rounds of 24
BATCH="${BATCH:-24}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"   # generation: weights ~8.5 GB + static KV ~24 GB + margin
PROBE_MIN_FREE_MIB="${PROBE_MIN_FREE_MIB:-18000}"  # probe: weights ~8.5 GB + short-seq activations
IDLE_HOLD_SEC="${IDLE_HOLD_SEC:-60}"
CHECK_EVERY="${CHECK_EVERY:-30}"
RATE_OUT="${RATE_OUT:-/data2/ds85/bgcmodel_runs/go_zeroshot_rate_n${NUM}}"

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_gpu() {
  local held=0
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$free" -ge "$MIN_FREE_MIB" ]]; then
      held=$((held + CHECK_EVERY))
      if [[ "$held" -ge "$IDLE_HOLD_SEC" ]]; then
        log "GPU has ${free} MiB free (held ${held}s) — proceeding."
        return 0
      fi
    else
      [[ "$held" -gt 0 ]] && log "GPU dropped to ${free} MiB free — resetting hold."
      held=0
    fi
    sleep "$CHECK_EVERY"
  done
}

# ---------------------------------------------------------------- 1. rate estimate
log "=== STEP 1: zero-shot BGC rate estimate, n=${NUM} ==="
log "Waiting for >= ${MIN_FREE_MIB} MiB free..."
wait_for_gpu

# --cache-implementation static is the important one: the default DynamicCache
# re-concatenates the whole KV cache every decode step and cost us ~9x on the n=24 run.
$GO genomeocean/scripts/generate_bgc_go.py \
    --num "$NUM" --preset creative_long --backend hf \
    --hf-batch-size "$BATCH" --cache-implementation static \
    --seed 20260727 \
    --out "$RATE_OUT"
rc=$?
if [[ $rc -ne 0 ]]; then log "generation FAILED (rc=$rc)"; else log "generation done -> $RATE_OUT"; fi

if [[ -s "$RATE_OUT/gen.jsonl" ]]; then
  log "Scoring with the antiSMASH gate (bgcmodel env)..."
  micromamba run -n bgcmodel python scripts/eval_suite_driver.py \
      --gen "$RATE_OUT/gen.jsonl" \
      --antismash-db /data2/ds85/antismash_db \
      --pfam-hmm /data2/ds85/pfam/Pfam-A.hmm \
      --skip-checks protein_homology kmer_novelty \
      --output "$RATE_OUT/eval.json"
  log "scoring done -> $RATE_OUT/eval.json"
fi

# ---------------------------------------------------------------- 2. linear probe
log "=== STEP 2: compound-class linear probe ==="
for spec in "bgcFM:pGenomeOcean/GenomeOcean-4B-bgcFM" "base:pGenomeOcean/GenomeOcean-4B"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  log "probe: $tag ($model)"
  # The probe needs only ~12 GB (weights 8.5 + short-sequence activations), unlike the
  # generation step's ~33 GB static KV cache. Using the generation-sized gate here made the
  # base-4B control wait indefinitely behind an unrelated 42 GB job. Gate per step.
  MIN_FREE_MIB="$PROBE_MIN_FREE_MIB" wait_for_gpu
  $GO genomeocean/scripts/class_probe_go.py \
      --model "$model" \
      --from-jsonl /data2/ds85/bgcmodel_data/splits_core/val.jsonl \
      --per-class 120 --min-class 40 --max-nt 4096 --seed 42 \
      --out "genomeocean/experiments/class_probe_${tag}.json"
  log "probe $tag done"
done

log "=== ALL DONE ==="
