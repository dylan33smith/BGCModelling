#!/usr/bin/env bash
# Taxon confound controls for the compound-class linear probe.
#
# WHY: compound class correlates with taxonomy (Actinomycetota is high-GC and PKS-rich),
# and nucleotide composition encodes taxonomy — so a strong "class" probe may really be a
# taxon probe. The layer-0 result (balanced_acc 0.345 from pure BPE composition, vs chance
# 0.091) proves composition alone carries a large chunk of the apparent class signal.
#
# Two controls, both on bgcFM (the model that showed the effect):
#   A. probe PHYLUM from the same activations — class only means something if it beats this
#   B. probe CLASS *within* a single phylum — taxonomy held ~constant; survives => real
#
# Run from the repo root. Waits for GPU headroom like run_rate_and_probe.sh.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export MAMBA_ROOT_PREFIX=/home/ds85/.local/share/mamba
export HF_HOME=/data2/ds85/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GO="micromamba run -p /data2/ds85/envs/genomeocean python"

MODEL="${MODEL:-pGenomeOcean/GenomeOcean-4B-bgcFM}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
MIN_FREE_MIB="${MIN_FREE_MIB:-25000}"
CHECK_EVERY="${CHECK_EVERY:-30}"
OUTDIR="genomeocean/experiments"

log() { echo "[$(date '+%F %T')] $*"; }

wait_for_gpu() {
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [[ "$free" -ge "$MIN_FREE_MIB" ]] && { log "GPU ${free} MiB free — proceeding."; return 0; }
    sleep "$CHECK_EVERY"
  done
}

# Wait for any still-running probe from run_rate_and_probe.sh to clear.
while pgrep -f "class_probe_go.py --model" >/dev/null 2>&1; do
  log "waiting for the main probe chain to finish..."
  sleep 60
done

# --- Control A: can the same activations predict PHYLUM? --------------------
log "=== CONTROL A: probe PHYLUM (upper bound on the taxon confound) ==="
wait_for_gpu
$GO genomeocean/scripts/class_probe_go.py \
    --model "$MODEL" --from-jsonl "$VAL" \
    --target phylum --per-class 120 --min-class 40 --max-nt 4096 --seed 42 \
    --out "$OUTDIR/probe_bgcfm_phylum.json"

# --- Control B: class WITHIN one phylum, taxonomy held ~constant ------------
for PHY in Pseudomonadota Bacillota Actinomycetota; do
  log "=== CONTROL B: probe compound_class within ${PHY} ==="
  wait_for_gpu
  $GO genomeocean/scripts/class_probe_go.py \
      --model "$MODEL" --from-jsonl "$VAL" \
      --target compound_class --restrict-phylum "$PHY" \
      --per-class 120 --min-class 25 --max-nt 4096 --seed 42 \
      --out "$OUTDIR/probe_bgcfm_class_within_${PHY}.json"
done

log "=== TAXON CONTROLS DONE ==="
