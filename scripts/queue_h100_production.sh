#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 PRODUCTION launcher (H100, shared GPU).
#
# Default: a FRESH run on the curated, leakage-free split, with all the
# post-audit corrections baked in:
#   - curated data (no leakage, no N / no contig-edge, class-balanced)   [C1/C2]
#   - chunk mode + EOS + continuation prefixes (long-seq support)        [M11]
#   - first-window validation + early stopping                          [M2/M7]
#   - auto-resume on OOM/preemption + grad-accum-aligned resume          [C6/M1]
#   - emergency-checkpoint rotation (won't refill the disk)              [m3]
#
# AUTO-RESUME: if the run dies on a non-interrupt error mid-training (e.g.
# another user grabs the shared GPU and OOMs us), it re-waits for the GPU to go
# idle and relaunches from the newest checkpoint, up to --max-retries times.
# A manual Ctrl-C (exit 130) is respected and NOT auto-resumed.
#
# RESUME a stopped/crashed run manually with:
#   scripts/queue_h100_production.sh --resume-from <run-dir-or-checkpoint-dir>
# ─────────────────────────────────────────────────────────────────────────────

usage() {
  cat <<'EOF'
Usage:
  scripts/queue_h100_production.sh [options]

Launches the Phase 1 production run on the curated leakage-free split. Default is
a fresh start; pass --resume-from to continue a stopped/crashed run. The run
auto-resumes itself from the newest checkpoint if it dies on a non-interrupt
error (shared-GPU OOM/preemption).

Options:
  --resume-from PATH      Resume a run: a checkpoint dir (has adapter/) OR a run
                          dir (has checkpoints/ — newest checkpoint is used).
  --max-seq-len N         Sequence length (default: 32768).
  --batch-size N          Micro-batch size (default: 1).
  --grad-accum N          Gradient accumulation steps (default: 128).
  --lr FLOAT              Learning rate (default: 5e-5).
  --warmup-steps N        LR warmup steps (default: 50).
  --max-epochs N          Epoch ceiling; early stopping usually cuts it (default: 6).
  --val-every N           Validate every N steps (default: 50).
  --save-every N          Checkpoint every N steps (default: 50).
  --early-stopping-patience N   Stop after N validations with no improvement (default: 4; 0 disables).
  --early-stopping-min-delta F  Min val-loss decrease counted as improvement (default: 0.001).
  --max-retries N         Auto-resume attempts after a non-interrupt failure (default: 10; 0 disables).
  --retry-backoff-sec N   Seconds to wait before each auto-resume attempt (default: 120).
  --check-every-sec N     Poll interval while waiting for idle GPU (default: 20).
  --idle-hold-sec N       Continuous idle time required before launch (default: 60).
  --min-free-mib N        Minimum free MiB required (default: 78000).
  --gpu-index N           GPU index to check (default: 0).
  --output-root PATH      Root output dir (default: /data2/ds85/bgcmodel_runs).
  --hf-home PATH          HF cache path (default: /data2/ds85/hf_cache).
  --env-name NAME         micromamba env name (default: bgcmodel).
  --train PATH            Train JSONL (default: curated split on /data2).
  --val PATH              Val JSONL (default: curated split on /data2).
  --wandb-mode MODE       WandB mode: online|offline|disabled (default: online).
  --wandb-project NAME    WandB project name (default: bcg-evo2-phase1).
  --dry-run               Print the command without running.
  -h, --help              Show this help.
EOF
}

# ── Defaults ──────────────────────────────────────────────────────────────
RESUME_CKPT=""
MAX_SEQ_LEN=32768
BATCH_SIZE=1
GRAD_ACCUM=128
LR="5e-5"
# Tuned for the curated ~18K set (~225 optimizer steps/epoch in chunk mode):
# short warmup, frequent validation, and a high epoch ceiling that early
# stopping cuts off once first-window val loss plateaus.
WARMUP_STEPS=50
MAX_EPOCHS=6
VAL_EVERY=50
SAVE_EVERY=50
EARLY_STOPPING_PATIENCE=4
EARLY_STOPPING_MIN_DELTA="0.001"
KEEP_SPECIAL_CKPTS=2          # rotate oom/interrupted/final checkpoints (m3)
MAX_RETRIES=10
RETRY_BACKOFF_SEC=120
CHECK_EVERY_SEC=20
IDLE_HOLD_SEC=60
MIN_FREE_MIB=78000
GPU_INDEX=0
OUTPUT_ROOT="/data2/ds85/bgcmodel_runs"
HF_HOME_PATH="/data2/ds85/hf_cache"
ENV_NAME="bgcmodel"
# Curated, leakage-free split (scripts/curate_dataset.py). The old record-level
# data/processed/splits_combined/ had 94.6% genome overlap and was removed.
TRAIN_JSONL="/data2/ds85/bgcmodel_data/splits_curated/train.jsonl"
VAL_JSONL="/data2/ds85/bgcmodel_data/splits_curated/val.jsonl"
WANDB_MODE="online"
WANDB_PROJECT="bcg-evo2-phase1"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume-from)      RESUME_CKPT="${2:?missing value}"; shift 2 ;;
    --max-seq-len)      MAX_SEQ_LEN="${2:?missing value}"; shift 2 ;;
    --batch-size)       BATCH_SIZE="${2:?missing value}"; shift 2 ;;
    --grad-accum)       GRAD_ACCUM="${2:?missing value}"; shift 2 ;;
    --lr)               LR="${2:?missing value}"; shift 2 ;;
    --warmup-steps)     WARMUP_STEPS="${2:?missing value}"; shift 2 ;;
    --max-epochs)       MAX_EPOCHS="${2:?missing value}"; shift 2 ;;
    --val-every)        VAL_EVERY="${2:?missing value}"; shift 2 ;;
    --save-every)       SAVE_EVERY="${2:?missing value}"; shift 2 ;;
    --early-stopping-patience)  EARLY_STOPPING_PATIENCE="${2:?missing value}"; shift 2 ;;
    --early-stopping-min-delta) EARLY_STOPPING_MIN_DELTA="${2:?missing value}"; shift 2 ;;
    --max-retries)      MAX_RETRIES="${2:?missing value}"; shift 2 ;;
    --retry-backoff-sec) RETRY_BACKOFF_SEC="${2:?missing value}"; shift 2 ;;
    --check-every-sec)  CHECK_EVERY_SEC="${2:?missing value}"; shift 2 ;;
    --idle-hold-sec)    IDLE_HOLD_SEC="${2:?missing value}"; shift 2 ;;
    --min-free-mib)     MIN_FREE_MIB="${2:?missing value}"; shift 2 ;;
    --gpu-index)        GPU_INDEX="${2:?missing value}"; shift 2 ;;
    --output-root)      OUTPUT_ROOT="${2:?missing value}"; shift 2 ;;
    --hf-home)          HF_HOME_PATH="${2:?missing value}"; shift 2 ;;
    --env-name)         ENV_NAME="${2:?missing value}"; shift 2 ;;
    --train)            TRAIN_JSONL="${2:?missing value}"; shift 2 ;;
    --val)              VAL_JSONL="${2:?missing value}"; shift 2 ;;
    --wandb-mode)       WANDB_MODE="${2:?missing value}"; shift 2 ;;
    --wandb-project)    WANDB_PROJECT="${2:?missing value}"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────

# Newest checkpoint dir (by mtime) under <run_dir>/checkpoints that has adapter/.
# Covers periodic (step_N) and emergency (step_N_oom/_interrupted/_final) saves.
find_latest_checkpoint() {
  local root="$1/checkpoints" d
  [[ -d "$root" ]] || return 0
  for d in $(ls -1dt "$root"/step_* 2>/dev/null); do
    if [[ -d "$d/adapter" ]]; then printf '%s' "$d"; return 0; fi
  done
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required command not found: $1" >&2; exit 2; }
}

# ── Sanity checks ─────────────────────────────────────────────────────────
require_cmd nvidia-smi
require_cmd micromamba
if ! micromamba run -n "$ENV_NAME" which deepspeed >/dev/null 2>&1; then
  echo "deepspeed not installed in micromamba env '${ENV_NAME}'." >&2
  exit 2
fi
[[ -f "$TRAIN_JSONL" ]] || { echo "Train JSONL not found: $TRAIN_JSONL" >&2; exit 2; }
[[ -f "$VAL_JSONL"   ]] || { echo "Val JSONL not found: $VAL_JSONL"   >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT" "$HF_HOME_PATH"

# ── Resolve resume target + run dir ───────────────────────────────────────
# Default (no --resume-from) = fresh run in a new timestamped dir. With
# --resume-from, accept a checkpoint dir or a run dir, and reuse that run dir so
# logs/checkpoints stay together.
RESUME_FROM=""
if [[ -n "$RESUME_CKPT" ]]; then
  if [[ -d "${RESUME_CKPT}/adapter" ]]; then
    RESUME_FROM="$RESUME_CKPT"
    RUN_DIR="$(dirname "$(dirname "$RESUME_FROM")")"
  elif [[ -d "${RESUME_CKPT}/checkpoints" ]]; then
    RESUME_FROM="$(find_latest_checkpoint "$RESUME_CKPT")"
    [[ -n "$RESUME_FROM" ]] || { echo "No checkpoint with adapter/ in ${RESUME_CKPT}/checkpoints" >&2; exit 2; }
    RUN_DIR="$RESUME_CKPT"
  else
    echo "Cannot resolve --resume-from: $RESUME_CKPT" >&2
    echo "Pass a checkpoint dir (with adapter/) or a run dir (with checkpoints/)." >&2
    exit 2
  fi
else
  RUN_DIR="${OUTPUT_ROOT}/phase1_lora_prod_$(date +%Y%m%d_%H%M%S)_L${MAX_SEQ_LEN}"
fi
mkdir -p "$RUN_DIR"
META_LOG="${RUN_DIR}/queue.log"

log() { printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "$META_LOG"; }

get_proc_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF>0' | wc -l | tr -d ' '
}
get_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | sed -n "$((GPU_INDEX + 1))p" | tr -d ' '
}
wait_for_gpu_idle() {
  local idle_start=0 now proc_count free_mib
  log "Waiting for GPU ${GPU_INDEX} idle: proc_count=0, free_mib>=${MIN_FREE_MIB}, hold=${IDLE_HOLD_SEC}s"
  while true; do
    proc_count="$(get_proc_count)"; free_mib="$(get_free_mib)"; now="$(date +%s)"
    if [[ -z "$free_mib" ]]; then
      log "Could not read GPU ${GPU_INDEX} free memory; retrying in ${CHECK_EVERY_SEC}s."
      idle_start=0; sleep "$CHECK_EVERY_SEC"; continue
    fi
    if [[ "$proc_count" -eq 0 && "$free_mib" -ge "$MIN_FREE_MIB" ]]; then
      [[ "$idle_start" -eq 0 ]] && { idle_start="$now"; log "GPU idle (free=${free_mib} MiB). Holding."; }
      (( now - idle_start >= IDLE_HOLD_SEC )) && { log "Idle window confirmed."; return 0; }
    else
      idle_start=0
      log "GPU busy: proc_count=${proc_count}, free_mib=${free_mib}; recheck in ${CHECK_EVERY_SEC}s."
    fi
    sleep "$CHECK_EVERY_SEC"
  done
}

# ── Run plan ──────────────────────────────────────────────────────────────
log "══════════════════════════════════════════════════════════════"
log "Phase 1 PRODUCTION RUN — L=${MAX_SEQ_LEN}"
log "  Output:     ${RUN_DIR}"
log "  Mode:       $([[ -n "$RESUME_FROM" ]] && echo "resume from ${RESUME_FROM}" || echo "fresh start")"
log "  Train data: ${TRAIN_JSONL}"
log "  Val data:   ${VAL_JSONL}"
log "  Epochs:     ${MAX_EPOCHS} max  (val-every=${VAL_EVERY}, save-every=${SAVE_EVERY})"
log "  EarlyStop:  patience=${EARLY_STOPPING_PATIENCE} min-delta=${EARLY_STOPPING_MIN_DELTA} (first-window val loss)"
log "  Batch:      ${BATCH_SIZE} x grad-accum ${GRAD_ACCUM} = effective batch $(( BATCH_SIZE * GRAD_ACCUM ))"
log "  LR:         ${LR}  (warmup=${WARMUP_STEPS} steps)"
log "  AutoResume: up to ${MAX_RETRIES} retries, ${RETRY_BACKOFF_SEC}s backoff"
log "  WandB:      ${WANDB_MODE} (project: ${WANDB_PROJECT})"
log "══════════════════════════════════════════════════════════════"

# ── DeepSpeed command (everything the corrected training path needs) ───────
DS_CMD=(
  deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py
  --train "$TRAIN_JSONL"
  --val "$VAL_JSONL"
  --output-dir "$RUN_DIR"
  --max-seq-len "$MAX_SEQ_LEN"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --lr "$LR"
  --warmup-steps "$WARMUP_STEPS"
  --max-epochs "$MAX_EPOCHS"
  --val-every "$VAL_EVERY"
  --save-every "$SAVE_EVERY"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
  --keep-special-ckpts "$KEEP_SPECIAL_CKPTS"
  --log-every 10
  --long-seq-strategy chunk
  --chunk-overlap 2048
  --eos-token
  --continuation-prefix
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --seed 42
)

RESUME_ARG=()
[[ -n "$RESUME_FROM" ]] && RESUME_ARG=(--resume-from "$RESUME_FROM")

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN — command:"
  log "  HF_HOME=$HF_HOME_PATH ${DS_CMD[*]} ${RESUME_ARG[*]}"
  log "DRY RUN complete."
  exit 0
fi

# ── Wait + launch, with auto-resume on OOM/preemption (C6) ─────────────────
attempt=0
while true; do
  attempt=$(( attempt + 1 ))
  wait_for_gpu_idle

  STARTED_AT="$(date '+%F %T')"
  if [[ ${#RESUME_ARG[@]} -gt 0 ]]; then
    log "Launch attempt ${attempt} at ${STARTED_AT} (resume: ${RESUME_ARG[1]})"
  else
    log "Launch attempt ${attempt} at ${STARTED_AT} (fresh)"
  fi

  set +e
  HF_HOME="$HF_HOME_PATH" \
    micromamba run -n "$ENV_NAME" \
    "${DS_CMD[@]}" "${RESUME_ARG[@]}" \
    2>&1 | tee -a "${RUN_DIR}/production.log"
  TRAIN_RC="${PIPESTATUS[0]}"
  set -e
  FINISHED_AT="$(date '+%F %T')"

  if [[ "$TRAIN_RC" -eq 0 ]]; then
    log "══════════════════════════════════════════════════════════════"
    log "PRODUCTION RUN COMPLETE  (finished ${FINISHED_AT})"
    log "  Output: ${RUN_DIR}   (best adapter: ${RUN_DIR}/checkpoints/best/adapter)"
    log "══════════════════════════════════════════════════════════════"
    exit 0
  fi

  if [[ "$TRAIN_RC" -eq 130 ]]; then
    log "Interrupted by user (exit 130) at ${FINISHED_AT}; not auto-resuming."
    log "  Resume later with: $0 --resume-from ${RUN_DIR}"
    exit 130
  fi

  log "Run FAILED (exit ${TRAIN_RC}) at ${FINISHED_AT}."
  LATEST_CKPT="$(find_latest_checkpoint "$RUN_DIR")"
  if [[ -z "$LATEST_CKPT" ]]; then
    log "  No checkpoint to resume from; aborting."
    exit 1
  fi
  if [[ "$MAX_RETRIES" -le 0 || "$attempt" -gt "$MAX_RETRIES" ]]; then
    log "  Auto-resume exhausted (attempt ${attempt}, max ${MAX_RETRIES})."
    log "  Resume manually with: $0 --resume-from ${RUN_DIR}"
    exit 1
  fi
  log "  Auto-resume ${attempt}/${MAX_RETRIES} from ${LATEST_CKPT}; backing off ${RETRY_BACKOFF_SEC}s."
  RESUME_ARG=(--resume-from "$LATEST_CKPT")
  sleep "$RETRY_BACKOFF_SEC"
done
