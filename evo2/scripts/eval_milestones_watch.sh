#!/usr/bin/env bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────────────────────
# eval_milestones_watch.sh — build a per-checkpoint quick_eval trajectory.
#
#   evo2/scripts/eval_milestones_watch.sh <run-dir> [options]
#
# Watches <run-dir>/checkpoints for milestone step_N checkpoints and runs
# evo2/scripts/quick_eval.sh on each, appending one row per checkpoint to a single
# master eval_track.jsonl so you can see how is_bgc / correct_class /
# class_markers / any_domain_rate / coding_density change across training.
#
# SINGLE-GPU SAFE (post-hoc, idle-gated). generate_bgc needs the GPU and loads
# its own copy of Evo2 7B, so this NEVER co-runs with training: each eval only
# launches after the GPU has been continuously idle (no compute process, free
# memory >= threshold) for --idle-hold-sec. In practice that means the sweep
# runs once training finishes (or during a long preemption gap). The idle hold
# defaults to 300 s so a brief training auto-resume gap does NOT steal the GPU.
#
# All intermediate checkpoints are kept (training launched with
# --keep-last-ckpts 0), so the full trajectory is always recoverable here.
# ─────────────────────────────────────────────────────────────────────────────

usage() {
  cat <<'EOF'
Usage:
  evo2/scripts/eval_milestones_watch.sh <run-dir> [options]

Options:
  --stride N            Evaluate step_N checkpoints where N % stride == 0 (default: 200).
                        The newest checkpoint is always included regardless of stride.
  --eval-root PATH      Output root (default: <run-dir>/quick_eval_milestones).
  --min-free-mib N      Minimum free GPU MiB required before an eval (default: 70000).
  --idle-hold-sec N     Continuous idle time required before an eval (default: 300).
  --check-every-sec N   Poll interval while waiting (default: 120).
  --gpu-index N         GPU index to check (default: 0).
  --once                Evaluate all currently-pending milestones, then exit
                        (do not keep watching for new checkpoints).
  -h, --help            Show this help.

Notes:
  - quick_eval knobs pass through via env: PER_CLASS, MAX_NEW, CLASSES, SEED, etc.
  - The master trajectory is <eval-root>/eval_track.jsonl (one row per checkpoint).
    View it sorted by step with:
      jq -s 'sort_by(.step)[] | {step,is_bgc,correct_class,class_markers,any_domain_rate,coding_density}' \
        <eval-root>/eval_track.jsonl
EOF
}

RUN_DIR=""
STRIDE=200
EVAL_ROOT=""
MIN_FREE_MIB=70000
IDLE_HOLD_SEC=300
CHECK_EVERY_SEC=120
GPU_INDEX=0
ONCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stride)          STRIDE="${2:?missing value}"; shift 2 ;;
    --eval-root)       EVAL_ROOT="${2:?missing value}"; shift 2 ;;
    --min-free-mib)    MIN_FREE_MIB="${2:?missing value}"; shift 2 ;;
    --idle-hold-sec)   IDLE_HOLD_SEC="${2:?missing value}"; shift 2 ;;
    --check-every-sec) CHECK_EVERY_SEC="${2:?missing value}"; shift 2 ;;
    --gpu-index)       GPU_INDEX="${2:?missing value}"; shift 2 ;;
    --once)            ONCE=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    -*)                echo "Unknown option: $1" >&2; usage; exit 2 ;;
    *)                 if [[ -z "$RUN_DIR" ]]; then RUN_DIR="$1"; shift;
                       else echo "Unexpected arg: $1" >&2; exit 2; fi ;;
  esac
done

[[ -n "$RUN_DIR" ]] || { echo "Missing <run-dir>." >&2; usage; exit 2; }
CKPT_ROOT="$RUN_DIR/checkpoints"
[[ -d "$CKPT_ROOT" ]] || { echo "No checkpoints dir under: $RUN_DIR" >&2; exit 2; }
EVAL_ROOT="${EVAL_ROOT:-$RUN_DIR/quick_eval_milestones}"
mkdir -p "$EVAL_ROOT"
MASTER="$EVAL_ROOT/eval_track.jsonl"
LOG="$EVAL_ROOT/watch.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUICK_EVAL="$SCRIPT_DIR/quick_eval.sh"
[[ -x "$QUICK_EVAL" || -f "$QUICK_EVAL" ]] || { echo "quick_eval.sh not found at $QUICK_EVAL" >&2; exit 2; }

log() { printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

get_proc_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF>0' | wc -l | tr -d ' '
}
get_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | sed -n "$((GPU_INDEX + 1))p" | tr -d ' '
}
# Block until the GPU is continuously idle (no compute proc + enough free mem)
# for IDLE_HOLD_SEC. This is what keeps the sweep from ever competing with the
# training run.
wait_for_gpu_idle() {
  local idle_start=0 now proc_count free_mib
  log "Waiting for GPU ${GPU_INDEX} idle: proc_count=0, free_mib>=${MIN_FREE_MIB}, hold=${IDLE_HOLD_SEC}s"
  while true; do
    proc_count="$(get_proc_count)"; free_mib="$(get_free_mib)"; now="$(date +%s)"
    if [[ -z "$free_mib" ]]; then
      idle_start=0; sleep "$CHECK_EVERY_SEC"; continue
    fi
    if [[ "$proc_count" -eq 0 && "$free_mib" -ge "$MIN_FREE_MIB" ]]; then
      [[ "$idle_start" -eq 0 ]] && { idle_start="$now"; log "GPU idle (free=${free_mib} MiB). Holding ${IDLE_HOLD_SEC}s."; }
      (( now - idle_start >= IDLE_HOLD_SEC )) && { log "Idle window confirmed."; return 0; }
    else
      idle_start=0
    fi
    sleep "$CHECK_EVERY_SEC"
  done
}

# Is training still active? (a deepspeed/finetune compute process exists, or the
# launcher has not logged completion). Used only to decide when to stop watching.
training_finished() {
  if pgrep -f "finetune_evo2_lora.py" >/dev/null 2>&1; then return 1; fi
  if [[ -f "$RUN_DIR/production.log" ]] && grep -q "PRODUCTION RUN COMPLETE" "$RUN_DIR/production.log"; then
    return 0
  fi
  # No training process and no explicit completion marker: treat as finished
  # (covers manual stops). Conservative: only after we've also drained pending.
  return 0
}

# Emit the list of checkpoint step dirs to evaluate, oldest step first:
#   - step_N where N % STRIDE == 0
#   - plus the newest step_N (final state), always
# Skips ones already evaluated (their out dir has quick_eval.json).
pending_checkpoints() {
  local d base step newest_step="" newest_dir=""
  declare -A want=()
  for d in "$CKPT_ROOT"/step_*; do
    [[ -d "$d/adapter" ]] || continue
    base="$(basename "$d")"
    # only pure numeric step_N (skip step_N_final / _oom / _interrupted)
    [[ "$base" =~ ^step_([0-9]+)$ ]] || continue
    step="${BASH_REMATCH[1]}"
    if [[ -z "$newest_step" || "$step" -gt "$newest_step" ]]; then newest_step="$step"; newest_dir="$d"; fi
    if (( step % STRIDE == 0 )); then want["$step"]="$d"; fi
  done
  [[ -n "$newest_dir" ]] && want["$newest_step"]="$newest_dir"
  # sort by step asc, drop already-done
  local s
  for s in $(printf '%s\n' "${!want[@]}" | sort -n); do
    [[ -f "$EVAL_ROOT/step_${s}/quick_eval.json" ]] && continue
    printf '%s\t%s\n' "$s" "${want[$s]}"
  done
}

log "════════════════════════════════════════════════════════════"
log "Milestone quick-eval watcher"
log "  Run dir:    $RUN_DIR"
log "  Stride:     every ${STRIDE} steps (+ newest checkpoint)"
log "  Eval root:  $EVAL_ROOT"
log "  Idle gate:  proc=0, free>=${MIN_FREE_MIB} MiB, hold=${IDLE_HOLD_SEC}s"
log "  Mode:       $([[ "$ONCE" -eq 1 ]] && echo once || echo continuous)"
log "════════════════════════════════════════════════════════════"

empty_scans=0
while true; do
  mapfile -t PENDING < <(pending_checkpoints)

  if [[ "${#PENDING[@]}" -eq 0 ]]; then
    if [[ "$ONCE" -eq 1 ]]; then log "No pending milestones; --once set. Done."; exit 0; fi
    if training_finished; then
      empty_scans=$(( empty_scans + 1 ))
      # require two consecutive empty scans after training to avoid racing a
      # just-written final checkpoint
      if [[ "$empty_scans" -ge 2 ]]; then
        log "Training finished and all milestones evaluated. Trajectory: $MASTER"
        exit 0
      fi
    else
      empty_scans=0
    fi
    sleep "$CHECK_EVERY_SEC"
    continue
  fi
  empty_scans=0

  # take the oldest pending checkpoint
  IFS=$'\t' read -r STEP CKPT <<< "${PENDING[0]}"
  log "Next milestone: step ${STEP}  ($CKPT)  [${#PENDING[@]} pending]"

  wait_for_gpu_idle

  OUT="$EVAL_ROOT/step_${STEP}"
  log "Running quick_eval on step ${STEP} → $OUT"
  set +e
  bash "$QUICK_EVAL" "$CKPT" "$OUT" >>"$EVAL_ROOT/step_${STEP}.log" 2>&1
  RC=$?
  set -e
  if [[ "$RC" -ne 0 ]]; then
    log "  quick_eval FAILED for step ${STEP} (rc=$RC); see $EVAL_ROOT/step_${STEP}.log. Will retry next scan."
    # leave it pending (no quick_eval.json) so it retries; brief backoff
    sleep "$CHECK_EVERY_SEC"
    continue
  fi
  if [[ -f "$OUT/eval_track.jsonl" ]]; then
    cat "$OUT/eval_track.jsonl" >> "$MASTER"
    log "  step ${STEP} done → appended to $MASTER"
    log "  $(tail -1 "$OUT/eval_track.jsonl")"
  else
    log "  step ${STEP} produced no eval_track.jsonl (unexpected); see $OUT."
  fi
done
