#!/usr/bin/env bash
set -euo pipefail

# H1 faithful-resume verification at L=32768 using a completed audit pilot.
#
# Resumes from pilot checkpoints/step_10 (micro_step_in_epoch=1280, rng_state)
# and runs to max-steps 20 in a fresh output dir. Compares step-20 metrics
# against the uninterrupted pilot's train_log.jsonl.
#
# See FINETUNE_GUIDE.md §6 "Resume reloads (post-H1)".

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  evo2/scripts/queue_h100_h1_pilot_resume_test.sh [options]

Verifies H1 (skip-ahead + RNG) at L=32k by resuming the audit pilot from
step_10 and comparing step 20 to the original uninterrupted run.

Options:
  --pilot-dir PATH        Audit pilot run dir (default: .../pilot_L32768_audit_20260514_182735).
  --resume-from-tag TAG   Checkpoint tag under pilot/checkpoints/ (default: step_10).
  --max-steps N           Optimizer steps target (default: 20).
  --check-every-sec N     GPU idle poll interval (default: 20).
  --idle-hold-sec N       Continuous idle hold (default: 60).
  --min-free-mib N        Minimum free MiB (default: 78000).
  --gpu-index N           GPU index (default: 0).
  --output-root PATH      Run root (default: /data2/ds85/bgcmodel_runs).
  --hf-home PATH          HF cache (default: /data2/ds85/hf_cache).
  --env-name NAME         micromamba env (default: bgcmodel).
  --dry-run               Print plan only.
  -h, --help              Show help.
EOF
}

PILOT_DIR="/data2/ds85/bgcmodel_runs/pilot_L32768_audit_20260514_182735"
RESUME_TAG="step_10"
MAX_STEPS=20
CHECK_EVERY_SEC=20
IDLE_HOLD_SEC=60
MIN_FREE_MIB=78000
GPU_INDEX=0
OUTPUT_ROOT="/data2/ds85/bgcmodel_runs"
HF_HOME_PATH="/data2/ds85/hf_cache"
ENV_NAME="bgcmodel"
TRAIN_JSONL="data/processed/splits_combined/train.jsonl"
VAL_JSONL="data/processed/splits_combined/val.jsonl"
DRY_RUN=0

# Match audit pilot config.json (HEAD 54c5aa3 run).
MAX_SEQ_LEN=32768
BATCH_SIZE=1
GRAD_ACCUM=128
VAL_EVERY=10
SAVE_EVERY=99
LOG_EVERY=10
WANDB_MODE="offline"
LOSS_ATOL="1e-4"
LR_ATOL="1e-10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot-dir)        PILOT_DIR="${2:?missing value}"; shift 2 ;;
    --resume-from-tag)  RESUME_TAG="${2:?missing value}"; shift 2 ;;
    --max-steps)        MAX_STEPS="${2:?missing value}"; shift 2 ;;
    --check-every-sec)  CHECK_EVERY_SEC="${2:?missing value}"; shift 2 ;;
    --idle-hold-sec)    IDLE_HOLD_SEC="${2:?missing value}"; shift 2 ;;
    --min-free-mib)     MIN_FREE_MIB="${2:?missing value}"; shift 2 ;;
    --gpu-index)        GPU_INDEX="${2:?missing value}"; shift 2 ;;
    --output-root)      OUTPUT_ROOT="${2:?missing value}"; shift 2 ;;
    --hf-home)          HF_HOME_PATH="${2:?missing value}"; shift 2 ;;
    --env-name)         ENV_NAME="${2:?missing value}"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 2
  }
}

require_cmd nvidia-smi
require_cmd python
require_cmd micromamba

if ! micromamba run -n "$ENV_NAME" which deepspeed >/dev/null 2>&1; then
  echo "deepspeed not installed in micromamba env '${ENV_NAME}'." >&2
  exit 2
fi

RESUME_CKPT="${PILOT_DIR}/checkpoints/${RESUME_TAG}"
REF_TRAIN_LOG="${PILOT_DIR}/train_log.jsonl"

if [[ ! -d "$RESUME_CKPT" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 2
fi
if [[ ! -f "$REF_TRAIN_LOG" ]]; then
  echo "Reference train_log not found: ${REF_TRAIN_LOG}" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_JSONL" || ! -f "$VAL_JSONL" ]]; then
  echo "Train/val JSONL missing under ${REPO_ROOT}" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$HF_HOME_PATH"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/h1_pilot_resume_${RUN_TS}"
mkdir -p "$RUN_DIR"
META_LOG="${RUN_DIR}/queue.log"

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "$META_LOG"
}

get_proc_count() {
  local count
  count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF>0' | wc -l)"
  printf "%s" "${count// /}"
}

get_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | sed -n "$((GPU_INDEX + 1))p" | tr -d ' '
}

wait_for_gpu_idle() {
  local idle_start=0
  local now proc_count free_mib

  log "Waiting for GPU ${GPU_INDEX} idle: proc_count=0, free_mib>=${MIN_FREE_MIB}, hold=${IDLE_HOLD_SEC}s"
  while true; do
    proc_count="$(get_proc_count)"
    free_mib="$(get_free_mib)"
    now="$(date +%s)"

    if [[ -z "$free_mib" ]]; then
      log "Could not read free memory for GPU ${GPU_INDEX}; retrying in ${CHECK_EVERY_SEC}s."
      idle_start=0
      sleep "$CHECK_EVERY_SEC"
      continue
    fi

    if [[ "$proc_count" -eq 0 && "$free_mib" -ge "$MIN_FREE_MIB" ]]; then
      if [[ "$idle_start" -eq 0 ]]; then
        idle_start="$now"
        log "GPU appears idle (free=${free_mib} MiB). Starting hold timer."
      fi
      if (( now - idle_start >= IDLE_HOLD_SEC )); then
        log "Idle window confirmed."
        return 0
      fi
    else
      idle_start=0
      log "GPU busy: proc_count=${proc_count}, free_mib=${free_mib}; recheck in ${CHECK_EVERY_SEC}s."
    fi

    sleep "$CHECK_EVERY_SEC"
  done
}

log "══════════════════════════════════════════════════════════════"
log "H1 faithful-resume test @ L=${MAX_SEQ_LEN}"
log "  Reference pilot:  ${PILOT_DIR}"
log "  Resume from:      ${RESUME_CKPT}"
log "  Resume output:    ${RUN_DIR}"
log "  Target steps:     ${MAX_STEPS} (continues past saved step in ckpt)"
log "  Micro-batch:      bs=${BATCH_SIZE} ga=${GRAD_ACCUM}"
log "══════════════════════════════════════════════════════════════"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN — would launch:"
  log "  deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \\"
  log "    --train $TRAIN_JSONL --val $VAL_JSONL --output-dir $RUN_DIR \\"
  log "    --max-seq-len $MAX_SEQ_LEN --batch-size $BATCH_SIZE --grad-accum $GRAD_ACCUM \\"
  log "    --max-steps $MAX_STEPS --val-every $VAL_EVERY --save-every $SAVE_EVERY \\"
  log "    --log-every $LOG_EVERY --wandb-mode $WANDB_MODE \\"
  log "    --resume-from $RESUME_CKPT"
  exit 0
fi

# Confirm H1 metadata in source checkpoint.
micromamba run -n "$ENV_NAME" python - "$RESUME_CKPT" <<'PYMETA' 2>&1 | tee -a "$META_LOG"
import sys
import torch
from pathlib import Path

ckpt = Path(sys.argv[1]) / "mp_rank_00_model_states.pt"
st = torch.load(ckpt, map_location="cpu", weights_only=False)
step = st.get("step")
epoch = st.get("epoch")
micro = st.get("micro_step_in_epoch")
has_rng = "rng_state" in st
print(f"  checkpoint step={step} epoch={epoch} micro_step_in_epoch={micro} rng_state={has_rng}")
if micro is None or int(micro) <= 0:
    raise SystemExit("ERROR: expected micro_step_in_epoch > 0 for H1 test")
if not has_rng:
    raise SystemExit("ERROR: checkpoint missing rng_state (pre-H1?)")
PYMETA

wait_for_gpu_idle

log "Launching resumed training (expect skip-ahead of ~${GRAD_ACCUM}×saved-step micro-batches in trainer log)..."

set +e
HF_HOME="$HF_HOME_PATH" \
  micromamba run -n "$ENV_NAME" \
  deepspeed --num_gpus=1 evo2/scripts/finetune_evo2_lora.py \
    --train "$TRAIN_JSONL" \
    --val "$VAL_JSONL" \
    --output-dir "$RUN_DIR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" \
    --val-every "$VAL_EVERY" \
    --save-every "$SAVE_EVERY" \
    --log-every "$LOG_EVERY" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project bcg-evo2-phase1 \
    --resume-from "$RESUME_CKPT" \
    2>&1 | tee "${RUN_DIR}/train.log"
TRAIN_RC="${PIPESTATUS[0]}"
set -e

if [[ "$TRAIN_RC" -ne 0 ]]; then
  log "❌ Training FAILED with exit code ${TRAIN_RC}. See ${RUN_DIR}/train.log"
  exit 1
fi

log "Training completed. Running verification..."

log ""
log "══════════════════════════════════════════════════════════════"
log "VERIFICATION (step ${MAX_STEPS} vs audit pilot)"
log "══════════════════════════════════════════════════════════════"

export REF_TRAIN_LOG RUN_DIR MAX_STEPS LOSS_ATOL LR_ATOL
micromamba run -n "$ENV_NAME" python <<'PYVERIFY' 2>&1 | tee -a "$META_LOG"
import json
import os
import sys
from pathlib import Path

ref_path = Path(os.environ["REF_TRAIN_LOG"])
run_dir = Path(os.environ["RUN_DIR"])
max_steps = int(os.environ["MAX_STEPS"])
loss_atol = float(os.environ["LOSS_ATOL"])
lr_atol = float(os.environ["LR_ATOL"])

errors = []

def load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def row_at_step(rows: list[dict], step: int) -> dict | None:
    for r in rows:
        if r.get("step") == step:
            return r
    return None

ref_rows = load_log(ref_path)
res_rows = load_log(run_dir / "train_log.jsonl")

print(f"\n--- Reference pilot ({ref_path}) ---")
for r in ref_rows:
    print(f"  step={r['step']:>3d}  loss={r['train_loss']:.6f}  lr={r['lr']:.2e}  "
          f"grad_norm={r.get('grad_norm', 'N/A')}")

print(f"\n--- Resumed run ({run_dir / 'train_log.jsonl'}) ---")
for r in res_rows:
    print(f"  step={r['step']:>3d}  loss={r['train_loss']:.6f}  lr={r['lr']:.2e}  "
          f"grad_norm={r.get('grad_norm', 'N/A')}")

ref = row_at_step(ref_rows, max_steps)
res = row_at_step(res_rows, max_steps)

if ref is None:
    errors.append(f"Reference pilot has no step {max_steps} in train_log")
if res is None:
    errors.append(f"Resumed run has no step {max_steps} in train_log")

train_log_text = (run_dir / "train.log").read_text() if (run_dir / "train.log").exists() else ""
if "Resume skip-ahead" in train_log_text:
    print("\n✅ Trainer log contains H1 skip-ahead message")
else:
    errors.append("FAIL: train.log missing 'Resume skip-ahead' (H1 path may not have run)")

if ref and res:
    if abs(ref["train_loss"] - res["train_loss"]) <= loss_atol:
        print(f"✅ train_loss @ step {max_steps}: {ref['train_loss']:.6f} (match)")
    else:
        errors.append(
            f"FAIL: train_loss @ step {max_steps}: ref={ref['train_loss']:.6f} "
            f"resumed={res['train_loss']:.6f} (atol={loss_atol})"
        )

    if abs(ref["lr"] - res["lr"]) <= lr_atol:
        print(f"✅ lr @ step {max_steps}: {ref['lr']:.2e} (match)")
    else:
        errors.append(
            f"FAIL: lr @ step {max_steps}: ref={ref['lr']:.2e} resumed={res['lr']:.2e}"
        )

    gn_ref = ref.get("grad_norm")
    gn_res = res.get("grad_norm")
    if gn_ref is not None and gn_res is not None:
        if abs(gn_ref - gn_res) <= 0.01:
            print(f"✅ grad_norm @ step {max_steps}: {gn_ref:.4f} (match)")
        else:
            errors.append(
                f"FAIL: grad_norm @ step {max_steps}: ref={gn_ref:.4f} resumed={gn_res:.4f}"
            )

    for key in ("first_record_idx", "collated_seq_len", "first_prefix_token_count"):
        if key in ref and key in res:
            if ref[key] == res[key]:
                print(f"✅ {key} @ step {max_steps}: {ref[key]}")
            else:
                errors.append(
                    f"FAIL: {key} @ step {max_steps}: ref={ref[key]} resumed={res[key]}"
                )

    resumed_steps = [r["step"] for r in res_rows if r["step"] > 10]
    if resumed_steps and min(resumed_steps) == max_steps:
        print(f"✅ Resumed run logged step(s): {resumed_steps} (expected only {max_steps} with log_every=10)")
    elif any(s > 10 for s in resumed_steps):
        print(f"ℹ️  Resumed steps logged: {resumed_steps}")

print()
if errors:
    print("❌ H1 PILOT RESUME TEST FAILED:")
    for e in errors:
        print(f"   {e}")
    sys.exit(1)

print("✅ ALL CHECKS PASSED — H1 faithful resume verified at L=32768.")
sys.exit(0)
PYVERIFY

VERIFY_RC=$?
if [[ "$VERIFY_RC" -eq 0 ]]; then
  log ""
  log "✅ H1 pilot resume test PASSED. Results: ${RUN_DIR}"
else
  log ""
  log "❌ H1 pilot resume test FAILED. Results: ${RUN_DIR}"
fi

exit "$VERIFY_RC"
