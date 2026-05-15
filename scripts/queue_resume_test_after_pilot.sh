#!/usr/bin/env bash
set -euo pipefail

# Wait until no `finetune_evo2_lora.py` process is still using --output-dir
# PILOT_DIR, verify the pilot reached its configured --max-steps, then
# immediately run `queue_h100_resume_test.sh` so the H100 is not left idle
# between the two workloads (shared-host race mitigation).
#
# Typical launch (from repo root, in tmux):
#   cd ~/projects/BCGModelling
#   nohup micromamba run -n bgcmodel bash scripts/queue_resume_test_after_pilot.sh \
#     --pilot-dir /data2/ds85/bgcmodel_runs/pilot_L32768_audit_20260514_182735 \
#     > /data2/ds85/bgcmodel_runs/pilot_L32768_audit_20260514_182735/chain_to_resume.log 2>&1 &
#
# Any arguments after `--` are forwarded to `queue_h100_resume_test.sh`.

PILOT_DIR=""
POLL_SEC=30
CHAIN_LOG=""
ENV_NAME="bgcmodel"

usage() {
  cat <<'EOF'
Usage:
  scripts/queue_resume_test_after_pilot.sh --pilot-dir RUN_DIR [options] [-- RESUME_TEST_ARGS...]

Waits until all trainer processes whose command line references RUN_DIR
have exited, verifies the pilot completed its configured max_steps (from
RUN_DIR/config.json vs RUN_DIR/train_log.jsonl), then runs
scripts/queue_h100_resume_test.sh (which does its own GPU-idle gating).

Options:
  --pilot-dir PATH     Output directory of the pilot run to wait on (required).
  --poll-sec N         Seconds between process checks (default: 30).
  --chain-log PATH     Append wrapper log here (default: RUN_DIR/chain_to_resume.log).
  --env-name NAME      micromamba env for verification Python (default: bgcmodel).
  -h, --help           Show this help.

Everything after a lone `--` is passed through to queue_h100_resume_test.sh.
EOF
}

RESUME_EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot-dir)   PILOT_DIR="${2:?missing value}"; shift 2 ;;
    --poll-sec)    POLL_SEC="${2:?missing value}"; shift 2 ;;
    --chain-log)   CHAIN_LOG="${2:?missing value}"; shift 2 ;;
    --env-name)    ENV_NAME="${2:?missing value}"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    --)            shift; RESUME_EXTRA+=("$@"); break ;;
    *)             echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PILOT_DIR" ]]; then
  echo "error: --pilot-dir is required" >&2
  usage
  exit 2
fi

if [[ ! -d "$PILOT_DIR" ]]; then
  echo "error: pilot directory does not exist: $PILOT_DIR" >&2
  exit 2
fi

if [[ -z "$CHAIN_LOG" ]]; then
  CHAIN_LOG="${PILOT_DIR}/chain_to_resume.log"
fi

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "$CHAIN_LOG"
}

mkdir -p "$(dirname "$CHAIN_LOG")"
touch "$CHAIN_LOG"

count_pilot_procs() {
  # Match the output-dir path appearing on the finetune worker command line.
  pgrep -af 'finetune_evo2_lora\.py' 2>/dev/null | grep -F -- "$PILOT_DIR" | wc -l | tr -d ' '
}

log "══════════════════════════════════════════════════════════════"
log "Chained resume-test launcher"
log "  Pilot dir:  $PILOT_DIR"
log "  Poll:       ${POLL_SEC}s"
log "  Chain log:  $CHAIN_LOG"
log "  Resume passthrough: ${#RESUME_EXTRA[@]} arg(s)"
log "══════════════════════════════════════════════════════════════"

n="$(count_pilot_procs)"
if [[ "$n" -eq 0 ]]; then
  log "No finetune_evo2_lora.py process referencing this pilot dir right now."
  log "Will still verify train_log vs config before launching resume test."
else
  log "Detected ${n} trainer process(es) for this pilot dir; waiting..."
fi

while [[ "$(count_pilot_procs)" -gt 0 ]]; do
  sleep "$POLL_SEC"
done

log "Trainer processes for pilot dir have exited."
sleep 3

# Verify pilot completed successfully (max_steps from config.json).
set +e
VERIFY_OUT="$(micromamba run -n "$ENV_NAME" python3 - <<PY
import json
import sys
from pathlib import Path

pilot = Path(r"""${PILOT_DIR}""")
cfg_path = pilot / "config.json"
log_path = pilot / "train_log.jsonl"

if not cfg_path.exists():
    print("MISSING_CONFIG")
    sys.exit(2)
cfg = json.loads(cfg_path.read_text())
hp = cfg.get("hyperparameters") or {}
ms = int(hp.get("max_steps") or 0)
if ms <= 0:
    print("MAX_STEPS_ZERO")
    sys.exit(2)
if not log_path.exists():
    print("MISSING_TRAIN_LOG")
    sys.exit(2)
rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
if not rows:
    print("EMPTY_TRAIN_LOG")
    sys.exit(2)
last = int(rows[-1]["step"])
if last < ms:
    print(f"INCOMPLETE last_step={last} want>={ms}")
    sys.exit(3)
print(f"OK last_step={last} max_steps={ms}")
sys.exit(0)
PY
)"
VERIFY_RC=$?
set -e

log "Pilot completion check: ${VERIFY_OUT}"
if [[ "$VERIFY_RC" -ne 0 ]]; then
  log "Aborting: pilot did not complete successfully (exit ${VERIFY_RC})."
  log "Fix the pilot run or launch resume test manually after investigation."
  exit 1
fi

log "Launching queue_h100_resume_test.sh (inherits its own GPU-idle wait)..."
exec micromamba run -n "$ENV_NAME" bash "${BASH_SOURCE%/*}/queue_h100_resume_test.sh" "${RESUME_EXTRA[@]}"
