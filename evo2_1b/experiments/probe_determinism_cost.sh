#!/usr/bin/env bash
# MEASURE WHAT THE DETERMINISM SETTINGS COST, before deciding to change them. Run from the repo root.
#
# WHY THIS IS A SCRIPT AND NOT A ONE-LINE EDIT. finetune_evo2_lora.py sets
# `torch.use_deterministic_algorithms(True, warn_only=True)` and `cudnn.benchmark = False`, but
# CUBLAS_WORKSPACE_CONFIG is unset -- so the GEMMs that dominate runtime are nondeterministic
# ANYWAY, while benchmark=False gives up cuDNN kernel autotuning. Evo2 is convolutional
# (StripedHyena: 21 of the 1B's 25 blocks are Hyena), which is exactly where autotuning pays, so
# the cost may be larger here than the usual attention-model intuition suggests. It is still a
# GUESS until measured, and adopting an unmeasured speedup is how the truncate default got in.
#
# ⚠️ RUN THIS BETWEEN ROUNDS, NEVER BETWEEN ARMS OF ONE COMPARISON. Both directions change the
# floating-point path. Applying it to arms 2 and 3 but not arm 1 makes the objective no longer the
# only difference between them -- the same confound as changing batch size mid-experiment.
set -euo pipefail
export HF_HOME=/data2/ds85/hf_cache
export EVO2_BASE_MODEL=evo2_1b_base

DATA=/data2/ds85/bgcmodel_data/splits_core
ANN=$DATA/train.domain_spans.jsonl
ROOT=${ROOT:-/data2/ds85/bgcmodel_runs/det_probe}
STEPS=${STEPS:-40}
L=${L:-8192}
mkdir -p "$ROOT"

# ONLY TWO OF THE THREE ARMS RUN HERE. (a) as-is and (b) CUBLAS_WORKSPACE_CONFIG=:4096:8 need no
# code change. (c) "drop determinism entirely + cudnn.benchmark=True" needs a `--no-deterministic`
# flag on the trainer, which DOES NOT EXIST yet (checked: finetune_evo2_lora.py sets both
# unconditionally at lines 361/367). Add the flag first if (b) does not already settle the
# question -- do not hand-edit those two lines for a probe, because the edited state is easy to
# leave behind and would silently change the default path of every later run.
probe () {
  local name="$1"; shift
  [[ -s "$ROOT/$name/train_log.jsonl" ]] && { echo "[det] $name done — skipping"; return 0; }
  echo "[det] $(date) $name  (env: ${*:-none})"
  env "$@" micromamba run -n bgcmodel deepspeed --num_gpus=1 --master_port 29520 \
    evo2/scripts/finetune_evo2_lora.py \
    --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" --output-dir "$ROOT/$name" \
    --max-seq-len "$L" --batch-size 1 --grad-accum 16 \
    --long-seq-strategy chunk --chunk-overlap 1024 \
    --warmup-steps 5 --max-epochs 1 --max-steps "$STEPS" \
    --log-every 5 --val-every 999 --save-every 999 \
    --wandb-mode offline --annotations "$ANN" --domain-weight 1.0 --frame-lambda 0.0 \
    > "$ROOT/$name.log" 2>&1 || echo "[det] $name FAILED — see $ROOT/$name.log"
}

probe as_is
probe cublas_det CUBLAS_WORKSPACE_CONFIG=:4096:8

echo
micromamba run -n bgcmodel python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print("DETERMINISM COST PROBE — steady-state tokens/sec (first 2 logs dropped as warmup)")
base = None
for name in ("as_is", "cublas_det"):
    p = root / name / "train_log.jsonl"
    if not p.exists():
        print(f"{name:>12}  (no log)"); continue
    rows = [json.loads(l) for l in p.open() if '"tokens_per_sec"' in l][2:]
    if not rows:
        print(f"{name:>12}  (too few steps)"); continue
    tps = sum(r["tokens_per_sec"] for r in rows) / len(rows)
    if base is None:
        base = tps
    print(f"{name:>12}  {tps:>8,.0f} tok/s   {tps/base:+.2%} vs as_is" if base else "")
print("\nAdopt the winner for a WHOLE round. If the spread is under ~5%, keep `as_is` —")
print("a change to the numerical path is not worth making for noise.")
PY
