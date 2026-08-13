#!/usr/bin/env bash
# DOMAIN WEIGHTING AT 10x. Run from the repo root.
#
# WHY. `--domain-weight 3.0` was reported as a null at n=152 and then shown NEVER TO HAVE LANDED:
# its in-domain/out-of-domain CE ratio is 0.9006 against baseline's 0.9011 (-0.06%), while plain
# fine-tuning moved that same ratio +0.34% and the spread across four models -- one of which never
# saw domain weighting -- is 0.4%. Not a plumbing bug (domain_weight=3.0, 47,524 annotations, 100%
# coverage, weighted loss differs from CE at every logged step). So the 3x null is UNINTERPRETABLE
# and domain weighting is untested rather than refuted.
#
# The working hypothesis is that the intervention is TOO DIFFUSE: 40.2% of positions are in-domain,
# so "attend 3x harder to 40% of the text" is a broad, mild nudge. The frame penalty bit hard
# because it was SHARP -- a rare, specific event (a stop-completing base at codon phase 2) with a
# large relative penalty. 10x tests the diffuseness hypothesis directly.
#
# EVERYTHING ELSE IS HELD FIXED -- same L=8192, chunk/1024, bs1 x ga16, 400 steps, same seed and
# data order -- so this arm is directly comparable to the existing baseline / frame / weighted set
# and needs no new control.
#
# ⚠️ THE GATE: verify the treatment BEFORE reading the outcome. That rule was violated twice in two
# days (once at n=24, once on the 3x arm) and both times produced a null reported as a closure. So
# this script runs the domain-weighting probe FIRST and only generates if the ratio actually moved.
# If 10x also fails to move it, generating 152 sequences would measure nothing and is skipped.
set -euo pipefail
export HF_HOME=/data2/ds85/hf_cache
export EVO2_BASE_MODEL=evo2_1b_base

ROOT=${ROOT:-/data2/ds85/bgcmodel_runs/phase2_1b}
ARM=weighted10
PY() { micromamba run -n bgcmodel python "$@"; }

echo "[w10] $(date) === STEP 1/3: train $ARM (domain-weight 10.0) ==="
ARMS="$ARM" PARALLEL=0 bash evo2_1b/experiments/run_objective_arms.sh

echo
echo "[w10] $(date) === STEP 2/3: DID THE TREATMENT LAND? (gate) ==="
PY evo2_1b/experiments/probe_domain_weighting.py --n 40 --root "$ROOT" \
   2>&1 | tee "$ROOT/dw_probe_w10.log" | grep -vE "UserWarning|warnings.warn|torch.load|Fetching|it/s|Extra keys|^\s*$|return |te_fp8|state = "

LANDED=$(PY - "$ROOT" <<'PYEOF'
import json, sys
from pathlib import Path
d = json.loads((Path(sys.argv[1]) / "domain_weight_probe.json").read_text())
b = d.get("baseline", {}).get("ratio")
w = d.get("weighted10", {}).get("ratio")
# 1% is the bar: the FULL spread across four models that never saw 10x weighting is 0.4%, so
# anything under ~1% is inside the noise this probe already demonstrated.
print("yes" if (b and w and w < b * 0.99) else "no")
PYEOF
)

echo
if [[ "$LANDED" != "yes" ]]; then
  echo "[w10] === STOPPING: the 10x treatment did NOT land either. ==="
  echo "[w10] Generation is skipped on purpose -- it would measure a model that was never changed."
  echo "[w10] That makes DIFFUSENESS-OF-WEIGHT the wrong explanation, and points at the remaining"
  echo "[w10] candidates: 6.7% of one epoch is too little training, or per-token loss weighting is"
  echo "[w10] simply not an effective lever on this substrate (in which case change the DATA --"
  echo "[w10] train on domain-dense regions -- rather than the loss)."
  exit 0
fi

echo "[w10] === STEP 3/3: treatment landed — generating 152 and scoring ==="
DATA=/data2/ds85/bgcmodel_data/splits_core
GEN="$ROOT/$ARM/gen_n150.jsonl"
[[ -s "$GEN" ]] || PY evo2/scripts/generate_bgc.py --adapter "$ROOT/$ARM/final_adapter" \
   --from-jsonl "$DATA/valtest_eval_4class.jsonl" \
   --per-class 38 --n 1 --max-new-tokens 8000 --seed 0 --batch-size 32 \
   --out-jsonl "$GEN" > "$ROOT/$ARM.gen_n150.log" 2>&1
PY evo2/scripts/score_ladder.py --gen "$GEN" --out-json "$ROOT/$ARM/ladder_n150.json" \
   > "$ROOT/$ARM.ladder_n150.log" 2>&1 || echo "[w10] ladder FAILED"
PY evo2_1b/experiments/compare_arms_highn.py --root "$ROOT"
