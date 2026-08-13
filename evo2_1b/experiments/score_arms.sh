#!/usr/bin/env bash
# SCORE THE PHASE-2 OBJECTIVE ARMS. Run from the repo root, after run_objective_arms.sh.
#
# Generates de novo from each arm's final adapter with IDENTICAL prompts, decoding and RNG seed —
# the objective is the only difference — then scores the validated ladder plus the novelty guard.
#
# READ IT ON best_bio_bits, NOT max_orf_aa. The frame-aware arm manipulates reading-frame length
# directly, so scoring it on ORF length would be scoring the manipulation. max_orf_aa does not
# track domain content de novo (r = 0.051 at 2 kb, -0.120 at 6 kb) and is reported only as a
# structural diagnostic. If ORF length rises and best_bio_bits does not, that is the informative
# negative: length was never the constraint.
#
# NOVELTY IS A CONSTRAINT, NOT A METRIC. Every ladder rung is maximised by copying training data,
# so an arm that improves with containment climbing toward 0.95 has learned to recite, not to build.
# WAIT=1 POLLS FOR EACH ARM'S ADAPTER AND SCORES IT THE MOMENT IT LANDS, instead of requiring all
# training to finish first. Arms do not finish together (a partly-run arm plus two fresh ones stagger
# by ~30 min), and generation is the serial GPU cost here — `score_ladder.py` is already CPU-parallel
# via ProcessPoolExecutor and runs no antiSMASH. Overlapping the first arm's generation with the
# others' training therefore spends GPU capacity that would otherwise idle.
#
# THIS CANNOT AFFECT THE COMPARISON. Generation is driven by --seed 0 with identical prompts and
# decoding for every arm; sharing the GPU changes scheduling, not sampling. What it CAN do is slow
# the still-training arms slightly, which costs wall-clock and nothing else.
set -euo pipefail
export HF_HOME=/data2/ds85/hf_cache
export EVO2_BASE_MODEL=evo2_1b_base

ROOT=${ROOT:-/data2/ds85/bgcmodel_runs/phase2_1b}
DATA=/data2/ds85/bgcmodel_data/splits_core
N_PER_CLASS=${N_PER_CLASS:-6}
MAX_NEW=${MAX_NEW:-6000}
BATCH=${BATCH:-24}
# BATCHED GENERATION IS 12.05x FASTER AND IS NOW VALIDATED (scripts/validate_batched_generation.py,
# which generate_bgc.py had referenced for months without the file existing). The first 24-per-arm
# run left this at the default 1 and generated ONE SEQUENCE AT A TIME -- that, not the model, is
# why scoring took hours.
#
# ⚠️ NEVER POOL BATCHED WITH SEQUENTIAL OUTPUT. vortex batches only equal-length prompts, so
# generate_bgc.py left-pads them, and a byte-level model CONDITIONS ON THE PAD BYTES -- a padded
# prompt is a different prompt. Proven, not assumed: of 8 prompts only the one needing zero padding
# reproduced its sequential output exactly. Padding is common-mode across arms at a fixed prompt
# set and seed, so it cannot manufacture a between-arm difference; but the existing sequential
# gen.jsonl files and any new batched ones are NOT the same experiment.
WAIT=${WAIT:-0}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-21600}   # 6 h, then give up rather than poll forever
PY() { micromamba run -n bgcmodel python "$@"; }

for arm in baseline frame weighted; do
  ADP="$ROOT/$arm/final_adapter"
  GEN="$ROOT/$arm/gen.jsonl"
  if [[ "$WAIT" == "1" ]]; then
    # GATE ON THE WEIGHTS FILE, AND ON ITS SIZE SETTLING. The trainer publishes the adapter with
    # shutil.copytree, which copies in directory order -- so `adapter_config.json` appears BEFORE
    # `adapter_model.safetensors` finishes streaming. Polling for the directory, or for the config,
    # fires on a half-written adapter and generation would load truncated weights: a corrupt arm
    # that still produces sequence, i.e. a silent wrong result rather than a crash.
    W="$ADP/adapter_model.safetensors"
    waited=0
    while (( waited < WAIT_TIMEOUT )); do
      if [[ -f "$W" ]]; then
        s1=$(stat -c%s "$W"); sleep 15; s2=$(stat -c%s "$W")
        [[ "$s1" == "$s2" && "$s1" != "0" ]] && break
      else
        sleep 60; waited=$((waited + 60))
      fi
    done
    [[ -f "$W" ]] || { echo "[score] $arm: no adapter weights after ${WAIT_TIMEOUT}s — skipped"; continue; }
    echo "[score] $(date) $arm adapter landed (${waited}s waited, $(stat -c%s "$W") bytes)"
  fi
  [[ -d "$ADP" ]] || { echo "[score] $arm: no final_adapter — skipped"; continue; }
  if [[ ! -s "$GEN" ]]; then
    echo "[score] $(date) generating from $arm"
    PY evo2/scripts/generate_bgc.py --adapter "$ADP" \
       --from-jsonl "$DATA/valtest_eval_4class.jsonl" \
       --per-class "$N_PER_CLASS" --n 1 --max-new-tokens "$MAX_NEW" --seed 0 \
       --batch-size "$BATCH" \
       --out-jsonl "$GEN" > "$ROOT/$arm.gen.log" 2>&1
  fi
  echo "[score] $(date) ladder on $arm"
  PY evo2/scripts/score_ladder.py --gen "$GEN" --out-json "$ROOT/$arm/ladder.json" \
     > "$ROOT/$arm.ladder.log" 2>&1 || echo "[score] $arm ladder FAILED — see $ROOT/$arm.ladder.log"
done

echo
PY - "$ROOT" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
arms = ["baseline", "frame", "weighted"]
rows = {}
for a in arms:
    p = root / a / "ladder.json"
    if p.exists():
        rows[a] = json.loads(p.read_text())
if not rows:
    raise SystemExit("[score] no ladder.json found for any arm")

print("PHASE-2 OBJECTIVE ARMS — 1B, de novo, identical prompts/decoding/seed")
print("PRIMARY = best_bio_bits. Reference on the 7B: LoRA 56.9, real cores 148.6.\n")
cols = [("best_bio_bits", "best_bio_bits *"), ("n_bio_domains", "n_bio_domains"),
        ("bio_span_frac", "bio_span_frac"), ("bio_fraction", "bio_fraction"),
        ("frac_with_bio_signal", "frac w/ signal"), ("max_orf_aa", "max_orf_aa (diag)"),
        ("novelty_max_containment", "novelty max"), ("novelty_verdict", "novelty")]
print(f"{'arm':>10} " + " ".join(f"{lab:>17}" for _, lab in cols))
for a in arms:
    r = rows.get(a)
    if not r:
        continue
    cells = []
    for k, _ in cols:
        v = r.get(k, "--")
        cells.append(f"{v:>17.3f}" if isinstance(v, (int, float)) else f"{str(v):>17}")
    print(f"{a:>10} " + " ".join(cells))

base = rows.get("baseline")
if base:
    print("\nDELTA vs baseline (primary metric):")
    for a in ("frame", "weighted"):
        if a in rows:
            d = rows[a].get("best_bio_bits", 0) - base.get("best_bio_bits", 0)
            print(f"  {a:>9}: {d:+.3f}")
    print("\nKILL CRITERION: no arm moving best_bio_bits above baseline at unchanged novelty is a")
    print("clean negative on the objective hypothesis — not a reason for another loss variant.")
    print("CAVEAT: n is small and these are single runs; read direction and magnitude, not p-values.")
PYEOF
