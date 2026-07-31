#!/usr/bin/env bash
# PHASE 3 — THE DECISIVE TEST: does steering change what the model WRITES?
#
# Everything before this measured the model's OPINION about text we hand it (teacher-forced
# log-likelihood). This measures its BEHAVIOUR. A per-base bias far too small to see in Phase 1
# can still compound across thousands of sampling steps -- or evaporate. Only generation says.
#
# THE EXPERIMENT — cross-class override. Seed the model with a real class-A core, steer the
# CONTINUATION toward class B, and ask whether B's machinery appears AND A's disappears. Both
# halves matter: markers appearing without the seed's disappearing is not an override.
#
# WHY THE SEEDED REGIME. It is the only setting with a measured floor AND ceiling
# (correct_class 0.067 mismatched vs 0.283 matched, n=60). Taxonomy-only generation sits on a
# 3.3% floor where nothing is detectable -- 9 GPU-hours were already burned proving that.
#
# THE COMPARISON THAT COUNTS IS A-vs-B (real direction vs SHUFFLED-LABEL direction).
# A-vs-C (vs unsteered) is reported but proves nothing: codon-shuffling the seed with NO steering
# at all already drives seed-class markers to 0/45, so "beats unsteered" is worth zero bits.
#
# DOSE. 1 class-unit -- where Phase 1 found the causal effect (p=0.040, 0/24 controls as
# extreme). Phase 2 confirmed no ORF damage anywhere in 0.5-4, so coherence is not the binding
# constraint; higher doses are noisier, not stronger.
#
# HOOK GATING. seed_generate.py steers ONLY generated positions (shape[1]==1), never the prefill.
# Steering the seed would corrupt the very exemplar carrying the class signal we are overriding.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
export TMPDIR="${TMPDIR:-/data2/ds85/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data2/ds85/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data2/ds85/cache/mpl}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
ENV=bgcmodel
PY(){ micromamba run -n "$ENV" python "$@"; }

V2="${V2:-/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200}"
SEEDSRC="${SEEDSRC:-/data2/ds85/bgcmodel_data/splits_core/valtest_eval.jsonl}"
DIRS="${DIRS:-/data2/ds85/bgcmodel_runs/class_probe_sweep/valtest30.steerdirs.npz}"
CLASSES="${CLASSES:-NRPS PKS TERPENE RIPP}"
LAYER="${LAYER:-16}"
DOSE="${DOSE:-1.0}"
NULL_PREFIX="${NULL_PREFIX:-perm0_}"
PER_CLASS="${PER_CLASS:-12}"
SEED_NT="${SEED_NT:-1000}"
MAX_NEW="${MAX_NEW:-3000}"
SEED="${SEED:-42}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/steer_phase3}"
mkdir -p "$ROOT"

EXCLUSIVE="${EXCLUSIVE:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"
wait_for_idle(){ local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if { [ "$EXCLUSIVE" -eq 0 ] || [ "${proc:-1}" -eq 0 ]; } && [ "${free:-0}" -ge "$MIN_FREE_MB" ]; then
      hold=$((hold+1)); [ "$hold" -ge 2 ] && return 0
    else hold=0; fi; sleep 10
  done; }

# Identical --seed across arms => the SAME seed exemplars in the SAME order => the arms are
# strictly PAIRED, which is what makes a per-exemplar comparison legitimate.
run_arm(){                              # $1=arm  $2=class-units  $3=dir prefix
  local arm="$1" dose="$2" pfx="$3"
  # NOTE: `out` must be a SEPARATE `local` -- bash declares every name in a single
  # `local` before assigning any, so ${arm} is still unset there and `set -u` aborts.
  local out="$ROOT/${arm}.jsonl"
  if [ ! -s "$out" ]; then
    echo "[p3] $(date) GEN $arm (dose=$dose, dir='${pfx:-real}')"; wait_for_idle
    PY evo2/scripts/seed_generate.py --adapter "$V2" --from-jsonl "$SEEDSRC" \
       --classes $CLASSES --per-class "$PER_CLASS" --seed-nt "$SEED_NT" \
       --max-new-tokens "$MAX_NEW" --no-class-tag --seed "$SEED" \
       --steer-dirs-npz "$DIRS" --steer-layer "$LAYER" \
       --steer-class-units "$dose" --steer-dir-prefix "$pfx" --steer-toward rotate \
       --out-jsonl "$out" > "$ROOT/gen_${arm}.log" 2>&1
  fi
  [ -s "$out" ] && echo "[p3] $arm: $(wc -l < "$out") records" \
                || echo "[p3] !! $arm produced nothing (see $ROOT/gen_${arm}.log)"
}

run_arm "A_real"     "$DOSE" ""
run_arm "B_shuffled" "$DOSE" "$NULL_PREFIX"
run_arm "C_unsteered" "0"    ""

echo "[p3] $(date) scoring class markers (target present AND seed absent)"
PY - "$ROOT" <<'PYEOF'
import json, sys, collections
from pathlib import Path
sys.path.insert(0, "/home/ds85/projects/BCGModelling/src")
from bgc_pipeline.evaluation import check_class_markers
root = Path(sys.argv[1])
# REQUIRED. Without it check_class_markers returns {"skipped": True,
# "reason": "Pfam HMM not found at None"} for EVERY sequence -- and a naive
# bool(result.get("markers_present")) turns that into False, i.e. "could not measure" silently
# becomes "measured absent". That produced an all-zero Phase 3 table on the first pass.
# Must be a Path: the function calls pfam_hmm_path.exists().
PFAM = Path("/data2/ds85/pfam/Pfam-A.hmm")
if not PFAM.exists():
    raise SystemExit(f"[p3] ABORT: Pfam HMM missing at {PFAM} -- marker scoring would be a no-op")
tsv = root / "per_sequence.tsv"
n_skip = 0
with tsv.open("w") as fh:
    fh.write("arm\tseed_class\ttarget_class\tidx\tlength\tskipped\tmarkers_target\tmarkers_seed\toverride\n")
    for arm in ("A_real", "B_shuffled", "C_unsteered"):
        p = root / f"{arm}.jsonl"
        if not p.exists():
            continue
        for i, line in enumerate(p.open()):
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("sequence", "") or ""
            sc = r.get("seed_class")
            tc = r.get("steer_target_class") or "-"
            skipped = False
            if len(s) < 200:
                mt = ms = False
                skipped = True
            else:
                rt = check_class_markers(s, expected_class=tc, pfam_hmm_path=PFAM) if tc != "-" else {}
                rs = check_class_markers(s, expected_class=sc, pfam_hmm_path=PFAM)
                # A SKIP is not a negative. Record it and exclude it from the rates.
                if rt.get("skipped") or rs.get("skipped"):
                    skipped = True
                    n_skip += 1
                mt = bool(rt.get("markers_present"))
                ms = bool(rs.get("markers_present"))
            # OVERRIDE = the target's machinery is there AND the seed's is gone. Appearance
            # without disappearance is not an override.
            fh.write(f"{arm}\t{sc}\t{tc}\t{i}\t{len(s)}\t{int(skipped)}\t{int(mt)}\t{int(ms)}\t"
                     f"{int(mt and not ms)}\n")
        print(f"[p3]   scored {arm}", flush=True)
print(f"[p3] per-sequence -> {tsv}   (skipped: {n_skip})")
PYEOF

echo "==================== PHASE 3 — CROSS-CLASS OVERRIDE ===================="
PY - "$ROOT/per_sequence.tsv" <<'PYEOF'
import sys, csv, collections
from math import comb
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
if not rows:
    print("no rows"); raise SystemExit(0)
by = collections.defaultdict(list)
n_skipped = sum(1 for r in rows if r.get("skipped") == "1")
for r in rows:
    if r.get("skipped") == "1":       # could not measure != measured absent
        continue
    by[r["arm"]].append(r)
if n_skipped:
    print(f"({n_skipped} sequences excluded: marker scan could not run)")
print(f"{'arm':>12} {'n':>4} {'target markers':>16} {'seed markers':>14} {'OVERRIDE':>10}")
rate = {}
for arm in ("A_real", "B_shuffled", "C_unsteered"):
    rs = by.get(arm)
    if not rs:
        continue
    n = len(rs)
    mt = sum(int(r["markers_target"]) for r in rs) / n
    ms = sum(int(r["markers_seed"]) for r in rs) / n
    ov = sum(int(r["override"]) for r in rs) / n
    rate[arm] = [int(r["override"]) for r in rs]
    print(f"{arm:>12} {n:>4} {mt:>16.3f} {ms:>14.3f} {ov:>10.3f}")
# PAIRED test on the contrast that counts: real vs shuffled-label, same exemplars.
a, b = rate.get("A_real"), rate.get("B_shuffled")
if a and b and len(a) == len(b):
    disc_a = sum(1 for x, y in zip(a, b) if x and not y)
    disc_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = disc_a + disc_b
    p = sum(comb(n, i) * 0.5 ** n for i in range(disc_a, n + 1)) if n else 1.0
    print(f"\nPAIRED real-vs-shuffled (the comparison that counts):")
    print(f"  override in A only: {disc_a}   in B only: {disc_b}   discordant: {n}")
    print(f"  one-sided p = {p:.4f}")
    print(f"  {'PROCEED to Phase 4' if p < 0.05 else 'NOT SIGNIFICANT at this n'}")
    if n < 10:
        print(f"  NOTE only {n} discordant pairs -- underpowered; the smallest attainable")
        print(f"       p at this discordance is {0.5 ** n:.4f}.")
print("\nA-vs-C (unsteered) is deliberately NOT the test: a codon-shuffled seed with no steering")
print("already drives seed markers to zero, so beating 'unsteered' carries no information.")
PYEOF
echo "[p3] ALL DONE $(date)"
