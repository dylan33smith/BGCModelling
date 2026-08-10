#!/usr/bin/env bash
# STEER AT LAYER 27, DOSE-SCALED TO THE LOCAL RESIDUAL NORM — is DILUTION the reason Phase 3 was null?
#
# THE HYPOTHESIS. Phase 3 injected the class direction at layer 16 and nothing changed in what the
# model wrote. One explanation is that the edit never survives to the output. Measured on the same
# activation cache the directions are fit from (n=3,430 real cores, acts_valtest_fit.npz):
#
#     layer       16      20      24      27       28        29        30
#     mean||h||   8.95    9.78    6.54    11.25    5.47e3    8.66e6    3.69e12
#     class AUC   0.923   0.927   0.885   0.835    0.590     0.610     0.553
#
# L27 is the LAST layer where the class direction is still real and the last one before the
# residual stream explodes by eleven orders of magnitude. An edit there passes through 4 blocks
# instead of 16.
#
# DOSE IS IN UNITS OF THE LOCAL RESIDUAL NORM (--steer-norm-frac), never absolute. ||h|| differs
# between the two injection points, so holding ||delta|| fixed would confound depth with dose.
# Measured LIVE at the hook during cached generation (NOT from the mean-pooled cache, which
# disagrees by 2.8x at L27): mean ||h|| = 6.69 at L16 and 31.97 at L27, so one class-unit is
# 0.082*||h|| at L16 -- Phase 3's actual operating point -- but only 0.056*||h|| at L27.
#
# LADDER, in multiples of Phase 3's operating point (frac 0.082 at L16). Reach scales ~ frac^2,
# and L27 delivers 3.5x less reach per unit frac (see steer_reach.py), so at L27:
#     frac 0.061 -> ~0.15x Phase 3's output-distribution impact   (under-dosed on purpose)
#     frac 0.16  -> ~1x    (REACH-MATCHED to Phase 3's L16 dose)  = 2.8 class-units
#     frac 0.32  -> ~4x                                           = 5.7 class-units
#     frac 0.64  -> ~16x                                          = 11.4 class-units
# The top of the ladder is far past the semantic range Phase 2 cleared for damage, which is
# exactly why coding_density is scored on every sequence.
#
# STAGE 1 (ladder)  real direction only, small n, four doses. Answers "is L27 steering survivable,
#                   and does anything move?" -- the coherence guard Phase 2 established only at L16.
# STAGE 2 (paired)  A_real vs B_shuffled at the dose stage 1 supports, full n, strictly paired.
#                   THIS is the test. Beating the SHUFFLED-LABEL arm is the only comparison that
#                   separates "the class direction did it" from "any perturbation does it".
#
# The unsteered floor is layer-independent (no hook at all), so Phase 3's C_unsteered arm is
# reused rather than regenerated -- same seeds, same params, same 48 exemplars.
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
DIRS="${DIRS:-/data2/ds85/bgcmodel_runs/class_probe_sweep/valtest30_multilayer.steerdirs.npz}"
CLASSES="${CLASSES:-NRPS PKS TERPENE RIPP}"
LAYER="${LAYER:-27}"
NULL_PREFIX="${NULL_PREFIX:-perm0_}"
SEED_NT="${SEED_NT:-1000}"        # identical to Phase 3, so the arms pair with its C_unsteered
MAX_NEW="${MAX_NEW:-3000}"
SEED="${SEED:-42}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/steer_l27}"
STAGE="${STAGE:-1}"
LADDER="${LADDER:-0.061 0.16 0.32 0.64}"
LADDER_PER_CLASS="${LADDER_PER_CLASS:-3}"
PAIRED_FRAC="${PAIRED_FRAC:-0.16}"
PAIRED_PER_CLASS="${PAIRED_PER_CLASS:-12}"
mkdir -p "$ROOT"

[ -s "$DIRS" ] || { echo "[l27] ABORT: no directions at $DIRS (run build_steer_dirs.py --layers 16 20 24 27)"; exit 1; }
[ -s "$PFAM" ] || { echo "[l27] ABORT: no Pfam HMM at $PFAM -- marker scoring would be a silent no-op"; exit 1; }

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

# Identical --seed across arms => the SAME seed exemplars in the SAME order => strictly PAIRED.
run_arm(){                              # $1=arm  $2=norm-frac  $3=dir prefix  $4=per-class
  local arm="$1" frac="$2" pfx="$3" nper="$4"
  local out="$ROOT/${arm}.jsonl"
  if [ -s "$out" ]; then echo "[l27] $arm exists, skip"; return; fi
  echo "[l27] $(date) GEN $arm (L=$LAYER frac=$frac dir='${pfx:-real}' n/class=$nper)"; wait_for_idle
  PY evo2/scripts/seed_generate.py --adapter "$V2" --from-jsonl "$SEEDSRC" \
     --classes $CLASSES --per-class "$nper" --seed-nt "$SEED_NT" \
     --max-new-tokens "$MAX_NEW" --no-class-tag --seed "$SEED" \
     --steer-dirs-npz "$DIRS" --steer-layer "$LAYER" \
     --steer-norm-frac "$frac" --steer-dir-prefix "$pfx" --steer-toward rotate \
     --out-jsonl "$out" > "$ROOT/gen_${arm}.log" 2>&1
  [ -s "$out" ] && echo "[l27] $arm: $(wc -l < "$out") records" \
                || echo "[l27] !! $arm produced nothing (see $ROOT/gen_${arm}.log)"
}

if [ "$STAGE" = "1" ]; then
  for F in $LADDER; do run_arm "ladder_f${F}" "$F" "" "$LADDER_PER_CLASS"; done
else
  run_arm "A_real_f${PAIRED_FRAC}"     "$PAIRED_FRAC" ""              "$PAIRED_PER_CLASS"
  run_arm "B_shuffled_f${PAIRED_FRAC}" "$PAIRED_FRAC" "$NULL_PREFIX"  "$PAIRED_PER_CLASS"
fi

echo "[l27] $(date) scoring: coherence (coding_density) AND class markers (target present / seed absent)"
PY - "$ROOT" "$PFAM" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/ds85/projects/BCGModelling/src")
from bgc_pipeline.evaluation import check_class_markers, check_coding_sanity
root, pfam = Path(sys.argv[1]), Path(sys.argv[2])
# REQUIRED. Without it check_class_markers returns {"skipped": True} for EVERY sequence and a
# naive bool(...) turns "could not measure" into "measured absent" -- which produced an all-zero
# Phase 3 table on its first pass.
if not pfam.exists():
    raise SystemExit(f"[l27] ABORT: Pfam HMM missing at {pfam}")
tsv = root / "per_sequence.tsv"
n_skip = 0
arms = sorted(p.stem for p in root.glob("*.jsonl") if not p.stem.startswith("_"))
with tsv.open("w") as fh:
    fh.write("arm\tseed_class\ttarget_class\tidx\tlength\tskipped\tcoding_density\t"
             "applied_frac\tapplied_class_units\tmarkers_target\tmarkers_seed\toverride\n")
    for arm in arms:
        for i, line in enumerate((root / f"{arm}.jsonl").open()):
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("sequence", "") or ""
            sc, tc = r.get("seed_class"), (r.get("steer_target_class") or "-")
            skipped, cd = False, ""
            mt = ms = False
            if len(s) < 200:
                skipped = True
            else:
                cs = check_coding_sanity(s)
                cd = round(cs.get("coding_density", float("nan")), 4)
                rt = check_class_markers(s, expected_class=tc, pfam_hmm_path=pfam) if tc != "-" else {}
                rs = check_class_markers(s, expected_class=sc, pfam_hmm_path=pfam)
                if rt.get("skipped") or rs.get("skipped"):   # a SKIP is not a negative
                    skipped = True
                    n_skip += 1
                mt, ms = bool(rt.get("markers_present")), bool(rs.get("markers_present"))
            fh.write(f"{arm}\t{sc}\t{tc}\t{i}\t{len(s)}\t{int(skipped)}\t{cd}\t"
                     f"{r.get('steer_realized_norm_frac','')}\t"
                     f"{r.get('steer_realized_class_units','')}\t"
                     f"{int(mt)}\t{int(ms)}\t{int(mt and not ms)}\n")
        print(f"[l27]   scored {arm}", flush=True)
print(f"[l27] per-sequence -> {tsv}   (skipped: {n_skip})")
PYEOF

echo "==================== LAYER-27 STEERING ===================="
PY - "$ROOT/per_sequence.tsv" <<'PYEOF'
import sys, csv, collections, statistics as st
from math import comb
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
if not rows:
    print("no rows"); raise SystemExit(0)
n_skipped = sum(1 for r in rows if r.get("skipped") == "1")
by = collections.defaultdict(list)
for r in rows:
    if r.get("skipped") == "1":       # could not measure != measured absent
        continue
    by[r["arm"]].append(r)
if n_skipped:
    print(f"({n_skipped} sequences excluded: marker scan could not run)")
print(f"{'arm':>22} {'n':>4} {'applied':>8} {'coding_density':>15} {'target mk':>10} "
      f"{'seed mk':>8} {'OVERRIDE':>9}")
rate = {}
for arm in sorted(by):
    rs = by[arm]; n = len(rs)
    cd = [float(r["coding_density"]) for r in rs if r["coding_density"]]
    ap = [float(r["applied_frac"]) for r in rs if r["applied_frac"]]
    rate[arm] = [int(r["override"]) for r in rs]
    print(f"{arm:>22} {n:>4} {st.mean(ap) if ap else float('nan'):>8.3f} "
          f"{st.mean(cd) if cd else float('nan'):>15.3f} "
          f"{sum(int(r['markers_target']) for r in rs)/n:>10.3f} "
          f"{sum(int(r['markers_seed']) for r in rs)/n:>8.3f} "
          f"{sum(int(r['override']) for r in rs)/n:>9.3f}")
# PAIRED test on the contrast that counts: real vs shuffled-label, SAME exemplars.
a_arms = [k for k in rate if k.startswith("A_real")]
b_arms = [k for k in rate if k.startswith("B_shuffled")]
for aa in a_arms:
    bb = next((b for b in b_arms if b.split("_f")[-1] == aa.split("_f")[-1]), None)
    if not bb or len(rate[aa]) != len(rate[bb]):
        continue
    a, b = rate[aa], rate[bb]
    da = sum(1 for x, y in zip(a, b) if x and not y)
    db = sum(1 for x, y in zip(a, b) if y and not x)
    n = da + db
    p = sum(comb(n, i) * 0.5 ** n for i in range(da, n + 1)) if n else 1.0
    print(f"\nPAIRED {aa} vs {bb} (the comparison that counts):")
    print(f"  override in A only: {da}   in B only: {db}   discordant: {n}   one-sided p = {p:.4f}")
    if n < 10:
        print(f"  NOTE only {n} discordant pairs -- underpowered; the smallest attainable p")
        print(f"       at this discordance is {0.5 ** n:.4f}.")
print("\nREAD: coding_density is the DAMAGE guard -- if it falls with dose, the model is being")
print("pushed out of distribution (CFG's failure mode) rather than conditioned, and any marker")
print("change at that dose is degradation, not control.")
PYEOF
echo "[l27] ALL DONE $(date)"
