#!/usr/bin/env bash
# MULTI-LAYER STEERING — re-assert the class direction at EVERY layer, not once.
#
# WHY THIS IS NOT JUST "MORE DOSE". Two prior results look like they close this and do not:
#   * single-layer reach FALLS with depth (L16 0.0101 -> L27 0.0029), and
#   * a bigger single-layer dose buys damage, not class (L27 ladder: 0/48 up to 11.9 class-units,
#     coherence 0.925 -> 0.684).
# Both measured ONE injection point at a time. A stack is mechanically different:
#   1. RE-ASSERTION. Nothing obliges blocks after L16 to preserve an added component -- it is not
#      a state the model would have produced itself, so downstream computation can overwrite it.
#      Falling reach with depth is exactly what such erasure looks like. Adding at every layer is
#      closer to CLAMPING the class coordinate than nudging it once, and no single-layer
#      measurement can see that difference.
#   2. DAMAGE IS PER-LAYER, EFFECT MAY BE CUMULATIVE. Coherence collapses when ONE edit gets
#      large. Nine small edits can sum to a much larger total push while each stays inside the
#      window Phase 2 cleared. The ladder tested "more push in one place", never "the same push
#      spread out".
#
# LAYERS: the probe plateau (class readable at 0.89-0.93 from L10 to L23) plus L24/L27, where the
# direction is weaker but still real. Each layer uses ITS OWN direction and ITS OWN class-unit.
#
# DOSE is per-layer, as a fraction of that layer's LIVE residual norm. Phase 3's single-layer
# operating point was 0.082, so a 9-layer stack at 0.027 / 0.082 / 0.16 delivers roughly 3x / 9x
# / 18x that total while each individual edit stays at or below the dose already shown to be
# damage-free on its own.
#
# EVERY DOSE GETS A SHUFFLED-LABEL TWIN. Nine perturbations move the model more than one; without
# the paired null, "the output changed" cannot be separated from "we perturbed it nine times".
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
DIRS="${DIRS:-/data2/ds85/bgcmodel_runs/class_probe_sweep/valtest30_stack.steerdirs.npz}"
CLASSES="${CLASSES:-NRPS PKS TERPENE RIPP}"
STACK="${STACK:-10,12,14,16,18,20,22,24,27}"
NULL_PREFIX="${NULL_PREFIX:-perm0_}"
FRACS="${FRACS:-0.027 0.082 0.16}"
PER_CLASS="${PER_CLASS:-3}"
SEED_NT="${SEED_NT:-1000}"      # identical to Phase 3 => pairs with its C_unsteered arm
MAX_NEW="${MAX_NEW:-3000}"
SEED="${SEED:-42}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/steer_stack}"
mkdir -p "$ROOT"

[ -s "$DIRS" ] || { echo "[stack] ABORT: no directions at $DIRS"; exit 1; }
[ -s "$PFAM" ] || { echo "[stack] ABORT: no Pfam HMM at $PFAM -- marker scoring would be a no-op"; exit 1; }

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

run_arm(){                              # $1=arm  $2=per-layer frac  $3=dir prefix
  local arm="$1" frac="$2" pfx="$3"
  local out="$ROOT/${arm}.jsonl"
  if [ -s "$out" ]; then echo "[stack] $arm exists, skip"; return; fi
  echo "[stack] $(date) GEN $arm (layers=$STACK frac=$frac dir='${pfx:-real}')"; wait_for_idle
  PY evo2/scripts/seed_generate.py --adapter "$V2" --from-jsonl "$SEEDSRC" \
     --classes $CLASSES --per-class "$PER_CLASS" --seed-nt "$SEED_NT" \
     --max-new-tokens "$MAX_NEW" --no-class-tag --seed "$SEED" \
     --steer-dirs-npz "$DIRS" --steer-layer "$STACK" \
     --steer-norm-frac "$frac" --steer-dir-prefix "$pfx" --steer-toward rotate \
     --out-jsonl "$out" > "$ROOT/gen_${arm}.log" 2>&1
  [ -s "$out" ] && echo "[stack] $arm: $(wc -l < "$out") records" \
                || echo "[stack] !! $arm produced nothing (see $ROOT/gen_${arm}.log)"
}

for F in $FRACS; do
  run_arm "A_real_f${F}"     "$F" ""
  run_arm "B_shuffled_f${F}" "$F" "$NULL_PREFIX"
done

echo "[stack] $(date) scoring coherence + class markers"
PY - "$ROOT" "$PFAM" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/ds85/projects/BCGModelling/src")
from bgc_pipeline.evaluation import check_class_markers, check_coding_sanity
root, pfam = Path(sys.argv[1]), Path(sys.argv[2])
if not pfam.exists():
    raise SystemExit(f"[stack] ABORT: Pfam HMM missing at {pfam}")
tsv = root / "per_sequence.tsv"
n_skip = 0
arms = sorted(p.stem for p in root.glob("*.jsonl"))
with tsv.open("w") as fh:
    fh.write("arm\tseed_class\ttarget_class\tidx\tlength\tskipped\tcoding_density\t"
             "applied_frac\tn_layers\tmarkers_target\tmarkers_seed\toverride\n")
    for arm in arms:
        for i, line in enumerate((root / f"{arm}.jsonl").open()):
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("sequence", "") or ""
            sc, tc = r.get("seed_class"), (r.get("steer_target_class") or "-")
            skipped, cd, mt, ms = False, "", False, False
            if len(s) < 200:
                skipped = True
            else:
                cd = round(check_coding_sanity(s).get("coding_density", float("nan")), 4)
                rt = check_class_markers(s, expected_class=tc, pfam_hmm_path=pfam) if tc != "-" else {}
                rs = check_class_markers(s, expected_class=sc, pfam_hmm_path=pfam)
                if rt.get("skipped") or rs.get("skipped"):   # a SKIP is not a negative
                    skipped = True
                    n_skip += 1
                mt, ms = bool(rt.get("markers_present")), bool(rs.get("markers_present"))
            fh.write(f"{arm}\t{sc}\t{tc}\t{i}\t{len(s)}\t{int(skipped)}\t{cd}\t"
                     f"{r.get('steer_realized_norm_frac','')}\t{r.get('steer_n_layers','')}\t"
                     f"{int(mt)}\t{int(ms)}\t{int(mt and not ms)}\n")
        print(f"[stack]   scored {arm}", flush=True)
print(f"[stack] per-sequence -> {tsv}   (skipped: {n_skip})")
PYEOF

echo "==================== MULTI-LAYER STEERING ===================="
PY - "$ROOT/per_sequence.tsv" <<'PYEOF'
import sys, csv, collections, statistics as st
from math import comb
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
if not rows:
    print("no rows"); raise SystemExit(0)
n_skipped = sum(1 for r in rows if r.get("skipped") == "1")
by = collections.defaultdict(list)
for r in rows:
    if r.get("skipped") == "1":
        continue
    by[r["arm"]].append(r)
if n_skipped:
    print(f"({n_skipped} sequences excluded: marker scan could not run)")
print(f"{'arm':>22} {'n':>4} {'frac':>6} {'layers':>7} {'coding_density':>15} "
      f"{'target mk':>10} {'seed mk':>8} {'OVERRIDE':>9}")
rate = {}
for arm in sorted(by):
    rs = by[arm]; n = len(rs)
    cd = [float(r["coding_density"]) for r in rs if r["coding_density"]]
    ap = [float(r["applied_frac"]) for r in rs if r["applied_frac"]]
    nl = [int(r["n_layers"]) for r in rs if r["n_layers"]]
    rate[arm] = [int(r["override"]) for r in rs]
    print(f"{arm:>22} {n:>4} {st.mean(ap) if ap else float('nan'):>6.3f} "
          f"{max(nl) if nl else 0:>7} {st.mean(cd) if cd else float('nan'):>15.3f} "
          f"{sum(int(r['markers_target']) for r in rs)/n:>10.3f} "
          f"{sum(int(r['markers_seed']) for r in rs)/n:>8.3f} "
          f"{sum(int(r['override']) for r in rs)/n:>9.3f}")
for aa in sorted(k for k in rate if k.startswith("A_real")):
    bb = "B_shuffled_f" + aa.split("_f")[-1]
    if bb not in rate or len(rate[aa]) != len(rate[bb]):
        continue
    a, b = rate[aa], rate[bb]
    da = sum(1 for x, y in zip(a, b) if x and not y)
    db = sum(1 for x, y in zip(a, b) if y and not x)
    n = da + db
    p = sum(comb(n, i) * 0.5 ** n for i in range(da, n + 1)) if n else 1.0
    print(f"\nPAIRED {aa} vs {bb}: override in A only {da}, in B only {db}, "
          f"discordant {n}, one-sided p = {p:.4f}")
    if n < 10:
        print(f"  underpowered: smallest attainable p at this discordance is {0.5 ** n:.4f}")
print("\nNOTE the binary marker gate has TPR 0.717 on real class DNA at this length, so it can")
print("only see a large effect. Run probe_score_generations.py on these arms for the continuous")
print("readout -- that is what decides whether a SMALL multi-layer effect exists.")
PYEOF
echo "[stack] ALL DONE $(date)"
