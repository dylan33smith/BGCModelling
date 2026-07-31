#!/usr/bin/env bash
# PHASE 2 — how hard can we push the CORRECTED class direction before the DNA stops looking
# like genes?  Output: the operating dose for Phase 3.
#
# WHY THIS IS NOT THE EARLIER TITRATION. run_steer_magnitude.sh titrated ||delta|| along the
# LEGACY direction, which was ~the sequence-length axis (PC1, 98% of variance, r=-0.9996 with
# ||h||) — a different subspace with different tolerance. The corrected direction is
# length-stripped and much lower-variance, so its damage curve has to be measured fresh.
#
# DOSE UNITS. Class-units, identical to Phase 1 (`--class-units`): 1 unit = the distance from the
# other-class mean to this class's mean along the direction. Phase 1 found the causal effect at
# **dose 1** (p=0.040, 0/24 shuffled controls as extreme) and NOT at dose 4 — where the control
# spread grew 8x while the real effect grew 4x. So the interesting band is at and below 1, and
# the grid is centred there rather than on the wide range the legacy titration used.
#
# READOUT. max_orf_aa and ORF count, NOT coding_density. Measured on real control generations,
# coding_density reads 0.888 for marker-PASS vs 0.879 for marker-FAIL (delta 0.009 — it cannot
# tell good output from bad), while max_orf_aa separates them 858 vs 605. Realized length is
# also primary: steering suppresses EOS, and length differences manufacture downstream artifacts.
#
# CONTROL ARM. A shuffled-label direction at the top dose. Without it, "the sequence degraded"
# cannot be separated from "any perturbation of that size degrades the sequence".
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
# Keep scratch off the shared /home (it has hit 100%, which breaks micromamba's process lock).
export TMPDIR="${TMPDIR:-/data2/ds85/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data2/ds85/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data2/ds85/cache/mpl}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
ENV=bgcmodel
PY(){ micromamba run -n "$ENV" python "$@"; }

V2="${V2:-/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200}"
TAGSRC="${TAGSRC:-/data2/ds85/bgcmodel_data/splits_core/valtest_eval.jsonl}"
DIRS="${DIRS:-/data2/ds85/bgcmodel_runs/class_probe_sweep/valtest30.steerdirs.npz}"
ACTS="${ACTS:-/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_valtest_fit.npz}"
CLASSES="${CLASSES:-NRPS PKS TERPENE ECTOINE RIPP}"
DOSES="${DOSES:-0 0.25 0.5 1 2 4}"
NULL_DOSE="${NULL_DOSE:-4}"            # shuffled-label arm runs at the top dose
NULL_PREFIX="${NULL_PREFIX:-perm0_}"
LAYER="${LAYER:-16}"
PER_CLASS="${PER_CLASS:-8}"            # 8 x 5 classes = 40 seqs/cell
MAX_NEW="${MAX_NEW:-2048}"
SEED="${SEED:-42}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/steer_phase2}"
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

SEQTSV="$ROOT/per_sequence.tsv"
printf "arm\tdose\tdir_prefix\tcompound_class\tseq_idx\tlength\thit_eos\tcoding_density\tmax_orf_aa\tn_orfs\n" > "$SEQTSV"

gen_and_score(){                       # $1=arm label  $2=dose  $3=dir prefix
  local arm="$1" dose="$2" pfx="$3"
  local out="$ROOT/${arm}.jsonl"
  if [ ! -s "$out" ]; then
    echo "[p2] $(date) GEN $arm (dose=$dose class-units, dir='${pfx:-real}')"; wait_for_idle
    PY evo2/scripts/steer_generate.py --adapter "$V2" --from-jsonl "$TAGSRC" \
       --dirs-npz "$DIRS" --acts-npz "$ACTS" \
       --classes $CLASSES --layer "$LAYER" --class-units "$dose" --dir-prefix "$pfx" \
       --per-class "$PER_CLASS" --max-new-tokens "$MAX_NEW" \
       --top-k 4 --top-p 1.0 --temperature 1.0 --seed "$SEED" \
       --out-jsonl "$out" > "$ROOT/gen_${arm}.log" 2>&1
  fi
  [ -s "$out" ] || { echo "[p2] !! $arm produced nothing (see $ROOT/gen_${arm}.log)"; return; }
  PY - "$arm" "$dose" "$pfx" "$out" "$SEQTSV" <<'PYEOF'
import json, sys, collections, statistics as st
sys.path.insert(0, "/home/ds85/projects/BCGModelling/src")
from bgc_pipeline.evaluation import check_coding_sanity
arm, dose, pfx, gen_p, seq_p = sys.argv[1:6]
idx = collections.Counter(); agg = collections.defaultdict(list)
with open(seq_p, "a") as fh:
    for line in open(gen_p):
        if not line.strip():
            continue
        r = json.loads(line); s = r.get("sequence", "") or ""; c = r.get("compound_class", "?")
        i = idx[c]; idx[c] += 1
        if len(s) < 100:
            cd, mo, no = 0.0, 0, 0
        else:
            m = check_coding_sanity(s)
            cd, mo, no = m.get("coding_density", 0.0), m.get("max_orf_aa", 0), m.get("n_orfs", 0)
        agg["len"].append(len(s)); agg["cd"].append(cd); agg["mo"].append(mo); agg["no"].append(no)
        fh.write(f"{arm}\t{dose}\t{pfx or 'real'}\t{c}\t{i}\t{len(s)}\t{r.get('hit_eos')}\t"
                 f"{cd:.4f}\t{mo}\t{no}\n")
m = lambda k: st.mean(agg[k]) if agg[k] else 0.0
print(f"[p2] {arm}: n={len(agg['len'])} max_orf_aa={m('mo'):.0f} n_orfs={m('no'):.2f} "
      f"len={m('len'):.0f} coding_density={m('cd'):.3f}")
PYEOF
}

for D in $DOSES; do gen_and_score "d${D}" "$D" ""; done
gen_and_score "null_d${NULL_DOSE}" "$NULL_DOSE" "$NULL_PREFIX"

echo "==================== PHASE 2 — USABLE DOSE (L$LAYER) ===================="
PY - "$SEQTSV" "$NULL_DOSE" <<'PYEOF'
import sys, csv, collections, statistics as st
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
if not rows:
    print("no rows"); raise SystemExit(0)
by = collections.defaultdict(list)
for r in rows:
    by[r["arm"]].append(r)
def agg(rs, k, cast=float):
    v = [cast(r[k]) for r in rs]
    n = len(v)
    return (st.mean(v), (st.stdev(v) / n ** 0.5 if n > 1 else 0.0), n)
base = by.get("d0")
print(f"{'arm':>10} {'dose':>6} {'max_orf_aa':>18} {'n_orfs':>13} {'length':>15} {'coding_den':>11}")
for arm in sorted(by, key=lambda a: (a.startswith("null"), float(by[a][0]['dose']))):
    rs = by[arm]
    mo, mose, n = agg(rs, "max_orf_aa"); no, _, _ = agg(rs, "n_orfs")
    ln, lnse, _ = agg(rs, "length"); cd, _, _ = agg(rs, "coding_density")
    print(f"{arm:>10} {rs[0]['dose']:>6} {mo:>11.0f} +/-{mose:>5.0f} {no:>13.2f} "
          f"{ln:>10.0f} +/-{lnse:>4.0f} {cd:>11.3f}")
if base:
    bmo, bmose, _ = agg(base, "max_orf_aa"); bln, _, _ = agg(base, "length")
    print("\nUSABLE-DOSE RULE: largest dose whose max_orf_aa 95% CI overlaps the unsteered arm")
    print("                  AND whose mean length is within 10% of unsteered.")
    ok = []
    for arm in sorted(by, key=lambda a: float(by[a][0]['dose'])):
        if arm.startswith("null") or arm == "d0":
            continue
        rs = by[arm]; mo, mose, _ = agg(rs, "max_orf_aa"); ln, _, _ = agg(rs, "length")
        # ONE-SIDED: we care about DAMAGE, not difference. A two-sided test flagged dose 4 as
        # "DEGRADED" when its ORFs were significantly LONGER than unsteered (419 vs 270).
        orf_ok = (mo - bmo) >= -2 * (mose ** 2 + bmose ** 2) ** 0.5
        len_ok = abs(ln - bln) / max(bln, 1) <= 0.10
        print(f"  dose {rs[0]['dose']:>5}: max_orf_aa {'OK ' if orf_ok else 'DEGRADED'} "
              f"({mo:.0f} vs {bmo:.0f}{'  IMPROVED' if mo > bmo else ''})   "
              f"length {'OK ' if len_ok else 'SHIFTED'} "
              f"({ln:.0f} vs {bln:.0f})")
        if orf_ok and len_ok:
            ok.append(float(rs[0]['dose']))
    print(f"\n  => usable dose = {max(ok) if ok else 'NONE >= 0.25'}")
    if not ok:
        print("     No dose preserves the sequence. Report and stop — a direction that cannot")
        print("     take a quarter-step without ORF collapse has no operating point.")
    null = [a for a in by if a.startswith("null")]
    if null:
        nrs = by[null[0]]; nmo, _, _ = agg(nrs, "max_orf_aa")
        top = by.get(f"d{sys.argv[2]}")
        if top:
            tmo, _, _ = agg(top, "max_orf_aa")
            print(f"\n  CONTROL at dose {sys.argv[2]}: real max_orf_aa {tmo:.0f} vs "
                  f"shuffled-label {nmo:.0f}")
            print("    (similar => the damage is generic to perturbation, not specific to the")
            print("     class direction, which is what we expect and want)")
PYEOF
echo "[p2] per-sequence: $SEQTSV"
echo "[p2] ALL DONE $(date)"
