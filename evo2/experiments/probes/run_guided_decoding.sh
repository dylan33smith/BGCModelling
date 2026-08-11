#!/usr/bin/env bash
# A1 — DISCRIMINATOR-GUIDED DECODING. Search output space instead of asking the model for a class.
#
# WHY THIS IS THE RIGHT NEXT EXPERIMENT. Every lever this project has closed required the GENERATOR
# to consume a class conditioner -- label token, soft prefix, activation edit -- and the 2026-08-10
# programme established it does not. Guided decoding needs none of that: sample continuations,
# score them with an EXTERNAL classifier, keep the best. The model only has to produce class-X-ish
# output sometimes; the scorer only has to recognise it. That sidesteps the exact failure mode.
#
# REGIME: SEEDED, MATCHED CLASS. Seed with 1000 nt of a real class-X core and guide the
# continuation toward X. This is not a convenience -- it is forced by the measurements. The
# UNSEEDED regime has no floor for selection to act on: antiSMASH is_bgc 0/25 (Phase 2) and
# correct_class 0/12 in every soft-prefix arm. Best-of-N there is choosing among N candidates
# none of which is a BGC, and it can only return one. Seeded-matched has a measured floor and
# ceiling (correct_class 0.283 matched vs 0.067 mismatched, n=60), i.e. real dynamic range.
# The claim this can support is therefore "an external scorer makes seeded generation MORE
# reliably on-class", which is narrower than de-novo class control -- state it that way.
#
# ARMS (each class run three ways, differing ONLY in the selection rule):
#   best    guided -- keep the candidate the discriminator scores highest for the target
#   random  CONTROL -- same candidates, keep a random one. Best-of-N with a USELESS scorer IS
#           random selection among N, so whatever `best` does not beat `random` by is resampling
#           variance and chunk-boundary effects, not the discriminator doing work.
# Identical --seed and --tag-class across every arm => identical taxa in the same order, and
# identical candidate sets at chunk 0. Paired per taxon.
#
# TWO QUESTIONS, KEPT SEPARATE -- this is pre-registered before the data lands:
#   Q1 MANIPULATION CHECK. Did selection actually raise the quantity we selected on (the guide
#      score), end-to-end? Greedy chunk-level selection can be MYOPIC: picking the best chunk-k
#      candidate can land in a region where every chunk-k+1 candidate is worse. Observed in the
#      n=1 smoke test. If Q1 FAILS, the experiment is INCONCLUSIVE about Q2 and the correct
#      follow-up is whole-sequence best-of-N (--n-chunks 1), not a claim that guidance fails.
#   Q2 THE RESULT. Did class-correctness rise, measured by instruments that took NO PART in
#      selection -- antiSMASH (FPR 0.000) and Pfam class markers?
#
# CIRCULARITY. We select on the discriminator score, so reporting that score as the result would
# be success by construction. `guide_*` fields are the manipulation check ONLY. The result comes
# from antiSMASH and markers. The summary below enforces the separation visually.
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
DISC="${DISC:-/data2/ds85/bgcmodel_runs/class_probe_sweep/prefix_discriminator_L16.joblib}"
TAXSRC="${TAXSRC:-/data2/ds85/bgcmodel_data/splits_core/valtest_eval.jsonl}"
CLASSES="${CLASSES:-NRPS PKS TERPENE RIPP}"
N="${N:-10}"
NCAND="${NCAND:-4}"
CHUNK="${CHUNK:-600}"
NCHUNK="${NCHUNK:-5}"
SEED_NT="${SEED_NT:-1000}"
SEED="${SEED:-42}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/guided_decoding}"
mkdir -p "$ROOT"

[ -s "$DISC" ] || { echo "[gd] ABORT: no discriminator at $DISC (run train_prefix_discriminator.py)"; exit 1; }
[ -s "$PFAM" ] || { echo "[gd] ABORT: no Pfam HMM at $PFAM -- marker scoring would be a no-op"; exit 1; }
TAGLIST=$(echo $CLASSES | tr ' ' ',')

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

# plain (N=1) runs first: it is 1/NCAND the cost and establishes the floor under THIS pipeline
# before any expensive arm, so a broken pipeline is caught cheaply.
for C in $CLASSES; do
  for SPEC in "plain 1 best" "random $NCAND random" "best $NCAND best"; do
    set -- $SPEC; ARM=$1; K=$2; RULE=$3
    OUT="$ROOT/gd_${C}_${ARM}.jsonl"
    [ -s "$OUT" ] && { echo "[gd] $(basename $OUT) exists, skip"; continue; }
    echo "[gd] $(date) GEN target=$C arm=$ARM (n_candidates=$K, rule=$RULE)"; wait_for_idle
    PY evo2/scripts/guided_generate.py --adapter "$V2" --discriminator "$DISC" \
       --from-jsonl "$TAXSRC" --seed-src "$TAXSRC" --seed-nt "$SEED_NT" \
       --target-class "$C" --tag-class "$TAGLIST" \
       --n "$N" --n-candidates "$K" --chunk-nt "$CHUNK" --n-chunks "$NCHUNK" \
       --select "$RULE" --seed "$SEED" --out-jsonl "$OUT" > "$ROOT/gen_${C}_${ARM}.log" 2>&1
    [ -s "$OUT" ] && echo "[gd] $(basename $OUT): $(wc -l < "$OUT") records" \
                  || echo "[gd] !! FAILED (see gen_${C}_${ARM}.log)"
  done
done

echo
echo "[gd] $(date) Q1 MANIPULATION CHECK — did selection move what we selected on?"
PY - "$ROOT" $CLASSES <<'PYEOF'
import json, sys, statistics as st
from math import comb
from pathlib import Path
root = Path(sys.argv[1]); CLASSES = sys.argv[2:]
print("Mean guide objective (log P(target)) of the CHOSEN candidate, per chunk.")
print("If 'best' does not exceed 'random' here, greedy selection is MYOPIC and Q2 is")
print("INCONCLUSIVE -- the correct follow-up is whole-sequence best-of-N, not a null claim.\n")
print(f"{'class':>9} {'n':>3} | {'best':>9} {'random':>9} {'delta':>9} {'up':>7} {'sign p':>8}")
def sign(d):
    up = sum(1 for x in d if x > 0); n = len(d)
    return up, n, min(1.0, 2*sum(comb(n,k)*0.5**n for k in range(max(up,n-up), n+1)))
overall = []
for C in CLASSES:
    try:
        b = [json.loads(l) for l in (root/f"gd_{C}_best.jsonl").open()]
        r = [json.loads(l) for l in (root/f"gd_{C}_random.jsonl").open()]
    except FileNotFoundError:
        continue
    rb = {x["tax_idx"]: x for x in b}; rr = {x["tax_idx"]: x for x in r}
    def mean_obj(rec):
        v = [t["guide_obj_chosen"] for t in rec["guide_trace"] if "guide_obj_chosen" in t]
        return st.mean(v) if v else float("nan")
    d = [mean_obj(rb[k]) - mean_obj(rr[k]) for k in rb if k in rr]
    d = [x for x in d if x == x]
    if not d: continue
    up, n, p = sign(d)
    overall += d
    print(f"{C:>9} {n:>3} | {st.mean([mean_obj(rb[k]) for k in rb]):>9.4f} "
          f"{st.mean([mean_obj(rr[k]) for k in rr]):>9.4f} {st.mean(d):>+9.4f} "
          f"{up:>3}/{n:<3} {p:>8.4f}")
if overall:
    up, n, p = sign(overall)
    print(f"{'POOLED':>9} {n:>3} | {'':>9} {'':>9} {st.mean(overall):>+9.4f} {up:>3}/{n:<3} {p:>8.4f}")
    print("\n=> " + ("selection DID raise the guide score; Q2 below is interpretable."
                     if st.mean(overall) > 0 and p < 0.05 else
                     "selection did NOT reliably raise the guide score. Treat Q2 as INCONCLUSIVE "
                     "and run whole-sequence best-of-N (--n-chunks 1) before concluding anything."))
PYEOF

echo
echo "[gd] $(date) Q2 THE RESULT — instruments that took no part in selection"
PY evo2/scripts/score_generations_antismash.py "$ROOT"/gd_*.jsonl --expected compound_class \
   --workers 10 --allow-legacy --out-tsv "$ROOT/antismash.tsv" > "$ROOT/antismash.log" 2>&1
PY - "$ROOT" "$PFAM" $CLASSES <<'PYEOF'
import json, sys, csv, collections, statistics as st
from math import comb
from pathlib import Path
sys.path.insert(0, "/home/ds85/projects/BCGModelling/src")
from bgc_pipeline.evaluation import check_class_markers, check_coding_sanity
root, pfam = Path(sys.argv[1]), Path(sys.argv[2]); CLASSES = sys.argv[3:]
asr = {}
f = root/"antismash.tsv"
if f.exists():
    for r in csv.DictReader(f.open(), delimiter="\t"):
        a = asr.setdefault(r["arm"], [0,0,0]); a[2]+=1
        a[0]+= r.get("is_bgc") in ("1","True","true")
        a[1]+= r.get("correct_class") in ("1","True","true")
print(f"{'arm':>20} {'n':>4} {'coding':>8} {'markers(own)':>13} {'antiSMASH is_bgc':>17} {'correct_class':>14}")
mk = {}
for C in CLASSES:
    for rule in ("plain","random","best"):
        p = root/f"gd_{C}_{rule}.jsonl"
        if not p.exists(): continue
        recs = [json.loads(l) for l in p.open() if l.strip()]
        cd, hits = [], {}
        for r in recs:
            s = r.get("sequence","") or ""
            if len(s) < 200: continue
            cd.append(check_coding_sanity(s).get("coding_density", float("nan")))
            hits[r["tax_idx"]] = bool(check_class_markers(
                s, expected_class=C, pfam_hmm_path=pfam).get("markers_present"))
        mk[(C,rule)] = hits
        d,c,n = asr.get(f"gd_{C}_{rule}", [0,0,0])
        print(f"{C+'/'+rule:>20} {len(recs):>4} {st.mean(cd) if cd else float('nan'):>8.3f} "
              f"{sum(hits.values())}/{len(hits):<11} {(d/n if n else float('nan')):>17.3f} "
              f"{(c/n if n else float('nan')):>14.3f}")
print("\nPAIRED per taxon, markers for the target class: guided vs random, same taxon.")
print(f"{'class':>9} {'pairs':>6} {'best only':>10} {'random only':>12} {'sign p':>8}")
tot_b = tot_r = 0
for C in CLASSES:
    b, r = mk.get((C,"best"), {}), mk.get((C,"random"), {})
    keys = [k for k in b if k in r]
    ob = sum(1 for k in keys if b[k] and not r[k]); orr = sum(1 for k in keys if r[k] and not b[k])
    tot_b += ob; tot_r += orr
    n = ob+orr
    p = min(1.0, 2*sum(comb(n,k)*0.5**n for k in range(max(ob,orr), n+1))) if n else 1.0
    print(f"{C:>9} {len(keys):>6} {ob:>10} {orr:>12} {p:>8.4f}")
n = tot_b+tot_r
p = min(1.0, 2*sum(comb(n,k)*0.5**n for k in range(max(tot_b,tot_r), n+1))) if n else 1.0
print(f"{'POOLED':>9} {'':>6} {tot_b:>10} {tot_r:>12} {p:>8.4f}")
print("\nBaseline for scale: the published matched-seeded correct_class is 0.283 (n=60);")
print("the mismatched floor is 0.067. The `plain` arm above is this pipeline's own floor.")
print("\nREAD: guide_* is what we OPTIMISED and is NOT evidence. The lines above are.")
print("Discordant counts below ~10 are underpowered regardless of p.")
PYEOF
echo "[gd] ALL DONE $(date)"
