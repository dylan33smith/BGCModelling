#!/usr/bin/env bash
# CFG diagnostic: does classifier-free guidance amplify a latent CLASS signal in the
# existing v2 adapter? NO retraining. Sweeps guidance weight w and scores each with
# antiSMASH (the gold is_bgc/correct_class gate). Read primarily via the SHORT classes
# (TERPENE/ECTOINE/BETALACTONE) where an 8k window fully covers the cluster.
#
# Interpretation:
#   * w=1 reproduces plain conditional generation (anchor; should match the confirm run).
#   * correct_class RISES with w  -> class signal exists, just needs amplifying
#       => train-with-class-dropout + CFG is worth it.
#   * correct_class FLAT across w  -> prefix carries ~no class info => per-class adapters.
#   * watch coding_density/is_bgc: if they COLLAPSE at high w, guidance pushed the model
#       out of distribution (expected failure mode of CFG without a trained null).
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
ENV=bgcmodel
PY(){ micromamba run -n "$ENV" python "$@"; }

ADAPTER="${ADAPTER:-/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ASDB="${ASDB:-/data2/ds85/antismash_db}"
CLASSES="${CLASSES:-TERPENE ECTOINE BETALACTONE}"
WEIGHTS="${WEIGHTS:-1 3 5}"
PER_CLASS="${PER_CLASS:-5}"
MAX_NEW="${MAX_NEW:-8192}"
SEED="${SEED:-42}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/cfg_diagnostic}"
mkdir -p "$ROOT"

# prompt pool filtered to the target classes (shared across all w)
POOL="$ROOT/prompts.jsonl"
PY - "$VAL" "$POOL" "$CLASSES" <<'PYEOF'
import json,sys
val,out,classes=sys.argv[1],sys.argv[2],set(sys.argv[3].split())
keep=[r for r in (json.loads(l) for l in open(val)) if r.get("compound_class") in classes]
open(out,"w").write("".join(json.dumps(r)+"\n" for r in keep))
print(f"[cfg] prompt pool: {len(keep)} records across {len(classes)} classes")
PYEOF

wait_for_idle(){ local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then hold=$((hold+1)); [ "$hold" -ge 3 ] && return 0
    else hold=0; fi; sleep 10
  done; }

echo "[cfg] $(date) waiting for idle GPU..."
wait_for_idle

# 1) VALIDATION GATE (cheap): w=1 greedy must equal the non-cached oracle.
echo "[cfg] $(date) validation gate..."
PY evo2/scripts/cfg_generate.py --adapter "$ADAPTER" --from-jsonl "$POOL" \
   --classes $CLASSES --validate-only --seed "$SEED" --out-dir "$ROOT" 2>&1 | tee "$ROOT/validate.log"
if ! grep -q "validation passed" "$ROOT/validate.log"; then
  echo "[cfg] VALIDATION FAILED — aborting (see validate.log). Do NOT trust w>1."; exit 3
fi

# 2) SWEEP w, writing cfg_w{W}.jsonl each.
echo "[cfg] $(date) generating w sweep: $WEIGHTS"
wait_for_idle
PY evo2/scripts/cfg_generate.py --adapter "$ADAPTER" --from-jsonl "$POOL" \
   --classes $CLASSES --per-class "$PER_CLASS" --guidance-weights $WEIGHTS \
   --max-new-tokens "$MAX_NEW" --top-k 4 --top-p 1.0 --temperature 1.0 \
   --seed "$SEED" --out-dir "$ROOT" 2>&1 | tee "$ROOT/generate.log"

# 3) score each w with antiSMASH (+cheap checks), tabulate correct_class vs w.
SUM="$ROOT/cfg_summary.tsv"
printf "w\tn\tis_bgc\tcorrect_class\tclass_markers\tany_domain\tcoding_density\n" > "$SUM"
for f in "$ROOT"/cfg_w*.jsonl; do
  [ -s "$f" ] || continue
  w=$(basename "$f" .jsonl | sed 's/^cfg_w//')
  echo "[cfg] $(date) scoring w=$w ..."
  # POSITIVE CONTROL: real held-out cores truncated to THIS run's own length distribution
  # and class mix. Until 2026-08-10 every probe driver passed an empty `_nopos.jsonl`, so
  # 0 of 25 reports had a ceiling and every rate was an uncalibrated fraction of an
  # unstated maximum. Generated per-eval because the ceiling depends on generation length.
  PY scripts/make_positive_control.py --gen "$f" --out "$ROOT/positive_control.jsonl" || true
  PY scripts/eval_suite_driver.py --gen "$f" --positive "$ROOT/positive_control.jsonl" \
     --skip-checks protein_homology kmer_novelty \
     --pfam-hmm "$PFAM" --antismash-db "$ASDB" --output "$ROOT/report_w$w.json" \
     > "$ROOT/eval_w$w.log" 2>&1
  PY - "$w" "$ROOT/report_w$w.json" "$SUM" "$ROOT/per_class_w$w.tsv" <<'PYEOF'
import json,sys,collections
w,rep_p,sum_p,pc_p=sys.argv[1:5]
rep=json.load(open(rep_p)); g=rep["generated"]; recs=rep.get("per_record",{}).get("generated",[])
qr=lambda q:g.get("per_question",{}).get(q,{}).get("pass_rate")
cr=lambda c:g.get("per_check",{}).get(c,{}).get("pass_rate")
any_dom=sum(1 for r in recs if (r.get("class_markers",{}) or {}).get("domain_count",0)>0)
cod=[ (r.get("coding_sanity",{}) or {}).get("coding_density") for r in recs ]; cod=[c for c in cod if c is not None]
row=[w,g.get("n"),qr("is_bgc"),qr("correct_class"),cr("class_markers"),
     round(any_dom/len(recs),3) if recs else None, round(sum(cod)/len(cod),3) if cod else None]
open(sum_p,"a").write("\t".join(str(x) for x in row)+"\n")
by=collections.defaultdict(lambda:[0,0,0])
for r in recs:
    c=r.get("expected_class"); a=r.get("antismash",{}) or {}; by[c][2]+=1
    if a.get("detected"): by[c][1]+=1
    if a.get("detected") and a.get("class_match"): by[c][0]+=1
with open(pc_p,"w") as fh:
    fh.write("class\tcorrect\tis_bgc\tn\n")
    for c in sorted(by): cc,ib,n=by[c]; fh.write(f"{c}\t{cc}\t{ib}\t{n}\n")
print(f"[cfg] w={w}: is_bgc={row[2]} correct_class={row[3]} coding_density={row[6]}")
PYEOF
done

echo "==================== CFG SWEEP SUMMARY ===================="
column -t "$SUM" 2>/dev/null || cat "$SUM"
echo "[cfg] ALL DONE $(date)"
