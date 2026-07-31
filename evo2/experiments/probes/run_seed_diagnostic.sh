#!/usr/bin/env bash
# Nucleotide-context SEEDING diagnostic. Does handing Evo2 a real class-defining ORF
# (in-context) make it continue in-class? Scores CONTINUATION ONLY (seed is prompt →
# stripped by vortex → never scored), so a correct-class continuation is the model's
# own contribution. Read against the no-seed floor (megasynthase correct_class ~0.01-0.07).
#
# Arms:
#   base_seed_notag : base Evo2, NO class tag, prompt = {tax}+seed  → pure native handle
#   v2_seed_tag     : v2 adapter + class tag,  prompt = |COMPOUND_CLASS:X|{tax}+seed → best practical
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
ENV=bgcmodel
PY(){ micromamba run -n "$ENV" python "$@"; }

V2="${V2:-/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ASDB="${ASDB:-/data2/ds85/antismash_db}"
CLASSES="${CLASSES:-NRPS PKS TERPENE}"
PER_CLASS="${PER_CLASS:-10}"
SEED_NT="${SEED_NT:-2000}"
MAX_NEW="${MAX_NEW:-6000}"
SEED="${SEED:-42}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/seed_diagnostic}"
mkdir -p "$ROOT"; : > "$ROOT/_nopos.jsonl"
SUM="$ROOT/seed_summary.tsv"
printf "arm\tclass\tcorrect\tis_bgc\tmarkers\tn\tcoding_density_mean\n" > "$SUM"

wait_for_idle(){ local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then hold=$((hold+1)); [ "$hold" -ge 3 ] && return 0
    else hold=0; fi; sleep 10
  done; }

# arm name | adapter-arg | extra seed_generate flags
run_arm(){
  local name="$1" adapter_arg="$2" extra="$3"
  echo "[seed] $(date) ARM $name"; wait_for_idle
  PY evo2/scripts/seed_generate.py $adapter_arg --from-jsonl "$VAL" --classes $CLASSES \
     --per-class "$PER_CLASS" --seed-nt "$SEED_NT" --max-new-tokens "$MAX_NEW" \
     --top-k 4 --top-p 1.0 --temperature 1.0 --seed "$SEED" $extra \
     --out-jsonl "$ROOT/$name.jsonl" > "$ROOT/gen_$name.log" 2>&1
  [ -s "$ROOT/$name.jsonl" ] || { echo "[seed] $name GEN FAILED (see gen_$name.log)"; return; }
  PY scripts/eval_suite_driver.py --gen "$ROOT/$name.jsonl" --positive "$ROOT/_nopos.jsonl" \
     --skip-checks protein_homology kmer_novelty \
     --pfam-hmm "$PFAM" --antismash-db "$ASDB" --output "$ROOT/report_$name.json" \
     > "$ROOT/eval_$name.log" 2>&1
  PY - "$name" "$ROOT/report_$name.json" "$SUM" <<'PYEOF'
import json,sys,collections
name,rep_p,sum_p=sys.argv[1:4]
rep=json.load(open(rep_p)); recs=rep.get("per_record",{}).get("generated",[])
by=collections.defaultdict(lambda:[0,0,0,0,[]])  # [correct,is_bgc,markers,n,coding]
for r in recs:
    c=r.get("expected_class"); a=r.get("antismash",{}) or {}; cm=r.get("class_markers",{}) or {}
    cs=r.get("coding_sanity",{}) or {}; by[c][3]+=1
    if a.get("detected"): by[c][1]+=1
    if a.get("detected") and a.get("class_match"): by[c][0]+=1
    if cm.get("domain_count",0)>0 and not cm.get("skipped"): by[c][2]+=1
    if "coding_density" in cs: by[c][4].append(cs["coding_density"])
with open(sum_p,"a") as fh:
    for c in sorted(by):
        cc,ib,mk,n,cd=by[c]; cdm=round(sum(cd)/len(cd),3) if cd else None
        fh.write(f"{name}\t{c}\t{cc}\t{ib}\t{mk}\t{n}\t{cdm}\n")
        print(f"[seed] {name} {c:10s} correct {cc}/{n}  is_bgc {ib}/{n}  markers {mk}/{n}  coding {cdm}")
PYEOF
  echo "[seed] $(date) $name DONE"
}

run_arm "base_seed_notag" ""                "--no-class-tag"
run_arm "v2_seed_tag"     "--adapter $V2"   ""

echo "==================== SEED DIAGNOSTIC SUMMARY ===================="
column -t "$SUM" 2>/dev/null || cat "$SUM"
echo "[seed] compare vs NO-SEED floor: megasynthase correct_class ~0.01-0.07 (v2), ~0 (base)"
echo "[seed] ALL DONE $(date)"
