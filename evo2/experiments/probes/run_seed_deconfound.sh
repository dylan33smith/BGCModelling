#!/usr/bin/env bash
# SEED DE-CONFOUND — settles whether the seeding lift (agg correct_class 0.37) is real
# class-conditioning or an artifact. Adversary verdict on the n=10 pilot: "likely-inflated"
# via (1) trivial gene-continuation, (2) memorization, (3) adapter-vs-tag confound.
#
# ARMS (seed sequence held IDENTICAL per record across arms; same RNG seed):
#   base_tag       base Evo2 + tag        ]  the 2x2 factorial: the critical never-run cell
#   base_notag     base Evo2, no tag      ]  is v2_notag — it separates "the ADAPTER converts
#   v2_tag         v2 adapter + tag       ]  the exemplar" from "the TAG is load-bearing".
#   v2_notag       v2 adapter, no tag     ]
#   v2_tag_trunc   v2 + tag, codon-truncated seed (NO orf spans seed->continuation)
#   v2_tag_shuf    v2 + tag, codon-shuffled seed (composition kept, gene structure destroyed)
#   v2_mismatch    v2 + tag of a DIFFERENT class than the seed (does the call track SEED or TAG?)
#
# GATES: novelty ON (memorization_check vs the training corpus) — a correct_class hit only
# counts if it PASSES novelty. Plus a byte-level leakage audit (scored seq must not contain
# the seed prefix).
#
# DECISION: REAL iff v2_notag ~ v2_tag AND survives novel-only AND shuffled/mismatch collapse
# AND truncated keeps the lift. Any collapse on novel-only or truncation => gene-continuation
# and/or memorization, NOT class installation.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
ENV=bgcmodel
PY(){ micromamba run -n "$ENV" python "$@"; }

V2="${V2:-/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
TRAIN="${TRAIN:-/data2/ds85/bgcmodel_data/splits_core/train.jsonl}"   # novelty reference corpus
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ASDB="${ASDB:-/data2/ds85/antismash_db}"
CLASSES="${CLASSES:-NRPS PKS PKS_NRPS_HYBRID TERPENE}"
PER_CLASS="${PER_CLASS:-15}"          # Dylan's reliability bar
SEED_NT="${SEED_NT:-2000}"
MAX_NEW="${MAX_NEW:-6000}"
SEED="${SEED:-42}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/seed_deconfound}"
mkdir -p "$ROOT"; : > "$ROOT/_nopos.jsonl"

wait_for_idle(){ local hold=0 proc free
  while true; do
    proc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ "${proc:-1}" -eq 0 ] && [ "${free:-0}" -ge 70000 ]; then hold=$((hold+1)); [ "$hold" -ge 3 ] && return 0
    else hold=0; fi; sleep 15
  done; }

# name | adapter | extra flags
ARMS=(
  "base_notag|| --no-class-tag"
  "base_tag||"
  "v2_notag|--adapter $V2| --no-class-tag"
  "v2_tag|--adapter $V2|"
  "v2_tag_trunc|--adapter $V2| --no-boundary-orf"
  "v2_tag_shuf|--adapter $V2| --shuffle-seed"
  "v2_mismatch|--adapter $V2| --mismatch-tag"
)

echo "[deconf] $(date) START — ${#ARMS[@]} arms x ${PER_CLASS}/class over: $CLASSES"
for entry in "${ARMS[@]}"; do
  IFS='|' read -r name adapter extra <<< "$entry"
  [ -s "$ROOT/$name.jsonl" ] && { echo "[deconf] $name already generated, skipping"; continue; }
  echo "[deconf] $(date) GEN $name"; wait_for_idle
  PY evo2/scripts/seed_generate.py $adapter --from-jsonl "$VAL" --classes $CLASSES \
     --per-class "$PER_CLASS" --seed-nt "$SEED_NT" --max-new-tokens "$MAX_NEW" \
     --top-k 4 --top-p 1.0 --temperature 1.0 --seed "$SEED" $extra \
     --out-jsonl "$ROOT/$name.jsonl" > "$ROOT/gen_$name.log" 2>&1
  [ -s "$ROOT/$name.jsonl" ] || echo "[deconf] !! $name GEN FAILED (see gen_$name.log)"
done

# ── novelty (memorization) scan: ALL arms' continuations vs the TRAINING corpus ──
echo "[deconf] $(date) novelty scan vs train corpus"
cat "$ROOT"/*.jsonl > "$ROOT/_allgen.jsonl" 2>/dev/null
PY scripts/memorization_check.py --query "$ROOT/_allgen.jsonl" --ref "$TRAIN" \
   --output "$ROOT/novelty.json" > "$ROOT/novelty.log" 2>&1 \
   && echo "[deconf] novelty scan done" || echo "[deconf] !! novelty scan failed (see novelty.log)"

# ── score every arm (antiSMASH + cheap checks), novelty passed in (NOT skipped) ──
SUM="$ROOT/deconfound_summary.tsv"
printf "arm\tn\tis_bgc\tcorrect_class\tcorrect_novel_only\ttracks_seed\tmarkers\tcoding_density\tleak\n" > "$SUM"
for entry in "${ARMS[@]}"; do
  IFS='|' read -r name adapter extra <<< "$entry"
  [ -s "$ROOT/$name.jsonl" ] || continue
  echo "[deconf] $(date) EVAL $name"
  NOVARG=""; [ -s "$ROOT/novelty.json" ] && NOVARG="--novelty $ROOT/novelty.json"
  PY scripts/eval_suite_driver.py --gen "$ROOT/$name.jsonl" --positive "$ROOT/_nopos.jsonl" \
     --skip-checks protein_homology $NOVARG \
     --pfam-hmm "$PFAM" --antismash-db "$ASDB" --output "$ROOT/report_$name.json" \
     > "$ROOT/eval_$name.log" 2>&1
  PY - "$name" "$ROOT/report_$name.json" "$ROOT/$name.jsonl" "$SUM" "$ROOT/per_class_$name.tsv" <<'PYEOF'
import json,sys,collections
name,rep_p,gen_p,sum_p,pc_p=sys.argv[1:6]
rep=json.load(open(rep_p)); recs=rep.get("per_record",{}).get("generated",[])
gen={}
for line in open(gen_p):
    r=json.loads(line); gen[r.get("sequence","")[:80]]=r      # match on sequence prefix
n=len(recs); ib=cc=cn=ts=mk=leak=0; cod=[]
by=collections.defaultdict(lambda:[0,0,0])
for r in recs:
    seq=r.get("sequence","") or ""
    g=gen.get(seq[:80],{})
    a=r.get("antismash",{}) or {}; cm=r.get("class_markers",{}) or {}; cs=r.get("coding_sanity",{}) or {}
    nov=r.get("kmer_novelty",{}) or {}
    exp=r.get("expected_class"); mapped=set(a.get("mapped_classes") or [])
    det=bool(a.get("detected")); match=bool(a.get("class_match"))
    novel_ok = (nov.get("pass") is not False)          # None (not run) treated as unknown-pass
    ib+=det; cc+= (det and match); cn += (det and match and novel_ok)
    seed_cls=g.get("seed_class")
    if det and seed_cls and seed_cls in mapped: ts+=1   # tracks the SEED's class
    if cm.get("domain_count",0)>0 and not cm.get("skipped"): mk+=1
    if "coding_density" in cs: cod.append(cs["coding_density"])
    sp=g.get("seed_prefix_64")
    if sp and sp in seq: leak+=1                        # leakage audit
    by[exp][2]+=1; by[exp][1]+=det; by[exp][0]+= (det and match)
f=lambda x: round(x/n,3) if n else None
open(sum_p,"a").write("\t".join(str(x) for x in
  [name,n,f(ib),f(cc),f(cn),f(ts),f(mk),
   round(sum(cod)/len(cod),3) if cod else None, leak])+"\n")
with open(pc_p,"w") as fh:
    fh.write("class\tcorrect\tis_bgc\tn\n")
    for c in sorted(by): a_,b_,c_=by[c]; fh.write(f"{c}\t{a_}\t{b_}\t{c_}\n")
print(f"[deconf] {name}: n={n} is_bgc={f(ib)} correct={f(cc)} correct_novel={f(cn)} tracks_seed={f(ts)} leak={leak}")
PYEOF
done

echo "==================== DE-CONFOUND SUMMARY ===================="
column -t "$SUM" 2>/dev/null || cat "$SUM"
cat <<'NOTE'

READ:
  correct_novel_only  <-- the number that matters (memorization-gated)
  tracks_seed         <-- fraction whose antiSMASH call matches the SEED's class
                          (in v2_mismatch: high tracks_seed + low correct_class
                           => class comes from the SEED, not the tag/model)
  leak                <-- MUST be 0 (scored sequence containing seed bytes)
  v2_notag vs v2_tag  <-- tag inert (~equal) => adapter+seed story; collapse => tag load-bearing
  v2_tag_trunc        <-- holds => de-novo class past the boundary; collapses => gene-continuation
  v2_tag_shuf         <-- should COLLAPSE if the real gene content (not composition) is what matters
NOTE
echo "[deconf] ALL DONE $(date)"
