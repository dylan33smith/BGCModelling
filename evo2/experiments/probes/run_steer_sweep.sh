#!/usr/bin/env bash
# ACTIVATION STEERING SWEEP — can injecting the class direction replace the (dead) class tag?
#
# Grid: layers {12,16,20} x alpha {1,2,4}, plus ONE alpha=0 control (layer-independent).
# Layers span the decodability PLATEAU (L10-L23 all tied ~0.89-0.91) rather than clustering
# at the noise-driven peak — the real question is where an injected edit still has enough
# downstream computation to take effect, which only the causal test answers.
#
# CLASSES: deliberately balanced across cluster size/complexity, not just megasynthases —
#   large : NRPS, PKS, PKS_NRPS_HYBRID
#   small : TERPENE, ECTOINE, RIPP      (major MiBIG classes; short, few-gene clusters)
# The summary reports LARGE vs SMALL separately: a lever that only moves megasynthases
# (or only the easy classes) is a different finding than one that moves both.
#
# PROMPT = taxonomy tag ONLY. The de-confound proved the |COMPOUND_CLASS| tag is inert, so
# this tests whether steering can REPLACE label conditioning outright. alpha=0 is the floor.
#
# READ: success = correct_class rises WHILE coding_density holds. If coherence collapses as
# alpha grows (CFG's failure mode), we are pushing out of distribution, not conditioning.
set -uo pipefail
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
# Keep scratch writes off the shared /home (it has hit 100%, which breaks the micromamba
# process lock, git, __pycache__ and antiSMASH temp files mid-run). See run_steer_titration.sh.
export TMPDIR="${TMPDIR:-/data2/ds85/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data2/ds85/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/data2/ds85/cache/mpl}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
ENV=bgcmodel
PY(){ micromamba run -n "$ENV" python "$@"; }

V2="${V2:-/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
SWEEP="${SWEEP:-/data2/ds85/bgcmodel_runs/class_probe_sweep}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ASDB="${ASDB:-/data2/ds85/antismash_db}"
CLASSES="${CLASSES:-NRPS PKS PKS_NRPS_HYBRID TERPENE ECTOINE RIPP}"
LARGE="${LARGE:-NRPS PKS PKS_NRPS_HYBRID}"
PER_CLASS="${PER_CLASS:-5}"
MAX_NEW="${MAX_NEW:-6144}"
LAYERS="${LAYERS:-16 12 20}"          # L16 first so the alpha curve lands early
ALPHAS="${ALPHAS:-1 2 4}"
SEED="${SEED:-42}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/steer_sweep}"
mkdir -p "$ROOT"; : > "$ROOT/_nopos.jsonl"

# EXCLUSIVE=0 shares the GPU (memory check only) — safe here since these metrics are not
# timing-sensitive. Default 1 keeps the project's exclusive-device convention.
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

gen_cell(){   # name layer alpha
  local name="$1" layer="$2" alpha="$3"
  [ -s "$ROOT/$name.jsonl" ] && { echo "[steer] $name exists, skip"; return; }
  echo "[steer] $(date) GEN $name (layer=$layer alpha=$alpha)"; wait_for_idle
  PY evo2/scripts/steer_generate.py --adapter "$V2" --from-jsonl "$VAL" \
     --dirs-npz "$SWEEP/acts_v2.dirs.npz" --acts-npz "$SWEEP/acts_v2.npz" \
     --classes $CLASSES --layer "$layer" --alpha "$alpha" \
     --per-class "$PER_CLASS" --max-new-tokens "$MAX_NEW" \
     --top-k 4 --top-p 1.0 --temperature 1.0 --seed "$SEED" \
     --out-jsonl "$ROOT/$name.jsonl" > "$ROOT/gen_$name.log" 2>&1
  [ -s "$ROOT/$name.jsonl" ] || echo "[steer] !! $name GEN FAILED (see gen_$name.log)"
}

# alpha=0 control is layer-independent -> generate once
gen_cell "a0_control" 16 0
for L in $LAYERS; do for A in $ALPHAS; do gen_cell "L${L}_a${A}" "$L" "$A"; done; done

SUM="$ROOT/steer_summary.tsv"
printf "cell\tlayer\talpha\tn\tis_bgc\tcorrect_class\tcc_large\tcc_small\tmarkers\tcoding_density\n" > "$SUM"
for f in "$ROOT"/a0_control.jsonl $(for L in $LAYERS; do for A in $ALPHAS; do echo "$ROOT/L${L}_a${A}.jsonl"; done; done); do
  [ -s "$f" ] || continue
  name=$(basename "$f" .jsonl)
  echo "[steer] $(date) EVAL $name"
  PY scripts/eval_suite_driver.py --gen "$f" --positive "$ROOT/_nopos.jsonl" \
     --skip-checks protein_homology kmer_novelty \
     --pfam-hmm "$PFAM" --antismash-db "$ASDB" --output "$ROOT/report_$name.json" \
     > "$ROOT/eval_$name.log" 2>&1
  PY - "$name" "$ROOT/report_$name.json" "$f" "$SUM" "$ROOT/per_class_$name.tsv" "$LARGE" <<'PYEOF'
import json,sys,collections
name,rep_p,gen_p,sum_p,pc_p,large=sys.argv[1:7]
large=set(large.split())
recs=json.load(open(rep_p))["per_record"]["generated"]
gen=[json.loads(l) for l in open(gen_p)]
layer=gen[0].get("steer_layer") if gen else None; alpha=gen[0].get("steer_alpha") if gen else None
n=len(recs); ib=cc=mk=0; cod=[]; by=collections.defaultdict(lambda:[0,0,0])
gl=[0,0]; gs=[0,0]      # [correct, n] for large / small
for r in recs:
    a=r.get("antismash",{}) or {}; cm=r.get("class_markers",{}) or {}; cs=r.get("coding_sanity",{}) or {}
    exp=r.get("expected_class"); det=bool(a.get("detected")); ok=det and bool(a.get("class_match"))
    ib+=det; cc+=ok
    if cm.get("domain_count",0)>0 and not cm.get("skipped"): mk+=1
    if "coding_density" in cs: cod.append(cs["coding_density"])
    by[exp][2]+=1; by[exp][1]+=det; by[exp][0]+=ok
    tgt = gl if exp in large else gs
    tgt[0]+=ok; tgt[1]+=1
f=lambda x,d=n: round(x/d,3) if d else None
open(sum_p,"a").write("\t".join(str(x) for x in
  [name,layer,alpha,n,f(ib),f(cc),f(gl[0],gl[1]),f(gs[0],gs[1]),f(mk),
   round(sum(cod)/len(cod),3) if cod else None])+"\n")
with open(pc_p,"w") as fh:
    fh.write("class\tcorrect\tis_bgc\tn\n")
    for c in sorted(by): a_,b_,c_=by[c]; fh.write(f"{c}\t{a_}\t{b_}\t{c_}\n")
print(f"[steer] {name}: correct={f(cc)} (large {f(gl[0],gl[1])} / small {f(gs[0],gs[1])}) coding={round(sum(cod)/len(cod),3) if cod else None}")
PYEOF
done

echo "==================== STEERING SWEEP ===================="
column -t "$SUM" 2>/dev/null || cat "$SUM"
cat <<'NOTE'

READ:
  cc_large / cc_small  <-- does steering move BOTH megasynthase and simple classes,
                           or only one? (a large-only or small-only lever is a
                           different, weaker claim than a general one)
  coding_density       <-- must HOLD as alpha rises. Collapsing coherence = out-of-
                           distribution push (CFG's failure mode), not conditioning.
  a0_control           <-- the floor: v2, taxonomy-only prompt, no steering.
NOTE
echo "[steer] ALL DONE $(date)"
