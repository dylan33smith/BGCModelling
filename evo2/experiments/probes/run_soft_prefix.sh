#!/usr/bin/env bash
# PER-CLASS SOFT PREFIXES — the first TRAINING-TIME conditioning handle this project has tried.
#
# WHY THIS, AFTER STEERING FAILED. The 2026-08-10 steering programme established that the class
# coordinate is present in the network but the generator does not CONSUME it: the direction can
# DELETE a class that is there and never INSTALL the target's. That is a property of training,
# not of inference, so no inference-time trick fixes it. Two other facts bracket the fix:
#   * LABEL conditioning is inert (v2_notag == v2_tag) -- a byte string with no pretrained prior.
#   * EXEMPLAR conditioning WORKS (seeded correct_class 0.283 vs a 0.067 floor).
# The model conditions on CONTENT and ignores LABELS. A soft prefix LEARNS a synthetic exemplar
# in embedding space: trained, so the generator can learn to consume it, and unconstrained by the
# byte-level tokenizer that gave the class tag nowhere to live.
#
# Trainable parameters: 16 x 4096 = 65k floats per class -- ~440x smaller than the LoRA.
#
# THE DESIGN IS INTERNALLY CONTROLLED. "Prefix vs no prefix" cannot separate "installs NRPS" from
# "makes output more BGC-like". So all four prefixes are generated under an IDENTICAL taxonomy
# pool with an identical seed -- the taxa are the same sequences in the same order in every arm,
# so the ONLY thing varying is which prefix is loaded, and every comparison is paired per taxon.
# The test is whether P(class X) is higher under prefix_X than under the three prefixes that were
# NOT trained on X. Those three are the control; no shuffled-label arm is needed.
#
# DE NOVO, NOT SEEDED. Seeded generation already works and its class comes from the exemplar, so
# a seeded result could not be attributed to the prefix. Taxonomy-only is the real goal and the
# regime where every previous lever scored zero.
#
# READOUTS, in order of sensitivity: the CONTINUOUS class probe (TPR 0.900, fit TRAIN-ONLY),
# then class markers (TPR 0.717), then antiSMASH (the gold standard, FPR 0.000 but it only
# detects ~1/3 of 3 kb generations). coding_density is the damage guard throughout.
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
TRAIN="${TRAIN:-/data2/ds85/bgcmodel_data/splits_core/train.jsonl}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
TAXSRC="${TAXSRC:-/data2/ds85/bgcmodel_data/splits_core/valtest_eval.jsonl}"
CLASSES="${CLASSES:-NRPS PKS TERPENE RIPP}"
N_PREFIX="${N_PREFIX:-16}"
STEPS="${STEPS:-400}"
MAX_NT="${MAX_NT:-4096}"
LR="${LR:-0.05}"
GEN_N="${GEN_N:-12}"
MAX_NEW="${MAX_NEW:-3000}"
SEED="${SEED:-42}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ASDB="${ASDB:-/data2/ds85/antismash_db}"
ROOT="${ROOT:-/data2/ds85/bgcmodel_runs/soft_prefix}"
mkdir -p "$ROOT"

[ -s "$PFAM" ] || { echo "[sp] ABORT: no Pfam HMM at $PFAM"; exit 1; }
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

# ---------------------------------------------------------------- 1. train one prefix per class
for C in $CLASSES; do
  if [ -s "$ROOT/$C/prefix_best.pt" ]; then echo "[sp] $C prefix exists, skip"; continue; fi
  echo "[sp] $(date) TRAIN prefix for $C"; wait_for_idle
  PY evo2/scripts/train_soft_prefix.py --adapter "$V2" --train "$TRAIN" --val "$VAL" \
     --compound-class "$C" --n-prefix "$N_PREFIX" --max-nt "$MAX_NT" --steps "$STEPS" \
     --lr "$LR" --seed "$SEED" --out-dir "$ROOT/$C" > "$ROOT/train_$C.log" 2>&1
  [ -s "$ROOT/$C/prefix_best.pt" ] && echo "[sp] $C: $(PY -c "
import json;d=json.load(open('$ROOT/$C/summary.json'));print(f\"val {d['baseline_val_loss']:.4f} -> {d['best_val_loss']:.4f} ({d['improvement']:+.4f})\")" 2>/dev/null)" \
                                  || echo "[sp] !! $C TRAIN FAILED (see train_$C.log)"
done

echo
echo "[sp] ===== did the prefixes actually LEARN anything? ====="
PY - "$ROOT" $CLASSES <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(f"{'class':>10} {'val before':>11} {'val after':>10} {'improvement':>12} {'step':>6}")
any_ok = False
for c in sys.argv[2:]:
    f = root / c / "summary.json"
    if not f.exists():
        print(f"{c:>10}   (no summary — training failed)"); continue
    d = json.load(open(f))
    ok = d["improvement"] > 0
    any_ok |= ok
    print(f"{c:>10} {d['baseline_val_loss']:>11.4f} {d['best_val_loss']:>10.4f} "
          f"{d['improvement']:>+12.4f} {d['best_step']:>6}{'' if ok else '   <- NEVER BEAT INIT'}")
if not any_ok:
    print("\nNo prefix beat its initialisation. A generation result from these would be")
    print("uninterpretable — fix training before reading anything downstream.")
PYEOF

# ---------------------------------------------------------- 2. generate: identical taxa per arm
# Same --seed and same --tag-class in EVERY arm => the same taxonomic tags in the same order,
# so arms are paired per taxon and the prefix is the only variable.
for C in $CLASSES; do
  OUT="$ROOT/gen_$C.jsonl"
  [ -s "$OUT" ] && { echo "[sp] gen_$C exists, skip"; continue; }
  echo "[sp] $(date) GEN with prefix=$C"; wait_for_idle
  PY evo2/scripts/generate_soft_prefix.py --adapter "$V2" --prefix-pt "$ROOT/$C/prefix_best.pt" \
     --from-jsonl "$TAXSRC" --tag-class "$TAGLIST" --n "$GEN_N" \
     --max-new-tokens "$MAX_NEW" --seed "$SEED" --out-jsonl "$OUT" > "$ROOT/gen_$C.log" 2>&1
  [ -s "$OUT" ] && echo "[sp] gen_$C: $(wc -l < "$OUT") records" || echo "[sp] !! gen_$C FAILED"
done
# floor: same taxa, same seed, NO prefix
if [ ! -s "$ROOT/gen_none.jsonl" ]; then
  echo "[sp] $(date) GEN no-prefix floor"; wait_for_idle
  PY evo2/scripts/generate_soft_prefix.py --adapter "$V2" --from-jsonl "$TAXSRC" \
     --tag-class "$TAGLIST" --n "$GEN_N" --max-new-tokens "$MAX_NEW" --seed "$SEED" \
     --out-jsonl "$ROOT/gen_none.jsonl" > "$ROOT/gen_none.log" 2>&1
fi

# ------------------------------------------------------------------ 3. score: continuous first
echo "[sp] $(date) continuous class probe (TRAIN-ONLY fit)"
PY evo2/scripts/probe_score_generations.py "$ROOT"/gen_*.jsonl --layer 16 \
   --out-json "$ROOT/probe_score.json" --emit-sidecar "$ROOT/probe_sidecar.json" \
   > "$ROOT/probe_score.log" 2>&1
echo "[sp] $(date) antiSMASH (gold standard) + markers + coherence"
PY evo2/scripts/score_generations_antismash.py "$ROOT"/gen_*.jsonl --expected compound_class \
   --workers 10 --allow-legacy --out-tsv "$ROOT/antismash.tsv" > "$ROOT/antismash.log" 2>&1

echo
echo "==================== SOFT PREFIX — THE CROSS-CLASS MATRIX ===================="
PY - "$ROOT" "$PFAM" $CLASSES <<'PYEOF'
import json, sys, csv, collections, statistics as st
from math import comb
from pathlib import Path
sys.path.insert(0, "/home/ds85/projects/BCGModelling/src")
from bgc_pipeline.evaluation import check_class_markers, check_coding_sanity
root, pfam = Path(sys.argv[1]), Path(sys.argv[2])
CLASSES = sys.argv[3:]
rows = json.load(open(root / "probe_score.json"))
by = collections.defaultdict(list)
for r in rows:
    by[r["arm"]].append(r)

print("\nP(class | prefix) from the continuous probe — ROWS are the prefix loaded, COLUMNS the")
print("class scored. The claim is the DIAGONAL standing above the rest of its own column.\n")
hdr = f"{'prefix':>12} {'n':>3} | " + " ".join(f"{c[:9]:>10}" for c in CLASSES)
print(hdr); print("-" * len(hdr))
mat = {}
for arm in [f"gen_{c}" for c in CLASSES] + ["gen_none"]:
    rs = by.get(arm)
    if not rs:
        continue
    lbl = arm.replace("gen_", "")
    mat[lbl] = {c: st.mean(r["probs"].get(c, 0.0) for r in rs) for c in CLASSES}
    cells = " ".join(f"{mat[lbl][c]:>10.4f}" + ("*" if c == lbl else " ")[0:0] for c in CLASSES)
    print(f"{lbl:>12} {len(rs):>3} | {cells}")

print("\nPAIRED per taxon: for each class X, prefix_X vs each OTHER prefix, same taxon.")
print("The other prefixes ARE the control — every arm is equally 'a trained prefix'.\n")
print(f"{'class X':>10} {'vs':>10} {'pairs':>6} {'mean dP(X)':>12} {'up':>7} {'sign p':>8}")
verdict = {}
for X in CLASSES:
    a = {r["tax_idx"]: r for r in by.get(f"gen_{X}", []) if "tax_idx" in r}
    deltas_all = []
    for Y in CLASSES + ["none"]:
        if Y == X:
            continue
        b = {r["tax_idx"]: r for r in by.get(f"gen_{Y}", []) if "tax_idx" in r}
        d = [a[k]["probs"].get(X, 0.0) - b[k]["probs"].get(X, 0.0) for k in a if k in b]
        if len(d) < 3:
            continue
        up = sum(1 for v in d if v > 0)
        p = min(1.0, 2 * sum(comb(len(d), k) * 0.5 ** len(d)
                             for k in range(max(up, len(d) - up), len(d) + 1)))
        print(f"{X:>10} {Y:>10} {len(d):>6} {st.mean(d):>+12.4f} {up:>3}/{len(d):<3} {p:>8.4f}")
        if Y != "none":
            deltas_all += d
    if deltas_all:
        up = sum(1 for v in deltas_all if v > 0)
        n = len(deltas_all)
        p = min(1.0, 2 * sum(comb(n, k) * 0.5 ** n for k in range(max(up, n - up), n + 1)))
        verdict[X] = (st.mean(deltas_all), up, n, p)
        print(f"{X:>10} {'ALL other':>10} {n:>6} {st.mean(deltas_all):>+12.4f} "
              f"{up:>3}/{n:<3} {p:>8.4f}   <== the test for {X}")

# --- the binary instruments, for the same arms ---
print("\nBinary readouts (lower sensitivity; report alongside, never instead):")
asr = {}
f = root / "antismash.tsv"
if f.exists():
    for r in csv.DictReader(f.open(), delimiter="\t"):
        a = asr.setdefault(r["arm"], [0, 0, 0])
        a[2] += 1
        a[0] += r.get("is_bgc") in ("1", "True", "true")
        a[1] += r.get("correct_class") in ("1", "True", "true")
print(f"{'prefix':>12} {'n':>4} {'coding':>8} {'markers(own)':>13} {'antismash is_bgc':>17} "
      f"{'correct_class':>14}")
for arm in [f"gen_{c}" for c in CLASSES] + ["gen_none"]:
    p = root / f"{arm}.jsonl"
    if not p.exists():
        continue
    recs = [json.loads(l) for l in p.open() if l.strip()]
    cd, mk = [], 0
    for r in recs:
        s = r.get("sequence", "") or ""
        if len(s) < 200:
            continue
        cd.append(check_coding_sanity(s).get("coding_density", float("nan")))
        cls = r.get("soft_prefix_class")
        if cls:
            mk += bool(check_class_markers(s, expected_class=cls,
                                           pfam_hmm_path=pfam).get("markers_present"))
    d, c, n = asr.get(arm, [0, 0, 0])
    print(f"{arm.replace('gen_',''):>12} {len(recs):>4} "
          f"{st.mean(cd) if cd else float('nan'):>8.3f} {mk}/{len(recs):<12} "
          f"{(d/n if n else float('nan')):>17.3f} {(c/n if n else float('nan')):>14.3f}")

print("\nREAD: a prefix that lifts its OWN class above what the other prefixes achieve on that")
print("class is real conditioning. A prefix that lifts every class equally is just making")
print("output more BGC-like — the thing the LoRA already does. coding_density must hold.")
PYEOF
echo "[sp] ALL DONE $(date)"
