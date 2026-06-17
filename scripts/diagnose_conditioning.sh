#!/usr/bin/env bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────────────────────
# diagnose_conditioning.sh — is the COMPOUND_CLASS tag actually being read?
#
#   scripts/diagnose_conditioning.sh <run-dir-or-checkpoint-dir> [out-dir]
#
# DECISIVE Step-0 diagnostic (see docs/archive/REDESIGN_PLAN.md). Holds taxonomy FIXED and
# varies only the class tag, with GREEDY decoding (top_k=1) so the generation is
# DETERMINISTIC — therefore any difference between two class-conditioned outputs
# from the same taxon is attributable SOLELY to the class tag.
#
# Reads three signals:
#   (1) cross-class divergence  — under greedy, if the class tag is ignored, the
#       outputs for different classes (same taxon) are byte-IDENTICAL. High
#       cross-class similarity ⇒ conditioning not read.
#   (2) class × obligate-domain — does each class's generation contain ITS own
#       obligate biosynthetic domains (NRPS C-A-T, PKS KS-AT-ACP, ...)?
#   (3) GC-by-taxon (bonus)     — sanity that taxonomy conditioning does anything.
#
# Interpretation:
#   - outputs ~identical across classes        → CONDITIONING DEAD  (axis 1)
#   - outputs differ but no obligate anywhere  → OBJECTIVE/DILUTION (axis 2)
#   - outputs differ WITH class-appropriate obligate signal → conditioning works
# ─────────────────────────────────────────────────────────────────────────────

CKPT_IN="${1:?usage: diagnose_conditioning.sh <run-dir-or-checkpoint-dir> [out-dir]}"
OUT="${2:-cond_diag_out}"
ENV_NAME="${ENV_NAME:-bgcmodel}"
export HF_HOME="${HF_HOME:-/data2/ds85/hf_cache}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
CLASSES="${CLASSES:-NRPS PKS PKS_NRPS_HYBRID TERPENE}"
N_TAXA="${N_TAXA:-3}"
MAX_NEW="${MAX_NEW:-16000}"   # moderate: divergence shows early, obligate needs length
SEED="${SEED:-42}"

if   [[ -d "$CKPT_IN/adapter" ]]; then CKPT="$CKPT_IN"
elif [[ -d "$CKPT_IN/checkpoints/best/adapter" ]]; then CKPT="$CKPT_IN/checkpoints/best"
else echo "Could not find adapter/ under $CKPT_IN" >&2; exit 2; fi

mkdir -p "$OUT"
: > "$OUT/_nopos.jsonl"   # empty positive set (driver tolerates it)
PY() { micromamba run -n "$ENV_NAME" python "$@"; }
echo "[diag] checkpoint: $CKPT"
echo "[diag] grid: {$CLASSES} x $N_TAXA taxa @ greedy max_new=$MAX_NEW"

# 1. build the class × taxon grid (same taxa across every class)
PY - "$VAL" "$OUT/grid.jsonl" "$CLASSES" "$N_TAXA" <<'PYEOF'
import json, sys, random
val, out, classes, n_taxa = sys.argv[1], sys.argv[2], sys.argv[3].split(), int(sys.argv[4])
recs = [json.loads(l) for l in open(val)]
rng = random.Random(0); rng.shuffle(recs)
taxa, seen = [], set()
for r in recs:
    t = r.get("taxonomic_tag", "")
    if t and t not in seen:
        seen.add(t); taxa.append(t)
    if len(taxa) >= n_taxa:
        break
rows = [{"compound_class": c, "taxonomic_tag": t, "sequence": ""}
        for c in classes for t in taxa]
open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
print(f"[diag] grid cells: {len(classes)} classes x {len(taxa)} taxa = {len(rows)}")
PYEOF

# 2. GREEDY generation (top_k=1 ⇒ deterministic; class is the only varying input)
PY scripts/generate_bgc.py --adapter "$CKPT" --from-jsonl "$OUT/grid.jsonl" \
  --per-class 0 --n 1 --max-new-tokens "$MAX_NEW" \
  --top-k 1 --temperature 1.0 --top-p 1.0 --batch-size 1 --seed "$SEED" \
  --out-fasta "$OUT/gen.fasta" --out-jsonl "$OUT/gen.jsonl"

# 3. domain scan only (M2)
PY scripts/eval_suite_driver.py --gen "$OUT/gen.jsonl" --positive "$OUT/_nopos.jsonl" \
  --skip-checks antismash protein_homology kmer_novelty --pfam-hmm "$PFAM" --output "$OUT/diag_eval.json"

# 4. parse: divergence + class×obligate matrix + verdict (join gen.id == per_record.accession)
PY - "$OUT/gen.jsonl" "$OUT/diag_eval.json" "$OUT/diag_report.json" <<'PYEOF'
import json, sys, statistics
from collections import defaultdict
gen_p, ev_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
gens = [json.loads(l) for l in open(gen_p)]
rep = json.loads(open(ev_p).read())
per = rep.get("per_record", {}).get("generated", [])
m2 = {r.get("accession"): (r.get("class_markers") or {}) for r in per}  # accession == gen id

def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

# (1) cross-class divergence per taxon (greedy ⇒ identical means class ignored)
by_taxon = defaultdict(dict)
for g in gens:
    by_taxon[g["taxonomic_tag"]][g["compound_class"]] = g
div, all_fracs, ident, tot = [], [], 0, 0
for tax, cm in by_taxon.items():
    cs = sorted(cm); pairs = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = cm[cs[i]]["sequence"], cm[cs[j]]["sequence"]
            frac = lcp(a, b) / max(1, min(len(a), len(b)))
            same = a == b
            pairs.append({"a": cs[i], "b": cs[j], "lcp_frac": round(frac, 3), "identical": same})
            all_fracs.append(frac); tot += 1; ident += int(same)
    div.append({"taxon": tax[:48], "pairs": pairs})

# (2) class × obligate-domain
class_obl, gc_by_class = defaultdict(list), defaultdict(list)
for g in gens:
    mm = m2.get(g["id"], {})
    ob = mm.get("obligate_domains") or []
    miss = set(mm.get("missing_obligate") or [])
    present = [d for d in ob if d not in miss]
    frac = (len(present) / len(ob)) if ob else None
    class_obl[g["compound_class"]].append({
        "taxon": g["taxonomic_tag"][:30], "len": len(g["sequence"]),
        "obligate_present": present, "obligate_missing": sorted(miss),
        "obligate_frac": (round(frac, 3) if frac is not None else None),
        "total_domains_found": mm.get("domain_count", len(mm.get("domains_found", []) or [])),
    })
    s = g["sequence"].upper()
    gc_by_class[g["compound_class"]].append(round((s.count("G") + s.count("C")) / len(s), 3) if s else 0.0)

def meanopt(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None

mean_lcp = meanopt(all_fracs)
obl_by_class = {c: meanopt([x["obligate_frac"] for x in v]) for c, v in class_obl.items()}
any_obl = any((f or 0) > 0 for f in obl_by_class.values())

if mean_lcp is not None and mean_lcp > 0.9:
    verdict = (f"CONDITIONING LIKELY NOT READ — cross-class outputs nearly identical "
               f"(mean LCP {mean_lcp}, {ident}/{tot} byte-identical pairs). The class tag "
               f"barely changes the output. → favor per-class adapters / stronger conditioning.")
elif not any_obl:
    verdict = (f"OBJECTIVE/DILUTION — outputs DO differ across classes (mean LCP {mean_lcp}) "
               f"but NO class shows its obligate domains. Conditioning has an effect; the model "
               f"just isn't building class machinery. → core-region trimming + objective.")
else:
    verdict = (f"CONDITIONING WORKS (partially) — class-differentiated output (mean LCP {mean_lcp}) "
               f"WITH some class-appropriate obligate domains {obl_by_class}.")

report = {"mean_cross_class_lcp_frac": mean_lcp, "identical_pairs": ident, "total_pairs": tot,
          "obligate_fraction_by_conditioned_class": obl_by_class,
          "gc_by_class": dict(gc_by_class), "divergence": div,
          "class_obligate_detail": {c: v for c, v in class_obl.items()}, "verdict": verdict}
open(out_p, "w").write(json.dumps(report, indent=2) + "\n")
print("\n[diag] ===== RESULT =====")
print("[diag] mean cross-class LCP frac:", mean_lcp, f"({ident}/{tot} identical pairs)")
print("[diag] obligate_fraction by conditioned class:", obl_by_class)
print("[diag] GC by class:", dict(gc_by_class))
print("[diag] VERDICT:", verdict)
PYEOF
echo "[diag] report → $OUT/diag_report.json"
