#!/usr/bin/env bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────────────────────
# diagnose_conditioning_stochastic.sh — is the COMPOUND_CLASS tag read? (STOCHASTIC)
#
#   evo2/scripts/diagnose_conditioning_stochastic.sh <run-dir-or-ckpt> [out-dir]
#
# Re-do of the Step-0 diagnostic with the PRODUCTION decoding (top_k=4, temp=1.0)
# instead of greedy. (Greedy collapsed to degenerate all-GC output, gc=1.0.)
#
# Design: T fixed taxa × C classes × N samples. Under sampling, two samples of the
# SAME (taxon,class) differ by sampling noise; two samples of DIFFERENT classes
# (same taxon) differ by sampling noise + any class effect. So:
#   WITHIN-class divergence  = noise floor
#   CROSS-class divergence   = noise + class effect
#   cross > within  ⇒ the class tag actually changes the output.
# Plus a class × obligate-domain matrix: is the effect class-APPROPRIATE (does each
# class show ITS obligate domains)?
#
# Verdict:
#   cross ≈ within                          → CONDITIONING DEAD (tag ignored)
#   cross > within, but no class machinery  → OBJECTIVE/DILUTION (weak/partial; not
#                                             building class domains)
#   cross > within, class-appropriate domains → conditioning works
# ─────────────────────────────────────────────────────────────────────────────

CKPT_IN="${1:?usage: diagnose_conditioning_stochastic.sh <ckpt> [out]}"
OUT="${2:-cond_diag_stoch_out}"
ENV_NAME="${ENV_NAME:-bgcmodel}"
export HF_HOME="${HF_HOME:-/data2/ds85/hf_cache}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
CLASSES="${CLASSES:-NRPS PKS PKS_NRPS_HYBRID TERPENE}"
N_TAXA="${N_TAXA:-2}"
N_SAMPLES="${N_SAMPLES:-3}"   # samples per (taxon,class) cell — need >=2 for within-class
MAX_NEW="${MAX_NEW:-16000}"
TOP_K="${TOP_K:-4}"           # PRODUCTION stochastic decoding
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"

if   [[ -d "$CKPT_IN/adapter" ]]; then CKPT="$CKPT_IN"
elif [[ -d "$CKPT_IN/checkpoints/best/adapter" ]]; then CKPT="$CKPT_IN/checkpoints/best"
else echo "Could not find adapter/ under $CKPT_IN" >&2; exit 2; fi

mkdir -p "$OUT"
PY() { micromamba run -n "$ENV_NAME" python "$@"; }
echo "[diag-stoch] ckpt=$CKPT  grid={$CLASSES} x $N_TAXA taxa x $N_SAMPLES samples  top_k=$TOP_K max_new=$MAX_NEW"

# 1. grid: same N_TAXA taxa across every class
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
    if len(taxa) >= n_taxa: break
rows = [{"compound_class": c, "taxonomic_tag": t, "sequence": ""} for c in classes for t in taxa]
open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
print(f"[diag-stoch] grid: {len(classes)} classes x {len(taxa)} taxa = {len(rows)} cells (x{__import__('sys').argv and ''} samples)")
PYEOF

# 2. STOCHASTIC generation (top_k=4); --n samples per cell
PY evo2/scripts/generate_bgc.py --adapter "$CKPT" --from-jsonl "$OUT/grid.jsonl" \
  --per-class 0 --n "$N_SAMPLES" --max-new-tokens "$MAX_NEW" \
  --top-k "$TOP_K" --temperature "$TEMP" --top-p "$TOP_P" --batch-size 1 --seed "$SEED" \
  --out-fasta "$OUT/gen.fasta" --out-jsonl "$OUT/gen.jsonl"

# 3. domain scan (M2)
# POSITIVE CONTROL: real held-out cores truncated to THIS run's own length distribution
# and class mix. Until 2026-08-10 every probe driver passed an empty `_nopos.jsonl`, so
# 0 of 25 reports had a ceiling and every rate was an uncalibrated fraction of an
# unstated maximum. Generated per-eval because the ceiling depends on generation length.
PY scripts/make_positive_control.py --gen "$OUT/gen.jsonl" --out "$OUT/positive_control.jsonl" || true
PY scripts/eval_suite_driver.py --gen "$OUT/gen.jsonl" --positive "$OUT/positive_control.jsonl" \
  --skip-checks antismash protein_homology kmer_novelty --pfam-hmm "$PFAM" --output "$OUT/diag_eval.json"

# 4. parse: within vs cross divergence + class×obligate + verdict
PY - "$OUT/gen.jsonl" "$OUT/diag_eval.json" "$OUT/diag_report.json" <<'PYEOF'
import json, sys, statistics, itertools, math
from collections import defaultdict, Counter
gen_p, ev_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
gens = [json.loads(l) for l in open(gen_p)]
rep = json.loads(open(ev_p).read())
per = rep.get("per_record", {}).get("generated", [])
m2 = {r.get("accession"): (r.get("class_markers") or {}) for r in per}
dom = {r.get("accession"): set(d["pfam_accession"] for d in (r.get("class_markers", {}) or {}).get("domains_found", []) or [])
       for r in per}

# DNA divergence must NOT use long-k-mer Jaccard — it SATURATES (any two distinct
# sequences share ~no 12-mers, so within≈cross≈1.0 regardless of conditioning).
# Use dense 5-mer COMPOSITION cosine + domain-set Jaccard instead.
def kvec(s, k=5):
    s = s.upper()
    c = Counter(s[i:i+k] for i in range(len(s) - k + 1) if set(s[i:i+k]) <= set("ACGT"))
    tot = sum(c.values()) or 1
    return {km: v / tot for km, v in c.items()}
def comp_dist(a, b):                  # 1 - cosine on 5-mer composition
    A, B = a["_v"], b["_v"]
    keys = set(A) | set(B)
    dot = sum(A.get(x, 0) * B.get(x, 0) for x in keys)
    na = math.sqrt(sum(v * v for v in A.values())); nb = math.sqrt(sum(v * v for v in B.values()))
    return 1.0 - (dot / (na * nb)) if na and nb else 1.0
def dom_dist(a, b):                   # 1 - Jaccard on Pfam domain sets
    A, B = dom.get(a["id"], set()), dom.get(b["id"], set())
    u = len(A | B)
    return 1.0 - (len(A & B) / u) if u else 1.0
for g in gens:
    g["_v"] = kvec(g["sequence"])

# group by (taxon, class)
cell = defaultdict(list)             # (tax,cls) -> [gen,...]
by_taxon = defaultdict(lambda: defaultdict(list))
for g in gens:
    cell[(g["taxonomic_tag"], g["compound_class"])].append(g)
    by_taxon[g["taxonomic_tag"]][g["compound_class"]].append(g)

# WITHIN-class pairs (same taxon+class) = noise floor; CROSS-class (same taxon,
# different class) = noise + class effect. Computed for composition AND domains.
within, cross, within_dom, cross_dom = [], [], [], []
for (tax, cls), gs in cell.items():
    for a, b in itertools.combinations(gs, 2):
        within.append(comp_dist(a, b)); within_dom.append(dom_dist(a, b))
for tax, cmap in by_taxon.items():
    for ci, cj in itertools.combinations(sorted(cmap), 2):
        for a in cmap[ci]:
            for b in cmap[cj]:
                cross.append(comp_dist(a, b)); cross_dom.append(dom_dist(a, b))

def m(xs): return round(statistics.mean(xs), 4) if xs else None
within_mean, cross_mean = m(within), m(cross)            # 5-mer composition cos-dist
within_dom_mean, cross_dom_mean = m(within_dom), m(cross_dom)
dom_ratio = round(cross_dom_mean / within_dom_mean, 3) if (within_dom_mean and within_dom_mean > 0) else None

# class × obligate-domain + GC sanity + any-domain
per_class = defaultdict(lambda: {"own_obl_frac": [], "any_domain": [], "gc": [], "n": 0})
for g in gens:
    mm = m2.get(g["id"], {})
    ob = mm.get("obligate_domains") or []
    miss = set(mm.get("missing_obligate") or [])
    own = ((len(ob) - len(miss)) / len(ob)) if ob else None
    dc = mm.get("domain_count", len(mm.get("domains_found", []) or []))
    s = g["sequence"].upper()
    pc = per_class[g["compound_class"]]
    pc["n"] += 1
    if own is not None: pc["own_obl_frac"].append(own)
    pc["any_domain"].append(int(dc > 0))
    pc["gc"].append(round((s.count("G") + s.count("C")) / len(s), 3) if s else 0.0)
obl_by_class = {c: m(v["own_obl_frac"]) for c, v in per_class.items()}
anydom_by_class = {c: m(v["any_domain"]) for c, v in per_class.items()}
gc_by_class = {c: m(v["gc"]) for c, v in per_class.items()}
any_obl = any((f or 0) > 0 for f in obl_by_class.values())

# verdict
ratio = round(cross_mean / within_mean, 3) if (within_mean and within_mean > 0) else None
comp_effect = (ratio is not None and ratio > 1.03) or (dom_ratio is not None and dom_ratio > 1.05)
if within_mean is None or within_mean == 0:
    verdict = (f"INCONCLUSIVE — within-class composition ~0 (samples not diverse). "
               f"within={within_mean} cross={cross_mean}")
elif not comp_effect:
    verdict = (f"CONDITIONING LIKELY DEAD — class tag has ~no effect beyond noise "
               f"(composition cross/within {ratio}, domain {dom_ratio}). → per-class adapters / "
               f"stronger conditioning.")
elif not any_obl:
    verdict = (f"WEAK/PARTIAL conditioning + OBJECTIVE/DILUTION — the tag shifts output "
               f"(composition ratio {ratio}, domain ratio {dom_ratio}) but NO class shows its obligate "
               f"domains. → core-region trimming + objective; conditioning may need strengthening.")
else:
    verdict = (f"CONDITIONING WORKS — class-differentiated (composition ratio {ratio}) AND "
               f"class-appropriate obligate domains present: {obl_by_class}.")

report = {"composition_5mer_within": within_mean, "composition_5mer_cross": cross_mean,
          "composition_cross_over_within": ratio,
          "domain_set_within": within_dom_mean, "domain_set_cross": cross_dom_mean,
          "domain_cross_over_within": dom_ratio,
          "n_within_pairs": len(within), "n_cross_pairs": len(cross),
          "own_obligate_fraction_by_class": obl_by_class, "any_domain_rate_by_class": anydom_by_class,
          "gc_by_class": gc_by_class, "decoding": {"top_k": __import__("os").environ.get("TOP_K", "4")},
          "verdict": verdict}
open(out_p, "w").write(json.dumps(report, indent=2) + "\n")
print("\n[diag-stoch] ===== RESULT =====")
print(f"[diag-stoch] composition 5-mer  within {within_mean} | cross {cross_mean} | ratio {ratio}")
print(f"[diag-stoch] domain-set         within {within_dom_mean} | cross {cross_dom_mean} | ratio {dom_ratio}")
print(f"[diag-stoch] own-obligate-fraction by class: {obl_by_class}")
print(f"[diag-stoch] any-domain-rate by class: {anydom_by_class}")
print(f"[diag-stoch] GC by class (sanity, want ~0.4-0.7 NOT 1.0): {gc_by_class}")
print(f"[diag-stoch] VERDICT: {verdict}")
PYEOF
echo "[diag-stoch] report → $OUT/diag_report.json"
