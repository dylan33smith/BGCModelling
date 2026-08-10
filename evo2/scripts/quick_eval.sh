#!/usr/bin/env bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────────────────────
# quick_eval.sh — fast checkpoint score for tracking FUNCTIONAL progress.
#
#   evo2/scripts/quick_eval.sh <run-dir-or-checkpoint-dir> [out-dir]
#
# Generates a small panel (one per module-bearing class, fixed seed) and scores the
# CHEAP checks: coding_sanity, antismash (is_bgc/correct_class — ~3 s/core), class_markers
# (Pfam proxy), module_architecture, taxon_faithfulness. Skips protein_homology (needs a
# DB) and kmer_novelty (needs the corpus scan); protein_foldability is opt-in. Appends a
# row to eval_track.jsonl.
#
# LENGTH MATTERS. We generate at the full training window (MAX_NEW=32768) on
# purpose: a complete obligate module is large (NRPS C-A-T ~3.5 kb, PKS KS-AT-ACP
# ~2.5 kb), and the model generates leading regulatory/intergenic sequence first.
# A short cap (e.g. 6 kb) truncates before the module completes and yields SILENT
# FAILURES (M2 fails for a length reason, not a capability reason). 32k is the
# in-distribution generation length and contains >=1 complete module.
#
# SIGNALS:
#   - is_bgc / correct_class : antiSMASH-gated QUESTION pass-rates (the headline)
#   - obligate_fraction : mean fraction of the class's markers present (graded EARLY signal)
#   - class_markers     : class_markers PASS rate (>=1 marker = right-class machinery, proxy)
#   - any_domain_rate   : fraction of generations with >=1 Pfam domain at all
#   - coding_density / module_count / in_order_fraction : coding_sanity / module quality
#
# NOTE: generation needs the GPU — run when the GPU is free (or pause training). It
# will NOT share the GPU with an active training run on one device. Panel restricted
# to module-bearing classes so every generation gets a real verdict (not no_verdict).
# ─────────────────────────────────────────────────────────────────────────────

CKPT_IN="${1:?usage: quick_eval.sh <run-dir-or-checkpoint-dir> [out-dir]}"
OUT="${2:-quick_eval_out}"
ENV_NAME="${ENV_NAME:-bgcmodel}"
export HF_HOME="${HF_HOME:-/data2/ds85/hf_cache}"
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
PFAM="${PFAM:-/data2/ds85/pfam/Pfam-A.hmm}"
ASDB="${ASDB:-/data2/ds85/antismash_db}"   # antiSMASH DBs -> real is_bgc/correct_class (~3 s/core)
PER_CLASS="${PER_CLASS:-1}"
MAX_NEW="${MAX_NEW:-32768}"   # full training window — short caps cause silent M2 failures (see header)
SEED="${SEED:-42}"
# Generation batch size. 1 = sequential (one prompt at a time; always correct).
# >1 / 0 = batched (left-padded) generation — FASTER but only used once the on-GPU
# equivalence gate (evo2/scripts/validate_batched_generation.py) has confirmed it
# matches sequential on this model. The orchestrator writes the validated choice
# to DECISION_FILE; absent that file (or an explicit env), we default to 1 (safe).
DECISION_FILE="${DECISION_FILE:-/data2/ds85/bgcmodel_runs/.batch_decision}"
if [[ -z "${GEN_BATCH_SIZE:-}" ]]; then
  if [[ -f "$DECISION_FILE" ]]; then
    GEN_BATCH_SIZE="$(tr -dc '0-9-' < "$DECISION_FILE")"
  fi
  GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
fi
echo "[quick_eval] generation batch size: $GEN_BATCH_SIZE (1=sequential)"
# Module-bearing classes whose obligate set needs real length (the length-sensitive
# signal). Add SACCHARIDE/SIDEROPHORE/TERPENE via CLASSES env for broader (single-
# domain) coverage. Fewer classes = faster (generation at 32k is the bottleneck).
CLASSES="${CLASSES:-NRPS PKS PKS_NRPS_HYBRID}"

# Resolve checkpoint: accept a checkpoint dir (has adapter/) or a run dir (best/).
if [[ -d "$CKPT_IN/adapter" ]]; then CKPT="$CKPT_IN"
elif [[ -d "$CKPT_IN/checkpoints/best/adapter" ]]; then CKPT="$CKPT_IN/checkpoints/best"
else echo "Could not find adapter/ under $CKPT_IN (checkpoint dir or run dir)." >&2; exit 2; fi

mkdir -p "$OUT"
PY() { micromamba run -n "$ENV_NAME" python "$@"; }
echo "[quick_eval] checkpoint: $CKPT"

# 1. small prompt pool filtered to the target classes
PY - "$VAL" "$OUT/prompts.jsonl" "$CLASSES" <<'PYEOF'
import json, sys
val, out, classes = sys.argv[1], sys.argv[2], set(sys.argv[3].split())
recs = [json.loads(l) for l in open(val)]
keep = [r for r in recs if r.get("compound_class") in classes]
open(out, "w").write("".join(json.dumps(r) + "\n" for r in keep))
print(f"[quick_eval] prompt pool: {len(keep)} records across {len(classes)} classes")
PYEOF

# Decoding controls (default to the historical quick_eval settings so behavior is
# unchanged unless overridden). TOP_P defaults to 1.0 (disabled).
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-4}"
TOP_P="${TOP_P:-1.0}"
# Chained-generation controls (default 1 window = unchanged behavior). MAX_WINDOWS>1
# lets long megasynthase cores complete across windows (mirrors training chunking).
MAX_WINDOWS="${MAX_WINDOWS:-1}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-2048}"
echo "[quick_eval] decoding: temperature=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P max_windows=$MAX_WINDOWS chunk_overlap=$CHUNK_OVERLAP"

# 2. generate the tiny panel (GPU)
PY evo2/scripts/generate_bgc.py --adapter "$CKPT" --from-jsonl "$OUT/prompts.jsonl" \
  --per-class "$PER_CLASS" --max-new-tokens "$MAX_NEW" --seed "$SEED" \
  --batch-size "$GEN_BATCH_SIZE" \
  --temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" \
  --max-windows "$MAX_WINDOWS" --chunk-overlap "$CHUNK_OVERLAP" \
  --out-fasta "$OUT/gen.fasta" --out-jsonl "$OUT/gen.jsonl"

# 2b. POSITIVE CONTROL — real held-out cores, truncated to the SAME length distribution and drawn
# to the SAME class mix as the generations. Until 2026-08-10 every driver passed
# `--positive "$OUT/_nopos.jsonl"`, a path chosen not to exist: 0 of 25 probe reports had a
# control, so every correct_class number was an uncalibrated fraction of an unstated maximum.
# It matters a lot. antiSMASH scores real curated BGCs at only 0.55 correct_class, and real
# splits_core cores truncated to 2048 nt score is_bgc 0.680 / correct_class 0.640 — so a
# generation at 0.10 is 16% of achievable, not 10% of perfect. It also separates "the model got
# worse" from "the instrument changed", which this project has twice confused.
POSCTRL="${POSCTRL:-$OUT/positive_control.jsonl}"
PY scripts/make_positive_control.py --gen "$OUT/gen.jsonl" --out "$POSCTRL" \
   --cores "${POSCTRL_CORES:-/data2/ds85/bgcmodel_data/splits_core/test.jsonl}" \
   || echo "[quick_eval] !! positive control generation failed; rates will be uncalibrated"

# 3. cheap checks (coding_sanity, antismash, class_markers, module_architecture,
#    taxon_faithfulness); skip the DB-bound/slow ones (protein_homology, kmer_novelty).
PY scripts/eval_suite_driver.py --gen "$OUT/gen.jsonl" --positive "$POSCTRL" \
  --skip-checks protein_homology kmer_novelty \
  --pfam-hmm "$PFAM" --antismash-db "$ASDB" --output "$OUT/quick_eval.json"

# 4. parse question pass-rates + GRADED check signals, append a tracking row
PY - "$CKPT" "$OUT/quick_eval.json" "$OUT/eval_track.jsonl" <<'PYEOF'
import json, re, sys
from datetime import datetime, timezone
ckpt, rep_p, track = sys.argv[1], sys.argv[2], sys.argv[3]
rep = json.loads(open(rep_p).read())
g = rep["generated"]
recs = rep.get("per_record", {}).get("generated", [])
def qr(q):   # question pass-rate
    return g.get("per_question", {}).get(q, {}).get("pass_rate")
def cr(c):   # check pass-rate
    return g.get("per_check", {}).get(c, {}).get("pass_rate")
# graded signals from per-record class_markers + coding_sanity + module_architecture
fracs, any_dom, n_cmk = [], 0, 0
coding, mods, ordmods, inorder = [], [], [], []
for r in recs:
    cmk = r.get("class_markers", {}) or {}
    if not cmk.get("skipped"):
        n_cmk += 1
        dc = cmk.get("domain_count", len(cmk.get("domains_found", []) or []))
        any_dom += int(dc > 0)
        ob = cmk.get("obligate_domains") or []     # the class's markers
        if ob:                                     # fraction of markers present (graded)
            miss = cmk.get("missing_obligate") or []
            fracs.append((len(ob) - len(miss)) / len(ob))
    cs = r.get("coding_sanity", {}) or {}
    if "coding_density" in cs:
        coding.append(cs["coding_density"])
    ma = r.get("module_architecture", {}) or {}
    if ma.get("applicable"):                       # module classes only
        mods.append(ma.get("module_count", 0))
        ordmods.append(ma.get("ordered_module_count", 0))
        if ma.get("in_order_fraction") is not None:
            inorder.append(ma["in_order_fraction"])
rnd = lambda v: round(v, 3) if v is not None else None
avg = lambda xs: rnd(sum(xs) / len(xs)) if xs else None
m = re.search(r"step_(\d+)", ckpt)
rec = {
    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "checkpoint": ckpt, "step": int(m.group(1)) if m else None, "n": g["n"],
    "is_bgc": qr("is_bgc"),                                                # GATE (antiSMASH)
    "correct_class": qr("correct_class"),                                 # GATE (antiSMASH)
    "obligate_fraction": rnd(sum(fracs) / len(fracs)) if fracs else None,  # graded EARLY signal
    "class_markers": cr("class_markers"),                                 # marker PASS rate (proxy)
    # denominator = records where class_markers actually RAN. Dividing by every record
        # reported any_domain_rate 0.0 ("not one generation has a single Pfam domain")
        # when the truth was "Pfam was not configured".
        "any_domain_rate": rnd(any_dom / n_cmk) if n_cmk else None,
        "any_domain_evaluated": n_cmk,
    "coding_density": avg(coding),                                         # coding_sanity (quality)
    "module_count": avg(mods),                                            # modules' worth
    "ordered_modules": avg(ordmods),                                      # in-order modules
    "in_order_fraction": avg(inorder),                                    # ordering quality
    "conditioning_faithful": qr("conditioning_faithful"),                # taxon faithfulness
}
open(track, "a").write(json.dumps(rec) + "\n")
print("[quick_eval] " + json.dumps(rec))
PYEOF
echo "[quick_eval] appended to $OUT/eval_track.jsonl"
