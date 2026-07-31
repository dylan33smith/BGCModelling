#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline for a trained checkpoint. Run when training finishes.
#
#   evo2/scripts/run_eval.sh <run-dir-or-checkpoint-dir> [out-dir] [per-class]
#
# Sequences (each step writes into OUT):
#   1. generate_bgc.py        — generate conditioned BGCs from held-out val prompts
#   2. memorization_check.py  — novelty vs full corpus (+ positive-control calibration)
#   3. eval_conditioning_adherence.py — likelihood adherence (+ base baseline)
#   4. conditioning_experiment.py     — causal class control + E. coli taxon control
#   5. eval_suite_driver.py   — acceptance suite (named checks → questions), gen vs positive
#
# GPU steps: 1, 3, 4, and (optionally) the foldability check in 5. They need the GPU
# free — run AFTER training has finished/stopped. Checks whose external tool is not
# installed (antiSMASH/ESMFold/MMseqs2/Pfam) self-skip. See evaluation.py for the
# CHECKS (coding_sanity, antismash, class_markers, kmer_novelty, protein_homology,
# module_architecture, taxon_faithfulness) → QUESTIONS (is_bgc, correct_class, novel, …).
# ─────────────────────────────────────────────────────────────────────────────

CKPT_IN="${1:?usage: run_eval.sh <run-dir-or-checkpoint-dir> [out-dir] [per-class]}"
OUT="${2:-eval_out_$(basename "$CKPT_IN")}"
PER_CLASS="${3:-3}"

ENV_NAME="${ENV_NAME:-bgcmodel}"
export HF_HOME="${HF_HOME:-/data2/ds85/hf_cache}"
# splits_curated is SUPERSEDED (CLAUDE.md: "do not use"). It overlaps the active
# splits_core training set, so every headline was computed over prompts that are
# not fully held out.
VAL="${VAL:-/data2/ds85/bgcmodel_data/splits_core/val.jsonl}"
# The novelty GATE must compare against the corpus the model ACTUALLY trained on.
# splits_combined_grouped is superseded; using it let sequences memorized from
# splits_core pass as novel.
REF="${REF:-/data2/ds85/bgcmodel_data/splits_core/train.jsonl}"
POS="${POS:-eval/positive_control_mibig.jsonl}"
MAX_NEW="${MAX_NEW:-16384}"

# Optional reference DBs for the suite. The PFAM/antiSMASH defaults point at the
# /data2 downloads; if a path doesn't exist yet, the driver self-skips that check,
# so these are safe to pass unconditionally. protein_homology (mmseqs2) stays off
# unless you point MMSEQS2_DB at a built DB. (MIBIG_GBK is reserved; no check uses it.)
PFAM_HMM="${PFAM_HMM:-/data2/ds85/pfam/Pfam-A.hmm}"
ANTISMASH_DB="${ANTISMASH_DB:-/data2/ds85/antismash_db}"
MMSEQS2_DB="${MMSEQS2_DB:-}"
MIBIG_GBK="${MIBIG_GBK:-}"
SUITE_DB_ARGS=()
[[ -n "$PFAM_HMM" ]]     && SUITE_DB_ARGS+=(--pfam-hmm "$PFAM_HMM")
[[ -n "$ANTISMASH_DB" ]] && SUITE_DB_ARGS+=(--antismash-db "$ANTISMASH_DB")
[[ -n "$MMSEQS2_DB" ]]   && SUITE_DB_ARGS+=(--mmseqs2-db "$MMSEQS2_DB")
[[ -n "$MIBIG_GBK" ]]    && SUITE_DB_ARGS+=(--mibig-gbk "$MIBIG_GBK")

# Resolve checkpoint: accept a checkpoint dir (has adapter/) or a run dir (use best/).
if [[ -d "$CKPT_IN/adapter" ]]; then
  CKPT="$CKPT_IN"
elif [[ -d "$CKPT_IN/checkpoints/best/adapter" ]]; then
  CKPT="$CKPT_IN/checkpoints/best"
else
  echo "Could not find an adapter/ under $CKPT_IN (checkpoint dir or run dir with checkpoints/best/)." >&2
  exit 2
fi

mkdir -p "$OUT"
run() { echo "[$(date '+%F %T')] >> $*" | tee -a "$OUT/run_eval.log"; "$@"; }
PY() { micromamba run -n "$ENV_NAME" python "$@"; }

echo "Checkpoint: $CKPT" | tee "$OUT/run_eval.log"
echo "Output:     $OUT"  | tee -a "$OUT/run_eval.log"

# 1. Generate from held-out conditioning prompts.
run PY evo2/scripts/generate_bgc.py --adapter "$CKPT" --from-jsonl "$VAL" \
  --per-class "$PER_CLASS" --max-new-tokens "$MAX_NEW" \
  --out-fasta "$OUT/generated.fasta" --out-jsonl "$OUT/generated.jsonl"

# 2. Novelty / anti-memorization vs the full corpus, with positive-control calibration.
run PY scripts/memorization_check.py --query "$OUT/generated.jsonl" --ref "$REF" \
  --positive-control "$POS" --output "$OUT/memorization.jsonl"

# 3. Conditioning-adherence (likelihood classifier) + base-model baseline.
run PY evo2/scripts/eval_conditioning_adherence.py --adapter "$CKPT" --val "$VAL" \
  --compare-base --output "$OUT/adherence.json"

# 4. Controlled conditioning experiment: causal class control + E. coli taxon control.
run PY evo2/scripts/conditioning_experiment.py --adapter "$CKPT" --experiment both \
  --output "$OUT/conditioning.json"

# 5. 8-metric suite + novelty gate, generated vs positive control.
run PY scripts/eval_suite_driver.py --gen "$OUT/generated.jsonl" --positive "$POS" \
  --novelty "$OUT/memorization.jsonl" --output "$OUT/eval_suite.json" \
  ${SUITE_DB_ARGS[@]+"${SUITE_DB_ARGS[@]}"}

echo "[$(date '+%F %T')] EVAL COMPLETE -> $OUT" | tee -a "$OUT/run_eval.log"
echo "  generated.jsonl, memorization.jsonl, adherence.json, conditioning.json, eval_suite.json"
