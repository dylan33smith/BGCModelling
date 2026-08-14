#!/usr/bin/env bash
# PHASE 3 — train the RIPP-only adapter on the 1B. Run from the repo root.
#
# WHAT MAKES THIS DIFFERENT FROM EVERY EARLIER RUN, and why each choice is what it is.
#
# 1. ONE CLASS, SO NO CLASS LABEL IS READ. A per-class adapter means the model never has to decode
#    a class tag — and every Phase-1 closure (inert prefix, CFG, steering, soft prefixes, activation
#    transplant) was about label-reading. Those negatives stop applying because the question stops
#    being asked. This is a cleaner design, not a workaround.
#
# 2. WHOLE-RECORD, NOT CHUNKED — and this is the point of picking a short class.
#    `|END|` has NEVER worked: hit_eos is 0/204 across two Phase-2 runs. The trainer appends the
#    marker to the FINAL WINDOW only (`--eos-token`, default on), so under chunking it lands at an
#    arbitrary stride boundary uncorrelated with content, and the model cannot learn "the cluster is
#    complete" from it. On the general corpus only 68.5% of records fit whole; **on RIPP ~89% do**.
#    At L=8192 with no chunking, nearly every training example therefore ends with `|END|` at the
#    TRUE cluster boundary — the first clean signal this marker has ever had.
#    ⇒ If it works, generation length becomes an OUTPUT rather than a hyperparameter we impose, and
#      "does the model know when a cluster is finished" becomes measurable.
#    ⇒ The ~11% of records that do not fit are DROPPED, not chunked. That is a real cost (it biases
#      training toward shorter RIPPs) and it is recorded here rather than hidden: mixing chunked and
#      whole records would reintroduce the arbitrary-boundary problem for exactly the records whose
#      endings we most want the model to learn.
#
# 3. DATA IS STRICT CORES WITH --flank 0. No promoters, no regulatory context. Any claim from this
#    model is about a BIOSYNTHETIC CORE, never a cluster ready to express.
#
# 4. NO CUSTOM OBJECTIVE. Phase 2 closed frame-aware and domain-weighted losses on measured
#    evidence (8x stop suppression that changed nothing downstream; a weighting whose in-domain loss
#    was identical at 3x and 10x). This run uses plain causal LM. Adding a loss variant here would
#    confound "does single-class work" with "does this objective work".
#
# The pre-registered evaluation is docs/phase3_preregistration.md. Do not score this model with
# anything else and then report it as the primary result.
set -euo pipefail
export HF_HOME=/data2/ds85/hf_cache
export EVO2_BASE_MODEL=evo2_1b_base

CLASS=${CLASS:-RIPP}
DATA=/data2/ds85/bgcmodel_data/splits_class/$CLASS
ROOT=${ROOT:-/data2/ds85/bgcmodel_runs/phase3_$CLASS}
STEPS=${STEPS:-2000}
SAVE=${SAVE:-500}
L=${L:-8192}

[[ -s "$DATA/train.jsonl" ]] || { echo "[ripp] ABORT: no $DATA/train.jsonl — run scripts/build_single_class_splits.py"; exit 1; }
mkdir -p "$ROOT"

# WHOLE-RECORD MEANS WHOLE-RECORD: filter, do not truncate.
# `--long-seq-strategy truncate` would CUT an over-long record and then append |END| to the cut --
# teaching the model to end a cluster in the middle of one. For a run whose entire purpose is to
# give |END| a clean signal, that is worse than dropping the record. So the over-long records are
# removed here, the count is reported, and truncate then never fires.
micromamba run -n bgcmodel python - "$DATA" "$ROOT" "$L" <<'PYEOF'
import json, sys
from pathlib import Path
data, root, L = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
budget = L - 200          # generous allowance for the prefix + the |END| marker
for split in ("train", "val"):
    keep, drop = [], 0
    for line in (data / f"{split}.jsonl").open():
        r = json.loads(line)
        if len(r["sequence"]) <= budget:
            keep.append(line)
        else:
            drop += 1
    out = root / f"{split}.whole.jsonl"
    out.write_text("".join(keep))
    n = len(keep) + drop
    print(f"[ripp]   {split}: kept {len(keep):,}/{n:,} whole ({len(keep)/n*100:.1f}%), "
          f"dropped {drop:,} over {budget:,} nt")
PYEOF

echo "[ripp] $(date) training $CLASS-only adapter"
echo "[ripp]   train $(wc -l < "$DATA/train.jsonl") records, val $(wc -l < "$DATA/val.jsonl")"
echo "[ripp]   whole-record (NO chunking) at L=$L so |END| lands at the true cluster boundary"

micromamba run -n bgcmodel deepspeed --num_gpus=1 --master_port 29520 \
  evo2/scripts/finetune_evo2_lora.py \
  --train "$ROOT/train.whole.jsonl" --val "$ROOT/val.whole.jsonl" \
  --output-dir "$ROOT/adapter_run" \
  --max-seq-len "$L" --batch-size 1 --grad-accum 16 \
  --long-seq-strategy truncate \
  --warmup-steps 50 --max-epochs 3 --max-steps "$STEPS" \
  --log-every 25 --val-every 250 --save-every "$SAVE" \
  --eos-token \
  --wandb-mode offline \
  > "$ROOT/train.log" 2>&1

echo "[ripp] $(date) DONE — adapter at $ROOT/adapter_run/final_adapter"
echo "[ripp] NEXT: evo2_1b/experiments/phase3_pilot.py has set n; run the pre-registered arms."
