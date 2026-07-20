#!/bin/bash
#SBATCH -J bgc-longctx
#SBATCH -p hopper
#SBATCH -q hopper
#SBATCH -A CHANGEME_ACCOUNT          # <-- your RT Project Slurm account
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00            # 2-day hopper max; chain with resubmit_chain.sh
#SBATCH -o longctx_%j.out
# ============================================================================
# One 2-day segment of the long-context whole-cluster run. Resumes the latest
# checkpoint if present (faithful --resume-from). Do NOT run blind to completion
# — pair with milestone eval (n>=15 every ~epoch) and stop if correct_class
# stays at the floor (see experiments/probes/run_optA.sh for the gate pattern).
#
# PREREQ: the 2-GPU prototype passed and the parallelism approach is wired in.
# Adjust --max-seq-len / --num_gpus to what the prototype proved fits:
#   L=131072 ~ 2 GPUs   |   L=262144 (all cores whole) ~ 4 GPUs
# ============================================================================
set -uo pipefail
module load conda cudatoolkit/12.6
conda activate bgcmodel
export HF_HOME=/N/slate/$USER/hf_cache
DATA=/N/slate/$USER/bgcmodel_data
RUN=/N/slate/$USER/bgcmodel_runs/mega_longctx     # Slate (persistent) — NOT scratch (purged)
mkdir -p "$RUN"; cd /N/slate/$USER/BGCModelling

# resume newest checkpoint with an adapter/, if any
RESUME=""
for d in $(ls -dt "$RUN"/checkpoints/step_* 2>/dev/null); do
  [ -d "$d/adapter" ] && { RESUME="--resume-from $d"; break; }
done
echo "resume: ${RESUME:-fresh from base}"

deepspeed --num_gpus=4 scripts/finetune_evo2_lora.py \
  --train "$DATA/mega_all.jsonl" --val "$DATA/splits_core/val.jsonl" \
  --output-dir "$RUN" \
  --max-seq-len 262144 --batch-size 1 --grad-accum 64 \
  --lora-dropout 0 --lr 5e-5 --warmup-steps 30 --max-epochs 6 \
  --val-every 60 --save-every 60 --log-every 10 \
  --long-seq-strategy chunk --chunk-overlap 2048 --eos-token --continuation-prefix \
  --keep-last-ckpts 0 --early-stopping-patience 0 \
  --wandb-mode offline --seed 42 $RESUME
