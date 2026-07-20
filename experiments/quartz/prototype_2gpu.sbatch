#!/bin/bash
#SBATCH -J bgc-proto
#SBATCH -p hopper
#SBATCH -q hopper
#SBATCH -A CHANGEME_ACCOUNT          # <-- your RT Project Slurm account
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH -o proto_%j.out
# ============================================================================
# GO/NO-GO PROTOTYPE. Can our LoRA + vortex/Evo2 stack train L=65,536 across
# 2 H100s? Just 3 steps. This decides whether the long-context run is reachable
# before any big compute is requested.
#
#   - If it TRAINS 3 steps: DeepSpeed ZeRO across 2 GPUs suffices at this L;
#     scale up (4 GPUs, larger L) in longcontext.sbatch.
#   - If it OOMs: the model's ACTIVATIONS aren't being split — implement
#     sequence/model parallelism (docs/quartz_setup.md §6: Evo2 model_parallel_size
#     or DeepSpeed-Ulysses) and re-run. That integration is the real engineering.
# ============================================================================
set -uo pipefail
module load conda cudatoolkit/12.6
conda activate bgcmodel
export HF_HOME=/N/slate/$USER/hf_cache
DATA=/N/slate/$USER/bgcmodel_data
cd /N/slate/$USER/BGCModelling

nvidia-smi --query-gpu=name,memory.total --format=csv
deepspeed --num_gpus=2 scripts/finetune_evo2_lora.py \
  --train "$DATA/mega_all.jsonl" --val "$DATA/splits_core/val.jsonl" \
  --output-dir /N/scratch/$USER/proto_2gpu \
  --max-seq-len 65536 --batch-size 1 --grad-accum 2 \
  --lora-dropout 0 --lr 5e-5 --warmup-steps 2 --max-epochs 1 --max-steps 3 \
  --val-every 99 --save-every 99 --log-every 1 \
  --long-seq-strategy chunk --chunk-overlap 2048 --eos-token \
  --wandb-mode offline --seed 42
echo "PROTO exit=$?  (OOM at L=65536 => add parallelism per docs/quartz_setup.md §6)"
