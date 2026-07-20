#!/usr/bin/env bash
# Build the bgcmodel conda env on the Quartz LOGIN node (has internet).
# Exact stack captured from the lab box (torch 2.5.1+cu124 / evo2 0.5.5 / ...).
set -uo pipefail
module load conda cudatoolkit/12.6 cmake
export HF_HOME=${HF_HOME:-/N/slate/$USER/hf_cache}
conda create -y -n bgcmodel python=3.12 || true
conda activate bgcmodel
set -e
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install evo2==0.5.5                                   # pulls vortex + most deps
pip install deepspeed==0.18.9 peft==0.19.0 transformers==4.46.3 accelerate==1.13.0
pip install pyrodigal==3.7.1 pyhmmer==0.12.0 biopython==1.81
set +e
pip install flash_attn==2.7.4.post1 --no-build-isolation \
  || echo "!! flash_attn build failed — see docs/quartz_setup.md §1 (usually a prebuilt wheel fixes it)"
# Optional (H100 FP8; unblocks the 1B model + speedups). Skip if it won't build.
# pip install transformer_engine[pytorch]
python -c "import torch,evo2,vortex,deepspeed,peft; print('OK: torch',torch.__version__,'cuda',torch.version.cuda,'gpus_visible',torch.cuda.device_count())"
