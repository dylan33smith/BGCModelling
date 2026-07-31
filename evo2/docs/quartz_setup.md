# Quartz (IU) long-context run — setup & execution guide

**Purpose.** Run the one experiment single-GPU couldn't: fine-tune Evo2 7B on **whole
megasynthase clusters** at long context (L up to 262,144) using **multi-GPU** on IU's
Quartz `hopper` partition (4× H100 SXM/NVLink per node). This tests whether whole-*cluster*
context converts the domain-level gains into `correct_class` — see
`docs/project_memory/decisions.md` (2026-07-10 → 07-13) for why every single-GPU-cheap lever
(coverage, imbalance, chunk-labeling, gene-aware chunking, whole-core-at-L=32k, LoRA rank)
came back **negative**. This is a real research-engineering bet with an uncertain prior; the
**2-GPU prototype (Phase 4) is the go/no-go** before committing serious compute.

> Read this top-to-bottom in a fresh Claude Code session started **on Quartz** (which has
> Bash access to the cluster). Also read `docs/project_memory/progress.md` +
> `decisions.md` for full context.

---

## 0. Prerequisites (must be true before anything runs)

- **Quartz login** works: `ssh ds85@quartz.uits.iu.edu` (+ Duo).
- **RT Project allocation** granted — this mints the Slurm account used as `-A`. Verify:
  ```bash
  sacctmgr -np show assoc where user=ds85 format=cluster,account,qos
  ```
  The `account` column is your `<ACCOUNT>`. Sanity-test it:
  ```bash
  srun -p debug -A <ACCOUNT> --time=00:03:00 --pty hostname   # prints c1/c2
  ```
  If this errors with "must include an RT Project", the allocation isn't attached yet — wait
  on the PI / projects.rt.iu.edu.

## Cluster facts (confirmed 2026-07)

- `hopper` partition: 12 nodes `g[25-36]`, each **4× H100 (gpu:h100:4)**, **515 GB RAM**,
  **96 cores**, local NVMe. **Max walltime = 2 days.** `AllowAccounts=ALL`, QOS `hopper`
  (also `long` QOS exists — check `sacctmgr show qos long format=name,maxwall`). Partition is
  usually **fully allocated → jobs queue.**
- Storage: `/N/slate/ds85` (personal Slate, ~1.6 TB — enough), `/N/scratch/ds85` (huge, but
  **purged** periodically — use for scratch only), home `/N/u/ds85/Quartz` (tiny, ~50 GB).
- Modules: `conda` (25.3/26.3), `cudatoolkit` up to 12.6, `cudnn/8.9_cuda12`, `cmake`, gcc,
  openmpi. **Login node has internet; assume compute nodes may not** — pre-stage everything.
- Set a project root once: `export BGC=/N/slate/ds85` (used throughout below).

---

## 1. Environment (build on the LOGIN node — it has internet)

Target stack (exact versions from the lab box): **python 3.12 · torch 2.5.1+cu124 · evo2 0.5.5
· deepspeed 0.18.9 · peft 0.19.0 · transformers 4.46.3 · flash-attn 2.7.4.post1**. Script:
`experiments/quartz/env_setup.sh`. Or manually:

```bash
module load conda cudatoolkit/12.6 cmake
export HF_HOME=/N/slate/ds85/hf_cache
conda create -y -n bgcmodel python=3.12 && conda activate bgcmodel
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install evo2==0.5.5
pip install deepspeed==0.18.9 peft==0.19.0 transformers==4.46.3 accelerate==1.13.0
pip install pyrodigal==3.7.1 pyhmmer==0.12.0 biopython==1.81
pip install flash_attn==2.7.4.post1 --no-build-isolation   # may compile (needs cudatoolkit+ninja); paste errors
python -c "import torch,evo2,vortex,deepspeed,peft; print(torch.__version__, torch.version.cuda)"
```

**Optional but valuable here:** the H100s support FP8, so Transformer Engine *can* be built on
Quartz — it was the blocker for the 1B model locally and gives FP8 speedups. Try
`pip install transformer_engine[pytorch]` after the above; if it fails to build, skip it (7B
runs fine without it via the bf16 fallback).

## 2. Data (stage onto Slate — ~14 GB, no 185 GB tar needed)

Long-context whole-cluster training needs only the Evo2 weights + `splits_core`. Script:
`experiments/quartz/stage_data.sh`.

```bash
export HF_HOME=/N/slate/ds85/hf_cache
huggingface-cli download arcinstitute/evo2_7b_262k      # ~13 GB, on login node
# splits_core comes with the repo clone? No — it's data, not code. Push from the lab box:
#   (run on the LAB box)  rsync -avP /data2/ds85/bgcmodel_data/splits_core/ \
#       ds85@quartz.uits.iu.edu:/N/slate/ds85/bgcmodel_data/splits_core/
```

## 3. Clone the code

```bash
cd /N/slate/ds85 && git clone https://github.com/dylan33smith/BGCModelling.git   # or the branch
cd BGCModelling
```
Edit any absolute paths: the training/eval scripts default to lab-box paths (`/data2/ds85/...`).
On Quartz, point them at `/N/slate/ds85/...` via the CLI flags (`--train`, `--val`,
`--output-dir`) and `export HF_HOME=/N/slate/ds85/hf_cache`. The `EVO2_BASE_MODEL` env override
and all training flags work unchanged.

## 4. GPU / NVLink verification (short interactive job)

```bash
srun -p hopper -q hopper -A <ACCOUNT> --nodes=1 --gpus-per-node=4 --cpus-per-task=16 \
     --mem=128G --time=00:20:00 --pty bash
# on the node:
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
nvidia-smi topo -m       # want NV# (NVLink) between all 4 GPUs, not SYS
module load conda; conda activate bgcmodel
python -c "import torch; print('gpus', torch.cuda.device_count())"   # expect 4
curl -sI --max-time 8 https://huggingface.co | head -1              # compute-node internet? (may be blocked)
```

## 5. Build the long-context dataset (mega-only, ALL lengths kept whole)

At high L, every mega core fits whole — **no length filter, no dropping** (this is what
single-GPU couldn't do). Just filter `splits_core` to the megasynthase classes:
```bash
python - <<'PY'
import json
mega={'NRPS','PKS','PKS_NRPS_HYBRID'}
o=open('/N/slate/ds85/bgcmodel_data/mega_all.jsonl','w'); n=0
for l in open('/N/slate/ds85/bgcmodel_data/splits_core/train.jsonl'):
    if json.loads(l).get('compound_class') in mega: o.write(l); n+=1
print('mega cores:', n)          # ~9,629, ~209 Mbp, max len 262,144
PY
python evo2/scripts/build_chunk_index.py --jsonl /N/slate/ds85/bgcmodel_data/mega_all.jsonl \
    --max-seq-len 262144 --chunk-overlap 2048
```

## 6. THE ENGINEERING: multi-GPU long context (do the 2-GPU prototype FIRST)

Our harness (`evo2/scripts/finetune_evo2_lora.py`) is **single-GPU DeepSpeed**. To fit L≥131k the
model/activations must be split across the 4 GPUs. This is the real work and the main risk.
Two routes:

- **Evo2/vortex native model parallelism.** StripedHyena's config exposes `model_parallel_size`
  — set it to 2 or 4 to shard weights + the activations that blow up at long L. **Risk:** LoRA
  (PEFT) over sharded weights is non-trivial (we already needed `target_parameters` tricks).
- **DeepSpeed sequence parallelism (Ulysses).** Shards the sequence dimension — natural for long
  context — but its interaction with StripedHyena's long convolutions is unverified.

**Go/no-go prototype — do this before requesting big compute:** get **L=65,536 training on
2 GPUs**, 3 steps, confirming (a) memory splits across GPUs and (b) loss computes with our
LoRA+vortex stack. Template: `experiments/quartz/prototype_2gpu.sbatch`. If the prototype works,
scale to 4 GPUs / L=131k–262k. If neither parallelism route wraps LoRA cleanly, that is itself a
finding — report it before sinking weeks in.

Memory math (single-GPU extrapolation ÷ N GPUs): L=131k ≈ 144 GB → 2 GPUs; L=262k ≈ 275 GB →
4 GPUs (one node = 320 GB NVLink). So one hopper node covers the full whole-cluster run.

## 7. The run: sbatch + resubmit chaining + milestone gating

Walltime is **2 days**; the full run is longer, so **chain 2-day jobs that resume the latest
checkpoint** (our `--resume-from` is faithful). Template: `experiments/quartz/longcontext.sbatch`
+ `experiments/quartz/resubmit_chain.sh`. Keep the **milestone-gated eval** (n≥15 every ~epoch,
auto-kill if `correct_class` stays at the floor) exactly as `experiments/probes/run_optA.sh` does
— do NOT run blind to completion.

## 8. Success criterion

`correct_class` (at n≥15) **climbs off the ~0.067 floor** across milestones → whole-cluster
context is the answer. If it stays flat / degrades (as every prior run did), that closes the
de-novo-generation direction and the recommendation becomes **reposition Evo2 as a BGC
evaluator/scorer** (its recalibrated antiSMASH+eval stack is already strong).

---

### Quick checklist
- [ ] RT Project `-A` account works (`srun -p debug` test passes)
- [ ] `bgcmodel` conda env builds; `import torch,evo2,vortex,deepspeed,peft` OK
- [ ] Evo2 7B weights on `/N/slate/ds85/hf_cache`; `splits_core` on Slate
- [ ] 4× H100 + NVLink confirmed (`nvidia-smi topo -m`)
- [ ] `mega_all.jsonl` + sidecar built at L=262144
- [ ] **2-GPU L=65k prototype trains 3 steps** ← go/no-go gate
- [ ] longcontext.sbatch + resubmit chain + milestone gate wired
- [ ] launch; watch `correct_class` per milestone
