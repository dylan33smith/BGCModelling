# GenomeOcean

> **STATUS 2026-08-14 — LIVE BUT HELD.** The long-standing blocker (unquantified SMC leakage
> against `splits_core`) was **measured and cleared**: bgcFM reconstructs our held-out cores at
> **0.0000** k=21 containment under greedy decoding, positive control demonstrated first
> (`scripts/quantify_smc_leakage.py`). It bounds the risk rather than proving zero overlap.
> GenomeOcean is not the working substrate for Phase 3 — the **1B is** — because running method
> comparisons on two models at once confounds method with model. Revisit for long-context work
> (it fits 64% of BGC regions whole vs Evo2's 0%) and as a third arm in any final comparison.
 track

Everything specific to **GenomeOcean** (Zhou et al., bioRxiv 2025.01.30.635558) lives
here. The Evo2 track lives in [`../evo2/`](../evo2/). Both tracks share the repo-root
dataset pipeline (`../scripts/`), eval suite (`../src/bgc_pipeline/evaluation.py`),
class map (`../config/`) and project memory (`../docs/plan.md`, `../docs/memory.md`, `../docs/terms.md`, `../docs/data.md`) — that shared
instrument is what makes the two models comparable.

Read [`../docs/model_comparison_evo2_vs_genomeocean.md`](../docs/model_comparison_evo2_vs_genomeocean.md)
for why this track exists and what we measured.

## What GenomeOcean is

| | |
|---|---|
| Architecture | `MistralForCausalLM` — 4.25B params, 24 layers, hidden 3072, 12 heads / 4 KV heads (GQA), SiLU, RMSNorm, RoPE (θ=1e6) |
| Tokenizer | BPE, **4,096** vocab, **5.15 bp/token measured on `splits_core`** |
| Context | 10,240 tokens trained (~53 kb); `max_position_embeddings` = 32,768 tokens (~169 kb) |
| Special tokens | `[UNK]`=0 `[CLS]`=1 `[SEP]`=2 (EOS) `[PAD]`=3 `[MASK]`=4; token 8 = `N` |
| Training data | 645 Gbp of metagenome co-assemblies (Tara Oceans, HMP, Lake Mendota, soils, Antarctic) |
| `bgcFM` variant | base 4B further trained on 1.72M deduplicated BGCs (43.5 Gbp) from JGI's SMC database |
| License | BSD-3-Clause (model + code) |

**`bgcFM` is unconditional.** It was fine-tuned on BGC sequences with no product-class
label in the input, so you cannot ask it for an NRPS. Their published "T1PKS" result
(paper §4.3.4) is generate-massively-then-filter: 258,260 generated → antiSMASH →
11,123 positive (4.3%) → 1,459 PKS → 1,044 T1PKS.

## Environment

GenomeOcean needs `transformers>=5.12`, which is incompatible with the Evo2/vortex
stack in `bgcmodel`. It has its own env, on `/data2` because `/home` is full:

```bash
export MAMBA_ROOT_PREFIX=/home/ds85/.local/share/mamba
export HF_HOME=/data2/ds85/hf_cache
micromamba run -p /data2/ds85/envs/genomeocean python <script>
```

Installed: torch 2.11.0**+cu128**, transformers 5.14.1, peft 0.19.1, pyrodigal, biopython.

> **vLLM is installed but non-functional on this host.** The vLLM 0.26 wheel is built
> against CUDA 13 (`libcudart.so.13`) and gputee's driver (575.64.03) is CUDA 12.9, so
> torch had to be refitted to `+cu128` and vLLM's compiled kernels no longer load. All
> scripts here default to the HF `generate()` backend. To recover the ~150× generation
> speedup, install a vLLM release built against cu128 (which will pin an older torch),
> or update the driver to 580+.

Upstream source is cloned (gitignored) at `external/genomeocean/` for reference —
`genomeocean/llm_utils.py` is where the prompting convention and sampling params live.

## Scripts

| Script | What it does |
|---|---|
| `scripts/analyze_tokenization.py` | Measures BPE compression and context fit on `splits_core`, against Evo2's byte-level tokenizer. |
| `scripts/probe_finetune_feasibility.py` | Gate: does bgcFM load, take a LoRA adapter, accept 22 new `[CLS_*]` class tokens, and survive fwd+bwd? Sweeps sequence length / batch size for peak memory. |
| `scripts/generate_bgc_go.py` | Replicates their zero-shot BGC generation, emitting our eval JSONL schema. |

Results land in `experiments/`.

## Reproducing what we ran

```bash
export MAMBA_ROOT_PREFIX=/home/ds85/.local/share/mamba HF_HOME=/data2/ds85/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GO="micromamba run -p /data2/ds85/envs/genomeocean python"

# 1. how much of our data fits (tokenizer only, no GPU)
$GO genomeocean/scripts/analyze_tokenization.py \
    --jsonl /data2/ds85/bgcmodel_data/splits_core/train.jsonl --limit 4000 \
    --out genomeocean/experiments/tokenization_report.json

# 2. fine-tuning + class-token feasibility gate (needs the H100)
$GO genomeocean/scripts/probe_finetune_feasibility.py \
    --sweep 4096 10240 16384 32768 \
    --out genomeocean/experiments/finetune_feasibility.json

# 3. their zero-shot BGC generation
$GO genomeocean/scripts/generate_bgc_go.py \
    --num 24 --preset creative_long --backend hf --hf-batch-size 8 \
    --out /data2/ds85/bgcmodel_runs/go_zeroshot_bgcfm

# 4. score it on OUR antiSMASH gate (bgcmodel env — that's where antiSMASH lives)
micromamba run -n bgcmodel python scripts/eval_suite_driver.py \
    --gen /data2/ds85/bgcmodel_runs/go_zeroshot_bgcfm/gen.jsonl \
    --skip-checks protein_homology kmer_novelty \
    --output /data2/ds85/bgcmodel_runs/go_zeroshot_bgcfm/eval.json
```

Note the env split in step 4: generation runs in `genomeocean`, scoring runs in
`bgcmodel` (antiSMASH 8.0.4 + Pfam live there). The JSONL is the handoff.
