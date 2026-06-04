# BCGModelling — State and Audit Document

*Generated 2026-05-27. Intended as a comprehensive onboarding and audit
reference for anyone joining or resuming work on this project.*

---

## 1  Project Objective

### What are we building?

A fine-tuned version of **Evo2 7B** (a 6.51-billion-parameter DNA language
model built on the StripedHyena 2 architecture) that generates
synthesis-ready **biosynthetic gene cluster (BGC)** nucleotide sequences on
demand.

### What is a BGC?

A biosynthetic gene cluster is a contiguous stretch of DNA (typically
20–150 kb) encoding all the genes needed for an organism to produce a
specific natural product — an antibiotic, pigment, siderophore, or other
secondary metabolite. BGCs are the molecular blueprints behind compounds
like penicillin, erythromycin, and violacein.

### What does "conditioned generation" mean here?

The model receives a structured text prefix that specifies:

1. **COMPOUND_CLASS** — the biosynthetic family (e.g. `PKS`, `NRPS`,
   `TERPENE`, `RIPP`).
2. **Taxonomic tag** — the target chassis organism's full Linnaean lineage
   in all-caps format (domain through species).
3. (Future phases) **COMPOUND** name or **SMILES** string for finer control.

Given that prefix, the model autoregressively generates a nucleotide
sequence that should:

- Be annotated by antiSMASH as the correct biosynthetic class.
- Contain the expected Pfam protein domains for that class.
- Use codon frequencies, GC content, and regulatory motifs appropriate for
  the target chassis (e.g. *E. coli*).
- Be synthesisable (no forbidden restriction sites, homopolymers, extreme
  GC windows).

### Transposition, not invention

The project is explicitly scoped to **transposition**: taking a known BGC
architecture and re-expressing it for a new chassis organism. It is **not**
designed to invent novel biosynthetic pathways. This distinction is
fundamental:

- **Transposition** works because the model learns biosynthetic
  architecture from MIBiG (2,636 curated BGCs) and chassis-specific
  sequence statistics from antiSMASH DB (343,923 BGCs across 55,950
  genomes). These are orthogonal signals from different parts of the
  training data.
- **Invention** would require the model to learn what a compound *is*
  chemically from its name token alone. Since 91% of MIBiG compound tokens
  appear in exactly one training record, the model cannot learn
  compound-level chemical semantics — it can only memorise and adapt the
  single example.

### Three target compounds for wet-lab validation

| Compound    | Class    | Detection              | Key genes       |
|-------------|----------|------------------------|-----------------|
| Violacein   | OTHER    | Blue-purple, HPLC 575nm| vioABCDE (5)    |
| Carotenoid  | TERPENE  | Yellow-orange, HPLC 450nm| crtEBIYZ (~6) |
| Indigoidine | NRPS     | Bright blue, HPLC 612nm| BpsA + sfp (2)  |

All are BSL-1, expressible in *E. coli* BL21(DE3), and visually
detectable before HPLC.

### Phased conditioning plan

| Phase | Format | Data | Status |
|-------|--------|------|--------|
| **1 (current)** | `\|COMPOUND_CLASS:{cls}\|{tax_tag}{seq}` | All 346,559 records, uniform | Ready to train |
| 2 | Add `\|COMPOUND:{tok}\|` for MIBiG compounds with ≥3 examples (~45 compounds) | Subset of MIBiG | Blocked on Phase 1 results |
| 3 | Replace COMPOUND with `\|SMILES:{canonical}\|` | 2,118 MIBiG records (80.3%) with SMILES | Future |

---

## 2  Architecture and Stack

### 2.1  Model: Evo2 7B

- **Architecture:** StripedHyena 2 (SH2) — a hybrid of Hyena
  long-convolution operators and standard multi-head attention.
- **Parameters:** 6.51 billion (6,509,764,352).
- **Context window:** 262,144 nucleotides (262 kb).
- **Vocabulary:** Single-nucleotide tokens (A, C, G, T) plus special tokens.
- **Source:** Arc Institute, installed via `pip install evo2==0.5.5`.
- **Checkpoint:** `arcinstitute/evo2_7b_262k` from HuggingFace (~14 GB),
  cached at `/data2/ds85/hf_cache/hub/models--arcinstitute--evo2_7b_262k/`.

**Why StripedHyena matters for memory:**

The dominant memory consumer in Evo2 is not the attention mechanism — it is
the Hyena long-convolution filter (`compute_filter()`), which materialises
a `[poles × channels × L]` tensor during every forward pass. This is an
FFT-based operation whose memory scales linearly with sequence length L.
At L=32768 without activation checkpointing, this single operation
consumes enough memory to OOM an 80 GB H100. This is the central hardware
constraint of the project.

### 2.2  Training strategy: LoRA

Full-parameter fine-tuning requires ~84 GB of GPU memory (14 GB bf16
weights + 14 GB gradients + 56 GB AdamW fp32 states) *before*
activations — it does not fit on 80 GB. LoRA is the adopted solution.

| LoRA parameter | Value | Effect |
|----------------|-------|--------|
| Rank (r)       | 16    | 28.7M trainable params |
| Alpha          | 32    | Scaling = alpha/r = 2.0 |
| Dropout        | 0.05  | Light regularisation |
| Target modules | All 133 nn.Linear layers | Wqkv, out_proj, out_filter_dense, l1, l2, l3 |
| **Trainable fraction** | **0.44%** | 28,704,768 / 6,509,764,352 |

Optimizer state drops from ~56 GB (full fine-tune) to ~336 MB (LoRA
adapters only). The base model's 14 GB of bf16 weights remain frozen in
memory but do not require optimizer states.

### 2.3  Orchestration: DeepSpeed ZeRO-2

- **DeepSpeed 0.18.9** with ZeRO Stage 2 (optimizer + gradient sharding).
- At `world_size=1` (single GPU), ZeRO-2 is effectively a thin bf16
  mixed-precision + gradient-accumulation wrapper.
- `exclude_frozen_parameters=True` ensures frozen base-model weights are
  excluded from ZeRO partitioning and checkpoint serialisation — without
  this, each checkpoint would be ~14 GB instead of ~120 MB.
- Launcher: `deepspeed --num_gpus=1`.

### 2.4  Activation checkpointing

Block-level activation checkpointing is **default-on** since 2026-04-26.
It is implemented via `torch.utils.checkpoint.checkpoint()` with
`use_reentrant=False` (required because LoRA dropout is non-zero).

**Impact:**

| L       | No-AC peak | AC-on peak | Savings |
|---------|----------:|----------:|--------:|
| 1,024   | 23.52 GB  | 16.35 GB  | -7.2 GB |
| 8,192   | 80.10 GB  | 22.77 GB  | -57.3 GB|
| 32,768  | OOM       | 43.92 GB  | ∞       |
| 65,536  | OOM       | 74.11 GB  | ∞       |

Without checkpointing, the project cannot train above L=8192. With
checkpointing, L=32768 fits comfortably and L=65536 is feasible with ~6 GB
headroom.

### 2.5  Hardware: gputee

| Resource | Spec |
|----------|------|
| GPU      | 1× NVIDIA H100 PCIe, 80 GB VRAM (81,559 MiB) |
| Driver   | 575.64.03 / CUDA 12.9 runtime |
| CPU      | 2× AMD EPYC 9124 (32c/64t) |
| RAM      | 376 GiB |
| Storage  | `/home` 1.8 TB (~16 GB free); `/data2` 7 TB (~1.5 TB free) |

**Critical storage note:** `/home` is essentially full. All run output
must go to `/data2` via `--output-dir /data2/ds85/bgcmodel_runs/...` and
`HF_HOME=/data2/ds85/hf_cache` must be exported in every training shell.

**Shared host:** gputee is a multi-user machine. The GPU is shared — there
is no reservation system (CUDA's `EXCLUSIVE_PROCESS` mode requires
`sudo`). Queue scripts gate on GPU idleness before launching.

### 2.6  Software versions (pinned)

| Package       | Version        | Pin reason |
|---------------|----------------|------------|
| Python        | 3.12.13        | Conda env |
| PyTorch       | 2.5.1+cu124    | 2.6 breaks flash-attn ABI |
| flash-attn    | 2.7.4.post1    | Prebuilt wheel for cu12+torch2.5 |
| transformers  | 4.46.3         | 5.x blocks .pt loading on torch<2.6 |
| peft          | 0.19.0         | LoRA adapters (3 compatibility fixes applied) |
| deepspeed     | 0.18.9         | ZeRO-2 engine |
| evo2          | 0.5.5          | Model loader |
| accelerate    | 1.13.0         | HF integration |
| wandb         | 0.26.0         | Experiment tracking |

### 2.7  Repository layout

```
BCGModelling/
├── docs/gputee/               # ACTIVE documentation
│   ├── PROJECT_GUIDE.md       # Living project status (§13 = task tracker)
│   ├── FINETUNE_GUIDE.md      # Hardware, hyperparams, launch templates, smoke findings
│   ├── BGC_Research_Plan.md   # Research plan v9 (11 sections)
│   └── MIGRATION_CHANGELOG.md # trojai → gputee port log
├── docs/trojai/               # ARCHIVED (4× A40 setup — read-only)
├── config/
│   └── compound_class_map.yaml # 91 antiSMASH/MIBiG product types → 20+ harmonised tokens
├── src/bgc_pipeline/
│   ├── class_map.py           # YAML class map loader
│   ├── taxonomy.py            # NCBI taxdump → Evo2 taxonomic tags
│   ├── mibig_record.py        # MIBiG JSON+GBK → training records
│   └── evaluation.py          # 8-metric evaluation suite
├── scripts/
│   ├── finetune_evo2_lora.py  # THE training script (~2000 lines)
│   ├── finetune_evo2.py       # Full fine-tune (reference only; OOMs)
│   ├── mibig_to_jsonl.py      # Step 1a: MIBiG → JSONL
│   ├── antismash_db_to_jsonl.py # Step 1b: antiSMASH DB → JSONL
│   ├── split_dataset.py       # Step 2: stratified splits
│   ├── queue_h100_smoke.sh    # Smoke sweep queue script
│   ├── queue_h100_pilot.sh    # Pilot run queue script ⚠️ SEE §7
│   └── queue_h100_resume_test.sh # Resume verification test
└── data/                      # All data (gitignored)
    ├── mibig/                 # MIBiG 4.0 (JSON + GBK)
    ├── antismash_db/          # antiSMASH DB v5 (taxa JSON only; 173GB tar absent)
    ├── ncbi_taxonomy/         # names.dmp + nodes.dmp
    ├── pfam/                  # Pfam-A.hmm 37.0
    ├── npatlas/               # NPAtlas 3.0 (36,454 compounds)
    ├── uniref50/              # MMseqs2 UniRef50 DB (29 GB)
    └── processed/
        └── mibig_train_records.jsonl      # 2,636 records (MIBiG source)
# NOTE (2026-06-04 disk cleanup): the large data now lives on /data2 (the /home
# disk was full). Locations:
#   /data2/ds85/bgcmodel_data/asdb5_train_records.jsonl   # 343,923 records, 22 GB (antiSMASH source)
#   /data2/ds85/bgcmodel_data/splits_combined_grouped/    # leakage-free FULL split
#   /data2/ds85/bgcmodel_data/splits_curated/             # ACTIVE: curated train (~18K) + full val/test
# REMOVED: data/processed/splits_combined/ (leaky, deprecated) and
#          data/processed/splits/ (historical MIBiG-only), plus all obsolete
#          /data2/ds85/bgcmodel_runs/* checkpoints (trained on the leaky split).
```

---

## 3  Implementation History (Chronological)

### Phase A: trojai (4× NVIDIA A40, 48 GB each)

| Date | Milestone |
|------|-----------|
| 2026-04-14 | Initial commit: MIBiG data acquisition |
| 2026-04-14 | antiSMASH DB v5 processing pipeline + compound_class_map.yaml |
| 2026-04-15 | Full fine-tune smoke test on trojai — 7 attempts, 3 Evo2↔DeepSpeed bugs fixed, OOM at `optimizer.step()` confirmed |
| 2026-04-15 | LoRA smoke test on trojai — 5 additional peft bugs fixed, 10-step test passes end-to-end |
| 2026-04-15 | Evaluation suite implemented (8 metrics), validated on 3 MIBiG BGCs + shuffled controls |
| 2026-04-15 | antiSMASH DB contig_edge annotation (343,923 records annotated) |
| 2026-04-15 | Combined train/val/test splits generated (277K/35K/35K stratified by COMPOUND_CLASS) |

### Phase B: Migration to gputee (1× NVIDIA H100, 80 GB)

| Date | Milestone |
|------|-----------|
| 2026-04-22 | Environment rebuilt from scratch on gputee (micromamba + pinned pip installs) |
| 2026-04-22 | Code + processed data migrated; docs split into `docs/trojai/` (archived) and `docs/gputee/` (active) |
| 2026-04-25 | H100 smoke benchmark sweep WITHOUT activation checkpointing: L=1024→8192 pass, L=16384+ OOM |
| 2026-04-26 | Block-level activation checkpointing implemented and validated |
| 2026-04-26 | H100 smoke benchmark sweep WITH checkpointing: L=1024→65536 pass, L=98304 OOM |
| 2026-04-26 | Padded long-L probes: ceiling bracketed between L=65536 (74.11 GB) and L=98304 (OOM) |
| 2026-04-28 | NPAtlas and UniRef50 restored on gputee |
| 2026-04-29→05-01 | Production-like preflight (batch=4, ga=32, AC on): L=40960→65536 pass at 20 steps each |

### Phase C: Production readiness (current)

| Date | Milestone |
|------|-----------|
| 2026-05-11 | Docs refreshed post-preflight; §13 NEXT retargeted to L=32768 pilot |
| 2026-05-14 | **Critical discovery:** bs=4 OOMs at L=32768 on clean GPU. FINETUNE_GUIDE updated to require bs=1 ga=128 |
| 2026-05-20 | Resume-from-checkpoint: 4 bugs fixed (adapter reload, ZeRO state, step counter, RNG state) |
| 2026-05-20 | Pilot queue script (`queue_h100_pilot.sh`) created and committed |
| 2026-05-20 | Pilot queued in bgcruns tmux session (waiting for GPU to become free) |

---

## 4  Core Mechanics of the Training Script

This section documents how `scripts/finetune_evo2_lora.py` works internally.

### 4.0  Dataset composition and characteristics

> **⚠️ SUPERSEDED PATHS / COUNTS (updated 2026-06-02).** The analysis below
> describes the **original full** combined dataset at
> `data/processed/splits_combined/` (277,238 train records). That split was
> later found to leak badly (94.6% genome overlap across splits — see
> AUDIT_FINDINGS.md C1/C2) and is **deprecated**. The project now uses:
> - `/data2/ds85/bgcmodel_data/splits_combined_grouped/` — group-aware,
>   leakage-free re-split of the same records (`scripts/split_dataset_grouped.py`).
> - `/data2/ds85/bgcmodel_data/splits_curated/` — **ACTIVE**: curated
>   leakage-free set (`scripts/curate_dataset.py`); train ~18K, quality-filtered
>   (no N / no contig-edge), per-class capped at 1000, diversity-stratified;
>   val/test kept full (~25K each).
>
> The distribution facts below (class imbalance, length structure, taxonomy
> skew, etc.) remain accurate descriptions of the underlying biology and the
> reasons curation was needed; only the splitting and the active counts changed.

This section characterises the combined training dataset to make its
structure, biases, and limitations concrete. All numbers are from the
original full training split at `data/processed/splits_combined/train.jsonl`
unless otherwise noted.

#### Split sizes

| Split | Records |
|-------|---------|
| Train | 277,238 |
| Val   | 34,655  |
| Test  | 34,655  |
| **Total** | **346,548** |

Splits are stratified by compound class (80/10/10).

#### Source composition

| Source | Records | % of train |
|--------|---------|-----------|
| antiSMASH DB v5 | 275,141 | 99.2% |
| MIBiG 3.1 | 2,097 | 0.8% |

The dataset is overwhelmingly antiSMASH-derived. MIBiG contributes only
2,097 records but these are the highest-quality, experimentally validated
BGCs. The model's understanding of "what a real BGC looks like" will be
shaped primarily by antiSMASH's computational predictions.

#### Sequence length distribution

| Statistic | Length (characters) |
|-----------|-------------------|
| Min       | 341               |
| P10       | 11,320            |
| P25       | 20,992            |
| **Median**| **23,075**        |
| Mean      | 33,033            |
| P75       | 42,561            |
| P90       | 58,996            |
| P95       | 73,844            |
| P99       | 123,858           |
| Max       | 262,337           |

The distribution is right-skewed: the median is 23K but the mean is 33K,
pulled up by a long tail of very large BGCs (mostly NRPS, PKS, and hybrids).
The maximum (262K) approaches Evo2's native context window (262,144 tokens).

#### Compound class breakdown

26 harmonised compound classes, with extreme imbalance:

| Class | Records | % of train | Median len | % > 32K | Category |
|-------|---------|-----------|------------|---------|----------|
| TERPENE | 69,131 | 24.9% | 21,077 | 4.7% | Short |
| RIPP | 67,988 | 24.5% | 20,852 | 4.2% | Short |
| NRPS | 32,940 | 11.9% | 46,064 | 89.5% | Long |
| PKS | 22,702 | 8.2% | 43,834 | 94.5% | Long |
| OTHER | 22,698 | 8.2% | 29,958 | 27.6% | Mixed |
| PKS_NRPS_HYBRID | 11,452 | 4.1% | 72,905 | 99.2% | Long |
| SIDEROPHORE | 11,269 | 4.1% | 55,102 | 99.2% | Long |
| BETALACTONE | 9,513 | 3.4% | 28,065 | 9.4% | Short |
| ARYLPOLYENE | 8,667 | 3.1% | 43,735 | 96.2% | Long |
| HSERLACTONE | 5,728 | 2.1% | 20,782 | 4.0% | Short |
| ECTOINE | 4,955 | 1.8% | 10,534 | 1.7% | Short |
| BUTYROLACTONE | 2,686 | 1.0% | 11,092 | 8.5% | Short |
| PHOSPHONATE | 1,488 | 0.5% | 18,669 | 18.5% | Mixed |
| MELANIN | 1,423 | 0.5% | 10,614 | 1.5% | Short |
| CDPS | 977 | 0.4% | 20,869 | 6.8% | Short |
| RESORCINOL | 964 | 0.3% | 42,040 | 93.8% | Long |
| PHENAZINE | 701 | 0.3% | 21,155 | 16.1% | Mixed |
| ALKALOID | 636 | 0.2% | 21,909 | 9.3% | Short |
| SACCHARIDE | 510 | 0.2% | 33,501 | 51.2% | Long |
| FURAN | 340 | 0.1% | 21,120 | 6.8% | Short |
| BETALACTAM | 154 | 0.1% | 23,630 | 17.5% | Mixed |
| PUFA | 151 | 0.1% | 55,943 | 98.7% | Long |
| NUCLEOSIDE | 133 | 0.0% | 21,389 | 20.3% | Mixed |
| PHOSPHOGLYCOLIPID | 16 | 0.0% | 28,911 | 18.8% | Mixed |
| LADDERANE | 14 | 0.0% | 46,866 | 100% | Long |
| PBDE | 2 | 0.0% | 25,386 | 0% | Short |

**Two distinct populations emerge:**

- **"Short" classes** (median < 30K): TERPENE, RIPP, ECTOINE, HSERLACTONE,
  BETALACTONE, etc. These encode small enzymes or ribosomally-synthesised
  peptides. They fit comfortably within L=32768 and need no chunking.

- **"Long" classes** (median > 40K): NRPS, PKS, PKS_NRPS_HYBRID,
  SIDEROPHORE, ARYLPOLYENE, RESORCINOL. These encode large multi-domain
  enzymatic assembly lines (particularly the NRPS/PKS megasynthases). Over
  89% of their records exceed L=32768 and require chunking.

This split is not random — it reflects fundamental biology. The
pharmacologically most important classes (NRPS, PKS, hybrids) are
inherently the longest and most expensive to train on.

**Class imbalance**: TERPENE + RIPP account for 49.4% of training data.
NRPS + PKS + hybrids (the most drug-relevant classes) account for 24.2%.
The long tail includes classes with as few as 2 records (PBDE), 14
(LADDERANE), and 16 (PHOSPHOGLYCOLIPID). The model will develop stronger
generation capability for the dominant short classes than for rare or
long classes. This is documented as risk B3 in §6.

#### Chunk-mode window expansion

With `--long-seq-strategy chunk` at L=32768 and overlap=2048, sequences
exceeding L are tiled into overlapping windows. This expands the dataset:

| Configuration | Windows | Multiplier | Steps/epoch (ga=128) |
|---------------|---------|-----------|---------------------|
| Truncate (no chunking) | 277,238 | 1.00x | 2,166 |
| **Chunk, overlap=2048** | **410,534** | **1.48x** | **3,207** |
| Chunk, overlap=512 | 408,084 | 1.47x | 3,188 |
| Chunk, overlap=0 | 406,803 | 1.47x | 3,178 |

The 1.48x expansion is driven almost entirely by the "long" classes:
NRPS, PKS, hybrids, siderophores, and arylpolyenes. Overlap has minimal
impact because most long sequences need only 2-3 windows, where the
overlap region is small relative to L.

**Length filter analysis**: filtering to sequences <= L (eliminating chunking)
would reduce training time by ~3x but catastrophically drops the long classes:

| Filter | Records kept | Time (1 ep) | NRPS | PKS | Hybrid | Siderophore |
|--------|-------------|-------------|------|-----|--------|-------------|
| <= 32K | 179,685 (65%) | ~7 days | 11% | 5% | 1% | 1% |
| <= 50K | 232,325 (84%) | ~13 days | 60% | 69% | 11% | 15% |
| <= 65K | 256,875 (93%) | ~16 days | 83% | 81% | 41% | 83% |
| <= 80K | 266,116 (96%) | ~18 days | 92% | 93% | 58% | 91% |
| All    | 277,238 (100%) | ~20 days | 100% | 100% | 100% | 100% |

The full dataset with chunk mode at L=32768 was chosen for the production
run to avoid losing the pharmacologically important long classes.

#### Taxonomy distribution

| Domain | Records | % |
|--------|---------|---|
| Bacteria | 270,997 | 97.7% |
| Eukaryota | 3,471 | 1.3% |
| Archaea | 2,762 | 1.0% |
| Other (metagenome, unidentified, etc.) | 8 | <0.01% |

Within Bacteria, three phyla dominate:

| Phylum | Records | % of train |
|--------|---------|-----------|
| Pseudomonadota | 108,952 | 39.3% |
| Actinomycetota | 85,989 | 31.0% |
| Bacillota | 52,980 | 19.1% |
| All others | 29,317 | 10.6% |

These three phyla account for 89.4% of training data. Generated sequences
conditioned on underrepresented taxa (eukaryotic fungi, archaea, rare
bacterial phyla) may lack diversity or default to bacterial-like codon
usage. See risk B4 in §6.

#### Conditioning prefix structure

Each training record has the format:
`|COMPOUND_CLASS:{cls}|{tax_tag}{nucleotide_sequence}`

Prefix lengths (the conditioning portion before the nucleotide sequence):

| Statistic | Characters |
|-----------|-----------|
| Min       | 59        |
| Median    | 137       |
| Mean      | 135       |
| Max       | 207       |

Prefixes are short relative to sequences (median 137 chars vs median
23,075 total). These tokens are masked from loss computation (§4.2) so
the model sees them for conditioning but is not penalised for predicting them.

#### Data quality notes

**Non-standard nucleotides**: 4.6% of records contain N characters
(IUPAC ambiguity code for unknown base). These arise from sequencing gaps
in the source genomes. The Evo2 tokenizer handles N as a valid token.
The model will learn to produce N characters at low frequency, which is
biologically appropriate for draft-quality genome contexts but undesirable
for synthesis-ready output. Post-generation filtering may be needed.

**Contig-edge BGCs**: 32,760 records (11.8%) are flagged as
`contig_edge=True`, meaning the BGC sits at the boundary of a sequencing
contig and may be biologically incomplete (truncated by the assembly, not
by our processing). These are included in training under assumption A4
(§5) — excluding them would lose a significant data fraction and the
model can learn from partial BGCs. However, the model may learn to
generate sequences that "end abruptly" if it trains on enough truncated
examples.

**Near-empty records**: 5 records have training_text < 500 characters.
These are negligible (0.002% of data) and will have minimal impact on
training.

**Sequence-level duplicates**: 966 records (0.3%) share an identical
nucleotide sequence with at least one other record. However, 49,165
records (17.7%) share an accession with another record, meaning the same
genomic region was annotated as containing multiple overlapping BGCs.
These are not exact duplicates but the model sees very similar sequence
neighbourhoods multiple times, which could inflate apparent validation
performance on sequences from those regions. Full deduplication has not
been performed (see risk B7 in §6).

#### Implications for the current production run

None of the issues identified above require changes to the current
production run configuration. They are documented here as known
limitations to evaluate after training:

- **Class imbalance**: evaluate per-class generation quality via M1
  (antiSMASH classification accuracy). If rare classes underperform,
  consider class-weighted sampling or oversampling in Phase 2.
- **Taxonomy skew**: evaluate via M7 (codon adaptation index) across
  different conditioning taxa. If non-bacterial taxa produce bacterial-like
  output, the conditioning signal may not be effective for those taxa.
- **Duplicate accessions**: when evaluating val/test performance, check
  whether high-scoring generated sequences correlate with accessions that
  appear multiple times in training.
- **Contig-edge truncation**: inspect generated sequences for abrupt
  endings that lack proper termination signals.
- **N characters**: count N frequency in generated sequences; filter if
  above biological background rate.

### 4.1  Data loading and collation

**`BGCTextDataset`**: Reads JSONL files, extracts `training_text` field from
each record. Each training text is a string of the form:

```
|COMPOUND_CLASS:PKS||D__BACTERIA;P__...;S__...|ATGCGATCG...
```

The Evo2 tokeniser converts this to integer token IDs. Sequences longer than
`--max-seq-len` are handled by `--long-seq-strategy` (audit M10 — there is **no
centre-crop**; the earlier docs claiming one were wrong):
- `truncate` (legacy default): **head-truncation** — keeps the prefix + the
  first `max_seq_len` tokens and discards the tail (`ids = ids[:max_seq_len]`).
- `chunk` (production): **forward tiling** from nt 0 with overlap — every
  window starts at `nt_start` advancing forward, re-prepending the prefix; full
  nucleotide coverage, no centering. See §4.3.

**`collate_pad`**: The collation function pads variable-length sequences in a
batch to the length of the longest sequence in that batch ("natural
collation"). This means GPU memory varies batch-to-batch depending on the
sequences drawn. The `--smoke-pad-to-max-seq-len` flag forces padding to
`--max-seq-len` for worst-case memory testing.

### 4.2  Prefix loss masking (H3 hypothesis)

The conditioning prefix (everything before the nucleotide sequence) is
masked from the cross-entropy loss computation using `IGNORE_INDEX = -100`.
The model sees the prefix during the forward pass (it conditions generation)
but is not penalised for "predicting" the prefix tokens. This prevents the
model from wasting capacity learning to reproduce the conditioning format
and focuses all learning on the nucleotide sequence itself.

### 4.3  Chunked long-sequence handling

For sequences exceeding `--max-seq-len`, the `--long-seq-strategy chunk`
flag enables deterministic nucleotide tiling with:

- Prefix-aware windowing (conditioning prefix is prepended to each chunk).
- Configurable overlap (`--chunk-overlap 2048`) to avoid hard boundaries.
- Deterministic chunking (same sequence always produces the same chunks).

### 4.4  LoRA application

The `apply_lora()` function uses peft's `get_peft_model()` with three
compatibility fixes for the Evo2/peft/torch version combination:

1. **dotdict `to_dict` injection** — Evo2's config is a `vortex.dotdict`
   that returns `None` for missing keys instead of raising `AttributeError`.
   A working `to_dict` lambda is injected before peft tries to call it.

2. **`autocast_adapter_dtype=False`** — peft 0.19 tries to detect
   `torch.float8_e8m0fnu`, which doesn't exist in torch 2.5.1.

3. **Inference tensor cloning** — Evo2 loads tensors in inference mode
   (`inference_mode: True` in vortex config). These must be cloned to
   regular tensors before autograd can use them in the backward pass.
   40 tensors are fixed (10 at world_size=1).

### 4.5  Activation checkpointing

`enable_block_activation_checkpointing()` wraps each StripedHyena block's
forward method with `torch.utils.checkpoint.checkpoint()`. This trades
compute for memory: each block's activations are recomputed during the
backward pass instead of being stored (~1.33× wall-clock overhead, already
baked into measured throughput numbers).

`use_reentrant=False` is required because LoRA dropout creates
non-deterministic paths through the computation graph that are incompatible
with reentrant checkpointing.

### 4.6  DeepSpeed configuration

Built by `build_ds_config()`:

- ZeRO Stage 2 with `exclude_frozen_parameters=True`.
- bf16 mixed precision.
- AdamW optimizer with beta1=0.9, beta2=0.95 (from Evo2 paper).
- WarmupCosineLR scheduler (warmup_steps=200, cosine decay to 10% of peak).
- Gradient clipping at 1.0.

### 4.7  How LoRA works and what training produces

This section explains the LoRA fine-tuning approach from the ground up:
what it is, why we use it, what it does to the model, and what files it
produces on disk.

#### 4.7.1  The problem: why we can't just train the whole model

Evo2 7B has 6.51 billion parameters. These parameters are stored as
numbers (weights) organised into matrices inside each layer of the model.
During normal "full" fine-tuning, you would adjust all 6.51 billion of
these numbers to teach the model about BGC sequences.

The problem is memory. To adjust a parameter during training, you need to
store not just the parameter itself, but also:

- **Its gradient** — which direction to adjust it (same size as the param)
- **Two AdamW optimizer states** — running averages that make the
  adjustments smoother (2x the size, in higher-precision fp32 format)

That adds up to:

```
Model weights (bf16):              14 GB
Gradients (bf16):                  14 GB
AdamW optimizer states (fp32):     56 GB   ← this is the killer
────────────────────────────────────────
Total before activations:          84 GB   > 80 GB GPU limit
```

84 GB exceeds our 80 GB GPU, and we haven't even counted the memory
needed for the actual computation (activations). Full fine-tuning
physically cannot run on our hardware.

#### 4.7.2  The LoRA solution: train a small overlay instead

**LoRA** (Low-Rank Adaptation) takes a different approach. Instead of
modifying the existing 6.51 billion parameters, it:

1. **Freezes** all original parameters (they become read-only)
2. **Adds** a small number of new, trainable parameters alongside them
3. Trains **only** the new parameters

The key insight is how those new parameters are structured. Every layer
in a neural network has weight matrices — for example, a matrix with
4096 rows and 4096 columns (16.7 million numbers). LoRA adds two much
smaller matrices next to it:

```
Original weight matrix W:     4096 x 4096  =  16,777,216 parameters (FROZEN)

LoRA adapter:
  Matrix A (down-project):    4096 x 16    =      65,536 parameters (TRAINED)
  Matrix B (up-project):        16 x 4096  =      65,536 parameters (TRAINED)
                                              ──────────
                                                 131,072 parameters
                                              (0.78% of the original matrix)
```

The "16" is the LoRA rank (`--lora-r 16`). It controls how expressive
the adaptation is. Higher rank = more trainable parameters = more
capacity to learn, but also more memory and risk of overfitting.

During a forward pass, the output of this layer becomes:

```
output = input x W  +  input x A x B
         ─────────     ──────────────
         original       LoRA adjustment
         (frozen)       (trained)
```

The adjustment `A x B` produces a matrix the same shape as `W`
(4096 x 4096), but it's constructed from only 131,072 trainable
numbers instead of 16.7 million. The "low-rank" name comes from
this factorisation — rank 16 is much lower than rank 4096.

#### 4.7.3  What this means for memory

With LoRA, the memory budget becomes:

```
Model weights (bf16):              14 GB   (still loaded, but frozen)
LoRA adapter weights (bf16):        0.05 GB (28.7M params — tiny)
LoRA gradients (bf16):              0.05 GB
LoRA AdamW states (fp32):           0.34 GB
────────────────────────────────────────────
Total before activations:          14.4 GB  (vs 84 GB for full fine-tune)
```

That leaves ~65 GB for activations (the intermediate computation results),
which is why we can train at L=32768 with activation checkpointing.

#### 4.7.4  Which layers get adapters

Not every part of the model gets a LoRA adapter. Our configuration
targets all 133 `nn.Linear` (fully connected) layers:

| Component | Layers | LoRA params | What it does |
|-----------|-------:|------------:|--------------|
| Attention Wqkv | 5 blocks | ~0.8M | How the model attends to different positions |
| Attention out_proj | 5 blocks | ~0.8M | How attention results are projected back |
| Hyena out_filter_dense | 27 blocks | ~3.7M | How the long-convolution filter output is mixed |
| MLP l1, l2, l3 | 32 blocks | ~23.2M | The feed-forward "thinking" layers |
| **Total** | **133 layers** | **28.7M** | **0.44% of 6.51B** |

The MLP layers dominate because they are the largest matrices in the model
and there are three per block (l1, l2, l3) across all 32 blocks.

#### 4.7.5  The two-piece model: base + adapter

The fine-tuned model is **two separate files** that get combined at
inference time:

**Piece 1 — Evo2 7B base model (~14 GB, never modified):**

The pretrained foundation model from Arc Institute. It lives permanently
in the HF cache:
```
/data2/ds85/hf_cache/hub/models--arcinstitute--evo2_7b_262k/
```

This file is read-only. Our training process never writes to it. Anyone
using Evo2 anywhere in the world has this exact same file. Think of it as
the "operating system" that knows how to read DNA — it doesn't know
anything about BGCs specifically, but it understands the grammar of
nucleotide sequences in general.

**Piece 2 — LoRA adapter (~120 MB, this is our training output):**

A directory containing the trained A and B matrices for all 133 target
layers. This is the entirety of what our multi-day training run produces.
It encodes everything the model learned about BGC sequences, compound
class conditioning, and chassis-organism adaptation.

Think of it as a "plugin" or "mod" — it cannot function alone (it needs
the base model to plug into), but it transforms the base model's
behaviour from "generic DNA language model" to "BGC-generating specialist."

**To use the fine-tuned model**, you load both pieces and stack them:

```python
from evo2 import Evo2
from peft import PeftModel

# 1. Load unchanged base model (14 GB, from HF cache — takes ~5 sec)
model = Evo2("evo2_7b_262k")

# 2. Apply our trained adapter on top (120 MB — instant)
model = PeftModel.from_pretrained(
    model,
    "/data2/ds85/bgcmodel_runs/phase1_lora_prod_.../final_adapter/"
)

# 3. Now this model generates BGC sequences with our fine-tuning
#    Give it a conditioning prefix and let it generate nucleotides:
prefix = "|COMPOUND_CLASS:NRPS||D__BACTERIA;P__PSEUDOMONADOTA;...;S__ESCHERICHIA|"
generated_sequence = model.generate(prefix, max_length=32768)
```

**Why this two-piece design is useful:**

- **Disk efficient:** Checkpoints are ~120 MB, not ~14 GB. A full
  training run with checkpoints every 500 steps produces ~1 GB of
  adapters, not ~120 GB of full model copies.
- **Shareable:** To share the fine-tuned model, you only distribute the
  120 MB adapter. The recipient downloads Evo2 7B themselves (it's public).
- **Composable:** You could train multiple adapters (one per compound
  class, for example) and swap them on the same base model.
- **Reversible:** The base model is never altered. If the fine-tuning
  fails, you just discard the adapter and still have the original.

#### 4.7.6  On-disk layout of a completed production run

```
/data2/ds85/bgcmodel_runs/phase1_lora_prod_<TS>_L32768/
│
├── final_adapter/                 ← THE FINISHED PRODUCT
│   ├── adapter_config.json        #   LoRA config (rank, alpha, target modules)
│   ├── adapter_model.safetensors  #   The trained A and B matrices (~120 MB)
│   └── README.md                  #   Auto-generated by peft
│
├── checkpoints/                   ← SNAPSHOTS DURING TRAINING
│   ├── step_500/
│   │   ├── adapter/               #   LoRA weights at step 500
│   │   └── mp_rank_0_model_states.pt  # DeepSpeed optimizer state (for resume)
│   ├── step_1000/
│   │   └── ...
│   ├── step_1500/
│   │   └── ...
│   └── best/                      ← BEST CHECKPOINT (lowest val loss)
│       └── adapter/
│
├── config.json                    # All hyperparameters (for reproducibility)
├── deepspeed_config.json          # DeepSpeed settings used
├── data_fingerprint.json          # SHA256 of training data (tamper detection)
├── env.txt                        # Full pip freeze (exact package versions)
├── train_log.jsonl                # Per-step metrics (loss, lr, memory, throughput)
├── val_log.jsonl                  # Validation metrics (val_loss, val_ppl)
└── production.log                 # Full stdout/stderr
```

**Which file do you actually use for generation?**

- **`final_adapter/`** — the model at the very end of training (step ~4,330)
- **`checkpoints/best/adapter/`** — the model at whatever step had the
  lowest validation loss

Usually you want `best/` — if the model starts overfitting near the end
of training (memorising the training data instead of generalising), the
final adapter will perform worse than the best mid-training checkpoint.
The training script tracks this automatically.

#### 4.7.7  Lifecycle summary

```
TRAINING PHASE:
  Evo2 base (14 GB, read-only)
       │
       ▼
  [Load into GPU memory]
       │
       ▼
  [Attach 133 LoRA adapters]──→ 28.7M new trainable parameters
       │
       ▼
  [Train for ~4,330 steps]
       │    ├── Every 500 steps: save adapter/ snapshot (~120 MB)
       │    └── Track best validation loss → save best/ adapter
       │
       ▼
  [Save final_adapter/] ──→ 120 MB file = the entire training output

INFERENCE PHASE:
  Evo2 base (14 GB, same read-only file)
       │
       ▼
  [Load into GPU memory]
       │
       ▼
  [Load final_adapter/ on top]
       │
       ▼
  [Generate BGC sequences]
       │
       ▼
  |COMPOUND_CLASS:NRPS|...|ATGCGATCG... (novel BGC nucleotide sequence)
```

### 4.8  Checkpoint save/load (mechanics)

**Saving** (`save_lora_checkpoint`):
- Saves LoRA adapter weights to `checkpoints/step_N/adapter/` via
  `model.save_pretrained()`.
- DeepSpeed handles optimizer/scheduler state in ZeRO partition files.
- Client state includes step counter and best val loss.

**Loading** (`load_lora_checkpoint`):
- Reloads adapter via `PeftModel.from_pretrained()`.
- Restores DeepSpeed optimizer/scheduler state.
- Restores step counter for faithful mid-epoch resume.

### 4.8  Faithful mid-epoch resume (H1)

When resuming from a checkpoint, the dataloader fast-forwards through
already-seen batches (skip-ahead) and restores the RNG state
(`gather_rng_state` / `set_rng_state`) so that training continues
identically to an uninterrupted run. This is critical for multi-day runs
on a shared GPU where interruptions are expected.

### 4.9  Training loop

Standard autoregressive language model training:

1. For each batch: tokenise → collate → forward pass → causal LM loss
   (with prefix masking) → backward pass → gradient accumulation.
2. Every `--grad-accum` micro-batches: optimizer step + LR scheduler step.
3. Every `--log-every` steps: write to `train_log.jsonl` and WandB.
4. Every `--val-every` steps: run validation on sampled val sequences.
5. Every `--save-every` steps: save checkpoint.

### 4.10  Default hyperparameters

| Parameter | Script default | Recommended gputee override |
|-----------|---------------|-----------------------------|
| max_seq_len | 32768 | — (keep) |
| batch_size | 4 | **1** (see §7 critical issue) |
| grad_accum | 8 | **128** (to maintain effective batch of 128) |
| lr | 5e-5 | — |
| warmup_steps | 200 | — |
| max_epochs | 2 | — |
| val_every | 250 | — |
| save_every | 500 | — |
| seed | 42 | — |

---

## 5  Assumptions Baked Into the Codebase

### A1: Long-sequence handling (CORRECTED — audit M10)

There is **no centre-crop** anywhere in the code; earlier versions of this
document wrongly claimed one. Actual behaviour (see §4.1):
- **Production uses `--long-seq-strategy chunk`** — forward tiling with overlap
  gives **full nucleotide coverage** of every BGC, so nothing is discarded; the
  "centre-crop preserves core genes" assumption is moot.
- **Truncate mode** (legacy default) does **head-truncation** — keeps the first
  `max_seq_len` tokens and discards the tail. It does not centre or preserve the
  middle. Not used in production.

The remaining open question for chunk mode is M11 (interior windows carry the
full class/taxon prefix though their local sequence may not justify it).

### A2: Validation loss is representative of generation (UPDATED — audit M2)

Validation now computes loss on the **first window only** (prefix-aligned start,
nt_start==0) of each held-out BGC — the same regime as inference — and reports
it length-stratified (`val_by_length`). This replaced the old interior-window
val loss, which did not reflect generation. A *generation-based* eval (generate
from held-out prompts → score with the eval suite) is still offline and depends
on the not-yet-built generation script (C3).

### A3: Phase 1 class-only conditioning is sufficient for first validation

The COMPOUND token is dropped in Phase 1 to maximise data uniformity
(346,559 records, one format). The assumption is that class-level
conditioning is enough to demonstrate the model can generate
architecturally correct BGCs. This is a deliberate simplification — if
class-only conditioning fails, the project has a fundamental problem.

### A4: contig_edge BGCs are safe to include in training

11.9% of antiSMASH records touch a contig boundary and may be truncated.
The assumption is that core biosynthetic logic is typically intact and
Evo2 tolerates partial context. These are included in v1 training; the
plan is to filter them only if generated sequences show pathological
early-termination patterns.

### A5: E. coli codon table is the right chassis reference

The CAI calculation in `evaluation.py` uses a hardcoded *E. coli* K-12
codon frequency table. All organism-compatibility metrics (M7) are
calibrated against *E. coli*. This is appropriate for the three target
compounds (all expressible in *E. coli*) but would need to be
parameterised for other chassis organisms.

### A6: Single-GPU training is sufficient

The project uses a single H100 with DeepSpeed at world_size=1. This is
adequate for LoRA fine-tuning but means:

- No data parallelism — training time scales linearly with dataset size.
- No model parallelism — the full 14 GB model must fit on one GPU alongside
  activations.
- Estimated wall-clock: ~2.7 days at L=32768, ~5.3 days at L=65536.

### A7: Effective batch size of 128 sequences is correct

The 128-sequence effective batch (bs=1 × ga=128 on gputee) was chosen to
match the original trojai configuration (bs=4 × 4 GPUs × ga=8). The
learning rate, weight decay, and warmup were tuned against this effective
batch size. Changing it without retuning would alter optimisation dynamics.

---

## 6  Blind Spots and Risks

### B1: Train/inference length mismatch

Evo2 supports 262k at inference, so the fine-tuned model *can* be asked to
generate full-length 50–150 kb BGCs. It would do so having only been
fine-tuned on ≤32k windows. The interaction between this mismatch and the
evaluation suite is not discussed in the current docs.

### B2: No long-context evaluation defined

There is no held-out evaluation set specifically for sequences >32k. Val
loss is computed on the same 32k crop as training. If the model fails to
generalise beyond the training window, current metrics won't detect it.

### B3: Class imbalance in training data

TERPENE and RIPP together account for ~49% of training data (driven by
antiSMASH DB). PKS and NRPS are better represented in MIBiG (33%+28%) but
are only 8%+12% in antiSMASH. The model may develop stronger chassis-
adaptation capability for TERPENE/RIPP than for PKS/NRPS.

### B4: Taxonomy distribution skew

97.9% of antiSMASH records are BACTERIA. EUKARYOTA (1.1%) and ARCHAEA
(1.0%) are severely underrepresented. *Streptomyces* alone accounts for
56,280 antiSMASH records (16.4%). Generated sequences conditioned on
underrepresented taxa may lack diversity.

### B5: Shared GPU contention

gputee is a shared machine with no GPU reservation mechanism. The queued
pilot has been waiting since ~2026-05-20 for the GPU to free up. Multi-day
runs are vulnerable to preemption — if another user starts a GPU process
mid-training, the training job will OOM. The queue script gates on startup
but does not protect against mid-run contention.

### B6: Version fragility

The software stack is pinned to specific versions due to a chain of
incompatibilities:

- torch 2.5.1 (2.6 breaks flash-attn ABI)
- transformers 4.46.3 (5.x blocks .pt loading on torch<2.6)
- peft 0.19.0 (with 3 manual compatibility fixes)

Any version upgrade risks cascading breakage. The stack works but is
frozen at April 2026 versions.

### B7: Deduplication not characterised

The `BGC_Research_Plan.md` §4.4 deduplication statistics table is marked
as TBD. The extent of near-duplicate sequences in the combined dataset
(particularly within antiSMASH DB, which contains multiple strains of the
same species) has not been quantified. This could affect both training
dynamics and evaluation metrics.

### B8: No automated end-to-end test

There is no CI pipeline or automated test that validates the full
data→training→generation→evaluation pipeline. Bugs are caught via manual
smoke tests and documented in the guides.

---

## 7  Current Status — Where Things Stand Right Now

### 7.1  What is complete

- ✅ Data pipeline: MIBiG + antiSMASH DB → JSONL → stratified splits
- ✅ Training script: LoRA fine-tuning with DeepSpeed, activation
  checkpointing, prefix masking, chunked long sequences, faithful resume
- ✅ Evaluation suite: 8 metrics implemented and validated
- ✅ Environment: fully built and verified on gputee
- ✅ Memory characterisation: comprehensive smoke sweeps with and without
  activation checkpointing, production-like preflight at multiple L values
- ✅ Resume-from-checkpoint: 4 bugs fixed and tested
- ✅ Documentation: three comprehensive guides (PROJECT_GUIDE, FINETUNE_GUIDE,
  BGC_Research_Plan)

### 7.2  What is in progress

**A pilot run is queued in the `bgcruns` tmux session**, waiting for the
GPU to become free. It has been waiting since approximately 2026-05-20.

### 7.3  ⚠️ CRITICAL ISSUE: Queued pilot will OOM

**The currently queued pilot run will almost certainly fail with an
out-of-memory error when the GPU becomes free.**

The `queue_h100_pilot.sh` script defaults to:
- `--batch-size 4` (line 47)
- `--grad-accum 32` (line 48)

However, the FINETUNE_GUIDE.md was updated on 2026-05-14 to document that
**batch_size=4 OOMs at L=32768 even on a clean 80 GB GPU**. The production
launch template was changed to require:
- `--batch-size 1`
- `--grad-accum 128`

The queue script was written before this discovery and was never updated.
The pilot run was launched with:

```bash
scripts/queue_h100_pilot.sh --max-steps 0 --val-every 250 --save-every 500
```

This inherits the default bs=4 ga=32 and will OOM.

**To fix:** Either update the script defaults or re-launch with explicit
overrides:

```bash
scripts/queue_h100_pilot.sh --batch-size 1 --grad-accum 128 \
  --max-steps 20 --val-every 10 --save-every 10
```

(Note: `--max-steps 0` also means "run forever" which is likely not the
intended pilot behaviour — the original plan was 20 steps.)

### 7.4  What has NOT been done

- ❌ No training run has completed (not even a pilot)
- ❌ No model checkpoint exists
- ❌ No sequences have been generated
- ❌ No evaluation has been run on generated sequences
- ❌ No WandB dashboard has live training metrics
- ❌ Deduplication audit not performed
- ❌ BiG-SCAPE metric (M6) not tested end-to-end
- ❌ No wet-lab collaboration established

### 7.5  Blocking chain

```
Fix queue script (bs=1, ga=128)
  → GPU becomes available
    → Pilot run completes (20 steps, ~30 min)
      → Verify pilot artefacts
        → Launch full production run (~2.7 days at L=32768)
          → Generate sequences from best checkpoint
            → Run 8-metric evaluation
              → Iterate / proceed to Phase 2
```

### 7.6  Time estimates

| Task | Estimated wall-clock |
|------|---------------------|
| Pilot run (20 steps, L=32768) | ~30–60 minutes |
| Full production run (2 epochs, L=32768) | ~2.7 days |
| Full production run (2 epochs, L=65536) | ~5.3 days |
| Sequence generation (after training) | ~hours |
| Full 8-metric evaluation | ~hours–day |

---

## 8  Action Plan

### Immediate (do now)

1. **Fix the queued pilot.** Kill the current tmux process. Update
   `queue_h100_pilot.sh` to default to `--batch-size 1 --grad-accum 128`.
   Re-launch the pilot with `--max-steps 20 --val-every 10 --save-every 10`.

2. **Verify the pilot** when it completes. The post-run verification in
   the script checks: train_log.jsonl (steps reached, memory envelope,
   throughput, natural collation), val_log.jsonl (validation cadence),
   config.json, checkpoints (size, adapter presence), loss trajectory.

### Short-term (after pilot passes)

3. **Launch the full production run.** Use FINETUNE_GUIDE Template A:
   ```
   --max-seq-len 32768 --batch-size 1 --grad-accum 128
   --val-every 250 --save-every 500 --max-epochs 2
   --wandb-project bcg-evo2-phase1 --wandb-mode online
   ```
   Expected duration: ~2.7 days. Expected steps: ~4,332.

4. **Monitor training.** Watch for: loss trajectory matching expected
   pattern (§8 of FINETUNE_GUIDE), val loss convergence, memory stability,
   no NaN grad norms. Use WandB dashboard and `tail -f train_log.jsonl`.

5. **Handle interruptions.** If the GPU is preempted mid-training, resume
   from the latest checkpoint using `--resume-from`. The faithful resume
   mechanism (H1) handles dataloader skip-ahead and RNG state restoration.

### Medium-term (after training completes)

6. **Generate BGC sequences.** Condition on each target class + *E. coli*
   taxonomy tag. Generate multiple candidates per class.

7. **Run the 8-metric evaluation suite.** Prioritise M1 (antiSMASH class
   match) and M2 (domain recovery) as the primary quality signals. M7
   (organism compatibility) validates chassis adaptation. GPU metrics
   (M3 ESMFold, M5 Evo2 perplexity) require the GPU to be free.

8. **Decide on L=65536.** If the L=32768 model shows strong results and
   wall-clock is acceptable, consider a stretch run at L=65536 to cover
   more of the long-tail sequences (83% → 97.8% coverage at full length).

### Longer-term

9. **Phase 2 conditioning.** Add compound-level tokens for the ~45
   well-represented MIBiG compounds. Test whether conditioning on
   `carotenoid` vs `ectoine` produces architecturally distinct outputs.

10. **Characterise deduplication.** Quantify near-duplicate sequences in
    the training set. Consider deduplication or downsampling of
    over-represented genera (*Streptomyces*, *Pseudomonas*, *Bacillus*).

11. **Wet-lab validation.** Identify a collaborator for synthesis and
    expression of top-scoring generated BGCs in *E. coli*.

---

## Appendix A: Key File Reference

| File | Role | Lines |
|------|------|------:|
| `scripts/finetune_evo2_lora.py` | Training script | ~2000 |
| `docs/gputee/FINETUNE_GUIDE.md` | Hardware + training runbook | ~1000 |
| `docs/gputee/PROJECT_GUIDE.md` | Living project status | ~1400 |
| `docs/gputee/BGC_Research_Plan.md` | Full research plan v9 | ~500 |
| `src/bgc_pipeline/evaluation.py` | 8-metric evaluation suite | ~300 |
| `config/compound_class_map.yaml` | 91 product types → tokens | ~127 |
| `scripts/queue_h100_pilot.sh` | Pilot queue script | ~391 |
| `scripts/queue_h100_smoke.sh` | Smoke sweep queue script | ~300 |

## Appendix B: Git Commit History

```
6687cec  fix resume-from-checkpoint (4 bugs) + add pilot queue script
576bf48  docs(gputee): align research plan and guides with current stack
a26904d  docs(gputee): refresh post-preflight + retarget §13 NEXT to L=32k pilot
8a30781  running preflight checks
5f476fc  Split docs into trojai/ and gputee/; port guides to H100
9d6ebc1  add Evo2 fine tuning pipeline with full and LoRA training scripts
3ebf134  antismash added. Antismash + mibig finetuning inputs generated
34188c8  added MiBIG data
132b2ce  Initial commit
```

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **BGC** | Biosynthetic Gene Cluster — contiguous DNA encoding a complete natural product pathway |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning that adds small trainable matrices to frozen model layers |
| **StripedHyena** | Evo2's architecture combining Hyena long-convolution operators with multi-head attention |
| **ZeRO-2** | DeepSpeed's optimizer + gradient sharding strategy for distributed training |
| **AC** | Activation Checkpointing — trading compute for memory by recomputing activations during backpropagation |
| **MIBiG** | Minimum Information about a Biosynthetic Gene cluster — curated database of ~3,000 characterised BGCs |
| **antiSMASH** | antibiotic & Secondary Metabolite Analysis Shell — tool for identifying BGCs in genomes |
| **antiSMASH DB v5** | Database of ~497,000 predicted BGCs from 56,846 genomes annotated by antiSMASH |
| **Pfam** | Protein family database; HMM profiles used for domain annotation |
| **CAI** | Codon Adaptation Index — measure of codon usage bias relative to a reference organism |
| **ESMFold** | Protein structure prediction model (used for foldability metric M3) |
| **BiG-SCAPE** | Biosynthetic Gene Similarity Clustering and Prospecting Engine (structural novelty metric M6) |
| **Chassis organism** | The host organism intended to express the generated BGC (default: *E. coli*) |
| **Transposition** | Adapting a known BGC architecture for expression in a different organism |
