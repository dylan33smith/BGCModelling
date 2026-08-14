# BCGModelling

Fine-tune a genome foundation model to generate novel, correctly-classified
**biosynthetic gene cluster (BGC)** nucleotide sequences conditioned on biosynthetic
**class** and taxonomic **lineage** (**Phase 1**, closed). **Phase 2** (2026-08-12, closed) tested
objective changes on the 1B track. **Phase 3** (opened 2026-08-14) narrows to one small compound
class at a time; a **compound**-conditioned FT for named-product design remains the eventual goal.

Three model tracks share one dataset and one eval instrument:

| Track | Model | Status |
|---|---|---|
| [`evo2/`](evo2/) | Evo2 7B + LoRA | incumbent; **every inference-time lever that edits the input or the activations is closed** — prefix labels (2026-07-21), CFG (2026-07-22), activation steering (2026-08-10) — and the cheap end of training-time coupling too (soft prefixes, 2026-08-10); **guided decoding is underpowered, not null** (Q2 5–0, p=0.0625, effective n=5) |
| [`evo2_1b/`](evo2_1b/) | Evo2 1B (`evo2_1b_base`) + LoRA | **THE TESTING SUBSTRATE for Phase 3.** Phase 2 (objective arms) closed here 2026-08-14; requires Transformer Engine 1.13.0 |
| [`genomeocean/`](genomeocean/) | GenomeOcean-4B / `bgcFM` | live but HELD — **leakage gate passed 2026-08-14** (0.0000 containment, greedy); fits 64% of BGC regions whole vs Evo2's 0%; takes a real trainable class token |

See [`docs/model_comparison_evo2_vs_genomeocean.md`](docs/model_comparison_evo2_vs_genomeocean.md)
for the head-to-head and the recommendation.

This README is the single current-state entry point. Deep operational runbooks and
dated audit records are preserved under [`docs/archive/`](docs/archive/); ongoing
working memory lives in [`docs/project_memory/`](docs/project_memory/).

---

## Current status (snapshot, 2026-08-14)

- **▶ PHASE 3 IS OPEN (2026-08-14) — one small class at a time, target RIPP.** Phase 2 closed
  the objective and training-budget levers on the general problem. A single small class **deletes**
  the long-context problem (RIPP median 1,931 nt vs the 1B's 8,192) and a per-class LoRA means the
  model never reads a class label, retiring every Phase-1 closure. Per-class datasets are split
  **from scratch** at `/data2/ds85/bgcmodel_data/splits_class/`.
  ⚠️ **ECTOINE looked ideal (396 nt, 2,492 records) and is disqualified — 85% of its held-out
  clusters are near-duplicates of training ones.** Length and diversity are anti-correlated across
  these classes. **RIPP is the target**: 8,129 train / 579 held out, 1,931 nt median (89% under
  8 kb), **43% near-dup loss — the most diverse** — and twice TERPENE's de novo detection.
  **Pre-registered before any model is trained:**
  [`docs/phase3_preregistration.md`](docs/phase3_preregistration.md).
  Informed by [Hie et al., *Science* 2026](https://www.science.org/doi/10.1126/science.aec2657),
  who ran the same single-family strategy for phages: consensus-sequence seeding, **4–8 nt seeds
  because longer ones caused memorisation**, and ~1000:1 overgeneration-and-filtering.
  **Substrate policy:** the **1B is the testing substrate**, the 7B confirms publishable claims, and
  GenomeOcean is available (leakage gate passed) but held so method is not confounded with model.

- **PHASE 2 (2026-08-12) — CLOSED. The 1B track.** `evo2_1b/` holds a clean, fast track for the
  objective-change experiments (frame-aware and domain-weighted loss) on `evo2_1b_base`:
  **0.990 nats/base vs 0.859 for the 7B base, and 3.34× the throughput** (8,770 vs 2,625 tok/s —
  a depth/width win only, since both models are byte-level). **Transformer Engine 1.13.0 is
  required**; without it the 1B loads and sits at chance. Everything model-agnostic is reused, not
  copied. See [`evo2_1b/README.md`](evo2_1b/README.md).

- **v2 LoRA is trained and stopped at `step_1200`** (run dir
  `/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768`, `L=32768`,
  `bs=1 ga=128`). It is the checkpoint every downstream experiment uses.

- **⚠️ REFRAMED 2026-08-12: class conditioning was never the binding constraint.**
  Decomposing `correct_class = P(detect) × P(right class | detect)` on the same adapter:

  | regime | n | P(detect) | 95% CI | P(right \| detect) |
  |---|---|---|---|---|
  | de novo (unseeded) | 81 | **0.012** | [0.000, 0.067] | 1 detection — unestimable |
  | seeded | 120 | **0.367** | [0.281, 0.459] | **0.932** |

  The seed multiplies detection **30×**. Seeded, class is already 0.932 — there is ~7% for a
  conditioning mechanism to win. De novo, there is nothing to install a class into. The whole
  conditioning programme was aimed at the smaller of two problems.

- **The real failure is capability, and it is specific.** Re-scored with permissive instruments
  (`evo2/scripts/soft_instrument_probe.py`) — no clustering required, just "does any single
  class-defining Pfam domain appear anywhere":

  | group | coding density | longest ORF | ≥1 class domain |
  |---|---|---|---|
  | real cores @3 kb | 0.972 | **702 aa** | **0.800** |
  | seeded @3 kb | 0.932 | 591 aa | 0.467 |
  | de novo @6 kb | 0.743 | 505 aa | **0.033** |
  | de novo @2 kb | 0.815 | 332 aa | **0.050** |

  Two instruments at very different strictness agree, so it is not an evaluation artifact.
  De novo output is *not* junk — and the first reading of this table ("the model cannot sustain a
  reading frame long enough to encode a module") **did not survive the ladder audit**. **100% of
  6 kb de novo generations hit some Pfam family: the model writes REAL protein, of the WRONG KIND.**
  Biosynthetic fraction 0.100 vs 0.836; `bio_span_frac` 0.051 vs 0.876. `max_orf_aa` is DEMOTED
  (AUROC 0.709; r = 0.051 / −0.120 *within* de novo generations — it does not track domain content
  where it would have to).
  ⇒ Track the **validated ladder**: **`best_bio_bits`** (PRIMARY, AUROC 0.950) → `n_bio_domains`
  (0.919) → `bio_span_frac` (0.896) → antiSMASH detect → class, with `biosynthetic_fraction` as a
  specificity diagnostic and `max_orf_aa` as a structural one — **all under a hard novelty
  constraint**, since every rung is maximised by copying training data. `correct_class` has read
  ~0 **de novo** since the project began (0.283–0.40 seeded).

- **Why the label was always inert, quantified** (`evo2/scripts/context_ablation.py`). Scoring the
  same 500 bases while varying preceding context: 10 nt already yields 73% of everything the model
  achieves (0.977 nats vs 1.386 uniform), and 1,000 → 6,000 nt buys 0.005. All long-range context
  is worth **0.149 nats**. Against that, **right-vs-wrong class tag = −0.0006 nats** (−0.0000 with
  the tag 200 nt away). Using the tag never reduced the loss, so nothing ever built a pathway to
  read it. This unifies the inert label, the absent CFG signal and the 0.003-nat soft prefix.

- **Inference-time conditioning that edits the input or the activations is closed, and the last
  closure is a positive demonstration.** Labels, CFG, steering (every layer/dose/recipe), affine
  concept editing, and cross-class activation transplants — plus the cheap end of *training-time*
  coupling (per-class soft prefixes: 65k trained parameters, input only). **The one arm not closed
  is discriminator-guided decoding**: selection works (guide score best−random +5.71, 39/40), but
  Q2 is UNDERPOWERED rather than null — 5–0 paired, p=0.0625, effective n=5 because 35 of 40 seeds
  returned identical outcomes in every arm. Patching showed the model **does** read mid-layer state — a
  real donor moves its behaviour 92% — while carrying the donor's class **0/48**. So the channel
  works; class is not what travels down it.

- **What does work today:** *exemplar-conditioned* generation. Seed a real core and the
  continuation is correct-class **0.283 vs a 0.067 floor**, memorization ruled out, all four
  pre-registered controls passed. The detection numbers explain the mechanism (the seed supplies
  the recognisability the model cannot generate), and this is the mode Evo's own published work
  validates experimentally.

- **Eval suite** is named **checks → questions**, antiSMASH the gold-standard gate, calibrated at
  **both** ends — negative control (false-positive rate 0.000 for `is_bgc`) and positive control —
  plus a **continuous** `class_probe` that never gates. On real cores at 3 kb, **31.4% of antiSMASH
  detections are off-class**, so `correct_class` genuinely discriminates; a high concordance with
  `is_bgc` in our generations is a fact about the generations, not the ruler.

- The live, detailed state is in **[`docs/project_memory/progress.md`](docs/project_memory/progress.md)**
  — read that first when resuming work. The steering program's full arc is in
  [`docs/steering_program.md`](docs/steering_program.md); the ranked conditioning list —
  superseded at the top level, citations still accurate — in
  [`docs/conditioning_next_steps.md`](docs/conditioning_next_steps.md).

**Retraction, 2026-08-11 (same day it was made).** A claim that the seeded readout was confounded —
that antiSMASH was recognising the seed rather than the generation — was **wrong**. Both generators
score the continuation only; across every seeded run ever produced, **0 of 1512 stored sequences
contain their seed**. Nothing needed rerunning and the 0.283 result is *cleaner* than the
retraction implied. Pinned by `tests/test_scored_span.py`. *Lesson recorded: a concordance rate is
meaningless without the same rate on a control, and a premise handed to a verifier is not verified
by it.*

**Leakage debt — CLEARED 2026-08-10.** The class probe and the steering directions had been fit
on **val+test** and applied to val/test-seeded generations. Both are now refit **train-only**
(`acts_v2_train500.npz`, provenance-verified; directions at 9 layers in
`trainonly.steerdirs.npz`, probe cached at `acts_v2_train500.probe_L16_s0.joblib`), and
`_fit_probe` **refuses** a non-train fit set —
every activation cache carries a `.provenance.json` and the guard is tested in both directions.
Clearing it cost one published finding: see the retraction above.

---

## Project memory protocol

Working knowledge is split into modular files under `docs/project_memory/`:

| File | Contents |
|------|----------|
| [`progress.md`](docs/project_memory/progress.md) | Exact state of the research when last stepped away + next actions. |
| [`decisions.md`](docs/project_memory/decisions.md) | Architecture/approach decisions and **why** (LoRA, strict cores, eval design, …). |
| [`bugs.md`](docs/project_memory/bugs.md) | Quirks, recurring errors, and the proven fixes. |

See the **Memory Protocol** section in [`CLAUDE.md`](CLAUDE.md): read `progress.md`
before starting a task; update these files after solving a major bug or making a
structural decision.

---

## Repository layout

Reorganized 2026-07-27 into **shared root + one folder per model track**. Anything
model-agnostic (dataset pipeline, eval suite, class map, tests) stays at the root so
both tracks are scored on the same instrument.

```
# ---- SHARED ----
src/bgc_pipeline/evaluation.py   # the eval suite (CHECKS → QUESTIONS); see Evaluation
src/bgc_pipeline/class_map.py    # load the antiSMASH-product → compound-class map
src/bgc_pipeline/objective.py    # domain-weighted + frame-aware training objective (model-agnostic)
src/bgc_pipeline/annotations.py  # window/prefix-offset alignment for the above
scripts/eval_suite_driver.py     # batch eval: gen vs positive control, --skip-checks
scripts/evaluate_bgc.py          # single-sequence eval
scripts/memorization_check.py    # k-mer novelty vs a reference corpus
scripts/build_class_map.py       # regenerate config/compound_class_map.yaml (antiSMASH 8)
scripts/build_core_records.py    # extract strict cores from antiSMASH-DB GBKs
scripts/{split_dataset_grouped,curate_dataset,dedup_core_splits,exclude_mibig_from_core}.py
scripts/{derive_class_markers,validate_m2_calibration}.py        # class-marker calibration
scripts/{calibrate_antismash,validate_antismash_calibration}.py  # antiSMASH calibration
config/compound_class_map.yaml   # antiSMASH/MIBiG product → our 22-class vocabulary
tests/                           # GPU-free unit tests (run tests/run_all.py)
docs/project_memory/             # decisions / bugs / progress (working memory)
docs/model_comparison_evo2_vs_genomeocean.md   # the two-track head-to-head
docs/conditioning_next_steps.md  # ranked plan + literature for what to try next
docs/steering_program.md         # the closed steering programme, start to finish

# ---- EVO2 1B TRACK ----  (see evo2_1b/README.md; PHASE 2, requires TE 1.13.0)
evo2_1b/scripts/evo2_1b_inference.py       # loader + substrate sanity check
evo2_1b/scripts/compare_1b_7b_loss.py      # 1B-vs-7B next-base CE on real cores
evo2_1b/experiments/run_objective_arms.sh  # baseline / frame / weighted
evo2_1b/experiments/score_arms.sh          # generate, then ladder + novelty
docs/archive/                    # archived runbooks, plans, and dated audits

# ---- EVO2 TRACK ----  (see evo2/README.md)
evo2/scripts/finetune_evo2_lora.py    # training implementation (LoRA on Evo2 7B)
evo2/scripts/queue_h100_production.sh # idle-GPU-gated production launcher (ckpt + auto-resume)
evo2/scripts/queue_h100_smoke.sh      # shared-GPU-safe memory smoke matrix
evo2/scripts/generate_bgc.py          # conditioned generation (sequential; batched gated off)
evo2/scripts/run_eval.sh              # full evaluation after training
evo2/scripts/quick_eval.sh            # fast per-checkpoint functional score
evo2/experiments/{probes,quartz}/     # the probe programme; Quartz long-context staging
evo2/docs/                            # evo2_lora_and_hyena.md, quartz_setup.md

# ---- GENOMEOCEAN TRACK ----  (see genomeocean/README.md)
genomeocean/scripts/analyze_tokenization.py        # BPE compression + context fit vs Evo2
genomeocean/scripts/probe_finetune_feasibility.py  # LoRA / class-token / memory gate
genomeocean/scripts/generate_bgc_go.py             # replicate their zero-shot BGC generation
genomeocean/experiments/                           # measurement outputs
genomeocean/external/                              # upstream clone (gitignored)
```

Shell wrappers in `evo2/scripts/` are still invoked **from the repo root**, e.g.
`evo2/scripts/queue_h100_smoke.sh`.

---

## Model & training stack

- **Base model:** Evo2 7B (`arcinstitute/evo2_7b_262k`), StripedHyena-2 hybrid
  (27 Hyena ops + 5 attention layers), byte-level `CharLevelTokenizer`, 262k context.
- **Strategy:** LoRA adapters (r=16, α=32, ~28.7M trainable ≈ 0.44%), targeting
  `Wqkv / out_proj / out_filter_dense / l1 / l2 / l3`. Embedding + LM head **frozen**.
- **Orchestration:** DeepSpeed (ZeRO-2) + PEFT + PyTorch, bf16.
- **Conditioning prefix:** `|COMPOUND_CLASS:{class}|{native lowercase GTDB tag}` then the
  nucleotide sequence. Loss is **masked over the prefix** (only the BGC half trains).
- **Host:** `gputee`, a single NVIDIA H100 PCIe (80 GB). Data / runs / HF cache live on
  `/data2` (home is near-full).

---

## Data

**Active dataset (v2):** `/data2/ds85/bgcmodel_data/splits_core/{train,val,test}.jsonl`

- **train 47,524 / val 8,048 / test 18,871**, 22 compound classes.
- Sequences are **strict antiSMASH core regions** — the contiguous span of
  `gene_kind="biosynthetic"` CDS, re-extracted from re-acquired antiSMASH-DB (asdb5)
  whole-genome GBKs (median ~3 kb, ~88% single-window).
- **Native lowercase GTDB** taxonomy tags (e.g.
  `|d__Bacteria;p__Pseudomonadota;…;s__Escherichia coli|`) — the old UPPERCASE_underscore
  tags were out-of-distribution for Evo2.
- **Leakage-clean:** genome-disjoint split (`split_dataset_grouped.py`) + exact-md5 +
  cross-split MMseqs2 near-dup removal (`dedup_core_splits.py`).
- **MiBIG held out** (`exclude_mibig_from_core.py`): near-dups of the 2,636 MiBIG BGCs
  removed from training, reserved for a later **compound-conditioned** FT.

Build pipeline: `build_core_records.py` → materialize strict cores →
`split_dataset_grouped.py` → `curate_dataset.py` → `dedup_core_splits.py` →
`exclude_mibig_from_core.py`.

**Deprecated (do not use):** `splits_curated/` (~18K), `splits_combined_grouped/`,
`splits_dedup/`, and `data/processed/splits_combined/` (leaky — 94.6% genome overlap).

---

## Training

```bash
cd ~/projects/BCGModelling
micromamba activate bgcmodel
export HF_HOME=/data2/ds85/hf_cache
```

**Production launch** (idle-GPU-gated; persistent tmux; checkpoints + auto-resume):

```bash
evo2/scripts/queue_h100_production.sh          # waits for a free GPU, then trains
```

Key constraints (H100 80 GB, `L=32768`):

- The only micro-batch shape that fits is **`--batch-size 1 --grad-accum 128`**
  (effective batch 128; `bs=4 ga=32` OOMs). LoRA hyperparameters remain valid.
- **Block-level activation checkpointing** is default-on (`--no-activation-checkpointing`
  to opt out). No-checkpoint is not viable above short contexts.
- Memory ceiling: `L=32768` passes with margin; `L=65536` is near-limit; `L=98304` OOMs.
- Production uses `--long-seq-strategy chunk --chunk-overlap 2048` (deterministic tiling,
  full nucleotide coverage); pre-build length sidecars with `evo2/scripts/build_chunk_index.py`.
- Validation: first-window-only (prefix-aligned) loss, length-stratified, with early stopping.

**Smoke / memory matrix** (shared-GPU-safe):

```bash
evo2/scripts/queue_h100_smoke.sh                          # default lengths
evo2/scripts/queue_h100_smoke.sh --lengths "49152 65536 98304"   # long-context probe
```

---

## Evaluation

The suite (in `src/bgc_pipeline/evaluation.py`) has two layers: **CHECKS** (compute
units, all sharing one gene caller — **pyrodigal/Prodigal**) combined into **QUESTIONS**
(what we actually want to know). GATE questions decide accept/reject; diagnostics inform.

| Question | Derived from | Gate? |
|----------|--------------|-------|
| `is_bgc` | `coding_sanity` ∧ `antismash.detected` (class_markers proxy) | ✅ |
| `correct_class` | `antismash.class_match` (class_markers proxy) | ✅ |
| `novel` | `kmer_novelty` (anti-memorization vs training) | ✅ |
| `proteins_plausible` | `protein_homology` (MMseqs2 vs known enzymes) | diag |
| `complete` | `module_architecture` (ordered NRPS/PKS modules) | diag |
| `class_probe_agrees` | `class_probe` — **continuous** class probability (opt-in) | diag |

`taxon_faithfulness` / `conditioning_faithful` were **removed** on 2026-08-10: it produced
`no_verdict` on 870/870 records and measures taxon conditioning, which is not what this
project tests. The function survives for `evo2/scripts/conditioning_experiment.py`.

### Calibration — measured at both ends, 2026-08-10

Every gate's false-positive rate used to be *asserted*. It is now measured, against 25 real
non-BGC windows cut from the same genomes outside every annotated region
(`scripts/make_negative_control.py`, which **refuses** to substitute shuffled sequence —
shuffling destroys codon structure so every gate passes trivially):

| instrument | false-positive rate | sensitivity (real class DNA @ 3 kb) |
|---|---|---|
| antiSMASH `is_bgc` | **0.000** (0/25) | 0.680 @ 2 kb |
| `class_markers` biosynthetic ≥2 | 0.040 | **0.717** |
| *(retired)* any-Pfam ≥1 proxy | **0.960** | — |
| `class_probe` argmax | n/a — see below | **0.900** |

The retired any-Pfam proxy would have called **96% of ordinary bacterial DNA** a BGC. Any
historical number computed with antiSMASH skipped used it; see `progress.md`.

- **antiSMASH is the gold-standard `is_bgc`/`correct_class` detector** (≈3 s/core),
  **recalibrated 0.15 → ≈0.97** on real held-out cores by completing the antiSMASH
  product→class map (`scripts/build_class_map.py` → `config/compound_class_map.yaml`,
  covering all 103 antiSMASH 8 products). `class_markers` (data-driven per-class Pfam
  markers, ANY-marker = right class, ≈0.87 on real cores) is the fast **proxy** when
  antiSMASH is skipped.
- **Gene caller:** pyrodigal (Prodigal) everywhere — replaced the legacy six-frame ORF
  finder, which fragmented megasynthases.
- **`class_probe` — the one CONTINUOUS readout (added 2026-08-10, diagnostic only).** Every
  other class instrument is binary, and a threshold gate bounds a *large* effect while saying
  nothing about a small one. With `class_markers` at TPR 0.717 and antiSMASH detecting only
  ~1/3 of seeded 3 kb generations, an intervention can move the model substantially toward a
  class and still score exactly 0.000. `class_probe` reports a probability instead, and it
  immediately found a p = 0.006 class-specific effect in sequences every binary gate scored
  as a flat zero.
  **It can never gate**, and the calibration is why: it is **0.900 confident on real non-BGC
  DNA** versus 0.986 on real clusters. It has no negative class and cannot abstain, so it
  measures *resemblance*, not validity — trustworthy only in **paired** comparisons where
  that shared bias cancels. (RIPP is its default guess for unremarkable DNA, 14/25 negatives.)
  Scores come from a model-specific scorer so the suite stays model-agnostic:
  `evo2/scripts/probe_score_generations.py --emit-sidecar` → `eval_suite_driver --probe-scores`.
  Calibrate with `evo2/scripts/calibrate_class_probe.py`, which **refuses to run without the
  negative control**.
- **Retired:** synthesis feasibility, Evo2 perplexity, BiG-SCAPE, `taxon_faithfulness`.
  E. coli expressibility is pruned from gating (the wet-lab axes are out of scope).
- Headline tiers: `generates_bgc` → `correct_class` → `biological_valid` (both) →
  **accept** (+ `novel`). Rates return `None`, never a fabricated `0.0`, when nothing was
  evaluated; a separate `funnel` block reports the monotone view on the common subset.

**Run it:**

```bash
# Fast per-checkpoint functional score (runs the cheap checks incl. antiSMASH;
# skips protein_homology + kmer_novelty). Appends a row to eval_track.jsonl.
evo2/scripts/quick_eval.sh <run-dir-or-checkpoint-dir> [out-dir]

# Full evaluation after training (generation → novelty → conditioning → suite).
evo2/scripts/run_eval.sh <run-dir-or-checkpoint-dir> [out-dir]

# Direct driver (named checks; skip by name):
python scripts/eval_suite_driver.py --gen gen.jsonl --positive pos.jsonl \
  --pfam-hmm /data2/ds85/pfam/Pfam-A.hmm --antismash-db /data2/ds85/antismash_db \
  --skip-checks protein_homology kmer_novelty --output eval.json
```

**Controls — build them, do not skip them.** Until 2026-08-10 every driver passed a
deliberately-nonexistent `--positive`, so 0 of 25 reports on disk had a ceiling and every
rate was a fraction of an unstated maximum:

```bash
# CEILING: real held-out cores at the generations' own length AND class mix
python scripts/make_positive_control.py --gen gen.jsonl --out pos.jsonl

# FLOOR: real non-BGC windows from the same genomes, outside every annotated region
python scripts/make_negative_control.py --gen gen.jsonl --gbk-tar <genomes.tar> --out neg.jsonl

# CONTINUOUS class readout (optional; paired comparisons only)
python evo2/scripts/probe_score_generations.py gen.jsonl --emit-sidecar probe.json \
  --out-json probe_scores.json
python scripts/eval_suite_driver.py --gen gen.jsonl --probe-scores probe.json ...
```

Calibration validators: `scripts/validate_antismash_calibration.py` (is_bgc/correct_class),
`scripts/validate_m2_calibration.py` (class_markers),
`scripts/marker_sensitivity.py` (marker TPR/FPR at generation length),
`evo2/scripts/calibrate_class_probe.py` (probe TPR + behaviour on non-BGC DNA).

---

## Environment recreation

```bash
micromamba create -n bgcmodel -f environment.yml   # gputee has micromamba, not conda
micromamba activate bgcmodel
```

> **`environment.yml` alone does not produce a working env on a fresh create.** The pip
> step crashes on `flash-attn` (its `setup.py` runs `import torch` before torch is
> installed). The conda side finishes cleanly. Working sequence: install torch first,
> then a prebuilt flash-attn wheel, re-run `env update`, then deepspeed/peft/wandb.
> See `docs/archive/gputee/FINETUNE_GUIDE.md §2`. `requirements.txt` lists the Python
> deps (incl. `pyrodigal>=3`, `pyhmmer`, `biopython`).

On conda hosts: `conda env create -f environment.yml` (same caveat).

---

## Known gotchas

- **vortex generation de-batches mixed-length prompts** (silently). Left-padding to
  equalize lengths perturbs StripedHyena and fails an on-GPU equivalence gate, so
  `generate_bgc.py` defaults to **sequential** generation.
- **Evo2 `eos_id = 0` is the null byte** — unusable as a stop token; generation runs the
  full `n_tokens`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reports **unsupported** here; not an
  OOM fix.
- NCCL "process group not destroyed" on shutdown is expected for short smoke runs.
- Shared-host GPU contention can invalidate memory measurements — use the queued,
  idle-gated runs.

More quirks + fixes: [`docs/project_memory/bugs.md`](docs/project_memory/bugs.md).

---

## Archived documentation

Detailed, host-specific, and historical docs live in [`docs/archive/`](docs/archive/):
the `gputee/` runbooks (`FINETUNE_GUIDE`, `PROJECT_GUIDE`, `BGC_Research_Plan`,
`MIGRATION_CHANGELOG`), the `trojai/` (old A40 host) snapshot, `EVAL_RUNBOOK.md`,
`REDESIGN_PLAN.md` (the eval-rewrite record), the dated audits (`AUDIT_FINDINGS.md`,
`STATE_AND_AUDIT.md`, `FABLE5_AUDIT.md`), and the TPU grant materials. They are kept
for reference but are **not** maintained as current — this README + `docs/project_memory/`
are the source of truth.
