# data.md — the map

**Purpose.** Single source of truth for data and artifact topology. Read before writing a loader,
a scoring script, or any path. Never guess a path or a record count — they are here, verified.

**Rule.** *Every number in this project is traceable to a (checkpoint, generation set, scoring
config, n) tuple.* If an artifact is not in this file, it is not quotable.

Verified against disk 2026-08-14. `tests/test_docs_contract.py` re-verifies.

---

## 1. Storage layout

| Root | Contents |
|---|---|
| `/data2/ds85/bgcmodel_data/` | datasets and splits |
| `/data2/ds85/bgcmodel_runs/` | training runs, checkpoints, generation sets |
| `/data2/ds85/pfam/` | `Pfam-A.hmm`, `biosynthetic_subset.hmm` (~91 models) |
| `/data2/ds85/hf_cache/` | `HF_HOME` — set it before any model load |
| `/data2/ds85/envs/genomeocean` | GenomeOcean env (separate from `bgcmodel`) |
| `/data2/ds85/asdb5_gbks/` | raw antiSMASH-DB GenBank source |

Nothing above lives in the repo. The repo holds code and docs only.

---

## 2. Record schema (all `*.jsonl` splits)

One JSON object per line. Identical across `splits_core/` and `splits_class/`.

| Field | Type | Notes |
|---|---|---|
| `accession` | str | `GCF_000022545.1.region2` — genome + region |
| `genome_accession` | str | **the grouping key for splitting** |
| `region_number` | int | |
| `compound_class` | str | one of 22 + `OTHER`; see `terms.md` |
| `antismash_products` | list[str] | raw product strings, pre-class-map |
| `contig_edge` | bool | `True` = truncated by assembly; quality filter |
| `taxonomic_tag` | str | native lowercase GTDB, `\|d__Bacteria;p__...` |
| `sequence` | str | nucleotide, strict antiSMASH **core** region |
| `region_start` / `region_end` | int | coordinates in source genome |
| `strict_core_genes` | int | count of core biosynthetic genes |
| `region_len` | int | nt |
| `training_text` | str | **what the model sees**: `\|COMPOUND_CLASS:RIPP\|\|d__Bacteria;...\|<seq>` |

⚠️ `training_text` is the trained-on field. `sequence` is the raw core. Loss is masked over the
prefix — only the sequence half contributes (see `memory.md`, prefix-masked loss).

---

## 3. Datasets — LIVE

### `splits_core/` — all-class, the Phase-1/2 workhorse
`/data2/ds85/bgcmodel_data/splits_core/`

| File | Records | Purpose |
|---|---|---|
| `train.jsonl` | 47,524 | training |
| `val.jsonl` | 8,048 | validation (first-window, prefix-aligned) |
| `test.jsonl` | 18,871 | held out |
| `train.domain_spans.jsonl` | 47,524 | derived: per-record domain spans for the weighted objective |
| `valtest_fit.jsonl` | 13,406 | derived: probe/steering-direction fitting |
| `valtest_eval.jsonl` | 13,513 | derived: probe evaluation |
| `valtest_eval_4class.jsonl` | 11,910 | derived: 4-class probe subset |

- **Construction:** strict antiSMASH core regions, native lowercase GTDB tags.
- **Leakage control:** genome-disjoint (grouped on `genome_accession`) + exact-duplicate removal +
  cross-split MMseqs2 pass.
- **MiBIG excluded** (56K → 47.5K), reserved for a later compound-conditioned FT.
- **22 classes.**
- ⚠️ **`valtest_fit.jsonl` IS A LEAKAGE TRAP, not merely a derived file.** Until 2026-08-10 the
  class probe was fit on `acts_valtest_fit.npz` — activations of **val+test** cores — and then used
  to score generations **seeded from those very cores**. The probe had seen the seeds; every number
  it produced was contaminated. The fit set MUST be train-only
  (`class_probe_sweep/acts_v2_train500.npz`); `probe_score_generations.py` now checks a
  `.provenance.json` on each activation cache and **refuses** a non-train set, with
  `--allow-leaky-probe` reserved for reproducing a historical number. Steering directions carry the
  same open debt (`memory.md` 2026-07-30).
- The other three derived files were simply undocumented before this file existed.

### `splits_class/<CLASS>/` — per-class, the Phase-3 substrate
`/data2/ds85/bgcmodel_data/splits_class/`

Built by `scripts/build_single_class_splits.py`. **Split FROM SCRATCH** — genome-disjoint with a
fresh MMseqs2 cross-split pass — **not filtered out of `splits_core`**. Manifest:
`splits_class/manifest.json`. Near-dup criterion: `mmseqs id>=0.8 cov>=0.5`. Fracs 0.8/0.1/0.1.

| Class | pooled | train | val | test | median nt | near-dup loss | in manifest | status |
|---|---|---|---|---|---|---|---|---|
| **RIPP** | 10,163 | **8,129** | 576 | 579 | **1,931** | 43% | ✅ | ✅ **PHASE-3 TARGET** |
| TERPENE | 14,122 | 11,297 | 793 | 732 | 960 | 46% | ✅ | available |
| PKS | 6,495 | 5,195 | 323 | 309 | 2,103 | — | ✅ | available |

`splits_class/RIPP/eval_prompts.jsonl` — 200 fixed generation prompts.

✅ **Manifest matches disk** (verified 2026-08-14 after cleanup). HSERLACTONE and BUTYROLACTONE
were built 11:51–11:52 on 2026-08-14 and orphaned when `build_single_class_splits.py` **rewrote the
whole manifest** at 12:34 covering only RIPP/PKS/TERPENE. Their leakage controls were therefore
unverifiable from the record, and both classes were already disqualified on diversity (69% / 81%
near-dup loss), so **they were deleted 2026-08-14** rather than given a fabricated provenance entry.
⚠️ The builder overwrites `manifest.json` wholesale — rebuilding one class drops every other entry.

⚠️ **ECTOINE and MELANIN are DISQUALIFIED, not merely unbuilt.** 85% and 95% of their held-out
clusters are near-duplicates of training clusters. Length and diversity are anti-correlated across
these classes — the short classes are short *because* they are conserved.

✅ **Target confirmed RIPP** (user, 2026-08-14). The archived `progress.md` header reading "the
target is TERPENE" is corrected in place at `docs/archive/pre-framework/progress.md`.

---

## 4. Datasets — DEPRECATED (DO NOT USE)

| Path | Why |
|---|---|
| `data/processed/splits_combined/` | ☠️ **LEAKY** — 94.6% genome overlap, 453 byte-identical seqs across splits |
| `splits_combined_grouped/` | superseded by the strict-core rebuild |
| `splits_curated/` | the ~18K curation; superseded |
| `splits_core_curated/` | superseded |
| `splits_core_grouped/` | superseded |
| `splits_core_premibig/` | pre-MiBIG-exclusion snapshot |
| `splits_dedup/` | superseded |
| `mega_whole_32k/` | whole-core-only run; starved the data (see `memory.md` 2026-07-12) |
| `probe_subsets/`, `probe_subsets_8k/` | probe-specific; not training data |

---

## 5. Run registry

**Status key:** ✅ LIVE (quotable) · 📦 SUPERSEDED (historical, do not quote as current) ·
☠️ INVALID (known-bad; never quote).

### Phase 3 — current

| Run dir | Date | Size | Contents | Status |
|---|---|---|---|---|
| `phase3_RIPP/` | 08-14 | 698M | **A0 + pilots** (merged 2026-08-14). `adapter_run/` (RIPP LoRA, 7,250 whole records, 3 ep / 1,350 steps, `loss_ce` 0.790→0.410); `A0_8k.jsonl`, `A0_noseed.jsonl` (150 de novo); `A0_8k_w2000.json` / `A0_8k_w8000.json`; `A0_battery.json`; `train.whole.jsonl`, `val.whole.jsonl`; `pilot_base.jsonl`, `pilot_general.jsonl`, `pilot_rates.json` | ✅ |

✅ **Case collision resolved 2026-08-14.** `phase3_ripp/` (pilot baselines) was `rsync`-merged
into `phase3_RIPP/` — no filename collisions, verified a strict subset before removal — and the
lowercase directory deleted. `evo2_1b/experiments/phase3_pilot.py` updated to the merged path.
See the naming convention in `CLAUDE.md`.

### Phase 2 — objective change (CLOSED)

| Run dir | Date | Size | Status |
|---|---|---|---|
| `phase2_long/` | 08-13 | 2.0G | ✅ two-arm long run (baseline + weighted 3×, 2,000 steps) |
| `phase2_1b/` | 08-13 | 2.9G | ✅ frame-aware × domain-weighted arms, L=8192 |
| `base_vs_lora/` | 08-12 | 308K | ✅ base-vs-adapter comparison |

⚠️ The **weighted arm's treatment never landed** — its null is uninterpretable. The Phase-2 closure
applies to the **frame arm only**. Do not quote the weighted arm as a negative result.

### Conditioning / steering programme (CLOSED — historical)

| Run dir | Date | Size | Status |
|---|---|---|---|
| `class_probe_sweep/` | 08-11 | 23G | ✅ linear probe 0.911 (chance 0.091) |
| `guided_decoding/` | 08-11 | 988K | ✅ **the one positive** — Q1 +5.71 (39/40); Q2 underpowered (5–0, p=0.0625, effective n=5) |
| `patch_generate/`, `patch_generate_BASE_nofloor/` | 08-11 | ~400K | ✅ cross-class activation transplant — 92% toward donor, class carried 0/48 |
| `soft_prefix/` | 08-10 | 2.6M | ✅ per-class soft prefixes; input-only, closed |
| `steer_stack/`, `steer_l27/`, `steer_reach/`, `steer_phase2/` | 08-10 | ~5M | 📦 |
| `steer_phase3/`, `steer_phase3_d2/`, `steer_phase3_d4/` | 07-31 | ~1.8M | 📦 |
| `steer_causal/`, `steer_magnitude/`, `steer_titration/`, `steer_sweep/` | 07-29/30 | ~6M | 📦 |
| `seed_deconfound/` | 07-28 | 7.4M | ⚠️ see `memory.md` — a same-day confound claim was **RETRACTED**; pinned by `tests/test_scored_span.py` |
| `cfg_diagnostic/`, `cfg_diagnostic_fine/` | 07-21/22 | 58M | 📦 classifier-free guidance; closed |
| `seed_diagnostic/` | 07-22 | 688K | 📦 |
| `simple_class_eval/`, `simple_class_confirm/` | 07-21 | 101M | 📦 |
| `class_probe/` | 07-27 | 56K | 📦 early probe |

### GenomeOcean

| Run dir | Date | Size | Status |
|---|---|---|---|
| `go_zeroshot_rate_n216/` | 07-27 | 24M | ✅ leakage gate **PASSED** — 0.0000 containment, greedy, positive control first |
| `go_zeroshot_bgcfm/` | 07-27 | 3.0M | ✅ |

### Phase 1 / infrastructure

| Run dir | Date | Size | Status |
|---|---|---|---|
| `probes_20260706/` | 07-13 | **28G** | ✅ probe sweep — de-chunking is the lever |
| `mega_whole_32k_run/` | 07-12 | 3.2G | ☠️ whole-core-only FAILED; starves the data |
| `phase1_lora_prod_20260617_095202_L32768/` | 07-06 | 11G | ✅ the Phase-1 production adapter |
| `phase1_lora_prod_20260604_151651_L32768/` | 06-16 | 3.6G | 📦 |
| `phase1_lora_prod_20260604_151300_L32768/`, `phase1_lora_prod_20260604_151541_L32768/` | 06-04 | — | ☠️ **DELETED 2026-08-14** — empty failed launches |
| `v2_smoke/` | 06-17 | 1.4G | 📦 |
| `quick_eval_step_250/` | 06-16 | 181M | 📦 |
| `conditioning_diag_step250/`, `conditioning_diag_stoch_step250/` | 06-16 | ~1.4M | 📦 |
| `_scripts/` | 07-29 | — | ☠️ **DELETED 2026-08-14** — empty |

### ⚠️ Loose files at the runs root — 69 files, unattributed

`/data2/ds85/bgcmodel_runs/` holds **21 result artifacts (4.0 MB)**, **47 logs (0.6 MB)** and one
shell script sitting *outside any run directory*. The results are not junk — several are
load-bearing and referenced by code: `ladder_audit.json` (the ladder validation itself),
`direction_audit.json`, `activation_patching_ksweep.json`, `context_ablation.json`,
`length_ceiling.json`, `class_probe_calibration_trainonly.json`, `denovo_decomposition.tsv`.

**They have no owning experiment**, which is the same disease as everything else here: a number
with nowhere to attach its provenance. **Not moved** — four are referenced by path in code, so
filing them is a change with blast radius, not a tidy-up. Queued as `plan.md` [P3-B5].
Going forward the naming convention in `CLAUDE.md` forbids creating more.

**Disk:** ~75 GB total, dominated by `probes_20260706` (28G) and `class_probe_sweep` (23G).

---

## 6. Training configuration (bears on data shape)

- **Context:** `L=32768` default; `L=65536` near-limit; `L=98304` OOMs.
- **Micro-batch:** `--batch-size 1 --grad-accum 128` is **the only shape that fits at L=32k** on
  this 80 GB H100. Effective batch 128. `bs=4` OOMs on forward, `bs=2` on backward.
- **Long sequences:** `--long-seq-strategy chunk --chunk-overlap 2048` (deterministic tiling, full
  nucleotide coverage). Phase-3 whole-record training uses L=8192 and **drops** over-length records
  rather than truncating (879 of 8,129 dropped) so `|END|` never lands on a cut sequence.
- **Sidecars:** `<split>.lengths.npy` + `.lengths.meta.json`; pre-build with
  `evo2/scripts/build_chunk_index.py`.
- **Length-bucketed batching** is carried forward to every future training run.
- **Loss masking:** `labels[:, :prefix_token_count] = IGNORE_INDEX`. Absolute loss values are not
  comparable to pre-2026-05-14 runs.

## 7. Model substrates

| Substrate | Selector | Role |
|---|---|---|
| **Evo2 1B** | `EVO2_BASE_MODEL=evo2_1b_base` | ✅ **the Phase-3 testing substrate** |
| Evo2 7B | default | confirmation of publishable results only |
| GenomeOcean-4B / `bgcFM` | separate env | live but held |

⚠️ The 1B **requires Transformer Engine 1.13.0**. Without it the model loads and is silently at
chance — a failure with no error message. Verify TE before trusting any 1B number.
