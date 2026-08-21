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
  `--allow-leaky-probe` reserved for reproducing a historical number. Steering directions carried the same debt and were **refit train-only on 2026-08-10**
  (`trainonly.steerdirs.npz`, 9 layers). The debt is CLEARED — see `memory.md` 2026-08-10.
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

⚠️ **`accession` is NOT unique** — RIPP train has 8,129 rows over 7,808 distinct accessions (321
collisions; val 576/558, test 579/560). Colliding rows are *different regions sharing one label*, not
duplicated data. **Splits remain genome- and accession-disjoint across train/val/test (verified
2026-08-18), so there is no leakage** — but any accession-keyed join must state its collision policy.
See `bugs.md`.

`splits_class/RIPP/eval_prompts.jsonl` — 200 fixed generation prompts. ⚠️ **100% drawn from
TEST** (199/199 accessions, 0% genome overlap with train), so tuning anything on it selects on the
test set. `splits_class/RIPP/val_prompts.jsonl` — 60 records from **val** (len ≥ 1000 so a 500-nt
seed leaves ≥ 500 nt of target), built 2026-08-17 for the Stage-1 seed-length sweep.

✅ **Manifest matches disk** (verified 2026-08-14 after cleanup). HSERLACTONE and BUTYROLACTONE
were built 11:51–11:52 on 2026-08-14 and orphaned when `build_single_class_splits.py` **rewrote the
whole manifest** at 12:34 covering only RIPP/PKS/TERPENE. Their leakage controls were therefore
unverifiable from the record, and both classes were already disqualified on diversity (69% / 81%
near-dup loss), so **they were deleted 2026-08-14** rather than given a fabricated provenance entry.
⚠️ The builder overwrites `manifest.json` wholesale — rebuilding one class drops every other entry.

### ⛔ `DEPRECATED_component_panels.json` — SCRAPPED 2026-08-19, do not use
Was `component_panels.json`. Precursor/transport/regulator/protease families selected by **regex
keyword match over Pfam-A `NAME`+`DESC` text** — never validated per family. The **precursor panel
is ~half enzyme** (PF14028 + PF04738 lantibiotic *dehydratases*, PF03515 colicin toxin) and
**overlaps `OBLIGATE_DOMAINS[RIPP]` via PF14028**, which made the "P+E" metric partly tautological.
Renamed rather than deleted so the provenance of the retracted numbers stays traceable.
**Replacement: antiSMASH/RODEO precursor calls — see `plan.md` [P5-DETECT].**

### `ripp_components.jsonl` — per-CDS component annotation, built 2026-08-19
`/data2/ds85/bgcmodel_data/ripp_components.jsonl`. One row per RIPP region: `accession`,
`genome_accession`, `compound_class`, region coordinates, `n_cds`, and **`genes`** — every CDS in
the region with `start`, `end`, `strand`, **`kind`** (antiSMASH `gene_kind`), `aa_len`, `product`.

Streamed once from `asdb5_gbks/asdb5_gbks.tar` (185 GB) with a genome allowlist of the 7,845 RIPP
genomes. **This is the annotation `build_core_records.py` computes per CDS and then discards** —
every component-level metric depends on it.

**gene_kind census inside RIPP regions:** none 66.4% · biosynthetic-additional 14.9% ·
biosynthetic 6.8% · transport 5.0% · regulatory 4.7% · other 2.1% · resistance 0.03%.

⚠️ **Rows are NOT unique by accession.** `GCF_x.regionN` repeats across contigs of the same
assembly because `region_number` restarts per contig — the same collision recorded in `bugs.md`.
**Dedupe on `(accession, region_start, region_end)` before any per-region statistic.**

### `splits_class_strictmatched/RIPP/` — size- and cluster-matched control, built 2026-08-18
**3,723 train / 258 val**, strict spans, restricted to **exactly the rows** the WIDE split kept
(same accessions, mirroring WIDE's last-wins collision rule). Test and eval_prompts copied unchanged.
Purpose: isolate span width from training-set size in [P4-WIDE-SEEDED]. Without it, a WIDE-vs-S2-1
difference confounds width with the 7,250 → 3,723 size drop.

### `splits_class_wide/<CLASS>/` — WIDE_KINDS spans, built 2026-08-18

`/data2/ds85/bgcmodel_data/splits_class_wide/RIPP/` — **7,808 train · 558 val · 560 test.**
Same accessions and **the same train/val/test assignment inherited verbatim** from
`splits_class/RIPP`, so WIDE vs STRICT is a clean A/B on span width alone. Sequence is
`wide_sequence` from `asdb5_core_records.jsonl` = the span of
`{"biosynthetic", "biosynthetic-additional"}` CDS. 8,926/8,926 accessions matched.

| | STRICT | **WIDE** | ratio |
|---|---|---|---|
| median nt | 1,854 | **8,494** | 4.58× |
| mean genes | 1.87 | **4.41** | 2.36× |
| **single-gene records** | **48.8%** | **14.8%** | — |
| **≥3 genes** | 21.9% | **67.7%** | — |
| share of the antiSMASH region kept | 9.8% | **46.7%** | — |

⚠️ **Context-window cost — this is the `mega_whole_32k` "starves the data" risk, quantified:**

| L | STRICT fit | WIDE fit |
|---|---|---|
| 8,192 (current Phase-3 config) | 89.6% | **48.6%** |
| 16,384 | 96.7% | **80.3%** |
| 32,768 | 98.9% | 97.5% |

⇒ WIDE at the current L=8192 would drop or truncate **half** the data. At L=16384 it keeps 80%.
**Not yet used for training** — see the strategic question in `plan.md`.
⚠️ Cross-split near-dup dedup was performed on the STRICT spans; widening could reintroduce
near-duplicates and **must be re-checked before any WIDE training run**.

⚠️ **ECTOINE and MELANIN are DISQUALIFIED, not merely unbuilt.** 85% and 95% of their held-out
clusters are near-duplicates of training clusters. Length and diversity are anti-correlated across
these classes — the short classes are short *because* they are conserved.

✅ **Target confirmed RIPP** (user, 2026-08-14). The archived `progress.md` header reading "the
target is TERPENE" is corrected in place at `docs/archive/pre-framework/progress.md`.

---

## 4. Datasets — DEPRECATED

**All deprecated split directories were DELETED 2026-08-17** (~41 GB reclaimed). They are recorded
here so an old number can still be traced to the dataset that produced it — the data is gone, the
provenance is not.

| Path | Why | Status |
|---|---|---|
| `data/processed/splits_combined/` | ☠️ **LEAKY** — 94.6% genome overlap, 453 byte-identical seqs across splits | deleted |
| `splits_combined_grouped/` | superseded by the strict-core rebuild (22 GB) | deleted 2026-08-17 |
| `splits_curated/` | the ~18K curation; superseded (4.1 GB) | deleted 2026-08-17 |
| `splits_core_curated/` | superseded (1.1 GB) | deleted 2026-08-17 |
| `splits_core_grouped/` | superseded (4.6 GB) | deleted 2026-08-17 |
| `splits_core_premibig/` | pre-MiBIG-exclusion snapshot (1.5 GB) | deleted 2026-08-17 |
| `splits_dedup/` | superseded (2.5 GB) | deleted 2026-08-17 |
| `probe_subsets/`, `probe_subsets_8k/` | probe-specific, not training data (7.6 GB) | deleted 2026-08-17 |
| `mega_whole_32k/` | whole-core-only run; starved the data | retained |

### ⚠️ KEEP — the pipeline intermediates

| File | Size | Why it must stay |
|---|---|---|
| `asdb5_train_records.jsonl` | 22 GB | **`splits_core` is built from these.** Deleting them means re-running antiSMASH extraction from `asdb5_gbks/` — days of compute — to rebuild any split. |
| `asdb5_core_records.jsonl` | 17 GB | ″ |
| `asdb5_core_strict.jsonl` | 4.7 GB | ″ |

## 5. Run registry

**Status key:** ✅ LIVE (quotable) · 📦 SUPERSEDED (historical, do not quote as current) ·
☠️ INVALID (known-bad; never quote).

### Phase 3 — current

| Run dir | Date | Size | Contents | Status |
|---|---|---|---|---|
| `phase3_RIPP/` | 08-14 | 698M | **A0 + pilots + controls.** `adapter_run/` (RIPP LoRA, 7,250 whole records, 3 ep / 1,350 steps, `loss_ce` 0.790→0.410); `A0_8k.jsonl`, `A0_noseed.jsonl`; `pilot_base.jsonl`, `pilot_general.jsonl`; `ctrl_base_n150_s1.jsonl`, `ctrl_general_n150_s1.jsonl` (B3, seed 1). **Stamped scores:** `A0_8k_w2000_RIPP.json`, `pilot_base_w2000_RIPP.json`, `pilot_general_w2000_RIPP.json`. `superseded/` holds the pre-fix generic-scored files. | ✅ |

✅ **Case collision resolved 2026-08-14.** `phase3_ripp/` (pilot baselines) was `rsync`-merged
into `phase3_RIPP/` — no filename collisions, verified a strict subset before removal — and the
lowercase directory deleted. `evo2_1b/experiments/phase3_pilot.py` updated to the merged path.
See the naming convention in `CLAUDE.md`.

### Phase 3 — WIDE_KINDS fine-tune

| Run dir | Date | Contents | Status |
|---|---|---|---|
| `phase3_RIPP_wide/` | 08-18 | **[P4-WIDE] fine-tune on the WIDE_KINDS substrate.** Same recipe as A0 (`train_class_adapter.sh`, renamed from `train_ripp.sh` 2026-08-19; LoRA, L=8192, bs=1 ga=16, **3 epochs**) with `DATA=splits_class_wide/RIPP`. ⚠️ **3,723/7,808 train records kept (47.7%)** — the rest exceed the 1B's 8,192 native context and are DROPPED, not chunked, so `\|END\|` still lands at a true boundary. val 258/558. Epochs matched to A0 rather than steps, because the dataset is smaller. `adapter_run/`, `train.log`, `train.whole.jsonl`, `val.whole.jsonl`. | 🔄 training |

⚠️ **This arm confounds span width with dataset size** (3,723 wide vs 7,250 strict records used by
A0). A size-matched STRICT control is required before the comparison is clean — see `plan.md`.

### Phase 6 / 7 — PKS and TERPENE (opened 2026-08-19)

⛔ **`DEPRECATED_<arm>_truncatepath.jsonl` in `phase6_PKS/` and `phase7_TERPENE/` — DO NOT USE.**
Six de novo arms (2026-08-20) generated before the `extract_sequence` truncation bug was found
(`bugs.md`). They keep only the leading ACGTN run, so PKS `A0` discarded a median of 6.2 kb of
99.9%-valid DNA per record and 45.5% of the arm fell below its scoring window against 2.0% of its
own base control. Regenerated on `--junk-policy mask`; retained only as evidence for the bug entry.


| Run dir | Date | Contents | Status |
|---|---|---|---|
| `phase6_PKS/` | 08-19 | **[P6-A0] STRICT-span PKS adapter.** `train_class_adapter.sh` with `CLASS=PKS`, `DATA=splits_class/PKS`, L=8192, LoRA, bs=1 ga=16, 3 epochs, whole-record. ⚠️ **3,906/5,195 train records kept (75.2%)**, val 233/323 — the rest exceed the 1B's 8,192 and are DROPPED, not chunked. ⚠️ The filter is **confounded with product type**: it shifts the real-core mix from 50% to 64% T3PKS, and the median of the fitting subset is 1,170 nt vs 2,103 nt for the whole split. ⚠️ **The kept records are ~59% chalcone-synthase (T3PKS, single gene) and ~31% ketosynthase (T1PKS-type modular), n=150 sample — so this adapter is T3PKS-DOMINATED and must never be described as 'generates PKS clusters'** (prereg §2.2). ✅ **TRAINED 2026-08-19: 732 steps / 3 epochs, train loss 0.794→0.753, best val 0.8635, 1h44m.** `adapter_run/final_adapter`, `train.log`, `train.whole.jsonl`, `val.whole.jsonl`. Pre-registered in `docs/phase6_PKS_preregistration.md`. `A0.jsonl` / `A0-C1.jsonl` / `A0-C2.jsonl` (200 each, `--junk-policy mask`) + `<arm>_w<win>_<CLS>.json` scorings. | ✅ **A0 SIGNIFICANT** |
| `phase7_TERPENE/` | 08-19 | **[P7-A0] STRICT-span TERPENE adapter.** Same recipe with `CLASS=TERPENE`, `DATA=splits_class/TERPENE`. **10,658/11,297 train records kept (94.3%)**, val 747/793. ✅ **TRAINED 2026-08-20: 1,998 steps, best val 0.8417, 4h13m.** ⚠️ **step 1,998 is at the `--max-steps 2000` cap** — 10,658/16 = 666 steps/epoch × 3 = 1,999, so 3 epochs completed with 1 step of headroom. A larger class would have been silently truncated mid-epoch. `adapter_run/final_adapter`. Pre-registered in `docs/phase7_TERPENE_preregistration.md`. `A0.jsonl` / `A0-C1.jsonl` / `A0-C2.jsonl` (200 each, `--junk-policy mask`) + `<arm>_w<win>_<CLS>.json` scorings. | ✅ **A0 SIGNIFICANT** |

⚠️ **Phase 6 and Phase 7 numbers are NOT comparable to each other or to Phase 3.** Three scoring
axes differ by design: class marker set, window (PKS **4,000** · TERPENE **2,000** · RIPP 2,000) and
antiSMASH `--minlength` (TERPENE **200**, everything else the 1,000 default). Cross-class reading is
of *shape* — does an intervention move the same direction — never of magnitude.

### Phase 5 — component detection

| Run dir | Date | Contents | Status |
|---|---|---|---|
| `phase5_classprobe/` | 08-19 | **[P5-CLASSPROBE] cross-class substrate comparison.** `real_RIPP_50.jsonl`, `real_PKS_50.jsonl`, `real_TERPENE_50.jsonl` — 50 held-out test cores per class, drawn `random.Random(0)`; `real_<CLASS>_50_w4000.json` — full reporting set at a **4,000 nt** window. Result: **PKS reaches `n_class_domains` >= 2 in 37/50 real cores (0.740) vs RIPP 0.200 and TERPENE 0.220** — PKS is the only built class whose training span carries cluster-grade domain content. Integrity guard clean (50/50 unique each). **Extended 08-19:** `real_<CLASS>_fit50.jsonl` + `real_<CLASS>_fit50_w<window>.json` — the **fits-the-1B (<=7,992 nt) ceilings**, which are the correct references for Phase 6/7 because that is the population each adapter trains on. `as_real_PKS/` + `as_real_PKS.tsv` — full-mode antiSMASH, **49/50 = 0.980** detected, product mix T3PKS 26 · T1PKS 20 · PKS-like 3 · T2PKS 2 · transAT-PKS 1. `as_real_TERPENE/` (default minlength, **23/50 could not run**) and `as_real_TERPENE_ml200/` + `.tsv` (**50/50 ran, 50/50 detected**, `terpene-precursor` 28 · `terpene` 22). | ✅ |
| `phase5_detect/` | 08-19 | **[P5-DETECT] full-mode antiSMASH.** `as_full/` — 24 real RIPP wide spans (12 mixed subclass, 12 module-covered) establishing precursor sensitivity **8% vs 50%**. `ab/` — the `--minimal` vs full A/B on identical sequences (**100% agreement on `is_bgc`**, so no prior number is retracted). `full_arms/` — 180 sequences across 5 arms in full mode, which produced the **subclass-specificity** finding (real ~70% specific, our best arm **0%**). Output dirs **retained**, unlike the `TemporaryDirectory` used by `evaluation.py:check_antismash`. | ✅ |

⚠️ **All prior antiSMASH results (833 sequences) used `--minimal`** — analysis modules disabled, so
**RODEO never ran** — and output went to a `TemporaryDirectory` that is deleted after `is_bgc` and
`class_match` are read. Those results are detection-only and cannot be mined for precursors.

### Phase 4 — WIDE vs STRICT span-width comparison

| Run dir | Date | Contents | Status |
|---|---|---|---|
| `phase3_RIPP_wide/` | 08-18 | **W-1, the WIDE adapter — REFUTED, significantly worse.** Trained on `splits_class_wide/RIPP` (3,723 whole records ≤L, 3 ep / 675 steps, `loss_ce` 1.309→0.844). `adapter_run/`; `wide_8k.jsonl` (150 de novo @8 kb, **uninformative — underpowered**); `W1_seeded_[a-d].jsonl` → `W1_seeded.jsonl` (188 seeded @L=8 nt, 2.2 kb). | ✅ |
| `phase3_RIPP_strictmatched/` | 08-18 | **W-2, the size+cluster-matched STRICT control.** Trained on `splits_class_strictmatched/RIPP` — the **same 3,723 rows** as W-1 with strict spans, so only span width differs. `adapter_run/`; `W2_seeded.jsonl` (188 @2.2 kb), `W2_seeded8k.jsonl` (188 @8 kb). **Corrected rate 0.043 / 0.085 — beat W-1 at Holm p=4.1e-04 / 3.2e-05.** | ✅ |
⛔ **THE FIVE SEEDED GENERATION SETS BELOW CARRY DUPLICATE RECORDS — see `bugs.md`, fan-out shard
collision (2026-08-19). Use the stated EFFECTIVE n, never the line count.**

| generation set | records | **effective n** |
|---|---|---|
| `phase3_RIPP/SF_seeded8k.jsonl` | 188 | **47** |
| `phase3_RIPP_strictmatched/W2_seeded.jsonl` | 188 | **47** |
| `phase3_RIPP_strictmatched/W2_seeded8k.jsonl` | 188 | **47** |
| `phase3_RIPP_wide/W1_seeded8k.jsonl` | 188 | **47** |
| `phase3_RIPP_wide/W1_seeded.jsonl` | 188 | **141** |

The per-shard files (`*_a.jsonl` … `*_d.jsonl`) are individually clean at 47 records each; for the
four 47-effective sets, shards b/c/d are byte-identical copies of shard a. **All Phase-3 sets
(`A0_*`, `ctrl_*`, `pilot_*`, `S2-*`, `s1_*`) were audited and are CLEAN.**

| `phase3_RIPP_widecmp/` | 08-19 | **[P4-WIDE-SEEDED] analysis — COMPLETE (`PIPELINE_OK`).** 11 × `<arm>_w<window>_RIPP.json` (full reporting set), `antismash_widecmp.tsv` (615 calls), `corrected_rates.json`, `as_<arm>_{pos,neg}.jsonl` stratified inputs, `logs/`. Generations live in each adapter's run dir. | ✅ |

### Phase 3 — Stage 2 confirmatory arms

| Run dir | Date | Contents | Status |
|---|---|---|---|
| `phase3_RIPP_stage2/` | 08-18 | **Stage 2 of [P3-B1-EXP] — COMPLETE**, pre-registered §8.5. Five arms × **n=188** (188 of 200 test prompts are ≥ seed_nt+500) at **L=8 nt**, `--no-boundary-orf`, TEST seeds, `--seed 11`, 2,200 nt, `evo2_1b_base`. `S2-1` LoRA **0.176** · `S2-2` general **0.000** · `S2-3` base **0.000** · `S2-4` LoRA+shuffled **0.186** · ~~`S2-5`~~ mismatch-tag **no-op, uninformative**. Files `S2-<n>.jsonl`, `S2-<n>_w2000_RIPP.json`, `S2-<n>_gen.log`, `S2-<n>_score.log`. | ✅ |

### Phase 3 — seed sweep

| Run dir | Date | Contents | Status |
|---|---|---|---|
| `phase3_RIPP_seedsweep/` | 08-17 | **Stage 1 of [P3-B1-EXP] — COMPLETE.** Exemplar-seeded generations, seed length ∈ {4, 8, 20, 100, 500} nt × {RIPP LoRA, base 1B}, n=50/cell, seeds from `splits_class/RIPP/val_prompts.jsonl` (60 val records, len ≥ 1000, 0% genome overlap with train). 10 × `s1_<model>_L<len>.jsonl` + 10 × `s1_<model>_L<len>_w2000_RIPP.json` (full Phase-3 reporting set) + `.log`. Generation 2,200 nt, `--seed 7`, substrate `evo2_1b_base`. **Result: L\* = 8 nt.** base 1B **0/50 at every length**; lora 0.140/0.160/0.100/0.100/0.240. ⚠️ Run **without** `--no-boundary-orf`, so the L=500 cells reconstruct the seeded cluster (12/12 domain match) — see `memory.md` 2026-08-17. `.claims/` holds the fan-out claim dirs. | ✅ |

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

⚠️ **Phase 8 opens on this track (`docs/phase8_GENOMEOCEAN_preregistration.md`, 2026-08-20).**
Arm = fine-tuned `GenomeOcean-4B` on `splits_class/TERPENE`; `bgcFM` is a declared reference, not a
control. Both checkpoints are local in `hf_cache`; env `/data2/ds85/envs/genomeocean`.
Class probe: base **0.878**, bgcFM **0.894**, chance 0.091 (`genomeocean/experiments/`).


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

### `_unfiled/` — the holding pen for unattributed artifacts

**59 orphaned files moved here 2026-08-17** (4.3 MB): logs, one-off JSONs and a shell script that
were written directly to the runs root and have no owning experiment. Kept, not deleted — several
are real measurements whose provenance is merely unknown, and deleting a measurement is worse than
failing to file it.

**11 files remain at the root deliberately.** Each is the named default output of exactly one
script — `ladder_audit.json` ← `ladder_audit.py`, `direction_audit.json` ← `direction_audit.py`,
and so on — so they are attributable by name even though they sit outside a run dir. Filing them
means changing 11 default output paths; queued as [P3-B5] rather than done silently.

⚠️ **Do not add to either.** The `CLAUDE.md` naming convention requires new artifacts to live in a
run directory.

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
