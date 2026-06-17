# Bugs & quirks — and the proven fixes

Recurring errors, environment/tooling quirks, and what actually fixed them. Add an entry
whenever a non-obvious bug is solved. See [decisions.md](decisions.md) for rationale and
[progress.md](progress.md) for current state.

---

## Evo2 / vortex / generation

- **vortex silently de-batches mixed-length prompts.** `Evo2.generate(..., batched=True)`
  checks `uniform_lengths` and falls back to per-sequence generation for ragged batches.
  *Fix:* generate **sequentially** (`generate_bgc.py` default). Left-padding to equalize
  lengths was tried and **failed an on-GPU equivalence gate** (left-pad perturbs
  StripedHyena: head-token LCP ~0.004, byte divergence). Keep batched behind that gate.
- **`eos_id = 0` is the null byte** in `CharLevelTokenizer` — unusable as a stop token;
  `stop_at_eos` is effectively off, so generation always runs the full `n_tokens`.
- **Generation needs the free GPU.** It will not share the device with active training;
  pause/finish training first.

## Eval suite

- **antiSMASH was ~15% on real BGCs — it was MAP COVERAGE, not parsing.** Parsing of
  `records[].areas[].products` was correct; antiSMASH 8 just emits **103 product types**
  and the old `compound_class_map.yaml` covered a fraction, so real clusters
  (`NRP-metallophore`, `PKS-like`, `NI-siderophore`, …) mapped to OTHER and failed.
  *Fix:* `scripts/build_class_map.py` regenerates the map from antiSMASH's own
  product→category grouping + overrides → ≈0.97.
- **Prodigal calls one long PARTIAL ORF in GC-repeat junk** (`"GC"*8000` → coding_density
  ~1.0, max ORF ~455 aa). So a "require ≥1 complete gene" rule does **not** catch the
  degenerate collapse. *Fix:* `coding_sanity` discriminates junk with a
  **dinucleotide-entropy complexity guard** (`< 2.5 bits` → fail), not completeness.
- **`coding_sanity` false-failed a real NRPS core** that is one 3599-aa megasynthase
  (edge-truncated by strict-core trimming → Prodigal flags it `partial` →
  `complete_gene_fraction = 0`). *Fix (two parts):* (1) `is_bgc` trusts
  `antismash.detected` when antiSMASH ran — coding_sanity is only the floor/proxy; (2)
  coding_sanity dropped the complete-gene requirement (see above).
- **antiSMASH JSON parse error left `class_match` unset** → `derive_questions` used
  antiSMASH for `is_bgc` but the class_markers proxy for `correct_class` (inconsistent,
  possible false PASS). *Fix:* on parse error, mark the antismash result **`skipped`** so
  both questions fall back to the proxy consistently.
- **six-frame ORF finder fragmented megasynthases** → PKS/NRPS detection low. *Fix:*
  replaced with **pyrodigal** everywhere (`find_orfs`); six-frame kept only as a fallback.
- **`class_markers` reported 0 obligate domains on real NRPS/PKS/TERPENE cores.** Not a
  gene-finder or extraction bug — the marker list was textbook-only (missing carotenoid,
  type-III PKS, NRPS-like subtypes). *Fix:* data-driven markers (`derive_class_markers.py`)
  + ANY-of pass semantics. Validated ≈0.87 on real cores.

## Conditioning diagnostics

- **Greedy `top_k=1` generation collapses to all-GC** (gc≈1.0) — degenerate, useless for
  a conditioning signal. Use stochastic sampling.
- **12-mer Jaccard saturated** (within ≈ cross ≈ 1.0) — wrong metric. *Fix:* use
  **5-mer composition cosine** + **domain-set Jaccard** (within-class vs cross-class).

## Data pipeline

- **`splits_core` records initially dropped `training_text`** →
  `training_prefix_for_chunking` raises. *Fix:* add
  `training_text = canonical_prefix + sequence` to each record.
- **`curate_dataset.py --min-len 1000` deletes short single-enzyme cores** (e.g. ectoine
  ~400 bp). *Fix:* `--min-len 300` for the core build.

## Environment / tooling

- **`flash-attn` install order:** its `setup.py` runs `import torch` at build time, so a
  plain `environment.yml` create crashes (torch not yet installed). *Fix:* install torch
  first → prebuilt flash-attn wheel → re-run env update → deepspeed/peft/wandb
  (`docs/archive/gputee/FINETUNE_GUIDE.md §2`).
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is unsupported on this platform** —
  do not rely on it as an OOM fix.
- **`bs=4 ga=32` OOMs at L=32k** (forward fails at bs=4, backward at bs=2). Only
  `bs=1 ga=128` fits. Effective batch is still 128.
- **NCCL "process group not destroyed"** warning on shutdown is expected for short runs.

## Agent / shell self-traps (when operating the repo)

- **`pkill -f` / `pgrep -f` / process scans self-match the agent's own command string** →
  false "STILL RUNNING" positives and one accidental self-kill. *Fix:* match by PID, or
  exclude the current shell's own command.
- **Double-backgrounding** (`nohup … &` *inside* a `run_in_background` tool call) makes the
  harness fire "completed" immediately for the launcher. *Fix:* use `run_in_background`
  directly without an inner `&`, or attach a watcher.
- **Timezone confusion:** the queue log uses local **EDT**; `date -u` is **UTC** (4 h
  offset). A run looked stalled for "3.5 h" but was ~24 min in and healthy. Compare like
  for like.
