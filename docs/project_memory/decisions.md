# Decisions — why the project is built the way it is

Architecture and approach decisions with their rationale. Newest context at the top of
each topic. See also [progress.md](progress.md) (current state) and [bugs.md](bugs.md)
(quirks/fixes). Full historical detail: `docs/archive/REDESIGN_PLAN.md`.

---

## Modelling

### [2026-07-13] Rank sweep closes the capacity question — expressiveness is not the limiter either
r=16/64/128 on mega_all (α=2r, n=15): correct_class 0.067 / 0.067 / **0.0** — no rank lifts the
functional gate; r=128 is worse (over-rank + α–r=2 over-shrink → the rsLoRA regime). r=64 gave a real
domain-marker bump (class_markers 0.133→0.267) that, like every other lever, did NOT convert to
correct_class. With probe B (coverage, flat), **LoRA capacity — both which layers and how expressive
— is ruled out.** This was the last cheap lever. The signature across the whole program is now
unmistakable: interventions move class-appropriate DOMAINS but never assemble a valid correct-class
cluster — strongly suggesting LoRA-conditioned Evo2 is near its ceiling for de-novo megasynthase
generation. Structural fork remains: long-context (multi-GPU) / full-or-partial FT / reposition Evo2
as an evaluator-scorer (where the recalibrated antiSMASH+eval stack is already strong).

### [2026-07-12] Option A real whole-core run FAILED — cheap fixes exhausted; whole-core-only starves the data
The milestone-gated mega-only whole-core run (L=32768, fresh-from-base) auto-killed at epoch 4:
correct_class 0.133 (2/15) at step 120 → **0.0 (0/15) at step 240**, with modules/obligate/is_bgc all
declining. More training made it WORSE — consistent with overfitting the small whole-core set
(80 Mbp; whole-core@L=32768 drops 62% of megasynthase nt = the long multi-module cores). Every cheap
lever now tested & failed: LoRA coverage(B), imbalance(D), chunk-label(P-tag), gene-aware,
whole-core-at-scale. Remaining = structural: long-context (multi-GPU, keeps all data whole) / higher
rank (see 2026-07-13, also negative) / full-vs-LoRA FT / reposition as evaluator.

### [2026-07-10] n=15 re-eval: C's correct_class win was small-n noise; whole-core helps DOMAINS, not the GATE
Re-evaluating P0 / mega_all / C at n=15 (was n=6) collapsed C's headline: **correct_class = 0.067
(1/15) for all three** — tied at the floor. The robust surviving effect is a **domain-level
gradient C > mega_all > P0** (class_markers 0.33/0.13/0.07; modules 0.27/0.13/0.07; obligate
0.147/0.072/0.044): whole megasynthase cores make the model produce ~3–5× more class-appropriate
obligate domains / partial modules; concentration (mega_all>P0) adds a smaller bump. But it does
**not** assemble into a valid correct-class cluster. **Implications:** the cheap 350-step probes
are exhausted; de-chunking and LoRA capacity are out; whole-core + concentration are weakly
supported ON DOMAINS but UNPROVEN on the functional gate. A real multi-epoch mega-only whole-core
run is the only way to test whether domain gains convert to correct_class — **but note the
whole-core ∩ feasible-L tension:** mega cores carry 209 Mbp, and whole-core-only training keeps
just **79 Mbp at L=32768 (drops the long assembly lines = 62% of the nt)** or 142 Mbp at L=65536.
The long multi-module cores — the ones we most want — don't fit at single-GPU L. Milestone-gate
any such run (kill if correct_class is flat by ~epoch 2–3); do not repeat the first run's mistake
of training to 1,200 steps before looking.

### [2026-07-09] Gene-aware chunking REFUTED — the lever is likely concentration and/or context length, not chunk boundaries
The gene-aware A/B (blind vs snap-to-gene on the same long-mega cores) showed **gene-aware does
not help** (`ga_geneaware` flat at 0 on every marker; `ga_blind` got class_markers 0.333 / module
0.167). So keeping *genes* whole doesn't recover C's benefit — a long cluster is still fragmented
across windows. Two things this forces us to confront:
- **C's win is confounded** (mega-only × whole × short ≤16k), and the *reliable* fact is that a
  mega-only probe shows life while the all-classes production run does not.
- **Crucially, at the production L=32768, ~79% of mega cores ALREADY fit whole** (see the length
  distribution) and the full run still produced correct_class=0. So "just make cores fit whole"
  (via longer L) is **not clearly sufficient** — the difference between the failing run and the
  C probe is at least as much **mega-only concentration** as whole-vs-chunked.
**Working stance:** treat long-context (larger L) as *one* lever for the ~7-21% of mega cores that
don't fit at L=32k, but do NOT assume it fixes conditioning on its own. The cleaner untested
variable is **training predominantly/only on the megasynthase classes at production scale** (C
was mega-only; D up-weighted to only 53% and stayed chunked). Recommend isolating concentration
next, and running any real conclusion at n≥15, not n=6.

### [2026-07-07] Probe sweep: DE-CHUNKING is the lever — overturns the diagnosis ranking
Four fast probes (350 steps, L=16384, fresh-from-base, vs a shared P0 control;
`probes_20260706/probe_summary.tsv`) tested the diagnosis fixes:
- **B** (unfreeze the frozen Hyena long-range input projection via `--lora-target-parameters
  projections.weight`) came out **identical to control (all functional gates 0)** → **LoRA
  capacity/coverage is NOT the bottleneck**, overturning the 2026-07-03 diagnosis's #1
  "leading suspect".
- **C** (train on **whole megasynthase cores**, no chunking) was the **only** probe to lift the
  gates — correct_class 0.33, class_markers 0.50, obligate_fraction 0.18, module_count 0.17.
- **D** (megasynthase upweighted to 53% but still full-length/chunked) stayed **flat (0)** → more
  mega data doesn't help if fragmented.
**Conclusion:** the lever is the **training signal** — the model must see the **complete
assembly line under its class label**. Chunking (diagnosis Lane 2, rated only "contributing")
is the primary cause; LoRA capacity (Lane 5, "leading suspect") is unsupported. **Next:
gene-aware chunking / whole-core training** (persist per-gene coords from the GBKs — parser
already exists in `build_core_records.py` — and snap chunk cuts to gene gaps); genes longer
than the window still need larger `L`. Caveats: n=6/probe, undertrained; C confounds
whole-core × mega-only × short — a long-mega chunked-vs-whole probe would fully isolate it.

### [2026-07-03] Stopped the 6-epoch continuation at step_1200 — conditioning failure is structural, not under-training
The continuous-resume run (step 400→1200, ~2 extra epochs) was launched on the hypothesis
that Phase-1 was under-trained. The step_1200 functional eval (pooled n=21, two decoding
temps) **falsified that**: `correct_class` stayed **0/21** and `module_count` **0/21** while
`is_bgc` sat at ~14% — and every antiSMASH-positive hit was a SIMPLE class (ectoine/terpene),
never the conditioned megasynthase (NRPS/PKS/hybrid). Robust to more samples and to lower
decoding temperature; val loss was flat the whole run. So the model emits generic gene-dense
DNA that occasionally forms an easy cluster but never builds the conditioned class's core
assembly-line machinery. **Decision:** halt the 6-epoch run (≈7 more days for no evidential
gain) and diagnose the root cause (data signal / long-gene chunking / class imbalance /
train-vs-gen prefix / LoRA capacity / gen-window truncation). This directly challenges the
"surface results = low training, not LoRA capacity" claim in the next section — capacity,
chunking, and conditioning strength are now live suspects.

### [2026-07-03] Diagnosis of the conditioning failure — data + prefix RULED OUT; leading suspect is frozen long-range (Hyena) adapter coverage
A 6-lane read-only diagnostic (multi-agent workflow) established:
- **RULED OUT — training-data signal.** 24/24 sampled NRPS/PKS/HYBRID cores carry their
  obligate domains (NRPS PF00501/PF00668; PKS KS/AT/ACP) with real multi-module architecture,
  positionally within the first 32k window. Labels are fine.
- **RULED OUT — train-vs-generation prefix.** The prefix builders are byte-identical (class,
  GTDB tag, delimiters, `|END|`, `|CONTINUATION|`, tokenization). The class tag reaches the
  model exactly as trained.
- **LEADING SUSPECT (structural) — LoRA coverage/capacity.** The Hyena input projection
  (`self.projections`, a TELinear producing the x1/x2/v gating streams into all 27 long
  convolutions — the long-range token-mixing pathway) is UNADAPTED/frozen; the long-filter
  params are never LoRA-eligible. Of 28.7M adapter params ~81% sit on position-wise MLPs
  (l1/l2/l3) that cannot mix across positions; only ~6.8% touch attention (just 5 of 32
  blocks). So LoRA adjusts local content but barely touches the long-range coordination an
  assembly-line module *is* — matching the symptom (accessory domains appear; ordered
  multi-domain modules never do). Directly refutes the prior untested "ample capacity" claim.
  **Full mechanism** — how LoRA attaches, the Hyena-block dataflow, why the conv kernels can't
  take LoRA, TELinear vs nn.Linear: [../evo2_lora_and_hyena.md](../evo2_lora_and_hyena.md).
- **CONTRIBUTING — chunking.** Megasynthase cores are heavily fragmented (HYBRID only 56% fit
  whole in the class-start window; interior windows train under `|CONTINUATION|`, not
  `|COMPOUND_CLASS|`). But 44–86% DO fit whole and still yield 0 modules → not sufficient alone.
- **CONTRIBUTING (amplifier) — class imbalance.** simple:mega record ratio 2.62:1 (mega is
  nucleotide-majority though) → mild pull toward easy attractors (ectoine/terpene).
- **EVAL CONFOUND to control — gen window.** 13/21 gens hit the 32k cap; but 8/21
  self-terminated far below it (390–9311 nt) with 0 obligate domains / 0 modules, and
  megasynthase-SIZED single ORFs (up to 10,793 aa) still scored module_count=0 → truncation is
  a confound to control for (re-run with `--max-windows 3-4`), not the root cause.

### LoRA adapters, not full fine-tuning
Evo2 7B is already pretrained on a huge genomic corpus; Phase-1's job is mostly to teach
the **conditioning interface** (LIMA-style: a small, high-quality adapter on top of broad
pretraining), not to relearn biology. LoRA (r=16, α=32, ~28.7M trainable ≈ 0.44%) keeps
the base frozen, fits on one H100, and avoids catastrophic forgetting of Evo2's sequence
prior. Full FT would be far more expensive and risk degrading the prior with our
comparatively tiny dataset. The embedding + LM head are frozen (we condition via a text
prefix, not new tokens). **Surface-level early results are expected from low training, not
from LoRA capacity** — the adapter has ample capacity for the conditioning task.

### Native lowercase GTDB taxonomy tags
Evo2 was pretrained with lowercase GTDB lineage strings
(`|d__Bacteria;p__Pseudomonadota;…;s__Escherichia coli|`). Our original data used
UPPERCASE_underscore tags, which are **out-of-distribution** and waste the conditioning
signal. We switched to native lowercase GTDB tags so the taxonomy prefix lands in-domain.

### Prefix-masked loss
The CE loss is masked over the conditioning prefix
(`labels[:, :prefix_token_count] = IGNORE_INDEX`); only the BGC nucleotide half trains.
This matches the intent ("generate a sequence *given* a fixed prefix") and makes absolute
loss values incomparable to pre-masking runs.

### Context L=32768, micro-batch `bs=1 ga=128`
On the 80 GB H100, `L=32768` is the conservative ceiling that passes with margin
(`65536` near-limit, `98304` OOMs). The only shape that fits at 32k is
`--batch-size 1 --grad-accum 128` (effective batch 128; `bs=4 ga=32` OOMs — see
[bugs.md](bugs.md)). Block-level activation checkpointing is default-on; the
no-checkpoint path is not viable above short contexts.

### Sequential generation (batched path gated off)
vortex silently de-batches mixed-length prompts, and the left-pad workaround perturbs
StripedHyena (empirically failed an on-GPU equivalence gate). So `generate_bgc.py`
defaults to **sequential** generation; the batched path stays behind a validated gate.

---

## Data

### Strict core-region trimming
We train on the **strict core** = the contiguous span of `gene_kind="biosynthetic"` CDS
(from antiSMASH `gene_kind`), not whole clusters or guessed windows. Rationale: focus the
model on the actual biosynthetic machinery, make ~88% of clusters single-window (so most
examples fit one context), and avoid diluting the class signal with flanking/regulatory
DNA. Cores are re-extracted from re-acquired antiSMASH-DB GBKs so boundaries are official,
not guessed.

### Group-aware (genome-keyed) splitting
The original record-level split leaked badly (94.6% genome overlap, 453 byte-identical
seqs across splits). Splits are now **genome-disjoint** (`split_dataset_grouped.py`) +
exact-md5-disjoint + cross-split MMseqs2 near-dup removed (`dedup_core_splits.py`). This
is what makes the novelty/positive-control comparisons meaningful.

### MiBIG held out for Phase-2
Near-dups of the 2,636 MiBIG BGCs are removed from training
(`exclude_mibig_from_core.py`). Reason: keep MiBIG as a genuinely held-out positive
control now, and **reserve it for a Phase-2 compound-conditioned fine-tune** (condition on
compound name, not just class — the eventual goal).

---

## Evaluation (rewritten from first principles, 2026-06-17)

### Named CHECKS → QUESTIONS, not metric_1..metric_11
The old flat metric numbering was a grab-bag of mixed value. The suite is now two layers:
**checks** (compute units) combined into **questions** (what we actually want to know:
is_bgc, correct_class, novel = gates; proteins_plausible, complete, conditioning_faithful
= diagnostics). The suite is deliberately scoped to *is-it-a-BGC / correct-class /
plausible / novel / complete* — the **wet-lab axes (synthesizability, E. coli
expressibility) were pruned** as out of scope.

### antiSMASH is the `is_bgc`/`correct_class` gate
Sequence quality alone does **not** verify "is this a BGC" — a region of housekeeping
genes would pass it. antiSMASH is the purpose-built BGC detector/classifier, so it owns
both gates; `class_markers` (Pfam) is the fast **proxy** used only when antiSMASH is
skipped (quick-eval). antiSMASH was previously ~15% on real BGCs — the cause was an
**incomplete product→class map** (antiSMASH 8 emits 103 product types), not parsing or
core-trimming. Fixed by `build_class_map.py` (regenerates the map from antiSMASH's own
product→category grouping + overrides) → **≈0.97** detection and class-match on real cores.

### pyrodigal (Prodigal), not six-frame or FragGeneScan
The legacy six-frame ORF finder fragmented megasynthases (tanking PKS/NRPS detection).
Chose Prodigal (pyrodigal) over FragGeneScan: it is the standard prokaryotic caller and
the one antiSMASH itself uses (internal consistency), it is strict (a frameshift → partial
genes — honest, not masked), and it flags partial/edge-truncated genes. One consistent
caller now feeds every protein-based check.

### `coding_sanity` uses a complexity guard, not gene-completeness
Prodigal calls one long *partial* ORF even in degenerate GC-repeat junk (coding_density
~1.0), and a legitimate strict core can be a single edge-truncated megasynthase. So
"require a complete gene" both fails to catch junk and false-fails real cores. The junk
discriminator is **dinucleotide-entropy complexity** instead. And `is_bgc` trusts
antiSMASH detection when it ran (coding_sanity is only the floor/proxy).

### Data-driven class markers, ANY-of semantics
`class_markers` (formerly M2) uses per-class Pfam markers **derived from real cores**
(`derive_class_markers.py`), not a textbook list — this captures subtype diversity
(type-III PKS, carotenoid "terpene", NRPS-like). Pass = contains **any** class marker
("has the class's machinery"); module *completeness/order* is owned separately by
`module_architecture`.

### quick_eval runs antiSMASH; full eval adds homology + novelty
antiSMASH is cheap (~3 s/core), so per-checkpoint `quick_eval` runs it for the real
`is_bgc`/`correct_class` signal and skips only the DB-bound/slow checks (`protein_homology`,
`kmer_novelty`).

---

## Open architectural fork (undecided)

**Per-class adapters vs one conditional model (Step 2).** Whether to train one
conditional adapter (current v2) or per-class adapters depends on whether class
conditioning is actually being used.

**Update 2026-06-24 (diagnostic run on v2 `best`, step 400):** the conditioning *is*
being used but **weakly**. Stochastic diagnostic: composition cross/within **ratio 1.08**
(class-differentiated), class-appropriate obligate domains for NRPS (0.056; PKS/TERPENE 0),
GC healthy (0.62–0.71) — script verdict "CONDITIONING WORKS". But the functional gates from
quick_eval are at the floor (`is_bgc=0`, `correct_class=0`, n=3): the model does not yet
build complete, antiSMASH-recognizable class machinery. This is a clear step up from the
2026-06-04 pilot ("CONDITIONING DEAD", ratio ≈1.0). **Decision deferred:** before splitting
into per-class adapters, first rule out under-training (early stop fired at epoch ~0.97 with
val loss still drifting) and the tiny-n eval — train longer / re-eval with more sequences,
then revisit. See [progress.md](progress.md) "Next actions".
