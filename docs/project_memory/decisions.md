# Decisions — why the project is built the way it is

Architecture and approach decisions with their rationale. Newest context at the top of
each topic. See also [progress.md](progress.md) (current state) and [bugs.md](bugs.md)
(quirks/fixes). Full historical detail: `docs/archive/REDESIGN_PLAN.md`.

---

## Modelling

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
