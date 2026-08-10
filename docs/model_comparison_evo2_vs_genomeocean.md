# Evo2 vs GenomeOcean for class-conditioned BGC generation

_Written 2026-07-27. Measurements in this doc were taken on gputee (1× H100 80 GB) against
`splits_core`, using this repo's own eval instrument — not quoted from either paper unless
explicitly marked "(their number)"._

Paper: Zhou et al., *GenomeOcean: An Efficient Genome Foundation Model Trained on
Large-Scale Metagenomic Assemblies*, bioRxiv [2025.01.30.635558](https://doi.org/10.1101/2025.01.30.635558)
· Code: [jgi-genomeocean/genomeocean](https://github.com/jgi-genomeocean/genomeocean)
· Weights: [pGenomeOcean](https://huggingface.co/pGenomeOcean) (BSD-3)

---

## 1. The one-paragraph answer

GenomeOcean is not a better Evo2; it is a **different shape of model that happens to remove
the three specific things that have blocked this project**. Evo2 is a 7B byte-level
StripedHyena with a `CharLevelTokenizer` — no trainable special-token slots, no usable EOS,
a bespoke `vortex` runtime, and a training step so expensive that we are pinned to
`batch_size=1` at 32 kb and n≈15 evaluations. GenomeOcean is a stock 4B `MistralForCausalLM`
with a real 4,096-entry BPE vocabulary, 5 special tokens, and a proper EOS. That means a
compound-class token can be a **genuine token with its own trainable embedding row**, which
is precisely the thing Evo2 structurally cannot offer and precisely where our conditioning
died (see `project_memory/decisions.md`, 2026-07-21). It is also ~5× cheaper per nucleotide
and ~3–8× cheaper per training step, which converts our n=15 evaluations into n=100+.

**It does not, however, solve class conditioning for free.** GenomeOcean's own BGC model
(`bgcFM`) is *unconditional* — it has no class handle at all, and their published class
results come from generate-massively-then-filter. The case for switching is that it is a
much better *substrate* to install conditioning into, not that conditioning comes built in.

---

## 2. Head-to-head

| | **Evo2 7B** (current) | **GenomeOcean-4B** |
|---|---|---|
| Architecture | StripedHyena 2 (hybrid conv/attn) | `MistralForCausalLM` — stock transformer decoder |
| Parameters | 7B | 4.25B (measured) |
| Tokenizer | `CharLevelTokenizer`, byte-level, 1 token = 1 bp | BPE, 4,096 vocab, **5.15 bp/token** (measured on `splits_core`) |
| Special tokens | none usable; no reliable EOS | `[UNK]`/`[CLS]`/`[SEP]`(EOS)/`[PAD]`/`[MASK]`; `N` = token 8 |
| Context (ours) | L = 32,768 tokens = **32.8 kb** | 10,240 tokens = **52.7 kb**; RoPE ceiling 32,768 tokens = **169 kb** |
| Runtime | `vortex` + DeepSpeed, bespoke | plain HF `transformers`; vLLM upstream |
| PEFT | LoRA works, but wrapping is fiddly (TELinear, `target_parameters`) | `get_peft_model` works out of the box |
| Native conditioning handle | GTDB lineage tag only — **no product-class prior** | none — but the vocabulary is extensible |
| Pretraining corpus | GTDB reference genomes (OpenGenome) | 645 Gbp metagenome co-assemblies |
| BGC-specialised checkpoint | none (we made ours) | **`bgcFM`**: base + 1.72M dedup'd SMC BGCs (43.5 Gbp) |
| License | Apache-2.0 | BSD-3-Clause |

---

## 3. Measured on our data and our hardware

### 3.1 Tokenization and context fit
`genomeocean/scripts/analyze_tokenization.py`, n=4,000 reservoir-sampled from
`splits_core/train.jsonl` → `genomeocean/experiments/tokenization_report.json`.

Compression is **5.15 bp/token** (median 5.15, range 4.57–5.60) — the paper's ~5× claim
holds on our BGC cores.

Fraction of records that fit **whole**, no chunking:

| | all classes | megasynthase (NRPS/PKS/HYBRID) |
|---|---|---|
| **strict cores** (what we train on today) | | |
| GenomeOcean @ 10,240 tok | 0.970 | 0.966 |
| GenomeOcean @ 32,768 tok (RoPE ceiling) | 0.997 | **1.000** |
| Evo2 @ 32,768 bp | 0.938 | 0.892 |
| **whole antiSMASH regions** (median 26.6 kb; mega 47.2 kb) | | |
| GenomeOcean @ 10,240 tok | 0.840 | **0.641** |
| Evo2 @ 32,768 bp | 0.576 | **0.000** |

**Read this carefully.** On strict cores the context advantage is modest (0.966 vs 0.892) —
context is *not* the differentiator on today's dataset, and it would be wrong to sell the
switch on that. The advantage becomes decisive only in the whole-region regime: the median
megasynthase *region* is 47.2 kb, so **Evo2 at any feasible single-GPU L can never see one
whole — literally 0% — while GenomeOcean sees 64% of them in a single window.** Probe C
(2026-07-07) found that seeing the complete cluster is the one thing that lifted
`correct_class` off the floor, and the follow-ups concluded the lever was "complete
*cluster*, not complete *genes*". That experiment is unrunnable on Evo2 at 1 GPU and
runnable on GenomeOcean today.

*(Whole-region token counts are estimated as `region_bp / 5.15`; the strict-core rows are
exact.)*

### 3.2 Fine-tuning feasibility
`genomeocean/scripts/probe_finetune_feasibility.py` →
`genomeocean/experiments/finetune_feasibility.json`. LoRA r=16 on all 7 projections,
plus trainable `embed_tokens`/`lm_head` (55.6M trainable, 1.31% of 4.25B), gradient
checkpointing on, bf16, real forward+backward.

| tokens | ≈ bp | peak GPU |
|---|---|---|
| 4,096 | 21 kb | 10.5 GB |
| 10,240 | 53 kb | **14.0 GB** |
| 16,384 | 84 kb | 17.6 GB |
| 32,768 | 169 kb | 27.0 GB |

Batch scaling at L=10,240: bs=1 → 14.0 GB, bs=2 → 19.8, bs=4 → 31.5, **bs=8 → 54.8 GB**.

Compare Evo2 (`CLAUDE.md`, audit 2026-05-14): at L=32,768 **only `bs=1` fits** on the same
80 GB H100 — `bs=2` fails on backward, `bs=4` on forward. So per micro-step GenomeOcean
carries **8 × 52.7 kb = 422 kb** of sequence against Evo2's **1 × 32.8 kb**, a ~12.8×
increase in nucleotides per step on identical hardware.

> Gotcha worth recording: `gradient_checkpointing_enable()` is a **silent no-op unless the
> model is in `.train()` mode** — transformers' `GradientCheckpointingLayer` gates on
> `self.training`. Before fixing that, L=10,240 OOM'd at 77 GB; after, it takes 14 GB.

### 3.3 Class tokens — the thing Evo2 cannot do
Gate result: **PASS**. `tokenizer.add_special_tokens()` accepts 22 `[CLS_<CLASS>]` tokens
(vocab 4096 → 4118), `resize_token_embeddings` covers **both** `embed_tokens` and `lm_head`
(`tie_word_embeddings=false`, so they are independent), and `[CLS_NRPS]` survives
tokenization as **a single atomic id (4096)** rather than being shredded into
nucleotide-like pieces. Adding the trainable embedding/lm_head copies costs essentially
nothing in memory (37.39 GB vs 37.48 GB at L=4,096 — within noise).

This is the structural contrast. On Evo2, `|COMPOUND_CLASS:NRPS|` is just bytes through a
byte-level tokenizer with no pretrained prior, so the LoRA has to install the entire class
concept through a low-rank bottleneck — which, across the probe programme, it never did
(`correct_class` 0.013 at n=75; base Evo2 0.00). On GenomeOcean the class is a first-class
vocabulary entry with a dedicated, fully-trainable 3072-dim embedding row and its own output
logit. That does not *guarantee* conditioning works, but it removes the specific mechanism we
identified as the failure.

---

## 4. How GenomeOcean actually does BGCs (and the honest caveat)

From the paper (§2.7, §4.1.3, §4.3.4) and `external/genomeocean/`:

- **`bgcFM` has no class conditioning.** Fine-tuned on 1.72M dedup'd SMC BGCs in two phases
  (16,000 steps @ 1,024 tokens, then 1,600 steps @ 10,240 tokens). No product label anywhere.
- **Generation is zero-shot**: the prompt is the literal `[CLS]` token and nothing else
  (`genomeocean/llm_utils.py` prepends `"[CLS]"` to every prompt; `--zero_shot` sets the rest
  to `""`). Token 8 (`N`) is suppressed at the logit level; EOS is token 2.
- **Their T1PKS result is a filtering result** (their numbers): 258,260 sequences generated →
  antiSMASH 7.0 → **11,123 positive (4.3%)** → 1,459 PKS → 1,044 T1PKS inspected for modules.
- Their second BGC application is **discovery, not generation**: subtract per-token loss of
  the base model from `bgcFM` across a genome, smooth over 1 kb, and call dips as candidate
  BGC regions.

**So the paper's impressive "correct T1PKS assembly-line architecture" figure is a
best-of-258,260 selection, not a conditioned generation.** Our project's goal —
"give me an NRPS" — is not something `bgcFM` can do as shipped. What the paper demonstrates
is that the *substrate* can produce well-formed long BGCs at a 4.3% base rate, cheaply
enough that filtering at that scale is affordable.

There is also no fine-tuning script in the public repo despite the package description
claiming "inference and fine-tuning", and the `gmeval` repo referenced in their Methods for
the BGC evaluation is **404 / not public**. So their eval pipeline is not directly
reproducible; ours substitutes for it.

### 4.1 We ran it — 24 sequences, scored on our own gate

`genomeocean/scripts/generate_bgc_go.py`, `creative_long` preset, prompt = `[CLS]` only →
`/data2/ds85/bgcmodel_runs/go_zeroshot_bgcfm/`. 1.156 Mbp, median 52.5 kb per sequence.

| | GenomeOcean `bgcFM` zero-shot | Evo2 v2_step1200 (conditioned) | base Evo2 |
|---|---|---|---|
| `is_bgc` | **3/24 = 0.125** | 3/21 ≈ 0.14 | 0.00 |
| `coding_density` | **0.900** | 0.893 | 0.606 |
| `coding_sanity` | 24/24 | — | — |

Products called: 1 NRPS, 2 RRE-containing (→ 1 NRPS + 2 RIPP). Composition is clean —
pure ACGT, mean GC 0.615 — with 1/24 low-complexity (83-bp homopolymer), which upstream
would have filtered and we deliberately did not.

**Superseded by the n=216 run below — read §4.2 before quoting anything here.**

**Do not over-read this.** (i) n=24 has a Wilson 95% CI of ~[4%, 31%] on 3/24, so it is
*consistent with* their 4.3% rather than a confirmation; n≥200 is needed for a rate.
(ii) Per-sequence rates flatter GenomeOcean because its sequences are 52 kb vs Evo2's
32 kb — per Mbp it is 2.6/Mbp vs Evo2's ~4.4/Mbp, i.e. Evo2 looks better on that
normalization. With 3 hits each, no normalization makes the difference significant.
The defensible claim is only this: **an off-the-shelf model with zero project-specific
training lands in the same range as our 50-hour fine-tuned Evo2.** That is a statement
about the starting point, not about which model wins.

### 4.2 Powered replication, n=216 — the rate does NOT reproduce, but the product mix is the story

`/data2/ds85/bgcmodel_runs/go_zeroshot_rate_n216/`, seed 20260727, `--cache-implementation
static`. 10.35 Mbp in 9,561 s = **1,083 bp/s** (3.4× the dynamic-cache run, not the ~9× I
first predicted). Median 51.2 kb, coding_density 0.908.

**`is_bgc` = 27/216 = 12.5%, Wilson 95% CI [8.7%, 17.6%] = 2.61 hits/Mbp.**
Their reported **4.3% sits outside this interval** — the method replicates, the number does
not. Ranked by how much I believe each explanation:

1. **Length.** All our sequences are ~51 kb; their sweep included `min_seq_len=1024` tokens
   (~5 kb). `is_bgc` is per-sequence, so shorter sequences mechanically score lower. Our
   2.61 hits/Mbp is the comparable quantity and they don't report theirs. This is §5's eval
   defect showing up as a literal reproducibility failure.
2. antiSMASH **8.0.4** (ours) vs **7.0** (theirs).
3. Our gate was deliberately recalibrated (~0.15 → 0.97 on real cores) and may be more
   permissive than their stock configuration.
4. They averaged over rep-pen [1.0–1.5] × temp [0.7–1.1]; we sat at one favourable point.

**Product mix (the part that matters for this project).** 27 hits: NRPS 11, RRE-containing
5, NRPS-like 4, **T1PKS 4**, transAT-PKS-like 2, + 4 singletons → in our vocabulary
**NRPS 15 / RIPP 8 / PKS 5 / TERPENE 1**, i.e. **74% megasynthase (20/27)**.

Set against Evo2 step_1200, where *conditioning on* NRPS/HYBRID yielded only simple classes
(ectoine, terpene), `correct_class` 0/21, `module_count` 0/21:

> **An unconditioned GenomeOcean produces megasynthases as its dominant output. A
> conditioned Evo2, explicitly asked for one, never produced a single one.**

That is the strongest substrate argument in this document. Two honest deflators: bgcFM's
SMC training corpus is itself NRPS/PKS-heavy, and 51 kb leaves room for an assembly line
that 32 kb does not — so some of this is corpus and context, not model quality.

---

## 5. What this means for this project

The project's live blocker (`project_memory/progress.md`) is that class conditioning fails on
Evo2 and the open question is whether the model *represents* compound class at all. Against
that, GenomeOcean changes the following:

**Removes:**
- the no-trainable-class-token problem (§3.3) — the diagnosed root cause
- the `bs=1` training-throughput ceiling (§3.2) — 12.8× more nucleotides per step
- the n=15 evaluation ceiling — cheaper generation makes n=100+ affordable, which matters
  given how many of this project's conclusions have been overturned by increasing n
  (C's 0.33 → 0.067; "terpene 1/4" → 0/15)
- the "whole megasynthase region never fits" wall (§3.1)
- the bespoke `vortex`/DeepSpeed stack — LoRA, CFG with proper class-dropout, per-class
  adapters and guided decoding are all standard HF operations here

**Does not remove:**
- the possibility that class simply isn't learnable from ~47K cores. The **class linear
  probe** queued as next action in `progress.md` is still the right first experiment — and
  it is *cheaper* to run on GenomeOcean (forward passes only, 4B, 5× fewer tokens).
- the need to actually build a conditioned fine-tune; `bgcFM` gives us a BGC-specialised
  starting point but no class handle.

**New things it enables that Evo2 could not:**
- start from `bgcFM` rather than a general genome model — a BGC-specialised init we never had
- the contrastive loss-score scan (base − bgcFM) as a *discriminator*, which is exactly the
  "fast external class scorer" the Arc-style guided-decoding plan needs
- whole-*region* training, and a path to 169 kb contexts on a single GPU

**Costs / risks:**
- 4B < 7B; less capacity, and GenomeOcean is metagenome- rather than GTDB-trained, so our
  native lowercase GTDB taxonomic tags have **no pretrained meaning** here. The taxon
  conditioning that does work on Evo2 would have to be re-installed (or dropped).
- BPE means a class token sits in a vocabulary the model has strong priors over; token-level
  novelty/memorization checks need re-calibrating for BPE.
- `bgcFM`'s training set (SMC) overlaps antiSMASH-derived BGCs, so **leakage against our
  `splits_core` is unquantified** and must be measured before any novelty claim.
- vLLM is not usable on gputee as installed (CUDA 13 wheels vs 12.9 driver) — see
  `genomeocean/README.md`.

### 5.1 UPDATE 2026-07-28 — the linear probe answered a different question than expected

The probe was meant to decide substrate. It didn't, because **both models separate compound
class strongly** (balanced accuracy, 11 classes, chance 0.091):

| model | best layer | balanced acc |
|---|---|---|
| Evo2 base | 16 | **0.911** |
| Evo2 v2 LoRA step_1200 | 16 | 0.906 |
| GenomeOcean bgcFM | 12 | 0.894 |
| GenomeOcean base-4B | 8 | 0.878 |

Confounds handled on the GenomeOcean side: taxonomy is a *shallow* property (phylum 0.657 at
layer 0, saturating by layer 4) while class is *computed* (0.345 -> 0.894 over 12 layers), and
class survives taxon stratification in three phyla (Pseudomonadota 0.907, Bacillota 0.948,
Actinomycetota 0.954).

**Do not over-read Evo2's 0.911 vs GenomeOcean's 0.894.** `--max-nt 4096` holds the biology
constant but gives Evo2 4,096 pooled positions against GenomeOcean's ~795 BPE tokens (and 32
blocks vs 24). Evo2's higher layer-0 floor (0.486 vs 0.345) is that confound made visible. Read
this as "both strongly separable", not a ranking.

**What it changes.** The single strongest argument for migrating — "GenomeOcean can represent
class and Evo2 can't" — is falsified. Evo2 represents class fine; it just can't be *steered* by
a prefix. That makes the cheapest decisive experiment **steering on Evo2 step_1200**, a model
already trained and evaluated: guided decoding scored by this probe, or activation addition at
block ~16. If that moves `correct_class` off the floor, no migration is needed.

GenomeOcean's case now rests entirely on the remaining axes — trainable class token, ~12.8x
nucleotides per micro-step, 51 kb context, unconditional megasynthase output, vLLM-ability.
Those are real and they matter for a *conditioned retrain*. They are no longer decisive on
their own.

> ### ⛔ UPDATE 2026-08-10 — the steering experiment above was run, and it FAILED
>
> Phases 0–6 of `docs/steering_program.md`: corrected length-stripped directions, dose in
> class-units, layer treated as a variable (16/20/24/27), multi-layer stacking, all with
> shuffled-label controls and a continuous readout 10x more sensitive than any binary gate.
> **Null throughout.** The mechanism is now identified: the class direction reliably **deletes**
> a class that is present (ΔP(seed) −0.308 vs a shuffled control, p = 0.0063) and never
> **installs** the target's. The model *represents* class; the generator does not *consume* it.
>
> "If that moves `correct_class` off the floor, no migration is needed" — it did not. So the
> conditional resolves the other way, and **GenomeOcean's remaining axes are decisive again**,
> the trainable class token above all: it is precisely the handle Evo2's byte-level tokenizer
> cannot provide, and "install a class handle from scratch through a low-rank bottleneck" is
> exactly what failed. Ranked against the Evo2 alternatives (per-class soft prefixes, per-class
> adapters) in `docs/project_memory/progress.md` → NEXT ACTIONS (2026-08-10).

### Recommendation (superseded in part by §5.1 — read that first)
Do **not** discard the Evo2 track — it is the incumbent with a full negative result and the
eval instrument is shared, so a head-to-head is cheap. But **run the next diagnostic on
GenomeOcean, not Evo2**: the class linear probe (`progress.md` next-action 3b) costs forward
passes only and answers the gating question — *is compound class linearly decodable from the
model's hidden states?* — on the substrate we would actually build on. If it separates on
GenomeOcean-bgcFM and not on Evo2, that is a decisive, cheap reason to switch. If it fails on
both, the problem is the data, and no amount of model swapping fixes it.

---

## 6. Reproducing

See [`../genomeocean/README.md`](../genomeocean/README.md) for the exact commands and the
environment split (generation in `genomeocean`, antiSMASH scoring in `bgcmodel`).
