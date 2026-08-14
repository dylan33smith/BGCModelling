# The 1B track — Phase 2 (closed), and the Phase-3 TESTING SUBSTRATE

> **STATUS 2026-08-14.** Phase 2 closed here: objective changes do not move de novo biosynthetic
> content, and neither does 5x the training budget (flat baseline curve, `progress.md`).
> **This model is now the designated testing substrate for Phase 3** — every method comparison runs
> here first; the 7B confirms anything publishable; GenomeOcean is held so method is not confounded
> with model. Phase-3 targets are ~1 kb, i.e. 1/8th of this model's context, which is a different
> regime from the one Phase 2 found it capacity-limited in.


**Opened 2026-08-12.** A clean track for the objective-change experiments, on a substrate small
enough to iterate on. Phase 1 (the 7B) answered *where the problem is*; Phase 2 asks *whether
changing the training objective moves it*, and that question needs many short training runs rather
than one long one.

---

## Why a separate track

Every remaining question in the B plan requires **training**. On the 7B a single cell of the
frame-aware × domain-weighted design is days, which makes a 4-cell factorial unrunnable and makes
any single null uninterpretable (was it the intervention, or did we simply not train long enough?).
A hypothesis test you can only run once is not a hypothesis test.

This track therefore keeps everything Phase 1 established and re-derives only what is
**model-specific**. Nothing here re-litigates the conditioning programme; that is closed and lives
in `docs/project_memory/`.

## What is REUSED from Phase 1, unchanged

These are model-agnostic — they consume a generations `.jsonl` and know nothing about which model
wrote it — so they are called directly rather than copied:

| what | where | why it transfers |
|---|---|---|
| eval suite (checks → questions) | `src/bgc_pipeline/evaluation.py` | scores sequence, not models |
| the validated **ladder** | `evo2/scripts/ladder_audit.py`, `score_ladder.py` | AUROC-validated on 7B data, but the metrics are sequence properties |
| **novelty guard** | `scripts/memorization_check.py` | containment vs the training corpus |
| per-domain nucleotide spans | `splits_core/train.domain_spans.jsonl` | a property of the DATA, not the model |
| positive / negative controls | `scripts/make_*_control.py` | real cores and real non-BGC windows |
| length ceilings, class map | `docs/project_memory/`, `config/` | measured on real DNA |

## What MUST be re-derived (model-specific)

Anything whose value depends on which network produced it. Phase-1 numbers are carried only as
**context**, never as a baseline for a Phase-2 comparison:

- next-base cross-entropy on real cores
- the ladder on de novo generations (`best_bio_bits`, `n_bio_domains`, `bio_span_frac`, …)
- base-vs-fine-tuned deltas
- anything about what the model *generates*

## Substrate: what actually happened

**`evo2_1b_base`** — 1.108B params, 25 blocks, hidden 1920, native context **8,192**, 4 attention
blocks (3/10/17/24) to 21 Hyena. **Transformer Engine is required.**

**The TE story, in order, because two steps of it were wrong:**

1. Project docs said for weeks that no small model existed because the 1B "requires TE/FP8".
2. The refusal in `evo2.models` is a **name check** — `if "7b" in model_name: bf16 fallback, else
   raise` — and the fallback is model-agnostic, so applying it by hand makes the 1B **load**
   cleanly: 1.108B params, finite logits. It looked like the docs were simply wrong.
3. **They were not.** On real held-out cores the bf16 1B reads **1.339 nats/base** with predictive
   entropy **1.357** against a uniform **1.386** — it emits a near-uniform distribution, i.e. it
   guesses. The 7B base under the *same* fallback is fine at 0.859. The checkpoint stores FP8 scale
   metadata in its `_extra_state` entries (`te_fp8_meta` on load) that TE needs to dequantise the
   projections; without TE they are read as raw bf16 and the model is destroyed. **The refusal is
   crude but substantively correct.**
4. **TE 1.13.0 installed.** 2.18 will not build against torch 2.5.1 (`SymmetricMemory.hpp` — a
   header from a later torch); 1.13 is contemporary with torch 2.5, does not upgrade torch, and the
   7B pipeline is verified unaffected afterwards.

### 1B vs 7B — the differences that matter

| | 1B + TE | 7B |
|---|---|---|
| parameters | 1.108B | 6.58B |
| blocks | 25 (4 attn / 21 Hyena) | 32 (5 attn / 27 Hyena) |
| hidden | 1,920 | 4,096 |
| native context | **8,192** | 32,768 (262k variant) |
| **nats/base on real cores** | **0.990** | 0.859 base · 0.820 +LoRA |
| 3 training steps @ L=4096 | **~2 s** | minutes |

The **+0.13 nats** handicap is real but modest: the 1B is a usable stand-in for asking whether an
objective change moves anything. It is **not** a substitute for a final number — a Phase-2 positive
should be confirmed on the 7B before it is reported as a project result.

⚠️ Both models share the byte-level tokenizer, so 1 token = 1 base in both. **The 1B's speed comes
from depth/width, not from shorter sequences** — 3 kb is still 3,000 tokens. Measured throughput is
**8,770 vs 2,625 tok/s = 3.34×**, well short of the ~6× the parameter ratio suggests, and it is why
GenomeOcean (5.15 bp/token) remains the better option if the *long*-context version is ever needed.

## Substrate sanity check — and why it took three tries

`verify_1b_sanity()` exists because a model that loads is not a model that works. Getting the check
right required fixing three errors, each an instance of a pattern this project keeps repeating:

1. **It used a HAND-WRITTEN sequence** and returned 1.3843 against a 1.386 uniform threshold —
   passing a substrate that was at chance, because invented DNA is out-of-distribution for every
   model. *Test on real data.*
2. **It then compared protocols.** A no-context measurement (1.25) was thresholded against a
   with-context reference (0.99). *The same number measured two ways is two numbers.*
3. **It thresholded on ONE core.** Single-sequence variance here is ~0.25 nats — larger than the
   gap between healthy and broken. *n=1 cannot support a threshold.*

Final form: 500 bases after 2,000 nt of real context, averaged over 8 real cores, healthy < 1.15.

## Training configuration, and why

**L = 8,192** — the 1B's native context. Not 4,096: the sequence budget there is ~3,897 nt after
the prefix, while ONE NRPS module is 3,000–4,500 nt, so the window could barely hold the thing the
frame-aware arm exists to teach. *"The window could not fit a module"* is a poor reason for a null.
Cost is affordable: 7.7 GB peak, ~8,700 tok/s.

**`--long-seq-strategy chunk --chunk-overlap 1024`** — 95,759 windows over all 467 Mbp. The
trainer's default is `truncate`, which was used unexamined until challenged and turned out to bias
the experiment against its own hypothesis:

| | truncate @4 kb | whole-records-only @8 kb | blind chunk @8 kb |
|---|---|---|---|
| DNA seen | 25.2% | 14.7% | **100%** |
| class-domain coverage | **49.0%** | — | **33.7%** (the true rate) |
| cost | biased AGAINST the weighted arm | drops long cores | cuts 13.1% of genes |

Gene-aware boundaries were already tested in Phase 1 ("gene-aware ≤ blind", n=6, on metrics reading
~0) and buy complexity against a measured null. Whole-records-only is cleaner but uses 14.7% of the
DNA and drops most NRPS and all hybrids — exactly the assembly-line classes the ORF question is
about. Chunking is **common-mode across all arms**: it can add noise, it cannot manufacture a
difference between them.

*Residual concern, recorded not fixed:* a window starting mid-gene penalises the model for frame it
cannot yet infer from its visible context. Overlap was raised 512→1024 for more run-up; the fuller
fix (skip the first ~100 positions of a window) waits on this pass rather than shipping untested
code before a run.

**Read on `best_bio_bits`, never `max_orf_aa`.** The frame arm manipulates ORF length directly, so
scoring it there scores the manipulation. `max_orf_aa` does not track domain content de novo
(r = 0.051 / −0.120). If ORF length rises and `best_bio_bits` does not, that is the informative
negative: length was never the constraint. **Novelty is a constraint, not a metric** — every rung is
maximised by copying training data.

## The arms

| arm | flags | tests |
|---|---|---|
| `baseline` | `--domain-weight 1.0 --frame-lambda 0.0` | bit-identical to `causal_lm_loss` (pinned by test) |
| `frame` | `--frame-lambda 0.5` | in-gene stop-completion penalty |
| `weighted` | `--domain-weight 3.0` | per-record-normalised domain weights |

Not the full 2×2 yet — the arms want different lengths (frame-aware is length-agnostic;
domain-weighted is least meaningful at short context), so the interaction cell runs only if a single
arm moves.

**Reference points for reading a result** (7B, de novo): `best_bio_bits` LoRA **56.9**, real cores
**148.6**, base **0.0**.

## Layout

```
evo2_1b/
  README.md
  scripts/
    evo2_1b_inference.py     # loader + substrate sanity check (real DNA, 8 cores, healthy < 1.15)
    compare_1b_7b_loss.py    # baseline: next-base CE on real cores, 1B vs 7B
  experiments/
    run_objective_arms.sh    # baseline / frame / weighted
    score_arms.sh            # generate with matched prompts, then the ladder + novelty
  docs/
```

Shared, model-agnostic tooling stays at the root (`CLAUDE.md` → Repository Layout) and is called,
not copied: `src/bgc_pipeline/{evaluation,objective,annotations}.py`,
`evo2/scripts/{ladder_audit,score_ladder}.py`, `scripts/{build_domain_spans,memorization_check}.py`.

Run everything from the repo root with `EVO2_BASE_MODEL=evo2_1b_base`.
