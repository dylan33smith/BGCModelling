# Phase 2 — the small-model track

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
from depth/width, not from shorter sequences** — 3 kb is still 3,000 tokens. That caps the speedup
at roughly the parameter ratio, and it is why GenomeOcean (5.15 bp/token) remains the better option
if the *long*-context version is ever needed.

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

## Layout

```
evo2_1b/
  scripts/
    evo2_1b_inference.py     # loader (+ the fp8/bf16 story), substrate sanity check
    compare_1b_7b_loss.py    # baseline 1: next-base CE on real cores, 1B vs 7B
  experiments/               # run drivers
  docs/                      # phase notes
```

Run everything from the repo root. Shared tooling stays at the root by the existing convention
(`CLAUDE.md` → Repository Layout).
