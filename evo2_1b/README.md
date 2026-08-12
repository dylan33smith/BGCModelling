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

**`evo2_1b_base`** — 1.108B params, 25 blocks, hidden 1920, **native context 8,192**, 4 attention
blocks (3/10/17/24) to 21 Hyena. The 8k context is a natural fit for the short-context pilot the B
plan already called for.

**The Transformer Engine story, in order, because two of these were wrong:**

1. Project docs said for weeks that no small model existed because the 1B "requires Transformer
   Engine / FP8". **Half right.**
2. The refusal in `evo2.models` is a **name check** — `if "7b" in model_name: fall back to bf16,
   else raise`. The fallback itself is model-agnostic, so the 1B *loads* fine in bf16 without TE:
   1.108B params, clean load, finite logits. It looked like the docs were simply wrong.
3. **They were not.** Measured on real held-out cores, the bf16 1B scores **1.339 nats/base** with
   a predictive entropy of **1.357** against ln(4) = **1.386** — it is emitting a near-uniform
   distribution, i.e. *guessing*. The 7B base under the *same* bf16 fallback is fine (0.859).
   **The fallback is safe for the 7B and destroys the 1B.** The name check is crude but substantively
   correct, and "it's just a string comparison" was wrong.
4. ⇒ TE is genuinely required for this model. It does **not** upgrade torch (`torch>=2.1`, we have
   2.5.1+cu124); it builds `transformer_engine_torch` from source and must be pinned to the **cu12**
   runtime, since the resolver otherwise pulls cu13 against our cu124 build.

**The general lesson, recorded because this project keeps paying for it:** *a model that loads is
not a model that works.* The check that caught this was predictive entropy against a uniform
baseline on **real** data — the first sanity check used a hand-written sequence and passed a model
that was at chance.

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
