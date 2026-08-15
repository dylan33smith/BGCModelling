# Evo2 + LoRA + the Hyena block — how conditioning is (and isn't) trained

Reference explainer written 2026-07-06 to accompany the 2026-07-03 conditioning-failure
diagnosis (see [docs/archive/pre-framework/decisions.md](../../docs/archive/pre-framework/decisions.md)). It explains
**how LoRA attaches to Evo2, where our adapters actually sit across the 32 blocks, and why
the long-range mixing pathway is currently un-trained** — which is the leading structural
hypothesis for why the model generates simple clusters (ectoine/terpene) but never the
conditioned NRPS/PKS megasynthase machinery. Probe **B** (`--lora-target-parameters
projections.weight`) is the direct test of the fix described here.

---

## 1. What LoRA is (and what it is not)

LoRA does **not** insert a module between blocks, and it does **not** modify the original
weights. It attaches a small trainable correction *in parallel* with a chosen weight matrix,
and freezes everything else.

A normal linear layer computes `y = W·x`, where `W` is a large frozen matrix
(e.g. 4096×4096 ≈ 16.8M numbers). LoRA freezes `W` and adds a low-rank parallel path:

```
y = W·x  +  (B·A)·x · (alpha/r)
    └frozen┘   └ trainable ┘
```

- `A` is `r×d_in`, `B` is `d_out×r`, with **rank `r`** (we use 16). `B·A` has the same shape
  as `W` but is built from ~130K numbers instead of 16.8M.
- **Only `A` and `B` are trained; `W` is never touched.** The update the model may make is
  forced through the rank-`r` bottleneck, so it is cheap but *limited in expressivity* — it
  lives in an `r`-dimensional subspace, whereas a full fine-tune could make any update.
- At inference the adapter can optionally be *merged* (`W' = W + B·A·alpha/r`); during
  training the base stays frozen and only the `A`/`B` pairs learn.

Mental model: **sticky-note corrections taped onto selected machines** — not a new machine
in the line, and not re-machining the originals.

## 2. Where our adapters sit across Evo2's 32 blocks

We select adapters by **layer-name suffix**; PEFT tapes one onto every matching matrix in
every block. Current target list (`finetune_evo2_lora.py` `LORA_TARGET_MODULES`):
`{Wqkv, out_proj, out_filter_dense, l1, l2, l3}`.

Evo2 (StripedHyena, 7B) is **32 stacked blocks**: **5 attention** (indices 3,10,17,24,31)
and **27 Hyena** long-convolution blocks (9 short `hcs`, 9 medium `hcm`, 9 long `hcl`).
Mapping the targets onto that:

| Target | What it is | Where | Trainable? |
|---|---|---|---|
| `l1,l2,l3` | GLU **MLP** | all 32 blocks | ✅ |
| `Wqkv,out_proj` | attention projections | 5 attention blocks | ✅ |
| `out_filter_dense` | Hyena mixer **output** projection | 27 Hyena blocks | ✅ |
| **`projections`** | Hyena mixer **input** projection (x1/x2/v) | 27 Hyena blocks | ❌ **frozen** ← the gap |
| `short_filter_weight`, `h`, poles/residues | the **convolution kernels** | 27 Hyena blocks | ❌ **cannot take LoRA** (see §5) |
| embeddings, unembed | token table + LM head | — | ❌ frozen |

Trainable-parameter split of the current 28.7M-param adapter (≈0.44% of 6.5B):

| Adapted | share |
|---|---|
| MLPs (`l1/l2/l3`) | **~81%** |
| Hyena output (`out_filter_dense`) | ~12% |
| Attention (`Wqkv/out_proj`) | ~7% |

**The problem is visible in that table:** ~81% of our capacity is on MLPs, which are
*position-wise* (they transform each nucleotide independently and **cannot mix across
positions**), and the one pathway that *does* mix across long distances — the Hyena mixer's
input projection and its convolution kernels — is 0% adapted.

## 3. Anatomy of a Hyena block (the dataflow)

A Hyena block replaces attention with a **gated long convolution**. Inside the mixer
(confirmed from `vortex/model/model.py`):

```
x ──► pre_norm ──► [ projections ] ──► split into x1, x2, v
                       (INPUT proj)          │
                                     x1 · v  │  (pre-gate)
                                             ▼
                                ┌─────────────────────────┐
                                │  short conv (FIR)        │  local mixing
                                │  long convolution (h /   │  ← the long-range operator
                                │  poles+residues)         │     (spans the whole sequence)
                                └─────────────────────────┘
                                             │
                                        y · x2  (post-gate)
                                             ▼
                                    [ out_filter_dense ] ──► + residual
                                       (OUTPUT proj)
```

Step by step:

1. **`pre_norm`** — normalizes the activations (RMSNorm-style) so the mixer sees
   well-scaled inputs. Stability, no mixing.
2. **`projections` (INPUT projection, a linear)** — maps the hidden state to **3× hidden
   size** and splits it into three streams **x1, x2, v**. These are Hyena's analogue of
   attention's Q/K/V: `v` is the content/"value" stream; `x1` and `x2` are multiplicative
   **gates**. This is the layer that decides *what gets fed into* the long-range operator.
3. **short conv (FIR filter, `short_filter_weight`)** — a length-3 depthwise convolution:
   each position mixes with its immediate neighbours. Cheap **short-range** context.
4. **long convolution (`h`, or log-poles/residues in `hcl`)** — the core **long-range
   operator**. A convolution whose kernel is as long as the sequence, implemented
   efficiently (FFT / state-space recurrence) so one position can be influenced by another
   thousands of bases away. **This is the machinery that would coordinate a KS domain with a
   downstream AT/ACP domain in an assembly line.**
5. **gating (`x1·v` before, `·x2` after)** — element-wise multiplies. The gates let the
   model modulate what enters and leaves the convolution (data-dependent control).
6. **`out_filter_dense` (OUTPUT projection, a linear)** — mixes channels back to hidden
   size for the residual add.

**Overlay what we train:** we adapt step 6 (output) and the block's MLP. We do **not** adapt
step 2 (input `projections`) or step 4 (the convolution kernels). So for the one operation
that mixes across long distances, **what goes in (2) and the operation itself (4) are frozen,
and we can only tune how the output is cleaned up afterward (6).** That is the mechanistic
reason the model can adjust *local* content (domain appearance) but cannot learn *new
long-range coordination* (assembly-line module order).

## 4. What Probe B changes

Probe **B** adds a LoRA adapter to the **input projection** (step 2) via
`--lora-target-parameters projections.weight` — making the x1/x2/v streams that feed the
long-range operator trainable for the first time (28.7M → 35.8M trainable params, the +7.1M
being 27 fresh adapters). It cannot touch step 4 (the kernels; see §5), but it lets the
model *shape what enters* the long-range mixing, instead of only reprojecting the output.

The companion lever is **rank**: even on adapted layers, rank-16 confines each update to a
16-dimensional subspace. If B shows life but is insufficient, raising rank — especially on
the mixer/attention paths rather than the MLPs — is the next dial.

## 5. Why the long-range convolution kernels can't take LoRA

Two reasons, one mechanical and one fundamental:

**(a) They are raw parameters, not layers.** In PyTorch, an **`nn.Module`** (like a linear
layer) is an object with a `forward()` method and registered weights; standard LoRA works by
**finding that module and replacing its forward** with one that computes `W·x + B·A·x`. The
Hyena kernels are **`nn.Parameter`** tensors (`self.h = nn.Parameter(...)`,
`self.short_filter_weight = nn.Parameter(...)`) — bare weight tensors used *directly* inside
the block's own forward, with no sub-module of their own. There is no `forward` to intercept,
so the standard "wrap the module" mechanism has nothing to hook onto. (This is a different
situation from `projections`, which *is* a module — see §6 — and can be reached by PEFT's
newer `target_parameters` path that reparametrizes a weight tensor in place.)

**(b) LoRA's math assumes a matmul weight; a conv kernel isn't one.** LoRA approximates a
weight *update* as a low-rank matrix `B·A` added to a matrix `W` that acts by **matrix
multiplication** (`W·x`). A convolution kernel doesn't multiply the input as a matrix — it is
*convolved* with it (and in `hcl`, the "kernel" is a set of spectral poles/residues, not a
matrix at all). A low-rank additive correction to a convolution/spectral kernel has neither
the shape nor the theoretical justification of a LoRA update, and PEFT does not implement it.
So even `target_parameters` (which reparametrizes 2-D linear weights) does not apply.

Consequence: the long-range **kernels stay frozen no matter what** — the model always relies
on Evo2's *pretrained* long-range filters. Probe B can reshape their *inputs*; nobody can
LoRA the filters themselves. (Changing them would require full/partial fine-tuning of those
parameters, a separate and heavier option.)

## 6. `TELinear` vs a vanilla `nn.Linear`

Both compute the same thing — a linear map `y = x·Wᵀ + b`. The differences are about
*implementation and type*, and they are why our input `projections` got skipped:

- **`nn.Linear`** is PyTorch's standard linear layer: stores `weight` (out×in) + `bias`,
  returns a plain tensor. PEFT has it in its registry of wrappable types, so a "adapt all
  Linear layers" pass finds and wraps it automatically.
- **`TELinear`** is a **Transformer Engine**-style linear (NVIDIA's FP8/mixed-precision
  library, tuned for H100 tensor cores). It carries extra machinery: FP8 scaling metadata
  (the `_extra_state` keys you see in the state-dict), optional tensor-parallel splitting,
  and — importantly — its forward returns a **tuple `(output, bias)`** rather than a bare
  tensor. In *this* environment Transformer Engine isn't installed, so the code uses a
  **pure-PyTorch fallback** `class TELinear(nn.Module)` (`vortex/model/layers.py`) that
  mimics TE's interface (same weight/bias naming, same `(output, bias_or_None)` return) but
  computes an ordinary linear.

Why it broke LoRA coverage:
1. **Type mismatch.** `TELinear` is not an `nn.Linear`, so PEFT's "wrap every `nn.Linear`"
   heuristic silently skips it — which is exactly how `projections` got left out of the
   target list.
2. **Non-standard return.** Its forward returns a tuple, which would break PEFT's standard
   forward-replacement (that path expects `module(x)` to return a tensor). PEFT's
   `target_parameters` route sidesteps this by reparametrizing the **`.weight` tensor** in
   place (`W → W + B·A`) and leaving the module's own forward untouched — which is why that
   is the route Probe B uses, and why it requires `lora_dropout=0` (PEFT's `ParamWrapper`
   forbids dropout).

---

## TL;DR

- LoRA = frozen base + small trainable low-rank correction **in parallel** with chosen weight
  matrices. Originals are never modified.
- ~81% of our adapter capacity is on **position-wise MLPs** that can't mix across positions;
  the Hyena **input projection** (x1/x2/v) is **frozen** and the **long convolution kernels
  can't take LoRA at all**.
- So the long-range pathway — the machinery that would build ordered assembly-line modules —
  is essentially un-trained. **Probe B** unfreezes the input projection to test whether that
  is the bottleneck; the kernels themselves need heavier fine-tuning if B is insufficient.
