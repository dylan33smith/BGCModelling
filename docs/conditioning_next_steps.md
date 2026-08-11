# Class conditioning — what to try next, and why

**Written 2026-08-11**, after the steering programme closed (`steering_program.md`) and per-class
soft prefixes also came back negative. Ranked from a 9-angle literature sweep (52 raw findings →
18 deduplicated ideas), scored by how directly each one addresses a **measured** failure of ours
rather than by how interesting it sounds.

> Provenance: the sweep used live web search, and the citations below were collected rather than
> recalled. They were **not** re-verified line by line afterwards. Treat the 2026 preprints
> (ideas A4, B3) as needing a fresh check before they are cited outside this repo.

---

## What we are actually stuck on

Stated precisely, because every idea below is judged against it:

| # | Measured finding | Where |
|---|---|---|
| 1 | Class is linearly decodable at **0.911** (chance 0.091) mid-network — and equally well in **base** Evo2, so the LoRA installed nothing | `progress.md` |
| 2 | The literal `\|COMPOUND_CLASS:X\|` prefix is **inert** (`v2_notag == v2_tag`) | 2026-07-21 |
| 3 | CFG found **no amplifiable signal**; coherence collapsed before class moved | 2026-07-22 |
| 4 | Steering **deletes** a class but never **installs** one, at any layer, dose, or 9-layer stack | `steering_program.md` |
| 5 | Soft prefixes trained and separated per class, but bought ~0.003 nats and gave `correct_class` **0/12** | `decisions.md` |
| 6 | **Exemplar conditioning works**: seed a real core → 0.283 vs a 0.067 floor | 2026-07-28 |
| 7 | Byte-level tokenizer gives a class tag no pretrained prior; the mid-network representation has no path to the output | — |

**The organising hypothesis this sweep produced.** Every mechanism we have tried injects the
condition at **one place**: one input token (2), one input position (5), an activation edit at a
few hand-picked layers (4). Meanwhile the model conditions perfectly well on content that is
present at **every** position (6). The literature's consistent answer is that conditioning which
works enters at **every layer** — and that is the axis we have never varied.

---

## Six cross-cutting themes

Each surfaced from more than one independent search angle, which is what makes them worth more
than any single citation.

1. **Depth of injection beats single-point injection.** Per-layer adapters (ProCALM), per-layer
   KV prefixes (Prefix-Tuning vs. embedding-only Prompt-Tuning), per-block modulation (DiT's
   adaLN-Zero vs. its own *worse* token-conditioning arm), per-position broadcast (ATGC-Gen),
   inter-layer gated cross-attention (Flamingo). The most repeated pattern in the sweep.
2. **Reading ≠ writing.** The direction that best *detects* a concept and the one that *causally
   installs* it are measurably different vectors — sometimes ~83° apart. This **predicts** our
   finding 4 rather than merely describing it.
3. **Ablation is structurally easier than injection**, and this is now an actively-studied general
   phenomenon with mechanistic accounts (angle/norm separation; injection competing against a
   growing residual norm that ablation never fights).
4. **Symbolic labels need full-model, large-scale exposure.** Every clean control-tag success
   (CTRL, ZymCTRL, ProGen, regLM) trained the tag through the whole model at large scale. Every
   LoRA-scale or embedding-only attempt in the sweep — including ours — underperformed.
5. **CFG has a hard prerequisite we never met:** a trained null branch via train-time condition
   dropout. Stated by the original paper, reconfirmed across diffusion, text, SMILES and DNA with
   zero counterexamples.
6. **The successful middle path is a small, gated, zero-initialised module** trained end-to-end
   against the real generative loss — never a hand-computed steering vector, never input-only.
   Zero-init specifically de-risks the coherence collapse we hit with ungated injection.

---

## Tier A — cheap (days), reuses infrastructure already on disk

### A1. Discriminator-guided decoding with our own `class_probe`
**Structurally different from everything we have closed.** Instead of one static vector, recompute
the guidance signal at *every* step from everything generated so far: FUDGE reweights the top-k
next-token distribution via Bayes' rule with a discriminator trained on **partial** prefixes; GeDi
does it contrastively with a small class-conditional model (~30× cheaper than PPLM); PPLM
backprops the classifier into the cached hidden state. ProteinGuide generalises the family to
Any-Order Autoregressive **biological sequence** models — Evo2's model class — and reports it
beating fine-tuning and post-hoc filtering.

- *Addresses:* **4** directly. Tests whether the null was specific to *static, context-insensitive*
  vectors rather than a property of all inference-time guidance. Needs no CFG null-branch (**3**)
  and no architecture change.
- *We already have:* `class_probe` at TPR 0.900, calibrated at both ends, is exactly the attribute
  model this family requires.
- *Caveat:* the probe has no negative class (0.900 mean confidence on real non-BGC DNA vs. 0.986 on
  cores). FUDGE's partial-prefix retraining is likely a prerequisite before trusting it as a live
  guide rather than a paired-comparison metric.
- FUDGE <https://arxiv.org/abs/2104.05218> · GeDi <https://arxiv.org/abs/2009.06367> ·
  PPLM <https://arxiv.org/abs/1912.02164> · ProteinGuide <https://arxiv.org/abs/2505.04823>

### A2. Affine Concept Editing — ablate, then **reset to a reference**
`v' = v − proj_r(v) + proj_r(r_ref) + α·r`. Combines the operation we already proved works
(directional ablation) with setting the vacated subspace to the **real class-X mean coordinate**,
instead of adding a delta from wherever the activation happened to sit. Plain `v + α·r` is
"non-standardised" — the resulting strength still depends on the original activation, so a fixed
dose never reaches a fixed target.

- *Addresses:* **4** with a concrete mechanistic fix, reusing our validated ablation machinery and
  the per-class means already computed for the closed steering programme.
- *Caveat:* validated on refusal — a coarse binary behaviour — not on installing a multi-kilobase
  generative style. Cheap enough to run before declaring steering fully closed.
- <https://arxiv.org/abs/2411.09003>

### A3. Re-run CFG after retraining the LoRA **with class dropout**
Ho & Salimans's Algorithm 1: null the condition on 10–20% of training examples so the same network
learns a genuine `P(x|null)`. Our LoRA saw the tag on 100% of examples, so its "unconditional"
stream at inference was an accidental byproduct, not a calibrated null mode — raising *w* had no
well-defined direction to amplify and simply pushed output OOD, exactly what we observed.

- *Addresses:* **3** completely — the textbook prerequisite was never met. Closes a real gap in our
  rigor even if the result stays negative.
- *Caveat:* extrapolation still needs *some* learnable conditional/unconditional delta, and finding
  **2** says the tag has no pretrained prior to build one from. Pair with B1 or A5, not standalone.
- <https://arxiv.org/abs/2207.12598> · <https://arxiv.org/abs/2306.17806>

### A4. Diagnostic: measure the angle between our probe direction and a **causal** direction
Derive a candidate injection direction by gradient ascent directly on `class_probe`'s output (a
causal objective), then measure its cosine to our existing diff-of-means direction. Published
measurements find detection and control directions can sit ~83° apart, with a ~15° rotation toward
causally-verified examples recovering 60–73% of control.

- *Addresses:* **1**, **4**, **7**. Our steering vectors were diff-of-means — itself a *detection*
  estimator, exactly the kind these papers show is misaligned with the causal direction.
- *Caveat:* explanatory, not a fix. Should **gate** any further steering spend rather than
  justify it.
- <https://arxiv.org/abs/2606.24952> · <https://arxiv.org/abs/2508.01892> ·
  AxBench <https://arxiv.org/abs/2501.17148>

### A5. Diagnostic: causally-localized injection (activation patching)
Injection demonstrably *can* work when localized to a small, causally-identified component set —
ITI raised TruthfulQA 32.5% → 65.1% by intervening on specific attention heads found via causal
mediation, not by picking the layer where a probe reads best. We located **where class is
decodable** but never **which components lie on the causal path to the next token**.

- *Addresses:* **4**, by reframing it as "the surgical form of injection is untried" rather than a
  closed negative.
- *Caveat:* the successes are coarse, single-decision behaviours (refuse / be truthful). Writing a
  multi-kb class-correct BGC is structurally harder, with no precedent at that complexity.
- ITI <https://arxiv.org/abs/2306.03341> · Function Vectors <https://arxiv.org/abs/2310.15213> ·
  reliability <https://arxiv.org/abs/2407.12404>

### A6. Redo the soft prefix as **every-layer KV-prefix** tuning
What we built was Lester et al.'s *Prompt Tuning* (embedding-only), not Li & Liang's
*Prefix-Tuning*, which prepends trainable vectors to the keys/values of **every** attention layer.
Both the original paper and P-Tuning v2 document embedding-only tuning as specifically failing on
hard tasks. `peft` is already a pinned dependency.

- *Addresses:* **5**, with the exact mechanistic reason from two independent papers — we chose the
  weak side of the one axis that matters.
- *Caveat, and it is a real one:* **this sits in tension with theme 4 and idea C4 below**, which
  argue further prefix-tuning spend is unlikely to pay off at LoRA scale. Worth **one** test
  because it changes the *mechanism* (every layer), not just the scale — but a null here should
  end prefix-tuning work rather than prompt another variant. Also: StripedHyena-2 is a hybrid, so
  KV-prefix injection applies cleanly only to the attention sublayers; the hyena-operator analogue
  would have to be improvised.
- <https://arxiv.org/abs/2101.00190> · <https://arxiv.org/abs/2110.07602>

---

## Tier B — moderate (weeks), strongest precedent

### B1. ProCALM-style **per-layer conditional adapters** on a frozen backbone
The closest published precedent in the entire sweep: same task family (autoregressive biological
sequence LM), same compute scale. A small conditioning encoder maps the categorical condition to a
latent; at **every** layer, a low-rank projection of that layer's hidden state is concatenated with
the latent, passed through a bottleneck MLP, and added back to the residual stream. Backbone never
updates. ~40–240 A100-hours vs. ~15,000 for training an equivalent control-tag model from scratch.

Crucially, their own controlled comparison found the **token/prompt-conditioned baseline —
structurally identical to our finding 2 — overfits and fails to generalise**, while the continuous
per-layer adapter generalises well.

- *Addresses:* **2** and **5** by replacing "a token the model may choose to attend to" with "a
  per-layer additive term it cannot skip"; **7** by giving the condition a dedicated pathway
  distinct from weights that have no prior for it.
- *Caveat:* demonstrated on ProGen2, a pure transformer. The per-layer hook point needs adapting
  for hyena/conv blocks — not done in the literature for this architecture family.
- <https://arxiv.org/abs/2410.03634> · <https://github.com/Profluent-Internships/ProCALM>

### B2. Per-class LoRA routing (class = which adapter, not a token to read)
Already our own documented next step; now with precedent for the routing mechanics. Since the class
is always known at generation time, the "router" is a hard lookup — no learned gate needed. Each
adapter learns *one* class's generative distribution instead of an entire conditional map.

- *Addresses:* **1**, **2**, **5**, **7** at once. A shared rank-16 LoRA must both learn to read an
  arbitrary byte-level label with zero prior **and** encode 22 classes in one low-rank subspace.
  Splitting removes the first requirement entirely — there is nothing for the tokenizer to fail to
  parse, because class identity is decided outside the forward pass.
- *Caveat:* less data per adapter; rare classes risk overfitting. MoCLE's universal-expert blend is
  the documented mitigation.
- MoCLE <https://arxiv.org/html/2312.12379v5> · LoRAHub <https://arxiv.org/abs/2307.13269>

### B3. LoRRA-style representation-alignment loss, trained **jointly** with generation
Add an auxiliary loss on hidden states — pushing class-X representations toward the class-X
centroid — into the *same backward pass* as the next-token loss. Every one of our findings 2–5
shares the shape "train first, intervene post-hoc second"; this couples the two objectives.

- *Addresses:* **1**, **2**, **4**, **7** collectively. The representation exists but is inert
  because nothing in training ever *required* the generative circuit to read it. This closes that
  loop, and we already have every ingredient except the loss term (LoRA infra, per-class centroids,
  the layer-16 direction).
- *Caveat:* Circuit Breakers' validated use is **suppression**; the installation direction is less
  battle-tested. Higher implementation risk than the suppression-only variant.
- RepE <https://arxiv.org/abs/2310.01405> · Circuit Breakers <https://arxiv.org/abs/2406.04313>

### B4. LoReFT — a **trained** low-rank intervention instead of a closed-form direction
`Φ(h) = h + Rᵀ(Wh + b − Rh)` at chosen layers, `{R, W, b}` trained by gradient descent against the
real class-X generation loss, backbone frozen. 15–65× fewer parameters than LoRA for comparable
effect.

- *Addresses:* **4** and **5** jointly — the first intervention whose parameters are optimised
  against a *causal* objective rather than heuristically computed (steering) or trained only via an
  indirect loss on unrelated tokens (soft prefix).
- *Caveat:* AxBench's own headline is that **prompting beat every representation-based method they
  tested**, ReFT included. A principled upgrade, not a guaranteed win over A1/A6.
- <https://arxiv.org/abs/2404.03592>

---

## Tier C — larger bets, none of them a first move

### C1. Gated, zero-initialised cross-attention (Flamingo / ControlNet)
New blocks between frozen layers; contribution multiplied by `tanh(α)` with `α` initialised to
**exactly zero**, so training begins as a true no-op and the model can only gradually learn to use
the new pathway. Directly de-risks the coherence collapse we hit with raw ungated injection.
*Caveat:* no genomic precedent found; for a purely categorical condition the K/V collapse to one
learned vector, which blurs into B1 — prototype the simpler gated adapter first.
<https://arxiv.org/abs/2204.14198> · <https://arxiv.org/abs/2302.05543>

### C2. AdaLN-Zero / FiLM per-block modulation
DiT's own controlled ablation ranked conditioning channels: token/in-context **FID ~35.2 (worst)**,
cross-attention ~26.1, adaLN ~25.2, **adaLN-Zero ~19.5 (best, and cheapest)**. Our prepended label
is structurally identical to their worst arm — independent evidence that finding **2** generalises
beyond "our LoRA didn't learn the tag" to "input-token conditioning is inherently low-bandwidth".
*Caveat:* DiT is non-causal and bidirectional; whether adaLN preserves autoregressive coherence is
untested by that paper. <https://arxiv.org/abs/2212.09748> · <https://arxiv.org/abs/1709.07871>

### C3. ATGC-Gen-style per-position broadcast
A DNA LM with **our exact character-level tokenizer constraint** conditions by concatenating the
class vector with every position's nucleotide encoding, removing the "read once at position zero,
carry it 32k characters" dependency shared by findings 2 and 5.
*Caveat:* purpose-built model, not a frozen-backbone retrofit; a constant additive signal at every
position risks shifting the input distribution away from what pretraining calibrated for.
<https://arxiv.org/abs/2507.19523>

### C4. Concept-bottleneck adapter
The strongest *number* in the sweep — CB-pLM reached ~96% directional-intervention accuracy vs.
~65–81% for tag/prompt baselines by making the concept layer architecturally unbypassable. But that
is a **from-scratch pretrain**; only the bolt-on-adapter retrofit is in scope, and it is the largest
speculative leap here. <https://arxiv.org/abs/2411.06090>

### C5. Substrate change — GenomeOcean's real trainable class token
Unchanged from `model_comparison_evo2_vs_genomeocean.md`: a proper vocabulary slot is precisely
what finding **7** says Evo2 lacks. Theme 4 tempers the enthusiasm — a new token only becomes
load-bearing with substantial training exposure, not a light adapter pass.

---

## Recommended order

1. **A1** (discriminator-guided decoding) — highest information per GPU-hour; the one inference-time
   family whose mechanism is genuinely untested here, and `class_probe` already exists.
2. **A4 + A5** (diagnostics, ~a day) — run *before* any further steering spend; they decide whether
   ideas in that family are worth anything.
3. **B1** (ProCALM per-layer adapters) — best precedent-to-cost ratio of the real fixes, and it
   tests the sweep's central hypothesis (depth of injection) directly.
4. **B2** (per-class LoRA) — highest prior of simply working, because it removes the symbolic
   conditioning problem instead of solving it.
5. **A2, A3, A6** as cheap fill-in; **B3/B4** if B1 shows partial signal; Tier C only after that.

**Standing caution.** Every idea above still has to clear the same bar the last three programmes
did: a **paired** design with the control built in (the off-diagonal arms, the shuffled-label twin),
a **continuous** readout alongside the binary gates, and an instrument whose sensitivity and
false-positive rate are measured *before* the result is read off it. Three separate findings in this
project were weakened or retracted for want of exactly that.
