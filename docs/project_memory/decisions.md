# Decisions — why the project is built the way it is

Architecture and approach decisions with their rationale. Newest context at the top of
each topic. See also [progress.md](progress.md) (current state) and [bugs.md](bugs.md)
(quirks/fixes). Full historical detail: `docs/archive/REDESIGN_PLAN.md`.

---

## Modelling

### [2026-08-12] Class conditioning was aimed at the smaller of two problems — the constraint is CAPABILITY
Two cheap measurements, both on data already on disk, reframe the programme.

**(a) The class tag is worth nothing to the loss** (`evo2/scripts/context_ablation.py`). Scoring the
same 500 bases of real cores while varying only preceding context: 10 nt already gives 73% of
everything the model achieves (0.977 nats vs 1.386 uniform); 1,000 → 6,000 nt buys 0.005; all
long-range context is worth **0.149 nats**. Right-vs-wrong class tag: **−0.0006 nats**, and
**−0.0000** with the tag only 200 nt away. ⇒ The tag is not ignored out of stubbornness — *using it
never reduced the loss*, so gradient descent had no incentive to build a pathway that reads it.
One number retro-explains the inert label, the absent CFG signal, and the 0.003-nat soft prefix.

**(b) De novo, almost nothing is detectable at all.** `correct_class = P(detect) × P(right|detect)`
on the same step_1200 adapter: de novo **0.012** (1/81, CI [0.000, 0.067]) vs seeded **0.367**
(44/120, CI [0.281, 0.459]); seeded class-given-detection **0.932**. The seed multiplies detection
**30×**, and with one de novo detection the conditional is unestimable — the pre-registered third
outcome, which is itself the answer.

⇒ **In neither regime is class-correctness binding.** Seeded, it is already 0.932. De novo, perfect
conditioning would have nothing to install class into. **This demotes per-layer conditional adapters
from "the next step" to third priority** — they target ~7% of the seeded gap and nothing of the de
novo one. Defer until de novo detection is non-trivial, at which point ProCALM applies directly.

**The failure is capability, not the instrument** (`evo2/scripts/soft_instrument_probe.py`). antiSMASH
needs *clustered* genes, so the obvious alternative was that a generation could contain real
biosynthetic DNA and still fail. Re-scored with instruments needing only ONE domain hit anywhere:

| group | coding density | longest ORF | ≥1 class domain | antiSMASH |
|---|---|---|---|---|
| real @3 kb | 0.972 | **702 aa** | 0.800 | ~0.58 |
| seeded @3 kb | 0.932 | 591 aa | 0.467 | 0.367 |
| de novo @6 kb | 0.743 | 505 aa | 0.033 | 0.034 |
| de novo @2 kb | 0.815 | 332 aa | 0.050 | 0.000 |

Two instruments at very different strictness agree. De novo output is *not* junk — but **the longest
ORF is 332–505 aa against ~1000–1500 aa for a single NRPS module**, so the model cannot hold a
reading frame long enough to encode one and there is nowhere for a domain to sit.

⇒ **Adopt a continuous ladder in place of the binary gate:** `max_orf_aa` → `domain_count` →
antiSMASH detect → class. The first two are non-zero today and can be optimised and tracked;
`correct_class` has read ~0 for a year and cannot.

⇒ **Why an objective change is warranted rather than merely plausible.** Next-base prediction is
*locally satisfiable*: predicting base 900 from bases 850–899 works whether or not base 1,400
introduces a stop codon. The objective never asks the model to keep a promise it made 1,000 bases
ago, which is exactly the failure measured above. Candidate fixes, in order: domain-weighted loss
(needs a one-off pyhmmer pass over the 47.5k training cores to get per-domain nucleotide
coordinates — the records carry `strict_core_genes` as a COUNT only, no per-domain spans),
reading-frame-aware penalties, an auxiliary head predicting upcoming domain content (NOT class —
the probe already recovers class at 0.911, so that head would teach nothing), and sequence-level
reward as a reserve.


### [2026-08-11] "The model does not read these activations" is FALSE — it reads them; our edit was the wrong shape
`evo2/scripts/activation_patching.py`. Substituting a **real** donor's activations (in-distribution,
all 4,096 coordinates consistent) over 10 of 1000 context positions at layer 16 moves the
next-token distribution **41%** of the way to the donor; 200 positions reaches **84%**. Controls:
same-class donor 0.128, position-shuffled 0.129, norm-matched noise −0.099. Our rank-1 steering
edit, at every position, at 2.8–11.4 class-units, moved it essentially nothing.

⇒ **The mechanism behind every steering null is the SHAPE of the edit, not the model's blindness.**
This is the direct experimental confirmation of what the ACE pre-check predicted from geometry
alone (a rank-1 edit fixes 1 coordinate of 4,096 and leaves the rest belonging to the source class,
landing 3–20 sd off-manifold). It does **not** reopen steering — a direction is still the wrong
shape — but it does open *transplantation* as a category we had never tried, and it means the
residual stream at mid-depth is a live channel rather than a dead one.

**Two design corrections, both caught by internal inconsistency rather than by a crash.**
(1) Patching ALL positions returns alignment 1.000 with an identical KL at layers 0, 16 and 31 —
the model just becomes the donor. Identical numbers across layers were the tell. It is a positive
control, not a measurement. (2) The one-position null at mid-layers was **leverage, not blindness**;
only a k-sweep separates those, and without it the experiment would have "closed" the last open
door on an artifact.

**RESOLVED by Phase B (`patch_generate.py`): what transfers is BEHAVIOUR, not CLASS.** Patch the
context at layer 16 or 22 over the last 50-200 positions, generate 3 kb, score with antiSMASH:
the donor's class appears in **0 of 48** cross-class transplants (95% upper bound 6.1%), against an
unpatched recipient-class rate of 0.333. If class travelled as well as behaviour does at these
settings (84-92% alignment), it should have appeared ~16 times. The degradation that does occur is
**generic** - the same-class control loses the recipient's class just as often (10 vs 3, p=0.09,
against cross-class 10 vs 4, p=0.18) - so there is no deletion/installation asymmetry here, unlike
steering. Simply no class transfer in either direction.

=> **This is the strongest closure of inference-time intervention in the project**, because it is
the first one that rests on a POSITIVE demonstration rather than a null: the channel provably
works (Phase A), the model provably reads it, its local output provably follows - and class does
not. Class is not controllable from a mid-layer context representation at generation time. What
remains is training-time coupling.

### [2026-08-11] A2 (Affine Concept Editing) closed by an offline pre-check; and a rank-1 edit cannot reach the class manifold
`evo2/scripts/ace_precheck.py`, CPU-only. ACE written out is a **rank-1 edit along the same
direction** as additive steering, differing only in a per-example dose that lands the coordinate on
the class mean `m_c`. Since `class_unit` is *defined* as the other-mean→class-mean distance,
**ACE ≈ dose 1.0 per example** — not a new mechanism, a better-calibrated dose.

So the deciding question is not "does it move the probe" (the direction audit showed any
sufficient move does) but "does the edited point resemble a real class member". Judged by k-NN
distance to a bank of real target activations, normalised by how far real held-out target
activations sit from that bank: **ACE removes 6.7% (L16) / 11.6% (L27) of the source's
off-manifold distance**, negative for two classes. Correcting 1 coordinate of 4096 leaves 4095
belonging to the source class. Pre-registered rule was "both far off-manifold ⇒ ACE is not the
fix". ⇒ **Skip A2; spend the GPU on Tier B.** Cost: zero GPU.

**Why the overdose finding does not reopen steering.** The doses the programme ran land 3.4 sd
(2.8 cu) and 19.5 sd (11.4 cu) past the target class mean — we were leaving the distribution, not
approaching the class, which is the mechanism behind "a bigger dose buys damage, not class". But
Phase 3 already ran doses 1, 2 and 4 at L16, and **dose 1 is the on-target, ACE-equivalent dose
(z≈+0.1) and gave 0/48**. The correctly-dosed cell was tested at n=48 (excluding effects above
~11%) and was null, so this is an explanation for the damage, not an untested cell.

**`class_probe` must never gate — now with a mechanism.** The edited off-manifold points score
**higher than genuine class members in 10/10 class-layer cells** (real NRPS 0.762 vs a 2.8-unit
edit at 0.963, sitting 3.9× further from the NRPS manifold). A linear readout of one direction out
of 4096 grows *more* confident as the point becomes *less* like the real class. This is
independent support for the rule already pinned by three tests.

### [2026-08-11] The steering edit LANDED at every layer — "no more steering variants" is now evidence-backed, not assumed
`evo2/scripts/direction_audit.py` (CPU-only, off the cached train activations) separates the two
explanations that every steering null was compatible with and that we had never distinguished:
the edit never landed (bad direction/dose ⇒ steering deserves another recipe), or it landed and
was ignored (⇒ the model does not read that subspace ⇒ depth of injection is the axis).

Take held-out **non-target** activations, add the same direction at the same doses, ask the probe
what class it sees. Across **all nine layers (10–27)** and all five classes, the readout flips to
the target at **1–2 class-units** — and the experiments dosed **2.8 / 5.7 / 11.4**. Real vs random
direction at 2.8 units: **0.94–1.00 vs 0.005–0.19**. The edit was 1.5–10× larger than needed to
completely convert a linear readout in the very layer it was applied to.

⇒ **Explanation (ii) holds. The decision stands, now for a measured reason.** The representation
moved exactly as intended at every depth we ever steered, and the output did not follow. The
failure is downstream of the edit, which is what the depth hypothesis predicts and what Tier B
tests. Had the flip needed >11.4 units, this would have REOPENED steering; it was written to be
able to overturn our own conclusion, and did not.

**The delete/install asymmetry is NOT geometric.** Ablation also works linearly (P(true class)
0.80 → 0.09–0.41 at every layer). So in activation space *both* operations succeed; through the
model only deletion survives to the output. The asymmetry is a property of how the model READS
the space, not of where the classes sit in it. This kills "the install direction is somehow
malformed" as an explanation.

**Corrects a reading of the literature in `conditioning_next_steps.md` A4.** Detection and
control directions being ~83° apart does NOT imply the detection direction is useless for
control. We measure **58–86°** between the diff-of-means direction and the probe's logit
direction — and the near-orthogonal direction still flips the readout completely at 2 units. A
large angle and full causal efficacy on the readout coexist. A4's gradient-ascent half (a
direction from the *model's output*, not the probe) is still unrun and still worth running; the
angle alone was never the diagnostic.

### [2026-08-11] Every rate at 3 kb must be quoted against its measured ceiling, and hybrids at 3 kb are withdrawn
Prompted by the question "aren't we failing because 3 kb is too short, not because conditioning
doesn't work?" — a fair challenge that turned out to be right for some classes and wrong for
others. Measured, not argued: real held-out cores truncated to generation length, same gate.

Two samples were measured and they disagree, so the right one matters:
- **PKS_NRPS_HYBRID: 0.00 at 1/2/3 kb in BOTH samples.** Structural — a hybrid call needs both
  machineries and 3 kb cannot hold both. All hybrid results at 3 kb are WITHDRAWN.
- **PKS: 0.40 at 3 kb vs 0.96 full** — a 2.4x compression, the largest for a non-hybrid class.
- **NRPS 0.76, TERPENE 0.88, RIPP 0.76** at 3 kb — only mildly affected.
- An earlier long-tail-only sample (cores ≥12 kb) read NRPS 0.25 / PKS 0.33 / pooled 0.40. That
  answers "what does truncation cost a LONG core" and **overstates the handicap for NRPS by 3x**.
  Use the population column. Pooled-excluding-hybrids is 0.70, agreeing with the independent
  positive control's 0.750.

**Consequence for how we report.** Absolute rates at 3 kb are fractions of 0.40-0.75, never of
1.0. Paired internally-controlled contrasts are unaffected (shared ceiling cancels). Going
forward, any de-novo generation experiment aimed at megasynthases should generate at >= 6-12 kb,
or restrict to classes whose natural cores fit the generation length.

### [2026-08-10] Soft prefixes fail too — the bound is "input-only conditioning", not "training-time coupling"
Per-class continuous soft prefixes (`evo2/scripts/train_soft_prefix.py`): 16 x 4096 = 65k learned
floats per class, base + v2 LoRA frozen and merged, trained on train-only data, evaluated de novo
(taxonomy-only, no seed) on held-out taxa, with all four prefixes generated under an IDENTICAL
taxonomy pool and seed so the only variable is which prefix is loaded.

**The run is not vacuous, which is what makes the negative worth something.** An accident of
string length gave a clean control: `|COMPOUND_CLASS:` is exactly 16 characters, so every prefix
initialised from a *provably identical* vector (pairwise cosine 1.0000 by construction) and
training moved them apart to 0.85–0.92 with norms 1.45 → 1.53. Class-specific learning
demonstrably happened. It just bought ~0.003 nats and did not reach generation: `correct_class`
**0/12 in every arm**, and the continuous probe's per-taxon paired test found no class beating the
other three prefixes (TERPENE's apparent +0.173 fails four ways — not significant after
Bonferroni, carried by 2 of 12 sequences with median +0.012, a coin flip against the no-prefix
floor, and it is the class the probe already drifts toward on non-BGC DNA).

**Why this is a narrow bound, stated deliberately.** The soft prefix changes only the model's
INPUT and totals 65k parameters. Per-class LoRA has 28.7M and modifies the *computation*; a
trainable class token in a model whose tokenizer supports one is different again. This result
bounds the cheap, input-only end of training-time coupling and nothing beyond it — do not cite it
as "training-time coupling failed".

**Corollary for how we spend next:** the pattern across steering (activation-space edit at a few
layers), label prefixes (one input token) and soft prefixes (one input position) is that every
mechanism we have tried injects the condition at ONE PLACE. See
`docs/conditioning_next_steps.md` for the literature review that makes this the organising
hypothesis for the next programme.

### [2026-08-10] ⚠️ RETRACTED IN PART — "the class direction DELETES class" was a leakage artefact
The finding below was produced by a class probe fit on **val+test** and applied to generations
**seeded from val+test cores**. Refit train-only, ΔP(seed) at the top dose goes from −0.308
(p = 0.0063) to **−0.177 (p = 0.146)** and the lowest dose changes sign. The generation-level
deletion claim is **withdrawn**; what remains is a non-significant negative trend at the two
higher doses. Phase 1's teacher-forced ablation asymmetry (z = 4.8) is independent of the probe
and still stands, as does the null on ΔP(target) — steering installs nothing, which was never the
contested part. Enforcement added: every activation cache now carries a `.provenance.json` and
`_fit_probe` REFUSES a non-train fit set. Original entry retained below, unedited.

### [2026-08-10] The class direction can DELETE a class but not INSTALL one — and binary gates could not see it
Multi-layer steering (9 layers, own direction + class-unit each, 3 doses, shuffled-label twins)
returned 0/12 target markers everywhere — the same flat zero every binary gate has produced all
program. The new CONTINUOUS readout (`class_probe`) on the identical sequences found something the
gates cannot express: paired real vs shuffled-label on the same exemplars, at the top dose,
**ΔP(seed class) = −0.308, 1/12 up, sign p = 0.0063** (Bonferroni over 6 tests: 0.038), while
ΔP(target) never reached significance (best +0.107, p=0.146, non-monotone). Controlled for
coherence damage: corr(Δcoding, ΔP(seed)) = +0.002, and on the 7 exemplars where the real arm was
NOT more damaged the effect is larger (−0.393).

**Interpretation: ablation works, injection does not.** The vector carries genuine class
information — enough to specifically erase a class that is present — but adding it does not make
the generator write the target's machinery. Erasing a coordinate the model already uses is easy;
writing one it does not consume changes nothing downstream. This reproduces Phase 1's asymmetry
(ablation z=4.8 strong, nudge marginal) in GENERATION rather than teacher-forced scoring, and it
is the mechanistic reason the whole inference-time family fails.

**Methodological consequence, which generalises past steering:** a threshold gate bounds a LARGE
effect and is silent on a small one. Every class readout in this project was binary until now
(marker TPR 0.717 at 3 kb; antiSMASH detects ~1/3 of seeded 3 kb generations), so a real effect of
this size was invisible by construction. `class_probe` is now a permanent DIAGNOSTIC check — never
a gate, because calibration showed it is 0.900 confident on real non-BGC DNA vs 0.986 on real
cores: it has no negative class and measures resemblance, not validity.

### [2026-08-10] Dilution is NOT why steering failed — injecting later gives LESS influence, not more
After Phase 3 killed layer-16 steering, the leading remaining excuse was that the edit is made 16
blocks from the output and never survives — the residual stream grows 11 orders of magnitude in
the last blocks (L27 11.25 → L30 3.69e12). We tested it at **layer 27**, the last layer where the
class direction is still real (held-out AUC 0.835) and the last before the blow-up.

New instrument: `evo2/scripts/steer_reach.py` measures `reach` = mean KL of the next-token
distribution under a steering edit whose magnitude is a fixed fraction of the **local** residual
norm. Measured on n=40 held-out cores at frac 0.16: **L16 0.01011 → L20 0.00604 → L24 0.00359 →
L27 0.00288.** Reach falls monotonically with depth; the same relative edit at L27 moves the
output **3.5x less** than at L16.

**Why that is the right answer rather than a surprise.** The residual stream is additive, so an
edit at L16 is both carried forward unchanged *and* read, amplified and re-expressed by the 11
blocks after it. An edit at L27 has 4 blocks left to be read by. "Closer to the output" means
**fewer opportunities to be used**. The late-layer norm explosion attenuates both injection
points equally, since it happens after both.

**Consequences.** (a) Multi-layer steering and later-layer steering are both dead as fixes — they
address a constraint that is not binding, and the deeper you inject the worse it gets. (b) The
remaining explanation for Phase 3 is the one the program flagged at the outset: the class
coordinate is *present* at L16 but the generator barely *consumes* it, so the answer is
**training-time coupling** (per-class adapters, or a class-prediction loss that forces the
last blocks to read the class coordinate), not any inference-time intervention.
(c) Doses must be quoted as a fraction of the **live** residual norm, never from the pooled
activation cache — see bugs.md 2026-08-10.

**Open leakage debt (unchanged):** all steering directions, including the new L27 build, are fit
on val+test. They must be refit on train-only before any number is reported externally.

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

### Native-format alignment is LOW leverage for CLASS conditioning (2026-07-21, workflow-verified)

Investigated whether re-aligning our conditioning prefix to Evo2's native pretraining
format is a lever for **class** conditioning. Verdict from code + the Evo2 paper +
an adversarial refutation pass (all converged; citations below): **no — spend effort on
CFG / per-class adapters instead, keep the GTDB tag as-is.**

- **Taxonomy tag is already native.** Our `|d__…;p__…;…;s__…|` is byte-for-byte Evo2's
  only structured pretraining conditioning field (lowercase 7-rank GTDB, pipe-wrapped,
  semicolon-separated, loss-masked, node-level `random_lineage_dropout`). No lever left
  there. NB the native `c__` rank is *taxonomic* class (Gammaproteobacteria), **not**
  biosynthetic product class.
- **Evo2 has ZERO native handle for product/function class.** Complete conditioning
  special-token inventory = GTDB phylo-tags + structural contig-stitch tokens `@`/`#`
  (Methods §4.1.2/4.1.3); functional priorities enter via *data weighting/windowing*,
  not tokens. CharLevelTokenizer is pure byte-level (vocab 512; reserved eod 0/eos 0/pad 1),
  so `|COMPOUND_CLASS:X|` tokenizes as raw UTF-8 with no dedicated encoding — LoRA must
  install class→sequence from scratch through the low-rank bottleneck. Reformatting cannot
  recruit a prior that does not exist.
- **Arc's own precedent = our exact pattern.** cas9/cas12/cas13 functional conditioning was
  a *separate finetuning stage* (prepend token + finetune on 8 kb sequences), not a base
  prior — confirming prepend-token+finetune *can* install functional conditioning, but note
  they used full-FT on a single coherent protein family vs. our LoRA on multi-gene BGCs
  (harder + thinner).
- **Two cheap ideas the investigation surfaced (queued, not levers-by-themselves):**
  (1) **Position-0 collision hypothesis** — our class block sits at byte 0, exactly where
  Evo2's prior expects the leading `|…|` to be *lineage*; it may corrupt the native
  lineage-reading pathway. Test by moving class *after* the tax tag or folding it in as a
  pseudo-rank `|d__…;s__…;b__<class>|`. Requires a retrain (adapter learned class-first), so
  **bundle with the next CFG-dropout retrain**, treat as formatting hygiene not a lever.
  (2) **Nucleotide-context seeding** — the one *native* functional handle: seed with a real
  class-diagnostic ORF (NRPS C-A-T / PKS KS-AT-ACP nt context) and see if Evo2 continues
  in-class (it autonomously learned coding/protein features). Pure-inference, no retrain —
  worth a cheap diagnostic alongside CFG.
- **Consequence for the fork:** levers ranked CFG (option 1) > per-class/class-embedded
  adapters (option 3) ≫ native-format alignment (~option 4, low). CFG diagnostic scaffolding
  written: `evo2/scripts/cfg_generate.py` + `experiments/probes/run_cfg_diagnostic.sh` (two-stream
  CFG on the existing v2 adapter, cond=`|COMPOUND_CLASS:X|{tax}` vs uncond=`{tax}`, w-sweep,
  w=1↔non-cached-oracle correctness gate).
- Citations: `vortex/model/tokenizer.py:126-193`; `evo2/utils.py:35-75` (make_phylotag_from_gbif);
  `evo2/configs/evo2-7b-262k.yml`; BioNeMo Evo2 Data-Prep docs (Evo2TaxonomyLineage,
  random_lineage_dropout); Evo2 paper Methods §4.1.2/4.1.3; Arc cas9/cas12/cas13 finetune.

### CFG (classifier-free guidance) diagnostic — the #1 lever, built (2026-07-21)

Since no native class prior exists, we amplify the (present-but-weak: ratio ~1.08, near-floor
correct_class) class signal at **sampling time** rather than via more capacity/data.
`evo2/scripts/cfg_generate.py` + `experiments/probes/run_cfg_diagnostic.sh` run two-stream CFG on the
**existing v2 adapter (no retrain)**: per step, `cond = |COMPOUND_CLASS:X|{tax}` and
`uncond = {tax}` (class dropped) each produce next-token logits; sample from
`logits = uncond + w·(cond − uncond)`, sweeping `w∈{1,3,5}`. Rising correct_class with `w` ⇒ signal
is real and amplifiable ⇒ retrain with random class-dropout + CFG. Flat ⇒ per-class adapters.
**Caveat:** v2 was NOT trained with class-dropout, so `uncond` is a *proxy* null, not a learned one —
a high-`w` coherence collapse (coding_density/is_bgc) is the expected OOD failure mode, read
cautiously; the clean version retrains with dropout first. **Correctness gate:** at `w=1`, CFG
reduces to plain conditional, so it must equal an independent non-cached-recompute greedy
token-for-token; the runner aborts on mismatch before trusting any `w>1` number. We drive
`wrapper.model(x, inference_params_dict=…)` directly for two states rather than the high-level
`generate()` (see [bugs.md](bugs.md): single-token-resume `seqlen_offset`). Read via the SHORT
classes (terpene/ectoine/betalactone) where 8 kb fully covers the cluster (no truncation confound).

**Result (2026-07-21):** validation gate PASSED (bookkeeping correct). Sweep w∈{1,3,5}: correct_class
0.067 → **0 → 0** while coding_density collapses 0.903 → 0.257 → **0.0**. Amplifying the class
direction did NOT raise correct_class — it pushed the model OOD into gibberish, with no transient
lift. **No amplifiable class signal** (consistent with the no-native-prior finding above). Caveat:
v2 wasn't trained with class-dropout, so the high-w collapse is partly the expected untrained-null
failure — so this doesn't fully rule out a *trained*-null CFG. Paired with the n=15 simple-class
confirmation (v2 correct_class 0.013, **base-Evo2 0.0** — LoRA adds coherence, not class), every
prefix-conditioning lever is now negative. **Leaning: per-class adapters** (resolves the "Open
architectural fork" toward per-class). Cheap tie-breakers before committing: a finer w-sweep
{1.5,2.0,2.5} (lift before OOD collapse?) and nucleotide-context seeding.

**Nucleotide-context seeding diagnostic (`evo2/scripts/seed_generate.py`, 2026-07-22).** The one *native*
functional handle Evo2 offers: prompt with the first ~2 kb of a REAL class-X core (its assembly-line
start) and let the model continue. Scores **continuation-only** — vortex `generate()` returns
generation with the prompt stripped, so the seed never enters the score and a correct-class result is
the model's own contribution, not seed leakage through antiSMASH. Arms: **base Evo2 + seed, no class
tag** (purest native-handle test) and **v2 + seed + class tag** (best practical). Read vs the no-seed
floor (~0.01-0.07 v2, ~0 base): a large lift ⇒ seeding is a usable control handle (provide a starter
gene → model extends the cluster in-class); flat ⇒ even a real exemplar doesn't hold the class.

**Outcome (2026-07-22):** the finer CFG sweep confirmed no amplifiable signal (correct_class flat
through w≤2, then collapses). Seeding gave a lift — v2+seed agg correct_class 0.37 on megasynthases;
base+seed 0/30 — but a next-directions workflow adversary rates it **likely-inflated**: (1) trivial
gene-continuation (the model may just finish the seeded megasynthase ORF — the tight is_bgc↔
correct_class coupling is that signature), (2) memorization (novelty was skipped), (3) confounded arm
(v2+seed+tag never isolated from v2+seed+no-tag). **Gated:** do not build on seeding until a
de-confound factorial settles it — 2×2 {base,v2}×{tag,no-tag} with seeds fixed, novelty gate ON,
housekeeping-seed negative, codon-truncated boundary (no ORF crosses seed→continuation), n≥15/class.
Parallel cheap gate for the whole program: a **class linear-probe** on Evo2 hidden states of real
cores (separable ⇒ decoding/steering problem, cheap fixes viable; not separable ⇒ must INSTALL a
representation — the only thing that justifies Quartz long-context).

### RESOLVED 2026-07-27/28: it's a STEERING problem, and seeding is exemplar- not label-conditioning

- **Probe verdict: class IS represented.** base Evo2 balanced_acc **0.911** (chance 0.091, shuffled
  0.089, n=991/11 classes); the v2 adapter is identical (0.906) so it added no class representation.
  Class peaks **mid-network (L16)** and **fades to 0.414 by L31** — i.e. the information exists but
  is not carried into the next-token distribution. **Decision: stop trying to INSTALL a class
  representation** (this de-prioritizes per-class adapters AND Quartz long-context for this purpose)
  **and start STEERING/DECODING with it.** The probe head doubles as the fast class scorer that
  guided decoding needs (one matmul at L16).
- **Seeding verdict: real but exemplar-driven.** All pre-registered criteria passed — novelty
  (420/420 PASS, max containment 0.024 ⇒ memorization ruled out), codon-truncation holds (0.217),
  shuffled seed collapses (0.0), leak 0. **But** the mismatch arm shows the continuation follows the
  **seed** (0.317) not the **tag** (0.067), and v2_notag == v2_tag ⇒ **the tag is inert**. So the
  honest claim is **exemplar-conditioned generation** ("extend/diversify a given cluster"), not
  label-conditioned de-novo generation; true effect ~0.283 (the n=10 pilot's 0.37 was optimistic).
  The adapter is still required (base_notag 0.0 → v2_notag 0.283).
- Together: **Evo2 knows the class but won't act on a label; it will act on an exemplar.** That is
  the project's core scientific finding and the basis for the next phase (steering + guided decoding,
  composed with retrieval-based exemplar seeding).

---

## 2026-07-27 — Repo split into two model tracks; GenomeOcean opened as a candidate substrate

**Decision: reorganize into `evo2/` + `genomeocean/` with shared infrastructure at the root.**
The dataset pipeline (`scripts/`), eval suite (`src/bgc_pipeline/evaluation.py` + its drivers),
class map (`config/`) and tests stay at the repo root; only model-specific code moved. *Why:*
the eval suite is the comparison instrument. Forking it per model (the obvious "two independent
folders" layout) would let the two copies drift and make any Evo2-vs-GenomeOcean number
untrustworthy. One dataset, one antiSMASH gate, two models. `tests/run_all.py` passes after the
move; `tests/` adds both `scripts/` and `evo2/scripts/` to `sys.path`.

**Decision: evaluate GenomeOcean-4B/`bgcFM` as a replacement substrate — but do NOT retire Evo2 yet.**
Measured on gputee against `splits_core` (`genomeocean/experiments/*.json`,
`docs/model_comparison_evo2_vs_genomeocean.md`):

- *The reason that matters:* GenomeOcean is a stock `MistralForCausalLM` with a **4,096-entry BPE
  vocabulary and 5 special tokens**, so a compound-class tag can be a **real token with its own
  trainable embedding row and output logit** (verified: 22 `[CLS_*]` tokens added, vocab 4096→4118,
  `[CLS_NRPS]` → single atomic id 4096, both `embed_tokens` and `lm_head` resized since
  `tie_word_embeddings=false`, ~0 GB cost). On Evo2's byte-level `CharLevelTokenizer` a class tag is
  just bytes with no pretrained prior — the mechanism we identified on 2026-07-21 as *the* reason
  conditioning failed. This does not guarantee conditioning works; it removes the diagnosed cause.
- *Throughput:* L=10,240 tok (52.7 kb) trains in **14.0 GB**; **bs=8 fits in 54.8 GB**. Evo2 at
  L=32,768 fits only `bs=1`. ~12.8× more nucleotides per micro-step, and generation at 74 steps/s
  × batch 24 ≈ 1,800 tok/s (~9,200 bp/s) on the plain HF backend. This directly attacks the
  project's chronic n≈15 evaluation ceiling, which has already overturned two conclusions.
- *Context, stated honestly:* on **strict cores** the advantage is modest (mega whole-fit 0.966 vs
  0.892) — context is not the differentiator on today's data and should not be used to sell the
  switch. On **whole antiSMASH regions** it is decisive: median mega region 47.2 kb ⇒ Evo2 fits
  **0%** whole, GenomeOcean **64%**. Probe C (2026-07-07) said the lever is seeing the *complete
  cluster*; that experiment is unrunnable on Evo2 at one GPU and runnable here today.
- *What it does NOT give us:* `bgcFM` is **unconditional**. It was fine-tuned on 1.72M dedup'd SMC
  BGCs with no product label; their T1PKS figure is a best-of-258,260 filtering result (4.3%
  antiSMASH-positive, their number), not a conditioned generation. There is also no public
  fine-tuning script, and the `gmeval` repo cited in their Methods is 404.
- *Costs:* 4B < 7B capacity; metagenome-trained, so our **GTDB lineage tags have no pretrained
  meaning** and taxon conditioning would need re-installing or dropping; SMC↔`splits_core` leakage
  is **unquantified** and must be measured before any novelty claim; BPE changes the basis for
  k-mer novelty calibration.

**Next experiment stays the same, but moves hosts:** the class linear-probe (progress.md action 3b)
is still the gating question — *is compound class linearly decodable from hidden states?* — and it is
cheaper on GenomeOcean (forward passes only, 4B, 5× fewer tokens). Run it on both; if it separates on
GenomeOcean-bgcFM and not Evo2, that is a decisive, cheap reason to switch. If it fails on both, the
problem is the data and no model swap fixes it.

## Activation steering — scaling parameterization (2026-07-29)

**Decision: steering strength is expressed in THREE explicit modes, and the coherence axis is
`--delta-norm` (absolute perturbation magnitude), not `--beta`.**

*Why.* The class direction norms at layer 16 span 17× (TERPENE 1.05 … ECTOINE 17.75). Two
different things we want to hold constant across classes are therefore in direct conflict:

| quantity | held constant by | what it controls |
|---|---|---|
| semantic push (class-mean offsets) | `--beta` | how far toward class X we move |
| physical perturbation ‖delta‖ | `--delta-norm` (or `--alpha`, relative) | whether the model survives |

You cannot hold both. Measured 2026-07-29: coherence damage tracks ‖delta‖, so **titrate
coherence on `--delta-norm`**, then convert the surviving ceiling into a per-class β via
`β = ‖delta‖_max / ‖v_class‖`. Every generated record now carries `steer_v_norm`,
`steer_applied_norm`, `steer_beta_equiv` so the two axes are always recoverable.

*Consequence, stated up front:* holding magnitude constant makes the semantic push vary
inversely. At ‖delta‖=2, TERPENE receives 1.90 class-mean offsets but ECTOINE only 0.11. If the
coherence ceiling lands low, **large-‖v‖ classes (ECTOINE, NRPS) may be un-steerable at layer 16
by this method** — a limit of the approach, not a tunable parameter.

## The LoRA supplies BGC-ness, not class — so base Evo2 is a control, not a fallback (2026-07-30)

**Decision: run Phases 0–4 on the v2 adapter, and add a BASE-model arm at Phase 1 as a
scientific control.**

*Why.* Measured, base vs v2:

| | base Evo2 | v2 (fine-tuned) |
|---|---|---|
| class readable at L16 | **0.9107** | 0.9062 |
| coding density | 0.606 | **0.893** |
| is_bgc (simple classes) | 0.00 | **0.12** |
| is_bgc, seeded | 0.183 | **0.417** |
| correct_class | ~0 | ~0 |

The fine-tune contributes **nothing** to class representation (base is marginally better) while
substantially improving the model's ability to write gene-cluster-like DNA at all. It therefore
does **job (1) of the seed** ("write a BGC") and not job (2) ("write THIS class") — see "The seed
does TWO jobs". This explains the whole conditioning history: coherent BGCs, zero class control.

*Consequences.*
- **Phase 4 needs v2**: with no seed, the adapter is the only thing supplying BGC-ness
  (is_bgc 0.00 → 0.12). Probably still not enough — hence the proposed "BGC-ness" direction.
- **Phase 1b (base arm) is cheap and diagnostic**: `acts_base.npz` is already cached so directions
  are a CPU build; only the causal tests need GPU (~1 h). If steering works on base, the result is
  a property of **Evo2 itself** rather than of our fine-tune — much stronger, and requires no
  training. If it works *better* on base, the fine-tune has dampened the class→output path
  (plausible: it was trained with class tags it learned to ignore), and only this test catches it.

*Stated limit.* "The LoRA adds no class information" rests on a single probe comparison
(0.9107 vs 0.9062, n=991). That is *linearly readable* class content only. It does not exclude
the fine-tune changing how **usable** that content is downstream — exactly what the base-vs-v2
steering comparison would measure, and what nothing run so far addresses.

## ⚠ OPEN DEBT: steering directions are fit on val+test — revisit the splits before publishing (2026-07-30)

**Decision: fitting directions on a genome-disjoint half of val+test is ACCEPTED FOR NOW to
establish whether the mechanism works, and MUST be redone before any reported result.**

*Why we did it.* `train` is length-biased by the curation pipeline — it was MMseqs2-deduped and
capped at ~4k/class while val/test were kept full, so long distinct cores survive and short
near-duplicate ones get collapsed. Measured: PKS median core 6,282 nt in train vs 1,158 (val) /
1,170 (test), a 5.37x mismatch, while val↔test agree to 1.01x. Directions fit on train did not
transfer (PKS cos 0.469, held-out AUC 0.740); refit on val+test they reach 0.928.

*The debt.* val was used for early stopping, and test is meant to be the untouched final holdout.
Fitting the *intervention* on test means a later "steering achieves X% correct_class" number is
not measured on clean data. This does not leak into model weights — it leaks into the tool — but
it is still leakage and it invalidates a headline claim.

*Mitigation in place:* the fit and evaluation halves are **genome-disjoint**
(`splits_core/valtest_{fit,eval}.jsonl`, 5,501 genomes each), so nothing scored in Phase 1 helped
define a direction. That makes the current mechanistic result sound; it does not make the split
publishable.

*The real fix, before any reported number:* carve a dedicated **direction-fitting split** that is
disjoint from train, val AND test. The cleanest route is to resample from `train` to match the
natural (val/test) length distribution — train has 47.5k cores, so the short PKS records needed to
match should exist; confirm before committing. Failing that, re-split from the pre-curation pool
into four parts: model-train / dir-fit / val / test.

## The seed does TWO jobs, and class-steering only addresses one (2026-07-29)

**Decision: seedless class-conditioned generation is treated as a TWO-problem milestone, not one,
and is sequenced after seeded steering is shown to work.**

*Why.* Taxonomy-only generation produces a valid BGC only ~3–12% of the time; seeded generation
reaches ~40% `is_bgc` and 0.283 `correct_class`. So the seed is doing two separable things:

1. **putting the model in "write a BGC" mode** (is_bgc 0.03–0.12 → 0.40), and
2. **specifying which class** (the part steering is meant to replace).

Activation class-steering only attacks (2). Drop the seed and (1) goes too — and **you cannot
steer the class of a sequence that is not a gene cluster.** That, not class control, is what the
3% floor has been reporting all along. Any seedless attempt must solve (1) separately; the natural
candidate is a **"BGC-ness" direction** built the same way (real cores vs ordinary genomic DNA)
and composed with the class direction.

*The sharper next experiment is a seed-length titration* — 2000 → 1000 → 500 → 200 → 100 → 0 nt,
with and without steering. The decisive comparison is whether **steering compensates for a
shortened seed**: if 500 nt + steering matches 2000 nt alone, steering is doing real work and
there is a measured path toward zero. If class control collapses as soon as the seed shortens
regardless of steering, the seed is load-bearing in a way steering cannot replace.

*Note the seeded product is not a consolation prize:* seeded generations are verified novel
(max training containment 0.024) and survive seed truncation, so "extend this cluster into a novel
one" is a real capability, and cross-class override (seed ectoine → generate PKS) would be new.

## Evaluation must have dynamic range before it is used to compare (2026-07-29)

**Decision: no steering configuration is evaluated with antiSMASH until the unsteered control
under the same prompting regime clears a usable floor.**

*Why.* The α sweep's unsteered control scored **1/30 is_bgc (3.3%)** under taxonomy-only
prompting; all nine steered cells scored 0/30. Detecting a lift off a 3% floor at n=5/class is
impossible, so ~9 GPU-hours produced no information in either direction. Meanwhile *seeded*
generation already reaches **0.283** — a regime with real dynamic range.

*Practical rule.* Before a comparison: state the baseline rate, the effect size worth detecting,
and the n required. Prefer graded readouts — the layer-16 probe head is one matmul and gives a
continuous class score — over a binary antiSMASH gate resting on the floor. Reserve antiSMASH
for confirming configurations that a cheap graded readout has already selected.
