# Steered class-specific generation — the plan, and how it ended

> ## ⛔ CLOSED 2026-08-10 — do not run another steering variant
>
> **Steering does not install a class, at any layer, dose, or geometry.** ΔP(target) is null in
> every arm on every instrument — binary and continuous, single-layer and 9-layer stacked, at
> doses from 0.15 to 11.9 class-units. The generator represents class (probe 0.93) and does not
> consume it, which is a training-time fact no inference-time intervention reaches.
>
> **Retracted 2026-08-10:** a companion claim that the direction reliably *deletes* a class
> (ΔP(seed) −0.308, p = 0.0063) was a **leakage artefact** — the probe had been fit on val+test
> and applied to val/test-seeded generations. Refit train-only it is −0.177 at p = 0.146: a
> consistent negative trend at the two higher doses, not a result. Phase 1's teacher-forced
> ablation asymmetry (z = 4.8) is independent of the probe and does stand.
>
> Not dose, not depth, not geometry — all three were tested after the early defects were fixed.
> Jump to [Verdict](#verdict--the-programme-is-closed-2026-08-10) for the full table and what to
> spend on next. **Soft prefixes were then tried and also failed** (2026-08-10, `decisions.md`);
> the ranked plan built from a literature sweep is `docs/conditioning_next_steps.md`.
>
> **Everything below is preserved as written**, including predictions that turned out wrong, so
> the reasoning can be audited against the results. Corrections are marked inline and in
> "How that assessment held up".

**Written 2026-07-29.** Every geometric number here was recomputed directly from
`class_probe_sweep/acts_v2.npz` and reproduces exactly. A denser, more technical version of the
same program (agent-generated, with additional audit findings) is kept alongside as
`steering_program_technical.md`.

**Goal:** make Evo2 write a gene cluster of a class we choose, by nudging its internal state
during generation rather than by asking with a label (labels are inert — established 2026-07-21).

---

## What we know going in

**1. The class information is real.** A trained classifier reads compound class off the model's
internal state at layer 16 with 91% accuracy (random guessing = 9%).

**2. Everything we injected so far was the wrong thing.** We built the nudge as
*average state of class X* − *average state of everything*. But 98% of how clusters differ from
each other is just **how long they are**, so that subtraction mostly captures length, not class.
All eleven "class directions" came out ~93% identical to each other, and for the pairs we care
about the arrow points **backwards** (steering toward PKS moves output toward NRPS). Zero
correct-class results were guaranteed before those experiments ran.

**3. The fix is free and works.** Removing the length component recovers real class directions:

| arrow recipe | NRPS→PKS | ECTO→TERP | NRPS→TERP | PKS→RIPP |
|---|---|---|---|---|
| class − global average (what we used) | 0.221 | 0.070 | 0.951 | 0.565 |
| class − average of other classes | 0.214 | 0.070 | 0.956 | 0.566 |
| class-vs-class | 0.839 | 0.933 | 0.940 | 0.566 |
| **class − average of others, length-stripped** | **0.985** | **0.999** | **0.995** | **0.951** |

(1.00 = perfectly separates the two classes, 0.50 = coin flip, below 0.50 = points backwards.)
The **length-stripping** is the load-bearing step — every recipe works once it is applied.
We use *class − average of all other classes, length-stripped*: it gives **one direction per
class** ("make this a PKS"), which is what generation needs, rather than one per pair.

**4. We were pushing far too hard.** The old strength setting injected a nudge 1.5–6× larger than
the entire spread of the real data. Along the corrected direction the natural scale is much
smaller — the data's spread along the class axis is 0.24–0.36 versus 8.1 along the length axis.
All strengths below are therefore quoted in **class-units**: 1 class-unit = the distance between
one class's average and the others' average. That is interpretable and comparable across classes.

**5. Our success test could not measure anything.** With no steering at all, the antiSMASH gate
fired 1 time in 30 (3.3%). Comparing 0/30 to 1/30 is not a measurement. Seeded generation
reaches 28%, so that is the regime with room to detect a change.

**6. The live risk — class may be present but unused.** Class readability peaks at layer 16
(0.906), holds to layer 23, then **collapses to 0.354 by the output layer**. The network appears
to discard class in exactly the last quarter that decides what gets written. And base Evo2 —
never fine-tuned, no class prior — reads class just as well as our model (0.911 vs 0.906).
Both are consistent with class being an *echo of the input* rather than a control the generator
uses. If so, no correction to direction or strength rescues steering. **Phase 1 tests this before
we spend anything meaningful.**

---

## The plan

| Phase | Question it answers | GPU-h | Cumulative |
|---|---|---|---|
| **0** | Do we have real class directions? | **0** | 0 |
| **1** | Does the model actually *use* class to decide what to write? | **1.5** | 1.5 |
| **2** | How hard can we push before the DNA stops looking like genes? | **1.5** | 3.0 |
| **3** | **Can we steer class in generation?** ← the decisive test | **4.0** | 7.0 |
| **4** | Does it hold up de novo, at the gold-standard gate? | **7.0** | 14.0 |

For scale: the failed sweep alone cost 8 GPU-h and returned no information.

---

### Phase 0 — Build correct directions (free, CPU only, ~1 h of work)

Build, from activations already cached on disk, one direction per class per layer:
*average of class X* − *average of all other classes*, with the length component removed.

Do this at **layers 16, 20, 24 and 27** — not just 16. Layer 16 is where class is most
*readable*, but readability collapses after 23, so the layer where class is still *usable* may be
later. Layer becomes a variable to test rather than an assumption.

Also build **shuffled-label directions**: the identical recipe, but with class labels randomly
scrambled first. These carry no class information by construction and are the control every later
phase compares against. Without them, "steering did something" cannot be separated from "poking
the model does something."

**Quality gate — a direction is admitted only if it passes all three:**
- separates its class from the others at ≥ 0.90
- is stable: computed on two random halves of the data, the two versions agree ≥ 0.70
- carries essentially none of the length component

*Already verified to pass at layer 16 on every pair tested (0.95–0.999). The gate mainly exists
to catch implementation bugs and to check the later layers.*

**Gets us closer by:** giving us, for the first time, a nudge that points at class instead of at
length. Nothing else can proceed without it.

---

### Phase 1 — Does the model actually use class? (1.5 GPU-h)

Three independent tests. All are forward passes only — no generation — so they are fast. Run at
each candidate layer.

**Test A — validate against the thing that already works.** Seeding with a real exemplar
demonstrably controls class (28% vs the 3% floor). Take those already-generated seeded sequences
off disk, look at their internal state, and ask: **did seeding move the state along our class
direction?** If the one intervention known to control class moves the state along this direction,
the direction is causally relevant. If seeding controls class *without* moving it, we are pushing
on the wrong variable and everything downstream is void. **Nearly free — run this first.**

**Test B — the two-sided nudge.** Feed the first half of a real held-out cluster; measure how
surprised the model is by the true second half. Nudge toward the true class and re-measure; nudge
toward a different class and re-measure. If the model uses class, the correct nudge makes the true
continuation *less* surprising and the wrong nudge makes it *more*. **The gap between the two
directions is the signal** — generic damage from poking the model affects both equally and cancels.

**Test C — the ablation.** Instead of adding class, *delete* it: remove the class direction from
the state as the model reads a real cluster, and check whether its predictions get worse. If
deleting class information costs nothing, the model was not using it.

Every test is compared against (i) the shuffled-label direction and (ii) a positive control — an
intervention already known to change generation, so we know the measurement can detect something.

**Decision:**
- Any test shows a real effect above the shuffled-label control → proceed to Phase 2.
- **All three flat, while the positive control registers clearly** → this is the thermometer
  verdict: class is readable but not used. Stop, and move to per-class adapters. Cost of learning
  this: 1.5 GPU-h instead of the ~14 a full program would spend.

**Gets us closer by:** answering the one question that can invalidate the whole approach, before
any expensive generation.

#### Phase 1b — BASE vs FINE-TUNED arm (+1 GPU-h, run alongside)

Everything so far has run on the **v2 LoRA adapter**. Two measurements say the fine-tune and the
class signal are largely independent of each other:

| | base Evo2 | v2 (fine-tuned) |
|---|---|---|
| class readable at L16 | **0.9107** | 0.9062 |
| coding density | 0.606 | **0.893** |
| is_bgc (simple classes) | 0.00 | **0.12** |
| is_bgc, seeded | 0.183 | **0.417** |
| correct_class | ~0 | ~0 |

**The LoRA taught "write a gene cluster", not "write THIS class".** It contributes nothing to the
class representation (base reads class marginally *better*) while substantially improving BGC-ness.
That is exactly job (1) of the seed — see "The seed does TWO jobs" in decisions.md — which is why
the fine-tune matters most in Phase 4, where no seed is available to supply it.

Because the LoRA adds no class information, our directions should transfer to base Evo2. Repeat
Phase 1 there (`acts_base.npz` is already cached, so the direction build is CPU-only):

- **works on base** ⇒ class-steerability is a property of **Evo2 itself**, not our fine-tune — a
  materially stronger and more general claim, and it needs no training at all.
- **works only on v2** ⇒ the fine-tune is doing something necessary; find out what.
- **works BETTER on base** ⇒ the fine-tune *dampened* the path from class representation to
  output — plausible, since it was trained with class tags it learned to ignore. This is the only
  test that would catch that.

**Honest limit on the premise:** "the fine-tune adds no class information" rests on ONE probe
comparison (0.9107 vs 0.9062, n=991 val cores). That measures *linearly readable* class content.
It does not rule out the fine-tune changing how **usable** that content is downstream — which is
precisely what a base-vs-v2 steering comparison measures and nothing we have run so far does.

**Model choice for the rest of the program:** v2 for Phases 0–3 (it is the better generator and
what we would deploy); v2 for Phase 4 (it is the only thing supplying BGC-ness once the seed is
gone); base as a scientific control at Phase 1b.

---

### Phase 2 — Find the usable push strength (1.5 GPU-h)

Generate 2,000-base sequences at strengths of 0.25, 0.5, 1, 2 and 4 class-units, plus unsteered
and a shuffled-label arm at the top strength. Six cells × 20 sequences.

Necessary even though we already ran a strength sweep, because **that sweep titrated the length
axis**. The corrected direction is a different, much lower-variance direction with unknown
tolerance.

**Measure longest-ORF and ORF count, not coding density.** Coding density is insensitive here —
on real data it reads 0.888 for sequences that pass the class check and 0.879 for those that
fail. Longest-ORF separates them 858 vs 605. Also track **realized sequence length**: steering
suppresses the model's stop signals, and length differences manufacture artifacts downstream.

**Decision:** pick the largest strength whose longest-ORF is statistically indistinguishable from
unsteered *and* whose length is within 10% of unsteered. **If no strength ≥ 0.5 class-units
survives, stop and report** — a direction that cannot take half a step without wrecking the
sequence has no operating point.

**Gets us closer by:** giving us the strongest usable setting for the real test.

---

### Phase 3 — The decisive test: steered class-specific generation (4 GPU-h)

**Cross-class override.** Start the model on a real cluster of class A, steer toward class B, and
ask whether it builds B's machinery instead of A's.

Run here because this is the only regime with a **measured floor and a measured ceiling**
(0.067 → 0.283 at n=60). In the taxonomy-only regime everything sits on a 3% floor where nothing
is detectable.

Three arms, paired on the same seed exemplars, 45 sequences each:

| arm | direction | strength |
|---|---|---|
| A | corrected class direction | best from Phase 2 |
| B | **shuffled-label direction** | matched |
| C | unsteered | none |

**Important:** the hook must be gated to apply only to *newly generated* positions, not to the
seed being read in. Otherwise we corrupt the exemplar that is carrying the class signal we are
trying to override.

**Primary measure:** target-class machinery present **and** seed-class machinery absent. Both
halves matter — appearing without disappearing is not an override.

**Decision:** proceed only if arm A beats arm B (the shuffled control), paired, at p < 0.05, with
A and B matched on sequence length and ORF size. **A-vs-C is reported but proves nothing** —
shuffling the seed with no steering at all already suppresses seed-class markers, so beating
"unsteered" is worth zero.

**Power, stated honestly up front:** 45 paired sequences detects a rise from ~5% to ~25%. It
**cannot** resolve a rise to 12% — that needs ~130 per arm and another 8 GPU-h. Declared out of
scope in advance so we do not over-read a null.

**Gets us closer by:** this *is* steered class-specific generation. Everything before it is setup.

---

### Phase 4 — De novo confirmation (7 GPU-h, only if Phase 3 succeeds)

No seed. Taxonomy prompt only, corrected direction, best strength, 90 sequences per arm at 6,144
bases, scored by antiSMASH. Size the run from Phase 3's measured effect, not from the 3% floor.

Two required additions: a **novelty check** against the training set (the directions are fit on
real cores, so an unchecked result cannot be distinguished from retrieval), and per-record IDs so
the novelty results can be joined.

**Gets us closer by:** turning "we can steer class" into "we can generate a requested class from
nothing," which is the actual project goal.

---

## Phase 5 — Was the edit being DILUTED? (2026-08-10) — **no, and the reverse is true**

Phase 3 killed steering at layer 16. The leading remaining excuse was **dilution**: the edit is
made 16 blocks from the output, and the residual stream explodes by eleven orders of magnitude in
the last blocks, so perhaps the nudge simply never survives to the token distribution. If so, the
fix is to inject later, and multi-layer steering becomes the obvious follow-up.

### The geometry that made L27 the right place to test

Recomputed from `class_probe_sweep/acts_valtest_fit.npz` (n=3,430 real cores, the same cache the
directions are fit from), via `scripts/` scratch analysis:

| layer | 16 | 20 | 24 | **27** | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|
| mean ‖h‖ (pooled) | 8.95 | 9.78 | 6.54 | **11.25** | 5.47e3 | 8.66e6 | 3.69e12 |
| class AUC (held-out) | 0.923 | 0.927 | 0.885 | **0.835** | 0.590 | 0.610 | 0.553 |
| PC1 share (length axis) | 97.1% | 93.9% | 13.0% | **17.4%** | 45.6% | 61.7% | 91.3% |

**L27 is the last layer where the class direction is still real, and the last one before the
residual stream blows up.** The blow-up is abrupt and sits entirely between L27 and L28 (486x in
one block). Directions were rebuilt at layers 16/20/24/27 by the identical recipe on the
identical data (`valtest30_multilayer.steerdirs.npz`, 30 shuffled-label control sets); as a
regression check, **0 of the L16 arrays differ** from the file Phase 1/3 ran on.

### Two measurements, both new

**A. A dose is only comparable across layers if it is a fraction of the LOCAL residual norm.**
`seed_generate.py --steer-norm-frac` recomputes ‖delta‖ = frac x ‖h‖ at every generated position,
and the output record now carries the *realized* ‖h‖, ‖delta‖ and dose rather than the requested
one. Measuring that live exposed a second-order version of the retired `_ref_norm` bug:

| layer | 16 | 20 | 24 | 27 |
|---|---|---|---|---|
| mean ‖h‖, **pooled cache** | 8.95 | 9.78 | 6.54 | 11.25 |
| mean ‖h‖, **live at the hook** | **6.69** | **8.34** | **13.77** | **31.97** |
| ratio | 0.75x | 0.85x | 2.10x | **2.84x** |
| 1 class-unit, in live ‖h‖ | **0.082** | 0.095 | 0.095 | **0.056** |

Pooling averages vectors that point in different directions, so it shrinks the norm — by a
different factor at each depth. Any dose derived from the pooled cache is therefore mis-scaled,
and increasingly so with depth. **In the units that actually govern a generated token, one
class-unit gets WEAKER with depth (0.082 -> 0.056), the reverse of what the cache implies.**
Consequence for the existing record: Phase 3's L16 dose of 1 class-unit was **0.082** of the local
norm, slightly *stronger* than the 0.061 the cache suggested — so Phase 3's null is not an
under-dosing artefact.

**B. `evo2/scripts/steer_reach.py` — does the edit reach the output at all?** For each layer, at a
dose fixed as a fraction of the local norm, it measures `reach` = mean KL(p_steered ‖ p_base) of
the next-token distribution over real held-out cores (n=40, 4 classes), against 3 shuffled-label
control directions.

| layer | reach @ frac 0.061 | reach @ frac 0.16 |
|---|---|---|
| **16** | **0.00136** | **0.01011** |
| 20 | 0.00087 | 0.00604 |
| 24 | 0.00053 | 0.00359 |
| **27** | **0.00044** | **0.00288** |

**Reach falls monotonically with depth. The same relative edit at L27 moves the output
distribution 3.5x LESS than at L16.** Dilution-by-depth is falsified, and in the direction
opposite to the hypothesis.

Mechanistically this should have been the expectation: the residual stream is *additive*, so an
edit at L16 is both carried forward unchanged **and** read, amplified and re-expressed by the 11
blocks that follow. An edit at L27 has only 4 blocks left to be read by. In a residual network,
"closer to the output" means **fewer opportunities to be used**, not more influence. The
late-layer norm explosion does not rescue this: it happens *after* both injection points and so
attenuates them equally.

**Two things this measurement does NOT show.** (1) The class-specific `gap` statistic is ~0 at
every layer here — but the context used is taxonomy + 600 nt of the real core, which is Phase 1's
`reinforce` regime, where Phase 1 also found nothing. It neither confirms nor contradicts Phase
1's `create`-context result (p = 0.040). (2) A `z` computed against a 3-control spread is not a z
— L24 reads z = 16.5 off a sd of 0.00003 on an effect of 0.00017. `steer_reach.py` now suppresses
z below 5 controls; the permutation p (floored at 1/(n+1) = 0.250 here) is the honest statistic.

### C. The generation test at L27 — 0/48, at doses up to 16x Phase 3's reach

Teacher-forced measurement is exactly what over-promised in Phase 1, so a teacher-forced null was
not allowed to close this. `evo2/experiments/probes/run_steer_l27.sh` runs the Phase 3 seeded
cross-class design (seed a real class-A core, steer the continuation toward class B) at layer 27,
n=12 per dose, doses fixed as fractions of the live residual norm.

| dose (frac of live ‖h‖) | = class-units | coding_density | **target-class markers** | seed-class markers |
|---|---|---|---|---|
| 0.061 | 1.1 | 0.925 | **0/12** | 6/12 |
| 0.16 (≈ reach-matched to Phase 3) | 2.9 | 0.883 | **0/12** | 5/12 |
| 0.32 (≈ 4x Phase 3's reach) | 5.9 | 0.896 | **0/12** | 6/12 |
| 0.64 (≈ 16x) | 11.9 | **0.684** | **0/12** | 3/12 |

0 of 48 sequences, 0 skipped. Pre-registered stop rule (set before the data landed): proceed to a
paired real-vs-shuffled arm only if some dose reached >=2/12 target markers with coding_density
>= 0.85. Nothing fired, so **Stage 2 was not run** and the 72 GPU-min were not spent.

**The dose-response is the informative part, and it is the Phase 3 signature exactly.** As dose
rises, the target class never appears; what happens instead is that the *seed's own* class
markers erode (6/12 -> 3/12) and coherence collapses at the top dose (0.925 -> 0.684). That is
**degradation, not redirection** — the model is being pushed out of distribution, which is CFG's
failure mode, not conditioned. The usable window at L27 is frac <= 0.32.

**Power, stated honestly.** 0/12 per dose rules out an effect of >=22% at that dose; it does not
rule out a small one. What makes this decision-grade is the conjunction: L16 was already null at
n=140 pooled (Phase 3), L27 has **3.5x less** influence on the output, and 0/48 here spans 1.1 to
11.9 class-units.

### Caveat on any L27 null

The L27 directions are genuinely weaker than the L16 ones — held-out AUC 0.861 / 0.818 / 0.867 /
0.789 for NRPS / PKS / TERPENE / RIPP versus 0.872 / 0.927 / 0.972 / 0.920 at L16 (shuffled-label
null p95 = 0.658; only 5 of 12 classes clear the 0.90 admissibility gate at L27 vs 9 at L16). All
four still beat the null, but a null *result* at L27 has "the direction was noisier there" as a
live competing explanation. A positive result would not have had that problem.

---

## Phase 6 — Multi-layer: the direction DELETES class but cannot INSTALL it (2026-08-10)

Phase 5 measured ONE injection point at a time, and that does not close the multi-layer case.
Two mechanisms make a stack different in kind, not just in dose:

1. **Re-assertion.** Nothing obliges the blocks after L16 to preserve an added component — it is
   not a state the model would have produced itself, so downstream computation can overwrite it.
   *Falling reach with depth is exactly what such erasure looks like.* Re-adding at every layer is
   closer to CLAMPING the class coordinate than nudging it once, and no single-layer measurement
   can see that difference. (Phase 5 originally called multi-layer closed on this evidence. That
   was wrong: the evidence was consistent with the mechanism it was meant to rule out.)
2. **Damage is per-layer; effect may be cumulative.** Coherence collapses when ONE edit gets
   large. Nine small edits can sum to a much larger total push while each stays in the safe band.

`evo2/experiments/probes/run_steer_stack.sh`: 9 layers (L10-L27, each with its OWN direction and
its OWN class-unit), 3 per-layer doses, every dose with a shuffled-label twin, seeded cross-class
design, n=12/arm.

**Binary gates: 0/12 target markers in every arm, real and shuffled alike.** Uninformative by
construction — the marker gate's TPR is 0.717 at this length.

**The continuous readout on the identical sequences**, paired real vs shuffled-label on the same
exemplars (the only comparison that isolates the class direction from generic perturbation):

| per-layer dose | ΔP(target) | p | **ΔP(seed)** | **p** |
|---|---|---|---|---|
| 0.027 | +0.075 (9/11) | 0.065 | −0.175 | 0.549 |
| 0.082 | +0.007 (6/12) | 1.000 | −0.090 | 0.388 |
| **0.16** | +0.107 (9/12) | 0.146 | **−0.308 (1/12 up)** | **0.0063** |

> **⚠️ CORRECTED 2026-08-10.** The table above used a probe fit on val+test and applied to
> val/test-SEEDED generations — it had seen the seeds. Refit train-only: ΔP(seed) at frac 0.16
> becomes **−0.177 (3/12 up, p = 0.146)**, not −0.308 at p = 0.0063, and the 0.027 cell changes
> sign to +0.056. **The generation-level deletion claim is withdrawn**, along with the damage
> control built on it. ΔP(target) remains null at every dose under both probes — steering
> installs nothing — and Phase 1's teacher-forced ablation asymmetry (z = 4.8) is independent of
> the probe and unaffected.

**The real direction strips the SEED's class identity 3x harder than an equal-magnitude random
direction** (Bonferroni over all 6 tests: 0.038, still significant). It never significantly
installs the TARGET's.

**Not an artefact of damage.** At frac 0.16 the real arm is more incoherent (coding 0.706 vs
0.834), but corr(Δcoding, ΔP(seed)) = **+0.002** — no relationship. Restricted to the 7 exemplars
where the real direction did NO more coding damage than the shuffled one, ΔP(seed) = **−0.393**
(same direction, larger; n=7, p=0.125 — underpowered, not contradicting).

### ⇒ ABLATION WORKS, INJECTION DOES NOT  — *(the deletion half is RETRACTED; see the correction above)*

The vector carries genuine class information — enough to specifically erase a class that is
present — but adding it does not make the generator write the target's machinery. Erasing a
coordinate the model already uses is easy; writing one it does not consume changes nothing
downstream. This reproduces Phase 1's asymmetry (ablation z=4.8 strong, nudge marginal) in
GENERATION rather than teacher-forced scoring, and it is the mechanistic reason the entire
inference-time family fails.

**A capability that falls out of the negative result:** class *suppression* works. "Generate a
BGC that is NOT an NRPS" is achievable with what is already built.

### The methodological finding, which outlives the steering programme

Until 2026-08-10 **every class readout in this project was binary**. A threshold gate bounds a
LARGE effect and is silent on a small one, and the thresholds here are not tight:
`class_markers` TPR 0.717 at 3 kb, antiSMASH detecting ~1/3 of seeded 3 kb generations. An effect
of the size found above was **invisible by construction** in every experiment we ran. The
continuous `class_probe` check (now permanent, diagnostic-only, calibrated at both ends) is the
fix; see README.md.

---

## Verdict — the programme is closed (2026-08-10)

Every original kill criterion was met, and the three defects that made the early nulls
uninterpretable (wrong axis, toxic dose, floor-bound readout) were all fixed first. Steering was
then tested properly and failed:

| stage | result |
|---|---|
| P0 directions | corrected, length-stripped, admissible; L16 arrays reproduce byte-identically |
| P1 teacher-forced | the model DOES use class — ablation z=4.8, nudge p=0.040 |
| P2 dose | damage-free band established |
| P3 generation @ L16 | **null** — 2/140 vs 1/142 shuffled, chance 3/144 |
| P5 depth | **reach FALLS with depth** (L16 0.0101 → L27 0.0029); 0/48 at L27 |
| P6 multi-layer | **never installs class**; 0/12 target everywhere. (A "deletes class" result at p=0.0063 was retracted 2026-08-10 as a probe-leakage artefact; clean p = 0.146.) |

**Do not run another steering variant.** Not other layers, not other dose schedules, not other
direction recipes. The mechanism is identified and it is none of those.

**Next spend is training-time coupling**, in this order:
1. ~~**Per-class soft prefixes**~~ — **RUN 2026-08-10, NEGATIVE.** Trained cleanly and separated
   per class from an identical init, but bought ~0.003 nats and gave `correct_class` 0/12 de novo.
   The bound is narrow: 65k parameters changing only the INPUT. See `decisions.md`, and
   `docs/conditioning_next_steps.md` for what replaced it at the top of the queue.
   *Original rationale, retained:* cheapest discriminating test (~1 GPU-day). Labels fail and
   exemplars work, so *learn a synthetic exemplar* in embedding space rather than assert a byte
   string with no pretrained prior. Needs hook-based plumbing: Evo2 is not an HF
   `PreTrainedModel`, so peft prompt-tuning will not drop in.
2. **Per-class LoRA adapters** — no conditioning interface to fail; class is which weights you
   load. Does not scale past a handful of classes; do 3-4 as proof.
3. **GenomeOcean + a real trainable class token** — removes the structural obstacle (Evo2's
   byte-level tokenizer gives the class tag no pretrained prior), and scales.

**Bank now:** exemplar-conditioned generation is a validated capability (0.283 vs a 0.067 floor,
memorization ruled out, four pre-registered controls passed).

**Standing debt:** directions AND the class probe are fit on val+test. Refit train-only before
any externally reported number.

---

## When to give up  — *(the criteria as written 2026-07-29; all of them fired. Retained
## because the programme's decisions should be readable against the rules set BEFORE the data.)*

**Stop and switch to per-class adapters if:**

- **Phase 1 is flat across all three tests while the positive control registers.** Class is
  readable but not used — the thermometer verdict. This is the most likely way the program dies,
  and it costs 1.5 GPU-h to find out.
- **Phase 2 finds no usable strength** below the point where sequences fall apart.
- **Phase 3's arm A does not beat the shuffled-label arm B**, with both properly matched. This is
  a real, controlled, adequately powered negative and it kills the approach.

**Do NOT accept as evidence against steering:**
- 0/30 on a 3% floor — that happens 37% of the time even when the intervention works
- beating the unsteered arm — a shuffled seed does that with no steering at all
- a coherence verdict based on coding density — it barely moves between good and bad sequences
- **any result from the old direction files** — those vectors are the length axis and score 0.07
  to 0.22 on the pairs that matter

**Honest assessment.** The wrong direction and the excessive strength fully explain every null
result on disk, and both cost nothing to fix — which means steering has never actually been
tested and deserves the 7 GPU-h to Phase 3. But the thing most likely to kill it is not a bug:
class carries ~2% of the variance, the network discards it before the output, and a model with no
class prior reads it just as well. If Phase 1 comes back flat, the honest reading is that class
at layer 16 is a readout of the input rather than a control variable, and per-class adapters are
the correct next spend.

### How that assessment held up (2026-08-10)

**Right about the outcome, wrong about the route.** Per-class adapters *are* the correct next
spend, and the reason is close to the one anticipated — but Phase 1 did **not** come back flat.
It came back positive (ablation z=4.8, nudge p=0.040), and the programme correctly proceeded on
that basis. The kill came at Phase 3, from the criterion written above: arm A did not beat the
shuffled-label arm B.

**What the plan did not anticipate**, and what took three extra phases to establish:

1. **Teacher-forced good news does not transfer to generation.** The plan flagged this as a
   caution; it turned out to be the whole story. A per-base effect far too small to redirect
   3,000 sequential sampling decisions still registers clearly in log-likelihood.
2. **"Present but unused" was too coarse.** The real finding is directional: the class coordinate
   can be **deleted** but not **installed**. Phase 1's own asymmetry (ablation strong, nudge
   marginal) was the tell, and nobody read it that way at the time.
3. **The readout ceiling was itself unmeasured.** Every gate in this programme was binary, and
   their sensitivities (marker TPR 0.717, antiSMASH ~0.33 on 3 kb generations) were only measured
   on 2026-08-10 — *after* the negative results they had produced. The programme reached the right
   verdict, but for most of its life it could not have distinguished a small real effect from
   zero. The continuous `class_probe` check exists because of that gap.

**The rule worth carrying forward:** an instrument's sensitivity and false-positive rate are part
of the experiment, not housekeeping to do later. A null is only as strong as the measured ceiling
it is read against.
