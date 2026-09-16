# DEFECTS AND AMENDMENTS

Split out of `SPEC.md` on 2026-09-16. This is the PROVENANCE record: every spec amendment, and
every defect found and fixed, dated. It is not the method — `SPEC.md` is — and it is not the
results, which are in `RESULTS.md`.

⚠ **`SPEC.md` still cites these by their original numbers** (`§12.A3` appears 6 times, `§12.A5`
5 times, and so on). Those references now resolve HERE. The numbering is deliberately unchanged
so that citations already written into artifacts, commit messages and FINDINGS keep resolving.

⚠ **Why this is kept rather than deleted.** Six of these amendments record defects that changed
published numbers, and three of those were introduced BY THE FIX FOR ANOTHER ONE. A reader
asking "why does GenomeOcean's α grid differ from Evo2's" or "why were these run directories
quarantined" cannot answer it from the method alone. §14.4 in particular is what distinguishes
a quarantined run from a lost one.

---

## Spec amendments (§12.A1 – §12.A10)


**A1 — 2026-09-07. `W3` is a learned per-attention-site conditioner, not KV-prefix tuning.**
The original text specified "learned key/value states supplied to every attention layer". That is
**not implementable on this substrate**, established by inspection [M]:
* peft's `PrefixTuning` injects through `past_key_values` and requires
  `prepare_inputs_for_generation`. Vortex's model has neither — its forward is
  `(x, inference_params_dict=None, padding_mask=None)`, with no HuggingFace KV interface.
* The attention kernel is `FlashSelfAttention` over a **packed** `qkv` of shape `(b, s, 3, h, d)`.
  Packed self-attention requires q, k and v to share a sequence length; a KV prefix by definition
  makes k/v longer than q. There is no hook point that prepends to k/v alone.
* The inference path *does* split them (`_update_kvcache_attention(qkv[:,:,0], qkv[:,:,1:], …)`)
  but only under `inference_params`, so training and generation would need two different prefix
  mechanisms that had to agree exactly — a correctness risk larger than the arm is worth.

⚠ **The paper must not describe `W3` as prefix tuning.** Prefix tuning is a named published method
with a different mechanism and a different capacity profile. The built arm keeps the three
properties the arm exists to test — learned by gradient descent, applied at every attention layer,
not input-only — and is described as what it is.

*This amendment was filed late: the change was first recorded only in a code comment
(`interventions.py`), which §0 forbids. A §10 unblinding diff run against the unamended text would
have flagged it as an implementation discrepancy rather than a decision.*

**A2 — 2026-09-08. `max_epochs` 12 -> 40, and a binding cap is a protocol failure, not a result.**

§6.0a's stated principle is "early stopping, not a guessed epoch count". Measured on two
independent training passes, `W2_REDOX_COFACTOR` **hit the 12-epoch cap both times without early
stopping firing**, and on the second pass its last checkpoint was its best — the signature of a run
still descending when it was cut off.

The residual slope is real, not noise. Over its final five checkpoints it improved **1.62e-3**,
against a measured trajectory-noise floor of **up to 5.1e-4** (FINDINGS 4a.7) — roughly 3x. So
for one of four classes the epoch count, not convergence, decided when training stopped, while the
other six arms early-stopped at 6-12 epochs. Comparing a converged arm against a truncated one
confounds the axis the benchmark reports (method) with one it does not (training budget).

The cap is therefore raised to a value chosen so that it **never binds**, leaving `patience` as the
single termination rule for every arm. It is not tuned to where any arm converges: a cap set from
one arm's observed convergence is the same guessed count with an extra step.

**Every training report now records `converged` (did early stopping fire), and a run that exhausts
its epochs is reported as a failure to converge rather than as a trained arm.** The defect was
visible in the overnight artifacts as `stopped_early: false` and went unread for a day because
nothing was looking at that field.

Raising the cap does not perturb the six arms that early-stopped: they terminate before it either
way. It changes their `train_config_hash`, which is correct — the config genuinely differs — and
all arms are retrained together so no comparison spans two values.

**A3 — 2026-09-08. `common_n` counts RECORDS, not clusters; a cluster's non-representative members are training data, not redundancy to discard.**

§4.4.3 as written selected `common_n` clusters and took one representative from each, on the
stated grounds that a draw over records "would over-represent large clusters and reinstate the
redundancy clustering exists to remove". That rationale conflates two different jobs the dataset
does:

* **Statistical independence** — the unit for a held-out claim. Near-duplicates are not independent
  samples, so the right count here is CLUSTERS, and one-per-cluster is correct.
* **Training signal** — how much gradient the arm sees. Here a cluster's other members are ordinary
  training data. They cannot leak, because the split boundary is drawn at the cluster.

Measured cost of the conflation, at the 8,192 nt bound: REDOX_COFACTOR has 10,207 records behind
820 clusters (12.4 per cluster), and the build trained on **820** of them. Because §4.4.3 then binds
every class to the smallest, TERPENE dropped 22,870 clusters to 820 (**96.4% discarded**) and RIPP
21,036 to 820 (**96.1%**). Every arm trained on **656** records where 9,400–91,600 were available
behind the same leakage wall — and the prior codebase's comparable adapter, which did produce
measurable output, trained on **7,250**.

Under this amendment each class gets ~10,100 records (~8,080 train), a **12-15x** increase, with the
cluster-level split and its near-duplicate verification unchanged. Cluster counts continue to be
reported, so any claim resting on independence uses them and not the record counts.

⚠ Equal RECORDS is chosen over equal CLUSTERS deliberately. Records-per-cluster varies by class
(REDOX 12.4, TERPENE 5.1), so equal clusters would hand one arm ~2.4x another's gradient volume —
the confound §6.0a exists to prevent. Equal records matches what the trainer actually sees. It
costs cluster-count equality, which is therefore reported per class rather than assumed.

⚠ The surrounding §4.4.3 prose still carried pre-reorientation numbers (`1,224`, `@16k`, "five
classes", ARYLPOLYENE binding). Those were stale before this amendment and are not what the
builder computes; the builder derives `common_n` and always did.

**Open decisions — require sign-off, not measurement.**

| id | decision | recommendation |
|---|---|---|
| D1 | K = 4 classes as in §4.4 | ✅ **RESOLVED — 4.** Threshold 0.55 sits in a natural gap (0.531 → 0.647); any cut in 0.54–0.64 selects the same set. |
| D2 | `bgcfm` coverage | ✅ **RESOLVED — all four classes, coverage reported.** |
| D3 | rank arms, or separate from floor? | ⏸ **DEFERRED to after Stage 1, deliberately.** Stage 2 is sized for beat-the-floor. If Stage 1 shows arms well separated, ranking comes nearly free; if all sit near the floor, ranking is unaffordable at any feasible n and the paper says so explicitly rather than under-powering it silently. |
| D4 | Stage 1 regime | ✅ **RESOLVED — both de novo and seeded.** Their rates differ enough that neither can power the other. |
| D5 | compute envelope | ✅ **RESOLVED — one shared H100; stay under ~50% utilisation for any job exceeding one day.** Evo2-1B and GO-4B both fit comfortably. **The binding resource is antiSMASH CPU throughput, not GPU** — G1 measures sequences/hour and total scoring cost before Stage 2 is sized. |

---


#### A4 — the I1 manipulation check gains a third part (2026-09-11)

§6's `I1` check specified two parts, (a) monotone projection and (b) KL against the unsteered
distribution, and correctly noted that (a) alone is circular. Building the arm showed the pair is
still not sufficient: **both are passed by a random vector.** (a) is satisfied by construction for
any direction — injecting `α·d` raises the projection onto `d` by `α‖d‖²` whatever `d` is — and
(b) is satisfied by any vector pushed hard enough to move the next-token distribution. So a pair
of checks intended to license a null could both pass on noise.

Part (c) closes it: derive the direction on TRAIN, derive a readout independently on VAL, and
require them to agree. A noise direction does not reproduce on held-out records and no amount of
injection makes it. Measured: TERPENE 0.755, REDOX_COFACTOR 0.805, RIPP 0.732, ARYLPOLYENE 0.307
(mean steered-site cosine at 192 records per side).

This ADDS to (a) and (b) rather than replacing them — the three test different things (the vector
is real / the hook fired / the output moved) and none implies another.


#### A5 — G9 is per class and per weight state (2026-09-11)

§6 specified G9 as one sweep of "injection site × magnitude". It was run on TERPENE and the
resulting α = 0.3 applied to all four classes, which cost two of the four `W2` arms: ARYLPOLYENE
fell to 29% of its baseline coding density and 0.015 termination against 0.290, REDOX_COFACTOR to
66% and 0.140 against 0.450. Both are uninformative under §6.4 rather than negative.

The failure was predicted before the arms ran (FINDINGS §15.2 named ARYLPOLYENE on the evidence
that its KL at α = 1 was 1.601 against TERPENE's 1.034) and it happened anyway, because the sweep
was skipped to save ~50 minutes. The cost was two arms and a day.

⚠ **It also confounds the one positive result.** TERPENE is the only class running at an α chosen
for it — the largest magnitude *it* tolerates. Every other class ran at a borrowed number, so
"steering worked for TERPENE and not the others" is not a comparison this design can make: RIPP's
0/200 at α = 0.3 is equally consistent with being under-steered relative to its own ceiling.

⇒ G9 is now **one sweep per (class, weight state)**, and an arm may only be read at its own α.


#### A6 — the benchmark is reported as EXPLORATORY; Stage 2 becomes conditional (2026-09-13)

§6.2 specifies Stage 2 as the powered, pre-registered benchmark, and §6.1 casts Stage 1 as a
shakedown whose rates exist to power it. **By decision, this paper reports Stage 1 as the
result**, framed as exploratory, and Stage 2 is deferred rather than scheduled.

**What that permits.** Descriptive and effect-size claims: *"seeding gave 0.130, steering 0.075,
de novo 0.014 on per-class weights."* Rates, confidence intervals, and contrasts whose evidence
is overwhelming at the realised n (seeding vs de novo is p = 1.1e-21 and needs no ceremony).

**What it forbids, and this is the operative half:**
* ⚠ **No claim that an arm or a substrate is BETTER than another** unless the interval separation
  is stated and the comparison's power is reported alongside it.
* ⚠ **No NEGATIVE claim from an underpowered arm.** FINDINGS §18 already found two: ARYLPOLYENE
  and REDOX_COFACTOR steering are blind below 2.3× and 5.7× lifts, so "steering does not work for
  this class" is not available. §18.1's four readable negatives remain readable.
* ⚠ **No p-value presented as confirmatory.** Cells, contrasts and α were all chosen adaptively
  after reading interim results — legitimate exploration, and exactly what voids a p-value's
  guarantee. Report them as descriptive statistics with the selection process disclosed.

**Requirements this imposes on the write-up**, all cheap and none needing new generation:
1. The exploratory status is stated in the abstract and the results section, not a footnote.
2. Every reported rate carries an interval.
3. Every comparison carries its minimum detectable effect at the realised n.
4. The adaptive choices are disclosed as a paragraph: which arms were chosen after seeing what.

⇒ **Stage 2 is PARKED, not dropped** — §14.7. It becomes necessary the moment the paper wants to
rank arms or substrates, or to assert a null it cannot currently support. Its sizing is already
computed (FINDINGS §21) so the decision can be taken later without re-deriving anything.

### 12.A7 AMENDMENT 2026-09-13 — DECODING PARAMETERS ARE PER SUBSTRATE. `top_k` was not.

**What was wrong.** §7.3 froze `temperature 1.0, top_k 4, top_p 1.0` and applied it to every arm
on **both** substrates. That is a configuration matched across substrates, which §14A forbids, and
it was not caught because the value is correct for Evo2 and the failure is invisible in the output.

**Measured**, 8 held-out TERPENE cores, teacher-forced, base weights, no prefix:

| substrate | vocab | probability mass surviving `top_k=4` | real nucleus at p=0.95 |
|---|---|---|---|
| Evo2-1B | 512, byte-level; 4 bases carry the mass | **0.9999** | 3.3 tokens |
| GenomeOcean-4B | 4,096, BPE at ~4.8 nt/token | **0.1937** | **1,012 tokens** |

On Evo2 the filter is a no-op — the alphabet IS four letters. On GenomeOcean it **discards 81% of
the probability mass at every step** and renormalises over the remaining 19%, sampling from 4
tokens where the model's own distribution spans about a thousand. Severe top-k truncation drives a
model into short, repetitive, early-terminating output, which is what `GO_STAGE1_FROZEN_1c2acf1b1ce67b8f`
shows (median 992–5,700 nt against an 8,192 nt budget, `hit_eos` up to 0.975) and it is the leading
explanation for that bundle's length confound (FINDINGS §24.2).

**The rule.** `temperature`, `top_k`, `top_p`, any repetition penalty, and any minimum-length floor
are **per substrate, selected by measurement**, exactly as rank, depth, injection sites and α
already are. What stays identical across substrates is the corpus, splits, classes, scoring config,
novelty gate, n, the **nucleotide** budget, and the endpoint. `FROZEN` therefore holds decoding
values **keyed by substrate family**, and a single shared scalar for these fields is a defect.

⚠ **Do NOT resolve this by copying the prior implementation's preset** (`temperature 0.9,
top_k off, top_p 1.0, repetition_penalty 1.2`). That preset was itself inherited from GenomeOcean's
upstream demo rather than measured here, and adopting it would import an unmeasured constant while
spending the §10 blind credit for nothing. It is chosen per substrate by measurement (§12.A8 records why no gate now does the choosing).

**Scope of the damage.** Generation only. GO's G2 likelihood health, G6 rank, G6b depth and the I1
site sweep all read held-out loss or KL and never sample, so they stand. Every Evo2 arm stands —
`top_k=4` is a no-op there, which is why this is a GenomeOcean-only defect and not a benchmark-wide
one. `GO_STAGE1_FROZEN_1c2acf1b1ce67b8f` remains a valid record of what was run and an invalid
source of any GenomeOcean rate.

### 12.A8 AMENDMENT 2026-09-13 — the decoding gate ⛔ RETIRED 2026-09-14

This amendment created **G11**, a gate that would SELECT each substrate's decoding
configuration by measuring generated sequence against the structural statistics of real
held-out sequence (distinct 21-mer fraction, prodigal coding density, median ORF length),
never against the endpoint (§2.4).

**It is retired without ever closing, because it could not discriminate.** Holding the
adapter, prompt and code fixed and merely resampling which 50 sequences were scored moved its
own deviation statistic over 0.0108–0.1117 (sd 0.0318), while its winner beat the runner-up by
0.0189 — **0.59 sd**. The entire range across the healthy rungs fit inside the noise band of a
single configuration, so no rung was distinguished from any other.

**Neither substrate needs the selection any more.** Evo2 keeps `top_k=4`, which over a
4-letter alphabet retains 0.9999 of its probability mass — unrestrictive, so nothing is being
chosen. GenomeOcean applies **no truncation at all** (`top_k=0`, the full 4,096-token
vocabulary): "we did not truncate" requires no gate to defend, where "we truncated to 256"
would require one this project does not have.

⚠ **What the gate measured is NOT retired**, and §12.A7 still stands: decoding is per
substrate. The measurements that rule out `top_k=4` on GenomeOcean — probability mass retained
0.1280 against Evo2's 0.9999, and distinct-21mer 0.3522 against 1.0000 — were taken on BASE
weights, teacher-forced, with no adapter, and are unaffected by everything that voided the
selection. See FINDINGS §25.

### 12.A9 AMENDMENT 2026-09-15 — THE §6.4 MANIPULATION CHECK WAS GEOMETRY-DEPENDENT, AND WAS SELECTING ON ITSELF

All four GenomeOcean I1 directions failed their §6.4 check and every I1 arm on that substrate
was refused at generation time. The directions were not the problem. Measured mean train/val
cosine per class, on the all-sites configuration:

| substrate | TERPENE | ARYLPOLYENE | RIPP | REDOX_COFACTOR |
|---|---|---|---|---|
| GenomeOcean-4B | 0.544 | 0.594 | 0.566 | 0.818 |
| Evo2-1B | 0.806 | 0.518 | 0.792 | 0.943 |

Not one GenomeOcean class failed the substantive criterion (mean cosine > 0.3). Four defects
were found, all of which make the check harder on a 24-layer decoder than on a 4-attention-site
hybrid at identical direction quality.

**(1) Two of the three pass criteria are ORDER STATISTICS OVER SITE COUNT.** Check (c) requires
`min cosine > 0` across every steered site; check (a) required monotone projection at every
steered site. Evo2 clears a min-of-3; GenomeOcean had to clear a min-of-24. TERPENE died on one
site out of 24 reading −0.083 against a mean of 0.544.

**(2) Check (a) required what its own sibling documents as impossible.** `manipulation_check`
explains that the sites are in series, that only the first site's response is predictable in
closed form, and that an earlier version "would have read propagation as a defect".
`projection_vs_alpha` then required monotonicity at every site. **RIPP and REDOX_COFACTOR
failed check (a) and nothing else** — RIPP on site 23 alone, REDOX on sites 2/8/12/23, all
downstream of up to 23 upstream injections. Both passed check (c) outright.
→ **Check (a) now gates on the FIRST STEERED SITE.** Per-site monotonicity is reported as
evidence (`n_sites_monotone`), not as a gate.

**(3) The probe magnitude was a constant shared across substrates**, which §14A forbids —
"probe magnitude" is named there as a treatment parameter. `--check-alpha` defaulted to 1.0 on
both, and check (a) swept a hard-coded `[0.0, 0.5, 1.0, 2.0, 4.0]`, a grid retired the same day.
→ **Checks (a) and (b) are now read at the substrate's OWN generation α grid**
(`substrate_config.I1_ALPHA_GRID`), so the check describes the magnitudes the arm actually runs
at, and no cross-substrate constant is involved.

⚠ **A unit error justified the original design and is corrected here.**
`directions.alpha_for_target_kl` argued GenomeOcean was "far outside any usable range" at α=1
(KL 3.44–7.81) against Evo2 at 0.197. Those figures are in different units: Evo2's token is
1 nt, GenomeOcean's is ~4.8 nt. Converted to **nats/NUCLEOTIDE** at α=1 on the trained
adapters, GenomeOcean reads 0.86–1.86 against Evo2's 0.83–1.74 — the same regime. The real
asymmetry is the argmax-flip rate (98–99% vs 64–72%), which is a rate and needs no conversion.
`kl_vs_unsteered` now returns `mean_kl_nats_per_nt` so the comparable quantity is to hand.

**(4) THE CHECK WAS SELECTING ON ITSELF.** `g9_sites` chose the injection site set by reading
the per-site cosines out of the direction artifact's `manipulation_check` — computed on val —
and the check was then re-read on the same records. The arm was selected for passing and the
pass was reported as the evidence. It did not bite Evo2 equally: with 4 sites `all` was
admissible without selection, while with 24 sites a passing subset is always findable, so the
rule degenerated into "search until the check passes".
→ **`val` is now partitioned into two disjoint halves by a fixed seed.** Half A is what
`g9_sites` selects on (`selection_readout`, a separate artifact key); half B is what the §6.4
check is read on. Nothing reads both. The direction is still derived on `train`, so the cosine
remains train-vs-held-out either way — what changed is that the fold deciding the gate was
never optimised over. A direction artifact without `selection_readout` is **refused** by
`g9_sites` rather than falling back, because a silent fallback restores the circularity.

**(5) The site-selection rule selected DEPTH, not class content.** Among admissible sets it took
the highest reach (mean KL against unsteered). Perturbing an early site changes the input to
every site downstream, so reach falls monotonically with depth for mechanical reasons. Measured
on GenomeOcean at α=1, in nats/token, every one of the four classes reads
`early_third` > `middle_third` > `late_third`:

| class | early_third | middle_third | late_third |
|---|---|---|---|
| ARYLPOLYENE | 5.900 | 3.500 | 0.226 |
| REDOX_COFACTOR | 8.176 | 1.719 | 0.146 |
| RIPP | 3.697 | 0.525 | 0.034 |
| TERPENE | 4.022 | 1.604 | 0.063 |

The class signal does not follow that ordering. The SMALLEST relative class-difference norm
sits in the early band for all four classes — 0.051 (ARYLPOLYENE), 0.140 (REDOX_COFACTOR),
0.092 (RIPP), 0.052 (TERPENE) — so the rule was maximising a quantity that peaks exactly where
the class content is thinnest. (The *peak* relative norm is at site 7 for three classes and
site 23 for REDOX_COFACTOR, so "the signal lives in the middle" is NOT a claim these artifacts
support; only the early-band minimum is.)
→ **Reach now GATES and fidelity CHOOSES**: eligible = admissible AND reach ≥ 10% of the best
admissible set's reach; among eligible, the highest mean cosine. The bar is relative so it is
invariant to probe magnitude and carries no unit across substrates — an absolute nats/token
floor would mean different things on a 1 nt token and a 4.8 nt one.

**(6) The chosen site set never reached the gate.** `attach_direction` gates on
`check_all_pass` in the direction artifact, written once for the all-sites configuration before
the site sweep ran; `chosen_sites` lived in a separate JSON nothing read back. TERPENE was
refused at generation time on the strength of a configuration it does not use, while an
admissible 8-site configuration sat in a file beside it.
→ **`g9_sites --finalize`** re-reads all three parts at the chosen sites on val_B and writes
them into the direction artifact. `attach_direction` now **defaults** its site set to the
artifact's `chosen_sites`, and **refuses** an explicit `--sites` that disagrees with the set the
check was read at.

⚠ **Two consequential side-effects of (6), both fixed here.** The §6.3 random-direction control
zeroed only *degenerate* sites, so against a subset-steered arm it would have pushed on every
site — 24 against TERPENE's 8 at the same α, which is not the magnitude-matched control §6.3
requires. And `intervention_active_sites` was reported from the derivation's `active_sites`
rather than the realised tensor, overstating §6.5 coverage by exactly the amount the site sweep
narrowed it.

**Endpoint hygiene (§2.4) holds throughout.** GenomeOcean's α-sweep run directories exist only
at α = 0.0 for all four classes: no endpoint has been read at any nonzero α on that substrate,
so the site set and α cannot be endpoint-contaminated. `endpoint_fields_read` remains `[]`.

⚠ **This changes the adjudication rule, so it changes Evo2 too.** The rule is part of the
protocol, not a treatment parameter, and a protocol that differs between substrates is a §10
discrepancy by construction. Evo2's phase B/B2/C/D results were taken under the superseded
rule and are listed in §14.4.


### 12.A10 AMENDMENT 2026-09-15 — THE NUCLEOTIDE BUDGET WAS ENFORCED IN TOKENS, SO ONLY ONE SUBSTRATE WAS HELD TO IT

`generate.py` capped generation at `int(budget_nt / approx_nt_per_token)` TOKENS. §4 holds the
**nucleotide** budget as part of the shared TASK — one of the few quantities deliberately
identical across substrates, because antiSMASH detection rises with the length of sequence it
is given. Enforcing it in tokens makes it identical only when the ratio is exact.

| substrate | nt/token | token cap | realised max length | generations over budget |
|---|---|---|---|---|
| Evo2-1B | 1.0 (byte-level) | 8,192 | **8,192** | **0 of 2,400** |
| GenomeOcean-4B | ~4.8 (BPE, a corpus AVERAGE) | 1,706 | **11,327** | **189 of 3,600 (5.2%)** |

4.8 nt/token is a mean over the corpus, not a property of any particular sample, so a draw of
longer BPE tokens overshoots. GenomeOcean was receiving up to **38% more sequence** than Evo2
in which to land a cluster, on the one metric the two are compared by.

**Did it reach the endpoint? Measured: NO.** 16 of GenomeOcean's 182 detections sat ON
over-budget generations, which looked alarming and was first reported here as a
"budget-conformant lower bound" of 17/200 for ARYLPOLYENE de novo against a measured 21/200.
⚠ **That framing was wrong and is retracted.** Sitting on a long generation is not the same as
depending on its length. Generation is seeded, so re-running an arm reproduces the same draws
and the only difference is the truncation — which makes the question directly testable:

| arm | max len before → after | over budget | detected before → after | on-target |
|---|---|---|---|---|
| GO W0 de novo | 8,423 → 8,192 | 1 → 0 | 0 → 0 | 0 → 0 |
| GO W1n de novo | 9,018 → 8,192 | 5 → 0 | 4 → 4 | 1 → 1 |
| GO W2 ARYLPOLYENE de novo | 9,270 → 8,192 | 26 → 0 | **21 → 21** | 21 → 21 |
| GO W2 TERPENE de novo | 8,992 → 8,192 | 11 → 0 | 4 → 4 | 4 → 4 |

Every detected cluster lay within the first 8,192 nt, so **no endpoint number moves**. The
defect was real, the fix is still required — an unbounded budget is not defensible whatever it
happens to produce on one corpus — but it did not bias the comparison. GenomeOcean's
ARYLPOLYENE de novo rate is 21/200 against Evo2's 16/200 both before and after.

**The fix.** `_enforce_budget_nt` truncates every generation to `budget_nt` after decoding, on
both substrates and every code path. The first `budget_nt` bases are exactly the bases the
model would have produced had it stopped there, so the prefix is the same measurement. A
truncated generation records **`hit_eos = False`**: its terminator landed outside the budget,
which is the same accounting Evo2 already got when it ran into the cap.

**What is re-run.** GenomeOcean's 12 endpoint arms (phase A0 de novo and phase A seeded) are
regenerated under the enforced budget; the superseded run directories are moved to
`runs_superseded_budget/` rather than deleted, so the before/after stays auditable. **Evo2 is
not re-run** — the fix is a no-op at 1 nt/token and its arms are already conformant.

⚠ **Not re-run, and recorded as a known limitation: GenomeOcean's phase C α selection.**
Those sweeps chose α from coding density, self-NLL and hit-EOS measured on un-truncated
generations. α is a per-arm TREATMENT parameter (§14A), and the phase D arms that use it
generate under the enforced budget, so the ENDPOINT is clean; what is imperfect is the choice
of magnitude, not the measurement of it. The clearest instance is REDOX_COFACTOR, which chose
α = 0.3 on a rung where 13 of 50 generations were over budget — under truncation its hit-EOS
would fall from 0.82 toward 0.56 and the rung would likely have failed the 0.15 termination
tolerance. Its phase D arm is still an honest measurement of "I1 REDOX_COFACTOR at α = 0.3".


---

## The deferred register (§14)

⚠ **A decision to skip is invisible later and looks identical to never having considered it.**
That is what this register exists to prevent.

### 14.1 Experiments

| id | what | why parked | cost |
|---|---|---|---|
| X1 | **Seeded × per-class CONFLICT matrix** — give the TERPENE adapter a RIPP seed, all 4×4 cells | The diagonal (each adapter seeded from its own class) answers "do the channels compose". The off-diagonal answers the sharper question: **when the weights and the seed disagree, which wins?** ⚠ **Its precondition is now met** — the diagonal ran 2026-09-10 and the channels DO compose (0.130 vs 0.014, 9.5×, `SEEDED_DIAGONAL_FROZEN_f1a5fa95dbdc07a1`, FINDINGS §11). Deferred by decision, not by evidence; ARYLPOLYENE at 0.315 and TERPENE at 0.150 are the two cells with enough signal to read a conflict against. | 3,200 generations, ~3–4 h |
| X2 | **Seed length beyond 128 as a claim** | L=256 and L=512 are measured and frozen, but their seed-only baselines are 0.165 and 0.412 — at 512 the baseline exceeds the generation rate. Recorded in FINDINGS 9.5–9.6, not claimed. Would need a contamination-corrected estimator to be usable. | data already exists |
| X3 | **Does genus/species in the lineage help?** | `LINEAGE_WIDTH` 186 keeps the whole lineage, so this is now testable by truncating deliberately — order-level vs species-level conditioning. Never run. | ~1 h per width |
| X4 | **Unprefixed RIPP control** | Skipped in the §10 control because RIPP is 0/200 prefixed, so an unprefixed zero is uninformative. Becomes worth running only if RIPP ever comes off the floor. | ~25 min |
| X5 | **Training-trajectory noise-floor replicate** | Queued then killed as a diagnostic. FINDINGS 4a.7 bounds the floor at ≤5.1e-4 from cross-commit comparisons; a same-commit replicate would measure it exactly. Matters for G6, which selects on held-out loss. | ~25 min |
| X6 | **Block-wise early exit for Evo2** | Generation runs to budget even after the terminator fires. A per-row exit would recover the wasted compute; batched decoding is bounded by its slowest row, so the win is smaller than it looks. | engineering, no science |
| X7 | **RIPP within-class strata** | `--strata` exists and is wired (SPEC 4.5.2) but has never been run. Would test whether RIPP's flat-zero is an averaging artefact over very different subclasses. | ~1 h |

### 14.2 Gates the spec defines that have no data

| gate | what it sets | status |
|---|---|---|
| G4 | decoding policy — temperature, top-k, top-p | `NEVER RUN`. Current values are inherited defaults, not swept. |
| G6 | **adapter rank sweep** | ✅ **RUN 2026-09-10 — RESOLVED, rank 16 retained.** Ranks {4,8,16,32,64} on the pooled `W1n` arm, held-out loss only. **A 16× parameter increase buys 0.00078 nats/nt — 0.48× the within-run evaluation noise (0.00164), so rank is indistinguishable from noise.** Rank 16 sits 0.00008 off the minimum, so the debt every W-arm owed is discharged. ⚠ This says nothing about the ENDPOINT — an earlier version of this row claimed capacity is not why the endpoint rates are low, which a loss-only sweep cannot support; retracted, see FINDINGS §13.2. Bounded to this protocol: every rank stopped at step 250, which is 12.4% of ONE epoch. See FINDINGS §13. |
| G6b | **adapter depth sweep** | ✅ **RUN 2026-09-11 — RESOLVED, all-blocks retained.** Rank 16 on `W1n`, six depth sets. Moving the same rank-16 adapter from all blocks to blocks 17-24 costs 0.360 nats/nt (228× the within-run noise), against 0.00078 (0.5×) for G6's 16× capacity increase. ⚠ An earlier version reported that contrast as "~460× more"; it is a ratio of two arbitrarily-scoped manipulations over a noise-valued denominator and is withdrawn as a statistic. Every set containing a block from 0-7 lands in [0.8956, 0.9011]; `middle` (8-16) and `late` (17-24) sit at 1.121 and 1.255 — ⚠ an ASSOCIATION, not an identified threshold: six points cannot separate it from a monotone count-of-early-blocks account. Capacity-matched: `early`/`middle`/`late` hold 3.32M/3.72M/3.44M parameters and span 0.359 nats/nt. `early` alone ties `all` within the noise floor at 0.32× the parameters. All-blocks is retained for every reported arm (it is the tied minimum and is what §8/§10/§11 already ran). Greedy: rank fixed from G6. See FINDINGS §14. |
| G8 | data scaling | `NEVER RUN` as a sweep. §12.A3 raised training data 12.3× and the endpoint did not move, which is one point on the curve, not the curve. |
| G9 | steering layer × magnitude | ✅ **RUN 2026-09-11/12 — RESOLVED, per class.** Swept on generation health, never the endpoint. `W2` ceilings: TERPENE 0.3, RIPP 0.3, ARYLPOLYENE 0.1, REDOX_COFACTOR 0.1 — **α does not transfer across classes** (§12.A5). The prior codebase's steering nulls remain unreportable (Pfam endpoints, no frozen config), which is why this had to be measured here rather than cited. See FINDINGS §15, §17. |
| G2 | substrate likelihood health | `PARTIAL` — Evo2 done; GO-4B and bgcfm load and generate but have no likelihood check. |


### 14.3 Arms and substrates

* **GO-4B / bgcfm training** — `PARKED to Stage 2` by decision. The comparison is scoped to
  unprefixed arms while the Evo2 side uses its native taxonomy prefix, which GO cannot receive
  (§4.3, which every prefixed arm already departs from in practice — see the ⚠ below).
  GenomeOcean's trainable class token is the genuine structural difference
  and the reason it is worth doing at all.
* **`I1` activation steering** — ✅ **BUILT 2026-09-10**: `model/directions.py` (difference-of-
  means derivation, record-weighted, unit-normalised), `run/derive_directions.py` (with the §6.4
  manipulation check against an independently derived validation readout), and the
  `--direction/--alpha/--random-direction` runner path. ✅ **G9 HAS RUN** — 2026-09-11/12, per class and per weight state (§12.A5): `W2` ceilings
  TERPENE 0.3, RIPP 0.3, ARYLPOLYENE 0.1, REDOX_COFACTOR 0.1. Directions are derived for all four
  classes on the base model AND on the four `W2` per-class adapters, all passing the **three-part**
  check of §12.A4 — the α-monotonicity and KL halves that were missing are built and pass. Arms
  run and frozen (`I1_STEERING_FROZEN_2b203f381cefe712`, `I1_PERCLASS_FROZEN_d974b3ee9b83274c`).
  G9
  and I1 are how a steering null gets into the paper on the frozen instrument (§14.6). The prior
  null is a reason to expect the outcome, never a substitute for measuring it here.
* **`I2` iterative refine** — `DROPPED` from the grid by decision, not deferred.

### 14.4 Results taken on a superseded pipeline

Both were measured before the terminator fix, i.e. on ~8,190 nt of post-termination sampling
(FINDINGS 4a.8). They are flagged in place and must not be cited until re-measured:

* **FINDINGS 4a.7** — training non-determinism moving the endpoint by ±1/200.
* **The §12.A3 data-volume null** — 12.3× more data with no endpoint movement.

⚠ **Added 2026-09-15 (§12.A9): every Evo2 I1 result on the `_tax_fx` weights.** Phases B, B2,
C and D were run under the superseded §6.4 adjudication — check (a) gating on all steered
sites, a probe α of 1.0 with the retired `[0, 0.5, 1, 2, 4]` grid, site selection reading the
same val fold the check was read on, and site choice by maximum reach. The *directions* are
unaffected (they are derived on `train` and nothing about the derivation changed), but the
chosen **site sets** and therefore the chosen **α** per class were produced by the old rule.
Replayed against the corrected selection rule on the recorded rows, three of four Evo2 classes
would choose a different site set: TERPENE `pair_01` → `single_0`, ARYLPOLYENE `all` →
`single_1`, REDOX_COFACTOR `single_0` → `all`; RIPP is unchanged at `single_0`. The α sweeps
in `/data2/ds85/bgcbench/g9/evo2-1b_*_W2_tax_fx_*.json` are therefore α values for site sets
that are no longer the chosen ones, and must not be cited until phases B2–D are re-run.

⚠ **QUARANTINED 2026-09-15 — two defects introduced BY THE FIX ITSELF, caught before they
reached a reported number.** Both are recorded here because a quarantined run directory that
is not explained looks identical to one that was simply lost.

1. **`runs_superseded_budget/`** — GenomeOcean's 12 endpoint arms measured before §12.A10's
   nucleotide-budget enforcement. Superseded, not wrong-by-accident; kept for the before/after.
2. **`superseded_reuse_bug/`** — Evo2's four phase C α artifacts and two phase D run
   directories. `g9_alpha` gained a "reuse an existing measurement instead of dying on the
   overwrite guard" path, whose glob `runs/stage1_<substrate>_<arm>_*` ends in a wildcard that
   swallows the realised-config HASH. It therefore matched on the arm NAME alone, and reused
   measurements taken at the site sets §12.A9 replaced — Evo2 phase C "completed" in 2.5
   minutes, reporting an α for RIPP read off a directory recording
   `intervention_site_subset=[0]` when the corrected protocol had chosen `all`=[0,1,2,3].
   Phase D then ran on those α. Reuse now compares injection sites, magnitude and weight
   state before accepting a directory (`_same_config`).

⚠ **ALSO 2026-09-15: phase D lost 10 of 12 arms to CUDA OOM.** `phaseD_steer.sh` checked free
GPU once at phase entry and then ran 12 arms; the shared card's other user returned to 38.9 GB
mid-phase. `g9_alpha` had already learned this and guards per α rung — phase D now guards per
arm, for the same reason. The lesson is recorded in `_lib.sh`: on a shared card a
once-per-phase check is not a guard, it is a snapshot.

✅ **RESOLVED 2026-09-16: the Evo2 re-run was completed.** Phases B, B2, C and D were re-run
under §12.A9, so BOTH substrates are now adjudicated by the same §6.4 protocol and the
cross-substrate steering comparison IS reportable. The paragraph below records the state while
it was deferred, and is kept because the deferral was real for ~8 hours.

⚠ **DEFERRED 2026-09-15 (user decision), SINCE LIFTED: the Evo2 re-run was not queued yet.** GenomeOcean's
steering chain is re-run under §12.A9 first; Evo2 keeps its superseded phase B/B2/C/D results
for now. **Cost of the deferral:** until Evo2 is re-run, the two substrates are adjudicated by
different §6.4 rules, and *no cross-substrate steering comparison is reportable* — that is a §10
discrepancy by construction, not a finding. The GenomeOcean result stands on its own (its gate
is internally consistent); the Evo2-vs-GenomeOcean steering contrast does not. Re-running Evo2
phases B→D closes it, at roughly 4–5 h on the shared card.

### 14.5 Methodology parked

* **`min_delta` below the measured noise floor.** §6.0a uses 1e-4 while trajectory noise
  reaches 5.1e-4, so early stopping and best-checkpoint selection are partly selecting noise.
  An amendment setting it from the measurement was drafted and **not filed** — it needs X5
  first, and the large deltas the project actually reads are 30–70× the floor.
* **Lift is undefined against a zero floor.** §3.5 divides by the unconditioned marginal, which
  is 0.000. Reports correctly emit `null` rather than dividing; the substitute in use is an
  absolute rate difference with an exact binomial interval.

### 14.7 Stage 2 — the powered, pre-registered benchmark `PARKED 2026-09-13`

The paper reports Stage 1 as exploratory (§12.A6), so Stage 2 is parked rather than scheduled.
**It is not dropped, and here is precisely when it becomes necessary:**

| trigger | why Stage 2 is required |
|---|---|
| ranking arms or substrates ("seeding beats steering", "Evo2 beats GO") | a ranked claim needs the contrast powered and pre-specified |
| asserting any null the current n cannot support | §18.2's two underpowered arms; a null needs power AND a manipulation check (§6.4) |
| presenting any p-value as confirmatory | every cell, contrast and α here was chosen adaptively after reading interim results |
| a reviewer asking for multiplicity control | no correction family was pre-specified across dozens of comparisons |

**Cost, already computed** (FINDINGS §21, so the decision needs no new derivation): n = 400 per
arm covers seeding, steering, the real-vs-random control and ARYLPOLYENE's gap — 32 arms, ~12 h
generation, 0.7 h scoring. Two contrasts are NOT affordable at any realistic n and would have to
be declared bounded rather than tested: pooled-vs-per-class de novo (~1,364/arm) and the
steering×seeding composition null (~2,459/arm).

⚠ **The pre-registration must be written and hashed BEFORE the first Stage 2 sequence is
generated** (§6.2). Running the arms first and writing the plan after produces Stage 1 again at
higher n, not Stage 2.

### 14.6 ⚠ Old results are NOT publishable — what must be re-measured

**A result we believe and a result we can report are different things.** Everything below was
established in the prior codebase and is *probably* true, but none of it was measured on this
benchmark's instrument. A paper whose premise is one frozen endpoint cannot cite numbers taken
on a different one, and "we already know the answer" is not a defence a reviewer accepts.

What disqualifies the prior data, in every case at least one of:
* a **Pfam / obligate-domain endpoint** rather than antiSMASH (§3.1 forbids it as an endpoint);
* **no frozen scoring config** — the antiSMASH invocation varied by phase and class;
* **per-class scoring windows** (2,000 nt TERPENE, 4,000 nt PKS) that are not cross-comparable;
* a **class token** in the input, which §4.3 forbids.

⚠ **§4.3 HAS NOT ACTUALLY BEEN AMENDED, and text here previously cited "§4.3 as amended" twice
as though it had.** §4.3 as written requires bare nucleotide input and forbids a taxonomy tag —
so **every prefixed arm in §8, §10, §11 and §11.5 departs from it.** The departure was a
deliberate decision (the GTDB lineage is Evo2's own pretraining format and names an organism, not
a compound class) and filing the amendment was explicitly deferred, not done. Until it is filed,
the honest statement is that the spec and the run record disagree, and the paper must describe
the input format from the runs rather than from §4.3. The class-token prohibition is untouched
and still holds.

| what | prior evidence | why it cannot be reported | to redo |
|---|---|---|---|
| **Steering is null** | six stages, all variants, closed 2026-08-10 | Pfam endpoint + probe readouts, no frozen config | ✅ **DONE** — G9 + I1 run on the frozen instrument. The prior null does NOT reproduce: TERPENE 15/200 vs 3/200, p=6.3e-03 (FINDINGS §17). Steering is not null here. |
| **CFG, soft prefixes, affine editing, cross-class transplants** | all closed | same, and no per-variant rates were ever recorded | each needs a benchmark arm, or the paper says "not tested here" |
| **Guided decoding is not the fix** | likelihood argument: adapters ARE the best model of their own target | a likelihood contrast, never an endpoint measurement | a decoding arm on the frozen endpoint |
| **Detection falls with target length** | r = −0.822 over 11 adapters | mixed instruments across the eleven; the correlation's own denominator problem is FINDINGS-documented | re-derive on frozen scoring |
| **GenomeOcean beats Evo2 de novo 9.8×** | 2,000 nt window, prior config | window and config both outside the benchmark | GO-4B on the rebuilt benchmark |

**`DROPPED` 2026-09-10 by decision — subclass conditioning.** The prior codebase's strongest
result (cyclactone 124/124) is NOT being redone. It is out of scope for this paper, which
benchmarks methods at the benchmark's own class level; a subclass arm changes the class
granularity rather than the method, so it belongs to a different comparison. ⚠ This is the one
row of §14.6 where the paper will say nothing at all, and FINDINGS §12.5's prediction therefore
stays untested — recorded as a hypothesis this benchmark generated, not as a result.

⇒ **The register's default is therefore REDO, not cite.** An item may only be carried into the
paper from prior work if it is a *methodological* fact that does not depend on the endpoint —
e.g. that Evo2's context is 8,192 tokens, or that vortex's `stop_at_eos` is dead code. Rates,
lifts and nulls all need re-measuring.

⚠ This inverts an earlier judgement recorded here: G9 was listed as a candidate for **DROPPED**
on the grounds that the prior programme had closed steering. That reasoning confused *knowing*
with *being able to show*, and would have left the paper unable to report the steering result at
all.

---

---

## §14A.2 — deferred arms

⚠ This subsection sat inside the `## 12` block in SPEC.md, orphaned from its `## 14A` parent,
which is one reason the amendment log had grown to 610 lines inside a section about gates.

### 14A.2 DEFERRED 2026-09-14 — `W1` and `W3` are not trained in the `_fx` generation

Both substrates train `W0` (base, no adapter), `W1n` (pooled, **nucleotide**-balanced) and
`W2`×4 (per class). Two arms are deferred for time, and this records them so a later reader
cannot mistake their absence for an oversight (§14's whole purpose):

| arm | what it was | why deferred | cost to add later |
|---|---|---|---|
| `W1` | pooled adapter, **record**-balanced | `W1n` is the pooled arm the benchmark reports, and the two differ only in how the four classes are weighted within one pooled corpus. `W1` was the weaker of the pair for the question being asked. | one training run per substrate (~4.3 h on Evo2), then one generation arm |
| `W3` | learned per-attention-site offset conditioner | It is a *different intervention*, not a weight state on the same ladder, and nothing else depends on it. FINDINGS §24 recorded it at 0/200 on GenomeOcean under the pre-fix defects, so it has no positive result to preserve. | one training run per substrate, then one generation arm. It reads the same corpus and produces an independent checkpoint, so nothing else needs redoing. |

⚠ **What this costs the write-up.** Without `W1`, the record-vs-nucleotide balancing contrast
is unavailable and must not be claimed. Without `W3`, the benchmark has **no trained
inference-time-style conditioner arm**, so the only intervention axis reported is `I1`
(derived-direction steering). Neither absence is a null; both are simply unmeasured.

