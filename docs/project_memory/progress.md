# Progress — current state of the research

**Read this first when resuming work.** It records the exact state at the last
checkpoint of activity. Update it at the end of a session or after a major change.
See [decisions.md](decisions.md) (the why) and [bugs.md](bugs.md) (quirks/fixes).

_Last updated: 2026-08-12 (PHASE 2 opened on the 1B; frame-aware + domain-weighted arms running)._

---

## Structural change 2026-07-27 — repo is now TWO MODEL TRACKS

The repo was split into `evo2/` and `genomeocean/`, with everything model-agnostic
(dataset pipeline `scripts/`, eval suite `src/bgc_pipeline/`, `config/`, `tests/`) left at
the root so both tracks are scored on the same instrument. Paths that used to be
`scripts/finetune_evo2_lora.py` are now `evo2/scripts/finetune_evo2_lora.py`; shell
wrappers are still invoked from the repo root. `tests/run_all.py` passes (8 files).
See [decisions.md](decisions.md) 2026-07-27 and `docs/model_comparison_evo2_vs_genomeocean.md`.

**GenomeOcean-4B is under evaluation as a replacement substrate.** Headline measurements
(all on gputee against `splits_core`, not quoted from their paper):

- BPE compression **5.15 bp/token** ⇒ 10,240-token context = **52.7 kb** (vs Evo2 32.8 kb);
  `max_position_embeddings` is 32,768 tokens ≈ **169 kb**.
- **Class tokens work.** 22 `[CLS_*]` special tokens added (vocab 4096→4118),
  `resize_token_embeddings` covers both `embed_tokens` and `lm_head`
  (`tie_word_embeddings=false`), and `[CLS_NRPS]` tokenizes to a **single atomic id**.
  This is the exact capability Evo2's byte-level `CharLevelTokenizer` cannot provide and
  where our conditioning died — it costs ~0 GB extra.
- **Training memory:** LoRA r=16 + trainable embed/lm_head, grad-checkpointing, bf16:
  L=10,240 tok → **14.0 GB**; L=32,768 tok (169 kb) → 27.0 GB; **bs=8 at L=10,240 → 54.8 GB**.
  Evo2 at L=32,768 only fits `bs=1` ⇒ ~**12.8× more nucleotides per micro-step**.
- **Context is NOT the differentiator on strict cores** (GO 0.966 vs Evo2 0.892 mega fit) —
  do not oversell it. It IS decisive for whole antiSMASH regions: median mega region is
  47.2 kb, so Evo2 fits **0%** whole while GenomeOcean fits **64%**.
- **`bgcFM` is unconditional** — no class handle at all. Their T1PKS result is
  generate-258,260-then-filter (4.3% antiSMASH-positive, their number).

**Zero-shot replication run (2026-07-27), `/data2/ds85/bgcmodel_runs/go_zeroshot_bgcfm/`.**
24 sequences, `creative_long` preset (min 9,600 tok, temp 0.9, rep-pen 1.2), prompt =
`[CLS]` only. 1.156 Mbp, median 52.5 kb, 3,602 s on the HF dynamic-cache backend
(320 bp/s — use `--cache-implementation static`, added after this run; measured 3.4× faster).
Scored on OUR antiSMASH gate (`scripts/eval_suite_driver.py`):

| | GenomeOcean `bgcFM` zero-shot | Evo2 v2_step1200 (conditioned) | base Evo2 |
|---|---|---|---|
| `is_bgc` | **3/24 = 0.125** | 3/21 ≈ 0.14 | 0.00 |
| `coding_density` | **0.900** (0.712–0.976) | 0.893 | 0.606 |
| `coding_sanity` | 24/24 = 1.00 | — | — |

Products: 1 NRPS, 2 RRE-containing (→ 1 NRPS + 2 RIPP in our vocabulary). `correct_class`
correctly ungraded (unconditional ⇒ no expected class).

**Honest reading — three caveats, all pointing the same way:**
1. **n=24 cannot estimate a 4.3% rate.** 3/24 has a Wilson 95% CI of roughly [4%, 31%], so
   this is *consistent with* their 4.3%, not a confirmation of it. A real estimate needs
   n≥200 (~8 h on the HF backend at the pre-fix throughput).
2. **Per-sequence is not per-nucleotide.** These sequences are 52 kb vs Evo2's 32 kb, so the
   per-sequence rates flatter GenomeOcean. Per Mbp it is 2.6 hits/Mbp vs Evo2's ~4.4/Mbp —
   i.e. Evo2 looks *better* on that normalization. With 3 hits on each side neither
   difference is significant on any normalization; treat both as "same order".
3. So the defensible claim is narrow but real: **an off-the-shelf model with zero
   project-specific training lands in the same range as our 50-hour fine-tuned Evo2 on
   `is_bgc` and coding density.** That is a statement about the starting point, not about
   which model wins.

1/24 outputs was low-complexity (83-bp homopolymer run, compressed to 26 kb). Upstream
filters these via `find_tandem_repeats_percentage`; we ran with filtering off
(`--max-repeats 100`) on purpose so the raw rate is visible.

**Powered rate estimate, n=216 (2026-07-27), `/data2/ds85/bgcmodel_runs/go_zeroshot_rate_n216/`.**
Same preset, seed 20260727, `--cache-implementation static`. 10.35 Mbp in 9,561 s
(**1,083 bp/s** — the static-cache fix gave **3.4×** over the dynamic-cache n=24 run's
321 bp/s, NOT the ~9× first predicted). Median 51.2 kb, coding_density 0.908.

- **`is_bgc` = 27/216 = 0.125, Wilson 95% CI [0.087, 0.176] → 2.61 hits/Mbp.**
  **Their reported 4.3% is OUTSIDE this CI — we do not reproduce their number.** The
  method replicates; the rate does not. Most likely we measured a different thing:
  (1) **length** — all our sequences are ~51 kb while their sweep included
  `min_seq_len=1024` tok (~5 kb), and `is_bgc` is per-sequence, not per-Mbp (see the
  eval defect below); (2) antiSMASH **8.0.4 vs their 7.0**; (3) our gate was deliberately
  recalibrated (~0.15→0.97) and may be more permissive; (4) they averaged over
  rep-pen [1.0-1.5] x temp [0.7-1.1], we sat at a single favourable point.
- **Product mix is the encouraging result.** 27 hits → NRPS 11, RRE-containing 5,
  NRPS-like 4, **T1PKS 4**, transAT-PKS-like 2, + 4 singletons. Mapped to our vocabulary:
  **NRPS 15 / RIPP 8 / PKS 5 / TERPENE 1 — i.e. 20/27 = 74% megasynthase.**
  Contrast the Evo2 step_1200 result: *conditioned* on NRPS/HYBRID it produced only
  SIMPLE classes (ectoine, terpene), `correct_class` 0/21, `module_count` 0/21. An
  UNCONDITIONED GenomeOcean makes megasynthases as its dominant output; a CONDITIONED
  Evo2 never made one. Deflators: SMC (bgcFM's training set) is NRPS/PKS-heavy, and
  51 kb leaves room for an assembly line that 32 kb does not.

### CLASS LINEAR PROBE on GenomeOcean-bgcFM (2026-07-27) - SEPARABLE, survives the taxon control

The fork gate from next-action 3b. Scripts: `genomeocean/scripts/class_probe_go.py` + shared
stats in `src/bgc_pipeline/linear_probe.py`; results in `genomeocean/experiments/`.
Protocol: RAW nucleotides (no class tag), mean-pooled hidden states, 5-fold CV logistic probe,
balanced accuracy, shuffled-label control on the same folds, `--max-nt 4096`.

| layer | CLASS all (chance .091) | PHYLUM (chance .111) | CLASS within Pseudomonadota (chance .111) |
|---|---|---|---|
| 0 | 0.345 | **0.657** | 0.415 |
| 4 | 0.770 | **0.899** peak | 0.818 |
| 8 | 0.892 | 0.892 | **0.907** peak |
| 12 | **0.894** peak | 0.889 | 0.904 |
| 16 | 0.878 | 0.886 | 0.885 |
| 20 | 0.830 | 0.891 | 0.869 |
| 24 | 0.792 | 0.879 | 0.832 |

**Verdict: compound class IS linearly decodable from bgcFM hidden states (~10x chance), and it
is NOT the taxon confound.** Two independent reasons:

1. **Curve shape.** Phylum is 0.657 at layer 0 - before the transformer does anything - and
   saturates by layer 4, flat thereafter. That is a *compositional* property (GC / k-mer bias
   ~ taxonomy; cf. TNF binning). Class is only 0.345 at layer 0 and needs 12 layers to reach
   0.894. Class is COMPUTED; taxonomy is READ OFF. Similar peaks, different signals.
2. **Taxon-stratified control (decisive), replicated across phyla.** Holding taxonomy roughly
   constant does not remove the class signal:
   - within **Pseudomonadota**: n=931, 9 classes (incl. NRPS/PKS/HYBRID), chance .111 -> **0.907**
   - within **Bacillota**: n=541, 6 classes, chance .167 -> **0.948**
   - within **Actinomycetota**: n=552, 5 classes, chance .200 -> **0.954**
   All three are at or ABOVE the unrestricted 0.894 (fewer classes => higher chance
   baseline). Three independent phyla, no collapse: the confound is ruled out.

Per-class recall (within Pseudomonadota, L8): BETALACTONE .982, RIPP .975, ARYLPOLYENE .962,
HSERLACTONE .958, TERPENE .958, ECTOINE .911, PKS .875, NRPS .833, PKS_NRPS_HYBRID .705.
Megasynthases are consistently hardest to decode - but still ~7x chance.

**IMPLICATION - this reverses the strategic default.** On GenomeOcean the class-conditioning
failure is a DECODING/STEERING problem, not a representation problem. The information is
demonstrably present. That puts the cheap levers back on the table - guided decoding scored
against this very probe, steering vectors / activation addition at layer ~8-12, soft prompts,
CFG with a trained null - and it means long-context compute (Quartz) is NOT justified by
"the model cannot represent class", at least not for GenomeOcean.

**BASE-4B CONTROL (done) — the class representation comes from PRETRAINING, not the BGC fine-tune.**

| layer | bgcFM | base-4B | delta |
|---|---|---|---|
| 0 | 0.345 | 0.346 | -0.001 |
| 4 | 0.770 | 0.771 | -0.001 |
| 8 | 0.892 | 0.878 | +0.014 |
| 12 | **0.894** | 0.877 | +0.017 |
| 16 | 0.878 | 0.842 | +0.036 |
| 20 | 0.830 | 0.805 | +0.025 |
| 24 | 0.792 | 0.770 | +0.022 |

base-4B peaks at **0.878 (L8)** vs bgcFM **0.894 (L12)** — the 1.72M-BGC fine-tune buys only
**+0.016** at peak. The two are literally indistinguishable through layer 4. So 645 Gbp of
METAGENOMIC PRETRAINING already taught compound-class discrimination; the BGC fine-tune mostly
taught BGC-shaped *generation*, plus a small edge in holding the class signal deeper into the
stack (largest at L16, +0.036, then narrowing again — it does NOT widen monotonically).

Practical consequences: (a) the steerable representation exists in BOTH checkpoints, so pick
between them on generation quality, not representation; (b) intervene at **layers 8-16**, where
separability peaks in both.

**EVO2 PROBE (other session, `/data2/ds85/bgcmodel_runs/class_probe/`) — EVO2 ALSO SEPARATES,
SLIGHTLY BETTER THAN GENOMEOCEAN. This reverses the switch recommendation.**

| model | best layer | balanced acc | chance |
|---|---|---|---|
| **Evo2 base** | 16 | **0.911** | 0.091 |
| Evo2 v2 LoRA step_1200 | 16 | 0.906 | 0.091 |
| GenomeOcean bgcFM | 12 | 0.894 | 0.091 |
| GenomeOcean base-4B | 8 | 0.878 | 0.091 |

Evo2 layer sweep (base): L0 .486, L4 .838, L8 .838, L12 .904, **L16 .911**, L20 .901, L24 .859,
L28 .605, L31 .414 (sharp collapse in the last blocks). n=991, 11 classes, shuffled ~.09.

Three consequences:

1. **Evo2 DOES represent compound class (~10x chance).** The Evo2 conditioning failure was
   therefore ALSO a decoding/steering problem, not a representation problem. The premise
   behind reserving Quartz long-context for "must INSTALL a representation" is now falsified
   for BOTH models — that spend is not justified by representation absence on either.
2. **The LoRA did not add class separability**: v2 0.906 vs base 0.911 (slightly lower, within
   noise). Perfectly consistent with the standing finding that the adapter added coherence
   (coding_density 0.61->0.89) but nothing about class.
3. **The strongest argument for switching substrate is GONE.** Class representation was never
   the differentiator; both models have it. The switch case now rests only on the other axes
   (trainable class token, 12.8x nt/micro-step, 51 kb context, unconditional megasynthase
   output, vLLM-ability) — real, but no longer decisive.

**COMPARABILITY CAVEAT — do not over-read the 0.911 vs 0.894 ranking.** The two probes are not
perfectly matched: `--max-nt 4096` holds the *biological* input constant, but Evo2's byte-level
tokenizer turns that into 4,096 pooled positions while GenomeOcean's BPE gives only ~795. Evo2
also has 32 blocks vs 24 layers. Evo2's higher layer-0 floor (.486 vs .345) is direct evidence
of the richer pooling. The 0.017 gap is within what that confound could produce; treat the two
as "both strongly separable", not "Evo2 wins".

**⚠️ SUPERSEDED 2026-08-10 — steering was run to completion (Phases 0/1/3/5/6, direction audit,
activation patching) and is CLOSED; see NEXT ACTIONS. The comparability caveat above stands.**

**REVISED NEXT ACTION (historical):** the cheapest decisive experiment is now **steering on Evo2 step_1200**,
which is already trained and already evaluated — guided decoding scored by this probe, or
steering vectors / activation addition at block ~16 where separability peaks. If steering moves
`correct_class` off the floor on Evo2, no migration is needed at all. GenomeOcean remains the
better substrate for a *conditioned retrain* (class token + throughput) if steering alone fails.

> **EVAL DEFECT (task #5, must fix before publishing any cross-model number):** `is_bgc`
> is a per-sequence binary, so a model emitting longer sequences scores higher for free.
> GenomeOcean 12.5%/seq vs Evo2 14.3%/seq, but 2.61 vs ~4.4 hits/Mbp — the ranking
> *flips* under length normalisation. Either fix generation to a common bp target or
> report hits/Mbp. Touches `src/bgc_pipeline/evaluation.py` +
> `scripts/eval_suite_driver.py::summarize_group`. This same defect is the leading
> explanation for the 12.5%-vs-4.3% gap above.

## ★★★ PHASE 6 (2026-08-10): the direction DELETES class but cannot INSTALL it — found only by
## the new continuous readout, which every binary gate reported as a flat zero.

**Multi-layer steering** (`run_steer_stack.sh`): the class direction re-asserted at 9 layers
(L10-L27, each with its own direction and class-unit) simultaneously, 3 per-layer doses, each
with a shuffled-label twin, seeded cross-class design, n=12/arm. Rationale: falling single-layer
reach with depth is what downstream ERASURE of an added component looks like, and re-asserting at
every layer is closer to *clamping* than nudging — a mechanism no single-layer run can test.

Binary gates: **0/12 target markers in every arm**, real and shuffled alike. Uninformative — the
marker gate's TPR is 0.717 at this length.

The CONTINUOUS readout (`class_probe`, TPR 0.900) on the same sequences, paired real vs
shuffled-label on the same exemplars:

| per-layer dose | ΔP(target) | p | ΔP(seed) — **leaky probe** | ΔP(seed) — **CLEAN (train-only)** | p |
|---|---|---|---|---|---|
| 0.027 | +0.015 | 1.000 | −0.175 | **+0.056** (sign flips) | 0.549 |
| 0.082 | −0.082 | 0.774 | −0.090 | −0.138 | 0.774 |
| 0.16 | +0.065 | 0.388 | **−0.308, p=0.0063** | **−0.177 (3/12 up)** | **0.146** |

### ⚠️ CORRECTED 2026-08-10 — the seed-deletion result did NOT survive the leakage fix

The first pass reported ΔP(seed) = **−0.308, p = 0.0063** and concluded "ABLATION WORKS,
INJECTION DOES NOT" as a *generation-level* finding. That probe was fit on **val+test** and then
applied to generations **seeded from val+test cores** — it had seen the seeds. Refit train-only
(22 classes, balanced acc 0.933, chance 0.045), the effect **halves to −0.177 and loses
significance (p = 0.146)**, and at the lowest dose it changes sign. The damage-control analysis
built on it (corr +0.002, −0.393 on damage-matched pairs) is void with it.

**What still stands:**
- **ΔP(target) is null at every dose under both probes** — steering does not install the target
  class. This was never the contested part and the leakage fix does not touch it.
- **Phase 1's teacher-forced ablation asymmetry (z = 4.8, p = 0.040)** is independent of the probe
  entirely — it is raw model log-likelihood — and stands.

**What does not:** the claim that the ablation asymmetry was confirmed *in generation*. The clean
numbers are a consistent negative trend at the two higher doses (−0.138, −0.177) that does not
reach significance at n=12. Suggestive, not a result. Do not cite −0.308 or p=0.0063.

**Why this matters beyond one number:** the leaky probe manufactured a significant finding from a
non-significant one, in the direction the analyst expected. It is the strongest argument in this
project's history for clearing instrument debt BEFORE reading results off the instrument.

⇒ Multi-layer is now tested and closed too, on a SENSITIVE instrument. The inference-time family
is exhausted. Next spend remains **training-time coupling** (per-class adapters, or a
class-prediction loss forcing the last blocks to read the class coordinate).

**New eval check `class_probe`** (continuous, diagnostic-only, never gates) is what made this
visible. Calibrated at both ends: TPR 0.900 on real cores at 3 kb (vs 0.717 for the Pfam marker
gate), but mean argmax confidence **0.900 on real NON-BGC DNA** vs 0.986 on real cores — it has
no negative class and cannot abstain, so it measures RESEMBLANCE, not validity, and is
trustworthy ONLY in paired comparisons. RIPP is its default attractor for unremarkable DNA
(14/25 negatives). See `evo2/scripts/calibrate_class_probe.py`.

## ★★★ PHASE 5 (2026-08-10): dilution is NOT the constraint. Inference-time steering is CLOSED.

The last standing excuse for Phase 3's null was **dilution** — the edit is made 16 blocks from the
output and the residual stream grows 11 orders of magnitude in the final blocks (L27 11.25 →
L30 3.69e12), so perhaps the nudge never survives. Tested at **layer 27**: the last layer where
the class direction is still real (held-out AUC 0.835) and the last before the blow-up.

**1. Reach falls with depth — the opposite of the hypothesis.** New instrument
`evo2/scripts/steer_reach.py` measures `reach` = mean KL of the next-token distribution under an
edit sized as a fraction of the **local** residual norm (n=40 held-out cores, 3 shuffled-label
controls). At frac 0.16: **L16 0.01011 → L20 0.00604 → L24 0.00359 → L27 0.00288.** The same
relative edit at L27 moves the output **3.5x less** than at L16.
*Why:* the residual stream is additive, so an L16 edit is carried forward **and** read, amplified
and re-expressed by the 11 blocks after it; an L27 edit has 4 blocks left to be read by. "Closer
to the output" means fewer opportunities to be used. The L28 norm explosion is *after* both
injection points and attenuates them equally.

**2. Generation confirms it at L27** (`evo2/experiments/probes/run_steer_l27.sh`, Phase 3's seeded
cross-class design, n=12/dose, 0 skipped):

| dose (frac live ‖h‖) | class-units | coding_density | target markers | seed markers |
|---|---|---|---|---|
| 0.061 | 1.1 | 0.925 | **0/12** | 6/12 |
| 0.16 | 2.9 | 0.883 | **0/12** | 5/12 |
| 0.32 | 5.9 | 0.896 | **0/12** | 6/12 |
| 0.64 | 11.9 | **0.684** | **0/12** | 3/12 |

**0/48 overall**, spanning 1.1–11.9 class-units (up to ~16x Phase 3's output-distribution impact).
The dose-response is Phase 3's signature exactly: the target class never appears, while the
*seed's own* markers erode (6/12 → 3/12) and coherence collapses at the top dose. **Degradation,
not redirection.** A pre-registered stop rule (≥2/12 target markers with coding ≥0.85) was set
before the data landed; nothing fired, so the paired Stage 2 was not run.

**3. Dosing bug found and fixed.** The activation cache stores *mean-pooled* states, so ‖h‖ read
from it disagrees with the live per-position norm at the hook by 0.75x at L16 but **2.84x at L27**
(pooled 11.25 vs live 31.97). Same failure family as the retired `_ref_norm`. Fixed by
`--steer-norm-frac` (dose = fraction of the live local norm, recomputed per position) with the
realized ‖h‖/‖delta‖/dose persisted per record. **Phase 3's L16 dose was 0.082 of the local norm
— slightly stronger than believed, so its null is not an under-dosing artefact.**

⇒ **Inference-time steering is closed**: wrong-axis, toxic-dose and floor-bound readout were all
fixed, the layer was made a variable, and the dose was made comparable — and it is still null.
Multi-layer / later-layer steering are dead too, since they target a non-binding constraint that
worsens with depth. **The next spend is training-time coupling**: per-class LoRA adapters, or a
class-prediction loss that forces the last blocks to read the class coordinate.

**Standing debt — CLEARED 2026-08-10**, see "Leakage debt: CLEARED" below. *(Historical text:*
all steering directions, including the L27 build, were fit on **val+test**.*)*

## ★★ PHASE 3 FINAL (2026-07-31): steering does NOT control generated class. Program answered.

Cross-class override — seed a real class-A core, steer the continuation toward class B, ask
whether B's machinery appears. Three paired arms, all three doses, `valtest_eval` seeds:

| dose | A real | B shuffled-label | seed markers (A) | seed markers (unsteered) |
|---|---|---|---|---|
| 1 | 0/48 | 0/47 | 0.375 | 0.396 |
| 2 | 2/46 | 0/47 | 0.457 | 0.396 |
| 4 | 0/46 | 1/48 | 0.326 | 0.396 |
| **pooled** | **2/140 = 0.014** | **1/142 = 0.007** | — | — |

Unsteered base rate of cross-class markers arising by chance: **3/144 = 0.021**.
`P(≥2 | p=0.021, n=140) = 0.79`. **Real and shuffled arms are indistinguishable from each other
and from chance, at every dose.** Steering slightly *perturbs* the seed's own class at dose 4
(0.326 vs 0.396) without installing the target's — degradation, not redirection.

**This is the plan's kill criterion**: arm A does not beat the shuffled-label arm B, properly
paired and nuisance-matched, across the entire dose range Phase 2 showed to be damage-free.

**The coherent picture across phases.** The model *does* use class in its next-token distribution
(Phase 1: p=0.040 on two independent interventions, ablation z=4.8) — but that effect is far too
weak to redirect 3,000 sequential sampling decisions. Exactly the gap flagged at the outset:
teacher-forced scoring can only ever give good news.

**Power, stated honestly:** this rules out a lift to ≥10% (we would have seen it with p≈0.994).
It does not rule out a lift to ≤5%. A 5%-of-generations effect is not a useful conditioning
mechanism, so this is a decision-grade negative, not a proof of zero.

⇒ **Next spend is per-class LoRA adapters**, the fallback the plan named. See decisions.md
"The seed does TWO jobs" — an adapter can install BOTH (BGC-ness and class), which is what
taxonomy-only generation needs and what steering could never supply.

## ★ PHASE 1 RESULT (2026-07-30): the model DOES use class — steering has something to act on

`evo2/scripts/steer_causal_tests.py`, layer 16, directions from a genome-disjoint half of
val+test, scored on the other half (`valtest_eval.jsonl`), **24 shuffled-label controls**:

| test | real | controls | z | controls beating it | p |
|---|---|---|---|---|---|
| **B — nudge, dose 1** | +0.00225 | ±0.00115 | 2.3 | **0/24** | **0.040** |
| B — nudge, dose 4 | +0.00911 | ±0.00956 | 1.0 | 5/24 | 0.240 |
| **C — ablation** | −0.00394 | −0.00127 ± 0.00055 | **4.8** | **0/24** | **0.040** |
| A — seeded state follows seed vs tag | 16 vs 7 | — | — | — | 0.047 (paired) |

Two *independent interventions* — adding the direction and deleting it — each beat all 24
controls at p = 0.040 (both AT the p-floor, so more controls would lower it). Deleting the real
class direction costs 3.1x more than deleting a shuffled one.

**Two corrections to earlier claims, both mine:**

1. **A "beat the max of the controls" decision rule is invalid.** It gets STRICTER as controls are
   added, so the identical effect read "USES CLASS" with 2 controls and "inside null" with 6. Now
   replaced by a proper permutation p = (#controls ≥ real + 1)/(#controls + 1) plus a z against the
   control distribution. Note p is FLOORED at 1/(n+1) — with 6 controls the best attainable p is
   0.143, so a "non-significant" verdict can be a limit of the control count, not the data.
2. **The dose-response argument was wrong.** I claimed "the gap doubles when the dose doubles —
   artefacts don't do that". But the CONTROL spread grows faster: dose 1→4 grows the real effect
   4x (0.00225→0.00911) and the control spread **8x** (0.00115→0.00956). Signal-to-noise gets
   *worse* with dose, which is why dose 4 is non-significant. Dose-response is expected of any
   perturbation and is not evidence.

**Operating point: dose ≈ 1 class-unit.** Higher doses are noisier, not stronger.

## ★★★★ PHASE-2 VERDICT (2026-08-13, n=152/arm): THE OBJECTIVE HYPOTHESIS IS CLOSED.
## The kill criterion now applies on its own terms — this test WAS powered.

`evo2_1b/experiments/rerun_arms_highn.sh` → `compare_arms_highn.py`. No retraining: the same three
adapters, regenerated at **152 records per arm** (the power-analysis target), at **8,000 nt** so the
6 kb ceiling can no longer censor one arm only, batched (12.05×, validated), scored per record.

| arm | n | detection | `best_bio_bits` mean | median ORF | novelty |
|---|---|---|---|---|---|
| baseline | 152 | **17/152** (0.112) | 7.43 | 528 | 0.011 PASS |
| frame | 152 | **10/152** (0.066) | 3.48 | **992** | 0.009 PASS |
| weighted | 152 | **16/152** (0.105) | 5.24 | 502 | 0.014 PASS |

| comparison | detection | `best_bio_bits` |
|---|---|---|
| frame vs baseline | Fisher **p = 0.226** | A = 0.477, p = 0.152 |
| weighted vs baseline | Fisher **p = 1.000** | A = 0.496, p = 0.810 |

**NOVELTY CLEAN IN ALL THREE** (max containment 0.009–0.014, `PASS_novel`). Nothing here is recited.

**ACHIEVED POWER — the reason this null counts and the n=24 null did not.** At the observed baseline
rate of 0.112, this design has power **0.74 to detect a doubling**, 0.96 for 2.5×, 1.00 for 3×. The
n=24 pass had ~0.15 for a doubling. ⚠️ Power for a **1.5× rise is only 0.29**, so a *modest*
improvement would still be missed — the closure is on effects of roughly 2× and above.

⇒ **THE PRE-REGISTERED KILL CRITERION NOW APPLIES.** It was correctly *withheld* at n=24, where the
test could not have rejected the hypothesis; it is correctly *applied* here, where it could.
**Neither objective moved de novo biosynthetic content. Do not build another loss variant.**

**AND THE INTERVENTION DEMONSTRABLY WORKED — that is what makes this a closure rather than a
failure.** The frame arm at n=152: median ORF **992 vs 528**, A = 0.736, **p = 1.1e-12**. It also
suppresses in-gene stop codons 8× (probe 2). The lever was pulled hard, verified three ways, and the
outcome did not follow.

**⚠️ CORRECTION TO THE n=24 READING.** At n=24 the *any*-Pfam comparison was a clean null (A=0.512,
p=0.89) and it was reported as "frame writes protein just as recognisable as baseline's, merely not
biosynthetic." **At n=152 that is false:** any-Pfam signal **21.94 vs 35.18, A = 0.406, p = 0.004**.
Frame's protein is significantly *less* recognisable in general, not just less biosynthetic. So
forcing the model past the point where it wants to stop produces **longer AND worse** protein, not
longer-but-equivalent. *Rule: a null at low n is not a finding; it is the absence of one.*

## ★★★ DOMAIN WEIGHTING AT 10x (2026-08-13): the lever has an almost FLAT DOSE-RESPONSE.
## Closed on a measured curve, not on a null.

`evo2_1b/experiments/run_weighted10.sh`. Identical to the 3x arm in every other respect (L=8192,
chunk/1024, bs1 x ga16, 400 steps, same seed and data order). The script GATES on the treatment
landing before it will spend an hour generating — the rule that was violated twice on 2026-08-12/13.
**The gate fired and generation was skipped.**

| condition | in/out CE ratio | shift vs untrained base | as % of what plain training buys |
|---|---|---|---|
| untrained base | 0.9042 | — | — |
| 3-step smoke runs (×5) | 0.9042–0.9043 | +0.0000 | ~0% |
| **400 steps, weight 1× (baseline)** | 0.9011 | **−0.0031** | **100%** |
| 400 steps, weight 3× | 0.9006 | −0.0036 | 117% |
| 400 steps, weight **10×** | 0.9002 | −0.0040 | **130%** |

⚠️ **CORRECTED SAME DAY — the ratio hid which side moved, and it was the wrong side.** The table
above reports `in/out`, which controls for a model that is simply better everywhere. It also
conceals whether the numerator fell or the denominator rose. Splitting them, with a PAIRED
Wilcoxon over the same 35 cores:

| | in-domain (the target) | non-domain (collateral) |
|---|---|---|
| plain fine-tuning (base → baseline) | **−0.00556, p < 0.0001** | −0.00284 |
| + weight 3× | −0.00039, p = 0.036 | +0.00012, p = 0.39 |
| + weight **10×** | **−0.00041, p = 0.15** | **+0.00054, p = 0.0001** |

⇒ **3× and 10× have IDENTICAL in-domain loss (0.8763 both).** The entire ratio difference between
them is the *denominator degrading*. So there is **no dose-response in domain learning at all** —
only a dose-response in damage to everything else.
⇒ The domain gain is **7.0% (3×) and 7.4% (10×) of what plain fine-tuning buys**, marginal at 3×
(p=0.036, one of several comparisons) and **not significant at 10×** (p=0.15).
⇒ **At 10× the harm (+0.00054, p=0.0001) is LARGER than the benefit (−0.00041, p=0.15)** — the only
thing that reliably scales with the weight is neglect of the down-weighted positions.
*Rule: when a ratio is the headline, print the numerator and denominator beside it; a ratio can
improve by fixing the thing you want or by breaking the thing you don't.*

⇒ ~~3.3× more weight bought 1.8× more effect.~~ Ordinary fine-tuning moves this ratio −0.0031; ten-
fold domain weighting adds only −0.0009 on top of that, i.e. **30% of what simply training does at
all**. The lever does not merely respond weakly to dose — on the quantity that matters it does not
respond to dose at all.

**An unintended control made this readable.** The probe auto-discovers arms, so it also scored five
3-step smoke runs. All five sit at 0.9042–0.9043, indistinguishable from the untrained base — which
pins the "no training happened" end of the scale and confirms the probe measures training, not noise.

⇒ **DIFFUSENESS WAS THE WRONG EXPLANATION.** The 3× null was hypothesised to be a too-broad nudge
(40.2% of positions). At 10× the contrast is unambiguous and the model still barely moves. So
**per-token loss weighting is a weak lever on this substrate**, and that is now established by a
dose-response curve rather than by a single null — a stronger form of evidence.

⚠️ **What this does NOT establish.** Both arms trained on 6.7% of one epoch. The flat dose-response
holds *at this training budget*; it does not rule out the weighting biting after 15× more training.
That is the one remaining variable never varied.

⇒ **NEXT, in order.** (1) If domain focus is still wanted, change the **DATA, not the loss** — train
on domain-dense regions directly, rather than reweighting them inside full windows; per-token
weighting has now been measured and found weak. (2) Train past 6.7% of an epoch before any further
objective claim. (3) Track A (exemplar conditioning) remains the path that works today.

## ⚠️ CORRECTION (2026-08-13, same day): THE CLOSURE APPLIES TO THE FRAME ARM ONLY.
## The weighted arm's treatment NEVER LANDED, so its null is uninterpretable.

`evo2_1b/experiments/probe_domain_weighting.py`. The weighted arm was declared closed alongside
frame. It should not have been: it is indistinguishable from baseline on **every** measured
quantity (gene length p=0.23, any-Pfam p=0.25, best_bio_bits p=0.81, n_bio_domains p=0.88,
bio_span_frac p=0.89, stop-completion mass 0.1228 vs 0.1227). For frame, the intervention was
verified BEFORE its null was trusted. For weighted, it never was — and the null was reported as a
closure anyway. Same error as the n=24 pass, one day later.

**The check.** Score fixed real held-out cores under each adapter; split per-position CE into
in-domain and out-of-domain (35 cores, 40.2% of positions in-domain, same pipeline that built the
training sidecar). The statistic is the RATIO in/out — a model simply better everywhere would show a
lower domain loss without having been steered at all.

| model | in-domain | out | ratio | vs baseline |
|---|---|---|---|---|
| base (no adapter) | 0.8823 | 0.9758 | 0.9042 | +0.34% |
| baseline | 0.8767 | 0.9729 | 0.9011 | — |
| frame *(control)* | 0.8829 | 0.9790 | 0.9019 | +0.09% |
| **weighted** | 0.8763 | 0.9731 | **0.9006** | **−0.06%** |

⇒ **−0.06% is nothing.** The whole spread across four models — including one that never saw domain
weighting at all — is 0.4%, and ordinary fine-tuning moved the ratio **+0.34%**, roughly 6× more
than the weighting did. Frame, the negative control, sits at +0.09% as expected, which fixes the
noise floor.

**NOT a plumbing bug — checked.** The trainer logged `domain_weight=3.0`, 47,524 annotation records,
**100% coverage** on the first 200 records, and the weighted loss differs from plain CE on all 40
logged steps. The weighting was applied and simply did not move the model.

**Two live explanations, both testable:**
1. **TOO DIFFUSE.** 40.2% of positions are in-domain. "Attend 3× harder to 40% of the text" is a
   broad, mild nudge. The frame penalty bit hard because it was SHARP — a rare, specific event
   (a stop-completing base at codon phase 2) with a large relative penalty. *Fix: much larger
   weight, or a narrower target.*
2. **TOO LITTLE TRAINING.** Every arm saw 6,400 windows = **6.7% of ONE epoch**; the 7B reference
   (best_bio_bits 56.9) saw 24× more. *Fix: train longer.*

⇒ **REVISED PLAN: the frame/length question is CLOSED. Domain weighting is UNTESTED, not refuted.**
Track A (bank exemplar conditioning) is still the safe path, — it works
(correct_class 0.283 vs a 0.067 floor, controls passed, memorisation ruled out) and is the mode
Evo's own published work validates. Track C (per-layer adapters) stays deferred: it targets class,
and class was never the binding constraint.

## ★★★ PHASE-2 PROBES (2026-08-13): the frame objective WORKED, and length is not the bottleneck

Two follow-up probes on the finished adapters. Together they turn the arm result from "underpowered
null" into a **positive demonstration with a clean dissociation**.

**PROBE 2 — did the frame arm learn to hold frame, or just to never stop?**
(`evo2_1b/experiments/probe_stop_suppression.py`.) Same real held-out cores, same pyrodigal gene
calls, only the model differs; the metric is the trained quantity itself — probability mass on the
base that would CLOSE an in-gene stop codon.

| model | stop-completion mass |
|---|---|
| base Evo2 (no adapter) | 0.1215 |
| baseline | 0.1227 |
| **frame** | **0.0147 (−88%)** |
| weighted | 0.1228 |

⇒ The frame arm is **8× less willing to end a gene**, and baseline/weighted are untouched — the
control works. Stops are **suppressed, not abolished** (abolition would read ~0), so the longer ORFs
are a real learned change rather than pure runaway.
⚠️ **This RETRACTS the same-day reading that the penalty was a weak intervention** because
`loss_stop_pen` was only 1–4% of the loss magnitude. A small loss term produced an 8× behavioural
change. *Rule: the size of a loss term is not the size of its effect.*

**PROBE 1 — extend generation 6,000 → 8,000 nt (the 1B's full context).**

| arm | median `max_orf_aa` | mean | max | ≥2000 (old cap) | at the 8 kb wall |
|---|---|---|---|---|---|
| baseline | 468 | 528 | 1,157 | 0/24 | 0/24 |
| **frame** | **1,038** | 1,330 | 2,666 | **6/24** | 4/24 |

A = 0.850, **p < 0.0001**.

⇒ **The 6 kb measurement was CENSORED, and only for one arm.** Baseline never approached the cap
(max 1,157), so it was measured cleanly; frame had 6/24 records above the old ceiling. The original
"1.5× median" was therefore an UNDERSTATEMENT — with room, it is **2.2×**, and the effect size rises
from A = 0.715 to A = 0.850. *Rule: a ceiling that binds on one arm only is a confound, not a
nuisance.* 4/24 do run to the new wall, so a runaway tail exists — but 20/24 terminate on their own.

**⇒ THE DISSOCIATION, which is the actual result.**

| | baseline | frame |
|---|---|---|
| gene length | 468 | **1,038** (p<0.0001) |
| **any**-Pfam signal (`best_any_bits`) | 31.34 | 22.92 — **A = 0.512, p = 0.89, no difference** |
| **biosynthetic** signal (`best_bio_bits`) | 17.12 | **1.19** |
| records with any bio signal | 4/24 | 1/24 (Fisher p = 0.348, n.s.) |

The frame arm writes protein that is **just as recognisable as baseline's** — the any-Pfam
comparison is a clean null — it is simply **not biosynthetic**. Making the model write LONGER genes
produced longer protein of the same wrong kind.

⇒ **LENGTH IS NOT THE BOTTLENECK — now shown three independent ways:** observationally (ladder
audit, r = 0.051 / −0.120), and causally at two generation lengths. The direction replicates across
6 kb and 8 kb; the two are **not pooled** (different conditioning — sequential@6kb vs batched@8kb,
see `bugs.md`), and neither is individually significant on domain content at n=24. It is the
*dissociation* that is strong, not the domain-content difference.

⇒ **DO NOT build another loss variant aimed at reading-frame length.** The remaining question is
domain *identity*, which the weighted arm targets and which has never been tested at adequate power.

## ★★★ PHASE-2 ARM RESULTS (2026-08-12, completed 20:45) — the frame loss MOVED ITS OWN
## VARIABLE AND DOMAIN CONTENT DID NOT FOLLOW. Primary metric was underpowered by construction.

All three arms trained (400 steps, L=8192, chunked), generated (24 de novo each, identical
prompts/decoding/seed) and scored. **Novelty PASS_novel in every arm** (max containment 0.012 /
0.003 / 0.003) — nothing here is memorisation.

| arm | `best_bio_bits` | detection | `n_bio_domains` | `max_orf_aa` (diag) | novelty max |
|---|---|---|---|---|---|
| baseline | 11.82 | **3/24** | 0.208 | 468.4 | 0.012 |
| frame | 2.14 | **3/24** | 0.125 | **826.2** | 0.003 |
| weighted | 11.83 | **3/24** | 0.375 | 437.5 | 0.003 |

**⚠️ THE HEADLINE DELTAS THE DRIVER PRINTED (frame −10.201, weighted −0.503) ARE NOISE.** Per-record
scoring shows **exactly 3 of 24 records carry any biosynthetic signal in every arm**, and the mean
is dominated by one draw: baseline's top record is **64% of its arm total** (182.6 bits), frame's is
45%. Pairwise Mann-Whitney on `best_bio_bits` gives **A = 0.508 / 0.497 / 0.492 — indistinguishable
from 0.5**, and detection is 3/24 in all three arms (**Fisher p = 1.000**). The primary metric did
not separate the arms in either direction.

**WHAT DID MOVE — and it is the pre-registered informative negative.** `max_orf_aa`, the variable
the frame loss manipulates directly: median **453.5 → 700.0**, A = 0.715, **p = 0.0109**. The
intervention worked. Domain content did not follow — same 3/24 detection, `n_bio_domains` if
anything *lower* (0.208 → 0.125). This is the second, now **causal**, line of evidence for the
ladder audit's demotion of `max_orf_aa`: the audit found r = 0.051 / −0.120 *observationally*; here
we pushed ORF length up ~54% at the median and domain content did not move. **Length was never the
constraint.**

**TWO HONEST QUALIFICATIONS, both of which cut against reading this as a strong result.**

1. **The frame arm partly Goodharted its own penalty.** 2/24 records have a single ORF spanning the
   *entire* 6 kb generation (`max_orf_aa` = 2000 = the cap: the model never emitted a stop at all),
   and 5/24 are ≥75% of cap. Dropping the saturated tail attenuates the effect to **A = 0.640,
   p = 0.1204** (≥50% of cap: p = 0.0651). Direction and median shift survive; significance does
   not. A stop-completion penalty can be satisfied by suppressing stops outright, which is not the
   same as learning to hold frame.
2. **THE DESIGN COULD NOT HAVE DETECTED A MODERATE EFFECT ON THE PRIMARY METRIC.** At a baseline
   detection of 3/24 = 0.125, the n needed per arm (α=.05, 80% power) is **152 to detect a doubling,
   46 to detect a tripling**. We ran **24**. The kill criterion in `score_arms.sh` was written
   assuming the test would be sensitive; that assumption failed. Firing it here would discard the
   hypothesis on the basis of a test that could not have rejected it.

⇒ **VERDICT.** The `max_orf_aa` dissociation is real and worth keeping — it corroborates the ladder
audit independently and causally. The comparison *between arms on domain content* is **not a clean
negative, it is an underpowered one**, and must not be written up as a closure.

⇒ **THE FIX IS CHEAP AND NEEDS NO RETRAINING.** The three adapters exist. Regenerating at
**n ≥ 150 per arm** (~34 s/record solo ⇒ ~4 h for all three) makes the existing arms interpretable.
Do that BEFORE running any further loss variant.

## What was running (now complete)

**PHASE 2, three objective arms on the 1B.** `evo2_1b/experiments/run_objective_arms.sh` →
`/data2/ds85/bgcmodel_runs/phase2_1b/{baseline,frame,weighted}`. Identical in every respect except
the objective:

| arm | flags | what it tests |
|---|---|---|
| baseline | `--domain-weight 1.0 --frame-lambda 0.0` | bit-identical to `causal_lm_loss` (pinned by test) |
| frame | `--frame-lambda 0.5` | in-gene stop-completion penalty |
| weighted | `--domain-weight 3.0` | per-record-normalised domain weights |

Config: **L=8192** (the 1B's native context), `--long-seq-strategy chunk --chunk-overlap 1024`
(95,759 windows over all 467 Mbp), batch 1 × grad-accum 16, 400 steps, LoRA. ~15 s per optimiser
step at ~8,700 tok/s ⇒ **~100 min per arm**. Actual: baseline 16:29→17:46, frame 17:48→19:16,
weighted 19:17→20:45, all scored by 21:00.

Scored afterwards by `evo2_1b/experiments/score_arms.sh`: de novo generation with identical
prompts/decoding/seed, then the validated ladder + novelty. **Primary = `best_bio_bits`**;
`max_orf_aa` is a diagnostic only (the frame arm manipulates it directly, so scoring there would be
scoring the manipulation).

## ★★★ LADDER AUDIT (2026-08-12): the primary metric is `best_bio_bits`, not the fraction

`evo2/scripts/ladder_audit.py`. `max_orf_aa` was adopted on BETWEEN-group evidence plus a
mechanistic story, and failed the WITHIN-group test. `biosynthetic_fraction` rested on exactly the
same kind of evidence, so it got the same test before adoption.

**PART 1 — within-group validation.** Seeded arm (n=120, 44 detections) is the only regime with
variance rather than a floor. AUROC for predicting the *independent* antiSMASH outcome:

| metric | predicts is_bgc | predicts correct_class |
|---|---|---|
| **`best_bio_bits`** (absolute) | **0.950** | **0.925** |
| `n_bio_domains` | 0.919 | 0.901 |
| `bio_span_frac` | 0.896 | 0.891 |
| `biosynthetic_fraction` (ratio) | 0.893 | 0.876 |
| `best_any_bits` | 0.804 | 0.784 |
| `max_orf_aa` | 0.709 | 0.728 |
| `co_orient` | 0.654 | 0.659 |
| `modules` | 0.500 | 0.500 |

⇒ **The absolute biosynthetic bitscore beats the ratio (0.950 vs 0.893), so it — not
`biosynthetic_fraction` — is the primary target.** The ratio's denominator adds noise: a sequence
with a strong biosynthetic hit AND a strong unrelated hit is penalised for no good reason. Keep the
ratio as a *specificity* diagnostic (it is what showed the model writes real-but-wrong proteins),
not as the objective. `max_orf_aa` retains weak signal here (0.709) but had **none** de novo
(r = 0.051 / −0.120), so it stays a structural diagnostic only.

**PART 2 — the previously unmeasured rungs.** Mean per group:

| metric | de novo | seeded | REAL | verdict |
|---|---|---|---|---|
| `n_bio_domains` | 0.20 | 1.71 | **2.48** | **RUNG** (AUROC 0.919) |
| `bio_span_frac` | 0.051 | 0.390 | **0.876** | **RUNG** (AUROC 0.896) — the clustering measure |
| `n_bio_orfs` | 0.16 | 0.75 | 1.36 | rung, weaker (0.826) |
| `n_orfs` | 4.02 | 2.94 | 2.12 | separates, but INVERTED — more ORFs is worse |
| `co_orient` | 0.805 | 0.885 | 0.973 | too weak (0.654) |
| `modules` / `in_order` | 0.000 | 0.000 | **0.000** | **length floor, not broken** — a module needs ~1000–1500 aa and cannot fit the 3 kb window this cohort used; the check is correct (5 ordered modules on a real 5,951 aa megasynthase). Diagnostic at ≥6 kb only. See bugs.md. |

`bio_span_frac` is the rung that was missing: it measures whether biosynthetic domains are SPREAD
across the sequence like a cluster rather than crammed in one spot. Real 0.876, de novo 0.051.

**⇒ THE VALIDATED LADDER**

| # | rung | de novo | seeded | REAL | status |
|---|---|---|---|---|---|
| 0 | coding_density | 0.74–0.82 | 0.93 | 0.97 | nearly closed |
| 1 | any Pfam hit | **1.00** | 0.94 | 1.00 | **SOLVED** — the model writes real protein |
| 2 | **`best_bio_bits`** | 0.7–15.7 | 72.3 | 148.6 | **PRIMARY TARGET** |
| 3 | `n_bio_domains` | 0.20 | 1.71 | 2.48 | |
| 4 | `bio_span_frac` | 0.051 | 0.390 | 0.876 | clustering |
| 5 | antiSMASH detect | 0.012 | 0.367 | ~0.58 | |
| 6 | correct_class | ~0 | 0.34 | ~0.45 | |
| — | `biosynthetic_fraction` | 0.076 | 0.528 | 0.836 | specificity diagnostic |
| — | `max_orf_aa` | 448 | 542 | 729 | structural diagnostic only |
| **guard** | **novelty** | — | — | — | **every rung above is maximised by copying training data** |

**The novelty guard is not optional, and here is what it actually does.** Rungs 2–4 all reward
sequence that looks like the training set, so the cheapest way to win any of them is memorisation.
`check_kmer_novelty` + `scripts/memorization_check.scan_corpus` measure **containment**: cut the
generation into every overlapping 21-nucleotide window, and ask what fraction of those windows also
occur in the single most similar real BGC. ≥0.95 → `FAIL_memorized`; ≥0.80 → `WARN`; below →
`PASS_novel`.

Three design points that matter for reading it:
* **Containment, not similarity.** "What share of MY k-mers appear in that reference" is the right
  question for a fragment: a 3 kb generation can be a verbatim copy of part of a 30 kb cluster, and
  a symmetric similarity score would read LOW because the reference has so much extra. Containment
  reads 1.0, correctly.
* **k=21** is long enough that a match is essentially never chance (4^21 ≈ 4×10¹²) and short enough
  to survive differences elsewhere in the sequence.
* **A skip is never a pass.** No scan supplied → SKIPPED. Scan present but missing the
  `max_containment` key → SKIPPED as malformed, explicitly refusing to default. That guard exists
  because an earlier version used `.get("max_containment", 0.0)`, and 0.0 means *maximal novelty* —
  so a malformed record certified a memorised sequence as novel.

It is a CONSTRAINT, not a rung: report it beside every ladder number, and treat an improvement with
an unverified novelty gate as uninterpretable rather than positive.

## ★★★ PHASE 2 (2026-08-12): the 1B track — substrate established, objective built and running

Full rationale in `decisions.md` 2026-08-12 and `evo2_1b/README.md`. Summary:

**Substrate.** `evo2_1b_base`, 1.108B params, 25 blocks (4 attn / 21 Hyena), hidden 1920, native
context **8,192**. **Transformer Engine 1.13.0 is REQUIRED and now installed** (2.18 will not build
against torch 2.5.1). Without TE the model loads and is **at chance** — 1.339 nats/base, predictive
entropy 1.357, uniform 1.386 — because the checkpoint stores FP8 scale metadata that TE must
dequantise. The long-standing "no small model exists" note was right about the conclusion and wrong
about the reason; "it's only a name check" was wrong about the substance.

| | 1B + TE | 7B |
|---|---|---|
| nats/base, real cores | **0.990** | 0.859 base · 0.820 +LoRA |
| throughput | **8,770 tok/s** | **2,625 tok/s** |

⇒ **3.34× faster, not 6×.** Both are byte-level, so there is no token-compression win; the speedup
is depth/width only. The +0.13 nats handicap is acceptable for *does this change anything*, but a
Phase-2 positive **must be confirmed on the 7B** before it is a project result.

**Objective** (`src/bgc_pipeline/objective.py`, shared by both tracks). Domain-weighted (per-record
normalised) and frame-aware (in-gene stop-completion penalty, real termini exempt at the whole
position). Defaults are bit-identical to `causal_lm_loss`. `loss_ce` / `loss_stop_pen` logged
separately; measured on the 1B at 0.10–0.29 penalty against 0.93–1.45 CE.

**Prerequisites built.** `scripts/build_domain_spans.py` now persists **gene spans + strand** and a
**row index** (47,524 rows); `src/bgc_pipeline/annotations.py` slices whole-record annotations to
the training window and offsets past the prefix; the trainer gained `--domain-weight` /
`--frame-lambda` / `--annotations` with a fail-loud guard and a ≥90% coverage check.

**Three bugs caught before they could corrupt a result** — see `bugs.md`: an accession-keyed join
that would have mismatched 12,217 records; a `truncate` default that biased the data 49.0% vs the
true 33.7% class-domain *against the arm under test*; and a sanity check that passed a
chance-level model three different ways.

## ⚙️ CARRIED TO THE NEXT TRAINING ROUND (2026-08-12) — throughput, decided but NOT applied

Two throughput items surfaced while the Phase-2 arms ran. **Neither is applied to the running
arms**, for the same reason batch size is not: the design is *identical in every respect except the
objective*, and a change made to arms 2 and 3 but not to arm 1 is a second difference.

1. **Resolve the half-enabled determinism** — the actionable one. `finetune_evo2_lora.py:361,367`
   sets `torch.use_deterministic_algorithms(True, warn_only=True)` and `cudnn.benchmark = False`,
   but `CUBLAS_WORKSPACE_CONFIG` is **unset** in the run environment, and torch says so at runtime.
   The GEMMs that dominate are nondeterministic anyway, while `benchmark = False` gives up kernel
   autotuning — and Evo2 is a *convolutional* architecture (StripedHyena; 21 of the 1B's 25 blocks
   are Hyena), which is precisely where cuDNN autotuning pays. So the cost may be larger here than
   in a pure-attention model.
   **Do not adopt blind — measure it first.** Written and ready:
   `evo2_1b/experiments/probe_determinism_cost.sh` runs 40-step smokes at L=8192, identical except
   for (a) as-is and (b) `CUBLAS_WORKSPACE_CONFIG=:4096:8` (real reproducibility), and reports
   steady-state `tokens_per_sec`. A third arm — determinism dropped + `cudnn.benchmark = True`, the
   pure-speed option — **is not runnable yet**: it needs a `--no-deterministic` flag that does not
   exist (both settings are unconditional at `finetune_evo2_lora.py:361,367`). Add the flag rather
   than hand-editing those lines, so a probe cannot leave the default path altered.
   **Whichever wins, fix it for a WHOLE round** — never between arms of one comparison — and if the
   spread is under ~5%, keep as-is: a change to the numerical path is not worth making for noise.

2. **Re-run `compare_1b_7b_loss.py` under TE** (2 min, GPU, after the arms). The on-disk result was
   produced *before* TE was installed and holds **1.327 nats** for the 1B — the broken bf16 number,
   not the 0.990 the working substrate gives. The file is honestly labelled but is a trap for anyone
   reading numbers rather than labels, so it is quarantined as
   `compare_1b_7b_loss.NO_TE.STALE.json` until the real one replaces it.

3. **CUDA MPS — REJECTED, do not enable.** MPS is the only thing that makes separate CUDA processes
   run concurrently rather than time-slice (see `bugs.md`), so it is the real fix for co-tenanting
   arms. **It is nonetheless off the table on `gputee`:** the daemon is host-wide and this box had
   35 logged-in users at the time. A throughput experiment does not justify changing the GPU
   execution model out from under other people's jobs. Sequential arms + `WAIT=1` scoring (which
   overlaps the finished arm's *generation* with the next arm's *training*) recovers the useful part
   at zero risk to anyone else. Revisit only on a machine we hold exclusively.

## NEXT ACTIONS — REWRITTEN 2026-08-12 after the detection/capability measurements

**The previous list ranked ways to condition class. That was the wrong target.** See LADDER AUDIT
(2026-08-12) above and the two detection numbers below. The ranked conditioning list survives as
reference in [`docs/conditioning_next_steps.md`](../conditioning_next_steps.md), now carrying a
superseded banner; its citations and caveats are still accurate, its *ranking* is not.

**Two numbers set the agenda.** De novo P(detect) = **0.012** vs seeded **0.367**, with seeded
class-given-detection already **0.932**. And the class tag is worth **−0.0006 nats** to the loss.
So there is ~7% for conditioning to win in the regime that works, and nothing to condition in the
regime that doesn't.

### A. Bank the result that works — a CHARACTERISATION paper, weeks, mostly CPU

**Framing (2026-08-12): descriptive paper about a known capability and its limits, not a novel
tool.** The contribution is the measurement programme — what works, what does not, and *why*, every
rate quoted against a measured ceiling and floor, with the negative results carried in full.

**Independent corroboration, found 2026-08-12.** *"Fundamental limitations of genomic language
models for realistic sequence generation"* (biorxiv 2026.01.17.700093 / PMC12871140) tested Evo 2
and megaDNA on whole-genome reconstruction and found synthetic sequences **"captured local sequence
statistics"** but **"consistently failed to preserve long-range genomic organization"** —
discriminator AUROC 0.97 (eukaryote) / 0.82 (prokaryote), with classification accuracy rising
monotonically with distance from the seed. That is our context-ablation result (73% of predictive
power from 10 bases) and our ORF scaling failure, reached by a completely different method. A strong
external check that we are not measuring an artefact of our own pipeline.

**Verified against the paper: it does NOT scoop us.** It covers neither biosynthetic gene clusters,
nor functional-class conditioning, nor ORF length; its readout is a CNN discriminator on genome
reconstruction, ours is functional (antiSMASH, Pfam domains, module architecture). Positioning:
*they showed gLMs fail at long-range structure in general; we show what that costs in a concrete
design task, quantify how much is recoverable by prompting, and close the conditioning mechanisms
one at a time.*

**The result being written up.** Exemplar conditioning is validated: `correct_class` **0.283 vs a 0.067 floor**, memorization ruled
out, all four pre-registered controls passed, and the scored span provably contains no seed
(0/1512). The detection numbers now supply the *mechanism* — the seed provides the recognisability
the model cannot generate — and this is the mode Evo's own published work validates experimentally
("genomic autocomplete"). **To do:** scale n, characterise which classes work and which don't
(quote every rate against its measured 3 kb ceiling), and add the novelty/diversity analysis.
Framed as *"extend and diversify a known cluster"*, not *"generate class X de novo"*.

### B. Attack de novo capability — the real bottleneck

**Primary target: `best_bio_bits`** (see LADDER AUDIT above — AUROC 0.950 for predicting the
independent antiSMASH outcome, against 0.893 for the ratio and 0.709 for `max_orf_aa`). Secondary
rungs `n_bio_domains` and `bio_span_frac`; `biosynthetic_fraction` retained as a specificity
diagnostic. **Novelty reported in every run as a guard** — every one of these rungs is maximised by
copying training data. `max_orf_aa` is a **structural diagnostic, not the objective**: within de novo generations
it does not track domain content (r = 0.051 at 2 kb, −0.120 at 6 kb) and it is gameable by avoiding
three codons.

**Structural diagnostic — `max_orf_aa`.** Length-matched against real cores it is **0.61 of real at a 2 kb
window and 0.48 at 6 kb** — and the gap WIDENS with length, because real DNA scales its longest ORF
with the room available (550 → 720 → 1,134 aa) while the model plateaus at ~340–550 aa. It also
emits more, shorter ORFs (6.0 vs 4.0 at 6 kb): it hits stops and restarts. This is a **scaling
failure**, and unlike `correct_class` it is continuous, non-zero today, and readable within hours
of a run starting.

**Form of the intervention: LoRA + a custom loss, NOT a full fine-tune.** Capacity has already been
ruled out twice — the rank sweep (16/64/128, all at the floor, 128 *worse*) and unfreezing the Hyena
long-range pathway (identical to control). A full FT would change capacity and objective at once and
the result could not be attributed. If LoRA + custom loss moves `best_bio_bits` at unchanged novelty, that is clean; if it
does not, full FT becomes the *next* question rather than a confound in this one.

1. ~~**Annotation pass**~~ **DONE 2026-08-12** — `scripts/build_domain_spans.py` →
   `train.domain_spans.jsonl`, 47,524 records with per-domain forward-strand nucleotide spans,
   round-trip tested on both strands.

**⚠️ SEQUENCING SUPERSEDED SAME DAY — all three arms (baseline / frame / weighted) were launched
together at L=8192; see "What is running right now" above.** The original argument, kept because the
length reasoning still holds: frame-aware is length-agnostic and can run short and fast, while
domain-weighted is least meaningful at short context (cores under 1 kb are already 78.6% domain, so
there is nothing to reweight). L=8192 proved affordable for every arm, so the trade-off was moot.
Weights are **per-record normalised**, so every core gets the same total up-weighting regardless of
its coverage — a flat multiplier would silently become a length reweighting (78.6% coverage under
1 kb vs 25.1% above 50 kb).

2. **THE 2×2 (kept as the eventual design): frame-aware × domain-weighted.** These are not ranked, deliberately. They attack the
   same measured failure from different angles and the relationship between "gradient weight on
   domain positions" and "reading-frame length achieved" is **unmeasured**. An earlier draft ranked
   frame-aware above domain-weighted on the argument that 33.7% of training nucleotides already sit
   inside class-defining domains so the gradient cannot be mis-aimed — that is a mechanistic story,
   not a measurement, and ranking on untested stories is the error that cost this programme months
   on conditioning. Four runs resolve it and attribute the result:

   | | domain-weighted OFF | domain-weighted ON |
   |---|---|---|
   | **frame-aware OFF** | baseline (current objective) | is reweighting alone enough? |
   | **frame-aware ON** | is a direct frame penalty enough? | do they compose? |

   - *Frame-aware:* penalise in-frame stop codons inside annotated genes, or weight by codon
     position. The most direct lever on the measured wall.
   - *Domain-weighted:* up-weight positions inside class-defining domains. A 2–3× reweighting at the
     measured 33.7% coverage — mild, but the classes and lengths where coverage IS low are exactly
     the hard ones (NUCLEOSIDE 9.4%, RESORCINOL 18.8%, RIPP 34.9%; cores >50 kb at 25.1%), so
     consider a per-length or per-class weight rather than a flat one.

3. **Auxiliary head predicting upcoming DOMAIN content** — explicitly *not* class, since the probe
   already recovers class at 0.911 and such a head would teach nothing. Forces the representation to
   carry a commitment about what is ahead, which next-base prediction never requires. Run only if
   the 2×2 moves nothing.

4. *Reserve:* sequence-level reward on domain presence. Directly optimises the target; expensive
   and high-variance.

**Kill criterion, stated in advance:** if no cell of the 2×2 moves `best_bio_bits` above baseline
(at unchanged novelty) within a single training run, that is a fast clean negative on the objective hypothesis — and the
question becomes scale/substrate, not another loss variant.

**Read it on the ladder, not the gate:** `best_bio_bits` (primary, AUROC 0.950) → `n_bio_domains`
(0.919) → `bio_span_frac` (0.896) → antiSMASH detect → class, with `biosynthetic_fraction` (0.893) a
specificity diagnostic and `max_orf_aa` (0.709) a structural diagnostic. Report the first three per
run under the novelty guard; the last two only once the first three move.

### C. Per-layer conditional adapters — DEFERRED, not dropped
The ProCALM-style design and its precedent are sound and written up in the plan doc. They now target
~7% of the seeded gap and nothing of the de novo one. **Revisit once de novo detection is
non-trivial**, at which point it becomes the natural next step.

### Also worth doing, cheap
- **Continuation-scoped and length-matched reporting.** Every rate quoted against its measured 3 kb
  ceiling (hybrids 0.00 — withdrawn; PKS 0.40; NRPS 0.76; RIPP 0.76; TERPENE 0.88).
- **Emit `accession` in `antismash.tsv`** — pairing currently relies on row order.
- **Add the max-vs-max Q1 test** to `analyze_guided_decoding.py`.
- **GenomeOcean** remains the substrate hedge: a real single-token class label, 52.7 kb context.

**Do NOT run:** another steering variant (layer, dose, or direction recipe — closed by a positive
demonstration); another input-only conditioning mechanism (the tag is worth −0.0006 nats).

**Standing methodological bar**, which three weakened-or-retracted findings in this project have
now paid for: a paired design with the control built in, a continuous readout alongside the binary
gates, and an instrument whose sensitivity AND false-positive rate are measured BEFORE a result is
read off it.


**Leakage debt: CLEARED 2026-08-10.** Probe and directions both refit train-only
(`acts_v2_train500.npz`, provenance-verified; `trainonly.steerdirs.npz` at 9 layers; probe cached
at `acts_v2_train500.probe_L16_s0.joblib`). `_fit_probe` now REFUSES a non-train fit set.

### Phase 0 + Phase 1 (2026-07-29) — directions rebuilt; instrument found broken

**Phase 0 built** (`evo2/scripts/build_steer_dirs.py`, CPU-only): one direction per class,
`μ_c − mean(μ_others)`, length-stripped, at layers 16/20/24/27, plus 10 shuffled-label controls
per layer. Held-out scored (in-sample scoring inflated the shuffled null to 0.852, nearly at the
0.90 gate; held-out drops it to 0.734). **All 11 classes at all 4 layers beat the null.**
The 0.90 gate is stricter than the evidence requires — NRPS reads 0.854 at L16, far above null.

**Length domination is a MID-LAYER phenomenon:** PC1 = 98.07% of variance at L16 and 96.10% at
L20 (both r≈−0.9996 with ‖h‖), but only 11.68% at L24 and 16.96% at L27. The original bug was
specific to where the steering was aimed.

**SAMPLE SIZE — sufficient to classify, NOT to steer.** Directions came from **real cores in
`val.jsonl`** (n=991; 100/class is a *cap*, and BUTYROLACTONE 40 / PHOSPHONATE 65 / ECTOINE 86
exhausted val's supply). Learning curve, two disjoint subsamples per n:

| n/class | split-half cosine | held-out AUC |
|---|---|---|
| 10 | 0.670 | 0.900 |
| 20 | 0.805 | 0.915 |
| 30 | 0.866 | 0.922 |
| 40 | **0.882 (still climbing)** | 0.919 |

AUC plateaus by n≈20–25, but **orientation has not converged** — ~10–15% of the vector is
estimation noise. Harmless for classification (projection separates anyway); **not** harmless for
steering, which *injects* the vector and so pumps that noise into the residual stream. Hence the
re-embed at 500/class from train (thousands available per class), to be validated on val/test.

**Phase 1 instrument is BROKEN — its nulls are uninterpretable.** The positive control
(real same-class vs different-class exemplar as prefix) came back
**−0.00126 ± 0.00107 — NOT detectable**. The measurement cannot see the class effect we know
exists (seeding, 0.283 vs 0.067), so Test B's "inside the null band" at every dose means nothing.
Diagnosed cause: the anchor also prepended 1000 nt of the *true* sequence, handing the model the
real thing immediately before the scored window. Fixed (exemplar-only context; the old design is
retained as `anchor_gap_withcond` to quantify the masking). Dose was *not* the problem on the
re-run — the damage guard shows 0.5% shift at dose 1 rising to 84% at dose 32.

Test A (does seeding move the state along our directions?): the mismatch arm leans to the **seed**
class over the **tag** class 11 vs 4, matching behaviour (0.317 vs 0.067), paired **p = 0.059** —
suggestive, not conclusive. Broken shuffled-seed arm matched no class (0/41, below chance).

### Assumption audit (2026-07-29) — six tested, none a showstopper

| assumption | verdict |
|---|---|
| directions pooled-over-positions but injected per-position | **cleared** — 1.8× scale ratio (NOT the 48× claimed in `steering_program_technical.md`); direction works per-position (0.859 vs 0.867 pooled) |
| block output normalized ⇒ injection washed out | **cleared** — `res_mlp_norm(x) = mlp(post_norm(x)) + x`; block output IS the residual stream |
| directions from full cores won't apply to short generations | **cleared** — cos(@1000nt, @full) = 0.85–0.98 |
| the 991 cores are independent | **fine** — 862 distinct genomes, max 4 per genome |
| "class direction" is really a taxonomy direction | **cleared** — separates phylum at 0.36–0.63 vs own class 0.86–0.99; phylum adds only +0.018 over baseline in the raw data |
| directions fit on real cores don't describe where generation lives | **cleared** — generation sits within 1–2σ of real cores; the behaviourally-working seeded arm is *closest* (±0.45σ), broken arm furthest (1.95×) |

- Magnitude titration COMPLETE 2026-07-29 14:03 — but see the geometry
  finding below: it titrated coherence along the **wrong axis**, so its ceiling is a valid
  measurement of a direction we should not be injecting. Coherence held to ‖delta‖=4 and
  collapsed at 8 (pooled 0.826 → 0.304); per-class "breakpoints" spread 2–8 but that spread is
  n=5 noise (NRPS reads 0.491 at ‖delta‖=2 then 0.950 at 4).
- Steering β-titration COMPLETE 2026-07-29 11:29 — **confounded, superseded** (see below).

### ★ THE ACTUAL BLOCKER (2026-07-29, verified by independent recomputation)

**The steering vectors are not class directions. They are ±the activation-norm/length axis.**
Every number below was recomputed directly from `class_probe_sweep/acts_v2.npz` (n=991, L16):

| measurement | value |
|---|---|
| PC1 share of centered variance | **98.07%** |
| corr(PC1 projection, ‖h‖) | **−0.9996** ⇒ PC1 *is* the norm/length axis |
| top singular value share of the 11×4096 direction matrix | **99.22%** ⇒ rank-≈1 |
| mean off-diagonal \|cos\| between the 11 "class directions" | **0.934** (94.5% of pairs >0.8) |
| class accuracy of diff-of-means as a classifier | **0.186** (logistic probe 0.881; whitened LDA 0.906; chance 0.091) |
| probe accuracy with PC1 **removed** | **0.909** (vs 0.908 full) — class lives in the residual 1.9% |
| probe accuracy from PC1 **alone** | **0.287** |

**The killer detail — the shipped direction points backwards for the pairs we care about:**

| intended steer | cos(true contrast, shipped v) | 1-D AUC |
|---|---|---|
| NRPS → PKS | **−0.856** | **0.221** |
| ECTOINE → TERPENE | **−0.789** | **0.070** |
| PKS → NRPS | +0.995 | 0.830 |

Because `v_class = μ_c − μ_global` is ~collinear with PC1, its **sign is set by whether the
class's cores are longer or shorter than average**, not by class identity. "Steer toward PKS"
and "steer toward NRPS" are the same intervention with opposite sign. **0/30 correct_class in
every α-sweep cell was structurally guaranteed** — independent of layer, magnitude, or n.

**Second confirmed defect:** `_ref_norm()` reads `X[:, -1, :]`, the *mean-pooled* activation
(‖·‖ = 9.97), so α ∈ {1,2,4} ⇒ ‖delta‖ ∈ {10,20,40} = **1.5–5.9× the entire between-sample
scatter** (mean ‖h−gm‖ = 6.752, recomputed). Even a correct direction would have been destroyed.

⇒ **Activation steering has never actually been tested.** Three independent, each-sufficient
defects — wrong axis, toxic dose, floor-bound readout — fully explain every null on disk.
Full diagnosis + costed 4-phase plan (P0 direction bank → P1 teacher-forced screen → P2 dose
response → P3 seeded cross-class override) and explicit kill criteria: see
`docs/steering_program.md`.

**The live risk (not a bug, not fixable by code):** class carries only **1.9%** of the variance,
activation sd along the class axis is 0.24–0.36 vs 8.1 along PC1, and **base Evo2 — which has no
class prior — decodes class as well as the fine-tuned model (0.911 vs 0.906)**. So L16 class may
be a readout *of the input* rather than a control variable *of the generator*. That is what the
P1 teacher-forced screen is designed to settle before any large GPU spend.

### THE BLOCKING PROBLEM: the steering readout has no dynamic range (2026-07-29)

The only steering run ever evaluated with antiSMASH (`/data2/ds85/bgcmodel_runs/steer_sweep/`,
grid L ∈ {12,16,20} × α ∈ {1,2,4}, n=5/class × 6 classes = 30/cell, 6144 tokens):

| cell | is_bgc | correct_class |
|---|---|---|
| **unsteered control (α=0)** | **1/30 (3.3%)** | 1/30 |
| all 9 steered cells | 0/30 | 0/30 |

Two independent fatal flaws, both worth remembering:

1. **Every steered cell was past the toxicity ceiling.** α=1 sets ‖delta‖ = mean‖h‖ ≈ 10, which
   already collapses coding_density 0.74 → 0.19. The grid *started* there. Nothing in it could
   have worked.
2. **The control was on the floor.** At n=5/class, 0/30 vs 1/30 is not a distinguishable
   difference. Even a working steering method would have produced a null result here.

⇒ ~9 GPU-hours yielded no information in either direction. **Do not evaluate another steering
config with antiSMASH until the baseline regime clears a usable floor.** Seeded generation
already reaches 0.283, so a regime with real dynamic range exists. See decisions.md
"Evaluation must have dynamic range before it is used to compare".

### Steering β-titration (2026-07-29) — β IS A CONFOUNDED SWEEP VARIABLE; redo needed

Layer 16, `steer_generate.py` on v2 `step_1200`, β ∈ {0, 0.1, 0.25, 0.5, 1, 2}, 2 kb ×
`PER_CLASS=3` over **5 classes** (NRPS/PKS/TERPENE/ECTOINE/RIPP) = 15 seqs per β,
coding-density only. Per-sequence recompute with pyrodigal (driver persisted aggregates only).

**The design is paired** (same 5 classes at every β), so the pooled n=15 statistics are the
wrong test. Per class:

| β | ECTOINE | NRPS | PKS | RIPP | TERPENE | pooled |
|---|---|---|---|---|---|---|
| 0 | 0.850 | 0.880 | 0.766 | 0.748 | 0.794 | 0.808 |
| 0.1 | 0.966 | 0.861 | 0.838 | 0.863 | 0.491 | 0.804 |
| 0.25 | 0.677 | 0.649 | 0.796 | 0.931 | 0.822 | 0.775 |
| 0.5 | **0.191** | 0.552 | 0.871 | 0.672 | 0.875 | 0.632 |
| 1 | **0.183** | 0.808 | 0.754 | 0.910 | 0.930 | 0.717 |
| 2 | **0.223** | 0.368 | 0.881 | 0.735 | 0.608 | 0.563 |

Paired t on per-class Δ (df=4): **no β differs significantly from baseline** (|t| ≤ 1.72,
needs 2.78). An earlier pooled z-test flagged β=2 as significant — that was an artifact of
treating 15 sequences from 5 heterogeneous classes as independent draws from one population.

**Root cause — β does not mean the same thing across classes.** β scales the RAW
difference-of-means vector, and ‖v_class‖ at layer 16 spans **17×**:

| class | ‖v‖ | behaviour |
|---|---|---|
| ECTOINE | 17.75 | collapses (0.85 → 0.19 by β=0.5) |
| NRPS | 5.80 | degrades (0.88 → 0.37 by β=2) |
| RIPP | 1.69 | flat / noise |
| PKS | 1.16 | flat / noise |
| TERPENE | 1.05 | flat / noise |

Degradation tracks ‖v‖ almost perfectly. At β=1 ECTOINE receives a ‖delta‖ of 17.75 while
TERPENE receives 1.05 — so a single global β applies a 17×-different physical perturbation
per class. **There is no single β that is both strong enough for TERPENE and survivable for
ECTOINE.** β is comparable in *semantic* units ("one class-mean offset") but not in
*perturbation magnitude*, which is what actually breaks the model. The deprecated `--alpha`
(unit-normalize v, scale by layer ref-norm) was solving exactly this; it was deprecated for
the opposite comparability problem. Neither knob is correct alone.

**Consequence:** this titration cannot answer its own question and should not gate the class
sweep. Re-run parameterized by **effective perturbation magnitude** — sweep ‖delta‖ ∈
{1, 2, 4, 8} with per-class β = ‖delta‖target / ‖v_class‖ — so every class gets a comparable
push. Only then does a coherence ceiling mean anything.

**Also established (holds regardless):** steering never produces garbage — 4-mer diversity
stays near-saturated (227–253 of 256) and GC 0.42–0.47 at every β including 2. High β
degrades *gene structure specifically*, not sequence statistics.

**Sample size:** 3 per class is far too thin; the class sweep needs n≈30–50 per cell.

- Class probe + seed de-confound both COMPLETE 2026-07-27/28
  (results in "What just finished"). **Decisive pair:** Evo2 *represents* class (linear probe 0.911
  vs chance 0.091) but won't act on a label; it DOES act on an exemplar (seeding 0.283, verified
  novel, class comes from the SEED not the tag). ⇒ **steering/decoding problem, not representation.**
  Next: activation steering + guided decoding using the probe head as the class scorer.
- **Quartz long-context** staging remains the OTHER structural fork (blocked on an RT Project
  allocation for the Slurm `-A` account; env/data prep can proceed on the login node);
  `evo2/docs/quartz_setup.md`, `evo2/experiments/quartz/`. Strategic fork now: **CFG / per-class adapters**
  (near-term) vs **long-context** vs **repositioning Evo2 as an evaluator/scorer**.

### (completed 2026-07-07) Fast capability-probe chain (launched 2026-07-06)

- **Fast capability-probe chain (launched 2026-07-06).** Tests the diagnosis fixes at reduced
  cost (L=16384, bs=1 ga=16, ~350 steps fresh-from-base, `lora_dropout=0`), each isolating one
  variable vs a shared P0 control, then quick_eval. Runner:
  `scratchpad/run_probe_chain.sh`; outputs under `/data2/ds85/bgcmodel_runs/probes_20260706/`
  (`probe_summary.tsv`). Probes: **P0** control · **B** +`projections.weight` (adapts the
  frozen Hyena input projection — 28.7M→35.8M trainable, validated) · **C** whole-core data
  (no chunking) · **D** mega-upweighted data. Read on the SENSITIVE proxies
  (class_markers / obligate_fraction / any_domain_rate) since 350 steps is deliberately
  undertrained. ~1 day sequential on the one GPU.
- The main run stays **STOPPED at step_1200** (23 checkpoints retained); resumable via
  `queue_h100_production.sh --resume-from <run-dir>` if a probe warrants scaling up.

### Code/data added for the probes (2026-07-06)
- `finetune_evo2_lora.py`: new `--lora-target-parameters` (peft 0.19 `target_parameters`,
  needs `lora_dropout=0` — peft's ParamWrapper forbids dropout).
- `quick_eval.sh`: `TEMPERATURE`/`TOP_K`/`TOP_P` + `MAX_WINDOWS`/`CHUNK_OVERLAP` env passthroughs.
- `evo2/scripts/build_probe_subsets.py` → `/data2/ds85/bgcmodel_data/probe_subsets/`
  (`subset_c_wholecore` 5,821 recs; `subset_d_megaup` 18,235 recs, 53% mega) + sidecars.

### Superseded plan (continuous-resume 2026-06-24 — did NOT deliver functional gains; kept for history)

- **v2 LoRA training — CONTINUOUS RESUME (started 2026-06-24 14:30 UTC).** tmux `bgc_v2`.
  - **Resumed from `checkpoints/step_400`** (best_val_loss 0.8179) into the same run dir
    `phase1_lora_prod_20260617_095202_L32768`; faithful H1 resume (RNG + data order restored).
  - **Early stopping DISABLED** (`--early-stopping-patience 0`) and **all checkpoints kept**
    (`--keep-last-ckpts 0`) — per the under-training hypothesis from the 2026-06-24 eval.
  - Target **6 epochs = 2,478 steps** (413 steps/epoch); ~52 h/epoch → ~11–13 days from
    step 400. Shape: `L=32768`, `bs=1 ga=128`, splits_core.
  - **LR schedule flattened in the later epochs:** `--lr-min-ratio 0.5` (was 0.1), so the
    cosine floor is 2.5e-5 instead of 5e-6 — later epochs keep a meaningful LR
    (~2.5–3e-5 by epoch 5) instead of decaying to near-zero. Horizon unchanged
    (`total_num_steps=2478`), so the restored schedule stays aligned. Rationale: the first
    pass plateaued in val loss at step 400 *while LR was still ~peak*, so we want later-epoch
    learning to actually have step size. **Watch `val_by_length` for overfitting** (flat/high
    LR can keep train loss dropping while val stalls); optional short cosine cooldown from the
    best long-run checkpoint can produce the final model.
  - Launched via `evo2/scripts/queue_h100_production.sh` (idle-GPU gated + auto-resume ×10). The
    launcher gained `--keep-last-ckpts` and `--lr-min-ratio` passthroughs (2026-06-24).

- **Milestone quick-eval watcher — running (tmux `bgc_eval`).** `evo2/scripts/eval_milestones_watch.sh`
  (new, 2026-06-24) watches `checkpoints/` and runs `quick_eval` on each `step_N` milestone
  (default stride 200 + newest), appending one row per checkpoint to
  `<run-dir>/quick_eval_milestones/eval_track.jsonl` (step → is_bgc / correct_class /
  class_markers / any_domain_rate / coding_density). **Single-GPU-safe / post-hoc:** it is
  idle-gated (proc=0, free≥70 GB, 300 s hold), so it never competes with training — the sweep
  runs once the GPU frees (training end or a long gap). View sorted:
  `jq -s 'sort_by(.step)[] | {step,is_bgc,correct_class,class_markers}' <eval-root>/eval_track.jsonl`.
  - Checkpoint cadence `save-every 50` (≈439 MB each incl. optimizer state); /data2 has 1.4 TB
    free so retaining the full trajectory (~50 ckpts/6 epochs ≈ 22 GB) is a non-issue.

## What just finished

- **CLASS LINEAR-PROBE (2026-07-27) — Evo2 ALREADY represents class; the adapter adds nothing to it.**
  `evo2/scripts/class_probe.py` (raw core nt, NO prefix/tag — else it would read the tag; mean-pooled
  hidden states via forward hooks; 5-fold logistic + shuffled-label control; n=991, 11 classes).
  **base Evo2 balanced_acc 0.911** (chance 0.091, shuffled 0.089); **v2 adapter 0.906** — identical,
  so the adapter did NOT install a class representation. Per-layer: 0.486 (L0) → **0.911 (L16)** →
  0.605 (L28) → 0.414 (L31): **class lives mid-network and FADES toward the output layer** — the
  mechanistic explanation for "represents class, won't generate it." Per-class recall @L16:
  BETALACTONE 0.99, TERPENE 0.97, RIPP 0.95, ECTOINE 0.93, PKS 0.86, NRPS 0.80, HYBRID 0.73.
  ⇒ **DECODING/STEERING problem, not a representation problem** (the good branch of the fork —
  cheap methods viable; does NOT justify Quartz spend on "installing" a representation).
  ⇒ **The probe head IS the fast class scorer** needed for guided decoding / steering (one matmul).
  Artifacts: `/data2/ds85/bgcmodel_runs/class_probe/probe_{base,v2}.json`.

- **SEED DE-CONFOUND (2026-07-28) — the seeding effect is REAL (~0.283, verified novel) but the
  class comes from the EXEMPLAR, not the label.** `evo2/experiments/probes/run_seed_deconfound.sh`,
  7 arms × n=15/class (NRPS/PKS/HYBRID/TERPENE, n=60/arm), seeds identical across arms.
  | arm | is_bgc | correct_class | tracks_seed | leak |
  |---|---|---|---|---|
  | base_notag | 0.0 | 0.0 | 0.0 | 0 |
  | base_tag | 0.183 | 0.133 | 0.133 | 0 |
  | v2_notag | 0.400 | **0.283** | 0.250 | 0 |
  | v2_tag | 0.417 | **0.283** | 0.283 | 0 |
  | v2_tag_trunc | 0.317 | 0.217 | 0.217 | 0 |
  | v2_tag_shuf | 0.0 | 0.0 | 0.0 | 0 |
  | v2_mismatch | 0.467 | **0.067** | **0.317** | 0 |
  All 4 pre-registered criteria PASS: **novelty** (all 420 continuations PASS; max_containment
  median 0.000 / max 0.024 → memorization RULED OUT; correct_novel_only == correct_class);
  **codon-truncation holds** (0.217 — class domains appear de-novo past the boundary, so NOT mere
  gene-continuation); **shuffled seed collapses to 0** (real gene content, not composition);
  **leak = 0** (verified). **Reinterpretation from the mismatch arm:** tag ≠ seed → the continuation
  follows the **SEED** (tracks_seed 0.317) and ignores the **TAG** (0.067 ≈ floor), and
  v2_notag == v2_tag ⇒ **the tag is inert; this is EXEMPLAR-conditioned, not label-conditioned,
  generation.** True effect ~0.283 (pilot's 0.37 was n=10 optimism). Adapter is required
  (base_notag 0.0 → v2_notag 0.283). Artifacts: `/data2/ds85/bgcmodel_runs/seed_deconfound/`.
  *Analysis bug caught+fixed:* the eval report has no `sequence` field, so the first summary's
  `tracks_seed`/`leak` join silently failed (vacuous zeros); recomputed with an index join validated
  against `sequence_length`. Core metrics were unaffected.

- **Finer CFG w-sweep + seeding diagnostic + next-directions workflow (2026-07-22).**
  - **Finer CFG w-sweep {1.5,2,2.5} → CFG CLOSED.** Combined correct_class: w=1 0.067 / 1.5 0.067 /
    2 0.067 / 2.5 0.0 / 3 0.0 / 5 0.0; coding_density 0.90/0.95/0.84/0.59/0.26/0. Flat at the floor
    through the coherent regime (w≤2), then collapses — no pre-collapse lift → no amplifiable class
    signal (untrained-null caveat removed). `bgcmodel_runs/cfg_diagnostic_fine/`.
  - **Seeding diagnostic (n=10/class) → a lift, but LIKELY-INFLATED.** Prompt = [tag] + first 2 kb
    of a real class-X core; score continuation-only. base+seed+no-tag: correct_class **0/30**.
    v2+seed+tag: correct_class NRPS 4/10, PKS 4/10, TERPENE 3/10 (**agg 0.37**), is_bgc 0.40,
    class_markers 29/30. First thing to move megasynthase correct_class off the floor — BUT a
    workflow adversary (verdict **likely-inflated**) flags 3 HIGH confounds: (1) **trivial
    gene-continuation** (model finishes the seeded megasynthase ORF; tight is_bgc↔correct_class
    coupling is the signature), (2) **memorization** (kmer_novelty was skipped — continuation may
    near-copy a train core), (3) **confounded arm** (v2+seed+tag moved adapter AND tag; the decisive
    v2+seed+no-tag cell never run). `bgcmodel_runs/seed_diagnostic/`. **Do NOT bank 0.37 until
    de-confounded; PI artifact deliberately NOT updated with it.**
  - **Next-directions workflow verdict:** DEAD = abstract-label conditioning. UNVERIFIED = exemplar
    seeding. ALIVE/untested (priority): (a) **linear-probe class from Evo2 activations** — does the
    model even REPRESENT class? gates all expensive compute (afternoon, forward-passes only);
    (b) **Arc-style guided decoding** with a fast external class scorer (Arc conditioned Evo2 on
    chromatin accessibility — a non-native handle — this way; never tried here); (c) **per-class
    adapters**; Quartz long-context reserved for "representation absent". Workflow transcript:
    `subagents/workflows/wf_f48679c0-871/`.

- **Simple-class n=15 confirmation + base-Evo2 control + CFG diagnostic (2026-07-21) — every
  prefix-conditioning lever now points to per-class adapters.**
  - **Simple-class conditioning fails generally (n=15/class, n=75).** `v2_step1200`: is_bgc 0.12,
    **correct_class 0.013 (1/75)** — the only hit is a single ectoine; TERPENE 0/15, the n=4
    preview's "terpene 1/4" was small-n noise (same collapse pattern as C 0.33→0.067). So the
    failure is GENERAL, not megasynthase-specific.
  - **Base-Evo2 (no-adapter) control is the clean result:** is_bgc **0.00**, correct_class **0.00**,
    coding_density **0.606**. Vs v2 (is_bgc 0.12, coding_density 0.893) → **the LoRA DOES contribute
    coherence/BGC-likeness (coding_density 0.61→0.89, is_bgc 0→0.12) but NOTHING to CLASS.** The
    adapter learned "make BGC-ish DNA," not "make the requested class."
  - **CFG diagnostic — no amplifiable class signal (untrained-null caveat).** Validation gate PASSED
    (w=1 == non-cached oracle → bookkeeping correct). Sweep on TERPENE/ECTOINE/BETALACTONE (5/class):
    w=1 correct_class 0.067 / coding_density 0.903; **w=3 correct_class 0 / coding 0.257; w=5 all 0 /
    coding 0.** Amplifying class did NOT raise correct_class — it destroyed coherence (OOD collapse),
    with no transient bump. Consistent with "no class signal to amplify." Caveat: v2 wasn't trained
    with class-dropout, so the high-w collapse is partly the expected untrained-null failure mode —
    doesn't fully rule out that a *trained*-null CFG would help. Artifacts:
    `/data2/ds85/bgcmodel_runs/{simple_class_confirm,cfg_diagnostic}/`.
  - **Direction:** per-class adapters (option 3) is now the pragmatic path. Cheap tests still worth
    running first: a finer w-sweep in the coherent regime {1.5,2.0,2.5} (does correct_class lift
    before OOD collapse?) and nucleotide-context seeding.

- **Evo2 native-conditioning-format investigation — option #2 CLOSED as low-leverage (2026-07-21,
  workflow-verified).** Code (installed evo2/vortex) + the Evo2 paper + an adversarial refutation
  pass all converge: Evo2 conditions ONLY on phylogenetic GTDB lineage tags (+ structural
  contig-stitch tokens `@`/`#`) and raw nucleotides — **ZERO native handle for biosynthetic product
  class.** Our GTDB tax tag is already native-aligned; our `|COMPOUND_CLASS:X|` block has no
  pretrained prior (CharLevelTokenizer is pure byte-level → LoRA installs class from scratch through
  the low-rank bottleneck). Arc's own cas9/cas12/cas13 conditioning was a SEPARATE finetune stage
  (prepend-token + finetune) — our exact pattern, confirming no base prior to tap. Full detail +
  citations in [decisions.md](decisions.md) (2026-07-21). **Levers re-ranked: CFG (#1) > per-class
  adapters (#3) ≫ native-format alignment (DROPPED).** Two cheap ideas surfaced & added to the plan:
  (i) class-token position (byte-0 collides with the native "leading `|..|` = lineage" prior → test
  class-AFTER-tax; needs a retrain) and (ii) nucleotide-context seeding (pure inference).
- **CFG diagnostic BUILT (2026-07-21).** `evo2/scripts/cfg_generate.py` + `experiments/probes/
  run_cfg_diagnostic.sh`: two-stream classifier-free guidance on the EXISTING v2 adapter (no
  retrain). Per step, cond=`|COMPOUND_CLASS:X|{tax}` and uncond=`{tax}` (class dropped) each yield
  next-token logits; sample from `logits = uncond + w·(cond − uncond)`, sweeping w∈{1,3,5}. Rising
  correct_class with w ⇒ signal amplifiable ⇒ train-with-class-dropout + CFG; flat ⇒ per-class.
  Correctness gate: w=1 must equal an independent non-cached-recompute greedy token-for-token
  (aborts on mismatch). Auto-runs after the simple-class base_evo2 control (watcher).

- **Rank sweep (r=16/64/128, mega_all, n=15) — capacity is NOT the limiter (2026-07-13).**
  correct_class: r=16 0.067 · r=64 0.067 · r=128 **0.0** — flat/worse; no rank lifts the gate. r=64
  bumped DOMAIN markers (class_markers 0.133→**0.267**, obligate 0.072→0.108) but same modules
  (0.133) and floor correct_class; r=128 collapsed (correct_class 0, modules 0 — over-rank / α–r=2
  over-shrink, rsLoRA regime). Same signature: capacity nudges DOMAINS, never the correct-class
  CLUSTER. **LoRA capacity now fully closed** (coverage via probe B + rank via this sweep). Runner:
  `scratchpad/run_ranksweep.sh`; rows `rank64_n15`/`rank128_n15` in probes_20260706/probe_summary.tsv.

- **Option A (real mega-only whole-core run, L=32768) — AUTO-KILLED at epoch 4; whole-core does NOT
  lift correct_class (2026-07-12).** Milestone n=15: step 120 (~ep2) is_bgc 0.267 / correct_class
  0.133 / modules 0.200 — a flicker; step 240 (~ep4) **is_bgc 0.133 / correct_class 0.0 / modules
  0.0** — everything DECLINED with more training. Self-gate fired (correct_class 0 < 0.15). So
  whole-core mega training at feasible single-GPU L does not convert to functional correct-class
  BGCs; likely overfitting the small whole-core set (80 Mbp — whole-core@L=32k drops 62% of mega nt
  / the long cores). Runner: `scratchpad/run_optA.sh`; run dir `bgcmodel_runs/mega_whole_32k_run`.

- **Concentration probe + n=15 re-eval — C's correct_class win was n=6 NOISE (2026-07-10).** At
  n=15 (PER_CLASS=5): **correct_class = 0.067 (1/15) for ALL of P0, mega_all, and C** — tied at the
  floor; C's earlier 0.33 (2/6) did not survive. What DOES survive is a domain-level gradient
  **C > mega_all > P0**: class_markers 0.33 / 0.13 / 0.07, obligate_frac 0.147 / 0.072 / 0.044,
  module_count 0.27 / 0.13 / 0.07. So whole megasynthase cores (C) and, to a lesser degree,
  mega-only concentration (mega_all > P0) make the model place ~3–5× more class-appropriate
  obligate domains / partial modules — but **none converts into an antiSMASH-valid correct-class
  cluster.** The fast 350-step fresh-from-base probes are exhausted; no config lifts the functional
  gate at reliable n. Next real test = a multi-epoch mega-only whole-core run (mind the
  whole-core∩feasible-L tension — see decisions.md).

- **Gene-aware chunking A/B — REFUTED the "de-chunking is the lever" reading (2026-07-09).**
  Two 200-step arms on the same 17,450 long-mega strict-core dataset (`ga_blind` arithmetic vs
  `ga_geneaware` snap-to-gene): **gene-aware did NOT help — it did worse.** `ga_blind`
  class_markers 0.333 / obligate 0.104 / module 0.167; `ga_geneaware` **all 0** (coding density
  also dropped 0.85→0.76). Neither reached correct_class. So snapping cuts to gene boundaries
  does not recover C's benefit. **Reinterpretation:** C's advantage is seeing the complete
  *cluster* (fits whole), not complete *genes* — points at **longer context (larger L)**, not
  smarter chunking. **Big caveat (see decisions.md):** at the production L=32768, ~79% of mega
  cores ALREADY fit whole and the full run still failed → long-context alone may not be the fix;
  C also confounds mega-only × whole × short. n=6/arm, 200 steps — weak screen. Implementation
  (`--gene-aware-chunking`, gene-bounds, build scripts) is retained and correct; the *hypothesis*
  is what's refuted.

- **Fast capability-probe sweep — DE-CHUNKING is the lever (2026-07-07).** Four 350-step
  fresh-from-base probes at L=16384, ga=16 (runner `scratchpad/run_probe_chain.sh`; results
  `/data2/ds85/bgcmodel_runs/probes_20260706/probe_summary.tsv`), each vs a shared P0 control:
  - **P0** control: correct_class 0, module_count 0.
  - **B** (+`projections.weight`, unfreezes the frozen Hyena long-range input projection):
    **identical to P0 (all 0)** → LoRA capacity/coverage is NOT the bottleneck at this scale;
    **overturns the diagnosis's #1 "leading suspect".**
  - **C** (whole-core / de-chunked megasynthase data): **correct_class 0.33, class_markers 0.50,
    obligate_fraction 0.18, module_count 0.17** — the ONLY probe to lift the functional gates,
    and the first thing in the project to produce correct-class BGCs with ordered modules +
    real obligate domains.
  - **D** (megasynthase upweighted to 53%, still full-length/chunked): **correct_class 0,
    module_count 0** (only a 0.06 obligate flicker) → more mega data does NOT help if chunked.
  - **Verdict:** the lever is the **training signal (whole-core / de-chunking)**, NOT model
    capacity (B flat) and NOT class concentration (D flat). The model must see the **complete
    assembly line under its class label**. Reorders the diagnosis: chunking (Lane 2, rated
    "contributing") is primary; LoRA capacity (Lane 5, "leading suspect") is unsupported.
  - Caveats: n=6/probe, 350 undertrained steps; C confounds whole-core × mega-only × shorter
    (≤16k). Clean isolation = gene-aware chunking (long mega cores chunked-well vs whole).
  - **Follow-up P-tag (2026-07-08):** re-ran D's data with `--no-continuation-prefix` (constant
    `|COMPOUND_CLASS|` on every chunk + `|END|`, no `|CONTINUATION|`) → **near-identical to D**
    (class_markers 0.167 & obligate 0.056 *literally the same*; correct_class/module still 0).
    So the continuation **TAG is NOT the culprit — it's the FRAGMENTATION.** Relabeling is a dead
    end; gene-aware chunking (chunks that contain complete genes) is the fix. Diluted test (only
    ~28% of subset_d chunks); a 100%-chunked long-mega subset would fully confirm.

- **step_1200 functional eval — the decisive negative (2026-07-03).** Pooled n=21 across two
  decoding temps (artifacts under the run dir: `quick_eval_step1200/`,
  `quick_eval_step1200_confirm_baseline/` [temp 1.0, n=9], `quick_eval_step1200_confirm_lowtemp/`
  [temp 0.7, n=6]):
  - `is_bgc` ≈ **3/21 (14%)** — NOT exactly 0 (the first n=6 caught 0/6 = small-sample noise);
    `correct_class` = **0/21**; `module_count` = **0/21**; obligate core domains
    (PF00501 NRPS-A / PF00668 C / PKS KS/AT) ≈ absent.
  - **Smoking gun:** every antiSMASH-positive hit was a SIMPLE class — requested NRPS→**ectoine**,
    requested HYBRID→**terpene** — never the conditioned megasynthase. Class-conditioning fails
    functionally: the model writes generic gene-dense DNA that occasionally forms an easy cluster
    but never builds the requested class's core assembly-line machinery.
  - **Robust:** more n didn't lift `correct_class`; conservative decoding (temp 0.7) gave cleaner
    coding density (0.98) but STILL 0 modules / 0 correct_class → not a sampling artifact. With flat
    val loss and no step-400→1200 gain, the gap is **structural, not under-training** — this
    challenges the "surface results = low training, not LoRA capacity" note in [decisions.md](decisions.md).
  - **Decision: STOP the 6-epoch run; diagnose the root cause** (workflow running).
  - Training was paused cleanly for this (wrapper SIGTERM'd, trainer SIGINT'd; step_1200 intact).

- **v2 LoRA training (first pass) — COMPLETE** (finished 2026-06-19 12:09 UTC; ~50 h wall).
  - Run dir: `/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768`
  - Config: Evo2 7B + LoRA, `L=32768`, `--batch-size 1 --grad-accum 128`, bf16, DeepSpeed.
  - **Early stop at step 400** (epoch ~0.97/6): no val improvement for 4 validations
    (patience=4, min-delta 0.001). Train loss 0.98 → 0.71; **best val_loss 0.8179** (ppl 2.27).
  - Checkpoints: `checkpoints/best/adapter`, `final_adapter/` (= step_400_final), plus
    step_{200,250,300,350,400}. GPU now idle; no tmux session.
  - Data: `splits_core` (train 47,524 / val 8,048 / test 18,871; 22 classes; strict
    antiSMASH cores; native GTDB tags; MiBIG held out).

- **Post-training eval on the v2 `best` checkpoint — done 2026-06-24** (GPU was free).
  Artifacts under the run dir: `quick_eval_best/`, `conditioning_diag_stoch_best/`,
  `post_train_eval.log`.
  - **quick_eval (n=3, 32k, antiSMASH gates):** `is_bgc=0.0`, `correct_class=0.0`,
    `class_markers=0.0`, `obligate_fraction=0.0`, `any_domain_rate=0.333`,
    `coding_density=0.913`, `module_count=0`. → produces coding-dense DNA with occasional
    Pfam domains, but **antiSMASH does not call any of the 3 as a BGC** and none are
    correct-class. (Tiny n; directional, not definitive.)
  - **conditioning diag (stochastic top_k=4, 24 seqs @16k):** composition 5-mer
    within 0.229 / cross 0.247 / **ratio 1.08**; domain-set ratio 1.02; own-obligate by
    class NRPS 0.056, PKS 0.0, HYBRID 0.021, TERPENE 0.0; any-domain 0.67–0.83;
    GC 0.62–0.71 (healthy, not degenerate). Script **VERDICT: "CONDITIONING WORKS"**
    (class-differentiated + NRPS shows some of its obligate domains).
  - **Honest read:** a real improvement over the 2026-06-04 pilot (which scored
    "CONDITIONING DEAD", ratio ≈1.0). v2 shows a **measurable, class-appropriate but WEAK**
    conditioning effect — yet it is **not yet producing antiSMASH-recognizable,
    correctly-classified BGCs** (functional gates at 0). The class tag is being read; the
    model is not yet building complete class machinery/modules.

## What is done and validated

- **Dataset v2 (`splits_core`) built & leakage-clean** — strict cores from re-acquired
  antiSMASH-DB GBKs, native lowercase GTDB tags, genome-disjoint + exact + MMseqs2-dedup,
  MiBIG excluded. Pre-MiBIG backup at `splits_core_premibig/`.
- **Eval suite rewritten** to named CHECKS → QUESTIONS (`src/bgc_pipeline/evaluation.py`).
  Gene caller is **pyrodigal** everywhere; synthesis/perplexity/BiG-SCAPE retired; E. coli
  expressibility no longer gates. All `tests/run_all.py` pass (18 files).
- **antiSMASH recalibrated** to the `is_bgc`/`correct_class` gate: **0.97 / 0.97** on 237
  real held-out cores (was ~0.15). Full product→class map in
  `config/compound_class_map.yaml` (via `build_class_map.py`). Calibration data:
  `/data2/ds85/bgcmodel_data/as_calib.jsonl`.
- **`class_markers` recalibrated** (data-driven, ANY-of) — ≈0.87 on real cores.
- **Eval consumers migrated** (driver, quick_eval, run_eval, diagnose scripts, standalone
  evaluate_bgc, tests). Adversarial review workflow run; its 1 high bug (antiSMASH
  parse-error proxy) + comment/coverage findings fixed.
- **Documentation consolidated** into this `README.md` + `docs/project_memory/`; detailed
  runbooks/plans/audits archived under `docs/archive/`.

## Next actions (in order) — updated 2026-07-21  ⚠️ **SUPERSEDED — see "NEXT ACTIONS — REWRITTEN 2026-08-12" above; `docs/conditioning_next_steps.md` is superseded at the top level too.**
Retained for history. Its ranking put activation steering first; steering was then run to
completion and closed on 2026-08-10 (Phases 0–6). Do not work from this list.

1. **[DONE 2026-07-21] Simple-class n=15 confirmation + base control + CFG.** Conditioning fails
   generally (v2 correct_class 0.013, base 0.0); LoRA adds coherence not class (coding_density
   0.61→0.89); CFG found no amplifiable class signal (correct_class flat-to-0, coherence collapses
   with w). See "What just finished".
2. **[DONE 2026-07-22] Finer CFG w-sweep + seeding** → CFG closed (no amplifiable signal); seeding
   agg correct_class 0.37 but **likely-inflated** (see "What just finished").
3. **[NEXT — follows directly from the probe] Steer/decode with the probe head as the class scorer.**
   The probe proves class is decodable at L16 (0.911) but fades by L31 — so make generation USE it:
   (a) **Activation steering** — add the layer-16 logistic weight vector for class X (scaled by α) to
   hidden states during generation; sweep α; watch correct_class vs coding_density (same OOD-collapse
   caution as CFG). No retraining. (b) **Guided decoding** (Arc's recipe) — sample K candidate chunks,
   score each with the probe head (one matmul, cheap enough for the decode loop), keep top-K, iterate;
   antiSMASH only as the final offline selector. (c) Compose with the exemplar seed, which
   independently works. Reuse `evo2/scripts/cfg_generate.py`'s two-stream plumbing + the probe
   artifacts in `/data2/ds85/bgcmodel_runs/class_probe/`.
4. **Harden exemplar-seeding into the actual method** (retrieval-conditioned generation): frame as
   "extend/diversify a known cluster" (honest — class comes from the exemplar), sweep seed length,
   retrieve exemplars by taxon/embedding, extend to more classes.
5. **[DE-PRIORITIZED by the probe] Per-class adapters / Quartz long-context.** The probe says the
   representation is NOT missing, so "install a class representation" is the wrong problem — do not
   spend Quartz on it until steering/decoding is exhausted.

**[DONE 2026-07-27/28] De-confound seeding + class linear-probe** — results above. Original plan:
   (a) **Seed de-confound factorial.** Edit `evo2/scripts/seed_generate.py` (add `--seed-source
   {bgc-core,housekeeping}` + `--no-boundary-orf` codon-truncation; eval driver emits per-hit CDS
   coords). Arms: 2×2 {base,v2}×{tag,no-tag} with seeds held fixed; **novelty gate ON** (kmer_novelty
   + MMseqs2 vs train); **housekeeping-seed negative**; **n≥15/class** across eligible classes. REAL
   only if v2+seed+no-tag ≈ v2+seed+tag AND survives novel-only AND housekeeping collapses AND the
   codon-truncated seed keeps the lift with the class domain appearing de-novo past the boundary.
   (b) **Class linear-probe** (new script) — logistic probe of `compound_class` from Evo2 hidden
   states of real cores vs a shuffled-label control. Separable ⇒ decoding/steering problem (cheap
   fixes viable); not ⇒ must INSTALL a representation (gates Quartz spend). Parallel-safe.
4. **[THEN, contingent on #3] Arc-style guided decoding** (sample → score with a fast class
   discriminator → top-K → iterate; antiSMASH only as final selector) · **per-class adapters**
   (class = which adapter you load) · **Quartz long-context** ONLY if the probe proves class is not
   represented (`evo2/docs/quartz_setup.md`, blocked on RT Project allocation).

**DROPPED / CLOSED levers:** native-format *alignment* as a class lever (option #2 — no native class
prior exists, 2026-07-21); LoRA capacity (probe B + rank sweep); chunking vs de-chunking; whole-core
at scale (Option A declined with training); class balancing. See "What just finished".

<details — superseded pre-2026-07-21 action list, kept for history:>

0. **[2026-07-07] Build gene-aware chunking / whole-core training — the validated lever.** The
   probe sweep showed de-chunking (whole cores) is what lifts the functional gates, not LoRA
   capacity (B) or class concentration (D). Plan:
   - **(i)** Persist per-gene coordinates: `build_core_records.py` already parses per-CDS
     coords (`cds_coords`) but only stores the count — add a `core_gene_bounds` field (rel
     offsets within the stored sequence). GBKs on disk at `/data2/ds85/asdb5_gbks`. Either
     re-run the core→split→dedup pipeline or emit an `accession→bounds` sidecar (non-destructive).
   - **(ii)** Snap chunk cuts to gene gaps in `build_nt_chunk_spans` (+ the duplicate in
     `build_chunk_index.py`) so no cut falls inside a gene; fall back to arithmetic when a single
     gene exceeds the budget (those need larger `L` / 7B long context).
   - **(iii)** Probe it: long-mega chunked-well vs whole-core (isolate de-chunking from the
     mega-only/short confound), then a real (not 350-step) run on 7B for publishable numbers.
   - Optional cleanup to test alongside: constant class tag every chunk + explicit START/END
     tokens (vs the current `|CONTINUATION|`) — see the 2026-07-07 chunking discussion.
   The earlier diagnosis-driven experiment list below is now **superseded by these results**.

1. **[SUPERSEDED — see item 0] Act on the diagnosis of the class-conditioning failure.**
   - **(a) [cheap control] Re-eval step_1200 with chained windows** (`generate_bgc.py
     --max-windows 3-4 --chunk-overlap 2048`) to remove the 32k-truncation confound. Expected:
     still 0 correct_class (deficit is upstream) — decisive and cheap.
   - **(b) [high-value ablation] LoRA coverage + rank.** Add the Hyena `projections` (TELinear)
     to LoRA targets (relax the isinstance check ~`finetune_evo2_lora.py:1200`; verify PEFT can
     wrap TELinear), bump rank 16→32/64 with a `rank_pattern` favouring mixer/attention over
     MLPs, optionally unfreeze prefix/class tokens (`trainable_token_indices`). Few-hundred-step
     controlled ablation watching `module_count`/`correct_class`.
   - **(c) [chunking ablation]** fine-tune on only whole-core-in-one-window NRPS/PKS/HYBRID; if
     modules emerge, chunking is the dominant lever. Structural fix: gene/module-aware splitting
     or `L=65536` for megasynthase classes; and/or label interior windows `|COMPOUND_CLASS|`.
   - **(d) [amplifier]** class-balanced sampling.
   The "train longer / raise LR" item below is **superseded**.

1. **Decide what the weak-conditioning + zero-functional-gate result means.** The class tag
   is read (ratio 1.08, class-appropriate domains) but no antiSMASH-valid BGC is produced.
   Likely levers, roughly in order of expected payoff:
   - **Train longer / harder.** Early stop fired at epoch ~0.97 (well under 1 full epoch);
     the prefix-masked loss was still drifting down. Consider loosening early stopping
     (higher patience / smaller min-delta) or raising LR, then re-eval. The conditioning
     interface may simply be under-trained (consistent with the LoRA "low-training ⇒
     surface results" note in [decisions.md](decisions.md)).
   - **Larger / less-tiny eval.** Re-run quick_eval with more sequences per class
     (`PER_CLASS`>1) so `is_bgc`/`correct_class` aren't estimated from n=3.
   - Only after the above: reconsider the **per-class-adapters vs one-conditional-model**
     fork (see [decisions.md](decisions.md) "Open architectural fork").
2. **If/when v2 conditions well** (functional gates lift off 0): start **Phase-2** — build a
   MiBIG core + compound-conditioned dataset and fine-tune for compound-level (named-product)
   generation.

_Done 2026-06-24: quick_eval + stochastic conditioning diagnostic on the v2 `best`
checkpoint (results above). Earlier "step 50" eval action is moot — the run already
completed at step 400._

## Known not-yet-done / deferred

- `protein_homology` (MMseqs2) DB is **not wired** for full-val — diagnostic-only; skipped
  in quick_eval. Wire a UniRef50 DB when running a full milestone eval.
- Generation-based offline eval depends on `generate_bgc.py` (built; sequential path).
- All work through 2026-06-17 is **committed and pushed to `main`** (commit `d337184`); the
  working branch `claude/laughing-hamilton-fdacc5` is synced to `main`. (Commit only when
  explicitly asked.)

## Pointers

- Eval suite + how to run: `README.md` → Evaluation; archived deep version
  `docs/archive/EVAL_RUNBOOK.md` and `docs/archive/REDESIGN_PLAN.md`.
- Training runbook: `docs/archive/gputee/FINETUNE_GUIDE.md`.
- Evo2 + LoRA + Hyena-block architecture explainer (why the long-range pathway is un-trained;
  what Probe B fixes): `evo2/docs/evo2_lora_and_hyena.md`.
- Auto-memory (cross-session, outside the repo):
  `~/.claude/projects/-home-ds85-projects-BCGModelling/memory/`.
