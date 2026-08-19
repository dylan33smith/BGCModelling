# plan.md — the dashboard

**Read at session start. This file holds only the active phase.** Completed interventions keep a
one-row summary in the Phase Ledger for the rest of the phase; their full write-up goes to
`memory.md` at completion. At phase close the ledger collapses to one line and the board resets.

**Last updated:** 2026-08-14

---

## Current State

Phase 3/4, target **RIPP**, substrate **Evo2 1B**. **Legs 1 and 2 closed positive; leg 3 and the
WIDE hypothesis both closed negative — all four on powered, pre-registered tests.**

**Best arm to date: STRICT-full regenerated at 8 kb — antiSMASH-corrected 0.116** against a
real-core ceiling of 0.760 and a base-model floor of 0.000. Seeding at L\*=8 nt is what buys it;
the class comes entirely from the adapter (shuffling the seed changes nothing, p=0.66).

**[P4-WIDE] is refuted with a mechanism.** Widening training spans to include
`biosynthetic-additional` genes made the model **significantly worse** (W-1 vs the size- and
cluster-matched W-2: Holm p=4.1e-04 at 2.2 kb, p=3.2e-05 at 8 kb), while the training-set size drop
itself cost nothing (p=0.79). **Cause measured: dilution.** The biosynthetic fraction of a training
span falls 0.683 → 0.477, i.e. **1.43× less biosynthetic signal per token** — and *not* from
intergenic space (coding density barely moves, 0.976 → 0.938) but from **additional
non-biosynthetic genes**. The WIDE adapter's 8 kb Pfam hits were **0/8 antiSMASH-confirmed**
(P≈0.004–0.009 under the other arms' rates): it emits isolated biosynthetic-looking genes with no
cluster context — exactly what dilution predicts.

**The number that has never moved:** `n_class_domains ≥ 2` is **0/188 in all five arms** — 940
sequences, five adapters, three windows, zero generations with two distinct RIPP markers. Real cores
14/68. Every intervention aimed at it has failed, and the model writes near-natural gene *counts*
while only ever one gene is biosynthetic.

**Next:** add span width **without** losing biosynthetic density — domain-weighted loss on WIDE
spans ([P5-WEIGHTED]), with a mandatory manipulation check since Phase-2's weighted arm never landed.

---

## ⚠️ OPEN STRATEGIC QUESTION — decide before committing to Phase 4

**The model was never shown clusters, so it cannot be failing to generate them.**

`n_class_domains >= 2` is **0 or near-0 in every arm run to date** — A0, base, general adapter, and
every seed length — against **29%** for real cores. That number has not moved once all day.

The cause is in the training data, not the model: **48.8% of RIPP training records are a single
biosynthetic gene** (median 2, mean 1.86), and the strict core is a **median 9.1% of the antiSMASH
region** (1,931 nt of 21,279). `STRICT_KINDS = {"biosynthetic"}` excludes transport, regulatory and
`biosynthetic-additional` — for RiPPs that drops the exporter/protease. See `memory.md` 2026-08-17.

So "generate a BGC" is not what any current arm tests. The honest description of every rate on this
board is **"produces a biosynthetic enzyme gene of the right class"**, and the 0.440 ceiling is
"a real trimmed core", not "a real BGC".

**The candidate intervention: a `WIDE_KINDS` training arm** (`{"biosynthetic",
"biosynthetic-additional"}`), giving 2–3 gene neighbourhoods instead of one gene.
- **For:** it is the only item on the board that targets the gap every metric keeps reporting. The
  window pressure that justified strict trimming has largely expired — Phase 3 trains at L=8192 on
  median 1.9 kb cores, an order of magnitude under the limit.
- **Against:** it needs a new training run, and whole-core-only training was tried before
  (`mega_whole_32k`) and **failed** — "starves the data". `WIDE_KINDS` is a middle point that has
  never been tried, not a repeat of that.
- **Do NOT redefine the endpoint mid-phase** (Standing Constraint 4). This is a Phase-4 scope
  decision or a deliberate re-open, not a quiet change.

⇒ **Stage 2 measures the same ceiling either way, so it is safe to run first — but decide this
before designing anything after it.**

## Reporting contract — every Phase-3 arm

**All Phase-3 interventions report THE PHASE-3 REPORTING SET in full** (`terms.md`), emitted by
`scripts/novelty_battery.py`. Never hand-assemble a subset — that is how A0 came to be quoted as a
bare hit rate for three days while `n_class_domains` sat at exactly 1.000.

Two arms are comparable **only** if their `scoring` stamps agree on all five axes. Each has already
caused a real error here:

| axis | required | what went wrong |
|---|---|---|
| Pfam subset | `OBLIGATE_DOMAINS[RIPP]`, 8 accessions | global set inverted A0 (08-14) |
| scoring window | **2,000 nt** | `_w8000` read 0.087 vs `_w2000` 0.027 on one arm |
| substrate | `evo2_1b_base` | unset env silently used the 7B (08-17), 150 gens discarded |
| generation path | batched (all current arms) | left-pad failed an equivalence gate — [P3-B8] |
| regime | de novo **or** seeded, never pooled | 0.012 vs 0.367 detection |

Generation *length* may differ safely — A0 generated 8,000 nt and the controls 4,000 — **because
the scored span is a fixed 2,000-nt prefix** and an autoregressive model writes the same first
2,000 tokens regardless of the total requested. That is what the fixed window is for.

Enforced by `tests/test_docs_contract.py`: arms missing the set FAIL; arms with divergent scoring
configs WARN.

## Phase Ledger — Phase 3

Endpoint names are `terms.md` identifiers. `memory.md` column = date anchor to grep.

| ID | Intervention | Endpoint | n | Result | Verdict | memory |
|---|---|---|---|---|---|---|
| P3-A0 | RIPP-only LoRA, **de novo** | `best_bio_bits` > 0 @ `OBLIGATE_DOMAINS[RIPP]`, 2000 nt | 150 | **4/150 = 0.027** | ✅ **SIGNIFICANT, p=0.0054** vs 0/400 | 2026-08-17 |
| P3-C1 | base 1B control, de novo | ″ | **200** | 0/200 = 0.000 | floor | 2026-08-17 |
| P3-C2 | general all-class adapter, de novo | ″ | **200** | 0/200 = 0.000 | floor (0.067 generic — other classes' domains) | 2026-08-17 |
| P3-CEIL | real RIPP cores | ″ | 50 | 22/50 = 0.440 | ceiling | 2026-08-14 |
| P3-NOV | novelty guard on A0 | `containment` | 150 | max 0.003; AAI med 0.000 / max 0.470; 150/150 distinct | ✅ pass | 2026-08-14 |
| P3-S1 | **seed-length sweep, LoRA** | ″ | 50/cell | L4 0.140 · **L8 0.160** · L20 0.100 · L100 0.100 · L500 0.240 | ✅ all beat de novo | 2026-08-17 |
| P3-S1c | **seed sweep, BASE 1B control** | ″ | 50/cell | **0/50 at every length incl. 500 nt** | ★ the seed alone does nothing | 2026-08-17 |
| P3-S1n | protein-novelty guard on the sweep | max AAI | 50/cell | 0.617 · 0.620 · 0.801 · 0.793 · **0.914** | ⚠️ memorisation at L=500 | 2026-08-17 |
| **P3-S2-1** | **RIPP LoRA + real seed @ L=8** | ″ | 188 | **33/188 = 0.176** [0.124,0.238] | ✅ **vs general p=2.5e-11** | 2026-08-18 |
| **P3-S2-2** | **general adapter + real seed** | ″ | 188 | **0/188 = 0.000** (0.181 *generic*) | ★ lift is class-specific | 2026-08-18 |
| P3-S2-3 | base 1B + real seed | ″ | 188 | 0/188 on RIPP **and** generic | floor | 2026-08-18 |
| **P3-S2-4** | **LoRA + SHUFFLED seed** | ″ | 188 | 35/188 = 0.186 | ★ **p=0.66 — seed content is irrelevant** | 2026-08-18 |
| ~~P3-S2-5~~ | ~~LoRA + mismatch tag~~ | ″ | 188 | ⚠️ **treatment did not land** — flag is a no-op with one class | UNINFORMATIVE | 2026-08-18 |
| **P3-AS** | **antiSMASH on S2-1 on-class** | `is_bgc`* / `correct_class`* | 33 | **0.485 / 0.485** | ★ gold standard, all detections on-class | 2026-08-18 |
| P3-AS-c | antiSMASH on real held-out cores | ″ | 50 | 0.760 / 0.740 | ceiling | 2026-08-18 |
| P3-AS-o | antiSMASH on S2-1 **off**-class | ″ | 50 | 0.040 / 0.040 | the Pfam gate hides no clusters | 2026-08-18 |
| P3-AAI | AAI among on-class vs real cores | `protein_aai`* | 33 | **0.496** vs real **0.641** | ✅ homologous but more divergent than nature | 2026-08-18 |
| **P3-WIN** | **window sweep, A0 de novo, fixed set** | `n_class_domains` | 85 | **1.00 at 2k/4k/8k** (real 1.60→1.69, bio 1.69→**2.67**) | ★ gap is NOT a window artefact | 2026-08-18 |
| **P3-PROBE** | **class probe within-positives** | probe P(RIPP) vs antiSMASH | 68 | **AUROC 0.337** (anti-correlated; saturated at ~0.997) | ⛔ **leg 3 has no instrument** | 2026-08-18 |
| P4-WIDE-dn | WIDE adapter, **de novo** 8 kb | `best_bio_bits` @ RIPP | 79 | 0/79 · 0/79 · 1/79 (n.s. vs A0) | ⚠️ **UNINFORMATIVE — not powered** | 2026-08-18 |
| **P4-W1** | **WIDE adapter, seeded L=8** | antiSMASH-corrected | 188 | **0.027** (2.2k) · **0.000** (8k) | ⛔ **WORSE than matched control** | 2026-08-19 |
| **P4-W2** | **STRICT size+cluster matched** | ″ | 188 | **0.043** (2.2k) · **0.085** (8k) | control — isolates span width | 2026-08-19 |
| P4-SF | STRICT-full regenerated @8 kb | ″ | 188 | **0.116** | best arm; gen length n.s. (p=0.50) | 2026-08-19 |
| **P4-DILUTE** | **biosynthetic fraction of training span** | paired, n=250 | 250 | STRICT **0.683** vs WIDE **0.477** | ★ **1.43× less signal/token — the cause** | 2026-08-19 |

**Provenance for the block above:**
`phase3_RIPP/adapter_run` (7,250 whole records, 3 ep / 1,350 steps, `loss_ce` 0.790→0.410) ·
`A0_8k.jsonl` + `phase3_ripp/pilot_*.jsonl` · scoring `OBLIGATE_DOMAINS[RIPP]` · window 2,000 nt ·
substrate Evo2 1B (TE 1.13.0 verified).

**Standing reading of the ledger:** A0 is **significant** (p=0.0054 vs 0/400 pooled controls,
pre-registered §8.4). But it reaches only ~6% of the 0.440 ceiling, and all four hits carry a
**single** RIPP domain where real cores carry 1.45 on average. The defensible claim is "a
class-specific LoRA puts RIPP-associated machinery into de novo output at a low but real rate" —
**not** "it generates RiPP clusters". One domain is not a cluster.

---

## In Progress

*(none — A0 complete, next intervention not yet started)*

### Intervention template — every entry MUST fill all eight fields

```
### [ID] <name>
Hypothesis:          what would have to be true, stated so it can be wrong.
Reasoning:           why this, why now, what it settles.
Technical:           what actually runs — script, substrate, data, config.
Primary endpoint:    a terms.md identifier + scoring config + window. One endpoint.
n and power:         n, from what pilot estimate, powered to detect what effect.
MANIPULATION CHECK:  the measurement proving the intervention LANDED.
                     Read BEFORE the endpoint. A null without this is uninformative.
Kill criterion:      the pre-registered result that closes this line.
Novelty guard:       containment reported alongside, always.
```

> The manipulation check is not boilerplate. Phase 2's domain-weighted arm consumed a full
> training run and returned an uninterpretable null **because the treatment never landed**, and
> that was only discovered afterwards. Read the check first, every time.

---

## Proposed interventions after [P4-WIDE] — ordered

Every one of these follows from a *measured* cause, not a guess. `n_class_domains ≥ 2` = 0/188 in
all five arms is the target; the STRICT-full 8 kb arm (corrected **0.116**) is the baseline to beat.

### [P5-WEIGHTED] WIDE spans + domain-weighted loss — DEMOTED
Still the right response to the measured dilution, but **no longer first**: the precursor finding
says the missing thing is a *component*, not signal density. Also note the user's hypothesis that
WIDE was secretly producing accessory machinery was **tested and refuted** (2/60 vs 8/60 carrying
the additional-domain vocabulary), so WIDE has no hidden upside to recover.
**Rationale (measured):** WIDE failed because biosynthetic density fell 0.683 → 0.477. Domain
weighting up-weights loss on biosynthetic spans, so the model can see wider context *without* the
signal being diluted. This is the only intervention that directly addresses the measured cause.
- **Machinery exists.** `finetune_evo2_lora.py` takes a `*.domain_spans.jsonl` sidecar;
  `scripts/build_domain_spans.py --data-dir splits_class_wide/RIPP --hmm-subset ripp_only.hmm`
  builds it. Never yet run on a WIDE split.
- ⚠️ **MANDATORY MANIPULATION CHECK.** Phase-2's weighted arm produced an uninterpretable null
  because **the treatment never landed**. Before reading any endpoint, verify the weights reach the
  optimiser — log the per-token weight distribution and confirm biosynthetic spans carry the
  intended multiple. A flat distribution means the run is void, not negative.
- **Arms:** WIDE+weighted vs WIDE unweighted (already have) vs STRICT-matched (already have). Only
  one new training run.

### [P5-PRECURSOR] ◀◀ NOW THE TOP PRIORITY — MEASURED, not speculative
**Evidence (2026-08-19).** Among single-marker generations, what separates antiSMASH-CONFIRMED from
REJECTED is **not** domain content (3.57 vs 3.50 distinct Pfam domains — identical). It is
**an extra gene, specifically a SHORT one of 20–80 aa: 0.43 in confirmed vs 0.00 in rejected.**
That is the **RiPP precursor peptide — the gene encoding the actual product.**

**Our instruments are structurally blind to it:**
- none of the 8 `OBLIGATE_DOMAINS[RIPP]` markers is a precursor (all modifying enzymes/binders);
- `find_orfs` defaults to **`min_aa=50`**, and Prodigal's own floor is ~30 aa.

**Actions, in order:**
1. **Re-score every existing arm with `min_aa=20`** and report `n_short_orfs` (20–80 aa) as a new
   metric. Costs nothing — no generation, no training. It may show the models are already producing
   precursor-sized ORFs we have been discarding.
2. **Add precursor detection**: a RiPP-precursor HMM set, or parse antiSMASH's own precursor calls
   out of the runs we have already done.
3. Only then consider training changes.
⚠️ Do not change the pre-registered PRIMARY mid-phase (Constraint 4). Add these as reported metrics.

### [P5-COMPOSE] Component-wise adapters, generated in sequence ◀ follows directly from the above
**User's proposal, and the precursor finding supports it.** A RiPP is not one thing — it is
precursor + modifying enzyme(s) + protease/transporter. Train a small adapter per component and
generate compositionally, each step seeded on the previous output:
`precursor → (precursor as seed) enzyme → (precursor+enzyme as seed) transporter`.

**Why it fits what we now know:**
- The binding constraint is a **missing component**, not weak content — so target components.
- We already know **seeding works and is class-specific** (0.176 vs 0.000 for the general adapter,
  p=2.5e-11), and that the model *continues* what it is given. Compositional seeding is that
  mechanism used deliberately.
- Component splits are buildable from `gene_kind` + our own ORF calls; no new data collection.

**Risks to design against:**
- ⚠️ **Error compounding** — a bad precursor poisons every later step. Measure each stage against
  its own control, not only the end product.
- ⚠️ **Seed length**: L\*=8 nt was chosen because longer seeds cause reconstruction (12/12 source-
  domain match at 500 nt). Compositional seeding *deliberately* uses long seeds, so the novelty
  gates must be read per stage, and `--no-boundary-orf` cannot protect a deliberate hand-off.
- Needs a joint-vs-monolithic control: does composition beat one adapter generating the whole thing?


**Rationale:** none of the 8 RIPP markers is a **precursor peptide** — they are all modifying
enzymes. And Prodigal calls **0 ORFs under 30 aa**, so typical short RiPP precursors are invisible
to our caller *and* absent from our marker set. We may be asking for clusters while measuring only
half of one.
- Add precursor detection (a dedicated RiPP-precursor HMM set, or antiSMASH's own precursor calls)
  as a **reported metric first** — do not make it an endpoint mid-phase (Constraint 4).
- If generations do contain precursors we cannot see, the "one gene" story is partly an artefact of
  the instrument, which would be the most important finding available.

### [P5-RL] Rejection-sampling / DPO on verified positives
**Rationale:** literature-supported ([P4-RL-0]) and now much better specified by today's results.
- **Reward on antiSMASH confirmation, not the Pfam gate** — the gate inflates 1.8× and, in the WIDE
  arm, confirmed 0/8. Rejection sampling is documented to hack exactly this kind of proxy.
- **Weight positives by `n_class_domains`** — the metric that has never moved.
- ⛔ **Do NOT rank candidates by any continuous score.** Measured: ladder metrics reach 0.575 and the
  class probe 0.337 for within-positives discrimination. Rewards must be built from **verified gate
  passes**, not scores.
- Cost: antiSMASH on every sampled batch. Feasible — 615 calls ran in 4 minutes.

### [P5-SUBSTRATE] Re-test the best arm on a longer-context model
**Rationale:** `evo2_1b_base` caps at **8,192** tokens, and real cores need 8 kb to show 2.67
domains. The 1B may be structurally unable to hold a cluster. GenomeOcean-4B (32,768 ctx, leakage
gate already passed) or the 7B would test this.
- ⚠️ Changes the substrate — a deliberate Phase-5 scope decision, not a mid-phase switch
  (Constraint 3). Run **only the single best arm** (STRICT-full seeded @8 kb), not the whole matrix.

### [P5-CEILING] Ask whether the target is achievable at all
**Rationale:** before spending more on interventions, establish what *any* model can do here. Take
real RIPP cores, corrupt them progressively (mask k% of the biosynthetic genes), and measure the
recovery curve. That gives an achievability ceiling for "produce a second marker" and tells us
whether 0/188 reflects a hard problem or a fixable one. Cheap, CPU-only, no training.

## Backlog — Phase 3

**The phase has three legs.** Leg 1 is done and negative-but-directional; legs 2 and 3 are untried.

| leg | status |
|---|---|
| 1. class-specific LoRA fine-tuning | ✅ **DONE — 0.027 vs 0/400, p=0.0054 significant** |
| 4. WIDE_KINDS span width | ⛔ **REFUTED 2026-08-19 — significantly worse; cause is dilution** |
| 2. class-specific seeding | ✅ **DONE — 0.176 at L=8, class-specific (p=2.5e-11), seed content irrelevant** |
| 3. inference pruning | ⬜ not started — two distinct mechanisms: [P3-B2a] prunes *during* generation, [P3-B2b] filters *after* |

Ordered. Top item is next.

### [P3-B0] ✅ DONE 2026-08-17 — scorer made class-specific
- `scripts/novelty_battery.py` takes `--cls` and **ignores it** for `on_class`; it scores against
  the global 91-model biosynthetic set. Every saved score file holds the generic number under the
  same key. See `bugs.md` 2026-08-17.
- **Nothing else in Phase 3 can run until this is fixed** — leg 2 would be scored the same way.
- Fix: subset the HMM to `OBLIGATE_DOMAINS[cls]` inside the scorer; emit the scoring set into the
  output filename (`_w2000_RIPP.json`), not just the window.
- **Acceptance test:** re-scoring A0 must return **4/150**, base 1B **0/50**, general adapter
  **0/50**. Reference implementation and expected output:
  `phase3_RIPP/A0_8k_w2000_RIPPSPECIFIC.json` (re-derived 2026-08-17).
- Then re-derive and **persist** the real-core ceiling — the recorded 0.440 is not reproducible
  from disk and an independent 50-record draw gave 0.62, so the sample used was never saved.

### [P3-B1] The SEED LADDER — leg 2 · ✅ **BOTH STAGES DONE 2026-08-18**
**The objection this exists to answer:** *seed a real BGC → get a BGC* is unimpressive; the model
could be finishing a cluster that already exists. The pre-registered arms (§7) are a ladder from
instance-copying toward generation from a class representation.

| rung | arm | seed | status |
|---|---|---|---|
| 0 | **A0** | none | ✅ **DONE, SIGNIFICANT** — p=0.0054. Partly defuses the objection already |
| 1 | **A1** | one real exemplar prefix | ceiling + novelty floor; the historical 0.283 mode |
| 2 | **A3** | consensus/centroid, per-sample bootstrap over exemplar subsets | ⚠️ see below |
| 3 | **A2** | **mosaic** — fragments from k different clusters, new k-subset per sample | real parts, novel combination; no new machinery |
| 4 | — | sample from a learned latent prior | build required, deferred |
| 5 | — | label only | CLOSED in Phase 1 |

**Seed length is a first-class variable, swept at 4 / 8 / 20 / 100 nt** — the phage paper found
4–8 nt optimal and longer seeds caused memorisation. Our historical seeds were ~500 nt, ~100×
their optimum.

#### ⚠️ MEASURED 2026-08-17: a consensus seed is not meaningful for RIPP
Position-wise base entropy over the first 20 nt of 8,129 RIPP training cores (2.00 bits = no
conservation whatsoever):

```
1.61 0.81 1.05 1.94 1.99 1.97 1.97 1.95 1.98 1.99 1.96 1.98 2.00 1.96 1.98 1.99 1.99 1.98 2.00 1.97
```

Only the first **three** positions carry information — that is the start codon (`ATGA` = 21% of all
cores) or a reverse-strand stop. **From position 4 onward RIPP core starts are indistinguishable
from random.** At 8 nt there are 2,651 distinct prefixes among 8,129 records; at 20 nt, 6,920.

Two consequences:
1. **A3 (consensus) is near-vacuous here.** The phage paper's consensus worked because ~15,000
   Microviridae genomes are *homologous and alignable*. RIPP was selected **for diversity** (43%
   near-dup loss); a consensus over non-alignable starts is noise, not a biological sequence.
   Run A3 only as a cheap negative control, and do not expect it to work.
2. **At 4–8 nt the objection largely evaporates anyway** — and so does the difference between the
   arms. `ATGA` is a start codon, not RIPP information. A 8-nt seed cannot be "filling in a
   cluster it memorised"; it carries ~16 bits. The exemplar-vs-consensus debate only bites at long
   seeds.

#### ❌ [P3-B1d] DOMAIN-ANCHORED SEEDING — MEASURED AND LARGELY DROPPED for RIPP
Proposed 2026-08-17 on the reasoning that class information is not at the 5′ end. **Measured the
same day: for RIPP that premise is false.** Position of the first RIPP marker domain, 400 cores:

| | first-domain offset |
|---|---|
| p50 | **0 nt** |
| p75 | **0 nt** |
| p90 | 433 nt |
| within 500 nt | 91.1% |
| **starting at nt 0** | **86.3%** |

antiSMASH **strict-core trimming already begins the region at the biosynthetic gene**. So for 86%
of RIPP cores the exemplar prefix *is* the domain start, and domain-anchored seeding collapses into
A1. There is no "everything before the domain" to generate — the concern is real in general and
does not apply here.

**What survives:**
- *Subsequent* domains are distributed (relative position p75 0.48, p90 0.73, ~1.5 domains/core),
  so a "seed from a non-first domain" variant exists — but it is a small subpopulation, not an arm.
- The idea keeps its force for **NRPS/PKS**, where cores are long multi-modular assembly lines.
  Revisit it there, not here.
- **A2 mosaic is the arm that actually carries the intent** — spans from k *different* clusters is
  a combination present nowhere in training, and it needs no new machinery.

### [P3-B1-EXP] The seeding experiment, fully specified

**Two facts that shape the whole design:**
1. **The seed never enters the scored span.** Both generators store the continuation only; 0 of
   1,512 seeded records on disk begin with their seed (`tests/test_scored_span.py`). The 2,000-nt
   window is 2,000 nt of *generated* sequence. There is no seed-recognition confound.
2. **`eval_prompts.jsonl` is 100% TEST** (199/199 accessions, 0% genome overlap with train).
   Clean — but it means tuning the seed length on it would be selecting on the test set.

#### Stage 1 — LENGTH SWEEP (tuning; **VAL** seeds; not confirmatory)
Hold seed *content* fixed at exemplar so length is the only variable.

| factor | levels |
|---|---|
| seed length | 0 (=A0, have) · 4 · 8 · 20 · 100 · 500 nt |
| model | RIPP LoRA · base 1B |
| seed source | **`splits_class/RIPP/val.jsonl`** (558 records) — build `val_prompts.jsonl` first |
| n | 50 per cell |

10 new cells × 50 ≈ **500 generations, ~1 h GPU.**
- **Primary readout for this stage is `containment`, not hit rate** — the question is *where does
  memorisation start*. The phage paper's whole point is that long seeds memorise.
- **Secondary:** `best_bio_bits > 0` @ `OBLIGATE_DOMAINS[RIPP]`.
- **Why base 1B is in the sweep:** it separates "the seed did it" from "the adapter did it". If
  base+500 nt matches LoRA+500 nt, the seed is carrying the result and the adapter is decorative.
- **Output: pick ONE L\*** — the longest length whose max containment stays under the 0.80 WARN
  threshold, and among those the best on-class rate. Record L\* before Stage 2.

#### Stage 2 — CONFIRMATORY ARMS at L\* (**TEST** seeds; pre-register n first)

| arm | seed | purpose |
|---|---|---|
| A0 | none | ✅ already significant, p=0.0054 |
| A1 | real exemplar @ L\* | ceiling + novelty floor |
| A2 | **mosaic** — fragments from k clusters, new k-subset per sample | the arm that answers "it is just completing a real cluster" |
| A3 | consensus, bootstrapped | ⚠️ expected to fail (entropy ≈ 2.0); cheap negative control |

× **two models: RIPP LoRA vs general all-class adapter** (the pre-registered comparison).
6 cells × n≈200 ≈ **1,200 generations, ~2.5 h GPU.** n=200 is powered for +10 pt at the 0.367
seeded base rate.

#### Gates that apply to every cell
- ⚠️ **`--no-boundary-orf` IS MANDATORY.** At L=500, 12/12 on-class hits reproduced their own
  source cluster's domain — the seed handed over most of a marker gene and the model finished it.
  The flag truncates the seed at its last in-frame stop so no ORF spans seed→continuation, forcing
  a class-defining domain in the continuation to be de novo. Run every Stage-2 arm with it, and
  report a without-flag arm as the adversary control.
- **Pre-register BOTH novelty gates** — `containment` AND protein AAI. Containment never exceeded
  0.021 in Stage 1 and would have passed the memorising L=500 configuration as clean.
- **MANIPULATION CHECK** (§9): seeded output must *differ from A0 on some measured axis*. A 4-nt
  seed that changes nothing is not a treatment, and its null is uninformative rather than negative.
- **Novelty:** `containment` reported per cell; computed on the continuation, which is already all
  that is stored.
- **Provenance:** every cell writes `<arm>_L<len>_w2000_RIPP.json` with the scoring stamp.
- **Substrate:** `export EVO2_BASE_MODEL=evo2_1b_base` — see `bugs.md`, this silently defaults to
  the 7B.
- ⚠️ **Do not run arms × lengths as a matrix.** Stage 1 fixes L\*, then Stage 2 varies arm only.

### [P3-B1c] Report the CLUSTERING rungs alongside the endpoint — cheap, do it in B1
The primary endpoint stays `best_bio_bits > 0` (Standing Constraint 4 — it does not change
mid-phase). But A0's breakdown showed all four hits carry **one** domain where real cores average
1.45, and a single domain is not a cluster. Two already-validated rungs measure exactly that and
are currently unreported:

- **`n_bio_domains`** (AUROC 0.919) — how many biosynthetic domains at all
- **`bio_span_frac`** (AUROC 0.896) — how far apart they sit, i.e. is it a *cluster*
  (real cores 0.876 vs de novo 0.051)

Both come free from `ladder_audit.one()`; the scorer already computes them. Add them as
**secondary outcomes** (§3, "reported always, decisive never") to every seeding cell. Expected
payoff: if seeding lifts `n_bio_domains` above 1 it is doing something qualitatively different
from A0, and that is a stronger claim than a hit-rate delta.

⚠️ **Do NOT add an ordering metric for RIPP.** `MODULE_PATTERNS` covers NRPS/PKS only, correctly —
those are collinear assembly lines. RiPP gene order is not collinear, and at 1.45 markers per core
order is undefined for most records. See `memory.md` 2026-08-17.

### [P3-WIN] MULTI-WINDOW SCORING — the 2 kb primary is now implicated
**Evidence that forced this (2026-08-18):** re-deriving the ladder AUROCs showed the cluster rungs
are **degenerate at a 2 kb window** — `n_class_domains` among on-class is 1.000, `bio_span_frac`
inverts to 0.173, and **real held-out cores average only 1.04 biosynthetic domains inside 2 kb.**
A window in which the *ceiling* shows one domain cannot measure cluster structure.

**Decision — add windows, do NOT replace the primary** (Standing Constraint 4):
- **Generate at 8 kb**, score at **2 kb (pre-registered PRIMARY), 4 kb, and 8 kb (declared
  secondaries).** One generation set, three scorings, filenames already carry the window
  (`_w2000` / `_w4000` / `_w8000`), so nothing is overwritten and nothing is confounded.
- The 2 kb number stays the headline so every Phase-3 arm remains comparable to A0 and Stage 2.
- The wider windows answer a different and now-necessary question: **does cluster structure exist
  further out than we have been looking?** If `n_class_domains` rises with window on the same
  sequences, the "one enzyme not a cluster" limitation is partly an artefact of the window rather
  than a property of the model.
- ⚠️ Do not compare a 4 kb or 8 kb number against any 2 kb number, including the 0.440 / 0.760
  ceilings. **Each window needs its own real-core ceiling**, scored identically.

### [P3-LEN] Re-generate the STRICT (A0) adapter at 8 kb for a length-matched comparison
The A0 adapter generated at 8 kb once (`A0_8k.jsonl`) but Stage 1/2 generated at 2.2 kb, so span
length is currently confounded with arm. Re-generate **both** adapters at 8 kb from the same seeds
and score all three windows. That isolates generation length from substrate width, and is the only
way to read the WIDE arm against A0 cleanly.
**Keep L\*=8 nt seeding** — it is the length at which the model is demonstrably *generating* rather
than reconstructing (0/8 source-domain match vs 12/12 at 500 nt), and changing it would reintroduce
the recall confound. Seed length and scoring window are independent axes; vary the window, hold the
seed.

### [P4-WIDE] WIDE_KINDS fine-tune — 🔄 RUNNING 2026-08-18
Substrate widened from `{"biosynthetic"}` to `{"biosynthetic","biosynthetic-additional"}`.
Same recipe as A0, `DATA=splits_class_wide/RIPP`, **epochs matched to A0 (3) rather than steps**.
⚠️ **3,723/7,808 records kept (47.7%)** — the rest exceed the 1B's **8,192 native context**
(`evo2_1b_base`, a hard model limit) and are dropped, not chunked, so `|END|` still lands true.
Read this arm as a **lower bound** on what WIDE can do: full WIDE is 4.41 genes/record, the
≤8,192 subset only 2.27 (vs 1.87 strict).

#### [P4-WIDE-CTRL] The size-matched control — SPEC (run after the WIDE arm reads out)
**The confound:** WIDE trains on 3,723 records, A0 on ~7,000. A difference could be span width
**or** dataset size. One extra arm separates them.

**Do NOT use a random 3,723-record subsample.** The WIDE subset is not random — it is the *short*
wide records, biased toward smaller clusters. Instead take **the STRICT spans of exactly the 3,723
accessions that survived the WIDE ≤8,192 filter.** Verified: 3,723/3,723 accessions are shared, so
the two arms contain **identical clusters, identical split membership, identical count** — the only
difference is span width. That is a perfectly matched pair, which a random subsample would not be.

| arm | records | total nt | median nt | status |
|---|---|---|---|---|
| A0 — strict, full | 6,963 | 14.58 M | 1,713 | ✅ done |
| **STRICT-matched** | **3,723** | **6.96 M** | 1,209 | ⬜ spec'd |
| WIDE | 3,723 | 13.69 M | 3,714 | 🔄 running |

**What each comparison isolates:**
- **WIDE vs STRICT-matched** → the effect of *span width*, at identical clusters and count.
- **STRICT-matched vs A0** → the effect of *dataset size* alone, at identical span width.

⚠️ **The one confound that cannot be removed: tokens.** At matched record count and matched epochs,
WIDE sees **1.97× the tokens** (13.69 M vs 6.96 M). You cannot match records, tokens *and* epochs
simultaneously — token-matching would require unequal epochs, and more passes over less data
carries its own memorisation risk. **Decision: match records and epochs, declare the 1.97× token
asymmetry, and report it with every WIDE number.** If WIDE wins, the honest claim is "wider spans
and the extra tokens they carry", not "wider spans alone".

**Endpoint:** unchanged (Standing Constraint 4) — `best_bio_bits > 0` @ `OBLIGATE_DOMAINS[RIPP]`,
2,000 nt. Report a wider window as a **declared secondary**, because WIDE markers spread over
~8.5 kb and a 2 kb window sees only 31.0% of them vs 70.7% on the full record.
**Primary question for this arm: does `n_class_domains ≥ 2` finally move off 2/188?**

#### ⚠️ [P4-WIDE-SEEDED] ◀ THE ACTUAL TEST — de novo was underpowered
The de novo WIDE arm ran and is **uninformative**: 0/79 · 0/79 · 1/79 vs A0's 2/85, n.s. at every
window, and at a ~0.024 base rate detecting even a doubling needs **n≈800/arm**. Generating de novo
was right for window-comparability with A0_8k and **wrong for power**.
⇒ **Run WIDE seeded at L\*=8 nt, n=188, `--no-boundary-orf`, test seeds — identical to S2-1** so it
reads directly against 0.176 in the regime that has power. Add the [P4-WIDE-CTRL] size-matched
STRICT arm at the same time. Watch `n_class_domains`: the one de novo WIDE hit carried **3**
biosynthetic domains, the most any generation has produced (n=1, an anecdote, not a result).

### [P4-RL] REINFORCEMENT LEARNING on our own gates — a Phase-4 track
**Idea (user, 2026-08-18):** generate in bulk, score, feed the winners back as positives and the
losers as negatives. In practice this is **rejection-sampling fine-tuning** (retrain on the winners)
or **DPO** (winners vs losers as explicit pairs). Tractable: LoRA on a 1B, and the scoring pipeline
already exists. At 0.176 a batch of 1,000 yields ~176 positives.

**Ordering — RL is DOWNSTREAM of the substrate work.** RL optimises what the model already puts in
its output distribution; it cannot teach structure the training data never contained. With 48.8% of
strict records being a single gene there is no gradient toward multi-gene output. **Run after
[P4-WIDE].**

#### [P4-RL-0] LITERATURE — what supports this, what warns against it (searched 2026-08-18)

**Supporting — the approach is established, and Evo's own authors endorse it.**
- **Evo 2 (Nature 2026)** states that *"supervised fine-tuning and reinforcement learning with
  feedback from biological experiments is likely to improve the efficiency and quality of sequences
  generated by Evo 2"* — an endorsement from the model's creators for exactly this.
- **DPO_pLM** (*Preference optimization of protein language models*, arXiv 2403.04187) is the closest
  analogue: DPO on an autoregressive protein LM against oracles (ESMFold pLDDT, the CLEAN enzyme
  classifier). **Single DPO round after SFT**, 9,439 base pairs → 66,898 preference triplets, ~1.3 h
  of DPO. Precedent that one round on modest data works.
- **RL with Verifiable Rewards / LLMol** (arXiv 2607.19044) — two-stage SFT→RL where a *verifiable*
  reward is the supervision. **Our gates are verifiable rewards** (a Pfam hit, an antiSMASH call);
  this is the framing to write the method up under.
- **DRAKES** — reward-guided DNA generation as single-step RL on a discrete diffusion model. DNA-
  specific precedent that reward-guided sequence design works on nucleotides, not just proteins.
- **ResiDPO** (*Designability Preference Optimization*) applies **residue-level** rewards and
  decouples optimisation across positions — the structural precedent for our **per-marker** weighting
  rather than one scalar per sequence.

**Warning against — two documented failure modes we are already exposed to.**
- ⚠️ **Rejection sampling reward-hacks the PROXY reward.** This is the literature's explicit finding,
  and it is our exact situation: our Pfam gate **inflates ~2× against antiSMASH** (0.176 vs 0.085).
  RL on the Pfam gate will optimise the proxy, not the goal. **Mitigation, and it is the user's own
  two-stage insight: use Stage A (Pfam, cheap) to select and Stage B (antiSMASH, expensive) to
  verify the winners — never train on Stage A alone.** The literature's other fix is adversarial
  training against the reward model.
- ⚠️ **Mode collapse when RL runs without the fine-tuned prior.** Reported directly: an RL-only seed
  *"collaps[ed] onto a very limited solution"*. We already show the precursor — PF05114 at
  near-natural rate and every other marker at 0–2%. **Keep a KL penalty to the SFT adapter and make
  intra-set distinctness a reward term, not a diagnostic.**

**What the literature does NOT do, and where our idea is a genuine addition.** DPO_pLM weights
preferred and dispreferred symmetrically and **does not discuss mode collapse or reward hacking at
all**. Graded negatives ([P4-RL-2]) are not standard practice in this literature. That is worth
stating as a contribution rather than assuming it is known.

**One method to add from the literature:** an explicit **KL-to-reference term** with a swept β.
DPO_pLM notes β controls "information retention from the reference model", and the mode-collapse
report is specifically about running without that anchor. Sweep it; do not pick one value blind.

#### [P4-RL-1] Reward = `JOINT_PASS`, never `on_class`
Standing Constraint 1's rationale is that every capability metric is maximised by copying training
data. RL optimises exactly what it is rewarded, so rewarding `on_class` rewards regurgitation. A
positive sample must pass **all** gates at once: on-class AND `containment` < 0.80 AND
`protein_aai` < 0.95 AND intra-set distinct. That is `JOINT_PASS`, already computed per record.

⚠️ **AND A HARD CONSTRAINT FROM 2026-08-18:** among on-class records **nothing we measure predicts
antiSMASH confirmation** (best AUROC 0.575; `bio_span_frac` inverts to 0.173). So a reward that
*ranks* on-class candidates by a ladder metric is **ranking on noise**. Rewards must be built from
**gate passes** (binary, verified) and **marker identity/count**, not from continuous ladder scores.

#### [P4-RL-2] GRADED NEGATIVES — weight by how many gates a record passes (user, 2026-08-18)
Run the full battery on **every** generation, not just the winners, and weight each record by its
gate count rather than a binary pass/fail.
- **Why it is better:** a record that is on-class but fails `protein_aai` is a *specific* kind of
  failure — "right answer, wrong route" — and is far more informative than one that failed
  everything. Binary labelling throws that distinction away.
- It also converts a sparse 17.6% binary signal into a dense one, which RL handles far better.
- No new machinery: the battery already emits every gate per record.
- **Open question to settle empirically:** whether to weight linearly in gate count or to treat
  novelty failures as hard zeros. Novelty-failing records are the ones RL will drift toward, so
  they may deserve negative rather than merely low weight.

#### [P4-RL-3] Marker-frequency weighting — run AFTER [P4-WIDE], unweighted first
See the measured marker table below. Weight a positive by the **rarity of the marker it produced**,
so PF02624 (YcaO) is worth more than another PF05114.
⚠️ **Do not stack this with [P4-WIDE] in one run.** One intervention at a time, or a gain cannot be
attributed. WIDE unweighted first; weighting only if the skew survives it.

#### MEASURED 2026-08-18 — the marker distributions that decide whether weighting is needed
Per-marker share of records carrying that marker (n=300 each, 2 kb window unless noted):

| marker | REAL held-out | STRICT train | WIDE train (full) | **GENERATED (S2-1)** |
|---|---|---|---|---|
| PF05114 DUF692 | 16.7% | 19.0% | 19.0% | **14.4%** |
| PF04055 RadSAM | 15.0% | 20.0% | 25.0% | **1.6%** |
| PF02624 YcaO | 14.0% | 13.7% | 19.3% | **0.0%** |
| PF05402 PqqD | 14.0% | 12.3% | 15.7% | **2.1%** |
| PF13353 Fer4_12 | 12.7% | 18.7% | 22.0% | **0.5%** |
| PF03070 TENA | 3.0% | 4.3% | 4.7% | 0.0% |
| PF14028 SkfB | 3.0% | 3.7% | 5.3% | 0.0% |
| PF00881 Nitrored | 2.7% | 3.0% | 7.3% | 0.0% |
| **any marker** | 58.3% | 63.7% | **70.7%** | 17.6% |

**Three things this settles.**
1. **The skew is in the MODEL, not the data.** STRICT train and REAL held-out agree closely on every
   marker. So "weight by training frequency" and "weight by real frequency" are the same thing — the
   choice of reference set does not matter, which was an open question.
2. **The collapse is sharper than "mostly PF05114".** The model emits PF05114 at **near-natural rate
   (14.4% vs 16.7%)** and essentially **none** of the other four common markers (0–2% against
   12–15%). It has learned one marker well and the rest not at all.
3. **WIDE raises marker content across the board** — every marker up, any-marker 63.7% → 70.7% —
   which is the mechanism by which it might fix the collapse without any reward shaping.

⚠️ **Scoring-window tension for WIDE.** The same WIDE records scored in a 2 kb window show only
31.0% any-marker, because the markers are spread across ~8.5 kb. A WIDE model may therefore
under-report on the pre-registered 2,000-nt endpoint. **Do not change the endpoint mid-phase**
(Standing Constraint 4) — report the 2 kb endpoint as primary and a wider window alongside as a
declared secondary.

### [P3-B2a] Pruning DURING generation (guided decoding) — ⛔ BLOCKED ON A SCORER
> **Not the phage-paper approach.** This scores partial candidates *mid-generation*. [P3-B2b] below
> generates complete sequences then discards most. Different mechanisms; they compose.

⛔ **CLOSED 2026-08-18 — MEASURED, not assumed. Two instrument families both fail.**
The prerequisite below was run. The class probe scores **AUROC 0.337** for predicting antiSMASH
confirmation among on-class records — *anti*-correlated, saturated at median P(RIPP) ≈ 0.997, from a
classifier with 0.933 held-out balanced accuracy. Ladder metrics reach 0.575 at best. **Nothing we
own separates a real cluster from a Pfam-passing near-miss.** Pruning on any of them ranks noise.
⇒ **Do NOT fit a 1B probe** — the failure is saturation against a target the probe cannot see, which
a refit inherits. ⇒ Reopen only if a *new* signal appears (e.g. an antiSMASH-derived reward, or a
structure model), not by re-testing what is already measured.

Historical note kept: the original blocker reasoning was —
Pruning only helps if the scorer can rank *among candidates that would pass the gate*. Re-deriving
the ladder in this regime showed **nothing we measure does that**: within the on-class pool the best
metric reaches 0.575 and `bio_span_frac` inverts to 0.173. Pruning on any of them ranks **noise**,
and would look like it worked because the retained set still passes the gate it was selected on.

**Prerequisite, and it is cheap:** the class probe is the only remaining candidate — it is
continuous, model-internal, and never yet tested for *within-positives* discrimination. Test it
against the 68 on-class records that now carry antiSMASH labels. **If the probe's AUROC for
predicting antiSMASH confirmation among on-class records is not clearly above 0.5, leg 3 has no
instrument and should stay closed** regardless of the Q1/Q2 history.
⚠️ Use the **train-only** probe (`acts_v2_train500.probe_L16_s0.joblib`), never the pre-2026-08-10 fit.


> **Not the phage-paper approach.** This scores partial candidates *mid-generation* and keeps the
> best, so a sequence is steered as it is written. [P3-B2b] below is the phage-paper approach —
> generate complete sequences, then throw most away. Different mechanisms; they compose.
- **Why this is leg 3, and why it is not the same as filtering below.** Guided decoding prunes
  candidates *during* generation using the class probe as a fast scorer (one matmul). Filtering
  throws away finished sequences. Different mechanisms, different costs, different ceilings.
- **It is the project's only positive conditioning result** and it was left underpowered, not
  closed: Q1 **+5.71 (39/40)**; Q2 **5–0, p=0.0625 at effective n=5**. A p-value that cannot go
  below 0.0625 is a design problem, not a null — n was the binding constraint.
- **It was closed when conditioning looked like the wrong target.** That reasoning does not
  transfer: the probe is a *class* scorer, and in a per-class regime the question changes from
  "can we steer class?" to "can we prune toward RIPP machinery?"
- **Prerequisite — check before spending anything:** the probe must be the train-only one
  (`acts_v2_train500.probe_L16_s0.joblib`, provenance-verified). Not the pre-2026-08-10 fit.
- **MANIPULATION CHECK:** confirm the scorer changes which candidate is kept — log kept-vs-best
  rank per step. `tests/test_guided_decoding.py` already pins which candidate is kept and how Q1
  is read.
- **Composes with [P3-B1]** — seeded generation with pruning is the strongest single arm available.

### [P3-B2b] Filtering AFTER generation (the phage-paper funnel)
> Generate complete sequences, score them, keep the survivors. No change to how any single
> sequence is written — purely a selection step over finished output.
- Adopt the phage-paper shape: generate **short**, sample **many**, filter **hard**. Measured
  support: block-0 detection 0.040 vs 0.024/0.028/0.022 later; 4× the tokens bought only 2.2× the
  hits (vs 0.151 predicted under independence). Tokens spent late are worth ~half those spent early.
- Their funnel: ~14,466 training genomes → thousands generated → 302 candidates → 285 synthesised
  → 16 viable. Roughly 1000:1 overgeneration.
- ⚠️ **`|END|` does not work and is not worth fixing** (0/150, previously 0/204; whole-record
  training did not change it). The phage paper used a **length filter (4–6 kb)**, not a stop token.

### [P3-B3] ✅ DONE 2026-08-17 — A0 powered to p=0.0054 by generating controls
**The obvious plan is wrong.** Generating more A0 sequences does **not** close this. Fisher's exact
against a fixed 0/100 control plateaus at p≈0.09 and never crosses 0.05:

| treat n | ctrl n | expected hits | p |
|---|---|---|---|
| 150 | 100 | 4 | 0.128 (current) |
| 300 | 100 | 8 | 0.098 |
| 500 | 100 | 13 | 0.091 |
| **150** | **300** | **4** | **0.012** ✅ |
| 200 | 400 | 5 | 0.004 ✅ |

**The control arm is the binding constraint.** ~200 more *control* generations converts the
existing A0 data — unchanged, already on disk — into a significant result. Controls are also the
cheap arm: base 1B and the general adapter, no training, generation only.

- **Conditional, and state it as such:** this holds only while the control rate stays at exactly 0.
  Both controls currently read 0.000 (0/50 each). A single control hit moves the p-value materially.
- **Pre-register n BEFORE generating** (Standing Constraint 4). Pick the n, run it once, read it
  once. Do not creep the sample — that is what makes this a test rather than a search.
- **Endpoint:** `best_bio_bits > 0` @ `OBLIGATE_DOMAINS[RIPP]`, 2,000 nt, identical generation
  config to A0.

### [P3-B4] Housekeeping — ✅ DONE 2026-08-14
- ✅ `phase3_ripp/` rsync-merged into `phase3_RIPP/` (lossless, verified a strict subset before
  removal); `phase3_pilot.py` repointed. No case collisions remain.
- ✅ Orphaned `HSERLACTONE/` + `BUTYROLACTONE/` splits **deleted** — both already disqualified on
  diversity, and the manifest cannot be honestly regenerated post-hoc.
- ✅ Empty run dirs removed: `_scripts/`, `phase1_lora_prod_20260604_151300_L32768/`,
  `phase1_lora_prod_20260604_151541_L32768/`.
- ✅ Archived `progress.md` TERPENE claim corrected in place; target reconfirmed **RIPP**.
- ✅ Filesystem naming convention added to `CLAUDE.md`.

### [P3-B9] ~~Fix `--mismatch-tag`~~ — **DROPPED 2026-08-18 (user)**
Not worth fixing: class-prepended tokens were already shown to have no effect in Phase 1 (the class
tag is worth **−0.0006 nats**), and Phase 3 side-steps label conditioning entirely by **routing to
class-specific adapters**. The S2-5 arm asked a question the architecture no longer poses. The
underlying no-op is still recorded in `bugs.md` so nobody re-runs into it.

### [P3-B8] Re-run the batched-generation equivalence gate
- Every Phase-3 number was generated through `generate_batch()`, which left-pads ragged prompts so
  vortex will batch them. That workaround **failed an on-GPU equivalence gate** historically
  (head-token LCP ~0.004, byte divergence) and the gate has not been re-run on the 1B.
- **The A0 result is unaffected as a comparison** — every arm used the same path. The open question
  is whether the absolute rates match sequential generation.
- Run `evo2/scripts/validate_batched_generation.py` on the 1B. If it fails, label Phase-3 rates as
  batched-path rates in `terms.md` and quote them only against other batched-path numbers.

### [P3-B7] Make the substrate explicit in code, not the shell
- `generate_bgc.py` has no `EVO2_BASE_MODEL` guard and **defaults to the 7B**. Every 1B script
  exports it at the top, so the substrate rides on the caller's shell. A bare invocation silently
  generates from the wrong model and only fails if an adapter happens to be shape-incompatible.
- Cost this already: 150 discarded control generations on 2026-08-17. See `bugs.md`.
- Fix: require the env var (or an explicit `--base-model`) and exit non-zero when absent. Also
  stamp the resolved checkpoint into every generation JSONL, the way scored outputs now stamp
  their scoring config.

### [P3-B5] File the 69 loose artifacts at the runs root
- `/data2/ds85/bgcmodel_runs/` holds 21 result files (4.0 MB), 47 logs, and 1 shell script outside
  any run directory. **The results are load-bearing** — `ladder_audit.json` is the ladder validation
  itself; `direction_audit.json`, `activation_patching_ksweep.json`, `context_ablation.json`,
  `length_ceiling.json` are all cited measurements.
- Four are referenced by path in code, so this is a change with blast radius, not a tidy-up. Do it
  deliberately: create owning run dirs, `git grep` each filename, move and repoint together.
- The `CLAUDE.md` naming convention now forbids creating more.

### [P3-B6] Decide on ~90 GB of superseded data — needs your call
Disk is at 84% (1.2 TB free), so this is not urgent, but nothing here is documented as keep-forever:
- **37 GB** — 7 deprecated split dirs, all marked DO-NOT-USE in `data.md`
- **46 GB** — `asdb5_*.jsonl` pipeline intermediates at the data root. ⚠️ Deleting these means
  `splits_core` cannot be rebuilt without re-running antiSMASH extraction from `asdb5_gbks/`.
- **7.6 GB** — `probe_subsets/` + `probe_subsets_8k/`

---

## Blocked

| Item | Blocked on |
|---|---|
| 7B confirmation of any Phase-3 result | **Unblocked in principle** — A0 is now significant (p=0.0054). Hold anyway until Stage 2 lands, then confirm the single best arm, not every arm. Standing Constraint 3. |
| Quartz multi-GPU long-context | PI allocation pending |

---

## Deferred (not dropped)

- **Per-layer conditional adapters** — deferred 2026-08-12, still coherent, but conditioning is no
  longer the binding constraint.
- **Characterisation paper (Track A)** — bank exemplar conditioning as a descriptive result. Weeks,
  mostly CPU. This is the fallback if Phase 3 does not reach significance.
- **GenomeOcean** — live but held; its leakage gate passed. Testing does not fan out across models.
