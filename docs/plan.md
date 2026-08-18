# plan.md — the dashboard

**Read at session start. This file holds only the active phase.** Completed interventions keep a
one-row summary in the Phase Ledger for the rest of the phase; their full write-up goes to
`memory.md` at completion. At phase close the ledger collapses to one line and the board resets.

**Last updated:** 2026-08-14

---

## Current State

Phase 3, target **RIPP**, substrate **Evo2 1B**. **Legs 1 and 2 both now have positive results.**
Leg 1: a RIPP-only LoRA produces RIPP machinery de novo at 4/150 = 0.027 vs **0/400** pooled
controls, **p = 0.0054** (pre-registered). Leg 2 Stage 1: seeding lifts that to **0.160 at an 8-nt
seed** (~6×), and — the strongest control in the project so far — **base Evo2-1B scores 0/50 at
every seed length up to 500 nt.** A real 500-nt RIPP prefix handed to the base model yields no RIPP
domain at all, so the seed is not doing the work; the adapter is.

**The seed length is where generation turns into recall.** At L=500, **12/12** on-class generations
reproduced a marker domain their **own source cluster** carries; at L=8, **0/8** did — 6 of 8 emitted
PF05114, simply the commonest RIPP marker, i.e. a class prior rather than a memory. Mechanism: 86%
of cores begin at the marker gene, so a long seed hands over most of that gene and the model
finishes it. **`containment` is blind to this** (max 0.021 at L=500 against a 0.80 gate) while
protein AAI rises to 0.914 (median 0.000 → 0.291). **L\* = 8 nt** — L=500's higher rate is not
statistically distinguishable (p=0.227) and carries the recall signal.

Next is Stage 2: arms A1/A2/A3 × LoRA/general at L=8 on TEST seeds, **`--no-boundary-orf` mandatory**,
with n and **both novelty gates** pre-registered first. The persistent gap is structural — best cell
is 4/50 records with ≥2 distinct RIPP markers against 29% for real cores. **One enzyme, not a
cluster** — consistent with 48.8% of training records being a single gene.

---

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

## Backlog — Phase 3

**The phase has three legs.** Leg 1 is done and negative-but-directional; legs 2 and 3 are untried.

| leg | status |
|---|---|
| 1. class-specific LoRA fine-tuning | ✅ **DONE — 0.027 vs 0/400, p=0.0054 significant** |
| 2. class-specific seeding | 🔄 **Stage 1 done, L\* = 8 nt.** Stage 2 arms next |
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

### [P3-B1] The SEED LADDER — leg 2 · **Stage 1 ✅ DONE (L\* = 8 nt); Stage 2 ◀ NEXT**
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

### [P3-B2a] Pruning DURING generation (guided decoding) ◀ the underrated one
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
