# plan.md — the dashboard

**Read at session start. This file holds only the active phase.** Completed interventions keep a
one-row summary in the Phase Ledger for the rest of the phase; their full write-up goes to
`memory.md` at completion. At phase close the ledger collapses to one line and the board resets.

**Last updated:** 2026-08-14

---

## Current State

Phase 3 is open: **one small class at a time, target RIPP**, on the **Evo2 1B** testing substrate.
Restricting to a single class *deletes* the long-context problem rather than working around it, and
a per-class LoRA means the model never reads a class label — so every Phase-1 conditioning closure
stops applying. The binding constraint is **capability, not conditioning**: de novo output is real
protein of the wrong kind (`biosynthetic_fraction` 0.100 vs 0.836 on real cores). **A0 has run** —
a RIPP-only adapter is the only non-real arm producing RIPP machinery at all (4/150 = 0.027 vs
0/50 for both the base 1B and the general adapter, against a 0.440 real-core ceiling), but at
p=0.152 it is **not yet significant**, and its first report was inverted by a scoring bug. Novelty
is clean throughout. **Nothing seeded has been run yet, and seeding is the mode that works** — so
the next intervention is seeded class-specific vs seeded generalist.

---

## Phase Ledger — Phase 3

Endpoint names are `terms.md` identifiers. `memory.md` column = date anchor to grep.

| ID | Intervention | Endpoint | n | Result | Verdict | memory |
|---|---|---|---|---|---|---|
| P3-A0 | RIPP-only LoRA, **de novo** | `best_bio_bits` > 0 @ `OBLIGATE_DOMAINS[RIPP]`, 2000 nt | 150 | **4/150 = 0.027** | ✅ positive direction, **p=0.152 n.s.** | 2026-08-14 |
| P3-C1 | base 1B control, de novo | ″ | 50 | 0/50 = 0.000 | floor | 2026-08-14 |
| P3-C2 | general all-class adapter, de novo | ″ | 50 | 0/50 = 0.000 | floor (0.080 under the *generic* set — other classes' domains) | 2026-08-14 |
| P3-CEIL | real RIPP cores | ″ | 50 | 22/50 = 0.440 | ceiling | 2026-08-14 |
| P3-NOV | novelty guard on A0 | `containment` | 150 | max 0.003; AAI med 0.000 / max 0.470; 150/150 distinct | ✅ pass | 2026-08-14 |

**Provenance for the block above:**
`phase3_RIPP/adapter_run` (7,250 whole records, 3 ep / 1,350 steps, `loss_ce` 0.790→0.410) ·
`A0_8k.jsonl` + `phase3_ripp/pilot_*.jsonl` · scoring `OBLIGATE_DOMAINS[RIPP]` · window 2,000 nt ·
substrate Evo2 1B (TE 1.13.0 verified).

**Standing reading of the ledger:** A0 reaches ~6% of ceiling. The direction changed on the
corrected metric; the significance did not. Do not report A0 as a win.

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
| 1. class-specific LoRA fine-tuning | ✅ A0 run — 0.027 vs 0.000 floor, p=0.152 n.s. |
| 2. class-specific seeding | ⬜ **not started** — and it is the regime that works |
| 3. inference pruning | ⬜ not started — two distinct mechanisms: [P3-B2a] prunes *during* generation, [P3-B2b] filters *after* |

Ordered. Top item is next.

### [P3-B0] ⛔ BLOCKER — make the scorer actually class-specific ◀ DO THIS FIRST
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

### [P3-B1] Seeded class-specific vs seeded generalist ◀ NEXT AFTER B0
- **Hypothesis:** With a real RIPP core as seed, the RIPP-only adapter beats the all-class adapter
  on RIPP-specific `best_bio_bits`. Seeding is the only regime with real variance in detection
  (0.367 vs 0.012 de novo) and the mode Evo's own published work validates.
- **Reasoning:** The informative contrast has never been run — A0 compared *unseeded* arms, where
  the floor is ~0 for everything. Seeded-vs-seeded is where a class-specific fine-tune can show up.
- **Technical:** `splits_class/RIPP/eval_prompts.jsonl`; both adapters; identical seed set.
- **Seed length:** ⚠️ **4–8 nt**, per Hie et al. Ours have historically been ~500 nt, and long
  seeds caused **memorisation** in that work. Sweep short; novelty-gate hard.
- **Primary endpoint:** `best_bio_bits > 0` @ `OBLIGATE_DOMAINS[RIPP]`, 2,000 nt window.
- **n and power:** from the A0 pilot rates; pre-register before generating.
- **MANIPULATION CHECK:** confirm the seed is present in the prompt and **absent from the scored
  span** (`tests/test_scored_span.py` pins this — 0/1512 records contained seed).
- **Novelty guard:** containment vs training; short seeds specifically to keep it clean.

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

### [P3-B3] Power A0 to significance — by generating CONTROLS, not more treatment
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
| 7B confirmation of any Phase-3 result | Nothing publishable yet — A0 is n.s. Do not spend 7B time until a 1B result is significant. |
| Quartz multi-GPU long-context | PI allocation pending |

---

## Deferred (not dropped)

- **Per-layer conditional adapters** — deferred 2026-08-12, still coherent, but conditioning is no
  longer the binding constraint.
- **Characterisation paper (Track A)** — bank exemplar conditioning as a descriptive result. Weeks,
  mostly CPU. This is the fallback if Phase 3 does not reach significance.
- **GenomeOcean** — live but held; its leakage gate passed. Testing does not fan out across models.
