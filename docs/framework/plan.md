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

Ordered. Top item is next.

### [P3-B1] Seeded class-specific vs seeded generalist ◀ NEXT
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

### [P3-B2] Overgenerate-and-filter funnel
- Adopt the phage-paper shape: generate **short**, sample **many**, filter **hard**. Measured
  support: block-0 detection 0.040 vs 0.024/0.028/0.022 later; 4× the tokens bought only 2.2× the
  hits (vs 0.151 predicted under independence). Tokens spent late are worth ~half those spent early.
- Their funnel: ~14,466 training genomes → thousands generated → 302 candidates → 285 synthesised
  → 16 viable. Roughly 1000:1 overgeneration.
- ⚠️ **`|END|` does not work and is not worth fixing** (0/150, previously 0/204; whole-record
  training did not change it). The phage paper used a **length filter (4–6 kb)**, not a stop token.

### [P3-B3] Power A0 to significance, or kill it
- 4/150 vs 0/100 pooled is p=0.152. Decide from a power analysis what n closes it, then run that n
  once. Do not creep the sample.

### [P3-B4] Housekeeping (cheap, do alongside)
- Rename `phase3_RIPP/` vs `phase3_ripp/` — they differ only in case (see `data.md`).
- Resolve the orphaned `HSERLACTONE/` + `BUTYROLACTONE/` splits: regenerate manifest or delete.
- Fix `progress.md:874` "the target is TERPENE" → RIPP (at cutover).
- Delete the empty run dirs: `_scripts/`, `phase1_lora_prod_20260604_151300_L32768/`, `phase1_lora_prod_20260604_151541_L32768/`.

---

## Blocked

| Item | Blocked on |
|---|---|
| 7B confirmation of any Phase-3 result | Nothing publishable yet — A0 is n.s. Do not spend 7B time until a 1B result is significant. |
| Quartz multi-GPU long-context | PI allocation pending |
| Steering directions fit on val+test | Open debt from 2026-07-30 — must be refit on `valtest_fit` alone before any of that work is published |

---

## Deferred (not dropped)

- **Per-layer conditional adapters** — deferred 2026-08-12, still coherent, but conditioning is no
  longer the binding constraint.
- **Characterisation paper (Track A)** — bank exemplar conditioning as a descriptive result. Weeks,
  mostly CPU. This is the fallback if Phase 3 does not reach significance.
- **GenomeOcean** — live but held; its leakage gate passed. Testing does not fan out across models.
