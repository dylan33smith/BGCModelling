# Phase 8 — GenomeOcean on TERPENE. Pre-registration.

**Opened 2026-08-20.** Written BEFORE any Phase-8 arm generates (Standing Constraint 4). Endpoints
in this file do not change mid-phase; deviations are recorded as dated amendments below, never by
editing the original text.

Companions: `docs/phase6_PKS_preregistration.md`, `docs/phase7_TERPENE_preregistration.md`. This
phase re-runs **Phase 7's exact endpoint on a different model**, so its comparison partner is
`[P7-A0]` and nothing else.

---

## 1. The one question this phase exists to answer

Three classes, three adapters, one model. Every Phase-3/6/7 finding — the ~7%-of-ceiling rate, the
single-marker limit, and above all **"the model only ever makes the easiest member of its class"** —
is confounded with **Evo2-1B**. This phase asks whether that is a property of *the method* or of
*that model*.

**It is the binding question** because the alternative interventions ([X2a–d]) are all designed
against "the method", and are aimed at the wrong cause if the answer is "the model".

## 2. ⚠️ WHY TERPENE, AND WHY NOT PKS — this is the whole design

GenomeOcean differs from Evo2-1B on **four axes at once**: parameters (4.25 B vs 1.1 B), context
(32,768 BPE positions ≈ **51,200 nt** vs 8,192 nt), tokenizer (BPE vs byte-level) and pretraining
corpus (metagenomic assemblies vs all-domains). **A win cannot be attributed to any single axis.**

TERPENE is chosen to remove the most dangerous of the four from contention:

| | fits Evo2's 7,992 nt budget | so a GenomeOcean win could be explained by context? |
|---|---|---|
| **TERPENE** | **94.0%** | **NO — Evo2 was not context-limited here** |
| RIPP | 89.4% | partly |
| PKS | 75.2%, and the median T1PKS (7,665 nt) barely fits | **YES — uninterpretable** |

⇒ **PKS is the tempting first class and the wrong one.** Its hard member is the one that does not
fit, so a GenomeOcean win there is uninterpretable between "better model" and "longer context".
⇒ TERPENE also has the cleanest easy/hard split (`terpene-precursor` vs the `terpene` cyclase rule)
and the highest ceiling on both instruments (Pfam 0.980, antiSMASH **1.000**).

## 3. T1 — THE BASE CHECKPOINT: `GenomeOcean-4B`, not `bgcFM`

Both are local and both passed the leakage gate. The decision rests on a measurement, not a
preference:

| | class probe, balanced acc | chance | shuffled control |
|---|---|---|---|
| `GenomeOcean-4B` (base) | **0.878** (layer 8) | 0.091 | 0.083 |
| `GenomeOcean-4B-bgcFM` | **0.894** (layer 12) | 0.091 | 0.082 |
| Evo2 (for reference) | ~0.91 | 0.09 | — |

⇒ **bgcFM's advantage is +0.016 — marginal.** Its extra BGC pretraining does **not** buy a stronger
class representation, so choosing it would add confounding without adding the thing we need.
⇒ **All three models already encode compound class at ~0.88–0.91 against ~0.09 chance.** The
bottleneck has never been representation; it is generation. Recorded here because it predicts that a
model swap alone may not move the endpoint.

**DECISION: the pre-registered arm is fine-tuned `GenomeOcean-4B`**, so the comparison against
`[P7-A0]` is fine-tune vs fine-tune.
**`bgcFM` runs zero-shot as a DECLARED REFERENCE CEILING, never as the arm** — it was trained on
12 M SMC BGC sequences, so a win by bgcFM would read "it already saw BGCs", which answers a
different question. Its zero-shot unconditioned rate is already measured: **`is_bgc` 27/216 =
0.125**.

## 3.1 T2 COMPLETE 2026-08-22 — the substrate is built, and it STRENGTHENS §2's argument

`genomeocean/scripts/build_class_substrate_go.py` · `phase8_TERPENE_GO/substrate_report.json`.
`splits_class/TERPENE` reused unchanged; only the length filter differs.

| | train | val |
|---|---|---|
| records in the split | 11,297 | 793 |
| median length | **960 nt = 192 tokens** (**4.974 nt/token**) | 975 nt = 197 tok |
| kept by **Evo2-1B** (<= 7,992 nt) | 10,658 = **0.943** | 747 = 0.942 |
| kept by **GenomeOcean** (<= 10,240 tok) | 11,260 = **0.997** | 788 = 0.994 |
| **records recovered** | **+602 (+5.3%)** | +41 (+5.2%) |

**Context available: 10,240 tokens x 4.974 nt/token = 50,934 nt — 6.4x Evo2-1B's 7,992.**

⇒ ★ **AND IT BUYS ALMOST NOTHING HERE — +5.3% of records.** That is not a disappointment, it is
**the design working as intended (§2).** TERPENE was chosen precisely because Evo2 was not
context-limited on it, and T2 now *measures* that rather than assuming it: a 6.4x context increase
recovers one record in nineteen. **So if `[P8-A0]` beats `[P7-A0]`, the 6.4x context cannot be the
explanation.** On PKS the same table would have looked entirely different and the phase would have
been uninterpretable.

⇒ **37 train records still do not fit** even at 10,240 tokens — the largest is **54,376 tokens
(~270 kb)**, which exceeds GenomeOcean's own 32,768 ceiling. No context setting rescues those; they
are dropped and counted, not hidden.

⇒ **Training context is FROZEN at 10,240 tokens** — the value the feasibility gate passed at, and it
already keeps 99.7%. Raising it to the 32,768 ceiling would buy at most 0.3% of records for ~3x the
memory.

✅ **EOS verified in code, not assumed:** the tokenizer auto-wraps `BOS=1 … EOS=2`, asserted by the
builder, which **refuses to emit a substrate if the wrap is absent**. `[X1]` therefore does not apply
to this phase — but junk-token and degeneracy rates are still reported (§6.3), because their absence
is itself a result.

## 4. Primary endpoint — INHERITED FROM PHASE 7, UNCHANGED

**`best_bio_bits` > 0 @ `OBLIGATE_DOMAINS[TERPENE]` (7 accessions), scoring window 2,000 nt**, then
full-mode antiSMASH at **`--minlength 200`**, output dirs retained.

⚠️ **Every scoring axis is inherited deliberately and must not be re-tuned**: same marker set, same
window, same `--minlength`, same ceiling file (`phase5_classprobe/real_TERPENE_fit50_w2000.json`),
same `scripts/novelty_battery.py`, same `scripts/antismash_full.py`. **A pipeline change invalidates
the only comparison this phase exists to make.** This is the one phase where inheriting rather than
re-deriving is correct, because the model is the variable.

## 5. Arms and n

| arm | model | role |
|---|---|---|
| **P8-A0** | fine-tuned `GenomeOcean-4B`, TERPENE | the treatment |
| **P8-C1** | un-fine-tuned `GenomeOcean-4B` | floor — isolates the fine-tune from the pretraining |
| **P8-REF** | `bgcFM` zero-shot | declared reference ceiling, **not** a control |
| *(existing)* `[P7-A0]` | Evo2-1B TERPENE adapter | **the comparison partner** — 14/200, corrected 0.065 |

**n = 200 per arm, on the SAME 200 prompts** from `splits_class/TERPENE/eval_prompts.jsonl` used by
`[P7-A0]`, so the cross-model comparison is **prompt-paired** and a McNemar test is available.

## 5.1 AMENDMENT 2026-08-22 — ⛔ §5's "prompt-paired / McNemar" CLAIM IS WITHDRAWN

§5 above stands unedited and stated that n=200 on **the same 200 prompts** as `[P7-A0]` makes the
cross-model comparison **prompt-paired**, with a McNemar test available. **That is impossible under
this design and was wrong when written.**

Evo2's 200 prompts are 200 distinct `(class, taxonomy)` pairs. **GenomeOcean cannot represent that
prefix at all** — 122 UNK of 132 ids (§3.1, [X4]) — so its only conditioning is the atomic
`[CLS_TERPENE]` token. **All 200 GenomeOcean generations therefore share ONE prompt** and differ
only by sampling. There is nothing to pair *on*.

⇒ **The comparison is UNPAIRED. McNemar is unavailable. Fisher's exact is the correct test**, on
`[P8-A0]` vs `[P7-A0]` counts out of 200 each.
⇒ **The within-Phase-8 comparisons are unaffected** — `[P8-A0]` vs `[P8-C1]` were always unpaired,
both being unconditional-or-single-token draws.
⇒ This costs statistical power, not validity. n=200 vs 200 with Evo2 at 14/200 remains adequately
powered for a difference of the size Phase 7 measured.

**A sixth unmatched axis, declared here:** **decoding parameters.** Evo2 used `temperature 1.0,
top_k 4` — sensible over a 4-letter byte alphabet, meaningless over a 4,096-token BPE vocabulary.
Phase 8 uses GenomeOcean's own upstream preset (`temperature 0.9, repetition_penalty 1.2,
top_k off, top_p 1.0`), which is also what the existing zero-shot runs used. ⚠️ **`repetition_penalty
1.2` is an intervention Evo2 never had** and it plausibly suppresses exactly the degenerate repeats
that plagued Evo2. Report it as part of the config; do not read a degeneracy difference as a model
difference.

**Generation length:** `min_new_tokens = 0` — EOS is allowed to fire on its own, because a working
stop signal is one of the things this model gets for free and forcing a minimum would suppress it.
`max_new_tokens = 900` ≈ 4,500 nt, matching Phase 7's 4,000 nt generation. **Report the length
distribution as its own row**, exactly as Phase 6/7 do, and state how many records fall below the
2,000 nt scoring window.

## 5.2 AMENDMENT 2026-08-24 — T9 ADDED: seeded arms, and a 2x2 that did not previously exist

**Registered BEFORE the seeded arms generate.** The de novo comparison (T5–T7) stands as the primary
and is unchanged.

⚠️ **CORRECTING A MISREADING WORTH RECORDING: `[P7-A0]` IS DE NOVO, NOT SEEDED.** Its records carry
no `seed_nt` / `seed_source` / `seed_accession` fields, and **no seeded TERPENE arm exists anywhere** —
seeding was Phase-3 RIPP only (`S2-*`, `SF_seeded*`). Any table pairing "GenomeOcean + seed" against
"Evo2" would therefore be comparing a seeded arm to a de novo one. **The honest design is a 2x2, and
two of its four cells did not exist:**

| | Evo2-1B | GenomeOcean-4B |
|---|---|---|
| **de novo** | `[P7-A0]` ✅ | `[P8-A0]` ✅ |
| **seeded (L\*=8)** | **`[P7-A0s]` — NEW** | **`[P8-A0s]` — NEW** |

**L\* = 8 nt, and it transfers with justification rather than by assumption.** L\*=8 was derived on
RIPP start-codon entropy. Measured on 300 TERPENE held-out cores the profile matches: positional
entropy **1.74 bits at position 0 rising to ~1.94–1.97 by positions 7–11** (RIPP: 1.53 → 1.98–1.99),
and ATG-start rates 0.35 vs 0.41. **The seed stops carrying information at the same place in both
classes**, so 8 nt is as defensible here as there.

⚠️ **Seeded rates are scored on the CONTINUATION ONLY** and are not comparable to de novo rates —
they are a different regime (Standing Constraint 9). The 2x2 is read **down columns** (does seeding
lift this model) and **across rows** (does the model matter at this seeding level), never diagonally.

⚠️ **GenomeOcean seeding is not mechanically identical to Evo2's.** An 8-nt seed is ~1.6 BPE tokens,
so it does not land on a clean token boundary the way it does for a byte-level tokenizer. Seeded rows
also have prompts of differing length and are left-padded with the pad token rather than truncated.
Both are recorded per record (`seed_nt`, `seed_prefix`, `scored_span`).

## 6. ⚠️ CONFOUNDS DECLARED BEFORE THE RESULT

1. **NOT parameter-matched — 4.25 B vs 1.1 B.** Never report this arm as if it were.
2. **NOT context-matched** — but TERPENE is chosen so this cannot explain a win (§2).
3. **NOT tokenizer-matched** — BPE vs byte-level. This also means **`[X1]` does not apply here**:
   GenomeOcean's tokenizer auto-wraps every sequence `BOS=1 … EOS=2`, so a proper single-token EOS
   is trained for free, and the Evo2 stray-byte pathology should be absent. **Report the junk-token
   and degeneracy rates anyway** — their absence is itself a result.
4. **NOT corpus-matched** — metagenomic assemblies vs all-domains.
5. **Generation throughput differs by ~an order of magnitude** (bgcFM zero-shot ran ~44 s/sequence).
   Budget accordingly; do not let it silently shrink n.

## 7. Novelty gates — unchanged, and one is load-bearing here

`containment` < 0.80 AND `protein_aai` < 0.95, plus intra-set distinctness, read against the
**TERPENE real-core distribution** (median 0.011, p90 0.034, max 0.140), not RIPP's.

⚠️ **The leakage gate is the one that could disqualify this track and it has PASSED**:
`genomeocean/experiments/smc_leakage.json`, containment **0.0000** across 48 true + 48 mismatched
against our test set. **Re-run it on the fine-tuned arm** — fine-tuning on our data changes what
memorisation is possible.

## 8. Kill criterion

If `[P8-A0]` reaches a rate indistinguishable from `[P7-A0]` at n=200 **and** still produces
**0 cyclase-rule detections**, then the "only the easiest member" limitation is **not** an Evo2-1B
artefact, it survives a 4× larger model with 6.4× the context and a different tokenizer, and
**[X2] should be un-held and pursued as a method problem.** That is a decisive, useful negative.

Conversely, if `[P8-A0]` produces cyclase-rule detections where Evo2 produced none, the finding is
**model-attributable**, and the [X2] interventions are aimed at the wrong cause.

## 9. What would make this exploratory rather than confirmatory

Changing the window, marker set or `--minlength`; using different prompts than `[P7-A0]`; swapping
the base checkpoint after seeing results; reporting `bgcFM` as the arm rather than as a reference;
pooling GenomeOcean and Evo2 rates; or quoting a Phase-8 number against a Phase-6 number. If any
occurs, the result is labelled exploratory in every document that reports it.
