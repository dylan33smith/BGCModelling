# plan.md — the dashboard

**Read at session start. This file holds only the active phase.** Completed interventions keep a
one-row summary in the Phase Ledger for the rest of the phase; their full write-up goes to
`memory.md` at completion. At phase close the ledger collapses to one line and the board resets.

**Last updated:** 2026-08-20

---

## Current State

**★★ 2026-08-20 — THE LIMITATION IS THE SAME IN ALL THREE CLASSES, AND IT IS THE DATA.** antiSMASH
confirms both new classes de novo (**PKS 0.040**, **TERPENE 0.065**, controls **0.000**, ceilings
0.980 / 1.000). And in every class the model produces **only the simplest member and never the
harder one**: PKS **T3PKS 8/8, T1PKS 0/8** (p=0.041) · TERPENE **precursor 13/13, cyclase 0/13**
(p=0.0024) · RIPP `RiPP-like` 7/7, specific subclass 0/7 (p=6.4e-06). Three rule systems, one
ceiling on complexity — so it is **not** an artefact of antiSMASH's RiPP hierarchy. PKS producing
zero T1PKS is its 59.3%-T3PKS training substrate reproducing itself. ⚠️ 8 and 13 detections are
**quote the direction, not the rate** — the contrasts are significant, the magnitudes are not
estimated. (The former ">=15-detection floor" is **withdrawn**, 2026-08-20: generation is the cheap
step, so if a contrast is n.s. the answer is to generate more, not to appeal to a threshold.)

**★ 2026-08-20 — THE METHOD TRANSFERS.** A class-specific LoRA reaches significance **de novo** on
**both** new classes (PKS and TERPENE, p=1.5e-07 each vs a pooled 0/400), with novelty gates clean
and `JOINT_PASS` == `on_class`. RIPP's [P3-A0] is no longer a single-class result. **But both land at
only ~7% of their own ceiling** — transfer of significance is not transfer of competence.

**2026-08-19: the board now carries THREE targets.** Phase 6 (**PKS**) and Phase 7 (**TERPENE**)
opened as their own phases, each with its own pre-registration. **RIPP work is PARKED IN THE WINGS,
not dropped** (user, 2026-08-19) — see *Parked: RIPP* below for exactly what is held and why.

| phase | target | state |
|---|---|---|
| **6** | **PKS** | ✅ **[P6-A0] SIGNIFICANT de novo — 14/200 = 0.070 vs 0/400, p=1.5e-07** (ceiling 0.920). `max_orf_aa` 633 vs real 673. ⚠️ T3PKS-dominated substrate; ⚠️ 27.5% degenerate records. |
| **7** | **TERPENE** | ✅ **[P7-A0] SIGNIFICANT de novo — 14/200 = 0.070 vs 0/400, p=1.5e-07** (ceiling 0.980). ⚠️ `n_class_domains`>=2 is 0/200 — RIPP's single-marker limit reproduces. |
| 5 | RIPP | ⏸️ **PARKED** — Level 2 achieved and defensible; open items listed below. |

⚠️ **[P6-A0] IS T3PKS-DOMINATED AND MUST BE DESCRIBED AS SUCH.** Of the 3,906 training records the
1B kept, **59.3% carry a chalcone synthase (T3PKS — a single ~350-aa gene)** and only **31.3% a
ketosynthase (T1PKS-type modular)**; median kept record 1,167 nt. Required phrasing and the
forbidden claim are pinned in `docs/phase6_PKS_preregistration.md` §2.2.

⚠️ **Per-class metric policy is now machine-readable** — `config/class_eval_policy.yaml` pins each
class's window, antiSMASH `--minlength`, and which metrics are void. `scripts/novelty_battery.py`
loads it, warns on a window mismatch, prints a `⛔ VOID` banner and stamps the policy into every
scored file. Void today: `bio_span_frac` and `n_class_domains` for **PKS**; `subclass_specificity`
for **TERPENE**.

⚠️ **No cross-phase number comparisons.** Three scoring axes differ by design — class marker set,
window (PKS **4,000** · TERPENE **2,000** · RIPP 2,000) and antiSMASH `--minlength` (TERPENE **200**,
else the 1,000 default). Cross-class reading is of **shape** — does an intervention move the same
direction — never of magnitude.

---

## Parked: RIPP — held, not dropped (2026-08-19, user)

| item | why it is parked, and what unparks it |
|---|---|
| **[P5-REGEN]** regenerate the five duplicated Phase-4/5 arms | ~611 generations + antiSMASH, overnight. Buys back the 8 kb WIDE contrast and a real `JOINT_PASS`; changes **no claim anyone will build on**. Unpark if the WIDE question is reopened or a write-up needs the 8 kb contrast at full power. |
| **[P5-BIOTRANS]** `bio + transport` training arm | ⚠️ **explicitly NOT dead** (user). Needs a re-stream of the 185 GB tar (`asdb5_core_records.jsonl` holds only strict and wide sequences). DEFINING-gene coverage 0.687, between STRICT 0.869 (works) and WIDE 0.576 (fails); 55.5% fits the 1B and only 58.4% of real RIPP regions have a transporter at all, so its ceiling is **0.584, not 1.0**. |
| **[P5-FILTER]** post-generation filtering | Selection only, never ranking — nothing we own ranks within positives. |
| **Write up Level 2** | Ready. The RIPP result is complete and controlled. |

**The RIPP claim as it stands, unchanged by the pivot:**
> *We generate short DNA that antiSMASH annotates as **RiPP-like biosynthetic gene clusters**, at
> ~15% of the real-core rate, novelty-verified at DNA and protein level, class supplied by the
> adapter rather than the seed.* Not "full BGCs" — training is on the biosynthetic **core**.

---

## Phase 5 (RIPP) — retained detail

Phase 5. Target **RIPP**, substrate **Evo2 1B**. **Level 2 is achieved and defensible. The precursor
line is dropped. The remaining gap is now stated in antiSMASH's own terms.**

⚠️ **READ FIRST — 2026-08-19 data-integrity correction.** The fan-out that produced the **Phase-4/5**
seeded arms wrote **four byte-identical copies** of the same units (`bugs.md`: `seed_generate.py`
has no shard argument). Effective n was **47–141, not 188**. Rates are essentially unchanged;
**n, CIs and p-values were not.** Consequences: the WIDE refutation now holds at **one window
(Holm p=0.021), not two** (8 kb fell to p=0.15, n.s.); `JOINT_PASS` = 0 on those arms was an
artefact and is **UNMEASURED**; the subclass gap rests on **7 unique generated detections, not 28**.
**Every Phase-3 set was audited and is CLEAN** — A0, the controls, the seed sweep and all of
Stage 2 are unaffected, so the Level-2 claim below stands. `scripts/novelty_battery.py` now refuses
to score a duplicated set.

**The claim, in plain language:**
> *We generate short DNA sequences that antiSMASH annotates as **RiPP-like biosynthetic gene
> clusters**, at ~15% of the rate for real held-out cores, with novelty verified at DNA and protein
> level, and with the class supplied by the adapter rather than the seed.*
⚠️ **Not "full BGCs"** — we train on the biosynthetic **core** (median 2,191 nt of a ~21,900 nt
region), so generations contain no transport, regulatory or resistance genes by construction.

**Established.** de novo p=0.0054 vs 0/400 · seeding lifts ~6× to **antiSMASH-corrected 0.116**
against a **0.760** real-core ceiling and **0.000** base floor · class-specific at **p=2.5e-11**
(general adapter 0/188 on the same seeds) · seed *content* irrelevant (shuffle p=0.66) · novelty
clean on both gates · L\*=8 nt is where the model generates rather than reconstructs (0/8 vs 12/12
source-domain match at 500 nt).

**The remaining gap — `subclass_specificity`.** Of detections, real cores get a **specific** RiPP
chemistry **0.909** of the time (30/33 detected sequences: lassopeptide, lanthipeptide class i–v,
thioamitides, azole-containing-RiPP); **our arms 0.000** — all 7 unique detections were the generic
`RiPP-like` (Fisher p≈1e-5; ⚠️ n=7 — the direction is established, the rate is not). The model trips the loose generic rule and never the
tight domain combination a subclass requires. This supersedes `n_class_domains ≥ 2`,
`bio_span_frac` and the precursor panels, each of which failed validation.

**Closed negative:** leg 3 inference pruning (powered — no instrument: ladder 0.575, class probe
0.337 for within-positives discrimination) and [P4-WIDE] span widening (**Holm p=0.021 at 2.2 kb on
unique records; the 8 kb contrast is n.s. after deduplication**, and the dilution mechanism was
retracted as partly circular — so WIDE closes on one window with an unknown cause).

**Next:** ✅ product specificity reported in full antiSMASH mode and ✅ `subclass_specificity`
pre-registered as a declared secondary (§9.2). **The open decision is whether to spend GPU
regenerating the five duplicated Phase-4/5 arms to restore n=188** (~611 generations + antiSMASH,
overnight) or to leave them at effective n with the correction documented. See NEXT STEPS.

---

## ✅ RESOLVED STRATEGIC QUESTION (asked 2026-08-18, answered 2026-08-19)

**Was: "the model was never shown clusters, so it cannot be failing to generate them."**
**Answer: tested and REFUTED.** Widening the training span made the model **significantly worse**
(WIDE vs a size- and cluster-matched control: Holm p=4.1e-04 at 2.2 kb, 3.2e-05 at 8 kb), and the
training-set size drop cost nothing (p=0.79). Wider spans are closed. The reasoning below is kept
for the record.

*Original text:*
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
| **P4-W1** | **WIDE adapter, seeded L=8** | antiSMASH-corrected | **141 / 47** ⚠️ | **0.028** (2.2k) · **0.000** (8k) | ⛔ **WORSE at 2.2k (Holm p=0.021); 8k now n.s.** | 2026-08-19 |
| **P4-W2** | **STRICT size+cluster matched** | ″ | **47** ⚠️ | **0.043** (2.2k) · **0.085** (8k) | control — isolates span width | 2026-08-19 |
| P4-SF | STRICT-full regenerated @8 kb | ″ | **47** ⚠️ | **0.128** | best arm; gen length n.s. | 2026-08-19 |
| ~~P4-DILUTE~~ | ~~biosynthetic fraction of training span~~ | paired, n=250 | 250 | ~~STRICT 0.683 vs WIDE 0.477~~ | ⛔ **RETRACTED — measure was circular**; honest gap 0.869→0.576 | 2026-08-19 |
| **P5-SUBCLASS** | **product specificity, full antiSMASH** | `subclass_specificity` | 33 real / **7 gen** ⚠️ | real **0.909**; generated **0/7 — all `RiPP-like`** (Fisher p≈1e-5) | ★ **the remaining gap, in the field's own terms** | 2026-08-19 |
| P5-AB | `--minimal` vs full mode, identical seqs | `is_bgc` | 10 | 8/10 vs 8/10 — **100% agreement** | ✅ no prior number retracted | 2026-08-19 |
| ⚠️ **P5-DEDUP** | **fan-out shard-collision audit** | effective n of every generation set | 68 sets | **5 Phase-4/5 sets 4x-duplicated; all Phase-3 sets CLEAN** | ⛔ **WIDE refutation halved; guard added to the scorer** | 2026-08-19 |
| **P5-CLASSPROBE** | **cross-class substrate comparison** | multi-domain / multi-gene content of real cores | 50/class | **catalytic units: PKS 0.300 · RIPP 0.080 · TERPENE 0.140.** ⚠️ **multi-GENE: 0.060 · 0.060 · 0.080 — level** | ★ **no class has multi-gene structure in the strict core; PKS's edge is INTRA-genic** | 2026-08-19 |
| **P6-A0-train** | **PKS strict adapter** | training convergence | 3,906 rec | 732 steps = 3 ep, loss 0.794→0.753, **best val 0.8635** | ✅ trained; ⚠️ 59.3% T3PKS / 31.3% T1PKS-type | 2026-08-20 |
| **P7-A0-train** | **TERPENE strict adapter** | ″ | 10,658 rec | 1,998 steps, loss 0.843→0.766, **best val 0.8417** | ✅ trained; ⚠️ **at the `--max-steps 2000` cap** | 2026-08-20 |
| **P6-A0** | **PKS class LoRA, de novo** | `best_bio_bits`>0 @ PKS, w4000 | 200 | **14/200 = 0.070** vs 0/400 (ceiling 0.920) | ✅ **SIGNIFICANT p=1.5e-07** | 2026-08-20 |
| **P7-A0** | **TERPENE class LoRA, de novo** | ″ @ TERPENE, w2000 | 200 | **14/200 = 0.070** vs 0/400 (ceiling 0.980) | ✅ **SIGNIFICANT p=1.5e-07** | 2026-08-20 |
| P6/P7-C | base 1B + general adapter, both classes | ″ | 200 each | **0/200 in all four control arms** | floor | 2026-08-20 |
| **P6-AS** | **antiSMASH on PKS A0** | corrected rate · product type | 14 pos + 100 neg | **0.040** vs 0.000 controls, ceiling 0.980; **T3PKS 8/8, T1PKS 0/8** | ✅ confirmed; ★ only the easy type | 2026-08-20 |
| **P7-AS** | **antiSMASH on TERPENE A0** | ″ | 14 pos + 100 neg | **0.065** vs 0.000, ceiling 1.000; **precursor 13/13, cyclase 0/13** | ✅ confirmed; ★ only the easy type | 2026-08-20 |
| P5-PREC | precursor detector sensitivity | antiSMASH RODEO motifs | 12 + 12 | mixed subclass **8%**; module-covered **50%** | ⛔ **too low to gate — precursor line dropped** | 2026-08-19 |

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

## WHERE WE ARE — and the next steps (2026-08-19)

**The defensible claim, in plain language:**
> *We generate short DNA sequences that antiSMASH annotates as **RiPP-like biosynthetic gene
> clusters**, at ~15% of the rate for real held-out cores, with novelty verified at both DNA and
> protein level, and with the class coming from the adapter rather than the seed.*

⚠️ **NOT** "we generate full biosynthetic gene clusters." We train on the **biosynthetic core only**
(median 2,191 nt of a ~21,900 nt region), so generations contain **no transport, regulatory or
resistance genes** by construction — and the detections are **exclusively antiSMASH's generic
`RiPP-like` catch-all**, never a specific subclass.

### THE SUBCLASS-SPECIFICITY GAP — the honest limitation
antiSMASH detects RiPPs with a **hierarchy of rules**: tight subclass rules (lanthipeptide-class-i…v,
lassopeptide, thiopeptide, sactipeptide…) each requiring a **specific combination** of domains, and
a loose generic **`RiPP-like`** rule that fires on weaker evidence when no subclass rule matches.

| | products called |
|---|---|
| real cores (33 detected) | lassopeptide 7 · RiPP-like 4 · lanthipeptide-class-iv 4 · class-i 3 · class-iii 3 · redox-cofactor 2 |
| **our best arm (12 detected)** | **RiPP-like 12 — nothing else** |

⇒ **~70% of real detections get a specific chemistry; 0% of ours do.** The model produces enough
signal to trip the loose generic rule and never the tight combination a subclass requires. This is
the same limitation `n_class_domains` was groping at, stated in the field's own terms — and it is
the thing to report and to target.

### What was SCRAPPED today, and what survived

| artifact | verdict |
|---|---|
| keyword-built **precursor panel** (81 families) | ⛔ SCRAPPED — ~half enzyme; quarantined to `DEPRECATED_component_panels.json` |
| real-core precursor ceiling; **P+E / P+E+T** counts | ⛔ SCRAPPED — inflated / partly tautological |
| transport, regulator, protease panels | ⚠️ PROVISIONAL — same unvalidated keyword method |
| **"1.43× dilution"** as WIDE's cause | ⛔ RETRACTED — measure was partly circular; real gap 0.869→0.576 |
| `n_class_domains ≥ 2` as a gate | ⛔ DEMOTED to diagnostic — only ~16% of *real* cores reach it |
| precursor as a general BGC component | ⛔ WRONG — RiPP-specific; NRPS/PKS have none |
| `ENZ` = `OBLIGATE_DOMAINS[RIPP]` | ✅ KEPT — data-derived |
| `ripp_components.jsonl` (27,171 regions) | ✅ KEPT — raw antiSMASH annotation |
| **WIDE failed** (Holm p=4.1e-04 / 3.2e-05 vs matched control) | ✅ KEPT — experimental, mechanism now uncertain |
| **generations produce zero precursors** | ✅ KEPT — zero on a superset panel implies zero on the subset |
| **Level 2: antiSMASH-confirmed 0.116 vs 0.760** | ✅ KEPT — the result |

### NEXT STEPS, in order

0. ⚠️ **[P5-REGEN] DECIDE: regenerate the five duplicated Phase-4/5 arms, or report at effective n.**
   The blocking question, because everything downstream inherits it.
   - **Cost:** ~611 additional unique generations (SF 141, W2_seeded 141, W2_8k 141, W1_8k 141,
     W1_seeded 47) plus antiSMASH on the new Pfam-positives and a negative sample — an overnight
     pipeline comparable to the original.
   - **Buys:** the 8 kb WIDE contrast back at full power, a real `JOINT_PASS` for those arms, and a
     a larger `subclass_specificity` denominator. At the measured detection rates, n=188 per arm
     yields ~12–16 detections for SF and ~16 for W2.
   - ⚠️ **Vary `--seed` per shard** (`bugs.md`). `seed_generate.py` still has no `--shard i --of N`.
   - **Does not buy:** any change to the Level-2 claim, which never depended on these arms.
1. ✅ **[P5-REPORT] DONE 2026-08-19** — all five arms scored in FULL antiSMASH mode
   (`phase5_detect/full_arms/`, output dirs retained), product specificity reported, and the whole
   set recomputed on unique records. Detection is unaffected by `--minimal` (100% agreement, n=10).
2. **[P5-BIOTRANS] One training arm: `bio + transport` spans.** The only expansion worth testing —
   DEFINING-gene coverage 0.687, between STRICT's 0.869 (works) and WIDE's 0.576 (fails), median
   6,578 nt with 55.9% fitting the 1B. Adds a functionally real component at the smallest cost.
   ⛔ Do **not** try "everything except `none`" (0.595 defining, 16,893 nt, 19.5% fit) — no better
   than WIDE on coverage and far worse on length.
3. ✅ **[P5-SUBCLASS] DONE 2026-08-19** — `subclass_specificity` adopted as a **declared secondary**
   (the primary endpoint is unchanged, Standing Constraint 4) and pre-registered in
   `docs/phase3_preregistration.md` **§9.2**, with its scoring config frozen, its real-core
   reference (**0.909**) stated. (Its power floor was withdrawn 2026-08-20 — §9.2 amendment.)
4. **[P5-FILTER] Post-generation filtering** — legitimate now *as a selection step only*
   (antiSMASH pass/fail, the phage-paper funnel). ⛔ Not as ranking: nothing we own ranks within
   positives (ladder 0.575, class probe 0.337).
5. **Write up Level 2.** The result is complete and controlled; further metric work has hit
   diminishing returns.

⛔ **DROPPED:** precursor-based endpoints (detector caps at 8–50%, and precursors are RiPP-specific
rather than a general BGC component), `n_class_domains ≥ 2` as a gate, WIDE and wider spans.

## Backlog — cross-phase, opened 2026-08-20 (user)

> **ORDER (user, 2026-08-20): [X3] GenomeOcean is the NEXT thing we try.** [X1] is a bug fix that
> can land alongside it. **[X2a–d] are HELD** — do not start the subclass interventions until
> GenomeOcean has reported, because [X3] may reattribute the whole [X2] finding from "the method"
> to "Evo2-1B's 8,192 context", and every [X2] intervention is designed against the wrong cause if
> it does.

### [X1] ⛔ THE MODEL EMITS NON-NUCLEOTIDE BYTES, AND OUR EOS IS 5 TOKENS WHEN IT COULD BE 1
**Three separate defects that compound.**

1. **A fine-tuned class adapter emits a non-ACGTN byte at ~7e-05/token (RIPP) to 1.5e-04 (PKS)** —
   30–100x the base model. Over 8,000 tokens that truncates 43–70% of records.
2. **`|END|` is FIVE tokens** (`|`,`E`,`N`,`D`,`|` = 124,69,78,68,124) and has never once fired:
   `hit_eos` is 0/150, 0/188, 0/200, 0/200 across every arm ever generated.
3. ✅ **CONFIRMED 2026-08-20 FROM LOGITS: the stray byte IS the EOS token.** At coherent positions
   it is **EOS 13/13 = 100%, at 16x–159x uniform**. **The model has learned to terminate and we have
   been discarding the signal.** `hit_eos` must test **token id 0**, not the string.
4. **The tokenizer ALREADY HAS a single-token EOS — id 0 — and we have never trained it.**
   ⚠️ And **ids 0 (EOS), 1 (PAD) and 32 (space) ALL detokenize to `' '`**, so once generation is
   decoded to a string these are indistinguishable. Our whole pipeline reads the string.

**✅ PROVENANCE ANSWERED 2026-08-20: we never trained token 0 — EVO2 DID.** Localization test on
12 real held-out cores: `P(EOS)` at the true end vs mid-core is **40.9x for the BASE model** and
**2,100x for our adapter**. Evo2's pretraining established the token; **our fine-tuning sharpened it
51x**, which is exactly why class adapters truncate 30–100x more than base.

⚠️ **AND THIS MAKES `--junk-policy mask` WRONG FOR EOS.** `truncate` stopped where the model stopped;
`mask` scores what the model wrote AFTER it said stop. **PKS `A0` has a stop event inside its scored
window in 44.5% of records vs 2.5–5% of its controls** — treatment-loaded, opposite direction to the
original bug. **TERPENE is balanced (5.0% vs 4.5%/5.5%) and unaffected.** Interim: `<arm>_stopateos.jsonl`
reconstructs stop-at-EOS from the same generations (masking is frame-preserving, the first `N` is the
stop), and **both scorings are reported as a pair** until generation is token-id aware.

**Interventions, cheapest first:**
- **[X1e] MAKE `hit_eos` TEST TOKEN ID 0 — the cheapest real fix on the board.** It has read 0 in
  every arm ever generated while the model was stopping all along. Requires capturing ids at
  generation time (vortex returns `logits`/`logprobs_mean`/`sequences`, not ids), which is also what
  separates EOS from genuine junk and makes the mask-vs-truncate choice unnecessary.
- **[X1a] CONSTRAINED DECODING — do this first, it is ~10 lines.** Mask the logits to
  `{A,C,G,T}` (+ the EOS id once trained) before sampling. NVIDIA's own Evo2 NIM docs state only
  the 4 base tokens are meaningful in output and the rest exist for technical reasons. This makes
  the stray byte **impossible by construction** rather than filtered after the fact.
  ⚠️ Keep an unconstrained arm as the diagnostic — constraining hides the behaviour we are studying.
- **[X1b] Train the EXISTING single-token EOS (id 0), not the 5-byte string** (user, 2026-08-20).
  ⚠️ **Do not ADD a token — id 0 is already there and the model already reaches for it.** One token
  is ~5x the per-record gradient of a 5-token marker and makes [X1a] trivial (mask to 5 ids).
  ✅ **SHIPPED 2026-08-20 — and with NO FLAG** (user): `--eos-token` and `--eos-mode` are both gone.
  The real EOS (id 0) is appended **unconditionally** after tokenisation to the window carrying a
  record's true end; the 5-byte `|END|` string is retired. `eos_reserve` is 1.
  ⛔ **UPWEIGHTING DROPPED** (user): the signal was never weak — masking EOS causally restores the
  median generation length from 4,583 to 8,000. We were discarding it. Fix the reader, not the writer.
  ⚠️ Upweighting it needs a **manipulation check** — Phase 2's weighted arm consumed a run and
  returned an uninterpretable null because the treatment never landed.
  ⚠️ Literature warns EOS becomes a **self-reinforcing attractor**: once emitted the model keeps
  emitting it, so a mis-placed early EOS collapses the record. Cap the upweight and measure.
- **[X1h] DEGENERACY — what it is NOT, measured 2026-08-20.** Two plausible mechanisms tested and
  **both rejected**: (1) *"the prior context was not BGC-like"* — the pre-collapse prefix is NOT worse
  on-class (degenerate 6/55 = 0.109 vs clean 8/144 = 0.056, Fisher **p=0.22**, if anything better);
  (2) *classic repetition self-reinforcement* — degenerate records show **no more within-alphabet
  repetition than clean ones** (longest homopolymer median 6 vs 6; the BASE model is worse at median
  9). ⇒ It is an **abrupt exit from the nucleotide manifold**, not a gradual decay, which is
  *encouraging* for [X1a]: there is no repetition loop for constraining to fall into.
  ⚠️ **But at those positions `P(ACGT) = 0.000`**, so renormalising over ACGT samples an essentially
  arbitrary base. **Whether constrained decoding yields USABLE sequence there is untested and is the
  measurement to make** — generate a constrained arm and score it, do not assume either way.
- **[X1i] SNIP-AND-REPLACE (user, 2026-08-20)** — detect a degeneration/zero-length record, discard
  it, and regenerate that slot. This is rejection sampling; it is legitimate as a **selection** step,
  needs no model change, and composes with [X1a]. It is also what the phage paper did (overgenerate,
  filter hard). ⚠️ Report the rejection rate as its own row — a filtered set with an unreported
  discard rate hides the failure it was built to remove.
- **[X1d] DEGENERACY IS A SEPARATE FAILURE AND [X1a] WILL NOT FIX IT.** In **0.42% of positions**
  the model collapses to a ~uniform distribution over all 512 tokens (`P(ACGT)=0.000`) — the cause of
  the **27.5% degenerate records in PKS `A0`**. Masking logits at such a position just forces an
  arbitrary nucleotide. Needs the `n_pass` / length-quality gate, and it is the one place a
  *capability* fix (better model, more context) may be required rather than a decoding fix.
- **[X1g] ◀ SHARED PREREQUISITE: MAKE GENERATION TOKEN-ID AWARE.** [X1e] (`hit_eos` on id 0) and
  [X1f] (early stopping) are the SAME build, and [X1a] constrained decoding needs the same hook.
  vortex returns `logits`/`logprobs_mean`/`sequences` but **never the sampled ids**, and ids 0/1/32
  all detokenize to `' '`, so the decoded string cannot distinguish EOS from junk. **One change
  unlocks all three:** capture ids at sampling time, then (a) `hit_eos` tests id 0, (b) a per-row
  done-mask stops each sequence at its own EOS, (c) the junk-vs-EOS distinction that makes the
  mask/truncate choice unnecessary. **Do this before any further generation spend.**
- **[X1f] EARLY STOPPING — ⚠️ BUILT, MEASURED, AND THE SIMPLE VERSION DOES NOT WORK.**
  vortex's `stop_at_eos` checks for EOS and only `print`s (no `break`), and inspects batch row 0
  only — but fixing that is not enough. An `all(rows done)` exit gives **1.01x**, because only
  **21/32 rows emit EOS at all** and one non-terminating row holds the whole batch.
  ⇒ **Per-row exit is worth building: 38.5% of decode compute** (157,490 tokens needed vs 256,000
  paid). EOS position median **2,869**, min 623, max 7,648. Needs vortex's cached
  `inference_params` rebuilt for surviving rows.
  ⇒ **Cheaper approximations first:** shorten `--max-new-tokens` (median EOS is 2,869 of 8,000
  requested), and use **smaller batches** — waste scales with the wait on the slowest row.
- **[X1c] Filter prematurely-ended sequences at the selection stage** (user's earlier idea). Cheap,
  legitimate as selection, and the phage paper used a plain length filter rather than a stop token.

### [X2] ⛔ ONLY THE SIMPLEST SUBCLASS — the finding that now defines the project
Measured in all three classes (`memory.md` 2026-08-20): **PKS T3PKS 8/8 and T1PKS 0/8** (p=0.041) ·
**TERPENE precursor 13/13, cyclase 0/13** (p=0.0024) · **RIPP `RiPP-like` 7/7, subclass 0/7**
(p=6.4e-06). Three different antiSMASH rule systems, one ceiling.

⚠️ **BEFORE calling this a model property, close the CONTEXT CONFOUND.** The harder member is in
every case the LONGER one, and the 1B's budget is 7,992 nt:

| class | easy member | hard member | fits the 1B? |
|---|---|---|---|
| PKS | T3PKS, median **1,083 nt** | T1PKS, median **7,665 nt** | **the median T1PKS barely fits; half do not** |
| TERPENE | precursor, median 928 nt | cyclase, median 2,009 nt | both fit |
| RIPP | generic | subclass rules need more domains | — |

⇒ For PKS the model may simply **never have seen a complete T1PKS**. That is a substrate defect, not
a capability limit, and it is testable.

**Ordered interventions — ⏸️ ALL HELD pending [X3] GenomeOcean (user, 2026-08-20):**
- **[X2a] Bigger denominators — but NOT a blocker.** All three contrasts are already **significant
  against their own controls** (p=0.041 / 0.0024 / 6.4e-06) on 7–13 detections, and generation is the
  cheap step: n=600/arm would roughly triple them. Worth doing to turn a *direction* into an
  *estimated rate*, not to rescue the finding. ⚠️ The ">=15-detection floor" is **withdrawn** (user,
  2026-08-20) — arbitrary where sampling is cheap.
- **[X2b] Seeded hard-subclass positive control.** Seed from a real T1PKS / cyclase exemplar at
  L\*=8. Phase 3 showed seeding lifts ~6x. **If seeded generation still yields 0 hard-subclass, the
  limitation is real; if it does not, it was the prior, not the capability.** This is the single
  most informative cheap experiment on the board.
- **[X2c] Inverse-subclass-frequency upweighting** (user, 2026-08-20). Reweight training by rarity
  of the subclass. ⚠️ **Do not run before [X2b]** — the PKS gap is 31.3% of training records
  carrying a ketosynthase producing 0% of output, which is far worse than frequency alone predicts,
  so reweighting may be aimed at the wrong cause.
- **[X2d] Subclass-conditioned adapters** — a T1PKS-only adapter (~1,200 records). The honest route
  to any modular-PKS claim, and it also tests whether one adapter per *subclass* recovers what one
  per class does not.

### [X3] ⛔ TEST GENOMEOCEAN ON TERPENE — the model-vs-method question is now the binding one

#### [X3] TASK BREAKDOWN — ordered, with what is already done

**Already built (2026-07-27, do not redo):**
- ✅ **Weights local** — `GenomeOcean-4B` and `GenomeOcean-4B-bgcFM` in `hf_cache`; env at
  `/data2/ds85/envs/genomeocean`.
- ✅ **Leakage gate PASSED** — `smc_leakage.json`, containment **0.0000** across 48 true + 48
  mismatched. This is the gate that would have disqualified the whole track; it is clear.
- ✅ **Fine-tune feasibility PASSED** — `finetune_feasibility.json`: `MistralForCausalLM`, 4.25 B
  params, 24 layers, **`max_position_embeddings` 32,768**, and all four gates green
  (`class_token_atomic`, `embedding_resize`, `gradient_checkpointing_active`, `train_step`) at
  seq_len 10,240. **22 class tokens added atomically**, vocab 4,096 → 4,118 — GenomeOcean can take a
  real trainable class token, which Evo2 cannot.
- ✅ **Tokenization measured** — `tokenization_report.json`.
- ✅ **Zero-shot rate + class probe** — `go_zeroshot_rate_n216/`, `go_zeroshot_bgcfm/`.

**T1 · ✅ DECIDED 2026-08-20 — the arm is fine-tuned `GenomeOcean-4B`; `bgcFM` is a reference only.**
Settled on a measurement: the class probe reads **0.878 (base) vs 0.894 (bgcFM)**, chance 0.091 —
bgcFM's extra BGC pretraining buys **+0.016**, i.e. essentially nothing representationally, while
adding the confound "it already saw BGCs". ⇒ base `GenomeOcean-4B` keeps the comparison
fine-tune-vs-fine-tune; `bgcFM` runs zero-shot as a **declared reference ceiling** (its
unconditioned `is_bgc` is already measured at **27/216 = 0.125**).
★ **And all three models encode class at ~0.88–0.91 vs ~0.09 chance** (Evo2 ~0.91). **The bottleneck
has never been representation — it is generation.** That predicts a model swap alone may not move
the endpoint, and it is recorded before the run rather than after.

**T2 · ✅ DONE 2026-08-22** (`phase8_TERPENE_GO/data/`, `substrate_report.json`).
`splits_class/TERPENE` reused unchanged. Measured: **4.974 nt/token**, median record 960 nt =
**192 tokens**; context 10,240 tok = **50,934 nt = 6.4x** Evo2's 7,992. Kept **11,260/11,297 =
0.997** train (Evo2: 0.943) and 788/793 val ⇒ **+602 records recovered (+5.3%)**.
★ **The 6.4x context buys +5.3% — which is the design working, not a letdown**: TERPENE was chosen
because Evo2 was not context-limited on it, so **a GenomeOcean win cannot be attributed to
context.** 37 records still exceed even the 32,768 ceiling (largest ~270 kb) and are dropped and
counted. Context **frozen at 10,240**. ✅ Tokenizer auto-wrap `BOS=1 … EOS=2` asserted in code.

**T3 · Confirm the EOS/class-token handling.** GenomeOcean's tokenizer **auto-wraps every sequence
`BOS=1 … EOS=2`**, so `[X1b]` is free here. Verify the class token survives fine-tuning as one
atomic id (already gated true) and that EOS lands in the training targets.

**T4 · Fine-tune.** Match the Evo2 recipe wherever it is meaningful — LoRA, 3 epochs, same data —
and record every axis where it cannot match (parameter count, tokenizer, context, optimizer).
⚠️ 4.25 B params vs 1.1 B: **this arm is not parameter-matched and must never be reported as if it
were.**

**T5 · Generate the three pre-registered arms**, n=200, **the same 200 `eval_prompts.jsonl`
prompts** as `[P7-A0]` so the comparison is prompt-paired: class adapter · base GenomeOcean ·
(optional) bgcFM zero-shot.

**T6 · Score through the IDENTICAL pipeline** — `novelty_battery.py --cls TERPENE --window 2000`,
then full-mode antiSMASH at `--minlength 200`. Same ceiling (`real_TERPENE_fit50_w2000.json`), same
floors, same gates. **Any pipeline change invalidates the comparison.**

**T7 · Report as a two-model table with `n_pass` and product type as rows.** The question is not
only "higher rate" but **"does it make the harder member"** — TERPENE cyclase vs precursor-only,
where Evo2 read **0/13**.

**T8 · ✅ DONE 2026-08-20 — `docs/phase8_GENOMEOCEAN_preregistration.md`** (user approved the new
file). Freezes the inherited endpoint, the arms, n=200 on the **same prompts as `[P7-A0]`**, the
five declared confounds, and a **two-way kill criterion**: an indistinguishable rate with still-zero
cyclase detections makes the "easiest member" limit method-attributable and **un-holds [X2]**;
cyclase detections where Evo2 had none make it model-attributable and aim [X2] at the wrong cause.

⚠️ **What a GenomeOcean win would and would not settle.** Context, parameters, tokenizer and
pretraining corpus ALL differ at once. A win says "this model does better"; it does **not** isolate
which axis did it. **TERPENE is chosen precisely to keep context out of the explanation** — 94% of
TERPENE already fits Evo2, so a win there cannot be attributed to the 6.4x context. That is what
makes TERPENE the right first class and PKS the wrong one.


Three classes, one model. Every finding above is **confounded with Evo2-1B**. GenomeOcean is the
control that separates them, and two facts make it the right instrument rather than merely a
different one:
- **Context: 10,240 BPE tokens ≈ 51,200 nt — 6.4x our 7,992 budget.** The median T1PKS (7,665 nt)
  fits trivially; so does a whole antiSMASH region (median 21,896 nt), which the 1B hosts for only
  1.9% of records. **It directly dissolves the [X2] context confound.**
- **It is bacterial-only and has a BGC-specific variant** — `bgcFM`, trained on 12M BGC sequences
  from SMC, reported to generate long BGC sequences unprompted. We already hold a
  `go_zeroshot_bgcfm` run dir.

**Scope it tightly** (Standing Constraint 3 — testing does not fan out across models): **ONE class,
ONE arm, the same pre-registered endpoint.** Recommend **TERPENE** — highest ceiling (antiSMASH
1.000), 94% context fit so the 1B is not handicapped, and its easy/hard split (precursor vs cyclase)
is the cleanest of the three. **PKS is the tempting choice and the wrong first one**: its context
confound means a GenomeOcean win there would be uninterpretable between "better model" and "longer
context".

## Backlog — Phase 3

**The phase had three legs.** Leg 1 ✅ significant · Leg 2 ✅ significant, class-specific · Leg 3 ⛔ closed (no instrument).

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

### [P4-WIDE] WIDE_KINDS fine-tune — ⛔ **REFUTED 2026-08-19** (Holm p=4.1e-04 / 3.2e-05)
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
| **STRICT-matched** | **3,723** | **6.96 M** | 1,209 | ✅ done |
| WIDE | 3,723 | 13.69 M | 3,714 | ✅ done — REFUTED |

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

### [P3-B2a] Pruning DURING generation (guided decoding) — ⛔ **CLOSED 2026-08-18, measured**
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
