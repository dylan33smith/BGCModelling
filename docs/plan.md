# plan.md — the dashboard

**Read at session start. This file holds only the active phase.** Completed interventions keep a
one-row summary in the Phase Ledger for the rest of the phase; their full write-up goes to
`memory.md` at completion. At phase close the ledger collapses to one line and the board resets.

**Last updated:** 2026-09-01

---

## Current State

**★★★★ 2026-09-01 — IT IS NOT LENGTH. IT IS HOW MANY GENES THE TARGET'S RULE REQUIRES.**
Across 11 subclass arms in 3 classes, **two orthogonal effects with two different predictors**:
**rule gene-count → specificity** (r = **−0.895**, p = 0.0002; partial −0.891) and **target length →
detectability** (r = **−0.822**, p = 0.0019; partial for specificity **+0.028**, i.e. nothing).
⇒ Mean specificity by genes the rule needs: **1 gene → 0.958** · 2 → 0.684 · 3 → **0.115**.
★ **hglE-KS settles it**: the LONGEST target (11,734 nt), one-domain rule, specificity **1.000**.
★★ **THE SINGLE-GENE CEILING IS NOW DIRECTLY OBSERVED, not inferred.** **62/65** azole detections
carry a **YcaO** domain — the azole rule's anchor — and get called `RiPP-like` because the *partner*
gene is missing. On a matched denominator the two arms make their anchor at a comparable rate
(**T1PKS 9.5%** of generations carry the full signature vs **azole 6.2%** carry YcaO); the entire
outcome difference is **whether the anchor ALONE satisfies the rule**.
⛔ **THREE CORRECTIONS THAT WEAKEN OUR CLAIMS, recorded together:**
(1) the real-core `n_bio_orfs` reference has been **too low all project** — per-subclass real cores are
**1.79–3.38**, not 1.454, so the multi-gene gap is **~2× larger** than reported;
(2) our long-target "successes" are **fragments** — T1PKS and hglE-KS generate their longest gene at
**0.63–0.65** of the real one, a truncated megasynthase that still trips the rule;
(3) **antiSMASH is a SIGNATURE DETECTOR, not a cluster validator** — a detection means the required
domains are present, never that the cluster is complete. ⇒ **"Generates the subclass" is not
supportable; "generates a gene carrying the subclass's defining signature" is.**
⚠️ **Every subclass finding is GenomeOcean-4B ONLY** (Evo2 was class-level, phases 3–9).
⚠️ `paper.md` is flagged **under revision** — its hypothesis predates this.


**★★★★ 2026-08-27 — IT IS NOT A SAMPLING PROBLEM, AND THE RANK HYPOTHESIS IS DEAD.**
`[P13-EVL-likelihood]` teacher-forced real held-out clusters through the adapters — the first
measurement in this project that is not a generation metric. **The azole adapter beats BASE on real
azole by only +0.0352 bits/nt (t(44)=2.49, better on 28/45 records); cyclactone's beats base by
+0.1341 and beats the WRONG adapter on 34/34.**
⇒ ⛔ **If azole failed only because its sampler goes elsewhere, it would still MODEL real azole well.
It does not.** Decoding interventions — guided decoding, beam search, rejection sampling — are not
the fix; that branch of the backlog is closed.
⇒ ✅ **Conditioning still lands at the likelihood level** — each adapter is the best model of its own
target, both significantly. Azole's is **~3x weaker and patchier** (35/45 vs 34/34).
⇒ ★★★ **THE DIAGNOSIS IS MEMORISATION WITHOUT GENERALISATION:** a **1.43-nat train/eval gap** (train
3.906 vs eval 5.331, eval *rising* after step 400) while gaining almost nothing on held-out records.
**That is a DATA limit, not an adapter-capacity limit** — and it points straight back at 14,266
phage training examples vs our **794** for a target of the same size.
⇒ ⛔ **`P13-TRN-lorarank` RE-SCOPED to a control.** We are **past** the LoRA knee, not short of it,
and **the arm that WORKS (cyclactone) overfits hardest of the three** — fit quality does not separate
success from failure here at all. ⚠️ Premise correction: the subclass adapters are **GenomeOcean-4B
(4.31 B)**, not Evo2-1B; 1.287% = 55.4 M trainable.

**★★★★ 2026-08-27 — THE PHAGE PAPER'S FIDELITY DIAL WORKS, AND IT DOES NOT BUY THE CHEMISTRY.**
`[P13-TRN-azolebucket]` ported their `+∼` identity-bucket conditioning onto azole — a second atomic
token after the class token, on `[P10]`'s exact records and byte-identical hyperparameters and
decoding, so **the bucket token is the only delta**.
⇒ ★ **The dial LANDS, hard:** `antismash_detection_rate` **0.280** at `[ID_80_95]` vs **0.010** at
`[ID_00_50]` — **p=1.78e-16, a 28x swing inside one adapter** — and **4.3x `[P10]`'s 0.065**
(p=3.9e-16). `JOINT_PASS` 0.133 → 0.460. Length fidelity 0.54x → **0.77x**.
⇒ ⛔ **NOT a length artefact** — the elevation holds in EVERY length stratum (<3 kb 0.175 vs 0.011,
p=2.5e-04; 3–4.5 kb 0.212 vs 0.029, p=0.025; >4.5 kb 0.355 vs 0.000, p=2.2e-10).
⇒ ★ **It controls NOVELTY as designed:** `protein_aai` ≥0.98 **5/400** high-fidelity vs **0/400**
low. ⚠️ **`containment` never fires — 0 FAIL, 0 even at WARN, in all six arms.** DNA novelty stays
clean while protein novelty degrades: **Standing Constraint 1's blind spot, observed live.**
⇒ ⛔ **AND THE TARGET CHEMISTRY DOES NOT MOVE.** Pooled high-fidelity **3/400 = 0.0075** vs `[P10]`
2/1000 = 0.002, **p=0.144 n.s.**; against its own ceiling **p=1.4e-58**. Collapse to bare generic
`RiPP-like` persists **36/37 and 54/56**. `n_bio_orfs` 1.093 vs `[P10]` 1.101 against a **1.454** ⚠️[too low — real azole cores are **3.378**]
ceiling — **4.3x the on-class DNA bought ZERO extra genes.**
⇒ ★★★ **AZOLE's boundary survives its most serious challenge: it is not a conditioning artefact,
and there is no fidelity price at which its chemistry can be bought.**
⚠️ **CORRECTED 2026-08-27 — this read "the `[P11]` compositional boundary" until `[P12-TRN-secondclass]`
(parallel session, same day) corrected `[P11]` itself.** That law was measured on
`own_subclass|detected`, whose denominator is class-dependent — RIPP has the generic `RiPP-like`
escape hatch, TERPENE and PKS have none — and over 11 arms it reads **r = -0.186**, with **PKS T1PKS
at 0.905 on a 7,594 nt target.** ✅ **This phase's result is untouched**: its primary was
`own_subclass_rate_all`, denominated on ALL generated records, never on detections.
⛔ **But one reading of it weakens:** the 28x detection swing is almost entirely generic `RiPP-like`
(36/37, 54/56) — the very escape hatch `[P12]` flags. Do not sell it as moving subclass competence.
**It makes the flat azole rate more damning, not less: the dial bought 28x more of what does not count.**
⚠️ Power: CI upper 0.022 vs a 1.000 ceiling — a jump to ceiling is excluded, **a modest rise to ~2%
is not.** ⚠️ `p13_nobucket` is off-distribution and itself below `[P10]` (p=0.0115), so the
load-bearing contrast is **high-vs-low bucket within this adapter**, which is internally controlled.
⚠️ **Gate T0 FAILED on DNA first** — `ani_to_ref` median **0.0000**, 76.5% of azole records have zero
alignable nucleotide identity to the medoid. ⇒ ***Microviridae* is a taxonomic FAMILY; an
"azole-containing RiPP" is a CHEMICAL annotation over unrelated clusters.** Their axis does not exist
in our data at the nucleotide level; on protein it does, but **bimodally** (40/115/14/2/628).

**★★★ 2026-08-27 — THE PHAGE PAPER SOLVED A DIFFERENT PROBLEM, AND TWO OF OUR ELIMINATIONS ARE
NARROWER THAN THEY READ.** `[P13-ANL-phagegap]` read Hie et al. Methods directly. Their prompt was
**`+∼`**, where `∼` is a trained token meaning **"95–100% nucleotide identity to ΦX174"**, and their
viable designs are **93.0–98.8% identical to a TRAINING sequence** (67–392 mutations each) under a
*soft* <95%-AAI preference applied to a curated shortlist. ⇒ **Their multi-gene 5.4 kb coherence is
largely INHERITED FROM A TEMPLATE, not composed** — and our `containment` ≥0.95 gate, applied
per-record to everything, is a different task. ⚠️ **Azole (6,293 nt) is LONGER than ΦX174 (5,386
nt)** — length is not why they win.
⇒ The engineering gaps, in order: **18.0x more unique training examples** (14,266 vs 794) and
**76.6x more genome-equivalents seen** (608,392 vs 7,940); **full-parameter SFT of a 7B on 16–32
H100s vs LoRA r=16 at 1.287% trainable**; **a template-fidelity dial we do not have**; 11x more
generations per target.
⛔ **HYPOTHESIS KILLED — within-target redundancy explains nothing.** `r(train_frac_distinct@0.80,
own-subclass rate) = +0.228`, `r(n_clusters@0.80, rate) = +0.007` vs `r(log10 nt, rate) = -0.933`
on the same five subclasses. **This STRENGTHENS the `[P11]` compositional claim** against a confound
it had never tested.
⚠️ **TWO HOLES IN `paper.md`'s ELIMINATION TABLE.** (1) "not enough data" is eliminated with
**r = -0.237 across 497–799 records — a 1.6x range**; that is a LOCAL null and does not license the
18x extrapolation. (2) **Adapter CAPACITY was never varied** — `lora_r=16` on all five subclass
adapters. Both queued as Phase 13.
⚠️ **AND A FACTUAL CORRECTION:** the phage funnel is **~36:1**, not the ~1000:1 this repo carried in
three places — ~11,000 SFT generations → 302 candidates → 285 synthesised → 16 viable. **Our n=1,000
azole pool is already ~1/11 of their entire per-target budget**, which re-scopes `[P3-B2b]`.

**★★★★ 2026-08-24 — THE LIMIT IS COMPOSITIONAL CAPACITY, NOT LENGTH CONTROL.** Forcing the azole
adapter to generate at full length (`min_new_tokens` 1270 → median **9,512 nt**, 1.51x target, up
from 3,390 = 0.54x) leaves its own-subclass rate **unchanged: 0.031 → 0.059, p=0.507**, still
collapsing to bare generic `RiPP-like` **16/17**. Against its ceiling: **p=6.2e-14**.
⇒ ★★ **The model does not fail because it stops writing too early — it fails because it cannot
assemble the domain combination the subclass requires, however much sequence it is given.**
⇒ The extra 6 kb IS filled with on-class material (primary `JOINT_PASS` **0.133 → 0.175**, detection
0.065 → 0.085) and **zero** subclass chemistry. **More length buys FILLER, not STRUCTURE** — the same
shape as the TERPENE `min_new_tokens` result.
⇒ Suppressing EOS shows there is **no second stopping point**: `hit_eos` 837/1000 → 112/200, the
model runs to the cap. Its stop is one decision, not a recoverable preference.
⚠️ **17 detections** — a jump to ceiling is excluded, a *modest* improvement is not. n=1000 would
resolve it (~1 h GPU); not run. ⚠️ Not length-matched — the claim is "length is not the lever
anywhere in **0.54x–1.51x**", not "at exactly 1.0x".

**★★★ 2026-08-24 — WHAT THIS METHOD CAN GENERATE IS PREDICTED BY TARGET LENGTH.** Five
subclass-conditioned adapters spanning **8.8x** in target length give
**r(log10 length, own-subclass rate) = -0.933**: cyclactone 715 nt → **1.000** · ranthipeptide
1,624 → **0.636** · redox-cofactor 2,191 → **0.200** · lassopeptide 2,738 → **0.417** · azole
6,293 → **0.031**. Extremes differ at **p=1.9e-48**; even adjacent points (cyclactone vs
ranthipeptide) at **p=1.2e-09**.
⚠️ **NOT monotonic** — lassopeptide sits above redox-cofactor, but on 12 and 15 detections with
overlapping CIs (**p=0.398**). **The trend is strong; the middle ordering is unresolved.**
★ **The failure mode degrades in THREE STAGES, which is the more useful finding:**
**right chemistry (715 nt) → wrong-but-REAL chemistry (1.6–2.7 kb, misses are `RRE-containing` or a
different genuine subclass) → no chemistry (6.3 kb, bare generic `RiPP-like` 62/63).** Only azole
reaches collapse, and only azole undershoots its length (**0.54x** vs the others' 0.85–1.24x).
⇒ **This is a SELECTION RULE: sub-1 kb subclasses are solved; past ~3 kb the method makes RiPP-like
DNA but not the chemistry asked for.** It predicts which of the 43 RiPP subclasses are worth a run.
⚠️ Detection rate falls with length too (0.620 → 0.165 → 0.075 → 0.060 → 0.065) — an earlier
"specificity but not detectability" claim compared inconsistent denominators and is **withdrawn**.

**★★★ 2026-08-24 — SUBCLASS CONDITIONING WORKS. THE FIRST CEILING EVER REACHED.**
`P10-TRN-cyclactone` (GenomeOcean, 661 records, `[CLS_CYCLIC_LACTONE_AUTOINDUCER]`) produces its own
subclass **124/124 = 1.000 of detections — indistinguishable from its real-core ceiling (p=1.0)**,
with **zero** falling back to the generic `RiPP-like`, and **124/200 = 0.620 of everything it
generates**. ✅ **NOT memorisation:** its generations are *more* novel than real held-out cores on
both gates (`containment` max 0.247 vs 0.456; `protein_aai` fails 2/200 vs the real cores' 2/36).
⛔ **But `P10-TRN-azole` is a clean null** — 1/63 = 0.016 against its own 45/45 = 1.000 ceiling
(p=8.2e-30), and no better than the class-level adapter (p=1.0). **The two arms differ at p=2.6e-49.**
⇒ ★ **What separates them is TARGET COMPLEXITY, not conditioning.** Cyclactone is one short peptide
(median 715 nt) and the model reproduces its length distribution exactly (708 nt, 0.99x); azole is
6,293 nt of multi-domain machinery and the model generates at **half** that (0.54x). **"Only the
simplest member" survives one level down — but given a simple enough target, conditioning now
delivers it COMPLETELY.**
⚠️ **Two instrument problems found and recorded.** (1) The Pfam endpoint is **VOID for cyclactone** —
real cores score only 2/36 = 0.056 on `OBLIGATE_DOMAINS[RIPP]`, so its 0/200 measures the instrument.
**Any future subclass arm must validate its Stage-A gate against real cores of that subclass first.**
(2) Azole is the **first arm where `JOINT_PASS` < `on_class`** (133 vs 158; 25 of 26 gate failures are
on-class hits) — small-data training memorises, so quote `JOINT_PASS`.

**★★★ 2026-08-24 — PHASE 9 CLOSES: THE SHAPE REPLICATES, THE MAGNITUDE DOES NOT.** GenomeOcean on
RIPP reproduces the TERPENE 2x2 **structure** exactly — de novo it beats Evo2 de novo (0.115 vs
0.027, **p=0.0020**), **ties Evo2 SEEDED** (p=0.471) and **gains nothing from a seed** (p=0.873).
⇒ "GenomeOcean reaches the seeded band without a seed" is now replicated on a second, independent
class. ⚠️ **But it reaches only 26.1% of its RIPP ceiling against 69.9% on TERPENE**, and the lift is
**4.3x, not 9.8x** — *"GenomeOcean roughly reaches the ceiling" was a TERPENE-only claim.*
★ **First specific RiPP chemistry ever generated** — a `ranthipeptide`, where every Evo2 detection
was the generic `RiPP-like`. ⛔ **BUT POWERED TO n=1,000 / 87 DETECTIONS THE RATE FELL to 2/87 =
0.023 strict, 10/87 = 0.115 lenient** — the 1/13 = 0.077 was small-sample inflation. The gap vs the
ceiling (0.500/0.740) is now **DECISIVE, p=1.7e-11**, superseding the earlier "uninformative" read.
★ **The real finding is RICHNESS: GenomeOcean made ONE chemistry (`ranthipeptide`, twice) across 87
detections where real cores span ELEVEN.** Not "produces subclasses rarely" — "produces one and
misses the other ten." ⚠️ vs Evo2 0/7 it is still p=1.0, and that is **Evo2's n, not ours**.
★★ **AND, among POSITIVES ONLY, the models barely differ — the model swap buys FREQUENCY, not
QUALITY** (TERPENE GO B/ceiling 0.90–1.05 on every structural metric). **TERPENE real cores are
themselves single-gene (`n_bio_orfs` 1.122) so TERPENE could never test multi-gene structure; RIPP's
1.454 ⚠️[class-level sample; per-subclass 1.79–3.38] can, and GO reaches only 0.71.** ⇒ **RIPP, not TERPENE, is the substrate for `bio + transport`.** ⛔ **And a measurement bug was found and fixed: the antiSMASH `--minlength` default
(1,000 nt) was silently rejecting 30–89% of every arm** — and 14/50 real cores — because GenomeOcean
generates at the real length distribution where Evo2 generated 4–8 kb. **Everything, ceiling
included, was re-derived at `--minlength 200`** (real-core detection 33 → **50/50**).

**★★★ 2026-08-24 — SEEDING ERASES THE MODEL GAP.** ⚠️ **"AND REVERSES IT" IS RETRACTED — the
reversal was an artefact of the broken seeded arm** (61/200 empty; `memory.md`, `[P8-T5-FIX]`).
De novo, GenomeOcean beats Evo2 9.8x (0.685 vs 0.070, p=3.4e-40). **Seeded, the two models are
INDISTINGUISHABLE** — Evo2 0.615 vs GenomeOcean 0.580, **p=0.541**; antiSMASH-corrected 0.545 vs
0.550. Seeding lifts Evo2 **8.8x** and still hurts GenomeOcean, but at **p=0.038, not 1.5e-08**.
★ **Three of the four cells cluster at 0.545–0.635 corrected — Evo2 seeded, GO seeded, GO de novo.
The only outlier is Evo2 de novo at 0.065.** ⇒ **GenomeOcean's advantage is reaching the seeded band
without a seed**, not being better in general. ⚠️ The tie is GO's *best* case: `min_new_tokens=100`
was applied only to GO's seeded arm. ⚠️ **And the cyclase claim is RETRACTED**: pooled to n=800,
Evo2 makes the harder member 3/48 = 0.062 — rare, not never — and GenomeOcean's 0.159 is **not
significantly higher (p=0.132)**. Both remain far below real cores (0.440). **The "easiest member"
limitation survived the model swap.**

**★★ 2026-08-20 — THE LIMITATION IS THE SAME IN ALL THREE CLASSES, AND IT IS THE DATA.** antiSMASH
confirms both new classes de novo (**PKS 0.040**, **TERPENE 0.065**, controls **0.000**, ceilings
0.980 / 1.000). ⚠️ **"NEVER" IS RETRACTED — read the 2026-08-24 block above first.** In every class
the model produces **only the simplest member and RARELY the harder one**: PKS **T3PKS 8/8, T1PKS
0/8** (p=0.041) · TERPENE **precursor 13/13, cyclase 0/13** (p=0.0024) · RIPP `RiPP-like` 7/7,
specific subclass 0/7 (p=6.4e-06). ⚠️ **All three zeros are at 7–13 detections.** The one that was
powered up — TERPENE cyclase, Evo2 pooled to n=800 — went **0/13 → 3/48 = 0.062**. Treat the PKS
and RIPP zeros as **untested at power**, not as established absences. Three rule systems, one
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
| **9** | **RIPP on GenomeOcean** | ✅ **COMPLETE.** ★★ **The 2x2 SHAPE replicates exactly** (GO de novo beats Evo2 de novo p=0.0020; ties Evo2 SEEDED p=0.471; gains nothing from a seed p=0.873) — the "reaches the seeded band without a seed" conclusion now holds on a class it was not derived from. ⚠️ **Magnitude does NOT transfer**: 4.3x not 9.8x, **26.1% of ceiling not 69.9%**. ★ First **specific** RiPP chemistry ever generated (a `ranthipeptide`), but ⛔ `subclass_specificity` **1/13 vs Evo2 0/7 is p=1.0 — UNDERPOWERED, not negative**. |
| **8** | **TERPENE on GenomeOcean** | ✅ **COMPLETE T1–T9 + fix.** ★ **The model gap is CONDITIONING-DEPENDENT**: de novo GO 0.685 vs Evo2 0.070 (p=3.4e-40); **seeded the two are INDISTINGUISHABLE, 0.615 vs 0.580, p=0.541** (⚠️ *"Evo2 wins, p=2.5e-05" RETRACTED* — it used the 61/200-empty arm). GO's advantage is reaching the seeded band **without a seed**. ⚠️ Cyclase claim **retracted**: Evo2 makes it 3/48 = 0.062, GO 20/126, **p=0.132 n.s.** |
| **10** | **RIPP SUBCLASS adapters** | ✅ **COMPLETE.** ★★ **cyclactone 1.000 = AT CEILING**; ⛔ azole 0.016 null. Was: 🔄 training (`P10-TRN-azole`, `P10-TRN-cyclactone`) — `[CLS_AZOLE_CONTAINING_RIPP]` (799 rec) and `[CLS_CYCLIC_LACTONE_AUTOINDUCER]` (664 rec) on GenomeOcean. Tests whether class-level conditioning was too COARSE. ⚠️ small data, not length-matched. |
| **11** | **RIPP subclass LENGTH series** | ✅ **COMPLETE.** ★★ **r = −0.933** over 5 subclasses × 8.8x length. Three-stage failure mode. Yields a **selection rule** for which subclasses are reachable. |
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

## Phase 10 queue — the subclass-specificity attack, in order (user, 2026-08-24)

⚠️ **IDs below use the CURRENT convention** `P<phase>-<KIND>-<slug>` (`terms.md`). The Phase
Ledger and everything in `memory.md` keep their ORIGINAL IDs and are read through the bridge
table there — the ledger is permanent and is never renamed.

**All three are GATED on `P10-TRN-azole` / `P10-TRN-cla` returning** — the two single-subclass adapters now training, plus
their generation, scoring and antiSMASH. Nothing below starts until that eval is in, because
the two running adapters tell us whether conditioning granularity is the lever at all.

**The problem they attack.** `subclass_specificity` is **2/87 = 0.023** (GenomeOcean) and **1/77 =
0.013** (Evo2) against a real-core **0.500**, p≈1e-11, and the two are indistinguishable from each
other (p=1.0). Both models make **one** chemistry; real cores span **eleven**. Training data is
**62.1% specific**, and the strict core retains its region's subclass **25/25 = 1.000** — so this is
neither a data-availability nor a label-provenance problem (`memory.md` 2026-08-24, `P10-ANL-subclass-dist`).

| id | intervention | what it changes | endpoint | what would FALSIFY it |
|---|---|---|---|---|
| **`P10-TRN-multitoken`** | **subclass-token routing — ONE adapter, N subclass tokens** | conditioning GRANULARITY at scale: `[CLS_<SUBCLASS>]` per record instead of one `[CLS_RIPP]`, over the top-N subclasses in a single model | `subclass_specificity` and **chemistry richness** (distinct chemistries generated, real cores = 11) | richness stays at 1 ⇒ granularity is not the lever, and the per-subclass adapters are the only route |
| ~~`P10-DAT-drop-generic`~~ | ⛔ **DROPPED 2026-08-24 (user)** — the `P11` series answered it | — | **Already falsified:** 4 of 5 subclass arms emit the generic rule **0 times** (cyclactone 0/124, ranthipeptide 0/33, redox 0/15, lasso 0/12). The attractor exists **only for azole**, and azole's problem is target length, not the training mixture. |
| ~~`P10-TRN-invfreq`~~ | ⛔ **DROPPED 2026-08-24 (user)** — the data answered it without the experiment | — | **Already falsified:** azole is the LARGER and MORE FREQUENT subclass (799 rec, 9.8%) and FAILS at 0.016; cyclactone (664 rec, 8.2%) reaches **1.000**. Matched on volume and frequency, opposite outcomes ⇒ frequency is not the lever. |

⚠️ **REPRIORITISED 2026-08-24 after Phase 10 returned.** `P10-TRN-invfreq` is **dropped** (above).
`P10-DAT-drop-generic` is **DROPPED** (2026-08-24) — `P11` showed 4 of 5 subclass arms emit the
generic rule zero times, so there is no attractor to remove.
`P10-TRN-multitoken` is **kept and is now more interesting**: the question changed from "does
granularity help?" to "does ONE adapter with N tokens PRESERVE cyclactone's 1.000 while sharing
capacity?" — the difference between a one-off and a usable method. **Top priority is now `P11`, the
length dose-response series.**

**Why this order.** `P10-TRN-multitoken` is the general form of the experiment already running and reuses its
substrate builder. `P10-DAT-drop-generic` is the cheapest thing on the board — a filter on an existing split, no
new data, no new code — and it isolates whether the 37.9% generic block is itself the attractor.
`P10-TRN-invfreq` is last because it was shelved once for a reason (old `[X2c] → `P10-TRN-invfreq``: 31% of PKS training carried a
ketosynthase and produced 0% of output, far worse than frequency alone predicts) and it only becomes
interpretable once `P10-TRN-multitoken` has said whether granularity matters.

⚠️ **Report chemistry RICHNESS alongside the rate for all three.** The `[P9-EVO2POOL]` result was
that both models make exactly ONE chemistry — a rate that doubles while richness stays at 1 has not
addressed the finding.
⚠️ **Novelty gates are load-bearing here.** Every arm below trains on a small, narrow slice; an
endpoint bought by memorisation is worthless. `containment` and `protein_aai` decide, not the rate.

---

## Phase 12 queue — write-up readiness (user, 2026-08-24)

**Framing, hypothesis and titles now live in `docs/paper.md`** (framing only; every claim there cites
`memory.md`).

| id | what | why it is on the list |
|---|---|---|
| **`P12-TRN-secondclass`** | **replicate the length dose-response with SUBCLASS-conditioned adapters in TERPENE or PKS** | ★ **THE NEXT EXPERIMENT.** The quantitative curve is RIPP-only. ⚠️ The *direction* already replicates at class level — PKS T3PKS 1,083 nt made / T1PKS 7,665 nt not; TERPENE precursor 928 nt made / cyclase 2,009 nt rarely — so this converts an existing directional observation into a second curve. Difference between a RIPP paper and a paper about genome language models. |
| **`P12-EVL-structure`** | **structure prediction on generated proteins** | Orthogonal to antiSMASH, which is HMM rules all the way down. A generated protein that folds like its class is evidence no homology search can give. Addresses "detection ≠ chemistry" without a wet lab. |
| **`P12-EVL-coevolution`** | **co-evolutionary plausibility of generated proteins** | Second orthogonal axis: do generated sequences carry the covariation real families do, or only the consensus? Directly probes whether output is a real family member or a centroid — which is exactly the collapse claim. |
| **`P12-EVL-proteomefilter`** | **broad filter = mmseqs hit COUNT vs the class's OWN training proteome** | Replaces the binary 8-accession Pfam gate, which is **blind to some subclasses** (real cyclactone cores score 2/36). Graded, cannot be blind by construction, and **we already run this search for the AAI gate** — the counts are free. Mirrors the phage paper's "≥7 protein hits". |
| **`P12-FIX-filterdiscipline`** | **make validity checks FILTERS, not footnotes** | `containment`/`protein_aai` ≥0.95, junk/N, **length conformance to target** (new — azole's 0.54× went unflagged). ⚠️ Emit BOTH denominators (all-generated and valid-only) so nothing already published silently changes meaning — Standing Constraint 9. |

**Principle for the filter work: filter on VALIDITY, never on the ENDPOINT.** A filter answers "is
this a legitimate candidate?"; a metric answers "is it any good?". Filtering on the endpoint is
circular. This is the line the phage paper holds and the one we currently blur.

---

## Phase 13 queue — the phage-paper gap (user, 2026-08-27)

**Opened from `[P13-ANL-phagegap]`** (`memory.md` 2026-08-27), which read the phage paper's Methods
directly and found two holes in `paper.md`'s elimination argument. Pre-registration:
`docs/phase13_IDENTITY_BUCKET_preregistration.md`.

| id | what | why it is on the list |
|---|---|---|
| **`P13-TRN-azolebucket`** | ✅ **DONE 2026-08-27 — identity-bucket conditioning on AZOLE.** **RESULT: the dial lands (detection 0.280 vs 0.010, p=1.78e-16, length-matched) and the subclass rate does NOT move (3/400 vs 2/1000, p=0.144).** `memory.md` 2026-08-27. Re-train the azole adapter with a second atomic token marking each training record's `ani_to_ref` bucket (their `∼`/`^`/`#`/`$`/`!` scheme), then generate across all five buckets, n=200 each. | ★ **THE NEXT EXPERIMENT.** The phage paper's multi-gene 5.4 kb success ran under **explicit template-fidelity conditioning** (`∼` = "95–100% identity to ΦX174"), and its viable outputs are **93.0–98.8% identical to a training sequence**. Every arm we have ever run sits at the maximum-novelty end of that dial. This builds the dial and measures the whole curve. ⇒ Converts `[P11]`'s "6.3 kb is a capability boundary" into **a measured exchange rate between fidelity and novelty** — or confirms the boundary at a known price. ⛔ **Gate T0 first:** if the top bucket holds <30 records the token has no training signal; switch to quintiles and drop the phage-threshold correspondence. |
| **`P13-EVL-likelihood`** | ✅ **DONE 2026-08-27 — teacher-force real held-out clusters through the adapter and read the per-nucleotide likelihood.** 9 cells: {azole, cyclactone} × {real held-out, own generations} × {adapter, base}, plus azole-adapter-on-cyclactone as a specificity control. | ⛔ **Every azole result to date measures GENERATION, which confounds two failures with opposite fixes:** (1) the model never learned what an azole cluster is — knowledge/capacity; (2) it models them fine but its sampling mass sits elsewhere — a MODE problem. **Teacher-forcing separates them for one forward pass.** The cyclactone column is what makes the azole number readable: if azole's adapter improves on real azole as much as cyclactone's does on real cyclactone, both know their target equally well and the 1.000-vs-0.031 gap is **where the sampler goes, not what the model knows.** ⇒ **RESULT: it is NOT the sampler.** azole +0.0352 vs base (28/45) against cyclactone +0.1341 (34/34 vs the wrong adapter). Memorisation without generalisation; decoding fixes are closed off. `memory.md` 2026-08-27. |
| **`P13-TRN-lorarank`** | **azole re-trained at `lora_r` 64 and 128** (α = 2r), everything else identical to `[P10-TRN-azole]`. ⚠️ **RE-SCOPED 2026-08-27 — this is a referee-facing CONTROL, not a hypothesis test.** | ⛔ **The 'we may be under the LoRA diminishing-returns knee' framing is WRONG, and our own loss curves already said so** (user, 2026-08-27). Azole at r=16 ends at train **3.906** vs eval **5.331** — a **1.43-nat generalisation gap** — with eval *rising* after step 400 (restored checkpoint-400 of 500). That is a model past the knee, not short of it; raising rank widens the gap and early stopping restores an even earlier checkpoint. ★ **And the decisive row is CYCLACTONE — the arm that reaches 1.000 has the WORST gap of the three**, peaking at step 150 then degrading to 5.633. **Fit quality does not separate our success case from our failure case at all.** ⇒ Run it only as a cheap box-check, quote it as a control, and **do not claim it closes the capacity hole**. ⚠️ Counter-argument, recorded because it cuts both ways: eval loss is per-token cross-entropy over ~1,270 tokens, so multi-gene architecture is a vanishing share of the signal — which makes loss weak evidence about architecture in EITHER direction, and is why `P13-EVL-likelihood` runs first. |

⚠️ **AND A CORRECTION THAT RE-SCOPES `[P3-B2b]`:** their funnel is **~36:1**, not ~1000:1 — ~11,000
SFT generations → 302 candidates → 285 synthesised → 16 viable. **Our n=1,000 azole pool is already
~1/11 of their entire per-target sampling budget**, so "overgenerate far harder" is not the lever
that intervention assumed. What their funnel buys is **filter discipline on a small pool**
(`P12-FIX-filterdiscipline`), not scale.

⛔ **The hypothesis `[P13-ANL-phagegap]` KILLED, recorded so it is not re-proposed:** within-target
training redundancy explains nothing. `r(train_frac_distinct@0.80, own-subclass rate) = +0.228`,
`r(n_clusters@0.80, rate) = +0.007` across the five `[P11]` subclasses. AZOLE is slightly *more*
redundant than CYCLIC_LACTONE_AUTOINDUCER (0.610 vs 0.645) and has the **most** effective distinct
examples (487) with the **worst** rate. And the phage training set is **more** diverse than ours at
the matched threshold (94.9% vs 87.7% distinct at 99% id), not less. See `terms.md`
`train_frac_distinct`.

---

## Phase 14 — THE COMPOSITE ENDPOINT (user, 2026-09-01)

**The eventual goal, pre-registered.** antiSMASH detection means *the signature is present*, never
that the cluster is complete — so the current defensible claim is **"generates a gene carrying the
subclass's defining signature"**. The composite below is what would license **"generates a complete,
architecturally correct cluster"** without a wet lab. Full definition and thresholds: `memory.md`
2026-09-01 `[P14-EVL-composite]`.

| # | component | instrument | where we are |
|---|---|---|---|
| 1 | all required genes present, not just the anchor | `definition_domains` per CDS | ⛔ azole 0.063 anchor-only |
| 2 | gene count matches real | `n_bio_orfs` vs per-subclass real cores | ⛔ 1.04–1.38 vs **1.79–3.38** |
| 3 | domain architecture correct AND ordered | `MODULE_PATTERNS` | ⚠️ never measured |
| 4 | ORF lengths match real | `max_orf_aa` vs real | ⚠️ median **0.82x**, worst 0.63x |
| 5 | plausible folds | structure prediction (`P12-EVL-structure`) | ⚠️ not run |

⚠️ **Every current arm would fail this.** That is deliberate — the gap between it and our present
rate is the size of the remaining problem. ⛔ **It still does not license "makes the compound."**
⚠️ **Never filter on components 4–5** — they correlate with the endpoint, so filtering is circular.

---

## Parked: RIPP — held, not dropped (2026-08-19, user)

| item | why it is parked, and what unparks it |
|---|---|
| **[P5-REGEN]** regenerate the five duplicated Phase-4/5 arms | ~611 generations + antiSMASH, overnight. Buys back the 8 kb WIDE contrast and a real `JOINT_PASS`; changes **no claim anyone will build on**. Unpark if the WIDE question is reopened or a write-up needs the 8 kb contrast at full power. |
| **[P5-BIOTRANS]** `bio + transport` training arm | ⛔ **CLOSED 2026-08-24 as a MULTI-GENE intervention** — measured on `ripp_components.jsonl` before building: it adds **0.000** biosynthetic genes, because the strict span runs first-to-last biosynthetic gene and so already contains them all. Still legitimate for generating a *complete* BGC with its exporter — a different goal. Original note: ⚠️ **explicitly NOT dead** (user). Needs a re-stream of the 185 GB tar (`asdb5_core_records.jsonl` holds only strict and wide sequences). DEFINING-gene coverage 0.687, between STRICT 0.869 (works) and WIDE 0.576 (fails); 55.5% fits the 1B and only 58.4% of real RIPP regions have a transporter at all, so its ceiling is **0.584, not 1.0**. |
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
(WIDE vs a size- and cluster-matched control: **Holm p=0.021 at 2.2 kb**; the 8 kb contrast is
**n.s., p=0.15**), and the training-set size drop cost nothing (p=0.79). Wider spans are closed.
⚠️ **CORRECTED 2026-08-24** — this line previously read "Holm p=4.1e-04 at 2.2 kb, 3.2e-05 at 8 kb".
Those are the **pre-deduplication** values. The [P5-DEDUP] audit found the fan-out wrote four
byte-identical copies, so effective n was 47–141, not 188. **WIDE is still refuted, but on ONE
window instead of two, and at p=0.021 instead of 4e-04.** See `memory.md` 2026-08-19 and the
READ FIRST block above. The reasoning below is kept for the record.

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

★ **STAGE-B-ONLY, from 2026-08-24 (user).** Report the **Stage-A RATE of positives** as the primary,
then report **every other metric among POSITIVES ONLY**, against a real-core reference that is also
Stage B on its own positives (22/50 RIPP, 49/50 TERPENE) — and quote the **B/ceiling** ratio beside
the raw value. Post-generation filtering discards negatives, and a Stage-A ladder value is the hit
rate rescaled anyway (`bio_span_frac` 0.641 all vs **0.934** positives). ⚠️ **`frac`, `co_orient` and
`bio_span_frac` SATURATE at Stage B** (1.20 / 1.10 / 1.07 of ceiling) — above 1.00 there is a
symptom of single-gene output, not quality. The metrics with real headroom are `n_bio_orfs`,
`n_bio_domains`, `n_class_domains`, `n_orfs`, `max_orf_aa`.

Two arms are comparable **only** if their `scoring` stamps agree on all five axes. Each has already
caused a real error here:

| axis | required | what went wrong |
|---|---|---|
| Pfam subset | `OBLIGATE_DOMAINS[RIPP]`, 8 accessions | global set inverted A0 (08-14) |
| scoring window | ⚠️ **RETIRED 2026-08-24 — score FULL length** | `_w8000` read 0.087 vs `_w2000` 0.027 on one arm. The window then became the opposite problem: it truncated the CEILING (real cores 0.440 windowed vs 0.680 full) while GenomeOcean, which stops itself, barely moved. **Phase 3–9 numbers are windowed and do not compare to full-length ones.** |
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
| P3-S1n | protein-novelty guard on the sweep | `protein_aai`* | 50/cell | ⚠️ **POOLED, RETRACTED** (was 0.617 · 0.620 · 0.801 · 0.793 · 0.914). Stage-B, among on-class: **L8 0.499 · L100 0.635 · L500 0.450 — no rise** | ⚠️ memorisation at L=500 **stands, on the domain-match evidence** (12/12 vs 0/8), not on AAI | 2026-08-18 |
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
| **P8-T2** | **GenomeOcean TERPENE substrate** | records kept vs Evo2 | 11,297 | **11,260 = 0.997 vs Evo2 0.943 → +602 (+5.3%)**; 4.974 nt/token, context 50,934 nt = 6.4x | ✅ **confound measured out: context cannot explain a P8 win** | 2026-08-22 |
| **P8-T4** | **GenomeOcean TERPENE adapter** | training convergence | 11,260 rec | 2,112 steps = 3 ep, eval 4.5192→**4.2798**; **1.2414 bits/nt vs Evo2 1.2143** | ✅ trained; ★ **loss parity — representation was never the bottleneck** | 2026-08-22 |
| **P8-T5/T7** | **GenomeOcean TERPENE de novo** | `best_bio_bits`>0 @ w2000 | 200 | **137/200 = 0.685** vs Evo2 0.070, ceiling 0.980; antiSMASH-corrected **0.635** | ✅ 9.8x de novo, novelty clean (0/200 gate fails) | 2026-08-24 |
| **P9-SEED** | **the 2x2: seeded arms both models** | ″ | 200/arm | **Evo2 seeded 0.615** (8.8x lift, p=6e-33) · **GO seeded 0.580** after the empty-generation fix (seeding still hurts GO, p=0.038) | ⛔ **"Evo2 BEATS GO" RETRACTED — p=0.541, NO DIFFERENCE** | 2026-08-24 |
| **P9-CYC** | **powered cyclase test** | harder-member rate | Evo2 n=800 | Evo2 **3/48 = 0.062** · GO 20/126 = 0.159 · real 0.440 | ⚠️ **p=0.132 n.s.; 'never' RETRACTED** | 2026-08-24 |
| **P8-CYC** | **cyclase vs the RIGHT comparator + training dist** | harder-member rate | — | **GO 0.159 vs Evo2-SEEDED 0.018, p=1.8e-04**; training data **0.331** — both under-produce, Evo2 by 18x | ★ **GenomeOcean IS the better substrate; NOT a data artifact** | 2026-08-24 |
| **P8-SEEDFIX** | **GenomeOcean seeded-arm repair** | empty rate + full reporting set | 200/arm | `min_new_tokens=100` → **0 empty, 200/200 unique**, primary **0.580** (corrected 0.550) vs de novo 0.685 (p=0.038) **and vs Evo2 seeded 0.615 p=0.541** | ⛔ seeding not worth it for GO; ⛔ **it also RETRACTS "Evo2 wins under seeding"** | 2026-08-24 |
| **P9-T4** | **GenomeOcean RIPP adapter** | training convergence | 8,090 rec | 1,518 steps = 3 ep, 1h50m, eval 5.1153→**4.9501** = **1.4462 bits/nt vs Evo2 1.4573** | ✅ **loss parity replicates — GO now marginally BETTER** | 2026-08-24 |
| **P9-A0** | **GenomeOcean RIPP de novo** | `best_bio_bits`>0 @ w2000 | 200 | **23/200 = 0.115** vs Evo2 0.027 (p=0.0020), ceiling 0.440; antiSMASH-corrected **0.085**, floor 0.000 | ✅ significant; ⚠️ only **26.1% of ceiling** vs TERPENE's 69.9% | 2026-08-24 |
| **P9-2x2** | **the RIPP 2x2 vs TERPENE's** | four Fisher contrasts | 200/arm | GO de novo **ties Evo2 SEEDED** (p=0.471) and **gains nothing from a seed** (p=0.873) — identical to TERPENE | ★★ **structure replicates on an independent class** | 2026-08-24 |
| **P9-SUBCLASS** | **`subclass_specificity`, the phase's designed endpoint** | specific vs generic, Stage B | 13 detected | ⛔ superseded by P9-POOL — the 1/13 = 0.077 was small-sample inflation | ⛔ **UNDERPOWERED at this n** | 2026-08-24 |
| **P9-POOL** | **powered subclass test (user)** | ″ + full Stage-B ladder | **n=1,000, 87 detected** | ⛔ **2/87 = 0.023 strict, 10/87 = 0.115 lenient** (real 0.500/0.740); **ONE chemistry generated vs ELEVEN**; corrected primary **0.104** | ⛔ **gap now DECISIVE, p=1.7e-11**; ⚠️ vs Evo2 still p=1.0 — Evo2's n, not ours | 2026-08-24 |
| **P9-STAGEB** | **Stage A vs Stage B across all arms (user)** | ladder among positives only | all arms | Among positives the models **barely differ** — TERPENE GO B/ceiling **0.90–1.05 on every metric**; Evo2 over-calls `n_orfs` **2.0x**. ★ TERPENE real cores are single-gene (1.122) so it **could never test multi-gene**; RIPP's 1.454 ⚠️[per-subclass real cores are 1.79–3.38] can, and GO reaches **0.71** | ★★ **the model swap buys FREQUENCY, not QUALITY**; multi-gene gap localised to RIPP | 2026-08-24 |
| **P9-ML200** | **antiSMASH floor re-derived** | records actually run | 6 runs | default rejected **7/23, 9/21, 178/200** and **14/50 real cores**; at `--minlength 200` all run, ceiling 33 → **50/50** | ⛔ **policy held only in the Evo2 length regime** | 2026-08-24 |
| **P9-EVO2POOL** | **model-vs-model subclass, detection-matched** | `subclass_specificity`, Stage B | GO 87 det / Evo2 77 det | ⛔ **2/87 = 0.023 vs 1/77 = 0.013, p=1.0** (lenient 0.115 vs 0.065, p=0.293); **BOTH make exactly ONE chemistry vs real cores' ELEVEN**; both ~5% of ceiling | ★★★ **POWERED NULL — the subclass limit is THE METHOD's, not either model's**; ⛔ "first specific chemistry" attribution RETRACTED (Evo2 makes a `redox-cofactor`) | 2026-08-24 |
| **P9-TIE** | **the primary tie at 18x power** | `best_bio_bits`>0 | 1000 vs 889 | GO de novo **0.128** vs Evo2 seeded **0.116**, **p=0.439** (old Evo2 0.160 at n=50 was inflated) | ✅ **third independent confirmation: GO de novo ≈ Evo2 seeded** | 2026-08-24 |
| **P9-CORR** | **antiSMASH-corrected, both pooled arms** | corrected rate | ″ | GO **0.104** vs Evo2 **0.133**, p=0.054 — but Evo2's `rn` is 2.7x higher because its negatives are **99.9% ≥2 kb** vs GO's 47.5% | ⚠️ **LENGTH ARTEFACT — not a win; quote the Pfam primary instead** | 2026-08-24 |
| **P9-WINDOW** | **w2000 vs w4000 diagnostic** | ladder + primary | pooled arms | `n_bio_orfs` B/ceiling **FLAT at 0.71**; but the primary ceiling rises **0.440 → 0.680** while GO stays flat → its share falls **29% → 20%** | ⛔ my "window depresses the ladder ceiling" prediction was WRONG (mixed instruments); ✅ window DOES flatter the primary | 2026-08-24 |
| **P9-BIOTRANS** | **`bio + transport` scoped before building** | biosynthetic genes added | 27,176 regions | **0.000** of 15,704 transporter-bearing regions gain a biosynthetic gene; span 3.9x longer | ⛔ **CLOSED — the strict span already contains every biosynthetic gene; 185 GB re-stream saved** | 2026-08-24 |
| **P10-DATA** | **RIPP subclass distribution + core retention** | records per subclass; label survival into the core | 8,129 train; 50 real cores | **43 subclasses**; 62.1% specific vs 37.9% generic-only; top is azole 9.8%. **Core retains the region's subclass 25/25 = 1.000** | ⛔ **the long tail is NOT the explanation** (27x under-production of the whole category); ⇒ granularity is the live hypothesis | 2026-08-24 |
| **P10-TRN-cyclactone** | **subclass adapter, short simple target** | own-subclass rate, Stage B | 200 gen / 124 det | ★★ **124/124 = 1.000, AT CEILING (p=1.0)**; 0.620 of all generated; novelty **cleaner than real held-out cores** | ★★★ **FIRST CEILING REACHED — subclass conditioning WORKS** | 2026-08-24 |
| **P10-TRN-azole** | **subclass adapter, long complex target** | ″ | 1,000 gen / 63 det | ⛔ **1/63 = 0.016** vs ceiling 1.000 (p=8.2e-30); no better than class-level (p=1.0); generates at **0.54x** its training length | ⛔ **clean null — target complexity binds** | 2026-08-24 |
| **P10-EVL-gate-void** | **Stage-A gate validity per subclass** | real cores on `OBLIGATE_DOMAINS[RIPP]` | 45 + 36 real | azole **45/45 = 1.000**; cyclactone **2/36 = 0.056** | ⛔ **Pfam endpoint VOID for cyclactone — validate the gate per subclass BEFORE using it** | 2026-08-24 |
| **P11-TRN-series** | **length dose-response, 5 subclass adapters** | own-subclass rate vs target length | 200/arm | **1.000 / 0.636 / 0.200 / 0.417 / 0.031** at 715 / 1,624 / 2,191 / 2,738 / 6,293 nt; **r = −0.933** | ★★★ **target length predicts what the method can generate**; ⚠️ middle two n.s. (p=0.398) | 2026-08-24 |
| **P11-ANL-failmode** | **what a MISS looks like, by length** | product census among misses | 5 arms | right → **wrong-but-real** (`RRE-containing`, other genuine subclasses) → **generic collapse** (azole 62/63) | ★ **three regimes, not a smooth fade** | 2026-08-24 |
| **P11-FIX-earlystop** | **early stopping, first outing** | steps saved | ranthipeptide | peaked epoch 2.9, stopped at 250 of 345 — **95 steps saved**, restore took checkpoint-100 | ✅ **works as designed** | 2026-08-24 |
| **P11-GEN-lengthfix** | **force azole to full length, re-measure** | own-subclass rate | 200 gen / 17 det | **0.031 → 0.059, p=0.507** at 1.51x target length; still 16/17 generic; primary rose 0.133 → 0.175 | ⛔ **NULL — the limit is COMPOSITIONAL CAPACITY, not length control**; ⚠️ underpowered for a modest effect | 2026-08-24 |
| **P11-ANL-collapse** | **does class-level conditioning collapse to the centroid?** | bare-generic rate by conditioning granularity | 87 vs 184 det | **0.885 → 0.000, p=4.3e-57**; named chemistry 0.023 → 1.000, p=1.1e-57 | ★★ **the generic output is a CONDITIONING ARTEFACT, not a capability limit** — explains the class-level ~7% of ceiling | 2026-08-24 |
| **P13-ANL-rulestructure** | **what predicts specificity — length or rule gene-count?** | own-subclass rate & detection rate vs both | 11 arms, 3 classes | **gene-count r=−0.895 (p=0.0002)** vs **length r=−0.186 n.s.**; length→detection **r=−0.822**; partials −0.891 / +0.028 | ★★★ **two orthogonal effects; length was the wrong variable for specificity** | 2026-09-01 |
| **P13-ANL-ycao** | **do depth-2 misses carry half the rule?** | anchor-domain presence among detections | azole, 65 det | **62/65 = 0.954 carry YcaO**, called `RiPP-like` for want of the partner gene | ★★★ **single-gene ceiling DIRECTLY OBSERVED** | 2026-09-01 |
| **P13-ANL-fragments** | **are the successes complete clusters?** | max ORF gen vs real; real-core `n_bio_orfs` | 5 arms | long-PKS longest gene **0.63–0.65×** real; real cores carry **1.79–3.38** bio genes vs our 1.04–1.38 | ⛔ **successes are fragments; the multi-gene gap is ~2× larger than reported** | 2026-09-01 |
| **P14-EVL-composite** | **pre-register the composite endpoint** | 5 components, thresholds declared in advance | — | all-genes-present · gene count · domain ORDER · ORF length · fold plausibility | ★★ **defines what would license "generates a complete cluster"**; every current arm fails it | 2026-09-01 |
| **P13-EVL-likelihood-corr** | **generation vs modelling, separated** | YcaO rate adapter vs base | 1000 vs 200 | adapter **63/1000 = 0.063** carry YcaO; base floor detects **0/200** | ⛔ **"the adapter learned nothing about azole" RETRACTED** — it learned the ANCHOR, not the joint structure | 2026-09-01 |
| P5-PREC | precursor detector sensitivity | antiSMASH RODEO motifs | 12 + 12 | mixed subclass **8%**; module-covered **50%** | ⛔ **too low to gate — precursor line dropped** | 2026-08-19 |
| **P8-AUDIT** | **refuted-claim audit across docs, code and cross-session auto-memory** | stale claims still asserted as fact outside `memory.md` | 16 sites | **16 corrected**; `memory.md` itself CLEAN. `hit_eos` had no `terms.md` entry — written. Auto-memory was the EOS leak (loads every session) | ⚠️ **corrections were reaching the ledger and not the docs people read** | 2026-08-24 |
| **P8-AUDIT-2** | **propagate `f386df0` (the 2x2 + cyclase retraction) to the surfaces it missed** | stale claims left after a concurrent commit | 3 sites | `plan.md` 08-20 block still said "never"; `terms.md` power note had no worked example; PI artifact rebuilt | ⚠️ **the artifact went stale within the hour — it has no verifier** | 2026-08-24 |

**Provenance for the block above:**
`phase3_RIPP/adapter_run` (7,250 whole records, 3 ep / 1,350 steps, `loss_ce` 0.790→0.410) ·
`A0_8k.jsonl` + `phase3_ripp/pilot_*.jsonl` · scoring `OBLIGATE_DOMAINS[RIPP]` · window 2,000 nt ·
substrate Evo2 1B (TE 1.13.0 verified).

**Standing reading of the ledger:** A0 is **significant** (p=0.0054 vs 0/400 pooled controls,
pre-registered §8.4). But it reaches only ~6% of the 0.440 ceiling, and all four hits carry a
**single** RIPP domain where real cores carry 1.45 on average ⚠️[that is `n_class_domains`; the `n_bio_orfs` reference of 1.454 is also too low — per-subclass 1.79–3.38]. The defensible claim is "a
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
| **our best arm (unique records: 3 detected)** ⚠️ | **`RiPP-like` 3 — nothing else** |

⚠️ **CORRECTED 2026-08-24.** This row previously read "our best arm (12 detected) | RiPP-like 12".
Those were the **duplicated** records ([P5-DEDUP]); on unique records it is **3**, and the W-2 arm is
**4**. The real-core column is unchanged. ⚠️ The rate below previously read "~70%", which came from
counting **product strings**; counted **per detected sequence** it is **0.909 (30/33)** — the figure
`terms.md` pins.

⇒ **0.909 of real detections get a specific chemistry; 0.000 of ours do** (Fisher p≈1e-5, ⚠️ n=7
unique generated detections — the direction is established, the rate is not). The model produces enough
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
| **WIDE failed** (**Holm p=0.021 at 2.2 kb; 8 kb n.s. at p=0.15** — corrected 2026-08-24 from the pre-dedup 4.1e-04 / 3.2e-05) | ✅ KEPT — experimental, one window, mechanism unknown |
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
- **[X1e] → `INF-FIX-hit-eos` MAKE `hit_eos` TEST TOKEN ID 0 — the cheapest real fix on the board.** It has read 0 in
  every arm ever generated while the model was stopping all along. Requires capturing ids at
  generation time (vortex returns `logits`/`logprobs_mean`/`sequences`, not ids), which is also what
  separates EOS from genuine junk and makes the mask-vs-truncate choice unnecessary.
- **[X1a] → `INF-FIX-constrained-decoding` CONSTRAINED DECODING — do this first, it is ~10 lines.** Mask the logits to
  `{A,C,G,T}` (+ the EOS id once trained) before sampling. NVIDIA's own Evo2 NIM docs state only
  the 4 base tokens are meaningful in output and the rest exist for technical reasons. This makes
  the stray byte **impossible by construction** rather than filtered after the fact.
  ⚠️ Keep an unconstrained arm as the diagnostic — constraining hides the behaviour we are studying.
- **[X1b] → `INF-FIX-eos-token` Train the EXISTING single-token EOS (id 0), not the 5-byte string** (user, 2026-08-20).
  ⚠️ **Do not ADD a token — id 0 is already there and the model already reaches for it.** One token
  is ~5x the per-record gradient of a 5-token marker and makes [X1a] → `INF-FIX-constrained-decoding` trivial (mask to 5 ids).
  ✅ **SHIPPED 2026-08-20 — and with NO FLAG** (user): `--eos-token` and `--eos-mode` are both gone.
  The real EOS (id 0) is appended **unconditionally** after tokenisation to the window carrying a
  record's true end; the 5-byte `|END|` string is retired. `eos_reserve` is 1.
  ⛔ **UPWEIGHTING DROPPED** (user): the signal was never weak — masking EOS causally restores the
  median generation length from 4,583 to 8,000. We were discarding it. Fix the reader, not the writer.
  ⚠️ Upweighting it needs a **manipulation check** — Phase 2's weighted arm consumed a run and
  returned an uninterpretable null because the treatment never landed.
  ⚠️ Literature warns EOS becomes a **self-reinforcing attractor**: once emitted the model keeps
  emitting it, so a mis-placed early EOS collapses the record. Cap the upweight and measure.
- **[X1h] → `INF-EVL-degeneracy-gate` DEGENERACY — what it is NOT, measured 2026-08-20.** Two plausible mechanisms tested and
  **both rejected**: (1) *"the prior context was not BGC-like"* — the pre-collapse prefix is NOT worse
  on-class (degenerate 6/55 = 0.109 vs clean 8/144 = 0.056, Fisher **p=0.22**, if anything better);
  (2) *classic repetition self-reinforcement* — degenerate records show **no more within-alphabet
  repetition than clean ones** (longest homopolymer median 6 vs 6; the BASE model is worse at median
  9). ⇒ It is an **abrupt exit from the nucleotide manifold**, not a gradual decay, which is
  *encouraging* for [X1a] → `INF-FIX-constrained-decoding`: there is no repetition loop for constraining to fall into.
  ⚠️ **But at those positions `P(ACGT) = 0.000`**, so renormalising over ACGT samples an essentially
  arbitrary base. **Whether constrained decoding yields USABLE sequence there is untested and is the
  measurement to make** — generate a constrained arm and score it, do not assume either way.
- **[X1i] → `INF-GEN-snip-replace` SNIP-AND-REPLACE (user, 2026-08-20)** — detect a degeneration/zero-length record, discard
  it, and regenerate that slot. This is rejection sampling; it is legitimate as a **selection** step,
  needs no model change, and composes with [X1a] → `INF-FIX-constrained-decoding`. It is also what the phage paper did (overgenerate,
  filter hard). ⚠️ Report the rejection rate as its own row — a filtered set with an unreported
  discard rate hides the failure it was built to remove.
- **[X1d] → `INF-EVL-degeneracy-gate` DEGENERACY IS A SEPARATE FAILURE AND [X1a] → `INF-FIX-constrained-decoding` WILL NOT FIX IT.** In **0.42% of positions**
  the model collapses to a ~uniform distribution over all 512 tokens (`P(ACGT)=0.000`) — the cause of
  the **27.5% degenerate records in PKS `A0`**. Masking logits at such a position just forces an
  arbitrary nucleotide. Needs the `n_pass` / length-quality gate, and it is the one place a
  *capability* fix (better model, more context) may be required rather than a decoding fix.
- **[X1g] → `INF-FIX-token-ids` ◀ SHARED PREREQUISITE: MAKE GENERATION TOKEN-ID AWARE.** [X1e] → `INF-FIX-hit-eos` (`hit_eos` on id 0) and
  [X1f] → `INF-FIX-per-row-stop` (early stopping) are the SAME build, and [X1a] → `INF-FIX-constrained-decoding` constrained decoding needs the same hook.
  vortex returns `logits`/`logprobs_mean`/`sequences` but **never the sampled ids**, and ids 0/1/32
  all detokenize to `' '`, so the decoded string cannot distinguish EOS from junk. **One change
  unlocks all three:** capture ids at sampling time, then (a) `hit_eos` tests id 0, (b) a per-row
  done-mask stops each sequence at its own EOS, (c) the junk-vs-EOS distinction that makes the
  mask/truncate choice unnecessary. **Do this before any further generation spend.**
- **[X1f] → `INF-FIX-per-row-stop` EARLY STOPPING — ⚠️ BUILT, MEASURED, AND THE SIMPLE VERSION DOES NOT WORK.**
  vortex's `stop_at_eos` checks for EOS and only `print`s (no `break`), and inspects batch row 0
  only — but fixing that is not enough. An `all(rows done)` exit gives **1.01x**, because only
  **21/32 rows emit EOS at all** and one non-terminating row holds the whole batch.
  ⇒ **Per-row exit is worth building: 38.5% of decode compute** (157,490 tokens needed vs 256,000
  paid). EOS position median **2,869**, min 623, max 7,648. Needs vortex's cached
  `inference_params` rebuilt for surviving rows.
  ⇒ **Cheaper approximations first:** shorten `--max-new-tokens` (median EOS is 2,869 of 8,000
  requested), and use **smaller batches** — waste scales with the wait on the slowest row.
- **[X1c] → `INF-GEN-filter-short` Filter prematurely-ended sequences at the selection stage** (user's earlier idea). Cheap,
  legitimate as selection, and the phage paper used a plain length filter rather than a stop token.

### `INF-ANL-taxonomy-prefix` ⬜ BACKLOG  *(was [X4])* — is the taxonomy prefix doing anything for Evo2? (user, 2026-08-22, NOT a focus)

**Not urgent, logged so it is not re-derived from scratch later.**

**Settled already:** GenomeOcean **cannot** take it — its BPE vocabulary is DNA-only, and the Evo2
prefix tokenizes to **122 UNK of 132 ids**. Phase 8 therefore conditions on `[CLS_TERPENE]` alone,
and losing taxonomy is a declared confound that biases **against** GenomeOcean (prereg §6).

**Open for Evo2.** We do not care about taxonomy as an *output* — `taxon_faithfulness` was retired
2026-08-10 because "it graded taxon conditioning, which is not what this project tests". But that is
not the same claim as "it does nothing as an *input*", and one specific fact argues for care:
**the GTDB lineage tag is the ONLY conditioning field Evo2 was pretrained on**
(`model_comparison_evo2_vs_genomeocean.md:45` — "GTDB lineage tag only — no product-class prior").
Our class half is something the base model never saw; the taxonomy half it genuinely understands,
and it varies across **1,260–1,494 values per class** rather than being constant like the class tag.

**What removing it would buy:** context — 6.8% of a median RIPP record, 10.1% PKS, **14.2% TERPENE**
— and, more usefully, **a cleaner Evo2↔GenomeOcean match**, since it would delete the Phase-8
confound. ⚠️ But that requires retraining every Evo2 adapter and regenerating every arm, breaking
continuity with Phases 3/6/7.

**The cheap test, ~30 min GPU:** one adapter, the same 200 prompts, three arms — real taxonomy ·
shuffled taxonomy (a real tag from the wrong record) · no taxonomy — compared on `on_class`.
⇒ **Measure before removing.** Reasoning instead of measuring was wrong three times this session
(the containment null, the left-pad hypothesis, the early-stopping speedup) and the measurement was
cheap every time.

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

**Ordered interventions — ⚠️ SUPERSEDED 2026-08-24 for RIPP. See *Phase 10 queue* at the top of this file.** The hold pending `[X3]` is lifted (GenomeOcean is complete and is now THE substrate). Mapping, so the two lists do not drift:
* `[X2d] → `P10-TRN-azole` / `P10-TRN-cla`` subclass-conditioned adapters → **RUNNING as `[P10-A]`** on RIPP (`AZOLE_CONTAINING_RIPP` 799 rec, `CYCLIC_LACTONE_AUTOINDUCER` 664 rec), on GenomeOcean not the 1B. The PKS/T1PKS framing below is the *original* scope and is not what is running.
* `[X2c] → `P10-TRN-invfreq`` inverse-frequency upweighting → **queued as `[P10-D]`**, still last, still for the reason given below.
* `[X2a]` bigger denominators → **DONE twice** (`[P9-POOL]` n=1,000, `[P9-EVO2POOL]` n=889). Both times the estimate moved DOWN. Do not re-run as a rescue.
* `[X2b] → `P10-GEN-seeded-subclass`` seeded hard-subclass control → **still open and still cheap**; not yet run on RIPP.
* **New, not in this list:** `[P10-B]` subclass-token routing and `[P10-C]` drop the generic block — both generated by the `[P10-DATA]` distribution analysis.

*Original text, kept for the record:*
- **[X2a] Bigger denominators — but NOT a blocker.** All three contrasts are already **significant
  against their own controls** (p=0.041 / 0.0024 / 6.4e-06) on 7–13 detections, and generation is the
  cheap step: n=600/arm would roughly triple them. Worth doing to turn a *direction* into an
  *estimated rate*, not to rescue the finding. ⚠️ The ">=15-detection floor" is **withdrawn** (user,
  2026-08-20) — arbitrary where sampling is cheap.
- **[X2b] → `P10-GEN-seeded-subclass` Seeded hard-subclass positive control.** Seed from a real T1PKS / cyclase exemplar at
  L\*=8. Phase 3 showed seeding lifts ~6x. **If seeded generation still yields 0 hard-subclass, the
  limitation is real; if it does not, it was the prior, not the capability.** This is the single
  most informative cheap experiment on the board.
- **[X2c] → `P10-TRN-invfreq` Inverse-subclass-frequency upweighting** (user, 2026-08-20). Reweight training by rarity
  of the subclass. ⚠️ **Do not run before [X2b] → `P10-GEN-seeded-subclass`** — the PKS gap is 31.3% of training records
  carrying a ketosynthase producing 0% of output, which is far worse than frequency alone predicts,
  so reweighting may be aimed at the wrong cause.
- **[X2d] → `P10-TRN-azole` / `P10-TRN-cla` Subclass-conditioned adapters** — a T1PKS-only adapter (~1,200 records). The honest route
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
`BOS=1 … EOS=2`**, so `[X1b] → `INF-FIX-eos-token`` is free here. Verify the class token survives fine-tuning as one
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

**T9 · 🔄 SEEDED ARMS (added 2026-08-24, user).** ⚠️ `[P7-A0]` is **de novo** — no seeded TERPENE
arm existed, so a "GenomeOcean+seed vs Evo2" table would compare seeded against de novo. Building the
missing **2x2**: `[P7-A0s]` (Evo2 seeded) and `[P8-A0s]` (GenomeOcean seeded), both L\*=8, scored on
the continuation only. L\*=8 transfers on a measurement — TERPENE start-codon entropy matches RIPP's
(1.74→1.97 bits vs 1.53→1.99). Plus **+600 Evo2 de novo** to power the cyclase contrast, which was
n.s. at p=0.21 purely because Evo2 had only 13 detections.

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
| 4. WIDE_KINDS span width | ⛔ **REFUTED 2026-08-19 — significantly worse (Holm p=0.021 @ 2.2 kb; 8 kb n.s.). ⚠️ CAUSE UNKNOWN** — the "dilution" explanation was RETRACTED 2026-08-19 as partly circular |
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

### [P4-WIDE] WIDE_KINDS fine-tune — ⛔ **REFUTED 2026-08-19** (**Holm p=0.021 @ 2.2 kb; 8 kb n.s.**)
Substrate widened from `{"biosynthetic"}` to `{"biosynthetic","biosynthetic-additional"}`.
Same recipe as A0, `DATA=splits_class_wide/RIPP`, **epochs matched to A0 (3) rather than steps**.
⚠️ **3,723/7,808 records kept (47.7%)** — the rest exceed the 1B's **8,192 native context**
(`evo2_1b_base`, a hard model limit) and are dropped, not chunked, so the stop token still lands at
a true record boundary. (⚠️ `|END|` **retired 2026-08-20**; the real EOS is token id 0.)
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
- ⚠️ **CORRECTED 2026-08-27 — this bullet previously read "Their funnel: ~14,466 training genomes
  → thousands generated → 302 candidates → 285 synthesised → 16 viable. Roughly 1000:1
  overgeneration."** The 1000:1 figure was never in the paper. Read from the source (bioRxiv
  2025.09.12.675911v1, Methods), the funnel is **~36:1**: 14,266 training genomes → **~10,000
  Evo1-SFT + ~1,000 Evo2-SFT generations** of length 6,000 at temperature 0.7 → **302** curated
  candidates → 285 synthesised → **16 viable**. Retention 10.4% (Evo1) / 17.2% (Evo2) through
  quality-control + tropism + diversification, then manual curation.
- ⇒ ⛔ **THIS WEAKENS [P3-B2b] SUBSTANTIALLY.** Our n=1,000 azole pool is already ~1/11 of their
  ENTIRE per-target sampling budget, so "generate far more" is not the lever this intervention
  assumed it was. What their funnel actually buys is **filter discipline on a small pool**, not
  scale. Re-scope accordingly; see `memory.md` 2026-08-27.
- ⚠️ **CORRECTED 2026-08-24 — this bullet previously read "`|END|` does not work and is not worth
  fixing (0/150, previously 0/204; whole-record training did not change it)."** The zero was a
  property of the **metric**, not the model. The 5-byte string `|END|` genuinely never fired and is
  now **retired**; but the tokenizer's **real EOS is token id 0**, Evo2 pretrained with it, and our
  adapters emit it (13/13 coherent stop positions, 16x–159x uniform; masking it restores median
  length 4,583 → 8,000). `hit_eos` tested the string, so it read a structural 0. See `memory.md`
  2026-08-20 and [X1]. The phage paper still used a **length filter (4–6 kb)**, not a stop token —
  which remains a reasonable selection step regardless.

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
