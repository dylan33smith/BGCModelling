# paper.md — the write-up frame

⛔ **HYPOTHESIS SECTION SUPERSEDED — REWRITE BEFORE DRAFTING (2026-09-01).** The document below was
built around *target length* as the capability boundary. `[P13-ANL-rulestructure]` shows that is the
**wrong variable for specificity**: across 11 arms in 3 classes, **how many genes the target's
annotation rule requires** predicts specificity at **r = −0.895 (p = 0.0002)** while length predicts
it at **−0.186 (n.s.)**, partial **+0.028**. Length governs a *different* endpoint — whether anything
detectable is produced at all (**r = −0.822**). **The hypothesis should be rebuilt around the
single-gene ceiling, which is now directly observed:** 62/65 azole detections carry the rule's anchor
domain (YcaO) and fail only for want of the partner gene.
⚠️ **And three findings WEAKEN what we can claim** — fold them in, do not omit them: the real-core
`n_bio_orfs` reference used throughout is too low (**1.79–3.38** per subclass, not 1.454, so the
multi-gene gap is ~2× larger); long-target successes are **fragments** (longest gene 0.63–0.65× the
real one); and **antiSMASH is a signature detector, not a cluster validator** — so the defensible
claim is *"generates a gene carrying the subclass's defining signature"*, never *"generates the
subclass"*. ⚠️ Every subclass result is **GenomeOcean-only**. See `memory.md` 2026-09-01.

**Created 2026-08-24 at user request** (the seventh doc; `CLAUDE.md` otherwise caps the set at six).
**This file holds FRAMING ONLY — no results.** Every number lives in `memory.md` and every claim
below must cite one there. If a claim here has no ledger entry, it is not yet a claim.

---

## The question

The phage work (*Generative design of bacteriophages with genome language models*, Science 2026 /
bioRxiv 2025.09.12.675911) asked **"can a genome language model design a working genome?"** and
answered it in a wet lab: ~300 synthesised, 16 viable, one target (ΦX174).

**We ask the question that comes after it, and that nobody has answered: WHERE does generative design
of functional DNA stop working, and WHAT stops it?**

That is not a smaller question and it is not their question. Theirs is a demonstration on one target;
ours is a **boundary characterised across a difficulty gradient**, which yields a rule that predicts
what is reachable *before* spending a run. We have no wet lab, so we must not claim viability — but
the boundary question does not need one.

---

## The hypothesis

> **The constraint on generative design of biosynthetic gene clusters is COMPOSITIONAL, not
> INFORMATIONAL.** The model has the information — it compresses held-out clusters as well as any
> alternative, and given a sufficiently simple target it produces the correct chemistry at the rate
> real clusters do. What it cannot do is *assemble* multiple domains into a specified combination,
> and that failure is predicted by the compositional complexity of the target rather than by data
> volume, model scale, context length, conditioning granularity, or generation length.

Falsifiable, quantitative, and the five obvious alternatives are already eliminated by controlled
comparison rather than argument.

---

## The structure: an elimination argument

| candidate explanation | test | verdict | ledger |
|---|---|---|---|
| not enough data | 5 subclasses, 497–799 records | **r = −0.237**; the LARGEST dataset has the WORST rate | `P11` |
| model scale / architecture | Evo2-1B vs GenomeOcean-4B | loss parity; subclass rates identical at matched detections, **p = 1.0** | `P9-EVO2POOL` |
| context length | 8,192 vs 50,934 nt | multi-gene structure unchanged | `P8`, `P9` |
| conditioning too coarse | subclass-conditioned adapters | ⛔ **REFUTED — conditioning WORKS** (ceiling reached) | `P10` |
| generation stops too early | forced 0.54× → 1.51× of target | **p = 0.507**, no change; filler not structure | `P11-GEN-lengthfix` |
| **target compositional complexity** | 5 subclasses × 8.8× length | **r = −0.933** | `P11` |

The fourth row is what makes this interesting rather than deflating: **conditioning demonstrably
works**, so the paper is not "the method fails" — it is "the method has a sharp, measurable boundary,
and here is where it sits."

---

## TWO DISTINCT FAILURE MODES, with different remedies — the spine of the paper

This is the claim to build the discussion around, and it is what separates a list of results from an
argument.

**1. Ambiguous conditioning ⇒ collapse to the class centroid. FIXABLE by conditioning.**

| conditioning | bare generic `RiPP-like` |
|---|---|
| whole compound class, `[CLS_RIPP]` | **77/87 = 0.885** |
| subclass — cyclactone | **0/124** |
| subclass — ranthipeptide | **0/33** |
| subclass — redox-cofactor | **0/15** |
| subclass — lassopeptide | **0/12** |
| **pooled subclass** | **0/184 = 0.000** |

**0.885 → 0.000, Fisher p = 4.3e-57.** And the mirror image on the specific side: named chemistry
goes **2/87 = 0.023** (class) → **124/124 = 1.000** (cyclactone), **p = 1.1e-57**.

⇒ **The generic output was never a capability limit; it is a CONDITIONING ARTEFACT.** Pointed at a
heterogeneous class — 43 subclasses under one token — the model produces the most permissive member
of the set, which is exactly what antiSMASH's loose `RiPP-like` rule detects. Point it at one
subclass and the generic annotation disappears entirely in four of five arms.
⇒ **This is a mode-seeking result, and it is the reason the class-level rate was ~7 % of ceiling.**
The class-level experiments were not measuring what the model can build; they were measuring what it
does when the instruction is ambiguous.

**2. Compositional complexity ⇒ failure to assemble. NOT fixable by conditioning.**
The one subclass arm that still collapses (azole, 6,293 nt) does so with maximally specific
conditioning, 100 %-pure training data, and — when forced — 1.5× the target length. The rate does
not move.

⇒ **Two failure modes that look IDENTICAL in the output — both surface as a generic annotation —
and have OPPOSITE remedies.** One is cured by narrowing the instruction; the other is untouched by
it. Distinguishing them required conditioning at two granularities *and* a length intervention;
neither alone separates them, which is why the class-level phases could not have found this.
⇒ ★ **This is the most novel part of the paper and the best candidate for the title** (see
*Conditioning is not the bottleneck*, below). A reader who takes only one thing should take this.

---

## The three load-bearing results

1. **A working system with a real ceiling.** One subclass reaches its measured real-core ceiling,
   with generations **more novel than real held-out cores** on both DNA and protein gates. That
   forecloses the referee's first question before it is asked.
2. **A quantitative boundary.** r = −0.933 across 8.8× in target length, with the failure degrading
   in **three regimes — right chemistry → wrong-but-real chemistry → no chemistry.** The middle
   regime is, as far as we know, undescribed: the model retains subclass competence well past the
   point where it stops hitting the target.
3. **A mechanism claim, tested and survived.** Length is the obvious confound for (2). We forced it
   and it did not move the rate — which is what promotes the correlation from curiosity to evidence
   about composition.

---

## ⚖️ THE COMPARISON, IN FULL — Hie et al. vs this work

*Numbers from `memory.md` `[P13-ANL-phagegap]` (2026-08-27), read from bioRxiv 2025.09.12.675911v1
Methods §B.1.4–B.1.5 directly, not from a summary.*

| axis | **Hie et al. (phage)** | **this work** |
|---|---|---|
| **question asked** | *can* a genome language model design a working genome? | **where** does it stop working, and **what** stops it? |
| targets | **one** — ΦX174 | **eleven** subclasses across **three** compound classes |
| target size | 5,386 nt, 11 genes | 715 – 11,734 nt (azole 6,293 — **longer than theirs**) |
| conditioning | `+∼` = family token **+ "95–100% identity to ΦX174"** | class token, then subclass token; **no fidelity axis until `[P13]`** |
| fine-tuning | **full-parameter SFT of a 7B**, 16–32 × H100 | **LoRA r=16 = 1.287 %** of a 4B, 1 × H100 |
| unique training examples | **14,266** | 794 (azole) · 661 (cyclactone) |
| genome-equivalents seen | **608,392** (42.6 epochs) | 7,940 (10 ep) · 6,610 (10 ep) |
| training-set diversity @0.99 | **0.949 distinct** | 0.877 (azole) · 0.786 (cyclactone) |
| generations per target | ~11,000 | 1,000 (azole) · 200 (most arms) |
| **novelty stance** | **soft preference** for <95 % AAI, applied to a curated shortlist | **hard per-record gates**, DNA containment AND protein AAI, on everything |
| **output identity to training** | **93.0–98.8 % nucleotide-identical** (67–392 mutations) | cyclactone max containment **0.247** — *more novel than real held-out cores* (0.456) |
| candidate triage | validity cascade: QC (≥7 protein hits, 4–6 kb, GC, homopolymers) → tropism (spike ≥60 %) → diversification (AAI, synteny, gene count) | validity gates + **antiSMASH on ALL generations** |
| role of the broad classifier | **geNomad = DESCRIPTIVE, never a gate**; CheckV explicitly *"analyzed, but not filtered with"* | Pfam gate was a filter — ⛔ **corrected `[P11]`**, it was blind to cyclactone (2/36) |
| ground truth | **wet lab** — 302 synthesised → 285 built → **16 viable** | antiSMASH + Pfam, **computational only** |
| funnel | ~11,000 → 302 → 16 (**~36:1** to candidates) | 200–1,000 per arm |
| headline outcome | 16 viable phages, cryo-EM confirmed | boundary + design rule + **two distinct failure modes** |

### ★ THE ONE PARAGRAPH THAT MATTERS

**Their prompt asked for a near-copy of one named genome, and that is what came back.** `∼` is a
trained token meaning "95–100 % identical to ΦX174"; the viable designs are **93.0–98.8 % identical
to a training sequence**. ⇒ **The 11-gene, 5.4 kb coherence is largely INHERITED FROM THE TEMPLATE,
not composed.** Our gates run per-record on everything generated, and `P10-TRN-cyclactone`'s output
is **more novel than genuine unseen biology**. **These are different tasks and the outputs are not
comparable.** Say this explicitly in the paper — a referee who does not have it will make the
comparison unfavourably and without the context.

### Three defences that do NOT work, checked rather than assumed

1. ⛔ **"Their target was shorter."** It is not — **5,386 nt vs azole's 6,293**. Length is not why
   they win; their target is the smaller one.
2. ⛔ **"They trained on near-duplicates."** False. Their set is **more** diverse at the matched
   threshold: **94.9 %** distinct @0.99 vs our azole's 87.7 %.
3. ⛔ **"We just need to overgenerate harder."** Their funnel is **~36:1**, not the ~1000:1 this
   project had recorded. **Our 1,000-record azole pool is already ~1/11 of their entire per-target
   sampling budget.** ⚠️ This corrected a claim that had been live since `[P3-B2b]`.

### ✅ And the challenge we ran against ourselves

`[P13-TRN-azolebucket]` **ported their fidelity dial onto azole** — a second atomic token marking
each record's identity bucket, on azole's exact training records, byte-identical hyperparameters and
decoding, **the bucket token the only delta**. Detection swung **28×** (0.010 → 0.280,
**p = 1.78e-16**), held in **every** length stratum, and steered protein novelty exactly as designed.
**The target chemistry did not move: 3/400 vs 2/1000, p = 0.144.**
⇒ **The compositional boundary is not a conditioning artefact.** We turned their dial to maximum and
the chemistry stayed where it was. **This is the strongest single piece of evidence in the paper**,
because it answers the obvious objection before it is raised.
⚠️ Their conditioning axis **does not exist in our data at the level they used it**: *Microviridae*
is a taxonomic family with a canonical member, so "identity to ΦX174" is real; "azole-containing
RiPP" is a **chemical annotation over genomically unrelated clusters** — **76.5 % have zero alignable
nucleotide identity** to the reference. That is itself a finding about what these two design problems
are.

## What we borrow from the phage paper, and what we do NOT

**Borrow: filter discipline.** Staged validity filters with pre-declared thresholds, applied BEFORE
evaluation rather than reported alongside it (their quality → tropism → diversification cascade;
AAI ≤ 95 % as a *discard*, not a footnote). Our `[P12]` methods phase adopts this.

**Do NOT borrow: the framing.** They demonstrate success on one target. We characterise a boundary
across many. Copying their structure would invite the comparison we lose (no wet lab) and hide the
comparison we win (a design rule).

---

## Known weaknesses — write these into the paper, do not wait to be asked

| weakness | mitigation | status |
|---|---|---|
| **No wet lab.** antiSMASH detection is a computational proxy for chemistry, NOT viability. | Frame the contribution as *computational design capability*. Add orthogonal computational validation — structure prediction on generated proteins, co-evolutionary plausibility. | `[P12]` queued |
| **The quantitative curve is RIPP-only.** All five subclass points are subclasses of one compound class. | Replicate the dose-response with subclass-conditioned adapters in a second class. ⚠️ Note the DIRECTION already replicates in PKS and TERPENE at class level — see below. | `[P12-TRN-secondclass]` **next experiment** |
| **Two middle points are thin** (12 and 15 detections, overlapping CIs, p = 0.398). | Generate more; cheap for the short subclasses. | queued |
| **The subclass series is single-model.** | Evo2 comparison exists at class level; a scale arm would strengthen it. | optional |

---

## ⚠️ CLASS vs SUBCLASS — the hierarchy, because it is easy to conflate

`OBLIGATE_DOMAINS` defines **23 compound CLASSES** (RIPP, PKS, TERPENE, NRPS, SACCHARIDE …). We have
built class-level adapters for **three**: RIPP, PKS, TERPENE.

**Within** RIPP, antiSMASH assigns **43 distinct SUBCLASSES** (lassopeptide, lanthipeptide-class-i…v,
ranthipeptide, cyclic-lactone-autoinducer, azole-containing-RiPP …). The Phase-10/11 subclass series
is **all inside RIPP** — five of those 43.

**The two experiment families ask different questions and the results compose:**

| | Phase 3/5/6/7 (CLASS level) | Phase 10/11 (SUBCLASS level) |
|---|---|---|
| conditioning | one adapter per compound class | one adapter per subclass |
| targets | RIPP, PKS, TERPENE | 5 subclasses, all within RIPP |
| endpoint | does it make on-class BGC DNA at all | does it make THE conditioned chemistry |
| ceiling | real cores of the class | real cores of that SUBCLASS |
| headline | significant vs 0/400 controls, but ~7 % of ceiling | 1.000 (short) → 0.031 (long) |

★ **AND THE CLASS-LEVEL WORK ALREADY CONTAINS THE SAME LENGTH PATTERN, INDEPENDENTLY, IN TWO OTHER
CLASSES.** Each class has a short member the model makes and a long member it does not:

| class | short member (made) | long member (not made) | class-adapter result |
|---|---|---|---|
| **PKS** | T3PKS, median **1,083 nt** | T1PKS, median **7,665 nt** | T3PKS 8/8, T1PKS 0/8 (p = 0.041) |
| **TERPENE** | precursor, median **928 nt** | cyclase, median **2,009 nt** | precursor 13/13, cyclase 0/13 → 3/48 = 0.062 at power |
| **RIPP** | cyclactone, median **715 nt** | azole, median **6,293 nt** | 1.000 vs 0.031 (subclass-conditioned) |

⇒ **The DIRECTION of the length effect already replicates across three compound classes.** What is
RIPP-only is the *quantitative curve* from subclass-conditioned adapters. `[P12-TRN-secondclass]`
converts an existing directional observation into a second curve — a much cheaper claim to make than
starting from nothing, and it should be said that way in the paper.

⚠️ The PKS and TERPENE observations rest on **8 and 13 detections** and were made with CLASS-level
adapters. Quote them as *direction*, never as rate (`memory.md` 2026-08-20).

---

## Candidate titles

1. **Compositional limits of generative design in biosynthetic gene clusters**
2. **Genome language models generate specified biosynthetic chemistry only for single-gene targets**
3. **Conditioning is not the bottleneck: compositional capacity limits generative BGC design**
4. **What can a genome language model design? A capability boundary for biosynthetic gene clusters**

⇒ (2) puts the finding in the title and is the most honest; (1) is the safest; (3) foregrounds the
two-failure-mode spine, which is the most novel part.

---

## Sequencing before any drafting

1. **`[P12-TRN-secondclass]` — replicate the dose-response in TERPENE or PKS.** The single
   highest-value experiment left: it is the difference between a RIPP paper and a paper about genome
   language models.
2. **`[P12]` methods phase** — orthogonal validation + filter discipline.
3. Power the thin middle points.
