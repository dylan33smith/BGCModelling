# FINDINGS — things measured during the rebuild that belong in the paper

**Purpose.** A running record of results and methodological findings uncovered while
building BGC-BENCH, so they reach the manuscript instead of being rediscovered or lost.
This is **not** a results file for the benchmark itself — those come from Stage 1 / Stage 2
and live in the run manifests.

**Rule.** Every entry carries the number, how it was measured, and one line on why it
matters for the paper. An entry without a number is a hypothesis and belongs in the spec,
not here. Append; do not rewrite. Mark corrections in place.

Tags: `[instrument]` `[design]` `[data]` `[stats]` `[limitation]`

---

## 1. Instrument findings — how antiSMASH behaves as a measurement device

### 1.1 `--minlength` silently discards short records, asymmetrically by class `[instrument]`
Default `--minlength 1000` filters the **input record** length, not the detected cluster
length. antiSMASH-DB was built on whole genomes and contigs, which clear it trivially; we
re-score **extracted cores**, which often do not. Retention of 60 real held-out cores/class:

| minlength | TERPENE | RIPP | PKS | BETALACTONE |
|---|---|---|---|---|
| 1 | 100% | 100% | 100% | 100% |
| 500 | 100% | 87% | 100% | 100% |
| 1000 (default) | **28%** | **52%** | 100% | 100% |

Ceilings on *scored* records barely move (RIPP 0.967 at ml=1 vs 0.935 at ml=1000), so the
entire effect is record loss. **A dropped record is not a non-detection.** Counting drops
as zeros makes the denominator class-dependent and biases short classes downward —
presenting exactly as "the model fails on short classes".
**Paper:** methods must state `--minlength 1`, and this is a caution for anyone scoring
extracted cores rather than genomes.

### 1.2 Ceiling ~1.0 against a near-zero floor `[instrument]`
Scored through the single invocation site at `--minlength 1`, on the rebuilt corpus
(held-out cores as ceiling, 300 real non-BGC coding sequences per class as floor):

| class | ceiling (on-target) | FPR (detect) | **FPR (on-target)** | real-core multi-gene | modal share | products |
|---|---|---|---|---|---|---|
| TERPENE | 1.000 | 0.000 | 0.000 | 0.180 | 0.574 | 2 |
| NRPS | 0.975 | 0.0033 | 0.000 | 0.254 | 0.371 | 10 |
| RIPP | 0.992 | 0.0033 | **0.0033** (1/300) | 0.353 | 0.370 | **19** |
| ARYLPOLYENE | 1.000 | 0.000 | 0.000 | 0.633 | 0.836 | 4 |
| BETALACTONE | 1.000 | 0.000 | 0.000 | 1.000 | 0.909 | 5 |

[CORRECTED 2026-09-06] An earlier version of this entry reported **0.000 for every class**.
That was measured on splits built before the accession-collision fix (§4.5) and is wrong.
RIPP carries **1 on-target false positive in 300**, so the floor is not exactly zero.
**Paper:** the floor must be quoted as measured, not as zero. A treatment rate near 0.003
is indistinguishable from RIPP's floor, which bears directly on the powering in §3.1.
Ceilings 0.975–1.000 still give essentially full dynamic range, and `--minlength 1` still
costs no specificity.

### 1.3 antiSMASH throughput is not the bottleneck `[instrument]`
~0.2 s/sequence at 8 CPUs in `--minimal` mode (60 sequences in 9–15 s; a 3-sequence probe
reads ~1.1 s/seq because startup dominates). 50,000 sequences ≈ 2.8 h.
**Paper:** supports the sample sizes; also worth stating because scoring cost is a common
reason benchmarks under-power.

### 1.4 Real cores already produce off-target products `[instrument]`
ARYLPOLYENE cores also emit `resorcinol` (12/60); BETALACTONE also `terpene-precursor`
(11/60); PKS real cores are 49 T3PKS vs 10 T1PKS. **Paper:** the confusion matrix has
genuine off-diagonal mass even for perfect generation, which is why specificity is scored
as lift against the real-core marginal and never against a diagonal ideal.

---

## 2. Design findings — confounds that shape what can be claimed

### 2.1 The length bound preferentially deletes multi-gene clusters `[design]`
Multi-gene clusters are long, so any corpus bound removes them selectively. PKS multi-gene
fraction: **38% unbounded → 16.9% at 16 kb → 6.7% at 8 kb**. PKS_NRPS_HYBRID loses 96% of
records at 8 kb (3,486 → 124).
**Paper:** any claim of the form "the model only produces single-gene output", measured on
a context-bounded corpus, is partly measuring the corpus. State the bound with the claim.

### 2.2 Multi-gene content and context fit are anti-correlated across classes `[design]`
Every class above ~65% multi-gene either fails the diversity gate or does not fit an 8 kb
context. **Across-class comparison therefore cannot separate "multi-gene is harder" from
"long is harder"**, and additionally varies class identity, effective_n, hybrid rate and
subclass structure simultaneously.
**Paper:** the ladder detects a multi-gene effect but cannot attribute it. Attribution
comes from the **seeded** regime, where each generation carries its seed's `core_gene_count`
and `seq_len` as per-record covariates, so the confound is separated at generation
resolution rather than across five class-level points (SPEC 4.5.1). Within-class
stratification is a fallback for the de novo regime only.

### 2.3 Subclass frequency and gene count are the same variable in most classes `[design]`
Spearman[log(product frequency), mean core-gene count], products with n≥20, 16 kb corpus:

| class | rho |
|---|---|
| PKS | **−0.867** |
| NRPS | **−0.786** |
| TERPENE | **−0.600** |
| **RIPP** | **−0.037** |

In PKS/NRPS/TERPENE, "the model only makes the easy member" cannot be separated from "the
model copied the data prior". RIPP decouples them: gene-rich common products
(`redox-cofactor` 3.06 genes at n=671, `azole-containing-RiPP` 2.66 at n=916) alongside
gene-poor rare ones (`cyanobactin` 1.36 at n=22, `sactipeptide` 1.59 at n=22).
**Paper:** subclass specificity is a RIPP analysis. Reporting it for PKS would be
uninterpretable, and this is a caution for the field.

### 2.4 The "class" level is a choice, not a given `[design]`
antiSMASH supplies **7 categories** and **~103 products**; the benchmark sits between them.
`arylpolyene` is a product inside category PKS; `betalactone` is inside `other`, which
antiSMASH does not assert is coherent and which is the single largest group in the corpus
(22,980 records, 30 products).
A **derived** promotion rule was attempted and is **vacuous**: "promote a product whose
trigger set fires no sibling rule in its category" passes for all 103 rules, because
antiSMASH rules are built from largely disjoint profile-HMM families. Size-only promotion
collapses the class level onto the product level, removing subclass structure everywhere.
**Paper:** promotion is disclosed as a design choice with its two entries and reasons, and
separability is checked empirically against the real-core confusion matrix.

### 2.5 `core_gene_count` counts a fifth of the genes in a core span `[design]` `[limitation]`
`gene_kind` composition inside core spans, 1,266 spans from 150 genomes:
unannotated **62.9%**, biosynthetic **20.9%**, biosynthetic-additional 10.5%, transport
3.6%, regulatory 1.4%.
Counting only `biosynthetic` is right for this benchmark — the endpoint is antiSMASH
detection, detection means satisfying the rule, so this counts the machinery the model must
produce to trip the detector. But 62.9% of CDS in the span carry no annotation and nothing
here can speak to them; `cds_count` records them so the question stays askable.
**⚠ RiPP-specific caveat:** RiPP precursor peptides are short and frequently unannotated,
so some of that 62.9% may be functionally essential genes we are not counting.

### 2.6 The strict core is where the biosynthetic genes are `[data]`
**99.5% of `biosynthetic` genes fall inside the `proto_core` span** (2,428 in core vs 32
in the surrounding region), while accessory genes sit mostly outside it
(biosynthetic-additional 5,409 outside vs 1,221 inside; transport 1,885 vs 423).
**Paper:** validates using `proto_core` rather than `region` as the unit. A `region`
carries a per-product neighbourhood of 5–20 kb on each side — one RiPP region spans
11,008 nt around a 1,008 nt core.

---

## 3. Statistical findings

### 3.1 The p-value has a floor set entirely by control n `[stats]`
For a treatment rate `r` against a control at the floor, one-tailed Fisher behaves as
`p ≈ (1 + n_c/n_t)^(−r·n_t)`, whose limit as treatment n grows with control n fixed is

```
p_floor = exp(−r · n_c)
```

**Treatment data cannot breach it.** `p < 0.05` requires `r · n_c > ln(20) ≈ 3.0`.
**Paper:** in a benchmark whose headline results are floors, control n is the binding
resource, and sizing treatments instead is wasted sampling.

---

## 4. Reproduction findings — defects that would corrupt results silently

### 4.1 Cluster-disjoint does not imply near-duplicate-free `[data]`
mmseqs defaults to `--cluster-mode 0` (greedy set cover), whose clusters are **not** the
transitive closure of the similarity relation. A cluster-disjoint split built on it still
leaked **3 forward and 3 reverse-complement** cross-split near-duplicates in TERPENE.
`--cluster-mode 1` (connected component) fixes it. Sensitivity matters independently: at
default sensitivity ARYLPOLYENE clustered to 2,774 groups, at `-s 7.5` to 2,025 — the
default was missing real similarity the verification search then found.
**Paper:** methods should state cluster mode and sensitivity, not just identity/coverage.

### 4.2 Component chaining defeats naive split assignment `[data]`
Records linked by shared genome **or** shared cluster chain into a few enormous components.
Hashing each component independently gave **93/3.5/3.5**; balancing on the union gave
**~60/20/20**. A per-class-aware greedy (largest component first, into the split whose
per-class deficits it most reduces) gives exactly 80/10/10, deterministically.

### 4.3 `effective_n` is meaningless without its length bound `[data]`
BETALACTONE: **2,139 clusters unbounded → 873 at 8 kb → 1,941 at 16 kb.** A
`pooled × (1 − neardup_loss)` proxy additionally underestimates every class by 15–35%,
because 68–82% of clusters are singletons.

### 4.4 Near-duplicate loss measures diversity, not leakage `[data]`
Once the split is cluster-aware, cross-split near-duplication is ~0 by construction and
what remains is reduced **effective sample size**. Held-out near-dup rates measured on a
genome-only split: RIPP 0.421, TERPENE 0.454, NRPS 0.482, PKS 0.520, then a natural gap to
SACCHARIDE 0.647, BETALACTONE 0.678, ARYLPOLYENE 0.766, ECTOINE 0.838.

---

## 5. Standing limitations to disclose

### 5.1 Negative controls are GC-poorer than the cores `[limitation]`
Median GC **0.475–0.481** (controls) vs **0.582–0.668** (cores), across all five classes,
with lengths matched to within ~0.5%. This is real biology — biosynthetic cores are GC-rich
— not a sampling defect, and it is not a shortcut antiSMASH can exploit, since detection
runs profile HMMs over translated ORFs rather than composition. But the measured
false-positive rate is plausibly a slight **under**estimate.

### 5.2 The corpus is antiSMASH-defined and the endpoint is antiSMASH `[limitation]`
This measures "can the model produce what the detector recognises", which is the intended
question, but it is circular in the strict sense. The MiBIG partition (24,694 of 340,044
in-class records, held out at build time and read once) is the partial external answer.

### 5.3 Hybrid records count for every class they map to `[limitation]`
Per-class hybrid fractions in the built corpus: TERPENE 3.9%, RIPP 8.2%, BETALACTONE 13.2%,
ARYLPOLYENE 18.5%, **NRPS 30.1%**. NRPS's rate in particular bounds how exclusively its
numbers can be read.
