# BGC-BENCH — Build Specification v2.0

**Status:** v2.4 — APPROVED. §4 (data) COMPLETE and verified 2026-09-06. §3, §5–§9 not built.
**Purpose:** the sole input to a blind reimplementation. An engineer with this document, the raw
data, and no access to the prior codebase must be able to build the benchmark.

---

## 0. What this document is, and the rule that governs it

This spec anchors a **clean-room rebuild**. The prior codebase (`phase3-ripp`, ~26k lines across
~150 scripts) is a **reference standard for later comparison, not an input**.

**THE BLIND RULE.** While implementing any component in §3–§9, the implementer MUST NOT read the
prior implementation of that component. Every number, threshold, and definition needed is in this
document. If something is missing, the fix is to amend this spec — not to consult the old code.
Amendments are dated and listed in §12.

**Provenance of the numbers in this spec.** Every quantitative claim below is one of:
- **[M]** measured fresh during this specification (method stated inline, reproducible), or
- **[C]** a design choice with a stated rationale, challengeable, or
- **[O]** an open value to be measured by a named gate before it is used.

**No prior result is asserted as fact anywhere in this document.** Prior findings enter only in
§10, as the thing the rebuild's output is diffed against.

---

## 1. The question

Given a generative genome model that has **no class-conditioning channel of any kind**, which
interventions induce generation of a *specified* biosynthetic gene cluster class?

"No channel" is literal and is a design commitment (§4.3): no class tag, no taxonomy tag, no
metadata. Model input is bare nucleotide sequence. This makes every model substrate take an
identical input format, so substrate and interface are not confounded.

**Success is class-SPECIFIC generation.** Producing more BGC-like sequence is a different
capability and is measured separately (§3.4). An intervention that raises all classes equally has
demonstrated capability, not control.

---

## 2. Non-goals — explicitly out of scope

Named so they cannot silently re-enter:

1. **Compound-level or product-structure prediction.** Class only.
2. **Expressibility, synthesis feasibility, or biological validity.** We measure what a detector
   recognises, not what a cell would make. Every claim says "biosynthetic core", never "cluster".
3. **Any conditioning channel in the model input** (§4.3).
4. **Optimising the benchmark's own metric.** No arm may be tuned against the primary endpoint.
5. **Non-antiSMASH quality metrics as endpoints.** See §3.7.

---

## 3. Metrics — the complete definition

### 3.1 The instrument

**antiSMASH 8.0.4 is the sole endpoint instrument.** One tool, one call, both factors.

Rationale [C]: a single instrument cannot drift relative to itself, its output is the field's
accepted definition of "is this a BGC and which kind", and the training corpus is defined by it,
so the achievable ceiling is structurally near 1.0 rather than compressed by proxy insensitivity.

**Circularity disclosure (required in the paper).** Training data is antiSMASH-defined and
evaluation is antiSMASH-based. This measures "can the model produce what the detector recognises."
That is the intended question. §4.5 specifies an externally-curated validation set as the partial
answer to it.

**Invocation is frozen** and identical for every sequence scored — generations, real cores,
negative controls, all arms, all substrates:

```
antismash --minimal --enable-html false --cpus <N> --genefinding-tool prodigal \
          --minlength <MINLENGTH> <input.fasta>
```

`MINLENGTH = 1` [M], provisional on the §4.8 false-positive check. It MUST NOT vary by class or
arm — a per-class value makes cross-class numbers incomparable by construction.

⚠ **Why not the default of 1000.** `--minlength` filters the **input record** length, not the
detected cluster length. antiSMASH-DB was built on whole genomes and contigs, which clear 1000
trivially; we re-score **extracted cores**, which often do not. Measured retention of 60 real
held-out cores per class [M]:

| minlength | TERPENE | RIPP | PKS | BETALACTONE |
|---|---|---|---|---|
| **1** | **100%** | **100%** | 100% | 100% |
| 200 | 100% | 100% | 100% | 100% |
| 500 | 100% | 87% | 100% | 100% |
| 1000 (default) | **28%** | **52%** | 100% | 100% |

The default silently discards 72% of TERPENE and 48% of RIPP cores, and **nothing** from the long
classes. Ceilings on *scored* records barely move (RIPP 0.967 at ml=1 vs 0.935 at ml=1000), so the
whole effect is record loss, not sensitivity.
**A DROPPED RECORD IS NOT A NON-DETECTION.** If dropped records are counted as zeros the denominator
becomes silently class-dependent and short classes are biased downward — indistinguishable from
"the model fails on short classes". 1 rather than 200 because *generations* can be far shorter than
real cores, and at any positive threshold the same bias reappears on the generation side.
**Every sequence submitted MUST receive a verdict; a scored-count below the submitted-count is a
build failure (§11).**

### 3.2 What is read from antiSMASH

Per scored sequence, the **full region table is retained** — region coordinates, the raw
`products` strings, and the rule name that fired. Storage is cheap and discarding it is what makes
a finding unrecoverable later.

Two fields are **endpoints**:
- `detected` — boolean, ≥1 region annotated.
- `products` — the set of raw product-type strings on the annotated region(s).

Everything else in the region table is retained for §3.6 and diagnostics, and is never an endpoint.

### 3.3 Class assignment

`products` are raw antiSMASH strings; the endpoint needs classes. The map
(product string → class) is **regenerated** from antiSMASH 8.0.4's own product→category grouping
(§10.3), not copied.

**Hybrid rule [C]: a sequence counts for EVERY class its products map to.**
A record whose products map to {PKS, NRPS} counts as a hit for a PKS target and for an NRPS
target. Rationale: hybrid clusters are real biology; excluding them discards genuine successes and
biases against multi-product classes.
**Required reporting:** the per-class hybrid fraction is reported in the paper alongside every
rate, because it bounds how much of a rate could be non-exclusive. Measured on the training corpus
[M]: TERPENE 5%, RIPP 10%, NRPS 26%, PKS 24%.

### 3.4 The endpoints

For arm `a` and target class `c`, over `n` generated sequences:

```
detect_rate(a,c)     = #{detected} / n                        CAPABILITY
precision(a,c)       = #{detected AND c in classes} / #{detected}   CONTROL
on_target_rate(a,c)  = #{detected AND c in classes} / n        PRIMARY
```

`on_target_rate = detect_rate × precision` by construction. All three are reported always.
Reporting the primary without both factors is prohibited — they answer different questions and an
intervention can move one while leaving the other flat.

### 3.5 The primary object is a matrix, not a rate

For K target classes, generate under every (arm, target) pair and populate:

```
M[a][c_target][c_observed] = fraction of generations conditioned toward c_target
                             whose antiSMASH classes include c_observed
```

**Class specificity is diagonal dominance, scored against the arm's own unconditioned marginal:**

```
lift(a,c) = on_target_rate(a,c) / on_target_rate(a, unconditioned)
```

`unconditioned` = the same arm generating with no target specified. An arm that raises every class
equally has lift ≈ 1 in every cell and has demonstrated **no control**, regardless of its rates.

**This makes the design self-controlling.** Conditioning toward class X is the off-target control
for class Y's row. K targets yield K×(K−1) off-target controls at no extra generation cost.

### 3.6 Subclass specificity — a pre-specified SECONDARY investigation (RIPP)

A class label is a **collapse** of antiSMASH's raw product strings (§3.3). `PKS` collapses `T1PKS`,
`T2PKS`, `T3PKS`, `hglE-KS`, …; `TERPENE` collapses `terpene`, `terpene-precursor`, …; `RIPP`
collapses `RiPP-like`, `lassopeptide`, `lanthipeptide-*`, `thiopeptide`, …

**Hitting the right class while producing only its most common member is a distinct, weaker
outcome than class-specific generation, and the class-level endpoint cannot see the difference.**

**Status [C]: SECONDARY. This is not a primary endpoint and no arm is optimised against it.**
It is nonetheless **pre-specified here**, before any data is generated, so that when it is reported
it is a planned investigation and not a post-hoc search. It carries its own section in the paper,
scoped to **RIPP** — the only class where the §3.6 confound does not apply — and the structural work
of the paper is done by the multi-gene ladder (§4.5), not by this.
It needs no additional antiSMASH call; only that §3.2's raw products are retained rather than
collapsed and discarded.

For arm `a` and target class `c`, over the generations that hit `c`:

```
subclass_profile(a,c)   = distribution over raw product strings among on-target generations
reference_profile(c)    = the same distribution over REAL held-out cores of class c,
                          scored under the identical config (§7)
modal_share(a,c)        = fraction of on-target generations that are the single most
                          common product string
subclass_coverage(a,c)  = # distinct product strings produced / # observed in reference_profile
```

**`modal_share` is the headline number.** A model producing one member has `modal_share` ≈ 1.0; the
comparison is always against the real-core `modal_share`, never against 1/K, because real cores are
themselves unevenly distributed.

`subclass_profile` vs `reference_profile` is reported as a distribution comparison with the raw
counts always shown — at the n's involved, a divergence statistic alone hides that a cell is 0/3.

⚠ **THE FREQUENCY/GENE-COUNT CONFOUND — subclass specificity is interpretable in RIPP and almost
nowhere else** [M]. Within most classes, the common subclasses ARE the few-gene subclasses, so
"the model produces only the easy member" cannot be separated from "the model copied the data
prior". Spearman[log(product frequency), mean core-gene count], products with n>=20, 16 kb corpus:

| class | rho | reading |
|---|---|---|
| PKS | **-0.867** | fully confounded |
| NRPS | **-0.786** | fully confounded |
| TERPENE | **-0.600** | confounded |
| **RIPP** | **-0.037** | **decoupled — the only interpretable class** |

RIPP carries gene-rich common products (`redox-cofactor` 3.06 genes at n=671, `azole-containing-RiPP`
2.66 at n=916) and gene-poor rare ones (`cyanobactin` 1.36 at n=22, `sactipeptide` 1.59 at n=22).
⇒ **The primary subclass analysis is RIPP.** Other classes report the breakdown descriptively with
their rho stated, and the confound itself is a reportable methods finding.

⚠ **Subclass rates are reported as a breakdown of an already-significant class-level result, never
as a standalone endpoint.** Splitting a small numerator across many product strings produces cells
with single-digit counts; those support a direction, not a rate.

### 3.7 Diagnostics — recorded, never decisive

Recorded for every sequence, reported in supplementary, **never an endpoint and never promoted**:
generated length, `hit_eos`, ORF count and coding density (single stated gene caller), GC content,
count of distinct antiSMASH regions per sequence.

**Prohibited as endpoints [C]:** any HMM/Pfam-derived score, any learned-probe score, any metric
whose value changes with a threshold not fixed in this document. Rationale: a second instrument
with its own sensitivity, drifting independently of the first, is how a benchmark stops being
comparable across arms.

**One exception, and it is a gate not a metric:** novelty (§3.8).

### 3.8 The novelty gate

Every ladder metric is maximised by reproducing training data, so novelty is an **absolute gate on
every arm, applied before any rate is read** — not a co-reported number.

```
containment(s) = max over training records t of
                 |canonical 21-mers(s) ∩ 21-mers(t)| / |canonical 21-mers(s)|
```

- Reference set: **the training split the arm was actually trained on**, per arm. An arm trained on
  a pooled corpus is gated against the pooled corpus. [C — asymmetric reference sets make novelty
  incomparable across arms, so this is specified per-arm and stated in every table.]
- `FAIL` at containment ≥ 0.95. Arm reported as a failure regardless of its rates.
- `WARN` at ≥ 0.80; reported, not disqualifying.
- **The gate MUST raise on a missing or empty k-mer set, never default to passing.** A test pins
  this (§11).
- Reported for every arm regardless of outcome, including arms that score zero.

### 3.9 The scored record — what scoring emits, per generation

A rate is a summary; the scored record is the evidence. Everything an analysis could need
must be written **at scoring time**, because a covariate not recorded then is unrecoverable
later no matter how the analysis is framed.

```
generation_id        str    unique
arm                  str    the §6 coordinate (weight state / regime / inference control)
substrate            str    §5 id
target_class         str    what was conditioned toward
seed_accession       str    S1 only; null in S0
seed_core_gene_count int    S1 only; null in S0   <- §4.5.1 covariate
seed_seq_len         int    S1 only; null in S0   <- §4.5.1 covariate
sequence             str    model output, seed excluded
seq_len              int
hit_eos              bool
scored_ok            bool   FALSE means antiSMASH returned no verdict -- a build failure
                            (§3.1), never silently a non-detection
detected             bool
products             list   RAW antiSMASH product strings, never collapsed (§3.6)
observed_classes     list   products mapped through the regenerated class map
on_target            bool   target_class in observed_classes
region_table         list   per region: coords, products, rule that fired
n_cds, coding_density        diagnostics (§3.7)
containment, novel           the gate (§3.8)
```

⚠ `seed_core_gene_count` and `seed_seq_len` exist so that cross-class claims can be
covariate-adjusted (§8.5). They are the reason §4.5.1 works without a second dataset, and
they cost one join at generation time.

---

## 4. Data

### 4.1 Source and provenance chain

Single raw source: **antiSMASH-DB GenBank records** at `/data2/ds85/asdb5_gbks/`.
The chain is rebuilt end to end; each stage writes a manifest (§4.6):

```
asdb5_gbks/ -> [extract]  core_records.jsonl
            -> [filter]   corpus.jsonl
            -> [split]    splits/<CLASS>/{train,val,test}.jsonl
```

### 4.2 Record schema

```
accession          str   stable unique id
genome_accession   str   THE GROUPING KEY FOR SPLITTING
compound_class     str   from the regenerated map (§3.3)
antismash_products list  raw product strings
sequence           str   nucleotide, strict core region, --flank 0
core_gene_count    int   number of core biosynthetic genes  <- the multi-gene axis
seq_len            int   len(sequence)
contig_edge        bool  truncated by assembly
```

**⚠ Naming hazard, pinned here because it caused a 10× error during this specification [M]:**
`seq_len` MUST be `len(sequence)`. A field describing the annotated *region* span (core plus
flanks) is a different quantity — one observed record had region span 11,008 nt against a core of
1,008 nt. The old schema's `region_len` was the region span. **This schema has no region-span
field.** If one is added it must not be named `*_len`.

### 4.3 Input format — the design commitment

```
model_input = sequence          # nothing else. no tag, no taxonomy, no delimiter.
```

Consequences, all intended:
- Evo2 and GenomeOcean receive byte-identical input format → substrate comparison is unconfounded.
- No conditioning channel exists, so §1's question is well-posed.
- Loss is computed over the whole input; there is no prefix to mask.

### 4.4 Class set

**Near-dup loss is NOT a disqualifier. It is a sample-size measurement, and it is superseded
by a direct cluster count.** [C]

A class's records form a similarity graph. The fraction of held-out records with a near-dup in
train depends on both the class's intrinsic redundancy and whether the split respected sequence
similarity. Since splits are built here, the split is made **cluster-aware** (§4.6) and cross-split
near-duplication goes to ~0 by construction. What remains is **reduced effective sample size**:

```
effective_n = number of clusters from mmseqs easy-cluster,
              --min-seq-id 0.8 -c 0.5 --cov-mode 2,
              computed ON THE LENGTH-BOUNDED CORPUS (§4.7), never before it
```

⚠ **Both halves of that definition were error sources during specification [M].** A
`pooled x (1 - neardup_loss)` proxy underestimated every class by 15-35% (BETALACTONE proxy ~1,500
vs true 2,139), because 68-82% of clusters are singletons. And computing it before the length bound
overstated the multi-gene classes badly — BETALACTONE 2,139 unbounded vs 873 at an 8 kb bound.
**effective_n is only meaningful with its length bound stated.**

### 4.4.1 The length bound drives the class set — measured [M]

Clustering the length-bounded corpus at three candidate bounds:

| class | effN@8k | ≥2gene@8k | effN@16k | ≥2gene@16k | effN@32k | ≥2gene@32k |
|---|---|---|---|---|---|---|
| TERPENE | 8,614 | 16.4% | 8,844 | 18.4% | 9,096 | 20.5% |
| RIPP | 6,048 | 45.4% | 6,512 | 49.5% | 6,689 | 50.7% |
| PKS | 2,839 | **6.7%** | 3,357 | 16.9% | 3,641 | 23.4% |
| NRPS | 2,637 | 16.0% | 3,285 | 25.9% | 4,000 | 35.4% |
| ARYLPOLYENE | 1,054 | 63.2% | 1,224 | 65.5% | 1,348 | 67.4% |
| BETALACTONE | **872** | 100% | **1,941** | 100% | 2,103 | 100% |
| PKS_NRPS_HYBRID | 94 | — | — | — | — | — |

**The bound preferentially deletes multi-gene clusters, because multi-gene clusters are long.** PKS
falls from 38% multi-gene unbounded to **6.7%** at 8 kb. This is the §4.7 selection effect, and it
strips out the exact signal §4.5 exists to measure. It is the single most consequential interaction
in this specification.

**8k → 16k more than doubles BETALACTONE and makes PKS/NRPS usable rungs. 16k → 32k buys almost
nothing** (+8% BETALACTONE, single digits elsewhere), while costing generation and antiSMASH time on
every sequence. **16 kb is therefore the preferred bound even where 32 kb is available.**

### 4.4.2 The class set — FINAL

**Bound: 16,000 nt** (§4.7, G2). **Five classes**, chosen to span the multi-gene ladder monotonically
while keeping hybrid fraction low and every class above the common effective_n:

| class | effN@16k [M] | ≥2 core genes [M] | med nt [M] | hybrid [M] | effSub [M] | ladder role |
|---|---|---|---|---|---|---|
| TERPENE | 8,844 | **18.4%** | 939 | **1.5%** | 2.18 | single-gene anchor; cleanest hybrid |
| NRPS | 3,285 | **25.9%** | 3,954 | 9.2% | 4.93 | assembly-line system |
| **RIPP** | 6,512 | **49.5%** | 1,832 | 4.8% | **10.47** | **keystone** — §4.5 stratification + §3.6 |
| ARYLPOLYENE | **1,224** | **65.5%** | 3,594 | 7.7% | 1.46 | multi-gene, short — binds the common n |
| BETALACTONE | 1,941 | **100%** | 7,790 | 3.1% | 1.20 | multi-gene extreme |

**Ladder: 18.4 → 25.9 → 49.5 → 65.5 → 100.** Monotone, five distinct rungs.

**PKS is excluded** [C] despite being a prominent class: at 16.9% it is a redundant rung with
TERPENE (18.4%), it carries the worst hybrid fraction measured (10.8%), and its subclass richness
is unusable for §3.6 at rho = -0.867. It was dropped on measurement, not on convenience, and the
paper says so. NRPS remains as the assembly-line representative.

Rejected on effective_n or on hybrid/context [M]: PKS_NRPS_HYBRID, SACCHARIDE, PHOSPHONATE,
SIDEROPHORE, ALKALOID, RESORCINOL, ECTOINE, CDPS, BETALACTAM.

### 4.4.3 Equal-n is fixed AT SPLIT TIME — there is exactly one dataset

**The common effective_n is applied by the split builder, once, producing a single canonical
dataset that every arm reads.** It is NOT a per-arm subsampling step. This is the strongest
available form of §9's equivalence rule: arms cannot differ in training data because **no other
dataset exists**.

```
common_n = min over the class set of effective_n@16k  =  1,224   (ARYLPOLYENE binds)
```

Per class: cluster the length-bounded corpus (§4.4), select `common_n` distinct clusters, take
**one representative per cluster** — never a uniform draw over records, which would over-represent
large clusters and reinstate the redundancy clustering exists to remove. The representative is the
mmseqs cluster representative, so selection is deterministic and needs no seed. Split 0.8/0.1/0.1
by cluster (§4.6).

Result: five classes x 1,224 clusters, ~980 train each.
- `W2` (per-class) trains on one class's ~980.
- `W1` (pooled) trains on the union, ~4,900 — balanced by construction, and seeing the *same
  amount of class C* as `W2` does.

**G8** confirms `common_n` sits above the saturation point. Because it needs a trained adapter it
runs early in Stage 1 rather than before the build; if the endpoint has not saturated at 1,224 then
every class is data-starved **equally**, which is interpretable but bounds what the benchmark can
claim, and is stated rather than discovered.

### 4.5 The multi-gene axis — measured INSIDE the benchmark dataset

**THE INVARIANT: every arm consumes `splits/<CLASS>/`, and nothing else.** There is one
training dataset (§4.4.3) and one set of reference corpora. An arm that trained on a
bespoke dataset would not be comparable to any other arm, which is the failure this whole
design exists to prevent. Sub-experiments may *analyse* the benchmark's own output, but
they may not substitute their own training data into the main grid.

**The ladder is the primary multi-gene evidence.** The class set spans
23.7 → 34.6 → 44.0 → 55.7 → 99.8% multi-gene [M], so degradation along it is the headline
result and it costs nothing extra: it falls out of the K×K matrix the benchmark already
produces.

**Its weakness is attribution, not detection.** Across five classes, multi-gene rank and
median-length rank correlate at ρ ≈ 0.70, and class identity, hybrid fraction and subclass
structure vary alongside. With n = 5 classes, "multi-gene is harder" cannot be separated
from "long is harder" at the class level.

#### 4.5.1 The seeded regime resolves the confound WITHIN the benchmark dataset

The `S1` arms seed each generation from a specific held-out record whose `core_gene_count`
and `seq_len` are both known. So each generation carries **per-record covariates**, and the
analysis is a per-generation model rather than a five-point class-level comparison:

```
on_target ~ seed_core_gene_count + seed_seq_len   (+ class as a factor)
```

n is the number of seeded generations, not 5. This is the **primary confound control**: it
uses the benchmark's own dataset and its own output, adds no training run, and separates
gene count from length at proper resolution. It is pre-specified here so that reporting it
is planned rather than post-hoc.

A second within-dataset analysis is free in every regime: among on-target generations,
compare the **produced** `core_gene_count` distribution against the real-core distribution
for that class. That asks "does it build multi-gene loci?" where the regression asks "is
multi-gene harder to hit?"

#### 4.5.2 The RIPP strata — a CONDITIONAL sub-experiment, off the critical path

`strata/RIPP_single/` and `strata/RIPP_multi/` are built (§4.9) and matched on length, with
`core_gene_count` 1 vs ≥2 as the only varying factor. They are **not part of the arm grid**
and nothing in the main benchmark depends on them.

**Gate:** run them only if the ladder (or §4.5.1) shows a multi-gene gradient that needs
attributing in the **de novo** regime, where per-record seed covariates do not exist. If
arms are flat across the ladder there is no effect to attribute and this is moot.

**If it runs:** subsample both strata to the benchmark's per-class train size (979) so the
result is directly comparable to every other number here. The full-size strata are retained
as a better-powered secondary only.

⚠ The strata overlap the main RIPP train set — same split side, drawn from the same pool —
so strata results and main-benchmark RIPP results are **not independent samples** and must
never be pooled or reported as replication.

### 4.6 Splitting

- **Grouped on BOTH `genome_accession` AND sequence cluster.** A genome appears in exactly one
  split, and a sequence cluster appears in exactly one split. Clusters from
  `mmseqs easy-cluster` at the §4.4 criterion. This drives cross-split near-duplication to ~0 by
  construction rather than by post-hoc filtering, and it is why near-dup loss is a sample-size
  number (§4.4) and not a leakage risk. Fractions 0.8/0.1/0.1 by cluster.
- Assignment by stable hash of the grouping key — **no RNG**, so the split is reproducible from the
  corpus alone with no seed to lose.
- Exact-duplicate removal before clustering. After the cluster-aware split, a **verification** pass
  re-runs the §4.4 near-dup search held-out vs train, forward **and reverse-complement**; a non-zero
  count is a build failure, not something to filter away.
- **Externally-curated held-out set (MiBIG):** excluded from every training corpus, declared as a
  named partition in the manifest **at build time**, used **once** at the end as external
  validation. Never iterated against. It is the partial answer to §3.1's circularity disclosure.
- **The manifest is written per class and MUST be additive.** [C] A builder that initialises an
  empty manifest and rewrites the whole file silently destroys sibling entries. A test pins this
  (§11) — during this specification, 12 existing subclass datasets were found carrying an
  all-zero manifest [M], i.e. no recorded leakage verification at all.
- **Split integrity assertions, enforced at build:** every class has non-empty train/val/test
  (an observed corpus had a class with 0 val and 0 test records [M]); genome intersection across
  splits is empty; near-dup count after filtering is 0.

---

### 4.7 Length bound — drop, never truncate

The corpus is bounded by the **smallest substrate's usable context**, and records exceeding it are
**dropped, not truncated**. Truncating manufactures a record that is not a biosynthetic core: a
severed terminal gene, an incomplete rule, a cluster that antiSMASH would never have called.

**This is a selection effect and MUST be reported, not just disclosed.** For every class, report the
`core_gene_count` and `seq_len` distributions of the retained subset against the full class. NRPS is
50% ≤8kb [M], so the retained half is the short half and plausibly the few-gene half — without this
reporting, "NRPS was easier than expected" is indistinguishable from "we kept the easy NRPS."

**Cross-substrate handling.** Substrates differ in context (§5). The **primary comparison uses one
shared bound** so the corpus is identical for every substrate. A substrate with more context
additionally runs a **declared secondary arm at its own full context**, reported separately and
never pooled with the primary. This makes the context difference a measurement rather than a
confound, consistent with §6.5.

### 4.8 The negative control corpus — real non-BGC DNA

**This does not exist and is the first component of the blind build.** Without it there is no
measured false-positive rate, so no rate anywhere in the benchmark has a floor, and `MINLENGTH = 1`
(§3.1) remains provisional.

**What it must be.** Real coding genomic DNA that contains no biosynthetic cluster. **Not shuffled
sequence** — shuffled DNA has no genes at all, so antiSMASH rejects it trivially and the resulting
"FPR 0.000" measures nothing. The control must be hard: real genes, real codon usage, real coding
density, and no cluster.

**Construction.**
1. Source the same genomes as the BGC corpus (`/data2/ds85/asdb5_gbks/`), so GC content, codon
   usage and taxonomy are matched by construction rather than by adjustment.
2. Sample intervals lying **outside every annotated region**, with a **≥5 kb margin** from any
   region boundary, so no partial cluster leaks in.
3. **Length-match to the class corpus being controlled**, per class — otherwise length, not
   content, is what separates control from core. A short class needs short negatives.
4. Apply the identical extraction and processing path as the cores (§4.1), with no exceptions.

**Verification, all required before use:**
- zero interval overlap with any annotated region, margin included;
- GC content and coding density distributions reported against the matched class corpus — a control
  that is obviously different is not a control;
- per-class length distributions match within a stated tolerance;
- `n >= 300` per class. An FPR estimated near zero needs a large denominator to give a useful upper
  bound; at n=300 an observed 0 bounds the FPR below ~0.01.

**What it delivers.** The class-specific false-positive rate — the floor every arm is measured
against (§6.3) — and confirmation that `MINLENGTH = 1` does not admit spurious detections on short
input. If the FPR at ml=1 is materially above zero, the minlength decision is revisited **before**
any arm is scored, not after.

### 4.9 BUILD RESULT — measured [M], 2026-09-06

Corpus: **540,697 core records from 56,846 genomes**, zero extraction errors. 14.0% exceed
the 16 kb bound and are dropped (SPEC 4.7). MiBIG partition: **24,694 of 340,044** in-class
records match a curated cluster and are excluded from train/val/test.

| class | train | val | test | clusters used/avail | med nt | ≥2 core genes | hybrid | products |
|---|---|---|---|---|---|---|---|---|
| TERPENE | 979 | 123 | 122 | 1,224 / 21,358 | 993 | 23.0% | 3.9% | 22 |
| NRPS | 979 | 123 | 122 | 1,224 / 10,723 | 4,789 | 34.2% | 30.1% | 31 |
| RIPP | 979 | 123 | 122 | 1,224 / 19,716 | 1,773 | 44.1% | 8.2% | **49** |
| ARYLPOLYENE | 979 | 123 | 122 | 1,224 / 2,500 | 3,576 | 55.1% | 18.5% | 22 |
| BETALACTONE | 979 | 123 | 122 | 1,224 / 4,107 | 8,770 | 99.8% | 13.2% | 18 |

Ladder **23.0 → 34.2 → 44.1 → 55.1 → 99.8%**, monotone. Every class identical in size at
every split. Verification: genome overlap 0, near-duplicates 0 forward and 0
reverse-complement, for all five classes.

**⚠ Clustering mode is load-bearing, and the default is wrong for this purpose [M].**
mmseqs defaults to `--cluster-mode 0` (greedy set cover), whose clusters are NOT the
transitive closure of the similarity relation — so a cluster-disjoint split still leaks.
The first build produced **3 forward and 3 reverse-complement cross-split near-duplicates
in TERPENE**. `--cluster-mode 1` (connected component) at `-s 7.5` fixes it. Sensitivity
matters independently: at default sensitivity ARYLPOLYENE clustered to 2,774 groups, at
`-s 7.5` to 2,025 — the default was missing real similarity the verification then found.
Residual removals (heuristic prefilter) are recorded per class in the manifest, never
absorbed; the built corpus removed **1** record, in RIPP.

**⚠ Split balance needs a per-class-aware greedy, not a hash [M].** Components chain hard
because a record links by genome OR cluster, so a few are enormous. Hashing each component
independently gave **93/3.5/3.5**; balancing on the union alone gave **~60/20/20**. Placing
components largest-first into the split whose per-class deficits they most reduce gives
exactly 80/10/10, deterministically and with no RNG.

**Within-RIPP strata (§4.5.2) [M] — built, but OFF the critical path:** drawn from the FULL RIPP pool, not the equal-n
subsample, since length matching is already lossy. Matched single/multi = **14,793/14,793**
train, 614/614 val, 801/801 test; median length 2,151 vs 2,101 nt; median `cds_count` 1 vs
3; `core_gene_count` 1 vs ≥2. Split assignment is inherited from `record_split.json`, so a
stratum record can never sit in another corpus's test set.

**Negative controls (§4.8): 300 per class, each from a DISTINCT genome**, length-matched to
within ~0.5% of the class median. The first build drew all 300 from one genome — 300
intervals from one genome is one control, not 300 — and a test now pins genome diversity.

**⚠ RECORDED LIMITATION: the controls are GC-poorer than the cores** — median GC 0.475–0.481
against 0.582–0.668. This is real biology (biosynthetic cores are GC-rich) rather than a
sampling defect, and it is not a shortcut antiSMASH can exploit, since detection runs
profile HMMs over translated ORFs and not composition. But it means the measured
false-positive rate is plausibly a slight **under**estimate, and the paper reports the gap
rather than claiming a matched control.

## 5. Substrates

| id | model | role | context |
|---|---|---|---|
| `evo2-1b` | Evo2 1B | primary substrate, all arms | [O] G2 |
| `go-4b` | GenomeOcean 4B | second substrate, **same arms** | [O] G2 |
| `bgcfm` | published BGC-finetuned GenomeOcean | external published baseline, generation only, no training | [O] G2 |

`bgcfm` is prior art run as-published: it establishes what an existing BGC generation model does
under this benchmark. It is never fine-tuned by us.

**Gate G2 — substrate health, before any arm runs.** For each substrate: (a) load and score a fixed
set of real held-out cores, confirming per-token likelihood is materially better than on
length-matched shuffled sequence; a substrate at chance must fail loudly. (b) record context limit,
dtype, and every dependency version into the run manifest. (c) confirm batched and sequential
generation produce equivalent distributions; if not, one mode is fixed and stated.

**Rationale [C]:** a substrate can load successfully and be silently at chance through a dependency
mismatch. G2 makes that an error rather than a negative result.

---

## 6. The arm grid

Arms are **factors**, not a flat list. Composability is the point: an arm is a coordinate.

**Weight state** (mutually exclusive — one is loaded):
- `W0` base — no training. The floor.
- `W1` pooled adapter over the benchmark classes at the common effective_n (§4.4.3), i.e.
  **balanced by construction**. This is the controlled comparison to `W2`: it sees the *same amount
  of class C* plus the other benchmark classes, so the contrast isolates class-exclusivity from
  data volume.
  **A full-corpus pooled arm is deliberately NOT included** [C]. It would reintroduce the exact
  confound §4.4.3 removes — trained on everything it sees ~8,844 TERPENE clusters against ~1,224
  ARYLPOLYENE, so its per-class results would be confounded by class frequency. The "was everything
  undertrained?" objection it would answer is answered better, and empirically, by **G8**.
- `W2` per-class adapter — class enters through *which weights are loaded*.
- `W3` **prefix tuning** — learned key/value states supplied to **every attention layer**, not a
  learned vector at the input embedding only. The input-only variant (prompt tuning) is a strictly
  weaker mechanism and is not what this arm tests. §6.5 covers the architectural asymmetry this
  creates between substrates.

**Adapter capacity is a declared, defended parameter, not an inherited default.**
Rank is chosen by a **sweep on held-out loss of the target class** — never on the benchmark
endpoint, which would tune an arm against the metric it is scored by (§2.4). The sweep is run once
per substrate, reported in full, and the selected rank stated with its criterion.
Because substrates differ in size, **the cross-substrate equivalence is trainable-parameter
fraction, not rank** — matching rank across a 1B and a 4B model matches neither capacity nor
proportion. Both the rank and the realised parameter fraction are reported for every arm.

**Context regime:**
- `S0` de novo — generation from nothing.
- `S1` seeded — a real held-out core fragment as prefix.
  **The seed is the first `L` nt of a held-out core of the target class**, `L` set per class by
  Gate G3. Held-out (never trained on), first-`L` (unambiguous and reproducible — no motif choice,
  no consensus construction, nothing to tune). The seed span is **excluded from the scored window**
  (§7) and a test pins that it never appears in scored text.
  **`L` is not inherited from any prior work or any other paper.** Values quoted elsewhere were
  measured on a different task, a different corpus and a different model, and carrying one over is
  precisely the assumption this rebuild exists to remove. G3 measures it here, per class.

**Inference-time control:**
- `I0` none.
- `I1` activation steering — inject a class direction during generation.
- `I2` iterative refine — generate, detect, retain the rule-satisfying span, re-seed from it,
  repeat to a fixed iteration cap.

### 6.1 Stage 1 — the shakedown (build verification, not inference)

Every arm run **once** at small n on `evo2-1b` / `W1r` for composable factors, to prove the harness
end-to-end and to estimate rates for powering.

**Stage 1 data is used for power estimation ONLY and is discarded from all final analysis.**
Reusing it is sample-creep — running until the p-value cooperates. Final runs are generated fresh.

### 6.2 Stage 2 — the powered benchmark

Cells and n fixed by pre-registration written *after* Stage 1 rates are read and *before* Stage 2
generates. Pre-registration is committed and hashed before the first generation.

### 6.3 Required control arms

| control | purpose |
|---|---|
| `W0/S0/I0` unconditioned | the floor |
| real held-out cores | the ceiling, scored identically |
| real non-BGC coding DNA | class-specific false-positive rate |
| shuffled-label `W2` | adapter trained on scrambled class labels — isolates class content from training-volume effects |
| random-direction `I1` | magnitude-matched random vector — isolates steering from resampling variance |

**Any post-processing applied to generations is applied identically to real cores.** No exceptions.

### 6.4 Manipulation checks — mandatory, read BEFORE the endpoint

**A null is interpretable only if the test was powered AND the intervention verifiably landed.**
Each arm declares its check; a null without a passing check is reported as **uninformative**, not
negative.

| arm | check |
|---|---|
| `W1`/`W2` | training loss falls on held-out data of the target class; monotone across ≥5 checkpoints |
| `W3` | prefix changes next-token distribution measurably vs no prefix |
| `I1` | the injected direction changes an independent readout of class in activations |
| `I2` | the retained span is present in the re-seeded prompt and absent from scored text |
| `S1` | seed present in prompt, absent from scored span |

---

### 6.5 Where substrates are not architecturally equivalent — declare, do not paper over

Some arms cannot be implemented identically on every substrate, and the honest response is to state
the asymmetry rather than pretend equivalence.

- **`W3` prefix tuning** attaches learned key/value states to attention layers. Substrates whose
  architecture is not uniformly attention (e.g. hybrid convolution/attention stacks) expose fewer
  attachment points than a standard transformer. **Record, for each substrate: total layers,
  attention layers, and the count actually carrying a prefix.** An arm attached at 4 of 25 layers
  on one substrate and 32 of 32 on another is not the same arm, and the comparison must say so.
- **`I1` activation steering** requires a defined residual stream at the injection site; the same
  constraint applies.
- **Context limit** differs by substrate and is a measurement (§5, G2), not something to equalise.
  Where a class exceeds one substrate's context and not another's, that is reported as a finding
  about the substrate, not corrected by truncating the class.

Any arm that cannot be implemented on a substrate is reported as **not applicable**, never as a
zero. A structural absence and a measured null are different results.

## 7. Generation protocol — frozen

1. **Identical `max_new_tokens` for every arm.** No arm gets more room.
2. **NO fixed scoring window. The entire generated sequence is scored.** [C]
   Length must not be a free variable across arms, but there are two ways to achieve that and only
   one is acceptable here. Truncating every sequence to a common window equalises exposure while
   **destroying exactly the signal §4.5 exists to measure** — a cluster requiring three genes across
   6 kb is undetectable in a 2 kb window, by construction. Instead, length is equalised **at the
   source**: every arm receives an identical generation budget (§7.1) and everything it produces is
   scored. The only excluded span is a seed (§6, `S1`), which is never model output.
3. **Identical decoding parameters and RNG seed across arms** within a comparison. Decoding
   parameters are a declared axis [O] G4 — swept deliberately or fixed, never varied incidentally.
4. **Batched and sequential outputs are never pooled.**
5. **Termination.** Each substrate's end-of-sequence token id is **identified and verified at G2**
   (emitted id, not an assumed one) and generation **stops on it**. `max_new_tokens` is a ceiling,
   not the termination mechanism. `hit_eos` and realised length are recorded per sequence for every
   arm. A substrate that never emits EOS is a finding to report, not a reason to disable the check.
6. **Truncation asymmetry, and how it is neutralised.** Generations are cut by the token budget;
   real cores are not. A gene running off the end of a generated sequence would otherwise be scored
   differently from a gene in a complete real core. Two requirements:
   a. The fixed scoring window (§7.2) is applied to **real cores and negative controls too**, so
      every scored sequence is truncated identically.
   b. **An ORF is a start codon to an in-frame stop codon within the scored span.** A region with a
      start and no terminal stop inside the span is **partial**, is flagged as such by the gene
      caller, and its treatment (counted or excluded) is fixed here and identical for generations
      and real cores. Partial-ORF counts are recorded per sequence.
7. One gene caller, one parameter set, stated here and identical everywhere.
8. **Realised length is reported, never corrected for.** An arm that terminates early has less
   scored material; that is a property of the arm, not a confound to adjust away. Report per arm:
   the realised-length distribution, `hit_eos` rate, and the endpoint **both per-sequence and
   per-nucleotide**. The two rates disagreeing is an informative result, not an error.
9. **Real cores and negative controls are scored WHOLE**, under the same length bound as
   generation (§4.7). Same envelope on both sides, so the ceiling is a real ceiling.
5. Every generation set carries a manifest: substrate, weights, arm coordinate, target class, n,
   decoding config, seed, code commit, dependency versions.

---

## 8. Statistical plan

### 8.1 Endpoint

`on_target_rate` — a rate, with an exact binomial CI. Not a mean of a continuous score: at the
rates involved a mean is an outlier detector.

### 8.2 Powering — control n is the binding constraint

For a treatment rate `r` with a control at the floor, one-tailed Fisher's exact behaves as
`p ≈ (1 + n_c/n_t)^(−r·n_t)`, whose limit as treatment n grows with control n fixed is

```
p_floor = exp(−r · n_c)
```

**The p-value has a hard floor set entirely by control n; treatment data cannot breach it.**
`p < 0.05` requires `r · n_c > ln(20) ≈ 3.0`.

Consequences, and they drive the budget:
- Controls are sized generously and **shared** across arms.
- §3.5's cross-class design supplies off-target controls for free.
- **Distinguishing two non-zero arms costs far more than beating the floor.** Whether the paper
  must rank arms or only separate them from the floor is Open Decision D3.
- Every n is fixed by Stage 1 rates; none is assumed here.

### 8.3 Nulls

Where an arm shows no effect, report an **equivalence bound** — "no effect larger than δ" — not
merely "not significant". A null with a passing manipulation check and a stated bound is a result;
without both it is uninformative.

### 8.4 Multiplicity

Arms × classes is a large family. The primary contrast per arm is pre-specified as a single cell;
everything else is exploratory and corrected. Fixed in the Stage 2 pre-registration.

---

### 8.5 Covariate adjustment — required for cross-CLASS claims, not cross-ARM

Seeds are held identical across arms (§7.3), so the seed distribution does not vary between
arms and cannot confound an arm comparison. It **does** vary between classes — that is the
ladder confound (§4.5) — so:

* **Across arms, within a class:** report the raw rate. No adjustment.
* **Across classes:** report the raw rate **and** the covariate-adjusted rate. An
  unadjusted cross-class claim is not reportable.

The adjustment, in the seeded regime, is a per-generation model over the §3.9 covariates:

```
on_target ~ seed_core_gene_count + seed_seq_len + (1 | target_class)
```

n is the number of seeded generations, not the number of classes, which is what makes gene
count separable from length at all.

⚠ **De novo has no per-generation covariates.** Cross-class claims in `S0` therefore cannot
be adjusted, and are reported with that stated. If an `S0` ladder gradient needs
attributing, that is the one condition under which §4.5.2's strata run.

## 9. Repository layout

Equivalence across arms is enforced **structurally**. An arm is a config; there is no per-arm
script, because a per-arm script is how two arms come to be scored differently.

```
bgcbench/
  conf/
    classes/<CLASS>.yaml       # class identity only
    arms/<ARM>.yaml            # one coordinate in the §6 grid
    substrates/<SUB>.yaml      # model id, context, dtype, versions
    scoring.yaml               # THE single frozen scoring config
  data/      build_corpus.py  split.py  manifest.py
  model/     load.py  train.py  generate.py       # substrate-agnostic
  score/     antismash.py  classmap.py  novelty.py  endpoints.py
  stats/     power.py  tests.py  report.py
  run/       stage1.py  stage2.py
  tests/
  reference/frozen/            # prior artifacts for §10 diff ONLY. never imported.
```

**Hard structural rules, each pinned by a test (§11):**
1. `score/` has exactly one antiSMASH invocation site and one endpoint computation. No arm-specific
   branch anywhere under `score/`.
2. `generate.py` takes an arm config; it contains no arm names.
3. Every scored artifact filename encodes its scoring-config hash. Two scorings of one generation
   set can never share a name.
4. No loose files at any run root; every artifact belongs to a run directory.
5. Run directory names: `<stage>_<SUBSTRATE>_<ARM>_<CLASS>`. Two names differing only in case are
   the same name and are rejected at creation.

---

## 10. The reconciliation protocol

### 10.1 Three steps, in order
1. **Build blind.** §0's blind rule. New branch from `main`; nothing from `phase3-ripp`.
2. **Run, record, freeze.** Stage 1 then Stage 2. Results written, hashed, committed before any
   comparison to prior work.
3. **Unblind and diff.** Compare against the prior codebase's numbers. Every discrepancy adjudicated
   in writing to a cause: new bug / old bug / definitional difference / stochastic.

### 10.2 Why this order

A blind rebuild alone cannot distinguish "the new code found a real correction" from "the new code
has a bug." Freezing before comparison is what makes the diff evidence rather than a negotiation.
The reconciliation record is itself a deliverable — it is the answer to "why believe these results."

### 10.3 Regenerate, then diff

The product→class map is **regenerated** from antiSMASH 8.0.4's grouping and then diffed against
the prior map. Freezing a copy would inherit whatever hand overrides drifted into it; regenerating
and diffing reveals what they were. The diff is a §10.3 artifact, not a blocker.

### 10.4 Known-wrong list

`reference/frozen/KNOWN_WRONG.md` records defects found in the prior codebase, each with a red test
in the new one, so the rebuild cannot reintroduce them. Seeded from this specification: the
region-span/sequence-length naming collision (§4.2); the manifest wholesale-overwrite (§4.6); a
novelty gate that can default to passing on an empty k-mer set (§3.7); split integrity holes
(§4.6). This list is **not** a channel for importing prior *results* — defects only.

---

## 11. Verification tests — written before the code they check

| test | pins |
|---|---|
| `test_novelty_gate_fails_closed` | empty/missing k-mer set raises; never passes |
| `test_manifest_additive` | writing class B preserves class A's entry |
| `test_split_disjoint` | zero genome overlap AND zero cluster overlap; zero near-dups on verification, forward and reverse |
| `test_split_nonempty` | every class has non-empty train/val/test |
| `test_seq_len_is_sequence` | `seq_len == len(sequence)` on every record |
| `test_single_scoring_site` | exactly one antiSMASH invocation in `score/` |
| `test_no_arm_names_in_generate` | `generate.py` contains no arm identifiers |
| `test_no_scoring_window` | no truncation of model output anywhere in `score/`; only a seed span may be excluded |
| `test_case_collision` | run-dir creation rejects case-insensitive duplicates |
| `test_seed_excluded_from_scored_span` | seeded arms: seed never in scored text |
| `test_input_is_bare_sequence` | no tag/taxonomy reaches the model |
| `test_identical_scoring_config` | all arms resolve to one scoring config hash |
| `test_every_sequence_gets_a_verdict` | antiSMASH scored-count == submitted-count; a shortfall raises |

---

## 12. Gates and open decisions

**Gates — measured before the work that depends on them.**

| id | gate | blocks |
|---|---|---|
| G1 | ✅ minlength **resolved = 1** [M] (§3.1). ⏸ class FPR still open — **blocked on §4.8**, which is build work, not a gate | final scoring config |
| G2 | ✅ Evo2-1B [M]: 16k @ 6.38 GiB, 32k @ 10.61, 64k @ 19.08 · health PASS (0.913 real vs 1.347 shuffled) · **EOS id 0 verified empirically**. ⏸ GO-4B / bgcFM outstanding | all generation |
| G3 | seed-length sweep for `S1` | seeded arms |
| G4 | decoding-parameter policy: swept or fixed | Stage 2 |
| G5 | ✅ at ml=1 [M], n=60/class: TERPENE 1.000 · RIPP 0.967 · PKS 1.000 · BETALACTONE 1.000. Re-run for the final class set | interpretation of every rate |
| G6 | adapter rank sweep on held-out loss, per substrate (§6) | every `W1`/`W2` arm |
| G7 | ✅ **~0.2 s/sequence** at 8 CPUs, `--minimal` [M] — 50,000 sequences ≈ 2.8 h. **Scoring is NOT the binding resource**, which reopens D3 | Stage 2 sizing |
| G8 | data-scaling: effective_n at which the endpoint saturates | **the class set (§4.4)** and equal-n subsampling |

**Bound resolved: 16 kb** [M]. G2 shows the 1B is not the constraint (64k fits in 19 GiB), so the
bound is a cost decision; 16k captures ~92% of 32k's data benefit at half the generation and
scoring cost. **The 7B is not required.**
**G8 now gates the class set and is the last gate before code.** G1 also carries G7's throughput measurement.
**G5**  It sets the dynamic range every arm is measured against, and if
any class's ceiling is low the whole benchmark for that class is compressed toward the floor.

**Open decisions — require sign-off, not measurement.**

| id | decision | recommendation |
|---|---|---|
| D1 | K = 4 classes as in §4.4 | ✅ **RESOLVED — 4.** Threshold 0.55 sits in a natural gap (0.531 → 0.647); any cut in 0.54–0.64 selects the same set. |
| D2 | `bgcfm` coverage | ✅ **RESOLVED — all four classes, coverage reported.** |
| D3 | rank arms, or separate from floor? | ⏸ **DEFERRED to after Stage 1, deliberately.** Stage 2 is sized for beat-the-floor. If Stage 1 shows arms well separated, ranking comes nearly free; if all sit near the floor, ranking is unaffordable at any feasible n and the paper says so explicitly rather than under-powering it silently. |
| D4 | Stage 1 regime | ✅ **RESOLVED — both de novo and seeded.** Their rates differ enough that neither can power the other. |
| D5 | compute envelope | ✅ **RESOLVED — one shared H100; stay under ~50% utilisation for any job exceeding one day.** Evo2-1B and GO-4B both fit comfortably. **The binding resource is antiSMASH CPU throughput, not GPU** — G1 measures sequences/hour and total scoring cost before Stage 2 is sized. |

---

## 13. What this spec deliberately does not contain

No expected results, no prior rates, no hypotheses about which arm will win. Those belong in the
Stage 2 pre-registration, written after Stage 1 and before Stage 2, and in the paper. A spec that
predicts its own outcome is a spec that will be read to confirm it.
