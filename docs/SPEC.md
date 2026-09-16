# BGC-BENCH — Build Specification v2.0

**Status:** v3.1 — APPROVED, and **§5–§9 ARE BUILT AND HAVE RUN.** ⚠ The header previously read
"§5–§9 NOT BUILT: `bgcbench/model/`, `bgcbench/stats/` and all of `bgcbench/conf/` are empty",
and cited corpus `0225546040b9`. Both are superseded:

* **Corpus.** The benchmark runs on `c74154974aff985f0a1677d1bf9d1f8491f7272f5fc6c27cb43546f195ff14bd`
  at `max_len 8192`, `common_n 10109` — NOT `0225546040b9` at 16,000 nt. Sections 4.4–4.9 below
  still describe the superseded 16 kb / five-class build; they are kept as the design record, and
  ⚠ **every class list, `common_n`, per-class record count and ladder in §4 predates the rebuild.**
  The built benchmark is **four** classes (TERPENE, RIPP, ARYLPOLYENE, REDOX_COFACTOR) — NRPS,
  PKS and BETALACTONE were dropped when the bound fell to 8,192 (FINDINGS §1.5c, §5.3).
* **Code.** `bgcbench/model/` (load, train, generate, interventions, directions, genconfig),
  `bgcbench/run/`, `bgcbench/score/` and `bgcbench/data/` are all built; 120 tests pass.
* **Gates run:** G1, G3, G5, G6, G6b, G7, G9, G10. Still open: G4 (decoding policy), G8
  (data-scaling sweep), G2 (partial — Evo2 done, GO-4B and bgcfm lack a likelihood check).
* **Results:** FINDINGS §6–§19, with eight frozen artifacts under `/data2/ds85/bgcbench/runs/`.
**Purpose:** the sole input to a blind reimplementation. An engineer with this document, the raw
data, and no access to the prior codebase must be able to build the benchmark.

---

## 0. What this document is, and the rule that governs it

This spec anchors a **clean-room rebuild**. The prior codebase (`phase3-ripp`, ~26k lines across
~150 scripts) is a **reference standard for later comparison, not an input**.

**THE BLIND RULE.** While implementing any component in §3–§9, the implementer MUST NOT read the
prior implementation of that component. Every number, threshold, and definition needed is in this
document. If something is missing, the fix is to amend this spec — not to consult the old code.
Amendments are dated and listed in [`DEFECTS.md`](DEFECTS.md) (§12.A1–A10), split out of §12 on 2026-09-16.

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

**TWO DIRECTIONS, AND THE GATE TAKES THE WORSE OF THEM.**

```
forward(s) = max over reference records t of |21mers(s) ∩ 21mers(t)| / |21mers(s)|
reverse(s) = max over reference records t of |21mers(s) ∩ 21mers(t)| / |21mers(t)|
gate on    = max(forward, reverse)
```

⚠ **FORWARD ALONE CANNOT FAIL AT THE GENERATION BUDGET** [M]. Its denominator is the whole
generation, so it decays as 1/length however much was copied. Measured on the real TERPENE
training split with the real instrument: **ten whole verbatim training records concatenated to
15,992 nt score forward 0.198 → PASS**, while antiSMASH calls them on-target with 15 core genes.
At the 16 kb bound only 0.6–3.7% of training records per class are long enough for a single
record to reach FAIL. Reverse containment does not dilute with length: the same collage scores
**1.000 → FAIL**, against 0.000–0.006 for held-out real cores plus filler and 0.000 for random
DNA. Reference records below 50 k-mers are excluded from the reverse direction.

**TWO REFERENCES, BOTH REPORTED. They answer different questions.**

| reference | question | applies to |
|---|---|---|
| **per-arm** — the training split the arm was actually trained on | "did this arm memorise its own training data?" | trained arms only |
| **corpus-level** — every eligible core record (~309k), searched with mmseqs | "did the model output a KNOWN BGC?" | **every arm, including the untrained floor** |

The corpus-level reference exists because the per-arm one is unconstructible for `W0` and
`bgcfm`, which we never trained — `Reference([])` raises by design — and because Evo2 and
GenomeOcean were **pretrained on public genome collections that include the source genomes of
this corpus**. A base-model arm reproducing a real cluster is memorising from pretraining, which
the per-arm reference cannot see at all. Reporting only the per-arm reference would leave the
floor arm's memorisation both ungated and unmeasurable [C].
The exact k-mer index does not scale to 309k records, so the corpus-level check uses mmseqs at
the §4.4 criterion; the per-arm check uses the exact index.
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
stage                str    "stage1" (shakedown) or "stage2" (powered). A field, not a policy:
                            SPEC 6.1 forbids pooling them and prose cannot be audited afterwards.
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

⚠ **REMOVED — the conclusion was reversed.** This argued "8k → 16k more than doubles BETALACTONE and makes PKS/NRPS usable rungs". FINDINGS §1.5c measured Evo2-1B near chance by ~14 kb; the corpus was rebuilt at 8,192 nt and PKS, NRPS and BETALACTONE were dropped in consequence. The surviving point is only the diminishing return: 16k → 32k buys almost
nothing** (+8% BETALACTONE, single digits elsewhere), while costing generation and antiSMASH time on
every sequence. **16 kb is therefore the preferred bound even where 32 kb is available.**

### 4.4.2 The class set — FINAL

**Bound: 8,192 nt** — Evo2-1B's usable context (FINDINGS §1.5c: NLL rises from 0.8049 at 8,192 to 1.2388 at 15,900 against ln(4)=1.386, i.e. near chance). **Four classes** (TERPENE, RIPP, ARYLPOLYENE, REDOX_COFACTOR). ⚠ An earlier bound of 16,000 nt and a five-class set including NRPS and BETALACTONE were REMOVED at the rebuild: the bound removed precisely those classes' multi-gene members, so they would have entered as biased short-tail subsamples. Chosen to span the multi-gene ladder monotonically
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
common_n = min over the class set of RECORDS@8,192 (§12.A3)  =  10,109   (REDOX_COFACTOR binds)
⚠ REPLACED: this read "min over the class set of effective_n@16k = 1,224 (ARYLPOLYENE binds)", from the superseded 16 kb build. Realised splits are 8,044 train / ~1,010 val / ~654 test per class.
```

Per class: cluster the length-bounded corpus (§4.4), assign **clusters** to train/val/test, and
take **every record in an assigned cluster** up to a common RECORD budget. Splitting by cluster is
what prevents leakage; discarding a cluster's other members does not add safety, it only removes
training signal. **AMENDED 2026-09-08 from "one representative per cluster" — see §12.A3.**

`common_n` is therefore a count of RECORDS, derived as the minimum across classes of the records
available behind their clusters, and each split is trimmed to the smallest class so record counts
are exactly equal per class per split. Selection order is the deterministic `_frac`-ordered cluster
order, so it still needs no seed. Cluster counts are reported alongside record counts, because the
two answer different questions: clusters bound **statistical independence**, records bound
**training signal**.

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

**If it runs:** subsample both strata to the benchmark's per-class train size (**8,044**, §12.A3; an earlier version said 979, from the superseded build) so the
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
- **MiBIG is NOT a test set** [C]. This is a generation benchmark: we do not predict on held-out
  inputs, so there is nothing for a test set to do. Its role is narrower and is stated as such:
  1. **Novelty reference** — "did the model regenerate an *experimentally characterised*
     cluster?" This is the sharpest form of the §3.8 corpus-level question.
  2. **External ceiling** — antiSMASH detection on independently curated data, a check on §3.1's
     circularity disclosure that does not depend on our own extraction.
  3. **Likelihood / surprise** — per-token likelihood on curated clusters, a capability measure.
  It is excluded from every training corpus, declared in the manifest at build time, and read
  once. Never iterated against.
  ⚠ **EXCLUSION IS AT CLUSTER LEVEL, NOT RECORD LEVEL** [M]. Removing the matching accessions
  before clustering left their near-duplicates eligible for training: the held-out set measured
  **26.3% near-duplicate to pooled train against an 8.2% background** — 3.2× enriched. A set a
  quarter of which is a near-copy of training is not external under any of the three uses.
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

### 4.9 BUILD RESULT — ⚠ SUPERSEDED, describes the 16 kb five-class build

⚠ **THIS SECTION IS NOT THE BUILT BENCHMARK.** It records the earlier build on corpus
`0225546040b9` at a 16,000 nt bound with five classes, and it is retained only as the design
record — every count, ladder, class list and instrument figure below is superseded. **The built
benchmark is:**

| | value |
|---|---|
| corpus | `c74154974aff985f0a1677d1bf9d1f8491f7272f5fc6c27cb43546f195ff14bd` (542,414 records) |
| bound | **8,192 nt** |
| classes | **four**: TERPENE, RIPP, ARYLPOLYENE, REDOX_COFACTOR |
| `common_n` | **10,109** (records, §12.A3; derived) |
| realised splits | 8,044 train / ~1,010 val / ~654 test per class |
| instrument (G5 / G1) | on-target **1.000** for all four at n=82; false-positive **0.000** at n=300 |
| clusters used | TERPENE 2,448 · RIPP 2,977 · ARYLPOLYENE 1,676 · REDOX_COFACTOR 793 |

⇒ Read nothing below as current. The superseded record follows.

#### 4.9-old (superseded) — corpus `0225546040b9`

**Everything below is re-measured from the built artifacts, not carried forward.** The corpus
was rebuilt six times during development as defects were found; earlier versions of this section
were stale on essentially every cell, so the corpus SHA is quoted with the numbers and every
gate artifact binds it (§9.3).

Corpus: **542,414 core records from 56,846 genomes**, zero extraction errors, `sha256`
`0225546040b9…`. Bound 16,000 nt; cores snapped outward to whole genes. MiBIG partition:
**31,838 records excluded at CLUSTER level** (§4.6).

| class | train | val | test | clusters avail | med nt | ≥2 core genes | hybrid (class) | products | leaking replaced |
|---|---|---|---|---|---|---|---|---|---|
| TERPENE | 979 | 123 | 122 | 23,960 | 1,367 | 23.5% | 4.2% | 23 | 0 |
| NRPS | 979 | 123 | 122 | 15,792 | 5,161 | 25.9% | 17.1% | 42 | 0 |
| RIPP | 979 | 123 | 122 | 23,453 | 2,263 | 45.5% | 4.3% | **48** | 0 |
| ARYLPOLYENE | 979 | 123 | 122 | 2,048 | 3,842 | 54.5% | 19.0% | 25 | 2 |
| BETALACTONE | 979 | 123 | 122 | 3,704 | 9,017 | 99.8% | 10.8% | 17 | 24 |

Ladder **23.5 → 25.9 → 45.5 → 54.5 → 99.8%**. Monotone, but ⚠ **TERPENE and NRPS are now nearly
tied** (23.5 vs 25.9) — closer than the ~12-point gap the class set was chosen on, because the
gene-snapping and fuzzy-coordinate fixes moved both. They remain distinct rungs only weakly.

**Equal-n holds exactly**: 979/123/122 for every class, at every split. Leaking held-out records
are REPLACED by backfill rather than deleted (§4.6), so the count is recorded without the
denominators shrinking — an earlier build deleted them and published 979/123/122 while disk held
979/107/110.

Verification: genome overlap 0, near-duplicates 0 forward and 0 reverse-complement, all classes.
Within-RIPP strata (§4.5.2, off the critical path): 18,528 matched pairs train.
Negative controls: 300 per class, each from a distinct genome, spread across the whole index by
stable hash.

#### 4.9.1 Instrument verification — measured [M]

| class | ceiling (on-target) | FPR (on-target) | real-core modal share | products in reference | **ORACLE on-target** |
|---|---|---|---|---|---|
| TERPENE | 1.000 | 0.000 (0/300) | 0.607 | 2 | 1.000 |
| NRPS | 0.975 | 0.000 | 0.496 | 5 | 0.967 |
| RIPP | 0.992 | 0.000 | 0.413 | **15** | 0.983 |
| ARYLPOLYENE | 1.000 | 0.000 | 1.000 | 1 | 1.000 |
| BETALACTONE | 1.000 | 0.000 | 1.000 | 1 | 1.000 |

**The ORACLE column is the load-bearing one.** The ceiling is measured by calling antiSMASH
directly; the oracle pushes a held-out real core through the COMPLETE arm path — `record.build`,
the novelty gate, `rates`, `confusion`, `lift`. Without it, "the methods do not work" and "the
harness is broken" are indistinguishable when every arm reads zero. It passes at 0.967–1.000 with
precision 1.000 and a clean confusion diagonal, and `novel=60/60` with every gate PASS, so the
strengthened novelty gate passes real cores rather than false-positiving on them.

It immediately earned its keep: it caught that **antiSMASH silently sanitises record ids** — it
strips colons, so `oracle::TERPENE::X` returns as `oracleTERPENEX` and all 60 verdict joins
missed. The totality check turned that into a loud failure instead of silently mismatched
verdicts. Scoring now submits opaque positional ids and maps back.

## 5. Substrates

| id | model | role | context |
|---|---|---|---|
| `evo2-1b` | Evo2 1B | primary substrate, all arms | [O] G2 |
| `go-4b` | GenomeOcean 4B | second substrate, same arms — **DEFERRED TO STAGE 2** [C] | [O] G2 |
| `bgcfm` | published BGC-finetuned GenomeOcean | external published baseline, generation only, no training | [O] G2 |

`bgcfm` is prior art run as-published: it establishes what an existing BGC generation model does
under this benchmark. It is never fine-tuned by us.

### 5.1 Termination and tokenisation differ between substrates — measured [M]

Neither substrate does what is wanted by default, and they fail in OPPOSITE places:

| | terminator | auto-appended at encode? | declared as `eos_token`? | tokenisation |
|---|---|---|---|---|
| **Evo2** | byte **0** | **NO** | yes (`eos_id=0`) | byte-level; 1 token ≈ 1 nt |
| **GenomeOcean** | `[SEP]` = **2** | **YES** | **NO** (`eos_token_id` is `None`) | BPE, vocab 4096 |

Consequences, each of which must be implemented rather than assumed:

1. **Evo2 will never learn to stop unless the terminator is appended to training sequences.**
   Its tokenizer does not add one (`tokenize("ACGT") -> [65,67,71,84]`). Token 0 is native — we
   invent nothing — but it must be written into the training text.
2. **GenomeOcean appends `[SEP]` automatically but will not stop on it**, because
   `eos_token_id` is `None`, so `generate()` has no stop criterion unless id 2 is passed
   explicitly.
3. **Token count ≠ nucleotide count on GenomeOcean.** `ACGTACGTACGT -> ACG/TACG/TACG/T`. The
   length bound (§4.7) and the generation budget (§7.1) are specified in **NUCLEOTIDES** and
   converted per substrate. A budget in tokens would give the two substrates different amounts
   of sequence.

**Gate G10 — termination, per substrate, before any arm runs.** Verify the terminator id from
what the model actually emits; verify training text carries it; verify generation stops on it;
report `hit_eos` rate and the realised length distribution against real held-out cores.
⚠ **This gate bears directly on the multi-gene axis.** If generations run to the full budget
while real cores are ~1–9 kb, `frac_multigene` compares a 16 kb budget against a ~1 kb reference
(a 15,992 nt concatenation produced 15 core genes). If termination works, lengths become
comparable and the confound largely dissolves. Until G10 passes, gene-count comparisons are
reported per-nucleotide as well as per-sequence.

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
- `W1n` pooled adapter, **balanced by NUCLEOTIDE**: per-class loss weights so the classes
  contribute equally to the gradient.
- `W1` pooled adapter over the benchmark classes at the common effective_n (§4.4.3), i.e.
  balanced by RECORD.
  ⚠ **EQUAL RECORDS IS NOT EQUAL TOKENS, and an earlier version of this spec wrongly said
  the two were the same** [M]. ⚠ The per-class nucleotide shares that stood here are REMOVED —
  they came from the superseded 16 kb five-class build (979 records/class; NRPS 24.8%,
  BETALACTONE 33.9%) and neither class is in the benchmark. On the built corpus the imbalance
  persists for the same reason: median core length runs TERPENE ~1,157 nt to ARYLPOLYENE
  ~3,599 nt, a ~3× spread, at 8,044 records per class — and the loss is per token, so `W1`'s gradient
  is dominated by the long classes. `W1` is therefore the RAW-mixture arm and `W1n` is the
  balanced one; data mixture is itself a control method (§6) and the pair measures it. This is the controlled comparison to `W2`: it sees the *same amount
  of class C* plus the other benchmark classes, so the contrast isolates class-exclusivity from
  data volume.
  **A full-corpus pooled arm is deliberately NOT included** [C]. It would reintroduce the exact
  confound §4.4.3 removes — trained on everything it sees ~8,844 TERPENE clusters against ~1,224
  ARYLPOLYENE, so its per-class results would be confounded by class frequency. The "was everything
  undertrained?" objection it would answer is answered better, and empirically, by **G8**.
- `W2` per-class adapter — class enters through *which weights are loaded*.
- `W3` **learned per-attention-site conditioner** — a trainable vector (plus an optional low-rank
  term) added at every attention site, learned by gradient descent on a FROZEN base model.
  **AMENDED 2026-09-07 from "prefix tuning / learned key-value states" — see §12.A1.** The
  input-only variant (prompt tuning) is a strictly weaker mechanism and is not what this arm
  tests; §6.5 covers the architectural asymmetry between substrates.
  It pairs with `I1`, which injects a DERIVED direction at the same sites: W3 learns the
  direction, I1 derives it from class means. One code path (`model/interventions.py`).
  **Capacity is swept, not fixed** — the bare offset is 7,680 parameters against LoRA's
  10,475,520 on the same model, so a null at that capacity would say "too few parameters"
  rather than "activation-space conditioning does not work". Default rank 16 = 253,440.

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
  **The seed is the first `L` nt of a held-out core of the target class**, `L` set by Gate G3.
  **RESOLVED 2026-09-10: L = 64**, frozen in `genconfig.FROZEN`. Chosen as the longest rung
  that is NOT confounded rather than the highest-scoring one — at L>=128 the seeds are
  themselves antiSMASH-detectable, and at L=512 the seed-only baseline (0.412) exceeds the
  generation rate (0.316). See FINDINGS 9.5-9.7. Held-out (never trained on), first-`L` (unambiguous and reproducible — no motif choice,
  no consensus construction, nothing to tune). The seed span is **excluded from the scored window**
  (§7) and a test pins that it never appears in scored text.
  **`L` is not inherited from any prior work or any other paper.** Values quoted elsewhere were
  measured on a different task, a different corpus and a different model, and carrying one over is
  precisely the assumption this rebuild exists to remove. G3 measures it here, per class.

**Inference-time control:**
- `I0` none.
- `I1` activation steering — inject a class direction into the residual stream during
  generation. Fully specified here because every other free parameter carries an `[O]` tag and a
  named gate, and an unspecified arm cannot be built blind.
  - **Direction: DIFFERENCE OF MEANS, not probe weights.** For class *C* at layer *L*:
    `d = mean(activations over C's TRAIN records) − mean(activations over all TRAIN records)`,
    length-normalised. **Train only** — the prior codebase carried a leakage debt from fitting
    directions on val+test. No probe is fitted, so §3.7's ban on learned-probe *endpoints* is
    not even in tension; a probe would be an intervention, never a metric.
  - **Injection site and magnitude:** swept together by **Gate G9**, **PER CLASS AND PER WEIGHT
    STATE — see §12.A5**. The sweep criterion is NOT the endpoint (§2.4) — it is the largest α
    at which generation quality is not degraded beyond a stated tolerance, read as per-token
    likelihood on the model's own output and coding density, both **one-sided**: an improvement
    in either is not degradation. Termination rate is a third axis, also one-sided.
    ⚠ **α IS NOT TRANSFERABLE ACROSS CLASSES.** Measured, not assumed: α = 0.3 chosen on
    TERPENE left ARYLPOLYENE at 29% of its baseline coding density and REDOX_COFACTOR at 66%,
    destroying both arms, while TERPENE and RIPP were unaffected. An arm run at another class's
    α is uninformative under §6.4 regardless of what it scores.
  - **MANIPULATION CHECK, probe-free, THREE parts, all required. AMENDED 2026-09-11 — see
    §12.A4.** (a) projection onto `d` increases monotonically with α — confirms the hook fired;
    (b) KL divergence between steered and unsteered next-token distributions is non-trivial —
    confirms it reached generation; (c) **the direction REPRODUCES**: `d` derived on TRAIN and a
    readout derived independently on VAL have a positive cosine at every steered site, and a mean
    above 0.3 over the steered sites.
    (a) alone is circular: it measures the quantity we injected. (b) is the one that licenses
    reading a null. **(c) is the one that distinguishes a class direction from a push**, and
    neither (a) nor (b) can do it: a random vector passes (a) *by construction*, and passes (b)
    too at any α large enough to perturb the distribution. Without (c) the only thing separating
    signal from magnitude is the §6.3 random-direction control, which is an ARM — so a failure
    would be discovered after 200 generations rather than at derivation.
    ⚠ The mean in (c) is taken over the sites the arm actually STEERS. A site whose class
    difference falls below the relative-norm floor is zeroed and not injected, so its cosine is
    identically 0; counting it would penalise the arm for a site it does not touch, and measured
    on ARYLPOLYENE that single arithmetic choice flipped the verdict (0.230 over four sites,
    0.307 over the three steered).
  - **Control:** magnitude-matched random direction (§6.3).
  - ⚠ Substrates expose different residual streams (§6.5); the injection site is recorded per
    substrate and an arm that cannot be implemented is NOT APPLICABLE, never a zero.
- `I2` iterative refine — ⚠ **DROPPED from the grid by decision (§14.3), not deferred.** Retained here only so the coordinate it would have occupied is legible. Generate, detect, retain the rule-satisfying span, re-seed from it,
  repeat to a fixed iteration cap.

### 6.0 Degenerate coordinates — collapse before sizing

With no conditioning channel in the input (§4.3, the project's premise), **an arm with no
class-bearing factor produces the SAME generations regardless of target class.** `W0/S0/I0` and
`W1/S0/I0` are one run each, not five, and their output populates every row of the confusion
matrix identically — which is exactly the unconditioned marginal §3.5's lift divides by. Six
cells collapse to two. State this when sizing; counting them as thirty overstates the budget.

**Sequencing [C]:** run the weight-state arms first (`W0`, `W1`, `W2`, `W3`), read them, and pick
the base for the composable `S` and `I` factors from that data rather than assuming `W1`.

### 6.0a Training protocol — the same for every weight-state arm

Arms differ in their DATA, never in how they are optimised. Any difference in training
procedure between arms is a confound on the axis the benchmark reports.

* **Early stopping, not a guessed epoch count** [C]. `max_epochs=40`, held-out evaluation
  every 25 optimizer steps, `patience=4`, `min_delta=1e-4`. **AMENDED 2026-09-08: `max_epochs`
  12 -> 40, and the cap must not bind — see §12.A2.** A cap that terminates a run is a
  guessed epoch count wearing early stopping's name. A fixed count cannot know
  whether an arm converged — measured on a first pass at 3 epochs, three of six arms still
  had headroom while two had already turned over, and only luck kept the rest from being
  under-trained.
* **Generation uses the BEST held-out checkpoint, never the last** [M]. `final` is whatever
  the last step produced; on that same first pass `W1` and `W2_RIPP` both had a `final`
  worse than their best, which handicaps exactly those two arms and nothing else.
* **Reported train loss is the mean since the last evaluation**, not the single batch that
  landed on a checkpoint. A one-batch train loss is too noisy to read a curve from, which
  is why overfitting was invisible on the first pass even where it had begun.
* **No arm is resumed from another arm's state.** Resuming restores adapter weights but not
  optimizer moments, so a resumed arm and a from-scratch arm have different optimisation
  histories. Every arm is trained from scratch under identical settings.

### 6.1 Stage 1 — the shakedown (build verification, not inference)

Every arm run **once** at small n on `evo2-1b` / `W1r` for composable factors, to prove the harness
end-to-end and to estimate rates for powering.

Stage 1 exists to prove the harness end to end and to estimate rates for powering. Its output
is **labelled `stage: "stage1"` in the scored record (§3.9)** and is never pooled with Stage 2.
That is a field rather than a paragraph because a prose prohibition cannot be audited afterwards.

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
| checkpoint selection | the arm is evaluated at its BEST held-out checkpoint, not its last. Training uses **early stopping** (§6.0a: `max_epochs=40`, evaluation every 25 steps, patience 4), so `final` is whatever the patience window ended on rather than the best point; ⚠ an earlier version of this row said "a fixed epoch count with no early stopping", contradicting §6.0a and describing a protocol no arm has ever run; measured on the first six arms, `W1` and `W2_RIPP` both had a `final` worse than their best, which would have handicapped exactly those two arms |
| `W1`/`W2` | training loss falls on held-out data of the target class; monotone across ≥5 checkpoints |
| `W3` | held-out loss WITH the conditioner attached vs WITHOUT it, on the same records. Zero-init makes the unintervened model an exact baseline, so the difference is measurable rather than asserted. A one-sided number — loss with the conditioner only — is not a check: it has no reference |
| `I1` | **three parts, all required (§12.A4)**: (a) projection onto `d` monotone in α, (b) non-trivial KL between steered and unsteered next-token distributions, (c) the direction REPRODUCES — train-derived vs independently val-derived cosine positive at every steered site, mean > 0.3. (a) and (b) are both passed by a RANDOM vector; (c) is what separates content from push |
| `I2` | the retained span is present in the re-seeded prompt and absent from scored text |
| `S1` | seed present in prompt, absent from scored span |

---

### 6.4a Length-bucketed batching — permitted, with one guard

Padding to the longest member of a batch wastes compute at these length spreads, so batches may
be bucketed by length. **One caveat, specific to the pooled arm** [C]: class medians run from
TERPENE ~1,157 nt to ARYLPOLYENE ~3,599 nt (manifest medians at the 8,192 nt bound; an earlier version said "TERPENE ~1.3 kb to BETALACTONE ~9.0 kb", a class the benchmark does not contain), so on `W1` a pure length bucket is still substantially a
CLASS bucket, and gradient updates would alternate between class-homogeneous batches — a
training dynamic introduced by accident, on the one arm whose premise is that it sees all
classes together.

Required: bucket by length, then require each batch to draw from **≥2 classes** wherever the
lengths permit; and verify empirically that held-out loss matches an unbucketed run before the
optimisation is trusted. It is a speed change and must be shown to be only that.

### 6.5 Where substrates are not architecturally equivalent — declare, do not paper over

Some arms cannot be implemented identically on every substrate, and the honest response is to state
the asymmetry rather than pretend equivalence.

- **`W3`** attaches a learned conditioner at each attention site. Substrates whose architecture is
  not uniformly attention (e.g. hybrid convolution/attention stacks) expose far fewer attachment
  points than a standard transformer: **measured, Evo2-1B has 4 attention blocks of 25 (blocks
  3, 10, 17, 24) — 16% coverage.** **Record, for each substrate: total blocks, attention sites,
  and the count actually carrying the conditioner**, and record them in the RUN artifact, not only
  the training one. An arm attached at 4 of 25 sites on one substrate and 32 of 32 on another is
  not the same arm, and the comparison must say so.
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
   a. ⚠ **There is NO fixed scoring window** — §7.2 above: the entire generated sequence is
      scored, and a common window would destroy the multi-gene signal §4.5 exists to measure.
      An earlier version of this clause said the opposite while citing §7.2 as its authority.
      What IS applied identically to real cores and negative controls is the **whole-sequence
      scoring itself** plus any post-processing (§6.3), so no class of input is truncated,
      trimmed or padded differently from another.
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
5. **Every artifact records the RUN, not the intent.** A provenance field that states what
   was configured rather than what happened is decoration; the test for it is whether a
   deliberately drifted run produces a different fingerprint. Each run records and hashes:
   n, budget, batch size, RNG seed, decoding parameters, seeded flag and seed length,
   weight state, resolved adapter path **and its content hash**, row class, substrate and
   checkpoint, **termination mode**, corpus sha256, scoring-config hash, **the class-map
   hash**, the novelty gate parameters and reference classes, and **the antiSMASH version
   that ACTUALLY RAN** — not the one expected. The run directory is
   `<stage>_<SUBSTRATE>_<ARM>_<CLASS>_<run hash>` and **writing into an existing one
   raises.**
6. **Termination is enforced on both substrates, by different mechanisms, and the mechanism
   is recorded** (§5.1). GenomeOcean halts natively on its terminator id; Evo2 generates to
   budget and its output is truncated at the first terminator, because vortex's
   `stop_at_eos` prints and does not break. Output is equivalent; internal compute is not.
7. **Empty generations are scored as non-detections, never dropped.** A draw whose
   terminator lands at position 0 is a real outcome of a model that learned to stop.
   Removing it inflates the rate, and the bias is DIRECTIONAL — only a model that emits its
   terminator can produce one, so the deletion concentrates in trained arms and is absent
   from the base control they are compared against.

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

**δ IS COMPUTED FROM TWO MEASURED QUANTITIES, NEVER CHOSEN BY TASTE.**

```
δ = max( instrument resolution , smallest effect detectable at 80% power at the planned n )
```

* **Instrument resolution** is the exact-binomial upper bound on the measured false-positive
  rate. At the current floor of 0/300 that is **0.0099** [M] — you cannot honestly claim to
  resolve an effect smaller than your own instrument's uncertainty.
* **Detectable-at-n** falls out of the pre-registration once n is fixed.

Both components are reported. Both are computable **before any arm runs**, which is what makes δ
pre-registrable rather than a number chosen after seeing the data.

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

⚠ **CORRECTED 2026-09-16. The tree below is what exists; the previous version of this section
described a layout that was never built** — a `conf/` directory of per-class, per-arm and
per-substrate YAML (configuration ended up in code), `data/build_corpus.py`,
`score/classmap.py`, a `stats/` package of `power.py`/`tests.py`/`report.py` (never written),
and `run/stage1.py`/`stage2.py` as the entry points. A reader following it would not have
found a single file at the stated path.

```
bgcbench/
  provenance.py                  # code_version, corpus_max_len, freeze_id
  data/      extract.py genbank.py tar_index.py mibig.py negative.py   # raw -> corpus
             classmap.py cluster.py split.py manifest.py taxonomy.py   # corpus -> splits
  model/     load.py train.py generate.py genconfig.py                 # substrate-agnostic
             interventions.py directions.py substrate_config.py        # I1 steering
  score/     antismash.py novelty.py record.py endpoints.py            # ONE scoring site
  run/       build_data.py train_arm.py derive_directions.py           # chain steps 2-4
             g9_sites.py g9_alpha.py arm.py                            # steps 5-7
             results_table.py checkpoint_ledger.py                     # step 8 + provenance
             oracle.py score_gates.py g2_health.py g10_termination.py  # calibration
pipeline/                        # the 8 scripts that invoked the 48 arms, in order
tests/                           # 136 tests; see §11
reference/frozen/                # KNOWN_WRONG.md, sweep_evidence.json. never imported
docs/                            # SPEC (method) DEFECTS (provenance) RESULTS FINDINGS
```

The four calibration entry points under `run/` are not part of the chain and nothing imports
them, but they are what licenses reading a near-zero arm: the G5 ceiling and G1 floor
(`score_gates.py`), the harness proof (`oracle.py`), the models-beat-shuffled-DNA check
(`g2_health.py`) and the terminator mechanism (`g10_termination.py`). See `RESULTS.md` §1.

**Hard structural rules, each pinned by a test (§11):**
1. `score/` has exactly one antiSMASH invocation site and one endpoint computation. No arm-specific
   branch anywhere under `score/`.
2. `generate.py` takes an arm config; it contains no arm names.
3. Every scored artifact filename encodes its scoring-config hash. Two scorings of one generation
   set can never share a name.
4. No loose files at any run root; every artifact belongs to a run directory.

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

⚠ **AUDITED AND CORRECTED 2026-09-16: 12 of this table's 13 test names did not exist.** They
were written before the code, as the section title says, and were never reconciled with the
names the suite actually grew — so a section whose purpose is "this constraint is pinned by a
test" was naming tests that are not there. `test_known_wrong_and_spec11_test_names_all_exist` now
parses this table and fails if any name is absent.

| test | pins |
|---|---|
| `test_novelty_gate_fails_closed_on_empty_kmers` | empty/missing k-mer set raises; never passes |
| `test_manifest_additive` | writing class B preserves class A's entry |
| `test_split_uses_every_member_of_an_assigned_cluster_without_leaking`, `test_built_splits_are_leak_free` | zero genome overlap AND zero cluster overlap; zero near-dups forward and reverse |
| `test_built_splits_are_balanced` | every class has non-empty train/val/test at equal effective_n |
| `test_seq_len_is_len_sequence` | `seq_len == len(sequence)` on every record |
| `test_single_antismash_invocation_site` | exactly one antiSMASH invocation in `score/` |
| `test_generate_has_no_arm_specific_branch`, `test_no_arm_or_class_branching_in_score` | `generate.py` and `score/` contain no arm identifiers |
| `test_no_truncation_of_model_output_in_score` | no truncation of model output anywhere in `score/`; only a seed span may be excluded |
| `test_seeded_generation_never_scores_the_seed` | seeded arms: seed never in scored text |
| `test_scoring_config_is_frozen_and_hashed`, `test_check_uniform_fires_on_realised_not_frozen` | all arms resolve to one scoring config hash, checked on the REALISED value |
| `test_every_sequence_gets_a_verdict_rejects_duplicates` | antiSMASH scored-count == submitted-count; a shortfall raises |
| `test_run_directory_refuses_to_overwrite_a_measurement` | a second run cannot destroy the first's measurement |

**Two rows from the original table are retired rather than renamed:**

⚠ **test_case_collision** (name unbackticked: it is not a test that exists) — "run-dir creation rejects case-insensitive duplicates" was **never
implemented**. `test_run_directory_refuses_to_overwrite_a_measurement` guards exact-name
collision, which is what actually bit (the realised-config hash makes two configurations differ
in name); case-insensitive collision was a hypothesised failure that never occurred in 48 arms.
Registered here rather than quietly dropped.

⚠ **test_input_is_bare_sequence** (likewise) — "no tag/taxonomy reaches the model" **is not a constraint
this benchmark holds**, and the row was wrong as written. §4.3's default is bare sequence and no
class tag or metadata ever reaches either model, but a TAXONOMY prefix is permitted for Evo2
because GTDB lineages are its native pretraining format: all 24 final Evo2 arms run
`prefix=taxonomy`, all 24 GenomeOcean arms run `prefix=none`. What is pinned instead is that the
prefix is declared, per substrate, and cross-checked —
`test_a_direction_cannot_be_attached_under_a_different_prefix` and
`test_lineage_prompts_are_one_uniform_width`.

---

## 12. Gates and open decisions

**Gates — measured before the work that depends on them.**

| id | gate | blocks |
|---|---|---|
| G1 | ✅ **CLOSED** [M]. minlength = 1; class FPR **0.000 (0/300) for every class** on real non-BGC coding DNA. Going low costs no specificity | final scoring config |
| G2 | ⏸ **PARTIAL.** Evo2-1B ✅ [M]: 16k @ 6.38 GiB, 32k @ 10.61, 64k @ 19.08 · health PASS (0.913 real vs 1.347 shuffled). GO-4B and bgcFM now **load and generate** (G10), but their health check — likelihood on real cores vs shuffled — has not been run | all generation |
| G3 | seed-length sweep for `S1` | seeded arms | ✅ **RUN 2026-09-10** — `G3_FROZEN_816d66441f8624ca`. Threshold between 16 and 32 nt. **Reportable value: L=64, on-target 0.077, p=7.0e-21, seed-only baseline 0.003.** Full ladder `G3_FULL_FROZEN_00820b3404a2acc4` (8 points, 8-512 nt). ⚠ **L=64 is the last uncontaminated rung**: above it the seeds are themselves antiSMASH-detectable, and at L=512 the seed-only baseline (0.412) EXCEEDS the generation rate (0.316) — past ~64 nt the rise is the seed, not the method. Batch size does not move the endpoint (L=128 at batch 80 vs 48: 0.184 vs 0.168). See FINDINGS 9.5-9.6. |
| G4 | decoding-parameter policy: swept or fixed | Stage 2 |
| G5 | ✅ **CLOSED** [M], re-measured on the built corpus `c74154974aff` (`gates/gates_ee8c025c1593_c74154974aff.json`): **on-target 1.000 for all four** of TERPENE, RIPP, ARYLPOLYENE, REDOX_COFACTOR at n=82 each, against a 0/300 negative-control floor (G1 false-positive 0.000 for every class). ⚠ An earlier row quoted corpus `0225546040b9` with NRPS 0.975 and BETALACTONE 1.000 — two classes the benchmark does not contain — and omitted REDOX_COFACTOR | interpretation of every rate |
| G6 | adapter rank sweep on held-out loss, per substrate (§6) | every `W1`/`W2` arm | ✅ RUN 2026-09-10, rank 16 retained (FINDINGS §13) |
| G6b | adapter **depth** sweep — which blocks carry adapters — on held-out loss, at the rank G6 selects | every `W1`/`W2` arm | ✅ RUN 2026-09-11, all-blocks retained (FINDINGS §14) |
| G7 | ✅ **~0.2 s/sequence** at 8 CPUs, `--minimal` [M] — 50,000 sequences ≈ 2.8 h. **Scoring is NOT the binding resource**, which reopens D3 | Stage 2 sizing |
| G8 | data-scaling: effective_n at which the endpoint saturates | **the class set (§4.4)** and equal-n subsampling |
| G9 | steering layer × magnitude, swept on generation quality — never on the endpoint (§2.4) | the `I1` arm | ✅ RUN 2026-09-11/12, **per class** (§12.A5); FINDINGS §15, §17 |
| G10 | ✅ **MECHANISM CLOSED** [M] for all three substrates: terminator id round-trips, training text carries it, truncation detects it. Base-model behaviour at 4 kb: Evo2 **0/12** hit_eos (runs to budget) · GO-4B **12/12** (median 553 nt) · **bgcFM 0/12** (median 4,792 nt — the published fine-tune LOST its base model's stopping). BPE ratio measured at **4.8 nt/token**. ⏸ Whether a *fine-tuned* Evo2 emits its terminator is a post-training measurement | every generation arm, and the gene-count axis (§5.1) |
| G11 | ⛔ **RETIRED 2026-09-14, NEVER CLOSED.** It existed to SELECT a `top_k` per substrate and it could not: its winner beat the runner-up by **0.59 sd** of the statistic's own sampling noise, and the whole healthy-rung range fit inside one configuration's noise band. Neither substrate now needs a selection — Evo2's `top_k=4` retains 0.9999 of its mass over a 4-letter alphabet, and GenomeOcean applies **no truncation** (`top_k=0`, all 4,096 tokens). ⚠ The MEASUREMENTS survive and still rule out `top_k=4` on GO (mass 0.1280, distinct-21mer 0.3522) — FINDINGS §25 | — |

**Bound resolved: 8,192 nt** [M]. ⚠ REVERSED — this read "Bound resolved: 16 kb" and reasoned from G2's memory measurements that the bound was a COST decision. FINDINGS §1.5c showed that mistook *it runs* for *it works*: held-out NLL rises 0.8049 at 8,192 → 1.2388 at 15,900 against ln(4)=1.386 for a uniform model, i.e. near chance. It is a MODEL-QUALITY decision. Superseded reasoning follows: G2 shows the 1B is not the constraint (64k fits in 19 GiB), so the
bound is a cost decision; 16k captures ~92% of 32k's data benefit at half the generation and
scoring cost. **The 7B is not required.**
⚠ **SUPERSEDED.** This read "G8 now gates the class set and is the last gate before code." G8 has **never been run** (§14.2) and the code was built, run and frozen without it — the class set was instead fixed by the 8,192 nt rebuild (four classes; §4.4.2 as amended in the header). G8 remains open as a data-scaling sweep, gating nothing. G1 also carries G7's throughput measurement.
**G5**  It sets the dynamic range every arm is measured against, and if
any class's ceiling is low the whole benchmark for that class is compressed toward the floor.

**Amendments and the deferred register live in [`DEFECTS.md`](DEFECTS.md).** §0 requires every
spec change to be filed and dated; that file holds §12.A1–§12.A10 and the §14 register, split
out on 2026-09-16 when this section had reached 610 lines. Cross-references of the form §12.A3
and §14.4 throughout this document resolve there, and their numbering is unchanged.

## 14A. STANDING RULE — nothing is matched across substrates except the TASK

**The two models are treated differently and given the same task. What is compared is the task
outcome, never the configuration.**

⇒ Every substrate-dependent parameter is chosen **per substrate, by measurement**: the adapter
rank, the adapter depth set, the injection site set, the injection magnitude α, the probe
magnitude used to select any of those, the batch size, the token budget's conversion to
nucleotides. A configuration that resembles the other model's gets **no privilege** and is not a
tie-breaker.

⚠ **What IS held identical**, because it is the task: the corpus, the splits, the class set, the
frozen scoring config, the novelty gate, `n_per_row`, the **nucleotide** budget, the seed length
in nucleotides, and the endpoint. Anything a substrate needs in its own units converts from the
nucleotide figure (§15.4) rather than being copied as a token count.

**Measured cost of getting this wrong.** GenomeOcean's I1 site set was nearly imposed as Evo2's
relative depths (`evo2_matched`: layers 3, 10, 16, 23, one per depth region, mimicking Evo2's
4-of-25 spacing). Measured against GenomeOcean's own best set (`early_third`: layers 0–7,
concentrated): **0.023 against 1.093 reach — a ~47× penalty.** GenomeOcean wants injection
concentrated early; Evo2's spacing spreads it across depth. Had the matched set been used, GO's
steering arm would have run at ~2% of its achievable effect and any null from it would have been
an artifact of the site choice rather than a property of the substrate.

⚠ **The rule reaches the measuring instrument too, and that is where it was first broken.** After
correctly refusing to match the SITES, the site sweep still probed both substrates at one effect
size (KL ≈ 1.0). For Evo2 that is α ≈ 1.0, above its own measured health ceiling of 0.3 — so its
site sets were ranked in a regime its generation does not survive. **Probe each substrate inside
its own working regime**, defined by its own G9 ceiling.


### 14A.1 GATE-BY-GATE CLASSIFICATION — added 2026-09-14 after G3 was found misfiled

Every gate is either a property of the **TASK** (shared; a per-substrate answer voids the
cross-substrate comparison) or of the **TREATMENT** (per substrate; a shared answer imposes one
model's configuration on the other). Getting this backwards fails SILENTLY in both directions:
both arms run, both report, and only the comparison is wrong.

| gate | what it sets | class | status |
|---|---|---|---|
| G1 | scoring config, `minlength` | **TASK** | shared ✅ |
| G2 | substrate likelihood health | treatment (diagnostic) | per substrate ✅ |
| **G3** | **seed length `L`** | **TASK** | ⚠ **WAS MISFILED** — §15.6 told GO to re-select L. Corrected 2026-09-14; both substrates use L=64 |
| G4 | decoding policy | treatment | per substrate ✅ — Evo2 `top_k=4` (unrestrictive), GenomeOcean `top_k=0` (no truncation). G11 retired, §12.A8 |
| G5 | oracle on real cores | **TASK** | no model involved; shared ✅ |
| G6 | adapter rank | treatment | per substrate ✅ |
| G6b | adapter depth | treatment | per substrate ✅ |
| G7 | scoring throughput | infrastructure | n/a |
| **G8** | **data scaling — effective_n** | **TASK** | ⚠ **NOT YET RUN. It must NOT be answered per substrate.** §4.4.3 fixes equal effective_n at split time and there is exactly one dataset; a per-substrate `n` would mean the two models trained on different amounts of data. G8 may REPORT a saturation curve; it may not re-cut the corpus for one substrate |
| G9 | steering site × magnitude | treatment | per substrate AND per class ✅ |
| G10 | termination mechanism | treatment (diagnostic) | per substrate; the derived floor is shared in **nucleotides** ✅ |
| G11 | decoding configuration | treatment | ⛔ RETIRED — could not discriminate; §12.A8 |

⚠ **`budget_nt` = 8,192 is a TASK constant set by ONE substrate's limit.** It comes from
`evo2-1b`'s `max_seqlen` and the measured NLL degradation past it. GenomeOcean's context is far
larger (10,240 BPE tokens ≈ 50 kb), so GO is generating well inside its own capacity. That is the
correct choice — an equal nucleotide budget IS the task — but it is a constraint imposed by Evo2
and must be reported as such rather than presented as each model's natural operating point.

## 15. GenomeOcean Stage 1 — the mirror of Evo2, and where it cannot be one

Stage 1 is repeated on `go-4b` to make the substrate comparison the paper's subject. The rule is
**same data, same endpoint, same protocol, same order**; only substrate-dependent constants are
re-measured. Where a structural difference makes an arm non-identical, §6.5 applies — it is
reported as a difference, never equalised by handicapping one side and never scored as a zero.

### 15.1 What is shared, unchanged

The corpus, the splits, the class set, the frozen scoring config `ee8c025c1593`, `--minlength 1`,
the novelty gate, `n_per_row` 200, `budget_nt` 8,192, temperature 1.0, top-k 4, top-p 1.0,
`rng_seed` 0, the L = 64 nt seed, and the arm grid (W0/W1/W1n/W2/W3 × S0/S1 × I0/I1). **One
dataset exists** (§4.4.3): GenomeOcean trains on exactly the records Evo2 trained on.

### 15.2 The four structural differences, measured not assumed

| | Evo2-1B | GenomeOcean-4B | consequence |
|---|---|---|---|
| tokenizer | byte-level, 1 nt/token | BPE 4096, **~4.8 nt/token** | every token budget converts; the nucleotide budget is what is held equal |
| attention sites | **4 of 25** blocks (3,10,17,24) | **24 of 24** layers | `I1`'s site set is ambiguous — see §15.3 |
| adapter targets | `l1/l2/l3`, `out_filter_dense`, `Wqkv`, `out_proj` | `q/k/v/o_proj`, `gate/up/down_proj` | mirrored **by role**; GO has no Hyena filter and that absence is reported, not padded |
| termination | does not stop natively; needs a decoder shim | **stops natively and eagerly** | both need the same nucleotide floor — see §15.4 |

⚠ A fifth difference is deliberately NOT exploited. GenomeOcean can take a **trainable class
token** and Evo2 cannot. §4.3 forbids a class token on both, so GO runs without the one
conditioning channel it uniquely supports. That is the honest comparison of the two as they
stand, and it is also the single most interesting thing a follow-up could test.

### 15.3 `I1` site selection — matched on DEPTH, not on count

Evo2's four attention blocks sit at fractional depths 0.12, 0.40, 0.68, 0.96. GenomeOcean has
attention everywhere, so "the same arm" is ambiguous: inject at all 24 and the substrates differ
in how hard they are pushed as well as in what they are; inject at four matched positions and the
contrast is about the substrate.

⇒ **Primary GO `I1` arm: layers 3, 10, 16, 23** (`interventions.matched_depth_subset`), the same
relative depths. The all-24 variant is a secondary arm if time allows, and the realised site list
and coverage are recorded per arm regardless (§6.5).

### 15.4 ⚠ The min-token floor was MISSING on the HF path, and would have broken the comparison

`suppress_terminator` gives Evo2 a floor of `min_new_tokens` and **returns early for every other
family**, so the HuggingFace generation path had no floor at all. GenomeOcean terminates natively
and eagerly — the prior project measured EOS firing straight after the seed, 61/200 empty
generations, and a floor taking that arm from 0.400 to 0.580.

⇒ Without it, GO would have been scored on truncated output and the deficit attributed to the
substrate. Fixed, and **the floor is in NUCLEOTIDES**: 1,000 nt is 1,000 tokens for Evo2 and
**208** for GenomeOcean at 4.8 nt/token. Passing 1,000 tokens to both would demand 4.8× more
sequence from GO — not the same floor.

### 15.5 Gates to re-measure for `go-4b` — every substrate-dependent constant

| gate | why it cannot be inherited |
|---|---|
| **G2** likelihood health | never run for GO (§14.2 records it PARTIAL). Real vs shuffled cores. **Blocking: run first.** |
| **G3** seed length | L = 64 nt is ~13 BPE tokens against 64 byte tokens. The threshold is a property of the tokenizer as much as the model. |
| **G6** adapter rank | swept on held-out loss, per substrate — §6 says so explicitly. |
| **G6b** adapter depth | 24 layers, all with attention; Evo2's answer (placement decisive, early blocks) may not transfer. |
| **G9** α × site | per class AND per weight state (§12.A5), on a different residual stream. |
| G10 termination | ✅ already measured for GO: 12/12 `hit_eos`, median 553 nt. |
| G5 / G1 instrument | ✅ substrate-independent — antiSMASH does not know which model wrote the sequence. |

### 15.6 Order of operations

1. **G2** likelihood health — if GO cannot model these cores, nothing downstream is interpretable.
2. **G6** rank, then **G6b** depth, on the pooled `W1n` arm (held-out loss only, §2.4).
3. Train `W0`/`W1`/`W1n`/`W2`×4/`W3` at the selected rank and depth.
4. Stage 1 de novo, both prefix conditions (see §15.7).
5. Seeded arms at the **SHARED** `seed_len_nt` (64). ⚠ **G3 IS NOT RE-RUN FOR GenomeOcean AND
   L IS NOT RE-SELECTED** — corrected 2026-09-14. This step previously read "G3 seed sweep, then
   the seeded arms at the selected L", which contradicts §14A: the seed length is part of the
   **TASK**, not of the treatment, and §14A lists it among what is held identical. Two substrates
   given prefixes of different lengths are not solving the same problem, and the seeded
   cross-substrate comparison — the one the paper exists to make — would be void.
   Everything that is a property of the MODEL (rank, depth, injection sites, α, decoding) is still
   chosen per substrate by measurement. The seed is not one of those. G3 may be re-run on
   GenomeOcean as a *reported measurement*, but the benchmark arms use L = 64 either way.
6. Derive `I1` directions (three-part check, §12.A4), **G9** per class, then the `I1` arms with
   health-matched random controls (§20.1).
7. Freeze, then the composition cell.

### 15.7 ⚠ The taxonomy prefix is NOT run on GenomeOcean — DROPPED 2026-09-13

Evo2 was pretrained with GTDB lineages, which is why the prefix is its native input format and
why §10 could show it causal. **GenomeOcean was not.** A lineage prefix on GO is a format it has
never seen.

⚠ **An earlier version of this section required BOTH prefix conditions on GO**, reasoning that
Evo2 has both (`W2_*_noprefix` and `W2_*_tax`) so GO should too. **That is matching for its own
sake and §14A forbids it.** Evo2 has both conditions because the prefix is a real channel for
Evo2 and the pair measures it. On GO there is no such channel to measure — the prefixed arm would
exist only to resemble Evo2's grid.

⇒ **GenomeOcean runs on bare nucleotide sequence, its native condition.** Every GO arm is
`--prefix none`. The comparison remains valid because §14A holds the TASK identical, not the
input formatting: both substrates receive the same corpus, splits, classes, endpoint and
nucleotide budget, each in the form it was pretrained to read.

⚠ Consequence to state in the paper: Evo2's headline arms carry a taxonomy prefix and GO's do
not, so the two are not input-identical. That is a property of the substrates, not a confound
introduced here — equalising it would mean either stripping Evo2 of its native format or handing
GO one it cannot use, and both make the comparison worse.


