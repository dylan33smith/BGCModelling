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

### 1.0 antiSMASH silently sanitises record ids `[instrument]`
It strips colons: a record submitted as `oracle::TERPENE::GCF_x.region2` is returned as
`oracleTERPENEGCF_x.region2`. All 60 verdict joins missed, and only a totality check
(submitted-count vs scored-count) turned it into a loud failure rather than silently mismatched
verdicts served to the wrong records.
**Paper:** anyone joining antiSMASH output back to input by record id must either verify the
round trip or, better, submit opaque ids they control and map back. This is a general trap.

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

### 1.2 Ceiling ~1.0 against a floor of 0/300, verified end to end `[instrument]`
Corpus `0225546040b9`, `--minlength 1`, through the single invocation site:

| class | ceiling | FPR (on-target) | ORACLE (full arm path) | real-core multi-gene | modal share | products |
|---|---|---|---|---|---|---|
| TERPENE | 1.000 | 0.000 (0/300) | 1.000 | 0.167 | 0.607 | 2 |
| NRPS | 0.975 | 0.000 | 0.967 | 0.224 | 0.496 | 5 |
| RIPP | 0.992 | 0.000 | 0.983 | 0.424 | 0.413 | **15** |
| ARYLPOLYENE | 1.000 | 0.000 | 1.000 | 0.600 | 1.000 | 1 |
| BETALACTONE | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1 |

The ORACLE column is the one that licenses reading any of the others: it pushes a held-out
real core through the COMPLETE arm path (record build, novelty gate, rates, confusion, lift),
where the ceiling column only calls antiSMASH. `novel=60/60` with every gate PASS.
**Paper:** the floor is 0/300 per class — an exact-binomial upper bound of ~0.0099, which is
also what sets δ (§3.1) — and it is a measured bound, never quoted as zero.

[CORRECTED TWICE] Earlier versions reported RIPP FPR 0.0033 (1/300) and before that 0.000 for
every class. The 1/300 came from negative controls drawn from the alphabetical head of the
genome index — 312 distinct genomes across all 1,500 controls, so the five per-class FPRs were
approximately one measurement reported five times.

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

### 1.5b Early stopping on Evo2: three mechanisms, none usable `[instrument]`
Investigated in response to "is there really no other way?". All three checked against the
local weights.

1. **`Generator.generate(stop_at_eos=True)` — CRASHES.** It is not merely dead code that
   prints without breaking; reaching that line threw `RuntimeError: CUDA error: an illegal
   memory access was encountered`. That is very likely why the wrapper one layer up
   hardcodes it to `False`.
2. **Batched per-row stopping — impossible in principle, on either substrate.** In any
   batched autoregressive loop every row advances together, so a row that terminates at
   token 500 still costs compute until its batch-mates finish. HuggingFace behaves the same
   way: it breaks when `unfinished_sequences.max() == 0`, i.e. when ALL rows are done. So
   GenomeOcean's "native stopping" is BATCH-level, not row-level, and describing it as
   per-row is wrong.
3. **Block-wise with cache carry-over — works, but is NOT equivalent.** `Generator.generate`
   documents passing and returning `inference_params_dict` for exactly this. It runs
   (after `model.eval()`, which `vortex_generate` does and a direct `Generator` does not),
   but **200 tokens generated in one call and in two 100-token blocks agree for only 119 of
   200 characters at the same seed**, diverging 19 tokens past the boundary. Both are valid
   samples, but the cached state is not restored exactly — the class of defect the prior
   codebase recorded as `seqlen_offset` corruption on resumed calls.

⇒ Post-hoc truncation remains the mechanism. It is exact, uniform across substrates, and
costs only compute. Adopting block-wise would trade a verified generation path for an
unverified one to buy a saving that is still hypothetical — whether a FINE-TUNED Evo2
terminates early enough to matter is a post-training measurement that has not been made.
**Revisit only if that measurement shows early termination**, and if adopted, apply it to
every arm and record the mode, since it changes outputs.

### 1.5c Evo2-1B degrades toward chance past ~10 kb `[instrument]` `[design]`
`evo2-1b`'s config `max_seqlen` is **8192**. A forward pass RUNS at 16k/32k/64k without
OOM — StripedHyena's convolutions have no positional limit and only its 4 attention blocks
of 25 do — so "it runs" was mistaken for "it works". Measured properly: mean NLL of the
last 1000 tokens of a prefix, on real cores ≥15.9 kb [M]:

| prefix length | NLL |
|---|---|
| 8,000 | 0.8090 |
| **8,192** (config limit) | **0.8049** |
| 9,000 | 0.8181 |
| 10,000 | 0.8505 |
| 12,000 | **1.0395** |
| 14,000 | **1.2093** |
| 15,900 | **1.2388** |

ln(4) = 1.386 is chance. **The model degrades progressively past ~10 kb and is near chance
by 14 kb.** `vortex` sizes its cache to prompt+tokens so long generation runs, but
`config.max_seqlen` is what flash attention uses — the library's own comment notes the two
diverging "leads to minor logit differences".
**Paper:** both the generation budget and the CORPUS BOUND must sit at or below the usable
context. A 16 kb bound trains on record tails the model reads at near-chance, and the long
classes carry the most of it — so the deficit would have read as a class effect on exactly
the axis the benchmark reports.

### 1.6 The three substrates terminate in three incompatible ways `[instrument]` `[design]`
Base models, no fine-tuning, 4,000 nt budget, 12 probes each [M]:

| substrate | terminator | auto-appended at encode | `hit_eos` | median length |
|---|---|---|---|---|
| Evo2-1B | byte 0 | **no** | **0/12** | 4,000 — runs to budget every time |
| GenomeOcean-4B | `[SEP]`=2 | yes | **12/12** | **553** (min 485, max 1,131) |
| **GenomeOcean-bgcFM** | `[SEP]`=2 | yes | **0/12** | **4,792** |

Three consequences, all load-bearing:

1. **Evo2 will never stop unless its terminator is written into the training text.** Its
   tokenizer does not add one, and `vortex.model.generation.generate` calls the inner
   generator with a hardcoded `stop_at_eos=False`, forwarding `**kwargs` after it — so
   `stop_at_eos=True` is a duplicate-keyword error, not an override. Termination must
   therefore be handled post hoc (generate to budget, truncate at the first terminator),
   which is the mechanism used uniformly across substrates.
2. **The published bgcFM checkpoint has LOST the termination behaviour its own base model
   has** — 0/12 against 12/12 for GenomeOcean-4B on identical prompts and settings.
   Fine-tuning on BGCs appears to have destroyed it. That is a result about prior art, and
   it means bgcFM cannot be compared to GO-4B on any length-sensitive axis without saying
   so.
3. **GenomeOcean stops at ~550 nt.** That is near real-core length for TERPENE (median
   1,367) but an order of magnitude short of BETALACTONE (9,017), so GO may terminate
   before it can physically emit a multi-gene cluster. Any multi-gene claim about GO has
   to be read against its own realised length distribution, not against the budget.

**⚠ `stop_at_eos` IS DEAD CODE IN VORTEX** [M]. `Generator.generate()` accepts it and
documents it as defaulting to True, but its entire implementation is a condition that
prints `"Stopping generation at EOS"` and does not break; the only `break` in the function
is in an unrelated `verbose` display block. Generation always runs to `num_tokens`. And the
condition inspects `generation[0]` — row 0 only — so had it been implemented, batched
generation would have cut the whole batch when the first sequence terminated. The wrapper
one layer up hardcodes it to False anyway. **Post-hoc truncation is therefore the only
correct termination mechanism for Evo2, not a workaround.** Anyone relying on this
parameter is generating to the full budget while believing otherwise.

Also measured: GenomeOcean's BPE ratio is **~4.8 nt/token**, not the 4.0 assumed — 1,000
tokens decoded to 4,792 nt. Budgets are specified in nucleotides and converted with the
measured ratio, or the two substrates get different amounts of sequence.

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

### 2.7 Equal records is not equal tokens `[design]` `[data]`
At 979 records per class the pooled training corpus is, by NUCLEOTIDE: TERPENE **9.9%**,
RIPP 12.9%, ARYLPOLYENE 18.5%, NRPS 24.8%, **BETALACTONE 33.9%** — a **3.4× imbalance** [M].
The loss is per token, so a record-balanced pooled arm's gradient is dominated by the long
classes. **An equal-n split does NOT give a balanced pooled model**, and an earlier version
of this project's spec asserted that it did.
**Paper:** the pooled arm must say which balance it used. `W1` (record-balanced) is the raw
mixture by token; `W1n` (nucleotide-balanced, per-class loss weights) is the balanced one.
The pair is what measures data mixture as a control method.

### 2.8 A fixed epoch count cannot tell you whether an arm converged `[design]`
First pass at 3 epochs, six arms, held-out loss: three still improving (ARYLPOLYENE,
BETALACTONE, RIPP), two already turned over (`W1` best at step 732 then rose;
`W2_RIPP` best at step 180 then rose), one flat. Generation loaded `final`, so the two that
had turned were evaluated at a checkpoint worse than their own best — an arbitrary handicap
on those two arms alone. Total improvements were small throughout, 0.005–0.022 nats.
**Paper:** report the stopping rule and the checkpoint used. "Trained for N epochs" is not
a description of a converged model, and a one-batch train loss is too noisy to see the
turn — the first pass logged exactly that and the overfitting was invisible.

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

### 4.0 A novelty gate can be structurally unable to fail `[instrument]` `[design]`
Containment defined as `max over training records t of |kmers(s) ∩ kmers(t)| / |kmers(s)|`
puts the WHOLE GENERATION in the denominator, so it decays as 1/length however much was
copied. Measured on the real TERPENE training split with the real instrument: **ten whole
verbatim training records concatenated to 15,992 nt score forward containment 0.198 →
PASS**, while antiSMASH calls them on-target with 15 core genes. At the 16 kb budget only
0.6–3.7% of training records per class are even long enough for a single record to reach
the FAIL threshold.
**Fix, empirically separated by three orders of magnitude:** add REVERSE containment,
`|kmers(s) ∩ kmers(t)| / |kmers(t)|`, which does not dilute with generation length, and
gate on the worse of the two. Same collage → reverse **1.000, FAIL**; held-out real core
plus filler → 0.000–0.006; random DNA → 0.000.
**Paper:** this is a general trap for anyone gating generative-genomics results on
containment, and it gets worse the better the model does, because a stronger arm produces
longer output.

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

## 4a. Uniformity findings — how a benchmark loses "every arm measured identically"

### 4a.1 Freezing the INTENT is not freezing the RUN `[design]`
A frozen-config module whose hash is `sha256(FROZEN_literal)` produces the SAME hash for
every run on a given code state, whatever was actually passed. Every mechanism built on it
is then inert, and each one LOOKS like enforcement:
* the run directory carried the constant, so all five per-class adapters resolved to ONE
  path and each truncated the last — and the survivor rendered the four destroyed rows as
  `{"n": 0, "detect_rate": null}`, **byte-identical to the NOT-APPLICABLE encoding**. Four
  deleted measurements would have published as four structural absences.
* the "refuses to compare arms with different configs" guard grouped on that constant, so
  it could not fire — and had no production caller.
* the "off-frozen runs are flagged in their report" field was a hardcoded `None`.
**The fix is to hash the REALISED values** — what n, budget, seed, temperature, adapter
sha, row class were actually used — and to put that in the directory name.
**Paper:** any provenance field that records intent rather than measurement is decoration.
The test for it is whether a deliberately drifted run produces a different fingerprint.

### 4a.2 Dropping empty generations biases toward the hypothesis `[design]` `[stats]`
A generation whose terminator lands at position 0 is a real outcome of a model that has
learned to stop. Removing it from the denominator before scoring inflates the rate — two
arms with identical biology, 70 on-target of 200 draws, report **0.35 vs 0.50** if one had
60 instant terminations — and `hit_eos_rate` computed over the survivors reads 0.0 against
a true 0.30, so **the drop erases its own evidence**.
⚠ The bias is DIRECTIONAL. Only a model that emits its terminator can lose a draw this
way, and the terminator is written into training text specifically so trained arms learn to
stop. Base models measure 0/12 hit_eos. So the deletion concentrates in exactly the
treatment arms and is absent from the control they are compared against.
**Paper:** empty generations are scored as non-detections, which is what they are.

### 4a.3 A declared parameter that nothing reads `[design]`
`rng_seed` was set from the frozen config, stamped into every report, and consumed by no
code on the generation path — generation was unseeded on both substrates while every
artifact asserted `rng_seed: 0`. Unlike the parameters above there was no truthful field
anywhere to contradict it.
**Paper:** a config value is only real if something consumes it. Grep for the consumer, not
the declaration.

### 4a.4 A manipulation check placed outside its own context measures nothing `[design]` `[instrument]`
The SPEC 6.4 two-sided check for W3 sat one indentation level **outside** the
`with iv.attached()` block that owns the hooks, while a comment on its first line asserted
"hooks still attached". Both of its measurements were therefore of the unintervened base
model, and it guarded the second one with `if not iv.is_attached()` — a condition that is
always true there. The published result was
`delta: 0.0, landed: false`, which is what it would have reported for **every possible
conditioner**, including a perfect one.

Measured after the fix, on the same `best.pt`: **with 0.90423, without 0.92148, delta
0.01725, landed true.** The arm had landed the whole time; W3's own training curve had
already moved 0.017 nats, which is the contradiction that gave it away. For scale, W1
(LoRA, 10.5 M trainable) reaches 0.88778 on the same held-out pool — so the 253 K-parameter
conditioner captures **about half** of LoRA's held-out gain at 1/41 the parameters.
**Paper:** a check whose two arms can be shown to be the same computation is not a weak
check, it is a decoration. Attachment is now established per measurement and *verified* —
`raise` on both branches — rather than narrated in a comment. It also loads the best
checkpoint first, so the check applies to the model generation will actually use.

### 4a.5 Keying a pooled arm on `classes[0]` invents strata the benchmark does not have `[design]` `[data]`
The trainer took each record's class as `classes[0]`, the first antiSMASH product in its
list. For a hybrid that is whichever product sorted first, not the class the record was
assigned to at split time. Measured on the pooled training set: **18 of 2624 records
(0.7%)** keyed to `NRPS` or `OTHER` — classes the four-class benchmark does not contain.

Three consequences, in increasing order of damage: the batch-mixing diagnostic counted six
classes; the nucleotide-balancing denominator became `sum/6` instead of `sum/4` (a uniform
scale, harmless inside a weighted mean); and those 18 records received per-record loss
weights of **22.9x (OTHER) and 56.3x (NRPS)** against 0.56–0.94 for everything else. With
`micro_batch=1` and `grad_accum=16`, one such record dominates the accumulated gradient of
the window it lands in. That is W1n's entire mechanism, contaminated.

Corrected weights, keyed on the assigned split: TERPENE **1.359**, RIPP **1.098**,
REDOX_COFACTOR **0.865**, ARYLPOLYENE **0.835** — monotone in the inverse of each class's
total nucleotides, which is what the arm was for. The val splits are clean (0 misfiled), so
held-out loss, early stopping and checkpoint selection were never affected.
**Paper:** the class a record was *assigned* and the class it *first maps to* are different
facts. Only the split directory knows the first one, so the loader has to carry it.

### 4a.6 A pipeline of arms is a pipeline of code versions `[design]` `[instrument]`
The overnight pass trained seven arms as one shell pipeline. Each arm is a separate
process, and a process imports whatever is on disk when it starts — so two commits that
landed mid-run split the pass into **three code versions**: five arms
(`W2_TERPENE`, `W2_RIPP`, `W2_ARYLPOLYENE`, `W2_REDOX_COFACTOR`, `W1`) on one, `W1n` and
`W3` on another. Nothing refused, nothing warned, and the seven reports looked uniform.

It surfaced by luck. One of those commits added two fields to `TrainConfig`, so the two
groups' recorded config dicts have **different key sets** — five carry 16 keys, two carry
18. A change that had not touched the dataclass would have left no trace anywhere.

**`train_config_hash` structurally cannot catch this.** It hashes the config, and the
config is precisely what is *supposed* to be identical between arms that differ only in
their data. The identity of the code that read that config is a separate fact and needs a
separate field. `provenance.code_version()` now records the commit and whether the tree was
dirty, in every training report. Dirty is recorded rather than forbidden — it is the
truthful answer during development, and its absence is what would let a run from
uncommitted code pass as a run from the commit it happens to sit on.

Magnitude, for calibration: retraining `W2_TERPENE` under the corrected code moved its best
held-out loss by **5e-5** (0.94038 → 0.94033), with the same epoch count and the same best
step. Small — but "small" is a measurement obtained by re-running, not a licence to publish
arms built by different code.
**Paper:** "every arm measured identically" is a claim about the code as much as the config,
and only one of the two was being recorded.

### 4a.7 Training is not reproducible under the seed its own report records `[instrument]` `[stats]`
`W2_ARYLPOLYENE` was retrained under a commit whose only differences on the LoRA path are
`_cls` substitutions, two unused `TrainConfig` fields and a provenance field — and its
record order is **exactly** invariant to that fix (verified element-wise, not as a set).
Same data, same order, same seed, same length bound, same rank. **Every one of its 15
checkpoints still moved**, by up to **2.4e-4**; the other two completed arms moved by up to
**4.5e-4** and **5.1e-4**.

Decomposed: **forward evaluation is exactly deterministic.** Five repeated `evaluate()`
calls on identical weights return one distinct value, spread **0.0**. So each recorded val
loss is an exact read — the divergence is in the **backward pass** (non-deterministic CUDA
kernels; Evo2 runs FlashAttention and TransformerEngine, neither of which offers a
deterministic backward here). Two runs of the same arm follow different trajectories.

**What this invalidates.** `min_delta = 1e-4` (§6.0a) is **below the noise floor** — early
stopping's "improved" test and best-checkpoint selection are partly selecting noise. And
overnight, `W1` and `W1n` differed by **7e-5** in best held-out loss, an order of magnitude
below the floor: as run, that contrast was unmeasurable, and reporting it either way would
have been reporting a coin flip.

**What survives comfortably.** W3's conditioner delta of **0.0173** and W1's **0.0337** are
34x and 66x the floor.
**Paper:** held-out loss needs the same treatment §8.3 already gives the endpoint — a
resolution measured on the instrument, with no difference read below it. A seed in a report
is a claim of reproducibility, and here it was not one. A replicate of `W2_ARYLPOLYENE`
under an identical commit is queued to put a number on the floor.

## 5. Standing limitations to disclose

### 5.1 Negative controls are GC-poorer than the cores `[limitation]`
Median GC **0.514–0.523** (controls) vs **0.563–0.668** (cores), lengths matched to within
~1%. Detection runs profile HMMs over translated ORFs rather than composition, so this is
not a shortcut antiSMASH can exploit, but the measured false-positive rate is plausibly a
slight **under**estimate.

[CORRECTED] An earlier version reported a gap of 0.475–0.481 vs 0.582–0.668 and asserted it
was "real biology rather than a sampling defect". **Both halves were wrong.** The controls
were drawn from the alphabetical head of the genome index, which is GC-poor fungi and
archaea; holding the genome constant, cores run only ~0.01–0.02 GC above non-core coding
DNA from the same organism. Spreading the sample by stable hash narrowed the gap to
0.05–0.15. Some residual gap is real; most of the original was a sampling artifact.

### 5.2 The corpus is antiSMASH-defined and the endpoint is antiSMASH `[limitation]`
This measures "can the model produce what the detector recognises", which is the intended
question, but it is circular in the strict sense. The MiBIG partition (24,694 of 340,044
in-class records, held out at build time and read once) is the partial external answer.

### 5.3 Hybrid records count for every class they map to `[limitation]`
Per-class hybrid fractions in the **built four-class splits**: TERPENE 1.0%, RIPP 1.5%
(2.2% multi-product), REDOX_COFACTOR 3.4%, **ARYLPOLYENE 10.1%**. ARYLPOLYENE's rate bounds
how exclusively its numbers can be read.

[CORRECTED] An earlier version of this entry quoted TERPENE 3.9%, RIPP 8.2%, BETALACTONE
13.2%, ARYLPOLYENE 18.5%, NRPS 30.1%. Those were corpus-wide fractions from the five-class
set that preceded the 8,192 nt reorientation, and two of the classes no longer exist. The
numbers above are read from the manifest of the splits actually trained on. The drop is
expected rather than surprising: cluster selection takes `common_n` clusters per class and
hybrids are over-represented among the long records the 8,192 nt bound excludes.

---

## 6. Stage 1 results — FROZEN 2026-09-08, before any comparison to prior work

**SPEC 10.1 step 2.** Written, hashed and committed *before* unblinding, because a blind rebuild
cannot otherwise distinguish "the new code found a real correction" from "the new code has a bug".

Bundle: `/data2/ds85/bgcbench/runs/STAGE1_FROZEN_f50c43b09042609c.json`
sha256[:16] = **f50c43b09042609c** — 8 run directories.

Regime: **de novo**, `evo2-1b`, n=200/row, budget 8,192 nt, `batch_size` 100, antiSMASH 8.0.4
(`ee8c025c1593`), corpus `c74154974aff`. Every arm `off_frozen: {}`.

| arm | trained on | core median | detected | on-target | rate | p vs floor |
|---|---|---|---|---|---|---|
| `W0` base | — | — | 0 | 0 | 0.000 | — |
| `W1` pooled LoRA | all 4 | — | 0 | 0 | 0.000 | 1.000 |
| `W1n` nt-balanced | all 4 | — | 0 | 0 | 0.000 | 1.000 |
| `W3` conditioner | all 4 | — | 0 | 0 | 0.000 | 1.000 |
| `W2_TERPENE` | TERPENE | 1,182 | 0 | 0 | 0.000 | 1.000 |
| `W2_RIPP` | RIPP | 1,931 | 0 | 0 | 0.000 | 1.000 |
| `W2_REDOX_COFACTOR` | REDOX_COFACTOR | 3,031 | 1 | 0 | 0.005 | 0.500 |
| `W2_ARYLPOLYENE` | ARYLPOLYENE | 3,579 | 3 | 3 | 0.015 | 0.124 |

Instrument, same run: oracle (real held-out cores) **1.000** detect / 1.000 on-target, n=60/class;
negative controls (real non-BGC DNA) **0.000**, n=300/class. Full dynamic range at both ends.

**Four detections in 1,600 generations, three on-target, none significant at n=200.**

Standing observations, recorded before unblinding:
* `hit_eos = 0.0` on **all eight arms** — not one generation ever emitted a terminator, and every
  arm ran the full 8,192 nt against class cores of median 1.2–3.6 kb.
* Detection rank-orders exactly with class core length (0, 0, 1, 3).
* The only arm producing on-target output was the **class-exclusive** one; the pooled arm saw the
  same records plus three other classes and produced nothing.
* `W3`'s manipulation check PASSES (delta 0.0188, landed true), so its zero is a **negative**;
  `W1` vs `W1n` is **uninformative** — both at the floor, and 1.5e-4 apart on held-out loss, which
  is below the measured noise floor (4a.7).
* No generation on any arm matched a known BGC.
