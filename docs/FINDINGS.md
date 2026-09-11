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
* `hit_eos = 0.0` on **all eight arms**, and every arm ran the full 8,192 nt against class
  cores of median 1.2–3.6 kb.
  **[INCORRECT — see 4a.8]** This was read as "not one generation ever emitted a terminator".
  It is a structural zero of the METRIC: the terminator could not survive decoding, so the
  detector could never fire. The models were emitting terminators the whole time — after
  **2 nt**.
* Detection rank-orders exactly with class core length (0, 0, 1, 3).
* The only arm producing on-target output was the **class-exclusive** one; the pooled arm saw the
  same records plus three other classes and produced nothing.
* `W3`'s manipulation check PASSES (delta 0.0188, landed true), so its zero is a **negative**;
  `W1` vs `W1n` is **uninformative** — both at the floor, and 1.5e-4 apart on held-out loss, which
  is below the measured noise floor (4a.7).
* No generation on any arm matched a known BGC.

---

## 7. The termination defect — found at §10 reconciliation, 2026-09-08

### 4a.8 A terminator that cannot survive decoding makes every generation post-mortem `[instrument]` `[design]`
vortex's `CharLevelTokenizer` decodes with `chr(max(32, min(id, vocab)))`, so ids **0 (EOS),
1 (PAD) and 32 (space) all render as a space** and are indistinguishable. `Substrate` searched
the DECODED string for `chr(0)`, which can never appear. Consequences, in order of damage:

1. **`hit_eos` was a structural zero** in all 13 Stage 1 run reports — a property of the
   metric, not of the model.
2. **`clean()` then DELETED that space**, shifting the reading frame by one base and
   destroying every ORF downstream of it. Measured on the frozen bundle: **1,422 of 1,600
   generations (88.9%)** lost at least one character, mean 3.13, up to 13. Recovered by
   patching the decoder and regenerating: **231 of 231 dropped characters were id 0** — the
   model's own stop token, nothing else.
3. **Generation ran 8,192 nt past the stop.** With the terminator made visible, Evo2 emits it
   at index **2** — in 56% of base generations and **91%** of fine-tuned ones. Every sequence
   in `STAGE1_FROZEN_f50c43b09042609c` is ~8,190 nt of **post-termination sampling**.

**Gate G10 passed while the property it gates was false.** T1 tested only the ENCODE
direction; T3 tested truncation on a probe string built in Python, never on model output.
Both now test the direction that failed, and T1 additionally requires the decoded terminator
to be distinguishable from PAD and space.

⚠ **Two earlier findings were measured on post-termination output and are not safe to read
as written:** 4a.7 (training non-determinism moving the endpoint by ±1/200) and the §12.A3
data-volume null. Both need re-measurement before they can be cited.

**Paper:** a round-trip test must test the round trip. Encoding a terminator proves nothing
about whether you can ever detect one.

### 4a.9 Unconditioned generation collapses immediately on this substrate `[data]` `[design]`
With termination fixed, the honest de novo measurement is that the model stops at **2 nt**.
An arm that terminates immediately produces nothing to score, so a `min_new_tokens` floor is
required for the regime to yield a measurement at all. It is a **decoding policy, not a
conditioning channel** — one number, identical for every arm and class, carrying no class
information, so SPEC 4.3 is not in tension. The prior project applied the same fix to
GenomeOcean, where EOS firing straight after the seed gave 61/200 empty generations.

At a 1,000 nt floor, `W2_RIPP` terminates at a median of **3,402 nt** against a training
median of 2,154 — the right scale for the first time. Detection was still **0/40**, so the
termination defect was real and is **not** what separates this rebuild from the prior work.


---

## 8. Stage 1 with the phylogeny prefix — FROZEN 2026-09-09

Bundle: `/data2/ds85/bgcbench/runs/STAGE1_TAX_FROZEN_4f754b369b13eaf3.json`
sha256[:16] = **4f754b369b13eaf3** — 8 runs, 26 hits with full organism provenance.

Regime: **de novo with Evo2's native GTDB lineage prefix**, `evo2-1b`, n=200/row, 8,192 nt,
`batch_size` 100, `min_new_tokens` 1000, antiSMASH 8.0.4. Every arm `off_frozen: {}`.
The prefix is loss-masked in training, space-padded to a uniform 186 characters, and drawn
from held-out records. **No class token anywhere.**

| arm | rows | n | det | on-target | rate | p | hit_eos | med len |
|---|---|---|---|---|---|---|---|---|
| `W0_tax` floor | 4 | 200 | 0 | 0 | 0.000 | — | 0.00 | 8192 |
| `W2_TERPENE_tax` | 1 | 200 | 3 | **3** | 0.015 | 0.124 | 0.83 | 3271 |
| `W2_RIPP_tax` | 1 | 200 | 0 | 0 | 0.000 | 1.000 | 0.83 | 2593 |
| **`W2_ARYLPOLYENE_tax`** | 1 | 200 | 7 | **7** | **0.035** | **0.007** | 0.29 | 8192 |
| `W2_REDOX_COFACTOR_tax` | 1 | 200 | 4 | 1 | 0.020 | 0.062 | 0.45 | 8192 |
| `W1_tax` pooled | 4 | 200 | 1 | **0** | 0.005 | 0.500 | 0.84 | 2855 |
| `W1n_tax` pooled | 4 | 200 | 1 | 1 | 0.005 | 0.500 | 0.83 | 2908 |
| `W3_tax` conditioner | 4 | 200 | 1 | 1 | 0.005 | 0.500 | 0.39 | 8192 |

### 8.1 The floor stayed at zero `[result]`
`W0_tax` is the untrained model given the SAME lineage prompts: **0/200**. The prefix alone
produces nothing. The adapter does the work; the prefix is what lets it express itself. That
control is what makes the rest of the table readable.

### 8.2 Class-exclusive training produces specificity; pooled training does not `[result]`
Every per-class arm that detected anything was **perfectly on-target** — TERPENE 3/3,
ARYLPOLYENE 7/7. The three pooled arms produced **three detections and two on-target**
between them, and their confusion rows are ~all zero.

[CORRECTED] An earlier version of this table read 4 detections in n=800 for each pooled arm
at p=0.062. A non-class-bearing arm generates **200 sequences ONCE** and scores them against
all four targets (SPEC 6.0 degenerate collapse), so summing the per-row cells counted the
same detection four times. Each pooled arm has **1 detection in 200**, p=0.500. The rates
were unaffected; the counts, the denominators and the p-values were not. Pooled training yields BGC-like sequence with
no class control. This is the contrast the benchmark exists to measure, and it is the first
time it has had data rather than a floor-to-floor null.

### 8.3 Termination returned `[result]`
`hit_eos` 0.83 on TERPENE and RIPP at median 3,271 and 2,593 nt, against training medians of
1,154 and 2,154. Those arms produce cluster-shaped objects rather than 8 kb continuations —
the first time any arm in this project has.

### 8.4 What is NOT established
Only ARYLPOLYENE clears p<0.05, and it is the class with the longest real cores (3,593 nt
median) — the same class that led the unprefixed run. TERPENE at 3/200 and REDOX at 1/200
on-target are suggestive, not established. RIPP remains at zero in both regimes.

⚠ These numbers are **not comparable to the unprefixed bundle `f50c43b09042609c`**: that one
was measured before the terminator fix, on 656-record splits, with no prefix. Four things
differ at once. The floor (0/800 here, 0/200 there) is the only directly comparable cell.


---

## 9. G3 — the seed-length sweep, and the control that reframes it

Bundle: `/data2/ds85/bgcbench/runs/G3_FROZEN_816d66441f8624ca.json`
Arm: `W1n_tax` (pooled, nucleotide-balanced, lineage-prefixed). 800 generations per point
(4 rows x 200). `batch_size` 80 against the frozen 100, declared `--off-frozen` and recorded:
the sweep must be internally uniform, and batch size does not affect the sampling
distribution.

| L | det | on-target | on-target rate | p | per-class on-target (of 200) |
|---|---|---|---|---|---|
| 8 | 5 | 1 | 0.001 | 3.1e-02 | TERP 1 |
| 16 | 5 | 1 | 0.001 | 3.1e-02 | TERP 1 |
| 32 | 43 | 42 | 0.052 | 6.4e-14 | TERP 10, RIPP 2, ARYL 30 |
| 64 | 65 | 62 | 0.077 | 7.0e-21 | TERP 17, RIPP 4, ARYL 41 |
| 128 | 164 | 147 | **0.184** | 3.8e-54 | TERP 75, RIPP 17, ARYL 53, REDOX 2 |

### 9.1 A threshold between 16 and 32 nt `[result]`
L=8 and L=16 are identical nulls — 1 on-target in 800, indistinguishable from the same arm's
de novo rate. At 32 nt it rises 42x and climbs monotonically to 0.184. That matches the
entropy measurement behind 5.3: below ~20 nt a core's 5' end is a start codon plus noise.

**The class is coming from the SEED, not the weights.** This is the POOLED arm — no class in
its weights, no class in its lineage prefix. De novo it managed 1 detection with no class
control; seeded at 128 nt it produces 147 on-target across all four classes, including RIPP,
which is at zero in every other configuration run so far.

### 9.2 Novelty is clean at every length `[instrument]`
A long seed hands the model real BGC sequence, so this is where copying would appear.
Worst-case containment rises with L exactly as it must — and stays far below the gate:

| L | median worst | max worst | gate | matched known BGC |
|---|---|---|---|---|
| 8 | 0.0000 | 0.0161 | 800 PASS | 0 |
| 32 | 0.0000 | 0.0324 | 800 PASS | 0 |
| 64 | 0.0000 | 0.0426 | 800 PASS | 0 |
| 128 | 0.0001 | 0.0634 | 800 PASS | 0 |

Max 0.063 against a fail threshold of **0.95**, and zero known-BGC matches at any length.

### 9.3 ⚠ THE SEEDS THEMSELVES BECOME DETECTABLE ABOVE 64 nt `[limitation]`
antiSMASH run on the seeds alone, n=100 per class, on-target counts. The seed is excluded
from scored text — Evo2 returns only the continuation, verified empirically — so a marker in
the seed cannot be counted as a generation. But it changes what a hit MEANS: completing a
fragment the detector already recognises is a weaker claim than building the marker.

| L | TERPENE | RIPP | ARYLPOLYENE | REDOX_COFACTOR |
|---|---|---|---|---|
| 8 / 16 / 32 | *below antiSMASH's length floor — "all records skipped"* | | | |
| 64 | **0** | 1 | **0** | 0 |
| 128 | 3 | 5 | 12 | 0 |
| 256 | 13 | 23 | 30 | 0 |
| 512 | **82** | 35 | 48 | 0 |
| full core | 100 | 99 | 100 | 99 |

**L=64 is the clean point.** Seeds are invisible to the instrument there — 1 detection in 400
across all classes — while generations reach 0.077 on-target. TERPENE and ARYLPOLYENE go from
a 0/100 seed baseline to 0.085 and 0.205: the model is building the marker, not extending one.

At L=128 the generation rate still exceeds the seed baseline substantially — TERPENE 12.5x
(0.030 -> 0.375), ARYLPOLYENE 2.2x, RIPP 1.7x — but the baseline is no longer zero and must be
quoted with the rate. **At L=512, 82% of TERPENE seeds are already on-target**, so points at
and above 256 nt are confounded by construction and cannot support an unqualified claim.

### 9.7 A generation is NOT more like its own seed's source `[instrument]` `[result]`
Frozen: `/data2/ds85/bgcbench/runs/SEED_SOURCE_SIMILARITY_be9ddeb59893a3bc.json`

The novelty gate takes a MAX over a whole reference set. That catches wholesale copying but
would bury the failure mode seeding specifically enables: hand the model the first L nt of
held-out record X and it reconstructs the rest of X. That is a PER-RECORD question, so it
needs a paired test.

For every seeded generation: containment against **the record its seed came from** (matched)
versus **20 random other held-out records of the same class** (background). Same k=21
canonical-k-mer containment the gate uses. **The seed region is excluded from the source**,
or the prompt would appear in both sides and inflate the matched arm for free.

| L | matched > 0 | background > 0 | wins / losses | sign p | matched max | bg max |
|---|---|---|---|---|---|---|
| 8 | 0.0% | 0.0% | 0 / 2 | 0.50 | 0.0000 | 0.0014 |
| 32 | 0.4% | 0.1% | 2 / 18 | 4.0e-04 | 0.0018 | 0.0049 |
| 64 | 0.5% | 0.2% | 4 / 21 | 9.1e-04 | 0.0067 | 0.0064 |
| 128 | 2.6% | 0.5% | 21 / 36 | 0.063 | 0.0120 | 0.0164 |
| 256 | 2.6% | 0.5% | 21 / 47 | 0.0022 | 0.0076 | 0.0080 |
| 512 | 2.2% | 0.5% | 18 / 44 | 0.0013 | 0.0098 | 0.0116 |

**Two things are true at once and they pull opposite ways.** Generations are ~5x more likely
to share ANY k-mer with their own source than with a random record of the same class (2.6% vs
0.5% at L>=128), and that gap grows with seed length — the seed leaves a local trace. But the
PAIRED comparison runs the other way: against its own background, the source LOSES, 21 wins
against 36–47 losses, sign p down to 0.0013. And the matched maximum never exceeds the
background maximum past L=64.

So the trace is a handful of shared k-mers in a small subset of generations, not
reconstruction. Every value here is **79x below the 0.95 gate threshold** — the largest
matched containment across 4,800 generations is **0.012**.

⇒ **The L=512 inversion (9.6) is not memorisation.** That was the live worry: if the seed-only
baseline outscoring the generation rate were caused by regurgitation, every seeded result
would be worthless. It is not. And this is a sharper claim than the gate's — "nothing copies
the one thing it had the best opportunity to copy" — which the gate's max-over-references
could not have made.

⚠ A first cut of this used "matched exceeds the background 99th percentile". The background
p99 is 0.00000, so that statistic collapsed to "is nonzero" and was replaced by the paired
sign test above.

### 9.5 The extension, and where the curve stops meaning anything `[result]` `[limitation]`
Full ladder frozen as `/data2/ds85/bgcbench/runs/G3_FULL_FROZEN_00820b3404a2acc4.json`, 8
points. L=256 and L=512 run at `batch_size` 48 (probed: 256 fits at 64, 512 at 48; an OOM
poisons the CUDA context, so each candidate was probed in a FRESH process — sequential
probing turns the next honest OOM into a spurious `CUFFT_INVALID_SIZE`).

| L | batch | on-target / 800 | rate | seed-only baseline | reading |
|---|---|---|---|---|---|
| 8 | 80 | 1 | 0.001 | *too short to score* | clean |
| 16 | 80 | 1 | 0.001 | *too short* | clean |
| 32 | 80 | 42 | 0.052 | *too short* | clean |
| **64** | 80 | **62** | **0.077** | **0.003** | **clean — the defensible point** |
| 128 | 80 | 147 | 0.184 | 0.050 | 3.7x baseline |
| 128 | 48 | 134 | 0.168 | 0.050 | 3.4x baseline |
| 256 | 48 | 192 | 0.240 | 0.165 | 1.5x baseline |
| 512 | 48 | 253 | 0.316 | **0.412** | **BELOW baseline** |

**BATCH SIZE DOES NOT MOVE THE ENDPOINT.** L=128 was run at both 80 and 48: 0.184 vs 0.168.
That overlap point is what licenses reading the two batch regimes as one curve, and it is
why the repeat was worth its hour.

### 9.6 ⚠ AT L=512 THE SEEDS ALONE OUTSCORE THE GENERATIONS `[limitation]`
The seed-only baseline reaches **0.412** while the generations reach **0.316**. The fragments
handed to the model are MORE on-target than what the model produces from them.

They are not strictly comparable objects — a 512 nt slice of a real core against up to 8 kb
of model output, and antiSMASH's rules are arity- and length-sensitive — so this is not a
clean "the model makes it worse" claim. But the direction is unambiguous and it settles how
to read the ladder: **past ~64 nt the rise is the seed, not the method.** A curve that keeps
climbing while its own baseline climbs faster is not evidence of capability.

⇒ **The reportable seeded result is L=64: 0.077 on-target, p = 7.0e-21, against a seed-only
baseline of 0.003.** L=128 survives as a supported secondary point at 3.4-3.7x its baseline.
L=256 and L=512 are recorded and not claimed.

### 9.4 REDOX_COFACTOR's marker is not at the 5' end `[data]`
Its seeds are on-target **0/100 at every length up to 512**, then 99/100 at the full core —
while its detected-but-off-target rate sits at a flat 6/100 (`RRE-containing`). No other class
behaves this way. Its generation on-target rate is correspondingly flat at ~0 across the whole
sweep. So the seed only helps when the class's marker lies near the 5' end of the core, and
for REDOX it does not. That is a property of the class, not of the method, and it predicts
which classes seeding can be expected to help before spending a run.


---

## 10. The phylogeny prefix is causal — the control

Frozen: `/data2/ds85/bgcbench/runs/PREFIX_CONTROL_FROZEN_98edca7b6319ae01.json`

Between the unprefixed Stage 1 table (§6) and the prefixed one (§8), **four things changed at
once**: the terminator fix, 12x more training data, the frame-preserving cleaner, and the
prefix. So §8 could only say "this configuration clears the floor", never "the prefix caused
it". This closes that.

**Design.** Unprefixed arms trained on the SAME 8,044 records, at the SAME commit, generated
through the SAME fixed pipeline. Only `--prefix` differs. An unprefixed adapter already
existed but was trained at `c82878d` — before the terminator fix, by code whose `TrainConfig`
had no `prefix` field — so reusing it would have reintroduced the mixed-code-version confound
of 4a.6, which is the thing this control exists to remove. Both arms were retrained.

| class | arm | detected | on-target | rate | p | hit_eos | median GC |
|---|---|---|---|---|---|---|---|
| ARYLPOLYENE | no prefix | 0 | 0 | 0.000 | — | 0.025 | 0.483 |
| ARYLPOLYENE | **+ phylogeny** | 7 | **7** | **0.035** | **0.007** | 0.290 | **0.617** |
| TERPENE | no prefix | 1 | 1 | 0.005 | — | 0.365 | 0.467 |
| TERPENE | **+ phylogeny** | 3 | **3** | 0.015 | 0.312 | 0.830 | **0.586** |

**Pooled: 10/400 on-target with the prefix against 1/400 without — a 10x difference,
Fisher one-sided p = 5.6e-03.**

### 10.1 What it establishes `[result]`
The phylogeny prefix **causes** the effect. Everything §8 reports survives, and the claim
strengthens from "this configuration clears the floor" to "the prefix is what lifts it". The
pipeline fixes were necessary — nothing worked before them — but they are not sufficient: an
arm with every fix and no prefix still sits at the floor.

Two secondary readouts move the same way and were not the endpoint, so they are independent
corroboration rather than the same measurement twice:
* **Composition.** Median GC 0.483 -> 0.617 and 0.467 -> 0.586, onto the real-core value of
  ~0.64. The prefix moves the output distribution toward real biology.
* **Termination.** `hit_eos` 0.025 -> 0.290 and 0.365 -> 0.830. Unprefixed arms run to the
  full budget; prefixed ones stop.

### 10.2 What it does NOT establish `[limitation]`
Only ARYLPOLYENE clears significance on its own (p = 0.007); TERPENE at 3 vs 1 is p = 0.31 and
carries the pooled result rather than standing alone. Two classes is a thin base for a general
claim, and RIPP was deliberately excluded because it is 0/200 prefixed — so the control speaks
for the classes where the prefix already appeared to work, not for the method everywhere.


## 11. Seeding × per-class weights — the diagonal, FROZEN 2026-09-10

`SEEDED_DIAGONAL_FROZEN_f1a5fa95dbdc07a1`. Four arms, `W2_<CLASS>_tax_S1`, 200 generations each,
64-nt real seeds drawn from the class test split, taxonomy prefix, scoring `ee8c025c1593`. Every
arm ran `rc=0`; `batch_size` 80 against a frozen 100, recorded in each report's `off_frozen`.

**The 2×2 is now complete.** Each cell is on-target rate, pooled over the classes it covers:

| weights | de novo | seeded L=64 |
|---|---|---|
| pooled (`W1n_tax`) | 0.005 | 0.077 |
| per-class (`W2_*_tax`) | **0.014** (11/800) | **0.130** (104/800) |

Seeding and per-class weights **compose**: 9.5× over the same adapters de novo, p = 1.1e-21. The
two interventions were measured separately and neither predicted the other; the diagonal is where
the benchmark first produces a rate worth reporting rather than a floor.

### 11.1 Per class — and where the result actually stands up

| class | seeded | rate | detected | precision | de novo | p vs de novo | seed-only | p vs seed-only |
|---|---|---|---|---|---|---|---|---|
| ARYLPOLYENE | 63/200 | 0.315 | 64 | 0.984 | 7/200 | 1.0e-14 | 0/100 | 1.5e-13 |
| TERPENE | 30/200 | 0.150 | 30 | 1.000 | 3/200 | 2.7e-07 | 0/100 | 2.4e-06 |
| REDOX_COFACTOR | 7/200 | 0.035 | 20 | 0.350 | 1/200 | 3.4e-02 | 0/100 | 5.7e-02 |
| RIPP | 4/200 | 0.020 | 4 | 1.000 | 0/200 | 6.2e-02 | 1/100 | 4.6e-01 |

⚠ **Two of the four classes carry this.** ARYLPOLYENE and TERPENE are unambiguous. REDOX_COFACTOR
clears the de novo comparison at p = 0.034 but **fails against its own seed-only baseline**
(p = 0.057) — it must be reported as suggestive, not positive. RIPP clears neither. The pooled
9.5× is real but it is not four independent replications, and writing it up as though every class
moved would misrepresent it.

### 11.2 The seed is not doing the work

At L = 64 the seeds are, by themselves, invisible to the instrument: 0/100 TERPENE, 0/100
ARYLPOLYENE, 0/100 REDOX_COFACTOR, 1/100 RIPP (§9). This is the whole reason L = 64 is the
frozen choice — at L ≥ 128 the seed is independently detectable (12/100 ARYLPOLYENE at 128,
48/100 at 512) and the arm stops measuring generation. So 63/200 against a 0/100 seed floor is
attributable to what the model wrote, not to what it was handed. Combined with §9.7 — generations
are not more similar to their own seed's source record than to any other — the seeded rate is not
retrieval.

### 11.3 REDOX_COFACTOR's precision collapse is a finding, not noise

REDOX detects at 0.100 but is on-target at 0.035: **precision 0.35**, against 0.98–1.00 for every
other class. The 13 off-target detections are not scattered — 11 are `RRE-containing` and 2
`ranthipeptide`, both RiPP-category products, so the arm scores as RIPP at 0.065.

⚠ **CORRECTED 2026-09-10.** This paragraph previously offered, as corroboration, "REDOX seeds
detect 6/100 and 0 are on-target". That figure is from the **L = 128-512 rungs**, not from the
frozen L = 64 arm this section reports — at L = 64 REDOX seeds detect **0/100** (§9, §11.2). The
sentence therefore cited a confounded rung as evidence for an unconfounded one and contradicted
§11.2 two paragraphs above it. The 6/100 RRE-only signature at longer seeds is real and is
consistent with the account below, but it is a *separate* observation and cannot corroborate the
L = 64 result. **The precision collapse at L = 64 stands on the arm's own 20 detections.**

⇒ The confusion is in the **class definition**, not in the model: RRE domains are shared between
redox-cofactor clusters and RiPP machinery, and the frozen rule set assigns them to RIPP. An
arm cannot be on-target for a class whose members the instrument reads as another class. This
belongs in the paper as a property of antiSMASH's category boundaries; it also means REDOX's
on-target rate understates what the adapter learned, and its detect rate overstates it.

**⚠ This was PRE-REGISTERED, which is what makes it evidence rather than a rescue.** The class
map wrote it as the standing justification for promoting `redox-cofactor` out of its antiSMASH
category, months before this arm ran (`bgcbench/data/classmap.py`, `PROMOTIONS`):

> ⚠ IT IS A RiPP SUBTYPE, so REDOX_COFACTOR and RIPP are biologically nested even though
> promotion makes them disjoint at class level (measured: 0 records carry both once promoted).
> **Off-diagonal mass between these two rows is expected and must be read as relatedness, not
> as a specificity failure.**

⚠ **One clause of that quotation is FALSE and is corrected at source.** "measured: 0 records
carry both once promoted" does not hold: **220 corpus records carry both `REDOX_COFACTOR` and
`RIPP`** (e.g. `GCF_000012325.1.NC_003910.region2`, products `['RiPP-like', 'redox-cofactor']`).
Promotion makes the two classes disjoint as *labels*, not as *records*. This does not weaken the
pre-registered prediction — it strengthens it, since the classes demonstrably overlap in the
corpus — but the parenthetical was wrong and §11.3 quoted it without checking.

`redox-cofactor`'s antiSMASH category *is* RiPP; it was promoted out because the category could
not express it (42.0% multi-gene against redox-cofactor's 100%). The biology agrees: PQQ and
mycofactocin are ribosomally synthesised precursor peptides and the RRE is genuine shared
recognition machinery. So §11.3 is a prediction the class map made in advance and the endpoint
confirmed — the paper should present it in that order.

**A mechanism was proposed and REJECTED.** The obvious story is truncation: the RRE fires early
in a cluster while the `redox-cofactor` rule needs more of it, so off-target hits should be the
shorter generations. They are not. On-target median 4,046 nt vs off-target 3,454 nt,
Mann-Whitney p = 0.48. The RRE/redox split is not a completeness effect and no length-based
account of it survives. (Incidentally: detections of either kind run ~3,400-4,000 nt while
non-detections sit at the 8,192 budget median — a detected cluster is one that terminated.)

### 11.4 Novelty

All 800 generations `PASS` (k=21 containment, fails at 0.95). One REDOX generation matched a
known BGC below threshold. No arm approached the gate.

### 11.5 Seeding alone does nothing — the base-model control

`stage1_evo2-1b_W0_tax_S1_ALLROWS_f7c72ce0ff0d`, run 2026-09-10. The BASE model, real 64-nt
held-out seeds, taxonomy prefix, otherwise the frozen config. **0 detections in 800.**

⚠ An earlier `W0_S1` run exists and is NOT this control: it used `seed_len 8` and no prefix,
from before L = 64 was frozen. L = 8 is below the threshold G3 located (between 16 and 32 nt),
so it could not have detected anything and says nothing about the frozen configuration.

This is the row §11 was missing. Without it, "seeding and weights compose" could not be told
apart from "seeding works and the weights are incidental":

| weights | de novo | seeded L=64 |
|---|---|---|
| `W0` base | 0.000 (0/800) | **0.000 (0/800)** |
| `W1n` pooled | 0.005 | 0.077 |
| `W2` per-class | 0.014 | 0.130 |

⇒ **A seed is not an intervention.** It multiplies what the weights already provide, and
multiplying zero gives zero. The 0.130 in the corner requires both.

**This null is interpretable under §6.4, and here is its warrant.**
* **Power.** 0/800 gives a 95% upper bound of 3/800 = **0.375%** (rule of three). That excludes
  the seeded per-class rate (0.130, **35×** the bound), the seeded pooled rate (0.077, 21×) and
  the de novo per-class rate (0.014, 4×). The null is a bound, not an absence of evidence.
* **Manipulation check.** Both channels verifiably landed: 800/800 generations carry a real
  `seed_accession` from a held-out record and 800/800 carry a `prefix_tag`, and
  `test_seed_excluded_from_scored_span` pins that the seed never enters the scored text. So
  "nothing was detected" cannot be explained by "nothing was applied".

Consistent with G10: the base model does not stop — `hit_eos` 0.0013, median length at the full
8,192 budget — against 0.290–0.830 for the trained arms (§10).

## 12. Why the classes differ — heterogeneity, held-out loss, and what does NOT follow

§11 leaves four classes spread over a 16× range of on-target rate. This section asks what
separates them. It is **exploratory**, it is powered at n = 4 classes, and §12.4 states plainly
why no correlation here can reach significance. It is written up because it generates a sharp,
pre-specifiable prediction (§12.5), not because it establishes a cause.

### 12.1 The measurements

Corpus figures from `/data2/ds85/bgcbench/corpus/core_records.jsonl`; loss from each arm's
`train_report.json`; detection from `SEEDED_DIAGONAL_FROZEN_f1a5fa95dbdc07a1`.

| class | corpus records | products | H(prod) | 2^H | modal share | length IQR (nt) | val loss | perplexity | detect |
|---|---|---|---|---|---|---|---|---|---|
| ARYLPOLYENE | 15,914 | 71 | 1.75 | 3.4 | 0.763 | 2,485–7,723 | **0.689** | 1.99 | **0.320** |
| TERPENE | 136,184 | 89 | 2.18 | 4.5 | 0.437 | 999–3,844 | 0.944 | 2.57 | 0.150 |
| REDOX_COFACTOR | 11,206 | 50 | **0.90** | **1.9** | **0.897** | 2,269–4,443 | 0.739 | 2.09 | 0.100 |
| RIPP | 122,270 | 93 | **3.91** | **15.0** | **0.354** | 1,449–6,229 | **1.037** | 2.82 | **0.020** |

⚠ **Corpus records are NOT training volume.** Training is equal-n by §4.4.3 — every arm above
saw **8,044 records**. RIPP's 122,270 and REDOX's 11,206 describe the class in nature, not what
the adapter was given. Data volume is therefore controlled and cannot explain any row.

**What the shorthands are.**
* **products** — antiSMASH's ~103 fine-grained labels (`lassopeptide`, `terpene-precursor`,
  `arylpolyene`), one level below the benchmark's four classes. The column counts distinct
  labels appearing anywhere in that class's records.
* **H(prod)** — Shannon entropy of the product distribution in bits, `-Σ pᵢ log₂ pᵢ`.
* **2^H** — the readable form: the **effective number of equally-common products**. REDOX behaves
  like ~2 things, RIPP like ~15. The long tails are comparable (50–93 labels); what differs is
  how the mass spreads. 90% of the mass takes 2 products for REDOX and **19** for RIPP.
* **val loss** — held-out cross-entropy in nats/nt on validation records the adapter never
  trained on, taxonomy-prefix positions masked (§8 defect, since fixed).
* **perplexity** — `e^loss`: how many bases the model is effectively still choosing between.
  Uniform over A/C/G/T is 4.00. ARYLPOLYENE has narrowed to 1.99, RIPP only to 2.82.
* **detect** — any antiSMASH call, on- or off-target; distinct from on-target rate.

### 12.2 RIPP's heterogeneity is real, not co-occurrence

A RIPP region can carry an unrelated neighbour (`NRPS`), which would inflate its diversity
without the class being internally varied. Restricting to each class's **own** antiSMASH
category kills that confound:

| class | own-category share of product mass | effective subtypes within own category |
|---|---|---|
| REDOX_COFACTOR | 0.916 | 1.1 |
| ARYLPOLYENE | 0.791 | 1.2 |
| TERPENE | 0.843 | 2.0 |
| RIPP | 0.851 | **9.0** |

Co-occurrence is comparable across all four (0.79–0.92). RIPP still carries 9 effective
subtypes against 1.1–2.0. ⇒ "RIPP" names a **union**, and the adapter is asked to learn it as
one target.

⚠ **CORRECTED: it is THREE measurements, not four.** An earlier version of this paragraph
claimed subtype diversity, length spread, held-out loss and detection all rank RIPP last.
**Length spread does not** — on the interquartile width the table above actually prints,
ARYLPOLYENE is the widest (5,238 nt) and RIPP is second (4,780 nt). The three that do rank RIPP
last are **subtype diversity, held-out loss and detection**, and the first two never touch
antiSMASH's endpoint. §12.1's length column is still worth reporting; it just is not evidence
for this claim.

### 12.3 RIPP and REDOX are low for OPPOSITE reasons

This is the statement that survives, and it comes from the precision column, not a correlation:

| | RIPP | REDOX_COFACTOR |
|---|---|---|
| val loss | **1.037** — worst | 0.739 — 2nd best |
| detect | 0.020 | 0.100 |
| precision | **1.000** | **0.350** |
| reading | a genuine **generation** failure: rarely produces anything, but right when it does | **not a learning failure at all**: a class-boundary failure at scoring (§11.3) |

Collapsing these into one "hard classes" story would be wrong. Only RIPP is hard to *model*.

### 12.4 ⚠ What does NOT follow, and why n = 4 forbids it

**Diversity does not predict the endpoint.** Spearman ρ(H, detect) = **−0.40** — effectively
nothing. REDOX is the counterexample: the least diverse class of the four and second-worst
on-target. What holds is a two-step chain, and only directionally:

| relation | ρ | p |
|---|---|---|
| H(prod) → val loss | +0.80 | 0.20 |
| val loss → detect | −0.80 | 0.20 |
| H(prod) → detect | −0.40 | 0.60 |
| modal share → detect | +0.40 | 0.60 |

**No correlation over four classes can ever be significant.** Spearman is a *rank* statistic; with
n = 4 there are 4! = 24 orderings, so even a **perfect** match arises by chance with probability
2/24 = **0.083**. That is the floor, and these sit at 0.20. ⇒ §12 reports shape, never evidence.
The benchmark has four classes by design (§4.4.2, NRPS and PKS dropped at the 8,192 nt bound), so
this cannot be fixed by measuring harder — only by treating §12.5 as the actual test.

### 12.5 The prediction this generates

If RIPP is hard **because** it is a union of ~9 subtypes, then conditioning on a *subclass* should
rescue it. That is a sharp, pre-specifiable prediction on a class currently flooring at 0.020.
The prior codebase's subclass result (cyclactone 124/124 — unreportable here: unfrozen config
plus a class token) was on `cyclic-lactone-autoinducer`, **a RiPP subtype**, with 6,497 records in
this corpus, which is why the prediction has a direction rather than being a bare guess.

⚠ **The subclass arm was DROPPED by decision on 2026-09-10** (SPEC §14.6). It changes the class
*granularity* rather than the method, so it sits outside a paper that benchmarks methods at the
benchmark's own class level. ⇒ **This prediction is therefore recorded and left untested.** It is
stated here so that it is a standing, falsifiable claim with a direction rather than a hindsight
explanation if anyone tests it later; the paper reports §12 as exploratory with an untested
prediction attached, and claims nothing about whether subclass conditioning would rescue RIPP.

## 13. G6 — the rank sweep. Capacity is not the binding constraint

Ranks {4, 8, 16, 32, 64} on the pooled nucleotide-balanced arm `W1n`, taxonomy prefix, equal-n
splits, `max_epochs=40`. **Held-out loss only — no generation, no antiSMASH** (§2.4: an arm
tuned against the endpoint that scores it is not a measurement).

Sweeping the pooled arm rather than one class is deliberate: §6 asks for one rank **per
substrate**, and choosing on a single class would have moved the undefended parameter — from
"rank 16 is undefended" to "the class we swept on is undefended" — rather than removed it.

| rank | best held-out loss | perplexity | trainable | % of model | best step | plateau spread |
|---|---|---|---|---|---|---|
| 4 | 0.89628 | 2.4505 | 2,618,880 | 0.23% | 250 | 0.00233 |
| 8 | 0.89593 | 2.4496 | 5,237,760 | 0.47% | 250 | 0.00170 |
| 16 | 0.89558 | 2.4487 | 10,475,520 | 0.94% | 250 | 0.00159 |
| 32 | 0.89553 | 2.4486 | 20,951,040 | 1.87% | 250 | 0.00130 |
| 64 | **0.89550** | 2.4486 | 41,902,080 | 3.75% | 250 | 0.00128 |

### 13.1 The result is not "rank 64 wins" — it is that rank does not matter

**A 16× increase in trainable parameters buys 0.00078 nats/nt.** Perplexity moves from 2.4505 to
2.4486: the model is choosing among ~2.45 bases either way.

The comparison that settles it is against the sweep's own noise. Within a single run, val loss
wanders by a mean of **0.00164** nats/nt across the plateau after its best step. The entire
across-rank effect is **0.00078** — **0.48×** the evaluation noise it sits inside. Rank is not
weakly beneficial here; it is **indistinguishable from measurement noise**, and reading the
ordering r64 < r32 < r16 < r8 < r4 as a capacity trend would be reading the noise.

Every rank also stopped identically — best step 250, `max_epochs` never binding.

⚠ **CORRECTED: no arm completed an epoch, and an earlier version of this paragraph said they
did.** `n_batches` is 32,176 at `grad_accum` 16, so one epoch is 2,011 optimiser steps; best step
250 is **12.4% of a single epoch**. The report's `epochs_run: 1` is `log[-1]["epoch"] + 1` and
every entry still carries `epoch: 0`. ⚠ A second claim in that paragraph — "train loss still
falling while val loss turned", cited as rank 16 going 0.769→0.736 — **is not in the artifact**:
over steps 250→350 train loss reads 0.76258, 0.78913, 0.79149, 0.73560, 0.77097, so the endpoints
*rise*, 0.736 is an interior dip and 0.769 appears nowhere in that log. The train series is a
sawtooth tracking batch composition, and the same shape appears at every rank. There is no
overfitting signature here to cite, and the val excursion at the stop (+0.00016) is an order of
magnitude below this section's own noise floor.

What stands without it: all five ranks stop at the same step with near-identical trajectories, so
the plateau is set by the data and the schedule rather than by capacity.

### 13.2 What this discharges, and what it does not

⇒ **Rank 16 is now defended**, and every `W1`/`W2` number in §8, §10, §11 and §11.5 keeps it —
not because 16 is optimal, but because the curve is flat and 16 sits 0.00008 nats/nt off the
minimum, well inside the noise. §6's requirement that capacity be "a declared, defended
parameter, not an inherited default" is met, and the debt §14.2 recorded against every W-arm is
paid.

⚠ **RETRACTED 2026-09-11.** This paragraph previously read: *"Adapter capacity is not why the
endpoint rates are low. A 0.020 RIPP rate and a 0.130 seeded per-class rate will not be raised by
a bigger adapter. That is a real negative result and the paper should report it as one."* **That
claim is not supported by this sweep and must not appear in the paper.** G6 read held-out loss and
nothing else — by design, because §2.4 forbids selecting an arm on the endpoint. No arm here was
generated or scored; the 0.020 and 0.130 rates come from §11's frozen run, not from these ten
adapters. Inferring an endpoint null from a loss null is exactly the step the benchmark exists to
refuse, and under §6.4 it would be a null with neither power nor a manipulation check.

⇒ What G6 supports is narrower and still useful: **held-out loss is flat in rank**, so an
objection of the form "the adapters were too small to model the target classes" is answered on
the modelling axis. Whether a larger adapter would move the *endpoint* is **untested**, and the
only way to test it is to generate and score at two ranks — which would then be an endpoint
comparison, not a selection, and is therefore permitted but has not been run.

⚠ **Bounded, and here is the bound.** This sweeps rank *under this training protocol* — same LR,
same schedule, same patience, same early stopping, all of which fired at the same step for every
rank. It does not show that capacity could never matter under a longer or differently tuned
schedule. It answers exactly what §6 asks (rank chosen on held-out loss under the arm's own
protocol) and no more.

⚠ **The selected rank is upper-leaning for the per-class arms.** `W1n` trains on 32,176 records
across four classes; a `W2` arm sees 8,044 of one. A pooled arm with more data and a more
heterogeneous target should if anything want *more* capacity, so a rank chosen here is not an
underestimate for `W2`. Given the curve is flat this is immaterial, but it would matter if the
curve were not.

⇒ **G6b (depth) becomes more interesting, not less.** If adapter *width* is irrelevant across a
16× range, the open question is whether *placement* is — whether the four attention blocks of 25
that carry adapters are the right ones. That is a different axis and this sweep says nothing
about it.

## 14. G6b — the depth sweep. Placement is decisive where capacity was not

Rank 16 (G6's value) on the pooled `W1n` arm, varying only **which of Evo2-1B's 25 blocks carry
adapters**. Held-out loss only — no generation, no antiSMASH (§2.4). The `all` row is
`G6_r16_W1n` itself, reused rather than re-run: it is exactly this arm at every block, and a
second run would only add noise to its own baseline.

| depth set | blocks | trainable | best held-out loss | vs `all` | × noise | best step |
|---|---|---|---|---|---|---|
| `all` | 0–24 | 10,475,520 | **0.89558** | — | — | 250 |
| `every_other` | 0,2,…,24 | 5,437,440 | 0.89669 | +0.00111 | 0.7 | 325 |
| `early` | 0–7 | 3,317,760 | 0.89681 | +0.00123 | 0.8 | 275 |
| `attention_only` | 3,10,17,24 | 2,088,960 | 0.90110 | +0.00552 | 3.5 | 550 |
| `middle` | 8–16 | 3,717,120 | 1.12099 | +0.22541 | **143** | 950 |
| `late` | 17–24 | 3,440,640 | 1.25533 | +0.35975 | **228** | 750 |

All five converged on early stopping with `max_epochs=40` never binding, and all plateaued —
`middle` and `late` are genuinely worse, not undertrained. Mean within-run plateau noise across
the six arms is 0.00157 nats/nt.

### 14.1 Every arm containing an early block did well; both that lack one did badly

| contains ≥1 block from 0–7? | arms | held-out loss |
|---|---|---|
| yes | `all`, `every_other`, `early`, `attention_only` | 0.8956 – 0.9011 |
| no | `middle`, `late` | 1.1210 – 1.2553 |

The separation is complete. Every arm touching blocks 0–7 lands inside a 0.0055 band; the two
that do not are 143× and 228× the noise away from it. `attention_only` carries just **one** early
block (block 3) among its four and still lands with the first group.

⚠ **The BINARY reading is not identified by this design.** Six points cannot separate "contains
an early block" from a monotone account — the depth sets also differ in how many early blocks they
contain (`all` 8, `early` 8, `every_other` 4, `attention_only` 1, `middle` 0, `late` 0), and that
ordering fits the losses just as well. Distinguishing them needs single-block or
one-block-at-a-time arms, which have not been run. The claim this section can carry is the
**association**: no arm containing an early block performed badly, and no arm lacking one
performed well.

### 14.2 It is placement, not capacity — and the comparison is capacity-matched

`early`, `middle` and `late` hold 3.32M, 3.72M and 3.44M trainable parameters: within 12% of each
other. Their held-out losses are 0.897, 1.121 and 1.255 — a spread of **0.359 nats/nt, 228× the
noise, at essentially identical capacity**. `middle` has the *most* parameters of the three and
is second-worst. No capacity account survives this.

Set against §13, the two gates answer the same question and disagree completely:

| axis | manipulation | effect | relative to noise |
|---|---|---|---|
| **capacity** (G6) | rank 4 → 64, **16× the parameters** | 0.00078 | **0.50×** |
| **placement** (G6b) | same rank, blocks moved | 0.35975 | **228×** |

⚠ **"460×" is rhetorical, not a measured quantity, and should not be written as one.** It is
0.35975 / 0.00078 — a ratio of two manipulations whose *ranges were chosen arbitrarily*. Rank
4→64 and all-blocks→late-blocks are not commensurable moves; a wider rank range or a milder depth
contrast changes the number at will, and the denominator is itself a noise-scale quantity. The
paper should state the two effects and their noise multiples side by side and let them speak.

⇒ What survives: **moving a fixed-capacity adapter changes held-out loss by two orders of
magnitude more than scaling that capacity 16× does.** §13's "capacity is not the binding
constraint" is consistent with this and is sharpened — capacity is not binding *at the placement
already in use*.

### 14.3 What this buys, and what it does not

⇒ **`early` is statistically indistinguishable from `all`** — +0.00123 against a 0.00157 noise
floor — at **0.32× the parameters** and 17 fewer blocks. Adapting blocks 0–7 recovers everything
adapting all 25 does. `attention_only` gets within 3.5× noise at 0.20× the parameters.

⇒ **Rank 16 on all blocks is retained for every reported arm.** `all` is the (statistically tied)
minimum, it is what §8, §10, §11 and §11.5 already used, and switching now would invalidate those
runs to buy nothing. G6b's value is the *knowledge*, not a configuration change.

⚠ **This does not say late blocks are useless, and an earlier version overstated it.** That
version said the arm "simply cannot improve held-out loss by adapting them". It can: LoRA is
zero-init, so step 0 *is* the base model, and the `late` arm falls monotonically from 1.30914 at
its first evaluation to 1.25533 — an improvement of **0.0538 nats/nt, 34× the noise floor** —
before plateauing. `middle` likewise goes 1.28980 → 1.12099. The supportable statement is that
late placement **improves an order of magnitude less than early placement**, not that it does
nothing.

⚠ **And no un-adapted baseline was ever measured.** No arm logs a step-0 evaluation and no
un-adapted held-out loss for the `W1n` validation set appears anywhere in this document, so every
"improvement" above is read off a first-eval-at-step-25 proxy rather than a true zero point. A
step-0 eval costs one forward pass per arm and should be added before any of this is written up.

⚠ **Greedy, as §14's gate row states.** Rank was fixed at 16 from G6 and depth swept at that
rank. A joint 2D sweep could differ — though with the rank axis flat at 0.50× noise and the depth
axis at 228×, a strong interaction would be surprising.

### 14.4 ⚠ The I1 measurement does NOT point the same way — corrected

An earlier version of this section claimed I1's direction derivation "points the same way" as the
depth sweep, and offered the joint reading that *class-relevant structure sits early and is gone by
the final block*. **Its own table contradicts that**, and the claim is withdrawn.

Relative class difference per attention site, all four classes at the same n (192 per side,
`I1_<CLASS>_base_n192`, all passing their §6.4 check):

| class | block 3 | block 10 | block 17 | block 24 |
|---|---|---|---|---|
| TERPENE | 0.2097 | 0.2119 | 0.2027 | **0.00058** |
| REDOX_COFACTOR | 0.1848 | 0.2121 | 0.2234 | **0.00064** |
| RIPP | 0.1145 | 0.1485 | 0.1708 | **0.00045** |
| ARYLPOLYENE | 0.1165 | 0.1265 | **0.1894** | **0.00014** |

⚠ **There is no early-to-late gradient here.** Across blocks 3 → 10 → 17 the class difference is
flat for TERPENE and *rises* for the other three — for ARYLPOLYENE block 17 is the **largest** of
the four sites, 1.6× block 3. And **block 17 is the first block of `late`** (`DEPTH_SETS["late"]`
= 17–24), the placement G6b measures as the single worst at 1.25533.

⇒ So at block 17 the two measurements point in **opposite** directions: I1 finds the most class
difference there, G6b finds adaptation there catastrophic. **They agree at exactly one site,
block 24**, where I1 finds ~300–1,300× less class difference than at the other three and G6b
finds late placement worst. One shared point is not a converging result.

⚠ **Block 24's near-zero reading may not even mean "no class structure".** Its target and other
class means are 0.10383 and 0.10383 — the site's mean output is essentially the same number
regardless of input, so the near-zero difference is consistent with the site being close to
input-independent at this pooling, which is a different fact from "the class signal is gone".
Distinguishing them needs the per-record variance at that site, which is not recorded.

⇒ **Nothing in §14 should be read as a claim about where class information lives in the model.**
G6b measures where LoRA adaptation helps held-out loss; I1 measures a difference of pooled means
at four attention sites. They are different quantities, they agree at one of four sites, and the
earlier "converging evidence" framing overstated a coincidence.

⚠ Two further defects in the withdrawn version, recorded so they are not repeated: it quoted
TERPENE at n=64 and ARYLPOLYENE at n=192 **in the same table without saying so**, and the
ARYLPOLYENE row came from an artifact that was **failing its manipulation check** at the time it
was cited. The table above is one n for all four classes, all passing.

## 15. G9 — the injection-magnitude sweep. α = 0.3

SPEC §6's G9 row fixes the criterion and it is **not** the endpoint (§2.4): *the largest α at
which generation quality is not degraded beyond a stated tolerance, read as per-token likelihood
on the model's own output and coding density.* A ceiling-finding sweep, not an optimisation —
there is no "best α" to locate.

`run.g9_alpha` reads exactly four quantities per arm — median coding density, median self-NLL,
`hit_eos` rate and k-mer distinctness — and never `detected`, `on_target`, `products` or
`observed_classes`. Those fields sit in the scored records it walks past, which is why the
restriction is enforced in the reading code and recorded in each artifact as `selection_fields`
alongside an empty `endpoint_fields_read`.

| α | W0 coding | W0 self-NLL | W0 adm. | W2 coding | W2 `hit_eos` | W2 med len | W2 adm. |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.2044 | 1.3599 | baseline | 0.9746 | 0.840 | 2,384 | baseline |
| 0.05 | 0.2766 | 1.3305 | ✓ | 0.9645 | 0.900 | 2,492 | ✓ |
| 0.10 | 0.2435 | 1.3295 | ✓ | 0.9667 | 0.980 | 1,983 | ✓ |
| 0.20 | 0.2512 | 1.3281 | ✓ | 0.9754 | 0.980 | 2,014 | ✓ |
| **0.30** | 0.1845 | 1.3505 | **✓** | **0.9857** | 1.000 | 1,934 | **✓** |
| 0.50 | 0.1207 | 1.2835 | ✗ −41% | 0.2201 | **0.040** | 8,192 | ✗ −77% |

⇒ **α = 0.3 on both weight states.** The cliff between 0.3 and 0.5 is sharp rather than gradual:
on `W2`, coding density falls 0.986 → 0.220 and 90% of generations stop terminating at all.

### 15.1 Two corrections to the sweep itself

⚠ **The first pass reported "no admissible α" and was wrong.** Its grid began at 0.5 — past the
entire admissible region — so a gate outcome was nearly recorded from a sweep that never sampled
below its own first failure.

⚠ **Both tolerances were initially two-sided, and both had to become one-sided.** Coding density
was fixed *before* this run: the two-sided version rejected `W0` at α=4 for coding density 0.3153
against a 0.2044 baseline — i.e. **for being 54% better**. `hit_eos` was fixed *after*, and that
matters more because it changed a selected value: two-sided, it rejected `W2` at α=0.3 for
terminating **more** (0.840 → 1.000) and selected 0.2 instead.

That amendment was made after seeing the data, so the check came first: at α=0.3 there is no
truncation (minimum length 1,002, nothing under 500 nt), median 1,934 against real TERPENE cores
at 1,375 (§12.1), and coding density 0.986 — **the highest in the sweep**. The two-sided rule was
rejecting the healthiest arm measured. Both values are stored (`chosen_alpha` 0.3,
`chosen_alpha_two_sided_eos` 0.2) so the amendment is visible rather than absorbed.

### 15.2 ⚠ Limitations that bound every I1 result

**The two baselines are not comparable.** `W0` at α=0 has coding density 0.204 and **never
terminates** (`hit_eos` 0.000, median 8,192); `W2` has 0.975 and stops 84% of the time.
Degradation is relative to each arm's own α=0, so "admissible" on `W0` means *no worse than an
already-poor baseline*, not *healthy*.

**α was swept on TERPENE only and applied to all four classes by decision.** This is a stated
transfer assumption, not a measured one. There is direct evidence against free transfer: at α=1
the KL against unsteered was 1.034 for TERPENE-`W2` but **1.601** for ARYLPOLYENE-`W2`, so the
same magnitude is not the same disruption per class — and the classes differ in direction
strength too (steered-site cosine 0.755 for TERPENE on base against 0.307 for ARYLPOLYENE).
Given the 0.3 → 0.5 cliff, 0.3 could sit inside tolerance for one class and past it for another.
⇒ Coding density, `hit_eos` and self-NLL are recorded per arm, so **whether 0.3 held for each
class is checkable after the fact**, and any arm that degraded must be read as uninformative
rather than negative. It was not checked before the arms ran.
