# FABLE5 Scientific Audit — BCGModelling Evo2 BGC-Generation Project

*Adversarial, science-first audit. Each finding below was cross-examined by two independent verifiers (Critical findings via steelman-then-rebut). "Confirmed" = both verifiers agree it is real; "Disputed" = they disagree; "Refuted" = both reject.*

**Goal under audit:** fine-tune Evo2 7B (LoRA) to generate synthesis-ready, class+taxonomy-conditioned BGCs that are correctly-classified, domain-complete, **NOVEL**, synthesizable, and wet-lab validatable.

> **SUPERSEDED (2026-06-17) — point-in-time audit record; do not read as current.**
> The eval suite has since been rewritten to named **CHECKS → QUESTIONS** scoped to
> is-it-a-BGC / correct-class / plausible-proteins / NOVEL / complete (the wet-lab axes —
> synthesizability, E. coli expressibility — were pruned). antiSMASH is now the
> recalibrated `is_bgc`/`correct_class` gate (~0.97 on real cores); pyrodigal replaced
> the six-frame ORF finder; the active dataset is `splits_core`. See
> [`REDESIGN_PLAN.md`](REDESIGN_PLAN.md) and `src/bgc_pipeline/evaluation.py`.

---

## Verdict

As built, the project is **not yet on track to credibly demonstrate the goal**, even though much of the engineering plumbing (leakage-free genome/exact-sequence splitting, prefix masking, EOS placement, resume machinery, conditioning-adherence likelihood eval) is sound and verified. The decisive problem is that **none of the five non-negotiable success criteria — especially NOVELTY and E. coli expressibility — is actually instrumented in the pipeline that produces the PASS/FAIL verdict.** The single biggest risk is that **the novelty/anti-memorization safety gate is simultaneously (a) absent from the scored eval suite, (b) miscomputed when run standalone (Jaccard-ranked top-3 misses memorized fragments), (c) uncalibrated (real held-out BGCs already score "memorized"), and (d) scoped to only 18K of ~346K known BGCs** — so a model that regurgitates a training/leaked BGC could be declared novel and routed to wet-lab synthesis. Layered on top, the headline "E. coli-expressible" deliverable has **zero training signal** (0 NRPS, 0 OTHER E. coli examples; no recoding objective), making two of the three wet-lab targets unsupported by the data.

---

## Confirmed Findings

### CRITICAL

---

#### C1 — Memorization check selects nearest-neighbor by Jaccard but reports containment, missing memorized fragments [CRITICAL / HIGH]

- **Dimension:** Correctness bugs
- **Location:** `scripts/memorization_check.py:127-144` (`nearest()`); CLI call at `:197` uses default `top_m=3`
- **What is wrong:** `nearest()` ranks all training BGCs by bottom-k MinHash **Jaccard**, then computes exact **containment** only for the top-3 Jaccard candidates and reports that as `max_containment`. Jaccard (`|A∩B|/|A∪B|`) and containment (`|q∩ref|/|q|`) diverge for asymmetric sizes: a short query that is an exact subsequence of a long training BGC has containment ≈1.0 but tiny Jaccard. Both verifiers reproduced end-to-end on **real training data**: a 16,384 bp exact slice of a real 262 kb training BGC (true containment 1.0) ranked #6 by Jaccard, fell outside top-3, and was reported `max_containment=0.3652`, **flagged NOT memorized**.
- **Why it matters for the goal:** Novelty (criterion 3) is the defining safety gate before committing synthesis budget. With default `--max-new-tokens=16384`, generated sequences are shorter than **78.3%** of training BGCs (median ~24.8 kb), so "memorized fragment of a longer BGC" is the *common* case — exactly the case the bug silently passes.
- **How to verify:** Build an index with one long record + ≥3 partial-homolog distractors that out-share k-mers with a query that is an exact slice of the long record; call `nearest(query, index, top_m=3)` and observe `max_containment << 1.0` while `exact_containment(query, source) == 1.0`.
- **Suggested fix:** Rank/refine candidates by a **containment-aware** estimator (asymmetric MinHash containment, or exact containment over all refs sharing any hash, or a global k-mer inverted index); raise `top_m` substantially. Add an integration test driving `nearest()` with a memorized fragment of a long reference asserting `max_containment≈1.0`. Note `tests/test_memorization.py` never calls `nearest()`.
- **Verifier note:** Failure regime is conditional (requires ≥3 training seqs that out-Jaccard the true source), but BGC gene-cluster families make this regime common, so a conservative gate cannot rely on it being rare. *(See also M11 — same defect, scoped at Major.)*

---

#### C2 — Near-duplicate nucleotide leakage across genome-disjoint splits (val↔train) [CRITICAL / HIGH]

- **Dimension:** Scientific / methodological soundness
- **Location:** `/data2/ds85/bgcmodel_data/splits_curated/{train,val}.jsonl`; `scripts/split_dataset_grouped.py:56-71,204-209` (dedup is exact-md5 only)
- **What is wrong:** Genome and exact-md5 overlap are genuinely 0, but genome-disjoint ≠ near-duplicate-disjoint. Per-record nearest-neighbor scans found real near-twins crossing the held-out boundary (e.g. val→train containment up to 0.975–0.989; several at 0.85–0.91 with identical lengths and same species, different genome). One verifier found a val record whose **first-2048nt containment = 1.0000** to a train record — i.e. a verbatim copy in exactly the window the val loss scores. Species-level val/train overlap is 31.2%.
- **Why it matters for the goal:** First-window teacher-forced val loss drives best-checkpoint selection and early stopping (`finetune_evo2_lora.py:2167-2238`). Near-twins concentrated in the first window deflate that signal and corrupt model selection — the failure the group-aware re-split was meant to fix — and weaken any generalization claim.
- **How to verify:** Run canonical-kmer containment of val queries vs train; confirm the per-record (not global-union) tail. Inspect first-window containment of top pairs.
- **Suggested fix:** Add an identity-cluster pass (mmseqs2 / minimap2 / CD-HIT-EST at ~90–95% over the BGC interval) and assign whole clusters to one split; or drop/flag val/test records with ≥0.90 first-window containment to train and re-validate checkpoint decisions. Report the residual near-dup distribution.
- **Verifier note:** Both confirmed; one rated it Major rather than Critical because the high-containment tail (~1–2.5% ≥0.95) is smaller than the finding's headline (the finding's 5%/21.7% used a looser global-union metric that overstates per-record similarity ~2×). The first-window=1.000 verbatim case and the corrupted model-selection mechanism keep it goal-critical.

---

#### C3 — Memorization threshold (0.95 k-mer containment) is uncalibrated; real held-out BGCs trip the flag [CRITICAL / HIGH]

- **Dimension:** Scientific / methodological soundness
- **Location:** `scripts/memorization_check.py:181-199` (`--memorized-threshold 0.95` on `max_containment`); `eval/positive_control_mibig.jsonl`
- **What is wrong:** Both verifiers reproduced exactly: running the check on the 20 real held-out MiBIG BGCs (the project's own calibration set) yields **2/20 ≥ 0.95** (BGC0000744 = 0.996 vs train near-twin BGC0000741; BGC0002052 = 0.955) and **3/20 ≥ 0.8**. Real, legitimate, non-memorized BGCs already exceed the flag threshold. Separately, the metric (k-mer containment) differs from the research plan's stated criterion (`BGC_Research_Plan.md:119,133`: >95% **nucleotide identity** over the BGC interval), with no containment-vs-identity calibration. The BGC0002052 case (containment 0.955, true Jaccard 0.367 — a short cluster *embedded* in a 2.55× longer region) is a demonstrable false positive.
- **Why it matters for the goal:** If real-but-novel BGCs score in the same 0.95–0.996 band as a true near-duplicate, a generated copy of a training BGC passes/fails the same way a genuinely novel one does — the novelty claim is unfalsifiable as instrumented.
- **How to verify:** `scripts/memorization_check.py --query eval/positive_control_mibig.jsonl` → max 0.996, 2/20 ≥ 0.95.
- **Suggested fix:** Replace/augment containment with alignment-based percent identity over the aligned interval (minimap2/megablast); calibrate the threshold empirically from the positive-control distribution (report separation/ROC, not a hard 0.95); require BOTH high identity AND high aligned fraction; de-duplicate the BGC0000744/741 cross-split pair; finish/implement M6 (it is a non-parsing stub) so novelty is not single-metric.

---

#### C4 — E. coli chassis is almost absent from training (106/18,270; 0 NRPS, 0 OTHER) [CRITICAL / HIGH]

- **Dimension:** Data gaps
- **Location:** `/data2/ds85/bgcmodel_data/splits_curated/train.jsonl`; per-class: SIDEROPHORE 60, ARYLPOLYENE 24, RIPP 10, TERPENE 5, PKS_NRPS_HYBRID 4, SACCHARIDE 2, PKS 1, **NRPS 0, OTHER 0**
- **What is wrong:** The wet-lab deliverable is E. coli-expressible BGCs for three targets: violacein (OTHER), carotenoid (TERPENE), indigoidine (NRPS). The taxonomic tag is part of the conditioning prefix at both train (`finetune_evo2_lora.py:489-490`) and inference (`generate_bgc.py:48,55`). But the curated train set has **0 Escherichia NRPS and 0 Escherichia OTHER** — the two classes of two of three named targets — and only 5 TERPENE. The model is asked to generate an NRPS conditioned on an E. coli lineage it never saw paired with NRPS. Curation *worsened* this: uncurated grouped train had 8 NRPS / 48 OTHER / 673 TERPENE E. coli records → 0/0/5 after the chassis-blind per-class cap + phylum×length stratification.
- **Why it matters for the goal:** Criteria 4/5 (synthesizable / plausibly E. coli-expressible / wet-lab validatable). The (class, E. coli) conditioning cell is out-of-distribution for exactly the targets; the model will most plausibly default to source-organism (Actinomycetota/Pseudomonadota, ~0.71 GC) statistics. This is the upstream/data-side root cause of the already-"resolved" eval-side fix (AUDIT C4), whose training-side half is explicitly still open.
- **How to verify:** `grep ESCHERICHIA` curated train, bucket by `compound_class` (reproduces 0/0/5). Confirm `generate_bgc.py` builds prefix from `--taxon`.
- **Suggested fix:** Document the chassis contract honestly — either (a) Phase-1 = faithful generation in a well-represented source taxon then recode for E. coli as a separate step, or (b) add an explicit E. coli-conditioned data arm / codon-optimization objective. Restrict wet-lab target prompts to taxa with adequate (class, taxon) support.

---

#### C5 — Chassis-transfer objective is encoded nowhere in training; model only learns P(seq | class, SOURCE organism) [CRITICAL / HIGH]

- **Dimension:** Scientific-reasoning gaps
- **Location:** `scripts/finetune_evo2_lora.py:483-505` (prefix = source `taxonomic_tag`); every curated record pairs a BGC with its own native organism; `docs/gputee/PROJECT_GUIDE.md:971-988`
- **What is wrong:** The project is framed as "transposition" (re-express a known BGC in E. coli), but training has **no (source→chassis-recoded) pairs and no codon-optimization/recoding objective**. Each record conditions on its OWN source taxonomy; the loss supervises only the native sequence (prefix masked). So the model learns to reproduce native source statistics. Conditioning on an E. coli tag at inference retrieves the sparse native-E.-coli prior (~219 of 18,270 records, mostly not the target classes), not an E. coli-optimized version of a foreign cluster.
- **Why it matters for the goal:** Criterion 4 and the entire wet-lab plan rest on chassis adaptation the loss never teaches. "Transpose a known cluster to E. coli" is unsupported by the training setup as built.
- **How to verify:** Grep training/data for any recoding/codon/chassis-target field (none). After a run, generate the same class under Streptomyces vs E. coli tags and compare GC/CAI — if output tracks the conditioned taxon's native stats rather than E. coli-optimal codons, the chassis claim fails.
- **Suggested fix:** Either (a) add a deterministic E. coli codon-optimization/recoding post-step (DNA Chisel / codon harmonization) and evaluate THAT output, (b) construct chassis-transfer training pairs, or (c) rescope the deliverable to "native-organism-faithful BGCs" until a recoding stage exists.
- **Verifier note:** One verifier rated this Major (it is documented as AUDIT C4, training-side deliberately deferred, eval side fixed). Both agree the gap is real; it overlaps C4 and M13 — treat as one chassis-transfer cluster. The finding's "93 E. coli" figure should read ~219 curated / ~5,310 grouped.

---

#### C6 — Novelty/memorization gate is not in the scored evaluation pipeline (orphaned script + stub metric) [CRITICAL / HIGH]

- **Dimension:** Goal alignment & justification
- **Location:** `src/bgc_pipeline/evaluation.py:666-719` (metric_6 stub), `:1019-1098` (summary runs only metrics 1-8); `scripts/memorization_check.py` (called by nothing)
- **What is wrong:** (a) `metric_6_bigscape` runs BiG-SCAPE but never parses distances — sets only `result["note"]="...distance parsing requires version-specific logic"`, never sets `"pass"`, so it is permanently `no_verdict`. (b) `memorization_check.py` is imported/called by nothing in the eval drivers (`grep` confirms only the script + its unit test). (c) `evaluate_bgc()`'s summary loops `range(1,9)` — only metrics 1-8, and the novelty-adjacent ones (M6 stub, M8 protein-identity vs UniRef50 default None) never gate. So a near-verbatim copy of a training BGC passes the entire scored suite and is reported a success.
- **Why it matters for the goal:** Novelty is the defining pre-synthesis safety criterion; the artifact a user trusts (the PASS/FAIL summary) *structurally cannot* fail a sequence for being memorized.
- **How to verify:** Read `evaluation.py:1080-1098`; `grep -rn 'memorization_check' --include=*.py .` returns only the script; read `metric_6` lines 716-717.
- **Suggested fix:** Wire `memorization_check.nearest()` (or an MMseqs2/minimap2 nucleotide-identity vs `train.jsonl`) into `evaluate_bgc` as a first-class gating metric (FAIL ≥0.95, WARN ≥0.8) on the leak-free split; finish or remove the M6 stub. Fix C1 first so the gate is correct.

---

#### C7 — Residual cross-split near-duplicate leakage for MIBiG / MIBiG↔antiSMASH clusters [CRITICAL / HIGH]

- **Dimension:** Goal alignment & justification
- **Location:** `scripts/split_dataset_grouped.py:60-71` (fallback to `accession::{acc}` for `genome_accession=None`); reproduced: BGC0002052 (test, OTHER) `max_containment=0.955` vs train `GCF_026239535.1.region25`
- **What is wrong:** All 2,389 MIBiG records have `genome_accession=None`, so each becomes its own singleton group keyed by its unique BGC accession and is scattered across train(304)/val(1038)/test(1047). MIBiG curated clusters frequently re-appear (different coordinates/accession) in the antiSMASH bulk; exact-md5 dedup and genome grouping cannot link `BGC*` to its `GCF_*` twin. Both verifiers reproduced the 0.955 case plus **two perfect 1.000 copies** (BGC0000142→GCF_000016425.1.region14, BGC0000001→GCF_000204155.1.region7), concentrated in drug-relevant PKS/NRPS. 79 MIBiG species appear in both train and test.
- **Why it matters for the goal:** MIBiG is the experimentally-validated subset and the source of the wet-lab targets. Leakage here contaminates val/test loss, model selection, and (with no novelty gate at selection time) the positive-control calibration baseline itself — a "novel" generation could be a memorized leaked cluster precisely where it matters most.
- **How to verify:** Confirm the only cross-split genome overlap is `genome_accession==None`; run `memorization_check.py` on the positive control vs train (BGC0002052 at 0.955).
- **Suggested fix:** Cluster sequences by content (MMseqs2/MinHash at ~50–90%) BEFORE splitting and assign whole clusters to one split; or hold all MIBiG in one split. Rebuild the positive control with a containment-based (not md5-only) disjointness guard. Add a pre-train assertion failing on any cross-split pair >0.9 containment.
- **Verifier note:** Severity anchored on novelty-falsifiability + contaminated positive control, not val-loss deflation (MIBiG is ~4% of val, so the selection-corruption leg is modest).

---

### MAJOR

---

#### M1 — Taxon-faithfulness fix (M7/C4) is implemented but never wired into the eval driver [MAJOR / HIGH]

- **Dimension:** Scientific / methodological soundness (Claimed-fix verification)
- **Location:** `scripts/evaluate_bgc.py:139-145,156,162`; fix lives in `src/bgc_pipeline/evaluation.py:759-801,1019-1031`
- **What is wrong:** The library fix is correct, but `evaluate_bgc.py` builds `EvalConfig` without `load_taxon_profiles()` (config.taxon_profiles stays empty) and calls `evaluate_bgc(seq, acc, cls, config)` without `expected_taxon`. So `resolve_taxon_profile` returns None for every record, and M7's verdict is always `None` (`no_verdict`). `data/processed/taxon_profiles.json` exists and loads cleanly — the fix is one wiring line away. *(This is the same defect as C15 below; listed once.)*
- **Why it matters for the goal:** M7 backs criterion 4 (organism compatibility). As wired it produces no PASS/FAIL signal, so "C4 resolved" is only half-true and a reviewer would believe faithfulness is graded when it is not.
- **How to verify:** Read `evaluate_bgc.py:139-156`; `grep -rn 'load_taxon_profiles\|expected_taxon' scripts/` → only `build_taxon_profiles.py`. Run on the positive control: M7 = `no_verdict`.
- **Suggested fix:** In `evaluate_bgc.py`, load `config.taxon_profiles = load_taxon_profiles(...)` and pass `expected_taxon=rec.get('taxonomic_tag','')`; add a `--taxon-profiles` flag and an end-to-end assertion that M7 returns a non-None verdict.

---

#### M2 — No causal class-conditioning control at the generation level [MAJOR / HIGH]

- **Dimension:** Scientific / methodological soundness
- **Location:** `scripts/evaluate_bgc.py:160-164`, `scripts/eval_smoke.py:188-196` (only negative control = mononucleotide shuffle); `eval_conditioning_adherence.py` (likelihood ranking, not generation)
- **What is wrong:** Criterion 1 is a generative causal claim about the COMPOUND_CLASS prefix. The only negative control is a mononucleotide shuffle (preserves GC, destroys all structure — proves almost nothing). The adherence eval is *discriminative* (scores fixed real held-out sequences under each class prefix). No class-swap experiment GENERATES under class A and checks via antiSMASH (M1) that output is called A, vs a scrambled/mismatched prefix. The repo's own `AUDIT_FINDINGS.md` prescribes this control and confirms it is unimplemented.
- **Why it matters for the goal:** Generation prompts are drawn from val/test (54% TERPENE+RIPP), so a high M1 class-match rate is confounded by the natural prior and Evo2's pretrained grammar. "Class-conditioned generation works" (gating Phase 2 + wet-lab spend) would be unsupported.
- **How to verify:** Confirm `score_model` scores `rec['sequence']` (real); confirm `generate_bgc.py` has no swap/mismatch mode.
- **Suggested fix:** Add a generation arm: per class, generate N from the true prefix and N from a mismatched/scrambled prefix (fixed taxon, matched seed), run antiSMASH M1, report a requested-vs-recovered confusion matrix; require the diagonal to beat the majority-class prior AND base-Evo2 generations from identical prefixes.

---

#### M3 — In-loop model selection uses teacher-forced first-window val loss as a proxy for generation quality [MAJOR / HIGH]

- **Dimension:** Scientific / methodological soundness
- **Location:** `scripts/finetune_evo2_lora.py:1096-1139` (token-weighted overall), `2167-2213` (best/early-stop on val_loss); val set natural-imbalanced (TERPENE+RIPP ~54%)
- **What is wrong:** Validation is first-window teacher-forced CE only — measures calibration of the conditional distribution, not generation quality. No demonstrated correlation with generation exists (no run has completed). The aggregate is token-weighted over the imbalanced val set with **no `val_by_class` breakdown** (only `val_by_length`), so a regression on drug-relevant long classes (NRPS/PKS) can be masked. The validated subset is the first ~500 file-order records (shuffle=False), preserving the skew.
- **Why it matters for the goal:** Best-checkpoint/early-stop choose the adapter used for all downstream generation. A signal dominated by short TERPENE/RIPP may not pick the best NRPS/PKS generator.
- **How to verify:** Read `run_validation` (no per-class breakdown) and the early-stop block. After a run, correlate per-class first-window val loss vs per-class antiSMASH M1 pass rate.
- **Suggested fix:** Add `val_by_class`; macro-average (or class-weight) the early-stop signal; once generation exists, periodically select on a small generation-based eval (M1/M2); shuffle/stratify the val subset.

---

#### M4 — Rare classes (PBDE=2, LADDERANE=14, PHOSPHOGLYCOLIPID=13, …) cannot learn conditioning and will memorize [MAJOR / HIGH]

- **Dimension:** Data gaps
- **Location:** `splits_curated/train.jsonl` per-class counts; val/test <10 for NUCLEOSIDE/PHOSPHOGLYCOLIPID/LADDERANE
- **What is wrong:** 12 of 26 classes never reach the 1000 cap; six have 2–136 records. With so few examples the model can only memorize, contradicting novelty for those classes. Worse, val/test have <10 records for three classes, so per-class eval is statistically meaningless (e.g. 1/1 recall reported as success).
- **Why it matters for the goal:** Criteria 1/3 (correct class AND novel). For low-count classes any class-match success is almost certainly regurgitation, and the harness cannot detect it (val/test n<10).
- **How to verify:** Per-class counts on curated train (12 < 1000) and val/test (3 < 10). Cross-reference with `memorization_check.py` containment per class.
- **Suggested fix:** Define a minimum-train-count threshold (e.g. ≥50–100 distinct-genome records); report per-class memorization containment; suppress/footnote per-class metrics where eval n<~20; consider dropping degenerate classes (PBDE/LADDERANE/PHOSPHOGLYCOLIPID) from the Phase-1 conditioning vocabulary.
- **Verifier note:** One verifier corrected "the eval cannot even detect it" — corpus-wide memorization *is* detectable via `memorization_check.py`; what lacks power is per-class generation-quality eval. Does not invalidate well-populated drug-relevant classes (all at cap).

---

#### M5 — Curation drops 93.5% of training-eligible BGCs and all but ~93–219 E. coli records [MAJOR / HIGH]

- **Dimension:** Scientific-reasoning gaps
- **Location:** `scripts/curate_dataset.py:90-100,293` (train-cap 1000/class); grouped train 280,448 → curated 18,270
- **What is wrong:** The LIMA "we only teach the interface" rationale justifies the cut, but the same docs claim the model learns chassis sequence statistics (codon/GC/operon spacing) — that is teaching CONTENT, not format, which LIMA-style minimal data does not instill for a barely-represented chassis. After curation only ~93–219 E. coli BGCs remain (native, mostly off-target). The argument cannot simultaneously be "minimal data is fine (format only)" AND "the model learns chassis sequence statistics."
- **Why it matters for the goal:** If Phase-1 only teaches the interface, generated content is Evo2's prior steered by a thin prefix — threatening criteria 1/2 (content control) and 4 (chassis-appropriate sequence).
- **How to verify:** Count curated E. coli; compare per-class generated GC/CAI vs conditioned-taxon native stats after training; run `eval_conditioning_adherence.py --compare-base`.
- **Suggested fix:** Separate the two claims: validate that Phase-1 moves adherence above the base-model baseline; treat chassis-statistics learning as a separately-validated hypothesis or move it to a recoding step. If chassis content is required, retain far more E. coli / Gammaproteobacteria records rather than capping uniformly.
- **Verifier note:** One verifier confirmed (doc contradiction is real, PROJECT_GUIDE cites "6,239 E. coli" which curation reduced to ~93–219). One refuted as overstated (faithfulness is graded at phylum granularity where Pseudomonadota is abundant; class-conditioned content not E.-coli-count-dependent). Status = confirmed; impact debated.

---

#### M6 — In-loop validation uses only the first 32k window, so selection never measures completion/termination [MAJOR / HIGH]

- **Dimension:** Scientific-reasoning gaps
- **Location:** `scripts/finetune_evo2_lora.py:841-847` (first_window_only filters to nt_start==0), `2192-2238` (best/early-stop on this loss)
- **What is wrong:** For long classes, 89.6–99.8% of BGCs exceed 32k, so val loss is computed purely on the first ~32 kb prefix and never on continuation past a window, the `|END|` marker, or coherent chaining. Decisively, `|END|` is appended only to the *last* window — so for long BGCs the first-window val loss **never includes the termination token**. The generation eval that would test this is offline and has never run.
- **Why it matters for the goal:** The goal needs complete, domain-complete, cleanly-terminated BGCs (criteria 2/4), especially for drug-relevant long classes. A model can minimize first-window loss while being unable to terminate or chain, and best/early-stop will select it.
- **How to verify:** Confirm `first_window_only` and that best/early-stop read this loss; note no in-loop generation metric. After a checkpoint, run chained generation on long classes and check `|END|` rate and antiSMASH class retention on the full output.
- **Suggested fix:** Add an offline/periodic generation-based selection signal for long classes (EOS-recall on final windows, multi-window full-sequence loss) and gate best/early-stop on it — not first-window loss alone.

---

#### M7 — Chained long-BGC generation has a train/inference seam mismatch [MAJOR / MEDIUM]

- **Dimension:** Scientific-reasoning gaps
- **Location:** Training `scripts/finetune_evo2_lora.py:919-931`; inference `scripts/generate_bgc.py:132-145`
- **What is wrong:** In training, an interior window is `|CONTINUATION:cls|tax` immediately followed by `seq[nt_start:nt_end]` (the whole sub, including the overlap, is supervised) — the model is trained to emit the interior chunk **from scratch**. At inference, `generate_bgc.py` prepends the model's own 2048nt generated tail as context after the continuation prefix and continues. The "continuation-prefix + 2048nt real context → continue" layout was never seen in training, risking boundary ORF truncation and loss of module-order coherence across seams.
- **Why it matters for the goal:** Long NRPS/PKS BGCs can only be produced via this chaining. Incoherent seams → antiSMASH class failure (1), broken domain order (2), non-synthesizable output (4).
- **How to verify:** Inspect interior-window construction (no seed) vs `generate_bgc.py:133-135` (seed = `full[-overlap:]`). After a checkpoint, run `--max-windows>1` and check boundary ORF truncation / class retention.
- **Suggested fix:** Make training match inference — train interior windows with the overlap GIVEN and masked from loss (supervise only post-overlap content); or generate long BGCs in a single ~262k-context pass to remove the seam. Validate seam coherence before wet-lab use.
- **Verifier note:** Strike the "verbatim duplication" sub-claim — vortex returns only generated tokens, so the seed is not re-appended; the load-bearing failure modes are boundary truncation and module incoherence. Default `--max-windows=1`, so the primary single-window results are unaffected.

---

#### M8 — Novelty check compares only against curated 18K, ignoring 262K dropped records + Evo2 pretraining [MAJOR / HIGH]

- **Dimension:** Scientific-reasoning gaps
- **Location:** `scripts/memorization_check.py:33` (`DEFAULT_TRAIN = splits_curated/train.jsonl`), `127-144`
- **What is wrong:** Two compounding gaps: (1) scope — the index is only the 18,270 curated records; a generated BGC could be a near-copy of one of the 262,178 dropped grouped records or any genome in Evo2's pretraining corpus and be reported novel. (2) method — the top-3 Jaccard shortlist (same defect as C1/M11) can miss a long-BGC neighbor a short generation is a fragment of.
- **Why it matters for the goal:** Criterion 3 + wet-lab go/no-go. With only ~1,350 LoRA steps the model is more likely to echo its strong pretraining prior, which this check cannot detect.
- **How to verify:** Confirm `DEFAULT_TRAIN` = curated subset; re-run a held-out sequence against the full grouped set and observe containment rise.
- **Suggested fix:** Default the index to `splits_combined_grouped/train.jsonl` (+ val/test) rather than the 18K subset; use a containment-oriented sketch (sourmash containment index); document that novelty-vs-pretraining cannot be fully ruled out.

---

#### M9 — Conditioning-adherence is the only controllability evidence and is a likelihood proxy, not a causal generation test [MAJOR / MEDIUM]

- **Dimension:** Scientific-reasoning gaps
- **Location:** `scripts/eval_conditioning_adherence.py:113-135`; no class-swap generation control anywhere
- **What is wrong:** M9 scores a fixed real held-out sequence's loglik under each class prefix and checks the true class ranks highest — it measures whether the prefix shifts likelihood over an already-correct sequence, not whether GENERATING under class A yields antiSMASH-class-A output. High M9 top-1 can coexist with generations that ignore the class token (real sequences are self-consistent; the base already has genomic grammar).
- **Why it matters for the goal:** The core Phase-1 claim is CONTROLLABLE conditioned generation. Claiming it from M9 alone is an unsupported leap. (Closely related to M2 — both call for the same generate-then-classify confusion matrix.)
- **How to verify:** `score_model` scores `rec['sequence']` (real), never generates; no generate-then-classify swap arm exists.
- **Suggested fix:** Add the generation-based class-swap control (confusion matrix vs class-frequency prior + scrambled-prefix negative) as the controllability evidence; keep M9 as a cheap supporting signal.

---

#### M10 — E. coli expressibility objective is encoded nowhere; model learns high-GC Streptomyces codon usage [MAJOR / HIGH]

- **Dimension:** Goal alignment & justification
- **Location:** `scripts/finetune_evo2_lora.py:490` (prefix = source taxon, no recoding); `src/bgc_pipeline/evaluation.py:759-801` (M7 expressibility informational, never gates); curated train STREPTOMYCES=4692 (25.7%, ~0.72 GC), ESCHERICHIA in tag=106 (0.58%)
- **What is wrong:** Training reproduces source-organism codon/GC statistics (overall train GC 0.60, Streptomyces 0.72 vs E. coli target 0.508). The M7 fix correctly stops auto-failing non-E.coli sequences but did so by making E. coli expressibility purely informational — so the objective now exists in NO loss term and NO gating metric. *(Overlaps C5; both are the chassis-transfer cluster — track together as AUDIT C4.)*
- **Why it matters for the goal:** A Streptomyces-conditioned ~71% GC NRPS expresses poorly in E. coli (rare codons, synthesis-vendor rejection). The chain from "trained model" to "validated synthesizable BGC" has no link enforcing expressibility.
- **How to verify:** Compute per-taxon GC vs 0.508; grep for codon-optimization (none); read M7 (`chassis_expressibility` never sets `pass`).
- **Suggested fix:** Add a deterministic E. coli codon-optimization/recoding post-step and evaluate THAT output, OR train/condition for E. coli with enough signal; until one exists, scope wet-lab claims to "faithful architecture," not "E. coli-ready."

---

#### M11 — Memorization check can MISS a memorized fragment (Jaccard-ranked top-3 drops the true source) [MAJOR / HIGH]

- **Dimension:** Claimed-fix verification
- **Location:** `scripts/memorization_check.py:127-144` (`nearest`)
- **What is wrong:** Same mechanism as C1, scoped as a conditional false-negative: the bug fires when the memorized fragment's source has ≥3 near-duplicate training BGCs that out-share the query's k-mers. Verifiers found the trigger requires a shared stretch ≥~3.5kb / ≥21% of a 16kb query across ≥3 homologs — realistic for over-represented PKS/NRPS genera (Streptomyces/Pseudomonas; antiSMASH DB 17.7% shared accessions). When it fires, true containment 1.0 is reported as ~0.15–0.40, `memorized=False`.
- **Why it matters for the goal:** This is the only nucleotide-level anti-memorization gate; the false-negative regime is exactly the drug-relevant long classes (40–262 kb).
- **How to verify:** Long source + ≥3 decoys sharing a ≥3.5kb stretch of an exact-substring query → observe `max_containment` far below 1.0 with the wrong `nearest_accession`.
- **Suggested fix:** Rank by an asymmetric containment estimator or raw shared-k-mer count (not size-penalized Jaccard); raise `top_m`; refine containment for all candidates above a low Jaccard floor; add a regression test for the short-query-in-long-ref-with-near-dup-decoys case. *(Fix once; resolves C1 and M8's method leg too.)*

---

#### M12 — C4 fix (taxon-faithful M7) internally correct but M7 is permanently `no_verdict` in the run path [MAJOR / HIGH]

- **Dimension:** Claimed-fix verification
- **Location:** `scripts/evaluate_bgc.py:139-145,156,162` vs `src/bgc_pipeline/evaluation.py:759-801,1019-1031`
- **What is wrong:** Duplicate of M1 from the fix-verification dimension: the C4 rewrite is correct, profiles exist, but the only driver never loads `taxon_profiles` or passes `expected_taxon`, so M7's faithfulness verdict is never computed end-to-end. *(Consolidate with M1.)*
- **Why it matters for the goal:** Every real eval run leaves M7 ungraded; a reviewer trusting the resolution log believes faithfulness is graded when it is not.
- **How to verify / fix:** Same as M1.

---

#### M13 — Nucleotide novelty is not integrated into the eval suite's PASS/FAIL summary [MAJOR / HIGH]

- **Dimension:** Claimed-fix verification
- **Location:** `src/bgc_pipeline/evaluation.py:666-719` (M6 stub); `scripts/evaluate_bgc.py` (never calls `memorization_check`)
- **What is wrong:** The fix-verification view of C6: `memorization_check.py` is standalone and never contributes to the suite summary; M6 ends at a stub with no verdict; there is no nucleotide-identity-vs-train metric inside `evaluation.py`. So a memorized sequence passes the integrated 8-metric suite unflagged. *(Consolidate with C6.)*
- **Why it matters for the goal:** The integrated eval the team will actually run has zero novelty gating.
- **How to verify / fix:** Same as C6; also fix the underlying C1/M11 containment-miss first.

---

#### M14 — Wall-clock/epoch budget is ~6–7× the documented estimate (~18-day ceiling on a contended shared GPU) [MAJOR / HIGH]

- **Dimension:** Goal alignment & justification
- **Location:** live `train_log.jsonl` (~1,140–1,157 s/step steady-state); `schedule.json` (total_steps=1350, 225/epoch, max_epochs=6); `STATE_AND_AUDIT.md:954,1114` claims "~2.7 days at L=32768"
- **What is wrong:** Measured steady-state is ~2,600 tok/s, ~1,155 s/step → ~3.0 days/epoch, ~18 days for the 6-epoch ceiling (~6 days for 2 epochs). The doc's "~2.7 days" is wrong on two axes: it computed steps for a 2-epoch run over the full 277K set (~4,332 steps) while the live run uses curated 18K at 225 steps/epoch × 6 epochs, AND the doc table contains a ~24× hour/day units error (4,332 × 21.7 min ≈ 65 days, mislabeled "~65 hours / ~2.7 days"). Per-step time itself matches reality.
- **Why it matters for the goal:** On an unreserved shared H100 with documented preemption risk, a 6–7× planning error affects whether a usable checkpoint is reached before preemption and whether early stopping has room to act on a near-flat val curve.
- **How to verify:** Per-step deltas from `train_log.jsonl` × 1,350 steps → ~18 days; compare to "~2.7 days."
- **Suggested fix:** Recompute from `schedule.json` × measured s/step; correct the docs to ~3 days/epoch / ~18-day ceiling; reconcile the 6-epoch config with the early-stop horizon; consider lowering `max_epochs` or documenting the preemption-resume budget.

---

### MINOR

---

#### m1 — `find_latest_checkpoint` picks resume checkpoint by mtime, not step number [MINOR / MEDIUM]

- **Dimension:** Correctness bugs
- **Location:** `scripts/queue_h100_production.sh:134-140`
- **What is wrong:** `ls -1dt "$root"/step_*` sorts by mtime; a `touch`/copy/rsync/restore can make an earlier-step dir newer, causing auto-resume to silently resume from a LOWER step. Reproduced. The A4 LR-schedule guard only fires when total_steps changes (it does not here), so it would not warn.
- **Why it matters for the goal:** The C6 auto-resume loop calls this on every preemption of a multi-day run. *No in-codebase path triggers the mtime inversion* (autonomous resume keeps mtime↔step aligned; loads are read-only); requires an operator action. Worst case is lost compute, not a corrupted model — resume restores LR/optimizer/data-position from the chosen checkpoint's internal state.
- **How to verify:** Create `step_100/adapter` and `step_200/adapter`, `touch step_100`, run `find_latest_checkpoint` → returns step_100.
- **Suggested fix:** Sort by numeric step parsed from the dir name (matching the existing Python `cleanup_old_checkpoints` logic), taking max step with an `adapter/` subdir.

---

#### m2 — ~31% species-level overlap between curated train and val/test (grouping is by genome, not species) [MINOR / HIGH]

- **Dimension:** Data gaps
- **Location:** `split_dataset_grouped.py:60-71` (`group_field=genome_accession`); val species also in train 1012/3240 (31.2%), test 30.5%, genus 44.6%
- **What is wrong:** Different genomes of the same species land in different splits, so ~31% of val/test species and ~45% of genera also appear in train. The script honestly reports this as accepted "~48% residual species overlap," but the eval interpretation does not surface it. Record-weighted, ~50% of val records share a species with train.
- **Why it matters for the goal:** Criterion 3 generalization. Val/test loss and adherence look better than true novel-taxon performance, biasing model selection and any "generalizes to new organisms" claim. Does NOT invalidate the leakage fix (genome/exact-seq leak = 0).
- **How to verify:** Intersect `S__`/`G__` token sets between curated train and val/test.
- **Suggested fix:** Add a novel-vs-seen-species (and genus) breakdown to `eval_conditioning_adherence.py` and val-loss reporting; optionally offer a species/ANI-cluster grouping mode for headline novelty claims.

---

#### m3 — Default generation settings cannot produce a complete long-class BGC (16K window vs 44–75K medians) [MINOR / HIGH]

- **Dimension:** Goal alignment & justification
- **Location:** `scripts/generate_bgc.py:181-184` (`--max-new-tokens` default 16384, `--max-windows` default 1)
- **What is wrong:** Default = one 16,384 nt window with `max_windows=1`. Drug-relevant long classes have medians 44–75 kb (97–99.7% exceed 16k), so a default run truncates them mid-cluster; chaining only engages when `max_windows>1`. The script's own documented eval example omits `--max-windows`, reproducing the truncation.
- **Why it matters for the goal:** A 16k NRPS/PKS fragment is missing most assembly-line domains and silently fails antiSMASH class (1) and domain (2) checks — making the most important classes look like model failures rather than config failures.
- **How to verify:** Read defaults; compare 16384 to per-class medians.
- **Suggested fix:** Make generation length class-aware (or loop to EOS up to a high cap); warn/refuse to emit a non-EOS-terminated long-class sequence; fix the docstring example to include `--max-windows`.
- **Verifier note:** One verifier argued Major (every default long-class generation is truncated); kept Minor since chaining exists and it is fixable per-run.

---

## Weakest Link

**The novelty / anti-memorization safety gate is the single weakest link**, because four independent, confirmed defects converge on it and it sits directly before the most expensive, irreversible decision (committing wet-lab synthesis budget):

1. **C6/M13** — it is not in the scored eval suite at all (M6 is a non-parsing stub; the memorization tool is orphaned), so the PASS/FAIL summary structurally cannot fail a memorized sequence.
2. **C1/M11** — even run standalone, it ranks candidates by Jaccard and misses the memorized-fragment-of-a-longer-BGC case, which is the *common* case at the default 16k generation length.
3. **C3** — its threshold is uncalibrated; real held-out BGCs already score 0.95–0.996, so it cannot separate novel from memorized.
4. **M8** — it compares only against 18K of ~346K known BGCs and never against Evo2's pretraining corpus.

Because novelty is one of the five non-negotiable criteria and the gate before synthesis spend, a model that regurgitates a training/leaked BGC (made *more* likely by the residual near-duplicate leakage in C2/C7 and the short ~1,350-step fine-tune) would be reported as a novel success. Until this gate is wired in, made containment-correct, calibrated, and scoped to the full known-BGC corpus, **the headline novelty claim is unfalsifiable as instrumented** — and the parallel chassis-transfer gap (C4/C5/M10) means even a genuinely novel output is not demonstrably E. coli-expressible.

---

## Disputed Findings

- **Novelty index built against curated 18K, not full corpus** — `memorization_check.py:33`. *Major.* One verifier confirms (lets a near-copy of a known-but-dropped BGC pass); one refutes as overstated (for the *memorization* question the curated set is the correct reference; the broader-novelty gap belongs to M6/M8, already tracked). *(Overlaps M8.)*
- **Generation `top_k=4` may prevent emitting `|END|` / N** — `generate_bgc.py:179`. *Major.* Mechanism (char-level vocab, hard top-k, `stop_at_eos=False`) is confirmed and the EOS contract has never run, but whether `|` actually fails to reach top-4 at the *terminal* position (where it is supervised to be argmax) is unproven — uncertain vs refuted. Verify by measuring `hit_eos` rate on a checkpoint; consider top-p or allow-listing the EOS tokens.
- **Hard long classes get the same ~1000 class-start examples as trivial classes** — `curate_dataset.py:293`. *Major.* Window-count facts confirmed; refuted as stated because the central claim that CONTINUATION windows are "label-free" is wrong (the prefix carries class+taxon). Residual true kernel (uniform start-example cap discards abundant NRPS/PKS start data) is real and already documented as risk B3.
- **~30% of curated train have no species token (S__NONE); Streptomyces 25.7%** — taxonomy probe. *Major.* Numbers confirmed; refuted as overstated because the faithfulness eval operates at phylum level (strong coverage) and the E. coli sub-claim conflates chassis vs conditioning-target. Residual: weak support for rare phyla.
- **No generation-based or base-model validation in the loop; near-flat loss** — val path + live run. *Major.* Mechanics confirmed; refuted as overstated because the "majority-prior" premise is false for the balanced train set and the generate-then-classify + `--compare-base` infrastructure already exists (just unrun). Residual: run those evals before claiming Phase-1 success.
- **`extract_sequence` truncates at first internal `|` while reporting `hit_eos=True`** — `generate_bgc.py:63-79`. *Minor.* Reproduced exactly; one confirms (Minor), one refutes for goal impact (off-distribution, rare, self-flags via `trailing_junk_trimmed`). Real residual: premature chaining halt in multi-window mode.

---

## Refuted (considered & dismissed)

- **Chained continuation re-predicts an overlap the model was trained to GENERATE but inference supplies as context** — `generate_bgc.py:130-145`. Both verifiers refute: teacher forcing co-trains BOTH "predict the overlap from the prefix" AND "continue conditioned on the full overlap"; the inference seam reconstructs a genuinely supervised training context (windows tile with `stride = budget − overlap`), so there is no regime gap. The only adjacent real issue is generic exposure bias, distinct from the named mechanism. *(Note: M7 captures a related but real seam concern — the inference layout "continuation-prefix + 2048nt real context" — which this refuted finding mis-framed.)*

---

## Verified Sound

**Correctness bugs**
- Prefix loss masking correct (`collate_pad` sets prefix positions to `IGNORE_INDEX`; shift makes last prefix logit predict first nucleotide; no off-by-one).
- EOS placement correct (`|END|` only on final window, after the prefix so supervised; `eos_reserve` budgets room; `tests/test_chunk_eos_windows.py` passes).
- Continuation prefix (M11 fix) correctly applied; B1 seam guard raises on a merging tokenizer.
- Mid-epoch resume micro_step alignment correct (snap-down to grad-accum boundary; OOM rollback traced).
- Skip-ahead data-stream faithfulness correct (seeded `DistributedSampler` + `set_epoch`; guarded finally-save).
- Checkpoint rotation correct (keeps newest numeric + special, always preserves `best/`, refuses `keep_last<=0`).
- `data_fingerprint` fix real (full-file streamed sha256 + compare-on-resume with loud warning).
- Group-aware split genuinely leakage-free at genome + exact-md5 level (0 overlap measured on grouped and curated splits).
- vortex/evo2 generation contract handled correctly (returns generated tokens only; `stop_at_eos=False`; trim at literal `|END|`).
- `sequence_loglik` correct (scores only sequence positions; returns sum+count for length-normalization).
- `make_positive_control` md5 disjointness guard real; RNG/determinism handling honest.

**Scientific / methodological soundness**
- Genome + exact-sequence leakage genuinely eliminated in `splits_curated` (0 measured).
- Train vs generation conditioning prefix byte-identical.
- Prefix loss-masking + chunk seam invariant guarded (B1 assertion).
- Interior-window continuation (M11) real; EOS only on final window.
- Conditioning-adherence likelihood classifier methodologically sound as a *discriminative* test (length-fair, balanced, random baseline, base comparison).
- M7 no longer auto-fails non-E.coli at the library level (faithfulness-driven; `no_verdict` without a profile).
- First-window-only, length-stratified validation (M2 fix) real and active.
- Dependency pinning (C5) + GPU-free test suite real.

**Data gaps**
- Leakage fix real at genome/exact-seq level; quality filtering verified (0 N, 0 contig-edge in curated train).
- The 14 well-represented classes hit the cap AND retain genuine genus diversity (NRPS 223 genera, PKS 266, etc.) — not 1000 near-duplicate Streptomyces.
- Curation preserves leakage-freedom by operating within disjoint splits; force-keeps MIBiG gold; val/test kept full.
- `tests/test_data_pipeline.py` asserts disjointness/dedup/cap/quality on a fixture.

**Scientific-reasoning gaps**
- LoRA choice justified by the 80GB memory constraint (full FT needs ~84GB; math checks out).
- Prefix masking + EOS-after-prefix correct; M11 distinct continuation prefix sound; seam-guard real.
- Group-aware split structurally valid; EOS-only-on-final + `eos_reserve` internally consistent.
- DeepSpeed grad-accum idiom correct (`step()` every micro-step, optimizer steps at boundaries).
- Dependency pinning real; positive-control design sound.

**Goal alignment & justification**
- Training-side conditioning machinery well-aligned and live; group-aware splitter fixes bulk antiSMASH leakage (0 GCF_* overlap).
- Curation preserves drug-relevant long classes at cap; M7 chassis fix real at metric level; first-window length-stratified val + early stopping active; M9 adherence design sound; `generate_bgc.py` exists and is consistent with training; production launch used the correct OOM-safe curated config.

**Claimed-fix verification**
- C1/C2 leakage fix, M2 first-window val, M11 EOS+continuation, M1 grad-accum resume, C3 generation script, M9 adherence + `sequence_loglik`, C4 M7 metric rewrite (library), m1 data-fingerprint, m2 determinism, C6 auto-resume, curation claims, and the full GPU-free test suite — all verified real/correct.

---

## Open Questions

These cannot be resolved from code/data alone and the authors must answer them — most require a completed GPU run + offline generation eval (none exists yet per `STATE_AND_AUDIT §7.4`):

- **Does first-window teacher-forced val loss correlate with generation quality** (antiSMASH M1/M2)? Needs a real run + generation eval.
- **Does ~1,350 LoRA steps (r=16) shift generation above the untuned Evo2 base**, or is output the base prior steered by a thin prefix? Needs `--compare-base` delta.
- **Does Evo2's pretraining corpus already contain these antiSMASH/RefSeq genomes?** If so, novelty is fundamentally limited regardless of the train-set memo check.
- **Does `top_k=4` actually block the `|END|` token at the terminal position?** Measure `hit_eos` rate on a checkpoint vs top-p / larger top-k.
- **Does `merge_and_unload` (peft-0.19) correctly merge LoRA deltas into Evo2's non-standard StripedHyena layers** (Wqkv, out_filter_dense, Hyena MLPs)? Could silently no-op/corrupt; untestable without a GPU.
- **Does chained `max_windows>1` generation produce coherent joins?** The continuation path has never run on a real checkpoint (relates to M7).
- **Full-population near-duplicate rate** (val↔train and across all 2,389 MIBiG) — only sampled here; needs an exhaustive all-vs-all containment/identity pass.
- **Is the multi-product → single-label harmonization** (e.g. NRPS+metallophore→SIDEROPHORE vs NRPS+T1PKS→PKS_NRPS_HYBRID) internally consistent, or a source of class-label noise? Depends on unread priority logic in `mibig_record.py` / `antismash_db_to_jsonl.py`.
- **Will early stopping (patience 4 × val-every 50, min-delta 0.001) trigger before the ~18-day ceiling** given the already-flat val curve?
- **Can the model generate full-length 50–150 kb BGCs in a single ~262k-context pass** (avoiding the seam entirely)? The train(≤32k)/inference(262k) length mismatch is unaddressed.
- **Goal-relevance of evaluating only on quality-filtered (N-free, non-contig-edge) val/test**, when real-world inputs include those cases.
- **Run `micromamba run -n bgcmodel python tests/run_all.py`** to confirm the claimed GPU-free test coverage actually passes (could not execute here).

---

## Recommended Priority Actions

1. **Wire a correct, calibrated novelty gate into the scored eval suite (C6/M13 + C1/M11 + C3 + M8).** Add a nucleotide-identity-vs-train metric to `evaluate_bgc()` with a real PASS/FAIL (FAIL ≥0.95, WARN ≥0.8); fix `nearest()` to rank by containment-aware score (not Jaccard top-3); index the full grouped corpus (+ MIBiG), not the 18K subset; calibrate the threshold from the positive-control distribution using alignment identity. Finish or remove the M6 stub.
2. **De-leak the splits against near-duplicates (C2 + C7).** Cluster all records (MIBiG + antiSMASH) by sequence identity (~80–95% over the BGC interval) BEFORE splitting and assign whole clusters to one split; rebuild the positive control with a containment-based disjointness guard; add a pre-train assertion failing on any cross-split pair >0.9 containment; re-validate any checkpoint decisions.
3. **Resolve the chassis-transfer contract honestly (C4 + C5 + M10).** Either add a deterministic E. coli codon-optimization/recoding post-step and evaluate THAT output, or rescope the deliverable to "faithful native-organism architecture." At minimum, restrict wet-lab target prompts to taxa with adequate (class, taxon) support, and re-curate to guarantee E. coli/Enterobacteriaceae coverage for the three target classes.
4. **Add the causal class-conditioning generation control (M2 + M9).** Generate under true vs scrambled/mismatched class prefixes (fixed taxon, matched seeds), run antiSMASH M1, report a requested-vs-recovered confusion matrix; require the diagonal to beat the majority-class prior and base-Evo2 generations.
5. **Wire taxon profiles into the eval driver (M1/M12).** Load `taxon_profiles.json` into `EvalConfig` and pass `expected_taxon` per record so M7 produces a verdict; add an end-to-end assertion.
6. **Make model selection generation-aware and class-balanced (M3 + M6).** Add `val_by_class`; macro-average/class-weight the early-stop signal; add a periodic generation-based eval (EOS-recall + antiSMASH class retention on full chained output for long classes) and gate selection on it.
7. **Fix the train/inference seam and long-generation defaults (M7 + m3).** Train interior windows with the overlap given-and-masked (or generate in a single long-context pass); make generation length class-aware so long classes are not truncated by default; warn on non-EOS-terminated long-class output.
8. **Correct the wall-clock budget and chassis/data docs (M14 + M4 + M5).** Recompute to ~3 days/epoch / ~18-day ceiling; reconcile the epoch ceiling with the preemption window; scope/footnote low-count classes (n<~100 train, n<~20 eval) and the curated E. coli counts so downstream claims are honest.
