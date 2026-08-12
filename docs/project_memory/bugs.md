# Bugs & quirks — and the proven fixes

Recurring errors, environment/tooling quirks, and what actually fixed them. Add an entry
whenever a non-obvious bug is solved. See [decisions.md](decisions.md) for rationale and
[progress.md](progress.md) for current state.

---

## Analysis / tooling

- **[2026-08-12] Running training arms concurrently on one GPU bought NOTHING — without MPS, CUDA
  contexts time-slice.** The 1B at `batch=1` reaches only ~75 TFLOPS of an H100's ~756 (**10% of
  peak**): a 1.1B model on 8 k tokens with no batch dimension is launch-latency and HBM-bandwidth
  bound and never fills the SMs. That headroom is real, and the inference drawn from it — "so a
  second and third process can claim it" — was **wrong**. Measured: baseline alone **8,437 tok/s**;
  the same process with two co-tenants **2,799**, i.e. almost exactly one third, with an aggregate
  of 7,957 = **0.94x** solo. Separate CUDA processes do not run concurrently under
  `Compute Mode: Default` — the driver time-slices whole contexts, so N processes get 1/N each plus
  ~6% switching overhead. `nvidia-smi` "utilization.gpu 100%" never contradicted this: that field is
  *fraction of time a kernel was resident*, not occupancy. **Fix:** reverted to sequential arms
  (`PARALLEL=0`, the default) and kept the free half — `WAIT=1` in `score_arms.sh`, which scores
  each arm the moment its adapter lands, so the first arm's generation overlaps the others'
  *training*. Sequential ordering makes that overlap **better**, not worse. **The real fix for
  concurrent processes is CUDA MPS** (`nvidia-cuda-mps-control -d`), which is not something to
  enable mid-experiment on a host with 35 logged-in users. Cost: ~25 min of GPU time.
  *Rule: idle SM capacity does not mean a second PROCESS can use it. Occupancy and concurrency are
  different questions, and only MPS connects them.*

- **[2026-08-12] Polling for `final_adapter/` (or for `adapter_config.json`) fires on a
  half-written adapter.** The trainer publishes with `shutil.copytree`, which creates the directory
  first and copies in directory order — so `adapter_config.json` lands **before**
  `adapter_model.safetensors` finishes streaming. A watcher gated on either would launch generation
  against truncated weights: an arm that still emits sequence, so the failure is a silent wrong
  result rather than a crash. **Fix:** gate on `adapter_model.safetensors` *and* require its size to
  be unchanged across a 15 s re-stat. *Rule: when waiting on a file another process is writing,
  wait on the LAST byte of the LARGEST file, not on the existence of the first one.*

- **[2026-08-12] Determinism is half-enabled, so we pay for it without getting it.**
  `finetune_evo2_lora.py` sets `torch.use_deterministic_algorithms(True, warn_only=True)` and
  `cudnn.benchmark = False`, but `CUBLAS_WORKSPACE_CONFIG` is **not** set in the run environment —
  and torch says so at runtime ("this operation is not deterministic because it uses CuBLAS"). The
  GEMMs that dominate runtime are nondeterministic regardless, while `benchmark = False` gives up
  kernel autotuning. NOT changed mid-experiment (it would be a second difference between arms).
  *Rule: resolve this in one direction or the other before the next training round — either set
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and get real reproducibility, or drop the determinism flags and
  get the speed.*

- **[2026-08-12] An accession-keyed join silently mismatches 12,217 training records.** The
  Phase-2 annotation sidecar was keyed by `accession`, but **5,219 accessions are shared by 12,217
  records with DIFFERENT sequences** (`GCF_043836905.1.region1` appears at 3,603 nt *and* 11,163
  nt). One record's domain spans and codon phases would have been attached to another record's DNA
  — confidently wrong weights, plausible-looking output, undetectable downstream. **Fix:** key by
  **row index**; the sidecar emits `row`, and the trainer cross-checks that the sidecar row names
  the same accession as the training record (so a sidecar built from a different split raises
  instead of mislabelling everything). *Rule: before using a field as a join key, count its
  duplicates.*

- **[2026-08-12] `truncate` is the trainer's default, and it biased the experiment against its own
  hypothesis.** Left unexamined, the Phase-2 arms trained on the first ~4 kb of every core: 25.2%
  of the training DNA, and a slice that is **49.0% class-domain against the true 33.7%**. The
  domain-weighted arm would therefore have had *less linker to down-weight than reality contains*
  — a bias against the intervention being tested. It also pins `nt_start` at 0, so the
  window-offset code in `annotations.py` never runs outside its unit test. **Fix:**
  `--long-seq-strategy chunk --chunk-overlap 1024`, giving 95,759 windows over all 467 Mbp at the
  true coverage, with `first_nt_start` verified non-zero (3,380 / 6,760 / 20,892) in a smoke run.
  *Rule: a default is a decision you did not make; check what it does to YOUR hypothesis.*

- **[2026-08-12] A substrate sanity check that passed a model which was at chance — three separate
  errors, each a pattern this project repeats.**
  1. **Hand-written test data.** The check scored an invented ORF and returned 1.3843 against a
     1.386 uniform threshold — passing a model that was guessing, because invented DNA is
     out-of-distribution for *every* model. *Test on real data.*
  2. **Threshold from a different protocol.** A no-context measurement (1.25) was compared against
     a with-context reference (0.99). *The same quantity measured two ways is two quantities.*
  3. **n = 1.** It thresholded on a single core, where variance is ~0.25 nats — larger than the gap
     between a healthy model and a dead one. *n=1 cannot support a threshold.*
  Final form: 500 bases after 2,000 nt of real context, averaged over 8 real cores, healthy < 1.15.

- **[2026-08-12] `deepspeed` is not on PATH under `nohup`.** A driver calling `deepspeed` bare
  died with exit 127 *after* printing its START line, so the log looked like a run in progress.
  **Fix:** `micromamba run -n bgcmodel deepspeed ...` inside the script. *Rule: a launcher that
  prints START before the first real work can report a failure as progress.*

- **[2026-08-12] Transformer Engine: the 1B needs it, and the version matters.** TE 2.18 will not
  build against torch 2.5.1 (`SymmetricMemory.hpp: No such file` — a header added in a later
  torch). **TE 1.13.0** is contemporary with torch 2.5, builds, does not upgrade torch, and leaves
  the 7B pipeline working (verified after install). Pin the **cu12** runtime; the resolver
  otherwise pulls cu13 against a cu124 torch. Without TE the 1B loads and is *at chance* — see
  decisions.md 2026-08-12.


- **[2026-08-12] ~~`module_architecture` is broken~~ — RETRACTED SAME DAY. It works; it has a
  LENGTH FLOOR, and my diagnosis blamed the wrong component twice.** The first claim was that the
  check never fires and that pyrodigal was fragmenting megasynthase genes. Both are wrong:
  * **pyrodigal is not fragmenting.** Its longest ORF matches the raw six-frame longest stop-free
    stretch on real cores at ratio **1.00–1.04** — the DNA genuinely contains stops. Some real
    cores do carry a full megasynthase (one NRPS core has a **5,951 aa** ORF), others top out
    at 561–920 aa.
  * **The check is correct.** On the core with the 5,951 aa megasynthase it finds 5 condensation,
    5 adenylation and 5 carrier domains and reports **module_count = 5, ordered = 5**.
  * **The zeros were a length artefact of my own measurement.** Everything in the ladder audit was
    truncated to 3 kb, which holds at most ~1,000 aa, while one module needs ~1,000–1,500 aa. A
    module cannot fit in the window. Measured on real ≥20 kb NRPS/PKS cores:

    | window | mean modules | % with ≥1 module |
    |---|---|---|
    | 3,000 nt | 0.25 | 25% |
    | 6,000 nt | 0.50 | 40% |
    | 12,000 nt | 0.80 | 40% |
    | 20,000 nt | 1.25 | 40% |

  **Correct usage:** a DIAGNOSTIC at ≥6 kb only, never at 3 kb, and never a gate — its own ceiling
  on real DNA is 0.40 even at 20 kb. Its AUROC of 0.500 in the ladder audit is right and
  uninformative: it was constant-zero across a 3 kb cohort. *Same family as the antiSMASH hybrid
  ceiling (0.00 at 3 kb): the instrument cannot fire in the window, which is not the same as the
  instrument being broken.* **Rule: before calling an instrument broken, check it at the length it
  was designed for.**

- **[2026-08-12] Reading a metric out of a result dict by a key that does not exist returns 0 and
  looks like a measurement — TWICE in one day.** `soft_instrument_probe.py` read `any_pfam_hit` /
  `n_pfam_hits` (neither exists; the real field is `markers_present`, a LIST) and reported
  `any Pfam = 0.000` for REAL cores. `ladder_audit.py` read `complete_modules` / `n_modules` (real
  fields: `module_count` / `ordered_module_count`) and reported 0 everywhere. Both were caught only
  because the REAL-data row was implausible. **Rule: when a new column reads 0 for the positive
  control, suspect the key before the biology** — and prefer `d["k"]` over `d.get("k")` for fields
  that must exist, so a rename raises instead of silently zeroing.


- **[2026-08-11] A missing input rendered as a null result: `direction_audit.py` pointed at a
  directions file that had only 2 of the 9 layers.** `train500.steerdirs.npz` carries L16/L20;
  `trainonly.steerdirs.npz` carries all nine (10,12,14,16,18,20,22,24,27) with class-units. Asking
  for L27 skipped every class, then printed a full set of empty tables ending in the verdict
  "0/0 classes landed" — which reads exactly like "the edit never landed". **Fix:** default to
  `trainonly.steerdirs.npz`, and `SystemExit` when a requested layer yields zero usable classes.
  The same fail-loud rule as `BGC_EVAL_STRICT`: a missing resource must never become a silent
  negative. Note the `.report.json` sidecar records `acts`/`n`/`stripped`/`prefix_index` — check
  it to confirm a directions file matches the activations you are scoring against.

- **[2026-08-11] `git checkout <file>` silently does nothing for an UNTRACKED file, so mutation
  tests accumulated instead of reverting.** Mutation-verifying a brand-new (uncommitted) script,
  each `sed`-then-`git checkout` cycle left the previous mutation in place; every arm reported
  "PASS 0" and looked like a clean kill when the file was in fact four mutations deep. **Fix:**
  `cp` the file to the scratchpad first and restore from that copy, and assert the baseline
  PASS count both before and after the sweep. A mutation sweep that never re-verifies the
  restored baseline cannot tell a real kill from a corrupted file.

- **[2026-08-11] A test asserting sklearn's multinomial coefficients are un-centred cannot fail.**
  `LogisticRegression` with the multinomial solver returns coefficients already centred across
  classes (measured max `|coef_.mean(axis=0)|` = 9e-16), so subtracting the other-class mean is a
  no-op and an assertion against a fitted pipeline passes whether or not the code does it. **Fix:**
  exercise the contract with a stub carrying deliberately un-centred coefficients. (The audit's
  reported angles are unaffected — subtracting ~1e-16 changes nothing.)

## Evo2 / vortex / generation

- **[2026-08-10] Training a prompt: AdamW's step is per-COORDINATE, so the update VECTOR is
  `lr*sqrt(D)`.** At D=4096 an lr of 0.05 moves the prefix by 3.2 — against a token-embedding
  norm of 1.45 that is **221% of the prefix's own length every step**. Measured: the prefix left
  the readable region during warmup, val went 0.884 → 1.404 by step 50, and the gradient norm
  collapsed 0.52 → 0.005 as it settled somewhere flat. It looked exactly like training (loss
  printed, steps ticked, a `prefix_best.pt` was written) and produced a dead vector.
  *Fix:* lr 1e-3 (~4% of ||e|| per step), plus two guards in `train_soft_prefix.py` — a startup
  check that prints the step size AS A FRACTION of ||e|| and refuses anything over 25%, and an
  early abort when val exceeds 1.25x baseline. *Rule:* quote a learning rate in units of the
  thing being updated, never in the abstract.
- **[2026-08-10] The inference loader's weights are INFERENCE TENSORS and cannot be backwarded
  through.** `load_evo2_wrapper_for_inference` merges the LoRA under `no_grad`, so any training
  on top of it dies with "Inference tensors cannot be saved for backward" — even when the weights
  are frozen and only a new parameter needs a gradient, because autograd still routes through
  them. *Fix:* re-materialise every parameter (and inference-mode buffer) with
  `.detach().clone()` inside `with torch.inference_mode(False):` after loading.
- **[2026-08-10] A dropped join key makes a paired analysis print NOTHING, which reads as "no
  data" rather than "bug".** `probe_score_generations.py` rebuilt records with an explicit key
  list and did not carry `tax_idx`, so the soft-prefix driver's paired table — joining on it —
  found zero pairs and printed an empty section under a populated matrix. *Fix:* carry every
  join key a caller might pair on (`tax_idx`, `seed_acc`) and say so in the code.
- **[2026-08-10] Pooling non-independent comparisons inflates n and the significance with it.**
  Comparing prefix_X against three control arms and pooling gives "36 pairs" from 12 generations
  used three times. TERPENE read p=0.0288 pooled and p=0.146 with the taxon as the independent
  unit. *Rule:* the independent unit is the item, not the comparison. Also report the MEDIAN —
  TERPENE's mean of +0.173 came from 2 of 12 sequences and its median was +0.012.

- **[2026-08-10] A mean-POOLED activation norm is not the norm a generated token sees — and the
  error grows with depth.** The activation cache stores mean-pooled hidden states, so ‖h‖ read
  from it disagrees with the live per-position ‖h‖ at the steering hook by 0.75x at L16 but
  **2.84x at L27** (pooled 11.25 vs live 31.97). Pooling averages vectors pointing in different
  directions and shrinks the norm by a depth-dependent factor. Any dose derived from the cache is
  therefore mis-scaled, and a cross-layer comparison built on it is confounded by exactly the
  quantity under test. This is the same failure family as the retired `_ref_norm` bug (which read
  `X[:, -1, :]`, the pooled vector, and made every alpha 1.5–5.9x the between-sample scatter).
  *Fix:* `seed_generate.py --steer-norm-frac` recomputes ‖delta‖ = frac × ‖h‖ from each generated
  position's own residual, and every record persists the **realized** ‖h‖ / ‖delta‖ / dose
  (`steer_mean_h_norm`, `steer_realized_norm_frac`, `steer_realized_class_units`) so no analysis
  ever has to re-derive a dose from stderr again — which the β-titration had to.
  Guarded by `tests/test_steer_hooks.py`.
- **[2026-08-10] Three "looks like a measurement, is an artefact of my own code" bugs, all found
  by reading a table that was about to be reported.** Same family; worth recognising on sight.
  1. *Comparing against `None` and printing the result as a rate.* The unsteered control arm has
     no `steer_target_class` by construction, so `argmax == target` compared to `None`, was False
     every time, and printed **0.000** — indistinguishable from "unsteered never hits the target",
     which is exactly the claim the column appeared to support. *Fix:* print `--` when an arm has
     no target; store the FULL probability vector so a baseline can be scored against the STEERED
     record's class.
  2. *A pairing predicate that silently drops every pair.* The same `None` also made the paired
     join require `target == None`, so all 48 pairs were dropped and the table read "too few
     matched pairs" — a null presented as a shortage of data.
  3. *Inferring a bucket instead of tagging it.* `marker_sensitivity.py` selected its
     "full-length" rows with `length == nt`; a core LONGER than the truncation has
     `nt == length == 3000`, so 29 truncated rows leaked into the full-length bucket and inflated
     it from 60 to 89. *Fix:* tag the bucket explicitly at write time, never re-derive it.
  **The pattern:** each produced a plausible number rather than an error. Guard by asking of every
  printed rate "what is the denominator, and can this cell be produced by a missing value?"

- **[2026-08-10] A z-score against 3 permutation controls is not a z-score.** `steer_reach.py`
  reported z = 16.5 at L24 — from a control sd of 0.00003 estimated off three points, on an
  effect of 0.00017. With few controls the sd is mostly noise, so a small spread manufactures a
  huge z. Same family as the retired "beat the max of the controls" rule, which got *stricter*
  as controls were added. *Fix:* suppress z below 5 controls and quote the permutation p, which
  is honest but floored at 1/(n+1).

- **vortex silently de-batches mixed-length prompts.** `Evo2.generate(..., batched=True)`
  checks `uniform_lengths` and falls back to per-sequence generation for ragged batches.
  *Fix:* generate **sequentially** (`generate_bgc.py` default). Left-padding to equalize
  lengths was tried and **failed an on-GPU equivalence gate** (left-pad perturbs
  StripedHyena: head-token LCP ~0.004, byte divergence). Keep batched behind that gate.
- **`eos_id = 0` is the null byte** in `CharLevelTokenizer` — unusable as a stop token;
  `stop_at_eos` is effectively off, so generation always runs the full `n_tokens`.
- **vortex `Generator.generate()` can't be resumed one token at a time** for stepwise
  decoding (e.g. CFG). On a resumed call (passing `inference_params_dict`) it derives
  `seqlen_offset` from the *passed input's* length (`input.shape[-1]`, `generation.py:176`) —
  which is just the single new token (=1), **not** the accumulated context — so positions get
  corrupted. Within a single multi-token `generate()` call it's fine (input = full prompt).
  *Fix (used by `evo2/scripts/cfg_generate.py`):* drive `model(x, inference_params_dict=…)` directly
  and manage `seqlen_offset` yourself — set it to `prompt_length` after prefill, then `+=1` per
  token (mirrors `generation.py:168-192`). Good news for logit-level work: `generate()` DOES
  return per-token `scores` (logits, shape `(B,T,vocab)`) and accepts/returns
  `inference_params_dict`. Validate any hand-rolled cached loop against a **non-cached
  full-recompute** greedy (the w=1 gate).
- **Generation needs the free GPU.** It will not share the device with active training;
  pause/finish training first.

## Eval suite

- **[2026-08-11] THE 3 kb GENERATION LENGTH IS A CEILING, and for hybrids it is a ceiling of ZERO.**
  Nearly every generation experiment ran at 2-3 kb. Measured directly (`scripts/length_ceiling.py`)
  by truncating REAL held-out cores to the lengths we generate and running the same antiSMASH gate
  — a real BGC is the best case, so this is the ceiling no generation can beat:

  | correct_class at 3 kb | POPULATION (n=25/class) | long-tail (cores ≥12 kb) | full length |
|---|---|---|---|
| NRPS | **0.76** | 0.25 | 0.84 |
| **PKS** | **0.40** | 0.33 | **0.96** |
| **PKS_NRPS_HYBRID** | **0.00** | **0.00** | 0.96 |
| TERPENE | 0.88 | 0.75 | 0.96 |
| RIPP | 0.76 | 0.67 | 0.92 |
| POOLED | 0.56 | 0.40 | 0.93 |

**Use the POPULATION column** — it samples each class at its natural length distribution, which
is what a generation is actually competing against. The long-tail column required cores ≥12 kb so
the long columns would be meaningful, and therefore answers only "what does truncation cost a
LONG core". Quoting it as the ceiling overstates the handicap for NRPS by 3x.

Reconciliation with the existing positive control (0.750 pooled at 3 kb): pooled-excluding-hybrids
here is (0.76+0.40+0.88+0.76)/4 = **0.70**. The 0.56 figure is dragged down only by including
hybrids at 0.00. Two independent methods agreeing at ~0.70-0.75.

**The two classes that are genuinely length-limited:**
- **PKS_NRPS_HYBRID — ceiling 0.00 at 1/2/3 kb in BOTH samples.** Structural: antiSMASH calls a
  hybrid only by seeing both machineries, and 3 kb cannot contain both. Every hybrid result at
  3 kb is WITHDRAWN — those arms could not have produced a positive whatever the model did.
- **PKS — 0.40 at 3 kb against 0.96 at full length**, a 2.4x compression, the largest for any
  non-hybrid class. Median PKS core is 9 kb in this sample, so most are heavily truncated.

NRPS (0.76), TERPENE (0.88) and RIPP (0.76) are only mildly affected: their cores are short enough
that 3 kb captures most of a typical one.

  *What this does NOT overturn:* paired, internally-controlled comparisons (real vs shuffled-label
  direction; guided vs random selection; prefix_X vs prefix_Y) share the ceiling, so it cancels in
  the contrast. Phase 1 never used antiSMASH at all. *What it does change:* every ABSOLUTE rate
  quoted at 3 kb was against a ceiling of 0.40-0.75, never 1.0.

- **[2026-08-11] Several arms were doubly underpowered: a compressed ceiling AND an n that could
  only see large effects.** Binomial power against the ~2% cross-class base rate, 80% power:

  | experiment | n/arm | smallest detectable rate |
  |---|---|---|
  | Phase 3 steering (pooled) | 140 | 6.5% |
  | Phase 3 steering (per dose) | 48 | 11.2% |
  | L27 ladder / multi-layer / soft prefix | **12** | **23%** |

  An n=12 arm reads 0/12 whether the true effect is 0% or 15%. "0/12 in every arm" therefore
  means *no LARGE effect*, not *no effect* — the L27 and multi-layer writeups lean on it harder
  than the n supports. The conclusion survives because the big experiment (Phase 3, n=140) rules
  out >=6.5% and the continuous probe has far more resolution than any binary gate.
  **Direction-estimation n was NOT a problem and was already fixed**: split-half cosine is
  0.97-0.99 at n=500/class (it was 0.67-0.88 at n=10-40, which is what triggered the re-embed).

- **Small-n quick_eval is noisy — `is_bgc` read 0/6 when the true rate was ~14%.** At
  step_1200, a single n=6 quick_eval showed `is_bgc=0.0`; pooling n=21 across two decoding
  temps gave `is_bgc≈3/21 (14%)` (still `correct_class=0/21`). *Lesson:* don't treat one tiny
  quick_eval as ground truth for a headline gate — use `PER_CLASS≥3` and pool (≥15) before
  concluding a gate sits at 0. `quick_eval.sh` now accepts `TEMPERATURE`/`TOP_K`/`TOP_P` env
  overrides (defaults 1.0/4/1.0) for decoding sweeps; conservative decoding (temp 0.7) did NOT
  reveal core-domain structure, so the megasynthase-conditioning failure is structural, not a
  sampling artifact.
- **antiSMASH was ~15% on real BGCs — it was MAP COVERAGE, not parsing.** Parsing of
  `records[].areas[].products` was correct; antiSMASH 8 just emits **103 product types**
  and the old `compound_class_map.yaml` covered a fraction, so real clusters
  (`NRP-metallophore`, `PKS-like`, `NI-siderophore`, …) mapped to OTHER and failed.
  *Fix:* `scripts/build_class_map.py` regenerates the map from antiSMASH's own
  product→category grouping + overrides → ≈0.97.
- **Prodigal calls one long PARTIAL ORF in GC-repeat junk** (`"GC"*8000` → coding_density
  ~1.0, max ORF ~455 aa). So a "require ≥1 complete gene" rule does **not** catch the
  degenerate collapse. *Fix:* `coding_sanity` discriminates junk with a
  **dinucleotide-entropy complexity guard** (`< 2.5 bits` → fail), not completeness.
- **`coding_sanity` false-failed a real NRPS core** that is one 3599-aa megasynthase
  (edge-truncated by strict-core trimming → Prodigal flags it `partial` →
  `complete_gene_fraction = 0`). *Fix (two parts):* (1) `is_bgc` trusts
  `antismash.detected` when antiSMASH ran — coding_sanity is only the floor/proxy; (2)
  coding_sanity dropped the complete-gene requirement (see above).
- **antiSMASH JSON parse error left `class_match` unset** → `derive_questions` used
  antiSMASH for `is_bgc` but the class_markers proxy for `correct_class` (inconsistent,
  possible false PASS). *Fix:* on parse error, mark the antismash result **`skipped`** so
  both questions fall back to the proxy consistently.
- **six-frame ORF finder fragmented megasynthases** → PKS/NRPS detection low. *Fix:*
  replaced with **pyrodigal** everywhere (`find_orfs`); six-frame kept only as a fallback.
- **`class_markers` reported 0 obligate domains on real NRPS/PKS/TERPENE cores.** Not a
  gene-finder or extraction bug — the marker list was textbook-only (missing carotenoid,
  type-III PKS, NRPS-like subtypes). *Fix:* data-driven markers (`derive_class_markers.py`)
  + ANY-of pass semantics. Validated ≈0.87 on real cores.

## Conditioning diagnostics

- **Greedy `top_k=1` generation collapses to all-GC** (gc≈1.0) — degenerate, useless for
  a conditioning signal. Use stochastic sampling.
- **12-mer Jaccard saturated** (within ≈ cross ≈ 1.0) — wrong metric. *Fix:* use
  **5-mer composition cosine** + **domain-set Jaccard** (within-class vs cross-class).

## Data pipeline

- **`splits_core` records initially dropped `training_text`** →
  `training_prefix_for_chunking` raises. *Fix:* add
  `training_text = canonical_prefix + sequence` to each record.
- **`curate_dataset.py --min-len 1000` deletes short single-enzyme cores** (e.g. ectoine
  ~400 bp). *Fix:* `--min-len 300` for the core build.

## Environment / tooling

- **`flash-attn` install order:** its `setup.py` runs `import torch` at build time, so a
  plain `environment.yml` create crashes (torch not yet installed). *Fix:* install torch
  first → prebuilt flash-attn wheel → re-run env update → deepspeed/peft/wandb
  (`docs/archive/gputee/FINETUNE_GUIDE.md §2`).
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is unsupported on this platform** —
  do not rely on it as an OOM fix.
- **`bs=4 ga=32` OOMs at L=32k** (forward fails at bs=4, backward at bs=2). Only
  `bs=1 ga=128` fits. Effective batch is still 128.
- **NCCL "process group not destroyed"** warning on shutdown is expected for short runs.

## Agent / shell self-traps (when operating the repo)

- **`pkill -f` / `pgrep -f` / process scans self-match the agent's own command string** →
  false "STILL RUNNING" positives and one accidental self-kill. *Fix:* match by PID, or
  exclude the current shell's own command.
- **Double-backgrounding** (`nohup … &` *inside* a `run_in_background` tool call) makes the
  harness fire "completed" immediately for the launcher. *Fix:* use `run_in_background`
  directly without an inner `&`, or attach a watcher.
- **Timezone confusion:** the queue log uses local **EDT**; `date -u` is **UTC** (4 h
  offset). A run looked stalled for "3.5 h" but was ~24 min in and healthy. Compare like
  for like.
- **Pooling a paired design and calling it n.** The β titration generated 3 seqs × 5 classes
  per cell. Treating the 15 as one sample gave "β=2 significantly degrades (z=−2.6)"; the
  correct paired t over per-class deltas (df=4) says **nothing is significant**. Between-class
  variance was masquerading as an effect. *Fix:* before any stats, check what the n is composed
  of — if the same strata appear in every cell, it is paired and must be analyzed per stratum.
- **Persisting aggregates instead of per-sequence values.** The β titration driver wrote only
  cell means to `titration.tsv`, so getting error bars required re-running pyrodigal over every
  sequence. *Fix:* drivers write a per-sequence TSV (`run_steer_magnitude.sh` → `per_sequence.tsv`)
  including the *actually applied* steering parameters, so analysis never re-derives from logs.

## ★ SILENT DEGRADATION — the highest-severity bug class in this repo (2026-07-31)

**A missing resource that yields a plausible wrong number is worse than a crash.** A crash
announces itself; a silent skip coerced to `False` looks exactly like a scientific result — and
this project already lost weeks to a 0/30 that was an instrument artifact.

Four hit in one afternoon, each indistinguishable from a real negative:

| call | missing | what it looked like |
|---|---|---|
| `check_antismash` | `databases_dir` | "prerequisite DBs not downloaded" → looked like a broken install |
| `check_antismash` | `class_map` | real cores scored correct_class **0.125 vs 0.750** → looked like "the model can't hit the right class" |
| `check_class_markers` | `pfam_hmm_path` | every sequence skipped; caller's `bool(...)` → "markers absent" → all-zero Phase 3 table |
| `find_orfs` | pyrodigal (behind `except Exception`) | silently swapped to the RETIRED six-frame scanner |

The gene caller was the worst: it feeds *every* check. Measured on one real 9.4 kb PKS core,
Prodigal → six-frame gives coding_density 0.9736 → **1.0**, n_orfs 9 → **35** (megasynthase
fragmentation, the exact failure the 2026-06-17 rewrite retired it for), and
complete_gene_fraction **pinned to exactly 1.0** (the six-frame path never sets `ORF.partial`).
No marker anywhere in the output. `except Exception` also swallowed an API break identically to
a missing package.

**Fix — `evaluation.py` now has `EvalResourceError` + `_resource_missing()`:** gating checks
(`antismash`, `class_markers`, `find_orfs`) RAISE; opt-in diagnostics skip but tag
`skip_kind="resource"`; `BGC_EVAL_STRICT=0` restores the old behaviour for triage.

### A 4-lens audit then confirmed 36 more sites. Fixed:

- **Rates divided by denominators containing unmeasured records.** `eval_suite_driver`
  computed every headline as k/n over ALL records while numerators counted only `"PASS"`. Since
  `quick_eval.sh` always passes `--skip-checks kmer_novelty`, the project's ACCEPT rate
  (`biological_valid_and_novel`) was **structurally 0.000 in every quick-eval ever run**. Rates
  now divide by what was actually evaluated and return **None** — never 0.0 — when nothing was.
  Same fix in `run_steer_sweep.sh`, `run_seed_deconfound.sh`, `quick_eval.sh`.
- **`no_gate_fail` was the inverse bug** — an unmeasured gate is not "FAIL", so skipped gates
  counted as clean passes, inflating it exactly when the instrument was least configured.
- **The novelty GATE failed OPEN.** `novelty.get("max_containment", 0.0)` returned *maximal
  novelty* for a record missing the key. A gate must never default to its passing value.
- **`correct_novel_only` was a no-op** — `nov.get("pass") is not False` counted "not run" as
  novel, making it numerically identical to `correct_class` under a name claiming otherwise.
- **`tracks_seed` was hard 0.000 in every row** of `deconfound_summary.tsv`: the summariser
  joined on `r["sequence"][:80]`, but `evaluate_bgc` records carry no `sequence` key. Recomputed
  by index, v2_mismatch is **tracks_seed 0.317 vs tracks_tag 0.067** — the number quoted in the
  artifact was right, but the shipped script could not have produced it.
- **`conditioning_faithful` was null on EVERY row ever.** `phylum_token` matched `"P__"`
  case-sensitively against lowercase GTDB tags, and profile keys are uppercase. Both ends now
  case-insensitive.
- **`protein_homology`** ignored the mmseqs return code and only set `pass` when hits existed —
  so pass_rate averaged over only the sequences that already had a hit (9 hit-less + 1 hit
  reported 1.000, not 0.100). Zero hits is now `pass=False`.
- **`protein_foldability`** counted per-ORF runtime crashes in the denominator, reporting
  "these proteins do not fold" for "ESMFold could not run here".
- **`run_eval.sh` defaulted to SUPERSEDED corpora** (`splits_combined_grouped`,
  `splits_curated`), so the novelty gate compared against a corpus the model never trained on.
- **Unresolvable supplied paths now warn.** A typo in `--pfam-hmm` and a genuinely
  unprovisioned host previously produced byte-identical output.
- **Fully-skipped arms print `n/a`, not `0.000`**, so they cannot be read as total failure.

**Lesson worth keeping:** every one of these was found by *running a check*, never by reasoning
that one was needed. The positive control is the cheapest instrument in the repo — real
held-out cores scored at the same length/settings as the generations — and it caught three of
these in minutes.

## Activation steering (2026-07-29)

- **★ THE STEERING VECTORS ARE NOT CLASS DIRECTIONS — they are ±the length/norm axis.**
  `v_class = μ_c − μ_global` computed on mean-pooled L16 activations is ~collinear with PC1,
  which holds **98.07%** of the centered variance and correlates **−0.9996** with ‖h‖. The 11×4096
  direction matrix is rank-≈1 (**99.22%** of Frobenius energy in the top singular value; mean
  off-diagonal \|cos\| **0.934**). As a classifier, diff-of-means scores **0.186** where the
  logistic probe scores 0.881. Worst of all, the sign is set by whether a class's cores are
  longer or shorter than average, so the shipped vector **points backwards** for key pairs:
  steering NRPS→PKS with `v_PKS` has cos = **−0.856** with the true contrast (1-D AUC 0.221);
  ECTOINE→TERPENE is −0.789 (AUC 0.070). "Steer toward PKS" and "steer toward NRPS" are the same
  intervention with opposite sign ⇒ **0/30 correct_class was structurally guaranteed** in every
  α-sweep cell regardless of layer, magnitude, or n. Class lives in the residual 1.9%: removing
  PC1 leaves the probe at 0.909 (vs 0.908 full), while PC1 alone gives 0.287.
  *Fix:* rebuild directions as PC1-orthogonalized class-vs-class contrasts (not whitened LDA —
  rank-990 covariance in D=4096 is unstable); see `docs/steering_program.md` P0.
- **Teacher-forced scoring with a true-sequence prefix masks the very effect it measures.**
  `steer_causal_tests.py` conditioned on taxonomy + the first 1000 nt of the *true* sequence, then
  scored the next 1000 nt. With the real thing sitting immediately before the scored window, the
  model already knows the class and its exact local position, so neither an exemplar nor a nudge
  has anything left to contribute. Measured consequence: the positive control (real same-class vs
  different-class exemplar) read **−0.00126 ± 0.00107 — undetectable**, which made every steering
  cell uninterpretable. *Fix:* score TWO context conditions side by side — `create` (taxonomy
  only; the intervention is the sole carrier of class — this is the question we actually care
  about) and `reinforce` (the old design, retained to quantify the masking).
- **The steering hook perturbed the CONTEXT as well as the scored region.** `_add_hook` added the
  vector at every position including the prompt. During real generation we would steer only what
  the model *writes*, so the test corrupted the context whose continuation it was scoring.
  *Fix:* `start_pos` gating on both `_add_hook` and `_project_out_hook`.
- **Core selection silently filtered on LENGTH — the original confound.** Requiring
  `cond_nt + score_nt` = 2000 nt admits 98% of NRPS but only 29% of PKS, 19% of TERPENE and
  **5% of ECTOINE** (median core 393 nt). Selecting cores *on length* is the worst possible
  sampling rule for a program whose directions were wrecked by a length axis. *Fix:* window as a
  FRACTION of each core's own length (`--split-frac`, default 0.4), so every class is
  represented at its natural scale.
- **`_ref_norm()` does not measure what its docstring says.** It reads `X[:, -1, :]` — the
  **mean-pooled** activation over the full sequence (‖·‖ = 9.97 at L16), not a per-position
  activation. So `--alpha` is denominated in units of the DC component: α ∈ {1,2,4} ⇒
  ‖delta‖ ∈ {10,20,40}, which is **1.5–5.9× the entire between-sample scatter**
  (mean ‖h−gm‖ = 6.752). Every class-scored steering cell ever run was far past any usable dose.
  *Corollary:* directions are estimated in mean-pooled space but injected **per position** — a
  units mismatch independent of the magnitude question.
- **A sweep variable that is not comparable across strata.** `--beta` scales the RAW
  difference-of-means vector, and `‖v_class‖` at layer 16 spans **17×** (TERPENE 1.05, PKS 1.16,
  RIPP 1.69, NRPS 5.80, ECTOINE 17.75). A single global β therefore applies a 17×-different
  *physical* perturbation per class — it destroyed ECTOINE (coding_density 0.850 → 0.191 at
  β=0.5) while leaving PKS untouched at β=2. Neither knob is universally right: **β is constant
  in semantics** (class-mean offsets), **‖delta‖ is constant in magnitude** (what actually breaks
  the model), and with ‖v‖ spanning 17× you cannot hold both. *Fix:* `steer_generate.py` now
  exposes all three modes (`--delta-norm` absolute magnitude, `--alpha` ref-norm-relative
  magnitude, `--beta` semantic) and records `steer_v_norm` / `steer_applied_norm` /
  `steer_beta_equiv` per sequence. Titrate coherence on `--delta-norm`.
- **The first α sweep was run entirely past the toxicity ceiling.** α=1 sets ‖delta‖ = mean‖h‖
  ≈ 10, which already collapses coding_density 0.74 → 0.19. The grid started at α=1, so all nine
  steered cells (L∈{12,16,20} × α∈{1,2,4}) measured a wrecked model. *Fix:* always titrate
  coherence cheaply (coding_density, no antiSMASH) before spending hours on the real readout.
- **A readout with no dynamic range cannot detect success.** That α sweep's *unsteered control*
  scored **1/30 is_bgc (3.3%)**; every steered cell scored 0/30. At n=5/class, 0/30 vs 1/30 is
  indistinguishable — the experiment could not have detected steering working even if it had.
  *Fix:* before running a comparison, state the baseline rate and the n needed to detect the
  claimed lift. Prefer graded readouts (probe-head logit, Pfam `class_markers`) over a binary
  gate sitting on the floor.
- **`/home` at 100% breaks `micromamba run` itself**, not just installs: the process lock at
  `~/.cache/mamba/proc/` cannot be written, so every `micromamba run` fails and jobs die
  mid-flight. Calling the env's python binary directly is **not** a safe workaround — it skips
  activation, so `PATH` never gets the env's `bin/`, and antiSMASH/hmmscan/diamond/prodigal
  (all env binaries) silently fail or resolve elsewhere. *Fix (2026-07-29):* repo `data/` (32 GB,
  gitignored) moved to `/data2/ds85/bcgm_data` + symlink; `~/.cache/huggingface` (5.5 GB) moved to
  `/data2/ds85/cache/huggingface` + symlink; all probe drivers now export `TMPDIR`,
  `XDG_CACHE_HOME`, `MPLCONFIGDIR` to `/data2`.

## GenomeOcean track (2026-07-27)

- **`gradient_checkpointing_enable()` is a SILENT NO-OP unless the model is in `.train()` mode.**
  transformers' `GradientCheckpointingLayer` gates on `self.training`, and `from_pretrained`
  returns a model in eval mode. Symptom: memory identical with and without checkpointing, and
  L=10,240 OOM'd at 77 GB. With `model.train()` the same step takes **14 GB**. Also enable
  checkpointing on the **base** model *before* `get_peft_model`, with
  `gradient_checkpointing_kwargs={"use_reentrant": False}`. Always assert the flag actually
  landed (`model.base_model.model.model.gradient_checkpointing`) rather than trusting the call.
- **`/home` on gputee is 100% full** (1.8 TB shared, ~6 GB free). A `pip install vllm` into
  `~/.local/share/mamba/envs/` hit ENOSPC mid-install and also broke an unrelated file write.
  *Fix:* the GenomeOcean env lives at `/data2/ds85/envs/genomeocean`, with
  `PIP_CACHE_DIR=/data2/ds85/pip_cache` and `TMPDIR` on /data2 too.
- **vLLM is installed but unusable on gputee.** The vLLM 0.26 wheel is built against CUDA 13
  (`import vllm` → `ImportError: libcudart.so.13`) and this host's driver is 575.64.03 (CUDA 12.9).
  torch had to be refitted to `2.11.0+cu128` (and `torchvision`/`torchaudio` **force-reinstalled**
  to matching `+cu128` builds, or `transformers` dies with
  `RuntimeError: operator torchvision::nms does not exist`). All GenomeOcean scripts default to
  the HF `generate()` backend. To recover vLLM: install a cu128-built vLLM release, or driver 580+.
- **First CUDA generate() call looks catastrophically slow** — ~3 steps/s for the first ~300
  steps (warmup/JIT), then **74 steps/s** steady. Do not diagnose throughput, or kill a run, off
  the first minute. Measuring progress by watching `nvidia-smi` memory growth is also unreliable
  under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **`peft` 0.19 + `torch` 2.5.1 (the `bgcmodel` env) is broken:** `get_peft_model` →
  `AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'`. GenomeOcean PEFT work must
  run in the `genomeocean` env (torch 2.11).

- **[2026-08-11] ~~`correct_class` on SEEDED generations measures detectability, not class.~~
  RETRACTED THE SAME DAY — see the retraction entry below.** The claim was that antiSMASH agreeing
  with `is_bgc` on 117/120 records meant it was recognising the seed. **The scored sequence contains
  no seed** (0/1512 records across every seeded run). What the concordance actually reflects: on
  REAL cores at 3 kb, 31.4% of antiSMASH detections are off-class, so the metric discriminates
  fine — our generations simply land on-class when they land at all (~92% vs ~69% for a
  length-matched real core). *The durable lesson is the methodological one: a concordance rate is
  meaningless without the same rate on a control.*

- **[2026-08-11] A "final full-sequence score" comparison is not a myopia test when the score is
  cumulative.** `guided_generate.py` scores `seq + cand`, so the guided arm's final number is a
  max-of-4 and the control's a uniform-of-4 — the guided arm wins by construction even under total
  myopia. The valid comparison is **max-vs-max**: the guided arm's chosen score against the control's
  own `guide_p_target_max`. Chunk 0 then gives 40/40 exact ties (both arms share candidate sets), a
  free harness control that the naive test cannot produce.

- **[2026-08-11] A paired sign test can be at its DESIGN FLOOR and look like a null.** best-vs-plain
  gave p=0.5000 with 2 discordant pairs — but 2×0.5² = 0.5 is the *minimum attainable* two-sided p at
  n_discordant=2, reached when the treatment wins BOTH. Reading it as "no effect" inverts the
  evidence. Always report the discordant count next to p; and with zero reversals, 6-0 is the
  smallest result that can reach p<0.05.

- **[2026-08-11] `pgrep -f <script>` inside a waiter matches the waiter's own command line.** A
  background poller written as `while pgrep -f run_guided_decoding.sh; do sleep 15; done` never
  exits: `pgrep -f` matches full command lines, including the poller's. Watch for a completion marker
  in the log instead, or exclude self with `pgrep -f ... | grep -v $$`.

- **[2026-08-11] RETRACTED same-day: "the seeded readout is confounded by the seed".** Claimed on
  the strength of `correct_class` agreeing with `is_bgc` on 117/120 guided-decoding records. **The
  seed is not in the scored sequence.** `guided_generate.py` starts `seq = ""` and only does
  `seq += cands[pick]` (the seed reaches the model via `prompt_seqs`); `seed_generate.py` stores
  `extract_sequence(...)`, the prompt-stripped generation. Across every seeded run on disk,
  **0/1512 stored sequences begin with their seed**. Now pinned by `tests/test_scored_span.py`.
  **Two lessons.** (1) *A concordance rate is meaningless without the same rate on a control.* Real
  cores at 3 kb show 31.4% of detections landing off-class and 84.7% overall agreement — so the
  metric discriminates fine and the 97.5% was a fact about the generations. Measuring that first
  would have prevented the whole error. (2) *A premise handed to a verifier is not verified by it.*
  Five adversarial reviewers were given the confound as background context; none checked the
  generator source. Put the premises in the attack list, not the preamble.

- **[2026-08-11] An intervention that saturates is a positive control, not a measurement.**
  Activation patching with `mode='all'` (substitute every position at layer L) returned alignment
  **1.000 at layers 0, 16 and 31 alike, with an identical KL of 0.8508 at each** — because once
  every position at depth L is the donor's, layers L+1..31 compute from the donor alone and the
  model simply becomes the donor. The tell was that the numbers were *identical across layers*, and
  a layer profile that does not vary with layer is not a layer profile. Kept as a positive control
  (it proves the patch propagates and nothing leaks past the hook). The informative version
  substitutes only the last k positions.

- **[2026-08-11] A null from a low-leverage intervention is not a null about the mechanism.**
  Patching ONE position out of 1000 at mid-layers gave alignment 0.056–0.13, indistinguishable from
  the noise control — which reads as "the model does not consult this layer" and would have closed
  the last open door in the programme. Sweeping k showed the opposite: 10 positions gives 0.414 at
  layer 16, 200 gives 0.837. It was **leverage, not blindness**. *Rule: before reading a null off an
  intervention, vary its magnitude along the axis that controls leverage and show the effect stays
  flat. A single operating point cannot distinguish "no mechanism" from "too small to see".*

- **[2026-08-11] An intervention experiment run on BASE Evo2 when the capability lives in the LoRA
  has no floor, and its null is unreadable.** `patch_generate.py` defaults `--adapter None`. The
  first Phase-B run therefore generated from base Evo2 and every arm — including the unpatched
  control — returned `is_bgc = 0.000`, so "the transplant did not install the donor's class" was
  indistinguishable from "nothing in this run was detectable at all". The seeded BGC capability
  comes from the adapter (`.../phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200`),
  which every earlier seeded experiment passed explicitly. **Fix:** rerun with `--adapter`, and
  `analyze_patch_generate.py` now ABORTS when the unpatched control detects nothing rather than
  printing a table of zeros. *Rule: a treatment arm can only be read against a control that is off
  the floor — check the control first, and make the analyzer refuse otherwise.*
