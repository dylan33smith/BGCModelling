# Bugs & quirks — and the proven fixes

Recurring errors, environment/tooling quirks, and what actually fixed them. Add an entry
whenever a non-obvious bug is solved. See [decisions.md](decisions.md) for rationale and
[progress.md](progress.md) for current state.

---

## Evo2 / vortex / generation

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
