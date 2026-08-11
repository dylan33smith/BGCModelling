# Evo2 track

Everything specific to **Evo2 7B** (Arc Institute; StripedHyena 2) lives here — the
LoRA trainer, the conditioned generation path, the queue wrappers, and the probe
programme that diagnosed the class-conditioning failure. The GenomeOcean track lives
in [`../genomeocean/`](../genomeocean/).

Shared across both tracks and therefore **still at the repo root**: the dataset
pipeline (`../scripts/`), the eval suite (`../src/bgc_pipeline/evaluation.py` and its
drivers `../scripts/eval_suite_driver.py`, `../scripts/evaluate_bgc.py`,
`../scripts/memorization_check.py`), the class map (`../config/`), the tests
(`../tests/`) and project memory (`../docs/project_memory/`).

> **Paths changed 2026-07-27.** Scripts that used to be `scripts/finetune_evo2_lora.py`
> are now `evo2/scripts/finetune_evo2_lora.py`. Shell wrappers still expect to be
> invoked **from the repo root**, e.g. `evo2/scripts/queue_h100_smoke.sh`. Shared
> scripts (`scripts/eval_suite_driver.py`, `scripts/curate_dataset.py`, …) did not move.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Training (`finetune_evo2_lora.py`), generation (`generate_bgc.py`), conditioning probes (`cfg_generate.py`, `seed_generate.py`, `conditioning_experiment.py`), the class probe + steering stack (`class_probe_sweep.py`, `build_steer_dirs.py`, `steer_generate.py`, `steer_causal_tests.py`, `steer_reach.py`), the continuous class readout (`probe_score_generations.py`, `calibrate_class_probe.py`), chunk indexing, the `queue_h100_*.sh` wrappers, `quick_eval.sh` / `run_eval.sh`. |
| `experiments/probes/` | The 2026-07 probe programme (capability chain, rank sweep, gene-aware A/B, concentration, CFG, seeding) and the steering programme (`run_steer_phase2.sh`, `run_steer_phase3.sh`, `run_steer_l27.sh`, `run_steer_stack.sh`; `run_steer_magnitude.sh` / `run_steer_titration.sh` are superseded). |
| `experiments/quartz/` | Multi-GPU long-context staging for IU Quartz (blocked on an RT Project allocation). |
| `docs/` | `evo2_lora_and_hyena.md` (why the long-range pathway is untrained), `quartz_setup.md`. |

### Activation steering — strength flags

> **Steering is CLOSED as of 2026-08-10.** These flags are documented because the code and the
> negative results are worth preserving, not because another sweep is warranted. See
> "Where this track stands" below.

`steer_generate.py` / `seed_generate.py` add a class direction to the residual stream. Class
direction norms span **17×** at layer 16, so there is no single strength knob; pick the one
matching the question (see `../docs/project_memory/decisions.md`):

| flag | holds constant | use for |
|---|---|---|
| `--steer-norm-frac` | ‖delta‖ / **live, per-position** ‖h‖ | **anything where the LAYER varies** — the only dose comparable across depths |
| `--delta-norm` | absolute ‖delta‖ | coherence titration within one layer |
| `--class-units` | class-mean offsets | class-effect comparisons within one layer |
| `--alpha` | ‖delta‖ / **pooled-cache** mean‖h‖ | legacy; see the warning below |

Exactly one is required. Every record carries the **realized** `steer_mean_h_norm` /
`steer_applied_norm` / `steer_realized_norm_frac` / `steer_realized_class_units`, so the dose
never has to be re-derived from logs (the β-titration had to).

**Never derive a dose from the mean-POOLED activation cache.** Pooling averages vectors that
point different ways and shrinks ‖h‖ by a depth-dependent factor — measured, cache vs live:
6.69 vs 8.95 at L16 (0.75×) but **31.97 vs 11.25 at L27 (2.84×)**. A cache-derived dose is
mis-scaled, and increasingly so with depth. `--steer-norm-frac` measures ‖h‖ at the hook.

`--steer-layer` accepts a **comma list** (`10,12,14,16,18,20,22,24,27`) to stack the direction
at every listed layer, each using its own direction and its own class-unit.

## Environment

```bash
micromamba activate bgcmodel      # torch 2.5.1+cu124, transformers 4.46.3, evo2 + vortex
export HF_HOME=/data2/ds85/hf_cache
```

This env also carries antiSMASH 8.0.4 and Pfam, so it is where **both** tracks run the
eval suite.

## Where this track stands

**Every inference-time conditioning lever is now closed.** The full arc, with the instrument
defects each stage exposed, is in [`../docs/steering_program.md`](../docs/steering_program.md).

| lever | verdict | date |
|---|---|---|
| prefix / label conditioning | dead — the tag is provably inert (`v2_notag` == `v2_tag`) | 2026-07-21 |
| CFG | dead — no amplifiable signal; coherence collapses first | 2026-07-22 |
| LoRA capacity (rank, coverage) | not the limiter | 2026-07-13 |
| data (whole-core, chunking, class concentration) | moves DOMAINS, never the gate | 2026-07-12 |
| activation steering — all variants | **dead** | 2026-08-10 |
| per-class soft prefixes (input-embedding only) | **dead at this scale** | 2026-08-10 |

The LoRA adds BGC-likeness (coding density 0.61 → 0.89) but nothing about class
(`correct_class` 0.013 at n=75; the class probe reads 0.906 on the adapter vs **0.911 on base
Evo2**, so the adapter installed no class representation).

**The mechanism, established 2026-08-10.** The model *represents* class (probe 0.911–0.933,
chance 0.091/0.045) but the generator does not *consume* it. ΔP(target) is null in every arm on
every instrument. A companion claim that the direction reliably **deletes** a class (ΔP(seed)
−0.308, p = 0.0063) was **retracted** the same day as a probe-leakage artefact — refit train-only
it is −0.177 at p = 0.146. Phase 1's teacher-forced ablation asymmetry (z = 4.8) is independent
of the probe and stands. Two supporting measurements: an injected edit's influence on the output *falls* with depth
(L16 0.0101 → L27 0.0029), so "inject later / inject everywhere" cannot help; and the
continuous `class_probe` readout — 10× more sensitive than any binary gate — finds no
class-specific movement toward the target at any dose or layer.

**What works today: exemplar conditioning.** Seed a real core and the continuation is
correct-class 0.283 vs a 0.067 floor, memorization ruled out, all four pre-registered controls
passed. The class comes from the **seed**, never the label.

**The cheap end of training-time coupling is now also closed.** Per-class soft prefixes
(`evo2/scripts/train_soft_prefix.py`, 65k learned floats/class, frozen backbone) trained cleanly
and separated per class from an identical initialisation, but moved validation loss only ~0.003
nats and produced `correct_class` 0/12 in every generation arm. The bound is narrow and
specific: **parameters that change only the INPUT do not install class**. It does not bound
per-class LoRA (28.7M params, modifies the computation) or a real trainable class token.

**Next spend** is ranked in `../docs/conditioning_next_steps.md` — a literature sweep (2026-08-11)
found the recurring pattern is that conditioning must enter at **every layer**, via a small
gated, zero-initialised, end-to-end-trained module, rather than at one input position or as a
hand-computed activation edit.

**Standing debt:** steering directions and the class probe are fit on **val+test**; refit
train-only before reporting any number externally.
