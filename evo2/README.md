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
| `scripts/` | Training (`finetune_evo2_lora.py`), generation (`generate_bgc.py`), conditioning probes (`cfg_generate.py`, `seed_generate.py`, `conditioning_experiment.py`), the class probe + steering pair (`class_probe_sweep.py`, `steer_generate.py`), chunk indexing, the `queue_h100_*.sh` wrappers, `quick_eval.sh` / `run_eval.sh`. |
| `experiments/probes/` | The 2026-07 probe programme (capability chain, rank sweep, gene-aware A/B, concentration, CFG, seeding) and the 2026-07-29 steering titrations (`run_steer_magnitude.sh` — current; `run_steer_titration.sh` — superseded β sweep). |
| `experiments/quartz/` | Multi-GPU long-context staging for IU Quartz (blocked on an RT Project allocation). |
| `docs/` | `evo2_lora_and_hyena.md` (why the long-range pathway is untrained), `quartz_setup.md`. |

### Activation steering — strength flags

`steer_generate.py` adds a class direction to the residual stream at layer L. Class direction
norms span **17×** at layer 16, so there is no single strength knob; pick the one matching the
question (see `docs/project_memory/decisions.md`):

| flag | holds constant | use for |
|---|---|---|
| `--delta-norm` | absolute ‖delta‖ | **coherence titration** — damage tracks magnitude |
| `--alpha` | ‖delta‖ / mean‖h‖ | same, in ref-norm units (older runs) |
| `--beta` | class-mean offsets | class-effect comparisons |

Exactly one is required. Each record carries `steer_v_norm` / `steer_applied_norm` /
`steer_beta_equiv`, so either axis is recoverable after the fact.

## Environment

```bash
micromamba activate bgcmodel      # torch 2.5.1+cu124, transformers 4.46.3, evo2 + vortex
export HF_HOME=/data2/ds85/hf_cache
```

This env also carries antiSMASH 8.0.4 and Pfam, so it is where **both** tracks run the
eval suite.

## Where this track stands

Prefix class-conditioning on Evo2 is closed as a dead end — see
`../docs/project_memory/progress.md` and the 2026-07-21 entry in
`../docs/project_memory/decisions.md`. The LoRA adds BGC-likeness (coding density
0.61 → 0.89) but nothing about class (`correct_class` 0.013 at n=75). CFG found no
amplifiable class signal; LoRA capacity and chunking were both ruled out.
