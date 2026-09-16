# BGC-BENCH

Do two DNA language models — **Evo2-1B** and **GenomeOcean-4B** — generate biosynthetic gene
clusters, and does class-conditioning help? antiSMASH 8.0.4 is the sole endpoint instrument.

**Results: [`docs/RESULTS.md`](docs/RESULTS.md)** — 48 arms, 24 per model, all at n=200 per class.

In short: unaided, both models sit near the floor and are statistically indistinguishable.
Seeding with a real 64 nt core prefix is the only intervention that works (up to 6.8×), and
Evo2 converts a seed significantly better than GenomeOcean. Activation steering does not move
the endpoint on either architecture.

## Documents

| file | what it is |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | the final numbers, with significance tests and the calibration that licenses reading them |
| [`docs/SPEC.md`](docs/SPEC.md) | the method — metrics, data, arm grid, generation protocol |
| [`docs/DEFECTS.md`](docs/DEFECTS.md) | provenance: every spec amendment and defect, dated. Six changed published numbers |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | instrument and design findings. ⚠ its *rates* predate the retrain — see its status header |
| [`reference/frozen/KNOWN_WRONG.md`](reference/frozen/KNOWN_WRONG.md) | defects inherited from the prior codebase, each with a red test |

## Reproducing

Artifacts live on `/data2/ds85/bgcbench`, never in this repo. The chain runs in order; each
stage consumes the previous one's output.

```bash
conda env create -f environment.yml && conda activate bgcmodel

# 1. raw GenBank -> corpus of BGC cores
python -m bgcbench.data.extract /data2/ds85/bgcbench/corpus/core_records.jsonl

# 2. corpus -> class map, cluster-disjoint splits, negatives, manifest
python -m bgcbench.run.build_data

# 3. splits -> the 10 LoRA adapters (5 per substrate: W1n pooled + W2 per class)
pipeline/… see run order below

# 4-7. directions -> injection sites -> alpha -> the 48 scored arms
# 8. the comparison tables
python -m bgcbench.run.results_table
```

`pipeline/` holds the eight scripts that actually invoked the 48 arms, in dependency order:

| script | stage | produces |
|---|---|---|
| `phaseA0_denovo.sh` | 3 → arms | the 12 de novo arms |
| `phaseA_seeded.sh` | | the 12 seeded (`_S1`) arms |
| `phaseB_directions.sh` | 4 | 8 I1 directions, derived on each arm's own weights |
| `phaseB2_sites.sh` | 5 | injection site sets, and the §6.4 check at those sites (`--finalize`) |
| `phaseC_alpha.sh` | 6 | chosen α per class, against generation health only |
| `extend_alpha.sh` | 6 | the downward grid extension; the only admissible α for GO TERPENE and Evo2 REDOX |
| `phaseD_steer.sh` | 7 | the 24 steering arms (I1, I1rand, I1xS1) |
| `_lib.sh` | — | shared; reads every substrate parameter from `model/substrate_config.py` |

⚠ **`_lib.sh`'s `wait_gpu` is checked per arm, not per phase.** A once-per-phase check is a
snapshot, not a guard: on a shared card it lost 18 arms to OOM when the other user's job
returned mid-phase.

## What makes the comparison valid

Configuration is deliberately **not** matched across substrates — rank, injection sites, α,
decoding and prefix are chosen per substrate and per class by measurement (SPEC §14A). What is
held identical is the *measurement*, and `bgcbench/run/results_table.py` asserts all four on
every run before printing anything:

* `n_per_class` = 200, every arm
* `budget_nt` = 8,192 nt, every arm, none exceeding it
* one antiSMASH scoring config, `ee8c025c1593`
* the novelty gate passed by every generation

## Tests

```bash
python tests/run_all.py     # 136 tests; pytest is not in the env
```

They are mostly not unit tests. About half guard a **measurement invariant** of the frozen
arms (every sequence gets a verdict; empty generations stay in the denominator; the novelty
gate fails closed; the budget is in nucleotides, not tokens) and about half are **red tests for
defects that actually happened**, cross-referenced to `KNOWN_WRONG.md` and `DEFECTS.md`.

⚠ Roughly 55 of them assert on the **source text** of chain modules (`assert "STRIP THE PROMPT"
in src`). That makes them sensitive to moving code between files — deleting a function can red
a test that was guarding something else entirely. Read the assertion before assuming a failure
means what it says.

## Provenance

The 48 arms were produced by the code at tag **`results-frozen-2026-09-16`**. The tree was
tidied afterwards; that tag is what to check out to see exactly what ran.
