# Activation steering — diagnosis and program

> **⚠ CORRECTIONS (2026-07-29, verified by direct measurement).** This document is
> agent-generated. Its core geometric findings reproduce exactly, but **two claims below are
> wrong** and one design recommendation was rejected. Read `docs/steering_program.md` for the
> plan actually being executed.
>
> 1. **"per-position activation ~485 vs pooled 9.97" is FALSE.** Measured on 54 real cores at
>    L16: mean single-position ‖h‖ = **11.53**, pooled ‖h‖ = **6.52** — a **1.8×** ratio, not
>    48×. The claim would have invalidated the whole dose calibration; it does not. The class
>    direction also separates classes nearly as well on *single positions* as on the pooled
>    state (NRPS 0.859 vs 0.867; TERPENE 0.990 vs 0.996), so injecting per-position is sound.
> 2. **The `res_mlp_norm` reading is misleading.** `ParallelGatedConvBlock.forward` ends
>    `y = self.res_mlp_norm(z_in)`, which looks like a normalization that would wash out an
>    injection. It is not: `res_mlp_norm(x) = self.mlp(self.post_norm(x)) + x` — an MLP
>    sub-block with its own residual. **The block output is the true residual stream**, so an
>    added vector persists downstream, and capture/injection happen in the same space.
> 3. **Whitened LDA was rejected** in favour of length-stripped, other-class-balanced contrasts
>    (rank-990 covariance in D=4096 is unstable). See `build_steer_dirs.py`.
>
> **Assumption audit (all cleared).** Directions are *not* disguised taxonomy directions (they
> separate phylum at only 0.36–0.63 vs their own class at 0.86–0.99); the 991 cores are near
> independent (862 distinct genomes, max 4 per genome); directions built from 1000 nt agree with
> full-core ones (cos 0.85–0.98); and generated sequences sit **within 1–2σ of the real-core
> distribution**, so the directions are applied roughly where they were estimated — with the
> behaviourally-working seeded arm sitting *closest* to real cores (±0.45σ).


**Status:** 2026-07-29. Produced by a 34-agent diagnostic workflow (4 independent lenses →
dedup/rank → per-cause experiment design → adversarial critique → synthesis). **Every headline
geometric number was independently recomputed from `class_probe_sweep/acts_v2.npz` before this
document was written** — PC1 variance share, direction-matrix rank, pairwise cosines, the
sign-inversion table, the estimator comparison, and the scatter scale all reproduce exactly.

**One-line summary:** activation steering has never been tested. The vector being injected is
±the activation-norm/length axis rather than a class direction, it was injected at 1.5–5.9× the
entire between-sample scatter, and it was scored with a binary gate sitting on a 3.3% floor.
Three independent, each-sufficient defects.

**Read `docs/project_memory/progress.md` → "THE ACTUAL BLOCKER" for the verified geometry table.**

---

# Program: making activation steering work on Evo2 7B + v2 LoRA

All numbers below marked **(measured this session)** were computed directly from `/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_v2.npz` (n=991, full-prefix pool, layers 12/16/20). Everything else is cited from files on disk.

---

## 1. The issues, in blocking order

**I1 — The vector you have been injecting is the mean-pool activation-norm axis, not a class axis, and for the pair you care most about it points backwards.** At L16, PC1 holds **98.07%** of centered variance and correlates with ‖h‖ at **r = −0.9996**; every `v_class = μ_c − μ_global` has |cos(v_c, global mean)| **0.776–0.977** (mean 0.94) and mean pairwise |cos| **0.874**. The decisive number: using unit `v_PKS` as the "steer NRPS→PKS" direction gives 1-D AUC **0.230** — i.e. pushing along it makes activations look *more* NRPS, not less (cos(û_{NRPS→PKS}, v̂_PKS) = −0.856). `v_TERPENE` for ECTOINE→TERPENE scores AUC **0.070**. *(measured this session)*

**I2 — Every class-scored cell was 15–100× past one class-unit; the one experiment that used a sane magnitude was on the wrong axis — so no experiment on disk has ever placed a class-carrying vector at a plausible dose.** `_ref_norm()` reads `X[:, -1, :]`, the *pooled* activation (‖·‖ = 9.97 at L16), so α ∈ {1,2,4} ⇒ ‖Δ‖ ∈ {10,20,40}. One class-unit along the *corrected* axis is ‖Δ‖ = **0.68–0.89** (⊥-contrast) or **0.33–0.47** (probe discriminant) *(measured this session)*. The magnitude titration's ‖Δ‖ ∈ {0.5,1,2} was the right size but ~90° off the class direction.

**I3 — No experiment has ever run a magnitude- and subspace-matched null direction**, so "steering is toxic" and "steering does class-specific work" are not separated in any result on disk. `run_steer_sweep.sh`, `run_steer_titration.sh`, `run_steer_magnitude.sh` all compare steered vs Δ=0 only.

**I4 — The forward hook adds Δ at every position including prefill, so in the only regime with dynamic range the intervention destroys the signal it is measuring.** `vortex/model/generation.py` runs the full prompt through the model at i=0; `_install_steer_hook` (`evo2/scripts/steer_generate.py:56`) adds Δ across the whole sequence dimension. Seeded generation with a 2000-nt seed therefore perturbs ~2000 prompt positions — and the seed is where the class lives (`tracks_seed` 0.317 vs `tracks_tag` 0.067).

**I5 — Every binary class readout on taxonomy-only generations sits on the same ~3% floor, and the "16× dynamic-range advantage" that has been used to justify the Pfam fallback does not exist.** `report_a0_control.json`: antiSMASH `correct_class` 1/30 = 0.033 **and** `class_markers` pass 1/30 = 0.033. The 0.533 figure is `run_steer_sweep.sh`'s `domain_count > 0` — any Pfam family, class-agnostic. Rename it `any_domain_rate`.

**I6 — The continuous surrogates are confounded with coherence and length, so none can be adopted on raw AUROC.** On the 420 `seed_deconfound` records an arm-identity-only score scores AUROC 0.789 and a coding-density-only score 0.644, both inside the band any probe would be judged in. At L16 the pooled activation decodes length bucket at 0.960 and length bucket alone decodes class at 0.226.

**I7 — Panel and generation length are mismatched to the gate, but not for the reason assumed.** antiSMASH is *not* length-limited: joining `as_calib.jsonl` to `splits_core/test.jsonl`, real cores ≤2048 nt score is_bgc **74/74** and correct_class **72/74**. The real problems are (a) `PKS_NRPS_HYBRID` — median core 21 kb, marker set is a near-union of NRPS's and PKS's (11/20 shared with PKS, 6/20 with NRPS), 20/45 unsteered triad records already fire its markers; (b) ECTOINE/TERPENE/PKS cores are 94%/81%/83% single-gene, so `correct_class` on them is a one-gene Pfam detector; (c) `check_antismash()` never passes `--minlength`, silently skipping every <1000 nt sequence (binds on short real cores, never on generations).

**I8 — MOST LIKELY FATAL: the class-discriminative subspace is one the residual stream essentially never varies along, and class is decodable equally well in BASE Evo2, which has no class prior.** After removing PC1 the probe is unchanged (0.916 → **0.911**) while PC1 alone gives only **0.315** — so class lives entirely in the ~1.9% residual. Along that residual axis the activation sd is **0.24–0.36** (⊥-contrast) or **0.10–0.14** (probe discriminant), versus **8.1** along PC1, so one class-unit is a **~3σ** off-manifold excursion. Combined with probe balanced accuracy 0.911 on base Evo2 (no class prior, no conditioning), the live hypothesis is that L16 mean-pooled class identity is an *epiphenomenon of the input*, not a variable the generator reads. **No fix to direction, magnitude, readout, or panel can rescue this**; it is the only issue in the list that is not addressable by code.

### Measured geometry (L16, full-pool, n=991)

| Quantity | Legacy `v_class` | Raw contrast μ_B−μ_A | **PC1-⊥ contrast** | Probe discriminant |
|---|---|---|---|---|
| 1-D AUC (A vs B) | 0.070 – 0.962 (*sign-inverted for NRPS↔PKS, ECTOINE↔TERPENE*) | 0.503 – 0.940 | **0.952 – 0.999** | 1.000 |
| \|cos with PC1\| | 0.78 – 0.98 | 0.85 – 1.00 | 0 (by construction) | 0.001 – 0.008 |
| ‖Δ‖ for 1 class-unit | n/a (not a contrast) | 1.04 – 16.91 | **0.68 – 0.89** | 0.33 – 0.47 |
| Activation sd along axis | 8.1 | 6.1 – 8.2 | **0.24 – 0.36** | 0.10 – 0.14 |
| Class gap in σ | — | 0.17 – 2.07 | **2.0 – 3.2** | 2.9 – 3.9 |
| Split-half \|cos\| (20 reps) | ~1.00 (stability *of the length axis*) | 0.73 – 0.999 | **0.77 – 0.92** | — |
| Window stability cos(@1000 nt, @full) | — | 0.09 – 1.00 | **0.91 – 0.98** | — |

Probe balanced accuracy: full features **0.916**; PC1 removed **0.911**; PC1 projection only **0.315** (chance 0.091).

---

## 2. Candidate fixes per issue

| Issue | Options | Verdict |
|---|---|---|
| I1 | (a) class-vs-class contrast; (b) shrinkage-whitened LDA; (c) **PC1-orthogonalized contrast**; (d) probe-discriminant contrast | (a) alone **fails** — the raw contrast is still 0.85–1.00 collinear with PC1. (b) **reject**: rank-990 covariance in D=4096, split-half cos goes negative at shrinkage 0.05, and it selects a below-random-variance axis. **(c) is the primary**; (d) as a labelled secondary — it is cleaner (AUC 1.000) but 2–3× lower-variance, hence more off-manifold per unit norm. |
| I2 | Retire `--alpha`/`--beta`; express every dose in **class-units** and in **σ-along-axis**; log absolute ‖Δ‖ per record | Adopt. Note `--delta-norm` already unit-normalizes, so the "17× ‖v‖ spread" was never the applied confound — the spread is just \|projection onto PC1\| per class (long-core classes negative, short-core positive), a mean-pooling length artifact. |
| I3 | (a) isotropic random unit vector; (b) **label-permuted direction through the identical estimator**; (c) the PC1 axis itself at matched ‖Δ‖ | **(b) is the real null** — same estimator, same subspace, no class signal. (a) is a straw null (an isotropic vector puts ~1.1% of its energy in the top-50 PC subspace). Run (c) as a diagnostic: it directly quantifies how much of the historical null was the wrong axis. |
| I4 | Gate the hook to post-prefill positions (`if out[0].shape[1] != 1: return out`) | Adopt, and run a delta>0 vs delta=0 n=10 sanity check that seeded is_bgc stays at ~0.33–0.47 before spending on P3. |
| I5 | (a) more n on the binary gate; (b) move to the seeded regime; (c) continuous surrogate | (a) is unaffordable (0.033→0.10 at 80% power needs n≈216/arm ≈ 5.8 GPU-h/arm at 6144). **(b) is the operating point** — seeded mk_SEED 0.600, antiSMASH as_SEED 0.333, n=60 already measured. (c) only as a screen, never as an endpoint. |
| I6 | Adopt on **incremental** AUROC over a coding-density baseline, within-stratum, CI bootstrapped over the 104 distinct seed accessions | Adopt. Raw pooled AUROC ≥ 0.75 is reachable with zero class information. |
| I7 | Drop `PKS_NRPS_HYBRID`; panel = NRPS, PKS, TERPENE (+RIPP where marker sets allow); land `--minlength 1`; stratify every rate by `strict_core_genes` | Adopt. Keep generation at 6144 for the gold-gate arms; 2048 is fine for coherence titration only. |
| I8 | No code fix exists. Only a causal test can resolve it. | This is what P1 and P3 are for, and it is the kill criterion. |

---

## 3. Ordered experiment plan

Measured cost constants used throughout: teacher-forced forward ≤4096 nt with one hook ≈ **0.25 s/seq**; autoregressive generation ≈ **72 tok/s** steady **+ ~313 s fixed model load per `steer_generate.py` invocation** (⇒ 2048 tok ≈ 32–44 s/seq, 6144 tok ≈ 96 s/seq, seeded-6000-with-2000-nt-seed ≈ 93–133 s/seq); antiSMASH 7.6–20.7 s/seq serial, ~4275 seq/h at 8-way; `class_markers` 9–13 s/seq (**one** hmmsearch per sequence — `expected_class` only selects which accessions you read off it, so scoring 11 classes costs the same as scoring 1).

| Phase | Deliverable | GPU-h | CPU | Cum. GPU-h |
|---|---|---|---|---|
| **P0** | Direction bank + dose units + readout export + audit fixes | **0** | ~3 h dev | 0 |
| **P1** | Teacher-forced ± selectivity screen vs matched nulls | **1.0** | — | 1.0 |
| **P2** | Generation dose-response + coherence envelope on the corrected axis | **1.5** | ~20 min | 2.5 |
| **P3** | Seeded cross-class override — **the decisive test** | **4.2** | ~25 min | 6.7 |
| **P4** | De novo taxonomy-only confirmation at the gold gate | **7.2** | ~35 min | 13.9 |

For reference, the α sweep alone cost **8.0 GPU-h** (`steer_sweep/master.log`, 15:43→23:45) and returned zero bits.

### P0 — Direction bank and instrument fixes (0 GPU-h)

Write `/home/ds85/projects/BCGModelling/evo2/scripts/build_steer_dirs.py`, reading `acts_v2.npz`, writing a **new** file `/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_v2.dirs_v2.npz`. Do **not** write to `acts_v2.dirs.npz` — it holds 352 legacy keys and `np.savez_compressed` truncates.

Keys per layer L ∈ {12,16,20} and ordered pair (A,B): `perp_L{L}_{A}__{B}`, `probe_L{L}_{A}__{B}`, `raw_L{L}_{A}__{B}`, `pc1_L{L}`, plus `perm{k}_L{L}_{A}__{B}` for k = 1…10 (label-permuted through the identical estimator). Also persist `class_unit_L{L}_{A}__{B}` = (μ_B − μ_A)·û and `sigma_L{L}_{A}__{B}` = held-out sd of projections. Every downstream summary row logs dose in **class-units**, in **σ**, and in absolute ‖Δ‖.

Also: export the L16 logistic head (`coef_`, `intercept_`, scaler `mean_`/`scale_`, class order) — `class_probe_sweep.py::_fit` currently discards every fitted model inside `cross_val_score`; export a PC1-removed variant as the readout head. Fix the join at `evo2/experiments/probes/run_seed_deconfound.sh:97` (join on accession/index — the report per-record dict has no `sequence` key, so `tracks_seed` and the leak audit were structurally 0.0 in all seven arms). Add `min_length: int = 1` to `check_antismash()` and thread it through `EvalConfig` and `eval_suite_driver.py`.

**Admissibility gate (per L, per ordered pair) — all four must hold on held-out folds:** 1-D AUC ≥ 0.90; |cos(û, PC1)| ≤ 0.10; split-half |cos| ≥ 0.70; prefix-window |cos(û@1000nt, û@full)| ≥ 0.85. At L16 the ⊥-contrast passes on all six panel pairs (0.952–0.999 / 0 / 0.77–0.92 / 0.91–0.98). The legacy `v_class` fails on every criterion. **State plainly in the write-up that the gate's informative content is the falsification of the legacy directions, not the admission of the new ones** — the latter was known before the script was written.

### P1 — Teacher-forced ± selectivity screen (1.0 GPU-h)

New `evo2/scripts/steer_nll_screen.py`, reusing `_install_steer_hook` (with a scope flag) and `sequence_loglik` from `conditioning_experiment.py:172`. No sampling — one forward per (core, kind, magnitude, sign).

Data: `splits_core/test.jsonl` — genuinely held out (the 991 cached cores are val.jsonl, confirmed by class counts 100×8 / 86 / 65 / 40). n=25 cores/class for NRPS, PKS, RIPP, TERPENE, **length-stratified across classes**, all ≥2000 nt; condition on nt 1–1000, score nt 1001–2000 so the scored window is identical for every class (this avoids pushing TERPENE/PKS into their extreme length tails — TERPENE median core is 942 nt).

Statistic, fully paired within core: **T(x) = ΔNLL(x; +m·û_{c→o}) − ΔNLL(x; −m·û_{c→o})**. Because û_{o→c} = −û_{c→o} exactly, any sign-symmetric coherence damage cancels. Damage guard D = mean|ΔNLL| over both signs.

Grid at L16: kind ∈ {perp, probe, raw_legacy, pc1_null, perm-null ×10} × m ∈ {0.25, 0.5, 1, 2, 4} class-units × 2 signs × 100 cores. L12/L20 and the generated-only hook scope are run only at the winning m.

**Mandatory positive-control anchor:** the same T computed for an intervention already known to move generation — a real class-B exemplar core as prefix (seeding, `correct_class` 0.283). Report every T as a fraction of T_anchor.

**Decision rule.** PASS a cell iff (i) T_perp exceeds the 95th percentile of the ten permuted-null T's at the same ‖Δ‖, in **≥3 of 4 classes**; (ii) T_perp ≥ **0.20 × T_anchor**; (iii) D ≤ D_anchor. Carry (L*, m*, kind*); prefer `perp` over `probe` on a tie (2–3× more on-manifold variance). Expected diagnostic: `raw_legacy` and `pc1_null` should not pass at any m — **if `raw_legacy` passes and `perp` does not, the diagnosis in §1 is wrong and the program must be re-derived before any generation spend.** No-pass at every cell → skip P2 and run the reduced 2-arm P3 directly; do **not** kill here (teacher forcing supplies the class in context, so it measures reinforcement not creation, and a per-token bias below the detection floor can still dominate 2000 sampling steps).

### P2 — Generation dose-response and coherence envelope (1.5 GPU-h)

Necessary because the existing titration measured coherence along ~PC1; the corrected axis is a different subspace with unmeasured toxicity. Taxonomy-only, 2048 new tokens, L*, kind=perp, m ∈ {0.5, 1, 2, 4} class-units, plus m=0 and the permuted null at the top m. 6 cells × 20 seqs = 120 gens ≈ 1.5 GPU-h including 6 model loads.

**Coherence primary is `max_orf_aa` and ORF count, not `coding_density`** — on the actual control arm, coding_density for marker-PASS vs marker-FAIL is 0.888 vs 0.879 (Δ=0.009) while max_orf_aa is 858 vs 605; and the existing titration's coding_density series is non-monotonic (0.826 / 0.728 / 0.837 / 0.737 at ‖Δ‖ = 0 / 0.5 / 1 / 2) i.e. pure n=25 noise. **Also report realized length per arm as a primary**: steering suppresses EOS (a0_control mean 5179 nt with 80% reaching cap, vs steered cells at 6144/6144), and length is the confound that manufactures every downstream artifact.

**Decision rule.** m* = the largest m whose arm's mean `max_orf_aa` 95% CI overlaps the m=0 arm **and** whose mean realized length is within 10% of the m=0 arm. If no m ≥ 0.5 class-units satisfies this, stop and report the coherence ceiling in class-units — a direction that cannot take half a class-mean step without ORF collapse has no operating point.

### P3 — Seeded cross-class override (4.2 GPU-h) — the decisive test

Run here because this is the only regime with a *measured floor and ceiling*: mk_SEED 0.600 and antiSMASH as_SEED 0.333 at n=60, versus 0.033 taxonomy-only. The attribution question is settled in its favour: with the join fixed, `tracks_seed` = 0.317 vs `tracks_tag` = 0.067 — the class signal is genuinely present in the seed, which is what makes it something an override can move.

Build: merge `_install_steer_hook` into `evo2/scripts/seed_generate.py`, **gated to post-prefill positions** (`if out[0].shape[1] != 1: return out`), using the real flag names (`--layer`, `--delta-norm`) plus `--steer-dirs-npz`, `--steer-kind`, `--steer-cross-class`. Sanity-check at n=10 that seeded is_bgc holds at ~0.33–0.47 with the hook active before committing the full run.

Three concurrent arms, same session, same code path, `--seed 42` so records are paired by seed exemplar; `--classes NRPS PKS TERPENE --per-class 15` ⇒ 45 records/arm; `--seed-nt 2000 --max-new-tokens 6000 --no-class-tag`:

| Arm | Direction | Dose |
|---|---|---|
| A | ⊥-contrast, seed-class → rotating other panel class | m* |
| B | label-permuted null, same subspace | matched ‖Δ‖ |
| C | unsteered | 0 |

135 gens × ~110 s ≈ 4.1 GPU-h + ~25 min antiSMASH.

Readouts, all paired by exemplar, **McNemar** (the arms are exactly paired; an unpaired two-proportion test throws away the design):
- **Primary (installation):** *exclusive* mk_TARGET = target-class markers present AND seed-class markers absent; plus antiSMASH exclusivity (target ∈ `mapped_classes` AND seed ∉ `mapped_classes`). Do not use `correct_class` — `evaluation.py:449` is set membership, and in the steered arms `compound_class` records the *seed* class unless a `--target-class` flag is added.
- **Secondary (suppression):** mk_SEED.
- **Nuisance match (gating):** realized length, max_orf_aa, ORF count. If A and B are not matched, the comparison is void.

**Decision rule.** PROCEED to P4 iff, on the **A-vs-B** contrast: exclusive mk_TARGET higher in A at McNemar p < 0.05, discordant-pair ratio ≥ 2.0, and A/B nuisance-matched (|Δ mean length| ≤ 10%, Δ mean max_orf_aa inside its 95% CI). Power: n=45 paired against a ~0.05 base rate detects a rise to ~0.25 at 80%; it **cannot** resolve a rise to 0.12 — that needs n≈130/arm (+8 GPU-h) and should be declared out of scope in advance.

**A-vs-C is reported but is not inferential.** `v2_tag_shuf` (codon-shuffled seed, *no steering*) already gives triad mk_SEED 0/45 at coding_density 0.692, and `base_notag` gives 8/45 — both would pass any suppression-vs-unsteered rule. Suppression relative to unsteered is worth zero bits.

### P4 — De novo confirmation at the gold gate (7.2 GPU-h)

Only if P3 proceeds. Three arms (⊥-contrast at m*, permuted null at matched ‖Δ‖, unsteered) × n=90 at 6144 tokens = 270 × 96 s = 7.2 GPU-h + ~35 min antiSMASH. Size n from P3's measured effect rather than from the 0.033 floor. Two required additions: add an `id` field to `steer_generate.py` records (currently absent, so `eval_suite_driver.py` falls back to `str(index)` and the novelty map cannot join), and pass `--novelty` from `scripts/memorization_check.py` against `splits_core/train.jsonl` — the directions are fit on val cores whose taxonomic tags are the prompts, so an ungated confirm is indistinguishable from retrieval. Report `correct_novel_only` as the headline, and report is_bgc within length strata.

---

## 4. Kill criterion

**Abandon activation steering and move to per-class LoRA adapters (or the GenomeOcean class-token route) if:**

**K1 (build check).** P0 admissibility fails — no (L, ordered pair) reaches AUC ≥ 0.90 with |cos(û,PC1)| ≤ 0.10 and split-half ≥ 0.70. *Already measured to pass at L16 on all six panel pairs, so this is not a live risk; it exists to catch an implementation bug.*

**K2 (mechanism).** P1 returns T_perp inside the permuted-null band at **every** (L, m, class, hook scope) **while the positive-control anchor T_anchor registers cleanly**. Interpretation: the model's own next-token distribution is unmoved by the corrected class direction even with the class fully in context, while an intervention known to change generation registers strongly ⇒ decodable ≠ steerable at this site. This is exactly what I8 predicts. **K2 alone downgrades but does not kill** — it licenses only "constant additive steering at blocks 12/16/20 over 0.25–4 class-units does not move teacher-forced NLL", not a route-level abandonment.

**K3 (behaviour) — this one kills.** P3's A-vs-B exclusive-target McNemar 95% CI **upper bound excludes a 2× lift**, with A and B nuisance-matched on length and max_orf_aa. That is a properly controlled, properly paired, adequately powered negative in the one regime with measured dynamic range, against the only null that matters.

**K4 (no operating point).** P2 puts m* below 0.5 class-units — the corrected direction cannot take half a class-mean step without measurable ORF collapse. Report and stop; no statistical design rescues an intervention with no dose that both does something and preserves the sequence.

**K2 ∧ K3 is the high-confidence kill.** K3 alone is sufficient. K2 alone triggers P3 at reduced scope (arms A and B only, 2.8 GPU-h) rather than a stop.

**Do not accept any of the following as evidence against steering:** 0/30 on a 3% floor (P(0 | p=0.033, n=30) = 0.365 — nine such cells is unsurprising under the null); suppression relative to an unsteered arm (codon-shuffling the seed does it with no steering at all); a coherence pass based on `coding_density` (Δ = 0.009 between marker-PASS and marker-FAIL); or any result from a cell whose injected vector was `v_class` from `acts_v2.dirs.npz` (AUC 0.070–0.962 with sign inversions on the panel's key pairs).

**Blunt assessment.** I1 and I2 fully explain every null on disk and are ~zero-cost to fix — which means the program has never actually been tested, and it deserves the 6.7 GPU-h to P3. But the thing most likely to kill it is I8, and I8 is not a bug: the class axis carries **1.9%** of the variance, the residual stream's sd along it is **0.24–0.36** against **8.1** along PC1, and base Evo2 — which has no class prior and no conditioning — decodes class just as well as the fine-tuned model. If P1 comes back inside the null band, the honest reading is that class at L16 is a readout of the input, not a control variable of the generator, and per-class adapters are the correct next spend.
