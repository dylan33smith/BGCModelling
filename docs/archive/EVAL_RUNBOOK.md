# EVAL RUNBOOK — BCGModelling Phase-1 LoRA eval

Goal of the eval: answer, for each generated BGC, the in-scope questions —
**is it a BGC** (`is_bgc`), is it the **correct class** (`correct_class`), are its
**proteins plausible** (`proteins_plausible`), is it **novel** (not memorized,
`novel`), and is it **complete** (correctly-ordered assembly-line modules,
`complete`) — plus the conditioning-faithfulness diagnostic. The suite is
deliberately scoped to in-silico BGC validity; it does **not** cover wet-lab axes
(synthesis feasibility, E. coli expression, wet-lab validation — those old metrics
are retired). This runbook is the ready-to-run plan for the moment the training run
finishes and frees the H100.

Target run dir (v2, active): `/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768`
(the earlier v1 run `phase1_lora_prod_20260604_151651_L32768` is **superseded** — do
not eval it). Repo root (cwd for everything): `/home/ds85/projects/BCGModelling`
Env: `micromamba activate bgcmodel` (or `micromamba run -n bgcmodel ...`),
`export HF_HOME=/data2/ds85/hf_cache`

Active data (v2): `/data2/ds85/bgcmodel_data/splits_core/{train,val,test}.jsonl`
(strict antiSMASH **core** regions; train 47,524 / val 8,048 / test 18,871; 22
compound classes; native lowercase GTDB tags; leakage-clean). Earlier splits
(`splits_combined`, `splits_combined_grouped`, `splits_dedup`, `splits_curated`)
are deprecated — do not present them as current.

---

## The eval suite: CHECKS → QUESTIONS

Eval is two named layers (see `src/bgc_pipeline/evaluation.py` + `REDESIGN_PLAN.md`).
There is **no** `metric_1..metric_N` numbering anymore.

**CHECKS** (compute units; one consistent gene caller — pyrodigal/Prodigal — for all
ORF/gene calling):

| Check | What it computes | External dep |
|---|---|---|
| `coding_sanity` | gene-rich, complete coding DNA (vs degenerate junk), via pyrodigal | none |
| `antismash` | **gold-standard** BGC detection (`detected` → is_bgc) + classification (`class_match` → correct_class), ~3 s/core | antiSMASH 8 DBs |
| `class_markers` | per-class Pfam markers (data-driven, `derive_class_markers.py`); fast PROXY for antiSMASH class when antiSMASH is skipped | Pfam-A.hmm |
| `kmer_novelty` | anti-memorization k-mer containment vs the training corpus | training jsonl |
| `protein_homology` | MMseqs2 homology of predicted proteins to known enzymes | MMseqs2 DB |
| `module_architecture` | ordered NRPS/PKS assembly-line modules (from marker positions) | (uses class_markers) |
| `taxon_faithfulness` | codon/GC faithfulness to the **conditioned** taxon | taxon_profiles.json |
| `protein_foldability` (OPTIONAL) | ESMFold pLDDT — opt-in, GPU-expensive | ESMFold (GPU) |

**QUESTIONS** (derived verdicts via `derive_questions`):

| Question | Derived from | Role |
|---|---|---|
| `is_bgc` | `coding_sanity` ∧ `antismash.detected` (class_markers proxy if antiSMASH skipped) | **GATE** |
| `correct_class` | `antismash.class_match` (class_markers proxy if skipped) | **GATE** |
| `novel` | `kmer_novelty` | **GATE** |
| `proteins_plausible` | `protein_homology` | diagnostic |
| `complete` | `module_architecture` | diagnostic |
| `conditioning_faithful` | `taxon_faithfulness` | diagnostic |

**antiSMASH is the gold standard** for `is_bgc` + `correct_class` (recalibrated: on
real held-out cores, detection ~0.97 and correct_class ~0.97). The earlier ~0.15
correct_class was an incomplete product→class map; fixed by `scripts/build_class_map.py`,
which regenerates `config/compound_class_map.yaml` covering all 103 antiSMASH-8
products. `class_markers` (M2 pass = ANY class marker present, ~0.87 on real cores)
is the cheap PROXY used when antiSMASH is skipped (e.g. quick-eval without DBs).

**Headline tiers:** `generates_bgc` (is_bgc PASS) → `correct_class` →
`biological_valid` (is_bgc ∧ correct_class) → **ACCEPT** (∧ novel).

**Retired / removed entirely:** synthesis feasibility (old M4), Evo2 perplexity (old
M5), BiG-SCAPE coherence (old M6). E. coli expressibility is pruned from gating (kept
only as an informational sub-score for the conditioning experiment). The legacy
six-frame ORF finder is gone — pyrodigal is used everywhere (it no longer fragments
megasynthases; `requirements.txt` pins `pyrodigal>=3`).

---

## Run it (when training finishes + GPU is free)

### Step 0 — prerequisites

1. Confirm the v2 checkpoint is final and the GPU is idle (training released the
   H100), and `HF_HOME=/data2/ds85/hf_cache`.
2. Reference DBs that widen coverage (all optional — any check whose tool/DB is
   absent **self-skips**, it does not FAIL the run):
   - antiSMASH 8 DBs at `/data2/ds85/antismash_db` → real `is_bgc`/`correct_class`.
   - `Pfam-A.hmm` (hmmpressed) at `/data2/ds85/pfam/Pfam-A.hmm` → `class_markers`.
   - MMseqs2 protein DB (e.g. Swiss-Prot at `/data2/ds85/mmseqs_swissprot/swissprot`)
     → `protein_homology`; pass via `MMSEQS2_DB`.
3. Sanity-check the orchestrator parses:

   ```bash
   cd /home/ds85/projects/BCGModelling
   bash -n scripts/run_eval.sh
   ```

### Step 1 — run the full pipeline

Must be invoked **from the repo root** so the relative `POS=eval/positive_control_mibig.jsonl`
and the taxon-profile default `data/processed/taxon_profiles.json` resolve.

```bash
cd /home/ds85/projects/BCGModelling
export HF_HOME=/data2/ds85/hf_cache
VAL=/data2/ds85/bgcmodel_data/splits_core/val.jsonl \
REF=/data2/ds85/bgcmodel_data/splits_core/train.jsonl \
scripts/run_eval.sh \
  /data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768 eval_out
```

(`run_eval.sh` still ships older `VAL`/`REF` defaults; set them to the v2
`splits_core` paths as above so generation prompts and the novelty reference come
from the active data.)

`run_eval.sh` resolves the run dir → `checkpoints/best` (whose `adapter/` holds
`adapter_config.json` + `adapter_model.safetensors`) and runs 5 steps in order:

| Step | Script | GPU? | Output |
|---|---|---|---|
| 1 | `generate_bgc.py` | **GPU** | `eval_out/generated.{fasta,jsonl}` |
| 2 | `memorization_check.py` | CPU | `eval_out/memorization.jsonl` |
| 3 | `eval_conditioning_adherence.py` | **GPU** | `eval_out/adherence.json` |
| 4 | `conditioning_experiment.py` | **GPU** | `eval_out/conditioning.json` |
| 5 | `eval_suite_driver.py` → `evaluate_bgc` | CPU (foldability GPU, opt-in) | `eval_out/eval_suite.json` |

> The `/data2/.../eval/positive_control_mibig.jsonl` path does **not** exist; the
> real positive control is the repo-relative `eval/positive_control_mibig.jsonl`
> (the `POS` default). Running from any other cwd silently yields an empty positive
> control and no `taxon_faithfulness` / class-marker control verdicts — so always
> launch from the repo root.

### Quick per-checkpoint score (track functional progress during/between training)

For a fast checkpoint score (not the full suite), use `quick_eval.sh`. It generates
a small panel (module-bearing classes, fixed seed, full 32k window) and runs only the
**cheap** checks — `coding_sanity`, `antismash` (real is_bgc/correct_class, ~3 s/core),
`class_markers`, `module_architecture`, `taxon_faithfulness` — and **skips**
`protein_homology` (needs a DB) and `kmer_novelty` (needs the corpus scan). It appends
a row to `eval_track.jsonl`.

```bash
cd /home/ds85/projects/BCGModelling
scripts/quick_eval.sh \
  /data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768 quick_eval_out
```

> Generate at the full training window (`MAX_NEW=32768`, the default) on purpose: a
> complete obligate module is large (NRPS C-A-T ~3.5 kb, PKS KS-AT-ACP ~2.5 kb) and
> the model emits leading regulatory/intergenic sequence first. A short cap truncates
> before the module completes and produces SILENT `complete`/`class_markers` failures
> for a length reason, not a capability reason.

### Reading each output

- **`generated.{fasta,jsonl}`** — the generated BGCs. JSONL records:
  `{id: gen_XXXX_S, sequence, compound_class, taxonomic_tag, ...}`. Sanity:
  FASTA non-empty, some records hit the `|END|` EOS marker, sequences trimmed.
  This `id` is the join key into memorization → eval_suite.

- **`memorization.jsonl`** — anti-memorization input for the `novel` question (the
  `kmer_novelty` check). Per record:
  `{id, length, max_containment, max_estimate, nearest_accession, label, verdict}`.
  `max_containment` is max canonical-21-mer containment vs the nearest reference in
  the training corpus (`REF`). Verdict thresholds: `>=0.95` → **FAIL_memorized**,
  `>=0.80` → **WARN** (no_verdict), else **PASS_novel**. Want most generated records
  PASS_novel and the MIBiG positive controls PASS.

- **`adherence.json`** — `P(seq|class,taxon)` as a likelihood classifier over
  held-out val records. Read **top-1/3/5 accuracy, MRR, per-token margin,
  per-class recall**, and (with `--compare-base`) the **delta vs base Evo2**. This
  is the primary "conditioning works / correctly-classified" signal.

- **`conditioning.json`** — two causal controls. **CLASS control**: generate per
  class at a fixed taxon, recover the class via the likelihood classifier, read the
  confusion matrix vs the majority prior. **TAXON control**: generate the same class
  under E. coli vs the source taxon and compare GC / codon usage (via
  `taxon_faithfulness`) to test E. coli steering. Read whether class is recovered
  above prior and whether GC/codon usage shift toward the E. coli profile. (E. coli
  expressibility is informational here only — it is not a gate.)

- **`eval_suite.json`** — per-CHECK and per-QUESTION PASS / FAIL / no_verdict /
  skipped counts and pass-rates for generated vs positive control. Read the QUESTION
  layer first: the GATES `is_bgc`, `correct_class`, `novel`, then the diagnostics
  `proteins_plausible`, `complete`, `conditioning_faithful`. Any check whose
  tool/DB is absent shows as **skipped** (not FAIL). Confirm the novelty join landed
  by checking the `novel` entry count equals the generated count (not 0).

### Component-only commands (if you don't want the full chain)

```bash
# Novelty component standalone (CPU):
micromamba run -n bgcmodel python scripts/memorization_check.py \
  --query eval_out/generated.jsonl \
  --ref /data2/ds85/bgcmodel_data/splits_core/train.jsonl \
  --positive-control eval/positive_control_mibig.jsonl \
  --output eval_out/memorization.jsonl

# Eval suite standalone (CPU; pass DBs to enable antismash/class_markers/homology):
micromamba run -n bgcmodel python scripts/eval_suite_driver.py \
  --gen eval_out/generated.jsonl --positive eval/positive_control_mibig.jsonl \
  --novelty eval_out/memorization.jsonl \
  --antismash-db /data2/ds85/antismash_db --pfam-hmm /data2/ds85/pfam/Pfam-A.hmm \
  --output eval_out/eval_suite.json

# Same, but skip the slow / DB-bound checks (quick CPU pass):
micromamba run -n bgcmodel python scripts/eval_suite_driver.py \
  --gen eval_out/generated.jsonl --positive eval/positive_control_mibig.jsonl \
  --skip-checks protein_homology kmer_novelty \
  --output eval_out/eval_suite.json

# Generation standalone (GPU):
micromamba run -n bgcmodel python scripts/generate_bgc.py \
  --adapter /data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/best \
  --from-jsonl /data2/ds85/bgcmodel_data/splits_core/val.jsonl \
  --per-class 1 --max-new-tokens 2048 --out-fasta /tmp/gen.fasta --out-jsonl /tmp/gen.jsonl

# Adherence + conditioning standalone (GPU) — NOTE: adherence uses --val:
micromamba run -n bgcmodel python scripts/eval_conditioning_adherence.py \
  --val /data2/ds85/bgcmodel_data/splits_core/val.jsonl \
  --adapter /data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/best \
  --compare-base --per-class-cap 20 --score-len 8192 --output adherence.json \
&& micromamba run -n bgcmodel python scripts/conditioning_experiment.py \
  --adapter /data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/best \
  --experiment both --output conditioning.json
```

---

## Readiness matrix

| Component | Status | What it verifies | Run command |
|---|---|---|---|
| Generation (`generate_bgc.py` + `evo2_inference.load_evo2_wrapper_for_inference`) | **needs-GPU+checkpoint** | Loads Evo2 7B + LoRA, builds `\|COMPOUND_CLASS:{cls}\|{tax}` prefix, samples, trims at `\|END\|`, writes FASTA+JSONL feeding the suite. | `scripts/generate_bgc.py --adapter .../checkpoints/best --from-jsonl .../splits_core/val.jsonl --per-class 1 --max-new-tokens 2048 --out-fasta /tmp/gen.fasta --out-jsonl /tmp/gen.jsonl` |
| Novelty / memorization (`memorization_check.py` + `kmer_novelty` → `novel`) | **ready-now** (CPU) | Max canonical-21-mer containment vs leakage-free `train.jsonl`; gates FAIL≥0.95 / WARN≥0.80 / PASS_novel. | standalone `memorization_check.py ...` then `eval_suite_driver.py ...`; full chain via `scripts/run_eval.sh` |
| Adherence (`eval_conditioning_adherence.py`) + conditioning (`conditioning_experiment.py`) | **needs-GPU+checkpoint** | Likelihood-classifier top-k/MRR/margin/recall (+base delta); CLASS & TAXON causal controls incl. GC/codon E. coli steering. | `eval_conditioning_adherence.py --val ... --adapter ... --compare-base` && `conditioning_experiment.py --adapter ... --experiment both` |
| Eval suite (`src/bgc_pipeline/evaluation.py` + `eval_suite_driver.py`) | **ready-now** (CPU; foldability opt-in GPU) | `evaluate_bgc` over generated vs positive control; CHECKS → QUESTIONS aggregation. Checks whose DB/tool is absent self-skip. | `eval_suite_driver.py --gen .../generated.jsonl --positive eval/positive_control_mibig.jsonl --novelty .../memorization.jsonl --antismash-db ... --pfam-hmm ... --output .../eval_suite.json` |
| Orchestrator `scripts/run_eval.sh` (5-step chain) | **ready-when-GPU-free** | run-dir→checkpoints/best resolution, 5 steps, id/novelty join. Pass v2 `splits_core` paths via `VAL`/`REF`. | `VAL=.../splits_core/val.jsonl REF=.../splits_core/train.jsonl scripts/run_eval.sh /data2/.../phase1_lora_prod_20260617_095202_L32768 eval_out` |
| Dependencies + reference data | **present** | Adapter, `splits_core/{val,train}.jsonl`, positive control, `taxon_profiles.json` all present; antiSMASH/Pfam/MMseqs2 DBs optional (self-skip if absent). | `bash -n scripts/run_eval.sh`; data presence checks |

Status legend: **ready-now** = runs on CPU today; **needs-GPU+checkpoint** = code
verified, waiting on the freed H100 + final checkpoint; **ready-when-GPU-free** =
orchestrator is correct, just needs the GPU released by training.

---

## Coverage of the in-scope questions

| Question (role) | Check(s) | Runnable now? |
|---|---|---|
| **is_bgc** (GATE) | `antismash.detected` (primary); `coding_sanity`; `class_markers` proxy | antiSMASH = needs DBs at `/data2/ds85/antismash_db` (else falls back to class_markers proxy). With DBs, real verdicts in ~3 s/core. |
| **correct_class** (GATE) | `antismash.class_match` (primary); `class_markers` proxy | antiSMASH gold-standard (correct_class ~0.97 after the product→class map fix). class_markers (~0.87) is the proxy when antiSMASH skipped. |
| **novel** (GATE) | `kmer_novelty` (k-mer containment vs training) | **runnable now on CPU.** `protein_homology` adds depth once an MMseqs2 DB is wired. |
| **proteins_plausible** (diagnostic) | `protein_homology` (MMseqs2 vs known enzymes) | needs an MMseqs2 protein DB (`MMSEQS2_DB`); self-skips otherwise. |
| **complete** (diagnostic) | `module_architecture` (ordered NRPS/PKS modules) | **runnable now on CPU** (uses class_markers positions; needs Pfam-A.hmm for the marker calls). |
| **conditioning_faithful** (diagnostic) | `taxon_faithfulness` (codon/GC vs conditioned taxon) | **runnable now on CPU** (needs `taxon_profiles.json`, present). |

ACCEPT is the conjunction of the three gates (`is_bgc ∧ correct_class ∧ novel`); the
diagnostics characterize quality but do not gate. Wet-lab axes (synthesizability,
E. coli expression, wet-lab validation) are **out of scope** for this suite.

---

## Calibration & data-build scripts (provenance)

- `scripts/build_class_map.py` — regenerates `config/compound_class_map.yaml`
  (all 103 antiSMASH-8 products → compound class); fixes the old correct_class undercount.
- `scripts/calibrate_antismash.py` + `scripts/validate_antismash_calibration.py` —
  antiSMASH `is_bgc`/`correct_class` recalibration + validation on real cores.
- `scripts/derive_class_markers.py` + `scripts/validate_m2_calibration.py` —
  data-driven per-class Pfam marker derivation + validation (the `class_markers` proxy).
- `scripts/build_taxon_profiles.py` → `data/processed/taxon_profiles.json` — codon/GC
  reference profiles for `taxon_faithfulness`.

---

## Known limitations carried in

- **`novel` is a hard anti-memorization floor, not a calibrated novelty score.**
  `kmer_novelty` gates on max canonical-21-mer containment (exact-verify after MinHash
  candidates). It catches exact and fragment-of-reference memorization, but
  near-duplicate (high-identity-but-not-byte-identical) calibration is not yet tuned,
  and the WARN band (0.80–0.95 → no_verdict) is a placeholder. `protein_homology`
  (MMseqs2) would add protein-homology depth once a DB is built/wired.

- **E. coli chassis compatibility is informational, not a gate.** `taxon_faithfulness`
  grades faithfulness vs the *conditioned* taxon and is no longer hardcoded to E. coli;
  E. coli expressibility is reported only as a sub-score in the conditioning experiment.
  The TAXON control in `conditioning.json` probes whether conditioning steers GC/codon
  usage toward the E. coli profile — read it as a hypothesis under test, not a guarantee
  of expression, and not part of ACCEPT.

- **Checks self-skip without their DB/tool.** `antismash` (DBs), `class_markers`
  (Pfam-A.hmm), `protein_homology` (MMseqs2 DB), and `protein_foldability` (ESMFold,
  opt-in GPU) each self-skip when their dependency is absent — they record **skipped**,
  not FAIL. When antiSMASH is skipped, `is_bgc`/`correct_class` fall back to the
  `class_markers` proxy consistently. Install the DBs (Step 0) for authoritative
  gate verdicts.

- **Positive-control length caveat.** `eval/positive_control_mibig.jsonl` mixes short
  fragments with full BGCs. The detection/class checks are only meaningful on
  full-length controls — a short fragment legitimately lacks a complete module and may
  FAIL by design. When reading positive-control pass-rates, weight the long records;
  do not read a short fragment's FAIL as a suite defect.

- **`conditioning_experiment` taxon prefix is slightly off-distribution.** Its
  hardcoded `ECOLI_TAXON` / `STREPTO_TAXON` start with `|` but lack the **trailing**
  `|` that training_text uses (`|COMPOUND_CLASS:RIPP||D__...;S__GRIMESII|`). It does
  not crash, and CLASS/TAXON deltas remain valid because both arms share the prefix
  convention, but absolute generation quality in that control may be mildly understated.
  The adherence script is unaffected (it derives the taxon from val records, which
  retain the trailing `|`). Optional fix: append a trailing `|` to those constants.

- **GPU-gated paths are code-verified, not yet executed against the v2 adapter.**
  vortex `cached_generation` sampling, `merge_and_unload` footprint on one H100, and
  chained multi-window continuation/EOS are signature/parity/unit-tested. On first GPU
  availability, run a `--per-class 1 --max-new-tokens 2048` smoke (non-empty FASTA,
  some `hit_eos`, no OOM) before the full sweep, and exercise `--max-windows>1` for
  long classes. `quick_eval.sh` defaults to sequential generation (`GEN_BATCH_SIZE=1`)
  unless the on-GPU batched-equivalence gate has written the decision file.

- **`best/` is overwritten in place as training proceeds.** `.../checkpoints/best/adapter`
  is a real dir (not a symlink), so the eval uses whatever the best checkpoint is at
  run time. Confirm the v2 checkpoint is final before launching.

- **Run from the repo root.** `POS`, `--positive`, and the taxon-profile defaults
  are repo-relative; wrong cwd silently drops the positive control and the
  `taxon_faithfulness` verdicts with no error. The `/data2/.../eval/positive_control_mibig.jsonl`
  path in some briefs does **not** exist.
