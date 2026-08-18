# BCGModelling — Agent Contract

## Project Core

* **Mission:** Fine-tune a genome foundation model (Evo2 / GenomeOcean) to *generate* biosynthetic
  gene clusters, and measure honestly whether it can. Negative results are results; every rate is
  quoted against a measured ceiling (real cores) and floor (real non-BGC DNA).
* **Environment:** `gputee`, single H100 80 GB, shared host. IU Quartz for multi-GPU long-context.
  Data and runs live on `/data2`, never in the repo.
* **Stack:** `micromamba activate bgcmodel` (torch 2.5.1+cu124, transformers 4.46.3, antiSMASH
  8.0.4 + Pfam). GenomeOcean uses a separate env: `micromamba run -p /data2/ds85/envs/genomeocean`.
  Bash/tmux/HMMER/MMseqs2 matter as much as Python here — YOU MUST NOT assume a task is Python.

## Documentation Architecture

* `CLAUDE.md` — **auto-loaded every session.** This contract. **Zero findings.**
* `plan.md` — **read at session start.** Current state, active interventions, phase ledger.
* `terms.md` — **search before naming any metric.** Definitions + provenance of every metric.
* `data.md` — **read before touching data, runs or paths.** Datasets, schemas, splits, run registry.
* `memory.md` — **never read on startup; `grep` it.** Chronological ledger of results + decisions.
* `bugs.md` — **`grep` by symptom.** `[Symptom] → [Proven fix]`, indexed by subject.

## Standing Constraints (hard rules; rationale lives in `memory.md`)

1. **Novelty gates every rung, never co-reported** — every ladder metric is maximised by copying.
   **Both gates apply: `containment` AND `protein_aai`** — DNA containment is blind to
   protein-level reconstruction, so it can pass a memorising arm on its own.
2. **Live datasets: `splits_core/`, `splits_class/<CLASS>/`, `splits_class_wide/<CLASS>/`** —
   DEPRECATED (`data.md`) is off-limits; `splits_combined/` leaked (94.6% genome overlap).
3. **The 1B is the Phase-3 testing substrate**; the 7B confirms. Testing does not fan out across
   models. ⚠️ `evo2_1b_base` context is **8,192 — a hard model limit**, not a config choice; it
   constrains substrate design (see `data.md`).
4. **Pre-registered endpoints do not change mid-phase** (`docs/phase3_preregistration.md`).
5. **A null is interpretable only if the test was powered AND the intervention verified to have
   landed** — otherwise the result is "uninformative", not "negative".
6. **MiBIG stays held out.** Reserved for a later compound-conditioned fine-tune.
7. **Never MIX antiSMASH and `class_markers` proxy numbers in one comparison** — the proxy inflates
   substantially (`terms.md`). This is not a reason to skip antiSMASH: **run it, report it as its
   own row.**
8. **`correct_class` under a class-specific adapter is UNMEASURED, not zero** (rewritten
   2026-08-18; the pre-Phase-3 rule asserted it reads ~0 de novo). Measure it before treating it
   as either a target or a dead end — see `memory.md`.

## Agent Behavior & Prohibitions

* **Verify before acting.** Never guess paths, shapes, record counts, or splits — use `ls`/`grep`
  or `data.md`. A path that "should" exist has repeatedly not.
* **No sweeping changes.** No global `sed` or multi-file refactors without permission.
* **Strict documentation limits.** Do not create new doc files; the six above are the set. ASK
  FIRST with a justification if you believe one is needed.
* **Prevent definition drift.** Search `terms.md` before defining a metric, writing a pipeline or
  labelling a table row; use the established name exactly. Never invent synonyms.
* **Report the failure, not the workaround.** A missing tool/resource/gate must never silently
  become a negative result (`BGC_EVAL_STRICT` enforces this in code; hold to it in prose).

## IMPORTANT: Results Reporting Format

**Metrics are ROWS. Experiments/arms are COLUMNS.** Never the transpose — a wide table of arms
invites cherry-picking which metrics to show, and the same metric must be visible in every report
to be comparable across them.

1. **Report THE PHASE-3 REPORTING SET in full, every time** (`terms.md`) — every metric, every arm,
   including the ones that did not move. Never omit a row; print `n/a` with a reason.
2. **Label every number with its MEASUREMENT STAGE and n** (`terms.md`, THE TWO MEASUREMENT STAGES).
   **Stage A** = all generated sequences (selection). **Stage B** = positives only
   (characterisation). Quoting a Stage-B metric over all sequences yields a different, usually wrong
   quantity — that error produced a retraction. Every Stage-B number needs a real-core reference.
3. **Row labels are the exact `terms.md` identifier** — snake_case, no prose synonyms. Not "bio
   bits" — `best_bio_bits`.
4. **Order rows by importance:** primary endpoint → novelty gates → cluster structure → context →
   demoted/diagnostic. Mark every gate metric with `*` after the identifier.
5. **Carry a provenance line** — checkpoint · generation set · n · scoring config · window. A number
   without provenance is not a result.
6. **State the ceiling (real cores) and floor (base / non-BGC)** as their own columns, so every rate
   is read against both without the reader hunting for them.
7. **Follow the table with a PER-METRIC reading, then a SYNTHESIS.** One line per row: what it
   measures, which direction is good, and how to read *this* table's value against its reference.
   Then: what the rows say together, what the table does not show, and which comparison is
   load-bearing. A table without both is data, not a result.
8. Two arms are comparable only if their `scoring` stamps match on Pfam subset, window, substrate,
   generation path and regime. Emit with `scripts/novelty_battery.py`; never hand-assemble.

## IMPORTANT: Filesystem Naming Convention

A name that cannot be grepped, or that collides, is a data-integrity bug — this repo has already
lost a 698 MB run dir to a case collision and ~6,600 training records to a manifest overwrite.

* **Run directories:** `<phase>_<TARGET>[_<arm>]` — `phase3_RIPP_A0`, `phase2_1b_weighted`.
  Phase lowercase, **class UPPERCASE exactly as in `OBLIGATE_DOMAINS`**, arm lowercase.
* **YOU MUST NOT create two names differing only in case.** `phase3_RIPP` and `phase3_ripp` are
  the same name. Check with `ls | sort -f | uniq -di` before creating a run dir.
* **YOU MUST NOT write loose files to `/data2/ds85/bgcmodel_runs/`.** Everything belongs inside a
  run directory. A result at the root has no owning experiment and becomes unattributable.
* **Inside a run dir:** `adapter_run/` (weights) · `train.log` · `<arm>.jsonl` (generations) ·
  `<arm>_w<window>.json` (windowed scores) · `<arm>_gen.log`. Window size goes **in the filename**
  — two scorings of one generation set must never share a name.
* **Datasets:** `splits_<scope>/` or `splits_class/<CLASS>/`. Every per-class split MUST have a
  `manifest.json` entry; the builder **overwrites the whole manifest**, so rebuilding one class
  drops the others unless you rebuild all of them.
* **No brace/glob shorthand in docs.** Write `run_a/`, `run_b/` — not `run_{a,b}/`. Shorthand is
  not greppable and the verifier cannot check it.
* **Deprecated data is renamed, not left in place:** prefix `DEPRECATED_`, or delete it. Record
  either in `data.md`.

## Execution & Long-Running Tasks

* **Parallel execution.** Anything over ~60 s (training, generation, antiSMASH, MMseqs2) runs in a
  detached `tmux` session.
* **GPU jobs go through the queue wrapper**, never raw `python train.py` — contention silently
  invalidates memory/throughput measurements.
* **Status sentinel, not desktop notifications** (`notify-send` no-ops on this headless host).
  Poll the sentinel (`0` = success); never report a run finished without reading it:
  `tmux new-session -d -s <n> '<cmd> > logs/<n>.log 2>&1; echo $? > logs/<n>.status'`
* **Read synchronous results.** After a fast command, read and summarize the output unprompted.
* **Fan out when the batched path is gated.** Generation is one-sequence-at-a-time (vortex batching
  is gated, `bugs.md`), which leaves the H100 at ~41% util / 4 GB of 80 GB. Run N *sequential*
  processes on disjoint units instead: semantics are unchanged (each still generates serially, so
  outputs are identical). **N is not a constant — MEASURE IT.** Check `nvidia-smi` utilisation
  first: fan-out only helps while the GPU is *under-utilised*. Once utilisation is ~100%, more
  workers time-slice a saturated device and aggregate throughput FALLS. Verify by counting outputs
  over a ~3 min window and comparing seq/hour against the previous N; if it did not rise, back off.
  **Re-measure whenever the model, sequence length, batch shape or host changes** — a 7B, longer
  generations or a busier host all move the optimum. *Measured 2026-08-17 for 1B generation at
  2.2 kb on an idle H100:* N=1 → 124 seq/h (41% util), **N=3 → 432 seq/h (100% util)**,
  N=5 → ~300 seq/h (regression). Treat that as a starting point, not a rule.
  `scripts/fanout.sh <N> <claim_dir> <unit_file> '<cmd with {}>'` — claims units atomically with
  `mkdir`, one tmux session + status sentinel per worker. Write to `<out>.partial` and `mv` on
  success so an interrupted unit never looks complete. **Not for throughput/memory benchmarks** —
  contention is exactly what invalidates those.
* ⚠️ **Never `pkill -f <pattern>` where the pattern can match your own command line.** `pkill -f
  seed_generate.py` issued from a shell whose command string contains that text kills the issuing
  shell. Kill by PID from `ps -eo pid,cmd`.

## IMPORTANT: The In-Place Memory Correction Rule

`memory.md` is a permanent ledger. YOU MUST NEVER delete or overwrite a historical entry. When an
old assumption, decision, or result is proven wrong:

1. Locate the exact original line.
2. Prepend `[INCORRECT] - ` to it, preserving the text.
3. Insert directly below: `[CORRECTION - YYYY-MM-DD]: ` with the new finding and what changed.

This project has already retracted a leakage artefact, inverted a Phase-3 result on a scoring bug,
and half-closed a track whose treatment never landed. Preserving the wrong version is what makes
those legible later.

## YOU MUST EXECUTE: The Wrap-Up Protocol

When an intervention, script, or experiment is confirmed complete:

1. **Archive to memory.** Append hypothesis, method, provenance, and results to `memory.md` under
   today's date. Full prose lives here.
2. **Compress to the ledger.** Add or update the one-row summary in `plan.md`'s Phase Ledger. The row
   stays for the rest of the phase — do not make the next session grep `memory.md` for it.
3. **Document fixes.** Append any `[Symptom] → [Fix]` to `bugs.md` under its subject heading.
4. **Define new terms.** Any new metric, variant, or pipeline step gets a `terms.md` entry with the
   full schema (Is / Computed by / Changes meaning with / Valid vs / Status). Tag it.
5. **Register artifacts.** Any new checkpoint, generation set, or dataset gets a `data.md` row.
6. **Reset the board.** Rewrite `plan.md`'s Current State paragraph and queue the next task.
7. **Run the verifier.** `python tests/test_docs_contract.py` must pass before the session ends.
