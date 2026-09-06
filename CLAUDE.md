# BCGModelling — Agent Contract

## Project Core

* **Mission:** Fine-tune a genome foundation model (Evo2 / GenomeOcean) to *generate* biosynthetic
  gene clusters, and measure honestly whether it can. Negative results are results; every rate is
  quoted against a measured ceiling (real cores) and floor (real non-BGC DNA).
* **Environment:** `gputee`, single H100 80 GB, shared host. IU Quartz for multi-GPU long-context.
  Data and runs live on `/data2`, never in the repo.
* **Stack:** `micromamba activate bgcmodel` (torch 2.5.1+cu124, transformers 4.46.3, antiSMASH
  8.0.4 + Pfam); GenomeOcean has its own env (`data.md`). Bash/tmux/HMMER/MMseqs2 matter as much as
  Python — YOU MUST NOT assume a task is Python.

## Documentation Architecture

* `CLAUDE.md` — **auto-loaded every session.** This contract. **Zero findings.**
* `plan.md` — **read at session start.** Current state, active interventions, phase ledger.
* `terms.md` — **search before naming any metric.** Definitions + provenance of every metric.
* `data.md` — **read before touching data, runs or paths.** Datasets, schemas, splits, run registry.
* `memory.md` — **never read on startup; `grep` it.** Chronological ledger of results + decisions.
* `bugs.md` — **`grep` by symptom.** `[Symptom] → [Proven fix]`, indexed by subject.
* `paper.md` — **FRAMING ONLY, zero results** (7th doc, user-authorised 2026-08-24). Every
  claim in it must cite a `memory.md` entry; a claim with no ledger entry is not a claim.

## Standing Constraints (hard rules; rationale lives in `memory.md`)

1. **Novelty gates every rung, never co-reported** — every ladder metric is maximised by copying.
   **Both gates apply: `containment` AND `protein_aai`** — DNA containment is blind to
   protein-level reconstruction, so it can pass a memorising arm on its own.
2. **Live datasets: `splits_core/`, `splits_class/<CLASS>/`, `splits_class_wide/<CLASS>/`** —
   DEPRECATED (`data.md`) is off-limits; `splits_combined/` leaked (94.6% genome overlap).
3. **GenomeOcean-4B is THE substrate for all new work** (user, 2026-08-24). `evo2_1b_base` is
   retained ONLY as the comparison case until the final picture is drawn — never as the default,
   never as the arm a new phase opens on. Testing does not fan out across models. ⚠️ Evo2's context
   is **8,192, a hard model limit**; GenomeOcean's is **10,240 tokens ≈ 50,934 nt** (`data.md`).
4. **Pre-registered endpoints do not change mid-phase** (`docs/phase3_preregistration.md`).
5. **A null is interpretable only if the test was powered AND the intervention verified to have
   landed** — otherwise the result is "uninformative", not "negative".
6. **MiBIG stays held out.** Reserved for a later compound-conditioned fine-tune.
7. **Never MIX antiSMASH and `class_markers` proxy numbers in one comparison** — the proxy inflates
   substantially (`terms.md`). This is not a reason to skip antiSMASH: **run it, report it as its
   own row.**
8. **`correct_class` under a class-specific adapter is MEASURED and NON-ZERO** — "UNMEASURED" and
   "~0 de novo" are both superseded; "~0" held for LABEL conditioning only (`memory.md`).

9. **A validation holds only in the REGIME it was measured in.** Any discrimination score,
   threshold or calibration must be re-derived when the scoring set, window, seed length, substrate
   or generation path changes; the Phase-2 ladder rungs did not survive re-testing (`memory.md`).

## Agent Behavior & Prohibitions

* **Verify before acting.** Never guess paths, shapes, counts or splits — use `ls`/`grep`/`data.md`.
* **No sweeping changes.** No global `sed` or multi-file refactors without permission.
* **Strict documentation limits.** Do not create new doc files; the six above are the set. ASK
  FIRST with a justification if you believe one is needed.
* **Prevent definition drift.** Search `terms.md` before defining a metric, writing a pipeline or
  labelling a table row; use the established name exactly. Never invent synonyms.
* **Report the failure, not the workaround.** A missing tool/resource/gate must never silently
  become a negative result — `BGC_EVAL_STRICT` enforces this in code; hold to it in prose.

## IMPORTANT: Results Reporting Format

**Metrics are ROWS. Experiments/arms are COLUMNS.** Never the transpose — a wide table of arms
invites cherry-picking which metrics to show, and the same metric must be visible in every report
to be comparable across them.

1. **Report THE PHASE-3 REPORTING SET in full, every time** (`terms.md`) — every metric, every arm,
   including the ones that did not move. Never omit a row; print `n/a` with a reason.
2. **PRIMARY = the Stage-A RATE of positives. EVERY other metric is STAGE B — positives only**
   (`terms.md`; user, 2026-08-24). Post-generation filtering discards negatives, and a Stage-A
   ladder value only re-measures the hit rate. Label stage and n; every Stage-B number needs a
   real-core reference **on its own positives**, quoted as a ratio to it.
3. **Row labels are the exact `terms.md` identifier** — snake_case, no prose synonyms. Not "bio
   bits" — `best_bio_bits`.
4. **Order rows by importance:** primary endpoint → novelty gates → cluster structure → context →
   demoted/diagnostic. Mark every gate metric with `*` after the identifier.
5. **Carry a provenance line** — checkpoint · generation set · n · scoring config · window. A number
   without provenance is not a result.
6. **State the ceiling (real cores) and floor (base / non-BGC)** as their own columns, so every rate
   is read against both without the reader hunting for them.
7. **EVERY table is followed by THREE things, in this order, as BULLET LISTS — never prose:**
   **(a) COLUMNS** — one bullet per column header: what that arm *is* and how it differs from the
   others. **(b) ROWS** — one bullet per metric: what it measures, which direction is good, how to
   read *this* value against its reference. Every row, every time, including ones that did not
   move. **(c) SYNTHESIS** — prose. Omitting (a) or (b) makes the table unreadable to anyone who
   did not build it.
8. **EVERY NUMBER IN PROSE MUST BE TRACEABLE TO A TABLE CELL.** If a figure is not in the table
   directly above, name its source — which arm, which window, which table. Never put values scored
   under different configs in one sentence without saying so; quoting an 8 kb-window `n_orfs`
   beside a 2 kb-window table is exactly the confusion this prevents. Prefer adding the number to
   the table over explaining it in text.
9. **EXPLAIN WHAT YOU DID, AND HOW IT DIFFERS FROM WHAT WAS DONE BEFORE.** Every results report
   opens with a plain-language **METHOD** paragraph — what was actually built or changed, in
   sequence — followed by **WHAT'S DIFFERENT**: an explicit contrast against the closest prior arm,
   naming the run dir it is being contrasted with and every knob that moved (substrate, data,
   conditioning, adapter config, generation path, scoring). A number is uninterpretable when the
   reader has to infer the intervention from the table. ⛔ **"Same recipe as before" is not
   acceptable** — state the recipe and state the delta, every time, even when the delta is one flag.
10. Two arms are comparable only if their `scoring` stamps match on Pfam subset, window, substrate,
   generation path and regime. Emit with `scripts/novelty_battery.py`; never hand-assemble.
   ⚠️ **THE SCORING WINDOW IS RETIRED (2026-08-24) — score the FULL sequence.** It was an Evo2-era
   fix for a model with no working EOS; with GenomeOcean it truncates the REFERENCE, not the arm.
   **Every Phase 3–9 number was windowed and is NOT comparable to a full-length one** — re-derive
   the reference, never carry it over.

## IMPORTANT: Experiment ID Convention

**`P<phase>-<KIND>-<slug>`**, or **`INF-<KIND>-<slug>`** for cross-phase tooling. `<KIND>` is exactly
one of **`DAT` `TRN` `GEN` `EVL` `ANL` `FIX`**, so an ID says what the work is and what it costs
without a lookup; `<slug>` is lowercase-hyphenated (`azole`, never `A0`). This replaced five
colliding schemes. ⚠️ **`memory.md` IDs are FROZEN** — the ledger is permanent, so the new scheme
covers **new and queued work only**; read old entries through the bridge table in `terms.md`, never
rewrite them. Run directories and filenames are the SEPARATE convention below and are not renamed.

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
* **Fan out when the batched path is gated** (Evo2 generation is one-at-a-time). N *sequential*
  processes on disjoint units — semantics unchanged, outputs identical. **N is not a constant —
  MEASURE IT**; fan-out helps only below ~100% utilisation, and past saturation throughput FALLS.
  Re-measure whenever model, length, batch shape or host changes, and **check whose load it is**
  (shared host — `nvidia-smi --query-compute-apps`). `scripts/fanout.sh <N> <claim_dir> <unit_file>
  '<cmd with {}>'`; give each shard a **distinct `--seed`** or they write identical output.
  **Not for throughput/memory benchmarks.**
* **Any fan-out MUST assert its own completeness and fail loudly.** A script that filters results
  and reports a count cannot tell "found nothing" from "ran nothing" (`bugs.md`). Assert the
  expected count and throw.
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

1. **Archive to memory.** Hypothesis, method, provenance, results → `memory.md` under today's date.
2. **Compress to the ledger.** One-row summary in `plan.md`'s Phase Ledger; it stays for the phase,
   so the next session need not grep `memory.md` for it.
3. **Document fixes.** Append any `[Symptom] → [Fix]` to `bugs.md` under its subject heading.
4. **Define new terms.** Any new metric/variant/pipeline step gets a full `terms.md` entry, tagged.
5. **Register artifacts.** Every new checkpoint, generation set or dataset gets a `data.md` row.
6. **Reset the board.** Rewrite `plan.md`'s Current State, add the Phase-Ledger row, and **bump
   `Last updated`** — the verifier fails if it predates the newest `memory.md` entry.
7. **Verify.** `python tests/test_docs_contract.py` must pass before the session ends.
