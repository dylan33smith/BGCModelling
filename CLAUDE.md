# BGC-BENCH — Agent Contract

## What this branch is

A **clean-room rebuild**. `docs/SPEC.md` is the single source of truth and the sole input to
implementation. This branch is an **orphan** — it shares no history with `main` or `phase3-ripp`.

## THE BLIND RULE — the reason this branch exists

**YOU MUST NOT read the prior implementation of any component you are building.**
The prior codebase lives on `main` and `phase3-ripp` (~26k lines, ~150 scripts). It is a
**reference standard for later comparison, not an input**.

* Do NOT `git show main:...` or `git show phase3-ripp:...` for any file under §3–§9 of the spec.
* Do NOT consult prior results, rates, or hypotheses. They enter only at §10 reconciliation.
* If the spec is missing something you need, **amend the spec** — never resolve it by reading old
  code. Amendments are dated in §12.

Permitted without restriction: published library APIs (evo2, transformers, antiSMASH, mmseqs2),
public documentation, and the raw data on `/data2`.

## The three-step protocol (spec §10)

1. **Build blind.** 2. **Run, record, freeze.** 3. **Unblind and diff** against the prior
codebase; adjudicate every discrepancy in writing to a cause.
Step 3 is a deliverable. Freezing before comparison is what makes the diff evidence.

## Standing constraints

1. **antiSMASH is the sole endpoint instrument** (§3.1). No Pfam/HMM score, no learned probe, is
   ever an endpoint.
2. **Model input is bare nucleotide sequence** (§4.3). No class tag, no taxonomy tag, no metadata.
3. **One dataset exists** (§4.4.3). Equal effective_n is fixed at split time; arms cannot differ in
   training data because no other dataset exists.
4. **Every sequence submitted to antiSMASH gets a verdict.** A dropped record is not a
   non-detection. `--minlength 1`.
5. **Novelty is a gate on every arm**, applied before any rate is read (§3.8). It fails closed.
6. **A null requires power AND a passing manipulation check** (§6.4), or it is reported as
   uninformative, not negative.
7. **No arm is optimised against the benchmark endpoint** (§2.4). Rank sweeps read held-out loss.

## Filesystem

* Artifacts live on `/data2`, never in the repo.
* Run dirs: `<stage>_<SUBSTRATE>_<ARM>_<CLASS>`. Two names differing only in case are the same name.
* Nothing loose at a run root; every artifact belongs to a run directory.

## Documentation

`docs/SPEC.md` is the spec. `reference/frozen/KNOWN_WRONG.md` records prior defects with a red test
each. Do not create other documentation files without asking.
