# MIGRATION — cutting over to the six-file framework

**Status: STAGED. Nothing existing has been modified.** Every file in `docs/framework/` is new.
The live `CLAUDE.md`, `README.md`, and `docs/project_memory/*` are untouched.

**Revert point:** tag `pre-docs-overhaul` (commit `e78ee31`), pushed to origin.
```bash
git reset --hard pre-docs-overhaul
```

---

## What was built

| Draft | Lines | Replaces | Notes |
|---|---|---|---|
| `docs/framework/CLAUDE.md` | 107 | `CLAUDE.md` (362) | governor only, **zero findings** |
| `docs/framework/plan.md` | 129 | `progress.md` NEXT ACTIONS | dashboard + Phase Ledger |
| `docs/framework/terms.md` | 269 | *(new)* | 24 entries, provenance schema |
| `docs/framework/data.md` | 216 | scattered | datasets + schema + **41-run registry** |
| `docs/framework/memory.md` | 223 | `progress.md` + `decisions.md` | ledger from 2026-08-12 forward |
| `docs/framework/bugs.md` | 686 | `bugs.md` (664) | **existing content verbatim** + new header |
| `tests/test_docs_contract.py` | 739 | *(new)* | 24 checks, currently **0 failed** |

**Session-start context cost: ~37k tokens → ~2.5k.** Today `CLAUDE.md` (22 KB) auto-loads and the
memory protocol says read `progress.md` (126 KB) first. After cutover `CLAUDE.md` is ~6 KB and
`plan.md` ~7 KB; `memory.md`, `bugs.md`, and the archive are grep-on-demand.

---

## Findings surfaced while building this

These are real problems the drift was hiding. None are fixed yet — they are logged in
`plan.md` [P3-B4] and flagged in `data.md`.

1. **`progress.md:874` says the Phase-3 target is TERPENE.** `CLAUDE.md` says RIPP. RIPP is
   correct — it is what A0 trained on and the only class with a built adapter. Two most-read docs,
   contradicting each other, on the single most important fact in the current phase.
2. **`phase3_RIPP/` and `phase3_ripp/` differ only in case** and hold different things (the A0
   adapter vs the pilot baselines). One careless `rsync`/`tar` merges and destroys them.
3. **`HSERLACTONE/` and `BUTYROLACTONE/` splits are orphaned.** Built 11:51–11:52 on 2026-08-14;
   `manifest.json` was rewritten at 12:34 covering only RIPP/PKS/TERPENE, dropping them. Their
   leakage controls are unverified from the record. ~6,600 training records with no provenance.
4. **Four `splits_core/` derived files were undocumented** — `train.domain_spans.jsonl`,
   `valtest_fit.jsonl`, `valtest_eval.jsonl`, `valtest_eval_4class.jsonl`.
5. **Four ladder metrics the code computes had no definition anywhere** — `n_orfs`, `co_orient`,
   `modules`, `in_order`. Found by the verifier, not by reading. `modules`/`in_order` are
   hard-coded to 0 for every non-assembly-line class **including RIPP**, so a 0 there is not a
   measurement — a trap for exactly the current phase.
6. **Two code-key/doc-name drifts:** `bio` → `best_bio_bits`, `frac` → `biosynthetic_fraction`.
   Both now documented as aliases and pinned by the verifier.

---

## Cutover procedure

Run from a clean tree on `docs-framework`. Steps 1–7 are one commit; verify between each.

### 1. Archive the pre-framework docs (do not delete)
```bash
mkdir -p docs/archive/pre-framework
git mv docs/project_memory/progress.md  docs/archive/pre-framework/progress.md
git mv docs/project_memory/decisions.md docs/archive/pre-framework/decisions.md
git mv docs/project_memory/bugs.md      docs/archive/pre-framework/bugs.md
```

### 2. Move the drafts into place
```bash
git mv docs/framework/plan.md   docs/plan.md
git mv docs/framework/terms.md  docs/terms.md
git mv docs/framework/data.md   docs/data.md
git mv docs/framework/memory.md docs/memory.md
git mv docs/framework/bugs.md   docs/bugs.md
git mv docs/framework/CLAUDE.md CLAUDE.md          # overwrites the 362-line governor
```

### 3. Fix the archived stale claim in place
Per the in-place correction rule, do **not** edit the archived line silently:
```
[INCORRECT] - ...the target is TERPENE, not ectoine
[CORRECTION - 2026-08-14]: The target is RIPP. TERPENE was the target for ~2 hours on
2026-08-14 before the diversity audit; A0 trained on RIPP and RIPP is what splits_class holds.
```

### 4. Repoint references in code and docs
`docs/project_memory/` is referenced from `README.md`, `evo2/README.md`, `evo2_1b/README.md`,
`genomeocean/README.md`, and several script docstrings. Find them before editing:
```bash
grep -rn "project_memory" --include="*.md" --include="*.py" --include="*.sh" . | grep -v "^./docs/archive/"
```
Do these individually. **No global `sed`** — CLAUDE.md prohibits it and several of these are inside
prose that needs rewording, not just repathing.

### 5. Reconcile `README.md`
`README.md` (446 lines) is the consolidated current-state overview and overlaps heavily with the new
`plan.md`. **Decide explicitly** — it is not covered by the six-file framework:
- **Recommended:** keep it as the *external-facing* project README (what this repo is, how to run
  it, how to cite it), and strip the current-state/findings sections that now live in `plan.md`
  and `memory.md`. A reader arriving from GitHub needs orientation; they do not need the ledger.
- The alternative — deleting it — loses the only doc addressed to someone who is not you.

### 6. Verify
```bash
python tests/test_docs_contract.py -v
```
Must report **0 failed**. The two warnings are expected: they are the documented orphaned splits
and the documented case collision, and they will clear when [P3-B4] is done.

### 7. Commit
```bash
git add -A && git commit -m "Docs: cut over to the six-file framework"
```

---

## After cutover

- **Delete `docs/framework/`** — it exists only to stage this change. This file goes with it (or
  moves to `docs/archive/` if you want the findings list preserved).
- **Wire the verifier into CI** alongside the existing tests so drift fails a run rather than
  waiting to be noticed.
- **Work [P3-B4]** — the housekeeping items above. They are cheap and two of them are data-integrity
  risks, not tidiness.

## What the verifier does NOT check

Be clear about the boundary. It checks *structure and consistency*, not truth. It cannot tell you:
- whether a number in `memory.md` is correct — only that it is present and has provenance;
- whether a hypothesis is worth testing;
- whether a manipulation check actually validates the manipulation — only that the field is filled.

The framework makes drift *visible*. It does not make judgments for you.
