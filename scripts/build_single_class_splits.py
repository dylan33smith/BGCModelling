#!/usr/bin/env python
"""Build PER-CLASS datasets for Phase 3 — SPLIT FROM SCRATCH, one small compound class at a time.

WHY SINGLE-CLASS, AND WHY IT IS NOT A RETREAT. Phase 2 closed the objective and budget levers on
the general problem. Two distinct problems had been tangled together throughout:

  1. LONG-CONTEXT COHERENCE — 2% of real BGC genes exceed the 1B's ENTIRE context; Evo2 fits 0% of
     whole BGC *regions*.
  2. BGC SPECIFICITY — the model writes real protein of the wrong kind (bio fraction 0.100 vs 0.836).

Restricting to a small class does not work around (1), it DELETES it: an ectoine core is 396 nt
median, 1/20th of the 1B's context. What remains is (2), alone, with thousands of examples.

AND IT RETIRES THE PHASE-1 CLOSURES RATHER THAN FIGHTING THEM. One adapter per class means the
model never READS a class label, and every Phase-1 negative (inert prefix, CFG, steering, soft
prefixes, activation transplant) was about label-reading.

────────────────────────────────────────────────────────────────────────────────────────────────
WHY THIS RE-SPLITS INSTEAD OF FILTERING THE GLOBAL SPLIT.

The first version of this script filtered `splits_core` by class. That INHERITS leakage-cleanliness
for free — a subset of genome-disjoint groups is still genome-disjoint — but it also inherits
whatever the *global* split happened to do to each class, and what it did was unusable:

    MELANIN        2,877 train / 0 val / 0 test   <- every melanin genome landed in train
    CDPS           1,395 / 15 / 20                <- untestable
    ALKALOID       1,202 /  9 / 12                <- untestable
    TERPENE        3,567 / 2,991 / 7,564          <- twice as much test as train

The global splitter balanced 22 classes at once; small classes were the residual. Re-splitting each
class on its own gives every class a proper 80/10/10 and rescues four classes that were dead.

WHAT RE-SPLITTING COSTS, AND HOW IT IS PAID BACK. Re-splitting BREAKS the inherited guarantee: new
cross-split neighbour pairs appear that the original MMseqs2 pass never examined. So this script
re-earns the property rather than assuming it, per class:

  1. GENOME-DISJOINT. Group by genome accession, assign whole groups (never records) to splits.
     A genome contributing to train can contribute to nothing else.
  2. EXACT-DUPLICATE. Inherited safely — the source corpus is already exact-dedup'd globally, and a
     subset of a set with no duplicates has no duplicates. Re-checked anyway; it is cheap.
  3. CROSS-SPLIT NEAR-DUPLICATE. Re-run from scratch with the SAME criterion the v2 corpus used
     (MMseqs2, >=80% identity over >=50% of the query, `--search-type 3`), dropping val/test records
     that are near-duplicates of anything in train. This is the step that would silently rot if it
     were skipped.

⚠️ VIABILITY GATE. A class that can be trained but never measured is the most expensive dead end
available, so classes are checked AFTER splitting and dropped if they cannot support an evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

SPLITS = ("train", "val", "test")
FRACS = {"train": 0.80, "val": 0.10, "test": 0.10}

MIN_TRAIN = 500      # below this a LoRA fine-tune is fitting noise
MIN_TEST = 60        # below this no generation-eval reaches n>=50 distinct prompts
MIN_GENOMES = 100    # below this the class is a few genomes' worth of near-duplicates


def genome_of(rec: dict) -> str:
    acc = rec.get("accession", "")
    return acc.split(".region")[0] if ".region" in acc else acc


def split_by_genome(recs: list[dict], seed: int) -> dict[str, list[dict]]:
    """Assign whole GENOMES to splits, largest group first, to the split furthest below quota.

    Groups are assigned, never records — that is what makes the split genome-disjoint. Largest-first
    because big groups are the hardest to place without overshooting; a random order lets a late
    500-record genome blow past a 10% test target it cannot then leave.
    """
    by_g: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_g[genome_of(r)].append(r)
    total = len(recs)
    target = {s: FRACS[s] * total for s in SPLITS}
    cur = {s: 0 for s in SPLITS}
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    rng = random.Random(seed)
    keys = sorted(by_g, key=lambda k: (-len(by_g[k]), k))
    for k in keys:
        deficit = {s: (target[s] - cur[s]) / max(target[s], 1e-9) for s in SPLITS}
        best = max(SPLITS, key=lambda s: (deficit[s], -SPLITS.index(s)))
        out[best].extend(by_g[k])
        cur[best] += len(by_g[k])
    for s in SPLITS:
        rng.shuffle(out[s])
    return out


def drop_cross_split_neardups(parts: dict[str, list[dict]], env: str, workdir: Path,
                              idthr: float, cov: float) -> tuple[dict[str, list[dict]], dict]:
    """Remove val/test records that are near-duplicates of TRAIN. Re-earned, not inherited."""
    from build_clean_holdout import mmseqs_neardup_query_ids
    stats = {}
    tr = parts["train"]
    tids = [f"tr_{i}" for i in range(len(tr))]
    for s in ("val", "test"):
        q = parts[s]
        if not q or not tr:
            stats[s] = 0
            continue
        qids = [f"{s}_{i}" for i in range(len(q))]
        nd = mmseqs_neardup_query_ids(q, qids, tr, tids, env, str(workdir),
                                      idthr=idthr, cov=cov)
        keep = [r for i, r in enumerate(q) if qids[i] not in nd]
        stats[s] = len(q) - len(keep)
        parts[s] = keep
    return parts, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_core"))
    ap.add_argument("--out", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_class"))
    ap.add_argument("--classes", nargs="*", default=[
        "ECTOINE", "HSERLACTONE", "BUTYROLACTONE", "TERPENE", "CDPS", "MELANIN",
        "ALKALOID", "PHENAZINE", "FURAN", "NUCLEOSIDE"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env", default="bgcmodel")
    ap.add_argument("--idthr", type=float, default=0.8)
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--eval-prompts", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # Pool the WHOLE v2 corpus (train+val+test) and re-split it. splits_core is already
    # exact-dedup'd and MiBIG-excluded, so pooling it recovers the full clean class corpus.
    pool: dict[str, list[dict]] = defaultdict(list)
    for split in SPLITS:
        for line in (args.src / f"{split}.jsonl").open():
            r = json.loads(line)
            if r.get("compound_class") in args.classes:
                pool[r["compound_class"]].append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest, built, skipped = {}, [], []
    print(f"{'class':<15} {'pooled':>7} {'train':>6} {'val':>5} {'test':>5} {'genomes':>8} "
          f"{'dropped':>8} {'med nt':>7}  verdict")
    for c in args.classes:
        recs = pool.get(c, [])
        if not recs:
            print(f"{c:<15} {'0':>7}  — not present in source")
            continue

        # exact-dup check (inherited, verified)
        seen, uniq = set(), []
        for r in recs:
            h = hashlib.md5(r["sequence"].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                uniq.append(r)
        n_exact = len(recs) - len(uniq)

        parts = split_by_genome(uniq, args.seed)
        parts, nd_stats = drop_cross_split_neardups(parts, args.env, args.out,
                                                    args.idthr, args.cov)
        n = {s: len(parts[s]) for s in SPLITS}
        ngen = len(set(genome_of(r) for r in parts["train"]))
        lens = sorted(len(r["sequence"]) for r in parts["train"]) or [0]
        dropped = n_exact + sum(nd_stats.values())

        fails = []
        if n["train"] < MIN_TRAIN:
            fails.append(f"train<{MIN_TRAIN}")
        if n["test"] < MIN_TEST:
            fails.append(f"test<{MIN_TEST}")
        if ngen < MIN_GENOMES:
            fails.append(f"genomes<{MIN_GENOMES}")
        ok = not fails
        print(f"{c:<15} {len(recs):>7,} {n['train']:>6,} {n['val']:>5,} {n['test']:>5,} "
              f"{ngen:>8,} {dropped:>8,} {st.median(lens):>7.0f}  "
              f"{'BUILD' if ok else 'SKIP: ' + ','.join(fails)}")

        # A genome must never appear in two splits. Assert it rather than trusting the loop.
        gsets = {s: set(genome_of(r) for r in parts[s]) for s in SPLITS}
        for a in SPLITS:
            for b in SPLITS:
                if a < b and gsets[a] & gsets[b]:
                    raise SystemExit(f"[class] ABORT: {c} genome overlap {a}/{b} — split is broken")

        manifest[c] = {"pooled": len(recs), "n": n, "genomes_train": ngen,
                       "exact_dups_removed": n_exact, "cross_split_neardups_removed": nd_stats,
                       "median_nt": st.median(lens), "viable": ok, "fails": fails,
                       "split_from_scratch": True, "fracs": FRACS,
                       "neardup_criterion": f"mmseqs id>={args.idthr} cov>={args.cov}"}
        if not ok and not args.force:
            skipped.append(c)
            continue

        cdir = args.out / c
        cdir.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            with (cdir / f"{s}.jsonl").open("w") as w:
                for r in parts[s]:
                    w.write(json.dumps(r) + "\n")
        with (cdir / "eval_prompts.jsonl").open("w") as w:
            for r in (parts["test"] + parts["val"])[: args.eval_prompts]:
                w.write(json.dumps(r) + "\n")
        built.append(c)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n[class] BUILT {len(built)}: {', '.join(built)}")
    if skipped:
        print(f"[class] SKIPPED {len(skipped)}: {', '.join(skipped)}")
    print(f"[class] genome-disjointness ASSERTED per class; cross-split near-dups re-earned "
          f"(mmseqs id>={args.idthr}, cov>={args.cov})")
    print(f"[class] manifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
