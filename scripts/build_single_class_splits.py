#!/usr/bin/env python
"""Build PER-CLASS datasets for Phase 3 — one small compound class at a time.

WHY SINGLE-CLASS, AND WHY IT IS NOT A RETREAT. Phase 2 established that this substrate cannot be
pushed on de novo biosynthetic content by changing the objective or the training budget. Two
distinct problems were tangled together the whole time:

  1. LONG-CONTEXT COHERENCE — 2% of real BGC genes are longer than the 1B's ENTIRE context, and
     Evo2 fits 0% of whole BGC *regions*.
  2. BGC SPECIFICITY — the model writes real protein of the wrong kind (biosynthetic fraction
     0.100 vs 0.836).

Restricting to a SMALL class does not work around problem 1, it DELETES it: an ectoine core has a
median length of 396 nt, which is 1/20th of the 1B's context. What remains is problem 2, alone, on
a target where thousands of examples exist.

AND IT RETIRES THE PHASE-1 CLOSURES RATHER THAN FIGHTING THEM. One adapter per class means the
model never has to READ a class label — and every Phase-1 negative (inert prefix, CFG, steering,
soft prefixes, activation transplant) was about label-reading. Those results stop applying because
the question stops being asked.

LEAKAGE IS INHERITED, NOT RE-DERIVED. `splits_core` is already genome-disjoint, exact-dedup'd and
MMseqs2-cross-split-clean, so filtering it BY CLASS preserves every one of those properties: a
subset of disjoint groups is still disjoint. Re-splitting per class would THROW THAT AWAY and have
to re-earn it. This script therefore only ever filters; it never re-splits.

⚠️ IT REFUSES CLASSES THAT CANNOT BE EVALUATED. MELANIN has 2,877 training records and **zero** in
val and test — genome-disjoint splitting put every melanin-bearing genome in train. A class with no
held-out data can be trained and never measured, which is the most expensive kind of dead end. Such
classes are reported and skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")

# Minimums, with the reason each one exists.
MIN_TRAIN = 500      # below this a LoRA fine-tune is fitting noise
MIN_TEST = 60        # below this no generation-eval reaches n>=50 distinct prompts
MIN_GENOMES = 100    # below this the class is a few genomes' worth of near-duplicates


def genome_of(rec: dict) -> str:
    acc = rec.get("accession", "")
    return acc.split(".region")[0] if ".region" in acc else acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_core"))
    ap.add_argument("--out", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_class"))
    ap.add_argument("--classes", nargs="*", default=[
        "ECTOINE", "HSERLACTONE", "BUTYROLACTONE", "TERPENE", "CDPS", "MELANIN",
        "ALKALOID", "PHENAZINE", "FURAN", "NUCLEOSIDE"])
    ap.add_argument("--eval-prompts", type=int, default=200,
                    help="size of the held-out prompt file used to drive generation")
    ap.add_argument("--force", action="store_true",
                    help="emit classes that fail the viability gate anyway")
    args = ap.parse_args()

    by_class: dict = {c: {s: [] for s in SPLITS} for c in args.classes}
    for split in SPLITS:
        src = args.src / f"{split}.jsonl"
        if not src.exists():
            raise SystemExit(f"[class] ABORT: missing {src}")
        for line in src.open():
            r = json.loads(line)
            c = r.get("compound_class")
            if c in by_class:
                by_class[c][split].append(r)

    manifest, emitted, skipped = {}, [], []
    print(f"{'class':<15} {'train':>7} {'val':>6} {'test':>6} {'genomes':>8} {'med nt':>7} "
          f"{'p90 nt':>7} {'%<8kb':>6}  verdict")
    for c in args.classes:
        d = by_class[c]
        n = {s: len(d[s]) for s in SPLITS}
        lens = sorted(len(r["sequence"]) for r in d["train"]) or [0]
        ngen = len(set(genome_of(r) for r in d["train"]))
        p90 = lens[int(0.9 * (len(lens) - 1))]
        frac8 = sum(1 for x in lens if x < 8192) / len(lens) * 100

        fails = []
        if n["train"] < MIN_TRAIN:
            fails.append(f"train<{MIN_TRAIN}")
        if n["test"] < MIN_TEST:
            fails.append(f"test<{MIN_TEST}")
        if ngen < MIN_GENOMES:
            fails.append(f"genomes<{MIN_GENOMES}")
        ok = not fails
        print(f"{c:<15} {n['train']:>7,} {n['val']:>6,} {n['test']:>6,} {ngen:>8,} "
              f"{st.median(lens):>7.0f} {p90:>7.0f} {frac8:>5.0f}%  "
              f"{'BUILD' if ok else 'SKIP: ' + ','.join(fails)}")

        manifest[c] = {"n": n, "genomes_train": ngen, "median_nt": st.median(lens),
                       "p90_nt": p90, "pct_under_8kb": round(frac8, 1),
                       "viable": ok, "fails": fails}
        if not ok and not args.force:
            skipped.append(c)
            continue

        cdir = args.out / c
        cdir.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            with (cdir / f"{s}.jsonl").open("w") as w:
                for r in d[s]:
                    w.write(json.dumps(r) + "\n")
        # Prompt file for generation drivers: held-out records only, so a generation prompt can
        # never be a training record's taxon+class pair drawn from the training genomes.
        pool = d["test"] + d["val"]
        with (cdir / "eval_prompts.jsonl").open("w") as w:
            for r in pool[: args.eval_prompts]:
                w.write(json.dumps(r) + "\n")
        emitted.append(c)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n[class] BUILT {len(emitted)}: {', '.join(emitted)}")
    if skipped:
        print(f"[class] SKIPPED {len(skipped)}: {', '.join(skipped)}")
        for c in skipped:
            m = manifest[c]
            if m["n"]["test"] == 0:
                print(f"  ⚠️ {c}: {m['n']['train']:,} train records and ZERO held out. Genome-disjoint"
                      f" splitting placed every {c}-bearing genome in train. Trainable, never"
                      f" measurable — do not use without re-splitting from source.")
    print(f"[class] manifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
