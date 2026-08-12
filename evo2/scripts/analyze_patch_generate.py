#!/usr/bin/env python
"""PHASE B readout: did the transplant carry CLASS, or only local sequence?

THE CONTRAST THAT MATTERS. Every generation here starts from the SAME recipient context. The arms
differ only in whose activations were substituted at layer L over the last k context positions:

    unpatched     nothing substituted            -> the recipient's own floor
    same_class    a different core of the RECIPIENT's class  -> "any real transplant perturbs"
    cross_class   a core of a DIFFERENT class     -> the treatment

So the question is not "did the output change" (Phase A already showed it changes a lot) but
whether the DONOR'S CLASS appears in the continuation, above what a same-class transplant produces.

WHY BOTH CLASSES ARE SCORED. A transplant could add the donor's class, remove the recipient's, do
both, or do neither while still scrambling the sequence. Reading only "is it the donor's class"
cannot tell those apart, and this project has already seen a class direction that DELETES reliably
and never INSTALLS. `mapped_classes` from antiSMASH is a set, so both memberships are read from one
run and the four outcomes stay distinguishable.

Scoring uses antiSMASH (measured FPR 0.000) and Pfam class markers, neither of which took any part
in choosing the patch. Paired by pair index, since every arm shares the recipient context.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def sign_test(deltas):
    d = [x for x in deltas if x == x and x != 0]
    up, n = sum(1 for x in d if x > 0), len(d)
    if n == 0:
        return 0, 0, 1.0
    return up, n, min(1.0, 2 * sum(comb(n, k) * 0.5 ** n for k in range(max(up, n - up), n + 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/data2/ds85/bgcmodel_runs/patch_generate"))
    ap.add_argument("--pfam", type=Path, default=Path("/data2/ds85/pfam/Pfam-A.hmm"))
    args = ap.parse_args()

    tsv = args.root / "antismash.tsv"
    if not tsv.exists():
        raise SystemExit(f"[pga] no {tsv} — run score_generations_antismash.py first "
                         f"(the command is printed at the end of patch_generate.py)")

    # antismash.tsv carries no per-record id, so it is joined by ROW ORDER within each arm file.
    # That assumption is checked against the `length` column rather than assumed, the same way the
    # guided-decoding analysis had to.
    by_arm: dict[str, list] = {}
    for r in csv.DictReader(tsv.open(), delimiter="\t"):
        by_arm.setdefault(r["arm"], []).append(r)

    arms: dict[str, list] = {}
    for f in sorted(args.root.glob("pg_*.jsonl")):
        key = f.stem
        recs = [json.loads(l) for l in f.open() if l.strip()]
        rows = by_arm.get(key, [])
        if len(rows) != len(recs):
            raise SystemExit(f"[pga] ABORT: {key} has {len(recs)} records but {len(rows)} "
                             f"antiSMASH rows — the row-order join is invalid.")
        if [int(x["length"]) for x in rows] != [r["length"] for r in recs]:
            raise SystemExit(f"[pga] ABORT: {key} length column does not match record order — "
                             f"the row-order join would pair the wrong sequences.")
        for rec, row in zip(recs, rows):
            mapped = {m for m in (row.get("mapped_classes") or "").split(",") if m}
            rec["_mapped"] = mapped
            rec["_detected"] = row.get("is_bgc") in ("1", "True", "true")
            rec["_donor_hit"] = rec["donor_class"] in mapped
            rec["_recip_hit"] = rec["recip_class"] in mapped
        arms[key] = recs

    print(f"[pga] {args.root}")
    print(f"\n{'arm':>26} {'n':>4} {'is_bgc':>8} {'DONOR class':>12} {'recip class':>12} "
          f"{'neither':>9}")
    for key in sorted(arms):
        recs = arms[key]
        n = len(recs)
        print(f"{key.replace('pg_',''):>26} {n:>4} "
              f"{sum(r['_detected'] for r in recs) / n:>8.3f} "
              f"{sum(r['_donor_hit'] for r in recs) / n:>12.3f} "
              f"{sum(r['_recip_hit'] for r in recs) / n:>12.3f} "
              f"{sum(not r['_donor_hit'] and not r['_recip_hit'] for r in recs) / n:>9.3f}")

    # FLOOR CHECK. If the unpatched arm produces nothing antiSMASH can detect, no arm can show an
    # installed class and every cell reads 0.000 — a result indistinguishable from "transplanting
    # does not transfer class". The first run of this experiment hit exactly that: it was launched
    # against BASE Evo2 while the seeded BGC capability lives in the LoRA adapter, so is_bgc was
    # 0.000 everywhere including the control. Refuse to report a null without a floor.
    base = arms.get("pg_unpatched", [])
    base_det = sum(r["_detected"] for r in base) / len(base) if base else 0.0
    if base_det == 0.0:
        raise SystemExit(
            f"[pga] ABORT: the UNPATCHED control has is_bgc = 0.000 (n={len(base)}). There is no "
            f"floor to move, so every arm reads 0.000 whether or not the transplant works, and "
            f"this run cannot distinguish 'class does not transfer' from 'nothing was detectable "
            f"in the first place'.\n"
            f"        Most likely cause: generation ran against BASE Evo2. The seeded BGC "
            f"capability comes from the LoRA — pass --adapter to patch_generate.py.")

    print("\nPAIRED, per recipient context — does a CROSS-class transplant install the donor's "
          "class\nmore often than a SAME-class transplant at the same layer and k?")
    print(f"{'layer/k':>12} {'pairs':>6} {'cross only':>11} {'same only':>10} {'sign p':>8}")
    keys = [k for k in arms if k.startswith("pg_cross_class_")]
    for ck in sorted(keys):
        sk = ck.replace("cross_class", "same_class")
        if sk not in arms:
            continue
        c = {r["pair"]: r for r in arms[ck]}
        s = {r["pair"]: r for r in arms[sk]}
        shared = sorted(set(c) & set(s))
        d = [int(c[p]["_donor_hit"]) - int(s[p]["_donor_hit"]) for p in shared]
        co = sum(1 for x in d if x > 0)
        so = sum(1 for x in d if x < 0)
        _, _, pv = sign_test(d)
        print(f"{ck.replace('pg_cross_class_', ''):>12} {len(shared):>6} {co:>11} {so:>10} "
              f"{pv:>8.4f}")

    print("\nDELETION vs INSTALLATION — the asymmetry this project has seen before.")
    print(f"{'arm':>26} {'recip class kept':>17} {'donor class gained':>19}")
    base_recip = {r["pair"]: r["_recip_hit"] for r in base}
    for key in sorted(keys):
        recs = arms[key]
        kept = [r for r in recs if base_recip.get(r["pair"])]
        print(f"{key.replace('pg_',''):>26} "
              f"{(sum(r['_recip_hit'] for r in kept) / len(kept) if kept else float('nan')):>17.3f} "
              f"{sum(r['_donor_hit'] for r in recs) / len(recs):>19.3f}")
    print("\n(left column = of the contexts whose UNPATCHED continuation carried the recipient's "
          "class,\n how many still do after the transplant. A drop with no matching gain on the "
          "right is\n deletion without installation — the same asymmetry activation steering "
          "showed.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
