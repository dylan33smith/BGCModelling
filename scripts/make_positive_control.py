#!/usr/bin/env python
"""Emit a LENGTH-MATCHED positive control set: real held-out cores, scored exactly like the
generations they accompany.

WHY THIS EXISTS. Every eval driver in this repo disables the positive control by passing
`--positive "$OUT/_nopos.jsonl"` -- a path deliberately chosen not to exist. Measured 2026-08-10:
0 of 25 probe reports contain one. Consequences:

  * `correct_class` is an uncalibrated fraction of an UNSTATED maximum. Where a control WAS run
    (go_zeroshot_*/eval.json, n=20 real curated BGCs), antiSMASH detected 17/20 = 0.85 but got
    the class right only 11/20 = 0.55. A generation scoring 0.10 against a ceiling of 0.55 is
    18% of achievable -- a very different claim from 10% of perfect.
  * There is no way to tell "the model got worse" from "the instrument changed". This project
    has twice mistaken the second for the first.
  * CLAIMED ceilings do not reproduce. CLAUDE.md says the gate was recalibrated to ~0.97 on real
    cores; measured here, real splits_core cores truncated to generation length score
    correct_class 0.750 (2048 nt) / 0.750 (3000 nt) / 0.812 (6000 nt), and full MIBiG BGCs 0.55.

LENGTH MATCHING IS THE POINT. antiSMASH's verdict depends strongly on how much sequence it sees,
so a control of full-length 30 kb reference BGCs does not calibrate 2 kb generations. This
samples real cores and truncates them to the SAME length distribution as the generations, so the
control answers: "scored exactly like these generations, what does a REAL BGC achieve?"
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True,
                    help="The generations this control accompanies (sets the length distribution).")
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"),
                    help="Real HELD-OUT cores to draw from.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = match the generation count.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gens = [json.loads(l) for l in args.gen.open() if l.strip()]
    if not gens:
        raise SystemExit(f"{args.gen} is empty")
    lengths = sorted(len(g.get("sequence", "") or "") for g in gens)
    classes = [g.get("compound_class") for g in gens if g.get("compound_class")]
    want = args.n or len(gens)

    byc: dict[str, list] = {}
    for line in args.cores.open():
        r = json.loads(line)
        c = r.get("compound_class")
        if c:
            byc.setdefault(c, []).append(r)

    rng = random.Random(args.seed)
    out = []
    # Mirror the generations' CLASS mix as well as their length mix: a control drawn from a
    # different class distribution calibrates a different question.
    pool_classes = classes or [c for c in byc]
    for i in range(want):
        cls = pool_classes[i % len(pool_classes)]
        cand = byc.get(cls) or []
        target_len = lengths[i % len(lengths)]
        # need a core at least as long as the generation it stands in for
        usable = [r for r in cand if len(r["sequence"]) >= target_len] or cand
        if not usable:
            continue
        r = rng.choice(usable)
        out.append({
            "sequence": r["sequence"][:target_len] if target_len else r["sequence"],
            "compound_class": cls,
            "taxonomic_tag": r.get("taxonomic_tag", ""),
            "accession": f"POSCTRL_{r.get('accession', i)}",
            "positive_control": True,
            "truncated_to": target_len,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    ol = sorted(len(r["sequence"]) for r in out)
    print(f"[posctrl] {len(out)} real held-out cores -> {args.out}")
    print(f"[posctrl]   generation lengths: median {st.median(lengths):.0f} "
          f"[{lengths[0]}..{lengths[-1]}]")
    print(f"[posctrl]   control    lengths: median {st.median(ol):.0f} [{ol[0]}..{ol[-1]}]")
    print(f"[posctrl]   classes matched to the generation mix "
          f"({len(set(r['compound_class'] for r in out))} distinct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
