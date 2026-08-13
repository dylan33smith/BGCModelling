#!/usr/bin/env python
"""Compare the Phase-2 arms PER RECORD, with tests rather than means.

WHY NOT JUST PRINT THE MEANS. That is what the first pass did, and it printed
"frame -10.201 vs baseline" from a distribution where only 3 of 24 records were non-zero and
baseline's single best draw was 64% of its arm total. `best_bio_bits` is heavily zero-inflated and
heavy-tailed, so its mean is an outlier detector, not a summary. Everything here is therefore
reported as a rate or a rank test, with the mean shown only alongside them.

Novelty is a CONSTRAINT, not a rung: every metric here is maximised by copying training data, so an
arm that improves with containment climbing is reciting. It is printed beside each arm and gates the
verdict.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))
sys.path.insert(0, str(REPO / "src"))

ARMS = ["baseline", "frame", "weighted"]


def score(root: Path, arm: str) -> list[dict]:
    from ladder_audit import one
    gen = root / arm / "gen_n150.jsonl"
    recs = [json.loads(line) for line in gen.open()]
    jobs = [(arm, r.get("sequence") or r.get("generated"), r.get("compound_class"), i)
            for i, r in enumerate(recs)]
    with ProcessPoolExecutor(max_workers=24) as ex:
        return list(ex.map(one, jobs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/data2/ds85/bgcmodel_runs/phase2_1b")
    args = ap.parse_args()
    root = Path(args.root)

    from scipy.stats import fisher_exact, mannwhitneyu

    data, novelty = {}, {}
    for a in ARMS:
        if not (root / a / "gen_n150.jsonl").exists():
            print(f"[compare] {a}: no gen_n150.jsonl — skipped")
            continue
        data[a] = score(root, a)
        lp = root / a / "ladder_n150.json"
        if lp.exists():
            novelty[a] = json.loads(lp.read_text())
    if "baseline" not in data:
        raise SystemExit("[compare] no baseline — nothing to compare against")

    print("=" * 78)
    print("PHASE-2 ARMS AT n=152 — per record, tests not means")
    print("=" * 78)
    hdr = f"{'arm':>9} {'n':>4} {'detect':>9} {'bio mean':>9} {'bio median':>11} {'orf med':>8} {'novelty':>16}"
    print(hdr)
    for a in ARMS:
        if a not in data:
            continue
        bio = [r["bio"] for r in data[a]]
        nz = sum(1 for x in bio if x > 0)
        nv = novelty.get(a, {})
        nvs = f"{nv.get('novelty_max_containment', float('nan')):.3f} {nv.get('novelty_verdict','?')}"
        print(f"{a:>9} {len(bio):>4} {nz:>4}/{len(bio):<4} {st.mean(bio):>9.2f} "
              f"{st.median(bio):>11.2f} {st.median([r['max_orf_aa'] for r in data[a]]):>8.0f} {nvs:>16}")

    b_bio = [r["bio"] for r in data["baseline"]]
    b_nz = sum(1 for x in b_bio if x > 0)
    print("\nvs baseline — DETECTION RATE (the powered comparison) and RANK test on best_bio_bits:")
    verdict_moved = []
    for a in ARMS:
        if a == "baseline" or a not in data:
            continue
        bio = [r["bio"] for r in data[a]]
        nz = sum(1 for x in bio if x > 0)
        _, pf = fisher_exact([[nz, len(bio) - nz], [b_nz, len(b_bio) - b_nz]])
        u, pm = mannwhitneyu(bio, b_bio, alternative="two-sided")
        A = u / (len(bio) * len(b_bio))
        better = nz / len(bio) > b_nz / len(b_bio)
        print(f"  {a:>9}  detection {nz}/{len(bio)} vs {b_nz}/{len(b_bio)}  Fisher p={pf:.4f}"
              f"   |  best_bio_bits A={A:.3f} p={pm:.4f}")
        if pf < 0.05 and better:
            verdict_moved.append(a)

    print("\n" + "-" * 78)
    if verdict_moved:
        clean = [a for a in verdict_moved
                 if novelty.get(a, {}).get("novelty_verdict", "").startswith("PASS")]
        if clean:
            print(f"  ⇒ MOVED at unchanged novelty: {', '.join(clean)}. The objective hypothesis")
            print("    survives; confirm on the 7B before reporting it as a project result.")
        else:
            print(f"  ⇒ {', '.join(verdict_moved)} improved but FAILED novelty — that is recitation,")
            print("    not capability. Not a result.")
    else:
        print("  ⇒ NO ARM MOVED DETECTION at n=152. Unlike the n=24 pass, this test WAS powered")
        print("    to see a doubling, so the pre-registered kill criterion now applies on its own")
        print("    terms: a clean negative on the objective hypothesis, not a reason for another")
        print("    loss variant.")
    print("-" * 78)
    json.dump({a: [{k: r[k] for k in ("bio", "any", "max_orf_aa", "n_bio_domains", "bio_span_frac")}
                   for r in v] for a, v in data.items()},
              open(root / "compare_n150.json", "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
