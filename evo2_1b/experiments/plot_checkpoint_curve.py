#!/usr/bin/env python
"""Read the checkpoint curve: detection vs training budget, baseline vs weighted(3x).

TWO QUESTIONS, TWO READINGS — kept separate because conflating them is how the 400-step pass went
wrong. Q1 is about the SUBSTRATE (does the baseline curve climb?), Q2 is about the INTERVENTION
(does the gap between curves open up?). A climbing baseline with a constant gap means more training
helps and the objective does not.

Detection is reported as a RATE with an exact binomial CI, not as a mean of `best_bio_bits`. The
first Phase-2 pass was misread precisely because that metric is zero-inflated and heavy-tailed, so
its mean tracks the single luckiest draw.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))
sys.path.insert(0, str(REPO / "src"))

ARMS = ["baseline_long", "weighted_long"]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(gen: Path, arm: str) -> list[dict]:
    from ladder_audit import one
    recs = [json.loads(line) for line in gen.open()]
    jobs = [(arm, r.get("sequence") or r.get("generated"), r.get("compound_class"), i)
            for i, r in enumerate(recs)]
    with ProcessPoolExecutor(max_workers=24) as ex:
        return list(ex.map(one, jobs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/data2/ds85/bgcmodel_runs/phase2_long")
    ap.add_argument("--steps", nargs="*", type=int, default=[500, 1000, 1500, 2000])
    args = ap.parse_args()
    root = Path(args.root)

    from scipy.stats import fisher_exact

    data: dict = {}
    for arm in ARMS:
        for st in args.steps:
            g = root / arm / f"gen_step{st}.jsonl"
            if not g.exists():
                continue
            res = score(g, arm)
            nz = sum(1 for r in res if r["bio"] > 0)
            nv = root / arm / f"ladder_step{st}.json"
            nvj = json.loads(nv.read_text()) if nv.exists() else {}
            data[(arm, st)] = {
                "n": len(res), "detect": nz,
                "orf_med": sorted(r["max_orf_aa"] for r in res)[len(res) // 2],
                "any_mean": sum(r["any"] for r in res) / len(res),
                "novelty": nvj.get("novelty_max_containment"),
                "verdict": nvj.get("novelty_verdict", "?"),
            }
    if not data:
        raise SystemExit("[curve] no generations found")

    print("=" * 84)
    print("CHECKPOINT CURVE — detection rate vs training budget (all points batched, n as shown)")
    print("=" * 84)
    print(f"{'arm':<15} {'step':>5} {'% of epoch':>11} {'detect':>10} {'rate':>7} {'95% CI':>16} "
          f"{'orf med':>8} {'novelty':>9}")
    for arm in ARMS:
        for st in args.steps:
            d = data.get((arm, st))
            if not d:
                continue
            lo, hi = wilson(d["detect"], d["n"])
            frac = st * 16 / 95759 * 100
            nv = f"{d['novelty']:.3f}" if isinstance(d["novelty"], (int, float)) else "?"
            print(f"{arm:<15} {st:>5} {frac:>10.1f}% {d['detect']:>4}/{d['n']:<5} "
                  f"{d['detect']/d['n']:>7.3f} [{lo:>5.3f},{hi:>5.3f}] {d['orf_med']:>8.0f} {nv:>9}")
        print()

    # ── Q1: does the BASELINE climb? substrate question ─────────────────────────────────────
    pts = [(st, data[("baseline_long", st)]) for st in args.steps if ("baseline_long", st) in data]
    print("-" * 84)
    if len(pts) >= 2:
        (s0, d0), (s1, d1) = pts[0], pts[-1]
        _, p = fisher_exact([[d1["detect"], d1["n"] - d1["detect"]],
                             [d0["detect"], d0["n"] - d0["detect"]]])
        r0, r1 = d0["detect"] / d0["n"], d1["detect"] / d1["n"]
        print(f"Q1 SUBSTRATE — baseline step {s0} -> {s1}: {r0:.3f} -> {r1:.3f}  (Fisher p={p:.4f})")
        if p < 0.05 and r1 > r0:
            print("   ⇒ BUDGET-LIMITED. Phase 2 was measured at the floor; its objective")
            print("     comparisons were run before there was anything to redirect, and must be")
            print("     redone at a budget where the baseline is off the floor.")
        else:
            print("   ⇒ NOT budget-limited at this range. 4x more training did not move detection,")
            print("     so the 1B is capacity-limited and no objective change was going to work on")
            print("     it. Change the substrate, not the loss.")

    # ── Q2: does the GAP open? intervention question ────────────────────────────────────────
    print()
    print("Q2 INTERVENTION — weighted(3x) minus baseline, at matched steps:")
    any_gap = False
    for st in args.steps:
        a, b = data.get(("weighted_long", st)), data.get(("baseline_long", st))
        if not a or not b:
            continue
        _, p = fisher_exact([[a["detect"], a["n"] - a["detect"]],
                             [b["detect"], b["n"] - b["detect"]]])
        flag = ""
        if p < 0.05:
            flag = "  <-- SIGNIFICANT"
            any_gap = True
        print(f"   step {st:>4}: weighted {a['detect']}/{a['n']} vs baseline {b['detect']}/{b['n']}"
              f"   Fisher p={p:.4f}{flag}")
    if not any_gap:
        print("   ⇒ the gap never opens. Domain weighting does not begin to work at any budget")
        print("     tested — consistent with the flat dose-response (3x and 10x gave IDENTICAL")
        print("     in-domain loss, 0.8763).")
    print("-" * 84)
    json.dump({f"{a}@{s}": v for (a, s), v in data.items()},
              open(root / "checkpoint_curve.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
