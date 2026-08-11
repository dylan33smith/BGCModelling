#!/usr/bin/env python
"""Q1 for guided decoding, read TWO ways — because they disagree exactly under myopia.

WHY THIS EXISTS SEPARATELY. The inline Q1 in `run_guided_decoding.sh` averages the guide
objective ACROSS CHUNKS. That answers "did each local selection stick?". It does not answer "is
the finished sequence more NRPS-like?", which is the end-to-end claim guided decoding actually
makes. The two coincide unless selection is MYOPIC — a chunk chosen because it looks like the
target locally, which then leads the continuation somewhere worse. Under myopia the chunk-mean
rises while the final score does not, and reporting only the mean would show a pass where the
method had in fact failed in the specific way the pre-registration warned about.

  mean-across-chunks UP + final UP    -> selection works and compounds. Q2 is interpretable.
  mean-across-chunks UP + final FLAT  -> MYOPIA. Q2 inconclusive; run whole-sequence best-of-N
                                         (--n-chunks 1) before drawing any conclusion.
  mean-across-chunks FLAT             -> selection is not moving its own objective at all;
                                         look for a bug in the selection rule before anything else.

It also adds a check the inline analysis does not do: PLAIN vs RANDOM. Both are unguided --
`plain` draws one candidate, `random` draws four and keeps one at random, so in expectation they
must agree. If they differ, the difference is coming from the harness (candidate pooling,
truncation, seeding) rather than from guidance, and the best-vs-random contrast is standing on
sand. A control that is never checked against another control is an assumption.

This is a READ-ONLY analysis over the driver's own output. It is deliberately a separate file:
the driver may still be running, and bash reads a script incrementally as it executes, so editing
it in place risks corrupting a run in progress.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from math import comb
from pathlib import Path

ARMS = ("plain", "random", "best")


def sign_test(deltas):
    """Two-sided sign test. Ties are dropped, which is the conservative convention."""
    d = [x for x in deltas if x == x and x != 0]
    up, n = sum(1 for x in d if x > 0), len(d)
    if n == 0:
        return 0, 0, 1.0
    p = min(1.0, 2 * sum(comb(n, k) * 0.5 ** n for k in range(max(up, n - up), n + 1)))
    return up, n, p


def load_arm(root: Path, cls: str, arm: str):
    p = root / f"gd_{cls}_{arm}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.open():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["tax_idx"]] = r
    return out


def mean_obj(rec):
    v = [t["guide_obj_chosen"] for t in rec.get("guide_trace", []) if "guide_obj_chosen" in t]
    return st.mean(v) if v else float("nan")


def final_p(rec):
    """P(target) after the LAST chunk — the score of the finished continuation."""
    tr = rec.get("guide_trace") or []
    return tr[-1].get("guide_p_target_chosen", float("nan")) if tr else float("nan")


def paired(a: dict, b: dict, fn):
    keys = sorted(k for k in a if k in b)
    return keys, [fn(a[k]) - fn(b[k]) for k in keys]


def contrast(root, classes, lo, hi, fn, label, note=""):
    print(f"\n--- {label} ---")
    if note:
        print(note)
    print(f"{'class':>17} {'pairs':>6} | {hi:>9} {lo:>9} {'delta':>9} {'up':>7} {'sign p':>8}")
    pooled = []
    for c in classes:
        A, B = load_arm(root, c, hi), load_arm(root, c, lo)
        if not A or not B:
            continue
        keys, d = paired(A, B, fn)
        d = [x for x in d if x == x]
        if not d:
            continue
        pooled += d
        up, n, p = sign_test(d)
        print(f"{c:>17} {len(keys):>6} | {st.mean([fn(A[k]) for k in keys]):>9.4f} "
              f"{st.mean([fn(B[k]) for k in keys]):>9.4f} {st.mean(d):>+9.4f} "
              f"{up:>3}/{n:<3} {p:>8.4f}")
    if pooled:
        up, n, p = sign_test(pooled)
        print(f"{'POOLED':>17} {'':>6} | {'':>9} {'':>9} {st.mean(pooled):>+9.4f} "
              f"{up:>3}/{n:<3} {p:>8.4f}")
    return pooled


def per_chunk_gap(root, classes):
    """Does the best-vs-random advantage GROW with chunk index, or wash out?

    A gap that appears early and then flattens is the signature of selection buying a local win
    that the continuation does not build on."""
    print("\n--- per-chunk gap: mean P(target) of the chosen candidate, best minus random ---")
    print("A gap that grows with chunk index means selection compounds; one that flattens or")
    print("shrinks means each chunk's win is being spent rather than banked.")
    rows = {}
    for c in classes:
        A, B = load_arm(root, c, "best"), load_arm(root, c, "random")
        for k in sorted(set(A) & set(B)):
            for arm, recs in (("best", A), ("random", B)):
                for t in recs[k].get("guide_trace", []):
                    rows.setdefault(t["chunk"], {"best": [], "random": []})[arm].append(
                        t.get("guide_p_target_chosen", float("nan")))
    if not rows:
        return
    ch = sorted(rows)
    print(f"{'chunk':>8} " + " ".join(f"{i:>8}" for i in ch))
    for arm in ("best", "random"):
        print(f"{arm:>8} " + " ".join(
            f"{st.mean([v for v in rows[i][arm] if v == v]):>8.3f}"
            if any(v == v for v in rows[i][arm]) else f"{'--':>8}" for i in ch))
    print(f"{'gap':>8} " + " ".join(
        f"{st.mean([v for v in rows[i]['best'] if v == v]) - st.mean([v for v in rows[i]['random'] if v == v]):>+8.3f}"
        if any(v == v for v in rows[i]['best']) and any(v == v for v in rows[i]['random'])
        else f"{'--':>8}" for i in ch))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/guided_decoding"))
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE", "RIPP"])
    args = ap.parse_args()

    have = {c: {a: len(load_arm(args.root, c, a)) for a in ARMS} for c in args.classes}
    print(f"[gd-an] {args.root}")
    for c, d in have.items():
        print(f"[gd-an]   {c:>17}: " + "  ".join(f"{a}={n}" for a, n in d.items()))
    partial = [c for c, d in have.items() if len(set(d.values())) > 1 or min(d.values()) == 0]
    if partial:
        print(f"[gd-an] NOTE: unequal/absent arms for {partial} — the run is still in progress. "
              f"Pairing drops to the intersection, so these are PARTIAL reads, not the result.")

    contrast(args.root, args.classes, "random", "best", mean_obj,
             "Q1a  MEAN ACROSS CHUNKS (matches the driver's inline check)",
             "The quantity each local selection was made on.")
    contrast(args.root, args.classes, "random", "best", final_p,
             "Q1b  FINAL FULL-SEQUENCE P(target)  <- the end-to-end claim",
             "The score of the finished continuation. Diverges from Q1a exactly under myopia.")
    contrast(args.root, args.classes, "plain", "random", final_p,
             "Q1c  HARNESS NULL-CHECK: plain vs random (both unguided — expect NO difference)",
             "A real difference here is the harness, not guidance, and undermines the contrast above.")
    per_chunk_gap(args.root, args.classes)

    print("\nNOTE ON SPREAD. The guide score is close to bimodal — most sequences land near 0 or "
          "near 1, few in between. Means over ~10 pairs are therefore coarse and unstable; the "
          "paired sign test above, not the mean, is the quantity to read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
