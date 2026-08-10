#!/usr/bin/env python
"""How SENSITIVE is the class-marker readout on DNA the length of our generations?

WHY THIS EXISTS. Every steering phase is scored with `check_class_markers`: does at least one of
the target class's characteristic Pfam domains appear? Phase 3 and the L27 ladder both report
0 target-class hits, and that is only informative if the readout would have FIRED on real DNA of
the same class at the same length. Its false-positive rate was measured 2026-08-10 (1/100 on real
non-BGC windows). Its TRUE-POSITIVE rate at generation length has never been measured -- so the
denominator of "0 out of 48" is unknown.

The warning sign is already in the data: in the L27 ladder, the SEED's own class markers appear in
only 5-6 of 12 continuations. If a continuation of a real class-X core shows class-X markers half
the time, the readout's ceiling is nowhere near 1.0, and "0/48" has to be read against that
ceiling rather than against perfection.

WHAT THIS MEASURES, on real HELD-OUT cores truncated to the generation length:
  * TPR  -- markers_present for the core's OWN class. The ceiling any steering arm could hit.
  * FPR  -- markers_present for each OTHER class. What a steered arm scores by accident.
The Pfam scan is the expensive step, so each sequence is scanned ONCE and every class verdict is
recomputed from the resulting accession set -- exactly what check_class_markers does internally.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bgc_pipeline.evaluation import OBLIGATE_DOMAINS, check_class_markers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE", "RIPP"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[3000, 0],
                    help="Truncation lengths to test; 0 = full core (the ceiling at any length).")
    ap.add_argument("--n-per-class", type=int, default=15)
    ap.add_argument("--pfam", type=Path, default=Path("/data2/ds85/pfam/Pfam-A.hmm"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/marker_sensitivity.json"))
    args = ap.parse_args()

    if not args.pfam.exists():
        raise SystemExit(f"ABORT: no Pfam HMM at {args.pfam} -- every scan would skip")

    byc: dict[str, list] = {c: [] for c in args.classes}
    for line in args.cores.open():
        r = json.loads(line)
        c = r.get("compound_class")
        if c in byc and len(r.get("sequence", "")) >= 1000:
            byc[c].append(r)
    rng = random.Random(args.seed)
    for c in args.classes:
        rng.shuffle(byc[c])
        byc[c] = byc[c][: args.n_per_class]
    print("[sens] cores: " + ", ".join(f"{c}={len(byc[c])}" for c in args.classes), flush=True)

    rows = []
    for L in args.lengths:
        for c in args.classes:
            for r in byc[c]:
                seq = r["sequence"][:L] if L else r["sequence"]
                if len(seq) < 200:
                    continue
                # ONE scan; every class verdict recomputed from its accession set. Passing
                # expected_class="" is what makes the scan class-agnostic.
                res = check_class_markers(seq, expected_class="", pfam_hmm_path=args.pfam)
                if res.get("skipped"):
                    raise SystemExit(f"[sens] ABORT: scan skipped ({res.get('reason')})")
                accs = set(res.get("unique_domain_accessions", []))
                rows.append({
                    # `bucket` is the REQUESTED truncation, tagged explicitly. Inferring it by
                    # comparing length to nt silently merged the two: a core longer than the
                    # truncation has nt == length == 3000, so every truncated row from a long
                    # core also matched the "full-length" test and inflated that bucket.
                    "bucket": str(L) if L else "full",
                    "true_class": c, "length": L or len(seq), "nt": len(seq),
                    "n_orfs": res.get("orf_count"), "n_domains": res.get("domain_count"),
                    # replicate check_class_markers' ANY-of rule for every class
                    "hits": {k: sorted(set(OBLIGATE_DOMAINS.get(k, [])) & accs)
                             for k in args.classes},
                })
            print(f"[sens]   L={L or 'full'} {c} done", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=1))

    print("\n" + "=" * 86)
    print("SENSITIVITY OF THE CLASS-MARKER READOUT (any-of the class's Pfam accessions)")
    print("TPR = fires on real DNA of that class. This is the CEILING a steering arm could reach.")
    print("=" * 86)
    for L in args.lengths:
        sub = [r for r in rows if r["bucket"] == (str(L) if L else "full")]
        if not sub:
            continue
        print(f"\n--- truncated to {L} nt ---" if L else "\n--- FULL-LENGTH cores ---")
        print(f"{'true class':>12} {'n':>4} {'median nt':>10} {'TPR (own)':>10}   "
              + " ".join(f"{'FPR:' + c[:7]:>12}" for c in args.classes))
        for c in args.classes:
            rs = [r for r in sub if r["true_class"] == c]
            if not rs:
                continue
            n = len(rs)
            tpr = sum(1 for r in rs if r["hits"][c]) / n
            med = sorted(r["nt"] for r in rs)[n // 2]
            fpr = [sum(1 for r in rs if r["hits"][k]) / n if k != c else None
                   for k in args.classes]
            cells = " ".join(f"{'--':>12}" if f is None else f"{f:>12.3f}" for f in fpr)
            print(f"{c:>12} {n:>4} {med:>10} {tpr:>10.3f}   {cells}")
        allr = sub
        print(f"{'POOLED':>12} {len(allr):>4} {'':>10} "
              f"{sum(1 for r in allr if r['hits'][r['true_class']]) / len(allr):>10.3f}")
    print("\nREAD: a steering arm reporting 0/N target-class hits must be read against the TPR at")
    print("the SAME length. If the TPR is 0.5, the readout misses half of REAL class DNA, so the")
    print("arm's effective sample size is half what it looks like -- and a small true effect is")
    print("invisible by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
