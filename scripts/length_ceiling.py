#!/usr/bin/env python
"""IS SEQUENCE LENGTH THE BINDING CONSTRAINT? A per-class, per-length ceiling measurement.

THE QUESTION. Nearly every generation experiment in this project ran at 2-3 kb. Real BGC cores
are often far longer -- median 5.2 kb for NRPS and 27.6 kb for PKS/NRPS hybrids -- so a natural
worry is that `correct_class` reads 0 because 3 kb is simply too short to contain a recognisable
cluster, regardless of whether conditioning works. If true, every null in the programme is
confounded with length and most of them would have to be re-read.

THE TEST. Take REAL held-out cores of known class, truncate them to the lengths we actually
generate, and run the same antiSMASH gate. A real BGC is the best case: if it cannot be detected
and classified at 3 kb, no generation could be either, and length is the binding constraint. If
it CAN, then a generated 3 kb sequence scoring 0 is a statement about the generation, not the
ruler.

Reported PER CLASS, because the pooled number hides exactly the case the worry is about: hybrids
have a median core of 27.6 kb, so a 3 kb window is ~11% of one, while a terpene core has a median
of 966 nt and 3 kb is three times the whole thing.

This is the positive control the eval suite already builds (`make_positive_control.py`), run as a
function of length rather than at one point.
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

from bgc_pipeline.class_map import load_class_map  # noqa: E402
from bgc_pipeline.evaluation import check_antismash, check_class_markers  # noqa: E402


def _score_one(job):
    """Module-level so it is picklable. A closure inside main() cannot be sent to a worker
    process, and the failure surfaces only at dispatch, after the core selection has already run."""
    c, L, nt, acc, s, cmap, pfam, asdb = job
    a = check_antismash(s, accession=f"{acc}_{L}", expected_class=c, class_map=cmap,
                        databases_dir=asdb)
    m = check_class_markers(s, expected_class=c, pfam_hmm_path=Path(pfam))
    return {"cls": c, "req_len": L, "nt": nt, "acc": acc,
            "as_skipped": bool(a.get("skipped")),
            "is_bgc": bool(a.get("detected")),
            "correct_class": bool(a.get("class_match")),
            "markers": bool(m.get("markers_present")) if not m.get("skipped") else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--classes", nargs="+",
                    default=["NRPS", "PKS", "PKS_NRPS_HYBRID", "TERPENE", "RIPP"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[1000, 2000, 3000, 6000, 12000, 0],
                    help="0 = full, untruncated core (the true ceiling).")
    ap.add_argument("--n-per-class", type=int, default=12)
    ap.add_argument("--pfam", type=Path, default=Path("/data2/ds85/pfam/Pfam-A.hmm"))
    ap.add_argument("--antismash-db", default="/data2/ds85/antismash_db")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/length_ceiling.json"))
    args = ap.parse_args()

    if not args.pfam.exists():
        raise SystemExit(f"ABORT: no Pfam HMM at {args.pfam}")
    cmap, _ = load_class_map(REPO / "config" / "compound_class_map.yaml")

    byc: dict[str, list] = {c: [] for c in args.classes}
    for line in args.cores.open():
        r = json.loads(line)
        c = r.get("compound_class")
        if c in byc and len(r.get("sequence", "")) >= 1000:
            byc[c].append(r)
    rng = random.Random(args.seed)
    picked = []
    for c in args.classes:
        rng.shuffle(byc[c])
        # Require cores long enough to actually TEST the long end -- otherwise "6 kb" silently
        # means "this core's full length", and the long columns would just re-measure the short
        # ones while looking like an independent data point.
        long_enough = [r for r in byc[c] if len(r["sequence"]) >= max(args.lengths)] or byc[c]
        picked += [(c, r) for r in long_enough[: args.n_per_class]]
    print(f"[len] {len(picked)} cores: "
          + ", ".join(f"{c}={sum(1 for k, _ in picked if k == c)}" for c in args.classes),
          flush=True)

    jobs = []
    for c, r in picked:
        for L in args.lengths:
            s = r["sequence"][:L] if L else r["sequence"]
            if len(s) < 200:
                continue
            jobs.append((c, L, len(s), r.get("accession") or r.get("id"), s,
                         cmap, str(args.pfam), args.antismash_db))
    print(f"[len] {len(jobs)} antiSMASH runs ({len(args.lengths)} lengths)", flush=True)

    from concurrent.futures import ProcessPoolExecutor

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(_score_one, jobs)):
            rows.append(res)
            if (i + 1) % 25 == 0:
                print(f"[len]   {i + 1}/{len(jobs)}", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=1))

    print("\n" + "=" * 88)
    print("CEILING: antiSMASH correct_class on REAL cores, truncated to the lengths we generate")
    print("A generated sequence cannot beat this. If the 3000 column is ~0, length is the")
    print("binding constraint and every null at 3 kb is confounded with it.")
    print("=" * 88)
    lens = args.lengths
    hdr = f"{'class':>17} " + " ".join(f"{('full' if L == 0 else str(L)):>7}" for L in lens)
    for metric in ("correct_class", "is_bgc", "markers"):
        print(f"\n--- {metric} ---")
        print(hdr)
        for c in args.classes:
            cells = []
            for L in lens:
                sub = [r for r in rows if r["cls"] == c and r["req_len"] == L
                       and not r["as_skipped"] and r.get(metric) is not None]
                cells.append(f"{sum(bool(r[metric]) for r in sub) / len(sub):>7.2f}" if sub
                             else f"{'--':>7}")
            print(f"{c:>17} " + " ".join(cells))
        cells = []
        for L in lens:
            sub = [r for r in rows if r["req_len"] == L and not r["as_skipped"]
                   and r.get(metric) is not None]
            cells.append(f"{sum(bool(r[metric]) for r in sub) / len(sub):>7.2f}" if sub
                         else f"{'--':>7}")
        print(f"{'POOLED':>17} " + " ".join(cells))
    med = {c: sorted(r["nt"] for r in rows if r["cls"] == c and r["req_len"] == 0)
           for c in args.classes}
    print("\nmedian FULL length of the sampled cores: "
          + ", ".join(f"{c}={(v[len(v)//2] if v else 0):,}" for c, v in med.items()))
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
