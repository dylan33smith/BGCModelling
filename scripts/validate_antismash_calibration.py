#!/usr/bin/env python3
"""Calibration check for the antiSMASH is_bgc/correct_class gate.

Runs the LIVE check_antismash (the real eval code path) on a sample of REAL held-out
cores per class, in parallel, and reports per-class:
  - is_bgc       = antiSMASH detected >=1 cluster (detection rate)
  - correct_class = the detected cluster type matches the conditioned class
Real BGCs SHOULD score high on both — that is what validates antiSMASH as a gate
(analogous to validate_m2_calibration.py for class_markers). antiSMASH on a small
core is ~3 s, so this is minutes. CPU-only — safe alongside GPU training.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bgc_pipeline.evaluation import check_antismash  # noqa: E402
from bgc_pipeline.class_map import load_class_map     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--db", default="/data2/ds85/antismash_db")
    ap.add_argument("--class-map", type=Path, default=Path("config/compound_class_map.yaml"))
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-len", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cmap, _ = load_class_map(args.class_map)
    import random
    by_class: dict[str, list[dict]] = defaultdict(list)
    for line in args.test.open():
        r = json.loads(line)
        if len(r.get("sequence", "")) >= args.min_len:
            by_class[r["compound_class"]].append(r)
    rng = random.Random(args.seed)
    jobs = []
    for c, recs in by_class.items():
        rng.shuffle(recs)
        for r in recs[: args.per_class]:
            jobs.append((c, r["sequence"]))
    print(f"[as-calib] {len(jobs)} cores across {len(by_class)} classes", file=sys.stderr)

    det = defaultdict(lambda: [0, 0]); cm = defaultdict(lambda: [0, 0])

    def run(c, seq):
        r = check_antismash(seq, accession="x", expected_class=c, class_map=cmap,
                            databases_dir=args.db)
        return c, r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, c, seq) for c, seq in jobs]
        done = 0
        for fut in as_completed(futs):
            c, r = fut.result()
            if r.get("skipped"):
                continue
            det[c][0] += int(bool(r.get("detected"))); det[c][1] += 1
            if r.get("detected"):
                cm[c][0] += int(bool(r.get("class_match"))); cm[c][1] += 1
            done += 1
            if done % 25 == 0:
                print(f"[as-calib] {done}/{len(jobs)}", file=sys.stderr)

    print(f"\n{'class':18}{'is_bgc':>10}{'correct_class':>16}")
    tb = [0, 0]; tc = [0, 0]
    for c in sorted(det, key=lambda k: -det[k][1]):
        d = det[c]; x = cm[c]; tb[0] += d[0]; tb[1] += d[1]; tc[0] += x[0]; tc[1] += x[1]
        cr = f"{x[0]}/{x[1]}" if x[1] else "-"
        print(f"{c:18}{d[0]}/{d[1]:<7}{cr:>10}  {(x[0]/x[1] if x[1] else 0):.2f}")
    print(f"\nOVERALL is_bgc detection: {tb[0]}/{tb[1]} = {tb[0]/max(tb[1],1):.2f}")
    print(f"OVERALL correct_class (of detected): {tc[0]}/{tc[1]} = {tc[0]/max(tc[1],1):.2f}")
    print("(was ~15% before recalibrating the antiSMASH product->class map)")


if __name__ == "__main__":
    main()
