#!/usr/bin/env python3
"""Calibration data collection for the antiSMASH is_bgc/correct_class gate.

Runs antiSMASH (--minimal, prodigal) on a sample of REAL held-out cores per class,
in parallel, and records for each core: whether a cluster was DETECTED (is_bgc) and
the antiSMASH product types emitted (correct_class). Output feeds the product->class
map build + the pass-rate validation (analogous to validate_m2_calibration.py for M2).

antiSMASH on a small core is ~3 s, so this is minutes, not hours. CPU-only — safe to
run alongside GPU training.

Output JSONL rows: {"class","accession","detected","products"}.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one(seq: str, acc: str, db: str, cpus: int, tmproot: str) -> dict:
    with tempfile.TemporaryDirectory(dir=tmproot) as tmp:
        fa = Path(tmp) / "q.fasta"
        fa.write_text(f">{acc}\n{seq}\n")
        outdir = Path(tmp) / "out"
        cmd = ["antismash", str(fa), "--output-dir", str(outdir),
               "--genefinding-tool", "prodigal", "--minimal", "--cpus", str(cpus),
               "--databases", db]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        js = list(outdir.glob("*.json"))
        if proc.returncode != 0 or not js:
            return {"accession": acc, "detected": None,
                    "error": (proc.stderr or "")[-300:], "products": []}
        data = json.loads(js[0].read_text())
        products: list[str] = []
        for rec in data.get("records", []):
            for area in rec.get("areas", []):
                products.extend(area.get("products", []))
        return {"accession": acc, "detected": len(products) > 0, "products": products}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--db", default="/data2/ds85/antismash_db")
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cpus", type=int, default=2)          # antismash internal threads per run
    ap.add_argument("--min-len", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("/tmp/as_calib.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

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
            jobs.append((c, r.get("accession", f"{c}_{len(jobs)}"), r["sequence"]))
    print(f"[calib] {len(jobs)} cores across {len(by_class)} classes; "
          f"{args.workers} workers x {args.cpus} cpus", file=sys.stderr)

    results = []
    tmproot = tempfile.mkdtemp(prefix="ascal_")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, seq, acc, args.db, args.cpus, tmproot): (c, acc)
                for (c, acc, seq) in jobs}
        done = 0
        for fut in as_completed(futs):
            c, acc = futs[fut]
            row = fut.result(); row["class"] = c
            results.append(row)
            done += 1
            if done % 20 == 0:
                print(f"[calib] {done}/{len(jobs)}", file=sys.stderr)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in results))

    # quick summary: detection rate + product inventory per class
    det = defaultdict(lambda: [0, 0]); prods = defaultdict(set); errs = 0
    for r in results:
        if r["detected"] is None:
            errs += 1; continue
        det[r["class"]][0] += int(bool(r["detected"])); det[r["class"]][1] += 1
        prods[r["class"]].update(r["products"])
    print(f"\n[calib] wrote {args.out}  (errors={errs})", file=sys.stderr)
    print(f"\n{'class':18}{'detected':>10}{'  products seen'}")
    for c in sorted(det, key=lambda k: -det[k][1]):
        p, n = det[c]
        print(f"{c:18}{p:>3}/{n:<3}     {sorted(prods[c])}")


if __name__ == "__main__":
    main()
