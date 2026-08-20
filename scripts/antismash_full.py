#!/usr/bin/env python
"""Full-mode antiSMASH over a JSONL of sequences, with output dirs RETAINED.

Written 2026-08-19. Two lessons from Phase 5 are baked in:

1. **Never `--minimal`.** It disables the analysis modules (RODEO, domain analysis,
   CompaRiPPson), so precursors and refined products never appear. Detection itself is
   unaffected (100% agreement on `is_bgc`, n=10) -- what minimal costs is everything
   downstream of detection.
2. **Never a `TemporaryDirectory`.** `evaluation.py:check_antismash` deletes its output after
   reading two booleans, so 833 sequences of prior results cannot be re-mined. Output dirs are
   kept here, one per sequence, named by the record index.

Emits one TSV row per input sequence: index, accession, ran, detected, n_regions, products.
`products` is the `region` feature's `product` qualifiers, semicolon-joined.

⚠️ `subclass_specificity` counts DETECTED SEQUENCES, not product strings -- a region can carry
several products. Counting strings is what produced the withdrawn "~70%" RiPP figure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DB_DEFAULT = "/data2/ds85/antismash_db"


def products_from_json(js: Path) -> list[str]:
    data = json.loads(js.read_text())
    out: list[str] = []
    for rec in data.get("records", []):
        for feat in rec.get("features", []):
            if feat.get("type") == "region":
                out += feat.get("qualifiers", {}).get("product", [])
    return out


def run_one(job: tuple[int, str, str, str, str, int, int]) -> dict:
    i, acc, seq, outroot, db, cpus, minlength = job
    stem = f"seq_{i:04d}"
    d = Path(outroot) / stem
    js = d / f"{stem}.json"
    if js.exists():                       # resumable: never redo finished work
        prods = products_from_json(js)
        return {"i": i, "accession": acc, "ran": 1,
                "detected": int(bool(prods)), "products": prods}
    fa = Path(outroot) / f"{stem}.fasta"
    fa.write_text(f">{stem}\n{seq}\n")
    cmd = ["antismash", str(fa), "--output-dir", str(d),
           "--genefinding-tool", "prodigal", "--cpus", str(cpus), "--databases", db]
    if minlength:
        # antiSMASH REFUSES any record under 1,000 nt by default ("all input records smaller
        # than minimum length"). TERPENE's median strict core is 960 nt, so the default silently
        # drops ~46% of real cores -- a instrument limit that would otherwise be read as a rate.
        cmd += ["--minlength", str(minlength)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (Path(outroot) / f"{stem}.log").write_text((proc.stdout or "") + (proc.stderr or ""))
    if proc.returncode != 0 or not js.exists():
        return {"i": i, "accession": acc, "ran": 0, "detected": 0, "products": []}
    prods = products_from_json(js)
    return {"i": i, "accession": acc, "ran": 1,
            "detected": int(bool(prods)), "products": prods}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True, help="JSONL with a `sequence` field.")
    ap.add_argument("--out-dir", type=Path, required=True, help="RETAINED output root.")
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--window", type=int, default=0,
                    help="Score only the first N nt (0 = whole record). Goes in the filename.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cpus-per-worker", type=int, default=2)
    ap.add_argument("--databases", default=DB_DEFAULT)
    ap.add_argument("--minlength", type=int, default=0,
                    help="antiSMASH --minlength. 0 = leave at the default 1,000 nt. Lower it for "
                         "short-core classes (TERPENE), and then NEVER compare those numbers "
                         "against default-minlength numbers -- it is part of the scoring config.")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.gen.open()]
    if args.limit:
        recs = recs[:args.limit]
    seqs = [r["sequence"][:args.window] if args.window else r["sequence"] for r in recs]

    # INTEGRITY: exact duplicates make every rate carry a false n (bugs.md, fan-out collision).
    if len(set(seqs)) != len(seqs):
        raise SystemExit(
            f"[antismash_full] {len(seqs) - len(set(seqs))} of {len(seqs)} records are "
            f"byte-identical; effective n is {len(set(seqs))}. Deduplicate before scoring.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, r.get("accession", f"rec{i}"), s, str(args.out_dir),
             args.databases, args.cpus_per_worker, args.minlength)
            for i, (r, s) in enumerate(zip(recs, seqs))]

    print(f"[antismash_full] {len(jobs)} sequences, FULL mode, window "
          f"{args.window or 'whole'}, minlength {args.minlength or 'default(1000)'}, "
          f"{args.workers} workers x {args.cpus_per_worker} cpus", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run_one, jobs))

    # COMPLETENESS: a filtered count cannot tell "found nothing" from "ran nothing" (CLAUDE.md).
    if len(res) != len(jobs):
        raise SystemExit(f"[antismash_full] expected {len(jobs)} results, got {len(res)}")

    with args.tsv.open("w") as fh:
        fh.write("i\taccession\tran\tdetected\tn_products\tproducts\n")
        for r in sorted(res, key=lambda x: x["i"]):
            fh.write(f"{r['i']}\t{r['accession']}\t{r['ran']}\t{r['detected']}\t"
                     f"{len(r['products'])}\t{';'.join(r['products'])}\n")

    ran = sum(r["ran"] for r in res)
    det = sum(r["detected"] for r in res)
    if ran == 0:
        raise SystemExit("[antismash_full] ZERO sequences ran. That is a broken install, "
                         "not a rate of 0.0 -- refusing to report it as a result.")
    if ran < len(res):
        print(f"  ⚠️ {len(res) - ran}/{len(res)} sequences DID NOT RUN. That is an instrument "
              f"limit, not a negative -- check the per-sequence logs before quoting any rate. "
              f"The usual cause is antiSMASH's 1,000 nt minimum; see --minlength.")
    print(f"[antismash_full] ran {ran}/{len(res)}, detected {det}/{ran} = {det/ran:.3f}")
    print(f"[antismash_full] wrote {args.tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
