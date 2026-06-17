#!/usr/bin/env python3
"""Build a leakage-clean held-out val/test for CONTINUED training.

Context: the genome-keyed split kept genomes disjoint, but near-DUPLICATE BGCs
(>=~90% identity across different genomes) still crossed the train/test boundary
(7/20 positive-control BGCs were near-dups of train). Since we are *continuing*
training from an existing checkpoint, ``train`` is kept FIXED (the model already
learned it). We rebuild a clean held-out:

  1. Remove from val/test any record that is a near-duplicate of a TRAIN record
     (MMseqs2 nucleotide search, >=80% identity over >=50% of the record length).
     -> the resumed model (trained on train) provably never saw the clean test.
  2. Remove val records that are near-dups of the CLEAN TEST, so val (used for
     resume monitoring / early stopping) is independent of the final test set.

train.jsonl is copied unchanged. Writes train/val/test to --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def write_fasta(path: Path, recs: list[dict], ids: list[str]) -> None:
    with open(path, "w") as f:
        for i, r in zip(ids, recs):
            f.write(f">{i}\n{r['sequence']}\n")


def mmseqs_neardup_query_ids(query_recs, qids, target_recs, tids, env, workdir,
                             idthr=0.8, cov=0.5, threads=8) -> set[str]:
    """Return the set of query ids with a near-dup hit in target (>=idthr identity
    over >=cov of the QUERY length)."""
    if not query_recs or not target_recs:
        return set()
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        qf, tf, out = Path(tmp) / "q.fa", Path(tmp) / "t.fa", Path(tmp) / "h.tsv"
        write_fasta(qf, query_recs, qids)
        write_fasta(tf, target_recs, tids)
        subprocess.run(
            ["micromamba", "run", "-n", env, "mmseqs", "easy-search",
             str(qf), str(tf), str(out), str(Path(tmp) / "tmp"),
             "--search-type", "3", "--min-seq-id", str(idthr), "-c", str(cov),
             "--cov-mode", "2", "--format-output", "query,target,fident,qcov",
             "-e", "1e-5", "--threads", str(threads)],
            check=True, capture_output=True, text=True,
        )
        nd = set()
        if out.exists():
            for line in out.read_text().splitlines():
                if line.strip():
                    nd.add(line.split("\t")[0])
        return nd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_curated"))
    ap.add_argument("--hits", type=Path, default=Path("/data2/ds85/dedup_work/nd_hits.tsv"),
                    help="MMseqs2 (val+test vs train) hits TSV; query ids va_i / te_i.")
    ap.add_argument("--out", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_dedup"))
    ap.add_argument("--env", default="bgcmodel")
    ap.add_argument("--workdir", type=Path, default=Path("/data2/ds85/dedup_work"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    tr = load(args.src / "train.jsonl")
    va = load(args.src / "val.jsonl")
    te = load(args.src / "test.jsonl")

    # 1. drop val/test records that are near-dups of TRAIN (precomputed scan)
    nd_train = set()
    for line in args.hits.read_text().splitlines():
        if line.strip():
            nd_train.add(line.split("\t")[0])
    clean_te = [r for i, r in enumerate(te) if f"te_{i}" not in nd_train]
    clean_va0 = [r for i, r in enumerate(va) if f"va_{i}" not in nd_train]
    print(f"[1] train near-dups removed: test {len(te) - len(clean_te)}/{len(te)} "
          f"-> {len(clean_te)} | val {len(va) - len(clean_va0)}/{len(va)} -> {len(clean_va0)}")

    # 2. make val independent of the CLEAN TEST (val ⊥ test)
    nd_vt = mmseqs_neardup_query_ids(
        clean_va0, [f"cv_{i}" for i in range(len(clean_va0))],
        clean_te, [f"ct_{i}" for i in range(len(clean_te))], args.env, str(args.workdir))
    clean_va = [r for i, r in enumerate(clean_va0) if f"cv_{i}" not in nd_vt]
    print(f"[2] val⊥test near-dups removed: {len(clean_va0) - len(clean_va)}/{len(clean_va0)} "
          f"-> {len(clean_va)}")

    # 3. write outputs (train fixed)
    shutil.copy(args.src / "train.jsonl", args.out / "train.jsonl")
    (args.out / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in clean_va))
    (args.out / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in clean_te))

    # 4. verify: 0 byte-identical overlap test↔train, report class coverage
    htr = {md5(r["sequence"]) for r in tr}
    bad = sum(1 for r in clean_te if md5(r["sequence"]) in htr)
    print(f"[verify] byte-identical test∩train = {bad} (must be 0); "
          f"near-dup test∩train removed via MMseqs2 (>=80% id, >=50% cov)")
    print(f"WROTE {args.out}:  train={len(tr)} (UNCHANGED)  val={len(clean_va)}  test={len(clean_te)}")
    print(f"  test classes ({len(set(r['compound_class'] for r in clean_te))}): "
          f"{dict(Counter(r['compound_class'] for r in clean_te).most_common(8))} ...")


if __name__ == "__main__":
    main()
