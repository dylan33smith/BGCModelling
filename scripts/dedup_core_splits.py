#!/usr/bin/env python3
"""Cross-split near-duplicate removal for the strict-core splits (v2).

The grouped split already guarantees genome-disjoint + exact-sequence-disjoint
splits. This removes CROSS-GENOME near-duplicates (the same BGC core present in
different strains that landed in different splits) so the held-out val/test are
genuinely independent of train. Train is kept full; val/test are cleaned (then
val is made independent of the cleaned test). MMseqs2 nucleotide near-dup search
(>=80% identity over >=50% of the query), reusing build_clean_holdout helpers.

Input : splits_core_curated/{train,val,test}.jsonl
Output: splits_core/{train,val,test}.jsonl
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_clean_holdout import load, md5, mmseqs_neardup_query_ids  # noqa: E402

SRC = Path("/data2/ds85/bgcmodel_data/splits_core_curated")
OUT = Path("/data2/ds85/bgcmodel_data/splits_core")
WORK = Path("/data2/ds85/dedup_core_work")
ENV = "bgcmodel"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    tr = load(SRC / "train.jsonl")
    va = load(SRC / "val.jsonl")
    te = load(SRC / "test.jsonl")
    print(f"input: train={len(tr)} val={len(va)} test={len(te)}")

    # 1. drop val/test cores that are near-dups of TRAIN
    nd_va = mmseqs_neardup_query_ids(va, [f"va_{i}" for i in range(len(va))],
                                     tr, [f"tr_{i}" for i in range(len(tr))], ENV, str(WORK))
    nd_te = mmseqs_neardup_query_ids(te, [f"te_{i}" for i in range(len(te))],
                                     tr, [f"tr_{i}" for i in range(len(tr))], ENV, str(WORK))
    cva0 = [r for i, r in enumerate(va) if f"va_{i}" not in nd_va]
    cte = [r for i, r in enumerate(te) if f"te_{i}" not in nd_te]
    print(f"[1] vs TRAIN near-dups removed: val {len(va)-len(cva0)} -> {len(cva0)} | "
          f"test {len(te)-len(cte)} -> {len(cte)}")

    # 2. make val independent of the cleaned TEST
    nd_vt = mmseqs_neardup_query_ids(cva0, [f"cv_{i}" for i in range(len(cva0))],
                                     cte, [f"ct_{i}" for i in range(len(cte))], ENV, str(WORK))
    cva = [r for i, r in enumerate(cva0) if f"cv_{i}" not in nd_vt]
    print(f"[2] val⊥test near-dups removed: {len(cva0)-len(cva)} -> {len(cva)}")

    # 3. write (train full)
    (OUT / "train.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in tr))
    (OUT / "val.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in cva))
    (OUT / "test.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in cte))

    # 4. verify zero byte-identical cross-split overlap
    htr = {md5(r["sequence"]) for r in tr}
    bad_va = sum(1 for r in cva if md5(r["sequence"]) in htr)
    bad_te = sum(1 for r in cte if md5(r["sequence"]) in htr)
    print(f"[verify] byte-identical val∩train={bad_va} test∩train={bad_te} (must be 0)")
    print(f"WROTE {OUT}: train={len(tr)} val={len(cva)} test={len(cte)}")


if __name__ == "__main__":
    main()
