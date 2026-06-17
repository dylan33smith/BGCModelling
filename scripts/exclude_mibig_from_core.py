#!/usr/bin/env python3
"""Remove MiBIG-overlapping cores from splits_core so v2 never sees MiBIG content.

splits_core comes from antiSMASH-DB (asdb5) RefSeq genome regions — no MiBIG
ACCESSIONS, but the same clusters MiBIG characterizes are embedded via their
RefSeq source genomes. To reserve MiBIG for a later compound-conditioned Phase-2
(and to make the MiBIG positive-control eval genuinely held-out), drop any
splits_core core that is a near-duplicate of a MiBIG BGC (MMseqs2 nucleotide
search: >=80% identity over >=50% of the CORE length).

Backs up splits_core -> splits_core_premibig, writes the cleaned splits in place.
"""
from __future__ import annotations
import gzip
import json
import shutil
import sys
from pathlib import Path

from Bio import SeqIO  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_clean_holdout import load, mmseqs_neardup_query_ids  # noqa: E402

CORE = Path("/data2/ds85/bgcmodel_data/splits_core")
BACKUP = Path("/data2/ds85/bgcmodel_data/splits_core_premibig")
MIBIG_DIR = Path("/home/ds85/projects/BCGModelling/data/mibig/mibig_gbk_4.0")
WORK = Path("/data2/ds85/mibig_excl_work")
ENV = "bgcmodel"


def load_mibig_seqs() -> list[dict]:
    recs = []
    files = sorted(MIBIG_DIR.glob("*.gbk")) + sorted(MIBIG_DIR.glob("*.gbk.gz"))
    for fp in files:
        try:
            handle = gzip.open(fp, "rt") if fp.suffix == ".gz" else open(fp)
            for rec in SeqIO.parse(handle, "genbank"):
                s = str(rec.seq).upper()
                if s and set(s) <= set("ACGTN"):
                    recs.append({"sequence": s, "accession": fp.stem})
        except Exception as e:
            print(f"  WARN: skip {fp.name}: {e}", file=sys.stderr)
    return recs


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    mibig = load_mibig_seqs()
    print(f"MiBIG target sequences: {len(mibig)} from {MIBIG_DIR.name}")

    if not BACKUP.exists():
        shutil.copytree(CORE, BACKUP)
        print(f"backed up splits_core -> {BACKUP}")

    tids = [f"mb_{i}" for i in range(len(mibig))]
    totals = {}
    for split in ["train", "val", "test"]:
        recs = load(BACKUP / f"{split}.jsonl")   # read from backup (pristine)
        qids = [f"{split}_{i}" for i in range(len(recs))]
        nd = mmseqs_neardup_query_ids(recs, qids, mibig, tids, ENV, str(WORK))
        clean = [r for i, r in enumerate(recs) if f"{split}_{i}" not in nd]
        (CORE / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in clean))
        totals[split] = (len(recs), len(recs) - len(clean), len(clean))
        print(f"[{split}] {len(recs)} -> removed {len(recs)-len(clean)} MiBIG near-dups -> {len(clean)}")

    print("\nSUMMARY (split: before -> removed -> after):")
    for s, (b, rm, a) in totals.items():
        print(f"  {s:5} {b:6} -> -{rm:5} -> {a:6}")
    print(f"backup of pre-exclusion splits at {BACKUP}")


if __name__ == "__main__":
    main()
