#!/usr/bin/env python3
"""End-to-end invariant tests for the split + curation pipeline (audit C1/C2 + M8).

Runs scripts/split_dataset_grouped.py then scripts/curate_dataset.py over a tiny
synthetic fixture and asserts the guarantees we rely on:
  - group-aware split: genomes are DISJOINT across train/val/test
  - no exact sequence leaks across splits (global dedup)
  - curation quality filters: no N/ambiguous, no contig_edge, none below min-len
  - per-class TRAIN cap is respected
  - curation preserves leakage-freedom (genomes still disjoint)

These guard against a future edit silently reintroducing leakage or breaking the
caps. Fast: a 60-record fixture, no GPU, no real data.

Run: python tests/test_data_pipeline.py
"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import split_dataset_grouped as SPLIT  # noqa: E402
import curate_dataset as CURATE        # noqa: E402

MIN_LEN = 1000
CLASSES = ["TERPENE", "NRPS", "PKS", "RIPP"]
PHYLA = ["P__ACTINOMYCETOTA", "P__PSEUDOMONADOTA"]
DUP_SEQ = "ACGT" * 600  # shared by two records in different genomes (dedup test)


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def unique_seq(rid: int) -> str:
    """A distinct, ACGT-only ~2000 nt sequence per record id."""
    suffix = ""
    x = rid + 1
    for _ in range(100):
        suffix += "ACGT"[x % 4]
        x //= 4
    return ("ACGT" * 475) + suffix  # 1900 + 100 = 2000 nt, unique per rid


def make_fixture(path: Path):
    recs = []
    rid = 0
    for g in range(20):                      # 20 genomes -> 80/10/10 gives non-empty val/test
        genome = f"G{g:02d}"
        phy = PHYLA[g % 2]
        for _ in range(3):                   # 3 BGCs per genome = 60 records
            cls = CLASSES[rid % len(CLASSES)]
            tax = f"|D__BACTERIA;{phy};S__SP{g}"
            seq = unique_seq(rid)            # clean, distinct, >= MIN_LEN
            contig_edge = False
            if rid in (5, 17):               # N-containing -> must be dropped
                seq = seq[:1000] + "NNNNN" + seq[1005:]
            if rid in (8, 19):               # contig_edge -> must be dropped
                contig_edge = True
            if rid in (10, 25):              # identical sequence, different genomes
                seq = DUP_SEQ
            recs.append({
                "accession": f"REC_{rid:03d}",
                "genome_accession": genome,
                "compound_class": cls,
                "taxonomic_tag": tax,
                "sequence": seq,
                "training_text": f"|COMPOUND_CLASS:{cls}|{tax}{seq}",
                "contig_edge": contig_edge,
            })
            rid += 1
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return recs


def load(path: Path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def run_main(module, argv):
    saved = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            module.main()
    finally:
        sys.argv = saved


def genomes(recs):
    return {r["genome_accession"] for r in recs}


def seqset(recs):
    return {md5(r["sequence"]) for r in recs}


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fixture = tmp / "fixture.jsonl"
        grouped = tmp / "grouped"
        curated = tmp / "curated"
        make_fixture(fixture)

        # ── 1) Group-aware split ─────────────────────────────────────────────
        run_main(SPLIT, ["split", "--inputs", str(fixture),
                         "--output-dir", str(grouped), "--seed", "42"])
        tr = load(grouped / "train.jsonl")
        va = load(grouped / "val.jsonl")
        te = load(grouped / "test.jsonl")
        assert tr and va and te, f"all splits non-empty (got {len(tr)},{len(va)},{len(te)})"

        gtr, gva, gte = genomes(tr), genomes(va), genomes(te)
        assert gtr & gva == set(), f"train/val genome leak: {gtr & gva}"
        assert gtr & gte == set(), f"train/test genome leak: {gtr & gte}"
        assert gva & gte == set(), f"val/test genome leak: {gva & gte}"

        str_, sva, ste = seqset(tr), seqset(va), seqset(te)
        assert str_ & sva == set(), "train/val exact-sequence leak"
        assert str_ & ste == set(), "train/test exact-sequence leak"
        assert sva & ste == set(), "val/test exact-sequence leak"

        # Global dedup: the duplicated sequence survives exactly once.
        all_recs = tr + va + te
        assert sum(1 for r in all_recs if r["sequence"] == DUP_SEQ) == 1, \
            "exact-duplicate sequence should be deduped to a single copy"
        print(f"PASS split: {len(tr)}/{len(va)}/{len(te)} train/val/test — "
              f"genomes & exact sequences disjoint; duplicate deduped")

        # ── 2) Curation ──────────────────────────────────────────────────────
        run_main(CURATE, ["curate", "--input-dir", str(grouped),
                          "--output-dir", str(curated),
                          "--train-cap", "2", "--eval-cap", "0", "--seed", "42"])
        ctr = load(curated / "train.jsonl")
        cva = load(curated / "val.jsonl")
        cte = load(curated / "test.jsonl")
        assert ctr, "curated train non-empty"

        for name, recs in (("train", ctr), ("val", cva), ("test", cte)):
            for r in recs:
                s = r["sequence"].upper()
                assert set(s) <= set("ACGT"), f"{name}: ambiguous base survived in {r['accession']}"
                assert r.get("contig_edge") is not True, f"{name}: contig_edge survived"
                assert len(s) >= MIN_LEN, f"{name}: below min-len survived"

        cls_counts = Counter(r["compound_class"] for r in ctr)
        assert max(cls_counts.values()) <= 2, f"train cap=2 violated: {dict(cls_counts)}"
        assert max(cls_counts.values()) == 2, \
            f"expected at least one class to hit the cap, got {dict(cls_counts)}"

        # no within-train exact duplicates, and leakage still zero
        assert len(seqset(ctr)) == len(ctr), "curated train has duplicate sequences"
        assert genomes(ctr) & genomes(cva) == set(), "curated train/val genome leak"
        assert genomes(ctr) & genomes(cte) == set(), "curated train/test genome leak"
        print(f"PASS curate: train per-class {dict(cls_counts)} (cap 2 respected); "
              f"no N/contig-edge/short; leakage-free preserved")

    print("\nALL DATA-PIPELINE TESTS PASSED")


if __name__ == "__main__":
    main()
