"""Within-RIPP multi-gene stratification (SPEC 4.5).

WHY WITHIN A CLASS AND NOT ACROSS CLASSES. Multi-gene content and sequence length are
anti-correlated across classes -- every class above ~65% multi-gene is also long -- so an
across-class comparison cannot separate "multi-gene is harder" from "long is harder", and
additionally varies class identity, effective_n, hybrid rate and subclass structure all at
once. Stratifying inside one class holds every one of those fixed.

RIPP is the class that makes this possible: 44% multi-gene at a median of 1.8 kb, so both
strata exist in quantity and neither is forced against the context bound.

LENGTH MATCHING IS THE POINT. Multi-gene cores are longer than single-gene cores by
construction, so an unmatched contrast would measure length, which is the exact confound
this design exists to remove. Strata are matched by binning on length and taking equal
counts per bin, then truncated to equal size.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BIN_NT = 250          # length-matching resolution


def _bin(n: int) -> int:
    return n // BIN_NT


def stratify(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Return (single, multi, report) matched on the length distribution."""
    single = [r for r in records if r["core_gene_count"] == 1]
    multi = [r for r in records if r["core_gene_count"] >= 2]

    by_bin_s, by_bin_m = defaultdict(list), defaultdict(list)
    for r in single:
        by_bin_s[_bin(r["seq_len"])].append(r)
    for r in multi:
        by_bin_m[_bin(r["seq_len"])].append(r)

    keep_s, keep_m = [], []
    for b in sorted(set(by_bin_s) & set(by_bin_m)):
        # deterministic: order within a bin by accession, take the same count from each
        s = sorted(by_bin_s[b], key=lambda r: r["accession"])
        m = sorted(by_bin_m[b], key=lambda r: r["accession"])
        k = min(len(s), len(m))
        keep_s.extend(s[:k])
        keep_m.extend(m[:k])

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else 0

    report = {
        "available": {"single": len(single), "multi": len(multi)},
        "matched": {"single": len(keep_s), "multi": len(keep_m)},
        "bin_nt": BIN_NT,
        "median_len": {"single": med([r["seq_len"] for r in keep_s]),
                       "multi": med([r["seq_len"] for r in keep_m])},
        "median_core_genes": {"single": med([r["core_gene_count"] for r in keep_s]),
                              "multi": med([r["core_gene_count"] for r in keep_m])},
        "median_cds": {"single": med([r["cds_count"] for r in keep_s]),
                       "multi": med([r["cds_count"] for r in keep_m])},
        "n_products": {"single": len({p for r in keep_s
                                      for p in r["antismash_products"]}),
                       "multi": len({p for r in keep_m
                                     for p in r["antismash_products"]})},
    }
    return keep_s, keep_m, report


def build(split_dir: Path, out_dir: Path, cls: str = "RIPP",
          corpus_path: Path | None = None, max_len: int | None = None,
          exclude: set[str] | None = None) -> dict:
    """Stratify each split of `cls` independently, inheriting the train/val/test boundary
    the class corpus established rather than re-drawing it.

    Draws from the FULL class pool, not the equal-n subsample. The common_n subsample
    exists for the cross-class comparison; length matching is already lossy (the strata
    overlap only where their length distributions do), so inheriting that cut as well
    leaves ~150 records per stratum where several thousand are available. Split
    assignment still comes from `record_split.json`, so global consistency holds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pool: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    rs_path = split_dir / "record_split.json"
    if corpus_path and rs_path.exists():
        record_split = json.loads(rs_path.read_text())
        with open(corpus_path) as fh:
            for line in fh:
                r = json.loads(line)
                if max_len and r["seq_len"] > max_len:
                    continue
                if cls not in r["classes"]:
                    continue
                if exclude and r["accession"] in exclude:
                    continue
                part = record_split.get(r["accession"])
                if part:
                    pool[part].append(r)
    report: dict[str, dict] = {}
    for part in ("train", "val", "test"):
        rows = pool[part] or [json.loads(l)
                              for l in open(split_dir / cls / f"{part}.jsonl")]
        s, m, rep = stratify(rows)
        for name, sel in (("single", s), ("multi", m)):
            d = out_dir / f"{cls}_{name}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{part}.jsonl", "w") as fh:
                for r in sorted(sel, key=lambda x: x["accession"]):
                    fh.write(json.dumps(r) + "\n")
        report[part] = rep
    return report
