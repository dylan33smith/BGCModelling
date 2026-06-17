#!/usr/bin/env python3
"""Build empirical per-taxon ReferenceProfiles for Metric 7 (organism compatibility).

Metric 7 grades a generated sequence's *faithfulness* to the organism it was
conditioned on (GC, codon usage, dinucleotide signature). To do that it needs a
reference profile per taxon. This script derives those profiles empirically from
real BGC training sequences, grouped by phylum token (e.g. P__ACTINOMYCETOTA),
and writes them as JSON consumable by ``evaluation.load_taxon_profiles``.

See docs/archive/AUDIT_FINDINGS.md C4. Memory-safe: single streaming pass with incremental
per-phylum accumulators (sequences are never all held in RAM).

Example:
    python scripts/build_taxon_profiles.py \
        --input /data2/ds85/bgcmodel_data/splits_combined_grouped/train.jsonl \
        --output data/processed/taxon_profiles.json \
        --min-records 200 --max-per-phylum 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Reuse the canonical codon table and phylum parser from the eval module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bgc_pipeline.evaluation import _CODON_TABLE, phylum_token  # noqa: E402


class _Accum:
    """Incremental GC / codon / dinucleotide accumulator for one phylum."""

    __slots__ = ("n", "gc", "bases", "codons", "mono", "di", "di_total")

    def __init__(self) -> None:
        self.n = 0
        self.gc = 0
        self.bases = 0
        self.codons: Counter = Counter()
        self.mono: Counter = Counter()
        self.di: Counter = Counter()
        self.di_total = 0

    def add(self, seq: str) -> None:
        s = seq.upper()
        if not s:
            return
        self.n += 1
        self.gc += s.count("G") + s.count("C")
        self.bases += len(s)
        self.mono.update(s)
        for i in range(len(s) - 1):
            self.di[s[i:i + 2]] += 1
        self.di_total += max(len(s) - 1, 0)
        for i in range(0, len(s) - 2, 3):
            codon = s[i:i + 3]
            if codon in _CODON_TABLE:
                self.codons[codon] += 1

    def to_profile(self, name: str, gc_tol: float, cai_thr: float, dinuc_thr: float) -> dict:
        target_gc = self.gc / self.bases if self.bases else 0.0
        total_codons = sum(self.codons.values())
        codon_freq = {
            c: (self.codons.get(c, 0) / total_codons * 1000.0) if total_codons else 0.0
            for c in _CODON_TABLE
        }
        dinuc_ratios = {}
        for d, cnt in self.di.items():
            if len(d) == 2 and self.mono[d[0]] and self.mono[d[1]] and self.bases:
                expected = (self.mono[d[0]] / self.bases) * (self.mono[d[1]] / self.bases) * self.di_total
                if expected > 0:
                    dinuc_ratios[d] = round(cnt / expected, 5)
        return {
            "name": name,
            "target_gc": round(target_gc, 4),
            "codon_freq": {c: round(v, 4) for c, v in codon_freq.items()},
            "dinuc_ratios": dinuc_ratios,
            "gc_tol": gc_tol,
            "cai_threshold": cai_thr,
            "dinuc_threshold": dinuc_thr,
            "_n_records": self.n,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_combined_grouped/train.jsonl"),
                    help="Training JSONL to derive profiles from (use train, not val/test).")
    ap.add_argument("--output", type=Path, default=Path("data/processed/taxon_profiles.json"))
    ap.add_argument("--min-records", type=int, default=200,
                    help="Skip phyla with fewer than this many records.")
    ap.add_argument("--max-per-phylum", type=int, default=5000,
                    help="Cap records counted per phylum (0 = no cap).")
    ap.add_argument("--gc-tol", type=float, default=0.10)
    ap.add_argument("--cai-threshold", type=float, default=0.7)
    ap.add_argument("--dinuc-threshold", type=float, default=0.15)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    accums: dict[str, _Accum] = {}
    n = 0
    no_phylum = 0
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            phy = phylum_token(rec.get("taxonomic_tag", ""))
            if not phy:
                no_phylum += 1
                continue
            acc = accums.get(phy)
            if acc is None:
                acc = accums[phy] = _Accum()
            if args.max_per_phylum and acc.n >= args.max_per_phylum:
                continue
            acc.add(rec.get("sequence", ""))
            if n % 50000 == 0:
                print(f"  ...{n:,} records, {len(accums)} phyla", file=sys.stderr, flush=True)

    profiles = {}
    for phy, acc in sorted(accums.items(), key=lambda kv: -kv[1].n):
        if acc.n < args.min_records:
            continue
        profiles[phy] = acc.to_profile(
            phy.replace("P__", "").lower(), args.gc_tol, args.cai_threshold, args.dinuc_threshold,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profiles, indent=2))

    print(f"\nScanned {n:,} records ({no_phylum:,} without phylum tag).", file=sys.stderr)
    print(f"Wrote {len(profiles)} taxon profiles to {args.output}:", file=sys.stderr)
    for phy, p in profiles.items():
        print(f"  {phy:<32} GC={p['target_gc']:.3f}  n={p['_n_records']:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
