"""The negative control corpus: real non-BGC DNA (SPEC 4.8).

Without this there is no measured false-positive rate, so no rate in the benchmark has a
floor, and the `--minlength 1` decision (SPEC 3.1) stays provisional.

IT MUST BE HARD. Shuffled sequence is not a control: it contains no genes at all, so
antiSMASH rejects it trivially and the resulting "FPR 0.000" measures nothing except that
the tool needs open reading frames. The control here is real coding DNA from the SAME
genomes -- real genes, real codon usage, real coding density -- that simply contains no
annotated cluster.

Sampled intervals keep a margin from every annotated region so no partial cluster leaks
in, and are length-matched per class, because otherwise length rather than content is what
separates a control from a core.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from bgcbench.data import tar_index
from bgcbench.data.genbank import parse

MARGIN = 5000          # SPEC 4.8: keep this far clear of any region boundary
MIN_CODING_CDS = 1     # an interval with no CDS is not "real coding DNA"


def _frac(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2**64


def _free_intervals(length: int, regions: list[tuple[int, int]],
                    margin: int) -> list[tuple[int, int]]:
    """Genome span minus every region, each padded by `margin` on both sides."""
    blocked = sorted((max(0, s - margin), min(length, e + margin)) for s, e in regions)
    merged: list[list[int]] = []
    for s, e in blocked:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    free, cur = [], 0
    for s, e in merged:
        if s > cur:
            free.append((cur, s))
        cur = max(cur, e)
    if cur < length:
        free.append((cur, length))
    return free


def sample_from_genome(acc: str, index: dict, targets: list[int],
                       margin: int = MARGIN) -> list[dict]:
    """Draw one interval per requested length from this genome's non-region space."""
    text = tar_index.read_member(acc, index)
    out: list[dict] = []
    for rec in parse(text):
        regions = [(r.start, r.end) for r in rec.of("region")]
        cds = sorted((c.start, c.end) for c in rec.of("CDS"))
        starts = [c[0] for c in cds]
        free = _free_intervals(len(rec.sequence), regions, margin)
        if not free:
            continue
        for i, want in enumerate(list(targets)):
            usable = [(s, e) for s, e in free if e - s >= want]
            if not usable:
                continue
            # deterministic choice: hash on (genome, locus, index, length)
            key = f"{acc}|{rec.locus}|{i}|{want}"
            s, e = usable[int(_frac(key + "|iv") * len(usable))]
            off = int(_frac(key + "|off") * (e - s - want + 1))
            a, b = s + off, s + off + want
            seq = rec.sequence[a:b]
            if not seq or set(seq) <= {"N"}:
                continue
            lo = bisect.bisect_left(starts, a)
            hi = bisect.bisect_right(starts, b)
            n_cds = sum(1 for cs, ce in cds[max(0, lo - 5):hi + 5]
                        if cs < b and a < ce)
            if n_cds < MIN_CODING_CDS:
                continue
            gc = sum(seq.count(x) for x in "GC") / len(seq)
            out.append({
                "accession": f"{acc}.neg{i}",
                "genome_accession": acc,
                "locus": rec.locus,
                "sequence": seq,
                "seq_len": len(seq),
                "n_cds": n_cds,
                "gc": round(gc, 4),
                "source_interval": [a, b],
                "margin": margin,
            })
            targets[i] = -1                       # consume this request
        targets[:] = [t for t in targets if t > 0]
        if not targets:
            break
    return out


def length_targets(class_records: list[dict], n: int) -> list[int]:
    """Length-match to the class corpus: draw the empirical length distribution."""
    lens = sorted(r["seq_len"] for r in class_records)
    if not lens:
        return []
    # even quantile sweep reproduces the distribution without sampling noise
    return [lens[min(len(lens) - 1, int((i + 0.5) / n * len(lens)))] for i in range(n)]


def verify(negatives: list[dict], class_records: list[dict]) -> dict:
    """SPEC 4.8: a control that is obviously different from the cores is not a control."""
    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else 0
    neg_gc = [r["gc"] for r in negatives]
    pos_gc = [round(sum(r["sequence"].count(x) for x in "GC") / max(r["seq_len"], 1), 4)
              for r in class_records]
    return {
        "n": len(negatives),
        "median_len_neg": med([r["seq_len"] for r in negatives]),
        "median_len_pos": med([r["seq_len"] for r in class_records]),
        "median_gc_neg": med(neg_gc),
        "median_gc_pos": med(pos_gc),
        "median_cds_neg": med([r["n_cds"] for r in negatives]),
        "genomes": len({r["genome_accession"] for r in negatives}),
    }
