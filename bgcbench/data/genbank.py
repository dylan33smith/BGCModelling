"""Targeted GenBank reader.

Only the six things the corpus needs are parsed: LOCUS id/length, the ORIGIN sequence,
and the `region` / `protocluster` / `proto_core` / `CDS` features. A general-purpose
parser reads far more than that and is an order of magnitude slower across a corpus this
size.

Coordinate convention, fixed here once: GenBank locations are 1-based inclusive.
Everything this module returns is converted to **0-based half-open** [start, end), which
is what Python slicing expects. `region_start` in an emitted record is therefore one less
than the number printed in the GenBank file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_LOC = re.compile(r"(\d+)\.\.(\d+)")


@dataclass
class Feature:
    key: str
    start: int                       # 0-based inclusive  (bounding box)
    end: int                         # 0-based exclusive  (bounding box)
    strand: int                      # +1 / -1
    spans: list[tuple[int, int]] = field(default_factory=list)
    quals: dict[str, list[str]] = field(default_factory=dict)

    def overlaps(self, lo: int, hi: int) -> bool:
        """Test the ACTUAL intervals, not the bounding box.

        A join()/order() feature is a set of intervals. Collapsing it to min..max makes an
        origin-spanning gene on a circular replicon -- join(2000000..2009313,1..500) --
        look like a single 2 Mb feature that overlaps every cluster on that replicon.
        Measured on 120 genomes: 43 of 436,077 CDS exceeded 20 kb under the collapse, some
        spanning an entire chromosome.
        """
        for a, b in (self.spans or [(self.start, self.end)]):
            if a < hi and lo < b:
                return True
        return False

    def q1(self, name: str, default: str | None = None) -> str | None:
        v = self.quals.get(name)
        return v[0] if v else default


@dataclass
class Record:
    locus: str
    length: int
    sequence: str
    features: list[Feature]

    def of(self, key: str) -> list[Feature]:
        return [f for f in self.features if f.key == key]


def _parse_location(loc: str) -> tuple[int, int, int, list[tuple[int, int]]]:
    """Return (start0, end, strand, spans). `spans` holds EVERY interval of a
    join()/order(); start/end are the bounding box and must not be used for overlap."""
    strand = -1 if "complement" in loc else 1
    pairs = _LOC.findall(loc)
    if not pairs:
        single = re.findall(r"(\d+)", loc)
        if not single:
            raise ValueError(f"unparseable location: {loc!r}")
        p = int(single[0])
        return p - 1, p, strand, [(p - 1, p)]
    spans = [(int(a) - 1, int(b)) for a, b in pairs]
    return min(s for s, _ in spans), max(e for _, e in spans), strand, spans


def parse(text: str, want: frozenset[str] | None = None):
    """Yield Record per LOCUS. `want` limits which feature keys are retained."""
    want = want or frozenset({"region", "protocluster", "proto_core", "CDS"})
    locus = None
    length = 0
    feats: list[Feature] = []
    seq_parts: list[str] = []
    in_features = in_origin = False
    cur: Feature | None = None
    pending_key: str | None = None
    pending_val: list[str] = []

    def flush_qual():
        nonlocal pending_key, pending_val
        if cur is not None and pending_key is not None:
            cur.quals.setdefault(pending_key, []).append(
                "".join(pending_val).strip().strip('"')
            )
        pending_key, pending_val = None, []

    for line in text.splitlines():
        if line.startswith("LOCUS"):
            parts = line.split()
            locus, length = parts[1], int(parts[2])
            feats, seq_parts = [], []
            in_features = in_origin = False
            cur = None
            continue
        if line.startswith("FEATURES"):
            in_features, in_origin = True, False
            continue
        if line.startswith("ORIGIN"):
            flush_qual()
            in_features, in_origin = False, True
            continue
        if line.startswith("//"):
            flush_qual()
            if locus is not None:
                yield Record(locus, length, "".join(seq_parts).upper(), feats)
            locus, cur = None, None
            in_features = in_origin = False
            continue

        if in_origin:
            seq_parts.append("".join(line.split()[1:]))
            continue

        if in_features:
            if len(line) > 5 and line[5] != " " and not line[:5].strip():
                # new feature: 5 spaces, key, location
                flush_qual()
                key, _, loc = line[5:].strip().partition(" ")
                loc = loc.strip()
                if key in want and loc:
                    try:
                        s, e, st, spans = _parse_location(loc)
                    except ValueError:
                        cur = None
                        continue
                    cur = Feature(key, s, e, st, spans)
                    feats.append(cur)
                else:
                    cur = None
            elif cur is not None and line.strip().startswith("/"):
                flush_qual()
                body = line.strip()[1:]
                k, eq, v = body.partition("=")
                if not eq:
                    cur.quals.setdefault(k, []).append("")
                else:
                    pending_key, pending_val = k, [v]
            elif cur is not None and pending_key is not None:
                pending_val.append(" " + line.strip())

    if locus is not None:
        flush_qual()
        yield Record(locus, length, "".join(seq_parts).upper(), feats)
