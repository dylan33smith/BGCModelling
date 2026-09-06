"""The novelty gate (SPEC 3.8).

Every ladder metric is maximised by reproducing training data, so novelty is an ABSOLUTE
GATE applied before any rate is read -- not a number reported alongside one.

⚠ IT FAILS CLOSED. KNOWN_WRONG #3: the prior implementation computed containment with
`max(..., default=0.0)`, and 0.0 is the PASSING value, so an empty k-mer set silently
passed the gate. A gate that defaults to passing is not a gate. Every path here that
cannot compute a containment raises.
"""
from __future__ import annotations

from collections import defaultdict

K = 21
FAIL_AT = 0.95
WARN_AT = 0.80
_COMP = str.maketrans("ACGTN", "TGCAN")


def canonical_kmers(seq: str, k: int = K) -> set[str]:
    s = seq.upper()
    out: set[str] = set()
    for i in range(len(s) - k + 1):
        km = s[i:i + k]
        if "N" in km:
            continue
        rc = km.translate(_COMP)[::-1]
        out.add(min(km, rc))
    return out


class Reference:
    """Inverted k-mer index over ONE arm's training split.

    The reference set is per arm (SPEC 3.8): an arm trained on a pooled corpus is gated
    against the pooled corpus. Using one shared reference would under-gate whichever arm
    trained on more data.
    """

    def __init__(self, records: list[dict], k: int = K):
        self.k = k
        self.n = len(records)
        self.index: dict[str, set[int]] = defaultdict(set)
        self.sizes: list[int] = []
        for i, r in enumerate(records):
            km = canonical_kmers(r["sequence"], k)
            self.sizes.append(len(km))
            for x in km:
                self.index[x].add(i)
        if not self.index:
            raise ValueError(
                "novelty reference is empty — refusing to build a gate that cannot fail"
            )

    def containment(self, seq: str) -> float:
        """max over reference records t of |kmers(s) & kmers(t)| / |kmers(s)|."""
        km = canonical_kmers(seq, self.k)
        if not km:
            raise ValueError(
                f"sequence of length {len(seq)} yields no valid {self.k}-mers, so "
                f"containment is undefined. The gate raises rather than returning 0.0, "
                f"which is the PASSING value (KNOWN_WRONG #3)."
            )
        hits: dict[int, int] = defaultdict(int)
        for x in km:
            for i in self.index.get(x, ()):
                hits[i] += 1
        if not hits:
            return 0.0
        return max(hits.values()) / len(km)

    def verdict(self, seq: str) -> dict:
        c = self.containment(seq)
        return {
            "containment": round(c, 4),
            "novel": c < FAIL_AT,
            "gate": "FAIL_memorized" if c >= FAIL_AT else ("WARN" if c >= WARN_AT
                                                           else "PASS"),
        }
