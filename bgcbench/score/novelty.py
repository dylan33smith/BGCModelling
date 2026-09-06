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

#: A reference record shorter than this contributes no usable k-mer set. It also removes
#: the 1-nt records that a fuzzy-coordinate parse can emit.
MIN_REF_KMERS = 50
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

    def _hits(self, seq: str) -> tuple[dict[int, int], int]:
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
        return hits, len(km)

    def containment(self, seq: str) -> float:
        """FORWARD: max over reference records t of |kmers(s) & kmers(t)| / |kmers(s)|.

        ⚠ THIS ALONE CANNOT FAIL AT THE GENERATION BUDGET. The denominator is the WHOLE
        generation, so containment falls as 1/length no matter how much was copied.
        Measured: ten whole verbatim TERPENE training records concatenated to 15,992 nt
        score forward 0.198 -- PASS -- while antiSMASH calls them on-target with 15 core
        genes. Use `verdict()`, which gates on both directions.
        """
        hits, n = self._hits(seq)
        return max(hits.values()) / n if hits else 0.0

    def reverse_containment(self, seq: str) -> float:
        """REVERSE: max over reference records t of |kmers(s) & kmers(t)| / |kmers(t)|.

        "How much of some training record does this generation contain?" -- which does not
        dilute with generation length, so a collage of copied records is caught. On the
        same memorised concatenation this reads 1.000 against 0.000-0.006 for clean
        held-out and random sequence.
        """
        hits, _ = self._hits(seq)
        best = 0.0
        for i, c in hits.items():
            if self.sizes[i] >= MIN_REF_KMERS:
                best = max(best, c / self.sizes[i])
        return best

    def verdict(self, seq: str) -> dict:
        fwd = self.containment(seq)
        rev = self.reverse_containment(seq)
        worst = max(fwd, rev)
        return {
            "containment": round(fwd, 4),
            "containment_reverse": round(rev, 4),
            "containment_worst": round(worst, 4),
            "novel": worst < FAIL_AT,
            "gate": "FAIL_memorized" if worst >= FAIL_AT else ("WARN" if worst >= WARN_AT
                                                              else "PASS"),
        }
