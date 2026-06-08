#!/usr/bin/env python3
"""Tests for the memorization/novelty similarity core (no GPU, no data).

Run: python tests/test_memorization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import memorization_check as M  # noqa: E402

K, N = 5, 500  # small k + large sketch so bottom-k Jaccard is exact for these toys


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def exact_jaccard(a, b, k=K):
    sa, sb = M.kmer_set(a, k), M.kmer_set(b, k)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def test_identical():
    s = "ACGTACGTGGCCAATTACGGTACA"
    assert M.exact_containment(s, s, K) == 1.0
    sk = M.minhash_sketch(s, K, N)
    assert M.jaccard_bottomk(sk, sk, N) == 1.0
    print("PASS: identical -> containment 1.0, jaccard 1.0")


def test_strand_agnostic():
    s = "ACGTACGTGGCCAATTACGGTACA"
    rc = revcomp(s)
    # canonical k-mers make a sequence and its reverse complement identical
    assert M.kmer_set(s, K) == M.kmer_set(rc, K)
    assert M.exact_containment(s, rc, K) == 1.0
    print("PASS: canonical k-mers are strand-agnostic (seq == revcomp)")


def test_disjoint():
    a, b = "AAAAAAAAAAAA", "CCCCCCCCCCCC"
    assert M.exact_containment(a, b, K) == 0.0
    assert M.jaccard_bottomk(M.minhash_sketch(a, K, N), M.minhash_sketch(b, K, N), N) == 0.0
    print("PASS: disjoint sequences -> 0 containment / 0 jaccard")


def test_substring_containment():
    ref = "ACGTACGTGGCCAATTACGGTACATGACTGACTGACTGA"
    query = ref[10:30]  # a substring -> all its k-mers are in ref
    assert M.exact_containment(query, ref, K) == 1.0
    # reverse: ref is NOT fully contained in the short query
    assert M.exact_containment(ref, query, K) < 1.0
    print("PASS: substring is fully contained; the longer one is not")


def test_jaccard_matches_exact():
    # with N >> #k-mers, bottom-k Jaccard equals the exact set Jaccard
    a = "ACGTACGTGGCCAATTACGGTACATGACT"
    b = "GGCCAATTACGGTACATTTTGGGGCCCCA"  # shares a middle stretch with a
    est = M.jaccard_bottomk(M.minhash_sketch(a, K, N), M.minhash_sketch(b, K, N), N)
    exact = exact_jaccard(a, b)
    assert abs(est - exact) < 1e-9, f"{est} != {exact}"
    assert 0.0 < est < 1.0, "partial overlap should be strictly between 0 and 1"
    print(f"PASS: bottom-k Jaccard == exact Jaccard ({est:.3f}) for partial overlap")


def test_determinism_and_short():
    s = "ACGTACGTGGCCAATT"
    assert M.minhash_sketch(s, K, N) == M.minhash_sketch(s, K, N)  # deterministic
    assert M.minhash_sketch("ACG", K, N) == []                    # shorter than k
    assert M.exact_containment("AC", "ACGTACGT", K) == 0.0         # query < k
    print("PASS: sketch deterministic; sequences shorter than k handled")


def main():
    test_identical()
    test_strand_agnostic()
    test_disjoint()
    test_substring_containment()
    test_jaccard_matches_exact()
    test_determinism_and_short()
    print("\nALL MEMORIZATION TESTS PASSED")


if __name__ == "__main__":
    main()
