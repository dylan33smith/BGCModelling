#!/usr/bin/env python3
"""Tests for the memorization/novelty similarity core (no GPU, no data).

Includes the audit-C1/M11 regression: a query that is a FRAGMENT of a long
reference (with same-length distractors that out-share k-mers by Jaccard) must
still be caught as containment ~1.0 by scan_corpus.

Run: python tests/test_memorization.py
"""

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evo2" / "scripts"))
import memorization_check as M  # noqa: E402

K, N = 5, 500


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def rand_seq(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def exact_jaccard(a, b, k=K):
    sa, sb = M.kmer_set(a, k), M.kmer_set(b, k)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def test_identical():
    s = "ACGTACGTGGCCAATTACGGTACA"
    assert M.exact_containment(s, s, K) == 1.0
    sk = M.minhash_sketch(s, K, N)
    assert M.jaccard_bottomk(sk, sk, N) == 1.0
    assert M.containment_estimate(set(sk), M.kmer_hash_set(s, K)) == 1.0
    print("PASS: identical -> containment / jaccard / containment-estimate all 1.0")


def test_strand_agnostic():
    s = "ACGTACGTGGCCAATTACGGTACA"
    assert M.kmer_set(s, K) == M.kmer_set(revcomp(s), K)
    assert M.exact_containment(s, revcomp(s), K) == 1.0
    print("PASS: canonical k-mers are strand-agnostic")


def test_disjoint():
    a, b = "AAAAAAAAAAAA", "CCCCCCCCCCCC"
    assert M.exact_containment(a, b, K) == 0.0
    assert M.containment_estimate(set(M.minhash_sketch(a, K, N)), M.kmer_hash_set(b, K)) == 0.0
    print("PASS: disjoint -> 0 containment")


def test_substring_containment():
    ref = "ACGTACGTGGCCAATTACGGTACATGACTGACTGACTGA"
    query = ref[10:30]
    assert M.exact_containment(query, ref, K) == 1.0
    assert M.exact_containment(ref, query, K) < 1.0
    # containment estimate from the query's sketch vs ref's full hash set is ~1.0
    assert M.containment_estimate(set(M.minhash_sketch(query, K, N)), M.kmer_hash_set(ref, K)) == 1.0
    print("PASS: substring fully contained (exact + estimate)")


def test_containment_beats_jaccard_for_fragment():
    # A short query that is a slice of a long ref: containment ~1.0 but Jaccard tiny.
    long_ref = rand_seq(4000, 1)
    query = long_ref[1000:2000]
    assert M.exact_containment(query, long_ref, 15) >= 0.99   # fragment fully contained
    assert exact_jaccard(query, long_ref, 15) < 0.4           # but Jaccard is small
    print("PASS: fragment has high containment but low Jaccard (the C1 trap)")


def test_scan_catches_memorized_fragment():
    """Audit C1/M11 regression: distractors out-Jaccard the true source, but the
    containment-ranked scan must still report the fragment as ~1.0 memorized."""
    long_ref = rand_seq(4000, 7)
    query = long_ref[1000:2000]                       # exact 1000nt slice
    # 4 same-length distractors sharing the query's first half (higher Jaccard
    # with the query than the long source has) but NOT containing the full query.
    distractors = [query[:500] + rand_seq(500, 100 + i) for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        ref = Path(d) / "ref.jsonl"
        with ref.open("w") as f:
            for i, ds in enumerate(distractors):
                f.write(json.dumps({"accession": f"DISTRACT_{i}", "sequence": ds}) + "\n")
            f.write(json.dumps({"accession": "REF_LONG", "sequence": long_ref}) + "\n")
        # top_m=3 mirrors the OLD default: the fix is ranking by containment, not raising top_m
        res = M.scan_corpus([("q", query)], ref, k=15, sketch_n=200, top_m=3)
    r = res[0]
    assert r["max_containment"] >= 0.99, f"memorized fragment missed: {r}"
    assert r["nearest_accession"] == "REF_LONG", f"wrong nearest: {r}"
    print(f"PASS: scan catches memorized fragment (containment={r['max_containment']}, "
          f"nearest={r['nearest_accession']}) despite higher-Jaccard distractors")


def test_verdict_tiers():
    assert M.verdict_for(0.99, 0.95, 0.8) == "FAIL"
    assert M.verdict_for(0.85, 0.95, 0.8) == "WARN"
    assert M.verdict_for(0.5, 0.95, 0.8) == "PASS"
    print("PASS: verdict tiers (FAIL/WARN/PASS)")


def main():
    test_identical()
    test_strand_agnostic()
    test_disjoint()
    test_substring_containment()
    test_containment_beats_jaccard_for_fragment()
    test_scan_catches_memorized_fragment()
    test_verdict_tiers()
    print("\nALL MEMORIZATION TESTS PASSED")


if __name__ == "__main__":
    main()
