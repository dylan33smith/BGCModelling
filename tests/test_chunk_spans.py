#!/usr/bin/env python3
"""Boundary/coverage tests for build_nt_chunk_spans at the real L=32768.

Complements tests/test_chunk_eos_windows.py (which checks the dataset/prefix/EOS
layer with a mock tokenizer). Here we pin the pure nucleotide-span arithmetic:
full coverage, correct overlap/stride, exact window counts at real sizes, the
single-window fast path, the EOS budget reservation, and the error guards.

Run: python tests/test_chunk_spans.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evo2" / "scripts"))
import finetune_evo2_lora as F  # noqa: E402

L = 32768
PREFIX_CAP = 207        # representative real max prefix (bytes==tokens, byte-level)
OVERLAP = 2048
EOS = len(F.EOS_MARKER.encode())  # 5
BUDGET = L - PREFIX_CAP - EOS     # 32556
STRIDE = BUDGET - OVERLAP         # 30508


def spans(S, eos_reserve=EOS):
    return F.build_nt_chunk_spans(S, L, PREFIX_CAP, OVERLAP, 0, eos_reserve)


def check_invariants(S):
    sp = spans(S)
    assert sp[0][0] == 0, f"S={S}: first span must start at 0"
    assert sp[-1][1] == S, f"S={S}: last span must end at S (got {sp[-1][1]})"
    for a, b in sp:
        assert 0 <= a < b <= S, f"S={S}: bad span ({a},{b})"
        assert b - a <= BUDGET, f"S={S}: span width {b-a} exceeds budget {BUDGET}"
    for i in range(len(sp) - 1):
        # uniform stride and genuine overlap (no gaps)
        assert sp[i + 1][0] - sp[i][0] == STRIDE, f"S={S}: stride break at {i}"
        assert sp[i + 1][0] < sp[i][1], f"S={S}: gap/no-overlap at {i}"
    return sp


def main():
    # Budget reservation: prefix + max window + EOS must fit in L.
    assert PREFIX_CAP + BUDGET + EOS <= L, "EOS budget reservation is wrong"
    assert BUDGET == 32556 and STRIDE == 30508

    # Single-window fast path: S <= budget -> exactly [(0, S)].
    for S in (1, 1000, BUDGET - 1, BUDGET):
        assert spans(S) == [(0, S)], f"S={S} should be a single window (0,{S})"

    # Just over budget -> exactly 2 windows.
    assert spans(BUDGET + 1) == [(0, BUDGET), (STRIDE, BUDGET + 1)]

    # Exact counts at real sizes.
    expected_counts = {
        BUDGET: 1,
        BUDGET + 1: 2,
        100_000: 4,
        262_144: 9,    # the dataset's max sequence length -> 9 windows (matches maxwin)
    }
    for S, want in expected_counts.items():
        got = len(spans(S))
        assert got == want, f"S={S}: expected {want} windows, got {got}"

    # Coverage/overlap invariants across a sweep, incl. exact stride multiples.
    for S in [1, 5000, BUDGET, BUDGET + 1, 50_000, 100_000,
              STRIDE + BUDGET, 2 * STRIDE + BUDGET, 262_144, 261_999]:
        sp = check_invariants(S)
        # closed-form count cross-check
        want = 1 if S <= BUDGET else 1 + math.ceil((S - BUDGET) / STRIDE)
        assert len(sp) == want, f"S={S}: count {len(sp)} != closed-form {want}"
    print(f"PASS: coverage/overlap/stride invariants + exact counts "
          f"(budget={BUDGET}, stride={STRIDE}, max 262144 -> 9 windows)")

    # EOS reserve actually shrinks the budget (more/equal windows than without).
    big = 100_000
    assert len(spans(big, eos_reserve=0)) <= len(spans(big, eos_reserve=EOS))
    # And without reserve the budget is larger by exactly EOS.
    no_eos = F.build_nt_chunk_spans(BUDGET + EOS, L, PREFIX_CAP, OVERLAP, 0, 0)
    assert no_eos == [(0, BUDGET + EOS)], "without EOS reserve, budget is L-prefix"
    print("PASS: EOS reserve shrinks seq_budget by exactly the EOS token count")

    # Error guards.
    try:
        F.build_nt_chunk_spans(1000, L, PREFIX_CAP, BUDGET, 0, EOS)  # overlap >= budget
        raise AssertionError("expected ValueError for overlap >= seq_budget")
    except ValueError:
        pass
    try:
        F.build_nt_chunk_spans(1000, 100, 100, 10, 0, 5)  # seq_budget <= 0
        raise AssertionError("expected ValueError for non-positive seq_budget")
    except ValueError:
        pass
    print("PASS: error guards (overlap>=budget, budget<=0) raise ValueError")

    print("\nALL CHUNK-SPAN TESTS PASSED")


if __name__ == "__main__":
    main()
