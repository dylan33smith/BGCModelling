"""Pin T3.2 / T3.3 / T6.1 — the Phase-3 novelty tests.

These exist because the previous battery could be passed by a model that had invented nothing, and
a test that cannot demonstrate its own dynamic range is not evidence. Each case below has a KNOWN
answer, so a regression that silently makes a test always-pass is caught here rather than in a
result.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from novelty_battery import containment, intra_set_diversity, joint_pass  # noqa: E402


def _rand(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_containment_dynamic_range():
    """A metric whose floor is legitimately zero can only be trusted with a positive control."""
    s = _rand(2000, 1)
    assert containment(s, s) == 1.0, "identical must be 1.0"
    assert containment(s, _rand(2000, 2)) < 0.01, "unrelated DNA shares ~no 21-mers"


def test_intra_set_detects_total_collapse():
    """The hole this closes: every other novelty check compares to TRAINING data, so a model
    emitting one sequence N times passes all of them."""
    d = intra_set_diversity([_rand(2000, 3)] * 10)
    assert d["n_distinct_clusters"] == 1
    assert d["median_pairwise_containment"] > 0.99
    assert d["frac_distinct"] == 0.1


def test_intra_set_passes_genuine_diversity():
    d = intra_set_diversity([_rand(2000, i) for i in range(10)])
    assert d["n_distinct_clusters"] == 10
    assert d["median_pairwise_containment"] < 0.01
    assert d["frac_with_a_near_duplicate"] == 0.0


def test_intra_set_partial_collapse():
    """Half duplicated: 5 copies collapse to 1 cluster, plus 5 distinct = 6."""
    seqs = [_rand(2000, 4)] * 5 + [_rand(2000, 10 + i) for i in range(5)]
    d = intra_set_diversity(seqs)
    assert d["n_distinct_clusters"] == 6
    assert abs(d["frac_with_a_near_duplicate"] - 0.5) < 1e-9


def test_joint_pass_catches_onclass_equals_memorised():
    """THE case marginal rates cannot see: the on-class records are exactly the non-novel ones.
    Marginals read '4 on-class, 6 novel' — both unremarkable — while NO single record is both."""
    on_class = [True] * 4 + [False] * 6
    nt = [0.99] * 4 + [0.01] * 6           # the on-class four are memorised
    aai = [0.5] * 10
    distinct = [True] * 10
    j = joint_pass(on_class, nt, aai, distinct)
    assert j["on_class"] == 4
    assert j["nt_novel"] == 6
    assert j["JOINT_PASS"] == 0, "intersection must be empty"


def test_joint_pass_counts_a_real_success():
    on_class = [True] * 3 + [False] * 7
    nt = [0.05] * 10
    aai = [0.4] * 10
    distinct = [True] * 10
    j = joint_pass(on_class, nt, aai, distinct)
    assert j["JOINT_PASS"] == 3 and abs(j["joint_rate"] - 0.3) < 1e-9


def test_joint_pass_protein_novelty_can_veto():
    """Nucleotide-novel but a protein paraphrase: T3.1 passes it, T3.2 must not."""
    on_class = [True] * 4
    nt = [0.02] * 4                        # novel DNA
    aai = [0.99] * 4                       # identical protein — synonymous codon swaps
    distinct = [True] * 4
    j = joint_pass(on_class, nt, aai, distinct)
    assert j["nt_novel"] == 4 and j["protein_novel"] == 0 and j["JOINT_PASS"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("ALL NOVELTY-BATTERY TESTS PASSED")
