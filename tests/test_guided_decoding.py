"""Guard DISCRIMINATOR-GUIDED DECODING: does it actually select what it claims to select?

WHY THIS EXISTS. The guided arm and its control differ by exactly one line -- which candidate
index gets kept. If that mapping is wrong, the "guided" arm silently becomes a random arm while
still being labelled guided, and the experiment returns a null that reads as a real negative
result. This project has already retracted one finding and weakened two others for want of a
check like this, so the selection rule is unit-tested before any GPU time is spent on it.

The specific trap: `probs` has one row per USABLE candidate (empty candidates are dropped), so
argmax over it gives a LOCAL index that must be mapped back through `usable` to the original
candidate list. An off-by-one there is invisible in the output -- every field still populates,
the run still completes, the sequences still look fine.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))


def test_best_picks_the_highest_scoring_candidate():
    from guided_generate import select_candidate
    # 4 candidates, all usable; class 1 is the target. Candidate 2 is best for class 1.
    probs = np.array([
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.1, 0.8, 0.1],   # <- highest P(class 1)
        [0.4, 0.4, 0.2],
    ])
    pick, obj, local, tgt = select_candidate(probs, 1, [0, 1, 2, 3], "logp", "best",
                                             random.Random(0))
    assert pick == 2, f"picked {pick}, expected 2"
    assert local == 2
    assert abs(float(tgt[local]) - 0.8) < 1e-9
    print("PASS guided: 'best' picks the argmax of P(target)")


def test_usable_index_mapping_survives_dropped_candidates():
    """THE REGRESSION TEST. When empty candidates are dropped, the local argmax index is NOT the
    index into the original candidate list."""
    from guided_generate import select_candidate
    # Original candidates 0..4; 0 and 2 were empty, so usable = [1, 3, 4].
    usable = [1, 3, 4]
    probs = np.array([
        [0.6, 0.2, 0.2],   # -> original candidate 1
        [0.1, 0.9, 0.0],   # -> original candidate 3   <- best for class 1
        [0.3, 0.3, 0.4],   # -> original candidate 4
    ])
    pick, obj, local, tgt = select_candidate(probs, 1, usable, "logp", "best", random.Random(0))
    assert local == 1, f"local argmax {local}, expected 1"
    assert pick == 3, (f"picked ORIGINAL candidate {pick}, expected 3 -- the local index was not "
                       f"mapped back through `usable`, so the guided arm would keep the wrong "
                       f"sequence while still reporting the best score")
    print("PASS guided: local index correctly mapped back through `usable`")


def test_random_rule_ignores_the_scores():
    """The control must not be influenced by the discriminator at all -- otherwise it is not a
    control, it is a weaker version of the treatment."""
    from guided_generate import select_candidate
    strong = np.array([[0.01, 0.99], [0.99, 0.01], [0.5, 0.5]])
    flat = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    picks_strong = [select_candidate(strong, 0, [0, 1, 2], "logp", "random", random.Random(s))[0]
                    for s in range(40)]
    picks_flat = [select_candidate(flat, 0, [0, 1, 2], "logp", "random", random.Random(s))[0]
                  for s in range(40)]
    assert picks_strong == picks_flat, ("the random rule changed when the scores changed -- it is "
                                        "reading the discriminator and is not a valid control")
    assert len(set(picks_strong)) > 1, "the random rule always picks the same index"
    print("PASS guided: 'random' control is independent of the discriminator's scores")


def test_margin_objective_penalises_ambiguity():
    """logp and margin must genuinely differ: a candidate can have the highest P(target) while
    being more ambiguous than a rival. If they never diverge, one of them is not implemented."""
    from guided_generate import select_candidate
    probs = np.array([
        [0.45, 0.44, 0.11],   # highest P(class 0) = 0.45, but margin only +0.01
        [0.40, 0.05, 0.55],   # P(class 0) = 0.40, margin -0.15
        [0.42, 0.10, 0.48],
    ])
    pick_logp, _, _, _ = select_candidate(probs, 0, [0, 1, 2], "logp", "best", random.Random(0))
    assert pick_logp == 0
    probs2 = np.array([
        [0.45, 0.44, 0.11],   # margin +0.01
        [0.44, 0.05, 0.51],   # margin -0.07
        [0.43, 0.02, 0.55],
    ])
    _, obj_m, _, _ = select_candidate(probs2, 0, [0, 1, 2], "margin", "best", random.Random(0))
    assert abs(obj_m[0] - 0.01) < 1e-9, obj_m
    assert obj_m[1] < 0 and obj_m[2] < 0, obj_m
    print("PASS guided: margin objective computes target-minus-best-rival, distinct from logp")


def test_objective_is_monotone_in_target_probability():
    """Sanity: whatever the objective, a candidate the discriminator likes more for the TARGET
    must not score lower. A sign error here would guide away from the target -- and would look
    exactly like 'guidance does not work'."""
    from guided_generate import select_candidate
    for objective in ("logp", "margin"):
        probs = np.array([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2]])
        _, obj, _, tgt = select_candidate(probs, 0, [0, 1, 2], objective, "best", random.Random(0))
        order_by_obj = list(np.argsort(obj))
        order_by_tgt = list(np.argsort(tgt))
        assert order_by_obj == order_by_tgt, (objective, obj, tgt)
    print("PASS guided: objective increases with P(target) — guidance points toward, not away")


def main() -> int:
    for t in (test_best_picks_the_highest_scoring_candidate,
              test_usable_index_mapping_survives_dropped_candidates,
              test_random_rule_ignores_the_scores,
              test_margin_objective_penalises_ambiguity,
              test_objective_is_monotone_in_target_probability):
        t()
    print("\nALL GUIDED-DECODING TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
