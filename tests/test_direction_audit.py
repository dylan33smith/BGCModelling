"""Guard the DIRECTION AUDIT's arithmetic — the part that decides whether steering reopens.

WHY THIS EXISTS. `direction_audit.py` answers one question: did the steering edit move the class
readout at the doses we actually used? Its verdict flips a decision already written into
progress.md ("do NOT run another steering variant"). Two pieces of arithmetic carry that verdict
and both fail silently:

  * THE DOSE. The edit is `alpha * class_unit * u`. Drop `class_unit` and every dose on the x-axis
    is wrong by a per-class factor of ~0.3-0.7, so "landed at 2 class-units" might really mean
    "landed at 6" -- above what we ever gave the model, inverting the conclusion. The curve still
    looks completely normal.
  * THE ANGLE. The probe sees standardised features; the edit is applied to raw activations.
    Comparing the coefficient vector to the direction without dividing by the scaler's sigma
    compares two different spaces. The cosine still comes out in [-1, 1] and still prints.

Neither error raises, and both produce a plausible table. Hence assertions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))


class _CapturingPipe:
    """Records exactly what activations it was handed, and reports a fixed distribution."""

    def __init__(self, n_classes=3):
        self.seen = []
        self.n_classes = n_classes

    def predict_proba(self, X):
        self.seen.append(np.array(X, copy=True))
        p = np.full((len(X), self.n_classes), 1.0 / self.n_classes)
        return p


def test_dose_is_alpha_times_class_unit_times_direction():
    """THE REGRESSION TEST. A missing class_unit rescales the whole x-axis and moves the verdict."""
    from direction_audit import _dose_curve

    H = np.zeros((5, 8))
    u = np.zeros(8)
    u[0] = 1.0
    unit = 0.37                      # a realistic class-unit (they run ~0.27-0.67)
    doses = [0.0, 1.0, 2.8, 11.4]
    pipe = _CapturingPipe()
    _dose_curve(pipe, H, u, unit, doses, 0)

    assert len(pipe.seen) == len(doses)
    for a, seen in zip(doses, pipe.seen):
        got = seen[0, 0]
        assert abs(got - a * unit) < 1e-12, (
            f"dose {a} class-units produced a shift of {got}, expected {a * unit}. The class_unit "
            f"factor is missing or misapplied, so every dose on the x-axis is wrong and the "
            f"'landed / did not land' verdict is read at the wrong place.")
        assert np.allclose(seen[:, 1:], 0.0), "the edit leaked into coordinates off the direction"
    print("PASS audit: dose = alpha * class_unit * direction, applied along the direction only")


def test_raw_space_direction_divides_by_the_scaler_sigma():
    """The probe's coefficients live in standardised space; the edit lives in raw space."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from direction_audit import _raw_space_logit_direction

    rng = np.random.default_rng(0)
    # Feature 1 has ~10x the spread of feature 0, so scaling is not a no-op and a missing
    # division shows up as a large, silent distortion of the angle.
    X = rng.normal(size=(300, 3)) * np.array([1.0, 10.0, 3.0])
    y = np.array(["A", "B", "C"] * 100)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X, y)

    got = _raw_space_logit_direction(pipe, 1)
    lr = pipe.named_steps["logisticregression"]
    scale = pipe.named_steps["standardscaler"].scale_
    want = (lr.coef_[1] - lr.coef_.mean(axis=0)) / scale
    assert np.allclose(got, want), "raw-space direction is not coef-contrast / sigma"
    # And it must actually differ from the un-divided version, or the test proves nothing.
    naive = lr.coef_[1] - lr.coef_.mean(axis=0)
    cos = float(np.dot(got, naive) / (np.linalg.norm(got) * np.linalg.norm(naive)))
    assert cos < 0.999, (f"scaled and raw directions are nearly identical (cos={cos:.4f}) — this "
                         f"fixture cannot detect a missing sigma division")
    print(f"PASS audit: raw-space direction divides by sigma (differs from naive, cos={cos:.3f})")


def test_contrast_is_against_the_other_classes_not_zero():
    """The steering direction is mu_c - mean(mu_others), so the probe side must use the SAME
    contrast or the angle compares a difference-of-means against a one-vs-rest logit.

    A NOTE ON WHY THIS USES A STUB. sklearn's multinomial solver already returns coefficients
    centred across classes (measured: max |coef_.mean(axis=0)| = 9e-16), so with a real fitted
    model the subtraction is a no-op and an assertion against a fitted pipeline would pass
    whether or not the code performs it. The contract is therefore exercised directly, with
    deliberately UN-centred coefficients — which is also the case that would arise if the
    estimator were ever switched to a one-vs-rest or binary fit."""
    from direction_audit import _raw_space_logit_direction

    class _Stub:
        def __init__(self):
            self.coef_ = np.array([[3.0, 0.0], [1.0, 2.0], [2.0, 1.0]])   # mean = (2, 1) != 0
            self.scale_ = np.array([1.0, 4.0])
            self.named_steps = {"standardscaler": self, "logisticregression": self}

    stub = _Stub()
    got = _raw_space_logit_direction(stub, 0)
    want = (stub.coef_[0] - stub.coef_.mean(axis=0)) / stub.scale_     # (1.0, -0.25)
    assert np.allclose(got, want), f"got {got}, expected {want}"
    assert not np.allclose(got, stub.coef_[0] / stub.scale_), (
        "the mean-of-other-classes contrast was not subtracted — the angle would compare "
        "mu_c - mean(mu_others) against a raw one-vs-rest coefficient")
    print("PASS audit: probe direction uses the same vs-other-classes contrast as the steering dir")


def test_crossing_reports_the_first_dose_at_or_above_threshold():
    from direction_audit import _crossing

    curve = [{"dose": 0.0, "flip_rate": 0.01}, {"dose": 1.0, "flip_rate": 0.38},
             {"dose": 2.0, "flip_rate": 0.87}, {"dose": 4.0, "flip_rate": 1.0}]
    assert _crossing(curve, "flip_rate", 0.5) == 2.0, "did not return the FIRST crossing dose"
    assert _crossing(curve, "flip_rate", 0.38) == 1.0, "threshold must be inclusive (>=)"
    assert _crossing(curve, "flip_rate", 1.01) is None, (
        "a curve that never reaches the threshold must return None, not the last dose — "
        "otherwise 'never landed' would be reported as 'landed at the top of the scan'")
    print("PASS audit: crossing dose is the first at-or-above threshold, None if never reached")


def main() -> int:
    for t in (test_dose_is_alpha_times_class_unit_times_direction,
              test_raw_space_direction_divides_by_the_scaler_sigma,
              test_contrast_is_against_the_other_classes_not_zero,
              test_crossing_reports_the_first_dose_at_or_above_threshold):
        t()
    print("\nALL DIRECTION-AUDIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
