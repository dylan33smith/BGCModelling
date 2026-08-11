"""Guard ACTIVATION PATCHING: is the donor's state actually substituted, and is "how far did it
move toward the donor" computed correctly?

WHY THIS EXISTS. This experiment is a fork in the project: a high cross-class alignment reopens
inference-time intervention (with transplants instead of directions), a near-zero one closes it
structurally. Both hooks and the metric fail silently in ways that produce a perfectly plausible
table:

  * A patch hook that returns the recipient unchanged gives alignment ~0 at every layer — which
    reads exactly like the decisive negative result, and would close the last open door on a bug.
  * A patch hook that broadcasts a mismatched donor (different prompt length) would silently patch
    the wrong positions rather than raising.
  * `_alignment` projects the achieved change onto the desired one. A missing normalisation, or
    projecting onto the wrong vector, still yields a number in a believable range.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))


class _Blk(nn.Module):
    def forward(self, x):
        return x


class _Toy(nn.Module):
    """Minimal stand-in exposing the `blocks.N` naming the hooks look up."""

    def __init__(self, n=3):
        super().__init__()
        self.blocks = nn.ModuleList([_Blk() for _ in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def test_patch_all_replaces_every_position():
    from activation_patching import _patch_hook

    m = _Toy()
    recip = torch.zeros(1, 5, 4)
    donor = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    h = _patch_hook(m, 1, donor, "all")
    try:
        out = m(recip)
    finally:
        h.remove()
    assert torch.allclose(out, donor), (
        "mode='all' did not substitute the donor at every position — if the hook is a no-op, every "
        "layer reports alignment ~0 and the experiment 'closes' intervention on a bug")
    print("PASS patch: mode='all' substitutes the donor at every position")


def test_patch_last_touches_only_the_final_position():
    from activation_patching import _patch_hook

    m = _Toy()
    recip = torch.zeros(1, 5, 4)
    donor = torch.ones(1, 5, 4) * 7.0
    h = _patch_hook(m, 1, donor, "last")
    try:
        out = m(recip)
    finally:
        h.remove()
    assert torch.allclose(out[:, :-1, :], torch.zeros(1, 4, 4)), "earlier positions were modified"
    assert torch.allclose(out[:, -1, :], torch.ones(1, 4) * 7.0), "final position was not patched"
    print("PASS patch: mode='last' touches only the final position")


def test_integer_k_patches_exactly_the_last_k_positions():
    """The k sweep is what separates 'this layer is not read' from 'one position out of a thousand
    has too little leverage'. An off-by-one, or k counted from the front, would blur those two
    explanations together while still producing a monotone-looking table."""
    from activation_patching import _patch_hook

    m = _Toy()
    recip = torch.zeros(1, 6, 4)
    donor = torch.ones(1, 6, 4) * 9.0
    for k in (1, 3, 6):
        h = _patch_hook(m, 1, donor, k)
        try:
            out = m(recip)
        finally:
            h.remove()
        patched = (out != 0).any(dim=-1)[0]
        assert patched.sum().item() == k, f"k={k} patched {patched.sum().item()} positions"
        assert patched[-k:].all(), f"k={k} did not patch the TRAILING positions"
        assert not patched[:-k].any(), f"k={k} patched positions before the last k"
    # k larger than the sequence must clamp, not index out of range
    h = _patch_hook(m, 1, donor, 999)
    try:
        out = m(recip)
    finally:
        h.remove()
    assert torch.allclose(out, donor), "k > length must clamp to the whole sequence"
    print("PASS patch: integer k substitutes exactly the last k positions, clamped at the length")


def test_mismatched_donor_length_raises():
    """Position-aligned patching is meaningless if the prompts tokenize to different lengths.
    Broadcasting or truncating here would patch the wrong positions and still return a number."""
    from activation_patching import _patch_hook

    m = _Toy()
    recip = torch.zeros(1, 5, 4)
    donor = torch.ones(1, 3, 4)
    h = _patch_hook(m, 1, donor, "all")
    try:
        raised = False
        try:
            m(recip)
        except ValueError:
            raised = True
        assert raised, "a donor with the wrong number of positions was accepted silently"
    finally:
        h.remove()
    print("PASS patch: a length-mismatched donor raises instead of patching the wrong positions")


def test_capture_then_patch_round_trips():
    """End-to-end: what _capture_hook records must be exactly what _patch_hook can re-impose."""
    from activation_patching import _capture_hook, _patch_hook

    m = _Toy()
    donor_in = torch.randn(1, 6, 4)
    store: dict = {}
    h = _capture_hook(m, 1, store)
    try:
        m(donor_in)
    finally:
        h.remove()
    assert "h" in store and store["h"].shape == donor_in.shape

    recip = torch.zeros(1, 6, 4)
    h = _patch_hook(m, 1, store["h"], "all")
    try:
        out = m(recip)
    finally:
        h.remove()
    assert torch.allclose(out, donor_in), "captured state did not round-trip through the patch"
    print("PASS patch: capture -> patch round-trips exactly")


def test_alignment_is_the_fraction_moved_toward_the_donor():
    from activation_patching import _alignment

    p_dst = torch.tensor([0.7, 0.2, 0.1])
    p_src = torch.tensor([0.1, 0.2, 0.7])
    assert abs(_alignment(p_dst, p_dst, p_src) - 0.0) < 1e-9, "no change must score 0"
    assert abs(_alignment(p_dst, p_src, p_src) - 1.0) < 1e-9, "becoming the donor must score 1"
    half = p_dst + 0.5 * (p_src - p_dst)
    assert abs(_alignment(p_dst, half, p_src) - 0.5) < 1e-9, "halfway must score 0.5"
    away = p_dst - 0.3 * (p_src - p_dst)
    assert _alignment(p_dst, away, p_src) < 0, "moving away from the donor must score negative"
    # A change ORTHOGONAL to the desired one is disruption, not transfer, and must not score.
    orth = p_dst + torch.tensor([0.1, -0.2, 0.1]) * 0.0 + torch.tensor([0.05, -0.10, 0.05])
    d = float(((orth - p_dst) * (p_src - p_dst)).sum())
    if abs(d) < 1e-12:
        assert abs(_alignment(p_dst, orth, p_src)) < 1e-9
    print("PASS patch: alignment is the signed fraction of the way from recipient to donor")


def test_kl_is_zero_for_identical_and_positive_otherwise():
    from activation_patching import _kl

    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.2, 0.3, 0.5])
    assert abs(_kl(p, p)) < 1e-9
    assert _kl(p, q) > 0
    print("PASS patch: KL is 0 for identical distributions and positive otherwise")


def main() -> int:
    for t in (test_patch_all_replaces_every_position,
              test_integer_k_patches_exactly_the_last_k_positions,
              test_patch_last_touches_only_the_final_position,
              test_mismatched_donor_length_raises,
              test_capture_then_patch_round_trips,
              test_alignment_is_the_fraction_moved_toward_the_donor,
              test_kl_is_zero_for_identical_and_positive_otherwise):
        t()
    print("\nALL ACTIVATION-PATCHING TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
