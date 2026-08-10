"""Guard the STEERING HOOKS: what gets injected, where, and how big it is.

WHY THIS EXISTS. Every steering result in this project is a claim about an edit made inside a
forward pass, and three separate defects in that edit have already invalidated whole runs:
  * the shipped direction was the length axis, so "steer toward PKS" and "steer toward NRPS"
    were the same intervention with opposite sign;
  * `_ref_norm` read the mean-POOLED activation, making every dose 1.5-5.9x the entire
    between-sample scatter;
  * a global `beta` applied a 17x-different physical push per class, so the coherence ceiling
    it measured belonged to no single intervention.
None of those were type errors or crashes. They were correct-looking code applying the wrong
magnitude, and only an explicit numeric check on the applied delta would have caught them.

These tests run the real hooks on a stand-in module (no Evo2 weights, no GPU) and assert the
NUMBERS: which positions are touched, the exact ||delta||, and that the recorded provenance
matches what was applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

D = 32


class _Block(torch.nn.Module):
    """Stands in for a StripedHyena block: returns (hidden, extra), as Evo2's blocks do."""

    def forward(self, h):
        return h, None


class _Model(torch.nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(n)])

    def forward(self, h, layer):
        for i, b in enumerate(self.blocks):
            out = b(h)
            h = out[0]
        return h


def _unit(seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(D, generator=g)
    return v / v.norm()


def _run(model, h):
    """One pass through every block, mimicking how the hook sees a residual stream."""
    for b in model.blocks:
        h = b(h)[0]
    return h


def test_prefill_is_never_steered():
    """THE PHASE-3 INVARIANT. The seed exemplar carries the class signal being overridden;
    perturbing it as the model READS it measures a different experiment entirely."""
    from seed_generate import _install_generated_only_steer_hook as install
    m = _Model()
    u = _unit()
    hnd = install(m, 3, u, abs_norm=5.0)
    try:
        prefill = torch.ones(1, 7, D)                    # seq len > 1 => the prompt
        out = _run(m, prefill.clone())
        assert torch.equal(out, prefill), "prefill was modified — the seed is being steered"
        gen = torch.ones(1, 1, D)                        # seq len == 1 => a generated token
        out = _run(m, gen.clone())
        assert not torch.equal(out, gen), "generated position was NOT steered"
    finally:
        hnd.remove()
    print("PASS hook: prefill untouched, generated position steered")


def test_absolute_dose_applies_exactly_that_norm():
    from seed_generate import _install_generated_only_steer_hook as install
    m = _Model()
    u = _unit(1)
    hnd = install(m, 0, u, abs_norm=3.25)
    try:
        h = torch.zeros(1, 1, D)
        out = _run(m, h)
        assert abs(float(out.norm()) - 3.25) < 1e-4, float(out.norm())
        # direction preserved, not just magnitude
        assert float(torch.dot(out.reshape(-1) / out.norm(), u)) > 0.9999
    finally:
        hnd.remove()
    print("PASS hook: absolute dose applies exactly ||delta|| = abs_norm, along v")


def test_norm_relative_dose_tracks_the_local_residual():
    """THE POINT OF THE MODE. Mean ||h|| is 8.95 at L16 but 3.69e12 at L30, so a dose fixed in
    absolute units is not a comparable dose across layers. This asserts the delta really does
    scale with the residual it is added to, per position and per batch row."""
    from seed_generate import _install_generated_only_steer_hook as install
    m = _Model()
    u = _unit(2)
    frac = 0.16
    for base in (1.0, 1e3, 1e9):                 # spans the residual blow-up in the last blocks
        hnd = install(m, 2, u, norm_frac=frac)
        try:
            h = torch.zeros(1, 1, D)
            h[0, 0, 0] = base                    # ||h|| == base, and h is orthogonal to nothing
            hn = float(h.norm())
            out = _run(m, h.clone())
            delta = out - h
            assert abs(float(delta.norm()) / hn - frac) < 1e-4, (base, float(delta.norm()) / hn)
        finally:
            hnd.remove()
    print("PASS hook: ||delta|| = frac * ||h|| across 9 orders of magnitude of residual norm")


def test_recorded_stats_match_what_was_applied():
    """The beta titration had to re-derive its own doses from stderr after the fact. The record
    must carry the REALIZED magnitudes, and they must be right."""
    from seed_generate import _install_generated_only_steer_hook as install
    m = _Model()
    u = _unit(3)
    st = {"n": 0, "h_sum": torch.zeros(()), "d_sum": torch.zeros(())}
    hnd = install(m, 1, u, norm_frac=0.25, stats=st)
    try:
        for base in (2.0, 4.0):
            h = torch.zeros(1, 1, D)
            h[0, 0, 0] = base
            _run(m, h)
    finally:
        hnd.remove()
    assert st["n"] == 2, st["n"]
    assert abs(float(st["h_sum"]) / st["n"] - 3.0) < 1e-4        # mean of 2 and 4
    assert abs(float(st["d_sum"]) / st["n"] - 0.75) < 1e-4       # 0.25 * mean ||h||
    assert abs(float(st["d_sum"]) / float(st["h_sum"]) - 0.25) < 1e-6
    print("PASS hook: realized ||h|| / ||delta|| recorded correctly")


def test_ambiguous_or_absent_dose_is_refused():
    from seed_generate import _install_generated_only_steer_hook as install
    m = _Model()
    u = _unit(4)
    for kwargs in ({}, {"abs_norm": 1.0, "norm_frac": 0.1}):
        try:
            install(m, 0, u, **kwargs)
            raise AssertionError(f"{kwargs} must raise — an ambiguous dose is unmeasurable")
        except ValueError:
            pass
    print("PASS hook: refuses no dose and refuses two doses")


def test_stacked_hooks_all_fire_and_share_one_stats_sink():
    """MULTI-LAYER steering installs one hook per layer with a SHARED stats dict.

    Two ways that silently goes wrong: only the last hook survives (each install overwriting the
    previous), or the shared sink counts one layer's applications as if they were the whole
    stack. Either would make a stacked arm quietly identical to a single-layer one -- an arm that
    looks like it ran and tests nothing.
    """
    from seed_generate import _install_generated_only_steer_hook as install
    m = _Model(n=6)
    layers = [1, 3, 5]
    st = {"n": 0, "h_sum": torch.zeros(()), "d_sum": torch.zeros(())}
    hs = [install(m, L, _unit(10 + L), abs_norm=1.0, stats=st) for L in layers]
    try:
        out = _run(m, torch.zeros(1, 1, D))
    finally:
        for h in hs:
            h.remove()
    assert st["n"] == len(layers), f"{st['n']} applications for {len(layers)} hooks"
    # three DIFFERENT unit vectors of norm 1 each: the sum cannot be a single one of them
    assert 0.5 < float(out.norm()) < 3.0, float(out.norm())
    assert float(st["d_sum"]) == len(layers) * 1.0
    # and each layer really used its OWN direction
    single = _Model(n=6)
    h1 = install(single, 1, _unit(11), abs_norm=1.0)
    try:
        one = _run(single, torch.zeros(1, 1, D))
    finally:
        h1.remove()
    assert not torch.allclose(out, one), "the stack collapsed to a single layer's edit"
    print("PASS stacked hooks: every layer fires, each with its own direction, one shared sink")


def test_causal_test_hooks_gate_on_start_pos():
    """steer_causal_tests.py scores a continuation under an intervention applied ONLY to the
    scored positions. If the gate leaks into the context, the measurement is of a different
    conditional distribution than the one reported."""
    from steer_causal_tests import _add_hook, _project_out_hook
    m = _Model()
    u = _unit(5)
    n_ctx = 4
    hnd = _add_hook(m, 0, u * 2.0, start_pos=n_ctx)
    try:
        h = torch.zeros(1, 9, D)
        out = _run(m, h.clone())
        assert torch.equal(out[:, :n_ctx], h[:, :n_ctx]), "context positions were steered"
        assert float(out[:, n_ctx:].norm(dim=-1).min()) > 1.9, "scored positions were NOT steered"
    finally:
        hnd.remove()

    hnd = _project_out_hook(m, 0, u, start_pos=n_ctx)
    try:
        h = u.repeat(1, 9, 1).clone() * 3.0          # entirely along u
        out = _run(m, h.clone())
        assert torch.allclose(out[:, :n_ctx], h[:, :n_ctx]), "context was ablated"
        assert float(out[:, n_ctx:].abs().max()) < 1e-5, "u was not removed from scored positions"
    finally:
        hnd.remove()
    print("PASS causal hooks: add and ablate both respect start_pos")


def test_soft_prefix_replaces_only_the_prefix_positions():
    """SOFT PREFIX: overwrite the first P embeddings, touch nothing else, and never fire
    during incremental decoding.

    Two silent failure modes this pins. (1) If the hook also rewrote later positions it would
    be corrupting the taxonomy tag and the nucleotides -- the run would train, converge on
    something, and mean nothing. (2) If it fired on single-token steps it would re-stamp the
    prefix onto every GENERATED position, which is a completely different intervention from
    a prompt prefix and would make generation incomparable to training.
    """
    import torch as T
    from train_soft_prefix import install_soft_prefix

    class Emb(T.nn.Module):
        def forward(self, ids):
            return T.zeros(ids.shape[0], ids.shape[1], D)

    class M(T.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_layer = Emb()

    m, P = M(), 4
    pref = T.nn.Parameter(T.arange(P * D, dtype=T.float32).reshape(P, D))
    h = install_soft_prefix(m, pref, P)
    try:
        out = m.embedding_layer(T.zeros(1, 9, dtype=T.long))
        assert T.allclose(out[0, :P], pref), "prefix positions not written"
        assert float(out[0, P:].abs().max()) == 0.0, "hook modified NON-prefix positions"
        # incremental decoding: one token at a time must be left alone
        step = m.embedding_layer(T.zeros(1, 1, dtype=T.long))
        assert float(step.abs().max()) == 0.0, "hook fired during single-token decoding"
        # batch broadcast
        b = m.embedding_layer(T.zeros(3, 9, dtype=T.long))
        assert T.allclose(b[2, :P], pref), "prefix not broadcast across the batch"
    finally:
        h.remove()
    print("PASS soft prefix: writes exactly the first P positions, silent during decoding")


def main() -> int:
    for t in (test_prefill_is_never_steered,
              test_absolute_dose_applies_exactly_that_norm,
              test_norm_relative_dose_tracks_the_local_residual,
              test_recorded_stats_match_what_was_applied,
              test_ambiguous_or_absent_dose_is_refused,
              test_stacked_hooks_all_fire_and_share_one_stats_sink,
              test_causal_test_hooks_gate_on_start_pos,
              test_soft_prefix_replaces_only_the_prefix_positions):
        t()
    print("\nALL STEERING HOOK TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
