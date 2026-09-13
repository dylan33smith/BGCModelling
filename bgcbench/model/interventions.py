"""Activation-space interventions: shared infrastructure for W3 and I1 (SPEC 6).

⚠ WHY THIS IS NOT KV-PREFIX TUNING, measured on the architecture rather than assumed.

SPEC 6 originally specified `W3` as "learned key/value states supplied to every attention
layer". That is not implementable on Evo2:

  * peft's PrefixTuning injects through `past_key_values` and needs
    `prepare_inputs_for_generation`. Vortex's model has NEITHER -- its forward is
    `(x, inference_params_dict=None, padding_mask=None)`, with no HuggingFace KV interface.
  * The attention kernel is `FlashSelfAttention`, which takes a PACKED qkv of shape
    (b, s, 3, h, d). Packed self-attention requires q, k and v to share a sequence length,
    and a KV prefix by definition makes k/v LONGER than q. There is no hook point that
    prepends to k/v alone.
  * The inference path does split them (`_update_kvcache_attention(qkv[:,:,0], qkv[:,:,1:],
    inference_params)`) -- but only under `inference_params`. Training and generation would
    need two different prefix mechanisms that had to agree exactly, which is a correctness
    risk far larger than the arm is worth.

So `W3` is built as a LEARNED PER-LAYER RESIDUAL OFFSET: one trainable vector per attention
block, added to that block's output. It keeps the three properties the arm exists to test --
learned by gradient descent, applied at every attention layer, and NOT input-only -- and it
pairs exactly with `I1`, which adds a DERIVED direction at the same sites. W3 learns the
direction; I1 derives it from class means. Same mechanism, different origin, one code path.

⚠ Evo2-1B has only 4 attention blocks of 25 (3, 10, 17, 24), enumerated from the loaded
model. A standard transformer would expose all of its layers. SPEC 6.5 requires recording
total / attention / intervened counts per substrate, because an arm attached at 4 of 25
sites is not the same arm as one attached at 32 of 32.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


#: How each substrate family names its attention module. Enumerated from the loaded model,
#: never assumed -- Evo2 exposes `inner_mha_cls` on 4 of its 25 blocks, GenomeOcean exposes
#: `self_attn` on all 24 of its layers.
ATTENTION_SUFFIXES = ("inner_mha_cls", "self_attn")


def attention_sites(model, subset: list[int] | None = None) -> list[tuple[str, nn.Module]]:
    """The attention blocks, in model order. Enumerated, never assumed.

    ⚠ SUBSTRATES DIFFER IN HOW MANY THERE ARE, AND SPEC 6.5 REQUIRES THAT BE REPORTED
    RATHER THAN EQUALISED. Evo2-1B has 4 attention blocks among 25 (coverage 0.16);
    GenomeOcean-4B is a standard decoder with attention at all 24 layers (coverage 1.00).
    An arm attached at 4 of 25 sites is not the arm attached at 24 of 24.

    `subset` selects site INDICES after enumeration, which is how a cross-substrate arm can
    be matched on relative depth rather than on count -- see `matched_depth_subset`.
    """
    sites = [(n, m) for n, m in model.named_modules()
             if any(n.endswith(sfx) for sfx in ATTENTION_SUFFIXES)]
    if subset is None:
        return sites
    bad = [i for i in subset if i < 0 or i >= len(sites)]
    if bad:
        raise ValueError(f"site indices {bad} out of range for {len(sites)} attention sites")
    return [sites[i] for i in subset]


def matched_depth_subset(n_sites: int, reference_depths=(0.12, 0.40, 0.68, 0.96)) -> list[int]:
    """Site indices at the same RELATIVE depths Evo2 exposes.

    ⚠ WHY MATCH DEPTH RATHER THAN COUNT. Evo2's four attention blocks sit at 3, 10, 17 and
    24 of 25 -- fractional depths 0.12, 0.40, 0.68, 0.96. GenomeOcean has attention
    everywhere, so "the same arm" is ambiguous: inject at all 24 and the two substrates
    differ in how hard they are pushed as well as in what they are; inject at 4 matched
    positions and the comparison is about the substrate. This returns the latter, and the
    realised site list is recorded per arm either way (SPEC 6.5).
    """
    return sorted({min(n_sites - 1, max(0, round(d * n_sites))) for d in reference_depths})


def site_report(model) -> dict:
    sites = attention_sites(model)
    total = sum(1 for n, _ in model.named_modules() if n.count(".") == 1
                and n.startswith("blocks."))
    return {"attention_sites": [n for n, _ in sites],
            "n_attention_sites": len(sites),
            "n_blocks": total,
            "coverage": round(len(sites) / max(total, 1), 4)}


class Intervention(nn.Module):
    """Adds a per-site vector to each attention block's output.

    Subclasses differ only in where the vector comes from. Everything else -- the sites,
    the hook mechanics, the removal guarantee -- is shared, so W3 and I1 cannot diverge in
    how they are applied, only in what they apply.
    """

    def __init__(self, model, hidden_size: int, dtype=None):
        super().__init__()
        self.sites = attention_sites(model)
        if not self.sites:
            raise RuntimeError("no attention sites found; refusing to build an "
                               "intervention that would silently do nothing")
        self.hidden_size = hidden_size
        self._handles: list = []
        self._dtype = dtype

    def vector(self, i: int) -> torch.Tensor:            # noqa: D401
        raise NotImplementedError

    def scale(self) -> float:
        return 1.0

    def _make_hook(self, i: int):
        def hook(_mod, _args, output):
            v = self.vector(i)
            if isinstance(output, tuple):
                head, rest = output[0], output[1:]
                return (head + self.scale() * v.to(head.dtype),) + rest
            return output + self.scale() * v.to(output.dtype)
        return hook

    @contextmanager
    def attached(self):
        """Hooks are removed on exit even if the body raises. A leaked hook would silently
        contaminate every subsequent arm run in the same process."""
        try:
            for i, (_n, mod) in enumerate(self.sites):
                self._handles.append(mod.register_forward_hook(self._make_hook(i)))
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []

    def is_attached(self) -> bool:
        return bool(self._handles)


class LearnedOffset(Intervention):
    """W3. A trainable per-site conditioner, initialised so it is exactly the base model.

    Zero init means the UNTRAINED arm is bit-identical to the base model -- so any
    difference at step 0 is a bug, not an intervention, and the manipulation check has a
    clean baseline.

    ⚠ CAPACITY IS A SWEPT PARAMETER, NOT A HANDICAP. The bare offset is one vector per
    site: 4 x 1920 = 7,680 parameters, against LoRA's 10,475,520 on the same model -- a
    1,364x gap, and roughly what ONE virtual token of prefix tuning per layer would buy.
    A null from an arm that far below every other arm's capacity would say "too few
    parameters", not "activation-space conditioning does not work". `rank > 0` adds a
    per-site low-rank transform of the block output, so capacity can be matched to the
    other arms and swept by the same gate that sets LoRA's rank (G6).
    """

    def __init__(self, model, hidden_size: int, rank: int = 0, dtype=None):
        super().__init__(model, hidden_size, dtype)
        dt = dtype or torch.float32
        self.rank = int(rank)
        self.offsets = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, dtype=dt)) for _ in self.sites
        ])
        if self.rank > 0:
            # B initialised at zero so the whole term starts as the identity, preserving
            # the "untrained == base model" property above.
            self.A = nn.ParameterList([
                nn.Parameter(torch.randn(hidden_size, self.rank, dtype=dt) * 0.02)
                for _ in self.sites])
            self.B = nn.ParameterList([
                nn.Parameter(torch.zeros(self.rank, hidden_size, dtype=dt))
                for _ in self.sites])
        else:
            self.A = self.B = None

    def vector(self, i: int) -> torch.Tensor:
        return self.offsets[i]

    def _make_hook(self, i: int):
        base_hook = super()._make_hook(i)

        def hook(mod, args, output):
            out = base_hook(mod, args, output)
            if self.rank <= 0:
                return out
            head = out[0] if isinstance(out, tuple) else out
            delta = (head.to(self.A[i].dtype) @ self.A[i]) @ self.B[i]
            head = head + delta.to(head.dtype)
            return (head,) + out[1:] if isinstance(out, tuple) else head
        return hook

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DirectionInjection(Intervention):
    """I1. A FIXED direction per site, scaled by alpha. Not trained.

    SPEC 6: the direction is a DIFFERENCE OF MEANS over the arm's TRAIN activations, never
    probe weights -- so no probe is fitted and SPEC 3.7's ban on learned-probe endpoints is
    not in tension. `alpha` and the site set are swept by gate G9 on generation quality,
    never on the benchmark endpoint (SPEC 2.4).
    """

    def __init__(self, model, hidden_size: int, directions: torch.Tensor,
                 alpha: float, dtype=None):
        super().__init__(model, hidden_size, dtype)
        if directions.shape[0] != len(self.sites):
            raise ValueError(
                f"{directions.shape[0]} directions for {len(self.sites)} attention sites; "
                f"a silent mismatch would steer some layers and not others."
            )
        self.register_buffer("directions", directions)
        self.alpha = float(alpha)

    def vector(self, i: int) -> torch.Tensor:
        return self.directions[i]

    def scale(self) -> float:
        return self.alpha


def random_direction_control(model, hidden_size: int, seed: int, alpha: float,
                             dtype=None) -> DirectionInjection:
    """SPEC 6.3's magnitude-matched random-direction control for I1. Same norm, no class
    content -- so anything I1 achieves that this does not is the direction doing work."""
    g = torch.Generator().manual_seed(seed)
    n = len(attention_sites(model))
    d = torch.randn(n, hidden_size, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    return DirectionInjection(model, hidden_size, d, alpha, dtype)
