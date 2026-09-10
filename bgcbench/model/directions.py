"""I1 direction derivation: class means over TRAIN activations (SPEC 6, gate G9).

⚠ WHY A DIFFERENCE OF MEANS AND NOT A PROBE. SPEC 3.7 bans a learned probe as an ENDPOINT.
A probe fitted to separate classes and then used to steer would also make the arm's own
direction a trained object with its own capacity, so a null could always be answered with
"the probe was too weak". A difference of means has no fitted parameters at all: it is a
statistic of the activations, so `I1` tests activation-space conditioning rather than the
quality of a classifier. SPEC 6 fixes this, and it is the reason I1 pairs with `W3` --
W3 LEARNS the direction by gradient descent, I1 DERIVES it. One mechanism, two origins.

⚠ DIRECTIONS ARE UNIT-NORM PER SITE, and this is load-bearing rather than cosmetic.
SPEC 6.3 requires a **magnitude-matched** random-direction control, and
`interventions.random_direction_control` normalises its vectors to unit norm. If I1's
directions kept their raw norms the two arms would differ in magnitude as well as in
content, so "I1 beat random" could just mean "I1 pushed harder". Normalising both puts the
entire magnitude on `alpha`, which is the swept parameter. The raw norms are recorded in
the artifact so the scale that was discarded is still reportable.

⚠ THE CONTRAST IS TARGET-CLASS vs THE OTHER BENCHMARK CLASSES, not target vs zero. A mean
activation is dominated by whatever the model does on ANY nucleotide sequence; subtracting
the other classes' mean removes that shared component and leaves what is specific to the
class. Target-minus-zero would steer toward "DNA", which every arm already produces.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch

from bgcbench.model.interventions import attention_sites, site_report


class _MeanCollector:
    """Accumulates a mean of each attention site's output, weighted PER RECORD.

    Streaming rather than storing: one record at 8,192 positions x 1920 hidden x 4 sites is
    ~250 MB in fp32, and the derivation runs over hundreds of records.

    ⚠ PER RECORD, NOT PER TOKEN, and the distinction is load-bearing. Pooling every position
    from every record into one mean weights a long record more heavily than a short one.
    The benchmark classes differ systematically in length (FINDINGS 12: RIPP's interquartile
    range is 1,449-6,229 nt against TERPENE's 999-3,844), so a token-weighted
    target-minus-others contrast would partly encode LENGTH rather than class content --
    and the steering arm would be pushing on the wrong axis.

    It also matches the dataset. SPEC 4.4.3 fixes equal effective_n by RECORD, and SPEC 6
    records that equal records is NOT equal tokens (a measured 3.4x nucleotide imbalance).
    A token-weighted direction would silently be the raw-mixture statistic that `W1` is and
    `W1n` exists to correct.

    Each record contributes its own within-record mean, and those are averaged with equal
    weight. `token_counts` keeps the per-token denominator so the discarded weighting is
    still reportable.
    """

    def __init__(self, model):
        self.sites = attention_sites(model)
        if not self.sites:
            raise RuntimeError("no attention sites found; refusing to derive directions "
                               "from a model with nothing to hook")
        self.n = len(self.sites)
        # per-record accumulator: sum of within-record means, and how many records
        self.sums: list[torch.Tensor | None] = [None] * self.n
        self.counts = [0] * self.n
        # the token-weighted denominator, kept only so the discarded weighting is reportable
        self.token_counts = [0] * self.n
        self._handles: list = []

    def _hook(self, i: int):
        def hook(_mod, _args, output):
            head = output[0] if isinstance(output, tuple) else output
            # (batch, seq, hidden) -> the WITHIN-RECORD mean over positions, in fp32 so a
            # long accumulation does not lose precision in bf16.
            flat = head.detach().to(torch.float32).reshape(-1, head.shape[-1])
            m = flat.mean(dim=0)
            self.sums[i] = m if self.sums[i] is None else self.sums[i] + m
            self.counts[i] += 1
            self.token_counts[i] += flat.shape[0]
        return hook

    @contextmanager
    def attached(self):
        try:
            for i, (_n, mod) in enumerate(self.sites):
                self._handles.append(mod.register_forward_hook(self._hook(i)))
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []

    def means(self) -> torch.Tensor:
        if any(c == 0 for c in self.counts):
            raise RuntimeError(f"site record counts {self.counts} include a zero; a site "
                               f"that never fired would give a meaningless mean")
        return torch.stack([self.sums[i] / self.counts[i] for i in range(self.n)])


def _encode(sub, rec: dict, prefix_kind: str, max_len_nt: int) -> list[int]:
    """Identical text construction to training (`train._encode2`), so the activations the
    direction is derived from are the ones the arm actually produces."""
    prefix = rec.get("tax_tag", "") if prefix_kind == "taxonomy" else ""
    text = sub.training_text(rec["sequence"][:max_len_nt], prefix=prefix)
    return [int(x) for x in sub.tokenizer.tokenize(text)]


@torch.no_grad()
def class_means(sub, records: list[dict], prefix_kind: str, max_len_nt: int,
                device: str = "cuda:0", limit: int | None = None) -> tuple[torch.Tensor, int]:
    """Mean activation per attention site over `records`. Returns (means, n_used)."""
    base = sub.model.model if sub.family == "evo2" else sub.model
    col = _MeanCollector(base)
    used = 0
    with col.attached():
        for r in (records[:limit] if limit else records):
            ids = _encode(sub, r, prefix_kind, max_len_nt)
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            base(x)
            used += 1
    if used == 0:
        raise RuntimeError("no records produced activations; the direction would be empty")
    return col.means(), used


def derive(sub, target_records: list[dict], other_records: list[dict], prefix_kind: str,
           max_len_nt: int, device: str = "cuda:0",
           limit: int | None = None) -> dict:
    """The I1 direction: mean(target) - mean(others), unit-normalised per site.

    Returns an artifact dict; `save()` writes it. Both halves are recorded so the
    derivation can be audited without re-running it.
    """
    base = sub.model.model if sub.family == "evo2" else sub.model
    mt, n_t = class_means(sub, target_records, prefix_kind, max_len_nt, device, limit)
    mo, n_o = class_means(sub, other_records, prefix_kind, max_len_nt, device, limit)
    raw = mt - mo
    norms = raw.norm(dim=-1)
    if torch.any(norms == 0):
        raise RuntimeError(f"a site's direction has zero norm ({norms.tolist()}); "
                           f"normalising would divide by zero and steer nothing")
    unit = raw / norms.unsqueeze(-1)
    return {
        "directions": unit.cpu(),
        "raw_norms": norms.cpu().tolist(),
        "target_mean_norm": mt.norm(dim=-1).cpu().tolist(),
        "other_mean_norm": mo.norm(dim=-1).cpu().tolist(),
        "n_target_records": n_t,
        "n_other_records": n_o,
        "hidden": int(unit.shape[-1]),
        "sites": site_report(base),
        "prefix": prefix_kind,
        "max_len_nt": int(max_len_nt),
        "normalised": True,
    }


def save(art: dict, path: str | Path, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(art)
    if extra:
        out.update(extra)
    torch.save(out, p)
    meta = {k: v for k, v in out.items() if k != "directions"}
    p.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return p


@torch.no_grad()
def projection(sub, records: list[dict], directions: torch.Tensor, prefix_kind: str,
               max_len_nt: int, device: str = "cuda:0",
               limit: int | None = None) -> list[float]:
    """Mean projection of each site's activations onto its direction.

    This is the SPEC 6.4 manipulation check for I1 -- "the injected direction changes an
    independent readout of class in activations". It is a readout, not a probe: no
    parameters are fitted. Run it on HELD-OUT records with the intervention detached and
    attached; if injection does not move it, the direction did not land and any null from
    the arm is uninformative rather than negative.
    """
    m, _ = class_means(sub, records, prefix_kind, max_len_nt, device, limit)
    d = directions.to(m.device, m.dtype)
    return (m * d).sum(dim=-1).cpu().tolist()
