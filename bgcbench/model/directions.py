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

#: A site's class difference must be at least this fraction of its activation
#: magnitude for the direction there to be signal rather than rounding.
MIN_RELATIVE_NORM = 1e-3

#: Seed for the val A/B partition. Fixed so the two halves are reproducible from the
#: splits alone, without persisting an index list.
VAL_SPLIT_SEED = 20260915


def val_halves(records: list[dict], seed: int = VAL_SPLIT_SEED) -> tuple[list, list]:
    """Partition a val split into two DISJOINT halves: A for selection, B for the check.

    ⚠ WHY THIS EXISTS -- THE CHECK WAS SELECTING ON ITSELF. `g9_sites` picks an injection
    site set by reading the per-site train/val cosines out of the direction artifact, and
    those cosines were computed on val. Re-reading the SPEC 6.4 manipulation check at the
    chosen sites then returns, by construction, the quantity the selection just maximised
    subject to admissibility: the arm is selected for passing and the pass is reported as
    evidence. Measured on GenomeOcean, TERPENE's `all` set reads mean cosine 0.544 with a
    site at -0.083 (inadmissible) while the selected `early_third` reads 0.655 / +0.420 --
    the selection moved the number that decides the gate.

    ⚠ IT DOES NOT BITE EVO2 EQUALLY, which is why it survived. With 4 attention sites the
    candidate sets are few and `all` was admissible without selection for ARYLPOLYENE; with
    24 sites the subset that passes is always findable, so on GenomeOcean the rule degenerates
    into "search until the check passes".

    Selection reads half A, the check reads half B, and nothing reads both. The direction
    itself is still derived on TRAIN, so the cosine remains a train-vs-held-out comparison
    either way -- what changes is that the half deciding the gate was never optimised over.
    """
    import random
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    mid = len(idx) // 2
    a = [records[i] for i in sorted(idx[:mid])]
    b = [records[i] for i in sorted(idx[mid:])]
    return a, b


def restrict(directions: "torch.Tensor", sites: list[int] | None) -> "torch.Tensor":
    """Zero every site outside `sites`, keeping the tensor's shape.

    Shape is preserved rather than sliced so the row index stays the model's site index
    everywhere downstream -- SPEC 6.5 requires the realised site set be reportable, and a
    re-indexed tensor makes "site 3" ambiguous between the model's and the subset's.
    Mirrors `load.attach_direction`, so the configuration a check is read at and the one an
    arm generates with are produced by the same operation.
    """
    if sites is None:
        return directions
    keep = {int(i) for i in sites}
    bad = [i for i in keep if i < 0 or i >= directions.shape[0]]
    if bad:
        raise ValueError(f"site indices {sorted(bad)} out of range for "
                         f"{directions.shape[0]} sites")
    d = directions.clone()
    for i in range(d.shape[0]):
        if i not in keep:
            d[i] = 0.0
    if not float(d.norm()):
        raise ValueError(f"site subset {sorted(keep)} leaves no live direction; every "
                         f"selected site is degenerate")
    return d


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
    direction is derived from are the ones the arm actually produces.

    ⚠ THE TWO TOKENIZERS RETURN DIFFERENT THINGS FROM `tokenize()`. Evo2's byte-level
    tokenizer returns integer ids; a HuggingFace tokenizer returns STRING pieces, so
    `int("ATG")` raises and every GenomeOcean derivation died on its first record. This
    mirrors `train._encode2`, which already branched on family -- the divergence was here.
    """
    prefix = rec.get("tax_tag", "") if prefix_kind == "taxonomy" else ""
    text = sub.training_text(rec["sequence"][:max_len_nt], prefix=prefix)
    if sub.family == "evo2":
        return [int(x) for x in sub.tokenizer.tokenize(text)]
    return sub.tokenizer(text)["input_ids"]


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
    # ⚠ A RELATIVE FLOOR, NOT `== 0`. Measured on Evo2-1B at 8 records per side, site 3's
    # raw norm came out at 8.7e-05 against a mean-activation norm of 0.104 -- a ratio of
    # 8e-04. An exact-zero guard passed it, the division amplified pure numerical noise into
    # a unit vector, and that vector was then injected at the SAME alpha as the real
    # directions. A site with no class signal must stop the derivation, not contribute
    # noise to it.
    scale = 0.5 * (mt.norm(dim=-1) + mo.norm(dim=-1))
    rel = norms / scale.clamp_min(1e-12)
    degenerate = [i for i, r in enumerate(rel.tolist()) if r < MIN_RELATIVE_NORM]
    active = [i for i in range(len(rel)) if i not in degenerate]
    if not active:
        raise RuntimeError(
            f"no attention site carries a class difference above {MIN_RELATIVE_NORM:g} of "
            f"its activation magnitude (relative norms {[round(x, 6) for x in rel.tolist()]}); "
            f"there is no direction to inject.")
    unit = raw / norms.clamp_min(1e-12).unsqueeze(-1)
    # ⚠ A DEGENERATE SITE IS ZEROED, NOT NORMALISED. Measured on Evo2-1B at 64 records per
    # side, the four attention sites (blocks 3, 10, 17, 24) give relative norms
    # [0.209, 0.210, 0.202, 0.0005]: the first three carry ~20% of their activation
    # magnitude as class difference and the LAST carries 400x less. That is stable, not
    # small-n noise -- block 24 simply has no class-discriminative signal in its output.
    # Dividing by ~0 there would turn rounding error into a unit vector and inject it at
    # the same alpha as the real directions. Zeroing means the site is not steered, and
    # SPEC 6.5 already requires the intervened-site count be reported for exactly this
    # reason: an arm attached at 3 of 4 sites is not the arm attached at 4.
    for i in degenerate:
        unit[i] = 0.0
    return {
        "directions": unit.cpu(),
        "raw_norms": norms.cpu().tolist(),
        "relative_norms": rel.cpu().tolist(),
        "active_sites": active,
        "degenerate_sites": degenerate,
        "min_relative_norm": MIN_RELATIVE_NORM,
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
    """Mean projection of each site's record-weighted activations onto a given direction."""
    m, _ = class_means(sub, records, prefix_kind, max_len_nt, device, limit)
    d = directions.to(m.device, m.dtype)
    return (m * d).sum(dim=-1).cpu().tolist()


@torch.no_grad()
def manipulation_check(sub, val_target: list[dict], val_other: list[dict],
                       train_directions: torch.Tensor, prefix_kind: str, max_len_nt: int,
                       device: str = "cuda:0", limit: int | None = None,
                       alpha: float = 1.0, sites: list[int] | None = None) -> dict:
    """SPEC 6.4 for I1: does the injected direction move an INDEPENDENT readout of class?

    ⚠ THE OBVIOUS CHECK IS CIRCULAR AND THIS DELIBERATELY IS NOT IT. Injecting `alpha * d`
    and then measuring the projection onto `d` raises it by exactly `alpha * ||d||^2` --
    for a unit direction, exactly `alpha`. That holds for ANY vector, including pure noise,
    so it tests arithmetic rather than the arm. An earlier version of this module shipped
    that check, computed with nothing attached, so it could neither fail nor be informative.

    The readout here is derived INDEPENDENTLY, from the VALIDATION split, by the same
    difference-of-means recipe. It can fail: if the train direction is noise it will not
    align with the val direction, `cos` collapses toward 0 and the injected shift vanishes.

    Returns per-site cosines, the readout before and after injection, and the realised
    shift against what a perfectly aligned direction would give.
    """
    from bgcbench.model.interventions import DirectionInjection

    base = sub.model.model if sub.family == "evo2" else sub.model
    mt, _ = class_means(sub, val_target, prefix_kind, max_len_nt, device, limit)
    mo, _ = class_means(sub, val_other, prefix_kind, max_len_nt, device, limit)
    raw = mt - mo
    norms = raw.norm(dim=-1)
    if torch.any(norms == 0):
        raise RuntimeError("a validation readout has zero norm; the check cannot be read")
    d_val = raw / norms.unsqueeze(-1)

    # ⚠ THE CHECK IS READ AT THE SITE SET THE ARM ACTUALLY STEERS. Restricting here rather
    # than at the call site means `active` below is derived from the restricted tensor, so
    # the cosine mean, the anti-alignment test and the injection all describe one
    # configuration. Reading the check at "all sites" and then generating at a subset is how
    # GenomeOcean's four directions were refused for a configuration no arm used.
    d_tr = restrict(train_directions, sites).to(d_val.device, d_val.dtype)
    cos = (d_tr * d_val).sum(dim=-1)

    before = projection(sub, val_target, d_val, prefix_kind, max_len_nt, device, limit)
    iv = DirectionInjection(base, int(d_val.shape[-1]), d_tr.cpu(), alpha).to(d_val.device)
    with iv.attached():
        after = projection(sub, val_target, d_val, prefix_kind, max_len_nt, device, limit)

    shift = [a - b for a, b in zip(after, before)]

    # ⚠ THE SITES ARE IN SERIES, so `alpha * cos` predicts the shift at the FIRST site only.
    # Injection at site 0 perturbs the input to every later site, so their readouts move for
    # two reasons at once. Measured at alpha=1: site 0 matched its prediction to four
    # decimals (-0.4697 vs -0.4697) while site 2 came out -0.456 against a predicted +0.611.
    # An earlier version compared all sites to alpha*cos and would have read propagation as
    # a defect.
    cosl = [float(c) for c in cos]
    # ⚠ "FIRST" MEANS FIRST **STEERED**, NOT MODEL SITE 0, and this line had it wrong in the
    # same revision that fixed it in `projection_vs_alpha`. `restrict` zeroes every row
    # outside the chosen set, so for a subset that excludes site 0 the hook there adds
    # alpha*0; site 0 being the shallowest attention site there is nothing upstream to
    # perturb it, so `after[0] == before[0]` BITWISE and `shift[0]` is exactly 0.0. The
    # `shift[0] != 0.0` conjunct below then forced passes=False for any such subset however
    # good the direction was -- while `first_site_agrees` simultaneously recorded True
    # (|0-0| < 1e-2), so the artifact claimed a confirmed first-site prediction at a site it
    # never steered. Replayed on the recorded candidate rows, that is Evo2 ARYLPOLYENE ([1])
    # and GenomeOcean TERPENE ([4]), ARYLPOLYENE ([12]) and RIPP ([8..15]) -- reinstating the
    # exact refusal this revision exists to remove, and disguising it as a quality failure.
    active = [i for i in range(len(cosl)) if float(d_tr[i].norm()) > 0]
    active = active or list(range(len(cosl)))
    first = active[0]
    expected_first = alpha * float(cos[first])

    # ⚠ AVERAGE OVER THE SITES THE ARM ACTUALLY STEERS. A degenerate site's direction is
    # zeroed by `derive()`, so its cosine is identically 0 -- that is an EXCLUSION, not a
    # measurement, and averaging it in penalises the arm for a site it deliberately does
    # not touch. Measured: ARYLPOLYENE at 192 records per side reads 0.230 over all four
    # sites and 0.307 over the three active ones, which is the difference between FAIL and
    # PASS on the same data.
    mean_cos = sum(cosl[i] for i in active) / len(active)
    return {
        "alpha": float(alpha),
        "cosine_train_val": cosl,
        "mean_cosine": mean_cos,
        "active_sites_checked": active,
        "mean_cosine_all_sites": sum(cosl) / len(cosl),
        "min_active_cosine": min(cosl[i] for i in active),
        "readout_before": before,
        "readout_after": after,
        "shift": shift,
        "first_steered_site": first,
        "first_site_shift": shift[first],
        "first_site_shift_expected": expected_first,
        "first_site_agrees": bool(abs(shift[first] - expected_first)
                                  < 1e-2 * max(1.0, abs(expected_first))),
        # The PRIMARY evidence is the cosine: a train direction that is real class content
        # aligns with one derived independently on held-out records. A noise direction does
        # not, and no amount of injection makes it.
        # ⚠ A MEAN ALONE IS NOT ENOUGH: one strongly reproducing site can carry it while
        # another is ANTI-aligned, which is not a direction that landed. ARYLPOLYENE at 64
        # records per side has sites [-0.18, 0.038, 0.616]: a mean of 0.156 and a site
        # pointing the wrong way. Both conditions are required.
        "site_subset": (sorted(int(i) for i in sites) if sites is not None else None),
        "passes": bool(mean_cos > 0.3
                       and min(cosl[i] for i in active) > 0.0
                       and shift[first] != 0.0),
        "criterion": ("over the sites the arm actually steers (degenerate sites excluded, "
                      "not counted as zero): mean train/val direction cosine > 0.3 AND no "
                      "active site anti-aligned (min cosine > 0) AND the FIRST STEERED "
                      "site's readout moves. That site is the only one whose shift is "
                      "predictable in closed form, because injection propagates through "
                      "the later sites. It is the lowest STEERED index, not model site 0: "
                      "an unsteered site 0 shifts by exactly zero and would fail every "
                      "subset that excludes it."),
    }


@torch.no_grad()
def projection_vs_alpha(sub, records: list[dict], directions: torch.Tensor,
                        prefix_kind: str, max_len_nt: int, alphas: list[float],
                        device: str = "cuda:0", limit: int | None = None,
                        sites: list[int] | None = None) -> dict:
    """SPEC 6 I1 check (a): projection onto `d` rises monotonically with α.

    ⚠ CIRCULAR BY DESIGN, and kept anyway. Injecting `α·d` raises the projection onto `d` by
    `α‖d‖²` for ANY `d`, noise included -- so this cannot tell a class direction from a random
    one (that is part (c)'s job, §12.A4). What it CAN fail on is the mechanism: a hook that
    did not attach, a site set that does not match the directions, a magnitude that is not
    what α says. That failure mode -- the arm silently generating from the base model and
    reading as a clean null -- is worth a cheap check of its own.
    """
    from bgcbench.model.interventions import DirectionInjection

    base = sub.model.model if sub.family == "evo2" else sub.model
    d = restrict(directions, sites)
    proj = []
    for a in alphas:
        if a == 0:
            proj.append(projection(sub, records, d, prefix_kind, max_len_nt, device, limit))
            continue
        iv = DirectionInjection(base, int(d.shape[-1]), d, float(a))
        iv = iv.to(next(base.parameters()).device)
        with iv.attached():
            proj.append(projection(sub, records, d, prefix_kind, max_len_nt, device, limit))
    n_sites = len(proj[0])
    per_site = [[proj[k][i] for k in range(len(alphas))] for i in range(n_sites)]
    active = [i for i in range(n_sites) if float(d[i].norm()) > 0]
    mono = {i: all(per_site[i][k + 1] >= per_site[i][k] - 1e-6
                   for k in range(len(alphas) - 1)) for i in active}
    # ⚠ THE GATE IS THE FIRST STEERED SITE, AND THE REST IS EVIDENCE. This function used to
    # require monotonicity at EVERY steered site, which contradicts what its own sibling
    # `manipulation_check` documents three functions up: the sites are IN SERIES, injection
    # at site 0 perturbs the input to every later site, and only the first site's response
    # is predictable in closed form. `manipulation_check` records that an earlier version
    # "would have read propagation as a defect" and fixed it there; this check kept the bug.
    #
    # ⚠ AND IT IS AN ORDER STATISTIC, SO IT SCALES WITH SITE COUNT. all-of-them over 4 sites
    # and all-of-them over 24 are not the same requirement. Measured on GenomeOcean at the
    # retired alpha grid, RIPP and REDOX_COFACTOR failed this check and NOTHING else --
    # RIPP on site 23 alone, REDOX on sites 2/8/12/23, all of them downstream of up to 23
    # upstream injections -- while both passed the substantive train/val cosine check
    # outright (0.566 and 0.818 mean, no anti-aligned site). Evo2, with 3 live sites, never
    # tripped it. That is the check's geometry talking, not the substrate's.
    #
    # What the check EXISTS to catch is named in the docstring above: a hook that did not
    # attach, a site set that does not match the directions, a magnitude that is not what
    # alpha says. The first steered site detects every one of those, in the only place the
    # response is uncontaminated by propagation.
    first = active[0] if active else None
    n_mono = sum(1 for v in mono.values() if v)
    return {"alphas": list(alphas), "projection_per_site": per_site,
            "active_sites": active, "monotone_per_site": mono,
            "site_subset": (sorted(int(i) for i in sites) if sites is not None else None),
            "first_steered_site": first,
            "first_site_monotone": bool(first is not None and mono[first]),
            "n_sites_monotone": n_mono, "n_sites_steered": len(active),
            "frac_sites_monotone": (n_mono / len(active)) if active else None,
            "passes": bool(first is not None and mono[first]),
            "criterion": ("projection onto d non-decreasing in alpha at the FIRST steered "
                          "site -- the only site whose response to injection is not "
                          "confounded by propagation from the sites upstream of it. "
                          "Monotonicity at the remaining steered sites is reported as "
                          "evidence (n_sites_monotone) but does not gate: requiring it at "
                          "every site is an order statistic over site count, which makes "
                          "the same direction quality fail on a 24-layer stack and pass on "
                          "a 4-site one.")}


@torch.no_grad()
def kl_vs_unsteered(sub, records: list[dict], directions: torch.Tensor, prefix_kind: str,
                    max_len_nt: int, alpha: float, device: str = "cuda:0",
                    limit: int | None = None, min_kl: float = 1e-3,
                    sites: list[int] | None = None) -> dict:
    """SPEC 6 I1 check (b): the steered next-token distribution differs from the unsteered one.

    ⚠ THIS IS THE PART THAT LICENSES READING A NULL. (a) shows the hook fired and (c) shows
    the direction is real, but neither shows the intervention reached the OUTPUT. A direction
    can be genuine and land in activation space while changing the next-token distribution so
    little that generation is unaffected -- and then a zero endpoint means "α was too small",
    not "steering does not work".

    Reports mean KL(steered ‖ unsteered) in nats per position, averaged over positions and
    records.
    """
    import torch.nn.functional as F

    from bgcbench.model.interventions import DirectionInjection
    from bgcbench.model.train import _unwrap

    base = sub.model.model if sub.family == "evo2" else sub.model
    d = restrict(directions, sites)
    iv = DirectionInjection(base, int(d.shape[-1]), d, float(alpha))
    iv = iv.to(next(base.parameters()).device)

    kls, tops = [], []
    for r in (records[:limit] if limit else records):
        ids = _encode(sub, r, prefix_kind, max_len_nt)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        lo = _unwrap(base(x))
        if lo is None:
            raise RuntimeError("could not find the [B,T,V] logits; KL cannot be computed")
        p_un = F.log_softmax(lo.float(), dim=-1)
        with iv.attached():
            ls = _unwrap(base(x))
        p_st = F.log_softmax(ls.float(), dim=-1)
        # KL(steered || unsteered), per position, then averaged
        kl = (p_st.exp() * (p_st - p_un)).sum(-1)
        kls.append(float(kl.mean()))
        tops.append(float((p_st.argmax(-1) != p_un.argmax(-1)).float().mean()))
    if not kls:
        raise RuntimeError("no records produced logits; the KL check measured nothing")
    mean_kl = sum(kls) / len(kls)
    # ⚠ nats/TOKEN IS NOT COMPARABLE ACROSS SUBSTRATES and this number is read across them.
    # Evo2's token is 1 nt; GenomeOcean's is ~4.8 nt, so the same disruption per nucleotide
    # reads ~4.8x larger on GO. The project has already been bitten once by putting Evo2's
    # nats/NUCLEOTIDE beside GO's nats/TOKEN in adjacent rows (CLAUDE.md), and this module's
    # own `alpha_for_target_kl` docstring still asserts GO is "far outside any usable range"
    # on the strength of the unconverted figure. Measured at alpha=1 on the all-sites
    # configuration, per NUCLEOTIDE: GO 0.86-1.86 against Evo2 0.83-1.74 -- the same regime.
    # The real asymmetry is `frac_argmax_changed` (GO 98-99%, Evo2 64-72%), which is a rate
    # and needs no conversion. Both are reported so neither has to be recomputed by a reader.
    nt_per_token = float(getattr(sub, "approx_nt_per_token", 1.0) or 1.0)
    return {"alpha": float(alpha), "mean_kl_nats": mean_kl,
            "mean_kl_nats_per_nt": mean_kl / nt_per_token,
            "nt_per_token": nt_per_token,
            "per_record_kl": kls, "n_records": len(kls),
            "frac_argmax_changed": sum(tops) / len(tops),
            "min_kl": min_kl,
            "site_subset": (sorted(int(i) for i in sites) if sites is not None else None),
            "passes": bool(mean_kl > min_kl),
            "criterion": (f"mean KL(steered || unsteered) over next-token distributions "
                          f"> {min_kl} nats/TOKEN -- the intervention reached the output. "
                          f"mean_kl_nats_per_nt is the cross-substrate-comparable form; "
                          f"nats/token is not (this substrate: {nt_per_token} nt/token)")}


@torch.no_grad()
def alpha_for_target_kl(sub, records, directions, prefix_kind, max_len_nt,
                        target_kl: float = 1.0, device: str = "cuda:0",
                        limit: int = 8, lo: float = 1e-3, hi: float = 4.0,
                        iters: int = 8) -> dict:
    """Find the α at which injection produces a TARGET effect size, not a target magnitude.

    ⚠ THIS FUNCTION IS NOT ON THE GATE PATH, AND ITS ORIGINAL JUSTIFICATION WAS A UNIT ERROR.
    It was written on the claim that "at α = 1 Evo2-1B sits at KL 0.197 nats/position while
    GenomeOcean-4B sits at 3.44-7.81 ... far outside any usable range", i.e. that one model
    was being probed in its working regime and the other in wreckage. Those two figures are
    in DIFFERENT UNITS: nats per Evo2 token is nats per NUCLEOTIDE, nats per GenomeOcean
    token is nats per ~4.8 nucleotides. Measured on the trained per-class adapters at α = 1,
    converted to nats/NUCLEOTIDE, GO reads 0.86-1.86 against Evo2's 0.83-1.74 — the same
    regime, not a 5x gap. `kl_vs_unsteered` now returns `mean_kl_nats_per_nt` so the
    comparable quantity is the one to hand.

    ⚠ AND MATCHING ON REALISED EFFECT IS ITSELF A CROSS-SUBSTRATE MATCH, which SPEC §14A
    forbids: "probe magnitude" is named there as a TREATMENT parameter, chosen per substrate
    by measurement, not equalised. Checks (a) and (b) are therefore read at the substrate's
    OWN generation α grid — the magnitudes the arm actually runs at — which is per-substrate
    by construction and needs no calibration.

    Kept because bisecting α to a target effect is still the right tool if a future gate
    needs a magnitude it cannot read off the generation grid. Bisects α so mean KL against
    unsteered lands near `target_kl`, in nats/TOKEN.
    """
    def kl_at(a):
        return kl_vs_unsteered(sub, records, directions, prefix_kind, max_len_nt,
                               alpha=float(a), device=device, limit=limit)["mean_kl_nats"]

    k_hi = kl_at(hi)
    if k_hi < target_kl:                       # even the ceiling is gentle
        return {"alpha": float(hi), "realised_kl": k_hi, "target_kl": target_kl,
                "note": "target KL not reachable below hi; returning hi"}
    a_lo, a_hi = lo, hi
    best = (hi, k_hi)
    for _ in range(iters):
        mid = 0.5 * (a_lo + a_hi)
        k = kl_at(mid)
        if abs(k - target_kl) < abs(best[1] - target_kl):
            best = (mid, k)
        if k > target_kl:
            a_hi = mid
        else:
            a_lo = mid
    return {"alpha": float(best[0]), "realised_kl": float(best[1]),
            "target_kl": target_kl,
            "criterion": ("alpha bisected so mean KL against unsteered lands near the target "
                          "-- substrates are probed at matched EFFECT, not matched magnitude")}
