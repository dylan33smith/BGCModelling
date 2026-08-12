"""CUSTOM TRAINING OBJECTIVES: frame-aware and domain-weighted next-base loss.

Model-agnostic on purpose — these operate on logits, labels and per-position annotations, so the
same code serves the 7B and the 1B track.

────────────────────────────────────────────────────────────────────────────────────────────────
WHY THE STANDARD OBJECTIVE IS MISALIGNED WITH THE TASK

The current loss is plain next-base cross-entropy: every position counts equally, and every wrong
answer counts equally. Both are wrong for this task, in different ways, and each variant fixes one.

1. NOT EVERY POSITION MATTERS EQUALLY. 33.7% of training nucleotides sit inside class-defining
   domains (measured, `train.domain_spans.jsonl`); they receive 33.7% of the gradient.

2. NOT EVERY MISTAKE MATTERS EQUALLY, AND THIS IS THE SHARPER POINT. If the true base is C, then
   guessing A, G or T all cost the same under cross-entropy. At GENERATION time they do not: a
   synonymous third-position change is harmless, while a base that completes a stop codon
   *terminates the protein* and makes everything downstream junk. Cross-entropy never expresses
   that asymmetry. Measured consequence: de novo the model's longest ORF plateaus at 340-550 aa
   however much room it is given, while real DNA scales to 1,134 aa at a 6 kb window — it hits
   stops and restarts (6.0 ORFs per 6 kb against 4.0 for real).

────────────────────────────────────────────────────────────────────────────────────────────────
WHAT THESE ARE **NOT** EVIDENCE FOR, STATED UP FRONT

`max_orf_aa` was demoted as a target because within de novo generations it does not track domain
content (r = 0.051 at 2 kb, -0.120 at 6 kb). So the frame-aware term aims at a quantity we have
NOT shown to be sufficient. The bet is that reading-frame length is a *gate* the model is failing
(one adenylation domain is ~400 aa; de novo ORFs top out at 336-549) rather than a quantity to
maximise, and a gate can show zero within-group correlation while still being an absolute barrier.

**Therefore these runs are scored on `best_bio_bits`, not on ORF length.** If ORF length rises and
biosynthetic content does not follow, that is a clean informative negative: length was never the
constraint. The objective is the lever; the ladder is the measurement.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

# Base ids under CharLevelTokenizer are raw ASCII bytes.
_A, _C, _G, _T = ord("A"), ord("C"), ord("G"), ord("T")
STOP_CODONS = ("TAA", "TAG", "TGA")


def build_position_weights(
    length: int,
    domain_spans: list,
    weight: float,
    normalise_per_record: bool = True,
) -> torch.Tensor:
    """Per-position loss weights, up-weighting class-defining domain positions.

    PER-RECORD NORMALISATION IS THE POINT, NOT A DETAIL. Class-domain coverage is 78.6% in cores
    under 1 kb and 25.1% above 50 kb. A flat multiplier therefore barely touches a short core and
    transforms a long one, so an unnormalised "domain weighting" silently becomes a LENGTH
    weighting — the model would be told long cores matter more, which is not the hypothesis.
    Normalising so every record's weights average 1.0 makes each core contribute equally regardless
    of how much of it happens to be annotated.
    """
    w = torch.ones(length, dtype=torch.float32)
    for span in domain_spans or []:
        s, e = int(span[0]), int(span[1])
        is_class = bool(span[3]) if len(span) > 3 else True
        if is_class and e > s:
            w[max(s, 0):min(e, length)] = weight
    if normalise_per_record and float(w.mean()) > 0:
        w = w / float(w.mean())
    return w


def build_frame_mask(length: int, genes: list) -> torch.Tensor:
    """Codon phase per position: 0/1/2 inside a gene, -1 outside any gene.

    Phase is measured from the gene's START, which on the reverse strand is its HIGH coordinate --
    getting that backwards silently mislabels the phase of every reverse-strand gene (about half of
    them) while still producing a plausible-looking mask.
    """
    ph = torch.full((length,), -1, dtype=torch.int8)
    for g in genes or []:
        s, e, strand = int(g[0]), int(g[1]), int(g[2])
        s, e = max(s, 0), min(e, length)
        if e <= s:
            continue
        idx = torch.arange(s, e)
        ph[s:e] = (((idx - s) % 3) if strand >= 0 else ((e - 1 - idx) % 3)).to(torch.int8)
    return ph


def stop_completion_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    frame_phase: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Probability mass the model puts on COMPLETING an in-gene stop codon.

    Applied only where the next position is codon-phase 2 (the third base) inside a gene, so the
    two preceding bases are known from the context. Given them:
        TA + {A,G} -> TAA / TAG        TG + {A}   -> TGA
    We penalise exactly the base(s) that would close a stop there, and nothing else. Positions
    outside genes, at other phases, or where the true base IS a stop (a legitimate gene terminus)
    are untouched -- penalising a real stop would teach the model never to end a gene.

    Returns a scalar mean over the penalised positions; 0.0 when there are none.
    """
    B, Lm1, V = logits.shape
    probs = F.softmax(logits.float(), dim=-1)
    total = logits.new_zeros(())
    count = 0

    # position t in `logits` predicts input_ids[:, t+1]
    phase_next = frame_phase[:, 1:]                       # phase of the predicted position
    prev1 = input_ids[:, :-1]                             # base immediately before it
    prev2 = torch.cat([input_ids[:, :1], input_ids[:, :-2]], dim=1)  # two before

    is_third = (phase_next == 2)
    valid = is_third & (labels[:, 1:] != IGNORE_INDEX)
    if not bool(valid.any()):
        return total

    ta = (prev2 == _T) & (prev1 == _A) & valid            # TA_ -> A or G closes a stop
    tg = (prev2 == _T) & (prev1 == _G) & valid            # TG_ -> A closes a stop
    true_next = labels[:, 1:]

    # Accumulate the stop-completing probability MASS PER POSITION, then average over penalised
    # positions. Averaging over (position x base) pairs instead would halve the TA case -- a
    # position with all its mass on A would score 0.5 rather than 1.0, understating the penalty by
    # exactly the number of stop-completing bases at that context.
    for mask, bad in ((ta, (_A, _G)), (tg, (_A,))):
        if not bool(mask.any()):
            continue
        # EXCLUDE THE WHOLE POSITION at a real gene terminus. If the TRUE base completes a stop
        # here, this is where the gene legitimately ends, and ANY stop-completing base is correct
        # -- penalising the alternatives (mass on G at a real TAA) would still teach the model to
        # avoid terminating. Excluding per-BASE rather than per-POSITION leaves exactly that
        # residue behind.
        true_completes = torch.zeros_like(mask)
        for b in bad:
            true_completes = true_completes | (true_next == b)
        m = mask & ~true_completes
        if not bool(m.any()):
            continue
        mass = torch.zeros_like(probs[:, :, 0])
        for b in bad:
            mass = mass + probs[:, :, b]
        total = total + mass[m].sum()
        count += int(m.sum())
    return total / count if count else total


def custom_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    position_weights: Optional[torch.Tensor] = None,
    frame_phase: Optional[torch.Tensor] = None,
    frame_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Next-base CE, optionally position-weighted, optionally with a stop-completion penalty.

    With both extras off this is EXACTLY the existing `causal_lm_loss`, which matters: the baseline
    cell of the factorial must be the current objective bit for bit, or a difference cannot be
    attributed to the intervention. Guarded by a test.

    Returns (loss, components) so the driver can log the CE and the penalty separately -- a
    penalty that is silently zero, or that swamps the CE, is otherwise invisible in a single
    number.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    comp: dict = {}

    if position_weights is None:
        ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1), ignore_index=IGNORE_INDEX)
    else:
        per_tok = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1), ignore_index=IGNORE_INDEX,
                                  reduction="none").view(shift_labels.shape)
        w = position_weights[..., 1:].to(per_tok.dtype)
        keep = (shift_labels != IGNORE_INDEX).to(per_tok.dtype)
        denom = (w * keep).sum().clamp_min(1e-8)
        ce = (per_tok * w * keep).sum() / denom
    comp["ce"] = float(ce.detach())

    loss = ce
    if frame_lambda > 0 and frame_phase is not None and input_ids is not None:
        pen = stop_completion_penalty(shift_logits, input_ids, frame_phase, labels)
        comp["stop_penalty"] = float(pen.detach())
        loss = loss + frame_lambda * pen
    comp["total"] = float(loss.detach())
    return loss, comp
