"""Guard the CUSTOM TRAINING OBJECTIVES — the parts that fail silently and invalidate a run.

Four things here produce a plausible training curve while being wrong:

1. THE BASELINE DRIFTING. The factorial's control cell must be the CURRENT objective bit for bit.
   If `custom_lm_loss` with both extras off differs at all from `causal_lm_loss`, every delta is
   measured against the wrong reference and the whole design is void.
2. REVERSE-STRAND PHASE. Codon phase runs from the gene's START, which on the minus strand is its
   HIGH coordinate. Getting it backwards mislabels roughly half of all genes and still yields a
   mask of the right shape and range.
3. PER-RECORD NORMALISATION. Without it, "domain weighting" silently becomes LENGTH weighting,
   because coverage runs 78.6% in short cores and 25.1% in long ones.
4. PENALISING REAL STOPS. A gene must be able to end. If the penalty hits the true terminal stop,
   the model is taught never to terminate — which would look like longer ORFs (the thing we are
   hoping for) while actively breaking gene structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

IGNORE_INDEX = -100
A, C, G, T = ord("A"), ord("C"), ord("G"), ord("T")


def test_baseline_is_bit_identical_to_the_current_objective():
    """THE MOST IMPORTANT ONE. With both extras off, the new loss must equal the old one."""
    from bgc_pipeline.objective import custom_lm_loss

    torch.manual_seed(0)
    logits = torch.randn(2, 12, 512)
    labels = torch.randint(0, 512, (2, 12))
    labels[0, :3] = IGNORE_INDEX                       # prefix masking, as in training

    # the existing implementation, inlined so the test does not depend on importing the trainer
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    old = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1), ignore_index=IGNORE_INDEX)

    new, comp = custom_lm_loss(logits, labels)
    assert torch.allclose(old, new, atol=0, rtol=0), (
        f"baseline drifted: old {float(old):.10f} vs new {float(new):.10f} — every cell of the "
        f"factorial would be measured against the wrong control")
    assert abs(comp["ce"] - float(old)) < 1e-9
    print("PASS objective: extras-off baseline is bit-identical to the current objective")


def test_reverse_strand_phase_counts_from_the_gene_start():
    from bgc_pipeline.objective import build_frame_mask

    fwd = build_frame_mask(12, [[3, 12, 1]])
    assert fwd[:3].tolist() == [-1, -1, -1], "positions outside a gene must be -1"
    assert fwd[3:12].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2], f"forward phase wrong: {fwd.tolist()}"

    rev = build_frame_mask(12, [[3, 12, -1]])
    # on the minus strand the gene starts at the HIGH coordinate, so phase runs backwards
    assert rev[3:12].tolist() == [2, 1, 0, 2, 1, 0, 2, 1, 0], (
        f"reverse phase wrong: {rev.tolist()} — counting from the low coordinate mislabels every "
        f"minus-strand gene while still producing a plausible mask")
    print("PASS objective: codon phase runs from the gene start on BOTH strands")


def test_weights_normalise_per_record():
    from bgc_pipeline.objective import build_position_weights

    # a SHORT record that is mostly domain, and a LONG one that is mostly not — the real spread
    short = build_position_weights(100, [[0, 80, "PF00501", 1]], weight=3.0)
    long_ = build_position_weights(1000, [[0, 250, "PF00501", 1]], weight=3.0)
    assert abs(float(short.mean()) - 1.0) < 1e-5, "short record not normalised"
    assert abs(float(long_.mean()) - 1.0) < 1e-5, "long record not normalised"
    # and the domain positions are still up-weighted RELATIVE to the rest within each record
    assert float(short[0]) > float(short[-1]) and float(long_[0]) > float(long_[-1])
    # without normalisation the two records would carry very different total weight
    raw_s = build_position_weights(100, [[0, 80, "PF00501", 1]], weight=3.0,
                                   normalise_per_record=False)
    raw_l = build_position_weights(1000, [[0, 250, "PF00501", 1]], weight=3.0,
                                   normalise_per_record=False)
    assert abs(float(raw_s.mean()) - float(raw_l.mean())) > 0.4, (
        "fixture does not exercise the imbalance normalisation exists to fix")
    print("PASS objective: per-record normalisation equalises total weight across core lengths")


def test_only_class_defining_spans_are_up_weighted():
    from bgc_pipeline.objective import build_position_weights

    w = build_position_weights(30, [[0, 10, "PF99999", 0], [20, 30, "PF00501", 1]],
                               weight=4.0, normalise_per_record=False)
    assert float(w[0]) == 1.0, "a NON class-defining domain must not be up-weighted"
    assert float(w[25]) == 4.0, "the class-defining domain was not up-weighted"
    print("PASS objective: only class-defining spans are up-weighted")


def test_stop_penalty_fires_on_completions_and_spares_real_stops():
    from bgc_pipeline.objective import stop_completion_penalty

    V = 512
    # context ... T A ?   with the third position at codon phase 2 inside a gene
    ids = torch.tensor([[C, T, A, C]])
    labels = ids.clone()
    phase = torch.tensor([[0, 0, 1, 2]], dtype=torch.int8)
    # model puts everything on A, which would close TAA
    logits = torch.full((1, 3, V), -20.0)
    logits[0, 2, A] = 20.0
    pen_bad = stop_completion_penalty(logits, ids, phase, labels)
    assert float(pen_bad) > 0.9, f"penalty did not fire on a stop completion: {float(pen_bad)}"

    # same setup, but the model puts everything on C — a legitimate continuation
    logits_ok = torch.full((1, 3, V), -20.0)
    logits_ok[0, 2, C] = 20.0
    pen_ok = stop_completion_penalty(logits_ok, ids, phase, labels)
    assert float(pen_ok) < 0.05, f"penalty fired on a non-stop base: {float(pen_ok)}"

    # THE ONE THAT MATTERS: when the TRUE next base is the stop-completing one — a real gene
    # terminus — the penalty must NOT fire, or the model is taught never to end a gene.
    labels_stop = torch.tensor([[C, T, A, A]])
    ids_stop = labels_stop.clone()
    pen_real = stop_completion_penalty(logits, ids_stop, phase, labels_stop)
    assert float(pen_real) == 0.0, (
        f"penalised a REAL gene terminus ({float(pen_real)}) — this would teach the model never "
        f"to stop, which looks like longer ORFs while destroying gene structure")
    print("PASS objective: stop penalty fires on completions, spares real termini")


def test_frame_lambda_zero_changes_nothing():
    from bgc_pipeline.objective import custom_lm_loss

    torch.manual_seed(1)
    logits = torch.randn(1, 8, 512)
    labels = torch.randint(0, 512, (1, 8))
    ids = labels.clone()
    phase = torch.full((1, 8), 2, dtype=torch.int8)
    a, _ = custom_lm_loss(logits, labels)
    b, _ = custom_lm_loss(logits, labels, input_ids=ids, frame_phase=phase, frame_lambda=0.0)
    assert torch.allclose(a, b), "frame_lambda=0 must be a no-op"
    print("PASS objective: frame_lambda=0 is exactly the baseline")




def test_window_annotations_apply_both_offsets():
    """The window offset and the prefix offset must BOTH be applied, in that order.

    Either one omitted puts the weights on the wrong bases while producing a tensor of exactly the
    right shape and range — the failure is invisible downstream.
    """
    from bgc_pipeline.annotations import window_annotations

    # record of 30 nt; a class-defining domain covers [20,30). Window is [15,30) -> 15 nt.
    # Tokenised row = 5 prefix tokens + 15 sequence tokens = 20.
    w, ph = window_annotations(record_length=30, genes=[[15, 30, 1]],
                               domain_spans=[[20, 30, "PF00501", 1]],
                               nt_start=15, nt_end=30, prefix_token_count=5, total_tokens=20,
                               domain_weight=3.0, normalise_per_record=False)
    assert w.shape[0] == 20 and ph.shape[0] == 20
    assert torch.allclose(w[:5], torch.ones(5)), "prefix must not be weighted"
    assert (ph[:5] == -1).all(), "prefix must have no codon phase"
    # window nt 15..19 are OUTSIDE the domain -> tokens 5..9 weight 1
    assert torch.allclose(w[5:10], torch.ones(5)), f"pre-domain window weights wrong: {w[5:10]}"
    # window nt 20..29 are INSIDE the domain -> tokens 10..19 weight 3
    assert torch.allclose(w[10:20], torch.full((10,), 3.0)), f"domain weights misplaced: {w[10:20]}"
    # the gene starts at record nt 15, which is the first window position -> phase 0,1,2,...
    assert ph[5:11].tolist() == [0, 1, 2, 0, 1, 2], f"phase misaligned in window: {ph[5:11]}"
    print("PASS annot: window and prefix offsets both applied, phase preserved across the window")


def test_normalisation_is_over_the_record_not_the_window():
    """Normalising per WINDOW would make an all-domain window identical to an all-linker one,
    destroying the contrast the weighting exists to create."""
    from bgc_pipeline.annotations import window_annotations

    # a record that is domain only in its second half
    kw = dict(record_length=100, genes=[], domain_spans=[[50, 100, "PF00501", 1]],
              prefix_token_count=0, domain_weight=4.0, normalise_per_record=True)
    w_lo, _ = window_annotations(nt_start=0, nt_end=50, total_tokens=50, **kw)    # all linker
    w_hi, _ = window_annotations(nt_start=50, nt_end=100, total_tokens=50, **kw)  # all domain
    assert float(w_hi.mean()) > float(w_lo.mean()) * 3, (
        f"window means {float(w_lo.mean()):.3f} vs {float(w_hi.mean()):.3f} — normalisation was "
        f"applied per WINDOW, which erases the domain/linker contrast")
    print("PASS annot: normalisation is over the record, so window contrast survives")


def main() -> int:
    for t in (test_baseline_is_bit_identical_to_the_current_objective,
              test_reverse_strand_phase_counts_from_the_gene_start,
              test_weights_normalise_per_record,
              test_only_class_defining_spans_are_up_weighted,
              test_stop_penalty_fires_on_completions_and_spares_real_stops,
              test_frame_lambda_zero_changes_nothing,
              test_window_annotations_apply_both_offsets,
              test_normalisation_is_over_the_record_not_the_window):
        t()
    print("\nALL OBJECTIVE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
