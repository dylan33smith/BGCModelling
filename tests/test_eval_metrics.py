#!/usr/bin/env python3
"""Pure-logic unit tests consolidated from prior inline checks.

Covers the metric/aggregation logic that does not need a GPU:
  - M9  conditioning-adherence aggregation (eval_conditioning_adherence)
  - M9  balanced per-class sampling
  - M2  validation length-bucket labelling + first-window filter
  - M1  grad-accum-aligned resume pointer (make_client_state)
  - M7  early-stopping state machine (mirrors the training-loop logic)

Run: python tests/test_eval_metrics.py
"""

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_conditioning_adherence as E  # noqa: E402
import finetune_evo2_lora as F  # noqa: E402


def test_m9_adherence():
    classes = ["A", "B", "C", "D"]
    per = [
        {"true_class": "A", "scores": {"A": -1.0, "B": -2.0, "C": -3.0, "D": -4.0}},
        {"true_class": "B", "scores": {"A": -1.0, "B": -2.0, "C": -3.0, "D": -4.0}},
        {"true_class": "C", "scores": {"A": -5.0, "B": -4.0, "C": -1.0, "D": -2.0}},
    ]
    s = E.summarize_adherence(per, classes)
    assert s["n_scored"] == 3
    assert abs(s["top1_acc"] - 2 / 3) < 1e-3
    assert s["top3_acc"] == 1.0
    assert abs(s["mrr"] - (1 + 0.5 + 1) / 3) < 1e-3
    assert abs(s["mean_per_token_margin"] - 1 / 3) < 1e-3
    assert s["per_class_recall"] == {"A": 1.0, "B": 0.0, "C": 1.0}
    assert s["confusion_top1"]["B"] == {"A": 1}
    assert s["random_baseline_top1"] == 0.25
    # single-class score dict is skipped (need >= 2 to rank)
    assert E.summarize_adherence([{"true_class": "A", "scores": {"A": -1.0}}], classes)["n_scored"] == 0
    print("PASS M9: conditioning-adherence aggregation")


def test_m9_balanced_sample():
    recs = [{"compound_class": c} for c in ["X"] * 5 + ["Y"] * 3 + ["Z"] * 1]
    cc = Counter(r["compound_class"] for r in E.balanced_sample(recs, 2, random.Random(0)))
    assert cc["X"] == 2 and cc["Y"] == 2 and cc["Z"] == 1
    print("PASS M9: balanced per-class sampling")


def test_m2_length_buckets_and_first_window():
    b = F.VAL_LENGTH_BOUNDS
    assert F._length_bucket_label(5000, b) == "<=16k"
    assert F._length_bucket_label(20000, b) == "<=32k"
    assert F._length_bucket_label(200000, b) == ">131k"
    # first-window filter keeps exactly one (nt_start==0) window per record
    lengths = np.array([5000, 40000, 120000, 300000])
    allw = F.build_all_chunk_indices(lengths, 32768, 207, 2048, 0, 5)
    fw = [c for c in allw if c[1] == 0]
    assert len(fw) == len(lengths)
    assert sorted(c[0] for c in fw) == list(range(len(lengths)))
    print("PASS M2: length buckets + first-window filter")


def test_m1_resume_alignment():
    def aligned(micro, ga):
        return F.make_client_state(step=1, best_val_loss=0.5, epoch=0,
                                   micro_step_in_epoch=micro, grad_accum=ga)["micro_step_in_epoch"]
    assert aligned(6, 128) == 0       # mid-accumulation -> back to last boundary
    assert aligned(130, 128) == 128   # one boundary completed
    assert aligned(256, 128) == 256   # already on a boundary
    assert aligned(5, 1) == 5         # ga=1 -> no change
    print("PASS M1: grad-accum-aligned resume pointer")


def test_m7_early_stop():
    def simulate(losses, patience, min_delta):
        best, no_imp = float("inf"), 0
        for i, v in enumerate(losses):
            prev = best
            if v < best:
                best = v
            if v < prev - min_delta:
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    return i
        return None
    assert simulate([1.0, 0.9, 0.85, 0.85, 0.85, 0.85, 0.84], 3, 0.001) == 5
    assert simulate([1.0, 0.9, 0.8, 0.7, 0.6], 3, 0.001) is None  # steady improvement
    print("PASS M7: early-stopping state machine")


def main():
    test_m9_adherence()
    test_m9_balanced_sample()
    test_m2_length_buckets_and_first_window()
    test_m1_resume_alignment()
    test_m7_early_stop()
    print("\nALL EVAL-METRIC TESTS PASSED")


if __name__ == "__main__":
    main()
