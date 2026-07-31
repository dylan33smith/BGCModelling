#!/usr/bin/env python3
"""Tests for the controlled-conditioning experiment logic (no GPU).

Run: python tests/test_conditioning_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evo2" / "scripts"))
import conditioning_experiment as C  # noqa: E402


def test_plans():
    cp = C.build_class_plan(["|D__BACTERIA;G__STREPTOMYCES"], ["NRPS", "PKS"], 3)
    assert len(cp) == 1 * 2 * 3
    assert all(p["experiment"] == "class" and "requested_class" in p for p in cp)

    tp = C.build_taxon_plan(["NRPS", "PKS"], C.ECOLI_TAXON, C.STREPTO_TAXON, 2)
    assert len(tp) == 2 * 2 * 2  # classes x reps x {target,source}
    labels = {p["taxon_label"] for p in tp}
    assert labels == {"target", "source"}
    # matched pairs: equal target/source counts per class
    for cls in ("NRPS", "PKS"):
        t = sum(1 for p in tp if p["requested_class"] == cls and p["taxon_label"] == "target")
        s = sum(1 for p in tp if p["requested_class"] == cls and p["taxon_label"] == "source")
        assert t == s == 2
    print("PASS: class + taxon plans (counts, matched pairs)")


def test_class_confusion():
    rows = (
        [{"requested_class": "NRPS", "recovered_class": "NRPS"}] * 3 +
        [{"requested_class": "PKS", "recovered_class": "PKS"}] * 2 +
        [{"requested_class": "TERPENE", "recovered_class": "TERPENE"}] +
        [{"requested_class": "TERPENE", "recovered_class": "NRPS"}]
    )
    r = C.class_confusion(rows, ["NRPS", "PKS", "TERPENE"])
    assert r["n"] == 7
    assert abs(r["diagonal_accuracy"] - 6 / 7) < 1e-3
    assert abs(r["majority_class_prior"] - 4 / 7) < 1e-3   # NRPS recovered 4x
    assert r["beats_prior"] is True
    assert r["per_class_recall"]["NRPS"] == 1.0
    assert r["per_class_recall"]["TERPENE"] == 0.5
    assert r["confusion"]["TERPENE"] == {"TERPENE": 1, "NRPS": 1}
    # recovered_class=None rows are excluded
    assert C.class_confusion([{"requested_class": "X", "recovered_class": None}], ["X"])["n"] == 0
    print("PASS: class confusion (diagonal vs prior, per-class recall, confusion)")


def test_taxon_effect():
    rows = (
        [{"requested_class": "NRPS", "taxon_label": "target", "gc": 0.52, "cai_ecoli": 0.7},
         {"requested_class": "NRPS", "taxon_label": "target", "gc": 0.50, "cai_ecoli": 0.7},
         {"requested_class": "NRPS", "taxon_label": "source", "gc": 0.71, "cai_ecoli": 0.4},
         {"requested_class": "NRPS", "taxon_label": "source", "gc": 0.70, "cai_ecoli": 0.4}] +
        # PKS: target NOT closer to E. coli GC than source -> does not steer
        [{"requested_class": "PKS", "taxon_label": "target", "gc": 0.70},
         {"requested_class": "PKS", "taxon_label": "source", "gc": 0.71}]
    )
    e = C.taxon_effect(rows)
    nrps = e["per_class"]["NRPS"]
    assert nrps["ecoli_tag_steers_gc_toward_ecoli"] is True       # 0.51 closer to 0.508 than 0.705
    assert nrps["target_cai_ecoli"] == 0.7 and nrps["source_cai_ecoli"] == 0.4
    assert e["per_class"]["PKS"]["ecoli_tag_steers_gc_toward_ecoli"] is False
    assert e["n_classes_steered"] == 1 and e["n_classes"] == 2
    assert e["verdict"] == "steers"   # 1 >= max(1, 2//2)
    assert C.taxon_effect([])["verdict"] == "no_data"
    print("PASS: taxon effect (GC steering toward E. coli, CAI, verdict)")


def main():
    test_plans()
    test_class_confusion()
    test_taxon_effect()
    print("\nALL CONDITIONING-EXPERIMENT TESTS PASSED")


if __name__ == "__main__":
    main()
