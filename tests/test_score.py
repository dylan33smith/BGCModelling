"""SPEC 11 verification tests for the scoring layer."""
from __future__ import annotations

import json
from pathlib import Path

from bgcbench.score import antismash, endpoints
from bgcbench.score.novelty import Reference, canonical_kmers

SCORE_DIR = Path(antismash.__file__).parent


# ------------------------------------------------------------ structural (SPEC 9)
def test_single_scoring_site():
    """Exactly one place runs antiSMASH. A second invocation site is how two arms come
    to be scored differently without anyone deciding to."""
    callers = [p for p in SCORE_DIR.glob("*.py")
               if "subprocess.run" in p.read_text() and p.name != "antismash.py"]
    assert not callers, f"extra subprocess sites in score/: {[p.name for p in callers]}"


def test_no_arm_or_class_branching_in_score():
    """SPEC 9.1: no arm-specific branch anywhere under score/."""
    banned = ("W0", "W1", "W2", "W3", "S0", "S1", "I0", "I1", "I2")
    for p in SCORE_DIR.glob("*.py"):
        src = p.read_text()
        for tok in banned:
            assert f'"{tok}"' not in src and f"'{tok}'" not in src, f"{p.name} branches on {tok}"


def test_scoring_config_is_frozen_and_hashed():
    h1 = antismash.config_hash()
    assert len(h1) == 12 and h1 == antismash.config_hash()
    assert antismash.FROZEN["minlength"] == 1, (
        "minlength must be 1: the default of 1000 filters INPUT RECORD length and "
        "silently drops 72% of TERPENE / 48% of RIPP extracted cores")


# ------------------------------------------------------------ novelty (KNOWN_WRONG #3)
def test_novelty_gate_fails_closed_on_empty_kmers():
    ref = Reference([{"sequence": "ACGT" * 40}])
    try:
        ref.containment("AC")
    except ValueError:
        return
    raise AssertionError("containment returned instead of raising; 0.0 is the PASSING "
                         "value, so a gate that cannot compute must raise")


def test_novelty_reference_refuses_to_be_empty():
    try:
        Reference([])
    except ValueError:
        return
    raise AssertionError("an empty reference builds a gate that can never fail")


def test_novelty_detects_memorisation():
    seq = "ACGTTGCAGGATCCTAGCTAGCTAGGCTA" * 8
    ref = Reference([{"sequence": seq}])
    assert ref.containment(seq) == 1.0
    assert ref.verdict(seq)["gate"] == "FAIL_memorized"


def test_novelty_is_strand_agnostic():
    seq = "ACGTTGCAGGATCCTAGCTAGCTAGGCTA" * 8
    rc = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
    assert canonical_kmers(seq) == canonical_kmers(rc)


# ------------------------------------------------------------ endpoints (SPEC 3.4, 3.5)
def _rec(det, obs, gid="g"):
    return {"generation_id": gid, "detected": det, "observed_classes": obs,
            "products": obs, "produced_core_genes": 1}


def test_rates_decompose():
    rows = [_rec(True, ["RIPP"]), _rec(True, ["PKS"]), _rec(False, []), _rec(False, [])]
    r = endpoints.rates(rows, "RIPP")
    assert r["detect_rate"] == 0.5
    assert r["precision"] == 0.5
    assert r["on_target_rate"] == 0.25
    assert abs(r["detect_rate"] * r["precision"] - r["on_target_rate"]) < 1e-9


def test_precision_is_none_when_nothing_detected():
    r = endpoints.rates([_rec(False, []), _rec(False, [])], "RIPP")
    assert r["precision"] is None and r["on_target_rate"] == 0.0


def test_confusion_rows_need_not_sum_to_one():
    """A hybrid record counts for every class it maps to (SPEC 3.3)."""
    m = endpoints.confusion({"RIPP": [_rec(True, ["RIPP", "NRPS"])]}, ["RIPP", "NRPS"])
    assert m["RIPP"]["RIPP"] == 1.0 and m["RIPP"]["NRPS"] == 1.0


def test_lift_is_one_when_an_arm_raises_everything_equally():
    """An arm that lifts every class equally has demonstrated capability, not control."""
    uncond = [_rec(True, ["RIPP"]), _rec(True, ["NRPS"])]
    by_t = {"RIPP": [_rec(True, ["RIPP"]), _rec(True, ["NRPS"])],
            "NRPS": [_rec(True, ["RIPP"]), _rec(True, ["NRPS"])]}
    lf = endpoints.lift(by_t, uncond, ["RIPP", "NRPS"])
    assert lf["RIPP"] == 1.0 and lf["NRPS"] == 1.0


# ------------------------------------------------------------ built gate artifacts
GATES = Path("/data2/ds85/bgcbench/gates")


def test_gate_artifacts_encode_the_config_hash():
    """SPEC 9.3: two scorings of one set can never share a filename."""
    for p in GATES.glob("gates_*.json"):
        assert antismash.config_hash() in p.name


def test_measured_floor_is_zero_and_ceiling_is_high():
    files = list(GATES.glob("gates_*.json"))
    if not files:
        return
    doc = json.loads(files[0].read_text())
    for cls, v in doc.items():
        assert v["G1_false_positive"]["on_target_rate"] == 0.0, f"{cls} FPR non-zero"
        assert v["G5_ceiling"]["on_target_rate"] >= 0.95, f"{cls} ceiling too low"
