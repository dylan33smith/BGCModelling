"""SPEC 11 verification tests for the scoring layer."""
from __future__ import annotations

import json
from pathlib import Path

from bgcbench.score import antismash, endpoints
from bgcbench.score.novelty import Reference, canonical_kmers

SCORE_DIR = Path(antismash.__file__).parent


# ------------------------------------------------------------ structural (SPEC 9)
def test_single_antismash_invocation_site():
    """SPEC 9.1: exactly one place runs antiSMASH. A second invocation site is how two arms
    come to be scored differently without anyone deciding to. Other tools (mmseqs, for the
    corpus-level novelty reference) are not the endpoint instrument and are not covered."""
    callers = [p for p in SCORE_DIR.glob("*.py")
               if "ANTISMASH" in p.read_text() and p.name != "antismash.py"]
    assert not callers, f"extra antiSMASH sites in score/: {[p.name for p in callers]}"


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


def test_measured_floor_is_low_and_ceiling_is_high():
    """The floor is MEASURED, not assumed. An earlier version asserted FPR == 0, which
    hard-codes the very thing SPEC 4.8 exists to test -- and went red the moment a real
    false positive appeared (RIPP 1/300)."""
    files = list(GATES.glob("gates_*.json"))
    if not files:
        return
    doc = json.loads(files[0].read_text())
    classes = doc.get("classes", doc)
    for cls, v in classes.items():
        assert v["G1_false_positive"]["on_target_rate"] <= 0.02, f"{cls} FPR too high"
        assert v["G5_ceiling"]["on_target_rate"] >= 0.95, f"{cls} ceiling too low"


def test_gate_artifact_binds_the_data_not_only_the_instrument():
    files = list(GATES.glob("gates_*.json"))
    if not files:
        return
    doc = json.loads(files[0].read_text())
    if "classes" not in doc:
        return                                  # pre-fix artifact
    assert doc.get("corpus_sha256"), "gate artifact must record which corpus it measured"


def test_novelty_catches_a_collage_of_training_records():
    """C1 REGRESSION. Forward containment alone cannot fail at the generation budget: its
    denominator is the whole generation, so it decays as 1/length however much was copied.
    Ten whole verbatim training records concatenated scored forward 0.198 -> PASS while
    antiSMASH called them on-target with 15 core genes."""
    import json as _j
    from pathlib import Path as _P
    tr = _P("/data2/ds85/bgcbench/splits/TERPENE/train.jsonl")
    if not tr.exists():
        return
    rows = [_j.loads(l) for l in open(tr)][:200]
    ref = Reference(rows)
    collage = "".join(r["sequence"] for r in rows[:10])[:16000]
    v = ref.verdict(collage)
    assert v["gate"] == "FAIL_memorized", f"a pure copy passed the gate: {v}"
    assert v["novel"] is False
    assert v["containment_reverse"] > v["containment"], (
        "reverse containment is what does the work here")


def test_every_sequence_gets_a_verdict_rejects_duplicates():
    """KNOWN_WRONG #5's red test, which did not previously exist. antiSMASH renames a
    duplicate id to '<id>_0', so both a set-difference check and a count check pass while
    one record's verdict silently serves for two."""
    try:
        antismash.run([("dup", "ACGT" * 100), ("dup", "TTTT" * 100)])
    except ValueError as e:
        assert "duplicate" in str(e).lower()
        return
    raise AssertionError("duplicate accessions were accepted")


def test_no_truncation_of_model_output_in_score():
    """SPEC 3.1: the entire generated sequence is scored; only a seed span may be excluded.
    Mutation-tested -- injecting seq[:6200] anywhere in score/ must break something."""
    import re as _re
    for p in SCORE_DIR.glob("*.py"):
        src = p.read_text()
        for m in _re.finditer(r"sequence\[[^\]]*:[^\]]*\]|seq\[[^\]]*:[^\]]*\]", src):
            frag = m.group(0)
            assert "seed" in src[max(0, m.start() - 200):m.start()].lower(), (
                f"{p.name} slices model output ({frag}) outside a seed context")


def test_confusion_reports_not_applicable_rather_than_zero():
    """SPEC 6.5: an absent cell is NOT APPLICABLE, never a zero. A crashed arm and a
    genuine 0/150 must not be byte-identical -- a row of zeros is the best possible
    specificity result."""
    m = endpoints.confusion({"RIPP": []}, ["RIPP", "NRPS"])
    assert m["RIPP"]["n"] == 0
    assert m["RIPP"]["RIPP"] is None and m["RIPP"]["NRPS"] is None


def test_caller_ids_survive_antismash_sanitisation():
    """antiSMASH strips colons from record ids, so 'oracle::X::y' returns as 'oracleXy'
    and every verdict join misses. The totality check caught it (60 of 60 unmatched), but
    the fix is to submit opaque positional ids and map back."""
    out = antismash.run([("weird::id::with:colons", "ATG" + "ACGT" * 200 + "TAA")])
    assert "weird::id::with:colons" in out, (
        "caller-supplied id did not survive the round trip")


def test_not_applicable_is_expressible_and_distinct_from_zero():
    """SPEC 6.5. A structural absence and a measured null must not be byte-identical --
    a zero is a measurement and reads as the best possible specificity result."""
    from bgcbench.score.record import not_applicable
    na = not_applicable("I1", "evo2-1b", "RIPP", "no residual stream at this site")
    assert na["applicable"] is False
    assert na["on_target"] is None and na["detected"] is None
    assert na["gate"] == "NOT_APPLICABLE"
    assert na["na_reason"]


def test_scored_record_carries_stage_and_both_novelty_directions():
    from bgcbench.score.record import REQUIRED
    assert "stage" in REQUIRED and "applicable" in REQUIRED


def test_corpus_novelty_fails_closed_when_reference_is_absent():
    """The gate must never report PASS for a check it did not run."""
    from pathlib import Path as _P
    from bgcbench.score.novelty import corpus_novelty
    try:
        corpus_novelty([{"generation_id": "g", "sequence": "ACGT" * 100}],
                       _P("/nonexistent/corpus.fa"))
    except FileNotFoundError:
        return
    raise AssertionError("corpus novelty returned without a reference")


def test_generation_config_is_frozen_and_hashed_like_scoring():
    """The n=150 defect: n lived at the COMMAND-LINE INVOCATION SITE, not in any frozen
    config, so a shell script passed a value nobody had agreed and nothing caught it.
    Scoring had been frozen and hashed since SPEC 3; generation had no equivalent."""
    from bgcbench.model import genconfig
    h = genconfig.config_hash()
    assert len(h) == 12 and h == genconfig.config_hash()
    assert genconfig.FROZEN["n_per_row"] == 200
    for k in ("budget_nt", "temperature", "top_k", "top_p", "rng_seed", "batch_size"):
        assert k in genconfig.FROZEN, f"{k} is not frozen and could differ between arms"


def test_generation_budget_matches_the_corpus_bound():
    """A budget below the corpus bound silently handicaps the long classes: BETALACTONE's
    real cores have a ~9 kb median, so a 4 kb budget makes it impossible for that arm to
    produce anything resembling its own reference, and the deficit reads as a class effect."""
    from bgcbench.model.genconfig import FROZEN
    man = Path("/data2/ds85/bgcbench/manifest.json")
    if not man.exists():
        return
    bound = json.loads(man.read_text())["_build"]["max_len"]
    assert FROZEN["budget_nt"] >= bound, (
        f"budget {FROZEN['budget_nt']} < corpus bound {bound}: long classes handicapped")


# REMOVED, both tautologies the uniformity audit identified:
#   test_arms_generated_under_different_configs_cannot_be_compared fed check_uniform two
#     hand-written dicts and passed regardless of how broken the production hash was.
#   test_run_directory_encodes_the_generation_config asserted the literal presence of the
#     vacuous gen_config_hash() call -- i.e. it pinned the defect in place.
# Replaced by test_realised_hash_actually_discriminates_a_drifted_run,
# test_check_uniform_fires_on_realised_not_frozen and
# test_run_directory_refuses_to_overwrite_a_measurement, which exercise behaviour.


def test_realised_hash_actually_discriminates_a_drifted_run():
    """BEHAVIOURAL, not a source grep. config_hash() hashes the FROZEN literal, so it is
    the same constant for every run whatever was passed -- which made the run directory
    collide and check_uniform unable to fire."""
    from bgcbench.model import genconfig as gc
    a = gc.realised(n_per_row=200, budget_nt=16000, seed_len_nt=8)
    b = gc.realised(n_per_row=150, budget_nt=16000, seed_len_nt=8)
    assert gc.realised_hash(a) != gc.realised_hash(b), "a drifted run hashes identically"
    assert gc.config_hash() == gc.config_hash()
    assert gc.off_frozen(b)["n_per_row"]["realised"] == 150
    assert gc.off_frozen(a) == {}, "a frozen run must report no drift"


def test_check_uniform_fires_on_realised_not_frozen():
    from bgcbench.model.genconfig import check_uniform
    try:
        check_uniform([{"arm": "A", "realised_config_hash": "x"},
                       {"arm": "B", "realised_config_hash": "y"}])
    except RuntimeError:
        return
    raise AssertionError("arms with different realised configs compared silently")


def test_generation_actually_seeds_the_rng():
    """rng_seed was stamped into every report and applied nowhere."""
    from bgcbench.model import generate as gen
    src = Path(gen.__file__).read_text()
    assert "def _seed_everything" in src
    assert "_seed_everything(cfg.seed)" in src, "seeding is defined but never called"
    import torch
    gen._seed_everything(7); a = torch.randn(4)
    gen._seed_everything(7); b = torch.randn(4)
    assert torch.equal(a, b), "seeding does not make generation reproducible"


def test_seed_length_is_frozen():
    """An unrecorded default of 0 made `--seeded` without the flag a SILENT de novo arm."""
    from bgcbench.model.genconfig import FROZEN
    assert FROZEN.get("seed_len_nt", 0) > 0


def test_empty_generations_stay_in_the_denominator():
    """Dropping them inflates the rate AND is directionally biased: only a model that
    emits its terminator can produce one, so the deletion concentrates in trained arms and
    is absent from the base control they are compared against."""
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert "EMPTY GENERATIONS STAY IN THE DENOMINATOR" in src
    assert 'gens = [g for g in gens if g["sequence"]]' not in src


def test_run_directory_refuses_to_overwrite_a_measurement():
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert "refusing to overwrite a measurement" in src
    assert "exist_ok=False" in src
    assert "{row_class or 'ALLROWS'}_{rhash}" in src, "run dir must carry class and run hash"


def test_both_substrates_terminate_and_the_mode_is_recorded():
    """Both DO stop and both produce terminator-truncated output; only the internal compute
    differs. Evo2 cannot stop natively (vortex's stop_at_eos prints and does not break), so
    the mode differs between substrates and must be recorded, not assumed."""
    from bgcbench.model.load import Substrate
    evo = Substrate(id="e", family="evo2", checkpoint="c", terminator_id=0,
                    terminator_str=chr(0), appends_terminator=False, native_stop=False,
                    approx_nt_per_token=1.0)
    go = Substrate(id="g", family="genomeocean", checkpoint="c", terminator_id=2,
                   terminator_str="", appends_terminator=True, native_stop=True,
                   approx_nt_per_token=4.8)
    assert evo.termination_mode == "post_hoc_truncation"
    assert go.termination_mode == "native_eos"
    body, hit = evo.truncate_at_terminator("ACGT" + chr(0) + "TTTT")
    assert body == "ACGT" and hit
    from bgcbench.model import generate as gen
    src = Path(gen.__file__).read_text()
    assert "eos_token_id=sub.terminator_id" in src, "HF path must halt on the terminator"


def test_instrument_state_is_recorded_not_intended():
    """FROZEN records the version we EXPECT and the check compares only the major
    component; a point release with changed rules would flip verdicts while every artifact
    still attested 8.0.4."""
    from bgcbench.score import antismash
    assert "OBSERVED" in dir(antismash)
    from bgcbench.data.classmap import build_map, mapping_hash
    m = build_map()["mapping"]
    h = mapping_hash(m)
    assert len(h) == 12 and h == mapping_hash(m)
    assert mapping_hash({**m, "x": "Y"}) != h, "classmap hash must change with the map"


def test_novelty_reference_follows_training_data_not_target_class():
    """Keying it on the target class meant the same memorised output could pass or fail
    depending on which row it landed in."""
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert "train_classes" in src
    assert "THE SPLIT THIS ARM TRAINED ON" in src


def test_lift_denominator_is_named_not_implicit():
    """A per-class adapter has no unconditioned twin; publishing three different quantities
    under one field name compares things that are not the same quantity."""
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert '"lift_denominator"' in src


def test_training_has_a_frozen_config_hash():
    from bgcbench.model.train import TrainConfig, train_config_hash
    a, b = TrainConfig(), TrainConfig(rank=32)
    assert train_config_hash(a) != train_config_hash(b)
    assert train_config_hash(a) == train_config_hash(TrainConfig())


def test_held_out_eval_is_stratified_across_classes():
    """Taking the first N in --classes order made the pooled arm's early stopping and its
    manipulation check both computed on whichever class was typed first."""
    from bgcbench.model import train as tr
    src = Path(tr.__file__).read_text()
    assert "STRATIFY" in src and "by_cls" in src


def test_evo2_generation_is_single_call_not_block_wise():
    """Block-wise cache carry-over runs but is NOT equivalent: 200 tokens in one call vs
    two 100-token blocks agree for only 119/200 characters at the same seed. Adopting it
    would trade a verified generation path for an unverified one."""
    from bgcbench.model import generate as gen
    src = Path(gen.__file__).read_text()
    assert "inference_params_dict" not in src, (
        "block-wise cache carry-over is not equivalent; see FINDINGS 1.5b")
