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


MODEL_USABLE_CONTEXT = 8192          # evo2-1b config max_seqlen, measured (see below)


def test_budget_and_corpus_bound_both_fit_the_model_context():
    """Two constraints, and an earlier version of this test had them backwards.

    A budget BELOW the corpus bound handicaps the long classes -- BETALACTONE's cores have
    a ~9 kb median, so a 4 kb budget made it unable to reach its own reference.
    But a budget or a corpus bound ABOVE the model's usable context is worse: measured on
    evo2-1b, NLL of the last 1000 tokens rises 0.805 at 8,192 -> 1.040 at 12,000 -> 1.239
    at 15,900, against ln(4) = 1.386 chance. Beyond ~10 kb the model is barely modelling.

    So both must sit at or below the context, and the corpus bound is the one that has to
    move -- generation cannot be stretched to cover data the model cannot read.
    """
    from bgcbench.model.genconfig import FROZEN
    assert FROZEN["budget_nt"] <= MODEL_USABLE_CONTEXT
    man = Path("/data2/ds85/bgcbench/manifest.json")
    if not man.exists():
        return
    bound = json.loads(man.read_text())["_build"]["max_len"]
    assert bound <= MODEL_USABLE_CONTEXT, (
        f"corpus bound {bound} exceeds the model's usable context "
        f"{MODEL_USABLE_CONTEXT}: records are being trained on past the point where the "
        f"model degrades toward chance, and the long classes carry the most of it")


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
    from bgcbench.model.genconfig import FROZEN as F
    a = gc.realised(n_per_row=F["n_per_row"], budget_nt=F["budget_nt"],
                    seed_len_nt=F["seed_len_nt"])
    b = gc.realised(n_per_row=150, budget_nt=F["budget_nt"], seed_len_nt=F["seed_len_nt"])
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


def test_budget_respects_the_model_usable_context():
    """MEASURED on evo2-1b (config max_seqlen 8192), NLL of the last 1000 tokens of a
    prefix: 8,192 -> 0.805 (best), 10,000 -> 0.851, 12,000 -> 1.040, 15,900 -> 1.239,
    against ln(4) = 1.386 chance. A 16,000 budget had every arm generating kilobases of
    near-random sequence, and the long classes would have looked worst because their
    references are longest."""
    from bgcbench.model.genconfig import FROZEN
    assert FROZEN["budget_nt"] <= 8192


def test_budget_below_the_prodigal_mode_switch():
    """antiSMASH switches prodigal gene-calling mode above 20,000 nt per sequence; arms on
    either side would be scored by different callers."""
    from bgcbench.model.genconfig import FROZEN
    from bgcbench.run.arm import PRODIGAL_MODE_SWITCH_NT
    assert FROZEN["budget_nt"] < PRODIGAL_MODE_SWITCH_NT


def test_ragged_seeds_are_refused_not_silently_shortened():
    """vortex batches only when all prompts are the same length, so a short seed changes
    the decode path -- and the change correlates with the confusion row."""
    from bgcbench.model.generate import _seed_text, usable_seed_pool
    pool = [{"accession": "a", "sequence": "ACGT" * 10, "seq_len": 40},
            {"accession": "b", "sequence": "AC", "seq_len": 2}]
    ok = usable_seed_pool(pool, 8)
    assert [r["accession"] for r in ok] == ["a"], "short record must be filtered out"
    assert _seed_text(pool[0], 8) == "ACGTACGT"
    try:
        _seed_text(pool[1], 8)
    except ValueError:
        return
    raise AssertionError("a short record silently produced a ragged prompt")


def test_hf_generation_pads_left():
    """A decoder-only model conditions on the token before the first generated position;
    right padding makes that [PAD] for every row that is not the longest in its batch."""
    from bgcbench.model import generate as gen
    src = Path(gen.__file__).read_text()
    assert 'padding_side = "left"' in src


def test_adapter_arm_requires_a_class_coordinate():
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert "refusing to run: an adapter arm needs either --row-class" in src


def test_generation_batch_size_is_frozen_not_merely_defaulted():
    """The batch size is part of the frozen generation config, so changing it changes the
    run hash and therefore which runs are comparable to which. It used to be a plain CLI
    default, which meant a single flag could silently move a run into a different
    comparability class -- the same failure mode as the n=150 shell script that motivated
    freezing generation in the first place.

    It is sized from a measurement, not chosen: 2.08 GB model + CUDA context, plus 0.429 GB
    per in-flight sequence at the 8,192 nt budget. n=200 in one batch needs 87.9 GB and
    does not fit an 80 GB card; 100 is the largest exact divisor of 200 that does.
    """
    from bgcbench.model.genconfig import FROZEN
    assert FROZEN["batch_size"] == 100, "the frozen batch size moved"
    assert FROZEN["n_per_row"] % FROZEN["batch_size"] == 0, \
        "batch size does not divide n, so the last batch is ragged"

    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert 'ap.add_argument("--batch-size", type=int, default=None' in src, \
        "batch size is a plain CLI default again, so it can be overridden silently"
    assert "--off-frozen" in src, "no way to declare a deliberate departure"
    assert "refusing to run: --batch-size" in src, \
        "a batch size differing from frozen is not refused"


def test_frozen_batch_fits_the_card_it_was_sized_for():
    """A frozen batch size that does not fit is a run that dies an hour in. The sizing
    model is recorded next to the value so it can be rechecked on other hardware."""
    from bgcbench.model.genconfig import FROZEN
    BASELINE_GB, PER_SEQ_GB, CARD_GB = 2.08, 0.429, 80.0
    need = BASELINE_GB + PER_SEQ_GB * FROZEN["batch_size"]
    assert need < CARD_GB * 0.85, f"frozen batch needs {need:.1f} GB of a {CARD_GB} GB card"
    # and the value is the largest exact divisor of n that fits
    bigger = [b for b in range(FROZEN["batch_size"] + 1, FROZEN["n_per_row"] + 1)
              if FROZEN["n_per_row"] % b == 0 and BASELINE_GB + PER_SEQ_GB * b < CARD_GB * 0.85]
    assert not bigger, f"a larger divisor would also fit: {bigger}"


def _stub_vortex_tokenizer():
    """A stand-in with vortex CharLevelTokenizer's exact decode semantics."""
    class T:
        eos_id = 0
        pad_id = 1
        vocab_size = 512
        def clamp(self, n): return max(32, min(int(n), self.vocab_size))
        def decode_token(self, t): return chr(self.clamp(t))
        def tokenize(self, s): return list(s.encode())
        def detokenize(self, ids): return "".join(self.decode_token(i) for i in ids)
    return T()


def test_evo2_terminator_survives_decoding_and_is_distinguishable():
    """The terminator was searched for as chr(0) in the DECODED string, but vortex decodes
    with chr(max(32, min(id, vocab))) — ids 0 (EOS), 1 (PAD) and 32 (space) all render as a
    space. So `truncate_at_terminator` could never fire and `hit_eos` was 0.0 in all 13
    Stage 1 run reports: a structural zero of the METRIC, not a property of the model.

    Measured once the shim was installed: Evo2 emits its stop token after TWO nucleotides
    from the de novo prompt — 56% of base generations, 91% of fine-tuned. Every frozen
    Stage 1 sequence was ~8,190 nt of post-termination sampling.
    """
    from bgcbench.model.load import EVO2_TERMINATOR_SENTINEL, _install_terminator_shim

    tok = _stub_vortex_tokenizer()
    # the defect, reproduced: without the shim the terminator is invisible
    assert tok.detokenize([0]) == tok.detokenize([1]) == tok.detokenize([32]) == " "
    assert chr(0) not in tok.detokenize(tok.tokenize("ACGT") + [0])

    assert _install_terminator_shim(tok, 0, EVO2_TERMINATOR_SENTINEL)
    decoded = tok.detokenize(tok.tokenize("ACGT") + [0])
    assert decoded == "ACGT" + EVO2_TERMINATOR_SENTINEL, decoded
    assert tok.detokenize([0]) != tok.detokenize([1]), "EOS still collides with PAD"
    assert tok.detokenize([0]) != tok.detokenize([32]), "EOS still collides with space"
    assert EVO2_TERMINATOR_SENTINEL not in "ACGTN", "sentinel collides with the alphabet"


def test_training_text_appends_the_REAL_terminator_not_the_display_sentinel():
    """`terminator_str` is how the terminator LOOKS after decoding (a sentinel for Evo2).
    Training text must append the form that ENCODES to the terminator id. Appending the
    sentinel instead would teach the model to emit '*' (id 42) rather than its own stop
    token (id 0) — a bug introduced while fixing the one above, and caught by G10's T1."""
    from bgcbench.model.load import Substrate

    sub = Substrate(id="x", family="evo2", checkpoint="c", terminator_id=0,
                    terminator_str="*", terminator_encode_str=chr(0),
                    appends_terminator=False, native_stop=False, approx_nt_per_token=1.0,
                    tokenizer=_stub_vortex_tokenizer())
    ids = sub.tokenizer.tokenize(sub.training_text("ACGT"))
    assert ids[-1] == sub.terminator_id, f"training text ends in {ids[-1]}, not the terminator"
    assert "*" not in sub.training_text("ACGT"), "the display sentinel leaked into training"


def test_clean_masks_rather_than_deletes_so_the_reading_frame_survives():
    """Deletion shifts the frame by one base and destroys every downstream ORF, and ORFs are
    what the scoring stack is built on. Measured on the frozen bundle: 1,422 of 1,600
    generations (88.9%) lost at least one character, mean 3.13 — and every one of those
    characters was the model's own terminator."""
    from bgcbench.model.load import Substrate

    dirty = "AC*GT NNAC"
    out = Substrate.clean(dirty)
    assert len(out) == len(dirty), "clean() changed the length — the frame is shifted"
    assert set(out) <= set("ACGTN"), out
    assert out == "ACNGTNNNAC", out


def test_terminator_suppression_restores_the_sampler_it_patched():
    """The floor is applied by swapping the module-level `sample` vortex imported. A leaked
    patch would silently alter every later arm in the same process."""
    import vortex.model.generation as vg
    from bgcbench.model.generate import suppress_terminator
    from bgcbench.model.load import Substrate

    sub = Substrate(id="x", family="evo2", checkpoint="c", terminator_id=0,
                    terminator_str="*", appends_terminator=False, native_stop=False,
                    approx_nt_per_token=1.0)
    before = vg.sample
    try:
        with suppress_terminator(sub, 10):
            assert vg.sample is not before, "suppression never took effect"
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert vg.sample is before, "sampler patch leaked past the context"


def test_min_new_tokens_is_frozen_and_below_every_class_median():
    """Without a floor an unconditioned arm terminates at 2 nt and produces nothing to
    score. The floor is one number, identical for every arm and class, injecting no class
    information — a decoding policy, not a conditioning channel."""
    from bgcbench.model.genconfig import FROZEN
    assert FROZEN["min_new_tokens"] >= 500, "floor too low to clear the 2 nt collapse"
    # must not dictate cluster length: it sits below every benchmark class's median core
    assert FROZEN["min_new_tokens"] < 1154, "floor exceeds TERPENE's median core"


def test_evo2_generation_buckets_ragged_prompts_and_preserves_order():
    """vortex batches only when every prompt in a call shares a length; otherwise it silently
    generates ONE AT A TIME. Measured twice on real GTDB lineages: 42 prompts carry 19-21
    distinct lengths, turning one batch into 19-21 near-sequential calls -- hours at ~40% GPU.

    The order check is the load-bearing one: results must scatter back to their original
    positions, or seed accession, seed length and the confusion-matrix row all attach to the
    wrong sequence."""
    from bgcbench.model.generate import _run_evo2, ArmSpec, GenConfig

    calls = []

    class FakeModel:
        def generate(self, prompt_seqs, **kw):
            assert len({len(p) for p in prompt_seqs}) == 1, \
                f"ragged batch submitted: {sorted({len(p) for p in prompt_seqs})}"
            calls.append(list(prompt_seqs))
            # echo the prompt back so order can be verified downstream
            return ([p + "ACGT" for p in prompt_seqs],)

    class FakeSub:
        family = "evo2"
        model = FakeModel()
        terminator_id = 0
        def truncate_at_terminator(self, t): return t, False
        @staticmethod
        def clean(t): return t

    prompts = ["A", "AB", "ABC", "AB", "A", "ABCD", "ABC", "A"]
    arm = ArmSpec(arm_id="t", weight_state="base")
    cfg = GenConfig(budget_nt=8, batch_size=100, seed=0, min_new_tokens=0)
    texts, hits = _run_evo2(FakeSub(), arm, prompts, cfg)

    assert all(len({len(p) for p in c}) == 1 for c in calls), "a ragged batch got through"
    assert len(calls) == 4, f"expected one call per distinct length, got {len(calls)}"
    # ORDER: result i must correspond to prompt i
    assert texts == [p + "ACGT" for p in prompts], f"order not preserved: {texts}"
    assert len(hits) == len(prompts)


def test_taxonomy_prefix_is_not_a_class_channel():
    """A phylogeny prefix names an ORGANISM, never a compound class, so it must not make an
    arm class-bearing and must not vary with the row being filled.

    ⚠ The bug this pins: an earlier version set class_bearing whenever a prefix was present.
    That broke SPEC 6.0's degenerate collapse -- an arm carrying the class NOWHERE, neither
    in its weights nor its input, must fill every row from ONE distribution -- and it
    quadrupled the cost of every pooled arm.

    Taxa do correlate with the BGC classes they carry, and that is deliberately not treated
    as a conditioning channel: it is a property of the organism distribution, not a label
    handed to the model.
    """
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()

    cb = [l for l in src.splitlines() if l.strip().startswith("class_bearing =")]
    assert len(cb) == 1, f"expected one class_bearing assignment, found {len(cb)}"
    assert "prefix" not in cb[0], \
        f"a taxonomy prefix still makes an arm class-bearing: {cb[0].strip()}"

    # the pool must be built OUTSIDE the per-target loop, from the arm's training classes
    pool = src.index("tax_pool = None")
    loop = src.index("for cls in targets:")
    assert pool < loop, "the lineage pool is built inside the target loop — it varies by row"
    seg = src[pool:loop]
    assert "row_class" in seg and "train_classes" in seg, \
        "the pool does not follow the arm's training data"


def test_hits_ledger_fields_exist_in_the_scored_record():
    """The hits ledger is the only place a positive detection is tied back to the organism
    whose lineage prompted it -- the per-generation record is where the lineage survives.
    A typo'd field name would silently write nulls for every hit, which looks exactly like
    'no lineage was used'."""
    from bgcbench.score.record import REQUIRED
    from bgcbench.run import arm as armmod
    import re

    src = Path(armmod.__file__).read_text()
    block = src[src.index("# ---- HITS LEDGER"):src.index('with open(d / "hits.jsonl"')]
    pulled = set(re.findall(r'r\.get\("([a-z_]+)"\)', block))
    # ⚠ EVERY field the ledger pulls must be in REQUIRED. An earlier version of this test
    # whitelisted the prefix fields as "extra", so it PASSED while record.build() silently
    # dropped them and every hit recorded a null lineage -- indistinguishable from "no
    # lineage was used". A whitelist here defeats the only check that catches that.
    # Check against what build() ACTUALLY WRITES, not against a whitelist. A whitelist is
    # what let this test pass while the prefix fields were being dropped.
    import re as _re
    from bgcbench.score import record as recmod
    rsrc = Path(recmod.__file__).read_text()
    body = rsrc[rsrc.index("def build("):]
    written = set(_re.findall(r'^\s+"([a-z_]+)":', body, _re.M))
    unknown = pulled - set(REQUIRED) - written
    assert not unknown, f"hits ledger reads fields the scored record never writes: {unknown}"
    for must in ("prefix_tag", "prefix_source_genome", "prefix_source_accession"):
        assert must in REQUIRED, \
            f"{must} is not in the scored record schema, so the ledger will write null"
    for must in ("prefix_tag", "prefix_source_genome", "on_target", "products"):
        assert must in block, f"hits ledger does not record {must}"
