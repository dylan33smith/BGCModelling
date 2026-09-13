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


def test_seed_length_is_the_value_gate_g3_selected():
    """SPEC G3 set the seeded prompt length. 64 is chosen as THE LONGEST UNCONFOUNDED RUNG,
    not the highest-scoring one: at L>=128 the seeds themselves become antiSMASH-detectable
    (seed-only baseline 0.050 at 128, 0.165 at 256, 0.412 at 512), and at 512 that baseline
    EXCEEDS the generation rate. A higher value buys a bigger number and a weaker claim.

    Below 32 nt the rung is a null — a core's 5' end is a start codon plus noise — so the
    previously frozen 8 was measuring almost nothing (0.001 on-target)."""
    from bgcbench.model.genconfig import FROZEN
    assert FROZEN["seed_len_nt"] == 64, (
        "seed length moved off the G3-selected value; if this is deliberate, G3 must be "
        "re-read and FINDINGS 9 updated with the new contamination boundary")
    # the frozen value must sit inside the uncontaminated window G3 measured
    assert 32 <= FROZEN["seed_len_nt"] <= 64, "outside the window G3 established as clean"


def test_i1_arm_is_class_bearing():
    """A derived direction is per target class, so an I1 arm carries the class at
    generation time and must fill each confusion row from its own distribution.

    ⚠ The bug this pins: `ArmSpec` was constructed without `inference_control`, so it took
    the "none" default and `class_bearing` stayed False for every I1 arm. That routes a
    class-conditional arm through SPEC 6.0's degenerate collapse, which fills all four rows
    from ONE sample -- reporting a steering arm as though the class never entered.
    """
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    spec = src[src.index("arm = ArmSpec("):src.index("class_bearing =")]
    assert "inference_control=" in spec, \
        "ArmSpec is built without inference_control; every I1 arm reads as class-free"
    assert "args.direction" in spec, \
        "inference_control does not follow --direction, so I1 never sets it"


def test_i1_alpha_and_kind_enter_the_realised_identity():
    """I1 at two alphas is two arms, and an I1 arm is not a W3 arm.

    ⚠ Two bugs this pins, both silent. `intervention_method` was the literal "offset", so
    an I1 run was recorded as W3 and the two were indistinguishable in the frozen record.
    And alpha was absent from the realised fields, so alpha=1 and alpha=4 hashed to the
    SAME run directory and the second run destroyed the first -- exactly the collision the
    site and rank fields already exist to prevent.
    """
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    seg = src[src.index("intervention_method="):src.index("classmap_hash=")]
    assert '"offset" if intervention' not in seg, \
        "intervention_method is hardcoded to offset; I1 arms record as W3"
    assert "intervention_kind" in seg, "the method does not follow the attached kind"
    assert "intervention_alpha=" in seg, \
        "alpha is not a realised field; two alphas collide on one run directory"


def test_direction_injection_survives_the_realised_report():
    """`DirectionInjection` has no trainable parameters and no `n_trainable()`; the report
    called it unconditionally, which raised AttributeError for every I1 arm before a single
    sequence was scored."""
    from bgcbench.model.interventions import DirectionInjection
    assert not hasattr(DirectionInjection, "n_trainable")
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    seg = src[src.index("intervention_trainable="):src.index("classmap_hash=")]
    assert "hasattr" in seg, \
        "n_trainable() is called unconditionally; I1 arms crash building their report"


class _Sub:
    """Minimal substrate stand-in. `derive()` stubs out `class_means` in these tests but
    still calls `site_report(sub.model.model)`, so the inner model must expose real
    `inner_mha_cls` modules."""

    family = "evo2"

    def __init__(self, hidden=2, n_sites=2):
        class _Outer:
            pass
        self.model = _Outer()
        self.model.model = _fake_attention_model(hidden=hidden, n_sites=n_sites)

def test_i1_directions_are_unit_norm_so_the_random_control_is_matched():
    """SPEC 6.3's control is MAGNITUDE-MATCHED: `random_direction_control` normalises to
    unit norm, so if `derive()` did not, the two arms would differ in magnitude as well as
    in content and "I1 beat random" could mean only "I1 pushed harder".

    ⚠ THIS TEST USED TO GREP THE SOURCE for `raw / norms.unsqueeze(-1)` and was worthless:
    the audit showed it passes with the normalisation deleted, because the same expression
    also appears in `manipulation_check`. It now calls `derive()` with the activation means
    stubbed, so it fails if and only if the arithmetic changes.
    """
    import torch

    from bgcbench.model import directions as D

    calls = []

    def fake_means(sub, records, prefix_kind, max_len_nt, device="cuda:0", limit=None):
        calls.append(records)
        return (torch.tensor([[3.0, 4.0], [0.0, 5.0]]) if records == "T"
                else torch.tensor([[0.0, 0.0], [0.0, 2.0]])), len(records)

    real = D.class_means
    D.class_means = fake_means
    try:
        art = D.derive(_Sub(), "T", "O", "none", 8192)
    finally:
        D.class_means = real

    d = art["directions"]
    assert torch.allclose(d.norm(dim=-1), torch.ones(2), atol=1e-6), \
        f"directions are not unit-norm ({d.norm(dim=-1).tolist()}); the random control is "
    assert art["raw_norms"] == [5.0, 3.0], \
        f"raw norms {art['raw_norms']} != the norms of (target - other); the contrast is wrong"
    # target-minus-others, not target-minus-zero: site 0 is (3,4)-(0,0) -> (0.6,0.8)
    assert torch.allclose(d[0], torch.tensor([0.6, 0.8]), atol=1e-6)
    assert torch.allclose(d[1], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert calls == ["T", "O"], "derive() did not take target first, others second"


def test_i1_refuses_a_zero_direction_rather_than_normalising_to_nan():
    """Identical class means give a zero difference; dividing by its norm yields NaN, which
    would propagate silently through every generation as a direction of not-a-number."""
    import torch

    from bgcbench.model import directions as D

    def same(sub, records, prefix_kind, max_len_nt, device="cuda:0", limit=None):
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 4

    real = D.class_means
    D.class_means = same
    try:
        D.derive(_Sub(), "T", "O", "none", 8192)
    except RuntimeError:
        return
    finally:
        D.class_means = real
    raise AssertionError("a zero-norm direction was normalised instead of refused")




def _fake_attention_model(hidden=8, n_sites=3, tuple_output=False):
    """A minimal stand-in exposing modules named `inner_mha_cls`, which is what
    interventions.attention_sites() enumerates. Lets the I1 hook path be exercised without
    a GPU or a 1B-parameter model."""
    import torch.nn as nn

    class Site(nn.Module):
        def __init__(self, h, tup):
            super().__init__()
            self.lin = nn.Linear(h, h, bias=False)
            self.tup = tup

        def forward(self, x):
            y = self.lin(x)
            return (y, None) if self.tup else y

    class Block(nn.Module):
        def __init__(self, h, tup):
            super().__init__()
            self.inner_mha_cls = Site(h, tup)

        def forward(self, x):
            o = self.inner_mha_cls(x)
            return o[0] if isinstance(o, tuple) else o

    class Model(nn.Module):
        def __init__(self, h, n, tup):
            super().__init__()
            self.blocks = nn.ModuleList([Block(h, tup) for _ in range(n)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    return Model(hidden, n_sites, tuple_output)


def test_i1_hook_actually_adds_alpha_times_the_direction():
    """The I1 mechanism, exercised rather than asserted about: attaching a
    DirectionInjection must shift every attention site's output by exactly alpha*direction,
    and detaching must restore the model bit-for-bit.

    ⚠ A hook that silently does nothing is this arm's worst failure mode -- it reads as a
    clean null. Nothing in a source-text assertion can catch it.
    """
    import torch

    from bgcbench.model.interventions import DirectionInjection

    for tup in (False, True):          # sites may return a tensor or a tuple
        torch.manual_seed(0)
        m = _fake_attention_model(hidden=8, n_sites=3, tuple_output=tup)
        x = torch.randn(2, 5, 8)
        base_out = m(x).clone()

        d = torch.zeros(3, 8)
        d[:, 0] = 1.0                  # unit vectors along axis 0
        iv = DirectionInjection(m, 8, d, alpha=2.0)

        with iv.attached():
            steered = m(x).clone()
        assert not torch.allclose(base_out, steered), (
            f"attaching the intervention changed nothing (tuple_output={tup}); the arm "
            f"would generate from the base model and report a null")

        after = m(x)
        assert torch.allclose(base_out, after), \
            "hooks leaked past attached(); every later arm in the process is contaminated"

        # a single-site model isolates the injection, so the shift is exactly checkable
        m1 = _fake_attention_model(hidden=8, n_sites=1, tuple_output=tup)
        iv1 = DirectionInjection(m1, 8, d[:1], alpha=2.0)
        want = m1(x) + 2.0 * d[0]
        with iv1.attached():
            got = m1(x)
        assert torch.allclose(got, want, atol=1e-5), \
            f"injected shift is not alpha*direction (tuple_output={tup})"


def test_i1_alpha_zero_is_exactly_the_base_model():
    """alpha=0 must be an exact no-op. If it is not, alpha does not control the magnitude
    and the G9 sweep is measuring something else."""
    import torch

    from bgcbench.model.interventions import DirectionInjection
    torch.manual_seed(0)
    m = _fake_attention_model()
    x = torch.randn(2, 5, 8)
    want = m(x).clone()
    iv = DirectionInjection(m, 8, torch.randn(3, 8), alpha=0.0)
    with iv.attached():
        got = m(x)
    assert torch.allclose(want, got), "alpha=0 is not a no-op"


def test_i1_refuses_a_direction_count_that_does_not_match_the_sites():
    """Steering some layers and not others, silently, would be an arm nobody specified."""
    import torch

    from bgcbench.model.interventions import DirectionInjection
    m = _fake_attention_model(n_sites=3)
    try:
        DirectionInjection(m, 8, torch.randn(2, 8), alpha=1.0)
    except ValueError:
        return
    raise AssertionError("a 2-direction tensor was accepted for 3 attention sites")


def test_mean_collector_averages_over_tokens_and_matches_a_hand_computation():
    """_MeanCollector is where a shape or dtype error would silently corrupt every
    direction. Checked against an explicit mean over (batch x position)."""
    import torch

    from bgcbench.model.directions import _MeanCollector
    torch.manual_seed(0)
    for tup in (False, True):
        m = _fake_attention_model(hidden=8, n_sites=2, tuple_output=tup)
        xs = [torch.randn(1, 4, 8), torch.randn(1, 7, 8)]

        seen = []
        h = m.blocks[0].inner_mha_cls.register_forward_hook(
            lambda _m, _a, o: seen.append(o[0] if isinstance(o, tuple) else o))
        for x in xs:
            m(x)
        h.remove()
        # PER RECORD: each record's own mean, then averaged with equal weight -- not one
        # pooled mean over every token, which would weight the 7-token record more.
        want = torch.stack([s.reshape(-1, 8).mean(0) for s in seen]).mean(0)
        pooled = torch.cat([s.reshape(-1, 8) for s in seen], 0).mean(0)
        assert not torch.allclose(want, pooled, atol=1e-6), \
            "the fixture cannot distinguish record- from token-weighting"

        col = _MeanCollector(m)
        with col.attached():
            for x in xs:
                m(x)
        got = col.means()[0]
        assert got.shape == (8,), f"site mean has shape {got.shape}, expected (8,)"
        assert torch.allclose(got, want, atol=1e-5), \
            f"streaming mean != explicit mean over tokens (tuple_output={tup})"
        # ⚠ RECORDS, not tokens. Token-weighting would let class length differences leak
        # into the direction (FINDINGS 12: the classes differ systematically in length) and
        # would be the raw-mixture statistic SPEC 4.4.3's equal-n-by-record rejects.
        assert col.counts[0] == 2, "the mean is not weighted per record"
        assert col.token_counts[0] == 11, "the token denominator is no longer reportable"


def test_direction_derivation_is_unit_norm_and_is_target_minus_others():
    """Exercises the arithmetic derive() performs, without needing a substrate."""
    import torch

    mt = torch.tensor([[3.0, 4.0], [0.0, 5.0]])       # per-site target means
    mo = torch.tensor([[0.0, 0.0], [0.0, 2.0]])       # per-site other-class means
    raw = mt - mo
    norms = raw.norm(dim=-1)
    unit = raw / norms.unsqueeze(-1)
    assert torch.allclose(norms, torch.tensor([5.0, 3.0]))
    assert torch.allclose(unit.norm(dim=-1), torch.ones(2)), \
        "directions are not unit-norm, so SPEC 6.3's random control is not magnitude-matched"
    # and the contrast must be a DIFFERENCE: identical means give a zero direction, which
    # derive() must refuse rather than normalise into NaN
    from bgcbench.model import directions as D
    src = Path(D.__file__).read_text()
    assert "zero norm" in src, "a zero-norm direction is not refused; normalising gives NaN"


def _stub_check(cosines, zero_sites=(), alpha=1.0):
    """Drive the REAL `manipulation_check` with controlled activations.

    `class_means` is stubbed so the validation readout at every site resolves to axis 0, and
    the train direction is built so each site's train/val cosine is exactly `cosines[i]`. A
    zeroed site gets a genuinely zero vector, as `derive()` produces. Everything after that
    -- active-site selection, aggregation, the pass rule -- is the real function.
    """
    import math

    import torch

    from bgcbench.model import directions as D

    n, H = len(cosines), 4
    d_tr = torch.zeros(n, H)
    for i, c in enumerate(cosines):
        if i in zero_sites:
            continue
        c = max(-1.0, min(1.0, float(c)))
        d_tr[i][0] = c
        d_tr[i][1] = math.sqrt(max(0.0, 1 - c * c))

    # manipulation_check calls class_means four times: mt, mo, then the readout BEFORE
    # injection and the readout AFTER. The stub bypasses the model, so hooks never fire --
    # it has to supply the post-injection shift itself, or `shift[0]` stays 0 and the pass
    # rule fails for a reason that has nothing to do with the aggregation under test.
    calls = {"n": 0}

    def fake_means(sub, records, prefix_kind, max_len_nt, device="cuda:0", limit=None):
        m = torch.zeros(n, H)
        m[:, 0] = 1.0 if records != "O" else -1.0     # target - other = +axis0
        if calls["n"] >= 3:                            # the 4th call is the steered readout
            for i, c in enumerate(cosines):
                if i not in zero_sites:
                    m[i][0] += alpha * float(c)        # what injecting alpha*d_tr would do
        calls["n"] += 1
        return m, 8

    real = D.class_means
    D.class_means = fake_means
    try:
        return D.manipulation_check(_Sub(hidden=H, n_sites=n), "T", "O", d_tr,
                                    "none", 8192, alpha=alpha)
    finally:
        D.class_means = real


def test_manipulation_check_averages_only_the_sites_it_steers():
    """A degenerate site is zeroed by `derive()`, so its train/val cosine is identically 0.
    That is an EXCLUSION, not a measurement, and averaging it in penalises the arm for a site
    it deliberately does not touch.

    ⚠ Measured on real artifacts: ARYLPOLYENE at 192 records per side gives cosines
    [0.118, 0.186, 0.616, 0.0] with site 3 zeroed. Over all four sites the mean is 0.230 and
    the check FAILS; over the three steered sites it is 0.307 and it PASSES -- a verdict flip
    on identical data, caused purely by the aggregation.

    ⚠ An earlier version of this test GREPPED directions.py for the implementation line and
    did arithmetic on a literal list, so it never called the function and could not catch a
    regression in it -- the same defect the audit found in test_i1_directions_are_unit_norm.
    This one drives the real function.
    """
    chk = _stub_check([0.118, 0.186, 0.616, 0.0], zero_sites=(3,))
    assert chk["active_sites_checked"] == [0, 1, 2], \
        f"steered sites resolved to {chk['active_sites_checked']}, expected [0, 1, 2]"
    assert abs(chk["mean_cosine"] - 0.307) < 0.005, \
        f"mean over steered sites is {chk['mean_cosine']:.4f}, expected ~0.307"
    assert abs(chk["mean_cosine_all_sites"] - 0.230) < 0.005, \
        "the all-sites mean is not retained, so the excluded scale is unreportable"
    assert chk["passes"] is True, \
        "the zeroed site still drags the mean below threshold; the verdict flip is back"


def test_manipulation_check_requires_no_anti_aligned_site():
    """A mean alone can be carried by one strongly reproducing site while another points the
    WRONG way. ARYLPOLYENE at 64 records per side has steered cosines
    [-0.18, 0.038, 0.616] -- mean 0.156 and site 0 anti-aligned -- and must fail."""
    # ⚠ THE FIXTURE MUST ISOLATE THE CONDITION. The real ARYLPOLYENE n=64 cosines
    # [-0.18, 0.038, 0.616] have a mean of 0.158, already under the 0.3 threshold -- so they
    # fail whether or not the anti-alignment rule exists, and a test built on them cannot
    # catch its removal. (Verified: deleting the rule left that version passing.) This
    # fixture has a mean WELL ABOVE threshold and one site pointing the wrong way, so only
    # the anti-alignment rule can reject it.
    strong_but_anti = _stub_check([-0.20, 0.90, 0.90, 0.0], zero_sites=(3,))
    assert strong_but_anti["mean_cosine"] > 0.3, \
        "the fixture no longer clears the mean threshold, so it cannot isolate the rule"
    assert strong_but_anti["min_active_cosine"] < 0, "the anti-aligned site was not detected"
    assert strong_but_anti["passes"] is False, \
        "a direction with a mean above threshold and an ANTI-ALIGNED site passed"

    # the real ARYLPOLYENE n=64 case fails too, on both counts at once
    chk = _stub_check([-0.18, 0.038, 0.616, 0.0], zero_sites=(3,))
    assert chk["min_active_cosine"] < 0 and chk["passes"] is False

    weak = _stub_check([0.1, 0.1, 0.1, 0.0], zero_sites=(3,))
    assert weak["min_active_cosine"] > 0 and weak["passes"] is False, \
        "a uniformly weak but positive direction passed"

    # and a strong all-positive direction passes, so these are not vacuously failing
    good = _stub_check([0.9, 0.8, 0.7, 0.0], zero_sites=(3,))
    assert good["passes"] is True, "a strong reproducing direction was rejected"


def test_hf_generation_applies_the_min_token_floor_in_NUCLEOTIDES():
    """Evo2 gets a floor of `min_new_tokens` via `suppress_terminator`, which returns early
    for every other family -- so the HF path had NO floor at all.

    ⚠ Two defects in one. GenomeOcean terminates natively and eagerly (the prior project
    measured EOS straight after the seed, 61/200 empty generations), so with no floor a
    cross-substrate comparison would score GO on truncated output and blame the substrate.
    And the floor is 1,000 NUCLEOTIDES: passing 1,000 TOKENS to a BPE model at ~4.8 nt/token
    would demand ~4,800 nt, 4.8x the sequence Evo2 must produce. Neither is the same floor.
    """
    from bgcbench.model import generate as G
    src = Path(G.__file__).read_text()
    body = src[src.index("def _run_hf("):]
    assert "min_new_tokens=min_new" in body, \
        "the HF path does not pass a min-token floor; GO can stop immediately after the seed"
    assert "cfg.min_new_tokens / sub.approx_nt_per_token" in body, \
        "the floor is not converted from nucleotides to tokens per substrate"

    # and the conversion itself
    from bgcbench.model.genconfig import FROZEN
    nt = FROZEN["min_new_tokens"]
    assert int(nt / 1.0) == 1000, "byte-level substrate should keep a 1,000-token floor"
    assert 180 < int(nt / 4.8) < 230, \
        f"BPE substrate floor {int(nt / 4.8)} tokens is not ~1,000 nt at 4.8 nt/token"


def test_attention_sites_finds_both_substrate_families():
    """Evo2 names its attention module `inner_mha_cls`, GenomeOcean names it `self_attn`.
    A hardcoded suffix silently returns ZERO sites on the other family -- and `_MeanCollector`
    would then raise, but only after a 4B-parameter model load."""
    import torch.nn as nn

    from bgcbench.model.interventions import ATTENTION_SUFFIXES, attention_sites

    assert "inner_mha_cls" in ATTENTION_SUFFIXES and "self_attn" in ATTENTION_SUFFIXES

    class GoLayer(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.self_attn = nn.Linear(h, h)

    class GoModel(nn.Module):
        def __init__(self, h, n):
            super().__init__()
            self.layers = nn.ModuleList([GoLayer(h) for _ in range(n)])

    m = GoModel(8, 24)
    sites = attention_sites(m)
    assert len(sites) == 24, f"found {len(sites)} GO attention sites, expected 24"
    assert all(n.endswith("self_attn") for n, _ in sites)

    # subset selection, which is how a cross-substrate arm is depth-matched
    sub = attention_sites(m, subset=[3, 10, 16, 23])
    assert len(sub) == 4 and sub[0][0].endswith("layers.3.self_attn")
    try:
        attention_sites(m, subset=[99])
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-range site index was accepted")


def test_matched_depth_subset_puts_GO_at_evo2s_relative_depths():
    """Evo2's 4 attention blocks sit at 3/10/17/24 of 25 — fractional depths 0.12–0.96.
    Matching GO on COUNT rather than depth would make the arms differ in how hard the model
    is pushed as well as in what it is; matching on depth keeps the contrast about the
    substrate. The realised site list is recorded either way (SPEC 6.5)."""
    from bgcbench.model.interventions import matched_depth_subset
    got = matched_depth_subset(24)
    assert got == [3, 10, 16, 23], f"GO matched depths {got}, expected [3, 10, 16, 23]"
    assert len(matched_depth_subset(25)) == 4
    # never out of range at either extreme
    for n in (4, 5, 24, 25, 48):
        s = matched_depth_subset(n)
        assert all(0 <= i < n for i in s), f"n={n} produced {s}"


def test_lora_targets_are_per_substrate():
    """Evo2's module names match NOTHING in GenomeOcean. Handing them to peft trains an
    adapter over zero modules, which fails after a model load and reads like a config typo
    rather than a substrate mismatch."""
    import re

    from bgcbench.model.train import (GO_LORA_TARGETS, SUBSTRATE_LORA, depth_sets_for,
                                      go_target_regex)
    assert set(SUBSTRATE_LORA) == {"evo2", "genomeocean"}
    assert not (set(GO_LORA_TARGETS) & {"l1", "l2", "l3", "out_filter_dense", "Wqkv"}), \
        "GO targets overlap Evo2's module names, which cannot both be right"

    rx = go_target_regex([3, 10, 16, 23])
    for name, want in (("model.layers.3.self_attn.q_proj", True),
                       ("model.layers.3.mlp.gate_proj", True),
                       ("model.layers.23.self_attn.o_proj", True),
                       ("model.layers.7.mlp.up_proj", False),
                       ("blocks.3.mlp.l1", False)):
        assert bool(re.fullmatch(rx, name)) is want, f"{name} matched {not want}"

    d = depth_sets_for("genomeocean")
    assert len(d["all"]) == 24 and d["late"][-1] == 23
    assert len(depth_sets_for("evo2")["all"]) == 25


def test_unwrap_finds_logits_in_a_huggingface_output_object():
    """vortex returns nested tuples; HuggingFace returns a CausalLMOutputWithPast dataclass
    whose `.logits` is the tensor and which is NOT a tuple.

    ⚠ The tuple-only version fell through to None and raised "could not locate logits in
    model output" on GenomeOcean's FIRST training step — after a 4B-parameter model load.
    Checking `.logits` first also avoids walking `past_key_values`, which is large and full
    of 3-D tensors that are not logits.
    """
    import torch

    from bgcbench.model.train import _unwrap

    want = torch.zeros(2, 5, 7)

    class HFOut:                      # stand-in for CausalLMOutputWithPast
        def __init__(self, logits, past):
            self.logits = logits
            self.past_key_values = past

    # a plausible KV cache: 3-D tensors that must NOT be mistaken for logits
    past = tuple((torch.ones(2, 5, 9), torch.ones(2, 5, 9)) for _ in range(3))
    got = _unwrap(HFOut(want, past))
    assert got is want, "did not return the .logits tensor"
    assert got.shape == (2, 5, 7)

    # the vortex shape still works
    assert _unwrap((None, (torch.zeros(1, 3, 4),))).shape == (1, 3, 4)
    assert _unwrap(torch.zeros(2, 2)) is None          # 2-D is not logits
    assert _unwrap(None) is None


def test_direction_encoding_handles_both_tokenizer_conventions():
    """Evo2's byte-level `tokenize()` returns integer IDS; a HuggingFace tokenizer returns
    STRING pieces. `int("ATG")` raises, and every GenomeOcean direction derivation died on
    its first record.

    ⚠ `train._encode2` already branched on family for exactly this reason — `directions._encode`
    was written from it and dropped the branch, so the divergence was invisible until a second
    substrate ran. Pinned by exercising both conventions.
    """
    from bgcbench.model import directions as D

    class _Tok:
        def __init__(self, mode): self.mode = mode
        def tokenize(self, t):
            return [ord(c) for c in t] if self.mode == "ids" else list(t)
        def __call__(self, t): return {"input_ids": [7, 8, 9]}

    class _S:
        def __init__(self, fam, mode):
            self.family, self.tokenizer = fam, _Tok(mode)
        def training_text(self, seq, prefix=""): return prefix + seq

    rec = {"sequence": "ATGC"}
    assert D._encode(_S("evo2", "ids"), rec, "none", 100) == [65, 84, 71, 67]
    # the HF convention must NOT go through int(); it takes the __call__ path
    assert D._encode(_S("genomeocean", "str"), rec, "none", 100) == [7, 8, 9]

    src = Path(D.__file__).read_text()
    body = src[src.index("def _encode("):src.index("def class_means(")]
    assert 'sub.family == "evo2"' in body, "the family branch is gone; GO will crash again"


def test_bpe_detokenisation_does_not_inject_separators_that_clean_masks_to_N():
    """RED TEST for the bug that killed all eight GenomeOcean Stage 1 generation arms.

    `tokenizer.decode()` on GenomeOcean returns `" ".join(tokens)` -- its fast tokenizer
    has no `backend_tokenizer.decoder`, so HuggingFace falls back to space-joining. Every
    space is then masked to N by `clean()`, at ~4.8 nt/token an N every ~5 bases, so NO
    21-mer is N-free and the novelty gate's k-mer set is empty.

    The assertion is on the k-mer set, not on the string, because the k-mer set is what
    actually failed: the gate raised rather than returning 0.0 (KNOWN_WRONG #3 -- the gate
    fails closed, which is the only reason this surfaced instead of handing antiSMASH
    sequence with every ORF destroyed).
    """
    from bgcbench.model.load import Substrate
    from bgcbench.score.novelty import canonical_kmers

    TOKENS = {10: "ATGCGG", 11: "ATT", 12: "ACAGGCG", 13: "TGAG", 14: "CCA",
              15: "CCGCG", 16: "CCCGG", 17: "CCTTTT", 18: "TATG", 19: "TATTTT",
              20: "TAG", 21: "TAGAG", 22: "ACGGGG", 2: "[SEP]"}

    class _HFLikeTokenizer:
        all_special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]

        def convert_ids_to_tokens(self, ids):
            return [TOKENS[i] for i in ids]

        def decode(self, ids, skip_special_tokens=True):
            # exactly what HuggingFace does with no backend decoder
            return " ".join(TOKENS[i] for i in ids
                            if not (skip_special_tokens and TOKENS[i].startswith("[")))

    sub = Substrate(id="g", family="genomeocean", checkpoint="c", terminator_id=2,
                    terminator_str="", appends_terminator=True, native_stop=True,
                    approx_nt_per_token=4.8, tokenizer=_HFLikeTokenizer())

    ids = list(range(10, 23)) + [2]
    expected = "".join(TOKENS[i] for i in range(10, 23))

    # the old path: every 21-mer straddles a masked separator, so the gate sees nothing
    broken = sub.clean(sub.tokenizer.decode(ids, skip_special_tokens=True))
    assert "N" in broken
    assert canonical_kmers(broken) == set(), (
        "fixture no longer reproduces the bug, so passing proves nothing")

    # the fix: the nucleotides survive intact and the gate has something to measure
    got = sub.detokenize(ids)
    assert got == expected, f"detokenize lost or added characters: {got!r}"
    cleaned = sub.clean(got)
    assert "N" not in cleaned
    assert len(canonical_kmers(cleaned)) == len(expected) - 21 + 1


def test_generation_decode_path_survives_the_real_genomeocean_tokenizer():
    """The fixture above encodes what HuggingFace does; this checks it against the actual
    tokenizer, so the test cannot pass on a mimicry that has drifted from the library."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("pGenomeOcean/GenomeOcean-4B",
                                            trust_remote_code=True)
    except Exception as e:                                   # no cache, no network
        print(f"  SKIP: GenomeOcean tokenizer unavailable: {e}")
        return

    from bgcbench.model.load import Substrate
    sub = Substrate(id="g", family="genomeocean", checkpoint="c",
                    terminator_id=int(tok.convert_tokens_to_ids("[SEP]")),
                    terminator_str="", appends_terminator=True, native_stop=True,
                    approx_nt_per_token=4.8, tokenizer=tok)

    seq = ("ATGCGGATTACAGGCGTGAGCCACCGCGCCCGGCCTTTTTATGTATTTTTAGTAGAGACGGGG"
           "TTTCACCATGTTGGCCAGGCTGGTCTCGAACTCCTGACCTCAGGTGATCCGCCCGCCTCGGC")
    ids = tok(seq)["input_ids"]
    assert sub.detokenize(ids) == seq
    assert sub.clean(sub.detokenize(ids)) == seq
    assert "N" in sub.clean(tok.decode(ids, skip_special_tokens=True)), (
        "the library no longer space-joins; the guard in detokenize may be removable")


def test_freeze_id_convention_is_stable_and_content_addressed():
    """A freeze is the SPEC §10 step-2 artifact: it is what makes the later unblind-and-diff
    evidence rather than rationalisation. Two things must hold, and the second is the one
    that would rot silently.

    1. The id is CONTENT-ADDRESSED -- change any recorded number and the id changes, so a
       bundle cannot be edited after the fact while keeping its name.
    2. The convention MATCHES the bundles already on disk. Evo2's Stage 1 was frozen under
       sha256(body-without-freeze_id, sort_keys)[:16]; if GO's bundles used anything else,
       the two substrates' freezes would not be comparable artifacts and nobody would
       notice, because each would be internally consistent.
    """
    from bgcbench.run.freeze import freeze_id

    body = {"what": "x", "runs": {"a": {"n_detected": 12, "n": 200}}}
    base = freeze_id(body)
    assert len(base) == 16 and all(c in "0123456789abcdef" for c in base)

    # the id must not depend on its own previous value, or re-freezing would drift
    assert freeze_id({**body, "freeze_id": "deadbeefdeadbeef"}) == base
    assert freeze_id({**body, "freeze_id": None}) == base

    # content-addressed: one changed count changes the id
    moved = {"what": "x", "runs": {"a": {"n_detected": 13, "n": 200}}}
    assert freeze_id(moved) != base

    # key ORDER must not change the id, or the same bundle freezes twice under two names
    assert freeze_id({"runs": body["runs"], "what": "x"}) == base

    # and the convention must still reproduce a bundle frozen under it earlier
    import json
    from pathlib import Path
    ref = Path("/data2/ds85/bgcbench/runs/SEEDED_DIAGONAL_FROZEN_f1a5fa95dbdc07a1.json")
    if not ref.exists():
        print("  SKIP: reference bundle not on this filesystem")
        return
    d = json.loads(ref.read_text())
    assert freeze_id(d) == d["freeze_id"] == "f1a5fa95dbdc07a1", (
        "the freeze-id convention has changed; existing bundles are no longer reproducible")
