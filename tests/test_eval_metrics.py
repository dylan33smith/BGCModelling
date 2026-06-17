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


def test_novelty_gate_wired():
    """audit C6/M13: the novelty gate is a first-class metric that FAILS memorized
    sequences in the scored evaluate_bgc summary."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bgc_pipeline.evaluation import check_kmer_novelty, evaluate_bgc, EvalConfig
    assert check_kmer_novelty({"max_containment": 0.99})["pass"] is False   # FAIL_memorized
    assert check_kmer_novelty({"max_containment": 0.30})["pass"] is True    # PASS_novel
    assert check_kmer_novelty({"max_containment": 0.85})["pass"] is None    # WARN
    assert check_kmer_novelty(None)["skipped"] is True                      # unverified, not pass
    # skip every check except kmer_novelty -> the 'novel' question reflects it alone
    cfg = EvalConfig(skip_checks=["coding_sanity", "antismash", "class_markers",
                                  "protein_homology", "module_architecture", "taxon_faithfulness"])
    s = evaluate_bgc("ACGT" * 100, config=cfg, novelty={"max_containment": 0.99})["questions"]
    assert s["novel"] == "FAIL", s
    assert evaluate_bgc("ACGT" * 100, config=cfg)["questions"]["novel"] == "skipped"
    print("PASS: novelty gate (kmer_novelty -> 'novel') wired into evaluate_bgc — memorized => FAIL")


def test_metric1_skips_when_tool_cannot_run():
    """A tool that could not run (rc!=0, no results JSON — e.g. antiSMASH DBs not
    downloaded) must SKIP, not FAIL: a hard FAIL would make every generation look
    domain-empty. A genuine rc==0 'ran but found nothing' still FAILs."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import subprocess as _sp
    import bgc_pipeline.evaluation as ev

    class _Proc:
        def __init__(self, rc): self.returncode = rc; self.stdout = ""; self.stderr = "db missing"

    orig = _sp.run
    try:
        _sp.run = lambda *a, **k: _Proc(1)         # tool present but exits 1, writes no JSON
        r = ev.check_antismash("ACGT" * 50, expected_class="NRPS")
        assert r.get("skipped") is True and "pass" not in r, r
        _sp.run = lambda *a, **k: _Proc(0)         # ran cleanly, still no region found
        r0 = ev.check_antismash("ACGT" * 50, expected_class="NRPS")
        assert r0.get("skipped") is not True and r0.get("pass") is False, r0
        assert r0.get("detected") is False, r0     # clean run, no cluster -> not a BGC
    finally:
        _sp.run = orig
    print("PASS antismash: failed-to-run skips; clean-but-empty -> detected=False/FAIL")


def test_eval_suite_driver_wires_dbs():
    """eval_suite_driver.run_group must thread opt-in DB paths into EvalConfig so
    class_markers/protein_homology can run; absent/nonexistent paths leave them
    unset (self-skip)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import eval_suite_driver as D
    import bgc_pipeline.evaluation as ev
    captured = {}

    # run_group does `from bgc_pipeline.evaluation import evaluate_bgc` at call time,
    # so patching the module attribute before the call is picked up by that import.
    def fake_eval(seq, **kw):
        captured["cfg"] = kw["config"]
        return {"summary": {}, "questions": {}, "checks": {}}

    orig = ev.evaluate_bgc
    try:
        ev.evaluate_bgc = fake_eval
        D.run_group([{"sequence": "ACGT", "compound_class": "NRPS"}], {}, [],
                    pfam_hmm=Path("/nonexistent.hmm"), mmseqs2_db="UniRef50")
        cfg = captured["cfg"]
        assert cfg.pfam_hmm_path is None          # nonexistent path -> stays unset (self-skip)
        assert cfg.mmseqs2_db == "UniRef50"       # string db passed through
    finally:
        ev.evaluate_bgc = orig
    print("PASS: eval_suite_driver wires opt-in DBs (existing only) into EvalConfig")


def test_eval_suite_aggregation():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import eval_suite_driver as D
    results = [
        {"expected_class": "NRPS",
         "questions": {"is_bgc": "PASS", "conditioning_faithful": "FAIL", "novel": "PASS"},
         "checks": {"coding_sanity": "PASS", "class_markers": "PASS"}},
        {"expected_class": "NRPS",
         "questions": {"is_bgc": "PASS", "conditioning_faithful": "PASS", "novel": "FAIL"},
         "checks": {"coding_sanity": "PASS", "class_markers": "FAIL"}},
        {"expected_class": "PKS",
         "questions": {"is_bgc": "skipped", "conditioning_faithful": "PASS", "novel": "no_verdict"},
         "checks": {"coding_sanity": "skipped", "class_markers": "PASS"}},
    ]
    s = D.summarize_group(results)
    assert s["n"] == 3
    pq = s["per_question"]
    assert pq["is_bgc"]["PASS"] == 2 and pq["is_bgc"]["skipped"] == 1
    assert pq["is_bgc"]["pass_rate"] == 1.0                          # 2 PASS / 2 scored
    assert pq["conditioning_faithful"]["pass_rate"] == round(2 / 3, 3)
    assert pq["novel"]["FAIL"] == 1 and pq["novel"]["pass_rate"] == 0.5
    # role tags: is_bgc/novel are gates, conditioning_faithful is a diagnostic
    assert pq["is_bgc"]["role"] == "gate" and pq["novel"]["role"] == "gate"
    assert pq["conditioning_faithful"]["role"] == "diagnostic"
    # per-check tally is independent of questions
    assert s["per_check"]["class_markers"]["PASS"] == 2
    print("PASS: driver aggregation (per-question + per-check pass rates incl. skipped)")


def test_gate_keyed_headline():
    """Headline keys off the 3 gate QUESTIONS (is_bgc, correct_class, novel). The
    diagnostics never gate acceptance."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import eval_suite_driver as D
    from bgc_pipeline.evaluation import GATE_QUESTIONS, DIAGNOSTIC_QUESTIONS
    assert set(GATE_QUESTIONS) == {"is_bgc", "correct_class", "novel"}
    assert set(DIAGNOSTIC_QUESTIONS) == {"proteins_plausible", "complete", "conditioning_faithful"}
    recs = [
        {"questions": {"is_bgc": "PASS", "correct_class": "PASS", "novel": "PASS"}},        # ACCEPT
        {"questions": {"is_bgc": "FAIL", "correct_class": "PASS", "novel": "PASS"}},        # not a BGC
        {"questions": {"is_bgc": "PASS", "correct_class": "no_verdict", "novel": "PASS"}},  # class unknown
        {"questions": {"is_bgc": "PASS", "correct_class": "PASS", "novel": "PASS"}},        # ACCEPT
        {"questions": {"is_bgc": "PASS", "correct_class": "PASS", "novel": "FAIL"}},        # memorized
    ]
    h = D.summarize_group(recs)["headline"]
    assert h["generates_bgc"]["n"] == 4, h               # is_bgc PASS: recs 0,2,3,4
    assert h["correct_class"]["n"] == 4, h               # correct_class PASS: recs 0,1,3,4
    assert h["biological_valid"]["n"] == 3, h            # is_bgc & correct_class: recs 0,3,4
    assert h["biological_valid_and_novel"]["n"] == 2, h  # + novel: recs 0,3 (rec4 novel-FAIL drops)
    assert h["valid_and_novel"]["n"] == 2, h             # legacy alias = accept
    assert D.summarize_group(recs)["per_question"]["is_bgc"]["role"] == "gate"
    print("PASS: headline keys off is_bgc/correct_class/novel gate questions")


def test_coding_sanity():
    """coding_sanity (is_bgc floor): real gene-rich coding DNA passes; empty /
    low-complexity junk fails. Discriminator is the dinucleotide-ENTROPY complexity
    guard — NOT gene-completeness (a legitimate strict core can be one big edge-
    truncated megasynthase that Prodigal flags partial, yet it must still pass)."""
    import random
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bgc_pipeline.evaluation import check_coding_sanity
    # empty -> fail
    assert check_coding_sanity("")["pass"] is False
    # GC-repeat junk (greedy collapse): low dinuc entropy -> fail on complexity
    junk = check_coding_sanity("GC" * 8000)
    assert junk["pass"] is False and junk["dinuc_entropy"] < 2.5, junk
    # one big edge-truncated megasynthase (partial, no complete gene) must still PASS
    rng = random.Random(1)
    codons = ["GCA", "GAA", "AAA", "CTG", "GAT", "CGT", "ACC", "GGT", "TTT", "CAG",
              "AAC", "GTG", "TCT", "ATT", "CAT", "TGG", "TAC", "CCG", "AGC"]
    big = "ATG" + "".join(rng.choice(codons) for _ in range(3000))   # no stop -> partial
    good = check_coding_sanity(big)
    assert good["pass"] is True and good["max_orf_aa"] >= 100 and good["dinuc_entropy"] >= 2.5, good
    print("PASS: coding_sanity (complexity guard fails junk; partial megasynthase passes)")


def test_module_architecture():
    """module_architecture: ordered C-A-T modules pass; scrambled fail; non-module n/a."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bgc_pipeline.evaluation import check_module_architecture
    C, A, T = "PF00668", "PF00501", "PF00550"
    def dom(acc, orf, pos): return {"pfam_accession": acc, "target": f"orf_{orf}", "env_from": pos}
    ordered = {"orfs_meta": [{"i": 0, "start": 0, "end": 9000}, {"i": 1, "start": 9000, "end": 18000}],
               "domains_found": [dom(C, 0, 10), dom(A, 0, 400), dom(T, 0, 800),
                                 dom(C, 1, 10), dom(A, 1, 400), dom(T, 1, 800)]}
    r = check_module_architecture(ordered, "NRPS")
    assert r["module_count"] == 2 and r["ordered_module_count"] == 2 and r["pass"] is True, r
    scrambled = {"orfs_meta": [{"i": 0, "start": 0, "end": 9000}],
                 "domains_found": [dom(T, 0, 10), dom(A, 0, 400), dom(C, 0, 800)]}
    r2 = check_module_architecture(scrambled, "NRPS")
    assert r2["module_count"] == 1 and r2["ordered_module_count"] == 0 and r2["pass"] is False, r2
    # hybrid: one NRPS module (C-A-T) + one PKS module (KS-AT-ACP) -> 2 ordered modules
    KS, AT, ACP = "PF00109", "PF00698", "PF00550"   # ACP shares PF00550 with NRPS T
    hyb = {"orfs_meta": [{"i": 0, "start": 0, "end": 9000}, {"i": 1, "start": 9000, "end": 18000}],
           "domains_found": [dom(C, 0, 10), dom(A, 0, 400), dom(T, 0, 800),
                             dom(KS, 1, 10), dom(AT, 1, 400), dom(ACP, 1, 800)]}
    rh = check_module_architecture(hyb, "PKS_NRPS_HYBRID")
    assert rh["applicable"] is True and rh["module_count"] == 2 and rh["ordered_module_count"] == 2, rh
    # non-module class -> not applicable / no_verdict
    assert check_module_architecture(ordered, "TERPENE")["applicable"] is False
    # missing class_markers -> no_verdict (pass None)
    assert check_module_architecture(None, "NRPS")["pass"] is None
    print("PASS: module_architecture (ordered NRPS/PKS/hybrid pass; scrambled fail; non-module n/a)")


def test_derive_questions():
    """derive_questions: antiSMASH is AUTHORITATIVE for is_bgc/correct_class when it
    ran; class_markers (domains) is the PROXY when antiSMASH is skipped (incl. the
    parse-error->skipped path). Covers the gate-question derivation directly."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bgc_pipeline.evaluation import derive_questions
    # antiSMASH authoritative: detected + class_match -> is_bgc/correct_class PASS,
    # and the class_markers proxy is IGNORED while antiSMASH ran.
    q = derive_questions({
        "coding_sanity": {"pass": True},
        "antismash": {"detected": True, "class_match": True},
        "class_markers": {"pass": False, "domain_count": 0},
        "kmer_novelty": {"pass": True},
    })
    assert q["is_bgc"] == "PASS" and q["correct_class"] == "PASS" and q["novel"] == "PASS", q
    # antiSMASH detected=False -> is_bgc FAIL even though coding_sanity passed.
    q2 = derive_questions({"coding_sanity": {"pass": True},
                           "antismash": {"detected": False, "class_match": False}})
    assert q2["is_bgc"] == "FAIL" and q2["correct_class"] == "FAIL", q2
    # antiSMASH skipped (e.g. parse error) -> PROXY: domains present + coding floor.
    q3 = derive_questions({"coding_sanity": {"pass": True},
                           "antismash": {"skipped": True},
                           "class_markers": {"pass": True, "domain_count": 5}})
    assert q3["is_bgc"] == "PASS" and q3["correct_class"] == "PASS", q3
    # proxy with no domains -> not a BGC.
    q4 = derive_questions({"coding_sanity": {"pass": True},
                           "antismash": {"skipped": True},
                           "class_markers": {"pass": False, "domain_count": 0}})
    assert q4["is_bgc"] == "FAIL", q4
    # nothing ran -> skipped, not a false verdict.
    q5 = derive_questions({})
    assert q5["is_bgc"] == "skipped" and q5["novel"] == "skipped", q5
    print("PASS: derive_questions (antismash authoritative; class_markers proxy when skipped)")


def main():
    test_m9_adherence()
    test_m9_balanced_sample()
    test_m2_length_buckets_and_first_window()
    test_m1_resume_alignment()
    test_m7_early_stop()
    test_novelty_gate_wired()
    test_metric1_skips_when_tool_cannot_run()
    test_eval_suite_driver_wires_dbs()
    test_eval_suite_aggregation()
    test_gate_keyed_headline()
    test_coding_sanity()
    test_module_architecture()
    test_derive_questions()
    print("\nALL EVAL-METRIC TESTS PASSED")


if __name__ == "__main__":
    main()
