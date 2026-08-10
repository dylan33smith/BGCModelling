"""END-TO-END smoke test: every eval path actually EXECUTES and returns a sane shape.

WHY THIS EXISTS. The 2026-07-31 hardening pass rewrote ~40 sites across evaluation.py and the
drivers using string replacement, and shipped TWO regressions to main:
  * quick_eval.sh committed with an IndentationError in an embedded heredoc (bash -n passes,
    so it only fails at runtime, after the expensive generation step);
  * check_antismash made to RAISE on antiSMASH's <1000 nt input rejection -- a MODEL failure
    mode, not a configuration one, so a regressing model would crash its own evaluation.
Both were caught only by a later audit. The unit tests passed throughout, because they exercise
LOGIC (derive_questions on hand-built dicts) rather than the real call paths.

This test runs the real functions on real sequences and asserts the SHAPE and the KIND of each
result -- specifically the resource-vs-data distinction that regression (2) got wrong.
Deliberately does NOT assert specific rates: those move legitimately.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PFAM = Path("/data2/ds85/pfam/Pfam-A.hmm")
ASDB = "/data2/ds85/antismash_db"
CORES = Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl")


def _real_core(min_len: int = 3000) -> dict:
    with CORES.open() as f:
        for line in f:
            r = json.loads(line)
            if len(r.get("sequence", "")) >= min_len:
                return r
    raise SystemExit("no suitable core found")


def test_coding_sanity_and_gene_caller():
    from bgc_pipeline.evaluation import check_coding_sanity
    r = check_coding_sanity(_real_core()["sequence"][:4000])
    assert r["gene_caller"] == "prodigal", r          # NOT the retired six-frame fallback
    assert r["complete_gene_fraction"] is not None, r  # None only under the degraded caller
    assert 0.0 <= r["coding_density"] <= 1.0, r
    assert isinstance(r["pass"], bool), r
    print("PASS coding_sanity: prodigal caller, real complete_gene_fraction")


def test_class_markers_needs_pfam_and_works_with_it():
    from bgc_pipeline.evaluation import check_class_markers, EvalResourceError
    seq = _real_core()["sequence"][:3000]
    try:
        check_class_markers(seq, expected_class="NRPS")
        raise AssertionError("missing pfam_hmm_path must RAISE, not skip")
    except EvalResourceError:
        pass
    if PFAM.exists():
        r = check_class_markers(seq, expected_class="NRPS", pfam_hmm_path=PFAM)
        assert not r.get("skipped"), r
        assert "domain_count" in r, r
    print("PASS class_markers: raises without Pfam, runs with it")


def test_antismash_distinguishes_data_from_config():
    """THE REGRESSION TEST. A too-short sequence is a MODEL failure -> score it FAIL.
    A missing database is a CONFIG failure -> raise. These must not be conflated."""
    from bgc_pipeline.evaluation import check_antismash, EvalResourceError
    from bgc_pipeline.class_map import load_class_map
    if not Path(ASDB).is_dir():
        print("SKIP antismash (no DB on this host)")
        return
    cmap, _ = load_class_map(REPO / "config" / "compound_class_map.yaml")

    r = check_antismash("ATGC" * 54, expected_class="NRPS", databases_dir=ASDB, class_map=cmap)
    assert r.get("input_rejected") is True, r
    assert r.get("detected") is False and r.get("pass") is False, r
    assert not r.get("skipped"), r      # scored, NOT skipped -- it is a real negative

    try:
        check_antismash("ATGC" * 600, expected_class="NRPS",
                        databases_dir="/nonexistent", class_map=cmap)
        raise AssertionError("a missing database must RAISE")
    except EvalResourceError:
        pass

    try:
        check_antismash("ATGC" * 600, expected_class="NRPS", databases_dir=ASDB)
        raise AssertionError("expected_class without a class_map must RAISE")
    except EvalResourceError:
        pass
    print("PASS antismash: short input -> FAIL (data); missing DB/class_map -> RAISE (config)")


def test_novelty_gate_can_actually_fail():
    """A gate that cannot fail is not a gate. Verifies both directions."""
    from bgc_pipeline.evaluation import check_kmer_novelty
    assert check_kmer_novelty({"max_containment": 0.99})["pass"] is False
    assert check_kmer_novelty({"max_containment": 0.01})["pass"] is True
    # a record missing the key must SKIP, never default to the passing value
    r = check_kmer_novelty({"nearest_accession": "X"})
    assert r.get("skipped") and r.get("pass") is None, r
    print("PASS kmer_novelty: fails on memorized, passes on novel, skips on malformed")


def test_taxon_profile_resolves_from_a_real_gtdb_tag():
    from bgc_pipeline.evaluation import load_taxon_profiles, resolve_taxon_profile
    tp = REPO / "data" / "processed" / "taxon_profiles.json"
    if not tp.exists():
        print("SKIP taxon profiles (not present)")
        return
    prof = load_taxon_profiles(tp)
    tag = _real_core().get("taxonomic_tag", "")
    assert resolve_taxon_profile(tag, prof) is not None, (
        f"real GTDB tag {tag[:60]} did not resolve -- conditioning_faithful would be null")
    print("PASS taxon profiles: real lowercase GTDB tag resolves")


def test_derive_questions_records_its_source():
    from bgc_pipeline.evaluation import derive_questions
    q = derive_questions({"coding_sanity": {"pass": True},
                          "antismash": {"detected": True, "class_match": True}})
    assert q["_verdict_source"]["is_bgc"] == "antismash", q
    q2 = derive_questions({"coding_sanity": {"pass": True},
                           "antismash": {"skipped": True},
                           "class_markers": {"pass": True, "domain_count": 9,
                                             "unique_domain_accessions": ["PF00668", "PF00501"]}})
    assert q2["_verdict_source"]["is_bgc"] == "class_markers_proxy", q2
    print("PASS derive_questions: verdict provenance recorded")


def test_driver_rates_are_none_not_zero_when_unmeasured():
    from eval_suite_driver import summarize_group
    recs = [{"questions": {"is_bgc": "PASS", "correct_class": "PASS", "novel": "skipped"}}] * 4
    h = summarize_group(recs)["headline"]
    assert h["biological_valid_and_novel"]["rate"] is None, h
    assert h["generates_bgc"]["rate"] == 1.0, h
    print("PASS driver: unmeasured -> None, never a fabricated 0.0")


def test_adherence_returns_none_rates_when_nothing_scored():
    sys.path.insert(0, str(REPO / "evo2" / "scripts"))
    import eval_conditioning_adherence as E
    s = E.summarize_adherence([{"true_class": "A", "scores": {"A": -1.0}}], ["A", "B"])
    assert s["n_scored"] == 0 and s["top1_acc"] is None, s
    print("PASS adherence: no fabricated 0.000 when nothing scored")


def main() -> int:
    for t in (test_coding_sanity_and_gene_caller,
              test_class_markers_needs_pfam_and_works_with_it,
              test_antismash_distinguishes_data_from_config,
              test_novelty_gate_can_actually_fail,
              test_taxon_profile_resolves_from_a_real_gtdb_tag,
              test_derive_questions_records_its_source,
              test_driver_rates_are_none_not_zero_when_unmeasured,
              test_adherence_returns_none_rates_when_nothing_scored):
        t()
    print("\nALL EVAL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
