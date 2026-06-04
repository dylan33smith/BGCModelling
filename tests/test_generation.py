#!/usr/bin/env python3
"""Tests for the C3 generation post-processing (no GPU).

Covers: EOS/prefix consistency with training, EOS trimming + nucleotide
sanitation, N fraction, FASTA formatting, and prompt sampling.

Run: python tests/test_generation.py
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import generate_bgc as G  # noqa: E402


def test_consistency_with_training():
    import finetune_evo2_lora as F
    assert G.EOS_MARKER == F.EOS_MARKER, "EOS marker drifted from training"
    rec = {"compound_class": "NRPS", "taxonomic_tag": "|D__BACTERIA;P__X"}
    assert G.build_prefix("NRPS", rec["taxonomic_tag"]) == F.canonical_phase1_prefix(rec)
    assert G.build_continuation_prefix("NRPS", rec["taxonomic_tag"]) == F.continuation_phase1_prefix(rec)
    print("PASS: EOS marker + class/continuation prefixes match training exactly")


def test_extract_sequence():
    # EOS present, clean nucleotides before it
    r = G.extract_sequence("ACGTACGT|END|leftover")
    assert r["sequence"] == "ACGTACGT" and r["hit_eos"] and r["len"] == 8
    assert r["n_count"] == 0 and not r["trailing_junk_trimmed"]
    # no EOS -> whole valid run kept; N counted
    r = G.extract_sequence("ACGTNNGT")
    assert r["sequence"] == "ACGTNNGT" and not r["hit_eos"] and r["n_count"] == 2
    # junk after the nucleotide run (no EOS) -> trimmed at first invalid char
    r = G.extract_sequence("ACGT|FOO")
    assert r["sequence"] == "ACGT" and not r["hit_eos"] and r["trailing_junk_trimmed"]
    # EOS at very start -> empty sequence
    r = G.extract_sequence("|END|ACGT")
    assert r["sequence"] == "" and r["hit_eos"] and r["len"] == 0
    # lowercase handled
    assert G.extract_sequence("acgtACGT|END|")["sequence"] == "ACGTACGT"
    print("PASS: extract_sequence trims at EOS, keeps leading ACGTN run, flags junk")


def test_n_fraction():
    assert G.n_fraction("ACGT") == 0.0
    assert abs(G.n_fraction("ACGTNNNN") - 0.5) < 1e-9
    assert G.n_fraction("") == 0.0
    print("PASS: n_fraction")


def test_fasta():
    seq = "ACGT" * 30  # 120 nt -> 80 + 40 wrap
    rec = G.to_fasta_record("gen_1", seq, compound_class="NRPS", length=120, eos=True)
    lines = rec.rstrip("\n").split("\n")
    assert lines[0] == ">gen_1 compound_class=NRPS length=120 eos=True"
    assert all(len(l) <= 80 for l in lines[1:]), "FASTA must wrap at 80 cols"
    assert "".join(lines[1:]) == seq
    # empty sequence still yields a header + (empty) body line
    assert G.to_fasta_record("g", "", compound_class="X").startswith(">g compound_class=X")
    print("PASS: FASTA header + 80-col wrapping")


def test_sample_prompts():
    recs = ([{"compound_class": "A", "taxonomic_tag": f"|t{i}"} for i in range(5)] +
            [{"compound_class": "B", "taxonomic_tag": "|tb"} for _ in range(3)])
    prompts = G.sample_prompts(recs, per_class=2, rng=random.Random(0))
    cc = Counter(p["compound_class"] for p in prompts)
    assert cc["A"] == 2 and cc["B"] == 2
    assert all("taxonomic_tag" in p for p in prompts)
    print("PASS: sample_prompts caps per class")


def main():
    test_consistency_with_training()
    test_extract_sequence()
    test_n_fraction()
    test_fasta()
    test_sample_prompts()
    print("\nALL GENERATION TESTS PASSED")


if __name__ == "__main__":
    main()
