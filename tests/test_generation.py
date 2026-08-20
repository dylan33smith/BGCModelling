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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evo2" / "scripts"))
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
    # junk after the nucleotide run (no EOS), TRUNCATE policy -> cut at first invalid char.
    # This is the pre-2026-08-20 behaviour, kept so old runs stay reproducible.
    r = G.extract_sequence("ACGT|FOO", "truncate")
    assert r["sequence"] == "ACGT" and not r["hit_eos"] and r["trailing_junk_trimmed"]
    assert r["discarded_by_junk_policy" if "discarded_by_junk_policy" in r
            else "discarded_by_truncate"] == 4
    # EOS at very start -> empty sequence
    r = G.extract_sequence("|END|ACGT")
    assert r["sequence"] == "" and r["hit_eos"] and r["len"] == 0
    # lowercase handled
    assert G.extract_sequence("acgtACGT|END|")["sequence"] == "ACGTACGT"
    print("PASS: extract_sequence trims at EOS, keeps leading ACGTN run, flags junk")


def test_extract_sequence_junk_policy():
    """MASK is the default: a stray byte must NOT cost us the rest of the sequence.

    Measured 2026-08-20: 23/32 raw PKS-adapter generations carried a stray character (usually a
    single space at a codon boundary), and truncating at it discarded a median of 6.2 kb that was
    99.9% ACGT. Worse, it hit fine-tuned adapters ~35x more than the base model, so it biased the
    treatment-vs-control comparison itself. See bugs.md.
    """
    raw = "ACGTACGTGA TTGTCGAGTT"          # one stray space mid-sequence, valid DNA after it
    t = G.extract_sequence(raw, "truncate")
    m = G.extract_sequence(raw, "mask")
    assert t["sequence"] == "ACGTACGTGA", t["sequence"]
    assert m["sequence"] == "ACGTACGTGANTTGTCGAGTT", m["sequence"]

    # THE POINT: masking preserves LENGTH, i.e. FRAME. Deleting the byte instead would shift every
    # downstream codon by one and destroy the ORFs the whole scoring stack is built on.
    assert len(m["sequence"]) == len(raw), "mask must be frame-preserving"
    assert m["n_count"] == 1 and m["n_junk_chars"] == 1
    assert m["len"] > t["len"] == m["leading_run_len"]

    # mask must not resurrect anything after a real EOS -- |END| still wins.
    e = G.extract_sequence("ACGT|END|ACGTACGT", "mask")
    assert e["hit_eos"] and e["sequence"] == "ACGT"

    # default policy is mask
    assert G.extract_sequence(raw)["sequence"] == m["sequence"]
    assert G.JUNK_POLICY_DEFAULT == "mask"

    # an unknown policy is an error, not a silent fallback
    try:
        G.extract_sequence(raw, "strip")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown junk_policy must raise")
    print("PASS: junk_policy mask keeps downstream DNA and preserves frame; truncate reproduces old runs")


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
    test_extract_sequence_junk_policy()
    test_n_fraction()
    test_fasta()
    test_sample_prompts()
    print("\nALL GENERATION TESTS PASSED")


if __name__ == "__main__":
    main()
