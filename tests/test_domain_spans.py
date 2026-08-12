"""Guard the amino-acid -> nucleotide mapping behind the per-position domain labels.

WHY THIS EXISTS. `build_domain_spans.py` produces the per-position mask a domain-weighted loss
would train against. Its one genuinely error-prone step is mapping a domain's amino-acid envelope
back onto nucleotide coordinates, because a reverse-strand gene is translated from the reverse
complement: residue 1 sits at the ORF's HIGH coordinate and the envelope runs backwards.

Get that wrong and roughly half of all domains are mislabelled — but the spans are still plausible
lengths, still inside real ORFs, and still sum to a believable "fraction of the core is machinery".
Nothing downstream would raise. The loss would simply up-weight the wrong nucleotides, and the
training run that followed would be uninterpretable rather than obviously broken.

The round-trip test is the load-bearing one: take a known protein motif, place it on each strand,
and check the mapped nucleotides actually translate back to that motif.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


class _ORF:
    """Matches the fields aa_span_to_nt reads off bgc_pipeline.evaluation.ORF."""

    def __init__(self, start, end, strand):
        self.start, self.end, self.strand = start, end, strand


def test_forward_strand_maps_from_the_low_coordinate():
    from build_domain_spans import aa_span_to_nt

    orf = _ORF(100, 400, 1)                       # 300 nt = 100 codons
    assert aa_span_to_nt(orf, 1, 1) == (100, 103), "residue 1 must be the first codon"
    assert aa_span_to_nt(orf, 1, 10) == (100, 130), "10 residues = 30 nt from the ORF start"
    assert aa_span_to_nt(orf, 11, 20) == (130, 160), "residue 11 starts at offset 30"
    assert aa_span_to_nt(orf, 1, 100) == (100, 400), "the whole protein spans the whole ORF"
    print("PASS spans: forward strand maps from the ORF's low coordinate")


def test_reverse_strand_maps_from_the_high_coordinate():
    """THE REGRESSION TEST. On the minus strand residue 1 is at the ORF's END."""
    from build_domain_spans import aa_span_to_nt

    orf = _ORF(100, 400, -1)
    assert aa_span_to_nt(orf, 1, 1) == (397, 400), (
        "residue 1 of a reverse-strand gene must map to the LAST codon of the ORF, not the first — "
        "this is the error that mislabels half of all domains while looking entirely normal")
    assert aa_span_to_nt(orf, 1, 10) == (370, 400)
    assert aa_span_to_nt(orf, 11, 20) == (340, 370)
    assert aa_span_to_nt(orf, 1, 100) == (100, 400), "the whole protein still spans the whole ORF"
    print("PASS spans: reverse strand maps from the ORF's high coordinate")


def test_span_length_is_three_nt_per_residue_on_both_strands():
    from build_domain_spans import aa_span_to_nt

    for strand in (1, -1):
        orf = _ORF(0, 900, strand)
        for a, b in ((1, 1), (5, 25), (100, 200), (250, 300)):
            s, e = aa_span_to_nt(orf, a, b)
            assert e - s == 3 * (b - a + 1), f"strand {strand}, residues {a}-{b}: got {e - s} nt"
    print("PASS spans: span length is exactly 3 nt per residue on both strands")


def test_spans_stay_inside_the_orf():
    """HMMER envelopes can run past the translated length; clamping must not produce a negative
    or out-of-range span, which would silently corrupt a mask."""
    from build_domain_spans import aa_span_to_nt

    for strand in (1, -1):
        orf = _ORF(50, 200, strand)               # 150 nt = 50 codons
        s, e = aa_span_to_nt(orf, 1, 999)
        assert 50 <= s <= e <= 200, f"strand {strand}: span ({s}, {e}) escaped the ORF"
        s, e = aa_span_to_nt(orf, 900, 999)
        assert 50 <= s <= e <= 200, f"strand {strand}: far-past-end span ({s}, {e}) escaped"
    print("PASS spans: envelopes running past the ORF are clamped inside it")


def test_round_trip_through_translation():
    """THE ONE THAT WOULD CATCH A SUBTLE OFF-BY-ONE. Plant a known peptide in a synthetic ORF on
    each strand, map its residue envelope back to nucleotides, and translate those nucleotides —
    they must spell the peptide. Coordinate arithmetic that merely *looks* right fails here."""
    from Bio.Seq import Seq

    from build_domain_spans import aa_span_to_nt

    motif_aa = "MKWVTFIS"
    motif_nt = str(Seq(motif_aa).back_translate()) if hasattr(Seq, "back_translate") else None
    if motif_nt is None:                          # Biopython has no back_translate; build codons
        table = {"M": "ATG", "K": "AAA", "W": "TGG", "V": "GTG", "T": "ACG",
                 "F": "TTT", "I": "ATT", "S": "AGC"}
        motif_nt = "".join(table[c] for c in motif_aa)

    lead, tail = "ATG" * 5, "GGC" * 7             # flanking codons
    coding = lead + motif_nt + tail               # forward-strand coding sequence
    aa_from = len(lead) // 3 + 1                  # 1-based residue index of the motif
    aa_to = aa_from + len(motif_aa) - 1

    # forward strand: the ORF's nucleotides ARE the coding sequence
    orf = _ORF(0, len(coding), 1)
    s, e = aa_span_to_nt(orf, aa_from, aa_to)
    assert str(Seq(coding[s:e]).translate()) == motif_aa, (
        f"forward round-trip gave {Seq(coding[s:e]).translate()}, expected {motif_aa}")

    # reverse strand: the stored sequence is the reverse complement of the coding sequence
    stored = str(Seq(coding).reverse_complement())
    orf = _ORF(0, len(stored), -1)
    s, e = aa_span_to_nt(orf, aa_from, aa_to)
    got = str(Seq(stored[s:e]).reverse_complement().translate())
    assert got == motif_aa, f"reverse round-trip gave {got}, expected {motif_aa}"
    print("PASS spans: mapped nucleotides translate back to the motif on BOTH strands")


def main() -> int:
    for t in (test_forward_strand_maps_from_the_low_coordinate,
              test_reverse_strand_maps_from_the_high_coordinate,
              test_span_length_is_three_nt_per_residue_on_both_strands,
              test_spans_stay_inside_the_orf,
              test_round_trip_through_translation):
        t()
    print("\nALL DOMAIN-SPAN TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
