"""SPEC 11 verification tests for the data layer. Written against the spec, not against
any prior implementation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


from bgcbench.data import classmap, manifest, negative, split
from bgcbench.data.genbank import parse

# --------------------------------------------------------------------- KNOWN_WRONG #2
def test_manifest_additive():
    """Writing class B must preserve class A. The prior builder initialised an empty
    manifest and rewrote the file, silently destroying siblings."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.json"
        manifest.update(p, "RIPP", {"n": 1})
        manifest.update(p, "TERPENE", {"n": 2})
        doc = manifest.load(p)
        assert set(doc) == {"RIPP", "TERPENE"}
        assert doc["RIPP"]["n"] == 1


def test_manifest_has_no_wholesale_write():
    """There must be no API that replaces the whole document."""
    src = Path(manifest.__file__).read_text()
    assert "def replace" not in src and "def write_all" not in src


# --------------------------------------------------------------------- KNOWN_WRONG #1
GBK = """LOCUS       TESTREC                60 bp    DNA     linear   CON 01-JAN-2026
FEATURES             Location/Qualifiers
     region          11..50
                     /product="RiPP-like"
                     /region_number="1"
                     /contig_edge="False"
     proto_core      21..40
                     /product="RiPP-like"
     CDS             22..39
                     /locus_tag="X1"
ORIGIN
        1 aaaaaaaaaa ccccccccccg ggggggggtt tttttttttt aaaaaaaaaa cccccccccc
//
"""


def test_genbank_coordinates_are_zero_based_half_open():
    rec = next(parse(GBK))
    reg = rec.of("region")[0]
    core = rec.of("proto_core")[0]
    # GenBank 11..50 is 1-based inclusive -> 0-based half-open [10, 50)
    assert (reg.start, reg.end) == (10, 50)
    assert (core.start, core.end) == (20, 40)
    assert core.end - core.start == 20


def test_core_is_protocore_not_region():
    """The region carries a 5-20 kb neighbourhood on each side; using it as the sequence
    would overstate length by an order of magnitude (observed 11,008 vs 1,008)."""
    rec = next(parse(GBK))
    reg = rec.of("region")[0]
    core = rec.of("proto_core")[0]
    assert (core.end - core.start) < (reg.end - reg.start)


def test_seq_len_is_len_sequence():
    """SPEC 4.2: seq_len must equal len(sequence). There is no region-span length field."""
    from bgcbench.data.extract import CoreRecord
    fields = CoreRecord.__dataclass_fields__
    assert "seq_len" in fields
    assert "region_len" not in fields, "a *_len field must never hold a region span"


# ---------------------------------------------------------------------- class map
def test_benchmark_classes_survive_regeneration():
    m = classmap.build_map()
    classmap.validate(m["mapping"])          # raises if any benchmark class is missing
    for c in classmap.BENCHMARK_CLASSES:
        assert c in set(m["mapping"].values())


def test_promotions_are_single_product_classes():
    m = classmap.build_map()["mapping"]
    for p in classmap.PROMOTIONS:
        members = [k for k, v in m.items() if v == p.upper()]
        assert members == [p]


def test_promotions_carry_a_reason():
    assert all(len(r) > 40 for r in classmap.PROMOTIONS.values())


def test_unmapped_is_not_folded_into_other():
    m = classmap.build_map()["mapping"]
    assert classmap.classify(["a-product-that-does-not-exist"], m) == ["UNMAPPED"]


def test_hybrid_counts_for_every_class():
    """SPEC 3.3: a record counts for EVERY class its products map to."""
    m = classmap.build_map()["mapping"]
    assert classmap.classify(["arylpolyene", "betalactone"], m) == [
        "ARYLPOLYENE", "BETALACTONE"]


# ---------------------------------------------------------------------- split
def test_split_assignment_is_deterministic_and_seedless():
    a = [split._assign(f"k{i}") for i in range(500)]
    b = [split._assign(f"k{i}") for i in range(500)]
    assert a == b
    assert set(a) <= {"train", "val", "test"}
    assert all(x in a for x in ("train", "val", "test"))


def test_split_fractions_are_approximately_right():
    n = 20000
    got = [split._assign(f"key{i}") for i in range(n)]
    assert 0.78 < got.count("train") / n < 0.82


# ---------------------------------------------------------------------- negatives
def test_free_intervals_respect_margin():
    free = negative._free_intervals(10000, [(4000, 5000)], margin=500)
    assert free == [(0, 3500), (5500, 10000)]
    for s, e in free:
        assert not (s < 5500 and 3500 < e), "interval overlaps the padded region"


def test_free_intervals_merge_overlapping_regions():
    free = negative._free_intervals(10000, [(1000, 2000), (2200, 3000)], margin=500)
    assert free == [(0, 500), (3500, 10000)]


def test_length_targets_match_the_class_distribution():
    recs = [{"seq_len": v} for v in [100, 200, 300, 400, 500]]
    t = negative.length_targets(recs, 5)
    assert min(t) >= 100 and max(t) <= 500
    assert len(t) == 5
