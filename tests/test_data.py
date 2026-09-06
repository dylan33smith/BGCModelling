"""SPEC 11 verification tests for the data layer. Written against the spec, not against
any prior implementation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


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


MANIFEST = Path("/data2/ds85/bgcbench/manifest.json")


def test_built_splits_are_balanced():
    """Behavioural, not source-text. Components chain badly (a record links by genome OR
    cluster), so per-component hashing gave 93/3.5/3.5 and union-level balancing gave
    ~60/20/20. Assert the built artifact, since that is what the benchmark consumes."""
    if not MANIFEST.exists():
        return                                   # build has not run yet
    doc = json.loads(MANIFEST.read_text())
    for cls in classmap.BENCHMARK_CLASSES:
        if cls not in doc:
            continue
        n = doc[cls]["split"]["n"]
        total = sum(n.values())
        assert 0.72 <= n["train"] / total <= 0.86, f"{cls} train share {n}"
        for held in ("val", "test"):
            assert 0.06 <= n[held] / total <= 0.16, f"{cls} {held} share {n}"


def test_built_splits_are_leak_free():
    """SPEC 4.6: the built artifact must carry zero genome overlap and zero near-dups in
    both orientations. Any residual removal must be recorded, not absorbed."""
    if not MANIFEST.exists():
        return
    doc = json.loads(MANIFEST.read_text())
    for cls in classmap.BENCHMARK_CLASSES:
        if cls not in doc:
            continue
        v = doc[cls]["verification"]
        assert v["genome_overlap"] == 0
        assert v["neardup_fwd"] == 0 and v["neardup_revcomp"] == 0
        assert "residual_removed" in v, "residual count must be recorded"


def test_negative_controls_are_not_pseudoreplicated():
    """300 intervals from one genome is one control, not 300. The first build reported
    genomes=1."""
    if not MANIFEST.exists():
        return
    doc = json.loads(MANIFEST.read_text())
    for cls in classmap.BENCHMARK_CLASSES:
        if cls not in doc:
            continue
        nc = doc[cls]["negative_control"]
        assert nc["genomes"] >= 0.9 * nc["n"], (
            f"{cls}: {nc['n']} negatives from only {nc['genomes']} genomes")


def test_cluster_mode_is_connected_component():
    """cluster-mode 0 (greedy set cover) does not make 'same cluster' equal 'similar',
    so a cluster-disjoint split still leaked 3 fwd + 3 revcomp near-dups in TERPENE."""
    from bgcbench.data import cluster as clu
    assert clu.CLUSTER_MODE == 1
    assert clu.SENSITIVITY >= 7.0


def test_ripp_strata_are_length_matched_and_equal_sized():
    """SPEC 4.5: gene count must be the only varying factor. Unmatched strata would
    measure length, which is the confound the design exists to remove."""
    if not MANIFEST.exists():
        return
    doc = json.loads(MANIFEST.read_text())
    strata = doc.get("_ripp_strata")
    if not strata:
        return
    for part, r in strata.items():
        assert r["matched"]["single"] == r["matched"]["multi"], f"{part} unequal"
        s, m = r["median_len"]["single"], r["median_len"]["multi"]
        assert abs(s - m) / max(s, m) < 0.10, f"{part} median length {s} vs {m}"
        assert r["median_core_genes"]["single"] == 1
        assert r["median_core_genes"]["multi"] >= 2


def test_one_dataset_invariant_is_documented():
    """SPEC 4.5: every arm consumes splits/<CLASS>/ and nothing else. An arm with bespoke
    training data is not comparable to any other arm."""
    spec = Path(__file__).resolve().parents[1] / "docs" / "SPEC.md"
    text = spec.read_text()
    assert "THE INVARIANT" in text
    assert "strata" in text and "not part of the arm grid" in text
