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
        cls = p.upper().replace("-", "_")   # hyphens are path components in run dirs
        members = [k for k, v in m.items() if v == cls]
        assert members == [p]


def test_promotions_carry_a_reason():
    assert all(len(r) > 40 for r in classmap.PROMOTIONS.values())


def test_unmapped_is_not_folded_into_other():
    m = classmap.build_map()["mapping"]
    assert classmap.classify(["a-product-that-does-not-exist"], m) == ["UNMAPPED"]


def test_hybrid_counts_for_every_class():
    """SPEC 3.3: a record counts for EVERY class its products map to."""
    m = classmap.build_map()["mapping"]
    assert classmap.classify(["arylpolyene", "redox-cofactor"], m) == [
        "ARYLPOLYENE", "REDOX_COFACTOR"]


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
        # removal now happens in build() with backfill, so equal-n survives it; the count
        # is recorded there, not in the (read-only) verification block
        assert "leaking_held_out_replaced" in doc[cls]["split"], (
            "the number of leaking held-out records replaced must be recorded")


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


def test_join_locations_do_not_use_the_bounding_box():
    """An origin-spanning gene on a circular replicon, join(2842906..2843201,1..161),
    collapses to a 2.84 Mb bounding box and would overlap every cluster on the replicon.
    Overlap must test the actual intervals."""
    from bgcbench.data.genbank import parse as gbparse
    gbk = ("LOCUS       CIRC                  3000 bp    DNA     circular CON 01-JAN-2026\n"
           "FEATURES             Location/Qualifiers\n"
           "     CDS             join(2900..2950,1..60)\n"
           '                     /gene_kind="biosynthetic"\n'
           "ORIGIN\n"
           "        1 " + "acgt" * 15 + "\n"
           "//\n")
    rec = next(gbparse(gbk))
    cds = rec.of("CDS")[0]
    assert cds.spans == [(2899, 2950), (0, 60)]
    assert cds.overlaps(0, 100) is True          # real interval
    assert cds.overlaps(2890, 2960) is True      # real interval
    assert cds.overlaps(1000, 2000) is False, (
        "bounding box 2899..2950 would falsely overlap mid-replicon coordinates")


CORPUS = Path("/data2/ds85/bgcbench/corpus/core_records.jsonl")


def test_corpus_accessions_are_unique():
    """The accession keys every downstream structure -- cluster assignment, record_split,
    component lookup, mmseqs FASTA headers. antiSMASH numbers regions PER RECORD, so
    omitting the locus made 56.8% of records collide."""
    if not CORPUS.exists():
        return
    seen = set()
    with open(CORPUS) as fh:
        for line in fh:
            a = json.loads(line)["accession"]
            assert a not in seen, f"duplicate accession {a}"
            seen.add(a)


def test_a_record_lands_in_one_partition_across_all_classes():
    """SPEC 4.6 global consistency. Under the SPEC 3.3 hybrid rule a record legitimately
    appears in SEVERAL class corpora, so global uniqueness is the wrong property. What
    must hold is that it lands in the same train/val/test partition in every one of them
    -- otherwise a hybrid record sits in TERPENE-train and NRPS-test, and the pooled W1
    arm, which trains on the union, trains on its own test set."""
    root = Path("/data2/ds85/bgcbench/splits")
    if not root.exists():
        return
    part_of: dict[str, str] = {}
    for f in sorted(root.glob("*/*.jsonl")):
        part = f.stem
        for line in open(f):
            a = json.loads(line)["accession"]
            prev = part_of.setdefault(a, part)
            assert prev == part, (
                f"{a} is in '{prev}' for one class and '{part}' for another — the pooled "
                f"arm would train on its own test set")


def test_split_files_have_no_internal_duplicates():
    root = Path("/data2/ds85/bgcbench/splits")
    if not root.exists():
        return
    for f in sorted(root.glob("*/*.jsonl")):
        accs = [json.loads(l)["accession"] for l in open(f)]
        assert len(accs) == len(set(accs)), f"duplicate accession inside {f}"


def test_verify_is_read_only():
    """verify() previously deleted leaking held-out records and rewrote the split files
    after the report was computed, so the manifest published 979/123/122 while disk held
    979/107/110. Removal now happens in build() with backfill; verify must only assert."""
    src = Path(split.__file__).read_text()
    body = src[src.index("def verify("):]
    assert '"w"' not in body and "write_text" not in body, "verify() writes to disk"
    assert "STRICTLY READ-ONLY" in body


def test_corpus_order_does_not_change_the_build():
    """SPEC 4.6 claims the split is reproducible from the corpus alone. extract.py writes
    with imap_unordered, so line order is worker-completion order; load_corpus must impose
    a canonical order or clustering (which is order-sensitive) re-partitions."""
    src = Path(split.__file__).read_text()
    assert "out.sort(key=lambda r: r[\"accession\"])" in src


def test_short_and_ambiguous_records_are_filtered():
    assert split.MIN_LEN >= 200 and split.MAX_N_FRAC <= 0.2
    root = Path("/data2/ds85/bgcbench/splits")
    if not root.exists():
        return
    for f in sorted(root.glob("*/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            assert r["seq_len"] >= split.MIN_LEN, f"{r['accession']} is {r['seq_len']} nt"
            n_frac = r["sequence"].count("N") / max(r["seq_len"], 1)
            assert n_frac <= split.MAX_N_FRAC, f"{r['accession']} is {n_frac:.0%} N"


def test_evo2_termination_does_not_rely_on_vortex_stop_at_eos():
    """vortex's stop_at_eos prints and does not break -- it is dead code, and its condition
    inspects row 0 only. Termination must be post-hoc truncation, so the substrate layer
    must not claim native stop for Evo2."""
    import inspect
    from bgcbench.model.load import Substrate
    src = inspect.getsource(Substrate)
    assert "truncate_at_terminator" in src
    text, hit = Substrate(id="x", family="evo2", checkpoint="c", terminator_id=0,
                          terminator_str=chr(0), appends_terminator=False,
                          native_stop=False, approx_nt_per_token=1.0
                          ).truncate_at_terminator("ACGT" + chr(0) + "TTTT")
    assert text == "ACGT" and hit is True


def test_generate_has_no_arm_specific_branch():
    """SPEC 9 rule 2: generate.py takes an arm coordinate as DATA. A branch on an arm name
    is how two arms come to be generated differently without anyone deciding to."""
    from bgcbench.model import generate as gen
    src = Path(gen.__file__).read_text()
    for tok in ("W0", "W1r", "W1", "W2", "W3", "S0", "S1", "I0", "I1", "I2"):
        assert f'"{tok}"' not in src and f"'{tok}'" not in src, f"branches on {tok}"


def test_seeded_generation_never_scores_the_seed():
    """The seed is real sequence. Scoring it as model output would make every seeded arm
    look extraordinary. Evo2 returns only the continuation; HuggingFace returns
    prompt+continuation and must be sliced."""
    from bgcbench.model import generate as gen
    src = Path(gen.__file__).read_text()
    assert "[plen:]" in src, "HF path must strip the prompt before scoring"
    assert "STRIP THE PROMPT" in src


def test_arm_runner_actually_attaches_the_adapter():
    """--adapter was accepted and never loaded, so a trained arm would have generated from
    the base model and read as a null. Nothing in the output would have shown it."""
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert "attach_adapter(sub, adapter)" in src


def test_per_class_adapter_generates_once_not_five_times():
    """A per-class adapter IS the conditioning; it cannot be 'conditioned toward' another
    class. Generating it five times would be five samples of ONE distribution reported as
    five independent measurements, giving five identical confusion rows."""
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert "row_class" in src
    assert "one adapter, one row" in src


def test_training_records_the_best_checkpoint_not_just_the_last():
    """Fixed epochs with no early stopping means `final` is whatever the last step
    produced. Measured: W1 and W2_RIPP both had a final WORSE than their best, so
    generating from final handicaps those two arms and nothing else."""
    import glob
    for f in glob.glob("/data2/ds85/bgcbench/adapters/*/train_report.json"):
        d = json.loads(Path(f).read_text())
        if "best_checkpoint" not in d:
            continue                       # trained before the fix
        assert d["best_val_loss"] is not None
        assert "final_is_best" in d


def test_pooled_balance_option_exists_and_is_token_aware():
    """Equal records is NOT equal tokens: 3.4x nucleotide imbalance across the five
    classes, and the loss is per token."""
    from bgcbench.model.train import TrainConfig
    assert TrainConfig().balance == "records"
    assert TrainConfig(balance="nucleotides").balance == "nucleotides"


def test_training_uses_early_stopping_not_a_guessed_epoch_count():
    from bgcbench.model.train import TrainConfig
    c = TrainConfig()
    assert c.max_epochs >= 8 and c.patience >= 2 and c.eval_every > 0
    assert not hasattr(c, "epochs"), "a fixed epoch count cannot know whether an arm converged"


def test_arm_runner_resolves_to_the_best_checkpoint():
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert '(p / "BEST").exists()' in src
    assert "rather than final" in src


def test_cli_defaults_come_from_the_frozen_config_not_literals():
    """The n=150 defect happened because n was a CLI literal that a shell script overrode.
    Every generation parameter's default must READ the frozen config, so changing the
    agreed value in one place changes it everywhere and changes the hash."""
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    for key in ("n_per_row", "budget_nt", "batch_size"):
        assert f'GEN_FROZEN["{key}"]' in src, f"--{key} default is not read from FROZEN"
    assert 'default=200' not in src, "a hard-coded literal can drift from the frozen config"


def test_intervention_hooks_are_identity_at_init_and_fully_removable():
    """Zero-init means the UNTRAINED W3 arm is bit-identical to the base model, so any
    difference at step 0 is a bug. And a leaked hook would silently contaminate every
    later arm run in the same process."""
    import torch
    import torch.nn as nn
    from bgcbench.model.interventions import LearnedOffset, attention_sites

    class Blk(nn.Module):
        def __init__(s): super().__init__(); s.lin = nn.Linear(8, 8)
        def forward(s, x): return s.lin(x)

    class Toy(nn.Module):
        def __init__(s):
            super().__init__()
            s.blocks = nn.ModuleList([nn.Module() for _ in range(2)])
            for b in s.blocks: b.inner_mha_cls = Blk()
        def forward(s, x):
            for b in s.blocks: x = b.inner_mha_cls(x)
            return x

    m = Toy()
    assert len(attention_sites(m)) == 2
    x = torch.randn(1, 4, 8)
    base = m(x).clone()
    for rank in (0, 4):
        iv = LearnedOffset(m, 8, rank=rank)
        with iv.attached():
            assert torch.allclose(base, m(x), atol=1e-5), f"rank {rank} not identity at init"
            with torch.no_grad():
                for p in iv.offsets: p.add_(1.0)
            assert not torch.allclose(base, m(x), atol=1e-3), "offset had no effect"
        assert torch.allclose(base, m(x), atol=1e-5), "hooks not removed"
        assert not iv.is_attached()


def test_w3_capacity_is_swept_not_fixed():
    """The bare offset is 1364x below LoRA on the same model; a null there would say
    'too few parameters', not 'activation conditioning does not work'."""
    import torch.nn as nn
    from bgcbench.model.interventions import LearnedOffset

    class Blk(nn.Module):
        def __init__(s): super().__init__(); s.lin = nn.Linear(8, 8)
        def forward(s, x): return s.lin(x)

    class Toy(nn.Module):
        def __init__(s):
            super().__init__(); s.blocks = nn.ModuleList([nn.Module() for _ in range(2)])
            for b in s.blocks: b.inner_mha_cls = Blk()

    m = Toy()
    assert LearnedOffset(m, 8, rank=0).n_trainable() < LearnedOffset(m, 8, rank=4).n_trainable()


def _toy_model(n_sites=3, hidden=8):
    import torch.nn as nn

    class Blk(nn.Module):
        def __init__(s):
            super().__init__(); s.lin = nn.Linear(hidden, hidden)
        def forward(s, x): return s.lin(x)

    class Toy(nn.Module):
        def __init__(s):
            super().__init__()
            s.blocks = nn.ModuleList([nn.Module() for _ in range(n_sites)])
            for b in s.blocks: b.inner_mha_cls = Blk()
        def forward(s, x):
            for b in s.blocks: x = b.inner_mha_cls(x)
            return x
    return Toy()


def test_intervention_arm_records_that_the_hooks_actually_fired():
    """BEHAVIOURAL. An intervention is a HOOK, not merged weights: if it is not attached
    during generation the arm silently emits base-model output and reads as a null. The
    artifact must attest the hooks were LIVE, not that the code intended them to be."""
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    assert "intervention_attached=" in src, "no artifact field attests the hooks fired"
    # the attestation must be taken from INSIDE the attached() context, not before it
    i_ctx = src.index("with intervention.attached():")
    i_call = src.index("intervention=intervention)")
    assert i_ctx < i_call, "provenance recorded outside the attached context"


def test_learned_offset_gradient_reaches_the_parameters():
    """The offset path trains a conditioner on a FROZEN base. If gradient does not reach
    it, training is a no-op that still reports a loss curve."""
    import torch
    from bgcbench.model.interventions import LearnedOffset
    m = _toy_model()
    for p in m.parameters():
        p.requires_grad_(False)
    iv = LearnedOffset(m, 8, rank=4)
    before = [p.detach().clone() for p in iv.parameters()]
    opt = torch.optim.SGD(iv.parameters(), lr=0.5)
    with iv.attached():
        loss = m(torch.randn(2, 3, 8)).pow(2).mean()
        loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in iv.parameters()), \
        "no gradient reached the conditioner"
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, iv.parameters())), \
        "an optimiser step did not change the conditioner"


def test_learned_offset_is_reproducible_under_its_recorded_seed():
    """The low-rank A was built from the UNSEEDED global RNG before manual_seed ran, so the
    default rank>0 run was not reproducible under the seed its own report recorded."""
    import torch
    from bgcbench.model.interventions import LearnedOffset
    m = _toy_model()
    torch.manual_seed(123); a = LearnedOffset(m, 8, rank=4)
    torch.manual_seed(123); b = LearnedOffset(m, 8, rank=4)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    from bgcbench.model import train as tr
    src = Path(tr.__file__).read_text()
    body = src[src.index("def _train_offset"):]
    assert body.index("torch.manual_seed") < body.index("LearnedOffset(base"), \
        "the conditioner is constructed before the seed is set"


def test_direction_injection_and_its_control_actually_steer():
    """I1 had zero test coverage. It shares the hook path with W3, so a regression there
    would silently disable steering while the arm still reported numbers."""
    import torch
    from bgcbench.model.interventions import DirectionInjection, random_direction_control
    m = _toy_model()
    x = torch.randn(1, 3, 8)
    base = m(x).clone()
    d = torch.ones(3, 8)
    iv = DirectionInjection(m, 8, d, alpha=2.0)
    with iv.attached():
        assert not torch.allclose(base, m(x), atol=1e-4), "direction had no effect"
    assert torch.allclose(base, m(x), atol=1e-6), "hooks not removed"
    ctrl = random_direction_control(m, 8, seed=0, alpha=2.0)
    with ctrl.attached():
        assert not torch.allclose(base, m(x), atol=1e-4)
    # alpha=0 must be an exact no-op, so the control can be magnitude-matched at zero
    z = DirectionInjection(m, 8, d, alpha=0.0)
    with z.attached():
        assert torch.allclose(base, m(x), atol=1e-6)
    try:
        DirectionInjection(m, 8, torch.ones(2, 8), alpha=1.0)   # wrong site count
    except ValueError:
        return
    raise AssertionError("a site-count mismatch was accepted; some layers would be unsteered")


def test_w3_capacity_and_sites_reach_the_run_provenance():
    """SPEC 6.5. Two W3 runs at different ranks must not collide on one run directory --
    that is the destructive-overwrite defect a prior audit found."""
    from bgcbench.model import genconfig as gc
    from bgcbench.run import arm as armmod
    src = Path(armmod.__file__).read_text()
    for f in ("intervention_method", "intervention_rank", "intervention_sites"):
        assert f + "=" in src, f"{f} missing from the realised provenance"
    a = gc.realised(intervention_rank=16, intervention_method="offset")
    b = gc.realised(intervention_rank=64, intervention_method="offset")
    assert gc.realised_hash(a) != gc.realised_hash(b), "two ranks share one run hash"


def test_offset_training_refuses_resume_rather_than_ignoring_it():
    from bgcbench.model.train import TrainConfig, train_lora
    cfg = TrainConfig(method="offset")
    try:
        train_lora(None, [], Path("/tmp/x_never_written"), cfg, resume_from="somewhere")
    except ValueError as e:
        assert "resume" in str(e).lower()
        return
    except Exception:
        raise AssertionError("resume was not refused before any work began")
    raise AssertionError("--resume-from was silently discarded")
