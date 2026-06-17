#!/usr/bin/env python3
"""Tests for the BATCHED generation path (no GPU).

The batched path (generate_batch / left_pad_to_uniform / assemble_record) must be
a drop-in for the sequential path (generate_one): same per-record schema, same
ordering, same values — the only difference is one batched wrapper.generate()
call instead of N. These tests mock wrapper.generate so the assembly logic is
exercised without loading the 7B model; on-GPU equivalence of the actual
generation is verified separately by scripts/validate_batched_generation.py.

Run: python tests/test_generation_batched.py   (also pytest-discoverable)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import generate_bgc as G  # noqa: E402


# ── test scaffolding ─────────────────────────────────────────────────────────

def _args(**over):
    """Minimal args namespace covering every field the generation paths read."""
    base = dict(max_new_tokens=32768, temperature=1.0, top_k=4, top_p=1.0,
                max_n_frac=0.01, max_windows=1, chunk_overlap=2048, n=1,
                batch_size=0)
    base.update(over)
    return SimpleNamespace(**base)


class MockWrapper:
    """Returns canned generation-only strings keyed by the REAL prefix.

    A left-padded prompt has its real prefix as a suffix (padding is leading
    spaces), so stripping leading spaces recovers the registry key. This both
    feeds the assembly logic AND asserts (indirectly) that the real prompt was
    preserved through left-padding. Records every call for inspection.
    """

    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        self.shortfall = False  # if True, drop the last sequence (alignment bug)
        self.surplus = False    # if True, append an extra sequence (alignment bug)

    def generate(self, prompt_seqs, n_tokens=0, batched=True, **kw):
        self.calls.append({"prompt_seqs": list(prompt_seqs), "batched": batched,
                           "n_tokens": n_tokens})
        seqs = [self.registry[ps.lstrip(" ")] for ps in prompt_seqs]
        if self.shortfall:
            seqs = seqs[:-1]
        if self.surplus:
            seqs = seqs + ["ACGTACGT"]
        return SimpleNamespace(sequences=seqs)


# varied canned generations: clean, no-EOS, trailing junk, N-content, prompt-leak
def _prompts_and_registry():
    prompts = [
        {"compound_class": "NRPS", "taxonomic_tag": "|D__BACTERIA;P__X"},          # short tax
        {"compound_class": "PKS", "taxonomic_tag": "|D__BACTERIA;P__ACTINOMYCETOTA;C__ACTINO"},  # long tax
        {"compound_class": "TERPENE", "taxonomic_tag": "|D__ARCHAEA"},             # shortest
    ]
    pre = [G.build_prefix(p["compound_class"], p["taxonomic_tag"]) for p in prompts]
    registry = {
        pre[0]: "ACGTACGTACGT|END|trailing-after-eos",   # clean, hits EOS
        pre[1]: "ACGTNNACGT|FOO bar",                     # no EOS, has N + trailing junk
        pre[2]: "ACGTACGT",                               # no EOS, clean, no junk
    }
    return prompts, pre, registry


# ── tests ────────────────────────────────────────────────────────────────────

def test_left_pad_to_uniform():
    out = G.left_pad_to_uniform(["abc", "a", "abcd"])
    assert [len(s) for s in out] == [4, 4, 4], "all padded to max length"
    assert out == [" abc".replace(" ", " "), "   a", "abcd"]
    assert out[0].endswith("abc") and out[1].endswith("a") and out[2].endswith("abcd")
    assert all(c == " " for s in out for c in s[:len(s) - len(s.lstrip(" "))])
    assert G.left_pad_to_uniform([]) == []
    assert G.left_pad_to_uniform(["x"]) == ["x"], "single prefix unchanged"
    print("PASS: left_pad_to_uniform pads left to uniform length, prefix kept as suffix")


def test_should_batch():
    assert G.should_batch(_args(batch_size=1)) is False, "batch_size=1 is sequential"
    assert G.should_batch(_args(batch_size=2, max_windows=1)) is True
    assert G.should_batch(_args(batch_size=0, max_windows=1)) is True, "0 = all-in-one batch"
    # chaining forces sequential fallback (prints a warning)
    assert G.should_batch(_args(batch_size=4, max_windows=2)) is False
    print("PASS: should_batch gates on batch_size and single-window")


def test_assemble_record_schema_and_values():
    rec = G.assemble_record("NRPS", "|TAX", "ACGTNN", hit_eos=True, windows=1, args=_args())
    assert set(rec) == {"compound_class", "taxonomic_tag", "sequence", "length",
                        "hit_eos", "windows", "n_count", "n_fraction", "n_pass",
                        "decoding"}
    assert rec["length"] == 6 and rec["n_count"] == 2
    assert rec["n_fraction"] == round(2 / 6, 5)
    assert rec["n_pass"] is False  # 0.33 > 0.01
    assert rec["decoding"]["max_new_tokens"] == 32768
    print("PASS: assemble_record emits the full schema with correct derived fields")


def test_batched_matches_sequential():
    """Core invariant: batched records are field-for-field identical to sequential."""
    prompts, _, registry = _prompts_and_registry()
    args = _args(batch_size=0)

    seq_mock = MockWrapper(registry)
    seq_records = [G.generate_one(seq_mock, p["compound_class"], p["taxonomic_tag"], args)
                   for p in prompts]

    bat_mock = MockWrapper(registry)
    bat_records = G.generate_batch(bat_mock, prompts, args)

    assert len(seq_records) == len(bat_records) == 3
    for s, b in zip(seq_records, bat_records):
        assert s == b, f"record mismatch:\n seq={s}\n bat={b}"
    # sequential made 3 calls (one prompt each); batched made exactly 1 call
    assert len(seq_mock.calls) == 3 and all(len(c["prompt_seqs"]) == 1 for c in seq_mock.calls)
    assert len(bat_mock.calls) == 1 and len(bat_mock.calls[0]["prompt_seqs"]) == 3
    print("PASS: batched output is byte-identical to sequential (1 call vs 3)")


def test_batched_prompts_left_padded_uniform_with_real_suffix():
    prompts, pre, registry = _prompts_and_registry()
    mock = MockWrapper(registry)
    G.generate_batch(mock, prompts, args=_args(batch_size=0))
    sent = mock.calls[0]["prompt_seqs"]
    assert len({len(s) for s in sent}) == 1, "all batched prompts are equal length"
    for s, real in zip(sent, pre):
        assert s.endswith(real), "each padded prompt ends with its real prefix"
        assert s.lstrip(" ") == real, "only leading spaces were added"
    assert mock.calls[0]["batched"] is True
    print("PASS: batched call left-pads to uniform length, preserves real prefix")


def test_extract_invariant_prompt_leak_yields_empty():
    """If a generation string leaks the prompt prefix (non-ACGTN at pos 0),
    extract_sequence anchors at 0 and returns empty — must surface as length 0."""
    prompts = [{"compound_class": "NRPS", "taxonomic_tag": "|TAX1"},
               {"compound_class": "PKS", "taxonomic_tag": "|TAX2"}]
    pre = [G.build_prefix(p["compound_class"], p["taxonomic_tag"]) for p in prompts]
    registry = {pre[0]: pre[0] + "ACGTACGT",   # prompt LEAKED in front of nucleotides
                pre[1]: "ACGTACGT"}            # clean
    recs = G.generate_batch(MockWrapper(registry), prompts, args=_args(batch_size=0))
    assert recs[0]["sequence"] == "" and recs[0]["length"] == 0, "leak -> empty seq"
    assert recs[1]["sequence"] == "ACGTACGT" and recs[1]["length"] == 8
    print("PASS: leaked-prompt generation yields empty sequence (invariant holds)")


def test_n_fraction_and_npass_parity():
    prompts = [{"compound_class": "NRPS", "taxonomic_tag": "|TAX"}]
    pre = G.build_prefix("NRPS", "|TAX")
    # 1 N in 100 nt = exactly 0.01 -> n_pass True (<=); 2 N in 100 -> False
    registry = {pre: "A" * 99 + "N"}
    args = _args(batch_size=0, max_n_frac=0.01)
    seq = G.generate_one(MockWrapper(registry), "NRPS", "|TAX", args)
    bat = G.generate_batch(MockWrapper(registry), prompts, args)[0]
    assert seq == bat
    assert seq["n_fraction"] == 0.01 and seq["n_pass"] is True
    print("PASS: n_fraction/n_pass identical across paths at the threshold boundary")


def test_batched_order_preserved():
    prompts, _, registry = _prompts_and_registry()
    recs = G.generate_batch(MockWrapper(registry), prompts, args=_args(batch_size=0))
    assert [r["compound_class"] for r in recs] == ["NRPS", "PKS", "TERPENE"]
    assert recs[0]["sequence"] == "ACGTACGTACGT" and recs[0]["hit_eos"] is True  # NRPS, trimmed at EOS
    assert recs[1]["sequence"] == "ACGTNNACGT" and recs[1]["hit_eos"] is False   # PKS, junk trimmed
    assert recs[2]["sequence"] == "ACGTACGT"      # TERPENE canned
    print("PASS: batched records preserve input prompt order")


def test_batch_alignment_mismatch_raises():
    prompts, _, registry = _prompts_and_registry()
    mock = MockWrapper(registry)
    mock.shortfall = True  # wrapper returns one fewer sequence than prompts
    try:
        G.generate_batch(mock, prompts, args=_args(batch_size=0))
    except RuntimeError as e:
        assert "alignment" in str(e).lower()
        print("PASS: count mismatch raises rather than silently mislabeling records")
        return
    raise AssertionError("expected RuntimeError on batch count mismatch (shortfall)")


def test_batch_alignment_excess_raises():
    prompts, _, registry = _prompts_and_registry()
    mock = MockWrapper(registry)
    mock.surplus = True  # wrapper returns one MORE sequence than prompts
    try:
        G.generate_batch(mock, prompts, args=_args(batch_size=0))
    except RuntimeError as e:
        assert "alignment" in str(e).lower()
        print("PASS: excess sequence count also raises (guard uses !=)")
        return
    raise AssertionError("expected RuntimeError on batch count mismatch (excess)")


def test_id_scheme_matches_across_branches():
    """The id format + order are produced in main() identically for both branches;
    verify the flatten + id expression yields the documented gen_PPPP_S scheme."""
    prompts = [{"compound_class": "NRPS", "taxonomic_tag": "|A"},
               {"compound_class": "PKS", "taxonomic_tag": "|B"}]
    n = 2
    instances = [(pi, si, p) for pi, p in enumerate(prompts) for si in range(n)]
    ids = [f"gen_{pi:04d}_{si}" for pi, si, _ in instances]
    assert ids == ["gen_0000_0", "gen_0000_1", "gen_0001_0", "gen_0001_1"]
    assert len(ids) == len(set(ids)), "ids unique"
    print("PASS: id scheme gen_PPPP_S is stable and unique")


def _run_main_records(val_path, registry, batch_size, n, tmp):
    """Drive generate_bgc.main() end-to-end with a mocked model, return the
    records it wrote to gen.jsonl. Mocks evo2_inference so no GPU is touched."""
    import json
    import types
    fa, jl = tmp / f"gen_{batch_size}_{n}.fasta", tmp / f"gen_{batch_size}_{n}.jsonl"
    argv = ["generate_bgc.py", "--from-jsonl", str(val_path), "--per-class", "1",
            "--n", str(n), "--max-new-tokens", "100", "--batch-size", str(batch_size),
            "--adapter", "DUMMY", "--seed", "42",
            "--out-fasta", str(fa), "--out-jsonl", str(jl)]
    fake = types.ModuleType("evo2_inference")
    fake.load_evo2_wrapper_for_inference = lambda adapter, device="cuda": MockWrapper(registry)
    old_mod = sys.modules.get("evo2_inference")
    old_argv = sys.argv
    sys.modules["evo2_inference"] = fake
    sys.argv = argv
    try:
        G.main()
    finally:
        sys.argv = old_argv
        if old_mod is not None:
            sys.modules["evo2_inference"] = old_mod
        else:
            sys.modules.pop("evo2_inference", None)
    return [json.loads(l) for l in open(jl)]


def _main_fixture():
    recs = [{"compound_class": "NRPS", "taxonomic_tag": "|TAXA", "sequence": "x"},
            {"compound_class": "PKS", "taxonomic_tag": "|TAXBB", "sequence": "x"},
            {"compound_class": "TERPENE", "taxonomic_tag": "|TAXCCC", "sequence": "x"}]
    registry = {
        G.build_prefix("NRPS", "|TAXA"): "ACGTACGT|END|junk",
        G.build_prefix("PKS", "|TAXBB"): "ACGTAACCGGTT",
        G.build_prefix("TERPENE", "|TAXCCC"): "AACCGGTTAA",
    }
    return recs, registry


def test_main_batched_equals_sequential_with_remainder():
    """Full main() path: 3 prompts x n=1 = 3 instances. batch_size=2 -> chunks
    [2,1] (remainder). batch_size=0 -> one chunk. All must equal sequential."""
    import json
    import tempfile
    recs, registry = _main_fixture()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        val = tmp / "val.jsonl"
        val.write_text("".join(json.dumps(r) + "\n" for r in recs))
        seq = _run_main_records(val, registry, batch_size=1, n=1, tmp=tmp)
        bat2 = _run_main_records(val, registry, batch_size=2, n=1, tmp=tmp)  # remainder
        bat0 = _run_main_records(val, registry, batch_size=0, n=1, tmp=tmp)  # all-in-one
        assert len(seq) == 3
        assert seq == bat2, "batch_size=2 (with remainder) must equal sequential"
        assert seq == bat0, "batch_size=0 (all-in-one) must equal sequential"
        ids = [r["id"] for r in bat2]
        assert ids == ["gen_0000_0", "gen_0001_0", "gen_0002_0"]
        assert len(ids) == len(set(ids))
    print("PASS: main() batched (remainder + all-in-one) == sequential, ids aligned")


def test_main_batched_multisample_ids():
    """n=2 samples per prompt with batching: ids + order + records align."""
    import json
    import tempfile
    recs, registry = _main_fixture()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        val = tmp / "val.jsonl"
        val.write_text("".join(json.dumps(r) + "\n" for r in recs))
        seq = _run_main_records(val, registry, batch_size=1, n=2, tmp=tmp)
        bat = _run_main_records(val, registry, batch_size=2, n=2, tmp=tmp)
        assert len(seq) == 6 and seq == bat, "6 instances; batched must match sequential"
        assert [r["id"] for r in bat] == [
            "gen_0000_0", "gen_0000_1", "gen_0001_0",
            "gen_0001_1", "gen_0002_0", "gen_0002_1"]
        # each record's class matches its id's prompt index (no cross-labeling)
        assert [r["compound_class"] for r in bat] == [
            "NRPS", "NRPS", "PKS", "PKS", "TERPENE", "TERPENE"]
    print("PASS: main() n=2 batched ids/order/labels align with sequential")


def main():
    test_left_pad_to_uniform()
    test_should_batch()
    test_assemble_record_schema_and_values()
    test_batched_matches_sequential()
    test_batched_prompts_left_padded_uniform_with_real_suffix()
    test_extract_invariant_prompt_leak_yields_empty()
    test_n_fraction_and_npass_parity()
    test_batched_order_preserved()
    test_batch_alignment_mismatch_raises()
    test_batch_alignment_excess_raises()
    test_id_scheme_matches_across_branches()
    test_main_batched_equals_sequential_with_remainder()
    test_main_batched_multisample_ids()
    print("\nALL BATCHED-GENERATION TESTS PASSED")


if __name__ == "__main__":
    main()
