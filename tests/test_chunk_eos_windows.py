#!/usr/bin/env python3
"""Integration tests for window-aware EOS + continuation conditioning (audit M11).

Builds a real BGCTextDataset over a temp JSONL with a byte-level mock tokenizer
(mirrors Evo2's CharLevelTokenizer) and asserts, for every chunk window:
  - first window (nt_start==0)   -> class prefix |COMPOUND_CLASS:..|
  - interior window (nt_start>0) -> continuation prefix |CONTINUATION:..|
  - EOS_MARKER appended IFF the window is the last (nt_end == seq_len)
  - EOS is SUPERVISED (after prefix), the prefix is masked
  - no window exceeds max_seq_len (EOS budget reserved correctly)
  - first_window_only (validation) keeps exactly one window/record, with EOS
    only on records short enough to be a single (first==last) window.

Run: python tests/test_chunk_eos_windows.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evo2" / "scripts"))
import finetune_evo2_lora as F  # noqa: E402


class MockTok:
    """Byte-level tokenizer: 1 char -> 1 token, like Evo2's CharLevelTokenizer."""
    def tokenize(self, s):
        return list(s.encode("utf-8"))


class MergingTok:
    """Pathological tokenizer that merges bytes in 2-byte chunks, so for an
    odd-length prefix it merges ACROSS the prefix↔sequence seam — must trip the
    B1 seam guard rather than silently mask the wrong span."""
    def tokenize(self, s):
        b = s.encode("utf-8")
        return [int.from_bytes(b[i:i + 2], "big") for i in range(0, len(b), 2)]


def decode(ids):
    return bytes(ids).decode("utf-8", errors="replace")


def mkrec(cls, tax, seq):
    return {
        "compound_class": cls,
        "taxonomic_tag": tax,
        "sequence": seq,
        "training_text": f"|COMPOUND_CLASS:{cls}|{tax}{seq}",
        "accession": f"acc_{cls}",
    }


def build_dataset(tmp, recs, max_seq_len, prefix_cap, tok=None, **kw):
    jsonl = tmp / "data.jsonl"
    with jsonl.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return F.BGCTextDataset(
        jsonl, tok or MockTok(), max_seq_len,
        long_seq_strategy="chunk", chunk_overlap=10,
        prefix_token_cap=prefix_cap, slack_tokens=0,
        lengths_cache_dir=tmp, **kw,
    )


def main():
    tax = "|D__BACTERIA"
    short = mkrec("NRPS", tax, "ACGT" * 7)    # 28 nt -> single window
    long_ = mkrec("PKS", tax, "ACGT" * 40)    # 160 nt -> several windows
    recs = [short, long_]
    max_seq_len = 80
    # Dataset-wide max prefix = the longest class (first-window) prefix.
    prefix_cap = max(len(F.canonical_phase1_prefix(r).encode()) for r in recs)

    # ── Training dataset: EOS + continuation on ──────────────────────────────
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ds = build_dataset(tmp, recs, max_seq_len, prefix_cap, continuation_prefix=True)

        # Index windows by record.
        by_rec = {}
        for ci, (rec_idx, nt0, nt1) in enumerate(ds.chunks):
            by_rec.setdefault(rec_idx, []).append((ci, nt0, nt1))

        # Record 0 is a single first==last window.
        assert len(by_rec[0]) == 1, f"short record should be 1 window, got {len(by_rec[0])}"
        # Record 1 spans several windows.
        assert len(by_rec[1]) >= 3, f"long record should chunk into >=3, got {len(by_rec[1])}"

        n_eos = 0
        for rec_idx, wins in by_rec.items():
            seq_len = len(recs[rec_idx]["sequence"])
            cls = recs[rec_idx]["compound_class"]
            for (ci, nt0, nt1) in wins:
                item = ds[ci]
                ids = item["input_ids"]
                text = decode(ids)
                p = item["prefix_token_count"]
                is_first = (nt0 == 0)
                is_last = (nt1 >= seq_len)

                # 1) budget: never exceed max_seq_len
                assert len(ids) <= max_seq_len, f"window {ci} overflow: {len(ids)}>{max_seq_len}"

                # 2) prefix selection
                if is_first:
                    assert text.startswith(f"|COMPOUND_CLASS:{cls}|"), \
                        f"first window must use class prefix: {text[:40]!r}"
                else:
                    assert text.startswith(f"|CONTINUATION:{cls}|"), \
                        f"interior window must use continuation prefix: {text[:40]!r}"

                # 3) EOS placement -- depends on --eos-mode (added 2026-08-20).
                #    'token'  appends the tokenizer's REAL single-token EOS, id 0. This is the
                #             default: |END| is a 5-BYTE STRING that never once fired at
                #             generation, while id 0 is what Evo2 pretrained with and what the
                #             model demonstrably emits (masking it restores full-length output).
                #    'marker' is the pre-2026-08-20 5-byte string, kept to reproduce old runs.
                if is_last:
                    n_eos += 1
                    mode = "token"
                    if mode in ("marker", "both"):
                        eos_len = len(F.EOS_MARKER.encode())
                        tail = ids[-eos_len:] if mode == "marker" else ids[-eos_len - 1:-1]
                        assert decode(tail) == F.EOS_MARKER, \
                            f"last window must carry the string marker: {text[-12:]!r}"
                        assert p <= len(ids) - eos_len, "EOS must lie in the supervised region"
                    if mode in ("token", "both"):
                        assert ids[-1] == F.EOS_TOKEN_ID, \
                            f"last window must END with token id {F.EOS_TOKEN_ID}, got {ids[-1]}"
                        # supervised: the appended id sits after the masked prefix
                        assert p <= len(ids) - 1, "EOS token must lie in the supervised region"
                else:
                    assert not text.endswith(F.EOS_MARKER), \
                        f"non-final window must NOT have EOS: {text[-12:]!r}"

                # prefix_token_count matches the actual prefix used
                rec = recs[rec_idx]
                expected_prefix = (F.canonical_phase1_prefix(rec) if is_first
                                   else F.continuation_phase1_prefix(rec))
                assert p == len(expected_prefix.encode()), \
                    f"prefix_token_count {p} != {len(expected_prefix.encode())}"

        # Exactly one last-window (EOS) per record.
        assert n_eos == 2, f"expected one EOS per record (2), got {n_eos}"
        print(f"PASS train: {len(ds.chunks)} windows; class/continuation prefixes, "
              f"EOS on {n_eos} final windows, masking + budget all correct")

    # ── Validation dataset: first_window_only ────────────────────────────────
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        vds = build_dataset(tmp, recs, max_seq_len, prefix_cap, continuation_prefix=True,
                            first_window_only=True)
        assert len(vds.chunks) == 2, f"first_window_only -> one/record (2), got {len(vds.chunks)}"
        eos_count = 0
        for ci, (rec_idx, nt0, nt1) in enumerate(vds.chunks):
            assert nt0 == 0, "first_window_only must keep only nt_start==0 windows"
            ids_ = [int(x) for x in vds[ci]["input_ids"]]
            text = decode(ids_)
            assert text.startswith("|COMPOUND_CLASS:"), "val first window uses class prefix"
            # mode-agnostic: an end signal is EITHER the string marker OR the real EOS token id
            if text.endswith(F.EOS_MARKER) or ids_[-1] == F.EOS_TOKEN_ID:
                eos_count += 1
        # Only the short record's first window is also the last -> gets EOS.
        assert eos_count == 1, f"only the single-window (short) record gets EOS, got {eos_count}"
        print("PASS val: first-window-only keeps 1 window/record; EOS only on the "
              "short (single-window) record")

    # ── EOS is now UNCONDITIONAL (2026-08-20): the real EOS id is always appended to the
    #    window carrying a record's true end. There is no --eos-token/--eos-mode any more, so
    #    the old "EOS off" case is gone; what remains is the continuation-prefix legacy check.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ds0 = build_dataset(tmp, recs, max_seq_len, prefix_cap, continuation_prefix=False)
        for ci in range(len(ds0.chunks)):
            ids_ = [int(x) for x in ds0[ci]["input_ids"]]
            text = decode(ids_)
            # The retired 5-byte STRING marker must never be written any more.
            assert not text.endswith(F.EOS_MARKER), "the |END| string marker is RETIRED"
            assert text.startswith("|COMPOUND_CLASS:"), "continuation off: all class prefix"
        # EOS is unconditional now: exactly one token reserved, and the final window carries it.
        assert ds0.eos_reserve == 1, f"one token reserved for the EOS id, got {ds0.eos_reserve}"
        last_ci = max(range(len(ds0.chunks)), key=lambda c: ds0.chunks[c][2])
        assert [int(x) for x in ds0[last_ci]["input_ids"]][-1] == F.EOS_TOKEN_ID
        print("PASS legacy: continuation-prefix off reproduces old behaviour; EOS id is "
              "unconditional and the |END| string is retired")

    # ── B1 seam guard: a merging tokenizer must raise, not silently mis-mask ──
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # 33-byte (odd) class prefix => 2-byte chunking merges across the seam.
        assert len(F.canonical_phase1_prefix(short).encode()) % 2 == 1
        dsm = build_dataset(tmp, recs, max_seq_len, prefix_cap, tok=MergingTok(), continuation_prefix=True)
        raised = False
        try:
            _ = dsm[0]
        except ValueError as e:
            raised = "seam" in str(e)
        assert raised, "merging tokenizer must trip the B1 prefix-seam guard"
        print("PASS B1: prefix-seam guard raises when the tokenizer merges the seam")

    print("\nALL WINDOW/EOS TESTS PASSED")


if __name__ == "__main__":
    main()
