#!/usr/bin/env python3
"""Tests for the operational/integrity hardening (audit m3, m1, A2 wandb).

Run: python tests/test_hardening.py
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import finetune_evo2_lora as F  # noqa: E402


def test_m3_checkpoint_rotation():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # numeric periodic + emergency (oom/interrupted/final) + best
        for name in ["step_100", "step_200", "step_300", "step_400", "step_500",
                     "step_150_oom", "step_250_interrupted", "step_350_oom",
                     "step_450_final", "best"]:
            (root / name).mkdir()
        F.cleanup_old_checkpoints(root, keep_last=2, keep_special=2)
        kept = {p.name for p in root.iterdir()}
        # newest 2 numeric kept; older deleted
        assert "step_500" in kept and "step_400" in kept
        assert not {"step_100", "step_200", "step_300"} & kept, kept
        # newest 2 emergency kept (by step number: 150,250,350,450 -> 350,450)
        assert "step_450_final" in kept and "step_350_oom" in kept
        assert not {"step_150_oom", "step_250_interrupted"} & kept, kept
        # best/ always preserved
        assert "best" in kept
        print("PASS m3: rotates step_N and step_N_{oom,interrupted,final}; keeps best/")

    # keep_special=0 -> emergency dirs are NOT rotated (all kept)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for name in ["step_100", "step_200", "step_300",
                     "step_150_oom", "step_250_oom"]:
            (root / name).mkdir()
        F.cleanup_old_checkpoints(root, keep_last=1, keep_special=0)
        kept = {p.name for p in root.iterdir()}
        assert "step_300" in kept and "step_200" not in kept
        assert {"step_150_oom", "step_250_oom"} <= kept, "keep_special=0 keeps all emergency"
        print("PASS m3: keep_special=0 disables emergency rotation")


def test_m1_data_fingerprint():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        train = tmp / "train.jsonl"
        val = tmp / "val.jsonl"
        train.write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
        val.write_text('{"b":1}\n')
        out = tmp / "run"
        out.mkdir()
        args = SimpleNamespace(train=train, val=val)

        # first run writes the fingerprint
        F.save_data_fingerprint(args, out)
        fp_path = out / "data_fingerprint.json"
        saved = json.loads(fp_path.read_text())
        assert "sha256_full" in saved["train"] and saved["train"]["lines"] == 3
        orig_hash = saved["train"]["sha256_full"]

        # resume on UNCHANGED data -> no mismatch warning
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            F.save_data_fingerprint(args, out)
        assert "MISMATCH" not in buf.getvalue(), "unchanged data should not warn"

        # resume on CHANGED data -> loud mismatch warning, original NOT overwritten
        train.write_text('{"a":1}\n{"a":2}\n{"a":3}\n{"a":4}\n')  # 4 lines now
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            F.save_data_fingerprint(args, out)
        assert "MISMATCH" in buf.getvalue(), "changed data must warn"
        assert json.loads(fp_path.read_text())["train"]["sha256_full"] == orig_hash, \
            "fingerprint must NOT be overwritten on resume"
        print("PASS m1: full-file fingerprint detects data change on resume; no overwrite")


def test_a2_wandb_log_safe():
    class Boom:
        def log(self, *a, **k):
            raise RuntimeError("network down")
    # must not raise
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        F.wandb_log_safe(Boom(), {"x": 1}, step=5)
        F.wandb_log_safe(None, {"x": 1}, step=5)  # None run is a no-op
    assert "wandb.log failed" in buf.getvalue()
    print("PASS A2: wandb_log_safe swallows logging errors (never stalls training)")


def main():
    test_m3_checkpoint_rotation()
    test_m1_data_fingerprint()
    test_a2_wandb_log_safe()
    print("\nALL HARDENING TESTS PASSED")


if __name__ == "__main__":
    main()
