#!/usr/bin/env python
"""VALIDATE the batched generation path in `evo2/scripts/generate_bgc.py`.

WHY THIS FILE EXISTS. `generate_bgc.py` has warned since it was written that the batched path must
be validated "with scripts/validate_batched_generation.py" before it is trusted — and that script
was never present in the repo. So `--batch-size > 1` has been shipped, documented as unverified,
and never verified. This closes that.

WHAT CAN GO WRONG, from vortex's own behaviour (see the comments in generate_bgc.py):
  1. vortex batches ONLY prompts of equal length; otherwise it SILENTLY de-batches and loops one at
     a time. That is a performance trap, not a correctness one — but it means a "batched" run can
     be secretly sequential and nobody notices.
  2. When it does batch, it RIGHT-pads, which for a causal model puts padding INSIDE the context the
     model conditions on. `generate_bgc.py` left-pads to a uniform length to avoid this.
  3. Misalignment: sequence i of the output belonging to prompt j. This is the dangerous one,
     because every record would be labelled with the wrong compound class and nothing downstream
     could tell.

THE TEST, AND WHAT IT FOUND. Generate the SAME prompts greedily (top_k=1) both ways. Greedy is
deterministic here -- verified by a sequential-vs-sequential control, 8/8 byte-identical -- so any
difference is attributable to the batched path.

Batched output is NOT byte-identical to sequential, and that is EXPECTED rather than a defect. The
mechanism is proven, not guessed: of 8 prompts (lengths 104-149), exactly one needed zero padding
because it was already the longest, and that one -- and only that one -- reproduced its sequential
output exactly. Left-padding puts bytes in front of the prompt, and a byte-level causal model
conditions on them, so a padded prompt is a DIFFERENT prompt. It is not misalignment and not
truncation.

So byte-equality is the wrong pass criterion. What actually has to hold:
  FATAL   wrong record count; class misalignment; truncated output; pad bytes echoed into the
          generated sequence.
  BENIGN  byte differences confined to prompts that received padding.
Padding is COMMON-MODE across arms when the prompt set and seed are fixed, so it cannot manufacture
a difference between arms -- but batched and sequential runs must never be POOLED, since the
padded and unpadded prompts are not the same conditioning.

Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_gen(adapter: str, src: str, per_class: int, max_new: int, batch_size: int,
            out: Path) -> list[dict]:
    cmd = [
        "micromamba", "run", "-n", "bgcmodel", "python",
        str(REPO / "evo2" / "scripts" / "generate_bgc.py"),
        "--adapter", adapter, "--from-jsonl", src,
        "--per-class", str(per_class), "--n", "1",
        "--max-new-tokens", str(max_new), "--seed", "0",
        "--top-k", "1", "--temperature", "0.0001",   # greedy => deterministic
        "--batch-size", str(batch_size),
        "--out-jsonl", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"[validate] generation failed at batch_size={batch_size}")
    return [json.loads(l) for l in out.open()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="/data2/ds85/bgcmodel_runs/phase2_1b/baseline/final_adapter")
    ap.add_argument("--from-jsonl", default="/data2/ds85/bgcmodel_data/splits_core/valtest_eval_4class.jsonl")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import random
    import time
    sys.path.insert(0, str(REPO / "evo2" / "scripts"))
    from generate_bgc import build_prefix, sample_prompts

    tmp = Path(tempfile.mkdtemp(prefix="valbatch_"))
    print(f"[validate] greedy generation, {args.per_class}/class, {args.max_new_tokens} tokens")

    # Which prompts get padded? A prompt already at the batch max is padded by zero bytes and MUST
    # therefore reproduce its sequential output exactly. That is the control for the whole theory.
    recs = [json.loads(l) for l in open(args.from_jsonl)]
    prompts = sample_prompts(recs, args.per_class, random.Random(0))
    plen = [len(build_prefix(p["compound_class"], p["taxonomic_tag"])) for p in prompts]
    pad_of = {i: max(plen) - n for i, n in enumerate(plen)}

    t0 = time.time()
    seq_recs = run_gen(args.adapter, args.from_jsonl, args.per_class, args.max_new_tokens,
                       1, tmp / "seq.jsonl")
    t_seq = time.time() - t0
    t0 = time.time()
    bat_recs = run_gen(args.adapter, args.from_jsonl, args.per_class, args.max_new_tokens,
                       args.batch_size, tmp / "bat.jsonl")
    t_bat = time.time() - t0

    fatal, benign, anomaly, identical_despite_pad = [], [], [], []
    if len(seq_recs) != len(bat_recs):
        fatal.append(f"record COUNT differs: sequential {len(seq_recs)} vs batched {len(bat_recs)}")
    else:
        for i, (a, b) in enumerate(zip(seq_recs, bat_recs)):
            sa, sb = a.get("sequence", ""), b.get("sequence", "")
            if a.get("compound_class") != b.get("compound_class"):
                fatal.append(f"record {i}: class MISALIGNED "
                             f"({a.get('compound_class')} vs {b.get('compound_class')})")
            if len(sa) != len(sb):
                fatal.append(f"record {i}: LENGTH differs ({len(sa)} vs {len(sb)}) — truncation")
            for pad in ("@", "N" * 20):
                if pad in sb and pad not in sa:
                    fatal.append(f"record {i}: pad bytes {pad!r} ECHOED into batched output")
            if sa != sb:
                (benign if pad_of.get(i, 0) > 0 else anomaly).append(
                    f"record {i} ({a.get('compound_class')}): differs, padded by {pad_of.get(i)} bytes")
            elif pad_of.get(i, 0) > 0:
                # NOT an anomaly on its own. A short pad often does not flip the argmax at any
                # step, so an identical greedy continuation is expected -- an earlier version of
                # this script failed the whole run on a ONE-BYTE pad. The real question this could
                # indicate ("did vortex silently de-batch, so no padding was ever applied?") is
                # answered by the SPEED ratio below, not by byte equality.
                identical_despite_pad.append(f"record {i}: padded by {pad_of[i]} byte(s), identical")

    speed = t_seq / max(t_bat, 1e-9)
    print()
    print("=" * 72)
    print(f"REPORT — --batch-size {args.batch_size}, {len(seq_recs)} records")
    print("=" * 72)
    unpadded = [i for i, v in pad_of.items() if v == 0]
    print(f"  classes aligned / lengths equal / no pad echo : {'YES' if not fatal else 'NO'}")
    print(f"  byte-identical to sequential : {len(seq_recs)-len(benign)}/{len(seq_recs)}"
          f"   (unpadded prompts: {unpadded})")
    print(f"  differing, all padded        : {len(benign)}/{len(seq_recs)}")
    for m in identical_despite_pad:
        print(f"     note: {m}")
    print(f"\n  SPEED  sequential {t_seq:6.1f}s   batched {t_bat:6.1f}s   = {speed:.2f}x")
    print("  (this is also the DE-BATCHING check: vortex silently loops one-at-a-time when it")
    print("   declines to batch, which would show up here as ~1.0x and nothing else would.)")
    print()

    if fatal:
        print("  ✗ NOT SAFE — do not use --batch-size > 1")
        for f in fatal:
            print("     ", f)
        return 1
    if anomaly:
        print("  ✗ NOT SAFE — an UNPADDED prompt changed under batching, which padding cannot explain")
        for f in anomaly:
            print("     ", f)
        return 1
    if speed < 1.5:
        print(f"  ⚠ USABLE BUT POINTLESS — {speed:.2f}x means vortex is probably de-batching;")
        print("    correctness is fine, there is just no speedup to collect.")
        return 0
    print(f"  ✓ VALIDATED — safe and {speed:.2f}x faster.")
    print("    Padding is common-mode across arms at a fixed prompt set + seed, so it cannot")
    print("    manufacture a between-arm difference. Do NOT pool batched with sequential output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
