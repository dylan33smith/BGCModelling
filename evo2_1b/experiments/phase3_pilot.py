#!/usr/bin/env python
"""PHASE-3 PILOT — establish the RIPP floor `p0`, then compute the confirmatory `n`.

THIS IS STEP 1 OF `docs/phase3_preregistration.md` AND IT EXISTS TO SET SAMPLE SIZE.
The pre-registration deliberately does NOT fix `n`, because `n` depends on the floor rate and the
floor rate is unknown. Guessing it is how the Phase-2 pass ended up at n=24 with 15% power for a
doubling, and read the resulting null as a closure.

⚠️ **PILOT DATA IS NOT REUSED IN THE CONFIRMATORY ANALYSIS.** Estimating an effect and testing it on
the same data is how a floor becomes whatever the first sample happened to show. This script writes
its own output file and the confirmatory run regenerates from scratch.

WHAT IT MEASURES — exactly the pre-registered primary endpoint, no substitutes:
    on_class_rate = fraction of generations with >=1 RIPP-defining biosynthetic Pfam domain
                    (`best_bio_bits > 0` for cls=RIPP), on a FIXED 2,000-nt scoring window.

CONTROL ARMS (no Phase-3 model is trained yet, so all of these exist today):
    base_1b          absolute floor — no adapter, no seed
    general_adapter  the Phase-2 all-class adapter — does specialising later beat generalising now?
    real_cores       CEILING — real held-out RIPP cores through the IDENTICAL window and scorer
    shuffled         instrument false-positive rate (di-nucleotide-preserving shuffle of real cores)

The `real_cores` and `shuffled` arms are what make the generated numbers readable: without a ceiling
a low rate is uninterpretable, and without a floor the scorer's own false-positive rate is unknown.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))
sys.path.insert(0, str(REPO / "src"))

CLASS = "RIPP"
WINDOW = 2000          # pre-registered fixed scoring window
DATA = Path("/data2/ds85/bgcmodel_data/splits_class") / CLASS


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(seqs: list[str], workers: int = 24) -> list[dict]:
    """Score the FIXED WINDOW of each sequence, class-aware. Same call for every arm."""
    from ladder_audit import one
    jobs = [("pilot", s[:WINDOW], CLASS, i) for i, s in enumerate(seqs)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, jobs))


def dinuc_shuffle(seq: str, rng: random.Random) -> str:
    """Preserve di-nucleotide composition. A naive base shuffle destroys local composition and
    makes the negative control artificially easy to beat."""
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def generate(adapter: str | None, n: int, max_new: int, out: Path, log: Path) -> list[str]:
    if out.exists() and out.stat().st_size:
        return [json.loads(l).get("sequence", "") for l in out.open()]
    cmd = ["micromamba", "run", "-n", "bgcmodel", "python",
           str(REPO / "evo2" / "scripts" / "generate_bgc.py"),
           "--from-jsonl", str(DATA / "eval_prompts.jsonl"),
           "--per-class", str(n), "--n", "1",
           "--max-new-tokens", str(max_new), "--seed", "0",
           "--batch-size", "32", "--out-jsonl", str(out)]
    if adapter:
        cmd[cmd.index("--from-jsonl"):cmd.index("--from-jsonl")] = ["--adapter", adapter]
    with log.open("w") as lg:
        r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"  !! generation failed, see {log}")
        return []
    return [json.loads(l).get("sequence", "") for l in out.open()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=4000,
                    help="generation length; SCORING always uses the fixed 2,000-nt window")
    ap.add_argument("--general-adapter",
                    default="/data2/ds85/bgcmodel_runs/phase2_long/baseline_long/final_adapter")
    ap.add_argument("--out", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase3_ripp"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    arms: dict[str, list[str]] = {}

    print(f"[pilot] {CLASS}: n={args.n}, generate {args.max_new_tokens} nt, "
          f"score fixed {WINDOW} nt window\n")

    # --- real cores: the CEILING. Scored through the identical window and scorer. ---
    real = [json.loads(l)["sequence"] for l in (DATA / "test.jsonl").open()]
    real = [s for s in real if len(s) >= WINDOW][: args.n]
    arms["real_cores (CEILING)"] = real

    # --- negative control: the instrument's own false-positive rate ---
    arms["shuffled (FP rate)"] = [dinuc_shuffle(s, rng) for s in real]

    # --- generated arms ---
    print("[pilot] generating base_1b (no adapter) …", flush=True)
    arms["base_1b (FLOOR)"] = generate(None, args.n, args.max_new_tokens,
                                       args.out / "pilot_base.jsonl",
                                       args.out / "pilot_base.log")
    print("[pilot] generating general_adapter …", flush=True)
    arms["general_adapter"] = generate(args.general_adapter, args.n, args.max_new_tokens,
                                       args.out / "pilot_general.jsonl",
                                       args.out / "pilot_general.log")

    print("\n" + "=" * 78)
    print(f"PHASE-3 PILOT — {CLASS}, primary endpoint on_class_rate (fixed {WINDOW} nt window)")
    print("=" * 78)
    print(f"{'arm':<24} {'n':>4} {'on-class':>10} {'rate':>7} {'95% CI':>16}")
    rates = {}
    for label, seqs in arms.items():
        if not seqs:
            print(f"{label:<24} {'--':>4}  (generation failed)")
            continue
        res = score(seqs)
        k = sum(1 for r in res if r["bio"] > 0)
        lo, hi = wilson(k, len(res))
        rates[label] = (k, len(res))
        print(f"{label:<24} {len(res):>4} {k:>5}/{len(res):<4} {k/len(res):>7.3f} "
              f"[{lo:>5.3f},{hi:>5.3f}]")

    # --- sample size for the confirmatory run ---
    floor = rates.get("base_1b (FLOOR)")
    print("\n" + "-" * 78)
    if floor and floor[1]:
        p0 = max(floor[0] / floor[1], 1.0 / floor[1])   # never 0; a zero floor makes n infinite
        print(f"CONFIRMATORY SAMPLE SIZE, from the measured floor p0 = {p0:.3f}")
        print(f"{'target effect':<22} {'p1':>7} {'n per arm (80% power, a=0.05)':>32}")
        for mult in (1.5, 2.0, 3.0):
            p1 = min(p0 * mult, 0.95)
            pbar = (p0 + p1) / 2
            n = ((1.96 * math.sqrt(2 * pbar * (1 - pbar))
                  + 0.84 * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / ((p1 - p0) ** 2)
            print(f"{f'{mult:.1f}x the floor':<22} {p1:>7.3f} {math.ceil(n):>32,}")
        print("\nRECORD THE CHOSEN n IN docs/phase3_preregistration.md §8 BEFORE RUNNING.")
        print("Pilot data is NOT reused in the confirmatory analysis.")
    else:
        print("No floor measured — cannot size the confirmatory run.")
    json.dump({k: {"on_class": v[0], "n": v[1]} for k, v in rates.items()},
              open(args.out / "pilot_rates.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
