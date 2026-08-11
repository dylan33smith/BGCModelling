#!/usr/bin/env python3
"""Run the whole (GPU-free) test suite. Usage: python tests/run_all.py

These tests cover the data + loader + metric logic that does not require the
Evo2 weights or a GPU. The remaining end-to-end checks (real tokenizer at the
32k boundary; a forward/generate pass) are done in the GPU smoke run.
"""

import subprocess
import sys
from pathlib import Path

TESTS = [
    "test_chunk_spans.py",       # build_nt_chunk_spans boundary/coverage at L=32768
    "test_chunk_eos_windows.py", # dataset chunking: prefix selection, EOS, masking, budget
    "test_data_pipeline.py",     # split + curation invariants (leakage-free, caps, quality)
    "test_eval_metrics.py",      # M9 / M2 / M1 / M7 pure-logic
    "test_hardening.py",         # m3 checkpoint rotation, m1 fingerprint, A2 wandb safe-log
    "test_generation.py",        # C3 generation post-processing (EOS trim, FASTA, prompts)
    "test_memorization.py",      # nucleotide memorization/novelty similarity core
    "test_conditioning_experiment.py",  # controlled class/taxon conditioning logic
    "test_generation_batched.py",       # batched generation matches the sequential path
    # The two guards below exist because the 2026-07-31 hardening pass shipped two regressions
    # to main while every test above passed. They were added and then NOT wired here, which is
    # the same failure one level up: a guard that does not run is not a guard.
    "test_eval_smoke.py",               # eval paths EXECUTE on real cores (not just logic)
    "test_shell_embedded_python.py",    # every PYEOF heredoc in the shell drivers compiles
    "test_steer_hooks.py",              # WHERE and HOW BIG the injected steering edit is
    "test_guided_decoding.py",          # WHICH candidate guided decoding keeps + how Q1 is read
    "test_direction_audit.py",          # the dose/angle arithmetic behind the steering verdict
]


def main():
    here = Path(__file__).resolve().parent
    failed = []
    for t in TESTS:
        print(f"\n{'=' * 70}\nRUN {t}\n{'=' * 70}")
        rc = subprocess.run([sys.executable, str(here / t)]).returncode
        if rc != 0:
            failed.append(t)
    print(f"\n{'=' * 70}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"ALL {len(TESTS)} TEST FILES PASSED")


if __name__ == "__main__":
    main()
