"""Every python heredoc embedded in a shell driver must COMPILE.

WHY THIS EXISTS. The 2026-07-31 eval-hardening pass edited these heredocs with string
replacement and shipped evo2/scripts/quick_eval.sh with an IndentationError plus an
uninitialised variable. `bash -n` passes on such a file — the shell only sees a quoted string —
so the failure surfaced only at RUNTIME, after the expensive 32k-token generation step, and it
reached main. The per-checkpoint tracking driver was committed unable to run.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GLOBS = ("evo2/scripts/*.sh", "evo2/experiments/probes/*.sh", "scripts/*.sh")
# Match on the project's convention: python heredocs are tagged PYEOF (`PY - <<'PYEOF'`).
# A looser "does it look like python?" heuristic false-positived on a SHELL heredoc whose
# comments happened to start with `if `.
HEREDOC = re.compile(r"<<'(PYEOF|PYTHON|PY)'\n(.*?)\n\1", re.S)


def main() -> int:
    checked = failed = 0
    for g in GLOBS:
        for sh in sorted(REPO.glob(g)):
            text = sh.read_text()
            for i, (_tag, body) in enumerate(HEREDOC.findall(text), 1):
                checked += 1
                try:
                    ast.parse(body)
                except SyntaxError as e:
                    failed += 1
                    print(f"FAIL {sh.relative_to(REPO)} block {i}: "
                          f"line {e.lineno}: {e.msg}", file=sys.stderr)
    if failed:
        print(f"\n{failed}/{checked} embedded python blocks do not compile", file=sys.stderr)
        return 1
    print(f"PASS: all {checked} shell-embedded python blocks compile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
