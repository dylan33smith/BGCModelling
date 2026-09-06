"""Minimal test runner — pytest is not in the env and adding a dependency to run four
dozen assertions is not worth it. Collects test_* callables and reports."""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    mods = sorted(p.stem for p in Path(__file__).parent.glob("test_*.py"))
    passed = failed = 0
    failures: list[tuple[str, str]] = []
    for name in mods:
        mod = importlib.import_module(f"tests.{name}")
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
            except Exception:
                failed += 1
                failures.append((f"{name}.{attr}", traceback.format_exc()))
    for who, tb in failures:
        print(f"\nFAIL {who}\n{tb}", file=sys.stderr)
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
