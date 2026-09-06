"""Additive dataset manifest (SPEC 4.6).

KNOWN_WRONG #1 in the prior codebase: the split builder initialised an empty manifest and
rewrote the whole file, so building one class silently destroyed every sibling entry.
Twelve subclass datasets were found on disk carrying an all-zero manifest -- no recorded
leakage verification at all, and no error anywhere.

Every write here is read-modify-write of a single key. There is deliberately no API that
replaces the whole document.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def update(path: Path, key: str, entry: dict) -> dict:
    """Set one key, preserving every other. Written atomically."""
    doc = load(path)
    doc[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return doc
