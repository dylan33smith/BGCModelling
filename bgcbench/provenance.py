"""What code produced an artifact, and what data bound it was built under.

⚠ WHY THIS EXISTS. The overnight training pass ran seven arms as one pipeline, and each arm
is a separate process that imports whatever is on disk when it starts. Two commits landed
while it was running, so five arms were trained by one version of `train.py` and two by
another. Nothing refused, nothing warned, and the reports looked uniform: the difference
only showed up because `TrainConfig` had gained two fields, so the two groups' recorded
config dicts had different KEY SETS. A change that had not touched the dataclass would have
left no trace at all.

`train_config_hash` cannot catch this -- it hashes the config, and the config is exactly
what is identical between two arms that were meant to differ only in data. The version of
the code that read that config is a separate fact and has to be recorded separately.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_version() -> dict:
    """The commit an artifact was produced at, and whether the tree was dirty.

    `dirty: true` is not a failure -- it is the truthful answer during development. It is
    recorded so that a run made from uncommitted code can never be mistaken for one made
    from the commit it happens to sit on.
    """
    def git(*a: str) -> str | None:
        try:
            return subprocess.run(("git", "-C", str(ROOT)) + a, capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()
        except Exception:
            return None
    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": sha,
            "dirty": (None if status is None else bool(status.strip())),
            "dirty_files": (sorted(l[3:] for l in status.splitlines())[:20]
                            if status else [])}


def corpus_max_len(default: int = 8192) -> int:
    """The length bound the SPLITS WERE BUILT AT, read from the manifest rather than
    carried as a literal.

    No training record can exceed it, so a training-time `max_len_nt` above it is dead
    configuration that still enters the run hash. The CLI default was 16000 -- a leftover
    from before the 8,192 nt reorientation, and above the model's own context, which is the
    same class of error already fixed once for the generation budget.
    """
    try:
        m = json.loads((Path("/data2/ds85/bgcbench/manifest.json")).read_text())
        v = m.get("_build", {}).get("max_len")
        return int(v) if v else default
    except Exception:
        return default


def freeze_id(body: dict) -> str:
    """Content address for a frozen result bundle: sha256 of the canonical body with the id
    itself removed, truncated to 16 hex.

    ⚠ THIS CONVENTION IS LOAD-BEARING AND MUST NOT CHANGE. Published numbers are cited BY
    BUNDLE ID, not by file path -- SPEC pins the seed length L=64 to
    G3_FROZEN_816d66441f8624ca, and SPEC and FINDINGS both pin the seeded composition result
    to SEEDED_DIAGONAL_FROZEN_f1a5fa95dbdc07a1. Change the hashing and the same content
    yields a different id, so a citation no longer resolves to the evidence it names. The
    test reproduces f1a5fa95dbdc07a1 from that bundle on disk.

    ⚠ MOVED HERE 2026-09-16 from run/freeze.py, whose writer (collect / endpoint_summary /
    main) was deleted. That writer produced 10 bundles and not one of them references an
    `_fx` run -- they all freeze superseded generations -- so nothing needs writing again.
    Three of those bundles are still cited for evidence the 48 arms cannot supply
    (G3_FULL_FROZEN's seed-only control, SEED_SOURCE_SIMILARITY's per-record memorisation
    test, PREFIX_CONTROL_FROZEN's matched prefix on/off), and this function is how their ids
    stay verifiable.
    """
    stripped = {k: v for k, v in body.items() if k != "freeze_id"}
    return hashlib.sha256(json.dumps(stripped, sort_keys=True).encode()).hexdigest()[:16]
