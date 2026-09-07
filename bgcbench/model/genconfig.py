"""THE FROZEN GENERATION CONFIG (SPEC 7).

Scoring has been frozen and hashed since SPEC 3 was built: one antiSMASH invocation site,
a FROZEN dict, and a `config_hash()` stamped into every artifact filename. Generation had
no equivalent -- `n`, the budget, and the decoding parameters were CLI arguments read at the
invocation site. So a shell script passed n=150 where the agreed value was 200 and nothing
anywhere caught it, because nothing was watching.

That is the same class of failure as the Stage-1 prohibition being prose: a property the
benchmark depends on, enforced by discipline rather than construction. This module makes
the generation side symmetric with the scoring side.

⚠ CHANGING ANY VALUE HERE CHANGES THE HASH, which changes every run directory name and
makes old and new artifacts non-comparable BY CONSTRUCTION rather than by anyone noticing.
That is the intended behaviour.
"""
from __future__ import annotations

import hashlib
import json

FROZEN = {
    # SPEC 6.1: Stage 1 n per row. Agreed value; an earlier run used 150 because n lived
    # at the invocation site and nothing recorded or checked it.
    "n_per_row": 200,

    # SPEC 7.1: identical for every arm. Set to the CORPUS BOUND (SPEC 4.7), not lower.
    # A smaller budget silently handicaps the long classes -- BETALACTONE's real cores have
    # a median of ~9 kb, so a 4 kb budget makes it impossible for that arm to produce
    # anything resembling its own reference, and the deficit would read as a class effect.
    "budget_nt": 16000,

    # SPEC 7.3: identical decoding across arms.
    "temperature": 1.0,
    "top_k": 4,
    "top_p": 1.0,
    "rng_seed": 0,

    # batch size affects padding and kernel selection, not the sampling distribution, but
    # it is recorded so two runs are fully reconstructible from the artifacts.
    "batch_size": 8,
}


def config_hash() -> str:
    return hashlib.sha256(json.dumps(FROZEN, sort_keys=True).encode()).hexdigest()[:12]


def check_uniform(reports: list[dict]) -> None:
    """Refuse to compare arms that were not generated the same way.

    A drift that is recorded is survivable; a drift that is silent is fatal. This makes it
    loud at the point of comparison, which is the last moment it can still be caught.
    """
    seen: dict[str, list[str]] = {}
    for r in reports:
        h = r.get("generation_config_hash", "MISSING")
        seen.setdefault(h, []).append(r.get("arm", "?"))
    if len(seen) > 1:
        raise RuntimeError(
            "arms were generated under DIFFERENT frozen configs and are not comparable:\n"
            + "\n".join(f"  {h}: {sorted(a)}" for h, a in seen.items())
        )
