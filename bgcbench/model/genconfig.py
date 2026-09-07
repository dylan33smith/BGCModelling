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

    # SPEC 6 S1. An unrecorded invocation-site default of 0 made `--seeded` without an
    # explicit length a SILENT DE NOVO ARM -- the prompt was the empty string and nothing
    # in any artifact showed it. Value itself is set by gate G3.
    "seed_len_nt": 8,
}


def config_hash() -> str:
    """Fingerprint of the FROZEN literal — the INTENT. Use `realised_hash` for a run.

    ⚠ This value is identical for every run made on a given code state, whatever was
    actually passed. Naming a run directory with it made all five per-class adapters
    resolve to ONE path, each truncating the last: four of five confusion rows destroyed,
    and the survivor rendered them as {"n": 0, "detect_rate": null} -- byte-identical to
    the SPEC 6.5 NOT-APPLICABLE encoding. Four deleted measurements would have published
    as four structural absences.
    """
    return hashlib.sha256(json.dumps(FROZEN, sort_keys=True).encode()).hexdigest()[:12]


def realised(**kw) -> dict:
    """The values a run ACTUALLY used. Every field here is measured, never assumed."""
    return {k: kw[k] for k in sorted(kw)}


def realised_hash(r: dict) -> str:
    return hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:12]


def off_frozen(r: dict) -> dict:
    """Which realised values differ from FROZEN. Empty dict means a frozen-config run."""
    return {k: {"frozen": FROZEN[k], "realised": r[k]}
            for k in FROZEN if k in r and FROZEN[k] != r[k]}


def check_uniform(reports: list[dict]) -> None:
    """Refuse to compare arms that were not generated the same way.

    A drift that is recorded is survivable; a drift that is silent is fatal. This makes it
    loud at the point of comparison, which is the last moment it can still be caught.
    """
    seen: dict[str, list[str]] = {}
    for r in reports:
        # group on the REALISED hash. Grouping on the frozen-literal hash made this
        # function unable to fire: every run carried the same constant.
        h = r.get("realised_config_hash") or "MISSING"
        seen.setdefault(h, []).append(r.get("arm", "?"))
    if len(seen) > 1:
        raise RuntimeError(
            "arms were generated under DIFFERENT frozen configs and are not comparable:\n"
            + "\n".join(f"  {h}: {sorted(a)}" for h, a in seen.items())
        )
