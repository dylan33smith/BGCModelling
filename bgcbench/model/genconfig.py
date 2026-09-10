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

    # SPEC 7.1: identical for every arm, and bounded by the MODEL'S USABLE CONTEXT.
    # evo2-1b's config max_seqlen is 8192. Measured per-token NLL on real >=15.9 kb cores,
    # scoring the last 1000 tokens of a prefix of each length:
    #   8,000 -> 0.809   8,192 -> 0.805 (best)   9,000 -> 0.818   10,000 -> 0.851
    #   12,000 -> 1.040   14,000 -> 1.209   15,900 -> 1.239   (ln 4 = 1.386 is chance)
    # So the model degrades progressively past ~10 kb and is near chance by 14 kb.
    # An earlier value of 16000 was set to match the corpus bound WITHOUT checking the
    # model: it would have had every arm generating 8 kb of near-chance sequence, and the
    # long classes would have looked worst because their references are longest.
    "budget_nt": 8192,

    # SPEC 7.3: identical decoding across arms.
    "temperature": 1.0,
    "top_k": 4,
    "top_p": 1.0,
    "rng_seed": 0,

    # batch size affects padding and kernel selection, not the sampling distribution, but
    # it is recorded so two runs are fully reconstructible from the artifacts.
    #
    # RAISED 8 -> 100 (2026-09-08) for throughput, not for any scientific reason. Generation
    # is autoregressive over 8,192 tokens, so wall time is set by the number of SEQUENTIAL
    # passes: n/batch_size. At batch 8 the measured W0 run took 62 min while occupying
    # 5.5 GB of an 80 GB card -- the card was idle, not busy.
    #
    # SIZED FROM A MEASUREMENT, not picked. Model + CUDA context is 2.08 GB and each
    # in-flight sequence costs 0.429 GB at the 8,192 nt budget, so:
    #     batch   8 ->  5.5 GB, 25 passes      batch 100 -> 45.0 GB,  2 passes
    #     batch  50 -> 23.5 GB,  4 passes      batch 200 -> 87.9 GB -- EXCEEDS the 80 GB card
    # Generating all 200 at once does not fit. 100 is the largest exact divisor of 200 that
    # does, with ~35 GB of headroom, and leaves no ragged batch.
    #
    # ⚠ This changes the frozen hash, so every arm must be generated at this value and the
    # batch-8 W0 run is NOT comparable to anything produced after it. That is the intended
    # behaviour of the hash, and W0 is regenerated rather than reconciled.
    "batch_size": 100,

    # SPEC 7: TERMINATOR FLOOR. Measured with the terminator finally visible, Evo2 emits its
    # stop token after TWO nucleotides from the de novo prompt -- 56% of base generations and
    # 91% of fine-tuned ones. Without a floor an unconditioned arm produces nothing to score.
    # One number, identical for every arm and class, injecting no class information: a
    # decoding policy, not a conditioning channel (SPEC 4.3 is not in tension).
    # 1000 sits below every benchmark class's median core (1,154-3,593 nt), so it forbids the
    # immediate collapse without dictating the length of a cluster.
    "min_new_tokens": 1000,

    # SPEC 6 S1. An unrecorded invocation-site default of 0 made `--seeded` without an
    # explicit length a SILENT DE NOVO ARM -- the prompt was the empty string and nothing
    # in any artifact showed it.
    #
    # SET BY GATE G3, 2026-09-10: 8 -> 64. G3 swept 8/16/32/64/128/256/512 at 800
    # generations each (G3_FULL_FROZEN_00820b3404a2acc4), and 64 is chosen because it is
    # THE LONGEST RUNG THAT IS NOT CONFOUNDED, not because it is the highest scoring:
    #
    #   L=8, L=16   on-target 0.001 -- nulls. A core's 5' end is a start codon plus noise
    #               below ~20 nt, so the old frozen 8 was measuring almost nothing.
    #   L=64        on-target 0.077, p = 7.0e-21, against a seed-only baseline of 0.003.
    #   L>=128      the SEEDS THEMSELVES become antiSMASH-detectable (baseline 0.050 at
    #               128, 0.165 at 256, 0.412 at 512). At 512 the seed-only baseline
    #               EXCEEDS the generation rate -- past ~64 nt the rise is the seed, not
    #               the method, so a higher rung buys a bigger number and a weaker claim.
    #
    # Not memorisation either way: a generation is NOT more like the record its seed came
    # from than like other held-out records of its class (FINDINGS 9.7, paired sign test).
    "seed_len_nt": 64,
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
