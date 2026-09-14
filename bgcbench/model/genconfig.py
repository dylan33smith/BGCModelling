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

    # SPEC 7.3 / §12.A7: DECODING IS PER SUBSTRATE, and it is keyed by family here so a
    # single shared scalar cannot exist to be matched across substrates by accident.
    #
    # ⚠ THIS WAS ONE SHARED TRIPLE AND IT WAS A DEFECT. `top_k: 4` is correct for Evo2 and
    # catastrophic for GenomeOcean, and the difference is invisible in the output. Measured
    # on 8 held-out TERPENE cores, teacher-forced, base weights:
    #
    #   Evo2-1B         vocab   512 (byte-level)  top-4 keeps 0.9999 of the mass  nucleus  3.3
    #   GenomeOcean-4B  vocab 4,096 (BPE 4.8nt)   top-4 keeps 0.1937 of the mass  nucleus 1012
    #
    # So on Evo2 the filter is a no-op -- the alphabet IS four letters -- while on
    # GenomeOcean it throws away 81% of the distribution at every step and renormalises
    # over the rest. Eight GO Stage 1 arms were generated through that before it was found
    # (GO_STAGE1_FROZEN_1c2acf1b1ce67b8f; those rates are not reportable).
    #
    # Values per family are selected by GATE G11 against the structural statistics of real
    # held-out sequence, never against the endpoint (§2.4) and never copied from the other
    # substrate or from the prior implementation's preset.
    "decoding": {
        # Evo2: G11 is N/A. top_k=4 over a 4-letter alphabet is already unrestrictive.
        "evo2": {"temperature": 1.0, "top_k": 4, "top_p": 1.0},
        # GenomeOcean: NO TRUNCATION. Sampling from the model's full 4,096-token vocabulary.
        #
        # ⚠ THIS IS A DELIBERATE NON-DECISION, AND THAT IS THE POINT. G11 previously selected
        # top_k=256 here, and that selection is VOID: holding everything else fixed and merely
        # resampling which 50 sequences get scored moves the gate's own deviation statistic
        # over 0.0108-0.1117 (sd 0.0318), while the winner beat the runner-up by 0.0189 =
        # 0.59 sd. The entire healthy-rung range fit inside one configuration's noise band.
        #
        # What the evidence DOES support is only that top_k=4 is catastrophic here, on two
        # legs that no defect touches: probability mass retained 0.1280 vs Evo2's 0.9999
        # (teacher-forced, BASE weights, no adapter), and distinct-21mer 0.3522 vs 1.0000 --
        # 65% of positions repeats. Measured mass retention across the vocabulary:
        #     k=1 0.0507 · k=4 0.1280 · k=16 0.2717 · k=64 0.4931
        #     k=256 0.7390 · k=1024 0.9298 · k=4096 1.0000
        #
        # Truncation's usual justification does not transfer to this substrate: every one of
        # GenomeOcean's 4,096 tokens is VALID DNA, so the tail is the model's real uncertainty
        # about the next k-mer, not garbage to be filtered. k=0 was also measured healthy
        # (distinct-21mer 1.0000, coding density 0.9333). Reporting "no truncation was
        # applied" needs no gate to defend it; "we truncated to 256" would need one we do
        # not have.
        "genomeocean": {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
    },
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
    #
    # ⚠ LOWERED 1000 -> 200 (2026-09-14). The old value was justified as sitting "below every
    # benchmark class's median core (1,154-3,593 nt)" -- but A FLOOR BINDS ON THE LEFT TAIL,
    # NOT THE MEDIAN. Measured on the held-out test splits:
    #
    #   class            shortest core   cores BELOW the 1000 nt floor
    #   TERPENE                  546            238/654 = 36.4%
    #   RIPP                     210             85/654 = 13.0%
    #   ARYLPOLYENE              243              7/651 =  1.1%
    #   REDOX_COFACTOR         2,035              0/652 =  0.0%
    #
    # So it forbade a correctly-sized cluster for over a THIRD of real TERPENE, and it did so
    # CLASS-ASYMMETRICALLY -- the exact shape of KNOWN_WRONG #5, where antiSMASH --minlength
    # 1000 discarded 72% of TERPENE and 48% of RIPP while dropping nothing from the long
    # classes and presented as "short classes are harder". That defect was fixed in the
    # scoring instrument and then reintroduced here, in the generator.
    #
    # 200 sits just below the shortest real core anywhere in the corpus (210 nt), so it never
    # binds on a legitimate output, while still being 100x the 2 nt collapse it exists to
    # forbid. The floor's job is to prevent degenerate immediate termination, not to dictate
    # how long a cluster may be.
    "min_new_tokens": 200,

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


def decoding_for(family: str) -> dict:
    """The decoding triple for a substrate family (§12.A7).

    ⚠ RAISES on an unknown family rather than falling back to a default. A silent fallback
    is exactly how one substrate's decoding came to be applied to the other: the wrong value
    produces valid-looking sequence and nothing downstream can tell.
    """
    table = FROZEN["decoding"]
    if family not in table:
        raise KeyError(
            f"no decoding configuration for substrate family {family!r}. Decoding is per "
            f"substrate (SPEC 12.A7) and must be selected by G11 against real sequence -- "
            f"never inherited from another family. Known: {sorted(table)}")
    return dict(table[family])


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
    """Which realised values differ from FROZEN. Empty dict means a frozen-config run.

    ⚠ DECODING IS NESTED AND FLATTENED IN `realised`. FROZEN holds `decoding[family]` while
    a run records `temperature`/`top_k`/`top_p` at the top level, so the generic loop below
    can never see them and drift would go unreported -- which is the auditing half of the
    §12.A7 defect. They are compared explicitly, against the family the run recorded.
    """
    out = {k: {"frozen": FROZEN[k], "realised": r[k]}
           for k in FROZEN if k != "decoding" and k in r and FROZEN[k] != r[k]}
    fam = r.get("substrate_family")
    if fam and fam in FROZEN["decoding"]:
        for k, v in FROZEN["decoding"][fam].items():
            if k in r and r[k] != v:
                out[k] = {"frozen": v, "realised": r[k], "family": fam}
    elif any(k in r for k in ("temperature", "top_k", "top_p")):
        out["decoding"] = {"frozen": "per-family table", "realised": "UNRESOLVABLE",
                           "why": f"substrate_family={fam!r} is not in the decoding table, "
                                  f"so this run's decoding cannot be audited"}
    return out


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
