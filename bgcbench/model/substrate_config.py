"""Per-substrate gated choices, in ONE place (SPEC §14A).

⚠ WHY THIS FILE EXISTS. Every value here is a property of a MODEL, chosen for that model by
measurement. Until now they lived as bare literals in shell scripts -- `--rank 4` appears six
times across work/*.sh and `--rank 16` six times, with nothing connecting either to a substrate.
A script that passed the wrong one would train a valid-looking arm and nothing downstream could
tell. That is the same shape as the two defects found on 2026-09-13/14: one frozen `top_k` applied
to both substrates (it keeps 0.9999 of Evo2's probability mass and 0.1937 of GenomeOcean's), and
one frozen `lora_alpha` applied across ranks (peft scales by alpha/r, so GenomeOcean ran at 8.0x
while Evo2 ran at 2.0x).

⚠ WHAT DOES **NOT** BELONG HERE. Anything that is the TASK is shared and lives in `genconfig`:
the corpus, the splits, the class set, the scoring config, the novelty gate, `n_per_row`, the
**nucleotide** budget, `seed_len_nt`, and the endpoint. Two substrates given different seed
lengths are not solving the same problem -- see the §15.6 correction of 2026-09-14.

The split is: TASK is shared, TREATMENT is per substrate.
"""
from __future__ import annotations

#: Provenance is part of the value. A number whose gate is not recorded looks identical to a
#: number that was measured, and this project has now been bitten by that twice.
SUBSTRATE: dict[str, dict] = {
    "evo2": {
        "rank":          (16,         "G6 rank sweep, FINDINGS §13"),
        "depth":         ("all",      "G6b depth sweep, FINDINGS §14"),
        "prefix":        ("taxonomy", "Evo2's native pretraining format (GTDB lineages)"),
        "lora_scaling":  (2.0,        "⚠ ADOPTED, NOT MEASURED — matches this project's "
                                      "original rank16/alpha32 pairing and the prior "
                                      "implementation. Deferred to §14."),
    },
    "genomeocean": {
        "rank":          (16,         "⚠ INHERITED, NOT MEASURED (user decision 2026-09-14). "
                                      "GO's own G6 chose rank 4, but that sweep is void: it "
                                      "varied alpha/r as 8/4/2/1 alongside rank, so rank and "
                                      "update magnitude were never separated."),
        "depth":         ("all",      "G6b depth sweep, FINDINGS §23"),
        "prefix":        ("none",     "SPEC §15.7 — GenomeOcean was not pretrained on GTDB "
                                      "lineages, so a lineage prefix is a format it has "
                                      "never seen"),
        "lora_scaling":  (2.0,        "⚠ ADOPTED, NOT MEASURED — see evo2"),
    },
}

#: I1 injection sites, per substrate AND per class (the site sweep is per class).
#: ⚠ EVERY VALUE HERE IS KEYED TO A WEIGHT STATE and these were measured on the PRE-2026-09-14
#: adapters. The re-train replaces those weights, so all of them must be re-measured before any
#: I1 arm runs. They are recorded rather than deleted so the re-measurement has a baseline.
I1_SITES_STALE: dict[str, dict[str, str]] = {
    "evo2":        {"TERPENE": "pair_01", "RIPP": "single_0",
                    "ARYLPOLYENE": "single_0", "REDOX_COFACTOR": "all"},
    "genomeocean": {"TERPENE": "early_third", "RIPP": "early_third",
                    "ARYLPOLYENE": "early_third", "REDOX_COFACTOR": "all"},
}

#: The α grid G9 sweeps, per substrate. This is the SEARCH SPACE, not a chosen value — the
#: chosen α is per class and lives in the G9 artifact.
#:
#: ⚠ CHECKS (a) AND (b) READ THIS SAME GRID. SPEC §14A names "probe magnitude" as a TREATMENT
#: parameter: chosen per substrate, never matched across them. The manipulation check therefore
#: probes at the magnitudes the arm actually generates at, rather than at a separate constant.
#: It used to probe at a hard-coded [0.0, 0.5, 1.0, 2.0, 4.0] with the gate at α = 1.0 — a grid
#: retired on 2026-09-15 and, for GenomeOcean, three rungs of it are magnitudes no arm will ever
#: run at. A check read outside the arm's operating range cannot license reading that arm's null.
I1_ALPHA_GRID: dict[str, list[float]] = {
    "evo2":        [0.0, 0.1, 0.2, 0.3, 0.4],
    "genomeocean": [0.0, 0.1, 0.2, 0.3, 0.4],
}

#: Steering magnitude, per substrate AND per class AND per weight state (SPEC §12.A5: alpha does
#: not transfer across classes). ⚠ ALSO INVALIDATED BY THE RE-TRAIN, for the same reason.
I1_ALPHA_STALE: dict[str, dict[str, float]] = {
    "evo2": {"TERPENE": 0.3, "RIPP": 0.3, "ARYLPOLYENE": 0.1, "REDOX_COFACTOR": 0.1},
    "genomeocean": {},        # never measured on trained GO weights
}


#: Peak GPU a generation arm actually takes, per substrate, MEASURED on this card.
#:
#: ⚠ ONE THRESHOLD FOR BOTH SUBSTRATES IS A DEADLOCK WAITING TO HAPPEN, and it is the same
#: §14A error as one `top_k` or one `lora_alpha` for both models. `g9_alpha` defaulted
#: --need-mib to 34,000 for every substrate: right for Evo2 (measured 29,690-30,490 MiB) and a
#: 2.5x over-estimate for GenomeOcean (measured 13,430-16,830 MiB). With another user holding
#: ~40 GB of an 81.5 GB card, a GenomeOcean sweep demanding 34,000 free stalls between arms on
#: memory it never needed. _lib.sh already records what that costs: 13 hours of idle GPU on
#: 2026-09-14, from a threshold set above what the job actually used.
#:
#: Values are the measured peak plus ~25% headroom, rounded.
GPU_MIB: dict[str, int] = {
    "evo2":        (30000, "measured 29,690 MiB (run.arm, batch 50) and 30,490 MiB (phase D)"),
    "genomeocean": (18000, "measured 13,430 MiB (g9 alpha arm) and 16,830 MiB (phase C peak)"),
}


def gpu_mib(family: str) -> int:
    """Free GPU a job on this substrate should wait for before starting."""
    if family not in GPU_MIB:
        raise KeyError(
            f"no measured GPU footprint for substrate family {family!r}. Inheriting another "
            f"model's threshold either deadlocks (too high) or OOMs (too low). "
            f"Known: {sorted(GPU_MIB)}")
    return GPU_MIB[family][0]


def alpha_grid(family: str) -> list[float]:
    """The α grid this substrate sweeps, and the grid its §6.4 checks are read at.

    ⚠ THE TWO SUBSTRATES CURRENTLY HOLD THE SAME LITERAL, AND THAT IS A COINCIDENCE, NOT A
    MATCH. Both were lowered to [0, 0.1, 0.2, 0.3, 0.4] by the same user decision on
    2026-09-15, for the same measured reason (large α throws generation off on both). They
    are keyed separately so either can move without dragging the other, which is what §14A
    requires of a treatment parameter.
    """
    if family not in I1_ALPHA_GRID:
        raise KeyError(
            f"no I1 alpha grid for substrate family {family!r}. The grid is a TREATMENT "
            f"parameter (SPEC §14A) and must not be inherited from another model. "
            f"Known: {sorted(I1_ALPHA_GRID)}")
    return list(I1_ALPHA_GRID[family])


def max_alpha(family: str) -> float:
    """The strongest magnitude this substrate will generate at.

    Check (b) is read here: if the intervention reaches the output anywhere on the grid it
    reaches it at the top rung, and a reach measured at a magnitude the arm never uses says
    nothing about the arm.
    """
    return max(alpha_grid(family))


def for_substrate(family: str) -> dict:
    """The gated choices for a substrate FAMILY, values only.

    ⚠ RAISES on an unknown family rather than returning a default. A silent fallback is how one
    substrate's configuration reached the other in the first place (see `decoding_for`).
    """
    if family not in SUBSTRATE:
        raise KeyError(
            f"no gated configuration for substrate family {family!r}. Every value in this "
            f"module is chosen PER SUBSTRATE by measurement (SPEC §14A) and must not be "
            f"inherited from another model. Known: {sorted(SUBSTRATE)}")
    return {k: v[0] for k, v in SUBSTRATE[family].items()}


def provenance(family: str) -> dict[str, str]:
    """Where each value came from — the gate, or an explicit note that it was not measured."""
    if family not in SUBSTRATE:
        raise KeyError(f"unknown substrate family {family!r}")
    return {k: v[1] for k, v in SUBSTRATE[family].items()}


def unmeasured(family: str) -> list[str]:
    """Fields whose recorded provenance says they were adopted rather than measured.

    Used to keep §14's deferred register honest: a configuration that was inherited should be
    visible as inherited in the artifacts, not indistinguishable from a gated result.
    """
    return sorted(k for k, v in SUBSTRATE[family].items()
                  if "NOT MEASURED" in v[1] or "INHERITED" in v[1])
