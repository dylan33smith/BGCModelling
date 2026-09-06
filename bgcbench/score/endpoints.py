"""Endpoints and the confusion matrix (SPEC 3.4, 3.5, 3.6).

Three numbers, always reported together. Reporting the primary without both factors is
prohibited: an intervention can move one while leaving the other flat, and they answer
different questions.

    detect_rate    = P(antiSMASH detects anything)              CAPABILITY
    precision      = P(target in classes | detected)            CONTROL
    on_target_rate = detect_rate x precision                    PRIMARY
"""
from __future__ import annotations

from collections import Counter, defaultdict


def rates(records: list[dict], target: str) -> dict:
    n = len(records)
    if not n:
        return {"n": 0, "detect_rate": None, "precision": None, "on_target_rate": None}
    det = [r for r in records if r["detected"]]
    on = [r for r in det if target in r["observed_classes"]]
    return {
        "n": n,
        "n_detected": len(det),
        "n_on_target": len(on),
        "detect_rate": round(len(det) / n, 4),
        "precision": round(len(on) / len(det), 4) if det else None,
        "on_target_rate": round(len(on) / n, 4),
    }


def confusion(records_by_target: dict[str, list[dict]],
              classes: list[str]) -> dict[str, dict[str, float]]:
    """M[target][observed] = fraction of generations conditioned toward `target` whose
    antiSMASH classes include `observed`. Rows need not sum to 1: a hybrid record counts
    for every class it maps to (SPEC 3.3)."""
    m: dict[str, dict[str, float]] = {}
    for t in classes:
        rows = records_by_target.get(t, [])
        n = max(len(rows), 1)
        m[t] = {o: round(sum(1 for r in rows if o in r["observed_classes"]) / n, 4)
                for o in classes}
    return m


def lift(records_by_target: dict[str, list[dict]], unconditioned: list[dict],
         classes: list[str]) -> dict[str, float | None]:
    """on_target_rate(arm, c) / on_target_rate(arm, unconditioned).

    An arm that raises every class equally has lift ~1 everywhere and has demonstrated
    CAPABILITY, not CONTROL. This is why specificity is never read off a raw rate.
    """
    out: dict[str, float | None] = {}
    n_u = max(len(unconditioned), 1)
    for c in classes:
        base = sum(1 for r in unconditioned if c in r["observed_classes"]) / n_u
        got = rates(records_by_target.get(c, []), c)["on_target_rate"]
        out[c] = None if (not base or got is None) else round(got / base, 3)
    return out


def subclass_profile(records: list[dict], target: str) -> dict:
    """SPEC 3.6 -- SECONDARY, and a breakdown of an already-significant class-level
    result, never a standalone endpoint. Raw product strings, never collapsed."""
    on = [r for r in records if r["detected"] and target in r["observed_classes"]]
    prods = Counter(p for r in on for p in r["products"])
    total = sum(prods.values())
    return {
        "n_on_target": len(on),
        "counts": dict(prods.most_common()),
        "modal_share": round(max(prods.values()) / total, 4) if total else None,
        "n_distinct": len(prods),
    }


def gene_count_profile(records: list[dict], target: str) -> dict:
    """SPEC 4.5.1 second within-dataset analysis: among on-target generations, how many
    core genes did the model actually build? Asks 'does it build multi-gene loci?', where
    the covariate regression asks 'is multi-gene harder to hit?'."""
    on = [r for r in records if r["detected"] and target in r["observed_classes"]]
    counts = Counter(r.get("produced_core_genes", 0) for r in on)
    return {"n_on_target": len(on), "distribution": dict(sorted(counts.items())),
            "frac_multigene": round(
                sum(v for k, v in counts.items() if k >= 2) / max(len(on), 1), 4)}
