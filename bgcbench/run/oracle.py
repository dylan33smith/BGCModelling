"""THE ORACLE ARM — the end-to-end positive control (SPEC 3, 6).

WHY THIS EXISTS. `score_gates.py` measures the ceiling by calling antiSMASH directly. It
never touches `record.py`, `novelty.py`, `rates()`, `confusion()` or `lift()` — so the
ceiling is measured through a path NO ARM WILL EVER RUN. If every arm then reads 0.000,
"the methods do not work" and "the harness is broken" are indistinguishable, and a null
result is unpublishable.

The oracle substitutes a held-out REAL CORE for a generation and pushes it through the
complete arm path. It must read on_target_rate ~= 1.0. Until it does, no arm's number
means anything.

It is not an arm and never appears in the benchmark table. It is the instrument check that
licenses reading the table at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgcbench.data.classmap import BENCHMARK_CLASSES, build_map
from bgcbench.score import antismash
from bgcbench.score.endpoints import confusion, gene_count_profile, lift, rates, subclass_profile
from bgcbench.score.novelty import Reference
from bgcbench.score.record import build as build_records

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
OUT = ROOT / "oracle"
WORK = ROOT / "work"

#: the oracle must clear this or nothing downstream is trustworthy
MIN_ON_TARGET = 0.90


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, default=16)
    ap.add_argument("--n", type=int, default=60, help="held-out cores per class")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = build_map()["mapping"]

    by_target: dict[str, list[dict]] = {}
    failures: list[str] = []
    for cls in BENCHMARK_CLASSES:
        cores = _load(SPLITS / cls / "test.jsonl")[: args.n]
        train = _load(SPLITS / cls / "train.jsonl")
        reference = Reference(train)

        # A held-out core stands in for a generation. Seed fields are populated exactly as
        # a seeded arm would populate them, so the SPEC 4.5.1 covariates are exercised too.
        gens = [{
            "generation_id": f"oracle::{cls}::{r['accession']}",
            "target_class": cls,
            "sequence": r["sequence"],
            "hit_eos": None,
            "seed_accession": r["accession"],
            "seed_core_gene_count": r["core_gene_count"],
            "seed_seq_len": r["seq_len"],
        } for r in cores]

        verdicts = antismash.run([(g["generation_id"], g["sequence"]) for g in gens],
                                 workdir=WORK, cpus=args.cpus)
        scored = build_records(gens, verdicts, mapping, reference,
                               arm="ORACLE", substrate="none")
        by_target[cls] = scored

        rt = rates(scored, cls)
        prof = subclass_profile(scored, cls, mapping)
        gp = gene_count_profile(scored, cls)
        novel = sum(1 for r in scored if r["novel"])
        gates = {}
        for r in scored:
            gates[r["gate"]] = gates.get(r["gate"], 0) + 1
        print(f"{cls:14s} n={rt['n']:3d} detect={rt['detect_rate']} "
              f"precision={rt['precision']} on_target={rt['on_target_rate']}  "
              f"novel={novel}/{len(scored)} gates={gates}  "
              f"multigene={gp['frac_multigene']} modal={prof['modal_share']}", flush=True)
        if (rt["on_target_rate"] or 0) < MIN_ON_TARGET:
            failures.append(f"{cls}: on_target_rate {rt['on_target_rate']} < "
                            f"{MIN_ON_TARGET}")

    # the full cross-class object an arm produces, exercised end to end
    classes = list(BENCHMARK_CLASSES)
    cm = confusion(by_target, classes)
    unconditioned = [r for rows in by_target.values() for r in rows]
    lf = lift(by_target, unconditioned, classes)
    payload = {"min_on_target": MIN_ON_TARGET,
               "per_class": {c: rates(by_target[c], c) for c in classes},
               "confusion": cm, "lift_vs_pooled": lf,
               "scoring_config": antismash.config_hash()}
    (OUT / "oracle.json").write_text(json.dumps(payload, indent=2))

    print("\nconfusion (row = conditioned toward, col = observed):")
    print(f"  {'':14s}" + "".join(f"{c[:9]:>11s}" for c in classes))
    for t in classes:
        print(f"  {t:14s}" + "".join(f"{str(cm[t][o]):>11s}" for o in classes))

    if failures:
        print("\nORACLE FAILED:")
        for f in failures:
            print("  " + f)
        print("\nThe harness cannot recognise a real core pushed through the arm path. "
              "No arm result is interpretable until this passes.")
        return 1
    print(f"\nORACLE PASSED — wrote {OUT / 'oracle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
