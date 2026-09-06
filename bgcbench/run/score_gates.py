"""G1 and G5 through the built scoring layer (SPEC 12).

G5  per-class CEILING on real held-out cores  -> the dynamic range every rate is read against
G1  per-class FALSE-POSITIVE RATE on real non-BGC DNA -> the floor, and the confirmation
    (or refutation) of the provisional --minlength 1

Both go through the single antiSMASH site, so the ceiling, the floor and every future arm
are scored by identical code under one frozen config hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgcbench.data.classmap import BENCHMARK_CLASSES, build_map
from bgcbench.score import antismash
from bgcbench.score.endpoints import subclass_profile

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
NEG = ROOT / "negatives"
OUT = ROOT / "gates"
WORK = ROOT / "work"


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap records per class (0=all)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    mapping = build_map()["mapping"]
    cfg = antismash.config_hash()
    print(f"scoring config {cfg}: {antismash.FROZEN}\n", flush=True)

    report: dict[str, dict] = {}
    for cls in BENCHMARK_CLASSES:
        cores = _load(SPLITS / cls / "test.jsonl")
        negs = _load(NEG / f"{cls}.jsonl")
        if args.limit:
            cores, negs = cores[: args.limit], negs[: args.limit]

        cv = antismash.run([(r["accession"], r["sequence"]) for r in cores],
                           workdir=WORK, cpus=args.cpus)
        nv = antismash.run([(r["accession"], r["sequence"]) for r in negs],
                           workdir=WORK, cpus=args.cpus)

        def summarise(recs, verds, target):
            rows = []
            for r in recs:
                v = verds[r["accession"]]
                from bgcbench.data.classmap import classify
                obs = classify(v["products"], mapping) if v["products"] else []
                rows.append({**v, "observed_classes": obs,
                             "on_target": target in obs})
            n = len(rows)
            det = sum(1 for x in rows if x["detected"])
            on = sum(1 for x in rows if x["on_target"])
            return rows, {"n": n, "detect_rate": round(det / n, 4) if n else None,
                          "on_target_rate": round(on / n, 4) if n else None,
                          "n_detected": det, "n_on_target": on}

        crow, cstat = summarise(cores, cv, cls)
        nrow, nstat = summarise(negs, nv, cls)
        prof = subclass_profile(crow, cls)
        report[cls] = {
            "G5_ceiling": cstat,
            "G1_false_positive": nstat,
            "reference_subclass_profile": prof,
            "produced_core_genes_real": {
                "median": sorted(x["produced_core_genes"] for x in crow)[len(crow) // 2]
                if crow else None,
                "frac_multigene": round(
                    sum(1 for x in crow if x["produced_core_genes"] >= 2)
                    / max(len(crow), 1), 4),
            },
        }
        print(f"{cls:14s} CEILING detect={cstat['detect_rate']} "
              f"on_target={cstat['on_target_rate']} ({cstat['n_on_target']}/{cstat['n']})"
              f"   FPR detect={nstat['detect_rate']} "
              f"on_target={nstat['on_target_rate']} ({nstat['n_on_target']}/{nstat['n']})"
              f"   modal_share={prof['modal_share']} n_products={prof['n_distinct']}",
              flush=True)

    (OUT / f"gates_{cfg}.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / f'gates_{cfg}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
