"""Freeze a set of runs into one immutable bundle (SPEC §10 step 2).

Freezing BEFORE comparison is what makes the later unblind-and-diff evidence rather than
a post-hoc rationalisation: a result that can still move while you read the prior
implementation proves nothing about whether the rebuild reproduced it independently.

The bundle records, per run, the arm identity, the full `realised` config (so the run can
be re-derived), the per-class endpoint counts, the novelty gate tally and the generation
health fields. `freeze_id` is sha256 of the canonical body with the id itself removed,
truncated to 16 hex -- the convention already in use by the Evo2 bundles, verified against
SEEDED_DIAGONAL_FROZEN_f1a5fa95dbdc07a1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUNS = Path("/data2/ds85/bgcbench/runs")

#: Fields lifted out of each report. Everything else stays in the run dir; the bundle is a
#: manifest, not a copy.
CARRIED = ("arm", "row_class", "class_bearing", "stage", "substrate",
           "hit_eos_rate", "median_len", "n_empty_generations", "n_per_class",
           "per_class", "confusion", "lift", "lift_denominator", "novelty",
           "novelty_reference_classes", "gene_counts", "scoring_config",
           "realised", "realised_config_hash", "off_frozen", "generation_config_frozen")


def freeze_id(body: dict) -> str:
    """⚠ MUST match the existing bundles. Any change here silently forks the convention
    and two bundles of the same content would carry different ids."""
    stripped = {k: v for k, v in body.items() if k != "freeze_id"}
    return hashlib.sha256(json.dumps(stripped, sort_keys=True).encode()).hexdigest()[:16]


def collect(run_dirs: list[Path]) -> dict:
    out = {}
    for d in run_dirs:
        p = d / "report.json"
        if not p.exists():
            raise SystemExit(f"no report.json in {d} -- refusing to freeze a partial set")
        r = json.load(open(p))
        out[d.name] = {k: r[k] for k in CARRIED if k in r}
    return out


def endpoint_summary(runs: dict) -> dict:
    """The counts a reader needs without opening any run dir. Counts, never rates alone:
    a rate with no denominator cannot be re-tested."""
    s = {}
    for name, r in runs.items():
        for cls, v in (r.get("per_class") or {}).items():
            if not v.get("n"):
                continue
            s[f"{r['arm']}::{cls}"] = {
                "n": v["n"], "n_detected": v["n_detected"],
                "n_on_target": v["n_on_target"],
                "median_len_nt": r.get("median_len"),
                "hit_eos_rate": r.get("hit_eos_rate"),
            }
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="bundle name, e.g. GO_STAGE1")
    ap.add_argument("--what", required=True, help="one line: what this set of runs IS")
    ap.add_argument("--runs", nargs="+", required=True, help="run directory names or paths")
    ap.add_argument("--note", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra recorded facts, e.g. config=rank4_all24layers")
    args = ap.parse_args()

    dirs = [Path(x) if "/" in x else RUNS / x for x in args.runs]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        raise SystemExit("missing run dir(s): " + ", ".join(missing))

    runs = collect(dirs)
    body = {"what": args.what, "n_runs": len(runs)}
    for kv in args.note:
        k, _, v = kv.partition("=")
        body[k] = v
    body["endpoint"] = endpoint_summary(runs)
    body["runs"] = runs
    body["freeze_id"] = freeze_id(body)

    out = RUNS / f"{args.name}_FROZEN_{body['freeze_id']}.json"
    if out.exists():
        raise SystemExit(f"{out} already exists -- a freeze is immutable, refusing to overwrite")
    out.write_text(json.dumps(body, indent=1, sort_keys=True))
    print(f"froze {len(runs)} run(s) -> {out}")
    for k, v in sorted(body["endpoint"].items()):
        print(f"  {k:42s} {v['n_detected']:3d}/{v['n']} detected, "
              f"{v['n_on_target']:3d}/{v['n']} on target, median {v['median_len_nt']} nt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
