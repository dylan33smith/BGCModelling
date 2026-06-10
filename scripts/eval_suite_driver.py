#!/usr/bin/env python3
"""Batch eval-suite driver: run the 8-metric suite + novelty gate over a group of
sequences and compare generated vs the real-BGC positive control.

This is the glue the audit (C6) asked for: novelty is a query-vs-corpus operation,
so it is computed ONCE in batch (memorization_check.scan_corpus) and the per-record
result is passed into evaluate_bgc as Metric 9. Each metric that lacks its tool
(antiSMASH, ESMFold, MMseqs2, BiG-SCAPE, Pfam HMM) self-skips; the driver records
what ran. Compare the generated group's verdicts against the positive control's to
read each metric as "generated vs real held-out BGC".

Inputs: --gen and --positive are JSONL with at least {sequence, compound_class,
taxonomic_tag} (and an id/accession). --novelty is the memorization_check report
(id -> max_containment) — without it, Metric 9 is reported UNVERIFIED, never a pass.

CPU-only by default (skips the GPU metrics 3/5 unless --include-gpu-metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def load_novelty_map(path: Optional[Path]) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    return {r.get("id"): r for r in load_jsonl(path) if r.get("id")}


def summarize_group(results: list[dict]) -> dict[str, Any]:
    """Per-metric verdict counts (PASS/FAIL/no_verdict/skipped) over a group, plus
    per-class diagonal for the metrics that have a pass verdict."""
    metrics = sorted({k for r in results for k in r.get("summary", {})},
                     key=lambda k: int(k.split("_")[1]))
    per_metric: dict[str, Counter] = {m: Counter() for m in metrics}
    per_class_pass: dict[str, Counter] = defaultdict(Counter)   # metric -> Counter(class->pass)
    per_class_tot: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        cls = r.get("expected_class", "?")
        for m, v in r.get("summary", {}).items():
            per_metric[m][v] += 1
            if v in ("PASS", "FAIL"):
                per_class_tot[m][cls] += 1
                per_class_pass[m][cls] += int(v == "PASS")
    n = len(results)
    out = {"n": n, "per_metric": {}}
    for m in metrics:
        c = per_metric[m]
        scored = c["PASS"] + c["FAIL"]
        out["per_metric"][m] = {
            "PASS": c["PASS"], "FAIL": c["FAIL"],
            "no_verdict": c["no_verdict"], "skipped": c["skipped"],
            "pass_rate": round(c["PASS"] / scored, 3) if scored else None,
        }
    return out


def run_group(records: list[dict], novelty: dict[str, dict], skip_metrics: list[int]) -> list[dict]:
    from bgc_pipeline.evaluation import evaluate_bgc, EvalConfig, load_taxon_profiles
    cfg = EvalConfig(skip_metrics=skip_metrics)
    tp = Path("data/processed/taxon_profiles.json")
    if tp.exists():
        cfg.taxon_profiles = load_taxon_profiles(tp)
    results = []
    for i, rec in enumerate(records):
        sid = rec.get("id") or rec.get("accession") or str(i)
        nov = novelty.get(sid)
        res = evaluate_bgc(
            rec.get("sequence", ""), accession=sid,
            expected_class=rec.get("compound_class", ""),
            expected_taxon=rec.get("taxonomic_tag", ""),
            config=cfg, novelty=nov,
        )
        results.append(res)
        if (i + 1) % 20 == 0:
            print(f"  evaluated {i + 1}/{len(records)}", file=sys.stderr, flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True, help="Generated sequences JSONL.")
    ap.add_argument("--positive", type=Path,
                    default=Path("eval/positive_control_mibig.jsonl"),
                    help="Real-BGC positive control JSONL.")
    ap.add_argument("--novelty", type=Path, default=None,
                    help="memorization_check report (id -> max_containment) for Metric 9.")
    ap.add_argument("--include-gpu-metrics", action="store_true",
                    help="Also run ESMFold (3) and Evo2 perplexity (5).")
    ap.add_argument("--output", type=Path, default=Path("eval_suite_report.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    skip = [] if args.include_gpu_metrics else [3, 5]
    novelty = load_novelty_map(args.novelty)
    gen = load_jsonl(args.gen)
    pos = load_jsonl(args.positive) if args.positive and args.positive.exists() else []

    print(f"Generated: {len(gen)} | positive control: {len(pos)} | "
          f"novelty entries: {len(novelty)} | skip metrics: {skip or 'none'}", file=sys.stderr)
    if not novelty:
        print("  WARNING: no --novelty map -> Metric 9 (anti-memorization) UNVERIFIED, "
              "not a pass. Run memorization_check.py first.", file=sys.stderr)
    if args.dry_run:
        print("[dry-run] not evaluating.", file=sys.stderr)
        return

    report: dict[str, Any] = {}
    g_res = run_group(gen, novelty, skip)
    report["generated"] = summarize_group(g_res)
    if pos:
        p_res = run_group(pos, novelty, skip)
        report["positive_control"] = summarize_group(p_res)
    report["per_record"] = {"generated": g_res, "positive_control":
                            (p_res if pos else [])}
    args.output.write_text(json.dumps(report, indent=2, default=str))

    print("\nmetric pass-rates (generated vs positive control):", file=sys.stderr)
    gm = report["generated"]["per_metric"]
    pm = report.get("positive_control", {}).get("per_metric", {})
    for m in gm:
        g = gm[m]["pass_rate"]
        p = pm.get(m, {}).get("pass_rate")
        print(f"  {m:<10} gen={g}  positive={p}", file=sys.stderr)
    print(f"  wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
