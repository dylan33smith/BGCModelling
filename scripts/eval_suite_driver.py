#!/usr/bin/env python3
"""Batch eval-suite driver: run the named-CHECK / QUESTION acceptance suite over a
group of sequences and compare generated vs the real-BGC positive control. See
evaluation.py for the CHECKS and the QUESTIONS they derive.

Novelty is a query-vs-corpus operation, so it is computed ONCE in batch
(memorization_check.scan_corpus) and the per-record result is passed into evaluate_bgc
as the kmer_novelty check. Each check that lacks its tool (antiSMASH, ESMFold,
MMseqs2, Pfam HMM) self-skips; the driver records what ran. Compare the generated
group's verdicts against the positive control's to read each question/check as
"generated vs real held-out BGC".

Inputs: --gen and --positive are JSONL with at least {sequence, compound_class,
taxonomic_tag} (and an id/accession). --novelty is the memorization_check report
(id -> max_containment) — without it, the 'novel' question is UNVERIFIED, never a pass.

Runs all CHECKS by default (use --skip-checks <names> to drop slow/DB-bound ones);
the optional ESMFold protein_foldability check is opt-in via --include-foldability.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    """Per-QUESTION and per-CHECK verdict counts + the gate-keyed headline.

    QUESTIONS are what we report (is_bgc / correct_class / novel are GATES; the rest
    diagnostic); CHECKS are the underlying compute units. antiSMASH owns is_bgc +
    correct_class, with class_markers as the quick-eval PROXY. See evaluation.py.
    """
    from bgc_pipeline.evaluation import (GATE_QUESTIONS, DIAGNOSTIC_QUESTIONS,
                                         QUESTIONS, CHECKS, OPTIONAL_CHECKS)
    n = len(results)

    def tally(key: str, names: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name in names:
            c: Counter = Counter()
            for r in results:
                v = r.get(key, {}).get(name)
                if v is not None:
                    c[v] += 1
            if sum(c.values()):
                scored = c["PASS"] + c["FAIL"]
                out[name] = {"PASS": c["PASS"], "FAIL": c["FAIL"],
                             "no_verdict": c["no_verdict"], "skipped": c["skipped"],
                             "pass_rate": round(c["PASS"] / scored, 3) if scored else None}
        return out

    per_q = tally("questions", list(QUESTIONS.keys()))
    for q, d in per_q.items():
        d["role"] = "gate" if q in GATE_QUESTIONS else "diagnostic"
    per_c = tally("checks", list(CHECKS) + list(OPTIONAL_CHECKS))

    # Headline (per-record). GATES = is_bgc, correct_class, novel.
    #   GENERATES_BGC = is_bgc PASS    (real coding DNA + antiSMASH cluster)
    #   CORRECT_CLASS = correct_class PASS
    #   BIOLOGICAL    = generates_bgc AND correct_class
    #   ACCEPT        = biological AND novel
    gen_bgc = correct_class = bio_valid = accept = no_gate_fail = 0
    for r in results:
        s = r.get("questions", {})
        g_bgc = s.get("is_bgc") == "PASS"
        c_cls = s.get("correct_class") == "PASS"
        novel = s.get("novel") == "PASS"
        bio = g_bgc and c_cls
        gen_bgc += int(g_bgc)
        correct_class += int(c_cls)
        bio_valid += int(bio)
        accept += int(bio and novel)
        no_gate_fail += int(all(s.get(q) != "FAIL" for q in GATE_QUESTIONS))

    def rate(k: int) -> Optional[float]:
        return round(k / n, 3) if n else None
    return {
        "n": n, "per_question": per_q, "per_check": per_c,
        "roles": {"gates": list(GATE_QUESTIONS), "diagnostics": list(DIAGNOSTIC_QUESTIONS)},
        "headline": {
            "generates_bgc": {"n": gen_bgc, "rate": rate(gen_bgc)},
            "correct_class": {"n": correct_class, "rate": rate(correct_class)},
            "biological_valid": {"n": bio_valid, "rate": rate(bio_valid)},
            "biological_valid_and_novel": {"n": accept, "rate": rate(accept)},
            "valid_and_novel": {"n": accept, "rate": rate(accept)},   # legacy alias = accept
            "no_gate_fail": {"n": no_gate_fail, "rate": rate(no_gate_fail)},
        },
    }


def run_group(records: list[dict], novelty: dict[str, dict], skip_checks: list[str],
              run_foldability: bool = False,
              pfam_hmm: Optional[Path] = None, mmseqs2_db: Optional[str] = None,
              mibig_gbk: Optional[Path] = None, antismash_db: Optional[Path] = None) -> list[dict]:
    from bgc_pipeline.evaluation import evaluate_bgc, EvalConfig, load_taxon_profiles
    cfg = EvalConfig(skip_checks=skip_checks, run_protein_foldability=run_foldability)
    tp = Path("data/processed/taxon_profiles.json")
    if tp.exists():
        cfg.taxon_profiles = load_taxon_profiles(tp)
    # antismash class match: map antiSMASH product types -> our harmonised compound
    # classes (T1PKS->PKS, lanthipeptide->RIPP, ...). Without this the antismash check
    # does an exact string match and fails real BGCs (PKS != T1PKS).
    cmap = Path("config/compound_class_map.yaml")
    if cmap.exists():
        from bgc_pipeline.class_map import load_class_map
        cfg.class_map, _ = load_class_map(cmap)
    # Wire optional reference DBs so antismash/class_markers/protein_homology can run.
    # Each check still self-skips if its path is missing, so passing them is safe.
    if pfam_hmm and pfam_hmm.exists():
        cfg.pfam_hmm_path = pfam_hmm
    if mibig_gbk and mibig_gbk.is_dir():
        cfg.mibig_gbk_dir = mibig_gbk
    if mmseqs2_db:
        cfg.mmseqs2_db = mmseqs2_db
    if antismash_db and antismash_db.is_dir():
        cfg.antismash_db_dir = str(antismash_db)
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
    ap.add_argument("--include-foldability", "--include-gpu-metrics",
                    dest="include_foldability", action="store_true",
                    help="Also run the optional ESMFold protein_foldability check (GPU).")
    ap.add_argument("--pfam-hmm", type=Path, default=None,
                    help="Pfam-A.hmm path -> enables the class_markers check.")
    ap.add_argument("--mmseqs2-db", type=str, default=None,
                    help="MMseqs2 protein DB (e.g. UniRef50) -> enables protein_homology.")
    ap.add_argument("--mibig-gbk", type=Path, default=None,
                    help="MIBiG GenBank dir (reserved; no check consumes it today).")
    ap.add_argument("--antismash-db", type=Path, default=None,
                    help="antiSMASH databases root -> points the antismash check at DBs.")
    ap.add_argument("--skip-checks", nargs="*", default=None,
                    help="Check NAMES to skip (see CHECKS). E.g. quick-eval uses "
                         "'--skip-checks protein_homology kmer_novelty' (slow / DB-bound).")
    ap.add_argument("--output", type=Path, default=Path("eval_suite_report.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    skip = sorted(set(args.skip_checks or []))   # check NAMES; protein_foldability is opt-in
    dbs = {"pfam_hmm": args.pfam_hmm, "mmseqs2_db": args.mmseqs2_db,
           "mibig_gbk": args.mibig_gbk, "antismash_db": args.antismash_db}
    novelty = load_novelty_map(args.novelty)
    gen = load_jsonl(args.gen)
    pos = load_jsonl(args.positive) if args.positive and args.positive.exists() else []

    enabled = [name for name, v in dbs.items() if v]
    print(f"Generated: {len(gen)} | positive control: {len(pos)} | "
          f"novelty entries: {len(novelty)} | skip checks: {skip or 'none'} | "
          f"opt-in DBs: {', '.join(enabled) or 'none (checks self-skip)'}", file=sys.stderr)
    if not novelty:
        print("  WARNING: no --novelty map -> the 'novel' question is UNVERIFIED, "
              "not a pass. Run memorization_check.py first.", file=sys.stderr)
    if args.dry_run:
        print("[dry-run] not evaluating.", file=sys.stderr)
        return

    report: dict[str, Any] = {}
    g_res = run_group(gen, novelty, skip, run_foldability=args.include_foldability, **dbs)
    report["generated"] = summarize_group(g_res)
    if pos:
        p_res = run_group(pos, novelty, skip, run_foldability=args.include_foldability, **dbs)
        report["positive_control"] = summarize_group(p_res)
    report["per_record"] = {"generated": g_res, "positive_control":
                            (p_res if pos else [])}
    args.output.write_text(json.dumps(report, indent=2, default=str))

    gh = report["generated"]["headline"]
    ph = report.get("positive_control", {}).get("headline", {})
    print("\nHEADLINES (generated | positive control):", file=sys.stderr)
    for key, label in (("generates_bgc", "generates a BGC (is_bgc)"),
                       ("correct_class", "correct class (correct_class)"),
                       ("biological_valid", "biologically valid (is_bgc + correct_class)"),
                       ("biological_valid_and_novel", "+ novel = ACCEPT")):
        g = gh.get(key, {}).get("rate")
        p = ph.get(key, {}).get("rate")
        print(f"  {label:<46} gen={g}  positive={p}", file=sys.stderr)

    def _rates(block_key: str, title: str) -> None:
        print(f"\n{title} (generated vs positive control):", file=sys.stderr)
        gm = report["generated"].get(block_key, {})
        pm = report.get("positive_control", {}).get(block_key, {})
        for m in gm:
            role = gm[m].get("role", "")[:4].upper().ljust(4)
            g = gm[m]["pass_rate"]
            p = pm.get(m, {}).get("pass_rate")
            print(f"  [{role}] {m:<22} gen={g}  positive={p}", file=sys.stderr)

    _rates("per_question", "per-QUESTION pass-rates")
    _rates("per_check", "per-CHECK pass-rates")
    print(f"  wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
