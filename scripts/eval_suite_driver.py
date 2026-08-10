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
    # A rate must be computed over the records where the question was actually EVALUATED.
    # Previously every rate was k/n over ALL records with numerators counting only == "PASS", so a
    # question that was never measured ("skipped" / "no_verdict") landed in the DENOMINATOR and
    # dragged the rate toward zero. Because evo2/scripts/quick_eval.sh always passes
    # `--skip-checks protein_homology kmer_novelty`, `novel` is "skipped" for every record — so
    # the project's ACCEPT rate (biological_valid_and_novel) was STRUCTURALLY 0.000 in every
    # quick-eval ever run, regardless of how good the generations were.
    gen_bgc = correct_class = bio_valid = accept = no_gate_fail = 0
    d_bgc = d_cls = d_bio = d_acc = d_gate = 0
    d_funnel = f_bgc = f_cls = f_bio = 0

    def _evaluated(v) -> bool:
        return v in ("PASS", "FAIL")

    for r in results:
        s = r.get("questions", {})
        g_bgc = s.get("is_bgc") == "PASS"
        c_cls = s.get("correct_class") == "PASS"
        novel = s.get("novel") == "PASS"
        bio = g_bgc and c_cls
        gen_bgc += int(g_bgc);       d_bgc += int(_evaluated(s.get("is_bgc")))
        correct_class += int(c_cls); d_cls += int(_evaluated(s.get("correct_class")))
        bio_valid += int(bio)
        d_bio += int(_evaluated(s.get("is_bgc")) and _evaluated(s.get("correct_class")))
        accept += int(bio and novel)
        d_acc += int(_evaluated(s.get("is_bgc")) and _evaluated(s.get("correct_class"))
                     and _evaluated(s.get("novel")))
        # no_gate_fail was the INVERSE bug: an unmeasured gate is not "FAIL", so skipped gates
        # counted as clean passes, inflating it exactly when the instrument was least configured.
        no_gate_fail += int(all(s.get(q) == "PASS" for q in GATE_QUESTIONS))
        d_gate += int(all(_evaluated(s.get(q)) for q in GATE_QUESTIONS))
        # Funnel numerators, counted ONLY on records where every funnel gate was evaluated, so
        # generates_bgc >= biological_valid >= ACCEPT holds by construction.
        if _evaluated(s.get("is_bgc")) and _evaluated(s.get("correct_class")) and _evaluated(s.get("novel")):
            d_funnel += 1
            f_bgc += int(g_bgc); f_cls += int(c_cls); f_bio += int(bio)

    # ONE DENOMINATOR FOR THE WHOLE FUNNEL. Per-metric `evaluated` counts fixed the
    # divide-by-unmeasured bug but broke MONOTONICITY: generates_bgc >= biological_valid >=
    # ACCEPT must hold by construction (each stage is a strict subset of the previous), yet
    # with d_bgc=10 and d_bio=2 a run scored generates_bgc 0.20 and biological_valid 1.00 --
    # a subset outscoring its superset, which is not interpretable as a funnel at all.
    # The funnel denominator is therefore the records where EVERY gate it depends on was
    # evaluated (d_acc), so all stages are comparable; each metric still reports its own
    # `evaluated` count so a reader can see how much was measurable.
    def rate(k: int, d: Optional[int] = None) -> Optional[float]:
        """None when nothing was evaluated — NEVER 0.0, which reads as a measured total failure."""
        d = n if d is None else d
        return round(k / d, 3) if d else None
    # Which instrument produced each gate verdict, aggregated. A correct_class rate derived from
    # the Pfam proxy is NOT comparable to one derived from antiSMASH (measured on 768 paired
    # records: proxy precision 0.366, and it reports 0.249 where antiSMASH reports 0.094).
    src_tally: dict[str, dict[str, int]] = {}
    for r in results:
        for gate, src in (r.get("questions", {}).get("_verdict_source", {}) or {}).items():
            src_tally.setdefault(gate, {}).setdefault(src, 0)
            src_tally[gate][src] += 1
    proxy_warn = [g for g, d in src_tally.items() if d.get("class_markers_proxy", 0) or
                  d.get("coding_floor_only", 0)]

    return {
        "n": n, "per_question": per_q, "per_check": per_c,
        # The monotone view: every stage on the SAME records (all funnel gates evaluated), so
        # generates_bgc >= biological_valid >= accept is guaranteed and the stages are comparable.
        "funnel": {"denominator": d_funnel,
                   "generates_bgc": rate(f_bgc, d_funnel),
                   "correct_class": rate(f_cls, d_funnel),
                   "biological_valid": rate(f_bio, d_funnel),
                   "accept": rate(accept, d_funnel)},
        "verdict_source": src_tally,
        "verdict_source_warning": (
            f"gate(s) {proxy_warn} were derived from a PROXY, not antiSMASH — the Pfam proxy has "
            f"precision 0.366 for correct_class and inflates it ~2.6x; these rates are NOT "
            f"comparable to antiSMASH-derived ones" if proxy_warn else None),
        "roles": {"gates": list(GATE_QUESTIONS), "diagnostics": list(DIAGNOSTIC_QUESTIONS)},
        "headline": {
            # TWO VIEWS, because one denominator cannot serve both purposes:
            #   * these per-metric rates use each metric's OWN evaluable records, so
            #     generates_bgc still reports when novelty was skipped (which quick_eval
            #     always does) instead of collapsing the whole block to None;
            #   * `funnel` below re-computes the same stages on the COMMON subset, where
            #     bgc >= valid >= accept holds by construction.
            # Reporting only the common-subset view hid measurable numbers; reporting only the
            # per-metric view produced a subset outscoring its superset (0.20 vs 1.00).
            "generates_bgc": {"n": gen_bgc, "evaluated": d_bgc, "rate": rate(gen_bgc, d_bgc)},
            "correct_class": {"n": correct_class, "evaluated": d_cls, "rate": rate(correct_class, d_cls)},
            "biological_valid": {"n": bio_valid, "evaluated": d_bio, "rate": rate(bio_valid, d_bio)},
            "biological_valid_and_novel": {"n": accept, "evaluated": d_acc, "rate": rate(accept, d_acc)},
            "valid_and_novel": {"n": accept, "evaluated": d_acc, "rate": rate(accept, d_acc)},   # legacy alias
            "no_gate_fail": {"n": no_gate_fail, "evaluated": d_gate, "rate": rate(no_gate_fail, d_gate)},
        },
    }


_REPO_ROOT = Path(__file__).resolve().parents[1]


def run_group(records: list[dict], novelty: dict[str, dict], skip_checks: list[str],
              run_foldability: bool = False, probe_scores: Optional[dict[str, dict]] = None,
              pfam_hmm: Optional[Path] = None, mmseqs2_db: Optional[str] = None,
              mibig_gbk: Optional[Path] = None, antismash_db: Optional[Path] = None) -> list[dict]:
    from bgc_pipeline.evaluation import evaluate_bgc, EvalConfig, load_taxon_profiles
    cfg = EvalConfig(skip_checks=skip_checks, run_protein_foldability=run_foldability)
    # antismash class match: map antiSMASH product types -> our harmonised compound
    # classes (T1PKS->PKS, lanthipeptide->RIPP, ...). Without this the antismash check
    # does an exact string match and fails real BGCs (PKS != T1PKS).
    # REPO-ANCHORED, not CWD-relative. This module is invoked from several working directories
    # (quick_eval.sh, run_eval.sh, the probe drivers, the GenomeOcean recipe); from any of them a
    # bare "config/..." silently missed and the class map was dropped -- which is failure mode #2
    # from 2026-07-31: antiSMASH "T1PKS" vs our "PKS", correct_class collapsing to ~0.
    _REPO = Path(__file__).resolve().parents[1]
    cmap = _REPO / "config" / "compound_class_map.yaml"
    if not cmap.exists():
        raise SystemExit(f"eval_suite_driver: class map not found at {cmap}. Without it "
                         f"correct_class is silently wrong (antiSMASH product names are not our "
                         f"class vocabulary).")
    from bgc_pipeline.class_map import load_class_map
    cfg.class_map, _ = load_class_map(cmap)
    # Wire optional reference DBs so antismash/class_markers/protein_homology can run.
    # Each check still self-skips if its path is missing, so passing them is safe.
    # A path that was SUPPLIED but does not resolve is a typo, not a deliberate omission. Warn
    # loudly and record it, so "I passed --pfam-hmm and got zeros" is distinguishable from
    # "I never passed it". Previously both produced byte-identical output.
    unresolved: list[str] = []
    for label, val, ok in (("--pfam-hmm", pfam_hmm, bool(pfam_hmm and pfam_hmm.exists())),
                           ("--mibig-gbk", mibig_gbk, bool(mibig_gbk and mibig_gbk.is_dir())),
                           ("--antismash-db", antismash_db, bool(antismash_db and antismash_db.is_dir()))):
        if val and not ok:
            unresolved.append(f"{label}={val}")
    if unresolved:
        print(f"WARNING eval_suite_driver: supplied but unresolvable, check(s) will DEGRADE: "
              f"{', '.join(unresolved)}", file=sys.stderr)
        cfg_unresolved = unresolved
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
        # PRECEDENCE MUST MATCH scripts/memorization_check.py:189, which builds ids as
        # `accession or id or str(i)`. The opposite order here meant the novelty lookup missed
        # and every arm was scored against the first arm's containment values.
        sid = rec.get("accession") or rec.get("id") or str(i)
        nov = novelty.get(sid)
        res = evaluate_bgc(
            rec.get("sequence", ""), accession=sid,
            expected_class=rec.get("compound_class", ""),
            expected_taxon=rec.get("taxonomic_tag", ""),
            config=cfg, novelty=nov,
            probe_scores=(probe_scores or {}).get(sid),
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
                    default=_REPO_ROOT / "eval" / "positive_control_mibig.jsonl",
                    help="Real-BGC positive control JSONL. Repo-anchored: a CWD-relative default "
                         "silently missed from every driver not run at the repo root.")
    ap.add_argument("--probe-scores", type=Path, default=None,
                    help="Sidecar {record id -> {class: probability}} from "
                         "evo2/scripts/probe_score_generations.py --emit-sidecar. Enables the "
                         "CONTINUOUS `class_probe` diagnostic, which can see class shifts too "
                         "small for the binary gates (class_markers TPR is 0.717 at 3 kb, and "
                         "antiSMASH detects only ~1/3 of seeded 3 kb generations). It NEVER "
                         "gates — see check_class_probe.")
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
    # A SUPPLIED-but-unresolvable sidecar is a typo, not an omission: without this the check
    # skips on every record and the report looks identical to never having asked for it.
    probe_scores: dict[str, dict] = {}
    if args.probe_scores:
        if not args.probe_scores.exists():
            raise SystemExit(f"eval_suite_driver: --probe-scores {args.probe_scores} does not "
                             f"exist. Generate it with probe_score_generations.py --emit-sidecar.")
        probe_scores = json.loads(args.probe_scores.read_text())
        if not probe_scores:
            raise SystemExit(f"eval_suite_driver: --probe-scores {args.probe_scores} is EMPTY — "
                             f"the class_probe check would skip on every record.")
    gen = load_jsonl(args.gen)
    pos = load_jsonl(args.positive) if args.positive and args.positive.exists() else []
    # LOUD. Without a control, every rate is an uncalibrated fraction of an UNSTATED maximum:
    # antiSMASH scores real curated BGCs at only 0.55 correct_class, and real splits_core cores
    # truncated to 2048 nt score is_bgc 0.680 -- so a generation at 0.10 is 16% of achievable,
    # not 10% of perfect. Measured 2026-08-10: 0 of 25 reports on disk had a control, because
    # six drivers passed a deliberately-EMPTY `_nopos.jsonl` (which exists, so an exists() check
    # never fired). Warn on empty as well as missing, and say how to fix it.
    if not pos:
        why = ("does not exist" if not (args.positive and args.positive.exists())
               else "is EMPTY")
        print(f"WARNING: positive control {args.positive} {why} — every rate in this report will "
              f"be UNCALIBRATED (no ceiling to compare against). Generate one with:\n"
              f"    python scripts/make_positive_control.py --gen {args.gen} "
              f"--out <dir>/positive_control.jsonl", file=sys.stderr)

    # Report what actually RESOLVED. Listing a DB as enabled because the flag was merely set
    # made the run log assert a configuration that run_group had silently dropped.
    def _resolves(v) -> bool:
        if not v:
            return False
        pv = Path(v)
        return pv.exists() or pv.is_dir() or not isinstance(v, (str, Path))
    enabled = [name for name, v in dbs.items() if _resolves(v)]
    missing = [f"{name}(unresolved)" for name, v in dbs.items() if v and not _resolves(v)]
    enabled += missing
    print(f"Generated: {len(gen)} | positive control: {len(pos)} | "
          f"novelty entries: {len(novelty)} | probe scores: {len(probe_scores)} | "
          f"skip checks: {skip or 'none'} | "
          f"opt-in DBs: {', '.join(enabled) or 'none (checks self-skip)'}", file=sys.stderr)
    if not novelty:
        print("  WARNING: no --novelty map -> the 'novel' question is UNVERIFIED, "
              "not a pass. Run memorization_check.py first.", file=sys.stderr)
    if args.dry_run:
        print("[dry-run] not evaluating.", file=sys.stderr)
        return

    report: dict[str, Any] = {}
    g_res = run_group(gen, novelty, skip, run_foldability=args.include_foldability,
                      probe_scores=probe_scores, **dbs)
    report["generated"] = summarize_group(g_res)
    if pos:
        # The control gets the SAME instrument, probe included, or the ceiling it establishes
        # would be for a different measurement than the one the generations got.
        p_res = run_group(pos, novelty, skip, run_foldability=args.include_foldability,
                          probe_scores=probe_scores, **dbs)
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
