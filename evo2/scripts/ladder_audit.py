#!/usr/bin/env python
"""AUDIT THE LADDER: validate the new primary metric the way max_orf_aa was NOT, and fill the rungs.

WHY. `max_orf_aa` was adopted on BETWEEN-group evidence (generated is shorter than real) plus a
mechanistic story ("cannot sustain a reading frame"). It failed the WITHIN-group test: inside de
novo generations it does not track domain content at all (r = 0.051 at 2 kb, -0.120 at 6 kb).
`biosynthetic_fraction` currently rests on exactly the same kind of evidence -- a clean between-group
ladder, 0.015 -> 0.100 -> 0.464 -> 0.836. That is not enough, and adopting it on that basis would
repeat the error one metric later.

SO THIS SCRIPT ASKS TWO THINGS.

(1) VALIDATION. Within a single group, does biosynthetic_fraction predict the outcomes we actually
    care about -- antiSMASH detection and correct class? The seeded arm is the place to ask,
    because it is the only regime with real variance in detection (0.367) rather than a floor.
    Reported as AUROC, and against max_orf_aa on the identical sequences so the two are directly
    comparable. If it fails here, it is not a target either.

(2) THE MISSING RUNGS. The ladder jumps from "has a biosynthetic domain" to "antiSMASH detects a
    cluster", and antiSMASH needs domains CLUSTERED and, for assembly-line classes, ORDERED. Those
    intermediate properties have never been measured on generations:
      * how many distinct ORFs carry a biosynthetic domain (one domain is not a cluster);
      * how far apart they sit, as a share of the sequence;
      * co-orientation -- real BGC genes are largely on one strand;
      * module architecture -- ordered C-A-T / KS-AT-ACP, which the eval suite can already score.
    Whichever of these separates real from generated is a rung; whichever does not, is not.

A NOTE ON WHAT THE LADDER CANNOT SEE. Every rung here can be maximised by copying training data.
Novelty is therefore a GUARD on the ladder, not a rung of it, and any run optimising these metrics
must report it alongside or the improvement is uninterpretable.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from bgc_pipeline.evaluation import (  # noqa: E402
    OBLIGATE_DOMAINS, check_class_markers, check_module_architecture, find_orfs,
)

BIO = Path("/data2/ds85/pfam/biosynthetic_subset.hmm")
PFAM = Path("/data2/ds85/pfam/Pfam-A.hmm")
PFAM_FULL_FOR_MARKERS = Path("/data2/ds85/pfam/Pfam-A.hmm")
_C: dict = {}


def _hits(dig, hmm_path):
    """(best bitscore, [(pfam_acc, orf_index, score)]) over INCLUDED hits."""
    import pyhmmer
    from pyhmmer.plan7 import HMMFile
    key = str(hmm_path)
    if key not in _C:
        with HMMFile(key) as fh:
            _C[key] = list(fh)
    best, out = 0.0, []
    for th in pyhmmer.hmmsearch(_C[key], dig, E=1e-3):
        acc = th.query.accession
        acc = (acc.decode() if isinstance(acc, bytes) else str(acc or "")).split(".")[0]
        for hit in th:
            if not hit.included:
                continue
            nm = hit.name
            nm = nm.decode() if isinstance(nm, bytes) else str(nm)
            best = max(best, float(hit.score))
            out.append((acc, int(nm.split("_")[1]), float(hit.score)))
    return best, out


def one(job):
    from pyhmmer.easel import Alphabet, TextSequence
    tag, seq, cls, key = job
    orfs = find_orfs(seq)
    base = {"tag": tag, "cls": cls, "key": key, "n_orfs": len(orfs),
            "max_orf_aa": 0, "any": 0.0, "bio": 0.0, "frac": 0.0,
            "n_bio_orfs": 0, "n_bio_domains": 0, "bio_span_frac": 0.0,
            "co_orient": 0.0, "modules": 0, "in_order": 0,
            # WHICH biosynthetic accessions were hit. `bio` is a bitscore against the GLOBAL
            # ~91-model set and is deliberately class-agnostic; callers that need a CLASS-SPECIFIC
            # rate must intersect this with OBLIGATE_DOMAINS[cls]. Reading `bio > 0` as "on-class"
            # is what inverted the Phase-3 A0 result on 2026-08-14 -- see bugs.md.
            "bio_accs": []}
    if not orfs:
        return base
    alpha = Alphabet.amino()
    dig = [TextSequence(name=f"orf_{i}".encode(), sequence=o.aa_seq).digitize(alpha)
           for i, o in enumerate(orfs)]
    anyb, _ = _hits(dig, PFAM)
    biob, biohits = _hits(dig, BIO)
    base["max_orf_aa"] = max(len(o.aa_seq) for o in orfs)
    base["any"], base["bio"] = anyb, biob
    base["frac"] = (biob / anyb) if anyb > 0 else 0.0
    # --- clustering rungs ---
    bio_orfs = sorted({i for _, i, _ in biohits})
    base["n_bio_orfs"] = len(bio_orfs)
    base["n_bio_domains"] = len(biohits)
    base["bio_accs"] = sorted({a for a, _, _ in biohits})
    if bio_orfs:
        lo = min(orfs[i].start for i in bio_orfs)
        hi = max(orfs[i].end for i in bio_orfs)
        base["bio_span_frac"] = (hi - lo) / max(len(seq), 1)
    strands = [o.strand for o in orfs]
    base["co_orient"] = max(strands.count(1), strands.count(-1)) / len(strands)
    # --- architecture rung (assembly-line classes only) ---
    if cls in ("NRPS", "PKS", "PKS_NRPS_HYBRID"):
        m2 = check_class_markers(seq, expected_class=cls, pfam_hmm_path=PFAM_FULL_FOR_MARKERS)
        ma = check_module_architecture(m2, cls)
        base["modules"] = int(ma.get("complete_modules") or ma.get("n_modules") or 0)
        base["in_order"] = int(bool(ma.get("in_order") or ma.get("collinear")))
    return base


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-real", type=int, default=25)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/ladder_audit.json"))
    args = ap.parse_args()

    R = Path("/data2/ds85/bgcmodel_runs")
    jobs = []
    # seeded arm: the ONLY group with real variance in antiSMASH detection, so the only place a
    # within-group validation can be run at all.
    as_rows = {}
    tsv = R / "guided_decoding" / "antismash.tsv"
    by_arm: dict[str, list] = {}
    for r in csv.DictReader(tsv.open(), delimiter="\t"):
        by_arm.setdefault(r["arm"], []).append(r)
    for c in ("NRPS", "PKS", "TERPENE", "RIPP"):
        for rule in ("plain", "random", "best"):
            arm = f"gd_{c}_{rule}"
            p = R / "guided_decoding" / f"{arm}.jsonl"
            if not p.exists():
                continue
            recs = [json.loads(l) for l in p.open() if l.strip()]
            rows = by_arm.get(arm, [])
            if len(rows) != len(recs):
                continue
            for i, (rec, row) in enumerate(zip(recs, rows)):
                key = f"{arm}#{i}"
                as_rows[key] = {"is_bgc": row.get("is_bgc") in ("1", "True", "true"),
                                "correct": row.get("correct_class") in ("1", "True", "true")}
                jobs.append(("seeded", rec["sequence"], c, key))
    for f, tag, L in [("steer_titration/L16_b0.jsonl", "denovo", 2000),
                      ("steer_sweep/a0_control.jsonl", "denovo", 6000)]:
        p = R / f
        if p.exists():
            for i, rec in enumerate(json.loads(l) for l in p.open() if l.strip()):
                jobs.append((tag, rec["sequence"][:L], rec.get("compound_class") or "NRPS",
                             f"{f}#{i}"))
    n = 0
    for line in open("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"):
        r = json.loads(line)
        if r.get("compound_class") in ("NRPS", "PKS", "TERPENE", "RIPP") and len(r["sequence"]) >= 3000:
            jobs.append(("REAL", r["sequence"][:3000], r["compound_class"], f"real#{n}")); n += 1
            if n >= args.n_real:
                break
    print(f"[ladder] {len(jobs)} sequences", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, jobs))
    args.out_json.write_text(json.dumps(rows, indent=1))

    print("\n" + "=" * 96)
    print("PART 1 — WITHIN-GROUP VALIDATION (seeded arm; the only one with variance in detection)")
    print("AUROC for predicting the independent antiSMASH outcome. 0.5 = no information.")
    print("=" * 96)
    sd = [r for r in rows if r["tag"] == "seeded" and r["key"] in as_rows]
    for outcome in ("is_bgc", "correct"):
        y = [as_rows[r["key"]][outcome] for r in sd]
        print(f"\n  predicting {outcome}  (n={len(sd)}, positives={sum(y)})")
        for m in ("frac", "bio", "any", "max_orf_aa", "n_bio_orfs", "n_bio_domains",
                  "bio_span_frac", "co_orient", "modules"):
            print(f"    {m:>15} AUROC {auroc([r[m] for r in sd], y):.3f}")

    print("\n" + "=" * 96)
    print("PART 2 — THE MISSING RUNGS: do they separate real from generated?")
    print("=" * 96)
    cols = ("n_orfs", "max_orf_aa", "frac", "n_bio_orfs", "n_bio_domains", "bio_span_frac",
            "co_orient", "modules", "in_order")
    print(f"\n{'group':>8} {'n':>4} " + " ".join(f"{c[:12]:>13}" for c in cols))
    for g in ("denovo", "seeded", "REAL"):
        sub = [r for r in rows if r["tag"] == g]
        if not sub:
            continue
        print(f"{g:>8} {len(sub):>4} " + " ".join(
            f"{st.mean(r[c] for r in sub):>13.3f}" for c in cols))
    print("\nHOW TO READ THIS. Part 1 is the test max_orf_aa failed: a metric that does not predict")
    print("the independent outcome WITHIN a group is not a target, however clean its between-group")
    print("ladder looks. Part 2 says which of the untested intermediate properties actually")
    print("separate real from generated — those are rungs; the rest are not.")
    print("\nGUARD, not a rung: every metric here is maximised by copying training data. Any run")
    print("optimising them must report novelty alongside or the improvement is uninterpretable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
