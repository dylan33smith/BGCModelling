#!/usr/bin/env python
"""SCORE THE LADDER on a generations file, for per-checkpoint training curves.

WHY THIS EXISTS. `eval_milestones_watch.sh` tracked `is_bgc / correct_class / class_markers /
any_domain_rate / coding_density`. De novo, the first two read ~0.012 and ~0, so a training run
would have printed zeros for days and told us nothing -- which is exactly how the original long
fine-tune went wrong (train loss fell, val flat, gates zero, no information). `quick_eval` also
explicitly skips `kmer_novelty`, so the anti-memorisation guard was not running either.

This scores the rungs that are NON-ZERO today and that were validated against the independent
antiSMASH outcome (`ladder_audit.py`, AUROC on n=120 with 44 detections):

    best_bio_bits   0.950   PRIMARY -- best bitscore vs biosynthetic Pfams
    n_bio_domains   0.919   how many biosynthetic domains at all
    bio_span_frac   0.896   how far apart they sit = is it a CLUSTER
    bio_fraction    0.893   specificity: biosynthetic share of all Pfam signal
    max_orf_aa      0.709   structural diagnostic ONLY (no de novo signal: r = 0.051 / -0.120)

plus NOVELTY, which is a CONSTRAINT and not a rung: every metric above is maximised by copying
training data, so an improvement with novelty unverified is uninterpretable. Reported here as max
k-mer containment against the training corpus; >= 0.95 is memorised.

Reference points for reading a curve (from `biosynthetic_fraction.py` / `ladder_audit.py`):

                    base Evo2   LoRA step_1200   REAL cores
    best_bio_bits       0.0          56.9           148.6
    n_bio_domains        --           0.20            2.48
    bio_span_frac        --           0.051           0.876
    bio_fraction       0.000          0.399           0.836
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "evo2" / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True, help="generations jsonl (quick_eval gen.jsonl)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--ref", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/train.jsonl"),
                    help="corpus the novelty guard scans (the TRAINING set — that is what could "
                         "have been memorised)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-novelty", action="store_true",
                    help="ONLY for a quick smoke test. A skipped novelty gate means novelty is "
                         "UNVERIFIED, never that it passed.")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.gen.open() if l.strip()]
    recs = [r for r in recs if len(r.get("sequence", "")) >= 200]
    if not recs:
        raise SystemExit(f"[ladder] ABORT: no scoreable sequences in {args.gen}")

    from ladder_audit import one
    from concurrent.futures import ProcessPoolExecutor
    jobs = [("gen", r["sequence"], r.get("compound_class") or "NRPS", f"g{i}")
            for i, r in enumerate(recs)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, jobs))

    out = {
        "n": len(rows),
        "best_bio_bits": round(st.mean(r["bio"] for r in rows), 3),
        "n_bio_domains": round(st.mean(r["n_bio_domains"] for r in rows), 3),
        "bio_span_frac": round(st.mean(r["bio_span_frac"] for r in rows), 4),
        "bio_fraction": round(st.mean(r["frac"] for r in rows), 4),
        "frac_with_bio_signal": round(sum(1 for r in rows if r["bio"] > 0) / len(rows), 3),
        "max_orf_aa": round(st.mean(r["max_orf_aa"] for r in rows), 1),
        "n_orfs": round(st.mean(r["n_orfs"] for r in rows), 2),
        "best_any_bits": round(st.mean(r["any"] for r in rows), 3),
    }

    # ---- NOVELTY: a CONSTRAINT, and a skip is never a pass ----------------------------------
    if args.skip_novelty:
        out["novelty"] = "SKIPPED — UNVERIFIED, not a pass"
    else:
        try:
            from memorization_check import scan_corpus
            nov = scan_corpus([(f"g{i}", r["sequence"]) for i, r in enumerate(recs)], args.ref)
            cont = [float(x["max_containment"]) for x in nov if "max_containment" in x]
            if not cont:
                out["novelty"] = "SKIPPED — scan returned no max_containment; UNVERIFIED"
            else:
                out["novelty_mean_containment"] = round(st.mean(cont), 4)
                out["novelty_max_containment"] = round(max(cont), 4)
                out["novelty_frac_memorised"] = round(sum(1 for c in cont if c >= 0.95) / len(cont), 3)
                out["novelty_verdict"] = ("FAIL_memorised" if max(cont) >= 0.95
                                          else "WARN" if max(cont) >= 0.80 else "PASS_novel")
        except Exception as e:                       # never let a novelty failure read as a pass
            out["novelty"] = f"SKIPPED ({e}) — UNVERIFIED, not a pass"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\n[ladder] reference: base 0.0 | LoRA step_1200 56.9 | REAL 148.6  (best_bio_bits)")
    print(f"[ladder] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
