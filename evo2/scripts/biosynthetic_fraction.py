#!/usr/bin/env python
"""THE CANDIDATE TARGET METRIC: what share of the protein the model writes is BIOSYNTHETIC?

WHY max_orf_aa IS NOT THE RIGHT TARGET. Measured 2026-08-12, within de novo generations, ORF length
does not track domain content at all: r = 0.051 at a 2 kb window and -0.120 at 6 kb (it does track
in the SEEDED regime, r = 0.473, which is the regime that already works). And only 3 of 64 codons
are stops, so a model can lengthen ORFs by learning to avoid three triplets without becoming any
more protein-like. Optimising it risks producing longer garbage.

WHAT THE DE NOVO OUTPUT ACTUALLY IS. Scanned against the FULL Pfam-A rather than the biosynthetic
subset, de novo generations hit real protein families 69-100% of the time (mean 13.6 families at
6 kb, best bitscore 102) - phage integrases, MFS transporters, ankyrin repeats, cyclins. The model
writes genuine, recognisable protein. It writes the WRONG protein.

So the failure is SPECIFICITY, not coherence, and the metric should measure specificity:

    biosynthetic_fraction = best bitscore against BIOSYNTHETIC Pfams
                            -------------------------------------------
                            best bitscore against ANY Pfam

Properties that max_orf_aa lacks: it is non-zero for essentially every sequence (the denominator
almost always fires), it cannot be gamed by avoiding stop codons, and it measures the actual gap
rather than a proxy for it. Both scans run on the SAME sequences here, because the ratio is
meaningless across different subsets.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bgc_pipeline.evaluation import find_orfs, OBLIGATE_DOMAINS  # noqa: E402

BIO = Path("/data2/ds85/pfam/biosynthetic_subset.hmm")
PFAM = Path("/data2/ds85/pfam/Pfam-A.hmm")
_C: dict = {}

def _best(dig, hmm_path, want=None):
    import pyhmmer
    from pyhmmer.plan7 import HMMFile
    key = str(hmm_path)
    if key not in _C:
        with HMMFile(key) as fh:
            _C[key] = list(fh)
    best = 0.0
    for th in pyhmmer.hmmsearch(_C[key], dig, E=1e-3):
        acc = th.query.accession
        acc = (acc.decode() if isinstance(acc, bytes) else str(acc or "")).split(".")[0]
        if want is not None and acc not in want:
            continue
        for hit in th:
            if hit.included:
                best = max(best, float(hit.score))
    return best

def one(job):
    from pyhmmer.easel import Alphabet, TextSequence
    tag, seq, cls, L = job
    orfs = find_orfs(seq[:L])
    if not orfs:
        return {"tag": tag, "cls": cls, "any": 0.0, "bio": 0.0, "cls_bio": 0.0, "frac": 0.0}
    alpha = Alphabet.amino()
    dig = [TextSequence(name=f"o{i}".encode(), sequence=o.aa_seq).digitize(alpha)
           for i, o in enumerate(orfs)]
    anyb = _best(dig, PFAM)
    biob = _best(dig, BIO)
    clsb = _best(dig, BIO, want=set(OBLIGATE_DOMAINS.get(cls, []) or []))
    return {"tag": tag, "cls": cls, "any": anyb, "bio": biob, "cls_bio": clsb,
            "frac": (biob / anyb) if anyb > 0 else 0.0,
            "cls_frac": (clsb / anyb) if anyb > 0 else 0.0}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-real", type=int, default=25)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/biosynthetic_fraction.json"))
    args = ap.parse_args()
    if not BIO.exists():
        raise SystemExit(f"[bf] ABORT: no biosynthetic subset at {BIO} — "
                         f"run scripts/build_domain_spans.py first, it builds this.")
    jobs = []
    for f, tag, L in [("steer_titration/L16_b0.jsonl", "denovo_2k", 2000),
                      ("steer_magnitude/L16_d0.jsonl", "denovo_2k", 2000),
                      ("steer_sweep/a0_control.jsonl", "denovo_6k", 6000),
                      ("guided_decoding/gd_NRPS_plain.jsonl", "seeded_3k", 3000),
                      ("guided_decoding/gd_PKS_plain.jsonl", "seeded_3k", 3000),
                      ("guided_decoding/gd_TERPENE_plain.jsonl", "seeded_3k", 3000),
                      ("guided_decoding/gd_RIPP_plain.jsonl", "seeded_3k", 3000)]:
        p = Path("/data2/ds85/bgcmodel_runs") / f
        if not p.exists():
            continue
        for r in (json.loads(l) for l in p.open() if l.strip()):
            if len(r["sequence"]) >= L:
                jobs.append((tag, r["sequence"], r.get("compound_class") or "NRPS", L))
    n = 0
    for line in open("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"):
        r = json.loads(line)
        if r.get("compound_class") in ("NRPS", "PKS", "TERPENE", "RIPP") and len(r["sequence"]) >= 3000:
            jobs.append(("REAL_3k", r["sequence"], r["compound_class"], 3000)); n += 1
            if n >= args.n_real:
                break
    print(f"[bf] {len(jobs)} sequences, both scans on each", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, jobs))
    args.out_json.write_text(json.dumps(rows, indent=1))
    print(f"\n{'group':>11} {'n':>4} {'best ANY':>9} {'best BIO':>9} {'best CLASS':>11} "
          f"{'BIO fraction':>13} {'>0':>5}")
    for g in ("denovo_2k", "denovo_6k", "seeded_3k", "REAL_3k"):
        sub = [r for r in rows if r["tag"] == g]
        if not sub:
            continue
        print(f"{g:>11} {len(sub):>4} {st.mean(r['any'] for r in sub):>9.1f} "
              f"{st.mean(r['bio'] for r in sub):>9.1f} {st.mean(r['cls_bio'] for r in sub):>11.1f} "
              f"{st.mean(r['frac'] for r in sub):>13.3f} "
              f"{st.mean(1 for r in sub if r['any'] > 0) * len([r for r in sub if r['any'] > 0]) / len(sub) / max(len([r for r in sub if r['any'] > 0]), 1):>5.2f}")
    print(f"\nfraction of sequences with a NON-ZERO denominator (i.e. metric is defined):")
    for g in ("denovo_2k", "denovo_6k", "seeded_3k", "REAL_3k"):
        sub = [r for r in rows if r["tag"] == g]
        if sub:
            print(f"  {g:>11}: {sum(1 for r in sub if r['any'] > 0) / len(sub):.2f}")
    print(f"\nwrote {args.out_json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
