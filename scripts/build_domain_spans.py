#!/usr/bin/env python
"""PER-POSITION DOMAIN LABELS: which nucleotides of a training core actually encode the machinery?

WHY THIS EXISTS. Next-base prediction spreads its gradient uniformly over every position in a
core. Measured 2026-08-12: 10 nt of context already delivers 73% of everything the model achieves,
all long-range context is worth 0.149 nats, and the class tag is worth -0.0006. So the objective
is dominated by local sequence statistics, and the handful of catalytic domains that actually make
a cluster an NRPS get the same per-position weight as the linkers and transporters around them.

A domain-weighted loss needs to know WHERE those domains are, in nucleotide coordinates. Nothing on
disk has that: training records carry `strict_core_genes` as a COUNT, region start/end, and the raw
sequence — no per-domain spans. This script produces them.

WHAT IT DOES, per record:
  1. pyrodigal calls genes (real starts, strand, no six-frame fragmentation) -> ORFs with
     nucleotide coordinates and their translations;
  2. pyhmmer scans those proteins against ONLY the biosynthetic Pfam accessions the eval suite
     already uses (`_BIOSYNTHETIC_PFAMS`, ~100 models, not all ~20,000 of Pfam-A);
  3. each domain hit's amino-acid envelope is mapped BACK to nucleotide coordinates, honouring
     strand, and written as a span.

Output sidecar `<split>.domain_spans.jsonl`, one line per record, aligned by accession.

IT ALSO MEASURES THE PREMISE. The rationale for domain-weighting is that the class-defining part of
a core is a small fraction of it and therefore a small fraction of the gradient. That fraction has
been asserted at "~5%" and never measured. The summary at the end reports it per class, so the
weighting scheme is designed against a number rather than a guess -- and if the fraction turns out
to be large, the rationale weakens and that should be visible before any training run is spent.

COORDINATE CONVENTION: 0-based, half-open [start, end), on the FORWARD strand of the stored
sequence, matching `find_orfs`. A reverse-strand domain still gets forward-strand coordinates, so a
consumer can mask positions without knowing about strand at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bgc_pipeline.evaluation import (  # noqa: E402
    OBLIGATE_DOMAINS, _BIOSYNTHETIC_PFAMS, find_orfs,
)


def aa_span_to_nt(orf, aa_from: int, aa_to: int) -> tuple[int, int]:
    """Map a 1-based inclusive amino-acid envelope onto forward-strand nucleotide coordinates.

    STRAND IS THE TRAP. For a reverse-strand gene the protein is translated from the reverse
    complement, so residue 1 sits at the ORF's HIGH nucleotide coordinate and the envelope runs
    backwards. Getting this wrong silently mislabels roughly half of all domains -- and because the
    spans would still be plausible lengths inside real ORFs, nothing downstream would complain.
    """
    n_aa = max(aa_to - aa_from + 1, 0)
    if orf.strand == -1:
        end = orf.end - 3 * (aa_from - 1)
        start = end - 3 * n_aa
    else:
        start = orf.start + 3 * (aa_from - 1)
        end = start + 3 * n_aa
    return max(orf.start, min(start, orf.end)), max(orf.start, min(end, orf.end))


def _subset_hmm(pfam: Path, out: Path, accessions: set[str]) -> Path:
    """Write a small HMM file holding only the biosynthetic models.

    Scanning all of Pfam-A per record would dominate the runtime; the eval suite tolerates it for a
    few dozen generations, but this pass covers 47.5k cores.
    """
    if out.exists():
        return out
    from pyhmmer.plan7 import HMMFile
    kept = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with HMMFile(str(pfam)) as fh, out.open("wb") as w:
        for hmm in fh:
            acc = hmm.accession
            acc = acc.decode() if isinstance(acc, bytes) else str(acc or "")
            if acc.split(".")[0] in accessions:
                hmm.write(w)
                kept += 1
    print(f"[spans] wrote {kept}/{len(accessions)} biosynthetic HMMs -> {out.name}", flush=True)
    if kept == 0:
        raise SystemExit(f"[spans] ABORT: no biosynthetic HMMs matched in {pfam} — a silently "
                         f"empty model set would label every core as domain-free.")
    return out


_HMM_CACHE: dict = {}


def annotate(job):
    """Module-level for ProcessPoolExecutor picklability."""
    rec, hmm_path, evalue = job
    import pyhmmer
    from pyhmmer.easel import Alphabet, TextSequence
    from pyhmmer.plan7 import HMMFile

    seq = rec["sequence"]
    try:
        orfs = find_orfs(seq)
    except Exception as e:                       # a gene-caller failure must not look domain-free
        return {"accession": rec.get("accession"), "error": f"find_orfs: {e}"}
    if not orfs:
        return {"accession": rec.get("accession"), "compound_class": rec.get("compound_class"),
                "length": len(seq), "n_orfs": 0, "spans": [], "n_domains": 0,
                "domain_nt": 0, "class_domain_nt": 0}

    alphabet = Alphabet.amino()
    digital = [TextSequence(name=f"orf_{i}".encode(), sequence=o.aa_seq).digitize(alphabet)
               for i, o in enumerate(orfs)]

    key = str(hmm_path)
    if key not in _HMM_CACHE:
        with HMMFile(key) as fh:
            _HMM_CACHE[key] = list(fh)
    hmms = _HMM_CACHE[key]

    cls = rec.get("compound_class") or ""
    class_accs = set(OBLIGATE_DOMAINS.get(cls, []) or [])
    spans = []
    for top_hits in pyhmmer.hmmsearch(hmms, digital, E=evalue):
        acc = top_hits.query.accession
        acc = (acc.decode() if isinstance(acc, bytes) else str(acc or "")).split(".")[0]
        for hit in top_hits:
            if not hit.included:
                continue
            hn = hit.name
            hn = hn.decode() if isinstance(hn, bytes) else str(hn)
            idx = int(hn.split("_")[1])
            for dom in hit.domains:
                if not dom.included:
                    continue
                a, b = int(dom.env_from), int(dom.env_to)
                s, e = aa_span_to_nt(orfs[idx], a, b)
                if e > s:
                    spans.append([s, e, acc, int(acc in class_accs)])
    spans.sort()
    # Union of covered nucleotides — overlapping domains must not be double-counted, or the
    # "what fraction of a core is machinery" number inflates past 1.0 without ever erroring.
    def _union(sp):
        tot, cur_s, cur_e = 0, None, None
        for s, e, *_ in sp:
            if cur_e is None or s > cur_e:
                tot += (cur_e - cur_s) if cur_e is not None else 0
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        return tot + ((cur_e - cur_s) if cur_e is not None else 0)

    # GENE spans are needed by the frame-aware loss: codon phase at position p inside a gene
    # starting at s is (p - s) % 3 on the forward strand, counted from the END on the reverse.
    # find_orfs already computes these and the first version of this script threw them away.
    genes = [[o.start, o.end, o.strand] for o in orfs]
    return {"accession": rec.get("accession"), "compound_class": cls, "length": len(seq),
            "n_orfs": len(orfs), "spans": spans, "n_domains": len(spans),
            "genes": genes,
            "domain_nt": _union(spans),
            "class_domain_nt": _union([s for s in spans if s[3]])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core"))
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--pfam", type=Path, default=Path("/data2/ds85/pfam/Pfam-A.hmm"))
    ap.add_argument("--hmm-subset", type=Path,
                    default=Path("/data2/ds85/pfam/biosynthetic_subset.hmm"))
    ap.add_argument("--evalue", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="0 = all records (debug aid).")
    args = ap.parse_args()

    if not args.pfam.exists():
        raise SystemExit(f"[spans] ABORT: no Pfam HMM at {args.pfam}")
    hmm = _subset_hmm(args.pfam, args.hmm_subset, _BIOSYNTHETIC_PFAMS)
    print(f"[spans] {len(_BIOSYNTHETIC_PFAMS)} biosynthetic accessions across "
          f"{len(OBLIGATE_DOMAINS)} classes", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    import statistics as st

    for split in args.splits:
        src = args.data_dir / f"{split}.jsonl"
        out = args.data_dir / f"{split}.domain_spans.jsonl"
        recs = [json.loads(l) for l in src.open()]
        if args.limit:
            recs = recs[: args.limit]
        print(f"\n[spans] {split}: {len(recs):,} records", flush=True)

        rows, errs = [], 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex, out.open("w") as w:
            for i, r in enumerate(ex.map(annotate,
                                         ((rec, str(hmm), args.evalue) for rec in recs),
                                         chunksize=8)):
                if r.get("error"):
                    errs += 1
                else:
                    rows.append(r)
                w.write(json.dumps(r) + "\n")
                if (i + 1) % 2000 == 0:
                    print(f"[spans]   {i + 1:,}/{len(recs):,}", flush=True)
        print(f"[spans] wrote {out}  ({errs} errors)")

        print(f"\n{'class':>18} {'n':>6} {'median nt':>10} {'domains':>8} "
              f"{'ALL-domain %':>13} {'CLASS-domain %':>15}")
        byc: dict[str, list] = {}
        for r in rows:
            byc.setdefault(r["compound_class"], []).append(r)
        for c in sorted(byc, key=lambda k: -len(byc[k])):
            v = byc[c]
            med = sorted(x["length"] for x in v)[len(v) // 2]
            print(f"{c:>18} {len(v):>6,} {med:>10,} "
                  f"{st.mean(x['n_domains'] for x in v):>8.2f} "
                  f"{st.mean(x['domain_nt'] / max(x['length'],1) for x in v):>12.1%} "
                  f"{st.mean(x['class_domain_nt'] / max(x['length'],1) for x in v):>14.1%}")
        allr = rows
        print(f"{'POOLED':>18} {len(allr):>6,} {'':>10} "
              f"{st.mean(x['n_domains'] for x in allr):>8.2f} "
              f"{st.mean(x['domain_nt'] / max(x['length'],1) for x in allr):>12.1%} "
              f"{st.mean(x['class_domain_nt'] / max(x['length'],1) for x in allr):>14.1%}")
        zero = sum(1 for x in allr if x["n_domains"] == 0)
        print(f"\nrecords with NO biosynthetic domain found: {zero:,}/{len(allr):,} "
              f"({zero / max(len(allr),1):.1%})")
        print("\nHOW TO READ THIS. The CLASS-domain %% is the share of each core that a "
              "domain-weighted\nloss would up-weight. If it is small, uniform next-base "
              "prediction really is spending\nalmost all of its gradient somewhere other than the "
              "machinery — which is the premise of\nthe whole objective change. If it is large, "
              "that premise is weaker than assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
