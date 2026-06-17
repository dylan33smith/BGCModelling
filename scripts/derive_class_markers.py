#!/usr/bin/env python3
"""Data-driven recalibration of per-class characteristic (obligate) Pfam domains.

M2's hand-listed OBLIGATE_DOMAINS are textbook-only and miss real subtype diversity
(e.g. carotenoids under "terpene" use SQS_PSY/lycopene-cyclase, not the classic
terpene cyclases). This scans a sample of splits_core cores per class — using the
SAME find_orfs->HMMER path M2 uses, so derived markers are detectable by M2 — and
reports, per class, the Pfam domains by FREQUENCY (fraction of the class's cores
that contain it) and ENRICHMENT (class freq / background freq). Those are the
candidates for a recalibrated, diversity-aware marker set.

One batched hmmsearch over all sampled ORFs (Pfam-A read once) for speed.
Output: JSON {class: [{pfam, name, class_freq, bg_freq, enrichment, n_cores}, ...]}.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bgc_pipeline.evaluation import find_orfs  # noqa: E402
import pyhmmer  # noqa: E402
from pyhmmer.easel import Alphabet, TextSequence  # noqa: E402
from pyhmmer.plan7 import HMMFile  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_core/train.jsonl"))
    ap.add_argument("--pfam", type=Path, default=Path("/data2/ds85/pfam/Pfam-A.hmm"))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--evalue", type=float, default=1e-10)
    ap.add_argument("--min-aa", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("/data2/ds85/bgcmodel_data/class_markers.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # sample cores per class
    by_class = defaultdict(list)
    for l in args.train.open():
        r = json.loads(l)
        by_class[r["compound_class"]].append(r["sequence"])
    rng = random.Random(args.seed)
    sample = {}
    for c, seqs in by_class.items():
        rng.shuffle(seqs)
        sample[c] = seqs[: args.per_class]
    n_cores = {c: len(v) for c, v in sample.items()}
    print(f"[markers] classes={len(sample)} cores sampled={sum(n_cores.values())}", file=sys.stderr)

    # collect ORFs (name encodes class + core index) — same find_orfs M2 uses
    alphabet = Alphabet.amino()
    digital = []
    name_to_cc = {}      # orf name -> (class, core_idx)
    for c, seqs in sample.items():
        for ci, s in enumerate(seqs):
            for oi, orf in enumerate(find_orfs(s, args.min_aa)):
                nm = f"{c}#{ci}#{oi}"
                name_to_cc[nm] = (c, ci)
                digital.append(TextSequence(name=nm.encode(), sequence=orf.aa_seq).digitize(alphabet))
    print(f"[markers] total ORFs: {len(digital)}; running one Pfam-A hmmsearch ...", file=sys.stderr)

    # per (class) -> pfam -> set of core indices that contain it
    cls_dom_cores = defaultdict(lambda: defaultdict(set))
    pfam_name = {}
    with HMMFile(str(args.pfam)) as hf:
        for th in pyhmmer.hmmsearch(hf, digital, E=args.evalue):
            acc = (th.query.accession.decode() if isinstance(th.query.accession, (bytes, bytearray))
                   else str(th.query.accession or "")).split(".")[0]
            nm = (th.query.name.decode() if isinstance(th.query.name, (bytes, bytearray)) else str(th.query.name or ""))
            if not acc:
                continue
            pfam_name[acc] = nm
            for hit in th:
                if not hit.included:
                    continue
                hn = hit.name.decode() if isinstance(hit.name, (bytes, bytearray)) else str(hit.name)
                cc = name_to_cc.get(hn)
                if cc:
                    cls_dom_cores[cc[0]][acc].add(cc[1])

    # background: fraction of ALL sampled cores containing each pfam
    total_cores = sum(n_cores.values())
    bg = Counter()
    for c, dom in cls_dom_cores.items():
        for acc, cores in dom.items():
            bg[acc] += len(cores)
    bg_freq = {acc: bg[acc] / total_cores for acc in bg}

    out = {}
    for c, dom in cls_dom_cores.items():
        rows = []
        for acc, cores in dom.items():
            cf = len(cores) / n_cores[c]
            rows.append({"pfam": acc, "name": pfam_name.get(acc, ""),
                         "class_freq": round(cf, 3), "bg_freq": round(bg_freq[acc], 3),
                         "enrichment": round(cf / max(bg_freq[acc], 1e-6), 2), "n_cores": len(cores)})
        # rank by class_freq * enrichment (frequent AND specific)
        rows.sort(key=lambda r: -(r["class_freq"] * min(r["enrichment"], 10)))
        out[c] = {"n_cores": n_cores[c], "top_markers": rows[:20]}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[markers] wrote {args.out}", file=sys.stderr)
    # human-readable top-6 per class
    for c in sorted(out, key=lambda k: -n_cores[k]):
        tops = out[c]["top_markers"][:6]
        print(f"\n{c} (n={out[c]['n_cores']}):")
        for r in tops:
            print(f"  {r['pfam']:9} {r['name']:22.22} freq={r['class_freq']:.2f} enr={r['enrichment']:.1f}")


if __name__ == "__main__":
    main()
