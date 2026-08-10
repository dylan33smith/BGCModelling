#!/usr/bin/env python
"""Emit a NEGATIVE control set: real non-BGC bacterial DNA, length- and taxon-matched.

WHY THIS EXISTS. The suite has never measured the FALSE-POSITIVE rate of any gate. Every
`is_bgc` number is a true-positive rate with an unstated false-positive rate, so no gate's
SPECIFICITY is known -- it is asserted. Concretely: `is_bgc = 0.12` means nothing until you know
whether ordinary genomic DNA of the same length also scores 0.12.

WHY NOT SHUFFLED SEQUENCE. scripts/evaluate_bgc.py already offers --include-negative-control,
which SHUFFLES the nucleotides. That is a very weak null: shuffling destroys codon structure, so
the gene caller finds nothing and every gate fails trivially. It proves the suite can reject
noise, which was never in doubt. The interesting question is whether the suite can reject REAL
DNA THAT IS NOT A BGC -- coding, gene-dense, correct GC, same organism -- which is exactly what
the model produces when it writes plausible-but-not-biosynthetic sequence. Measured: the
class_markers is_bgc proxy has specificity 0.598 against antiSMASH; against real non-BGC DNA it
is unknown and could be far worse, because housekeeping genes are what the generic Pfam families
(AMP-binding, DAO, adh_short) actually match.

SOURCE. Non-BGC windows are cut from the SAME genomes the BGC cores come from, OUTSIDE every
annotated BGC region, and truncated to the generations' own length distribution. That controls
organism, GC, codon usage and length simultaneously -- everything except "is it a BGC".
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _cut_nonbgc_from_genomes(tar_path: Path, max_genomes: int, min_len: int, rng) -> list[dict]:
    """Windows from whole genomes that lie OUTSIDE every annotated antiSMASH region.

    Everything is held constant except BGC-ness: same organisms, same GC, same codon usage,
    real coding DNA. A gate that fires here is producing a FALSE POSITIVE on exactly the kind of
    sequence a model produces when it writes plausible-but-not-biosynthetic output.
    """
    import gzip
    import tarfile
    from Bio import SeqIO

    out: list[dict] = []
    # STREAM the archive: this tar is truncated (`tarfile.ReadError: unexpected end of data`
    # partway through), so getmembers() -- which walks to the end -- raises before returning
    # anything. Iterating yields every member up to the damage, which is plenty.
    members = []
    with tarfile.open(tar_path) as tf:
        try:
            for m in tf:
                if m.name.endswith(".gbk.gz"):
                    members.append(m)
                if len(members) >= max_genomes:
                    break
        except tarfile.ReadError as e:
            print(f"[negctrl] archive truncated after {len(members)} entries ({e}); "
                  f"using what is readable", flush=True)
    with tarfile.open(tar_path) as tf:
        for m in members:
            fh = tf.extractfile(m)
            if fh is None:
                continue
            try:
                with gzip.open(fh, "rt") as g:
                    for rec in SeqIO.parse(g, "genbank"):
                        seq = str(rec.seq)
                        if len(seq) < 3 * min_len:
                            continue
                        # every antiSMASH region on this contig -> forbidden intervals
                        bad = [(int(f.location.start), int(f.location.end))
                               for f in rec.features if f.type == "region"]
                        for _ in range(6):          # a few windows per genome
                            st_ = rng.randrange(0, max(len(seq) - min_len, 1))
                            en = st_ + min_len
                            if any(st_ < b and en > a for a, b in bad):
                                continue           # overlaps a BGC -> not a negative
                            w = seq[st_:en]
                            if w.count("N") > 0.01 * len(w):
                                continue
                            out.append({"sequence": w,
                                        "accession": f"{rec.id}:{st_}-{en}",
                                        "taxonomic_tag": "",
                                        "source_genome": m.name})
                        break                      # first (largest) contig only
            except Exception:
                continue
    print(f"[negctrl] cut {len(out)} non-BGC windows from {len(members)} genomes "
          f"(outside every antiSMASH region)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True,
                    help="Generations this control accompanies (sets length + class mix).")
    ap.add_argument("--source", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"),
                    help="JSONL of explicitly non-BGC records (see --gbk-tar to build one).")
    ap.add_argument("--gbk-tar", type=Path, default=None,
                    help="Whole-genome GBKs. Cuts windows OUTSIDE every antiSMASH region: real "
                         "coding, gene-dense, correct-GC DNA from the same organisms, differing "
                         "from a BGC in exactly one respect. This is the control that measures "
                         "specificity; shuffled sequence does not.")
    ap.add_argument("--max-genomes", type=int, default=34)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = match the generation count.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gens = [json.loads(l) for l in args.gen.open() if l.strip()]
    if not gens:
        raise SystemExit(f"{args.gen} is empty")
    # Length AND class travel together, per generation (see make_positive_control.py: sorting
    # lengths separately from classes de-pairs them and calibrates a different question).
    spec = [(len(g.get("sequence", "") or ""), g.get("compound_class")) for g in gens]
    want = args.n or len(gens)

    # Records carry the BGC core plus its genomic coordinates. We cannot cut true intergenic DNA
    # without the full genome, so use the most conservative available proxy: the FLANK of a core
    # from a DIFFERENT class than the one being scored is still BGC DNA and would understate the
    # false-positive rate. Instead we require an explicit non-BGC source and fail loudly if the
    # records do not carry one, rather than silently substituting something weaker.
    nonbgc: list[dict] = []
    if args.gbk_tar:
        nonbgc = _cut_nonbgc_from_genomes(args.gbk_tar, args.max_genomes,
                                          max(L for L, _ in spec), rng=random.Random(args.seed))
    else:
        src = [json.loads(l) for l in args.source.open() if l.strip()]
        nonbgc = [r for r in src if r.get("non_bgc") or r.get("is_negative")]
    if not nonbgc:
        raise SystemExit(
            f"{args.source} carries no explicitly non-BGC records (no `non_bgc`/`is_negative` "
            f"field).\n"
            f"A negative control MUST be real non-BGC DNA. Two ways to get it:\n"
            f"  1. cut windows outside annotated regions from the source genomes "
            f"(scripts/build_core_records.py has the coordinates), or\n"
            f"  2. download a few complete genomes and sample intergenic/housekeeping windows.\n"
            f"REFUSING to substitute shuffled sequence: shuffling destroys codon structure, so "
            f"every gate fails trivially and the measured specificity is meaninglessly high.")

    rng = random.Random(args.seed)
    out, used = [], set()
    for i in range(want):
        target_len, cls = spec[i % len(spec)]
        usable = [r for r in nonbgc if len(r.get("sequence", "")) >= target_len] or nonbgc
        fresh = [r for r in usable if id(r) not in used] or usable
        if not fresh:
            continue
        r = rng.choice(fresh)
        used.add(id(r))
        out.append({
            "sequence": r["sequence"][:target_len] if target_len else r["sequence"],
            # The class we ASK for is the generation's class: the question is whether non-BGC DNA
            # gets called a BGC of the class we requested.
            "compound_class": cls,
            "taxonomic_tag": r.get("taxonomic_tag", ""),
            "accession": f"NEGCTRL_{r.get('accession', i)}",
            "negative_control": True,
            "truncated_to": target_len,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    ol = [len(r["sequence"]) for r in out]
    print(f"[negctrl] {len(out)} non-BGC windows -> {args.out}")
    print(f"[negctrl]   lengths: median {st.median(ol):.0f} [{min(ol)}..{max(ol)}]")
    print(f"[negctrl]   any gate that PASSES on these is a FALSE POSITIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
