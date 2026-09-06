"""Raw GenBank -> core records (SPEC 4.1, 4.2).

WHAT A "CORE" IS, fixed here. SNAPPED TO WHOLE GENES. An antiSMASH `region` is the cluster plus a per-product
`/neighbourhood` of 5-20 kb on EACH side, so a region is far larger than the biosynthetic
core -- one observed RiPP region spans 11,008 nt around a 1,008 nt core. The core is the
`proto_core` feature. Where a region contains several protoclusters the core is the
CONTIGUOUS span from the first proto_core start to the last proto_core end: splicing the
protocores together would emit a chimeric sequence that never existed.

The span is then EXTENDED OUTWARD to whole-gene boundaries. `proto_core` bounds are
cluster bounds and do not align to gene bounds: 16.4% of CDS overlapping a core straddle
its edge. Extracting on the raw bound truncates those genes, they lose their stop codon,
and prodigal will not call them -- measured on 121 RIPP held-out cores, 62 lost CDS calls
and every core gene that went missing went missing because the gene was never called at
all. Snapping costs a median of 0 nt and a p90 of ~1.1 kb.

Extension uses each CDS's INDIVIDUAL spans, never its bounding box, so an origin-spanning
gene on a circular replicon cannot drag the core across the whole replicon.

This is still `--flank 0`: no neighbourhood, no regulatory context, only the genes the core
already overlapped, made whole. Every claim says "biosynthetic core", never "cluster ready
to express" (SPEC 2.2).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from bgcbench.data import tar_index
from bgcbench.data.classmap import build_map, classify
from bgcbench.data.genbank import parse


@dataclass
class CoreRecord:
    accession: str
    genome_accession: str
    region_number: int
    locus: str
    classes: list[str]           # SPEC 3.3: a record counts for EVERY class it maps to
    antismash_products: list[str]
    sequence: str
    seq_len: int                 # == len(sequence). SPEC 4.2: there is no region-span field.
    core_gene_count: int          # CDS with /gene_kind="biosynthetic" in the core span
    cds_count: int                # ALL CDS in the span; diagnostic, not the multi-gene axis
    snap_nt: int                  # nt added by whole-gene snapping (0 = boundaries aligned)
    snap_capped: bool             # True if SNAP_CAP bound the extension
    n_protoclusters: int
    contig_edge: bool
    region_span: int             # diagnostic only, never a length statistic (KNOWN_WRONG #1)


#: A single gene should never extend a core by more than this on one side. Guards against a
#: pathological annotation dragging the core outward without bound.
SNAP_CAP = 5000


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def extract_genome(acc: str, index: dict, mapping: dict[str, str]) -> list[dict]:
    text = tar_index.read_member(acc, index)
    out: list[dict] = []
    for rec in parse(text):
        cores = rec.of("proto_core")
        cdss = rec.of("CDS")
        for reg in rec.of("region"):
            inner = [c for c in cores if c.overlaps(reg.start, reg.end)]
            if not inner:
                continue                       # a region with no protocore has no core
            lo = min(c.start for c in inner)
            hi = max(c.end for c in inner)
            raw_lo, raw_hi = lo, hi
            # snap outward to whole genes, per-span so a join() cannot run away
            for c in cdss:
                if not c.overlaps(lo, hi):
                    continue
                for a, b in (c.spans or [(c.start, c.end)]):
                    if a < hi and lo < b:
                        lo = min(lo, a)
                        hi = max(hi, b)
            capped = False
            if raw_lo - lo > SNAP_CAP:
                lo, capped = raw_lo - SNAP_CAP, True
            if hi - raw_hi > SNAP_CAP:
                hi, capped = raw_hi + SNAP_CAP, True
            lo, hi = max(0, lo), min(len(rec.sequence), hi)
            seq = rec.sequence[lo:hi]
            if not seq or set(seq) <= {"N"}:
                continue
            prods = reg.quals.get("product", [])
            classes = classify(prods, mapping)
            in_core = [c for c in cdss if c.overlaps(lo, hi)]
            # SPEC 4.2: core_gene_count is "core biosynthetic genes", NOT every CDS in
            # the span. antiSMASH marks the genes that satisfied the detection rule with
            # /gene_kind="biosynthetic"; counting all CDS instead inflates the multi-gene
            # fraction badly (TERPENE read 53% rather than 18% before this was fixed).
            n_genes = sum(1 for c in in_core
                          if c.q1("gene_kind") == "biosynthetic")
            rn = reg.q1("region_number") or "0"
            # THE LOCUS IS PART OF THE KEY. antiSMASH numbers regions PER RECORD, so
            # region_number restarts at 1 on every contig: a draft genome with 84 contigs
            # otherwise yields 84 records all keyed "<genome>.region1". Measured before
            # this fix: 307,069 of 540,695 records (56.8%) shared an accession, and 560
            # accessions appeared in more than one split file. Every downstream structure
            # -- cluster assignment, record_split, component lookup, mmseqs FASTA headers
            # -- is keyed on this string, so collisions silently overwrite.
            out.append(asdict(CoreRecord(
                accession=f"{acc}.{rec.locus}.region{rn}",
                genome_accession=acc,
                region_number=int(rn),
                locus=rec.locus,
                classes=classes,
                antismash_products=prods,
                sequence=seq,
                seq_len=len(seq),
                core_gene_count=n_genes,
                cds_count=len(in_core),
                snap_nt=(raw_lo - lo) + (hi - raw_hi),
                snap_capped=capped,
                n_protoclusters=len(inner),
                contig_edge=(reg.q1("contig_edge", "False") == "True"),
                region_span=reg.end - reg.start,
            )))
    return out


def _worker(args):
    acc, index_path, mapping = args
    try:
        idx = _worker.cache
    except AttributeError:
        idx = _worker.cache = json.loads(Path(index_path).read_text())
    try:
        return extract_genome(acc, idx, mapping)
    except Exception as e:                      # a bad member must not kill the pass
        return [{"__error__": f"{acc}: {type(e).__name__}: {e}"}]


def run(accessions: list[str], out_path: Path, workers: int = 12) -> dict:
    import multiprocessing as mp

    index_path = str(tar_index.TAR.with_suffix(".index.json"))
    mapping = build_map()["mapping"]
    n_rec = n_err = 0
    errors: list[str] = []
    with open(out_path, "w") as fh, mp.Pool(workers) as pool:
        for batch in pool.imap_unordered(
            _worker, ((a, index_path, mapping) for a in accessions), chunksize=8
        ):
            for r in batch:
                if "__error__" in r:
                    n_err += 1
                    if len(errors) < 50:
                        errors.append(r["__error__"])
                    continue
                fh.write(json.dumps(r) + "\n")
                n_rec += 1
    return {"records": n_rec, "genomes": len(accessions),
            "errors": n_err, "error_sample": errors}


if __name__ == "__main__":
    idx = tar_index.load_index()
    accs = sorted(idx)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if limit:
        accs = accs[:limit]
    stats = run(accs, Path(sys.argv[1]), workers=int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    print(json.dumps(stats, indent=2))
