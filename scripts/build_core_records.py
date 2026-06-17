#!/usr/bin/env python3
"""Build CORE-trimmed BGC records from the antiSMASH-DB GBK tar.

One streaming pass over asdb5_gbks.tar. For every antiSMASH region we:
  * compute TWO core spans from per-CDS ``gene_kind`` (see docs/archive/REDESIGN_PLAN.md Step 3):
      - strict : contiguous span of gene_kind == "biosynthetic" genes
      - wide   : contiguous span of {"biosynthetic","biosynthetic-additional"}
  * re-extract the core nucleotides DIRECTLY from the GBK contig (not the stored
    record sequence — the original ingestion center-truncates oversized regions),
  * emit a native lowercase GTDB taxonomy tag (Step-1 fix: `d__Bacteria;...;
    s__Escherichia coli`, not the old `D__BACTERIA;...;S__ESCHERICHIA_COLI`).

Both core sequences are stored so the strict-vs-wide choice can be made later from
per-class length stats WITHOUT re-touching the 172 GiB tar. Fallback: a region with
no qualifying CDS keeps the full region for that variant (flagged). Cores longer
than --max-len are center-truncated (rare).

Output: one JSONL of core records + a stats JSON (per-class length distributions,
%-single-window). This is the FULL core pool; curate / split / dedup come after.

Usage:
  python scripts/build_core_records.py \
      --tar /data2/ds85/asdb5_gbks/asdb5_gbks.tar \
      --taxa data/antismash_db/asdb5_taxa.json.gz \
      --out /data2/ds85/bgcmodel_data/asdb5_core_records.jsonl \
      --out-stats /data2/ds85/bgcmodel_data/asdb5_core_stats.json
  # smoke first:  --limit 40 --out /tmp/core_smoke.jsonl --out-stats /tmp/core_smoke_stats.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO  # noqa: F401  (ensures Bio available; used via helpers)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from antismash_db_to_jsonl import (  # noqa: E402
    _parse_gbk_bytes, map_region_products, load_taxa_json, load_class_map,
    _TAXA_JSON_RANK_ORDER,
)

STRICT_KINDS = {"biosynthetic"}
WIDE_KINDS = {"biosynthetic", "biosynthetic-additional"}
WINDOW = 32768  # single-window threshold for stats


def build_native_tag(entry: dict) -> str | None:
    """Native lowercase GTDB-style tag: |d__Bacteria;p__...;s__Escherichia coli|.
    Lowercase rank prefix, taxon name kept verbatim (real case + spaces), only
    delimiter-colliding chars sanitised."""
    segs = []
    for field, prefix in _TAXA_JSON_RANK_ORDER:
        name = (entry.get(field) or "").strip()
        if not name or name.lower() in ("unknown", "unclassified", ""):
            continue
        name = name.replace("|", "").replace(";", "").strip()
        if name:
            segs.append(f"{prefix.lower()}__{name}")
    return f"|{';'.join(segs)}|" if segs else None


def native_tag_for_gbk(gbk_text, taxa_map, deprecated_ids) -> str | None:
    m = re.search(r'/db_xref="taxon:(\d+)"', gbk_text)
    if not m:
        return None
    taxid = int(m.group(1))
    taxid = deprecated_ids.get(taxid, taxid)
    entry = taxa_map.get(taxid)
    return build_native_tag(entry) if entry else None


def _core_span(cds_coords, kinds, rs, re_):
    """min-start, max-end, n_genes over CDS whose gene_kind ∈ kinds and that lie
    within [rs, re_]. Returns None if no qualifying CDS."""
    sel = [(s, e) for (s, e, gk) in cds_coords if gk in kinds and s >= rs and e <= re_]
    if not sel:
        return None
    return min(s for s, _ in sel), max(e for _, e in sel), len(sel)


def _materialize(full_seq, span, rs, re_, flank, max_len):
    """Return (seq, start, end, n_genes, fallback). If span is None → full region."""
    if span is None:
        s, e, n, fb = rs, re_, 0, True
    else:
        cs, ce, n = span
        s, e, fb = max(rs, cs - flank), min(re_, ce + flank), False
    seq = full_seq[s:e]
    if len(seq) > max_len:  # center-truncate giant cores to the context window
        mid = (s + e) // 2
        half = max_len // 2
        ts = max(0, mid - half)
        seq = full_seq[ts:ts + max_len]
        s, e = ts, ts + len(seq)
    return seq, s, e, n, fb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", type=Path, required=True)
    ap.add_argument("--taxa", type=Path, default=ROOT / "data/antismash_db/asdb5_taxa.json.gz")
    ap.add_argument("--class-map", type=Path, default=ROOT / "config/compound_class_map.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-stats", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=262144, help="context window cap")
    ap.add_argument("--flank", type=int, default=0, help="bp added each side of the core")
    ap.add_argument("--min-region-len", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="stop after N genomes (smoke)")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    class_mapping, class_default = load_class_map(args.class_map)
    res = load_taxa_json(args.taxa)
    taxa_map, deprecated = res if isinstance(res, tuple) else (res, {})
    print(f"[core] class-map entries={len(class_mapping)} default={class_default} | "
          f"taxa entries={len(taxa_map)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: {"n": 0, "region_len": [], "strict_len": [], "wide_len": [],
                                 "strict_fallback": 0, "wide_fallback": 0})
    n_genomes = n_regions = 0
    t0 = time.time()

    with args.out.open("w") as out, tarfile.open(args.tar, "r:*") as tf:
        try:
            for member in tf:
                name = member.name
                if not (name.endswith(".gbk.gz") or name.endswith(".gbk")):
                    continue
                if args.limit is not None and n_genomes >= args.limit:
                    break
                stem = Path(name).name
                genome_acc = re.sub(r"\.gbk(\.gz)?$", "", stem)
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                raw = fobj.read()
                records = _parse_gbk_bytes(raw, name)
                if not records:
                    continue
                n_genomes += 1
                try:
                    gbk_text = (gzip.decompress(raw) if name.endswith(".gz") else raw).decode(
                        "ascii", errors="replace")
                except Exception:
                    gbk_text = ""
                tax_tag = native_tag_for_gbk(gbk_text, taxa_map, deprecated) or "||"

                for rec in records:
                    full_seq = str(rec.seq).upper()
                    cds_coords = [(int(f.location.start), int(f.location.end),
                                   f.qualifiers.get("gene_kind", [""])[0])
                                  for f in rec.features if f.type == "CDS"]
                    for f in rec.features:
                        if f.type != "region" or f.qualifiers.get("tool") != ["antismash"]:
                            continue
                        rs, re_ = int(f.location.start), int(f.location.end)
                        if re_ - rs < args.min_region_len:
                            continue
                        products = f.qualifiers.get("product", [])
                        cls = map_region_products(products, class_mapping, class_default)
                        region_number = f.qualifiers.get("region_number", ["?"])[0]
                        contig_edge = (f.qualifiers.get("contig_edge", ["False"])[0]
                                       .strip().lower() == "true")
                        s_span = _core_span(cds_coords, STRICT_KINDS, rs, re_)
                        w_span = _core_span(cds_coords, WIDE_KINDS, rs, re_)
                        s_seq, s0, s1, s_n, s_fb = _materialize(full_seq, s_span, rs, re_, args.flank, args.max_len)
                        w_seq, w0, w1, w_n, w_fb = _materialize(full_seq, w_span, rs, re_, args.flank, args.max_len)

                        out.write(json.dumps({
                            "accession": f"{genome_acc}.region{region_number}",
                            "genome_accession": genome_acc,
                            "region_number": int(region_number) if str(region_number).isdigit() else region_number,
                            "compound_class": cls,
                            "antismash_products": products,
                            "contig_edge": contig_edge,
                            "taxonomic_tag": tax_tag,
                            "region_start": rs, "region_end": re_, "region_len": re_ - rs,
                            "strict_core_start": s0, "strict_core_end": s1,
                            "strict_core_len": len(s_seq), "strict_core_genes": s_n,
                            "strict_fallback": s_fb, "strict_sequence": s_seq,
                            "wide_core_start": w0, "wide_core_end": w1,
                            "wide_core_len": len(w_seq), "wide_core_genes": w_n,
                            "wide_fallback": w_fb, "wide_sequence": w_seq,
                        }) + "\n")
                        n_regions += 1
                        st = stats[cls]
                        st["n"] += 1
                        st["region_len"].append(re_ - rs)
                        st["strict_len"].append(len(s_seq))
                        st["wide_len"].append(len(w_seq))
                        st["strict_fallback"] += int(s_fb)
                        st["wide_fallback"] += int(w_fb)

                if n_genomes % args.progress_every == 0:
                    print(f"[core] genomes={n_genomes:,} regions={n_regions:,} "
                          f"({(time.time()-t0)/60:.1f} min)", file=sys.stderr, flush=True)
        except (EOFError, tarfile.TarError) as exc:
            print(f"[core] tar ended early after {n_genomes:,} genomes "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)

    # aggregate stats
    def pct_sw(lens):
        return round(100 * sum(1 for x in lens if x <= WINDOW) / len(lens), 1) if lens else None
    def med(xs):
        return int(statistics.median(xs)) if xs else None
    summary = {"total_genomes": n_genomes, "total_regions": n_regions,
               "elapsed_min": round((time.time() - t0) / 60, 1), "window": WINDOW,
               "by_class": {}}
    for cls, st in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        summary["by_class"][cls] = {
            "n": st["n"],
            "region_median": med(st["region_len"]),
            "strict_median": med(st["strict_len"]), "strict_pct_single_window": pct_sw(st["strict_len"]),
            "strict_fallback": st["strict_fallback"],
            "wide_median": med(st["wide_len"]), "wide_pct_single_window": pct_sw(st["wide_len"]),
            "wide_fallback": st["wide_fallback"],
        }
    args.out_stats.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[core] DONE: {n_regions:,} regions from {n_genomes:,} genomes → {args.out}")
    print(f"[core] stats → {args.out_stats}")
    for cls, s in list(summary["by_class"].items())[:12]:
        print(f"  {cls:20} n={s['n']:6}  region~{s['region_median']}  "
              f"strict~{s['strict_median']} ({s['strict_pct_single_window']}% SW)  "
              f"wide~{s['wide_median']} ({s['wide_pct_single_window']}% SW)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
