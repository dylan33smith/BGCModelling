"""MiBIG as the external-validation partition (SPEC 4.6).

MiBIG entries are experimentally characterised clusters. They are the partial answer to
the circularity disclosure in SPEC 3.1 -- our corpus is antiSMASH-defined and our endpoint
is antiSMASH, so an externally curated set is the only outside check available.

Two rules:
  * MiBIG records are EXCLUDED from every training corpus.
  * The partition is declared in the manifest AT BUILD TIME, not bolted on later, and is
    read ONCE at the end. Iterating against it would destroy the one external check.

Matching is by SEQUENCE, not accession. MiBIG references nucleotide accessions
(AM492533.1) while the corpus is keyed by assembly accession (GCF_000015805.1); an
accession join would silently miss most of the overlap. A MiBIG entry is a whole cluster
with context and our records are strict cores, so the core is the query and coverage is
measured over it.
"""
from __future__ import annotations

import json
from pathlib import Path

from bgcbench.data import cluster as clu
from bgcbench.data.genbank import parse

MIBIG_GBK = Path("/data2/ds85/bcgm_data/mibig/mibig_gbk_4.0")


def load_mibig(gbk_dir: Path = MIBIG_GBK) -> list[dict]:
    out: list[dict] = []
    for p in sorted(gbk_dir.glob("*.gbk")):
        try:
            for rec in parse(p.read_text(), want=frozenset()):
                if rec.sequence:
                    out.append({"accession": p.stem, "sequence": rec.sequence})
                    break
        except Exception:
            continue
    if not out:
        raise RuntimeError(f"no MiBIG sequences read from {gbk_dir} — refusing to "
                           f"declare an empty external-validation partition")
    return out


def partition(records: list[dict], workdir: Path | None = None,
              threads: int = 16) -> set[str]:
    """Accessions in `records` that match a MiBIG cluster. These form the MIBIG
    partition and are excluded from train/val/test."""
    ref = load_mibig()
    return clu.neardup_query_ids(records, ref, workdir=workdir, threads=threads)
