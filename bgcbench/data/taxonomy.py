"""GTDB lineage per genome — Evo2's native conditioning format.

⚠ WHAT THIS IS AND IS NOT. Evo2 was pretrained with a lowercase GTDB lineage prefixed to the
sequence, so the lineage is the model's OWN input format, not a channel this project invents.
It names an ORGANISM, never a compound class. That distinction is the whole point: a
`|COMPOUND_CLASS:TERPENE|` token hands the model the answer, a lineage does not.

Measured on the prior codebase's TERPENE adapter, same sampling and scorer throughout:

    |COMPOUND_CLASS:TERPENE| + lineage   GC 0.607   25/42 on-target
    lineage only                         GC 0.654    0/42
    bare "A"                             GC 0.453    0/42

So the lineage alone moves composition ONTO the real-core value (~0.64) while producing no
detections. That result cannot separate "the class token carries information the lineage does
not" from "the adapter needs the format it was trained on" — it was trained on class+lineage,
so lineage-only is a format mismatch for it. A model TRAINED on lineage-only is untested, and
is what the pilot builds.
"""
from __future__ import annotations

import json
from pathlib import Path

#: The prior project's record table, which carries a GTDB lineage per genome. Read-only.
TAX_SOURCE = Path("/data2/ds85/bgcmodel_data/asdb5_core_records.jsonl")


def load_table(path: Path = TAX_SOURCE) -> dict[str, str]:
    """genome_accession -> GTDB lineage, e.g. `|d__Bacteria;p__Bacteroidota;...|`."""
    tax: dict[str, str] = {}
    if not path.exists():
        return tax
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            t = r.get("taxonomic_tag")
            if t:
                tax.setdefault(r["genome_accession"], t)
    return tax


def attach(records: list[dict], table: dict[str, str] | None = None) -> dict:
    """Set `tax_tag` on every record that has a lineage. Returns a coverage report.

    Records WITHOUT a lineage keep `tax_tag = ""` rather than being dropped: dropping them
    would silently change the training set size, and equal-n is fixed at split time.
    """
    table = load_table() if table is None else table
    n_hit = 0
    for r in records:
        t = table.get(r.get("genome_accession", ""), "")
        r["tax_tag"] = t
        n_hit += bool(t)
    return {"n": len(records), "with_lineage": n_hit,
            "coverage": round(n_hit / max(len(records), 1), 4),
            "source": str(TAX_SOURCE)}
