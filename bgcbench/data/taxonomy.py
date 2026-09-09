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


#: FIXED PROMPT WIDTH, RIGHT-PADDED WITH SPACES. vortex batches only when prompts share a
#: length, and real lineages run 22-186 characters (median 114) -- 200 generations fragmented
#: into 43 buckets, ~3 hours of wall time against minutes for one batch. One width makes every
#: prompt batchable, and it is applied IDENTICALLY in training and generation so the two
#: formats match.
#:
#: ⚠ PAD, DO NOT TRUNCATE. An earlier version truncated to 82 characters, which was uniform
#: but silently discarded taxonomy: measured on 400 records, 100% were cut, genus and species
#: were lost for ~98%, and the family name was severed mid-word ("f__Gloeobactera"). That
#: conditions the model at roughly ORDER level. 186 is the longest lineage in the table, so
#: padding to it keeps every lineage COMPLETE and loses nothing.
#:
#: The model learns the format either way -- under truncation, 0 of 200 generations continued
#: the taxonomy instead of emitting DNA -- because training and generation agree. Padding
#: simply means there is nothing to learn around.
LINEAGE_WIDTH = 186

#: Right-padding character. A space is a single byte (id 32) in Evo2's byte tokenizer and
#: cannot be confused with a nucleotide, so `clean()` would mask it rather than mistake it
#: for sequence. It never reaches the output: Evo2 returns only the continuation.
LINEAGE_PAD = " "


def canonical(tag: str, width: int = LINEAGE_WIDTH) -> str:
    """One fixed-width lineage, padded with spaces. The lineage itself is never cut.

    Raises if a tag exceeds `width` rather than truncating it: silently losing taxonomy is
    the failure this replaced, and a longer lineage appearing later should be loud.
    """
    if not tag:
        return ""
    if len(tag) > width:
        raise ValueError(
            f"lineage is {len(tag)} characters, above LINEAGE_WIDTH={width}: "
            f"{tag[:60]}... Raise the width rather than truncating -- truncation costs "
            f"genus and species and severs the family name mid-word."
        )
    return tag + LINEAGE_PAD * (width - len(tag))


def attach(records: list[dict], table: dict[str, str] | None = None,
           width: int | None = LINEAGE_WIDTH) -> dict:
    """Set `tax_tag` on every record that has a lineage. Returns a coverage report.

    Records WITHOUT a lineage keep `tax_tag = ""` rather than being dropped: dropping them
    would silently change the training set size, and equal-n is fixed at split time.
    """
    table = load_table() if table is None else table
    n_hit = 0
    for r in records:
        t = table.get(r.get("genome_accession", ""), "")
        r["tax_tag"] = canonical(t, width) if (width and t) else t
        n_hit += bool(t)
    widths = {len(r["tax_tag"]) for r in records if r["tax_tag"]}
    return {"n": len(records), "with_lineage": n_hit,
            "coverage": round(n_hit / max(len(records), 1), 4),
            "width": width, "realised_widths": sorted(widths),
            "source": str(TAX_SOURCE)}
