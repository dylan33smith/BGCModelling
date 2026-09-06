"""Build the scored record (SPEC 3.9).

A rate is a summary; this is the evidence. Every covariate an analysis could need is
written HERE, at scoring time, because one not written now is unrecoverable later no
matter how the analysis is framed. In particular `seed_core_gene_count` and
`seed_seq_len` are what make the SPEC 4.5.1 confound control possible without a second
training dataset.
"""
from __future__ import annotations

from bgcbench.data.classmap import classify
from bgcbench.score import antismash
from bgcbench.score.novelty import Reference

REQUIRED = (
    "generation_id", "arm", "substrate", "target_class",
    "seed_accession", "seed_core_gene_count", "seed_seq_len",
    "sequence", "seq_len", "hit_eos", "scored_ok", "detected", "products",
    "observed_classes", "on_target", "region_table", "n_cds", "coding_density",
    "produced_core_genes", "containment", "novel",
)


def build(generations: list[dict], verdicts: dict[str, dict],
          mapping: dict[str, str], reference: Reference,
          arm: str, substrate: str) -> list[dict]:
    """`generations` carry generation-side fields; `verdicts` come from the single
    antiSMASH site keyed on generation_id."""
    out = []
    for g in generations:
        gid = g["generation_id"]
        v = verdicts.get(gid)
        if v is None:
            raise RuntimeError(
                f"{gid} has no antiSMASH verdict. Scoring must be total — a missing "
                f"verdict is a build failure, not a non-detection (SPEC 3.1)."
            )
        products = v["products"]
        observed = classify(products, mapping) if products else []
        nov = reference.verdict(g["sequence"])
        rec = {
            "generation_id": gid,
            "arm": arm,
            "substrate": substrate,
            "target_class": g.get("target_class"),
            "seed_accession": g.get("seed_accession"),
            "seed_core_gene_count": g.get("seed_core_gene_count"),
            "seed_seq_len": g.get("seed_seq_len"),
            "sequence": g["sequence"],
            "seq_len": len(g["sequence"]),
            "hit_eos": g.get("hit_eos"),
            "scored_ok": v["scored_ok"],
            "detected": v["detected"],
            "products": products,
            "observed_classes": observed,
            "on_target": g.get("target_class") in observed,
            "region_table": v["region_table"],
            "n_cds": v["n_cds"],
            "coding_density": v["coding_density"],
            "produced_core_genes": v["produced_core_genes"],
            "containment": nov["containment"],
            "novel": nov["novel"],
            "gate": nov["gate"],
            "scoring_config": antismash.config_hash(),
        }
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            raise RuntimeError(f"scored record missing required fields: {missing}")
        out.append(rec)
    return out
