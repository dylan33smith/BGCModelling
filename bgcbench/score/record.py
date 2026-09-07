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
    "generation_id", "stage", "applicable", "arm", "substrate", "target_class",
    "seed_accession", "seed_core_gene_count", "seed_seq_len",
    "sequence", "seq_len", "hit_eos", "scored_ok", "detected", "products",
    "observed_classes", "on_target", "region_table", "n_cds", "coding_density",
    "produced_core_genes", "containment", "novel",
)


def not_applicable(arm: str, substrate: str, target_class: str, reason: str,
                   stage: str = "stage1") -> dict:
    """SPEC 6.5: an arm that cannot be implemented on a substrate is NOT APPLICABLE, never
    a zero. A zero is a measurement and reads as the best possible specificity result; a
    structural absence and a measured null are different results and must not be
    byte-identical."""
    return {"generation_id": f"NA::{substrate}::{arm}::{target_class}",
            "stage": stage, "applicable": False, "na_reason": reason,
            "arm": arm, "substrate": substrate, "target_class": target_class,
            "seed_accession": None, "seed_core_gene_count": None, "seed_seq_len": None,
            "sequence": "", "seq_len": 0, "hit_eos": None, "scored_ok": False,
            "detected": None, "products": [], "observed_classes": [], "on_target": None,
            "region_table": [], "n_cds": None, "coding_density": None,
            "produced_core_genes": None, "containment": None, "novel": None,
            "gate": "NOT_APPLICABLE"}


def build(generations: list[dict], verdicts: dict[str, dict],
          mapping: dict[str, str], reference: Reference,
          arm: str, substrate: str, stage: str = "stage1",
          corpus_verdicts: dict[str, dict] | None = None) -> list[dict]:
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
        # An empty generation is a real outcome (the model terminated immediately), not a
        # missing measurement. Containment is undefined on it, so it is recorded as such
        # rather than raising or being silently dropped from the denominator.
        nov = (reference.verdict(g["sequence"]) if g["sequence"]
               else {"containment": None, "containment_reverse": None,
                     "containment_worst": None, "novel": None, "gate": "EMPTY"})
        cv = (corpus_verdicts or {}).get(gid)
        rec = {
            "generation_id": gid,
            # SPEC 6.1 forbids pooling shakedown and powered runs. A field, not a policy:
            # prose cannot be audited after the fact.
            "stage": stage,
            "applicable": True,
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
            "containment_reverse": nov["containment_reverse"],
            "containment_worst": nov["containment_worst"],
            "novel": nov["novel"],
            "gate": nov["gate"],
            # SPEC 3.8 corpus-level reference: answers "did it output a KNOWN BGC?", which
            # the per-arm reference cannot ask of an untrained arm.
            "matched_known_bgc": (cv or {}).get("matched_known_bgc"),
            "corpus_best_identity": (cv or {}).get("best_identity"),
            "scoring_config": antismash.config_hash(),
        }
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            raise RuntimeError(f"scored record missing required fields: {missing}")
        out.append(rec)
    return out
