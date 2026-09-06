"""Cluster-aware, genome-aware, GLOBAL split (SPEC 4.4.3, 4.6, 4.7).

Three properties, all enforced by construction rather than by filtering afterwards:

1. GENOME-DISJOINT   a genome appears in exactly one split.
2. CLUSTER-DISJOINT  a sequence cluster appears in exactly one split, so cross-split
                     near-duplication is ~0 before any filtering runs.
3. GLOBALLY CONSISTENT  a record has ONE split assignment across every class corpus.

Property 3 is not optional and is easy to get wrong. Under the SPEC 3.3 hybrid rule a
record counts for every class its products map to, so the same record can sit in the
TERPENE corpus and the NRPS corpus. Splitting each class independently would then place
it in TERPENE-train and NRPS-test -- and the pooled `W1` arm, which trains on the union,
would be training on its own test set. So components are formed and assigned ONCE, over
the union of all benchmark classes, and every class corpus inherits that assignment.

Grouping unit: the connected component of the bipartite graph linking records that share
a genome OR a cluster. Assigning whole components is what makes 1 and 2 hold together;
assigning by genome alone leaves cluster leakage, and by cluster alone leaves genome
leakage.

Assignment is by stable hash of the component key -- no RNG, so a split is reproducible
from the corpus alone with no seed to lose.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from bgcbench.data import cluster as clu
from bgcbench.data import manifest as mf

FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


class _Union:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _frac(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2**64


def _assign(key: str) -> str:
    f = _frac(key)
    if f < FRACS["train"]:
        return "train"
    return "val" if f < FRACS["train"] + FRACS["val"] else "test"


def load_corpus(path: Path, classes: tuple[str, ...], max_len: int) -> list[dict]:
    """SPEC 4.7: over-length records are DROPPED, never truncated -- truncation would
    manufacture a record with a severed terminal gene that antiSMASH would never call."""
    want = set(classes)
    out = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["seq_len"] > max_len:
                continue
            if want & set(r["classes"]):
                out.append(r)
    return out


def build(corpus_path: Path, out_dir: Path, classes: tuple[str, ...],
          max_len: int, common_n: int, workdir: Path | None = None,
          threads: int = 16, exclude: set[str] | None = None) -> dict:
    records = load_corpus(corpus_path, classes, max_len)
    if exclude:
        records = [r for r in records if r["accession"] not in exclude]
    by_acc = {r["accession"]: r for r in records}

    # --- one global clustering over the union of all benchmark classes ---------------
    rep = clu.cluster(records, workdir=workdir, threads=threads)

    # --- components: linked by shared genome OR shared cluster -----------------------
    uf = _Union()
    for r in records:
        a = ("acc", r["accession"])
        uf.union(a, ("gen", r["genome_accession"]))
        uf.union(a, ("clu", rep[r["accession"]]))
    comp_of = {r["accession"]: uf.find(("acc", r["accession"])) for r in records}

    # stable component key: the smallest accession it contains, so the key does not
    # depend on iteration order or on union-find internals
    members = defaultdict(list)
    for acc, c in comp_of.items():
        members[c].append(acc)
    comp_key = {c: min(a) for c, a in members.items()}
    split_of_comp = {c: _assign(comp_key[c]) for c in members}

    # --- per class: take common_n clusters, honouring the global assignment ----------
    report: dict[str, dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        cls_recs = [r for r in records if cls in r["classes"]]
        clusters = defaultdict(list)
        for r in cls_recs:
            clusters[rep[r["accession"]]].append(r)
        # order clusters deterministically, then take common_n
        ordered = sorted(clusters, key=lambda k: (_frac(k), k))
        chosen = ordered[:common_n]
        if len(chosen) < common_n:
            raise RuntimeError(
                f"{cls}: only {len(chosen)} clusters at max_len={max_len}, "
                f"below common_n={common_n}. SPEC 4.4.2 must be revisited before "
                f"building — silently shrinking one class breaks equal-n."
            )
        buckets: dict[str, list[dict]] = {k: [] for k in FRACS}
        for ck in chosen:
            # one representative record per cluster (SPEC 4.4.3)
            recs = sorted(clusters[ck], key=lambda r: r["accession"])
            pick = next((r for r in recs if r["accession"] == ck), recs[0])
            buckets[split_of_comp[comp_of[pick["accession"]]]].append(pick)

        d = out_dir / cls
        d.mkdir(parents=True, exist_ok=True)
        for name, rows in buckets.items():
            with open(d / f"{name}.jsonl", "w") as fh:
                for r in sorted(rows, key=lambda x: x["accession"]):
                    fh.write(json.dumps(r) + "\n")

        picked = [x for b in buckets.values() for x in b]
        lens = sorted(x["seq_len"] for x in picked)
        report[cls] = {
            "clusters_available": len(clusters),
            "clusters_used": len(chosen),
            "n": {k: len(v) for k, v in buckets.items()},
            "genomes": {k: len({r["genome_accession"] for r in v})
                        for k, v in buckets.items()},
            "median_seq_len": lens[len(lens) // 2] if lens else 0,
            "multigene_frac": round(
                sum(1 for x in picked if x["core_gene_count"] >= 2) / max(len(picked), 1),
                4),
            "hybrid_frac": round(
                sum(1 for x in picked if len(x["antismash_products"]) > 1)
                / max(len(picked), 1), 4),
            "n_products": len({p for x in picked for p in x["antismash_products"]}),
        }
    return {"classes": report, "max_len": max_len, "common_n": common_n,
            "n_records_considered": len(records),
            "clustering": {"min_seq_id": clu.MIN_SEQ_ID, "coverage": clu.COVERAGE,
                           "cov_mode": clu.COV_MODE}}


def verify(out_dir: Path, classes: tuple[str, ...], workdir: Path | None = None,
           threads: int = 16) -> dict:
    """SPEC 4.6 integrity assertions. Every failure raises; none is filtered away."""
    res: dict[str, dict] = {}
    for cls in classes:
        d = out_dir / cls
        parts = {k: [json.loads(l) for l in open(d / f"{k}.jsonl")] for k in FRACS}
        for k, rows in parts.items():
            if not rows:
                raise RuntimeError(f"{cls}/{k} is EMPTY — an observed prior corpus had a "
                                   f"class with 0 val and 0 test and nothing caught it")
        gsets = {k: {r["genome_accession"] for r in v} for k, v in parts.items()}
        overlap = ((gsets["train"] & gsets["val"]) | (gsets["train"] & gsets["test"])
                   | (gsets["val"] & gsets["test"]))
        if overlap:
            raise RuntimeError(f"{cls}: {len(overlap)} genomes cross splits")
        held = parts["val"] + parts["test"]
        nd = clu.neardup_query_ids(held, parts["train"], workdir=workdir, threads=threads)
        rc = [{"accession": r["accession"], "sequence": clu.revcomp(r["sequence"])}
              for r in held]
        nd_rc = clu.neardup_query_ids(rc, parts["train"], workdir=workdir, threads=threads)
        if nd or nd_rc:
            raise RuntimeError(
                f"{cls}: cluster-aware split still leaks — {len(nd)} forward and "
                f"{len(nd_rc)} reverse-complement near-duplicates. This is a build "
                f"failure, not something to filter."
            )
        res[cls] = {"n": {k: len(v) for k, v in parts.items()},
                    "genome_overlap": 0, "neardup_fwd": 0, "neardup_revcomp": 0}
    return res
