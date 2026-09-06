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

#: A 1-nt record is a parse artifact, not a core. An all-N record carries no sequence.
MIN_LEN = 200
MAX_N_FRAC = 0.10


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
            if r["seq_len"] < MIN_LEN:
                continue
            if r["sequence"].count("N") / max(r["seq_len"], 1) > MAX_N_FRAC:
                continue
            if want & set(r["classes"]):
                out.append(r)
    # CANONICAL ORDER. extract.py writes with pool.imap_unordered, so corpus LINE order is
    # worker-completion order and differs on every rebuild. mmseqs easy-cluster is
    # order-sensitive -- reordering identical records changes the partition and ~24% of
    # representative labels -- and cluster selection is by sorted(_frac(representative)),
    # so a relabel moves clusters across the common_n boundary. Sorting here makes the
    # build a function of the corpus CONTENT, which is what SPEC 4.6 claims.
    out.sort(key=lambda r: r["accession"])
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

    # --- select each class's common_n clusters BEFORE assigning splits ---------------
    # Order matters. Assigning components first and selecting clusters afterwards makes
    # the balance uncontrollable, because a component's class composition is only known
    # once selection has happened.
    cls_clusters: dict[str, dict[str, list[dict]]] = {}
    cls_ordered: dict[str, list[str]] = {}
    selected: dict[str, list[str]] = {}
    for cls in classes:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            if cls in r["classes"]:
                groups[rep[r["accession"]]].append(r)
        ordered = sorted(groups, key=lambda k: (_frac(k), k))
        if len(ordered) < common_n:
            raise RuntimeError(
                f"{cls}: only {len(ordered)} clusters at max_len={max_len}, below "
                f"common_n={common_n}. SPEC 4.4.2 must be revisited before building — "
                f"silently shrinking one class breaks equal-n."
            )
        cls_clusters[cls] = groups
        cls_ordered[cls] = ordered          # per class; `ordered` alone leaks the last one
        selected[cls] = ordered[:common_n]

    # --- components: linked by shared genome OR shared cluster -----------------------
    uf = _Union()
    for r in records:
        a = ("acc", r["accession"])
        uf.union(a, ("gen", r["genome_accession"]))
        uf.union(a, ("clu", rep[r["accession"]]))
    comp_of_cluster: dict[str, object] = {}
    for cls in classes:
        for ck in selected[cls]:
            comp_of_cluster[ck] = uf.find(("clu", ck))

    members = defaultdict(list)
    for ck, c in comp_of_cluster.items():
        members[c].append(ck)
    comp_key = {c: min(v) for c, v in members.items()}

    # per-component, per-class selected-cluster counts
    sel_of = {cls: set(selected[cls]) for cls in classes}
    counts: dict[object, dict[str, int]] = {
        c: {cls: sum(1 for ck in v if ck in sel_of[cls]) for cls in classes}
        for c, v in members.items()
    }

    # GREEDY, PER-CLASS-AWARE BALANCED ASSIGNMENT.
    # Components chain hard (a record links by genome OR cluster), so a handful are huge.
    # Hashing each component independently gave 93/3.5/3.5; balancing on the union alone
    # gave ~60/20/20 because class composition varies between components. Place each
    # component, largest first, into the split whose per-class deficits it reduces most.
    # Ties break on the stable component key, so this stays deterministic and seedless.
    target = {cls: {k: FRACS[k] * common_n for k in FRACS} for cls in classes}
    have = {cls: {k: 0.0 for k in FRACS} for cls in classes}
    order = sorted(members, key=lambda c: (-sum(counts[c].values()), comp_key[c]))
    split_of_comp: dict = {}
    for c in order:
        best, best_gain = None, None
        for k in FRACS:
            gain = sum(
                min(counts[c][cls], max(0.0, target[cls][k] - have[cls][k]))
                for cls in classes
            )
            if best_gain is None or gain > best_gain:
                best, best_gain = k, gain
        split_of_comp[c] = best
        for cls in classes:
            have[cls][best] += counts[c][cls]

    # Components holding no selected cluster still need an assignment, because analyses
    # that draw from the FULL class pool (SPEC 4.5 stratification) must inherit the same
    # train/val/test boundary -- otherwise a stratum record could sit in another corpus's
    # test set. They cannot affect the balance of the selected clusters (their counts are
    # zero), so the seedless hash is the right assignment for them.
    # `members` only holds components carrying a SELECTED cluster, so members.get() always
    # missed here and the key degenerated to whichever record happened to come first --
    # i.e. it was corpus-order dependent, not canonical. Build the key from the component's
    # full membership instead.
    comp_members: dict = defaultdict(list)
    for r in records:
        comp_members[uf.find(("clu", rep[r["accession"]]))].append(r["accession"])
    for c, accs in comp_members.items():
        if c not in split_of_comp:
            split_of_comp[c] = _assign(min(accs))

    # cluster -> split, for EVERY cluster in the class union, not only selected ones
    cluster_split = {rep[r["accession"]]: split_of_comp[uf.find(("clu", rep[r["accession"]]))]
                     for r in records}
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "cluster_split.json").write_text(json.dumps(cluster_split))
    record_split = {r["accession"]: cluster_split[rep[r["accession"]]] for r in records}
    (out_dir / "record_split.json").write_text(json.dumps(record_split))

    # --- per class: take common_n clusters, honouring the global assignment ----------
    report: dict[str, dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        clusters = cls_clusters[cls]
        chosen = selected[cls]
        def _pick(ck):
            recs = sorted(clusters[ck], key=lambda r: r["accession"])
            return next((r for r in recs if r["accession"] == ck), recs[0])

        buckets: dict[str, list[dict]] = {k: [] for k in FRACS}
        for ck in chosen:
            buckets[split_of_comp[comp_of_cluster[ck]]].append(_pick(ck))

        # LEAK REMOVAL HAPPENS HERE, WITH BACKFILL, so equal-n survives it.
        # Previously verify() deleted leaking held-out records and rewrote the files after
        # the report was computed, so the manifest published 979/123/122 while disk held
        # 979/107/110 for BETALACTONE -- and the deletion is not random, it removes exactly
        # the held-out records most similar to training, biasing held-out difficulty
        # upward by a different amount in every class.
        spare = list(cls_ordered[cls][common_n:])
        removed = 0
        for _ in range(6):
            held = buckets["val"] + buckets["test"]
            if not held:
                break
            fwd = clu.neardup_query_ids(held, buckets["train"], workdir=workdir,
                                        threads=threads)
            rc = [{"accession": r["accession"],
                   "sequence": clu.revcomp(r["sequence"])} for r in held]
            rev = clu.neardup_query_ids(rc, buckets["train"], workdir=workdir,
                                        threads=threads)
            bad = fwd | rev
            if not bad:
                break
            for part in ("val", "test"):
                keep = [r for r in buckets[part] if r["accession"] not in bad]
                need = len(buckets[part]) - len(keep)
                removed += need
                while need and spare:
                    ck = spare.pop(0)
                    comp = comp_of_cluster.get(ck) or uf.find(("clu", ck))
                    if split_of_comp.get(comp) != part or not clusters.get(ck):
                        continue
                    keep.append(_pick(ck))
                    need -= 1
                buckets[part] = keep
        if removed:
            report_extra = {"leaking_held_out_replaced": removed}
        else:
            report_extra = {"leaking_held_out_replaced": 0}

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
                sum(1 for x in picked if len(x["classes"]) > 1) / max(len(picked), 1), 4),
            "n_products": len({p for x in picked for p in x["antismash_products"]}),
            # SPEC 3.3 defines hybrids at CLASS level; counting multi-PRODUCT records
            # inflated NRPS to 0.281 where the multi-class figure is 0.208, because
            # {NRPS, NRPS-like} and {NRP-metallophore, NRPS} both map to {NRPS} alone.
            "hybrid_frac_multiproduct": round(
                sum(1 for x in picked if len(x["antismash_products"]) > 1)
                / max(len(picked), 1), 4),
            **report_extra,
        }
    return {"classes": report, "max_len": max_len, "common_n": common_n,
            "n_records_considered": len(records),
            "clustering": {"min_seq_id": clu.MIN_SEQ_ID, "coverage": clu.COVERAGE,
                           "cov_mode": clu.COV_MODE}}


def verify(out_dir: Path, classes: tuple[str, ...], workdir: Path | None = None,
           threads: int = 16) -> dict:
    """SPEC 4.6 integrity assertions. STRICTLY READ-ONLY.

    This function previously deleted leaking held-out records and rewrote the split files
    in place, after the report had already been computed -- so the manifest published
    979/123/122 while disk held 979/107/110. Removal now happens in build() with backfill
    (which preserves equal-n); verify's only job is to confirm the property and RAISE if
    it does not hold.
    """
    res: dict[str, dict] = {}
    for cls in classes:
        d = out_dir / cls
        parts = {k: [json.loads(l) for l in open(d / f"{k}.jsonl")] for k in FRACS}
        for k, rows in parts.items():
            if not rows:
                raise RuntimeError(f"{cls}/{k} is EMPTY")
            accs = [r["accession"] for r in rows]
            if len(accs) != len(set(accs)):
                raise RuntimeError(f"{cls}/{k} contains duplicate accessions")
        gsets = {k: {r["genome_accession"] for r in v} for k, v in parts.items()}
        overlap = ((gsets["train"] & gsets["val"]) | (gsets["train"] & gsets["test"])
                   | (gsets["val"] & gsets["test"]))
        if overlap:
            raise RuntimeError(f"{cls}: {len(overlap)} genomes cross splits")
        held = parts["val"] + parts["test"]
        fwd = clu.neardup_query_ids(held, parts["train"], workdir=workdir, threads=threads)
        rc = [{"accession": r["accession"], "sequence": clu.revcomp(r["sequence"])}
              for r in held]
        rev = clu.neardup_query_ids(rc, parts["train"], workdir=workdir, threads=threads)
        if fwd or rev:
            raise RuntimeError(
                f"{cls}: {len(fwd)} forward and {len(rev)} reverse-complement near-"
                f"duplicates between held-out and train. build() should have replaced "
                f"them; this is a build failure and verify does not filter."
            )
        res[cls] = {"n": {k: len(v) for k, v in parts.items()},
                    "genome_overlap": 0, "neardup_fwd": len(fwd),
                    "neardup_revcomp": len(rev)}
    return res
