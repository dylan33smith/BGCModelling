"""Sequence clustering (SPEC 4.4, 4.6).

effective_n -- the number of distinct sequence clusters -- is the sample size that
matters. A class with 4,661 records and 873 clusters has 873 independent things to learn,
and quoting 4,661 overstates it by 5x.

The same clustering does double duty: it defines effective_n, and it supplies the grouping
key that makes the split cluster-disjoint by construction rather than by post-hoc
filtering.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

MMSEQS = "/home/ds85/.local/share/mamba/envs/bgcmodel/bin/mmseqs"

#: SPEC 4.4 [C] -- inherited from prior practice and stated so it can be varied.
MIN_SEQ_ID = 0.8
COVERAGE = 0.5
COV_MODE = 2          # coverage of the query


def cluster(records: list[dict], workdir: Path | None = None,
            min_seq_id: float = MIN_SEQ_ID, coverage: float = COVERAGE,
            threads: int = 8) -> dict[str, str]:
    """Return accession -> cluster representative accession."""
    if not records:
        return {}
    with tempfile.TemporaryDirectory(dir=str(workdir) if workdir else None) as td:
        tmp = Path(td)
        fa = tmp / "in.fa"
        with open(fa, "w") as fh:
            for r in records:
                fh.write(f">{r['accession']}\n{r['sequence']}\n")
        proc = subprocess.run(
            [MMSEQS, "easy-cluster", str(fa), str(tmp / "c"), str(tmp / "t"),
             "--min-seq-id", str(min_seq_id), "-c", str(coverage),
             "--cov-mode", str(COV_MODE), "--threads", str(threads)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"mmseqs easy-cluster failed:\n{proc.stderr[-2000:]}")
        out = tmp / "c_cluster.tsv"
        if not out.exists():
            raise RuntimeError("mmseqs produced no cluster table")
        member_to_rep: dict[str, str] = {}
        for line in out.read_text().splitlines():
            if not line.strip():
                continue
            rep, member = line.split("\t")[:2]
            member_to_rep[member] = rep
    missing = {r["accession"] for r in records} - set(member_to_rep)
    if missing:
        raise RuntimeError(
            f"{len(missing)} records absent from the cluster table (e.g. "
            f"{sorted(missing)[:3]}). Refusing to proceed: a silently dropped record "
            f"would understate effective_n."
        )
    return member_to_rep


def neardup_query_ids(query: list[dict], target: list[dict],
                      workdir: Path | None = None, threads: int = 8) -> set[str]:
    """Accessions in `query` with a near-duplicate in `target`. Used as a VERIFICATION
    pass after a cluster-aware split (SPEC 4.6) -- a non-zero count is a build failure,
    not something to filter away."""
    if not query or not target:
        return set()
    with tempfile.TemporaryDirectory(dir=str(workdir) if workdir else None) as td:
        tmp = Path(td)
        qf, tf, out = tmp / "q.fa", tmp / "t.fa", tmp / "h.tsv"
        for path, recs in ((qf, query), (tf, target)):
            with open(path, "w") as fh:
                for r in recs:
                    fh.write(f">{r['accession']}\n{r['sequence']}\n")
        proc = subprocess.run(
            [MMSEQS, "easy-search", str(qf), str(tf), str(out), str(tmp / "t2"),
             "--search-type", "3", "--min-seq-id", str(MIN_SEQ_ID), "-c", str(COVERAGE),
             "--cov-mode", str(COV_MODE), "--format-output", "query,target,fident,qcov",
             "-e", "1e-5", "--threads", str(threads)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"mmseqs easy-search failed:\n{proc.stderr[-2000:]}")
        if not out.exists():
            return set()
        return {l.split("\t")[0] for l in out.read_text().splitlines() if l.strip()}


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]
