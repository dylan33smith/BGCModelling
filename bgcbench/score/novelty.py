"""The novelty gate (SPEC 3.8).

Every ladder metric is maximised by reproducing training data, so novelty is an ABSOLUTE
GATE applied before any rate is read -- not a number reported alongside one.

⚠ IT FAILS CLOSED. KNOWN_WRONG #3: the prior implementation computed containment with
`max(..., default=0.0)`, and 0.0 is the PASSING value, so an empty k-mer set silently
passed the gate. A gate that defaults to passing is not a gate. Every path here that
cannot compute a containment raises.
"""
from __future__ import annotations

import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

K = 21
FAIL_AT = 0.95
WARN_AT = 0.80

#: A reference record shorter than this contributes no usable k-mer set. It also removes
#: the 1-nt records that a fuzzy-coordinate parse can emit.
MIN_REF_KMERS = 50
_COMP = str.maketrans("ACGTN", "TGCAN")


def canonical_kmers(seq: str, k: int = K) -> set[str]:
    s = seq.upper()
    out: set[str] = set()
    for i in range(len(s) - k + 1):
        km = s[i:i + k]
        if "N" in km:
            continue
        rc = km.translate(_COMP)[::-1]
        out.add(min(km, rc))
    return out


class Reference:
    """Inverted k-mer index over ONE arm's training split.

    The reference set is per arm (SPEC 3.8): an arm trained on a pooled corpus is gated
    against the pooled corpus. Using one shared reference would under-gate whichever arm
    trained on more data.
    """

    def __init__(self, records: list[dict], k: int = K):
        self.k = k
        self.n = len(records)
        self.index: dict[str, set[int]] = defaultdict(set)
        self.sizes: list[int] = []
        for i, r in enumerate(records):
            km = canonical_kmers(r["sequence"], k)
            self.sizes.append(len(km))
            for x in km:
                self.index[x].add(i)
        if not self.index:
            raise ValueError(
                "novelty reference is empty — refusing to build a gate that cannot fail"
            )

    def _hits(self, seq: str) -> tuple[dict[int, int], int]:
        km = canonical_kmers(seq, self.k)
        if not km:
            raise ValueError(
                f"sequence of length {len(seq)} yields no valid {self.k}-mers, so "
                f"containment is undefined. The gate raises rather than returning 0.0, "
                f"which is the PASSING value (KNOWN_WRONG #3)."
            )
        hits: dict[int, int] = defaultdict(int)
        for x in km:
            for i in self.index.get(x, ()):
                hits[i] += 1
        return hits, len(km)

    def containment(self, seq: str) -> float:
        """FORWARD: max over reference records t of |kmers(s) & kmers(t)| / |kmers(s)|.

        ⚠ THIS ALONE CANNOT FAIL AT THE GENERATION BUDGET. The denominator is the WHOLE
        generation, so containment falls as 1/length no matter how much was copied.
        Measured: ten whole verbatim TERPENE training records concatenated to 15,992 nt
        score forward 0.198 -- PASS -- while antiSMASH calls them on-target with 15 core
        genes. Use `verdict()`, which gates on both directions.
        """
        hits, n = self._hits(seq)
        return max(hits.values()) / n if hits else 0.0

    def reverse_containment(self, seq: str) -> float:
        """REVERSE: max over reference records t of |kmers(s) & kmers(t)| / |kmers(t)|.

        "How much of some training record does this generation contain?" -- which does not
        dilute with generation length, so a collage of copied records is caught. On the
        same memorised concatenation this reads 1.000 against 0.000-0.006 for clean
        held-out and random sequence.
        """
        hits, _ = self._hits(seq)
        best = 0.0
        for i, c in hits.items():
            if self.sizes[i] >= MIN_REF_KMERS:
                best = max(best, c / self.sizes[i])
        return best

    def verdict(self, seq: str) -> dict:
        fwd = self.containment(seq)
        rev = self.reverse_containment(seq)
        worst = max(fwd, rev)
        return {
            "containment": round(fwd, 4),
            "containment_reverse": round(rev, 4),
            "containment_worst": round(worst, 4),
            "novel": worst < FAIL_AT,
            "gate": "FAIL_memorized" if worst >= FAIL_AT else ("WARN" if worst >= WARN_AT
                                                              else "PASS"),
        }


# --------------------------------------------------------------------------------------
# CORPUS-LEVEL REFERENCE (SPEC 3.8)
#
# The per-arm reference answers "did this arm memorise its own training data?". It is
# unconstructible for W0 and bgcfm, which we never trained -- Reference([]) raises by
# design -- and it is blind to the deeper problem: Evo2 and GenomeOcean were PRETRAINED on
# public genome collections that include the source genomes of this corpus. A base-model
# arm reproducing a real cluster is memorising from pretraining, which the per-arm
# reference cannot see at all.
#
# This answers the other question -- "did the model output a KNOWN BGC?" -- for EVERY arm,
# including the untrained floor. The exact k-mer index does not scale to ~309k records, so
# it uses mmseqs at the SPEC 4.4 criterion.
# --------------------------------------------------------------------------------------

MMSEQS = "/home/ds85/.local/share/mamba/envs/bgcmodel/bin/mmseqs"
CORPUS_MIN_ID = 0.8
CORPUS_COV = 0.5


def corpus_novelty(generations: list[dict], corpus_fasta: Path,
                   workdir: Path | None = None, threads: int = 16) -> dict[str, dict]:
    """generation_id -> {matched_known_bgc, best_identity, best_target}."""
    if not generations:
        return {}
    if not corpus_fasta.exists():
        raise FileNotFoundError(
            f"corpus reference {corpus_fasta} absent — refusing to report corpus-level "
            f"novelty as PASS when it was never computed (the gate fails closed)."
        )
    with tempfile.TemporaryDirectory(dir=str(workdir) if workdir else None) as td:
        tmp = Path(td)
        q, out = tmp / "q.fa", tmp / "h.tsv"
        with open(q, "w") as fh:
            for g in generations:
                fh.write(f">{g['generation_id']}\n{g['sequence']}\n")
        proc = subprocess.run(
            [MMSEQS, "easy-search", str(q), str(corpus_fasta), str(out), str(tmp / "t"),
             "--search-type", "3", "--min-seq-id", str(CORPUS_MIN_ID),
             "-c", str(CORPUS_COV), "--cov-mode", "2",
             "--format-output", "query,target,fident,qcov", "-e", "1e-5",
             "--threads", str(threads)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"mmseqs easy-search failed:\n{proc.stderr[-2000:]}")
        best: dict[str, tuple[float, str]] = {}
        if out.exists():
            for line in out.read_text().splitlines():
                if not line.strip():
                    continue
                qid, tid, fid, _cov = line.split("\t")[:4]
                f = float(fid)
                if qid not in best or f > best[qid][0]:
                    best[qid] = (f, tid)
    res = {}
    for g in generations:
        gid = g["generation_id"]
        hit = best.get(gid)
        res[gid] = {"matched_known_bgc": hit is not None,
                    "best_identity": round(hit[0], 4) if hit else 0.0,
                    "best_target": hit[1] if hit else None}
    return res


def write_corpus_fasta(records: list[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(f">{r['accession']}\n{r['sequence']}\n")
    return out
