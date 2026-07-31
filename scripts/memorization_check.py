#!/usr/bin/env python3
"""Nucleotide-level memorization / novelty check vs a reference BGC corpus.

For each query (generated BGC, or a real positive control), find the reference
BGC it is most CONTAINED in and report:
  - max_containment : exact fraction of the query's canonical k-mers present in
                      its single most-similar reference BGC (the interpretable
                      "memorization %"). ~1.0 => essentially copied.
  - max_estimate    : the MinHash containment estimate used to find candidates.
  - nearest         : which reference accession it matched.
  - verdict         : FAIL (>= --fail-threshold), WARN (>= --warn-threshold), PASS.

WHY CONTAINMENT, NOT JACCARD (audit C1/M11): a generated sequence that is a
fragment of a much longer training BGC has containment ~1.0 but tiny Jaccard.
Ranking candidates by Jaccard (the old bug) dropped exactly those memorized
fragments. We instead rank by a *containment* estimate (query sketch hashes found
in each reference's FULL k-mer set), so fragments are caught, then exact-verify
the top candidates.

Reference corpus (audit M8): pass the FULL known-BGC corpus (the grouped split,
which includes everything the model trained on plus near-duplicates), not just the
curated subset. Evo2's pretraining corpus is not accessible and is a documented
limitation of any novelty claim.

Calibration (audit C3): pass --positive-control (real held-out BGCs) to print the
"real BGC" containment distribution alongside the queries — the gate threshold
should be read off that separation, not assumed. NB: the positive control must be
de-leaked first (audit C2/C7) for the calibration to be clean.

Canonical k-mers make this strand-agnostic. Pure stdlib, no GPU. Core functions
unit-tested in tests/test_memorization.py (incl. the fragment-of-long-ref case).
"""

from __future__ import annotations

import argparse
import heapq
import json
import zlib
from pathlib import Path
from typing import Optional

_COMP = bytes.maketrans(b"ACGTN", b"TGCAN")
# Default reference = the FULL leakage-free corpus (training universe + near-dups),
# not the 18K curated subset (audit M8).
DEFAULT_REF = Path("/data2/ds85/bgcmodel_data/splits_combined_grouped/train.jsonl")


# ── Pure-logic core (unit-tested) ───────────────────────────────────────────

def _canon(kmer: bytes) -> bytes:
    rc = kmer.translate(_COMP)[::-1]
    return kmer if kmer <= rc else rc


def kmer_set(seq: str, k: int = 21) -> set:
    """Set of canonical k-mers (strand-agnostic), as bytes."""
    b = seq.upper().encode()
    return {_canon(b[i:i + k]) for i in range(len(b) - k + 1)} if len(b) >= k else set()


def kmer_hash_set(seq: str, k: int = 21) -> set:
    """Set of canonical-k-mer hashes (same hash space as minhash_sketch)."""
    b = seq.upper().encode()
    return {zlib.crc32(_canon(b[i:i + k])) & 0xFFFFFFFF
            for i in range(len(b) - k + 1)} if len(b) >= k else set()


def minhash_sketch(seq: str, k: int = 21, n: int = 400) -> list[int]:
    """Bottom-n MinHash of canonical-k-mer hashes (deterministic)."""
    hs = kmer_hash_set(seq, k)
    return sorted(hs)[:n]


def jaccard_bottomk(a: list[int], b: list[int], n: int) -> float:
    """Bottom-k MinHash Jaccard estimate (kept for reference/tests)."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union_bottom = sorted(sa | sb)[:n]
    if not union_bottom:
        return 0.0
    inter = sum(1 for h in union_bottom if h in sa and h in sb)
    return inter / len(union_bottom)


def containment_estimate(query_sketch: set, ref_hashes: set) -> float:
    """MinHash containment estimate: fraction of the query's sketch hashes present
    in the reference's FULL k-mer-hash set. Unbiased for containment and — unlike
    Jaccard — ~1.0 when the query is a fragment of a much larger reference."""
    if not query_sketch:
        return 0.0
    return len(query_sketch & ref_hashes) / len(query_sketch)


def exact_containment(query_seq: str, ref_seq: str, k: int = 21) -> float:
    """Exact fraction of the query's canonical k-mers that occur in ref."""
    q = kmer_set(query_seq, k)
    if not q:
        return 0.0
    return len(q & kmer_set(ref_seq, k)) / len(q)


# ── Streaming containment scan over the reference corpus ─────────────────────

def scan_corpus(queries: list[tuple[str, str]], ref_path: Path, k: int = 21,
                sketch_n: int = 400, top_m: int = 8, max_refs: int = 0,
                progress_every: int = 5000) -> list[dict]:
    """For each (id, seq) query, find the reference BGC it is most CONTAINED in.

    Single streaming pass over ref_path: build each reference's full k-mer-hash set,
    score every query by containment estimate, keep each query's top-m candidate
    offsets, then exact-verify those few candidates. Correct for the fragment case.
    """
    q_sketches = [set(minhash_sketch(seq, k, sketch_n)) for _, seq in queries]
    heaps: list[list] = [[] for _ in queries]  # per query: min-heap of (estimate, offset)

    with ref_path.open("rb") as f:
        offset, i = 0, 0
        while True:
            line = f.readline()
            if not line:
                break
            R = kmer_hash_set(json.loads(line).get("sequence", ""), k)
            if R:
                for qi, qsk in enumerate(q_sketches):
                    if not qsk:
                        continue
                    est = len(qsk & R) / len(qsk)
                    if est > 0:
                        h = heaps[qi]
                        if len(h) < top_m:
                            heapq.heappush(h, (est, offset))
                        elif est > h[0][0]:
                            heapq.heapreplace(h, (est, offset))
            offset += len(line)
            i += 1
            if max_refs and i >= max_refs:
                break
            if progress_every and i % progress_every == 0:
                print(f"  scanned {i:,} reference BGCs...", flush=True)

    n_refs = i          # how many reference records were actually scanned (see --max-refs)

    def _read(off: int) -> dict:
        with ref_path.open("rb") as f:
            f.seek(off)
            return json.loads(f.readline())

    results = []
    for qi, (qid, qseq) in enumerate(queries):
        max_cont, acc = 0.0, None
        max_est = max((e for e, _ in heaps[qi]), default=0.0)
        for est, off in heaps[qi]:
            rec = _read(off)
            cont = exact_containment(qseq, rec.get("sequence", ""), k)
            if cont > max_cont:
                max_cont, acc = cont, rec.get("accession")
        # PROVENANCE. Without it, a scan against a truncated or wrong reference corpus is
        # indistinguishable from a scan against the real one: every query comes back
        # PASS_novel and the suite certifies the model as non-memorizing on the strength of a
        # partial comparison. The `novel` GATE is the pre-synthesis safety check, so the
        # corpus it was actually run against has to travel with the number.
        results.append({"id": qid, "length": len(qseq),
                        "max_containment": round(max_cont, 4),
                        "max_estimate": round(max_est, 4),
                        "nearest_accession": acc,
                        "ref_corpus": str(ref_path),
                        "ref_n_records": n_refs,
                        "k": k})
    return results


def verdict_for(max_containment: float, fail_thr: float, warn_thr: float) -> str:
    if max_containment >= fail_thr:
        return "FAIL"
    if max_containment >= warn_thr:
        return "WARN"
    return "PASS"


# ── Query I/O ───────────────────────────────────────────────────────────────

def load_seqs(path: Path) -> list[tuple[str, str]]:
    if path.suffix in (".fasta", ".fa", ".fna"):
        out, sid, buf = [], None, []
        for line in path.open():
            line = line.rstrip()
            if line.startswith(">"):
                if sid is not None:
                    out.append((sid, "".join(buf)))
                sid, buf = line[1:].split()[0], []
            else:
                buf.append(line)
        if sid is not None:
            out.append((sid, "".join(buf)))
        return out
    recs = [json.loads(l) for l in path.open()]
    return [(r.get("accession") or r.get("id") or str(i), r.get("sequence", ""))
            for i, r in enumerate(recs)]


def _summary(label: str, results: list[dict], fail_thr: float, warn_thr: float):
    conts = sorted(r["max_containment"] for r in results)
    if not conts:
        print(f"[{label}] no sequences"); return
    n_fail = sum(1 for c in conts if c >= fail_thr)
    n_warn = sum(1 for c in conts if warn_thr <= c < fail_thr)
    p = lambda q: conts[min(len(conts) - 1, int(q * (len(conts) - 1)))]
    print(f"[{label}] max_containment: median={p(0.5):.3f} p90={p(0.9):.3f} "
          f"max={conts[-1]:.3f} | FAIL(>= {fail_thr})={n_fail} "
          f"WARN={n_warn} PASS={len(conts) - n_fail - n_warn}  (n={len(conts)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF,
                    help="Reference corpus JSONL (default: full grouped train).")
    ap.add_argument("--positive-control", type=Path, default=None,
                    help="Real held-out BGCs to scan alongside for threshold calibration.")
    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--sketch-size", type=int, default=400)
    ap.add_argument("--top-m", type=int, default=8,
                    help="Candidates exact-verified per query (raise for safety).")
    ap.add_argument("--max-refs", type=int, default=0, help="Cap reference records (0=all; demo).")
    ap.add_argument("--fail-threshold", type=float, default=0.95)
    ap.add_argument("--warn-threshold", type=float, default=0.80)
    ap.add_argument("--label", default="query")
    ap.add_argument("--output", type=Path, default=Path("memorization_report.jsonl"))
    args = ap.parse_args()

    queries = load_seqs(args.query)
    # Scan queries (+ positive control, if given) in ONE pass each over the ref.
    print(f"Scanning {len(queries)} '{args.label}' sequences vs {args.ref} (k={args.k})...")
    q_res = scan_corpus(queries, args.ref, args.k, args.sketch_size, args.top_m, args.max_refs)
    for r in q_res:
        r["label"] = args.label
        r["verdict"] = verdict_for(r["max_containment"], args.fail_threshold, args.warn_threshold)

    pc_res = []
    if args.positive_control is not None:
        pc = load_seqs(args.positive_control)
        print(f"Scanning {len(pc)} positive-control sequences for calibration...")
        pc_res = scan_corpus(pc, args.ref, args.k, args.sketch_size, args.top_m, args.max_refs)
        for r in pc_res:
            r["label"] = "positive_control"
            r["verdict"] = verdict_for(r["max_containment"], args.fail_threshold, args.warn_threshold)

    with args.output.open("w") as out:
        for r in q_res + pc_res:
            out.write(json.dumps(r) + "\n")

    print()
    _summary(args.label, q_res, args.fail_threshold, args.warn_threshold)
    if pc_res:
        _summary("positive_control", pc_res, args.fail_threshold, args.warn_threshold)
        print("  NOTE: calibrate the threshold from the separation between the two "
              "distributions; the positive control must be de-leaked first (audit C2/C7).")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
