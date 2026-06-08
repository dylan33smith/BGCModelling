#!/usr/bin/env python3
"""Nucleotide-level memorization / novelty check vs the training set.

For each query sequence (generated BGCs, or the real-MiBIG positive control), find
its nearest training BGC by canonical-k-mer MinHash and report:
  - max_jaccard       : MinHash Jaccard to the closest training sequence
  - max_containment   : exact fraction of the query's k-mers found in that
                        training sequence (the interpretable "memorization" number)
  - nearest           : which training accession it matched

A generated sequence with containment ~1.0 is essentially copied from a training
BGC (memorized); low values mean novel content. Calibrate against the positive
control: real held-out BGCs have *some* k-mer overlap with training but should not
sit at ~1.0 — generated sequences should look like the real ones, not above them.

Canonical k-mers (min of a k-mer and its reverse complement) make this
strand-agnostic. The training index (per-sequence MinHash sketch + byte offset) is
built once and cached.

NOTE: pure-Python + stdlib only (no GPU, no external tools). Core similarity
functions are unit-tested in tests/test_memorization.py.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Optional

_COMP = bytes.maketrans(b"ACGTN", b"TGCAN")
DEFAULT_TRAIN = Path("/data2/ds85/bgcmodel_data/splits_curated/train.jsonl")


# ── Pure-logic core (unit-tested) ───────────────────────────────────────────

def _canon(kmer: bytes) -> bytes:
    rc = kmer.translate(_COMP)[::-1]
    return kmer if kmer <= rc else rc


def kmer_set(seq: str, k: int = 21) -> set:
    """Set of canonical k-mers (strand-agnostic)."""
    b = seq.upper().encode()
    return {_canon(b[i:i + k]) for i in range(len(b) - k + 1)} if len(b) >= k else set()


def minhash_sketch(seq: str, k: int = 21, n: int = 200) -> list[int]:
    """Bottom-n MinHash of canonical k-mers (deterministic crc32 hashing)."""
    b = seq.upper().encode()
    if len(b) < k:
        return []
    hashes = {zlib.crc32(_canon(b[i:i + k])) & 0xFFFFFFFF for i in range(len(b) - k + 1)}
    return sorted(hashes)[:n]


def jaccard_bottomk(a: list[int], b: list[int], n: int) -> float:
    """Bottom-k MinHash Jaccard estimate from two sorted sketches."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union_bottom = sorted(sa | sb)[:n]
    if not union_bottom:
        return 0.0
    inter = sum(1 for h in union_bottom if h in sa and h in sb)
    return inter / len(union_bottom)


def exact_containment(query_seq: str, ref_seq: str, k: int = 21) -> float:
    """Fraction of the query's canonical k-mers that occur in ref (memorization %)."""
    q = kmer_set(query_seq, k)
    if not q:
        return 0.0
    return len(q & kmer_set(ref_seq, k)) / len(q)


# ── Training index (build once, cache) ──────────────────────────────────────

def build_index(train_path: Path, k: int, n: int, max_records: int = 0) -> dict:
    records = []
    with train_path.open("rb") as f:
        offset = 0
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            rec = json.loads(line)
            seq = rec.get("sequence", "")
            records.append({
                "accession": rec.get("accession", f"train_{i}"),
                "len": len(seq),
                "offset": offset,
                "sketch": minhash_sketch(seq, k, n),
            })
            offset += len(line)
            i += 1
            if max_records and i >= max_records:
                break
            if i % 2000 == 0:
                print(f"  indexed {i} training records...", flush=True)
    return {"k": k, "n": n, "train_path": str(train_path), "records": records}


def load_or_build_index(train_path: Path, cache: Path, k: int, n: int,
                        max_records: int = 0, rebuild: bool = False) -> dict:
    if cache.exists() and not rebuild:
        idx = json.loads(cache.read_text())
        if idx.get("k") == k and idx.get("n") == n:
            print(f"Loaded training index ({len(idx['records']):,} records) from {cache}")
            return idx
    print(f"Building training index from {train_path} (k={k}, n={n})...")
    idx = build_index(train_path, k, n, max_records)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx))
    print(f"  cached {len(idx['records']):,} sketches to {cache}")
    return idx


def _fetch_train_seq(train_path: Path, offset: int) -> str:
    with train_path.open("rb") as f:
        f.seek(offset)
        return json.loads(f.readline()).get("sequence", "")


def nearest(query_seq: str, index: dict, top_m: int = 3) -> dict:
    """Nearest training neighbor by MinHash, refined with exact containment."""
    k, n = index["k"], index["n"]
    qsketch = minhash_sketch(query_seq, k, n)
    scored = [(jaccard_bottomk(qsketch, r["sketch"], n), r) for r in index["records"]]
    scored.sort(key=lambda t: t[0], reverse=True)
    max_jac = scored[0][0] if scored else 0.0

    # Exact containment for the top-M MinHash candidates (re-read their sequences).
    train_path = Path(index["train_path"])
    best_cont, best_acc = 0.0, None
    for jac, r in scored[:top_m]:
        cont = exact_containment(query_seq, _fetch_train_seq(train_path, r["offset"]), k)
        if cont > best_cont:
            best_cont, best_acc = cont, r["accession"]
    return {"max_jaccard": round(max_jac, 4),
            "max_containment": round(best_cont, 4),
            "nearest_accession": best_acc}


# ── Query I/O ───────────────────────────────────────────────────────────────

def load_query_seqs(path: Path) -> list[tuple[str, str]]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=Path, required=True,
                    help="Sequences to check (FASTA or JSONL): generated or positive control.")
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--index-cache", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/train_memo_index.json"))
    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--sketch-size", type=int, default=200)
    ap.add_argument("--max-train-records", type=int, default=0,
                    help="Cap training records indexed (0 = all; use for a quick demo).")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--memorized-threshold", type=float, default=0.95,
                    help="max_containment above this flags a query as likely memorized.")
    ap.add_argument("--label", default="query")
    ap.add_argument("--output", type=Path, default=Path("memorization_report.jsonl"))
    args = ap.parse_args()

    index = load_or_build_index(args.train, args.index_cache, args.k, args.sketch_size,
                                args.max_train_records, args.rebuild_index)
    queries = load_query_seqs(args.query)
    print(f"\nChecking {len(queries)} '{args.label}' sequences against "
          f"{len(index['records']):,} training BGCs (k={args.k})...")

    results = []
    n_mem = 0
    with args.output.open("w") as out:
        for i, (sid, seq) in enumerate(queries):
            res = nearest(seq, index)
            res.update({"id": sid, "label": args.label, "length": len(seq)})
            res["memorized"] = res["max_containment"] >= args.memorized_threshold
            n_mem += int(res["memorized"])
            results.append(res)
            out.write(json.dumps(res) + "\n")
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(queries)}...", flush=True)

    conts = sorted(r["max_containment"] for r in results)
    mid = conts[len(conts) // 2] if conts else 0.0
    print(f"\n[{args.label}] max_containment to nearest training BGC:")
    print(f"  median={mid:.3f}  min={conts[0]:.3f}  max={conts[-1]:.3f}" if conts else "  (none)")
    print(f"  flagged memorized (>= {args.memorized_threshold}): {n_mem}/{len(results)}")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
