#!/usr/bin/env python3
"""Pre-build per-record nucleotide length sidecars for chunked training.

Writes ``<stem>.lengths.npy`` (int32, one row per JSONL line) and
``<stem>.lengths.meta.json`` (path/size/mtime fingerprint) next to each
JSONL (or under ``--cache-dir``). Idempotent: skips rebuild when metadata
matches the JSONL on disk.

Logic is duplicated (intentionally) from ``scripts/finetune_evo2_lora.py`` so
this script runs without importing PyTorch.

Used by ``scripts/finetune_evo2_lora.py`` with ``--long-seq-strategy chunk``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _lengths_sidecar_paths(cache_dir: Path, jsonl_path: Path) -> tuple[Path, Path]:
    stem = jsonl_path.stem
    return cache_dir / f"{stem}.lengths.npy", cache_dir / f"{stem}.lengths.meta.json"


def _lengths_meta_matches(meta: dict[str, Any], jsonl_path: Path) -> bool:
    stat = jsonl_path.stat()
    return (
        meta.get("path") == str(jsonl_path.resolve())
        and meta.get("size") == stat.st_size
        and meta.get("mtime") == int(stat.st_mtime)
    )


def load_or_build_lengths_sidecar(jsonl_path: Path, cache_dir: Path) -> np.ndarray:
    """Load int32 per-record sequence lengths; build sidecar if missing or stale."""
    npy_path, meta_path = _lengths_sidecar_paths(cache_dir, jsonl_path)
    stem = jsonl_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    if npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if _lengths_meta_matches(meta, jsonl_path):
            arr = np.load(npy_path)
            if int(meta.get("count", -1)) == len(arr):
                return arr.astype(np.int32, copy=False)

    lengths: list[int] = []
    with jsonl_path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            lengths.append(len(rec["sequence"]))
    arr = np.asarray(lengths, dtype=np.int32)
    stat = jsonl_path.stat()
    meta = {
        "path": str(jsonl_path.resolve()),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "count": len(arr),
    }
    tmp_npy = cache_dir / f"{stem}.lengths.pending-{os.getpid()}.npy"
    tmp_meta = meta_path.with_suffix(".json.pending")
    np.save(tmp_npy, arr)
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_npy, npy_path)
    os.replace(tmp_meta, meta_path)
    return arr


def build_nt_chunk_spans(
    seq_len_nt: int,
    max_seq_len: int,
    prefix_token_cap: int,
    chunk_overlap_nt: int,
) -> list[tuple[int, int]]:
    seq_budget = max_seq_len - prefix_token_cap
    if seq_budget <= 0:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) must exceed prefix_token_cap ({prefix_token_cap})"
        )
    if chunk_overlap_nt >= seq_budget:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap_nt}) must be < sequence budget "
            f"(max_seq_len - prefix_token_cap) = {seq_budget}"
        )
    stride = seq_budget - chunk_overlap_nt
    if stride <= 0:
        raise ValueError(
            "stride = max_seq_len - prefix_token_cap - chunk_overlap must be > 0"
        )
    if seq_len_nt <= seq_budget:
        return [(0, seq_len_nt)]
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + seq_budget, seq_len_nt)
        spans.append((start, end))
        if end >= seq_len_nt:
            break
        start += stride
    return spans


def build_all_chunk_indices(
    lengths: np.ndarray,
    max_seq_len: int,
    prefix_token_cap: int,
    chunk_overlap_nt: int,
) -> list[tuple[int, int, int]]:
    chunks: list[tuple[int, int, int]] = []
    for rec_idx, slen in enumerate(lengths.astype(int).tolist()):
        for nt0, nt1 in build_nt_chunk_spans(
            slen, max_seq_len, prefix_token_cap, chunk_overlap_nt
        ):
            chunks.append((rec_idx, nt0, nt1))
    return chunks


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        nargs="*",
        default=[
            ROOT / "data/processed/splits_combined/train.jsonl",
            ROOT / "data/processed/splits_combined/val.jsonl",
            ROOT / "data/processed/splits_combined/test.jsonl",
        ],
        help="JSONL files to index.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to write sidecars (default: same directory as each JSONL).",
    )
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--chunk-overlap", type=int, default=2048)
    p.add_argument("--prefix-budget", type=int, default=256)
    args = p.parse_args()

    for jsonl_path in args.jsonl:
        jsonl_path = jsonl_path.resolve()
        if not jsonl_path.is_file():
            print(f"SKIP (missing): {jsonl_path}", file=sys.stderr)
            continue
        cache_dir = args.cache_dir if args.cache_dir is not None else jsonl_path.parent
        cache_dir = cache_dir.resolve()
        print(f"=== {jsonl_path.name} ===")
        print(f"  cache_dir: {cache_dir}")
        lengths = load_or_build_lengths_sidecar(jsonl_path, cache_dir)
        n = len(lengths)
        print(f"  records: {n:,}")
        chunks = build_all_chunk_indices(
            lengths,
            args.max_seq_len,
            args.prefix_budget,
            args.chunk_overlap,
        )
        print(f"  chunks @ L={args.max_seq_len}, overlap={args.chunk_overlap}, "
              f"prefix_token_cap={args.prefix_budget} (CLI preview; trainer may use "
              f"meta max_prefix_tokens when --auto-prefix-budget): {len(chunks):,}  "
              f"(x{len(chunks) / max(n, 1):.3f} vs records)")
        over = int((lengths > (args.max_seq_len - args.prefix_budget)).sum())
        print(f"  records with len(sequence) > (L - prefix_budget): {over:,} "
              f"({over / max(n, 1):.2%})")


if __name__ == "__main__":
    main()
