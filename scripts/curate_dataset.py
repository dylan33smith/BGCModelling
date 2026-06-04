#!/usr/bin/env python3
"""Curate a small, robust, representative training set from the grouped split.

RATIONALE (see AUDIT_FINDINGS.md and the curation discussion)
------------------------------------------------------------
The combined training set (~280K records) is prohibitively large for a single
H100 (~20 days/epoch) and highly redundant (Streptomyces-dominated, ~95% from
three phyla, many near-duplicate clusters). Evo2 7B is already pretrained on a
massive nucleotide corpus, so Phase-1 fine-tuning is largely teaching the
*conditioning interface* — a regime where a few thousand well-chosen examples
per class go a long way (cf. LIMA). This script curates a compact training set
that is balanced across classes, diverse across taxa and BGC lengths, and
quality-filtered, while leaving validation/test FULL and representative.

KEY DESIGN
----------
- Input is the leakage-free GROUPED split (genomes disjoint across train/val/
  test). Curating *within* each split therefore preserves leakage-freedom by
  construction — no re-splitting needed.
- TRAIN: quality-filter -> per-class budget cap -> stratified, diversity-aware
  selection (across phylum x length-bucket, preferring distinct genomes), with
  all MIBiG gold records kept.
- VAL/TEST: quality-filter only (kept full and representative for robust,
  unbiased evaluation). Optionally cap with --eval-cap.

QUALITY FILTERS (all default ON)
- drop sequences containing ambiguous bases (N / non-ACGT)  -> AUDIT M8
- drop contig_edge=True records (potentially incomplete BGCs)
- drop sequences shorter than --min-len

MEMORY
- Two-pass streaming per split: pass 1 collects per-record metadata only
  (class, phylum, length bucket, genome, flags); selection is decided from
  metadata; pass 2 re-streams and writes only the selected lines.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bgc_pipeline.evaluation import phylum_token  # noqa: E402

_VALID_BASES = set("ACGT")

# Production chunk-mode params, used only to ESTIMATE optimizer steps/epoch.
# Training iterates over chunk WINDOWS, not records: long BGCs (NRPS/PKS/hybrids)
# tile into several windows, so the real step count is well above records/128.
# Mirrors scripts/finetune_evo2_lora.py defaults.
EST_MAX_SEQ_LEN = 32768
EST_CHUNK_OVERLAP = 2048
EST_EOS_RESERVE = 5  # len("|END|")


def _chunk_window_count(length: int, seq_budget: int, overlap: int) -> int:
    """Number of chunk windows a sequence of `length` nt tiles into."""
    if length <= seq_budget:
        return 1
    stride = seq_budget - overlap
    return 1 + (length - seq_budget + stride - 1) // stride  # 1 + ceil((L-B)/stride)


def length_bucket(n: int, bounds: list[int]) -> str:
    """Label a sequence length by configurable bucket boundaries."""
    prev = 0
    for b in bounds:
        if n <= b:
            return f"<={b}"
        prev = b
    return f">{prev}"


def genome_key(rec: dict) -> str:
    g = rec.get("genome_accession")
    if g:
        return f"g::{g}"
    a = rec.get("accession")
    return f"a::{a}" if a else "unknown"


def is_mibig(rec: dict) -> bool:
    return str(rec.get("accession", "")).startswith("BGC")


def passes_quality(rec: dict, drop_ambiguous: bool, drop_edge: bool, min_len: int) -> bool:
    seq = rec.get("sequence", "")
    if len(seq) < min_len:
        return False
    if drop_edge and rec.get("contig_edge") is True:
        return False
    if drop_ambiguous and seq:
        # cheap fast path: any base outside ACGT (covers N and other IUPAC codes)
        if not set(seq.upper()) <= _VALID_BASES:
            return False
    return True


def allocate_budget(avail: dict[str, int], budget: int, floor_min: int = 1) -> dict[str, int]:
    """Allocate a per-class budget across (phylum x length) cells.

    Proportional to availability (preserves within-class shape), with a small
    per-cell floor so rare strata — notably long BGCs and minor phyla — are not
    dropped. Caps each cell at its availability. Largest-remainder rounding.
    """
    cells = [c for c, n in avail.items() if n > 0]
    total = sum(avail[c] for c in cells)
    if total <= budget:
        return {c: avail[c] for c in cells}

    raw = {c: budget * avail[c] / total for c in cells}
    alloc = {c: min(int(raw[c]), avail[c]) for c in cells}

    # Distribute the rounding remainder by largest fractional part (cap at avail).
    rem = budget - sum(alloc.values())
    order = sorted(cells, key=lambda c: raw[c] - int(raw[c]), reverse=True)
    i = 0
    guard = 0
    while rem > 0 and any(alloc[c] < avail[c] for c in cells) and guard < 10 * len(cells) + 10:
        c = order[i % len(order)]
        if alloc[c] < avail[c]:
            alloc[c] += 1
            rem -= 1
        i += 1
        guard += 1

    # Guarantee-min pass: give every non-empty cell at least min(floor_min, avail),
    # borrowing from the largest-allocated cells that are above their own floor.
    for c in cells:
        want = min(floor_min, avail[c])
        if alloc[c] >= want:
            continue
        need = want - alloc[c]
        for d in sorted(cells, key=lambda x: alloc[x], reverse=True):
            if need <= 0:
                break
            if d == c:
                continue
            spare = alloc[d] - min(floor_min, avail[d])
            if spare > 0:
                take = min(spare, need)
                alloc[d] -= take
                alloc[c] += take
                need -= take
    return alloc


def select_cell(records: list[tuple[int, str]], k: int, rng: random.Random) -> list[int]:
    """Pick k record indices from a cell, preferring distinct genomes first."""
    if k >= len(records):
        return [idx for idx, _ in records]
    shuffled = records[:]
    rng.shuffle(shuffled)
    seen: set[str] = set()
    picked: list[int] = []
    leftover: list[int] = []
    for idx, g in shuffled:
        if len(picked) >= k:
            break
        if g not in seen:
            seen.add(g)
            picked.append(idx)
        else:
            leftover.append(idx)
    for idx in leftover:
        if len(picked) >= k:
            break
        picked.append(idx)
    return picked


def curate_split(
    path: Path,
    cap: int,
    bounds: list[int],
    args,
    rng: random.Random,
) -> tuple[set[int], dict]:
    """Return the set of selected record indices for a split, plus a report dict."""
    # ── Pass 1: metadata ──────────────────────────────────────────────────────
    # Per record: (class, phylum, lenbucket, genome, is_mibig). Index = line no.
    meta: list[tuple] = []
    n_total = 0
    n_quality_drop = 0
    natural_class = Counter()
    max_prefix_bytes = 0
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            n_total += 1
            if not passes_quality(rec, args.drop_ambiguous, args.drop_contig_edge, args.min_len):
                n_quality_drop += 1
                meta.append(None)  # keep index alignment, mark dropped
                continue
            cls = rec.get("compound_class", "UNKNOWN")
            natural_class[cls] += 1
            phy = phylum_token(rec.get("taxonomic_tag", "")) or "P__UNKNOWN"
            seq_len = len(rec.get("sequence", ""))
            lb = length_bucket(seq_len, bounds)
            meta.append((cls, phy, lb, genome_key(rec), is_mibig(rec), seq_len))
            # byte-level tokenizer => prefix token count == prefix byte length
            pfx = len(f"|COMPOUND_CLASS:{cls}|{rec.get('taxonomic_tag', '')}".encode())
            if pfx > max_prefix_bytes:
                max_prefix_bytes = pfx

    # ── Selection ─────────────────────────────────────────────────────────────
    # Group surviving records by class, then by (phylum, lenbucket) cell.
    by_class_cells: dict[str, dict[tuple, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    mibig_by_class: dict[str, list[int]] = defaultdict(list)
    for idx, m in enumerate(meta):
        if m is None:
            continue
        cls, phy, lb, gkey, mib, _slen = m
        if mib:
            mibig_by_class[cls].append(idx)
        by_class_cells[cls][(phy, lb)].append((idx, gkey))

    selected: set[int] = set()
    per_class_selected = Counter()
    for cls, cells in by_class_cells.items():
        # Force-keep all MIBiG gold records (train only; eval keeps them too —
        # they are already leakage-isolated by genome).
        forced = set(mibig_by_class.get(cls, [])) if args.keep_all_mibig else set()
        selected |= forced

        if cap <= 0:
            # No cap: keep everything that passed quality.
            for cell_recs in cells.values():
                selected.update(idx for idx, _ in cell_recs)
            per_class_selected[cls] = sum(len(v) for v in cells.values())
            continue

        remaining_budget = max(cap - len(forced), 0)
        # Exclude already-forced indices from cell candidate pools.
        avail = {}
        pools = {}
        for cell, recs in cells.items():
            pool = [(idx, g) for idx, g in recs if idx not in forced]
            if pool:
                pools[cell] = pool
                avail[cell] = len(pool)
        alloc = allocate_budget(avail, remaining_budget) if remaining_budget > 0 else {}
        for cell, k in alloc.items():
            for idx in select_cell(pools[cell], k, rng):
                selected.add(idx)
        per_class_selected[cls] = len({i for c in cells.values() for i, _ in c} & selected)

    # Estimate chunk windows for the selected records (training iterates windows,
    # not records). Uses the production chunk params; long BGCs expand >1 window.
    seq_budget = EST_MAX_SEQ_LEN - max_prefix_bytes - EST_EOS_RESERVE
    n_windows_est = 0
    for idx in selected:
        m = meta[idx]
        if m is not None:
            n_windows_est += _chunk_window_count(m[5], seq_budget, EST_CHUNK_OVERLAP)

    report = {
        "path": str(path),
        "n_total": n_total,
        "n_quality_drop": n_quality_drop,
        "n_selected": len(selected),
        "n_windows_est": n_windows_est,
        "max_prefix_bytes": max_prefix_bytes,
        "natural_class": natural_class,
        "selected_class": per_class_selected,
    }
    return selected, report


def write_selected(path: Path, selected: set[int], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx in selected:
                fout.write(line if line.endswith("\n") else line + "\n")
                written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_combined_grouped"),
                    help="Leakage-free grouped split dir (train/val/test.jsonl).")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_curated"))
    ap.add_argument("--train-cap", type=int, default=1000,
                    help="Max TRAIN records per compound_class (0 = keep all).")
    ap.add_argument("--eval-cap", type=int, default=0,
                    help="Max VAL/TEST records per class (0 = keep all; recommended).")
    ap.add_argument("--length-bounds", type=int, nargs="+",
                    default=[16000, 32768, 65536, 131072],
                    help="Length-bucket upper boundaries (nt) for diversity stratification.")
    ap.add_argument("--min-len", type=int, default=1000)
    ap.add_argument("--keep-ambiguous", dest="drop_ambiguous", action="store_false", default=True,
                    help="Keep sequences with N/ambiguous bases (default: drop).")
    ap.add_argument("--keep-contig-edge", dest="drop_contig_edge", action="store_false", default=True,
                    help="Keep contig_edge=True records (default: drop).")
    ap.add_argument("--no-keep-all-mibig", dest="keep_all_mibig", action="store_false", default=True,
                    help="Do not force-keep all MIBiG records in train.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report curated distribution without writing files.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    splits = {
        "train": (args.input_dir / "train.jsonl", args.train_cap),
        "val": (args.input_dir / "val.jsonl", args.eval_cap),
        "test": (args.input_dir / "test.jsonl", args.eval_cap),
    }
    for name, (p, _) in splits.items():
        if not p.exists():
            print(f"ERROR: {name} input not found: {p}", file=sys.stderr)
            sys.exit(2)

    reports = {}
    selections = {}
    for name, (p, cap) in splits.items():
        print(f"\nCurating {name} (cap={cap or 'none'})...", file=sys.stderr, flush=True)
        sel, rep = curate_split(p, cap, args.length_bounds, args, rng)
        selections[name] = sel
        reports[name] = rep
        print(f"  {rep['n_total']:,} -> {rep['n_selected']:,} "
              f"(dropped {rep['n_quality_drop']:,} on quality)", file=sys.stderr)

    # ── Report: per-class natural vs curated ──────────────────────────────────
    print("\n" + "=" * 78, file=sys.stderr)
    print("CURATION REPORT", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    tr = reports["train"]
    print(f"\nTRAIN per-class (natural -> curated, cap={args.train_cap}):", file=sys.stderr)
    print(f"  {'Class':<20} {'natural':>9} {'curated':>9}", file=sys.stderr)
    print("  " + "-" * 42, file=sys.stderr)
    for cls, nat in tr["natural_class"].most_common():
        print(f"  {cls:<20} {nat:>9,} {tr['selected_class'][cls]:>9,}", file=sys.stderr)
    print(f"\n  TRAIN total: {tr['n_total']:,} natural -> {tr['n_selected']:,} curated", file=sys.stderr)
    print(f"  VAL:   {reports['val']['n_total']:,} -> {reports['val']['n_selected']:,}", file=sys.stderr)
    print(f"  TEST:  {reports['test']['n_total']:,} -> {reports['test']['n_selected']:,}", file=sys.stderr)
    tr_records = reports["train"]["n_selected"]
    tr_windows = reports["train"].get("n_windows_est", tr_records)
    est_steps = tr_windows / 128
    print(f"\n  TRAIN chunk windows (est.): {tr_windows:,}  "
          f"({tr_windows / max(tr_records, 1):.2f}x records — long BGCs tile into "
          f"several windows)", file=sys.stderr)
    print(f"  Est. optimizer steps/epoch (ga=128): {est_steps:,.0f}  "
          f"[records/128 alone would understate this as {tr_records / 128:,.0f}]",
          file=sys.stderr)

    if args.dry_run:
        print("\n[dry-run] No files written.", file=sys.stderr)
        return

    print("\nWriting curated splits...", file=sys.stderr, flush=True)
    for name, (p, _) in splits.items():
        out = args.output_dir / f"{name}.jsonl"
        n = write_selected(p, selections[name], out)
        print(f"  wrote {n:,} -> {out}", file=sys.stderr)

    # Persist the curation manifest for provenance.
    manifest = {
        "input_dir": str(args.input_dir),
        "seed": args.seed,
        "train_cap": args.train_cap,
        "eval_cap": args.eval_cap,
        "length_bounds": args.length_bounds,
        "min_len": args.min_len,
        "drop_ambiguous": args.drop_ambiguous,
        "drop_contig_edge": args.drop_contig_edge,
        "keep_all_mibig": args.keep_all_mibig,
        "counts": {k: {"total": reports[k]["n_total"], "selected": reports[k]["n_selected"]}
                   for k in reports},
    }
    (args.output_dir / "curation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Output dir: {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
