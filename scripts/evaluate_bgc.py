#!/usr/bin/env python3
"""Run the eval suite (named CHECKS → QUESTIONS) on BGC sequences.

Accepts input as JSONL (from mibig_to_jsonl.py) or FASTA. Runs every check whose
dependencies are installed; gracefully skips the rest. See evaluation.py for the
CHECKS / QUESTIONS the suite reports.

Usage examples:
    # Fast local run (skip the DB-bound / slow checks)
    python scripts/evaluate_bgc.py \\
        --jsonl data/processed/splits/test.jsonl \\
        --max-sequences 3 \\
        --skip-checks antismash protein_homology kmer_novelty

    # Full evaluation on a generated sequence
    python scripts/evaluate_bgc.py \\
        --fasta generated.fasta \\
        --expected-class PKS \\
        --pfam-hmm data/pfam/Pfam-A.hmm

    # Positive vs negative control comparison
    python scripts/evaluate_bgc.py \\
        --jsonl data/processed/splits/test.jsonl \\
        --max-sequences 5 \\
        --include-negative-control \\
        --skip-checks protein_homology kmer_novelty
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgc_pipeline.evaluation import EvalConfig, evaluate_bgc


def load_from_jsonl(path: Path, max_records: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(rows) >= max_records:
                break
            rows.append(json.loads(line))
    return rows


def load_from_fasta(path: Path, max_records: int, expected_class: str) -> list[dict]:
    from Bio import SeqIO

    rows: list[dict] = []
    for record in SeqIO.parse(str(path), "fasta"):
        if len(rows) >= max_records:
            break
        rows.append({
            "accession": record.id,
            "sequence": str(record.seq),
            "compound_class": expected_class,
        })
    return rows


def shuffle_sequence(seq: str, rng: random.Random) -> str:
    chars = list(seq.upper())
    rng.shuffle(chars)
    return "".join(chars)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path, help="JSONL from mibig_to_jsonl.py")
    src.add_argument("--fasta", type=Path, help="FASTA of BGC nucleotide sequences")
    parser.add_argument("--expected-class", default="", help="COMPOUND_CLASS for FASTA input")
    parser.add_argument("--max-sequences", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pfam-hmm",
        type=Path,
        default=ROOT / "data" / "pfam" / "Pfam-A.hmm",
        help="Path to Pfam-A.hmm",
    )
    parser.add_argument(
        "--mibig-gbk-dir",
        type=Path,
        default=ROOT / "data" / "mibig" / "mibig_gbk_4.0",
    )
    parser.add_argument("--mmseqs2-db", type=str, default=None, help="Path to MMseqs2 UniRef50 DB")
    parser.add_argument("--antismash-db", type=str, default="/data2/ds85/antismash_db",
                        help="antiSMASH prerequisite DB root. A getattr() fallback stood here for a\n                             flag this parser never defined, so the value was unreachable config.")
    parser.add_argument(
        "--skip-checks",
        nargs="*",
        default=[],
        help="Check NAMES to skip (e.g. antismash protein_homology kmer_novelty).",
    )
    parser.add_argument(
        "--include-negative-control",
        action="store_true",
        help="Also evaluate a shuffled version of each sequence as negative control",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    args = parser.parse_args()

    # Load sequences
    if args.jsonl:
        records = load_from_jsonl(args.jsonl, args.max_sequences)
    else:
        records = load_from_fasta(args.fasta, args.max_sequences, args.expected_class)

    if not records:
        print("No sequences loaded.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Evaluating {len(records)} sequences (skip checks: {args.skip_checks or 'none'})",
        file=sys.stderr,
    )

    # Build class map from compound_class_map.yaml for antiSMASH mapping
    class_map = None
    class_map_path = ROOT / "config" / "compound_class_map.yaml"
    if class_map_path.exists():
        sys.path.insert(0, str(ROOT / "src"))
        from bgc_pipeline.class_map import load_class_map
        mapping, _ = load_class_map(class_map_path)
        class_map = mapping

    # A path that was SUPPLIED but does not resolve is a typo, not an omission. Silently
    # coercing it to None made "I passed --pfam-hmm and got zeros" byte-identical to
    # "I never passed it".
    for _label, _val, _ok in (("--pfam-hmm", args.pfam_hmm, args.pfam_hmm.exists()),
                              ("--mibig-gbk-dir", args.mibig_gbk_dir, args.mibig_gbk_dir.is_dir())):
        if _val and not _ok:
            print(f"WARNING: {_label}={_val} does not resolve; that check will DEGRADE",
                  file=sys.stderr)


    config = EvalConfig(
        pfam_hmm_path=args.pfam_hmm if args.pfam_hmm.exists() else None,
        mibig_gbk_dir=args.mibig_gbk_dir if args.mibig_gbk_dir.is_dir() else None,
        mmseqs2_db=args.mmseqs2_db,
        class_map=class_map,
        antismash_db_dir=args.antismash_db,
        skip_checks=args.skip_checks or [],
    )

    rng = random.Random(args.seed)
    all_results: list[dict] = []

    for i, rec in enumerate(records):
        acc = rec.get("accession", f"seq_{i}")
        seq = rec.get("sequence", "")
        cls = rec.get("compound_class", args.expected_class)
        print(f"  [{i+1}/{len(records)}] {acc} ({len(seq):,} bp, class={cls})", file=sys.stderr)

        res = evaluate_bgc(seq, acc, cls, config)
        res["control"] = "positive"
        all_results.append(res)

        if args.include_negative_control:
            shuf_seq = shuffle_sequence(seq, rng)
            neg_res = evaluate_bgc(shuf_seq, f"{acc}_shuffled", cls, config)
            neg_res["control"] = "negative_shuffled"
            all_results.append(neg_res)

    # Output
    output = json.dumps({"evaluations": all_results}, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"\nResults written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Print summary table (one column per QUESTION)
    from bgc_pipeline.evaluation import QUESTIONS
    qkeys = list(QUESTIONS.keys())
    glyph = {"PASS": "✓", "FAIL": "✗", "skipped": "-", "no_verdict": "?"}
    abbr = {"is_bgc": "bgc", "correct_class": "cls", "novel": "nov",
            "proteins_plausible": "prot", "complete": "cmpl"}
    print("\n" + "=" * 90, file=sys.stderr)
    print(f"{'Accession':<20} {'Control':<12} "
          + " ".join(f"{abbr.get(q, q):>5}" for q in qkeys), file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    for r in all_results:
        acc = r["accession"][:19]
        ctrl = r.get("control", "?")[:11]
        cells = [f"{glyph.get(r.get('questions', {}).get(q, '—'), '—'):>5}" for q in qkeys]
        print(f"{acc:<20} {ctrl:<12} {' '.join(cells)}", file=sys.stderr)
    print("=" * 90, file=sys.stderr)


if __name__ == "__main__":
    main()
