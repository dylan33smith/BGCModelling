#!/usr/bin/env python3
"""Build a real-BGC positive control for evaluation calibration.

Selects MiBIG records (the most curated BGCs) that landed in the TEST split — so
they are real, experimentally-curated clusters the model never saw in training or
validation. Running these through the eval pipeline (and the memorization check)
calibrates "what good looks like": real held-out BGCs should score highly on the
validity metrics and should NOT look memorized (high but not ~identical k-mer
similarity to training).

Verifies zero exact-sequence overlap with train/val before writing.
Output: a FASTA (for the eval suite / generation comparison) + a JSONL (full records).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

D = Path("/data2/ds85/bgcmodel_data/splits_curated")


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def fasta_record(acc: str, seq: str, **meta) -> str:
    tags = " ".join(f"{k}={v}" for k, v in meta.items())
    body = "\n".join(seq[i:i + 80] for i in range(0, len(seq), 80))
    return f">{acc} {tags}\n{body}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, default=D / "test.jsonl")
    ap.add_argument("--train", type=Path, default=D / "train.jsonl")
    ap.add_argument("--val", type=Path, default=D / "val.jsonl")
    ap.add_argument("--per-class", type=int, default=4,
                    help="Records per compound class (length-spread).")
    ap.add_argument("--min-len", type=int, default=4000)
    ap.add_argument("--max-len", type=int, default=80000)
    # eval/ is tracked (data/ is gitignored) so this fixed reference is version-controlled.
    ap.add_argument("--out-fasta", type=Path,
                    default=Path("eval/positive_control_mibig.fasta"))
    ap.add_argument("--out-jsonl", type=Path,
                    default=Path("eval/positive_control_mibig.jsonl"))
    args = ap.parse_args()

    test = load(args.test)
    mibig = [r for r in test
             if str(r.get("accession", "")).startswith("BGC")
             and args.min_len <= len(r.get("sequence", "")) <= args.max_len]

    # Disjointness guard: none of the picks may share an exact sequence with train/val.
    train_md5 = {md5(r["sequence"]) for r in load(args.train)}
    val_md5 = {md5(r["sequence"]) for r in load(args.val)}
    mibig = [r for r in mibig if md5(r["sequence"]) not in train_md5
             and md5(r["sequence"]) not in val_md5]

    # Per class, pick `per_class` records spread across the length range (deterministic).
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in mibig:
        by_class[r["compound_class"]].append(r)
    picks: list[dict] = []
    for cls in sorted(by_class):
        pool = sorted(by_class[cls], key=lambda r: len(r["sequence"]))
        k = min(args.per_class, len(pool))
        if k == 0:
            continue
        # evenly-spaced indices across the length-sorted pool
        idxs = [round(i * (len(pool) - 1) / max(k - 1, 1)) for i in range(k)]
        picks.extend(pool[i] for i in sorted(set(idxs)))

    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with args.out_fasta.open("w") as fa, args.out_jsonl.open("w") as jl:
        for r in picks:
            jl.write(json.dumps(r) + "\n")
            fa.write(fasta_record(
                r["accession"], r["sequence"],
                compound_class=r["compound_class"],
                length=len(r["sequence"]),
                taxon=r.get("taxonomic_tag", "").split(";")[1] if ";" in r.get("taxonomic_tag", "") else "",
            ))

    from collections import Counter
    print(f"Positive control: {len(picks)} real MiBIG BGCs from the TEST split "
          f"(0 exact-seq overlap with train/val).")
    print(f"  classes: {dict(Counter(r['compound_class'] for r in picks))}")
    print(f"  lengths: {min(len(r['sequence']) for r in picks):,}"
          f"–{max(len(r['sequence']) for r in picks):,} nt")
    print(f"  wrote {args.out_fasta}  and  {args.out_jsonl}")


if __name__ == "__main__":
    main()
