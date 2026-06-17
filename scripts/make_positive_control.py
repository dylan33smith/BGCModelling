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

Completeness filtering (see docs/archive/AUDIT_FINDINGS.md / docs/archive/EVAL_RUNBOOK.md): the validity
checks antismash (region detection) and class_markers (class-marker recovery) are
*completeness* signals. A real but fragmentary BGC that is too short to contain its
class's full biosynthetic module would (correctly) FAIL those checks — making it a
poor positive-control exemplar. So we apply a class-aware minimum length: the modular
classes (NRPS/PKS/hybrids) need enough sequence to hold a complete C-A-T / KS-AT-ACP
module, so they draw only from full-length records. Classes whose obligate-domain
set is empty (RIPP/OTHER/…) are length-agnostic for M2 and use the general floor.
We filter by length (an independent proxy for completeness), NOT by passing the
metric we are calibrating — that would be circular.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

D = Path("/data2/ds85/bgcmodel_data/splits_curated")

# Class-aware completeness floor (nt). Only the multi-domain modular classes need a
# high floor; their obligate-domain module spans many kb, so a fragment can't hold it.
# Other classes fall back to the general --min-len. Pool is ample at these floors
# (test split has thousands of NRPS/PKS >= 20kb), so this costs no sample diversity.
CLASS_MIN_LEN: dict[str, int] = {
    "NRPS": 20000,             # C + A + T module (PF00668/PF00501/PF00550)
    "PKS": 20000,              # KS + AT + ACP module (PF00109/PF00698/PF00550)
    "PKS_NRPS_HYBRID": 25000,  # KS + C, and hybrids run large
    "SACCHARIDE": 12000,       # glycosyltransferase + sugar machinery
    "SIDEROPHORE": 12000,      # IucA_IucC + assembly
}


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
                    help="Records per compound class (length-spread within the complete pool).")
    ap.add_argument("--min-len", type=int, default=8000,
                    help="General length floor (raised per-class for modular classes; see CLASS_MIN_LEN).")
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

    # Per class, pick `per_class` records spread across the length range — but only
    # from the COMPLETE pool (>= class floor), so the control contains real BGCs we
    # genuinely expect to pass the completeness metrics.
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in mibig:
        by_class[r["compound_class"]].append(r)
    picks: list[dict] = []
    for cls in sorted(by_class):
        floor = max(args.min_len, CLASS_MIN_LEN.get(cls, 0))
        pool = sorted((r for r in by_class[cls] if len(r["sequence"]) >= floor),
                      key=lambda r: len(r["sequence"]))
        if not pool:
            print(f"  WARNING: class {cls}: no complete records >= {floor:,} nt "
                  f"(had {len(by_class[cls])} below floor); skipping")
            continue
        k = min(args.per_class, len(pool))
        # evenly-spaced indices across the length-sorted complete pool
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
    by_cls: dict[str, list[int]] = defaultdict(list)
    for r in picks:
        by_cls[r["compound_class"]].append(len(r["sequence"]))
    for cls in sorted(by_cls):
        ls = sorted(by_cls[cls])
        floor = max(args.min_len, CLASS_MIN_LEN.get(cls, 0))
        print(f"    {cls:<12} n={len(ls)} lengths={ls[0]:,}–{ls[-1]:,} (floor {floor:,})")
    print(f"  wrote {args.out_fasta}  and  {args.out_jsonl}")


if __name__ == "__main__":
    main()
