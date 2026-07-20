#!/usr/bin/env python
"""Build fast-probe training subsets for the 2026-07-03 conditioning diagnosis.

Two capability-probe datasets derived from splits_core/train.jsonl:

  subset_c_wholecore  — (c) CHUNKING probe: NRPS/PKS/PKS_NRPS_HYBRID cores whose
                        WHOLE sequence fits in one L window (no chunking, so the
                        entire assembly line is seen under the |COMPOUND_CLASS:|
                        start prefix). Tests whether removing fragmentation lets
                        ordered modules emerge.

  subset_d_megaup     — (d) IMBALANCE probe: megasynthase classes up-weighted to
                        ~parity with the aggregate of the other classes (each other
                        class down-sampled), so the model can't collapse to the
                        simple-class attractors (ectoine/terpene). Tests the
                        easy-attractor hypothesis.

Deterministic (fixed seed); writes to /data2 so the trainer + build_chunk_index
can consume them. Run afterwards:  python scripts/build_chunk_index.py <subset>.jsonl
"""
import argparse, json, random
from pathlib import Path

MEGA = {"NRPS", "PKS", "PKS_NRPS_HYBRID"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/train.jsonl"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/probe_subsets"))
    ap.add_argument("--l", type=int, default=16384,
                    help="Probe context length; whole-core cap = L - prefix - eos.")
    ap.add_argument("--max-prefix-tokens", type=int, default=199)
    ap.add_argument("--eos-reserve", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    budget = args.l - args.max_prefix_tokens - args.eos_reserve

    recs = [json.loads(l) for l in args.train.open()]
    by_class: dict[str, list] = {}
    for r in recs:
        by_class.setdefault(r.get("compound_class"), []).append(r)
    print(f"loaded {len(recs)} records, {len(by_class)} classes; whole-core budget={budget} nt")

    # ---- subset C: whole-core megasynthase cores ----------------------------
    c = [r for r in recs
         if r.get("compound_class") in MEGA and len(r.get("sequence", "")) <= budget]
    rng.shuffle(c)
    c_path = args.out_dir / "subset_c_wholecore.jsonl"
    c_path.write_text("".join(json.dumps(r) + "\n" for r in c))
    from collections import Counter
    cc = Counter(r["compound_class"] for r in c)
    print(f"[C] whole-core (<= {budget} nt) mega: {len(c)} records  {dict(cc)}")

    # ---- subset D: megasynthase up-weighted to ~parity ----------------------
    mega = [r for r in recs if r.get("compound_class") in MEGA]
    others = [cls for cls in by_class if cls not in MEGA]
    # down-sample each non-mega class so the non-mega total ~= mega total
    per_other = max(1, len(mega) // max(1, len(others)))
    d: list = list(mega)
    for cls in others:
        pool = by_class[cls]
        take = min(per_other, len(pool))
        d.extend(rng.sample(pool, take))
    rng.shuffle(d)
    d_path = args.out_dir / "subset_d_megaup.jsonl"
    d_path.write_text("".join(json.dumps(r) + "\n" for r in d))
    dc = Counter(r["compound_class"] for r in d)
    mega_frac = sum(dc[m] for m in MEGA) / len(d)
    print(f"[D] mega-up: {len(d)} records; mega fraction={mega_frac:.2f}; per-other cap={per_other}")
    print(f"    class spread: {dict(dc)}")

    print(f"\nwrote:\n  {c_path}\n  {d_path}")
    print("next: build sidecars with scripts/build_chunk_index.py on each subset.")


if __name__ == "__main__":
    main()
