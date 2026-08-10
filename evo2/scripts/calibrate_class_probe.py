#!/usr/bin/env python
"""Calibrate the `class_probe` eval check at BOTH ends, before anyone reads a number off it.

This project has twice mistaken an instrument change for a model change, and the 2026-08-10 audit
found that every gate's false-positive rate had been ASSERTED rather than measured -- the old
Pfam is_bgc proxy turned out to fire on 96% of ordinary bacterial DNA. A new readout does not get
to skip that check.

TWO MEASUREMENTS, both required:

  TPR  -- on real HELD-OUT cores, truncated to the generation length. The ceiling: how often the
          probe's argmax is the true class on DNA that genuinely IS that class.
  FPR  -- on the NEGATIVE control: real non-BGC genomic windows cut from the same organisms
          outside every annotated antiSMASH region (scripts/make_negative_control.py).

The FPR measurement is the important one, and it is why `class_probe` is wired as a diagnostic
that can never gate. The probe is trained on BGC classes ONLY -- it has no "not a BGC" output, so
on non-BGC DNA it cannot abstain. It will return a distribution and it will have an argmax. The
question this answers is not "does it misfire" (it must) but "how CONFIDENT is it when it does",
because a probe that is as confident on non-BGC DNA as on real clusters carries no information
about validity at all, only about resemblance.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402
from probe_score_generations import _embed, _fit_probe  # noqa: E402


def _load(path: Path, n: int, trunc: int, rng) -> list[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    rng.shuffle(rows)
    out = []
    for r in rows:
        s = (r.get("sequence") or "")[:trunc] if trunc else (r.get("sequence") or "")
        if len(s) >= 500:
            out.append({"seq": s, "cls": r.get("compound_class")})
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768"
                                 "/checkpoints/step_1200"))
    ap.add_argument("--acts-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_v2_train500.npz"),
                    help="Train-only activations; provenance is enforced by _fit_probe.")
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--negative", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/steer_phase2/negative_control.jsonl"),
                    help="Real NON-BGC windows. REQUIRED: without them the probe's behaviour on "
                         "non-BGC DNA is asserted, not measured — the exact defect the 2026-08-10 "
                         "audit found across every other gate.")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n-pos", type=int, default=60)
    ap.add_argument("--trunc", type=int, default=3000, help="Generation length; 0 = full core.")
    ap.add_argument("--max-nt", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_calibration.json"))
    args = ap.parse_args()

    if not args.negative.exists():
        raise SystemExit(
            f"ABORT: no negative control at {args.negative}. Build one with\n"
            f"  python scripts/make_negative_control.py --gen <generations> --gbk-tar <genomes> "
            f"--out {args.negative}\n"
            f"Calibrating only on positives measures sensitivity and calls it accuracy.")

    rng = random.Random(args.seed)
    pos = _load(args.cores, args.n_pos, args.trunc, rng)
    neg = _load(args.negative, 10 ** 6, args.trunc, rng)
    print(f"[cal] positives: {len(pos)} real cores  |  negatives: {len(neg)} real non-BGC windows")
    if not pos or not neg:
        raise SystemExit("[cal] ABORT: one of the two sets is empty")

    pipe, classes, cv_acc = _fit_probe(args.acts_npz, args.layer, args.seed)
    wrapper = load_evo2_wrapper_for_inference(Path(args.adapter) if args.adapter else None,
                                              device=args.device)
    seqs = [r["seq"] for r in pos] + [r["seq"] for r in neg]
    X = _embed(wrapper, seqs, [args.layer], args.device, args.max_nt)[args.layer]
    P = pipe.predict_proba(X)
    ci = {c: k for k, c in enumerate(classes)}

    rows = []
    for i, (r, row) in enumerate(zip(pos + neg, P)):
        rows.append({"kind": "positive" if i < len(pos) else "negative",
                     "true": r["cls"], "argmax": classes[int(row.argmax())],
                     "argmax_p": float(row.max()),
                     "p_true": float(row[ci[r["cls"]]]) if r.get("cls") in ci else None})

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=1))

    print("\n" + "=" * 84)
    print(f"CLASS-PROBE CALIBRATION — layer {args.layer}, sequences truncated to "
          f"{args.trunc or 'full'} nt")
    print(f"probe cross-validated balanced accuracy on real cores: {cv_acc:.3f} "
          f"(chance {1/len(classes):.3f})")
    print("=" * 84)

    p_rows = [r for r in rows if r["kind"] == "positive"]
    n_rows = [r for r in rows if r["kind"] == "negative"]
    tpr = sum(1 for r in p_rows if r["argmax"] == r["true"]) / len(p_rows)
    print(f"\nTPR  argmax == true class on REAL cores : {tpr:.3f}  (n={len(p_rows)})")
    print(f"     mean confidence in the true class   : {st.mean(r['p_true'] for r in p_rows if r['p_true'] is not None):.3f}")
    byc = collections.defaultdict(list)
    for r in p_rows:
        byc[r["true"]].append(r["argmax"] == r["true"])
    for c in sorted(byc):
        print(f"       {c:>16}: {sum(byc[c])}/{len(byc[c])}")

    print(f"\nNON-BGC DNA (n={len(n_rows)}) — the probe has NO negative class, so it cannot")
    print( "abstain. What matters is how confident it is when it is necessarily wrong:")
    print(f"     mean argmax confidence : {st.mean(r['argmax_p'] for r in n_rows):.3f}")
    print(f"     median                 : {st.median(r['argmax_p'] for r in n_rows):.3f}")
    print(f"     >= 0.5 confident       : {sum(1 for r in n_rows if r['argmax_p'] >= 0.5)}/{len(n_rows)}")
    pos_conf = st.mean(r["argmax_p"] for r in p_rows)
    print(f"     (real cores, for scale : {pos_conf:.3f})")
    hits = collections.Counter(r["argmax"] for r in n_rows)
    print(f"     argmax lands on        : {dict(hits.most_common(5))}")
    if n_rows and st.mean(r["argmax_p"] for r in n_rows) >= 0.8 * pos_conf:
        print("\n  ==> The probe is nearly as confident on NON-BGC DNA as on real clusters.")
        print("      It measures RESEMBLANCE, not validity. This is why class_probe is a")
        print("      diagnostic that never gates, and why it is only trustworthy in PAIRED")
        print("      comparisons where this shared bias cancels.")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
