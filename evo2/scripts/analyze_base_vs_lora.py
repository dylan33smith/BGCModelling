#!/usr/bin/env python
"""DID THE FINE-TUNE CHANGE WHAT THE MODEL GENERATES, OR ONLY WHAT IT SCORES?

THE QUESTION. Measured 2026-08-12: de novo output from the fine-tuned model hits real Pfam families
100% of the time at 6 kb -- phage integrases, MFS transporters, ankyrin repeats -- and biosynthetic
families almost never (biosynthetic_fraction 0.100 against 0.836 for real cores). The model writes
genuine protein of the wrong kind.

That raises a question nobody has asked in this project: **did the LoRA move the output distribution
into biosynthetic space at all?** A fine-tune can lower validation loss by sharpening likelihoods on
the training distribution while leaving what it actually GENERATES essentially unchanged. Every
downstream experiment -- steering, soft prefixes, guided decoding, the whole conditioning programme
-- assumed a generator that had been moved and merely needed steering.

THE TEST. Identical prompts, identical decoding, identical RNG seed; the only difference is whether
the adapter is loaded. Verified before running: the two arms' prompt plans are byte-identical apart
from the model line.

  * LoRA >> base on biosynthetic_fraction  -> the fine-tune did move generation; the remaining gap
    to 0.836 is how much further it has to go, and the objective work (B) is aimed correctly.
  * LoRA ~= base                           -> the fine-tune changed likelihoods without changing
    what is generated. That reframes the objective work before a training run is spent on it, and
    it would mean three months of conditioning experiments were run on a generator that had never
    left base-model behaviour.

Paired by prompt index, since both arms share the prompt list. A paired sign test is the primary
readout; means are reported alongside but the metric is skewed and n is small.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))


def sign_test(deltas):
    d = [x for x in deltas if x == x and x != 0]
    up, n = sum(1 for x in d if x > 0), len(d)
    if n == 0:
        return 0, 0, 1.0
    return up, n, min(1.0, 2 * sum(comb(n, k) * 0.5 ** n for k in range(max(up, n - up), n + 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/data2/ds85/bgcmodel_runs/base_vs_lora"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from biosynthetic_fraction import one  # same scorer, both scans on each sequence

    arms = {}
    for name in ("base", "lora"):
        p = args.root / f"{name}.jsonl"
        if not p.exists():
            raise SystemExit(f"[bvl] missing {p}")
        arms[name] = [json.loads(l) for l in p.open() if l.strip()]
    if len(arms["base"]) != len(arms["lora"]):
        print(f"[bvl] WARNING: unequal arms ({len(arms['base'])} vs {len(arms['lora'])}); "
              f"pairing on the shorter.")

    from concurrent.futures import ProcessPoolExecutor
    jobs, keys = [], []
    for name, recs in arms.items():
        for i, r in enumerate(recs):
            jobs.append((name, r["sequence"], r.get("compound_class") or "NRPS", 10 ** 9))
            keys.append((name, i, r.get("compound_class")))
    print(f"[bvl] scoring {len(jobs)} sequences (both Pfam scans each)", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(one, jobs))
    for (name, i, cls), r in zip(keys, res):
        r["arm"], r["i"], r["cls"] = name, i, cls

    b = {r["i"]: r for r in res if r["arm"] == "base"}
    l = {r["i"]: r for r in res if r["arm"] == "lora"}
    shared = sorted(set(b) & set(l))

    print(f"\n{'arm':>6} {'n':>4} {'best ANY':>9} {'best BIO':>9} {'BIO fraction':>13} "
          f"{'frac>0':>7} {'len':>7}")
    for name, d in (("base", b), ("lora", l)):
        v = [d[i] for i in shared]
        print(f"{name:>6} {len(v):>4} {st.mean(x['any'] for x in v):>9.1f} "
              f"{st.mean(x['bio'] for x in v):>9.1f} {st.mean(x['frac'] for x in v):>13.3f} "
              f"{sum(1 for x in v if x['frac'] > 0) / len(v):>7.2f} "
              f"{st.mean(len(arms[name][i]['sequence']) for i in shared):>7.0f}")

    print(f"\nPAIRED (same prompt, same seed, adapter is the only difference)")
    print(f"{'metric':>18} {'base':>9} {'lora':>9} {'delta':>9} {'up':>8} {'sign p':>8}")
    for key, lab in (("frac", "BIO fraction"), ("bio", "best BIO bits"),
                     ("cls_bio", "best CLASS bits"), ("any", "best ANY bits")):
        d = [l[i][key] - b[i][key] for i in shared]
        up, n, p = sign_test(d)
        print(f"{lab:>18} {st.mean(b[i][key] for i in shared):>9.3f} "
              f"{st.mean(l[i][key] for i in shared):>9.3f} {st.mean(d):>+9.3f} "
              f"{up:>3}/{n:<4} {p:>8.4f}")

    print(f"\nper class (BIO fraction)")
    print(f"{'class':>10} {'n':>4} {'base':>8} {'lora':>8} {'delta':>9}")
    for c in sorted({k[2] for k in keys if k[2]}):
        idx = [i for i in shared if b[i]["cls"] == c]
        if not idx:
            continue
        print(f"{c:>10} {len(idx):>4} {st.mean(b[i]['frac'] for i in idx):>8.3f} "
              f"{st.mean(l[i]['frac'] for i in idx):>8.3f} "
              f"{st.mean(l[i]['frac'] - b[i]['frac'] for i in idx):>+9.3f}")

    fb = st.mean(b[i]["frac"] for i in shared)
    fl = st.mean(l[i]["frac"] for i in shared)
    print("\nHOW TO READ THIS. Real held-out cores score 0.836 on this metric.")
    print(f"  base {fb:.3f}  ->  lora {fl:.3f}  |  real 0.836")
    print("  * A clear LoRA advantage means the fine-tune DID move generation toward biosynthetic")
    print("    space, and the objective work is aimed correctly at closing the rest of the gap.")
    print("  * base ~= lora means the fine-tune changed likelihoods without changing what is")
    print("    GENERATED — and the whole conditioning programme was run on a generator that had")
    print("    never left base-model behaviour. That must be known before spending a training run.")
    print("  * n is small and the metric is skewed: read the paired sign test, not the means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
