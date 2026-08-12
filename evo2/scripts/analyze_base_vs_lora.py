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

**THE CONFOUND THIS TEST HAS, AND THE GUARD FOR IT.** The LoRA was trained on 47,524 BGC cores;
base Evo2 was not. So "LoRA scores higher on biosynthetic metrics" has two very different readings:
it LEARNED to generate biosynthetic sequence, or it is REGURGITATING cores it memorised. Every
metric on the ladder rewards resemblance to the training set, so the two look identical on all of
them. Novelty is therefore scored here as a gate, not an afterthought: k-mer containment against the
reference corpus, FAIL at >= 0.95. A LoRA advantage carried by memorised sequences is not evidence
that the fine-tune moved generation, and without this the more interesting reading would be the one
reported.

Paired by prompt index, since both arms share the prompt list. A paired sign test is the primary
readout; means are reported alongside but the metric is skewed and n is small. PRIMARY METRIC is
`best_bio_bits` (absolute), not the ratio: the ladder audit measured AUROC 0.950 vs 0.893 for
predicting the independent antiSMASH outcome.
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
    print("  * = PRIMARY (ladder audit: AUROC 0.950 for the independent antiSMASH outcome)")
    print(f"{'metric':>18} {'base':>9} {'lora':>9} {'delta':>9} {'up':>8} {'sign p':>8}")
    for key, lab in (("bio", "best BIO bits *"), ("frac", "BIO fraction"),
                     ("cls_bio", "best CLASS bits"), ("any", "best ANY bits")):
        d = [l[i][key] - b[i][key] for i in shared]
        up, n, p = sign_test(d)
        print(f"{lab:>18} {st.mean(b[i][key] for i in shared):>9.3f} "
              f"{st.mean(l[i][key] for i in shared):>9.3f} {st.mean(d):>+9.3f} "
              f"{up:>3}/{n:<4} {p:>8.4f}")

    # ---- NOVELTY GATE -------------------------------------------------------------------
    # Runs on BOTH arms against the TRAINING corpus. A LoRA win carried by memorised sequence is
    # not evidence the fine-tune moved generation.
    print("\nNOVELTY GATE (k-mer containment vs the training corpus; FAIL at >= 0.95)")
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from memorization_check import scan_corpus
        ref = Path("/data2/ds85/bgcmodel_data/splits_core/train.jsonl")
        queries, qkeys = [], []
        for name, d in (("base", b), ("lora", l)):
            for i in shared:
                queries.append((f"{name}#{i}", arms[name][i]["sequence"]))
                qkeys.append((name, i))
        nov = scan_corpus(queries, ref)
        cont = {k: float(r.get("max_containment", float("nan"))) for k, r in zip(qkeys, nov)}
        print(f"{'arm':>6} {'mean containment':>17} {'max':>7} {'>=0.95 (memorised)':>20} "
              f"{'>=0.80 (warn)':>15}")
        for name in ("base", "lora"):
            v = [cont[(name, i)] for i in shared if cont[(name, i)] == cont[(name, i)]]
            if not v:
                continue
            print(f"{name:>6} {st.mean(v):>17.3f} {max(v):>7.3f} "
                  f"{sum(1 for x in v if x >= 0.95) / len(v):>20.2f} "
                  f"{sum(1 for x in v if x >= 0.80) / len(v):>15.2f}")
        memo = [i for i in shared if cont.get(("lora", i), 0) >= 0.95]
        if memo:
            print(f"  !! {len(memo)} LoRA sequences are MEMORISED — recomputing the headline "
                  f"metric with them excluded:")
            keep = [i for i in shared if i not in memo]
            if keep:
                d = [l[i]["bio"] - b[i]["bio"] for i in keep]
                up, n, pv = sign_test(d)
                print(f"     best_bio_bits, memorised excluded (n={len(keep)}): "
                      f"{st.mean(d):+.2f}, {up}/{n}, p={pv:.4f}")
        else:
            print("  no LoRA sequence reaches the memorisation threshold — a LoRA advantage, if "
                  "any, is not explained by copying.")
    except Exception as e:
        print(f"  SKIPPED: {e}")
        print("  NOTE: a skipped novelty gate means novelty is UNVERIFIED, never that it passed. "
              "Any LoRA advantage below is uninterpretable until this runs.")

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
