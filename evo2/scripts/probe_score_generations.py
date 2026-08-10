#!/usr/bin/env python
"""A CONTINUOUS class readout for generated DNA — does steering move it AT ALL?

WHY THIS EXISTS. Every steering phase has been scored with binary gates: did a class-defining
Pfam domain appear (`check_class_markers`), or did antiSMASH call a cluster of that class. Both
are threshold instruments, and both are insensitive at the length we generate:

  * antiSMASH detects a BGC in only ~0.33 of seeded 3 kb continuations, so a steering arm's
    CEILING is 0.33, not 1.0;
  * the Pfam any-of proxy fires on the seed's OWN class in only 5-6 of 12 continuations.

Against instruments like that, an intervention could shift the model substantially toward the
target class and still score exactly 0.000 -- because "somewhat more PKS-like" is not a domain.
A null from a binary gate therefore bounds a LARGE effect and says nothing about a small one.

WHAT THIS DOES INSTEAD. The class linear probe reads compound class off mean-pooled hidden
states at 0.91 balanced accuracy (chance 0.09). Here it is applied to the FINISHED generated
DNA -- re-embedded with NO hook installed -- which turns "is it the target class" into a
probability rather than a yes/no. If steering nudges generation toward the target at all, this
sees it long before a domain gate does.

NOT CIRCULAR, but read the caveat. The steering vector is derived from the same activation
space the probe reads, so the obvious worry is that we are measuring our own injection. We are
not: the hook is removed before scoring, and what gets embedded is the generated NUCLEOTIDE
SEQUENCE, which the model must have actually written differently for the score to move. The real
caveat is different -- probe and directions are both fit on val+test, the standing leakage debt.

THE COMPARISON THAT COUNTS is still real-direction vs SHUFFLED-LABEL direction, paired on the
same seed exemplar. A probability shift under any perturbation is not evidence; a shift the
shuffled arm does not reproduce is.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402


@torch.no_grad()
def _embed(wrapper, seqs: list[str], layers: list[int], device: str, max_nt: int):
    """Mean-pool each block's output over ALL positions — the identical recipe
    `class_probe_sweep._embed_all_layers` used to build the cache the probe is fit on.
    Any other pooling would put the generations in a different space from the training data."""
    model, tok = wrapper.model, wrapper.tokenizer
    mods = dict(model.named_modules())
    cap: dict[int, np.ndarray] = {}

    def mk(i):
        def hook(_m, _in, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            cap[i] = h.detach().float()[0].mean(dim=0).cpu().numpy()
        return hook

    handles = [mods[f"blocks.{i}"].register_forward_hook(mk(i)) for i in layers]
    X = {i: [] for i in layers}
    try:
        for k, s in enumerate(seqs):
            ids = tok.tokenize(s[:max_nt])
            if not torch.is_tensor(ids):
                ids = torch.LongTensor(list(ids))
            cap.clear()
            _ = model(ids.to(device).view(1, -1))
            for i in layers:
                X[i].append(cap[i])
            if (k + 1) % 25 == 0:
                print(f"[probe] embedded {k + 1}/{len(seqs)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return {i: np.stack(v) for i, v in X.items()}


def _fit_probe(acts_npz: Path, layer: int, seed: int):
    """Multinomial logistic on real cores. Returns (predict_proba, classes, held-out accuracy).

    The accuracy is reported because a probe that does not work is not a readout: if it cannot
    separate real cores it cannot detect a shift in generated ones, and the whole measurement
    would be vacuous in a way that looks like a null.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    z = np.load(acts_npz)
    y = z["y"]
    Xf = z[f"L{layer}"]
    X = (Xf[:, -1, :] if Xf.ndim == 3 else Xf).astype(np.float64)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = float(cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy").mean())
    pipe.fit(X, y)
    chance = 1.0 / len(set(y.tolist()))
    print(f"[probe] L{layer} probe: balanced acc {acc:.3f} (chance {chance:.3f}) on n={len(y)}")
    if acc < 2 * chance:
        raise SystemExit(f"[probe] ABORT: probe accuracy {acc:.3f} is near chance {chance:.3f} — "
                         f"it cannot detect a shift in generations either, and a null from it "
                         f"would be vacuous rather than informative.")
    return pipe, list(pipe.classes_), acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path, nargs="+", help="Generated .jsonl file(s); one arm each.")
    ap.add_argument("--adapter", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768"
                                 "/checkpoints/step_1200"))
    ap.add_argument("--acts-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_valtest_fit.npz"))
    ap.add_argument("--layer", type=int, default=16, help="Layer the probe reads (peak: 16).")
    ap.add_argument("--baseline-arm", default=None,
                    help="Arm stem to treat as the unsteered reference for the PAIRED delta "
                         "(joined on seed_accession). Without it only absolute scores print.")
    ap.add_argument("--max-nt", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/steer_probe_score.json"))
    ap.add_argument("--emit-sidecar", type=Path, default=None,
                    help="Also write {record id -> {class: probability}} for the eval suite's "
                         "`class_probe` check (scripts/eval_suite_driver.py --probe-scores). "
                         "Keyed by accession/id, matching the driver's own id precedence.")
    args = ap.parse_args()

    recs: list[dict] = []
    for p in args.jsonl:
        arm = p.stem
        for i, line in enumerate(p.open()):
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("sequence", "") or ""
            if len(s) < 200:
                continue
            recs.append({"arm": arm, "i": i, "seq": s,
                         # Same id precedence as eval_suite_driver's `sid`
                         # (accession or id or str(i)) so the sidecar joins without
                         # a silent miss -- a mismatched key would make every record
                         # skip the check while looking like it ran.
                         "sid": r.get("accession") or r.get("id") or str(i),
                         "seed_class": r.get("seed_class"),
                         "target": r.get("steer_target_class"),
                         "seed_acc": r.get("seed_accession"),
                         "dose_cu": r.get("steer_realized_class_units"),
                         "dose_frac": r.get("steer_realized_norm_frac")})
    if not recs:
        raise SystemExit("no scoreable records")
    print(f"[probe] {len(recs)} generations over {len(args.jsonl)} arms")

    pipe, classes, acc = _fit_probe(args.acts_npz, args.layer, args.seed)

    # Cache the embeddings: they are the only GPU cost here, and re-analysis (a different
    # pairing, a different statistic) should not require another model load.
    cache = args.out_json.with_name(args.out_json.stem + f"_emb_L{args.layer}.npz")
    ident = np.array([f"{r['arm']}#{r['i']}" for r in recs])
    X = None
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if z["ident"].shape == ident.shape and bool((z["ident"] == ident).all()):
            X = z["X"]
            print(f"[probe] reusing cached embeddings from {cache.name}")
    if X is None:
        wrapper = load_evo2_wrapper_for_inference(Path(args.adapter) if args.adapter else None,
                                                  device=args.device)
        X = _embed(wrapper, [r["seq"] for r in recs], [args.layer], args.device,
                   args.max_nt)[args.layer]
        np.savez_compressed(cache, X=X, ident=ident)
        print(f"[probe] cached embeddings -> {cache.name}")
    P = pipe.predict_proba(X)
    ci = {c: k for k, c in enumerate(classes)}
    for r, row in zip(recs, P):
        # Store the FULL distribution. Storing only p_target made the unsteered arm unusable as a
        # baseline: it has no steer_target_class by construction, so its p_target was None and
        # every pair dropped -- and `argmax == target` compared against None, printing a
        # fabricated 0.000 that looked like a real measurement of "unsteered never hits target".
        r["probs"] = {c: float(row[k]) for k, c in enumerate(classes)}
        r["p_target"] = float(row[ci[r["target"]]]) if r.get("target") in ci else None
        r["p_seed"] = float(row[ci[r["seed_class"]]]) if r.get("seed_class") in ci else None
        r["argmax"] = classes[int(row.argmax())]
        r.pop("seq")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(recs, indent=1))
    if args.emit_sidecar:
        side = {r["sid"]: r["probs"] for r in recs if r.get("sid")}
        if len(side) != len(recs):
            print(f"  WARNING: {len(recs) - len(side)} records had no accession/id and are "
                  f"absent from the sidecar — those will SKIP the class_probe check")
        args.emit_sidecar.parent.mkdir(parents=True, exist_ok=True)
        args.emit_sidecar.write_text(json.dumps(side, indent=1))
        print(f"[probe] sidecar ({len(side)} records) -> {args.emit_sidecar}")

    print("\n" + "=" * 96)
    print(f"CONTINUOUS CLASS READOUT — probe at L{args.layer} (balanced acc {acc:.3f}), applied to")
    print("the finished generated DNA with no hook installed.")
    print("=" * 96)
    print(f"{'arm':>24} {'n':>4} {'dose(cu)':>9} {'P(target)':>11} {'P(seed)':>10} "
          f"{'argmax==target':>15} {'argmax==seed':>13}")
    by = collections.defaultdict(list)
    for r in recs:
        by[r["arm"]].append(r)
    for arm in sorted(by):
        rs = [r for r in by[arm] if r["p_target"] is not None]
        if not rs:
            rs = by[arm]
        cu = [r["dose_cu"] for r in rs if r.get("dose_cu")]
        pt = [r["p_target"] for r in rs if r["p_target"] is not None]
        ps = [r["p_seed"] for r in rs if r["p_seed"] is not None]
        # An arm with no target of its own (the unsteered control) must print "--", not 0.000:
        # comparing argmax to None is always False, which reads as a measured zero.
        has_t = [r for r in rs if r.get("target")]
        amt = (f"{sum(1 for r in has_t if r['argmax'] == r['target']) / len(has_t):>15.3f}"
               if has_t else f"{'--':>15}")
        print(f"{arm:>24} {len(rs):>4} {st.mean(cu) if cu else float('nan'):>9.2f} "
              f"{st.mean(pt) if pt else float('nan'):>11.4f} "
              f"{st.mean(ps) if ps else float('nan'):>10.4f} {amt} "
              f"{sum(1 for r in rs if r['argmax'] == r.get('seed_class')) / len(rs):>13.3f}")

    from math import comb

    def _paired(label: str, ref_arm: str):
        """Delta in P(target) against `ref_arm`, joined on the seed exemplar.

        The target class always comes from the STEERED record and is applied to BOTH sides. The
        reference arm may legitimately have no target of its own (the unsteered arm never does),
        and requiring the two to match dropped every pair.
        """
        if ref_arm not in by:
            return
        base = {r["seed_acc"]: r for r in by[ref_arm] if r.get("seed_acc")}
        if not base:
            print(f"\n{label}: reference arm '{ref_arm}' carries no seed_accession — cannot pair")
            return
        print(f"\n{label} (reference = {ref_arm}, joined on seed_accession; the steered record's "
              f"target class is scored on both sides):")
        print(f"{'arm':>24} {'pairs':>6} {'mean dP(target)':>16} {'sd':>9} {'t':>7} "
              f"{'n up':>7} {'sign p':>8}")
        for arm in sorted(by):
            if arm == ref_arm:
                continue
            d = []
            for r in by[arm]:
                b = base.get(r.get("seed_acc"))
                tgt = r.get("target")
                if b is None or tgt is None or tgt not in r["probs"] or tgt not in b["probs"]:
                    continue
                d.append(r["probs"][tgt] - b["probs"][tgt])
            if len(d) < 3:
                print(f"{arm:>24} {len(d):>6}   (too few matched pairs)")
                continue
            m, sd = st.mean(d), st.pstdev(d)
            t = m / (sd / len(d) ** 0.5) if sd > 0 else float("nan")
            up = sum(1 for x in d if x > 0)
            # two-sided sign test: no distributional assumption, robust to the heavy tails a
            # bounded probability produces
            p = min(1.0, 2 * sum(comb(len(d), k) * 0.5 ** len(d)
                                 for k in range(max(up, len(d) - up), len(d) + 1)))
            print(f"{arm:>24} {len(d):>6} {m:>+16.5f} {sd:>9.5f} {t:>7.2f} "
                  f"{up:>3}/{len(d):<3} {p:>8.4f}")

    if args.baseline_arm:
        _paired("PAIRED vs UNSTEERED — does steering move it at all?", args.baseline_arm)
    # THE COMPARISON THAT COUNTS. Both arms are perturbed by the same amount; only one carries
    # class information. Anything the shuffled arm reproduces is not a class effect.
    for a, b in (("A_real", "B_shuffled"),):
        if a in by and b in by:
            _paired("PAIRED vs SHUFFLED-LABEL — is any of it CLASS-specific?", b)
            break
    print("\nREAD: a binary gate can only see an effect big enough to build a domain. This sees")
    print("any shift at all. If the paired dP(target) is ~0 here too, the null is not an artefact")
    print("of insensitive scoring — the model's output genuinely does not move toward the target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
