#!/usr/bin/env python
"""PHASE 0 — build CORRECTED class-steering directions. Pure CPU, no model, no GPU.

WHY THIS EXISTS. The shipped directions in `acts_*.dirs.npz` are `mu_class - mu_global`, which
at layer 16 is ~collinear with PC1 of the activations. PC1 holds 98.07% of the centered variance
and correlates -0.9996 with ||h||, i.e. it IS the sequence-length axis. Consequences, all
measured: the 11 "class directions" are rank-~1 (99.22% of Frobenius energy in one singular
value, mean pairwise |cos| 0.934); as a classifier the recipe scores 0.186 where a logistic probe
scores 0.881; and because the sign is set by whether a class's cores are longer or shorter than
average, the vector POINTS BACKWARDS for key pairs (NRPS->PKS AUC 0.221, ECTOINE->TERPENE 0.070).
Every 0/30 correct_class result was structurally guaranteed. See docs/steering_program.md.

WHAT THIS BUILDS. For each layer and class:

    v_c = mu_c - mean_{k != c} mu_k          # one direction PER CLASS ("make this a PKS"),
                                             # balanced against the other classes rather than
                                             # against a global mean the big classes dominate
    v_c <- v_c - (v_c . pc1) pc1             # remove the length component
    v_c <- v_c / ||v_c||                     # unit length; dose is applied separately

Measured: stripping is the load-bearing step. Same pairs, AUC before -> after:
NRPS->PKS 0.214 -> 0.985, ECTOINE->TERPENE 0.070 -> 0.999, PKS->RIPP 0.566 -> 0.951.

DOSE UNITS. Raw ||delta|| is uninterpretable and not comparable across classes. We persist
`classunit_L{L}_{C}` = the projected distance from the other-class mean to class C's mean along
v_c. A dose of "1 class-unit" therefore means "move by exactly the class-mean separation",
comparable across classes and layers. `sigma_L{L}_{C}` (spread of real data along the axis) is
also persisted so a dose can be quoted in sigma.

CONTROLS. `perm{k}_L{L}_{C}` are built by the IDENTICAL recipe on SHUFFLED class labels. They
carry no class information by construction and are the null every downstream phase compares
against — without them "steering changed something" cannot be separated from "perturbing the
residual stream changes something".

Writes a NEW file; never touches the legacy `acts_*.dirs.npz` (352 keys, still referenced by
older scripts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import collections

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _one_vs_rest_auc(X: np.ndarray, y: np.ndarray, cls: str, v: np.ndarray) -> float:
    """How well does projecting onto v separate `cls` from every other class?

    Rank-based AUC: P(a random cls sample projects higher than a random non-cls sample).
    0.5 = coin flip, <0.5 = the direction points the WRONG WAY.
    """
    s = X @ v
    a, b = s[y == cls], s[y != cls]
    if not len(a) or not len(b):
        return float("nan")
    # rank-based form: O(n log n) instead of the O(n^2) outer comparison
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    ra = ranks[: len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def _length_match(X: np.ndarray, y: np.ndarray, classes: np.ndarray, n_bins: int, rng) -> np.ndarray:
    """Return indices sampled so every class has the SAME length distribution.

    WHY. Stripping PC1 removes the *global* length axis but leaves *class-conditional* length
    structure. Measured 2026-07-29: PKS cores are 6282 nt median in train vs 1158 nt in val
    (5.4x), and the two independently-fit PKS directions agree at only cos 0.469 — while TERPENE
    (1.12x mismatch) agrees at 0.943. Across the panel, corr(length ratio, direction agreement)
    = **-0.923**. A direction fit on unmatched data partly encodes "how long is this class here",
    which does not transfer.

    Length is not stored in the activation cache, but ||h|| is a near-perfect proxy for it
    (PC1 correlates -0.9996 with ||h||, and PC1 is the length axis), so we can match on ||h||
    using only the cache. Bin by ||h|| quantile, then take an equal number per class per bin.
    """
    nrm = np.linalg.norm(X, axis=1)
    edges = np.quantile(nrm, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    keep: list[int] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        inbin = {c: np.where((y == c) & (nrm >= lo) & (nrm < hi))[0] for c in classes}
        present = [c for c in classes if len(inbin[c])]
        if len(present) < 2:
            continue                       # a bin only one class occupies carries no contrast
        k = min(len(inbin[c]) for c in present)
        for c in present:
            idx = inbin[c].copy()
            rng.shuffle(idx)
            keep.extend(idx[:k].tolist())
    return np.array(sorted(keep), dtype=int)


def _dirs_for(X: np.ndarray, y: np.ndarray, classes: np.ndarray, pc1: np.ndarray | None):
    """v_c = mu_c - mean(other class means), optionally length-stripped, unit-normalized.

    Averaging the OTHER CLASS MEANS (not the global mean) keeps the contrast balanced: a global
    mean is dominated by whichever classes have the most records, which biases every direction
    toward the majority classes.
    """
    mus = {c: X[y == c].mean(0) for c in classes}
    out = {}
    for c in classes:
        others = np.mean([mus[k] for k in classes if k != c], axis=0)
        v = mus[c] - others
        if pc1 is not None:
            v = v - np.dot(v, pc1) * pc1
        out[c] = _unit(v)
    return out, mus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_v2.npz"))
    ap.add_argument("--out-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_v2.dirs_v2.npz"))
    ap.add_argument("--report", type=Path, default=None, help="JSON report (default: alongside --out-npz)")
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24, 27],
                    help="Layer 16 is where class is most READABLE, but readability collapses "
                         "after ~23 (0.906 -> 0.354 by the output), so the layer where class is "
                         "still USABLE may be later. Layer is a variable, not an assumption.")
    ap.add_argument("--prefix-index", type=int, default=-1,
                    help="Which pooled prefix window to build from (-1 = full sequence).")
    ap.add_argument("--n-perm", type=int, default=10, help="Shuffled-label control directions per layer.")
    ap.add_argument("--no-strip", action="store_true", help="Diagnostic: skip length-stripping.")
    ap.add_argument("--length-match", action="store_true",
                    help="Resample so every class shares one length distribution (matched on "
                         "||h||, a -0.9996 proxy for length). Removes CLASS-CONDITIONAL length "
                         "structure, which PC1-stripping does not.")
    ap.add_argument("--match-bins", type=int, default=8)
    ap.add_argument("--min-auc", type=float, default=0.90)
    ap.add_argument("--min-splithalf", type=float, default=0.70)
    ap.add_argument("--max-pc1-leak", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # GUARD (2026-07-29): class_probe_sweep.py auto-writes its OWN legacy sidecar at
    # <cache-stem>.dirs.npz when it finishes. Writing here silently loses this build — it
    # happened: a 576-array build was overwritten by a 704-key legacy file (32 layers x 22
    # classes of un-stripped mu_c - mu_global), and the next run started with no shuffled-label
    # controls and every class-unit defaulted to 1.0.
    if args.out_npz.name.endswith(".dirs.npz"):
        sys.exit(f"REFUSING to write {args.out_npz}\n"
                 f"  '<stem>.dirs.npz' is the path class_probe_sweep.py auto-generates for its "
                 f"legacy sidecar; it will clobber this build.\n"
                 f"  Use e.g. {args.out_npz.with_name(args.out_npz.name.replace('.dirs.npz', '.steerdirs.npz'))}")

    z = np.load(args.acts_npz)
    y = z["y"]
    classes = np.array(sorted(np.unique(y).tolist()))
    prefix_labels = [str(p) for p in z["prefix_labels"]] if "prefix_labels" in z else []
    rng = np.random.default_rng(args.seed)

    print(f"acts     : {args.acts_npz}")
    print(f"classes  : {len(classes)}  n={len(y)}  prefix window={prefix_labels[args.prefix_index] if prefix_labels else args.prefix_index}")
    print(f"layers   : {args.layers}")
    print(f"stripping: {'OFF (diagnostic)' if args.no_strip else 'ON (remove length component)'}\n")

    payload: dict[str, np.ndarray] = {}
    report: dict = {"acts": str(args.acts_npz), "classes": classes.tolist(), "n": int(len(y)),
                    "prefix_index": args.prefix_index, "stripped": not args.no_strip, "layers": {}}

    for L in args.layers:
        key = f"L{L}"
        if key not in z:
            print(f"  !! {key} not in acts; skipping")
            continue
        Xf = z[key]
        X = (Xf[:, args.prefix_index, :] if Xf.ndim == 3 else Xf).astype(np.float64)
        y_L = y
        if args.length_match:
            sel = _length_match(X, y, classes, args.match_bins, rng)
            kept = collections.Counter(y[sel].tolist())
            print(f"L{L}: length-matched {len(sel)}/{len(y)} records  "
                  + ", ".join(f"{c}={kept.get(c, 0)}" for c in classes[:6]) + " ...")
            X, y_L = X[sel], y[sel]
        gm = X.mean(0)
        Xc = X - gm

        # --- identify the dominant axis, then CHECK what it is (we do not assume it is length) ---
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pc1 = Vt[0]
        var_share = float((S[0] ** 2) / (S ** 2).sum())
        norm_corr = float(np.corrcoef(Xc @ pc1, np.linalg.norm(X, axis=1))[0, 1])

        strip = None if args.no_strip else pc1
        dirs, mus = _dirs_for(X, y_L, classes, strip)

        # --- split-half stability: rebuild on two disjoint halves, compare ---
        idx = rng.permutation(len(y_L))
        h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
        d1, _ = _dirs_for(X[h1], y_L[h1], classes, strip)
        d2, _ = _dirs_for(X[h2], y_L[h2], classes, strip)

        # --- window stability: does the direction survive being built from a shorter prefix? ---
        wdirs = None
        if Xf.ndim == 3 and Xf.shape[1] > 1 and args.prefix_index in (-1, Xf.shape[1] - 1):
            Xw = Xf[:, 1, :].astype(np.float64)          # the 1000-nt window
            pcw = None
            if not args.no_strip:
                Uw, Sw, Vtw = np.linalg.svd(Xw - Xw.mean(0), full_matrices=False)
                pcw = Vtw[0]
            wdirs, _ = _dirs_for(Xw, y_L, classes, pcw)

        # HELD-OUT AUC. Scoring a direction on the same records it was built from inflates both
        # the real and the shuffled-label numbers (4096 dims, 991 records — a difference-of-means
        # partly fits noise). Build on one half, score on the other, average both ways. Measured
        # consequence: the shuffled-label 95th percentile drops from ~0.85 in-sample to ~chance,
        # which is what makes the admissibility gate mean anything.
        def heldout_auc(labels: np.ndarray) -> dict:
            out = {}
            da, _ = _dirs_for(X[h1], labels[h1], classes, strip)
            db, _ = _dirs_for(X[h2], labels[h2], classes, strip)
            for c in classes:
                a = _one_vs_rest_auc(X[h2], y_L[h2], c, da[c])   # built on h1, scored on h2
                b = _one_vs_rest_auc(X[h1], y_L[h1], c, db[c])   # built on h2, scored on h1
                out[c] = float(np.nanmean([a, b]))
            return out

        ho = heldout_auc(y_L)

        per_class = {}
        for c in classes:
            v = dirs[c]
            others = np.mean([mus[k] for k in classes if k != c], axis=0)
            class_unit = float(np.dot(mus[c] - others, v))     # dose scale, in raw activation units
            sigma = float((X @ v).std())                        # spread of real data along the axis
            rec = {
                "auc_heldout": round(ho[c], 4),
                "auc_in_sample": round(_one_vs_rest_auc(X, y_L, c, v), 4),
                "splithalf_cos": round(float(abs(np.dot(d1[c], d2[c]))), 4),
                "pc1_leak_abs_cos": round(float(abs(np.dot(v, pc1))), 4),
                "class_unit": round(class_unit, 4),
                "sigma_along_axis": round(sigma, 4),
                "n": int((y_L == c).sum()),
            }
            if wdirs is not None:
                rec["window_cos_1000_vs_full"] = round(float(abs(np.dot(v, wdirs[c]))), 4)
            rec["PASS"] = bool(
                rec["auc_heldout"] >= args.min_auc
                and rec["splithalf_cos"] >= args.min_splithalf
                and rec["pc1_leak_abs_cos"] <= args.max_pc1_leak
            )
            per_class[str(c)] = rec
            payload[f"L{L}_{c}"] = v.astype(np.float32)
            payload[f"classunit_L{L}_{c}"] = np.float32(class_unit)
            payload[f"sigma_L{L}_{c}"] = np.float32(sigma)

        # --- shuffled-label controls, identical recipe ---
        for k in range(args.n_perm):
            yp = rng.permutation(y_L)
            pdirs, _ = _dirs_for(X, yp, classes, strip)
            for c in classes:
                payload[f"perm{k}_L{L}_{c}"] = pdirs[c].astype(np.float32)
        # a permuted direction's AUC is the null band for the gate itself
        perm_aucs = []
        for k in range(args.n_perm):
            hp = heldout_auc(rng.permutation(y_L))
            perm_aucs.extend(hp[c] for c in classes)

        payload[f"pc1_L{L}"] = pc1.astype(np.float32)
        payload[f"globalmean_L{L}"] = gm.astype(np.float32)

        npass = sum(r["PASS"] for r in per_class.values())
        report["layers"][str(L)] = {
            "pc1_variance_share": round(var_share, 4),
            "pc1_corr_with_activation_norm": round(norm_corr, 4),
            "perm_auc_mean": round(float(np.mean(perm_aucs)), 4),
            "perm_auc_p95": round(float(np.percentile(perm_aucs, 95)), 4),
            "n_pass": npass, "n_classes": len(classes),
            "per_class": per_class,
        }
        print(f"L{L}: PC1={100*var_share:.2f}% of variance, corr with ||h|| = {norm_corr:+.4f} "
              f"{'<- length axis' if abs(norm_corr) > 0.9 else ''}")
        print(f"     shuffled-label control AUC: mean {np.mean(perm_aucs):.3f}, 95th pct {np.percentile(perm_aucs, 95):.3f}")
        print(f"     {npass}/{len(classes)} classes pass the admissibility gate")
        worst = sorted(per_class.items(), key=lambda kv: kv[1]["auc_heldout"])[:3]
        for c, r in worst:
            print(f"       lowest AUC  {c:>16}: {r['auc_heldout']:.3f} (in-sample {r['auc_in_sample']:.3f}) "
                  f"(split-half {r['splithalf_cos']:.2f}, PC1 leak {r['pc1_leak_abs_cos']:.3f})"
                  + ("" if r["PASS"] else "   <-- FAILS GATE"))
        print()

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **payload)
    rp = args.report or args.out_npz.with_suffix(".report.json")
    rp.write_text(json.dumps(report, indent=2))
    print(f"wrote {len(payload)} arrays -> {args.out_npz}")
    print(f"wrote report            -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
