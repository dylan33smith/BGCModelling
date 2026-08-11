#!/usr/bin/env python
"""DID WE OVERDOSE? An offline pre-check for Affine Concept Editing (A2), and a dose post-mortem.

WHAT ACE ACTUALLY IS, once you write it down. Affine Concept Editing ablates the concept
component and RESETS it to a reference value, rather than zeroing it or adding a fixed amount:

    additive steering  h' = h + (alpha * class_unit) * u        <- fixed dose, same for every h
    ACE                h' = h - ((h.u) - m_c) * u               <- per-example, lands at m_c

Both are rank-1 edits along the SAME direction u. The only difference is the size, and ACE's size
is chosen per example so the coordinate lands exactly on the target class's mean coordinate m_c.

WHICH MAKES THE INTERESTING QUESTION A DOSE QUESTION. `class_unit` is defined as the projected
distance from the other-class mean to class c's mean, so a dose of alpha=1 already lands the
AVERAGE non-target activation on the target class mean. The steering experiments ran alpha =
2.8 / 5.7 / 11.4. Those doses do not push activations toward the target class; they push them
2 to 10 class-units PAST it, along an axis whose real spread is a fraction of a class-unit.

So this script does not ask "does ACE move the probe" -- the direction audit already showed any
sufficient move along u flips the readout, and ACE is a smaller move than the ones we ran. It
asks the question that decides whether ACE is worth GPU time, and which also serves as a
post-mortem on every dose we have already spent:

  1. WHERE DOES THE EDITED POINT LAND, in units of the target class's own spread along the axis?
     Real target activations sit at z ~ 0 +/- 1 by construction. If our doses land at z = +5 or
     +20, we were not steering toward the class, we were leaving the distribution -- which would
     give "a bigger dose buys damage, not class" an actual mechanism.
  2. IS THE EDITED POINT ON THE TARGET MANIFOLD? A rank-1 edit fixes one coordinate out of 4096
     and leaves the other 4095 as the SOURCE class's. The probe is linear and only reads its own
     direction, so it will happily say "NRPS" about a point no real NRPS activation resembles.
     Measured as k-NN distance to a bank of real target activations, in units of how far real
     held-out target activations sit from that same bank. A ratio near 1 is on-manifold; 3 is not.

PRE-REGISTERED READING:
  * ADD off-manifold at our doses AND ACE near 1.0  -> ACE is a genuinely different intervention,
    the overdose has a mechanism, and A2 earns its GPU time.
  * BOTH far off-manifold                           -> a rank-1 edit cannot land on the manifold
    at all. ACE is not the fix; skip A2 and spend the GPU on Tier B.
  * BOTH near 1.0                                   -> the off-manifold story is wrong, and ACE's
    advantage over plain additive steering is not on this axis. Weak case for A2.

Note the probe cannot answer either question -- it is linear, reads one direction, and was
already shown to saturate at 1-2 class-units. It is reported alongside only to confirm that every
arm here would have "passed" a probe-based check, which is the point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from direction_audit import _fit_split_probe, _load_layer  # noqa: E402

DEFAULT_DOSES = (1.0, 2.0, 2.8, 5.7, 11.4)


def knn_dist(bank: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Mean distance from each query row to its k nearest neighbours in `bank`."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k).fit(bank)
    d, _ = nn.kneighbors(query)
    return d.mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/"
                                 "acts_v2_train500.npz"))
    ap.add_argument("--dirs-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/"
                                 "trainonly.steerdirs.npz"))
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 27])
    ap.add_argument("--classes", nargs="+",
                    default=["NRPS", "PKS", "PKS_NRPS_HYBRID", "TERPENE", "RIPP"])
    ap.add_argument("--doses", type=float, nargs="+", default=list(DEFAULT_DOSES))
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/ace_precheck.json"))
    args = ap.parse_args()

    prov = json.loads(args.acts_npz.with_name(args.acts_npz.stem + ".provenance.json").read_text())
    if prov.get("split") != "train":
        raise SystemExit(f"[ace] ABORT: acts are split={prov.get('split')!r}, want train-only.")

    dz = np.load(args.dirs_npz)
    rng = np.random.default_rng(args.seed)
    report = {"doses": args.doses, "layers": {}}

    for L in args.layers:
        print(f"\n[ace] === layer {L} ===", flush=True)
        X, y = _load_layer(args.acts_npz, L)
        pipe, ho, acc = _fit_split_probe(X, y, args.holdout_frac, args.seed)
        classes = list(pipe.classes_)
        print(f"[ace] probe holdout balanced acc {acc:.3f} (chance {1 / len(classes):.3f})")
        tr_mask = np.ones(len(y), dtype=bool)
        tr_mask[ho] = False

        lay = {"holdout_balanced_acc": acc, "classes": {}}
        for c in args.classes:
            dkey, ukey = f"L{L}_{c}", f"classunit_L{L}_{c}"
            if dkey not in dz or ukey not in dz or c not in classes:
                print(f"[ace]   {c}: no direction / not a probe class — skipped")
                continue
            u = np.asarray(dz[dkey], dtype=np.float64)
            u /= np.linalg.norm(u)
            unit = float(dz[ukey])
            ci = classes.index(c)

            # The reference bank and the edit's target coordinate come from TRAIN rows only.
            # Building the edit out of held-out activations would let the evaluation set define
            # the thing being evaluated.
            bank = X[tr_mask & (y == c)]
            m_c = float((bank @ u).mean())
            s_c = float((bank @ u).std())

            tgt_ho = X[ho[y[ho] == c]]
            pool = ho[y[ho] != c]
            if len(pool) > args.n_eval:
                pool = rng.choice(pool, args.n_eval, replace=False)
            H = X[pool]

            ref = float(np.median(knn_dist(bank, tgt_ho, args.knn)))   # on-manifold yardstick

            def summarise(A, label):
                coord = A @ u
                return {"arm": label,
                        "z_along_axis": float(((coord - m_c) / s_c).mean()),
                        "p_target": float(pipe.predict_proba(A)[:, ci].mean()),
                        "knn_ratio": float(np.median(knn_dist(bank, A, args.knn)) / ref)}

            arms = [summarise(tgt_ho, "REAL target (held-out)"),
                    summarise(H, "unedited (source)")]
            for a in args.doses:
                arms.append(summarise(H + (a * unit) * u, f"add @{a:g} cu"))
            ace = H - ((H @ u) - m_c)[:, None] * u[None, :]
            arms.append(summarise(ace, "ACE (reset to m_c)"))

            lay["classes"][c] = {"class_unit": unit, "sigma_along_axis": s_c,
                                 "class_units_per_sd": unit / s_c,
                                 "n_bank": int(len(bank)), "n_eval": int(len(H)),
                                 "knn_ref_distance": ref, "arms": arms}
            print(f"[ace]   {c:>16}  1 class-unit = {unit / s_c:.2f} sd  bank={len(bank)}",
                  flush=True)

        if not lay["classes"]:
            raise SystemExit(f"[ace] ABORT: layer {L} produced no usable classes — a missing "
                             f"input, not a null result.")
        report["layers"][str(L)] = lay
        del X

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1))

    for L, lay in report["layers"].items():
        print("\n" + "=" * 100)
        print(f"LAYER {L} — where does each edit actually put the activation?")
        print("z = position along the class axis in units of the TARGET class's own spread, "
              "measured from its mean.")
        print("knn = distance to real target activations, / how far real target activations sit "
              "from each other. 1.0 = on-manifold.")
        print("=" * 100)
        for c, r in lay["classes"].items():
            print(f"\n{c}   (1 class-unit = {r['class_units_per_sd']:.2f} sd along the axis; "
                  f"bank n={r['n_bank']}, eval n={r['n_eval']})")
            print(f"{'arm':>24} {'z':>8} {'knn ratio':>11} {'P(target)':>11}")
            for a in r["arms"]:
                print(f"{a['arm']:>24} {a['z_along_axis']:>+8.1f} {a['knn_ratio']:>11.2f} "
                      f"{a['p_target']:>11.3f}")

        # Verdict, pooled over classes.
        def pooled(name):
            v = [a for c in lay["classes"].values() for a in c["arms"] if a["arm"] == name]
            return (float(np.mean([x["z_along_axis"] for x in v])),
                    float(np.mean([x["knn_ratio"] for x in v]))) if v else (float("nan"),) * 2
        print(f"\n{'POOLED over classes':>24} {'z':>8} {'knn ratio':>11}")
        for name in (["REAL target (held-out)", "unedited (source)"]
                     + [f"add @{a:g} cu" for a in report["doses"]] + ["ACE (reset to m_c)"]):
            z, k = pooled(name)
            print(f"{name:>24} {z:>+8.1f} {k:>11.2f}")

        z_ace, k_ace = pooled("ACE (reset to m_c)")
        z_run, k_run = pooled("add @2.8 cu")
        print("\nHOW TO READ THIS.")
        print(f"  * Real target activations sit at z~0, knn 1.00 by construction — that is the "
              f"target we are trying to hit.")
        print(f"  * The dose we ran (2.8 class-units) lands at z={z_run:+.1f}, knn {k_run:.2f}.")
        print(f"  * ACE lands at z={z_ace:+.1f}, knn {k_ace:.2f}.")
        print("  * If ACE's knn ratio is far above 1.0 too, a rank-1 edit cannot reach the "
              "manifold at all: it corrects one coordinate and leaves 4095 belonging to the "
              "source class. A2 would then not be worth GPU time.")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
