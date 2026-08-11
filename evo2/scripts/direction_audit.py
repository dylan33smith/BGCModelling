#!/usr/bin/env python
"""WAS THE STEERING EDIT TOO SMALL, OR WAS IT IGNORED? A GPU-free separation.

THE CONFOUND. Every steering null in this project is compatible with two very different
explanations, and nothing we have run so far distinguishes them:

  (i)  THE EDIT NEVER LANDED. The direction was wrong, or the dose too small, so the class
       representation barely moved. The nulls would then be an artifact of our dosing, and
       steering would deserve another attempt with a better recipe.
  (ii) THE EDIT LANDED AND WAS IGNORED. The representation moved exactly as intended, and the
       model simply does not read that subspace when producing tokens. Steering-at-a-layer is
       then dead for a structural reason, and the answer is depth of injection (Tier B).

We concluded (ii) and wrote "do NOT run another steering variant" into progress.md. That
conclusion has never been tested directly. This script tests it, and can overturn it.

THE TEST. Everything needed lives in the cached activations -- no GPU, no generation. Take REAL
held-out activations of NON-target class, add the SAME direction at the SAME doses the
experiments used, and ask the linear probe what class it now sees. This is the linear
precondition for steering: if the edit cannot move a linear readout sitting in the very same
activation space, it certainly cannot move the model's output 11 blocks downstream.

  * Probe says TARGET at the doses we used  -> the edit landed. Explanation (ii). Our conclusion
    stands, and the depth hypothesis is the right next move.
  * Probe does NOT move                     -> the edit never landed even linearly. Explanation
    (i). The steering nulls are partly a dosing artifact and the "do not run another steering
    variant" decision must be REOPENED.

Note what this does and does not show. A pass is a *precondition*, not a proof: moving the
probe's readout is necessary for steering to work and nowhere near sufficient, because the probe
is a linear function of one layer while the model is not. A failure, though, is decisive in the
other direction -- an edit that cannot move a linear readout in its own layer is not a candidate
explanation for anything downstream.

CONTROLS, because a dose-response curve that rises for any large vector shows nothing:
  * RANDOM direction at identical doses -- the "did we just add a big vector" control. A class
    direction that beats it is doing something specific; one that does not is only adding norm.
  * ABLATION of the true class, on target-class rows -- deletion is the one operation we know
    works through the model. If the linear readout reproduces the delete-works/install-fails
    asymmetry, that asymmetry is a property of the GEOMETRY, not of the model's depth.

The probe is refit here on a TRAIN SPLIT and evaluated on a HELD-OUT split. The cached probe was
fit on all 10,022 rows, so scoring those same rows would be in-sample: the probe would be
unusually certain of each row's true class, resistance to flipping would be overstated, and a
failure to flip could be an artifact of the fit rather than of the geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# The doses the generation experiments actually ran, in class-units. Phase 3 steered at L16 at
# ~1 class-unit; the L27 ladder ran 2.8 / 5.7 / 11.4 (run_steer_l27.sh header). Anything the
# curve reaches only far above 11.4 is a dose we never gave the model.
DEFAULT_DOSES = (0.0, 0.5, 1.0, 2.0, 2.8, 4.0, 5.7, 8.0, 11.4, 16.0, 24.0)
EXPERIMENT_DOSES = (1.0, 2.8, 11.4)


def _load_layer(acts_npz: Path, layer: int):
    z = np.load(acts_npz)
    y = np.asarray(z["y"])
    Xf = z[f"L{layer}"]
    X = (Xf[:, -1, :] if Xf.ndim == 3 else Xf).astype(np.float64)
    return X, y


def _fit_split_probe(X, y, holdout_frac: float, seed: int):
    """Fit on a train split, return (pipe, holdout_idx, balanced accuracy on holdout)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx = np.arange(len(y))
    tr, ho = train_test_split(idx, test_size=holdout_frac, random_state=seed, stratify=y)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial"))
    pipe.fit(X[tr], y[tr])
    acc = float(balanced_accuracy_score(y[ho], pipe.predict(X[ho])))
    return pipe, ho, acc


def _raw_space_logit_direction(pipe, class_index: int) -> np.ndarray:
    """Direction in RAW activation space that most increases this class's logit.

    The probe sees standardised features, so its coefficient vector lives in scaled space. The
    steering edit is applied to raw activations. Comparing the two without dividing by the
    scaler's sigma compares vectors in different spaces and the angle is meaningless.
    """
    scaler = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    w_scaled = lr.coef_[class_index] - lr.coef_.mean(axis=0)   # contrast vs the other classes,
    return w_scaled / scaler.scale_                            # matching mu_c - mean(mu_others)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _dose_curve(pipe, H: np.ndarray, direction: np.ndarray, unit: float,
                doses, class_index: int):
    """Mean P(target) and argmax-flip rate over held-out NON-target rows, per dose."""
    out = []
    for a in doses:
        P = pipe.predict_proba(H + (a * unit) * direction)
        out.append({"dose": float(a),
                    "p_target_mean": float(P[:, class_index].mean()),
                    "p_target_median": float(np.median(P[:, class_index])),
                    "flip_rate": float((P.argmax(axis=1) == class_index).mean())})
    return out


def _crossing(curve, key: str, thresh: float):
    """Lowest dose at which the curve reaches `thresh` (None = never, within the scan)."""
    for pt in curve:
        if pt[key] >= thresh:
            return pt["dose"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/"
                                 "acts_v2_train500.npz"))
    # trainonly.steerdirs.npz carries all NINE layers (10..27) with class-units, and is the file
    # the multi-layer stack experiment steered with. train500.steerdirs.npz has only L16/L20 --
    # pointing at it silently produced an empty table for every other layer.
    ap.add_argument("--dirs-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/"
                                 "trainonly.steerdirs.npz"))
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[10, 12, 14, 16, 18, 20, 22, 24, 27])
    ap.add_argument("--classes", nargs="+",
                    default=["NRPS", "PKS", "PKS_NRPS_HYBRID", "TERPENE", "RIPP"])
    ap.add_argument("--doses", type=float, nargs="+", default=list(DEFAULT_DOSES))
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--n-eval", type=int, default=600,
                    help="Held-out non-target rows per class (subsampled for speed).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/direction_audit.json"))
    args = ap.parse_args()

    prov_p = args.acts_npz.with_name(args.acts_npz.stem + ".provenance.json")
    if not prov_p.exists():
        raise SystemExit(f"[dir] ABORT: no provenance sidecar at {prov_p}")
    prov = json.loads(prov_p.read_text())
    if prov.get("split") != "train":
        raise SystemExit(f"[dir] ABORT: acts are split={prov.get('split')!r}, want train-only.")
    print(f"[dir] acts={args.acts_npz.name} split={prov['split']} n={prov['n']}")

    dz = np.load(args.dirs_npz)
    rng = np.random.default_rng(args.seed)
    report = {"acts": str(args.acts_npz), "dirs": str(args.dirs_npz),
              "doses": args.doses, "layers": {}}

    for L in args.layers:
        print(f"\n[dir] === layer {L} ===", flush=True)
        X, y = _load_layer(args.acts_npz, L)
        print(f"[dir] X={X.shape}; fitting probe on "
              f"{1 - args.holdout_frac:.0%} / evaluating on {args.holdout_frac:.0%} ...",
              flush=True)
        pipe, ho, acc = _fit_split_probe(X, y, args.holdout_frac, args.seed)
        classes = list(pipe.classes_)
        chance = 1.0 / len(classes)
        print(f"[dir] holdout balanced acc {acc:.3f} (chance {chance:.3f})")
        if acc < 2 * chance:
            raise SystemExit(f"[dir] ABORT: probe at {acc:.3f} is near chance — it cannot serve "
                             f"as a readout and any null from it would be vacuous.")

        lay = {"holdout_balanced_acc": acc, "chance": chance, "classes": {}}
        for c in args.classes:
            dkey, ukey = f"L{L}_{c}", f"classunit_L{L}_{c}"
            if dkey not in dz or ukey not in dz:
                print(f"[dir]   {c}: no direction at L{L} — skipped")
                continue
            if c not in classes:
                print(f"[dir]   {c}: not a probe class — skipped")
                continue
            u = np.asarray(dz[dkey], dtype=np.float64)
            u = u / np.linalg.norm(u)
            unit = float(dz[ukey])
            ci = classes.index(c)

            w_raw = _raw_space_logit_direction(pipe, ci)
            cos_wu = _cos(u, w_raw)

            # Held-out rows that are NOT the target class: the population steering has to convert.
            pool = ho[y[ho] != c]
            if len(pool) > args.n_eval:
                pool = rng.choice(pool, args.n_eval, replace=False)
            H = X[pool]

            r_unit = rng.normal(size=u.shape)
            r_unit /= np.linalg.norm(r_unit)

            curve = _dose_curve(pipe, H, u, unit, args.doses, ci)
            curve_rand = _dose_curve(pipe, H, r_unit, unit, args.doses, ci)

            # Ablation, on TARGET-class held-out rows: remove the component along u and ask how
            # much of the class readout survives. This is the operation we know works downstream.
            tgt = ho[y[ho] == c]
            abl = None
            if len(tgt) >= 10:
                Ht = X[tgt]
                P0 = pipe.predict_proba(Ht)[:, ci]
                proj = (Ht @ u)[:, None] * u[None, :]
                P1 = pipe.predict_proba(Ht - proj)[:, ci]
                abl = {"n": int(len(tgt)),
                       "p_true_before": float(P0.mean()), "p_true_after": float(P1.mean()),
                       "kept_rate_before": float((pipe.predict_proba(Ht).argmax(1) == ci).mean()),
                       "kept_rate_after": float(
                           (pipe.predict_proba(Ht - proj).argmax(1) == ci).mean())}

            lay["classes"][c] = {"cos_dir_vs_probe": cos_wu,
                                 "angle_deg": float(np.degrees(np.arccos(abs(cos_wu)))),
                                 "class_unit": unit, "n_eval": int(len(pool)),
                                 "curve": curve, "curve_random": curve_rand, "ablation": abl}
            print(f"[dir]   {c:>16}  cos(dir, probe)={cos_wu:+.3f} "
                  f"({np.degrees(np.arccos(abs(cos_wu))):.1f} deg)  n={len(pool)}", flush=True)
        # FAIL LOUD ON AN EMPTY LAYER. The first run of this script pointed at a directions file
        # holding only L16/L20, so layer 27 skipped every class and then printed a full set of
        # empty tables ending in "0/0 classes landed" -- a missing input rendered as a result.
        # That is the exact failure mode BGC_EVAL_STRICT exists to prevent in the eval suite.
        if not lay["classes"]:
            raise SystemExit(
                f"[dir] ABORT: layer {L} produced no usable classes. {args.dirs_npz.name} has no "
                f"'L{L}_<CLASS>' / 'classunit_L{L}_<CLASS>' entries for {args.classes}. A layer "
                f"with no directions is a missing input, not a null result.")
        report["layers"][str(L)] = lay
        del X

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1))

    # ---------------------------------------------------------------- report
    for L, lay in report["layers"].items():
        print("\n" + "=" * 96)
        print(f"LAYER {L} — does the steering edit move the class readout IN ITS OWN LAYER?")
        print(f"probe holdout balanced acc {lay['holdout_balanced_acc']:.3f} "
              f"(chance {lay['chance']:.3f})")
        print("=" * 96)
        print("\n--- mean P(target) on held-out NON-target activations, by dose (class-units) ---")
        dh = " ".join(f"{d:>7.1f}" for d in report["doses"])
        print(f"{'class':>17} {'arm':>7} {dh}")
        for c, r in lay["classes"].items():
            print(f"{c:>17} {'real':>7} "
                  + " ".join(f"{p['p_target_mean']:>7.3f}" for p in r["curve"]))
            print(f"{'':>17} {'random':>7} "
                  + " ".join(f"{p['p_target_mean']:>7.3f}" for p in r["curve_random"]))
        print("\n--- argmax flip rate to target (the probe would now CALL it the target) ---")
        print(f"{'class':>17} {'arm':>7} {dh}")
        for c, r in lay["classes"].items():
            print(f"{c:>17} {'real':>7} "
                  + " ".join(f"{p['flip_rate']:>7.3f}" for p in r["curve"]))
            print(f"{'':>17} {'random':>7} "
                  + " ".join(f"{p['flip_rate']:>7.3f}" for p in r["curve_random"]))

        print(f"\n--- verdict per class (EXPERIMENT doses were {EXPERIMENT_DOSES} class-units) ---")
        print(f"{'class':>17} {'angle':>7} {'flip@1':>8} {'flip@2.8':>9} {'flip@11.4':>10} "
              f"{'dose->0.5':>10} {'verdict':>12}")
        for c, r in lay["classes"].items():
            at = {p["dose"]: p for p in r["curve"]}
            f1, f28, f114 = (at.get(d, {}).get("flip_rate") for d in EXPERIMENT_DOSES)
            cross = _crossing(r["curve"], "flip_rate", 0.5)
            landed = cross is not None and cross <= max(EXPERIMENT_DOSES)
            print(f"{c:>17} {r['angle_deg']:>6.1f}d "
                  + " ".join(f"{(v if v is not None else float('nan')):>8.3f}"
                             for v in (f1, f28, f114))
                  + f" {(f'{cross:.1f}' if cross is not None else '>scan'):>10} "
                  + f"{'LANDED' if landed else 'DID NOT LAND':>12}")

        print("\n--- ablation control (target-class rows, remove the component along the dir) ---")
        print(f"{'class':>17} {'n':>4} {'P(true) before':>15} {'after':>8} "
              f"{'kept before':>12} {'after':>8}")
        for c, r in lay["classes"].items():
            a = r.get("ablation")
            if not a:
                continue
            print(f"{c:>17} {a['n']:>4} {a['p_true_before']:>15.3f} {a['p_true_after']:>8.3f} "
                  f"{a['kept_rate_before']:>12.3f} {a['kept_rate_after']:>8.3f}")

        landed = [c for c, r in lay["classes"].items()
                  if (_crossing(r["curve"], "flip_rate", 0.5) or 1e9) <= max(EXPERIMENT_DOSES)]
        print(f"\nHOW TO READ THIS. {len(landed)}/{len(lay['classes'])} classes reach a 50% flip "
              f"rate at or below the largest dose we actually gave the model.")
        print("  * If MOST landed: the edit did move the class representation, and the generation "
              "null is downstream — steering-at-a-layer is closed for the right reason and the "
              "depth hypothesis (Tier B) is the next move.")
        print("  * If MOST did not: the edit never landed even in its own layer. The nulls are "
              "confounded with dose and the 'no more steering variants' decision REOPENS.")
        print("  * Either way, compare 'real' against 'random' at every dose: a real direction "
              "that only matches the random one is adding norm, not class.")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
