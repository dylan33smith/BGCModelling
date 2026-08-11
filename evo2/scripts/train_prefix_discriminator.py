#!/usr/bin/env python
"""A LENGTH-ROBUST class discriminator, for guiding generation while it is still partial.

WHY A NEW ONE. The existing `class_probe` is fit on FULL-sequence mean-pooled activations and
calibrated at 3 kb. Guided decoding has to score a generation while it is still 500 or 1000 nt
long, and the probe has never been asked to do that -- its accuracy in that regime is simply
unmeasured. FUDGE's central design point is exactly this: train the discriminator on PARTIAL
prefixes, because that is what it will actually see at decode time. Reusing a
trained-on-complete-sequences scorer as a live per-step guide is the documented way this family
of methods goes wrong.

The activation cache already stores mean-pooled windows at 500 / 1000 / 2000 / full nt, so this
costs no GPU time: we fit one probe over ALL prefix lengths pooled, which makes it length-robust
rather than length-specialised, and then report accuracy PER LENGTH so the guidance signal's
reliability is known at every point in a generation rather than assumed.

THE LEAK THIS GUARDS AGAINST. A record contributes four rows (its 500 / 1000 / 2000 / full
windows) that are near-duplicates of each other -- the 500 nt window is a strict prefix of the
full one. Ordinary k-fold would put a core's 500 nt window in train and its full window in test,
so the model would be scored on sequences it had already seen most of. Cross-validation here is
therefore GROUPED BY RECORD (GroupKFold), so every window of a given core lands in the same fold.
Measured below: the ungrouped estimate is optimistic by a wide margin, which is exactly the kind
of quietly-inflated number this project has been burned by before.

TRAIN-ONLY, enforced. The fit set must carry a `.provenance.json` declaring split == train --
the same guard `probe_score_generations._fit_probe` uses, for the same reason: a discriminator
fit on val/test and then used to steer generations evaluated against val/test has seen its own
evaluation data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_v2_train500.npz"),
                    help="TRAIN-ONLY activation cache with prefix windows.")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--out", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/prefix_discriminator_L16.joblib"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--also-report-ungrouped", action="store_true", default=True,
                    help="Also compute the (wrong) ungrouped CV number, to show the gap.")
    args = ap.parse_args()

    prov_p = args.acts_npz.with_name(args.acts_npz.stem + ".provenance.json")
    if not prov_p.exists():
        raise SystemExit(f"[disc] ABORT: no provenance sidecar at {prov_p}")
    prov = json.loads(prov_p.read_text())
    if not prov.get("fit_safe_for_val_test_evaluation"):
        raise SystemExit(f"[disc] ABORT: {args.acts_npz.name} is split={prov.get('split')!r}. "
                         f"A discriminator fit on val/test and used to guide generations that are "
                         f"then evaluated on val/test has seen its own evaluation data.")

    z = np.load(args.acts_npz)
    y_rec = z["y"]
    plabels = [str(p) for p in z["prefix_labels"]]
    Xf = z[f"L{args.layer}"]                       # (N, P, D)
    n_rec, n_pref, dim = Xf.shape
    print(f"[disc] {args.acts_npz.name}  split={prov.get('split')}  "
          f"{n_rec} records x {n_pref} prefix windows {plabels} x {dim}d")

    # Flatten to one row per (record, prefix window), carrying the record index as the CV group.
    X = Xf.reshape(n_rec * n_pref, dim).astype(np.float64)
    y = np.repeat(y_rec, n_pref)
    groups = np.repeat(np.arange(n_rec), n_pref)
    plen = np.tile(np.array(plabels), n_rec)
    print(f"[disc] fitting on {len(X)} rows ({n_pref} lengths pooled), "
          f"{len(set(y.tolist()))} classes")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold, KFold

    def _pipe():
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0))

    # ---- GROUPED CV: every window of a core stays in one fold ----
    from sklearn.metrics import balanced_accuracy_score
    gkf = GroupKFold(n_splits=args.folds)
    oof = np.empty(len(y), dtype=object)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        p = _pipe().fit(X[tr], y[tr])
        oof[te] = p.predict(X[te])
        print(f"[disc]   grouped fold {k + 1}/{args.folds} done", flush=True)
    grouped_acc = balanced_accuracy_score(y, oof)
    chance = 1.0 / len(set(y.tolist()))
    print(f"\n[disc] GROUPED balanced accuracy: {grouped_acc:.4f}  (chance {chance:.4f})")

    per_len = {}
    for L in plabels:
        m = plen == L
        per_len[L] = float(balanced_accuracy_score(y[m], oof[m]))
    print("[disc] per prefix length (this is the number that matters for guidance):")
    for L in plabels:
        print(f"[disc]     {L:>6} nt : {per_len[L]:.4f}")

    ungrouped_acc = None
    if args.also_report_ungrouped:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        oof2 = np.empty(len(y), dtype=object)
        for tr, te in kf.split(X):
            oof2[te] = _pipe().fit(X[tr], y[tr]).predict(X[te])
        ungrouped_acc = float(balanced_accuracy_score(y, oof2))
        print(f"\n[disc] UNGROUPED (WRONG) balanced accuracy: {ungrouped_acc:.4f}")
        print(f"[disc]   inflation from letting a core's own prefix windows straddle folds: "
              f"{ungrouped_acc - grouped_acc:+.4f}")
        print(f"[disc]   -> report the GROUPED number; the ungrouped one is scored on cores the "
              f"model already saw most of.")

    final = _pipe().fit(X, y)
    if grouped_acc < 2 * chance:
        raise SystemExit(f"[disc] ABORT: grouped accuracy {grouped_acc:.4f} is near chance "
                         f"{chance:.4f} -- this cannot guide anything, and a null result from "
                         f"guidance with it would be vacuous rather than informative.")

    import joblib
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": final, "classes": list(final.classes_), "layer": args.layer,
                 "grouped_balanced_acc": float(grouped_acc),
                 "ungrouped_balanced_acc": ungrouped_acc,
                 "per_prefix_len_acc": per_len, "prefix_labels": plabels,
                 "chance": float(chance), "n_rows": int(len(X)), "n_records": int(n_rec),
                 "acts": str(args.acts_npz), "provenance": prov}, args.out)
    print(f"\n[disc] wrote {args.out}")
    print(f"[disc] USE: this is the GUIDE. It must never also be the EVALUATOR -- selecting on a "
          f"score and then reporting that score is circular by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
