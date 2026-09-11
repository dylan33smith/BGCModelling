"""Derive an I1 class direction and run its SPEC 6.4 manipulation check.

The direction comes from TRAIN records (SPEC 6). The manipulation check is read on VAL
records, which the direction never saw -- a check computed on the same records the
direction was derived from would be guaranteed to pass and would measure nothing.

⚠ THIS SCRIPT NEVER GENERATES OR SCORES. Alpha is swept by G9 against the manipulation
check and generation health, never against the benchmark endpoint (SPEC 2.4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgcbench.data.classmap import BENCHMARK_CLASSES
from bgcbench.model import directions as D
from bgcbench.model.load import load
from bgcbench.provenance import code_version, corpus_max_len

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
OUT = ROOT / "directions"


def _load(p: Path, cls: str) -> list[dict]:
    out = []
    for line in open(p):
        r = json.loads(line)
        r["split_class"] = cls
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="evo2-1b")
    ap.add_argument("--target", required=True, choices=list(BENCHMARK_CLASSES))
    ap.add_argument("--adapter", default=None,
                    help="derive on top of a weight state. I1 composes with W, so the "
                         "direction must come from the model the arm actually runs.")
    ap.add_argument("--prefix", choices=["none", "taxonomy"], default="taxonomy")
    ap.add_argument("--limit", type=int, default=64,
                    help="records per side. The mean is over TOKENS, so 64 records at "
                         "~3,000 nt is ~200k positions per site -- ample for a mean.")
    ap.add_argument("--max-len-nt", type=int, default=corpus_max_len())
    ap.add_argument("--check-alpha", type=float, default=1.0,
                    help="alpha used ONLY for the SPEC 6.4 manipulation check, to show the "
                         "direction lands. Not the swept generation alpha (G9).")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    sub = load(args.substrate)
    adapter = None
    if args.adapter:
        from bgcbench.model.load import attach_adapter, resolve_best
        # ⚠ SPEC 6.4: the arm generates from the BEST held-out checkpoint. Deriving from the
        # directory would take `final` instead, so the direction would come from a DIFFERENT
        # model than the one it is injected into -- invisible in both reports.
        adapter = resolve_best(args.adapter)
        sub = attach_adapter(sub, adapter)
        print(f"derived on adapter {adapter}", flush=True)

    others = [c for c in BENCHMARK_CLASSES if c != args.target]
    tr_t = _load(SPLITS / args.target / "train.jsonl", args.target)
    tr_o: list[dict] = []
    per = max(1, args.limit // len(others))
    for c in others:
        tr_o += _load(SPLITS / c / "train.jsonl", c)[:per]
    va_t = _load(SPLITS / args.target / "val.jsonl", args.target)

    tbl = None
    taxonomy = None
    if args.prefix == "taxonomy":
        from bgcbench.data import taxonomy
        tbl = taxonomy.load_table()
        for grp in (tr_t, tr_o, va_t):
            taxonomy.attach(grp, tbl)

    print(f"deriving {args.target} from {min(len(tr_t), args.limit)} target / "
          f"{len(tr_o)} other train records", flush=True)
    art = D.derive(sub, tr_t, tr_o, args.prefix, args.max_len_nt, limit=args.limit)
    art["target_class"] = args.target
    art["contrast_classes"] = others
    art["adapter"] = adapter
    art["adapter_requested"] = args.adapter
    art["code_version"] = code_version()

    # SPEC 6.4 manipulation check, against a readout derived INDEPENDENTLY on the val split
    # and measured WITH the injection attached. Projecting onto the injected vector itself
    # would rise by exactly alpha for any vector, noise included, and could not fail.
    va_o: list[dict] = []
    for c in others:
        va_o += _load(SPLITS / c / "val.jsonl", c)[:per]
    if args.prefix == "taxonomy":
        taxonomy.attach(va_o, tbl)
    chk = D.manipulation_check(sub, va_t, va_o, art["directions"], args.prefix,
                               args.max_len_nt, limit=args.limit, alpha=args.check_alpha)
    art["manipulation_check"] = chk
    print(f"raw norms per site : {[round(x, 3) for x in art['raw_norms']]}")
    print(f"relative norms     : {[round(x, 5) for x in art['relative_norms']]}")
    print(f"train/val cosine   : {[round(c, 3) for c in chk['cosine_train_val']]} "
          f"(mean {chk['mean_cosine']:.3f})")
    print(f"readout shift      : {[round(x, 4) for x in chk['shift']]}")
    print(f"first-site shift   : {chk['first_site_shift']:.4f} vs predicted "
          f"{chk['first_site_shift_expected']:.4f} -> "
          f"{'agrees' if chk['first_site_agrees'] else 'DISAGREES'}")
    # SPEC 6 parts (a) and (b). (c) is `chk` above. All three are required (12.A4).
    mono = D.projection_vs_alpha(sub, va_t, art["directions"], args.prefix, args.max_len_nt,
                                 alphas=[0.0, 0.5, 1.0, 2.0, 4.0], limit=min(args.limit, 16))
    kl = D.kl_vs_unsteered(sub, va_t, art["directions"], args.prefix, args.max_len_nt,
                           alpha=args.check_alpha, limit=min(args.limit, 16))
    art["check_a_monotone"] = mono
    art["check_b_kl"] = kl
    art["check_all_pass"] = bool(mono["passes"] and kl["passes"] and chk["passes"])
    print(f"(a) monotone proj  : {'PASS' if mono['passes'] else 'FAIL'}  "
          f"{ {i: [round(v,3) for v in mono['projection_per_site'][i]] for i in mono['active_sites']} }")
    print(f"(b) KL vs unsteered: {'PASS' if kl['passes'] else 'FAIL'}  "
          f"mean {kl['mean_kl_nats']:.5f} nats/pos at alpha={args.check_alpha}, "
          f"argmax changed {kl['frac_argmax_changed']:.3f}")
    print(f"(c) reproduces     : {'PASS' if chk['passes'] else 'FAIL'} -- {chk['criterion']}")
    print(f"MANIPULATION CHECK : {'PASS' if art['check_all_pass'] else 'FAIL'} (all three required)")
    if not art["check_all_pass"]:
        print("⚠ the direction does not verifiably land; SPEC 6.4 makes any null from this "
              "arm UNINFORMATIVE rather than negative", flush=True)

    name = args.name or f"{args.substrate}_I1_{args.target}"
    p = D.save(art, OUT / f"{name}.pt")
    print(f"wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
