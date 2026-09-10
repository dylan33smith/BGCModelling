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
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    sub = load(args.substrate)
    if args.adapter:
        from bgcbench.model.load import attach_adapter
        sub = attach_adapter(sub, args.adapter)
        print(f"derived on adapter {args.adapter}", flush=True)

    others = [c for c in BENCHMARK_CLASSES if c != args.target]
    tr_t = _load(SPLITS / args.target / "train.jsonl", args.target)
    tr_o: list[dict] = []
    per = max(1, args.limit // len(others))
    for c in others:
        tr_o += _load(SPLITS / c / "train.jsonl", c)[:per]
    va_t = _load(SPLITS / args.target / "val.jsonl", args.target)

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
    art["adapter"] = args.adapter
    art["code_version"] = code_version()

    # SPEC 6.4 manipulation check, on HELD-OUT records the direction never saw.
    proj = D.projection(sub, va_t, art["directions"], args.prefix, args.max_len_nt,
                        limit=args.limit)
    art["val_projection"] = proj
    print(f"raw norms per site : {[round(x, 3) for x in art['raw_norms']]}")
    print(f"val projection     : {[round(x, 3) for x in proj]}")

    name = args.name or f"{args.substrate}_I1_{args.target}"
    p = D.save(art, OUT / f"{name}.pt")
    print(f"wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
