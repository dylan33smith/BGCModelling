#!/usr/bin/env python
"""BASELINE 1: next-base cross-entropy on REAL held-out cores, 1B vs 7B.

WHY THIS IS THE FIRST THING PHASE 2 MEASURES. The 1B is loaded in bf16 because Transformer Engine
is absent, while its checkpoint was trained with FP8 input projections. That is a real numerical
change, and a model that loads cleanly can still be damaged. The first sanity check written for
this used a HAND-WRITTEN sequence and returned 1.3843 nats against a uniform guess of 1.386 --
indistinguishable from chance -- which looked like a broken substrate but was actually a bad test:
invented DNA is out-of-distribution for both models. Real held-out cores are the correct probe.

It doubles as the first Phase-2 baseline, and as the headline 1B-vs-7B difference: if the 1B is far
worse at modelling BGC DNA, every downstream comparison has to be read against that handicap.

REFERENCE POINTS from the 7B (measured 2026-08-12, `evo2/scripts/context_ablation.py`):
    uniform guess over A/C/G/T           1.386 nats
    7B + LoRA, 10 nt of context          0.977
    7B + LoRA, 6,000 nt of context       0.829
Anything near 1.386 means the model is not predicting DNA at all.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2_1b" / "scripts"))
sys.path.insert(0, str(REPO / "evo2" / "scripts"))


@torch.no_grad()
def nats_per_base(model, tok, seq: str, device: str, ctx_nt: int, score_nt: int):
    """Cross-entropy over `score_nt` bases that follow `ctx_nt` bases of real context."""
    ids = [int(i) for i in tok.tokenize(seq[: ctx_nt + score_nt])]
    if len(ids) <= ctx_nt + 1:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)[0]
    labels = x[0, 1:]
    scored = lp[ctx_nt - 1:].gather(-1, labels[ctx_nt - 1:].unsqueeze(-1))
    return float(-scored.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE", "RIPP"])
    ap.add_argument("--n-per-class", type=int, default=6)
    ap.add_argument("--ctx-nt", type=int, default=2000)
    ap.add_argument("--score-nt", type=int, default=500)
    ap.add_argument("--models", nargs="+", default=["1b", "7b"])
    ap.add_argument("--adapter-7b",
                    default="/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/"
                            "checkpoints/step_1200")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase2_1b/compare_1b_7b_loss.json"))
    args = ap.parse_args()

    need = args.ctx_nt + args.score_nt
    byc: dict[str, list] = {c: [] for c in args.classes}
    for line in args.cores.open():
        r = json.loads(line)
        c = r.get("compound_class")
        if c in byc and len(r.get("sequence", "")) >= need:
            byc[c].append(r)
    rng = random.Random(args.seed)
    picked = []
    for c in args.classes:
        rng.shuffle(byc[c])
        picked += byc[c][: args.n_per_class]
    print(f"[cmp] {len(picked)} real held-out cores, ctx={args.ctx_nt} score={args.score_nt}",
          flush=True)

    rows = []
    for tag in args.models:
        if tag == "1b":
            from evo2_1b_inference import load_1b
            w = load_1b(device=args.device)
            model, tok, label = w.model, w.tokenizer, "1B base (bf16, no TE)"
        else:
            from evo2_inference import load_evo2_wrapper_for_inference
            w = load_evo2_wrapper_for_inference(Path(args.adapter_7b), device=args.device)
            model, tok, label = w.model, w.tokenizer, "7B + LoRA step_1200"
        vals = []
        for r in picked:
            v = nats_per_base(model, tok, r["sequence"], args.device, args.ctx_nt, args.score_nt)
            if v is not None:
                vals.append({"cls": r["compound_class"], "nats": v})
        rows.append({"model": tag, "label": label, "n": len(vals),
                     "mean_nats": st.mean(x["nats"] for x in vals),
                     "per_class": {c: st.mean(x["nats"] for x in vals if x["cls"] == c)
                                   for c in args.classes
                                   if any(x["cls"] == c for x in vals)}})
        print(f"[cmp] {label}: {rows[-1]['mean_nats']:.4f} nats/base over n={len(vals)}", flush=True)
        del model, w
        torch.cuda.empty_cache()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=1))

    print("\n" + "=" * 78)
    print("NEXT-BASE CROSS-ENTROPY ON REAL HELD-OUT CORES (lower is better)")
    print("uniform guess over A/C/G/T = 1.386 nats")
    print("=" * 78)
    print(f"\n{'model':>24} {'n':>4} {'nats/base':>10} " +
          " ".join(f"{c[:7]:>8}" for c in args.classes))
    for r in rows:
        print(f"{r['label']:>24} {r['n']:>4} {r['mean_nats']:>10.4f} " +
              " ".join(f"{r['per_class'].get(c, float('nan')):>8.4f}" for c in args.classes))
    if len(rows) == 2:
        d = rows[0]["mean_nats"] - rows[1]["mean_nats"]
        print(f"\n1B minus 7B: {d:+.4f} nats/base")
        print("  A small gap means the 1B is a usable stand-in for iterating on the objective.")
        print("  A large gap means every Phase-2 result is read against a weaker model, and a")
        print("  null could be the substrate rather than the intervention.")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
