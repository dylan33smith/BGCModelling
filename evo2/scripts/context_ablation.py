#!/usr/bin/env python
"""HOW MUCH OF THE MODEL'S PREDICTION COMES FROM THE LAST FEW BASES? The local-statistics budget.

WHY THIS EXISTS. Every conditioning mechanism we have tried has to compete, during training, with
whatever else predicts the next base. If the immediately preceding ~100 bases already predict it
almost perfectly, then a class label sitting 30,000 positions away has almost no loss to reduce,
gradient descent has almost no reason to build a pathway that reads it, and "the tag is inert" stops
being mysterious. This script measures that competition directly.

THE MEASUREMENT. Take real held-out cores. Score the SAME 500 bases every time, varying only how
much preceding context the model is allowed to see: 10 bases, 30, 100, ... up to several kb. Report
next-base cross-entropy in nats. Two reference points make the numbers readable:

    ln(4) = 1.386 nats   a uniform guess over A/C/G/T
    ~0.65-0.82 nats      where our fine-tune's train/val loss actually sits

The gap between the 10-base number and the several-kb number is the ENTIRE value of long-range
context to this objective. Anything a class label could contribute has to come out of that gap --
and the tag competes for it against every other long-range cue (GC content, codon phase, repeats,
the organism's own signature).

Also scored: the class tag present vs absent at the longest context. That is the same comparison
the conditioning programme has been making all along, expressed in the units that training actually
optimises, so it can be read against the local-context budget on the same axis.

This is a diagnostic, not an intervention. It cannot tell us whether conditioning is POSSIBLE; it
tells us how much room there is for it to matter to the loss, which bounds what any training-time
mechanism can be driven by.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402


@torch.no_grad()
def _score(model, tok, context: str, target: str, device: str):
    """Mean next-base cross-entropy (nats) over `target`, given `context` before it."""
    ids = [int(i) for i in tok.tokenize(context + target)]
    n_ctx = len(list(tok.tokenize(context)))
    if n_ctx >= len(ids) - 1:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)[0]
    labels = x[0, 1:]
    scored = lp[n_ctx - 1:].gather(-1, labels[n_ctx - 1:].unsqueeze(-1))
    return float(-scored.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE", "RIPP"])
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[10, 30, 100, 300, 1000, 3000, 6000])
    ap.add_argument("--score-nt", type=int, default=500)
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/context_ablation.json"))
    args = ap.parse_args()

    need = max(args.contexts) + args.score_nt
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
    print("[ctx] " + ", ".join(f"{c}={min(len(byc[c]), args.n_per_class)}" for c in args.classes)
          + f"  (need >= {need} nt)", flush=True)

    wrapper = load_evo2_wrapper_for_inference(Path(args.adapter) if args.adapter else None,
                                              device=args.device)
    model, tok = wrapper.model, wrapper.tokenizer

    rows = []
    for i, r in enumerate(picked):
        seq, cls = r["sequence"], r["compound_class"]
        tax = r.get("taxonomic_tag", "") or ""
        tgt = seq[max(args.contexts): max(args.contexts) + args.score_nt]
        for k in args.contexts:
            # The SAME target every time; only how much precedes it changes.
            ctx = seq[max(args.contexts) - k: max(args.contexts)]
            v = _score(model, tok, ctx, tgt, args.device)
            if v is not None:
                rows.append({"cls": cls, "ctx_nt": k, "tag": "none", "nats": v})
        # Longest context, now WITH the conditioning the training pipeline supplies.
        ctx = seq[:max(args.contexts)]
        for tag_name, tag in (("tax+class", f"{tax}|COMPOUND_CLASS:{cls}|"),
                              ("tax", tax),
                              ("wrong_class",
                               f"{tax}|COMPOUND_CLASS:"
                               f"{[c for c in args.classes if c != cls][0]}|")):
            v = _score(model, tok, tag + ctx, tgt, args.device)
            if v is not None:
                rows.append({"cls": cls, "ctx_nt": max(args.contexts), "tag": tag_name, "nats": v})
        print(f"[ctx] {i + 1}/{len(picked)}", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=1))

    import statistics as st
    print("\n" + "=" * 86)
    print("NEXT-BASE CROSS-ENTROPY (nats) vs HOW MUCH PRECEDING CONTEXT THE MODEL SEES")
    print("Same 500 bases scored every time. Uniform guess over A/C/G/T = 1.386 nats.")
    print("=" * 86)
    print(f"\n{'context (nt)':>13} {'nats/base':>10} {'vs 10 nt':>10}")
    base = None
    for k in args.contexts:
        v = [r["nats"] for r in rows if r["ctx_nt"] == k and r["tag"] == "none"]
        if not v:
            continue
        m = st.mean(v)
        if base is None:
            base = m
        print(f"{k:>13,} {m:>10.4f} {m - base:>+10.4f}")
    full = st.mean([r["nats"] for r in rows
                    if r["ctx_nt"] == max(args.contexts) and r["tag"] == "none"])
    print(f"\nTOTAL value of going from 10 nt to {max(args.contexts):,} nt of context: "
          f"{base - full:+.4f} nats")

    print(f"\n{'conditioning at ' + str(max(args.contexts)) + ' nt':>34} {'nats/base':>10} "
          f"{'vs untagged':>12}")
    for tag_name in ("none", "tax", "tax+class", "wrong_class"):
        v = [r["nats"] for r in rows if r["tag"] == tag_name
             and r["ctx_nt"] == max(args.contexts)]
        if v:
            print(f"{tag_name:>34} {st.mean(v):>10.4f} {st.mean(v) - full:>+12.4f}")

    tc = [r["nats"] for r in rows if r["tag"] == "tax+class"]
    wc = [r["nats"] for r in rows if r["tag"] == "wrong_class"]
    if tc and wc:
        gain = st.mean(wc) - st.mean(tc)
        print(f"\nRIGHT class tag vs WRONG class tag: {gain:+.4f} nats")
        print(f"  as a share of what long-range context is worth at all: "
              f"{gain / max(base - full, 1e-9):.2%}")
        print("\nHOW TO READ THIS. The 'right minus wrong tag' number is the entire loss signal")
        print("available to teach the model to use its class label. If it is a rounding error")
        print("next to what local context already provides, then gradient descent has almost")
        print("no incentive to build a pathway that reads the tag — and the tag being inert at")
        print("generation time is the expected outcome, not a surprise.")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
