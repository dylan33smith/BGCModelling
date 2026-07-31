#!/usr/bin/env python
"""Linear-probe compound_class from Evo2 hidden states — does the model REPRESENT class?

The fork gate for the whole conditioning program. We feed RAW core nucleotides (NO
prefix / NO class tag — else the probe would trivially read the tag tokens), capture
mean-pooled hidden states at several blocks, and fit a linear (logistic) probe to
predict compound_class, vs a shuffled-label control.

  * Linearly SEPARABLE (balanced_acc >> chance/shuffled) -> the model already encodes
    class in its activations; the problem is DECODING/STEERING (cheap fixes viable:
    guided decoding, steering vectors, soft prompts).
  * NOT separable -> class is not represented; you must INSTALL a representation
    (per-class adapter / whole-core / long-context) — the only thing that justifies
    heavy compute.

Runs base Evo2 and (optionally) the v2 adapter: if base is flat but v2 separates,
the adapter installed a class representation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402

_BLOCK_RE = re.compile(r"^blocks\.\d+$")


def _sample_cores(jsonl: Path, per_class: int, min_class: int, max_nt: int, seed: int):
    import random
    rng = random.Random(seed)
    byc: dict[str, list[str]] = {}
    for line in jsonl.open():
        r = json.loads(line)
        c, s = r.get("compound_class"), r.get("sequence", "")
        if c and len(s) >= 200:
            byc.setdefault(c, []).append(s)
    classes = sorted(c for c, v in byc.items() if len(v) >= min_class)
    data = []
    for c in classes:
        pool = byc[c][:]
        rng.shuffle(pool)
        for s in pool[:per_class]:
            data.append((c, s[:max_nt]))
    rng.shuffle(data)
    return data, classes


@torch.inference_mode()
def _embed(wrapper, data, layers, device):
    model, tok = wrapper.model, wrapper.tokenizer
    mods = dict(model.named_modules())
    handles, captured = [], {}

    def mk(i):
        def hook(_m, _in, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            captured[i] = h.detach().float().mean(dim=1).squeeze(0).cpu().numpy()
        return hook
    for i in layers:
        handles.append(mods[f"blocks.{i}"].register_forward_hook(mk(i)))

    X = {i: [] for i in layers}
    y = []
    for k, (c, s) in enumerate(data):
        ids = tok.tokenize(s)
        if not torch.is_tensor(ids):
            ids = torch.LongTensor(list(ids))
        ids = ids.to(device).view(1, -1)
        captured.clear()
        _ = model(ids)
        if k == 0:
            shp = {i: captured[i].shape for i in layers}
            print(f"[probe] captured per-layer embedding dims: {shp}", file=sys.stderr)
        for i in layers:
            X[i].append(captured[i])
        y.append(c)
        if (k + 1) % 200 == 0:
            print(f"[probe]   embedded {k + 1}/{len(data)}", file=sys.stderr, flush=True)
    for h in handles:
        h.remove()
    return {i: np.vstack(X[i]) for i in layers}, np.array(y)


def _probe(Xi, y, seed, classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(pipe, Xi, y, cv=skf)
    acc = balanced_accuracy_score(y, pred)
    ysh = y.copy()
    np.random.RandomState(seed).shuffle(ysh)
    predsh = cross_val_predict(pipe, Xi, ysh, cv=skf)
    accsh = balanced_accuracy_score(ysh, predsh)
    per_class = dict(zip(classes, np.round(recall_score(y, pred, labels=classes, average=None, zero_division=0), 3).tolist()))
    return float(acc), float(accsh), per_class


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA ckpt/run dir; omit for base Evo2.")
    ap.add_argument("--from-jsonl", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--min-class", type=int, default=40)
    ap.add_argument("--max-nt", type=int, default=4096)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="block indices; default every 4th + last")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data, classes = _sample_cores(args.from_jsonl, args.per_class, args.min_class, args.max_nt, args.seed)
    print(f"[probe] {len(data)} cores across {len(classes)} classes "
          f"(chance≈{1/len(classes):.3f}): {classes}", file=sys.stderr)

    adapter = args.adapter
    if adapter is not None:
        adapter = Path(adapter)
        if not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
            adapter = adapter / "adapter"
    print(f"[probe] model: {'base Evo2' if adapter is None else adapter}", file=sys.stderr)
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)
    nblocks = sum(1 for n in dict(wrapper.model.named_modules()) if _BLOCK_RE.match(n))
    layers = args.layers or sorted(set(list(range(0, nblocks, 4)) + [nblocks - 1]))
    print(f"[probe] {nblocks} blocks; probing layers {layers}", file=sys.stderr)

    embs, y = _embed(wrapper, data, layers, args.device)
    rows = []
    for i in layers:
        acc, accsh, per_class = _probe(embs[i], y, args.seed, classes)
        rows.append({"layer": i, "balanced_acc": round(acc, 3), "shuffled_acc": round(accsh, 3),
                     "gap": round(acc - accsh, 3), "per_class_recall": per_class})
        print(f"[probe] layer {i:2d}: balanced_acc {acc:.3f}  (shuffled {accsh:.3f}, chance {1/len(classes):.3f})",
              file=sys.stderr, flush=True)
    best = max(rows, key=lambda r: r["balanced_acc"])
    out = {"model": "base" if args.adapter is None else str(args.adapter), "n": len(data),
           "n_classes": len(classes), "classes": classes, "chance": round(1 / len(classes), 3),
           "best_layer": best["layer"], "best_balanced_acc": best["balanced_acc"], "layers": rows}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"[probe] BEST layer {best['layer']}: balanced_acc {best['balanced_acc']} "
          f"vs chance {1/len(classes):.3f} / shuffled {best['shuffled_acc']}  -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
