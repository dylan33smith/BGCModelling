#!/usr/bin/env python
"""Full-depth class-probe sweep: WHICH layer best encodes compound class, and HOW EARLY?

Supersedes the stride-4 sweep in class_probe.py, which sampled only 9 of 32 blocks and
found a PLATEAU (L12 0.904 / L16 0.911 / L20 0.901) — a spread of ~0.010, about one
standard error at n~991, so "layer 16 is best" was never actually established.

The forward pass is the expensive part; hooks are nearly free. So we capture ALL blocks
in ONE pass and, inside the same hook, also mean-pool over PREFIXES of the sequence
(500 / 1k / 2k / full). That answers two questions for the price of one:

  (1) WHICH LAYER encodes class best — with per-fold error bars, so ties read as ties.
  (2) HOW EARLY it is readable — the prerequisite for guided decoding, which must score
      a partial sequence mid-generation.

Activations are cached to .npz so steering directions (logistic weights AND
difference-of-means) can be derived later at ANY layer with no recomputation.

NB run this on the model you intend to STEER (the merged v2 adapter), not just base:
the adapter is merged into the weights, so its activation space is its own.

CAVEAT: probe accuracy = where class is most READABLE, which is not necessarily where it
is most STEERABLE (decoding != causal intervention). Use this to narrow 32 layers to a
few candidates, then let the steering sweep arbitrate.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402

_BLOCK_RE = re.compile(r"^blocks\.\d+$")


def _sample_cores(jsonl: Path, per_class: int, min_class: int, max_nt: int, seed: int):
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
def _embed_all_layers(wrapper, data, prefix_lens, device):
    """One forward pass per sequence; capture EVERY block, mean-pooled over each prefix.

    Returns X[layer] -> (N, P, D) and y (N,). Hooks reduce (1,L,D) to (P,D) immediately,
    so peak memory stays small even with all 32 blocks hooked."""
    model, tok = wrapper.model, wrapper.tokenizer
    mods = dict(model.named_modules())
    layers = sorted(int(n.split(".")[1]) for n in mods if _BLOCK_RE.match(n))
    captured: dict[int, np.ndarray] = {}

    def mk(i):
        def hook(_m, _in, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            h = h.detach().float()[0]                      # (L, D)
            vs = []
            for p in prefix_lens:
                n = h.shape[0] if p is None else min(p, h.shape[0])
                vs.append(h[:n].mean(dim=0).cpu().numpy())
            captured[i] = np.stack(vs)                     # (P, D)
        return hook

    handles = [mods[f"blocks.{i}"].register_forward_hook(mk(i)) for i in layers]
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
            print(f"[sweep] {len(layers)} blocks hooked; per-layer capture "
                  f"{captured[layers[0]].shape} (prefixes x hidden)", file=sys.stderr)
        for i in layers:
            X[i].append(captured[i])
        y.append(c)
        if (k + 1) % 100 == 0:
            print(f"[sweep]   embedded {k + 1}/{len(data)}", file=sys.stderr, flush=True)
    for h in handles:
        h.remove()
    return {i: np.stack(X[i]) for i in layers}, np.array(y), layers


def _fit(Xi, y, seed, n_jobs):
    """5-fold linear probe + shuffled-label control. Returns per-fold scores so that
    differences smaller than the fold spread can be called TIES rather than rankings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mk = lambda: make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = cross_val_score(mk(), Xi, y, cv=skf, scoring="balanced_accuracy", n_jobs=n_jobs)
    ysh = y.copy()
    np.random.RandomState(seed).shuffle(ysh)
    ssh = cross_val_score(mk(), Xi, ysh, cv=skf, scoring="balanced_accuracy", n_jobs=n_jobs)
    return (float(s.mean()), float(s.std()), float(s.std() / np.sqrt(len(s))),
            float(ssh.mean()), s.tolist())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA ckpt/run dir; omit for base Evo2.")
    ap.add_argument("--from-jsonl", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--min-class", type=int, default=40)
    ap.add_argument("--max-nt", type=int, default=4096)
    ap.add_argument("--prefix-lens", type=int, nargs="+", default=[500, 1000, 2000],
                    help="Prefix lengths to mean-pool (full sequence is always added).")
    ap.add_argument("--top-k-prefix", type=int, default=3,
                    help="Run the prefix-length analysis on the top-K layers (full-seq ranking).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--cache", type=Path, default=None, help="Write/read activations .npz here.")
    ap.add_argument("--from-cache", action="store_true", help="Skip the GPU pass; refit from --cache.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prefix_lens = list(args.prefix_lens) + [None]           # None = full sequence
    plabels = [str(p) for p in args.prefix_lens] + ["full"]

    if args.from_cache:
        z = np.load(args.cache, allow_pickle=True)
        y = z["y"]; layers = [int(v) for v in z["layers"]]
        X = {i: z[f"L{i}"] for i in layers}
        classes = sorted(set(y.tolist()))
        print(f"[sweep] loaded cache {args.cache} — {len(y)} seqs, {len(layers)} layers", file=sys.stderr)
    else:
        data, classes = _sample_cores(args.from_jsonl, args.per_class, args.min_class,
                                      args.max_nt, args.seed)
        print(f"[sweep] {len(data)} cores / {len(classes)} classes (chance {1/len(classes):.3f})",
              file=sys.stderr)
        adapter = args.adapter
        if adapter is not None:
            adapter = Path(adapter)
            if not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
                adapter = adapter / "adapter"
        print(f"[sweep] model: {'base Evo2' if adapter is None else adapter}", file=sys.stderr)
        wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)
        X, y, layers = _embed_all_layers(wrapper, data, prefix_lens, args.device)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.cache, y=y, layers=np.array(layers),
                                prefix_labels=np.array(plabels),
                                **{f"L{i}": X[i] for i in layers})
            print(f"[sweep] cached activations -> {args.cache}", file=sys.stderr)

    fi = len(plabels) - 1                                   # full-sequence index
    print(f"\n[sweep] === LAYER SWEEP (full sequence, {len(layers)} layers) ===", file=sys.stderr)
    rows = []
    for i in layers:
        m, sd, se, shm, folds = _fit(X[i][:, fi, :], y, args.seed, args.n_jobs)
        rows.append({"layer": i, "balanced_acc": round(m, 4), "std": round(sd, 4),
                     "sem": round(se, 4), "shuffled": round(shm, 4), "folds": [round(f, 4) for f in folds]})
        print(f"[sweep] L{i:2d}: {m:.3f} ± {se:.3f} (sem)   shuffled {shm:.3f}",
              file=sys.stderr, flush=True)

    best = max(rows, key=lambda r: r["balanced_acc"])
    # Anything within 1 SEM of the winner is a statistical TIE, not a ranking.
    thresh = best["balanced_acc"] - best["sem"]
    tied = sorted([r["layer"] for r in rows if r["balanced_acc"] >= thresh])
    topk = [r["layer"] for r in sorted(rows, key=lambda r: -r["balanced_acc"])[: args.top_k_prefix]]
    print(f"\n[sweep] best L{best['layer']} = {best['balanced_acc']:.3f} ± {best['sem']:.3f}",
          file=sys.stderr)
    print(f"[sweep] statistically TIED with best (within 1 sem): {tied}", file=sys.stderr)

    print(f"\n[sweep] === PREFIX LENGTH (top-{args.top_k_prefix} layers: {topk}) ===", file=sys.stderr)
    prefix_rows = []
    for i in topk:
        for pi, pl in enumerate(plabels):
            m, sd, se, shm, _ = _fit(X[i][:, pi, :], y, args.seed, args.n_jobs)
            prefix_rows.append({"layer": i, "prefix": pl, "balanced_acc": round(m, 4),
                                "sem": round(se, 4), "shuffled": round(shm, 4)})
            print(f"[sweep] L{i:2d} prefix {pl:>5}: {m:.3f} ± {se:.3f}", file=sys.stderr, flush=True)

    # class means per layer (full seq) -> difference-of-means steering directions, free
    means = {}
    for i in layers:
        Z = X[i][:, fi, :]
        gm = Z.mean(axis=0)
        means[str(i)] = {c: (Z[y == c].mean(axis=0) - gm).tolist() for c in classes}

    out = {"model": "base" if args.adapter is None else str(args.adapter),
           "n": int(len(y)), "classes": classes, "chance": round(1 / len(classes), 4),
           "prefix_labels": plabels, "layer_sweep": rows,
           "best_layer": best["layer"], "best_balanced_acc": best["balanced_acc"],
           "tied_with_best": tied, "top_layers": topk, "prefix_sweep": prefix_rows}
    args.out.write_text(json.dumps(out, indent=1))
    if args.cache:
        np.savez_compressed(args.cache.with_suffix(".dirs.npz"),
                            **{f"L{i}_{c}": np.array(v) for i, cs in means.items() for c, v in cs.items()})
        print(f"[sweep] steering directions (diff-of-means) -> {args.cache.with_suffix('.dirs.npz')}",
              file=sys.stderr)
    print(f"[sweep] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
