#!/usr/bin/env python
"""PROBE 3 — did the DOMAIN-WEIGHTED arm change the model at all?

WHY THIS IS NEEDED. The weighted arm was declared a null at n=152 (detection 16/152 vs baseline
17/152, Fisher p=1.000). But it is also **indistinguishable from baseline on every other quantity
measured** — gene length p=0.23, any-Pfam p=0.25, best_bio_bits p=0.81, n_bio_domains p=0.88,
bio_span_frac p=0.89, and stop-completion mass 0.1228 vs 0.1227. So the null has two readings:

  (a) the weighting was DELIVERED and did not help; or
  (b) the weighting never took effect, and there was no treatment to evaluate.

For the frame arm this ambiguity was resolved BEFORE trusting its null — an 8x drop in
stop-completion mass proved the intervention landed. The weighted arm never got that check, and its
null was reported as a closure anyway. This closes the gap.

THE MEASUREMENT. `--domain-weight 3.0` multiplies the per-token loss by 3 inside class-defining
domain spans (per-record normalised). If it worked, the trained model must predict domain
nucleotides *relatively* better than a model that never saw the weighting. So: score fixed real
held-out cores under each adapter and split the per-position cross-entropy into IN-domain and
OUT-of-domain positions.

THE STATISTIC IS THE RATIO, not the raw domain loss. A model that is simply better everywhere would
have a lower domain loss without having been steered at all; `loss_in / loss_out` divides that out
and isolates the reallocation the weighting was supposed to buy.

READING IT, decided before running:
  * weighted's ratio clearly BELOW baseline's -> the treatment landed; its null is a real negative.
  * weighted's ratio == baseline's            -> the treatment never landed; the n=152 weighted null
                                                 is UNINTERPRETABLE and 'domain weighting' is untested.

Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "evo2_1b" / "scripts"))

# Auto-discovered so a new arm is picked up without editing this file. `base` and `baseline` are
# pinned first: base fixes the untrained reference and baseline is the comparator every ratio is
# quoted against, so both must be present before any treatment arm is interpreted.
_PINNED = ["baseline", "frame", "weighted"]


def discover_arms(root: Path) -> dict:
    arms = {"base (no adapter)": None}
    found = [d.name for d in sorted(root.iterdir())
             if (d / "final_adapter" / "adapter_model.safetensors").exists()]
    for name in _PINNED + [f for f in found if f not in _PINNED]:
        if name in found:
            arms[name] = f"{name}/final_adapter"
    return arms


def annotated_cores(n: int, length: int, pfam: Path, workers: int) -> list[tuple[str, list]]:
    """Real held-out cores plus their CLASS-defining domain spans, via the same pipeline that built
    the training sidecar — so 'in-domain' means here exactly what it meant during training."""
    from concurrent.futures import ProcessPoolExecutor

    from build_domain_spans import _BIOSYNTHETIC_PFAMS, _subset_hmm, annotate
    hmm = _subset_hmm(pfam, REPO / "data" / "processed" / "bio_subset.hmm", _BIOSYNTHETIC_PFAMS)
    recs = []
    with open("/data2/ds85/bgcmodel_data/splits_core/test.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if len(r.get("sequence", "")) >= length:
                r = dict(r, sequence=r["sequence"][:length])
                recs.append(r)
                if len(recs) >= n:
                    break
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(annotate, ((r, str(hmm), 1e-5, i) for i, r in enumerate(recs))))
    keep = []
    for r, a in zip(recs, out):
        if a.get("error"):
            continue
        spans = [s for s in (a.get("spans") or []) if len(s) > 3 and s[3]]   # class-defining only
        if spans:
            keep.append((r["sequence"], spans))
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--len", type=int, default=6000)
    ap.add_argument("--pfam", type=Path, default=Path("/data2/ds85/pfam/Pfam-A.hmm"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--root", default="/data2/ds85/bgcmodel_runs/phase2_1b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arm names (base is always included)")
    args = ap.parse_args()

    cores = annotated_cores(args.n, args.len, args.pfam, args.workers)
    if not cores:
        raise SystemExit("[dw-probe] ABORT: no held-out cores carry class-defining domains — the "
                         "in/out split would be vacuous")
    frac = st.mean(sum(min(e, args.len) - max(s, 0) for s, e, *_ in sp) / args.len
                   for _, sp in cores)
    print(f"[dw-probe] {len(cores)} real held-out cores with class domains; "
          f"mean {frac*100:.1f}% of positions in-domain")

    from evo2_1b_inference import load_1b
    rows, per_core = {}, {}
    _arms = discover_arms(Path(args.root))
    if args.arms:
        _arms = {k: v for k, v in _arms.items() if v is None or k in args.arms}
    for label, rel in _arms.items():
        adapter = None if rel is None else Path(args.root) / rel
        if adapter is not None and not adapter.exists():
            continue
        print(f"[dw-probe] loading {label} …", flush=True)
        w = load_1b(device=args.device, adapter_dir=adapter)
        li, lo = [], []
        for seq, spans in cores:
            ids = [int(i) for i in w.tokenizer.tokenize(seq)]
            x = torch.tensor([ids], dtype=torch.long, device=args.device)
            with torch.no_grad():
                logits = w.model(x)
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)[0]
            nll = -lp.gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)      # per-position CE
            mask = torch.zeros(nll.shape[0], dtype=torch.bool, device=args.device)
            for s, e, *_ in spans:
                a, b = max(int(s) - 1, 0), min(int(e) - 1, nll.shape[0])
                if b > a:
                    mask[a:b] = True
            if bool(mask.any()) and bool((~mask).any()):
                li.append(float(nll[mask].mean()))
                lo.append(float(nll[~mask].mean()))
        rows[label] = (st.mean(li), st.mean(lo))
        per_core[label] = (li, lo)
        print(f"[dw-probe]   in-domain {rows[label][0]:.4f}  out {rows[label][1]:.4f}  "
              f"ratio {rows[label][0]/rows[label][1]:.4f}")
        del w
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("DOMAIN-WEIGHTING PROBE — cross-entropy on fixed real cores, split by domain membership")
    print("=" * 78)
    print(f"{'model':<20} {'in-domain':>10} {'out':>10} {'ratio in/out':>14} {'vs baseline':>13}")
    b = rows.get("baseline")
    for label, (i_, o_) in rows.items():
        r = i_ / o_
        delta = "" if not b or label == "baseline" else f"{(r/(b[0]/b[1]) - 1)*100:+.2f}%"
        print(f"{label:<20} {i_:>10.4f} {o_:>10.4f} {r:>14.4f} {delta:>13}")
    # VERDICT KEYS OFF THE NUMERATOR, NOT THE RATIO. Reporting the ratio alone once made 3x and
    # 10x look different when their in-domain losses are IDENTICAL (0.8763 both) and the whole gap
    # was the denominator degrading. A ratio improvement bought by damaging the down-weighted
    # positions is not the intervention working.
    treatments = [k for k in rows if k.startswith("weighted")]
    if b and treatments:
        print()
        base_in = rows.get("base (no adapter)", (b[0], b[1]))[0]
        train_effect = base_in - b[0]          # what plain fine-tuning bought on domain loss
        for k in treatments:
            gain = b[0] - rows[k][0]
            harm = rows[k][1] - b[1]
            print(f"  {k:<12} domain gain {gain:+.5f} = {gain/train_effect*100:5.1f}% of what training "
                  f"alone buys | non-domain harm {harm:+.5f}")
        print("\n  See the paired tests below before reading any of these as real.")
    from scipy.stats import wilcoxon
    if "baseline" in per_core:
        print("\n" + "=" * 78)
        print("PAIRED TEST ON THE IN-DOMAIN LOSS ITSELF (same cores, so pair them)")
        print("=" * 78)
        bi, bo = per_core["baseline"]
        for label, (li, lo) in per_core.items():
            if label == "baseline":
                continue
            di = [a - b for a, b in zip(li, bi)]
            do = [a - b for a, b in zip(lo, bo)]
            try:
                _, pi = wilcoxon(di)
                _, po = wilcoxon(do)
            except ValueError:
                pi = po = float("nan")
            print(f"  {label:<20} in-domain {st.mean(di):+.5f} (p={pi:.4f})   "
                  f"out {st.mean(do):+.5f} (p={po:.4f})")
        print("\n  A ratio can improve because the NUMERATOR fell or because the DENOMINATOR rose.")
        print("  Only the first is the intervention working; the second is collateral damage.")
    json.dump({k: {"in": v[0], "out": v[1], "ratio": v[0] / v[1],
                   "per_core_in": per_core[k][0], "per_core_out": per_core[k][1]}
               for k, v in rows.items()},
              open(Path(args.root) / "domain_weight_probe.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
