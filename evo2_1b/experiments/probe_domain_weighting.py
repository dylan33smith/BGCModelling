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

ARMS = {
    "base (no adapter)": None,
    "baseline": "baseline/final_adapter",
    "frame": "frame/final_adapter",
    "weighted": "weighted/final_adapter",
}


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
    rows = {}
    for label, rel in ARMS.items():
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
    if b and "weighted" in rows:
        wr = rows["weighted"][0] / rows["weighted"][1]
        br = b[0] / b[1]
        print()
        if wr < br * 0.99:
            print("  ⇒ THE TREATMENT LANDED. The weighted arm reallocated capacity toward domain")
            print("    positions, so its n=152 null is a REAL negative: domain weighting was")
            print("    delivered and did not improve de novo biosynthetic content.")
        else:
            print("  ⇒ THE TREATMENT NEVER LANDED. The weighted model predicts domain positions no")
            print("    better, relatively, than baseline — so `--domain-weight 3.0` changed the loss")
            print("    arithmetic without changing the model. The n=152 weighted null is")
            print("    UNINTERPRETABLE, and 'domain weighting' remains UNTESTED, not refuted.")
            print("    Next: raise the weight substantially, and/or train past 6.7% of one epoch.")
    json.dump({k: {"in": v[0], "out": v[1], "ratio": v[0] / v[1]} for k, v in rows.items()},
              open(Path(args.root) / "domain_weight_probe.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
