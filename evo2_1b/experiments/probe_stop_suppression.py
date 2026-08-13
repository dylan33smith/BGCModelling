#!/usr/bin/env python
"""PROBE 2 — did the frame arm learn to write longer genes, or just to never write a stop?

THE QUESTION THIS SETTLES. The frame arm's generations have longer ORFs (median 453.5 -> 700 aa,
p=0.0109), and 2 of 24 contain a single ORF spanning the WHOLE 6 kb generation. Two readings fit
that equally well:

  (a) REAL: it learned to sustain reading frame, i.e. to write megasynthase-like genes. Legitimate
      -- 3.6% of real BGC genes exceed 2,000 aa, and those are exactly the assembly-line enzymes
      the project is after.
  (b) DEGENERATE: it learned to suppress stop codons everywhere. The penalty is satisfiable that
      way, and length would then be an artifact rather than an achievement.

Generation length cannot separate these: a real megasynthase gene is LONGER than the 6 kb window,
so it would also fill it end to end. This probe measures the mechanism directly instead.

HOW. `stop_completion_penalty` is exactly the quantity the frame arm was trained against: the
probability mass the model puts on the base that would CLOSE an in-gene stop codon (TA+{A,G},
TG+{A}) at codon phase 2. Score the SAME real held-out cores under each adapter and compare. Real
gene termini are excluded by the penalty itself, so a legitimately-ending gene is never counted.

READING THE RESULT, decided before running:
  * frame >> 0 and close to baseline  -> the penalty barely bit; length came from somewhere else.
  * frame modestly below baseline      -> (a). It shifted the odds without abolishing stops.
  * frame at or near ZERO              -> (b). It cannot end a gene at all; the length is an
                                          artifact and the objective needs a term that does not
                                          reward never terminating.

Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "evo2_1b" / "scripts"))

from bgc_pipeline.objective import build_frame_mask, stop_completion_penalty  # noqa: E402

ARMS = {
    "base (no adapter)": None,
    "baseline": "baseline/final_adapter",
    "frame": "frame/final_adapter",
    "weighted": "weighted/final_adapter",
}


def real_cores(n: int, min_len: int, max_len: int) -> list[str]:
    """Real held-out cores. NOT generations: we want the same fixed text under every adapter, so
    the only thing that varies is the model."""
    out = []
    with open("/data2/ds85/bgcmodel_data/splits_core/test.jsonl") as f:
        for line in f:
            r = json.loads(line)
            s = r.get("sequence", "")
            if len(s) >= min_len:
                out.append(s[:max_len])
                if len(out) >= n:
                    break
    if not out:
        raise SystemExit("[stop-probe] ABORT: no real cores found")
    return out


def genes_for(seq: str) -> list[tuple[int, int, int]]:
    from bgc_pipeline.evaluation import find_orfs
    orfs = find_orfs(seq, min_aa=60)
    genes = []
    for o in orfs:
        s = getattr(o, "start", None)
        e = getattr(o, "end", None)
        strand = getattr(o, "strand", 1)
        if s is None or e is None:
            continue
        genes.append((int(s), int(e), int(strand)))
    return genes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--len", type=int, default=6000)
    ap.add_argument("--root", default="/data2/ds85/bgcmodel_runs/phase2_1b")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    seqs = real_cores(args.n, args.len, args.len)
    print(f"[stop-probe] {len(seqs)} real held-out cores, {args.len} nt each")

    # Gene calls are a property of the TEXT, so they are computed once and reused for every
    # adapter. Recomputing per arm would let a caller difference masquerade as a model difference.
    gene_sets = [genes_for(s) for s in seqs]
    ngenes = sum(len(g) for g in gene_sets)
    print(f"[stop-probe] {ngenes} genes called (shared across all arms)")
    if ngenes == 0:
        raise SystemExit("[stop-probe] ABORT: no genes called — the penalty would be 0 for every "
                         "arm and the comparison would be vacuous")

    from evo2_1b_inference import load_1b
    results = {}
    for label, rel in ARMS.items():
        adapter = None if rel is None else Path(args.root) / rel
        if adapter is not None and not adapter.exists():
            print(f"[stop-probe] {label}: no adapter at {adapter} — skipped")
            continue
        print(f"[stop-probe] loading {label} …", flush=True)
        w = load_1b(device=args.device, adapter_dir=adapter)
        vals = []
        for seq, genes in zip(seqs, gene_sets):
            ids = [int(i) for i in w.tokenizer.tokenize(seq)]
            x = torch.tensor([ids], dtype=torch.long, device=args.device)
            phase = build_frame_mask(len(ids), genes).unsqueeze(0).to(args.device)
            with torch.no_grad():
                logits = w.model(x)
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            pen = stop_completion_penalty(logits[:, :-1, :], x, phase, x)
            vals.append(float(pen))
        results[label] = vals
        print(f"[stop-probe]   mean stop-completion mass = {st.mean(vals):.5f}")
        del w
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("STOP-COMPLETION PROBABILITY MASS — same sequences, same gene calls, only the model differs")
    print("=" * 72)
    base = results.get("baseline")
    for label, vals in results.items():
        rel = ""
        if base and label != "baseline":
            rel = f"   {(st.mean(vals) / st.mean(base) - 1) * 100:+.1f}% vs baseline"
        print(f"  {label:<20} {st.mean(vals):.5f}  (median {st.median(vals):.5f}){rel}")
    if "frame" in results and base:
        fm, bm = st.mean(results["frame"]), st.mean(base)
        print()
        if fm < bm * 0.05:
            print("  ⇒ (b) DEGENERATE: the frame arm has essentially abolished in-gene stops. Its")
            print("     ORF length is an artifact. The objective needs a term that does not reward")
            print("     never terminating.")
        elif fm < bm * 0.7:
            print("  ⇒ (a) REAL, with a caveat: stops are suppressed but not abolished — the arm")
            print("     shifted the odds rather than removing the ability to end a gene.")
        else:
            print("  ⇒ the penalty barely bit. The ORF-length difference did not come from stop")
            print("     suppression, so it is either a weaker downstream effect or noise.")
    json.dump({k: v for k, v in results.items()},
              open(Path(args.root) / "stop_probe.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
