"""Train one weight-state arm (SPEC 6).

  --classes RIPP                 -> per-class adapter
  --classes TERPENE NRPS ...     -> pooled adapter over the benchmark classes

The pooled arm sees the SAME amount of each class as a per-class arm does, because equal-n
is fixed at split time (SPEC 4.4.3) -- so the contrast isolates class-exclusivity from data
volume, which is the only reason the comparison means anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgcbench.data.classmap import BENCHMARK_CLASSES
from bgcbench.model.load import load
from bgcbench.model.train import TrainConfig, train_lora

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
ADAPTERS = ROOT / "adapters"


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="evo2-1b")
    ap.add_argument("--name", required=True)
    ap.add_argument("--classes", nargs="+", default=list(BENCHMARK_CLASSES))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len-nt", type=int, default=16000)
    ap.add_argument("--balance", choices=["records", "nucleotides"], default="records",
                    help="'records' gives every record equal weight -- equal-n, but NOT "
                         "equal tokens (measured 3.4x nucleotide imbalance). "
                         "'nucleotides' weights the per-class loss so classes contribute "
                         "equally by token, which is what the pooled gradient actually sees.")
    ap.add_argument("--limit", type=int, default=0, help="cap train records (smoke tests)")
    args = ap.parse_args()

    sub = load(args.substrate, trainable=True)
    train, val = [], []
    for c in args.classes:
        train += _load(SPLITS / c / "train.jsonl")
        val += _load(SPLITS / c / "val.jsonl")
    if args.limit:
        train, val = train[:args.limit], val[:max(8, args.limit // 8)]

    cfg = TrainConfig(rank=args.rank, epochs=args.epochs, grad_accum=args.grad_accum,
                      max_len_nt=args.max_len_nt, balance=args.balance)
    out = ADAPTERS / f"{args.substrate}_{args.name}"
    print(f"training {args.name} on {sorted(args.classes)}: "
          f"{len(train)} train / {len(val)} val, rank {cfg.rank}, {cfg.epochs} epochs",
          flush=True)
    rep = train_lora(sub, train, out, cfg, val_records=val)
    print(f"\ntrainable {rep['trainable_params']:,} / {rep['total_params']:,} "
          f"= {100*rep['trainable_frac']:.3f}%")
    print(f"batching: {rep['batching']}")
    print(f"checkpoints: {rep['checkpoints']}  ->  {out}")
    print(f"best checkpoint: {rep['best_checkpoint']} (val {rep['best_val_loss']}) "
          f"| final_is_best={rep['final_is_best']}")
    if rep.get("class_weights"):
        print(f"class loss weights: "
              f"{ {k: round(v,3) for k,v in rep['class_weights'].items()} }")
    if rep["log"]:
        print("loss track:", [(d["step"], d["train_loss"], d["val_loss"])
                              for d in rep["log"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
