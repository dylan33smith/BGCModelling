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
from bgcbench.provenance import corpus_max_len

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
ADAPTERS = ROOT / "adapters"


def _load(p: Path, cls: str) -> list[dict]:
    """Tag every record with the split it came from.

    ⚠ Without this the trainer keyed on `classes[0]`, which for a hybrid is whichever
    antiSMASH product sorted first -- so 18 of the 2624 pooled training records keyed to
    NRPS or OTHER, classes the benchmark does not contain. See `train._cls`.
    """
    out = []
    for line in open(p):
        r = json.loads(line)
        r["split_class"] = cls
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="evo2-1b")
    ap.add_argument("--name", required=True)
    ap.add_argument("--classes", nargs="+", default=list(BENCHMARK_CLASSES))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--depth", default=None,
                    help="G6b: which blocks carry adapters. One of the DEPTH_SETS names "
                         "(all/early/middle/late/attention_only/every_other). Omit for "
                         "every block, which is what every arm before G6b used.")
    ap.add_argument("--method", choices=["lora", "offset"], default="lora",
                    help="'offset' is W3: a learned per-attention-site conditioner. It is "
                         "not KV-prefix tuning -- see interventions.py for why that is not "
                         "implementable on this architecture.")
    ap.add_argument("--offset-rank", type=int, default=16,
                    help="capacity of the W3 conditioner. 0 = bare offset (7,680 params, "
                         "~1364x below LoRA, so a null would be capacity-limited).")
    ap.add_argument("--lr-schedule", choices=["linear", "cosine", "constant"],
                    default="linear",
                    help="THERE WAS NO SCHEDULE BEFORE 2026-09-14 -- lr was flat for the "
                         "whole run. 'constant' reproduces that.")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--train-epochs", type=float, default=3.0,
                    help="the PLANNED run: both the early-stopping floor AND the LR decay "
                         "horizon, so the two cannot disagree. Default 3.0 matches the prior "
                         "implementation, whose loss fell monotonically across all three "
                         "epochs. Training time scales directly with this.")
    ap.add_argument("--max-epochs", type=int, default=40,
                    help="SPEC 12.A2: high enough that the cap never binds, so early "
                         "stopping is the single termination rule for every arm.")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--resume-from", default=None,
                    help="continue training an existing adapter instead of starting over")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len-nt", type=int, default=corpus_max_len(),
                    help="training length bound. DERIVED from the manifest's built corpus "
                         "bound, not carried as a literal: the default was 16000, left "
                         "over from before the 8,192 nt reorientation and above the "
                         "model's own context. No record can exceed the build bound, so "
                         "the excess was dead configuration that still entered the hash.")
    ap.add_argument("--balance", choices=["records", "nucleotides"], default="records",
                    help="'records' gives every record equal weight -- equal-n, but NOT "
                         "equal tokens (measured 3.4x nucleotide imbalance). "
                         "'nucleotides' weights the per-class loss so classes contribute "
                         "equally by token, which is what the pooled gradient actually sees.")
    ap.add_argument("--prefix", choices=["none", "taxonomy"], default="none",
                    help="'taxonomy' prepends each record's GTDB lineage -- Evo2's native "
                         "pretraining format -- LOSS-MASKED, so it is context to condition "
                         "on rather than text the adapter learns to emit.")
    ap.add_argument("--limit", type=int, default=0, help="cap train records (smoke tests)")
    args = ap.parse_args()

    sub = load(args.substrate, trainable=True)
    # held-out loss must cover EVERY class the arm trains on. Taking the first 32 val
    # records in --classes order made the pooled arm checkpoint-selected on one class.
    train, val = [], []
    for c in args.classes:
        train += _load(SPLITS / c / "train.jsonl", c)
        val += _load(SPLITS / c / "val.jsonl", c)
    if args.prefix == "taxonomy":
        from bgcbench.data import taxonomy
        tbl = taxonomy.load_table()
        ctr, cva = taxonomy.attach(train, tbl), taxonomy.attach(val, tbl)
        print(f"lineage coverage: train {ctr['with_lineage']}/{ctr['n']} "
              f"({ctr['coverage']}), val {cva['with_lineage']}/{cva['n']}", flush=True)
        if ctr["coverage"] < 0.99:
            raise SystemExit(f"only {ctr['coverage']:.1%} of training records carry a "
                             f"lineage; a partly-prefixed arm is two arms in one")
    if args.limit:
        train, val = train[:args.limit], val[:max(8, args.limit // 8)]

    cfg = TrainConfig(rank=args.rank, max_epochs=args.max_epochs,
                      train_epochs=args.train_epochs,
                      lr_schedule=args.lr_schedule, warmup_steps=args.warmup_steps,
                      eval_every=args.eval_every, patience=args.patience,
                      grad_accum=args.grad_accum, max_len_nt=args.max_len_nt,
                      balance=args.balance, method=args.method,
                      offset_rank=args.offset_rank, prefix=args.prefix,
                      depth=args.depth)
    out = ADAPTERS / f"{args.substrate}_{args.name}"
    print(f"training {args.name} on {sorted(args.classes)}: "
          f"{len(train)} train / {len(val)} val, rank {cfg.rank}, "
          f"max {cfg.max_epochs} epochs, eval every {cfg.eval_every} steps, "
          f"patience {cfg.patience}", flush=True)
    rep = train_lora(sub, train, out, cfg, val_records=val,
                     resume_from=args.resume_from)
    print(f"\ntrainable {rep['trainable_params']:,} / {rep['total_params']:,} "
          f"= {100*rep['trainable_frac']:.3f}%")
    print(f"batching: {rep['batching']}")
    print(f"train config hash: {rep['train_config_hash']}"
          f"{'  RESUMED from ' + str(rep['resumed_from']) if rep['resumed_from'] else ''}")
    print(f"epochs run {rep['epochs_run']}/{rep['max_epochs']}  "
          f"early_stop={rep['stopped_early']}  ->  {out}")
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
