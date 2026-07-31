#!/usr/bin/env python3
"""M9 — conditioning-adherence check via generative (likelihood) classification.

Does the fine-tuned model actually respond to the COMPOUND_CLASS prompt? We use
the conditional generator P(sequence | class, taxonomy) as a classifier: hold a
held-out BGC's taxonomy and sequence fixed, score the sequence's per-token
log-likelihood under EACH candidate class prefix, and check whether the TRUE
class scores highest. If conditioning "took", the true class should win.

Outputs top-1/3/5 accuracy, MRR, mean per-token margin, and per-class recall.
Optionally also scores the untouched base Evo2 (M5 baseline) so you can see how
much the LoRA fine-tune improved conditioning adherence (delta).

Usage:
  python scripts/eval_conditioning_adherence.py \
      --val /data2/ds85/bgcmodel_data/splits_curated/val.jsonl \
      --adapter /data2/ds85/bgcmodel_runs/<run>/checkpoints/best \
      --compare-base --per-class-cap 20 --score-len 8192 \
      --output eval_conditioning_adherence.json

NOTE: the scoring forward pass needs a GPU + evo2 weights; the metric
aggregation (summarize_adherence) is unit-tested independently.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def load_records(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def balanced_sample(
    records: list[dict], per_class_cap: int, rng: random.Random,
) -> list[dict]:
    """Up to per_class_cap records per compound_class (deterministic)."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r.get("compound_class", "UNKNOWN")].append(r)
    out: list[dict] = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        rng.shuffle(pool)
        out.extend(pool[:per_class_cap] if per_class_cap > 0 else pool)
    return out


def summarize_adherence(per_record: list[dict], classes: list[str]) -> dict[str, Any]:
    """Aggregate ranking metrics. Pure function (unit-tested).

    per_record: list of {"true_class": str, "scores": {class: per_token_loglik}}.
    """
    n = 0
    top1 = top3 = top5 = 0
    rr_sum = 0.0
    margin_sum = 0.0
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)

    for r in per_record:
        scores = r.get("scores") or {}
        true = r["true_class"]
        if true not in scores or len(scores) < 2:
            continue
        ranked = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        rank = ranked.index(true)  # 0-based
        n += 1
        per_class_total[true] += 1
        if rank == 0:
            top1 += 1
            per_class_correct[true] += 1
        if rank < 3:
            top3 += 1
        if rank < 5:
            top5 += 1
        rr_sum += 1.0 / (rank + 1)
        best_wrong = max(v for c, v in scores.items() if c != true)
        margin_sum += scores[true] - best_wrong
        confusion[true][ranked[0]] += 1

    denom = max(n, 1)
    return {
        "n_scored": n,
        "n_classes": len(classes),
        "random_baseline_top1": round(1.0 / max(len(classes), 1), 4),
        "top1_acc": round(top1 / denom, 4),
        "top3_acc": round(top3 / denom, 4),
        "top5_acc": round(top5 / denom, 4),
        "mrr": round(rr_sum / denom, 4),
        "mean_per_token_margin": round(margin_sum / denom, 5),
        "per_class_recall": {
            c: round(per_class_correct[c] / per_class_total[c], 4)
            for c in sorted(per_class_total)
        },
        "confusion_top1": {
            t: dict(confusion[t].most_common(3)) for t in sorted(confusion)
        },
    }


def score_model(model, tokenizer, samples, classes, args) -> list[dict]:
    """Score every sampled record under every candidate class. GPU path."""
    from evo2_inference import sequence_loglik

    per_record = []
    for i, rec in enumerate(samples):
        tax = str(rec.get("taxonomic_tag", ""))
        seq = rec.get("sequence", "")
        true_cls = rec.get("compound_class", "UNKNOWN")
        scores: dict[str, float] = {}
        for c in classes:
            prefix = f"|COMPOUND_CLASS:{c}|{tax}"
            res = sequence_loglik(
                model, tokenizer, prefix, seq, args.max_seq_len,
                device=args.device, score_len=args.score_len,
            )
            if res is not None:
                total, ntok = res
                scores[c] = total / max(ntok, 1)  # per-token log-lik (length-fair)
        per_record.append({"true_class": true_cls, "scores": scores})
        if (i + 1) % 25 == 0:
            print(f"    scored {i + 1}/{len(samples)} records", file=sys.stderr, flush=True)
    return per_record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_curated/val.jsonl"))
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA checkpoint dir (with adapter/). Omit to score base only.")
    ap.add_argument("--compare-base", action="store_true",
                    help="Also score the untouched base Evo2 (M5 baseline) for a delta.")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Candidate class set (default: all classes present in --val).")
    ap.add_argument("--per-class-cap", type=int, default=20,
                    help="Records sampled per class (0 = all). Balances the metric.")
    ap.add_argument("--score-len", type=int, default=8192,
                    help="Nucleotides of each sequence to score (speed; held fixed).")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("eval_conditioning_adherence.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the sample/class plan without loading the model.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = load_records(args.val)
    classes = args.classes or sorted({r.get("compound_class", "UNKNOWN") for r in records})
    samples = balanced_sample(records, args.per_class_cap, rng)

    print(f"Held-out records: {len(records):,}", file=sys.stderr)
    print(f"Candidate classes ({len(classes)}): {', '.join(classes)}", file=sys.stderr)
    print(f"Sampled {len(samples)} records "
          f"(<= {args.per_class_cap}/class); scoring first {args.score_len} nt each.",
          file=sys.stderr)
    print(f"Forward passes/model: ~{len(samples) * len(classes):,}", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] not loading model.", file=sys.stderr)
        return

    from evo2_inference import load_evo2_for_inference

    report: dict[str, Any] = {
        "val": str(args.val),
        "adapter": str(args.adapter) if args.adapter else None,
        "n_classes": len(classes),
        "score_len": args.score_len,
        "n_samples": len(samples),
    }

    if args.adapter is not None:
        print("Loading fine-tuned model (base + adapter)...", file=sys.stderr)
        model, tok = load_evo2_for_inference(args.adapter, device=args.device)
        report["finetuned"] = summarize_adherence(
            score_model(model, tok, samples, classes, args), classes)
        del model

    if args.compare_base or args.adapter is None:
        print("Loading base Evo2 (no adapter)...", file=sys.stderr)
        base, tok = load_evo2_for_inference(None, device=args.device)
        report["base"] = summarize_adherence(
            score_model(base, tok, samples, classes, args), classes)
        del base

    # Delta (fine-tune improvement over base), if both were run.
    if "finetuned" in report and "base" in report:
        report["delta"] = {
            k: round(report["finetuned"][k] - report["base"][k], 4)
            for k in ("top1_acc", "top3_acc", "top5_acc", "mrr", "mean_per_token_margin")
        }

    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}", file=sys.stderr)
    for key in ("base", "finetuned"):
        if key in report:
            s = report[key]
            print(f"  [{key:9s}] top1={s['top1_acc']:.3f} top3={s['top3_acc']:.3f} "
                  f"mrr={s['mrr']:.3f} margin={s['mean_per_token_margin']:+.4f} "
                  f"(random top1={s['random_baseline_top1']:.3f})", file=sys.stderr)
    if "delta" in report:
        d = report["delta"]
        print(f"  [delta    ] top1={d['top1_acc']:+.3f} mrr={d['mrr']:+.3f} "
              f"margin={d['mean_per_token_margin']:+.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
