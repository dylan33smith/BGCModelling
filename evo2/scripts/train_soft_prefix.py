#!/usr/bin/env python
"""PER-CLASS SOFT PREFIXES — learn a synthetic exemplar instead of asserting a label.

WHY THIS AND NOT MORE STEERING. Two facts sit next to each other in this project:

  * LABEL conditioning is inert. `|COMPOUND_CLASS:NRPS|` provably does nothing —
    v2_notag == v2_tag, and the tag has no pretrained prior because Evo2's tokenizer is
    byte-level, so a LoRA must install class->sequence from scratch through a low-rank
    bottleneck.
  * EXEMPLAR conditioning WORKS. Seed a real core and the continuation is correct-class
    0.283 against a 0.067 floor, with memorization ruled out.

The model conditions on CONTENT and ignores LABELS. A soft prefix is the obvious thing in
between: **learn a synthetic exemplar** directly in embedding space, unconstrained by the
tokenizer. It is not a byte string with no meaning, and it is not an inference-time nudge into
a variable the generator does not read (which the 2026-08-10 steering programme showed fails --
the class direction can DELETE a class but never INSTALL one). It is trained, so the generator
learns to consume it.

WHAT IS TRAINED. Exactly `--n-prefix x 4096` floats. The base model and the v2 LoRA are frozen
and merged. Per class, one run, one tensor.

MECHANISM. `--n-prefix` placeholder tokens are prepended to the input, and a forward hook on
`embedding_layer` overwrites their embeddings with the learned vectors. Prepending real tokens
rather than splicing a shorter sequence keeps every length, mask and cache offset consistent --
the model sees an ordinary sequence whose first P embeddings happen to be learned.

LOSS is masked over the soft prefix AND the taxonomy tag, so only nucleotides are supervised.
This matches the project's H3 convention; absolute losses are comparable to LoRA runs only in
so far as that masking matches.

THE CONTROL THAT MATTERS is not "prefix vs no prefix" -- any trained prefix might just make
output more BGC-like. It is **prefix_X vs prefix_Y**: does the prefix trained on class X produce
class X more than the prefix trained on class Y does? Train several, then read the diagonal of
the resulting matrix against its off-diagonal (see run_soft_prefix.sh).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402
from finetune_evo2_lora import (  # noqa: E402
    IGNORE_INDEX,
    enable_block_activation_checkpointing,
)

PLACEHOLDER_ID = ord("A")   # any valid single-byte id; its embedding is overwritten anyway


def install_soft_prefix(model, prefix: torch.nn.Parameter, n_prefix: int):
    """Overwrite the first `n_prefix` embeddings with the learned vectors.

    Gated on `shape[1] > 1`, which covers both uses: during training every pass is a full
    forward, and during cached generation only the PREFILL has length > 1 (later steps emit one
    token at a time and must not be touched). Returns the hook handle.
    """
    emb = dict(model.named_modules())["embedding_layer"]

    def hook(_m, _in, out):
        if out.shape[1] <= 1:
            return out
        h = out.clone()
        h[:, :n_prefix, :] = prefix.to(h.dtype).unsqueeze(0)
        return h

    return emb.register_forward_hook(hook)


def _init_prefix(model, tok, n_prefix: int, seed_text: str, device) -> torch.Tensor:
    """Initialise from the embeddings of REAL tokens rather than from noise.

    Random init starts the prefix far outside the distribution the blocks expect and, at this
    depth, mostly produces garbage gradients for the first few hundred steps. Seeding from real
    token embeddings starts it somewhere the model can already read.
    """
    emb = dict(model.named_modules())["embedding_layer"]
    ids = [int(i) for i in tok.tokenize(seed_text)]
    while len(ids) < n_prefix:
        ids += ids
    ids = torch.tensor(ids[:n_prefix], dtype=torch.long, device=device).view(1, -1)
    with torch.no_grad():
        v = emb(ids)[0].detach().float().clone()
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768"
                                 "/checkpoints/step_1200"),
                    help="Frozen LoRA to train the prefix on top of (it supplies BGC-likeness).")
    ap.add_argument("--train", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/train.jsonl"),
                    help="TRAIN split only — val/test are the evaluation set for this experiment.")
    ap.add_argument("--val", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/val.jsonl"))
    ap.add_argument("--compound-class", required=True)
    ap.add_argument("--n-prefix", type=int, default=16,
                    help="Soft tokens. 16 x 4096 = 65k trainable floats — four orders of "
                         "magnitude below the 28.7M-parameter LoRA.")
    ap.add_argument("--max-nt", type=int, default=4096,
                    help="Nucleotides per example. The class signal is readable from the first "
                         "few kb (the probe reads it at 4096 nt), and generation is ~3 kb, so a "
                         "long context buys nothing here and costs hours.")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05,
                    help="Soft prompts tune at a MUCH higher LR than weights (a handful of "
                         "embedding-scale vectors, no weight decay coupling).")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    def load(path: Path, n: int | None = None) -> list[dict]:
        rows = [json.loads(l) for l in path.open()
                if l.strip() and json.loads(l).get("compound_class") == args.compound_class]
        rng.shuffle(rows)
        return rows[:n] if n else rows

    train = load(args.train)
    val = load(args.val, args.val_n)
    if len(train) < 50:
        raise SystemExit(f"only {len(train)} {args.compound_class} records in {args.train} — "
                         f"too few to fit even 65k parameters honestly")
    print(f"[sp] class={args.compound_class}  train={len(train)}  val={len(val)}", flush=True)

    adapter = args.adapter
    if adapter and not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
        adapter = adapter / "adapter"
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)
    model, tok = wrapper.model, wrapper.tokenizer
    for p in model.parameters():
        p.requires_grad_(False)
    n_ckpt = enable_block_activation_checkpointing(model)
    print(f"[sp] frozen base + LoRA; activation checkpointing on {n_ckpt} blocks", flush=True)

    prefix = torch.nn.Parameter(
        _init_prefix(model, tok, args.n_prefix,
                     f"|COMPOUND_CLASS:{args.compound_class}|", args.device))
    handle = install_soft_prefix(model, prefix, args.n_prefix)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if trainable:
        raise SystemExit(f"[sp] ABORT: {len(trainable)} model parameters are still trainable — "
                         f"this run would fine-tune the model, not the prefix")
    print(f"[sp] trainable: prefix only, {prefix.numel():,} floats", flush=True)

    opt = torch.optim.AdamW([prefix], lr=args.lr, weight_decay=0.0)

    def batch(rec: dict):
        """(input_ids, labels). Supervise nucleotides only: the soft prefix and the taxonomy
        tag are conditioning, exactly as the LoRA runs mask their prefix."""
        tax = rec.get("taxonomic_tag", "") or ""
        seq = (rec.get("sequence", "") or "")[: args.max_nt]
        tax_ids = [int(i) for i in tok.tokenize(tax)]
        seq_ids = [int(i) for i in tok.tokenize(seq)]
        ids = [PLACEHOLDER_ID] * args.n_prefix + tax_ids + seq_ids
        lab = [IGNORE_INDEX] * (args.n_prefix + len(tax_ids)) + seq_ids
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        y = torch.tensor([lab], dtype=torch.long, device=args.device)
        return x, y

    def loss_of(rec: dict) -> torch.Tensor:
        x, y = batch(rec)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
                               y[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)

    @torch.no_grad()
    def evaluate() -> float:
        tot, n = 0.0, 0
        for r in val:
            l = loss_of(r)
            if torch.isfinite(l):
                tot += float(l); n += 1
        return tot / max(n, 1)

    log = (args.out_dir / "train_log.jsonl").open("w")
    base_val = evaluate()
    print(f"[sp] val loss BEFORE training (prefix = class-tag embeddings): {base_val:.4f}",
          flush=True)
    best, best_step = base_val, 0
    torch.save({"prefix": prefix.detach().cpu(), "n_prefix": args.n_prefix,
                "compound_class": args.compound_class, "step": 0, "val_loss": base_val},
               args.out_dir / "prefix_best.pt")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(args.warmup, 1)) * \
            (0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.grad_accum):
            l = loss_of(train[rng.randrange(len(train))]) / args.grad_accum
            if not torch.isfinite(l):
                continue
            l.backward()
            tot += float(l)
        gn = float(torch.nn.utils.clip_grad_norm_([prefix], 1.0))
        opt.step()
        if step % 10 == 0 or step == 1:
            print(f"[sp] step {step:>4}/{args.steps}  loss {tot:.4f}  lr {lr:.4f}  "
                  f"gnorm {gn:.3f}  {(time.time()-t0)/step:.1f}s/step", flush=True)
        log.write(json.dumps({"step": step, "loss": tot, "lr": lr, "grad_norm": gn}) + "\n")
        log.flush()
        if step % args.val_every == 0 or step == args.steps:
            v = evaluate()
            improved = v < best
            print(f"[sp]   val {v:.4f} (baseline {base_val:.4f}, "
                  f"delta {v - base_val:+.4f}){'  <- best' if improved else ''}", flush=True)
            log.write(json.dumps({"step": step, "val_loss": v, "baseline": base_val}) + "\n")
            log.flush()
            if improved:
                best, best_step = v, step
                torch.save({"prefix": prefix.detach().cpu(), "n_prefix": args.n_prefix,
                            "compound_class": args.compound_class, "step": step, "val_loss": v},
                           args.out_dir / "prefix_best.pt")
    handle.remove()

    summary = {"compound_class": args.compound_class, "n_prefix": args.n_prefix,
               "baseline_val_loss": base_val, "best_val_loss": best, "best_step": best_step,
               "improvement": base_val - best, "steps": args.steps,
               "n_train": len(train), "n_val": len(val), "max_nt": args.max_nt, "lr": args.lr}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[sp] {args.compound_class}: val {base_val:.4f} -> {best:.4f} "
          f"(improvement {base_val - best:+.4f}) at step {best_step}")
    print(f"[sp] wrote {args.out_dir}/prefix_best.pt")
    if best >= base_val:
        print("[sp] WARNING: the prefix never beat its initialisation. Either the LR is wrong "
              "or the model cannot use this handle — do not read a generation result from it "
              "without saying so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
