"""LoRA training for the weight-state arms (SPEC 6, 6.4a).

TERMINATOR IN THE TRAINING TEXT. Evo2's tokenizer does not append one and its generation
loop cannot stop (vortex's `stop_at_eos` is dead code -- it prints and does not break), so
a model that never sees a terminator in training will never emit one and every generation
runs to the full budget. `Substrate.training_text()` adds it where the tokenizer will not.
It is the model's own native token; nothing is invented, and it is not a conditioning
channel.

LENGTH-BUCKETED BATCHING, WITH THE GUARD SPEC 6.4a REQUIRES. Padding to the longest member
wastes compute at these length spreads. But class medians run ~1.3 kb (TERPENE) to ~9.0 kb
(BETALACTONE), so on the POOLED arm a pure length bucket is very nearly a pure CLASS
bucket, and gradient updates would alternate between class-homogeneous batches -- a
training dynamic introduced by accident on the one arm whose premise is that it sees every
class together. Batches are therefore drawn to mix classes within a length window wherever
the lengths permit, and the realised class-mixing is reported so the guard can be audited.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

#: All plain nn.Linear in Evo2-1B. `projections` is TransformerEngine's TELinear (21 of
#: them) and is left alone. Enumerated from the loaded model, not assumed.
EVO2_LORA_TARGETS = ["l1", "l2", "l3", "out_filter_dense", "Wqkv", "out_proj"]


@dataclass
class TrainConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 5e-5
    #: EARLY STOPPING replaces a guessed epoch count. Fixed epochs cannot know whether an
    #: arm converged: measured on the first six arms, three still had headroom while two
    #: had already turned over, and only luck kept the rest from being under-trained.
    max_epochs: int = 12
    eval_every: int = 25            # optimizer steps between held-out evaluations
    patience: int = 4               # evaluations without improvement before stopping
    min_delta: float = 1e-4         # smaller than this is not an improvement
    micro_batch: int = 1
    grad_accum: int = 16
    max_len_nt: int = 16000
    length_bucketed: bool = True
    #: "records" -> every record contributes equally (equal-n, but NOT equal tokens:
    #: measured 3.4x nucleotide imbalance across the five classes, and the loss is per
    #: token, so the pooled gradient is dominated by the long classes).
    #: "nucleotides" -> per-class loss weights make the classes contribute equally by token.
    balance: str = "records"
    min_classes_per_batch: int = 2
    seed: int = 0
    targets: list[str] = field(default_factory=lambda: list(EVO2_LORA_TARGETS))


def _unwrap(o):
    """vortex returns nested tuples; find the [B, T, V] tensor."""
    if torch.is_tensor(o):
        return o if o.dim() == 3 else None
    if isinstance(o, (tuple, list)):
        for x in o:
            r = _unwrap(x)
            if r is not None:
                return r
    return None


def build_batches(records: list[dict], cfg: TrainConfig,
                  rng: random.Random) -> list[list[dict]]:
    """Length-bucketed batches that still mix classes (SPEC 6.4a)."""
    if not cfg.length_bucketed:
        idx = list(range(len(records)))
        rng.shuffle(idx)
        return [[records[i] for i in idx[j:j + cfg.micro_batch]]
                for j in range(0, len(idx), cfg.micro_batch)]

    order = sorted(records, key=lambda r: r["seq_len"])
    window = max(cfg.micro_batch * 8, 32)
    batches: list[list[dict]] = []
    for i in range(0, len(order), window):
        chunk = order[i:i + window]
        # round-robin over classes inside the length window, so a batch drawn from it
        # carries several classes whenever the window does
        by_cls: dict[str, list[dict]] = {}
        for r in chunk:
            by_cls.setdefault(r["classes"][0], []).append(r)
        for v in by_cls.values():
            rng.shuffle(v)
        interleaved: list[dict] = []
        while any(by_cls.values()):
            for k in list(by_cls):
                if by_cls[k]:
                    interleaved.append(by_cls[k].pop())
        for j in range(0, len(interleaved), cfg.micro_batch):
            batches.append(interleaved[j:j + cfg.micro_batch])
    rng.shuffle(batches)
    return batches


def batch_class_mixing(batches: list[list[dict]]) -> dict:
    """Report the guard rather than assert it: with micro_batch=1 no batch can mix, and
    that is a fact about the batch size, not a failure."""
    sizes = [len(b) for b in batches]
    mixed = [len({r["classes"][0] for r in b}) for b in batches]
    multi = sum(1 for m in mixed if m > 1)
    return {"n_batches": len(batches),
            "median_batch_size": sorted(sizes)[len(sizes) // 2] if sizes else 0,
            "batches_with_multiple_classes": multi,
            "frac_multiclass": round(multi / max(len(batches), 1), 4),
            "note": "with micro_batch=1 mixing happens across accumulation steps, not "
                    "within a batch; the accumulated gradient is what matters"}


def _encode(sub, rec: dict, cfg: TrainConfig) -> list[int]:
    text = sub.training_text(rec["sequence"][: cfg.max_len_nt])
    if sub.family == "evo2":
        return [int(x) for x in sub.tokenizer.tokenize(text)]
    return sub.tokenizer(text)["input_ids"]


def train_lora(sub, records: list[dict], out_dir: Path, cfg: TrainConfig,
               val_records: list[dict] | None = None, device: str = "cuda:0",
               resume_from: str | None = None) -> dict:
    from peft import LoraConfig, get_peft_model

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    base = sub.model.model if sub.family == "evo2" else sub.model

    # peft calls model.config.to_dict(). Vortex's config is a dotdict that RETURNS None
    # for a missing key instead of raising, so `to_dict` resolves to None and peft calls
    # None(). Attach a real one rather than patching peft.
    cfgobj = getattr(base, "config", None)
    if cfgobj is not None and not callable(getattr(cfgobj, "to_dict", None)):
        def _to_dict(_c=cfgobj):
            d = getattr(_c, "__dict__", None)
            if isinstance(d, dict) and d:
                return {k: v for k, v in d.items() if not k.startswith("_")}
            return dict(_c) if isinstance(_c, dict) else {}
        try:
            cfgobj.to_dict = _to_dict
        except Exception:
            setattr(base, "config", type("C", (), {"to_dict": staticmethod(_to_dict)})())
    peft_cfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                          bias="none", target_modules=cfg.targets)
    # autocast_adapter_dtype=False: peft 0.19's cast probes torch.float8_e8m0fnu, which
    # does not exist in torch 2.5.1, and raises before any training starts. The cast is the
    # failing step, so it is skipped; adapter dtype then follows the base model's.
    if resume_from:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, resume_from, is_trainable=True,
                                          autocast_adapter_dtype=False)
        print(f"  resumed from {resume_from}", flush=True)
    else:
        model = get_peft_model(base, peft_cfg, autocast_adapter_dtype=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.95))

    cls_w: dict[str, float] = {}
    if cfg.balance == "nucleotides":
        per: dict[str, int] = {}
        for r in records:
            per[r["classes"][0]] = per.get(r["classes"][0], 0) + r["seq_len"]
        if per:
            target = sum(per.values()) / len(per)
            cls_w = {c: target / nt for c, nt in per.items()}

    batches = build_batches(records, cfg, rng)
    mixing = batch_class_mixing(batches)

    log: list[dict] = []
    step = 0
    pad = 1 if sub.family == "evo2" else (sub.tokenizer.pad_token_id or 0)
    best_val, best_step, since_improve = float("inf"), 0, 0
    run_loss, run_n = 0.0, 0
    stopped_early = False

    for ep in range(cfg.max_epochs):
        if stopped_early:
            break
        for bi, batch in enumerate(batches):
            ids = [_encode(sub, r, cfg) for r in batch]
            L = max(len(x) for x in ids)
            x = torch.full((len(ids), L), pad, dtype=torch.long, device=device)
            mask = torch.zeros((len(ids), L), dtype=torch.bool, device=device)
            for i, seq in enumerate(ids):
                x[i, :len(seq)] = torch.tensor(seq, device=device)
                mask[i, :len(seq)] = True

            logits = _unwrap(model(x))
            if logits is None:
                raise RuntimeError("could not locate logits in model output")
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            nll = -lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:]
            if cls_w:
                w = torch.tensor([cls_w.get(r["classes"][0], 1.0) for r in batch],
                                 device=device, dtype=nll.dtype).unsqueeze(1)
                loss = (nll * m * w).sum() / (m * w).sum().clamp(min=1)
            else:
                loss = (nll * m).sum() / m.sum().clamp(min=1)
            (loss / cfg.grad_accum).backward()
            # MEAN since the last evaluation, not the single batch that happened to land
            # on a checkpoint -- a one-batch train loss is far too noisy to read a curve
            # from, which is what made overfitting invisible on the first six arms.
            run_loss += float(loss.item()); run_n += 1

            if (bi + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.eval_every == 0:
                    vl = evaluate(sub, model, val_records, cfg, device) \
                        if val_records else None
                    tl = round(run_loss / max(run_n, 1), 5)
                    run_loss, run_n = 0.0, 0
                    improved = vl is not None and vl < best_val - cfg.min_delta
                    if improved:
                        best_val, best_step, since_improve = vl, step, 0
                        model.save_pretrained(str(out_dir / "best"))
                    else:
                        since_improve += 1
                    log.append({"step": step, "epoch": ep, "train_loss": tl,
                                "val_loss": vl, "improved": improved,
                                "since_improve": since_improve})
                    print(f"  step {step} ep{ep} train={tl} val={vl}"
                          f"{'  *best*' if improved else f'  (no gain x{since_improve})'}",
                          flush=True)
                    if since_improve >= cfg.patience:
                        print(f"  EARLY STOP: {cfg.patience} evaluations without "
                              f"improvement; best val {best_val} at step {best_step}",
                              flush=True)
                        stopped_early = True
                        break

    model.save_pretrained(str(out_dir / "final"))

    # BEST CHECKPOINT, NOT THE LAST. `final` is whatever the last step produced; measured
    # on the first six arms, W1 and W2_RIPP both had a final WORSE than their best, which
    # handicaps exactly those two arms. `best/` is written whenever val improves.
    best_dir = (out_dir / "best") if (out_dir / "best").exists() else None
    (out_dir / "BEST").write_text(json.dumps(
        {"step": best_step, "val_loss": best_val if best_val < float("inf") else None,
         "path": str(best_dir) if best_dir else None,
         "final_val_loss": log[-1]["val_loss"] if log else None,
         "final_is_best": bool(log) and log[-1]["step"] == best_step,
         "stopped_early": stopped_early,
         "epochs_run": (log[-1]["epoch"] + 1) if log else 0,
         "max_epochs": cfg.max_epochs}, indent=2))

    report = {"trainable_params": trainable, "total_params": total,
              "best_checkpoint": (str(best_dir) if best_dir else None),
              "best_val_loss": (best_val if best_val < float("inf") else None),
              "final_is_best": bool(log) and log[-1]["step"] == best_step,
              "trainable_frac": round(trainable / max(total, 1), 6),
              "rank": cfg.rank, "targets": cfg.targets,
              "max_epochs": cfg.max_epochs, "epochs_run": (log[-1]["epoch"] + 1) if log else 0,
              "stopped_early": stopped_early, "best_step": best_step,
              "n_train": len(records), "batching": mixing, "log": log,
              "balance": cfg.balance, "class_weights": cls_w,
              "checkpoints": len(log)}
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def evaluate(sub, model, records: list[dict], cfg: TrainConfig,
             device: str = "cuda:0", limit: int = 32) -> float:
    """Held-out loss — the SPEC 6.4 manipulation check for a weight-state arm, and the
    criterion for the G6 rank sweep (never the benchmark endpoint, SPEC 2.4)."""
    model.eval()
    tot, n = 0.0, 0
    pad = 1 if sub.family == "evo2" else (sub.tokenizer.pad_token_id or 0)
    for r in records[:limit]:
        ids = _encode(sub, r, cfg)
        x = torch.tensor([ids], device=device)
        logits = _unwrap(model(x))
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        tot += float(nll.mean().item()); n += 1
    model.train()
    return round(tot / max(n, 1), 5)
