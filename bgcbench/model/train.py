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
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from bgcbench.provenance import code_version

#: All plain nn.Linear in Evo2-1B. `projections` is TransformerEngine's TELinear (21 of
#: them) and is left alone. Enumerated from the loaded model, not assumed.
EVO2_LORA_TARGETS = ["l1", "l2", "l3", "out_filter_dense", "Wqkv", "out_proj"]

#: Evo2-1B's 25 blocks, enumerated from a trained adapter rather than assumed: every block
#: carries `mlp.l1/l2/l3`; the 21 Hyena blocks carry `out_filter_dense`; the 4 attention
#: blocks (3, 10, 17, 24) carry `inner_mha_cls.Wqkv/out_proj`. 104 adapted modules in all.
EVO2_N_BLOCKS = 25
EVO2_ATTENTION_BLOCKS = (3, 10, 17, 24)

#: GenomeOcean-4B, enumerated from the loaded model: a standard 24-layer decoder, hidden
#: 3,072, with `model.layers.N.self_attn.{q,k,v,o}_proj` and
#: `model.layers.N.mlp.{gate,up,down}_proj`. Every layer carries both, so unlike Evo2 there
#: is no attention/non-attention split.
#:
#: ⚠ THE TARGET SET MIRRORS EVO2'S BY ROLE, NOT BY NAME. Evo2 adapts its MLP (`l1/l2/l3`),
#: its attention projections (`Wqkv`, `out_proj`) and its Hyena output filter
#: (`out_filter_dense`). The GenomeOcean equivalent is the attention projections plus the
#: MLP; it has no Hyena filter, which is a structural absence and is reported as one
#: (SPEC 6.5), never padded with an unrelated module to make the counts match.
GO_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
GO_N_LAYERS = 24

#: family -> (targets, n_blocks). `target_regex` uses this so a depth set means the same
#: thing on both substrates: "which blocks carry adapters".
SUBSTRATE_LORA = {
    "evo2": (EVO2_LORA_TARGETS, EVO2_N_BLOCKS, "blocks"),
    "genomeocean": (GO_LORA_TARGETS, GO_N_LAYERS, "model.layers"),
}

#: G6b depth sets. WHICH blocks carry adapters is a free parameter that SPEC 6 never
#: declared -- it defends rank and is silent on placement -- so it is swept the same way,
#: on held-out loss, never on the endpoint (SPEC 2.4).
#:
#: ⚠ These sets differ in PARAMETER COUNT as well as placement, which would normally
#: confound a depth comparison with a capacity one. G6 licenses it: held-out loss moved
#: 0.00078 nats/nt across a 16x rank range, 0.48x the within-run noise, so capacity is not
#: binding in this regime and a difference between depth sets is placement. The realised
#: trainable count is recorded for every arm regardless.
DEPTH_SETS: dict[str, tuple[int, ...]] = {
    "all": tuple(range(EVO2_N_BLOCKS)),
    "early": tuple(range(0, 8)),
    "middle": tuple(range(8, 17)),
    "late": tuple(range(17, EVO2_N_BLOCKS)),
    "attention_only": EVO2_ATTENTION_BLOCKS,
    "every_other": tuple(range(0, EVO2_N_BLOCKS, 2)),
}


def go_target_regex(layers, targets=None) -> str:
    """peft `target_modules` regex for GenomeOcean, restricted to `layers`."""
    t = targets or GO_LORA_TARGETS
    b = "|".join(str(i) for i in sorted(set(layers)))
    att = "|".join(x for x in t if x.endswith("_proj") and x[0] in "qkvo")
    mlp = "|".join(x for x in t if x in ("gate_proj", "up_proj", "down_proj"))
    parts = []
    if att:
        parts.append(rf"self_attn\.({att})")
    if mlp:
        parts.append(rf"mlp\.({mlp})")
    return rf"model\.layers\.({b})\.({'|'.join(parts)})"


def depth_sets_for(family: str) -> dict:
    """DEPTH_SETS for a substrate, so G6b means the same thing on both."""
    n = SUBSTRATE_LORA[family][1]
    third = n // 3
    return {"all": tuple(range(n)),
            "early": tuple(range(0, third)),
            "middle": tuple(range(third, 2 * third)),
            "late": tuple(range(2 * third, n)),
            "every_other": tuple(range(0, n, 2))}


def target_regex(blocks: tuple[int, ...] | list[int],
                 targets: list[str] | None = None) -> str:
    """A peft `target_modules` regex restricted to `blocks`.

    peft accepts either a list of name suffixes or a single regex, and applies
    `re.fullmatch` to each module's name (e.g. `blocks.7.mlp.l1`). A suffix list cannot
    express "these blocks only", so depth selection needs the regex form.
    """
    t = targets or EVO2_LORA_TARGETS
    b = "|".join(str(i) for i in sorted(set(blocks)))
    leaf = "|".join(
        [rf"mlp\.({'|'.join(x for x in t if x in ('l1', 'l2', 'l3'))})"]
        + ([r"out_filter_dense"] if "out_filter_dense" in t else [])
        + ([rf"inner_mha_cls\.({'|'.join(x for x in t if x in ('Wqkv', 'out_proj'))})"]
           if ("Wqkv" in t or "out_proj" in t) else [])
    )
    return rf"blocks\.({b})\.({leaf})"


def train_config_hash(cfg) -> str:
    """Training had NO frozen config and no hash: nine of twelve CLI flags appeared in no
    artifact. Arms must differ in DATA, never in optimisation, and nothing recorded whether
    they did."""
    import hashlib
    from dataclasses import asdict
    d = {k: v for k, v in asdict(cfg).items()}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class TrainConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 5e-5
    #: EARLY STOPPING replaces a guessed epoch count. Fixed epochs cannot know whether an
    #: arm converged: measured on the first six arms, three still had headroom while two
    #: had already turned over, and only luck kept the rest from being under-trained.
    #: SPEC 12.A2: chosen so the cap NEVER BINDS, leaving `patience` as the single
    #: termination rule for every arm. At 12 it bound on W2_REDOX_COFACTOR twice, whose
    #: residual slope (1.62e-3 over its last five checkpoints) is ~3x the measured
    #: trajectory-noise floor -- so one class was cut off mid-descent while the other six
    #: converged, confounding method with training budget.
    max_epochs: int = 40
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
    #: "lora" -> low-rank adapters on the linear layers (W1/W1n/W2)
    #: "offset" -> per-attention-site learned conditioner (W3), see interventions.py
    method: str = "lora"
    offset_rank: int = 16
    min_classes_per_batch: int = 2
    #: "none" -> bare sequence (SPEC 4.3 default, unchanged).
    #: "taxonomy" -> prepend the record's GTDB lineage, Evo2's native pretraining format.
    #: The prefix is LOSS-MASKED: it is context to condition on, not text to learn to emit.
    prefix: str = "none"
    seed: int = 0
    targets: list[str] = field(default_factory=lambda: list(EVO2_LORA_TARGETS))
    #: G6b: name of a DEPTH_SETS entry, or None for every block (the default all arms used
    #: before G6b existed). Recorded in the report so an arm's placement is never implicit.
    depth: str | None = None


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


def _cls(r: dict) -> str:
    """The class this record was ASSIGNED to at split time -- the directory it was loaded
    from -- and NOT `classes[0]`.

    ⚠ For a hybrid, `classes[0]` is whichever antiSMASH product happened to sort first, and
    it is routinely a class the benchmark does not contain. Measured on the pooled training
    set: 18 of 2624 records key to NRPS or OTHER under `classes[0]`. That invented two
    spurious strata, made the nucleotide-balancing denominator 6 instead of 4, and handed
    those 18 records per-record loss weights of 22.9x and 56.3x against ~0.6-0.9x for
    everything else -- so a handful of hybrids dominated whatever batch they landed in.
    Falls back to `classes[0]` so a caller that did not tag its records still works.
    """
    return r.get("split_class") or r["classes"][0]


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
            by_cls.setdefault(_cls(r), []).append(r)
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
    mixed = [len({_cls(r) for r in b}) for b in batches]
    multi = sum(1 for m in mixed if m > 1)
    return {"n_batches": len(batches),
            "median_batch_size": sorted(sizes)[len(sizes) // 2] if sizes else 0,
            "batches_with_multiple_classes": multi,
            "frac_multiclass": round(multi / max(len(batches), 1), 4),
            "note": "with micro_batch=1 mixing happens across accumulation steps, not "
                    "within a batch; the accumulated gradient is what matters"}


def _encode(sub, rec: dict, cfg: TrainConfig) -> list[int]:
    """Token ids for one record. See `_encode2` for the prefix-aware form."""
    return _encode2(sub, rec, cfg)[0]


def _encode2(sub, rec: dict, cfg: TrainConfig) -> tuple[list[int], int]:
    """(token ids, number of PREFIX tokens to exclude from the loss).

    ⚠ THE PREFIX IS CONTEXT, NOT A TARGET. Supervising it would spend adapter capacity
    learning to emit GTDB lineages, which is not the task and is not what the prior project
    did -- it masked the prefix and supervised only the sequence
    ("prefix-mask train: idx=0 length=4326 prefix=131 supervised=4195").
    """
    prefix = rec.get("tax_tag", "") if cfg.prefix == "taxonomy" else ""
    text = sub.training_text(rec["sequence"][: cfg.max_len_nt], prefix=prefix)
    if sub.family == "evo2":
        ids = [int(x) for x in sub.tokenizer.tokenize(text)]
        plen = len(sub.tokenizer.tokenize(prefix)) if prefix else 0
    else:
        ids = sub.tokenizer(text)["input_ids"]
        plen = len(sub.tokenizer(prefix)["input_ids"]) if prefix else 0
    return ids, int(plen)


def train_lora(sub, records: list[dict], out_dir: Path, cfg: TrainConfig,
               val_records: list[dict] | None = None, device: str = "cuda:0",
               resume_from: str | None = None) -> dict:
    from peft import LoraConfig, get_peft_model

    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.method == "offset":
        if resume_from:
            raise ValueError(
                "--resume-from is not supported for --method offset: it was accepted and "
                "SILENTLY DISCARDED, so a run that looked resumed started from scratch."
            )
        return _train_offset(sub, records, out_dir, cfg, val_records, device)
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
    fam = getattr(sub, "family", "evo2")
    # ⚠ TARGETS ARE PER SUBSTRATE. `cfg.targets` defaults to Evo2's module names; handing
    # those to GenomeOcean matches nothing and peft trains an adapter over zero modules --
    # which raises, but only after a model load, and would otherwise look like a config typo
    # rather than a substrate mismatch.
    if fam == "genomeocean" and cfg.targets == EVO2_LORA_TARGETS:
        cfg = replace(cfg, targets=list(GO_LORA_TARGETS))
    dsets = depth_sets_for(fam) if fam in SUBSTRATE_LORA else DEPTH_SETS
    if cfg.depth:
        if cfg.depth not in dsets:
            raise ValueError(f"unknown depth set {cfg.depth!r} for {fam}; have {sorted(dsets)}")
        blocks = dsets[cfg.depth]
        tmods = (go_target_regex(blocks, cfg.targets) if fam == "genomeocean"
                 else target_regex(blocks, cfg.targets))
        print(f"  [{fam}] depth set {cfg.depth}: blocks {list(blocks)}", flush=True)
    else:
        tmods = cfg.targets
    peft_cfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                          bias="none", target_modules=tmods)
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
            per[_cls(r)] = per.get(_cls(r), 0) + r["seq_len"]
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
            enc = [_encode2(sub, r, cfg) for r in batch]
            ids = [e[0] for e in enc]
            L = max(len(x) for x in ids)
            x = torch.full((len(ids), L), pad, dtype=torch.long, device=device)
            mask = torch.zeros((len(ids), L), dtype=torch.bool, device=device)
            for i, (seq, plen) in enumerate(enc):
                x[i, :len(seq)] = torch.tensor(seq, device=device)
                mask[i, :len(seq)] = True
                if plen:
                    mask[i, :plen] = False        # prefix is context, never a target

            logits = _unwrap(model(x))
            if logits is None:
                raise RuntimeError("could not locate logits in model output")
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            nll = -lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:]
            if cls_w:
                w = torch.tensor([cls_w.get(_cls(r), 1.0) for r in batch],
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
         "stopped_early": stopped_early, "converged": stopped_early,
         "epochs_run": (log[-1]["epoch"] + 1) if log else 0,
         "max_epochs": cfg.max_epochs}, indent=2))

    report = {"train_config": {k: v for k, v in __import__("dataclasses").asdict(cfg).items()},
              "train_config_hash": train_config_hash(cfg),
              # WHICH CODE read that config: two arms meant to differ only in data can
              # still be produced by different versions of this file. See provenance.py.
              "code_version": code_version(),
              "resumed_from": resume_from,
              "trainable_params": trainable, "total_params": total,
              "best_checkpoint": (str(best_dir) if best_dir else None),
              "best_val_loss": (best_val if best_val < float("inf") else None),
              "final_is_best": bool(log) and log[-1]["step"] == best_step,
              "trainable_frac": round(trainable / max(total, 1), 6),
              "rank": cfg.rank, "targets": cfg.targets,
              "max_epochs": cfg.max_epochs, "epochs_run": (log[-1]["epoch"] + 1) if log else 0,
              # SPEC 12.A2: a run that exhausts its epochs did NOT converge. It was
              # visible as stopped_early:false in the overnight artifacts and went unread
              # for a day because nothing named it.
              "stopped_early": stopped_early, "converged": stopped_early,
              "best_step": best_step,
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
    # STRATIFY. Taking the first `limit` records in --classes order meant the pooled arm's
    # early stopping and its SPEC 6.4 manipulation check were both computed on whichever
    # class happened to be typed first.
    by_cls: dict[str, list[dict]] = {}
    for r in records:
        by_cls.setdefault(_cls(r), []).append(r)
    picked: list[dict] = []
    if by_cls:
        per = max(1, limit // len(by_cls))
        for v in by_cls.values():
            picked.extend(v[:per])
    for r in (picked or records[:limit]):
        ids, plen = _encode2(sub, r, cfg)
        x = torch.tensor([ids], device=device)
        logits = _unwrap(model(x))
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        # ⚠ SKIP THE PREFIX. It is masked in training, so scoring it here measures the
        # model's ability to predict arbitrary GTDB text rather than BGC sequence -- and
        # early stopping and best-checkpoint selection both read this number. Measured
        # before the fix: val loss 3.00-3.36 on a prefixed arm against ~0.94 unprefixed,
        # so the checkpoint was being chosen on lineage prediction.
        nll = nll[:, plen:] if plen else nll
        tot += float(nll.mean().item()); n += 1
    model.train()
    return round(tot / max(n, 1), 5)


# --------------------------------------------------------------------------------------
# W3 -- learned per-attention-site conditioner (SPEC 6; see interventions.py for why this
# is not KV-prefix tuning on this architecture).
# --------------------------------------------------------------------------------------

def _train_offset(sub, records, out_dir, cfg, val_records, device):
    import random as _r

    from bgcbench.model.interventions import LearnedOffset, site_report

    # SEED FIRST. LoRA's path seeds before peft builds its matrices; this one constructed
    # LearnedOffset's low-rank A from the UNSEEDED global RNG and seeded afterwards, so the
    # default rank>0 run was not reproducible under the seed its own report recorded.
    rng = _r.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    base = sub.model.model if sub.family == "evo2" else sub.model
    for p in base.parameters():
        p.requires_grad_(False)            # the BASE MODEL is frozen; only the conditioner trains
    hidden = int(getattr(base.config, "hidden_size", None)
                 if not isinstance(base.config, dict) else base.config["hidden_size"])
    iv = LearnedOffset(base, hidden, rank=cfg.offset_rank).to(device)
    sites = site_report(base)
    print(f"  {sites['n_attention_sites']} attention sites of {sites['n_blocks']} blocks "
          f"(coverage {sites['coverage']}); trainable {iv.n_trainable():,}", flush=True)

    cls_w: dict[str, float] = {}
    if cfg.balance == "nucleotides":
        per: dict[str, int] = {}
        for r in records:
            per[_cls(r)] = per.get(_cls(r), 0) + r["seq_len"]
        if per:
            tgt = sum(per.values()) / len(per)
            cls_w = {c: tgt / nt for c, nt in per.items()}

    opt = torch.optim.AdamW([p for p in iv.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.95))
    batches = build_batches(records, cfg, rng)
    mixing = batch_class_mixing(batches)
    pad = 1 if sub.family == "evo2" else (sub.tokenizer.pad_token_id or 0)
    log, step = [], 0
    best_val, best_step, since = float("inf"), 0, 0
    run_loss, run_n, stopped = 0.0, 0, False

    with iv.attached():
        for ep in range(cfg.max_epochs):
            if stopped:
                break
            for bi, batch in enumerate(batches):
                enc = [_encode2(sub, r, cfg) for r in batch]
                ids = [e[0] for e in enc]
                L = max(len(x) for x in ids)
                x = torch.full((len(ids), L), pad, dtype=torch.long, device=device)
                mask = torch.zeros((len(ids), L), dtype=torch.bool, device=device)
                for i, (sq, plen) in enumerate(enc):
                    x[i, :len(sq)] = torch.tensor(sq, device=device)
                    mask[i, :len(sq)] = True
                    if plen:
                        mask[i, :plen] = False    # prefix is context, never a target
                logits = _unwrap(base(x))
                lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                nll = -lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
                m = mask[:, 1:]
                if cls_w:
                    w = torch.tensor([cls_w.get(_cls(r), 1.0) for r in batch],
                                     device=device, dtype=nll.dtype).unsqueeze(1)
                    loss = (nll * m * w).sum() / (m * w).sum().clamp(min=1)
                else:
                    loss = (nll * m).sum() / m.sum().clamp(min=1)
                (loss / cfg.grad_accum).backward()
                run_loss += float(loss.item()); run_n += 1
                if (bi + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in iv.parameters() if p.requires_grad], 1.0)
                    opt.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % cfg.eval_every == 0:
                        vl = _eval_offset(sub, base, val_records, cfg, device) \
                            if val_records else None
                        tl = round(run_loss / max(run_n, 1), 5)
                        run_loss, run_n = 0.0, 0
                        imp = vl is not None and vl < best_val - cfg.min_delta
                        if imp:
                            best_val, best_step, since = vl, step, 0
                            torch.save({"state_dict": iv.state_dict(),
                                        "rank": cfg.offset_rank, "hidden": hidden,
                                        "sites": sites}, out_dir / "best.pt")
                        else:
                            since += 1
                        log.append({"step": step, "epoch": ep, "train_loss": tl,
                                    "val_loss": vl, "improved": imp})
                        print(f"  step {step} ep{ep} train={tl} val={vl}"
                              f"{'  *best*' if imp else f'  (no gain x{since})'}",
                              flush=True)
                        if since >= cfg.patience:
                            print(f"  EARLY STOP: best val {best_val} at step {best_step}",
                                  flush=True)
                            stopped = True
                            break

    # SPEC 6.4 MANIPULATION CHECK, two-sided, AT THE CHECKPOINT GENERATION WILL USE.
    #
    # ⚠ This block sits AFTER `with iv.attached()` has exited, so the hooks are GONE here.
    # The first version of it measured `with_iv` at this indentation and called the result
    # "hooks still attached" in a comment; both of its measurements were therefore of the
    # unintervened base model. delta was 0.0 BY CONSTRUCTION and `landed` was false for
    # every conditioner that could ever be trained -- a check that cannot fail its subject
    # and cannot pass it either. Attachment is now established explicitly per measurement
    # and VERIFIED, not narrated.
    #
    # It also loads best.pt first: the arm that generates is the best checkpoint (SPEC
    # 6.4), so a check run on the final step's weights is a check on a different model
    # than the one the endpoint is read from.
    manip = None
    if val_records:
        ckpt = out_dir / "best.pt"
        at = "final"
        if ckpt.exists():
            iv.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
            at = "best"
        with iv.attached():
            if not iv.is_attached():
                raise RuntimeError("manipulation check: hooks are not attached for the "
                                   "intervened measurement")
            with_iv = _eval_offset(sub, base, val_records, cfg, device)
        if iv.is_attached():
            raise RuntimeError("manipulation check: hooks leaked past the context, so the "
                               "unintervened measurement would not be unintervened")
        without_iv = _eval_offset(sub, base, val_records, cfg, device)
        manip = {"val_loss_with_intervention": with_iv,
                 "val_loss_without_intervention": without_iv,
                 "delta": round(without_iv - with_iv, 5),
                 "landed": bool(without_iv - with_iv > cfg.min_delta),
                 "measured_at": at}

    (out_dir / "BEST").write_text(json.dumps(
        {"step": best_step, "val_loss": best_val if best_val < float("inf") else None,
         "path": str(out_dir / "best.pt") if (out_dir / "best.pt").exists() else None,
         "final_val_loss": log[-1]["val_loss"] if log else None,
         "final_is_best": bool(log) and log[-1]["step"] == best_step,
         "stopped_early": stopped, "method": "offset"}, indent=2))
    report = {"train_config": {k: v for k, v in __import__("dataclasses").asdict(cfg).items()},
              "train_config_hash": train_config_hash(cfg), "resumed_from": None,
              "code_version": code_version(),
              "method": "offset", "sites": sites,
              "trainable_params": iv.n_trainable(),
              "total_params": sum(p.numel() for p in base.parameters()),
              "trainable_frac": round(iv.n_trainable()
                                      / max(sum(p.numel() for p in base.parameters()), 1), 8),
              "max_epochs": cfg.max_epochs, "epochs_run": (log[-1]["epoch"] + 1) if log else 0,
              "stopped_early": stopped, "converged": stopped, "best_step": best_step,
              # only claim a checkpoint that exists: a run ending before its first
              # evaluation wrote none, and naming one anyway sends generation at a
              # nonexistent file
              "best_checkpoint": (str(out_dir / "best.pt")
                                  if (out_dir / "best.pt").exists() else None),
              "manipulation_check": manip,
              "best_val_loss": best_val if best_val < float("inf") else None,
              "final_is_best": bool(log) and log[-1]["step"] == best_step,
              "rank": cfg.offset_rank, "targets": ["attention_sites"],
              "n_train": len(records), "batching": mixing, "log": log,
              "balance": cfg.balance, "class_weights": cls_w}
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def _eval_offset(sub, base, records, cfg, device, limit: int = 32):
    """Held-out loss WITH the conditioner attached -- it is already attached by the caller's
    context manager, so this measures the intervened model, which is what the manipulation
    check needs."""
    by: dict[str, list[dict]] = {}
    for r in records:
        by.setdefault(_cls(r), []).append(r)
    picked: list[dict] = []
    per = max(1, limit // max(len(by), 1))
    for v in by.values():
        picked.extend(v[:per])
    tot, n = 0.0, 0
    for r in (picked or records[:limit]):
        ids, plen = _encode2(sub, r, cfg)
        x = torch.tensor([ids], device=device)
        lp = torch.log_softmax(_unwrap(base(x))[:, :-1].float(), dim=-1)
        nll = -lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        nll = nll[:, plen:] if plen else nll        # skip the prefix, as in evaluate()
        tot += float(nll.mean().item())
        n += 1
    return round(tot / max(n, 1), 5)
