#!/usr/bin/env python
"""[P8-T4] Fine-tune GenomeOcean-4B on one compound class.

CONDITIONING — WHY IT IS A CLASS TOKEN AND NOT EVO2'S TEXT PREFIX
----------------------------------------------------------------
Evo2 conditions on `|COMPOUND_CLASS:X|` + a GTDB taxonomy tag, byte-level, so arbitrary text is
representable. **GenomeOcean's vocabulary is BPE over DNA**: pushing that same prefix through it
yields **122 UNK of 132 ids** — the conditioning is destroyed, not merely reshaped. So the Evo2
prefix cannot be ported, and the native route (validated by `probe_finetune_feasibility.py`) is an
**atomic special token**, `[CLS_<CLASS>]`, which survives a tokenizer round trip as one id.

⚠️ **CONSEQUENCE, DECLARED: Phase 8 LOSES the taxonomic conditioning.** Evo2's prefix carried a
taxonomy tag with ~1,494 distinct values on TERPENE; GenomeOcean gets the class token alone. This is
a fifth axis on which the two arms are not matched (prereg §6). It biases **against** GenomeOcean —
strictly less conditioning information — which is the safe direction: a win despite it is a stronger
win, and a loss is confounded and must be reported as such.

⚠️ The class token is **informationally constant** inside a single-class adapter, exactly as Evo2's
class tag is. It is here so that generation has a prompt to condition on and so train/generation
stay consistent — not because one value of a one-valued field carries signal.

WHY `modules_to_save`: the new class-token row is randomly initialised. Without making
`embed_tokens`/`lm_head` trainable the tag would carry no signal at all — a silent no-op.

PADDING: records are short (TERPENE median **192 tokens**) and the context is 10,240, so padding to
the context length would waste ~98% of every step. `batch_size=1` avoids padding entirely and gives
the same effective batch as Evo2 via `grad_accum=16`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cls", default="TERPENE")
    ap.add_argument("--model", default="/data2/ds85/hf_cache/hub/models--pGenomeOcean--"
                                      "GenomeOcean-4B/snapshots/"
                                      "2bed2fc3ed47c5f6955ba3e64563512c9b338dfb")
    ap.add_argument("--data", type=Path, required=True, help="dir with train.jsonl / val.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=10240)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--max-val", type=int, default=200)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/data2/ds85/hf_cache")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, PreTrainedTokenizerFast, Trainer,
                              TrainingArguments)

    CLASS_TOKEN = f"[CLS_{args.cls}]"
    args.out.mkdir(parents=True, exist_ok=True)

    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    vocab_before = len(tok)
    tok.add_special_tokens({"additional_special_tokens": [CLASS_TOKEN]})
    ids_probe = tok.encode(CLASS_TOKEN + "ATGCATGC", add_special_tokens=False)
    if tok.convert_ids_to_tokens([ids_probe[0]])[0] != CLASS_TOKEN:
        raise SystemExit(f"[P8-T4] FATAL: {CLASS_TOKEN} was shredded by the tokenizer. That is "
                         f"the Evo2 failure mode and makes the tag a silent no-op.")
    cls_id = ids_probe[0]
    print(f"[P8-T4] {CLASS_TOKEN} is atomic, id {cls_id}; vocab {vocab_before} -> {len(tok)}")

    # ── data ────────────────────────────────────────────────────────────────────
    class DS(torch.utils.data.Dataset):
        def __init__(self, path: Path, limit: int | None = None):
            self.recs = [json.loads(l) for l in path.open()]
            if limit:
                self.recs = self.recs[:limit]

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, i):
            seq = self.recs[i]["sequence"]
            # tokenizer auto-wraps BOS ... EOS; the class token goes AFTER BOS.
            body = tok(seq)["input_ids"]
            ids = [body[0], cls_id] + body[1:]
            ids = ids[: args.seq_len]
            labels = list(ids)
            labels[0] = -100          # BOS: never a target
            labels[1] = -100          # class token: supplied at generation, never predicted
            return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                    "attention_mask": torch.ones(len(ids), dtype=torch.long)}

    train_ds, val_ds = DS(args.data / "train.jsonl"), DS(args.data / "val.jsonl", args.max_val)
    print(f"[P8-T4] train {len(train_ds):,} · val {len(val_ds):,}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation="eager")
    model.resize_token_embeddings(len(tok))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],   # the new class row must be trainable
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[P8-T4] trainable {trainable:,} / {total:,} = {100*trainable/total:.3f}%")

    targs = TrainingArguments(
        output_dir=str(args.out / "hf"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr, warmup_steps=args.warmup_steps,
        bf16=True, logging_steps=25,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[], remove_unused_columns=False, gradient_checkpointing=False,
        dataloader_num_workers=2, seed=0,
    )

    def collate(feats):
        # batch_size=1 by design (median 192 tokens vs a 10,240 context -> padding would waste
        # ~98% of every step). Guard rather than silently mis-pad if that ever changes.
        if len(feats) != 1:
            raise ValueError("this trainer assumes batch_size=1; add real padding before raising it")
        return {k: v.unsqueeze(0) for k, v in feats[0].items()}

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate)
    trainer.train()

    final = args.out / "final_adapter"
    model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    hist = [h for h in trainer.state.log_history if "eval_loss" in h]
    (args.out / "train_summary.json").write_text(json.dumps({
        "cls": args.cls, "class_token": CLASS_TOKEN, "class_token_id": cls_id,
        "model": args.model, "seq_len": args.seq_len, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "trainable_params": trainable, "total_params": total,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "global_step": trainer.state.global_step,
        "best_eval_loss": min((h["eval_loss"] for h in hist), default=None),
        "eval_history": hist,
    }, indent=1))
    print(f"[P8-T4] DONE step={trainer.state.global_step} "
          f"best_eval_loss={min((h['eval_loss'] for h in hist), default=None)}")
    print(f"[P8-T4] adapter -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
