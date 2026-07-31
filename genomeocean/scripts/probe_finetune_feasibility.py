#!/usr/bin/env python3
"""Prove out the three things we need from GenomeOcean before committing to it.

  1. LOADS + TRAINS. Does GenomeOcean-4B-bgcFM load as a plain HF
     `MistralForCausalLM` and take a LoRA adapter + a real backward pass on our
     H100, at a sequence length that covers our BGC cores?
  2. CLASS TOKEN. Its BPE vocabulary is 4,096 with only 5 special tokens, and
     `tie_word_embeddings=false`. Can we add real `[CLASS_NRPS]`-style tokens
     (resize embed_tokens + lm_head) and train them? This is the thing Evo2's
     byte-level CharLevelTokenizer structurally cannot do -- there, a class tag is
     just more nucleotide-ish bytes with no pretrained prior, which is where our
     conditioning died (docs/project_memory/decisions.md, 2026-07-21).
  3. MEMORY. What does a training step actually cost at L=10,240 tokens (~52 kb)?

Nothing here is a training run -- it is a feasibility gate. It reports facts and
exits non-zero if a gate fails.

Usage:
  python genomeocean/scripts/probe_finetune_feasibility.py --seq-len 10240
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL = "pGenomeOcean/GenomeOcean-4B-bgcFM"

# Measured on splits_core by genomeocean/scripts/analyze_tokenization.py.
BP_PER_TOKEN = 5.15

# The 22 compound classes in splits_core (docs/project_memory/progress.md).
CLASS_TOKENS = [
    "[CLS_NRPS]", "[CLS_PKS]", "[CLS_PKS_NRPS_HYBRID]", "[CLS_TERPENE]",
    "[CLS_RIPP]", "[CLS_SACCHARIDE]", "[CLS_ARYLPOLYENE]", "[CLS_BETALACTONE]",
    "[CLS_SIDEROPHORE]", "[CLS_PHOSPHONATE]", "[CLS_ECTOINE]", "[CLS_MELANIN]",
    "[CLS_HSERLACTONE]", "[CLS_BUTYROLACTONE]", "[CLS_CDPS]", "[CLS_RESORCINOL]",
    "[CLS_PHENAZINE]", "[CLS_FURAN]", "[CLS_ALKALOID]", "[CLS_NUCLEOSIDE]",
    "[CLS_BETALACTAM]", "[CLS_PUFA]",
]


def gb(x: int) -> float:
    return x / 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seq-len", type=int, default=10_240,
                    help="Tokens per training example (10,240 = GenomeOcean's trained max).")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--sweep", type=int, nargs="*", default=None,
                    help="Sweep these token lengths instead of the single --seq-len.")
    ap.add_argument("--no-modules-to-save", dest="modules_to_save",
                    action="store_false", default=True,
                    help="Skip full-copy training of embed_tokens/lm_head "
                         "(cheaper, but the new class-token rows stay frozen).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    report: dict = {"model": args.model, "seq_len": args.seq_len,
                    "batch_size": args.batch_size, "gates": {}}
    failures: list[str] = []

    # --- 1. tokenizer + class tokens -------------------------------------------
    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    vocab_before = len(tok)
    n_added = tok.add_special_tokens({"additional_special_tokens": CLASS_TOKENS})
    vocab_after = len(tok)

    # A class token must survive a round trip as ONE token, otherwise it has been
    # silently shredded into nucleotide pieces (the Evo2 failure mode).
    probe = "[CLS_NRPS]ATGCATGCATGC"
    ids = tok.encode(probe, add_special_tokens=False)
    first_id = ids[0]
    roundtrip_ok = (tok.convert_ids_to_tokens([first_id])[0] == "[CLS_NRPS]")

    report["tokenizer"] = {
        "vocab_before": vocab_before,
        "class_tokens_added": n_added,
        "vocab_after": vocab_after,
        "class_token_is_atomic": roundtrip_ok,
        "example_ids_head": ids[:6],
    }
    report["gates"]["class_token_atomic"] = roundtrip_ok
    if not roundtrip_ok:
        failures.append("class token did not survive tokenization as a single id")

    # --- 2. load model ----------------------------------------------------------
    # trust_remote_code=False on purpose: the checkpoint is a stock Mistral decoder
    # and upstream's own llm_utils.py loads it this way for causal LM. Avoiding the
    # bundled modeling_mistral.py (written against transformers 4.38) keeps us off
    # a compatibility cliff.
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation="sdpa",
    )
    n_params = sum(p.numel() for p in model.parameters())
    report["model"] = {
        "class": type(model).__name__,
        "total_params": n_params,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "vocab_size_before_resize": model.config.vocab_size,
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "max_position_embeddings": model.config.max_position_embeddings,
    }

    # --- 3. resize for the class tokens ----------------------------------------
    model.resize_token_embeddings(vocab_after)
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    report["resize"] = {
        "embed_tokens_shape": list(emb.shape),
        "lm_head_shape": list(head.shape),
        "resize_ok": emb.shape[0] == vocab_after and head.shape[0] == vocab_after,
    }
    report["gates"]["embedding_resize"] = report["resize"]["resize_ok"]
    if not report["resize"]["resize_ok"]:
        failures.append("resize_token_embeddings did not cover both embed and lm_head")

    # --- 4. gradient checkpointing (BEFORE the PEFT wrap) -----------------------
    # Order matters: enabling it on the PeftModel after wrapping does not reliably
    # reach the decoder layers, and `use_reentrant=True` silently no-ops when the
    # only trainable params are adapters whose inputs don't require grad.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    # --- 5. attach LoRA ---------------------------------------------------------
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # The new class-token rows are randomly initialised, so they must be
        # trainable or the class tag carries no signal at all.
        modules_to_save=(["embed_tokens", "lm_head"] if args.modules_to_save else None),
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["lora"] = {
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / n_params,
        # LoraConfig normalises these to sets, which json.dumps cannot encode.
        "target_modules": sorted(lora.target_modules),
        "modules_to_save": sorted(lora.modules_to_save or []),
    }

    model.cuda()
    model.config.use_cache = False
    # Required: transformers' GradientCheckpointingLayer only checkpoints when the
    # module is in training mode, so leaving the model in eval() silently makes
    # gradient_checkpointing_enable() a no-op and memory looks unimprovable.
    model.train()

    # Verify checkpointing actually landed on the decoder stack rather than
    # trusting the flag we passed in -- a silent no-op here is the difference
    # between "fits on an H100" and OOM.
    decoder = model.base_model.model.model
    gc_active = bool(getattr(decoder, "gradient_checkpointing", False))
    report["lora"]["gradient_checkpointing_active"] = gc_active
    report["gates"]["gradient_checkpointing_active"] = (
        gc_active or not args.gradient_checkpointing)
    if args.gradient_checkpointing and not gc_active:
        failures.append("gradient checkpointing requested but not active on decoder")

    # --- 6. real forward + backward, swept over sequence length -----------------
    # Same shape as the Evo2 smoke matrix (evo2/scripts/queue_h100_smoke.sh): find
    # the largest L that survives a real fwd+bwd, and record peak memory at each.
    lengths = args.sweep or [args.seq_len]
    sweep: list[dict] = []
    for L in lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok_step, err, loss_val = True, None, None
        try:
            B = args.batch_size
            # [CLS] + class token + nucleotide body, mirroring the conditioned format.
            body = torch.randint(low=9, high=vocab_before, size=(B, L - 2), device="cuda")
            cls_id = torch.full((B, 1), tok.cls_token_id, device="cuda")
            cls_class = torch.full((B, 1), first_id, device="cuda")
            input_ids = torch.cat([cls_id, cls_class, body], dim=1)

            labels = input_ids.clone()
            # Prefix masking, same intent as Evo2 H3: only the body is scored.
            labels[:, :2] = -100

            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            loss_val = float(out.loss.detach())
            model.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError as exc:
            ok_step, err = False, f"OutOfMemoryError: {str(exc)[:200]}"
        except Exception as exc:  # noqa: BLE001 - record the exact failure
            ok_step, err = False, f"{type(exc).__name__}: {exc}"

        row = {
            "seq_len_tokens": L,
            "bp_covered": int(L * BP_PER_TOKEN),
            "ok": ok_step,
            "loss": loss_val,
            "peak_gpu_gb": round(gb(torch.cuda.max_memory_allocated()), 2),
            "error": err,
        }
        sweep.append(row)
        print(f"  L={L:>6} tok (~{row['bp_covered']:>6,} bp)  "
              f"{'OK  ' if ok_step else 'OOM '}  peak {row['peak_gpu_gb']:>6.2f} GB",
              flush=True)
        del body, cls_id, cls_class, input_ids, labels
        if 'out' in dir():
            del out

    report["sweep"] = sweep
    report["train_step"] = {
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_ok_seq_len_tokens": max((r["seq_len_tokens"] for r in sweep if r["ok"]),
                                     default=None),
        "max_ok_bp": max((r["bp_covered"] for r in sweep if r["ok"]), default=None),
    }
    any_ok = any(r["ok"] for r in sweep)
    report["gates"]["train_step"] = any_ok
    if not any_ok:
        failures.append("no swept sequence length completed a fwd+bwd step")

    # ---------------------------------------------------------------------------
    print(json.dumps(report, indent=2))
    print("\n=== GATES ===")
    for k, v in report["gates"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
