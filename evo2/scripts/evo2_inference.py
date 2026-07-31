#!/usr/bin/env python3
"""Shared Evo2 inference primitives: load (base + optional LoRA adapter) and score.

Factored out of scripts/finetune_evo2_lora.py so evaluation/inference tooling
(the M9 conditioning-adherence check; later the C3 generation script) can load
the fine-tuned model and run forward passes without pulling in DeepSpeed/training
machinery. The model-loading mirrors the proven training loader, including the
peft-0.19 compatibility shims.

NOTE: the forward-pass / loading paths require a GPU + the evo2 weights and have
not been smoke-tested headless here; the pure-logic consumers (metric
aggregation) are unit-tested separately.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

# Reuse constants/helpers from the training script (importing it does not run
# main(); it is guarded by __name__ == "__main__").
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_evo2_lora import (  # noqa: E402
    EVO2_MODEL_NAME,
    assert_pad_token_safe,
    count_prefix_tokens,
)


def _install_peft_compat_shims(model: Any) -> None:
    """The three peft-0.19 / Evo2 / transformers-4.46 shims from the trainer."""
    # Evo2's vortex dotdict returns None for missing attribute lookups; peft calls
    # model.config.to_dict().
    try:
        model.config["to_dict"] = lambda: {
            k: v for k, v in model.config.items() if not callable(v)
        }
    except Exception:
        pass
    # peft 0.19 imports transformers.integrations.tensor_parallel (only in
    # transformers >= 4.50; we pin 4.46.3). Provide a stub so the import succeeds;
    # the TP logic never fires on a single GPU.
    name = "transformers.integrations.tensor_parallel"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.ALL_PARALLEL_STYLES = {}
        stub.ColwiseParallel = None
        stub.EmbeddingParallel = None
        stub.RowwiseParallel = None
        stub.gather_state_dict_for_save = None
        sys.modules[name] = stub


def load_evo2_for_inference(
    adapter_dir: Optional[Path] = None,
    device: str = "cuda",
    eval_mode: bool = True,
) -> tuple[Any, Any]:
    """Load Evo2 7B and (optionally) overlay a trained LoRA adapter, for inference.

    adapter_dir: a checkpoint ``adapter/`` directory (peft format). If None, the
    untouched base model is returned — useful as the M5 baseline.
    Returns (model, tokenizer).
    """
    import evo2

    evo_wrapper = evo2.Evo2(EVO2_MODEL_NAME)
    model = evo_wrapper.model
    tokenizer = evo_wrapper.tokenizer
    assert_pad_token_safe(tokenizer)

    # Same tensor hygiene as the trainer: make params contiguous and clone any
    # inference-mode tensors (harmless for forward; keeps parity with training).
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_contiguous():
                p.data = p.data.clone().contiguous()

    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir)
        if not (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter").exists():
            adapter_dir = adapter_dir / "adapter"
        _install_peft_compat_shims(model)
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, str(adapter_dir), is_trainable=False, autocast_adapter_dtype=False,
        )

    model = model.to(device)
    if eval_mode:
        model.eval()
    return model, tokenizer


def load_evo2_wrapper_for_inference(
    adapter_dir: Optional[Path] = None,
    device: str = "cuda",
) -> Any:
    """Load the Evo2 *wrapper* (for its efficient `.generate()`), with the LoRA
    adapter MERGED into the base weights so generation uses the fine-tuned model.

    Unlike `load_evo2_for_inference` (which returns a PeftModel for scoring), this
    returns the `evo2.Evo2` wrapper whose `.model` is a plain StripedHyena with the
    adapter baked in — so `wrapper.generate(...)` (cached/efficient vortex
    generation) runs the fine-tuned model. If adapter_dir is None, the untouched
    base model is returned (M5 generation baseline).
    """
    import evo2

    wrapper = evo2.Evo2(EVO2_MODEL_NAME)
    with torch.no_grad():
        for p in wrapper.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.clone().contiguous()

    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir)
        if not (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter").exists():
            adapter_dir = adapter_dir / "adapter"
        _install_peft_compat_shims(wrapper.model)
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(
            wrapper.model, str(adapter_dir), is_trainable=False, autocast_adapter_dtype=False,
        )
        # Merge LoRA deltas into the base Linear weights and drop the peft wrapper,
        # leaving a plain adapted StripedHyena that vortex generation can use.
        wrapper.model = peft_model.merge_and_unload()

    wrapper.model = wrapper.model.to(device)
    wrapper.model.eval()
    return wrapper


def _to_id_list(tokens: Any) -> list[int]:
    if isinstance(tokens, (list, tuple)):
        return list(tokens)
    return [int(t) for t in tokens]


@torch.no_grad()
def sequence_loglik(
    model: Any,
    tokenizer: Any,
    prefix: str,
    sequence: str,
    max_seq_len: int,
    device: str = "cuda",
    score_len: Optional[int] = None,
) -> Optional[tuple[float, int]]:
    """Total log-likelihood the model assigns to ``sequence`` given ``prefix``.

    Mirrors training: causal next-token CE, scored ONLY over the sequence tokens
    (the prefix is conditioning, not supervised). ``score_len`` optionally caps
    how many nucleotides of the sequence are scored (speed; comparable across
    candidate prefixes as long as it is held fixed). Returns (sum_logprob,
    n_seq_tokens), or None if there are no sequence tokens to score.
    """
    sub = sequence if score_len is None else sequence[:score_len]
    text = prefix + sub
    ids = _to_id_list(tokenizer.tokenize(text))[:max_seq_len]
    prefix_n = count_prefix_tokens(tokenizer, prefix)
    if prefix_n >= len(ids):
        return None

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids)
    logits = out[0] if isinstance(out, (tuple, list)) else out

    # Next-token prediction: logits at position i predict token i+1.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits.float(), dim=-1)
    tok_lp = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)[0]  # (L-1,)

    # A predicted position i (0-based over shift) corresponds to original token
    # i+1; it is a SEQUENCE token iff (i+1) >= prefix_n.
    positions = torch.arange(1, input_ids.shape[1], device=device)
    seq_mask = positions >= prefix_n
    total = float(tok_lp[seq_mask].sum().item())
    n = int(seq_mask.sum().item())
    return total, n
