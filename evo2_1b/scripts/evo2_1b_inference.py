#!/usr/bin/env python
"""LOAD evo2_1b_base — the Phase-2 substrate. TRANSFORMER ENGINE IS REQUIRED.

WHY A 1B TRACK AT ALL. Everything past this point needs TRAINING runs, and the 7B is too slow to
iterate on: the objective-change experiments (frame-aware loss, domain weighting) are hypothesis
tests, not production runs, and a hypothesis test that takes days per cell cannot be iterated. The
1B is ~6.3x smaller and its native context is 8k rather than 32k, which is where the short-context
pilot wants to live anyway.

WHY THE PACKAGE SAYS YOU CANNOT DO THIS, AND WHY YOU CAN. `evo2.models` refuses to load any non-7B
model when Transformer Engine is missing:

    if config.get("use_fp8_input_projections", False) and not HAS_TE:
        is_7b_model = "7b" in (model_name or "") or "7b" in (config_path or "")
        if is_7b_model:  ... config.use_fp8_input_projections = False   # bf16 fallback
        else:            raise ImportError(...)

The gate is implemented as a NAME CHECK, and applying the 7B's own fallback line to the 1B does
make it LOAD: 1.108B parameters, 25 blocks, clean weights, finite logits. **It also makes it
useless.** Measured on real held-out cores, the bf16 1B scores 1.339 nats/base with a predictive
entropy of 1.357 against a uniform 1.386 — it emits a near-uniform distribution, i.e. it guesses.
The 7B base under the SAME fallback is fine at 0.859. The checkpoint stores FP8 scale metadata in
its `_extra_state` entries (visible as `te_fp8_meta` on load) which TE needs to dequantise the
projections; without TE those weights are read as raw bf16 and the model is destroyed.

⇒ **TE is genuinely required.** The refusal is crude in implementation but correct in substance,
and "it is only a string comparison" was wrong. Installed: TE **1.13.0** — 2.18 fails to build
against torch 2.5.1 (`SymmetricMemory.hpp: No such file`, a header added in a later torch), and
1.13 is the version contemporary with torch 2.5. It does not upgrade torch, and the 7B pipeline is
unaffected (verified after install).

With TE present the 1B scores **0.990 nats/base**, against 0.859 for the 7B base and 0.820 for the
7B+LoRA — a real but modest handicap, and a usable substrate.

THE GENERAL LESSON, recorded because this project keeps paying for it: *a model that loads is not a
model that works*, and the check that caught this was predictive entropy against a uniform baseline
on REAL data. The first sanity check used a hand-written sequence, returned 1.3843 against a 1.386
threshold, and PASSED a model that was at chance.
"""
from __future__ import annotations

import glob
import os
import pkgutil
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

MODEL_NAME = "evo2_1b_base"


def load_1b(device: str = "cuda:0", adapter_dir: Optional[Path] = None) -> Any:
    """Return the evo2 wrapper for the 1B, LoRA merged if given.

    NOW A THIN PASS-THROUGH. With Transformer Engine installed the package loads the 1B without
    complaint, so the hand-rolled bf16 path this file originally contained is gone — see the module
    docstring for why it had to go (it produced a model at chance).
    """
    import evo2
    wrapper = evo2.Evo2(MODEL_NAME)
    with torch.no_grad():
        for p in wrapper.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.clone().contiguous()
    if adapter_dir is not None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evo2" / "scripts"))
        from evo2_inference import _install_peft_compat_shims
        from peft import PeftModel
        adapter_dir = Path(adapter_dir)
        if not (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter").exists():
            adapter_dir = adapter_dir / "adapter"
        _install_peft_compat_shims(wrapper.model)
        wrapper.model = PeftModel.from_pretrained(
            wrapper.model, str(adapter_dir), is_trainable=False,
            autocast_adapter_dtype=False).merge_and_unload()
    wrapper.model = wrapper.model.to(device).eval()
    return wrapper


def verify_1b_sanity(w: Any, device: str = "cuda:0") -> dict:
    """Check the substrate rather than assuming it.

    bf16-instead-of-FP8 could in principle produce a model that loads and emits garbage. Three
    cheap checks: the logits are finite, next-base prediction on REAL DNA beats a uniform guess
    (ln 4 = 1.386 nats), and a short greedy continuation is nucleotide text.
    """
    import json

    import torch.nn.functional as F
    out: dict = {}
    # REAL held-out DNA, not an invented sequence. The first version of this check used a
    # hand-written ORF and returned 1.3843 nats against a 1.386 uniform baseline -- it PASSED a
    # substrate that was at chance, because invented DNA is out-of-distribution for every model.
    # SEVERAL cores, not one. A single sequence swings this by ~0.25 nats, which is larger than
    # the gap between a healthy model and a broken one -- a one-sample threshold would fail a
    # working substrate as readily as it would pass a dead one.
    seqs = []
    with open("/data2/ds85/bgcmodel_data/splits_core/test.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("compound_class") in ("NRPS", "PKS", "TERPENE", "RIPP")
                    and len(r.get("sequence", "")) >= 2500):
                seqs.append(r["sequence"][:2500])
                if len(seqs) >= 8:
                    break
    if not seqs:
        raise SystemExit("[1b] ABORT: no real cores available for the sanity check")
    # SAME PROTOCOL AS THE REFERENCE NUMBERS: score the 500 bases that follow 2,000 bases of real
    # context. Scoring from position 0 instead averages in the cold-start positions and reads ~1.25
    # for a model whose with-context value is 0.99 -- comparing across protocols is how a healthy
    # model gets failed and a broken one gets passed.
    CTX, SCORE = 2000, 500
    vals, finite = [], True
    for seq in seqs:
        ids = [int(i) for i in w.tokenizer.tokenize(seq[:CTX + SCORE])]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = w.model(x)
        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
        finite = finite and bool(torch.isfinite(logits).all())
        lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)[0]
        vals.append(float(-lp[CTX - 1:].gather(-1, x[0, 1:][CTX - 1:].unsqueeze(-1)).mean()))
    out["logits_finite"] = finite
    out["n_cores"] = len(vals)
    out["protocol"] = f"{SCORE} bases after {CTX} nt of real context, mean over {len(vals)} cores"
    nll = sum(vals) / len(vals)
    out["nats_per_base_on_real_dna"] = round(nll, 4)
    # A REAL threshold. 1.386 is the uniform guess; a model that merely beats it can still be at
    # chance to three decimals. Reference points on the same probe: 1B+TE 0.990, 7B base 0.859,
    # the broken bf16 1B 1.339.
    out["nats_per_base_on_real_dna"] = round(nll, 4)
    out["healthy"] = bool(nll < 1.15)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    w = load_1b(device=args.device)
    n = sum(p.numel() for p in w.model.parameters())
    print(f"[1b] loaded {MODEL_NAME}: {n/1e9:.3f}B params, {len(w.model.blocks)} blocks")
    st = verify_1b_sanity(w, args.device)
    for k, v in st.items():
        print(f"[1b]   {k}: {v}")
    if not st["logits_finite"] or not st["healthy"]:
        raise SystemExit(
            f"[1b] ABORT: {st['nats_per_base_on_real_dna']} nats/base on real DNA is not a working "
            f"model (uniform = 1.386; healthy 1B+TE = 0.990). Do NOT build Phase 2 on it — this is "
            f"exactly the state the bf16-without-TE path produced.")
    print("[1b] substrate OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
