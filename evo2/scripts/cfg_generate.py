#!/usr/bin/env python
"""CFG (classifier-free guidance) diagnostic for Evo2 BGC CLASS conditioning.

WHY. Conditioning is present but weak: correct-class generation sits near the floor
(~0.07-0.10) even though LoRA capacity is not the limiter (rank sweeps + broader
target modules were flat). CFG is the standard amplifier for "the signal is there
but the model won't commit." At each generation step we run TWO streams that share
the generated suffix but differ in prompt:
    cond   = |COMPOUND_CLASS:{cls}|{tax}   (class + taxonomy)
    uncond = {tax}                          (taxonomy only -> "class-dropped")
and sample from the extrapolated logits
    logits = uncond + w * (cond - uncond)          (w = guidance weight; w=1 -> plain cond)
`cond - uncond` isolates exactly what the class token adds; w>1 amplifies it. If
correct_class jumps under guidance, the class signal EXISTS and just needs
amplifying (-> train-with-class-dropout + CFG is worth it). If it stays flat even
at high w, the prefix carries ~no class information (-> per-class adapters).

CAVEAT. The base was NOT trained with class-dropout, so `uncond` is a different
conditional, not a true learned null. This is the near-free DIAGNOSTIC version
(no retraining): it still measures the class token's marginal, amplified. A clean
CFG would retrain with random class-dropout first. High w can also push logits
out of distribution and produce incoherent DNA -> we watch coding_density/is_bgc,
not just correct_class, across the w sweep.

MECHANICS. vortex's Generator.generate() drives `self.model(x, inference_params_dict)`
and manages seqlen_offset manually (generation.py:168-192): after a full-prompt
prefill, offset = prompt_length, then += 1 per token. We replicate that for two
independent inference-param states. We do NOT reuse the high-level generate() across
num_tokens=1 calls because on a resumed call it derives seqlen_offset from the
single new token's length, not the accumulated context (generation.py:176), which
would corrupt positions. Driving the model directly avoids that.

VALIDATION GATE. `--validate` checks that CFG at w=1 (greedy) reproduces an
INDEPENDENT non-cached full-recompute greedy, token-for-token. That certifies the
cached two-stream bookkeeping before any w>1 result is trusted.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402
from generate_bgc import (  # noqa: E402
    EOS_MARKER,
    assemble_record,
    build_prefix,
    extract_sequence,
)
from vortex.model.sample import sample  # noqa: E402

# StripedHyena carries its recurrent/attention cache across these four sub-states.
STATE_KEYS = ("mha", "hcl", "hcm", "hcs")


def _ids(tokenizer: Any, s: str, device: str) -> torch.Tensor:
    t = tokenizer.tokenize(s)
    if not torch.is_tensor(t):
        t = torch.LongTensor(list(t))
    return t.to(device).view(1, -1)


def _forward(model: Any, x: torch.Tensor, ipd: Any) -> tuple[torch.Tensor, Any]:
    """One model call with cache. Returns (last_position_logits (1,V), ipd)."""
    out = model(x, inference_params_dict=ipd)
    logits, ipd = out if isinstance(out, (tuple, list)) else (out, ipd)
    return logits[:, -1].float(), ipd


@torch.inference_mode()
def cfg_generate(
    wrapper: Any,
    cond_prompt: str,
    uncond_prompt: str,
    *,
    max_new: int,
    w: float,
    top_k: int,
    top_p: float,
    temperature: float,
    device: str,
    stop_marker: Optional[str] = EOS_MARKER,
    return_ids: bool = False,
):
    """Two-stream classifier-free-guided generation. Returns the decoded text
    (and, if return_ids, the list of generated token ids)."""
    model, tok = wrapper.model, wrapper.tokenizer

    xc, xu = _ids(tok, cond_prompt, device), _ids(tok, uncond_prompt, device)
    Lc, Lu = xc.shape[1], xu.shape[1]
    tot = max(Lc, Lu) + max_new + 2

    def fresh():
        ipd = model.initialize_inference_params(max_seqlen=tot)
        ipd["mha"].max_batch_size = 1
        return ipd

    ipd_c, ipd_u = fresh(), fresh()
    # prefill both prompts in parallel (seqlen_offset stays 0 through prefill)
    lc, ipd_c = _forward(model, xc, ipd_c)
    lu, ipd_u = _forward(model, xu, ipd_u)
    # after prefill: offset = own prompt length (mirrors generation.py:176)
    for k in STATE_KEYS:
        ipd_c[k].seqlen_offset = Lc
        ipd_u[k].seqlen_offset = Lu

    out_ids: list[int] = []
    text = ""
    for _ in range(max_new):
        logits = lu + w * (lc - lu)               # CFG extrapolation
        nxt = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)  # (1,)
        tid = int(nxt.item())
        out_ids.append(tid)
        text += tok.detokenize([tid])
        if stop_marker and stop_marker in text:
            break
        step = nxt.view(1, 1)
        lc, ipd_c = _forward(model, step, ipd_c)
        lu, ipd_u = _forward(model, step, ipd_u)
        for k in STATE_KEYS:                       # advance both states (generation.py:182-185)
            ipd_c[k].seqlen_offset += 1
            ipd_u[k].seqlen_offset += 1

    return (text, out_ids) if return_ids else text


@torch.inference_mode()
def _plain_greedy_noncached(wrapper: Any, prompt: str, k: int, device: str) -> list[int]:
    """Independent oracle: greedy decode by recomputing the FULL forward each step
    (no cache). Slow but bookkeeping-free -> a ground truth for the cached path."""
    model, tok = wrapper.model, wrapper.tokenizer
    ids = _ids(tok, prompt, device)
    out: list[int] = []
    for _ in range(k):
        o = model(ids)
        logits = o[0] if isinstance(o, (tuple, list)) else o
        nxt = logits[:, -1].float().argmax(dim=-1)
        out.append(int(nxt.item()))
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return out


def _validate(wrapper: Any, prompt: str, device: str, k: int = 24) -> bool:
    """CFG at w=1, greedy (top_k=1) must equal the non-cached greedy oracle."""
    # uncond == cond so lc==lu and logits == lc: this exercises the exact cached
    # two-stream path while reducing to pure conditional greedy.
    _, cfg_ids = cfg_generate(
        wrapper, prompt, prompt, max_new=k, w=1.0, top_k=1, top_p=0.0,
        temperature=1.0, device=device, stop_marker=None, return_ids=True,
    )
    ref_ids = _plain_greedy_noncached(wrapper, prompt, k, device)
    ok = cfg_ids == ref_ids
    first_div = next((i for i, (a, b) in enumerate(zip(cfg_ids, ref_ids)) if a != b), None)
    print(f"[cfg-validate] cached-CFG(w=1,greedy) vs non-cached oracle over {k} tokens: "
          f"{'MATCH' if ok else 'MISMATCH'}"
          + ("" if ok else f" (first divergence @ token {first_div})"), file=sys.stderr)
    if not ok:
        print(f"  cfg={cfg_ids}\n  ref={ref_ids}", file=sys.stderr)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA checkpoint dir (with adapter/) or run dir; omit for base Evo2.")
    ap.add_argument("--from-jsonl", type=Path, required=True,
                    help="Records to draw (compound_class, taxonomic_tag) prompts from.")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Restrict to these compound classes (default: all present).")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--guidance-weights", "-w", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0],
                    help="w values to sweep. w=1.0 == plain conditional (sanity anchor).")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Writes cfg_w{W}.jsonl per guidance weight (eval_suite_driver schema).")
    ap.add_argument("--validate", action="store_true",
                    help="Run the w=1 vs non-cached-oracle gate and exit non-zero on mismatch.")
    ap.add_argument("--validate-only", action="store_true", help="Only run the validation gate.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {'base Evo2 (baseline)' if args.adapter is None else args.adapter}", file=sys.stderr)
    print("Loading Evo2 + adapter (merging)...", file=sys.stderr, flush=True)
    # resolve run-dir -> checkpoint dir with adapter/
    adapter = args.adapter
    if adapter is not None:
        adapter = Path(adapter)
        if not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
            adapter = adapter / "adapter"
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)

    records = [json.loads(l) for l in args.from_jsonl.open()]
    if args.classes:
        keep = set(args.classes)
        records = [r for r in records if r.get("compound_class") in keep]

    # validation gate (uses the first available prompt)
    if args.validate or args.validate_only:
        vprompt = build_prefix(records[0]["compound_class"], records[0].get("taxonomic_tag", ""))
        ok = _validate(wrapper, vprompt, args.device)
        if not ok:
            print("[cfg] VALIDATION FAILED — cached two-stream bookkeeping is wrong; "
                  "do NOT trust w>1 results.", file=sys.stderr)
            return 3
        if args.validate_only:
            print("[cfg] validation passed.", file=sys.stderr)
            return 0

    # sample identical prompts per class (seeded)
    rng = random.Random(args.seed)
    by_class: dict[str, list[dict]] = {}
    for r in records:
        by_class.setdefault(r.get("compound_class", "UNKNOWN"), []).append(r)
    prompts: list[dict] = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        rng.shuffle(pool)
        for r in pool[: args.per_class]:
            prompts.append({"compound_class": cls, "taxonomic_tag": r.get("taxonomic_tag", "")})
    print(f"[cfg] {len(prompts)} prompts across {len(by_class)} classes; "
          f"w sweep = {args.guidance_weights}", file=sys.stderr)

    class _A:  # decoding-metadata shim for assemble_record
        temperature = args.temperature; top_k = args.top_k; top_p = args.top_p
        max_new_tokens = args.max_new_tokens; max_n_frac = 0.10

    for w in args.guidance_weights:
        out_path = args.out_dir / f"cfg_w{w:g}.jsonl"
        n_done = 0
        with out_path.open("w") as fh:
            for p in prompts:
                cls, tax = p["compound_class"], p["taxonomic_tag"]
                cond = build_prefix(cls, tax)
                uncond = tax                       # class-dropped null (taxonomy only)
                torch.manual_seed(args.seed)       # reproducible per-call sampling
                raw = cfg_generate(
                    wrapper, cond, uncond, max_new=args.max_new_tokens, w=w,
                    top_k=args.top_k, top_p=args.top_p, temperature=args.temperature,
                    device=args.device, stop_marker=EOS_MARKER,
                )
                ex = extract_sequence(raw)
                rec = assemble_record(cls, tax, ex["sequence"], ex["hit_eos"], 1, _A)
                rec["guidance_weight"] = w
                fh.write(json.dumps(rec) + "\n")
                n_done += 1
        print(f"[cfg] w={w:g}: wrote {n_done} -> {out_path}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
