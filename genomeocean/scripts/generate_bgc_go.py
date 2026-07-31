#!/usr/bin/env python3
"""Replicate GenomeOcean's zero-shot BGC generation, emitting OUR eval schema.

This mirrors what the GenomeOcean paper actually did for BGCs (Methods 4.3.4):
bgcFM is an **unconditional** BGC model -- it was fine-tuned on 1.7M deduplicated
SMC BGCs with no product-class label anywhere in the input. So "generate a T1PKS"
is not a thing you can ask it. Their pipeline is generate-massively-then-filter:
258,260 sequences -> antiSMASH -> 11,123 positive (4.3%) -> 1,459 PKS -> 1,044 T1PKS.

We reproduce the generation half here and hand the output straight to our own
antiSMASH gate (scripts/eval_suite_driver.py), so the number we get is measured on
the same instrument we use for Evo2 rather than quoted from their paper.

Upstream's prompting convention (genomeocean/llm_utils.py): every prompt is
prefixed with the literal "[CLS]" token; zero-shot means the prompt is *only*
"[CLS]". Token 8 ('N') is suppressed at the logit level. EOS is token 2.

Backends: vLLM when importable (what upstream uses, ~150x faster than Evo2), else
plain HF `generate()`. The output JSONL is identical either way.

Usage:
  # their 'creative_long' preset, 40 sequences of ~50 kb
  python genomeocean/scripts/generate_bgc_go.py \
      --num 40 --preset creative_long \
      --out genomeocean/experiments/zeroshot_run1

  # score with our suite (unconditional -> compound_class is empty, so the
  # correct_class gate is not graded; is_bgc is)
  #
  # --antismash-db IS REQUIRED. Without it antiSMASH cannot run, is_bgc falls back to the
  # coding_sanity floor alone, and `generates_bgc.rate` becomes "the fraction of generations
  # that are gene-rich, non-degenerate DNA" -- which nearly every 50 kb bgcFM generation
  # satisfies. That reads as a spectacular BGC hit rate and is not one. Likewise --pfam-hmm,
  # without which the class_markers proxy silently produces no verdict.
  python scripts/eval_suite_driver.py \
      --gen genomeocean/experiments/zeroshot_run1/gen.jsonl \
      --skip-checks protein_homology kmer_novelty \
      --pfam-hmm /data2/ds85/pfam/Pfam-A.hmm \
      --antismash-db /data2/ds85/antismash_db \
      --output genomeocean/experiments/zeroshot_run1/eval.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL_BGC = "pGenomeOcean/GenomeOcean-4B-bgcFM"
MODEL_BASE = "pGenomeOcean/GenomeOcean-4B"

# Upstream go_generate.py presets, verbatim. min_seq_len is in TOKENS; at the
# ~5.15 bp/token we measured on splits_core, 9600 tokens is ~49 kb.
PRESETS = {
    "conservative":      dict(min_seq_len=1024, temperature=0.7, repetition_penalty=1.0),
    "conservative_long": dict(min_seq_len=9600, temperature=0.7, repetition_penalty=1.0),
    "creative":          dict(min_seq_len=1024, temperature=0.9, repetition_penalty=1.2),
    "creative_long":     dict(min_seq_len=9600, temperature=0.9, repetition_penalty=1.2),
}

N_TOKEN_ID = 8      # 'N' -- suppressed so we never emit ambiguity codes
EOS_TOKEN_ID = 2    # [SEP]


def build_records(seqs: list[str], prompt: str, args) -> list[dict]:
    """Emit the {sequence, compound_class, taxonomic_tag, accession} shape that
    scripts/eval_suite_driver.py consumes.

    compound_class is deliberately EMPTY for unconditional generation: bgcFM was
    never told a class, so grading `correct_class` against one would be
    meaningless. evaluate_bgc() guards on a falsy expected_class and simply
    reports whatever class antiSMASH assigns.
    """
    recs = []
    for i, s in enumerate(seqs):
        recs.append({
            "accession": f"go_{args.preset}_{i:05d}",
            "id": f"go_{args.preset}_{i:05d}",
            "compound_class": args.expected_class,
            "taxonomic_tag": "",
            "sequence": s,
            "length": len(s),
            "source": {
                "model": args.model,
                "preset": args.preset,
                "prompt": prompt or "<zero-shot>",
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_tokens": args.min_seq_len,
                "max_tokens": args.max_seq_len,
                "seed": args.seed,
            },
        })
    return recs


def generate_vllm(args, prompt: str) -> list[str]:
    from vllm import LLM, SamplingParams
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    llm = LLM(
        model=args.model, trust_remote_code=False, seed=args.seed,
        dtype="bfloat16", max_model_len=args.max_seq_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True, skip_tokenizer_init=True,
    )
    # Upstream prepends the literal "[CLS]" to every prompt.
    ids = tok.encode("[CLS]" + prompt, add_special_tokens=False)
    params = SamplingParams(
        n=args.num, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        min_tokens=args.min_seq_len, max_tokens=args.max_seq_len,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=[EOS_TOKEN_ID], detokenize=False,
        logit_bias={N_TOKEN_ID: float("-inf")},
    )
    outs = llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=params)
    seqs = []
    for o in outs:
        for c in o.outputs:
            seqs.append(tok.decode(c.token_ids, skip_special_tokens=True)
                        .replace(" ", "").replace("\n", ""))
    return seqs


def generate_hf(args, prompt: str) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation="sdpa",
    ).cuda().eval()

    ids = torch.tensor([tok.encode("[CLS]" + prompt, add_special_tokens=False)],
                       device="cuda")
    # HF has no logit_bias; suppress 'N' with bad_words_ids instead.
    #
    # cache_implementation="static" matters a LOT here. The default DynamicCache
    # `torch.cat`s the entire KV cache every decode step; at batch 24 x 10k tokens
    # that is ~24 GB copied per step, which drops throughput from ~74 steps/s to
    # ~8 steps/s with the GPU sitting near 0% compute utilisation. Static
    # preallocates instead. (Upstream sidesteps all of this by using vLLM.)
    gen_kwargs = {}
    if args.cache_implementation != "dynamic":
        gen_kwargs["cache_implementation"] = args.cache_implementation

    seqs: list[str] = []
    remaining = args.num
    while remaining > 0:
        b = min(args.hf_batch_size, remaining)
        with torch.no_grad():
            out = model.generate(
                ids.repeat(b, 1),
                do_sample=True, temperature=args.temperature,
                top_p=args.top_p, top_k=(args.top_k if args.top_k > 0 else 0),
                min_new_tokens=args.min_seq_len, max_new_tokens=args.max_seq_len,
                repetition_penalty=args.repetition_penalty,
                bad_words_ids=[[N_TOKEN_ID]], eos_token_id=EOS_TOKEN_ID,
                pad_token_id=tok.pad_token_id,
                **gen_kwargs,
            )
        for row in out:
            gen = row[ids.shape[1]:]
            seqs.append(tok.decode(gen, skip_special_tokens=True)
                        .replace(" ", "").replace("\n", ""))
        remaining -= b
        print(f"  generated {len(seqs)}/{args.num}", flush=True)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_BGC,
                    help=f"default {MODEL_BGC}; pass {MODEL_BASE} for the un-finetuned control")
    ap.add_argument("--base-control", action="store_true",
                    help=f"shorthand for --model {MODEL_BASE}")
    ap.add_argument("--num", type=int, default=20, help="sequences to generate")
    ap.add_argument("--preset", default="creative_long", choices=sorted(PRESETS))
    ap.add_argument("--prompt", default="",
                    help="nucleotide seed; empty = zero-shot (their BGC method)")
    ap.add_argument("--expected-class", default="",
                    help="label the output with a class; leave empty for unconditional")
    ap.add_argument("--max-seq-len", type=int, default=10_240)
    ap.add_argument("--min-seq-len", type=int, default=None, help="override preset")
    ap.add_argument("--temperature", type=float, default=None, help="override preset")
    ap.add_argument("--repetition-penalty", type=float, default=None, help="override preset")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--hf-batch-size", type=int, default=4)
    ap.add_argument("--cache-implementation", default="static",
                    choices=["static", "dynamic"],
                    help="HF backend only. 'static' preallocates the KV cache; "
                         "'dynamic' re-concatenates it every step and is ~9x slower "
                         "at long context.")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args()

    if args.base_control:
        args.model = MODEL_BASE

    preset = PRESETS[args.preset]
    for k in ("min_seq_len", "temperature", "repetition_penalty"):
        if getattr(args, k) is None:
            setattr(args, k, preset[k])
    if args.min_seq_len > args.max_seq_len:
        raise SystemExit(f"--min-seq-len {args.min_seq_len} > --max-seq-len {args.max_seq_len}")

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except Exception as exc:  # noqa: BLE001
            print(f"vLLM unavailable ({type(exc).__name__}), falling back to HF: {exc}")
            backend = "hf"

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} backend={backend} preset={args.preset} "
          f"num={args.num} min_tok={args.min_seq_len} max_tok={args.max_seq_len} "
          f"temp={args.temperature} rep_pen={args.repetition_penalty}")

    t0 = time.time()
    seqs = (generate_vllm if backend == "vllm" else generate_hf)(args, args.prompt)
    elapsed = time.time() - t0

    seqs = [s for s in seqs if s]
    recs = build_records(seqs, args.prompt, args)

    gen_path = args.out / "gen.jsonl"
    with gen_path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    with (args.out / "gen.fa").open("w") as fh:
        for r in recs:
            fh.write(f">{r['accession']}\n")
            s = r["sequence"]
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")

    total_bp = sum(len(r["sequence"]) for r in recs)
    lens = sorted(len(r["sequence"]) for r in recs) or [0]
    stats = {
        "model": args.model, "backend": backend, "preset": args.preset,
        "requested": args.num, "produced": len(recs),
        "elapsed_sec": round(elapsed, 1),
        "total_bp": total_bp,
        "bp_per_sec": round(total_bp / elapsed, 1) if elapsed else None,
        "sec_per_sequence": round(elapsed / len(recs), 2) if recs else None,
        "len_min": lens[0], "len_median": lens[len(lens) // 2], "len_max": lens[-1],
        "params": {k: getattr(args, k) for k in
                   ("temperature", "repetition_penalty", "top_p", "top_k",
                    "min_seq_len", "max_seq_len", "seed")},
    }
    (args.out / "generation_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nwrote {len(recs)} sequences to {gen_path}")
    print(f"  {total_bp:,} bp in {elapsed:.1f}s "
          f"({stats['bp_per_sec']:,} bp/s, {stats['sec_per_sequence']}s/seq)")
    print(f"  length min/median/max = {lens[0]:,} / {lens[len(lens)//2]:,} / {lens[-1]:,} bp")


if __name__ == "__main__":
    main()
