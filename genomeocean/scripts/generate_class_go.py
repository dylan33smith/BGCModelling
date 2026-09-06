#!/usr/bin/env python
"""[gen_class_go] Generate TERPENE arms from GenomeOcean, for comparison with `[P7-A0]`.

⚠️ THE PROMPT IS ONE TOKEN, AND THAT BREAKS PAIRING
---------------------------------------------------
Evo2's 200 prompts were 200 distinct `(class, taxonomy)` pairs. GenomeOcean cannot represent that
prefix at all — it tokenizes to 122 UNK of 132 ids — so its only conditioning is the atomic
`[CLS_TERPENE]` token. **All 200 generations therefore share ONE prompt and differ only by
sampling.** Consequence, recorded in the prereg as an amendment: the Evo2-vs-GenomeOcean comparison
is **UNPAIRED**, McNemar is unavailable, and Fisher's exact is the correct test.

Reuses the proven pieces of `generate_bgc_go.py`: static KV cache (the default DynamicCache
re-concatenates the whole cache every step and collapses throughput), `bad_words_ids` to suppress
`N`, and the real EOS id so generation can stop on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

BASE = ("/data2/ds85/hf_cache/hub/models--pGenomeOcean--GenomeOcean-4B/snapshots/"
        "2bed2fc3ed47c5f6955ba3e64563512c9b338dfb")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit for the un-fine-tuned floor.")
    ap.add_argument("--class-token", default=None, help="e.g. [CLS_TERPENE]; omit for uncond.")
    ap.add_argument("--cls", default=None,
                    help="Class stamped into each record's `compound_class`. Defaults to the name "
                         "inside --class-token. REQUIRED for an unconditional arm, whose records "
                         "still belong to a class-specific comparison and must not inherit a "
                         "stale label.")
    ap.add_argument("--bucket-token", default=None,
                    help="[P13] identity-bucket token (e.g. [ID_95_100]) appended AFTER the class "
                         "token, matching the training prefix order. Omit for the [P10] path.")
    ap.add_argument("--tag", default=None,
                    help="Accession prefix, e.g. p9. Defaults to the --out stem.")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed-from-jsonl", default=None,
                    help="[T9] Real cores to draw seed prefixes from (e.g. the TERPENE test split). "
                         "Omit for de novo.")
    ap.add_argument("--seed-nt", type=int, default=8,
                    help="Seed prefix length in NUCLEOTIDES. L*=8 was derived on RIPP start-codon "
                         "entropy; measured on TERPENE the profile matches (entropy saturates to "
                         "~1.9 bits by position 3 in both), so 8 transfers with justification.")
    ap.add_argument("--max-new-tokens", type=int, default=900, help="~4,500 nt at 4.974 nt/token")
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help="0 = let EOS fire on its own. Forcing a minimum would suppress the very "
                         "stop signal this model gets for free.")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cls is None:
        if args.class_token:
            args.cls = args.class_token.strip("[]").removeprefix("CLS_")
        else:
            raise SystemExit("FATAL: an unconditional arm has no class token to infer from. "
                             "Pass --cls explicitly so records are not mislabelled.")
    tag = args.tag or args.out.stem.split(".")[0]

    os.environ.setdefault("HF_HOME", "/data2/ds85/hf_cache")
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    src = args.adapter if args.adapter and (Path(args.adapter) / "tokenizer.json").exists() \
        else args.model
    tok = PreTrainedTokenizerFast.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").cuda()
    if args.adapter:
        from peft import PeftModel
        model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[gen_class_go] adapter merged: {args.adapter}")
    model.eval()

    seeds = None
    if args.seed_from_jsonl:
        import random as _r
        pool = [json.loads(l)["sequence"] for l in open(args.seed_from_jsonl)]
        pool = [x for x in pool if len(x) >= args.seed_nt]
        _r.Random(args.seed).shuffle(pool)
        seeds = [pool[i % len(pool)][: args.seed_nt].upper() for i in range(args.n)]
        print(f"[gen_class_go] SEEDED: {args.seed_nt} nt prefixes from {args.seed_from_jsonl} "
              f"({len(set(seeds))} distinct of {args.n})")

    prompt_ids = [tok.bos_token_id if tok.bos_token_id is not None else 1]
    if args.class_token:
        cid = tok.convert_tokens_to_ids(args.class_token)
        if cid is None or cid == tok.unk_token_id:
            raise SystemExit(f"[gen_class_go] FATAL: {args.class_token} is not in this tokenizer. "
                             f"The conditioning would silently be nothing.")
        prompt_ids.append(cid)
        print(f"[gen_class_go] prompt = BOS + {args.class_token}(id {cid})")
    else:
        print("[gen_class_go] prompt = BOS only (unconditional floor)")

    if args.bucket_token:
        bid = tok.convert_tokens_to_ids(args.bucket_token)
        if bid is None or bid == tok.unk_token_id:
            raise SystemExit(f"[gen_class_go] FATAL: {args.bucket_token} is not in this tokenizer. "
                             f"The fidelity conditioning would silently be nothing -- which is the "
                             f"exact failure mode Gate T0 exists to prevent.")
        prompt_ids.append(bid)
        print(f"[gen_class_go] + bucket {args.bucket_token}(id {bid}); prompt_ids={prompt_ids}")

    n_id = tok.convert_tokens_to_ids("N")
    bad = [[n_id]] if n_id is not None and n_id != tok.unk_token_id else None
    ids = torch.tensor([prompt_ids], device="cuda")
    torch.manual_seed(args.seed)

    recs, t0, done = [], time.time(), 0
    while done < args.n:
        b = min(args.batch_size, args.n - done)
        if seeds is None:
            batch_ids = ids.repeat(b, 1)
            batch_seeds = [None] * b
        else:
            # Every row carries its own seed, so the prompts differ in length. Left-pad to the
            # longest and mask the pad, rather than silently truncating anyone's seed.
            batch_seeds = seeds[done:done + b]
            rows = [prompt_ids + tok.encode(sd, add_special_tokens=False) for sd in batch_seeds]
            w = max(len(r) for r in rows)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 3
            batch_ids = torch.tensor([[pad] * (w - len(r)) + r for r in rows], device="cuda")
        with torch.no_grad():
            out = model.generate(
                batch_ids, do_sample=True, temperature=args.temperature,
                top_p=args.top_p, top_k=(args.top_k if args.top_k > 0 else 0),
                min_new_tokens=args.min_new_tokens, max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                bad_words_ids=bad, eos_token_id=tok.eos_token_id or 2,
                pad_token_id=tok.pad_token_id, cache_implementation="static",
            )
        for bi, row in enumerate(out):
            gen = row[batch_ids.shape[1]:]
            hit_eos = bool((gen == (tok.eos_token_id or 2)).any())
            seq = tok.decode(gen, skip_special_tokens=True).replace(" ", "").upper()
            junk = sum(1 for c in seq if c not in "ACGTN")
            recs.append({"sequence": seq, "length": len(seq), "hit_eos": hit_eos,
                         "n_count": seq.count("N"), "n_junk_chars": junk,
                         "accession": f"{tag}_{len(recs):04d}",
                         "compound_class": args.cls,
                         "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                                      "top_k": args.top_k,
                                      "repetition_penalty": args.repetition_penalty,
                                      "max_new_tokens": args.max_new_tokens,
                                      "min_new_tokens": args.min_new_tokens},
                         "adapter": args.adapter, "class_token": args.class_token,
                         "bucket_token": args.bucket_token,
                         "seed_nt": (args.seed_nt if seeds is not None else 0),
                         "seed_prefix": batch_seeds[bi],
                         "scored_span": ("continuation_only" if seeds is not None else "full")})
        done += b
        print(f"  {done}/{args.n} ({time.time()-t0:.0f}s)", flush=True)

    uniq = len({r["sequence"] for r in recs})
    if uniq < len(recs):
        print(f"  ⚠️ {len(recs)-uniq} duplicate sequences of {len(recs)} — effective n is {uniq}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    L = sorted(r["length"] for r in recs)
    print(f"[gen_class_go] {len(recs)} records, {uniq} unique · median {L[len(L)//2]} nt · "
          f"hit_eos {sum(r['hit_eos'] for r in recs)}/{len(recs)} · "
          f"junk chars {sum(r['n_junk_chars'] for r in recs)} · {time.time()-t0:.0f}s")
    print(f"[gen_class_go] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
