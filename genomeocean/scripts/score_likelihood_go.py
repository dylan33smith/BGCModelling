#!/usr/bin/env python
"""[P13-EVL-likelihood] Does the adapter KNOW what a real target looks like, or only fail to WRITE one?

Every azole result so far measures GENERATION. Generation confounds two very different failures:

  (1) the model never learned what an azole cluster is        -> a knowledge/capacity problem
  (2) the model models them fine but its sampling mass sits elsewhere -> a MODE problem

Teacher-forcing separates them for the cost of a forward pass. We score the per-nucleotide
likelihood the adapter assigns to REAL held-out clusters of its own subclass, and compare it to:

  * the BASE model on the same records          -> how much did fine-tuning buy on real targets?
  * the adapter on its OWN generations          -> is its probability mass on real clusters, or
                                                   somewhere else it prefers?
  * the SAME two quantities for CYCLIC_LACTONE_AUTOINDUCER, the arm that reaches 1.000

That last column is what makes the azole number readable. If azole's adapter improves on real azole
by as much as cyclactone's improves on real cyclactone, then both models know their target equally
well and the difference between 1.000 and 0.031 is NOT knowledge -- it is where the sampler goes.

⚠️ Reported in BITS PER NUCLEOTIDE, never per token: the tokenizer is BPE and token counts differ
between sets, so per-token loss is not comparable across them (`terms.md` convention).
⚠️ NLL is computed over the SEQUENCE tokens only -- the BOS and class-token prefix are masked, so
the adapter is not credited for a prompt the base model never sees.
"""
import argparse, json, os, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/data2/ds85/hf_cache/hub/models--pGenomeOcean--"
                    "GenomeOcean-4B/snapshots/2bed2fc3ed47c5f6955ba3e64563512c9b338dfb")
    ap.add_argument("--adapter", default=None, help="omit for the base-model floor")
    ap.add_argument("--class-token", default=None)
    ap.add_argument("--gen", type=Path, required=True, help="jsonl with a 'sequence' field")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=10240)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/data2/ds85/hf_cache")
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    src = args.adapter if args.adapter and (Path(args.adapter) / "tokenizer.json").exists() \
        else args.base
    tok = PreTrainedTokenizerFast.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").cuda()
    if args.adapter:
        from peft import PeftModel
        model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[P13-EVL] adapter merged: {args.adapter}")
    model.eval()

    prefix = []
    if args.class_token:
        cid = tok.convert_tokens_to_ids(args.class_token)
        if cid is None or cid == tok.unk_token_id:
            sys.exit(f"[P13-EVL] FATAL: {args.class_token} not in tokenizer — the conditioning "
                     f"would silently be nothing.")
        prefix = [cid]
        print(f"[P13-EVL] prefix = BOS + {args.class_token}(id {cid})")
    else:
        print("[P13-EVL] prefix = BOS only (unconditional)")

    recs = [json.loads(l) for l in open(args.gen)]
    if args.limit:
        recs = recs[: args.limit]
    bos = tok.bos_token_id if tok.bos_token_id is not None else 1

    rows, tot_nll, tot_nt, n_trunc = [], 0.0, 0, 0
    with torch.no_grad():
        for i, r in enumerate(recs):
            seq = r["sequence"]
            if not seq:
                continue
            body = tok(seq, add_special_tokens=False)["input_ids"]
            ids = [bos] + prefix + body
            if len(ids) > args.max_tokens:
                ids = ids[: args.max_tokens]
                n_trunc += 1
            n_pre = 1 + len(prefix)
            x = torch.tensor([ids], device="cuda")
            logits = model(x).logits.float()
            # predict token t from position t-1; score only the sequence tokens
            lp = torch.log_softmax(logits[0, :-1], dim=-1)
            tgt = x[0, 1:]
            per_tok = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            seq_nll = per_tok[n_pre - 1:].sum().item()
            n_seq_tok = len(ids) - n_pre
            # nucleotides actually scored = the decoded length of the scored tokens
            nt = len(tok.decode(ids[n_pre:], skip_special_tokens=True))
            if nt == 0:
                continue
            tot_nll += seq_nll
            tot_nt += nt
            rows.append({"i": i, "accession": r.get("accession"), "n_tokens": n_seq_tok,
                         "n_nt": nt, "nll_nats": seq_nll,
                         "bits_per_nt": seq_nll / nt / 0.6931471805599453})
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(recs)}", flush=True)

    if not rows:
        sys.exit("[P13-EVL] FATAL: nothing scored.")
    agg = tot_nll / tot_nt / 0.6931471805599453
    per = sorted(x["bits_per_nt"] for x in rows)
    out = {"tag": args.tag, "gen": str(args.gen), "adapter": args.adapter,
           "class_token": args.class_token, "n_records": len(rows),
           "n_truncated": n_trunc, "total_nt": tot_nt,
           "bits_per_nt_pooled": agg,
           "bits_per_nt_median": per[len(per) // 2],
           "bits_per_nt_min": per[0], "bits_per_nt_max": per[-1],
           "per_record": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[P13-EVL] {args.tag}: n={len(rows)}  pooled {agg:.4f} bits/nt  "
          f"median {out['bits_per_nt_median']:.4f}  (truncated {n_trunc})")
    print(f"[P13-EVL] wrote {args.out}")


if __name__ == "__main__":
    main()
