#!/usr/bin/env python
"""[P8-T2] Build the TERPENE training substrate for GenomeOcean's BPE tokenizer.

WHAT THIS DOES AND DOES NOT CHANGE
----------------------------------
**Reuses `splits_class/<CLASS>` UNCHANGED** — same records, same train/val/test split, same
held-out test set as `[P7-A0]`. The ONLY difference between Phase 7 and Phase 8 is the model, and
that is the whole point of the phase: a substrate change would destroy the comparison.

What DOES change is the length filter. Evo2-1B trains at 8,192 **byte** tokens, so
`train_class_adapter.sh` drops every record over 7,992 nt. GenomeOcean is BPE at ~5 nt/token with
`max_position_embeddings` 32,768, so the same records cost far fewer tokens and **the filter
essentially stops binding**. This script measures that in tokens rather than assuming it, and
reports exactly how many records come back.

⚠️ The tokenizer AUTO-WRAPS every sequence `BOS=1 … EOS=2`, so a proper single-token EOS is trained
for free — the Evo2 `[X1]` work does not apply here. The wrap costs 2 tokens per record, counted.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
from pathlib import Path

GO_SNAPSHOT = ("/data2/ds85/hf_cache/hub/models--pGenomeOcean--GenomeOcean-4B/"
               "snapshots/2bed2fc3ed47c5f6955ba3e64563512c9b338dfb")
EVO2_BUDGET_NT = 7992          # what Evo2-1B keeps (L=8192 minus the prefix allowance)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cls", default="TERPENE")
    ap.add_argument("--splits", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_class"))
    ap.add_argument("--model", default=GO_SNAPSHOT)
    ap.add_argument("--seq-len", type=int, default=10240,
                    help="Training context in TOKENS. 10,240 is what the feasibility gate passed "
                         "at; the model's ceiling is 32,768.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/data2/ds85/hf_cache")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"cls": args.cls, "model": args.model, "seq_len_tokens": args.seq_len,
                    "evo2_budget_nt": EVO2_BUDGET_NT, "splits": {}}

    for split in ("train", "val"):
        recs = [json.loads(l) for l in (args.splits / args.cls / f"{split}.jsonl").open()]
        n = len(recs)
        nt = [len(r["sequence"]) for r in recs]
        # tokenize in batches; the tokenizer adds BOS/EOS itself, so this is the REAL cost
        toks: list[int] = []
        B = 256
        for i in range(0, n, B):
            enc = tok([r["sequence"] for r in recs[i:i + B]])["input_ids"]
            toks.extend(len(e) for e in enc)
        assert len(toks) == n, f"{split}: tokenized {len(toks)} of {n}"

        keep_go = [i for i, t in enumerate(toks) if t <= args.seq_len]
        keep_evo2 = [i for i, x in enumerate(nt) if x <= EVO2_BUDGET_NT]
        ratio = [x / t for x, t in zip(nt, toks) if t]

        out = args.out_dir / f"{split}.jsonl"
        with out.open("w") as fh:
            for i in keep_go:
                fh.write(json.dumps(recs[i]) + "\n")

        report["splits"][split] = {
            "n_total": n,
            "median_nt": int(st.median(nt)),
            "median_tokens": int(st.median(toks)),
            "max_tokens": max(toks),
            "nt_per_token_median": round(st.median(ratio), 3),
            "kept_genomeocean": len(keep_go),
            "kept_genomeocean_frac": round(len(keep_go) / n, 4),
            "kept_evo2": len(keep_evo2),
            "kept_evo2_frac": round(len(keep_evo2) / n, 4),
            "records_recovered_vs_evo2": len(keep_go) - len(keep_evo2),
            "out": str(out),
        }
        d = report["splits"][split]
        print(f"[{args.cls}/{split}] {n:,} records · median {d['median_nt']} nt = "
              f"{d['median_tokens']} tok ({d['nt_per_token_median']} nt/token) · max "
              f"{d['max_tokens']} tok")
        print(f"    Evo2-1B  keeps {d['kept_evo2']:,}/{n:,} = {d['kept_evo2_frac']:.3f} "
              f"(<= {EVO2_BUDGET_NT} nt)")
        print(f"    GenomeOcean keeps {d['kept_genomeocean']:,}/{n:,} = "
              f"{d['kept_genomeocean_frac']:.3f} (<= {args.seq_len} tokens)")
        print(f"    ⇒ RECOVERED {d['records_recovered_vs_evo2']:,} records "
              f"({d['records_recovered_vs_evo2']/n:+.1%} of the split)\n")

    # EOS sanity: the tokenizer must wrap, or [X1] would apply here after all
    probe = tok("ACGTACGTACGT")["input_ids"]
    report["tokenizer"] = {"vocab_size": tok.vocab_size,
                           "bos_id": probe[0], "eos_id": probe[-1],
                           "auto_wraps_bos_eos": probe[0] == 1 and probe[-1] == 2}
    if not report["tokenizer"]["auto_wraps_bos_eos"]:
        raise SystemExit("[P8-T2] FATAL: tokenizer did NOT auto-wrap BOS/EOS. The Phase-8 "
                         "pre-registration assumes it does (so [X1] does not apply). Refusing "
                         "to build a substrate whose EOS handling is unverified.")
    print(f"[tokenizer] vocab {tok.vocab_size} · auto-wraps BOS={probe[0]} … EOS={probe[-1]} ✓")

    args.report.write_text(json.dumps(report, indent=1))
    print(f"[P8-T2] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
