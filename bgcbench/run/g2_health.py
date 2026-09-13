"""G2: substrate likelihood health — can this model model these sequences at all?

Real held-out cores against LENGTH-MATCHED SHUFFLED sequence. A substrate that scores the
two alike has learned nothing about genomic structure, and every downstream rate from it is
uninterpretable. SPEC §5: "a substrate at chance must fail loudly."

⚠ THE UNIT IS NATS PER NUCLEOTIDE, NOT PER TOKEN, AND THIS IS THE WHOLE COMPARISON.
Evo2 is byte-level (1 token = 1 nt) so its per-token loss already is per-nucleotide.
GenomeOcean is BPE at ~4.8 nt/token: a per-token NLL of 2.0 there is 0.42 per nucleotide,
and comparing 2.0 against Evo2's 0.9 would rank the better model as the worse one by a
factor of ~4.8. Chance is ln(4) = 1.386 nats/nt for ANY substrate that has learned nothing
about DNA, which is what makes the per-nucleotide figure comparable across tokenizers.

⚠ SHUFFLING IS MONONUCLEOTIDE, WITHIN EACH RECORD. That preserves length and base
composition and destroys everything else, so the contrast isolates sequence structure from
GC content — a model could otherwise score real cores better purely by matching their GC.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

import torch

from bgcbench.data.classmap import BENCHMARK_CLASSES
from bgcbench.model.load import load
from bgcbench.provenance import code_version, corpus_max_len

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
OUT = ROOT / "gates"
CHANCE_NATS_PER_NT = math.log(4)


@torch.no_grad()
def _nll_per_nt(sub, seqs: list[str], device: str = "cuda:0") -> list[float]:
    """Mean NLL in nats per NUCLEOTIDE for each sequence."""
    import torch.nn.functional as F

    from bgcbench.model.train import _unwrap
    base = sub.model.model if sub.family == "evo2" else sub.model
    out = []
    for s in seqs:
        if len(s) < 32:
            continue
        if sub.family == "evo2":
            ids = [int(x) for x in sub.tokenizer.tokenize(s)]
        else:
            ids = sub.tokenizer(s)["input_ids"]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        lo = _unwrap(base(x)) if sub.family == "evo2" else base(x).logits
        lp = F.log_softmax(lo.float(), dim=-1)[:, :-1]
        tgt = x[:, 1:]
        nll_per_token = float(-lp.gather(-1, tgt.unsqueeze(-1)).mean())
        # per token -> per nucleotide, using this record's OWN ratio rather than a constant
        nt_per_token = len(s) / len(ids)
        out.append(nll_per_token / nt_per_token)
    return out


def _shuffle(s: str, rng: random.Random) -> str:
    c = list(s)
    rng.shuffle(c)
    return "".join(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="evo2-1b")
    ap.add_argument("--n-per-class", type=int, default=25)
    ap.add_argument("--max-len-nt", type=int, default=corpus_max_len())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sub = load(args.substrate)
    OUT.mkdir(parents=True, exist_ok=True)

    per_class, real_all, shuf_all = {}, [], []
    for C in BENCHMARK_CLASSES:
        recs = [json.loads(l) for l in open(SPLITS / C / "test.jsonl")][:args.n_per_class]
        seqs = [r["sequence"][:args.max_len_nt] for r in recs]
        real = _nll_per_nt(sub, seqs)
        shuf = _nll_per_nt(sub, [_shuffle(s, rng) for s in seqs])
        gap = (st.median(shuf) - st.median(real)) if real and shuf else 0.0
        per_class[C] = {"n": len(real),
                        "median_real_nats_per_nt": st.median(real) if real else None,
                        "median_shuffled_nats_per_nt": st.median(shuf) if shuf else None,
                        "gap": gap,
                        "real_below_chance": bool(real and st.median(real) < CHANCE_NATS_PER_NT)}
        real_all += real
        shuf_all += shuf
        print(f"  {C:16s} real {st.median(real):.4f}  shuffled {st.median(shuf):.4f}  "
              f"gap {gap:+.4f} nats/nt", flush=True)

    mr, ms = st.median(real_all), st.median(shuf_all)
    passes = bool(mr < ms and mr < CHANCE_NATS_PER_NT)
    art = {"gate": "G2", "substrate": args.substrate, "unit": "nats per NUCLEOTIDE",
           "chance_nats_per_nt": CHANCE_NATS_PER_NT,
           "median_real": mr, "median_shuffled": ms, "gap": ms - mr,
           "frac_of_chance_real": mr / CHANCE_NATS_PER_NT,
           "passes": passes,
           "criterion": ("real cores score BELOW length-matched mononucleotide-shuffled "
                         "sequence AND below chance (ln 4 = 1.386 nats/nt)"),
           "per_class": per_class, "n_per_class": args.n_per_class,
           "code_version": code_version()}
    p = OUT / f"g2_health_{args.substrate}.json"
    p.write_text(json.dumps(art, indent=1))
    print(f"\n{args.substrate}: real {mr:.4f} vs shuffled {ms:.4f} nats/nt "
          f"(chance {CHANCE_NATS_PER_NT:.4f}) — {'PASS' if passes else 'FAIL'}")
    print(f"  real is {mr / CHANCE_NATS_PER_NT:.1%} of chance")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
