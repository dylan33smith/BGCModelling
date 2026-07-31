#!/usr/bin/env python3
"""Linear-probe compound_class from GenomeOcean hidden states.

Counterpart to `evo2/scripts/class_probe.py`. Same question, same statistical protocol
(shared in `src/bgc_pipeline/linear_probe.py`), different model — so the two results are
directly comparable.

**The question.** Feed REAL BGC cores of known class through the FROZEN model as raw
nucleotides (no class tag — otherwise the probe just reads the tag back). Mean-pool the
hidden states at several layers. Fit a logistic regression to predict `compound_class`,
cross-validated, against a shuffled-label control on the same folds.

  * SEPARABLE (balanced_acc >> shuffled) -> the model already encodes class internally.
    Our conditioning failure is then a DECODING/STEERING problem, and cheap fixes are on
    the table: guided decoding against this very probe, steering vectors, soft prompts.
  * NOT SEPARABLE -> class is not represented at all. No prompt engineering can help; a
    representation has to be INSTALLED by training (per-class adapters, class tokens,
    whole-region context). Only that justifies heavy compute.

Run both `--model .../GenomeOcean-4B-bgcFM` and `--model .../GenomeOcean-4B`: bgcFM saw
1.7M BGCs, the base model did not, so a gap between them says the BGC fine-tune built
BGC-relevant structure even without labels.

Note on comparability with the Evo2 probe: `--max-nt` is in NUCLEOTIDES, deliberately.
Evo2 sees 4,096 nt as 4,096 tokens; GenomeOcean sees the same 4,096 nt as ~795 BPE
tokens. Holding the biological input constant is the right control.

Usage:
  python genomeocean/scripts/class_probe_go.py \
      --from-jsonl /data2/ds85/bgcmodel_data/splits_core/val.jsonl \
      --out genomeocean/experiments/class_probe_bgcfm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bgc_pipeline.linear_probe import probe, sample_cores  # noqa: E402

MODEL_BGC = "pGenomeOcean/GenomeOcean-4B-bgcFM"


@torch.inference_mode()
def embed(model, tok, data, layers, device, batch_size):
    """Mean-pooled hidden states at each requested layer.

    `output_hidden_states=True` returns len(layers)+1 tensors: index 0 is the embedding
    output, index i is the output of decoder layer i-1. We pool over real tokens only,
    using the attention mask, so right-padding cannot drag the mean toward the pad
    embedding.
    """
    X: dict[int, list[np.ndarray]] = {i: [] for i in layers}
    y: list[str] = []
    for start in range(0, len(data), batch_size):
        chunk = data[start:start + batch_size]
        enc = tok([s for _, s in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=10_240)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        m = mask.unsqueeze(-1).to(out.hidden_states[0].dtype)
        denom = m.sum(dim=1).clamp(min=1)
        for i in layers:
            pooled = (out.hidden_states[i] * m).sum(dim=1) / denom
            X[i].append(pooled.float().cpu().numpy())
        y.extend(c for c, _ in chunk)
        done = min(start + batch_size, len(data))
        if done % 200 < batch_size:
            print(f"[probe]   embedded {done}/{len(data)}", file=sys.stderr, flush=True)
    return {i: np.vstack(X[i]) for i in layers}, np.array(y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_BGC)
    ap.add_argument("--from-jsonl", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--min-class", type=int, default=40)
    ap.add_argument("--max-nt", type=int, default=4096,
                    help="NUCLEOTIDES per core (not tokens) — matches the Evo2 probe.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="hidden_states indices; default every 4th + last")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target", choices=["compound_class", "phylum"],
                    default="compound_class",
                    help="TAXON CONTROL: 'phylum' probes taxonomy from the same "
                         "activations. A class probe only means something if it beats it.")
    ap.add_argument("--restrict-phylum", default=None,
                    help="TAXON CONTROL: probe class WITHIN one phylum, holding taxonomy "
                         "roughly constant. Survives => real class signal.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    data, classes = sample_cores(args.from_jsonl, args.per_class, args.min_class,
                                 args.max_nt, args.seed, label_field=args.target,
                                 restrict_phylum=args.restrict_phylum)
    if not classes:
        raise SystemExit("no label had >= --min-class records; loosen the filters")
    print(f"[probe] target={args.target}"
          f"{' within ' + args.restrict_phylum if args.restrict_phylum else ''}: "
          f"{len(data)} cores across {len(classes)} labels "
          f"(chance≈{1/len(classes):.3f}): {classes}", file=sys.stderr)

    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = "[PAD]"
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation="sdpa").to(args.device).eval()
    model.config.use_cache = False

    n_layers = model.config.num_hidden_layers
    layers = args.layers or sorted(set(list(range(0, n_layers + 1, 4)) + [n_layers]))
    print(f"[probe] model {args.model}: {n_layers} layers; probing hidden_states {layers}",
          file=sys.stderr)

    embs, y = embed(model, tok, data, layers, args.device, args.batch_size)

    rows = []
    for i in layers:
        acc, accsh, per_class = probe(embs[i], y, args.seed, classes)
        rows.append({"layer": i, "balanced_acc": round(acc, 3),
                     "shuffled_acc": round(accsh, 3), "gap": round(acc - accsh, 3),
                     "per_class_recall": per_class})
        print(f"[probe] layer {i:2d}: balanced_acc {acc:.3f}  "
              f"(shuffled {accsh:.3f}, chance {1/len(classes):.3f})",
              file=sys.stderr, flush=True)

    best = max(rows, key=lambda r: r["balanced_acc"])
    out = {"model": args.model, "target": args.target,
           "restrict_phylum": args.restrict_phylum,
           "n": len(data), "n_classes": len(classes),
           "classes": classes, "chance": round(1 / len(classes), 3),
           "max_nt": args.max_nt, "best_layer": best["layer"],
           "best_balanced_acc": best["balanced_acc"],
           "best_shuffled_acc": best["shuffled_acc"], "layers": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"[probe] BEST layer {best['layer']}: balanced_acc {best['balanced_acc']} "
          f"vs chance {1/len(classes):.3f} / shuffled {best['shuffled_acc']}  -> {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
