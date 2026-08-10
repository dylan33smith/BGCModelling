#!/usr/bin/env python
"""Generate with a trained SOFT PREFIX — de novo, taxonomy only, no seed.

This is the actual project goal: ask for a class and get that class, from nothing. Seeding is
excluded on purpose. Seeded generation already works (correct_class 0.283) but the class comes
from the exemplar, so a seeded soft-prefix result could not be attributed to the prefix.

THE CROSS-CLASS MATRIX IS THE EXPERIMENT. Running one prefix and comparing it to "no prefix"
cannot distinguish "this prefix installs NRPS" from "any trained prefix makes output more
BGC-like". So every prefix is run against every target class: generate with prefix_X, then score
the output for class X, Y, Z... The claim is about the DIAGONAL standing above its OWN column
(does prefix_X produce X more than prefix_Y does?), which is internally controlled -- no
shuffled-label arm required, because the off-diagonal prefixes are themselves the controls.

`--no-prefix-arm` adds the unconditioned floor for scale. It is context, not the comparison.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402
from generate_bgc import _gen_sequences, assemble_record, extract_sequence  # noqa: E402
from train_soft_prefix import PLACEHOLDER_ID, install_soft_prefix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768"
                                 "/checkpoints/step_1200"))
    ap.add_argument("--prefix-pt", type=Path, default=None,
                    help="prefix_best.pt from train_soft_prefix.py. Omit for the no-prefix floor.")
    ap.add_argument("--from-jsonl", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/valtest_eval.jsonl"),
                    help="Source of real taxonomic tags. HELD OUT — the prefix was trained on "
                         "train.jsonl, so the taxa it is prompted with here are unseen.")
    ap.add_argument("--tag-class", default=None,
                    help="Comma list of classes to draw taxonomy tags from. GIVE THE SAME LIST "
                         "TO EVERY ARM: with a fixed --seed the tags are then IDENTICAL across "
                         "arms, so the only thing that varies is the prefix and the comparison "
                         "is paired per taxon. Using each prefix's own class here instead would "
                         "confound the prefix with the taxon it was prompted with. Defaults to "
                         "the prefix's own class (single-arm use only).")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    args = ap.parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pref, n_prefix, pref_class = None, 0, None
    if args.prefix_pt:
        blob = torch.load(args.prefix_pt, map_location="cpu")
        pref = blob["prefix"].to(args.device)
        n_prefix = int(blob["n_prefix"])
        pref_class = blob["compound_class"]
        print(f"[gsp] prefix {pref_class}: {tuple(pref.shape)} from step {blob.get('step')} "
              f"(val {blob.get('val_loss'):.4f})", file=sys.stderr)
    tag_class = args.tag_class or pref_class
    if tag_class is None:
        raise SystemExit("--tag-class is required when no prefix is given")

    adapter = args.adapter
    if adapter and not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
        adapter = adapter / "adapter"
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)

    want = [c.strip() for c in str(tag_class).replace(",", " ").split() if c.strip()]
    # Sort before shuffling: dict/file order is stable here, but sorting makes the tag sequence
    # a function of (--seed, --tag-class) alone, so two arms are paired even if the source file
    # is ever regenerated in a different order.
    tags = sorted({r["taxonomic_tag"] for r in
                   (json.loads(l) for l in args.from_jsonl.open() if l.strip())
                   if r.get("compound_class") in want and r.get("taxonomic_tag")})
    if not tags:
        raise SystemExit(f"no taxonomic tags for {want} in {args.from_jsonl}")
    rng = random.Random(args.seed)
    rng.shuffle(tags)
    print(f"[gsp] {len(tags)} candidate taxa from {want}; generating {args.n} "
          f"(identical taxa in every arm sharing this --seed/--tag-class)", file=sys.stderr)

    class _A:
        temperature = args.temperature; top_k = args.top_k; top_p = args.top_p
        max_new_tokens = args.max_new_tokens; max_n_frac = 0.10

    handle = install_soft_prefix(wrapper.model, pref, n_prefix) if pref is not None else None
    n = 0
    try:
        with args.out_jsonl.open("w") as fh:
            for i in range(args.n):
                tax = tags[i % len(tags)]
                # The placeholder characters occupy the positions whose embeddings the hook
                # overwrites; they are never read as nucleotides.
                prompt = (chr(PLACEHOLDER_ID) * n_prefix) + tax
                torch.manual_seed(args.seed + i)
                out = wrapper.generate(prompt_seqs=[prompt], n_tokens=args.max_new_tokens,
                                       temperature=args.temperature, top_k=args.top_k,
                                       top_p=args.top_p, cached_generation=True, verbose=0)
                ex = extract_sequence(_gen_sequences(out)[0])
                # compound_class = what we ASKED for. For a cross-class cell that is the
                # PREFIX's class, not the taxonomy's -- the prefix is the conditioning signal
                # under test, and the taxon is a nuisance held class-appropriate.
                rec = assemble_record(pref_class or tag_class, tax, ex["sequence"],
                                      ex["hit_eos"], 1, _A)
                rec.update({"soft_prefix_class": pref_class, "n_soft_prefix": n_prefix,
                            "tag_class": ",".join(want), "tax_idx": i % len(tags), "prompt": "soft_prefix+taxonomy",
                            "seeded": False,
                            "accession": f"SP_{pref_class or 'none'}_{i}"})
                fh.write(json.dumps(rec) + "\n")
                n += 1
                if (i + 1) % 4 == 0:
                    print(f"[gsp] {i + 1}/{args.n}", file=sys.stderr, flush=True)
    finally:
        if handle is not None:
            handle.remove()
    print(f"[gsp] wrote {n} -> {args.out_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
