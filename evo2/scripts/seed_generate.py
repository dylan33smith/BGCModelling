#!/usr/bin/env python
"""Nucleotide-context SEEDING diagnostic for BGC class conditioning.

The one genuinely NATIVE functional handle Evo2 offers: it learned coding-sequence
and protein-structure features in pretraining, so in-context continuation from a real
class-defining ORF may keep generation in-class — without relying on the (broken)
|COMPOUND_CLASS| tag. We test that directly: seed the model with the first `--seed-nt`
nucleotides of a REAL held-out core of class X (which contains the start of the
class's assembly-line gene), let it continue, and score ONLY the continuation.

Key mechanic: vortex `generate()` returns generation-ONLY (it strips the prompt), so
with prompt = [prefix] + [real seed], the returned string IS the continuation — the
seed never enters the score, so a "correct-class" continuation is the model's own
contribution, not the seed leaking through antiSMASH.

Arms (set by flags):
  * --no-class-tag  : prompt = {tax} + seed         → pure native-handle test (usually base Evo2)
  * (default)       : prompt = |COMPOUND_CLASS:X|{tax} + seed  → adapter + seed, best practical case

Read against the no-seed floor (megasynthase correct_class ~0.01-0.07): if seeded
continuations are correct-class far more often, seeding is a usable control handle
(provide a starter gene, the model extends the cluster in-class). If not, even a real
exemplar doesn't keep it in-class.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402
from generate_bgc import (  # noqa: E402
    _gen_sequences,
    assemble_record,
    build_prefix,
    extract_sequence,
)


_STOPS = ("TAA", "TAG", "TGA")


def _truncate_at_last_stop(seed: str) -> str:
    """Cut the seed at the last in-frame stop codon so NO open reading frame spans the
    seed->continuation boundary (adversary control for 'trivial gene-continuation').
    Scans all three frames and takes the latest stop; falls back to the untouched seed
    if none is found (caller can detect via the recorded seed_nt)."""
    s = seed.upper()
    best = -1
    for frame in range(3):
        for i in range(frame, len(s) - 2, 3):
            if s[i:i + 3] in _STOPS:
                best = max(best, i + 3)
    return seed[:best] if best > 0 else seed


def _codon_shuffle(seq: str, rng) -> str:
    """Shuffle the seed in codon (3-nt) blocks: preserves nucleotide/codon composition
    and GC, but destroys gene structure and domain order. Control for 'is the lift just
    composition / dense-coding-like statistics rather than the actual class machinery?'"""
    codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
    rng.shuffle(codons)
    return "".join(codons)


def _load_housekeeping_seeds(path: Path, seed_nt: int, rng) -> list[str]:
    """Non-BGC seeds (negative control): if the lift is real class-conditioning, seeding
    with ordinary genomic/housekeeping DNA must NOT produce correct-class clusters."""
    seqs = []
    for line in path.open():
        r = json.loads(line)
        s = r.get("sequence", "")
        if len(s) >= seed_nt + 500:
            seqs.append(s)
    rng.shuffle(seqs)
    return seqs



def _install_generated_only_steer_hook(model, layer: int, delta):
    """Add `delta` to block `layer`'s output ONLY during incremental generation.

    THE CRITICAL DETAIL FOR PHASE 3. vortex `generate()` runs the whole prompt through the model
    once (prefill, sequence length > 1), then emits tokens one at a time (sequence length == 1).
    An ungated hook would perturb the SEED as the model reads it -- corrupting the exemplar that
    is carrying the class signal we are trying to override, and making the experiment measure
    something else entirely. Gating on shape[1] == 1 confines the intervention to generated
    positions, which is what "steer the output" actually means.
    """
    block = dict(model.named_modules())[f"blocks.{layer}"]

    def _apply(h):
        if h.shape[1] != 1:          # prefill: the seed. Leave it untouched.
            return h
        return h + delta.to(h.dtype)

    def hook(_m, _in, out):
        if isinstance(out, (tuple, list)):
            return (_apply(out[0]),) + tuple(out[1:])
        return _apply(out)

    return block.register_forward_hook(hook)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA ckpt/run dir; omit for base Evo2.")
    ap.add_argument("--from-jsonl", type=Path, required=True, help="Real cores to draw seeds from (e.g. val).")
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE"])
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed-nt", type=int, default=2000, help="Length of the real-core seed prefix (nt).")
    ap.add_argument("--max-new-tokens", type=int, default=6000, help="Continuation length to generate + score.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-class-tag", action="store_true",
                    help="Drop |COMPOUND_CLASS|; prompt = {tax} + seed (pure native-handle test).")
    ap.add_argument("--seed-source", choices=["bgc-core", "housekeeping"], default="bgc-core",
                    help="bgc-core = real class-X core (the test); housekeeping = non-BGC DNA "
                         "(NEGATIVE CONTROL: must NOT yield correct-class clusters).")
    ap.add_argument("--housekeeping-jsonl", type=Path, default=None,
                    help="Records to draw non-BGC seeds from (required for --seed-source housekeeping).")
    ap.add_argument("--no-boundary-orf", action="store_true",
                    help="Codon-truncate the seed at its last in-frame stop so no ORF spans "
                         "seed->continuation (controls 'model just finishes the seeded gene').")
    ap.add_argument("--mismatch-tag", action="store_true",
                    help="CONTROL: tag a DIFFERENT class than the seed's. Record carries the TAG "
                         "class as compound_class and the seed's class as seed_class, so analysis "
                         "can see whether the continuation tracks the SEED or the TAG.")
    ap.add_argument("--shuffle-seed", action="store_true",
                    help="CONTROL: codon-shuffle the seed (keeps composition, destroys gene "
                         "structure). Tests whether composition alone drives the lift.")
    # --- PHASE 3: cross-class override ---
    ap.add_argument("--steer-dirs-npz", type=Path, default=None,
                    help="build_steer_dirs.py output. Enables steering of the CONTINUATION.")
    ap.add_argument("--steer-layer", type=int, default=16)
    ap.add_argument("--steer-class-units", type=float, default=0.0,
                    help="Dose in class-units (same scale as Phase 1 / Phase 2). 0 = unsteered arm.")
    ap.add_argument("--steer-dir-prefix", default="",
                    help="\"\" = real direction; \"perm0_\" = shuffled-label control arm.")
    ap.add_argument("--steer-toward", default="rotate",
                    help="Target class to steer the continuation toward. 'rotate' cycles through "
                         "the classes OTHER than the seed's -- the cross-class override test: "
                         "seed class A, steer toward B, ask whether B's machinery appears AND A's "
                         "disappears.")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    args = ap.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    adapter = args.adapter
    if adapter is not None:
        adapter = Path(adapter)
        if not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
            adapter = adapter / "adapter"
    print(f"Model: {'base Evo2 (baseline)' if adapter is None else adapter}  "
          f"| class_tag={not args.no_class_tag}", file=sys.stderr)
    print("Loading Evo2 + adapter (merging)...", file=sys.stderr, flush=True)
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)

    # cores strictly longer than the seed so the seed is a real *fragment*
    keep = set(args.classes)
    pool: dict[str, list[dict]] = {}
    for line in args.from_jsonl.open():
        r = json.loads(line)
        c = r.get("compound_class")
        if c in keep and len(r.get("sequence", "")) >= args.seed_nt + 500:
            pool.setdefault(c, []).append(r)

    rng = random.Random(args.seed)

    # Negative-control seed pool: real non-BGC DNA, same lengths, drawn deterministically.
    hk_pool: list[str] = []
    if args.seed_source == "housekeeping":
        if args.housekeeping_jsonl is None:
            ap.error("--seed-source housekeeping requires --housekeeping-jsonl")
        hk_pool = _load_housekeeping_seeds(args.housekeeping_jsonl, args.seed_nt, rng)
        if not hk_pool:
            ap.error(f"no usable housekeeping seeds (need len >= {args.seed_nt + 500})")
        print(f"[seed] housekeeping pool: {len(hk_pool)} seqs (NEGATIVE CONTROL)", file=sys.stderr)

    class _A:  # decoding-metadata shim for assemble_record
        temperature = args.temperature; top_k = args.top_k; top_p = args.top_p
        max_new_tokens = args.max_new_tokens; max_n_frac = 0.10

    n = 0
    classes_present = sorted(pool)
    with args.out_jsonl.open("w") as fh:
        for cls in classes_present:
            sel = pool[cls][:]
            rng.shuffle(sel)
            sel = sel[: args.per_class]
            for hk_i, r in enumerate(sel):
                tax = r.get("taxonomic_tag", "")
                # The class label + taxon stay identical across arms; ONLY the seed's
                # provenance changes, so the negative control differs by exactly one thing.
                src = r["sequence"] if args.seed_source == "bgc-core" else hk_pool[hk_i % len(hk_pool)]
                seed = src[: args.seed_nt]
                if args.shuffle_seed:
                    seed = _codon_shuffle(seed, rng)
                if args.no_boundary_orf:
                    # ADVERSARY CONTROL: end the seed so no open reading frame spans
                    # seed->continuation. We stop at the last in-frame stop codon within
                    # the seed, so the model cannot merely *finish the gene we handed it*;
                    # a class-defining domain in the continuation must then be de-novo.
                    seed = _truncate_at_last_stop(seed)
                # CONTROL: tag a different class than the seed, so we can see whether the
                # continuation's antiSMASH call tracks the SEED or the TAG.
                tag_cls = cls
                if args.mismatch_tag:
                    others = [c for c in classes_present if c != cls]
                    tag_cls = others[hk_i % len(others)] if others else cls
                prefix = tax if args.no_class_tag else build_prefix(tag_cls, tax)
                prompt = prefix + seed
                # --- Phase 3: pick the override target and install the generation-only hook ---
                steer_target, handle = None, None
                if args.steer_dirs_npz is not None and args.steer_class_units:
                    if args.steer_toward == "rotate":
                        others = [c for c in classes_present if c != cls]
                        steer_target = others[hk_i % len(others)] if others else cls
                    else:
                        steer_target = args.steer_toward
                    zz = np.load(args.steer_dirs_npz)
                    vkey = f"{args.steer_dir_prefix}L{args.steer_layer}_{steer_target}"
                    ukey = f"classunit_L{args.steer_layer}_{steer_target}"
                    if vkey not in zz or ukey not in zz:
                        raise SystemExit(f"[seed-steer] missing {vkey} or {ukey} in "
                                         f"{args.steer_dirs_npz} — is this a build_steer_dirs.py "
                                         f"output? (the legacy sidecar has no class-units)")
                    v = zz[vkey].astype(np.float64)
                    v = v / (np.linalg.norm(v) + 1e-12)
                    dv = v * args.steer_class_units * float(zz[ukey])
                    handle = _install_generated_only_steer_hook(
                        wrapper.model, args.steer_layer,
                        torch.tensor(dv, dtype=torch.float32, device=args.device))
                torch.manual_seed(args.seed)
                try:
                    out = wrapper.generate(
                        prompt_seqs=[prompt], n_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        cached_generation=True, verbose=0,
                    )
                finally:
                    if handle is not None:
                        handle.remove()      # never leak a hook into the next sequence
                cont = extract_sequence(_gen_sequences(out)[0])   # generation-only (prompt stripped)
                # compound_class = what we ASKED for (the tag) so correct_class measures
                # tag-following; seed_class is recorded separately for seed-following.
                rec = assemble_record(tag_cls, tax, cont["sequence"], cont["hit_eos"], 1, _A)
                rec["seed_class"] = cls
                rec["tag_class"] = tag_cls
                rec["mismatch_tag"] = bool(args.mismatch_tag)
                rec["shuffled_seed"] = bool(args.shuffle_seed)
                rec["seed_nt"] = len(seed)
                rec["seed_source_len"] = len(src)
                rec["scored_span"] = "continuation_only"
                rec["steer_target_class"] = steer_target
                rec["steer_class_units"] = args.steer_class_units
                rec["steer_dir_prefix"] = args.steer_dir_prefix or "real"
                rec["steer_layer"] = args.steer_layer if steer_target else None
                rec["class_tag"] = not args.no_class_tag
                rec["seed_source"] = args.seed_source
                rec["no_boundary_orf"] = bool(args.no_boundary_orf)
                rec["seed_prefix_64"] = seed[:64]        # leakage audit: scored seq must NOT start with this
                rec["seed_accession"] = r.get("accession") or r.get("id")
                fh.write(json.dumps(rec) + "\n")
                n += 1
            print(f"[seed] {cls}: {len(sel)} continuations", file=sys.stderr, flush=True)
    print(f"[seed] wrote {n} continuation records -> {args.out_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
