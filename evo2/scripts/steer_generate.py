#!/usr/bin/env python
"""Activation steering for BGC class control — inject the class direction, no retraining.

WHY. The probe showed class is linearly decodable from mid-network activations (~0.91,
chance 0.09) in BOTH base and the v2 adapter, on a broad plateau (L10-L23), then dumped
off a cliff in the last blocks (L27 0.84 -> L29 0.45). So the model KNOWS the class; the
information just never reaches the ~2-bit nucleotide output distribution. Prefix/label
conditioning is dead (tag provably inert). Steering intervenes where the signal actually
lives: mid-network.

METHOD (activation addition / steering vectors, standard interpretability practice — it
works precisely BECAUSE the concept is linearly represented, which the probe established):
at a chosen block, replace the hidden state h with

    h  +  alpha * ref_norm * (v_class / ||v_class||)

v_class is the difference-of-means direction for that class at that layer (mean activation
over real class-X cores minus the global mean), precomputed by class_probe_sweep.py.
ref_norm is the layer's mean activation norm, so ALPHA IS IN UNITS OF "fraction of a typical
activation magnitude" and is therefore comparable across layers.

PROMPT = taxonomy tag ONLY (no |COMPOUND_CLASS| block). The de-confound proved the tag is
inert, so this asks the real question: can steering REPLACE label conditioning? alpha=0 is
the floor control (and is layer-independent, so it is generated once).

WATCH BOTH correct_class AND coding_density: CFG failed by pushing the model out of
distribution (coherence collapsed to 0 at high w). Steering can do the same. The success
signature is a window where correct_class rises WHILE coding_density holds.
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
from generate_bgc import _gen_sequences, assemble_record, extract_sequence  # noqa: E402


def _install_steer_hook(model: Any, layer: int, delta: torch.Tensor):
    """Add `delta` to every position of the block's output. Returns the handle.

    The hook RETURNS the modified tensor (a forward hook's return value replaces the
    output), preserving tuple structure since StripedHyena blocks may return
    (hidden, inference_params)."""
    mods = dict(model.named_modules())
    block = mods[f"blocks.{layer}"]

    def hook(_m, _in, out):
        if isinstance(out, (tuple, list)):
            h = out[0] + delta.to(out[0].dtype)
            return (h,) + tuple(out[1:])
        return out + delta.to(out.dtype)

    return block.register_forward_hook(hook)


def _load_direction(dirs_npz: Path, layer: int, cls: str, prefix: str = "") -> np.ndarray:
    """`prefix` selects a shuffled-label control set, e.g. "perm3_" -> perm3_L16_NRPS."""
    z = np.load(dirs_npz)
    key = f"{prefix}L{layer}_{cls}"
    if key not in z:
        raise KeyError(f"{key} not in {dirs_npz}; available e.g. {list(z.keys())[:5]}")
    return z[key]


def _load_class_unit(dirs_npz: Path, layer: int, cls: str) -> float:
    """The dose scale from build_steer_dirs.py: distance from the other-class mean to this
    class's mean along the direction. Phase 1 established its effect at 1 class-unit, so
    generation must use the SAME units or the phases are not comparable."""
    z = np.load(dirs_npz)
    key = f"classunit_L{layer}_{cls}"
    if key not in z:
        raise KeyError(f"{key} missing from {dirs_npz} — not a build_steer_dirs.py output? "
                       f"(the legacy class_probe_sweep sidecar has no class-units)")
    return float(z[key])


def _ref_norm(acts_npz: Path, layer: int) -> float:
    """Mean L2 norm of the layer's (full-sequence) activation vectors — the scale that
    makes alpha comparable across layers. npz keys load lazily, so this reads one array."""
    z = np.load(acts_npz)
    X = z[f"L{layer}"]                      # (N, P, D); last prefix index = full sequence
    V = X[:, -1, :]
    return float(np.linalg.norm(V, axis=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA ckpt/run dir (the model to steer).")
    ap.add_argument("--from-jsonl", type=Path, required=True, help="Source of real taxonomic tags per class.")
    ap.add_argument("--dirs-npz", type=Path, required=True, help="acts_*.dirs.npz from class_probe_sweep.py")
    ap.add_argument("--acts-npz", type=Path, required=True, help="acts_*.npz (for the per-layer ref norm)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--layer", type=int, required=True)
    # THREE scaling modes. ||v_class|| spans 17x at layer 16 (TERPENE 1.05 ... ECTOINE 17.75),
    # so NO single knob holds both quantities constant — pick the one matching the question:
    #   --delta-norm : constant PERTURBATION MAGNITUDE across classes  -> the COHERENCE axis
    #   --beta       : constant SEMANTIC push (class-mean offsets)     -> the CLASS-EFFECT axis
    #   --alpha      : same as --delta-norm but in units of mean||h|| (kept for older runs)
    # Measured 2026-07-29: coherence damage tracks ||delta||, not beta. A global beta gave
    # ECTOINE a 17x larger physical push than TERPENE and destroyed it (0.85 -> 0.19 at beta=0.5)
    # while PKS was untouched at beta=2. Titrate coherence with --delta-norm.
    ap.add_argument("--alpha", type=float, default=None,
                    help="Scaling in ref-norm units: ||delta|| = alpha * mean||h||. Constant "
                         "magnitude across classes. Equivalent to --delta-norm, just relative; "
                         "prefer --delta-norm (absolute, no ref-norm bookkeeping).")
    ap.add_argument("--delta-norm", type=float, default=None,
                    help="Scaling by ABSOLUTE perturbation magnitude: ||delta|| = this, for every "
                         "class (v is unit-normalized first). The right knob for coherence "
                         "titration, since coherence damage tracks physical magnitude. "
                         "0 = unsteered control.")
    ap.add_argument("--class-units", type=float, default=None,
                    help="PREFERRED. Dose in class-units: delta = dose * classunit * unit(v). "
                         "Identical scaling to steer_causal_tests.py, so Phase 1 and Phase 2/3 "
                         "doses mean the same thing. Phase 1 found the effect at 1.0.")
    ap.add_argument("--dir-prefix", default="",
                    help="Direction key prefix. \"\" = real; \"perm3_\" = a shuffled-label "
                         "control direction, for the null arm of a generation sweep.")
    ap.add_argument("--beta", type=float, default=None,
                    help="Scaling by SEMANTIC offset: delta = beta * v_class (raw "
                         "difference-of-means). beta=1 = exactly one class-mean offset. Constant "
                         "semantic push, but NOT constant magnitude — a global beta hits "
                         "large-||v|| classes far harder. 0 = unsteered control.")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    args = ap.parse_args()

    # Validate BEFORE the (slow) model load so a bad invocation fails in milliseconds.
    given = [n for n, v in (("--alpha", args.alpha), ("--beta", args.beta),
                            ("--delta-norm", args.delta_norm),
                            ("--class-units", args.class_units)) if v is not None]
    if len(given) != 1:
        ap.error(f"give exactly one of --alpha / --beta / --delta-norm / --class-units "
                 f"(got: {given or 'none'})")
    mode = {"--beta": "beta_raw_diffmean", "--alpha": "alpha_refnorm",
            "--delta-norm": "delta_norm_absolute", "--class-units": "class_units"}[given[0]]
    strength = {"--beta": args.beta, "--alpha": args.alpha,
                "--delta-norm": args.delta_norm, "--class-units": args.class_units}[given[0]]
    use_beta = args.beta is not None

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    adapter = args.adapter
    if adapter is not None:
        adapter = Path(adapter)
        if not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
            adapter = adapter / "adapter"
    print(f"Model: {'base Evo2' if adapter is None else adapter} | layer {args.layer} "
          f"| alpha {args.alpha}", file=sys.stderr)
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)

    # real taxonomic tags per class (taxonomy stays class-appropriate, as in every prior run)
    pool: dict[str, list[str]] = {}
    for line in args.from_jsonl.open():
        r = json.loads(line)
        c = r.get("compound_class")
        if c in set(args.classes):
            t = r.get("taxonomic_tag", "")
            if t:
                pool.setdefault(c, []).append(t)
    rng = random.Random(args.seed)

    ref = 0.0
    if strength != 0 and mode == "alpha_refnorm":
        ref = _ref_norm(args.acts_npz, args.layer)
        print(f"[steer] layer {args.layer} ref_norm {ref:.2f}", file=sys.stderr)

    class _A:
        temperature = args.temperature; top_k = args.top_k; top_p = args.top_p
        max_new_tokens = args.max_new_tokens; max_n_frac = 0.10

    n = 0
    with args.out_jsonl.open("w") as fh:
        for cls in sorted(args.classes):
            tags = pool.get(cls, [])
            if not tags:
                print(f"[steer] !! no taxonomic tags for {cls}; skipping", file=sys.stderr)
                continue
            rng.shuffle(tags)
            handle = None
            if strength != 0:
                v = _load_direction(args.dirs_npz, args.layer, cls, args.dir_prefix)
                vn = float(np.linalg.norm(v))
                unit = v / (vn + 1e-8)
                if mode == "beta_raw_diffmean":      # constant semantic push
                    dv = v * strength
                elif mode == "alpha_refnorm":        # constant magnitude, ref-norm units
                    dv = unit * strength * ref
                elif mode == "class_units":          # SAME units as Phase 1
                    cu = _load_class_unit(args.dirs_npz, args.layer, cls)
                    dv = unit * strength * cu
                else:                                # constant magnitude, absolute
                    dv = unit * strength
                # beta_equiv = how many class-mean offsets this magnitude actually delivers.
                # It varies per class under the magnitude modes — that is the point, and the
                # number to quote when comparing the coherence axis to the class-effect axis.
                dvn = float(np.linalg.norm(dv))
                print(f"[steer]   {cls}: ||v||={vn:.2f} ||delta||={dvn:.2f} "
                      f"beta_equiv={dvn / (vn + 1e-8):.3f}", file=sys.stderr)
                delta = torch.tensor(dv, dtype=torch.float32, device=args.device)
                handle = _install_steer_hook(wrapper.model, args.layer, delta)
            try:
                for i in range(args.per_class):
                    tax = tags[i % len(tags)]
                    torch.manual_seed(args.seed + i)
                    out = wrapper.generate(
                        prompt_seqs=[tax],          # taxonomy ONLY — no class tag
                        n_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        cached_generation=True, verbose=0,
                    )
                    ex = extract_sequence(_gen_sequences(out)[0])
                    rec = assemble_record(cls, tax, ex["sequence"], ex["hit_eos"], 1, _A)
                    rec.update({"steer_layer": args.layer, "steer_alpha": args.alpha,
                                "steer_beta": args.beta, "steer_delta_norm": args.delta_norm,
                                "steer_class_units": args.class_units,
                                "steer_dir_prefix": args.dir_prefix,
                                "steer_scaling": mode,
                                # persist what was ACTUALLY applied, so analysis never has to
                                # re-derive it from logs (the beta titration had to)
                                "steer_v_norm": round(vn, 4) if strength != 0 else 0.0,
                                "steer_applied_norm": round(dvn, 4) if strength != 0 else 0.0,
                                "steer_beta_equiv": round(dvn / (vn + 1e-8), 4) if strength != 0 else 0.0,
                                "steer_dir": "diff_of_means", "prompt": "taxonomy_only"})
                    fh.write(json.dumps(rec) + "\n")
                    n += 1
            finally:
                if handle is not None:
                    handle.remove()      # never leak a hook into the next class
            print(f"[steer] {cls}: {args.per_class} generated", file=sys.stderr, flush=True)
    print(f"[steer] wrote {n} -> {args.out_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
