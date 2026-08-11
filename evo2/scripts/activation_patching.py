#!/usr/bin/env python
"""WHAT DOES THE MODEL ACTUALLY CONSULT? Activation patching, layer by layer.

THE QUESTION THIS ANSWERS. We know class is DECODABLE at layer 16 (linear probe 0.911 vs 0.091
chance) and we know an injected class direction LANDS and is IGNORED (direction_audit.py: the edit
flips a linear readout at 1-2 class-units and we dosed 2.8-11.4, at all nine layers, with no effect
on the output). Those two facts together say the class is present but not consulted -- or that our
edit was the wrong SHAPE to be consulted.

The ACE pre-check sharpened that into a testable fork. Our steering edit was rank-1: it corrected
one coordinate out of 4,096 and left the other 4,095 belonging to the source class, landing 3-20 sd
off the data manifold. Patching does the opposite. It substitutes a GENUINE activation pattern,
recorded from a real forward pass over real DNA of the target class -- in-distribution, all 4,096
coordinates mutually consistent.

  * PATCHING MOVES THE OUTPUT where steering did not  -> the model does read that layer; our
    failure was the edit's shape, not the model's blindness. That reopens intervention with a very
    different recipe (transplant, not translate).
  * PATCHING ALSO DOES NOTHING                        -> the model genuinely does not consult those
    activations when choosing the next base. Inference-time intervention is dead for a structural
    reason, and only training-time coupling remains.

Either answer is decisive, which is what makes this worth the GPU time.

WHAT THIS PHASE MEASURES, AND WHAT IT DOES NOT. Evo2's vocabulary is bytes: the next-token
distribution is effectively over A/C/G/T. Compound class is NOT legible in one token, so this phase
cannot measure class transfer. It measures INFLUENCE -- does substituting layer L's state change
what the model does next, and how much of the way does it move toward the donor's own behaviour.
That is the necessary precondition for class transfer and it is cheap. The functional class test
(generate under the patch, score with antiSMASH) is phase B, aimed at whichever layers survive here.

CONTROLS, because "replacing a layer's state changes the output" is true of any large perturbation:
  * SAME-CLASS donor -- a different real core of the RECIPIENT's class. Isolates "class transferred"
    from "any real donor perturbs".
  * SHUFFLED donor -- the donor's activations with positions randomly permuted. Same values, same
    norms, same marginal distribution, destroyed temporal structure.
  * NOISE donor -- Gaussian matched to the donor's per-position norm. The "any big vector" control
    that the steering work should have had from the start.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402


def _block(model, layer: int):
    return dict(model.named_modules())[f"blocks.{layer}"]


def _capture_hook(model, layer: int, store: dict):
    """Record layer L's output on this forward pass."""
    def hook(_m, _in, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        store["h"] = h.detach().clone()
        return out
    return _block(model, layer).register_forward_hook(hook)


def _patch_hook(model, layer: int, donor: torch.Tensor, mode):
    """Replace layer L's output with the donor's recorded activations.

    `mode='last'` substitutes only the final position. Downstream Hyena convolutions still re-mix
    all the EARLIER (unpatched) positions, so the recipient's context survives and the measurement
    is informative about how much layer L's final-position state is consulted. THIS IS THE
    EXPERIMENT.

    `mode='all'` substitutes every position, which MEASURED 1.000 alignment at layers 0, 16 and 31
    alike, with an identical KL of 0.8508 at each -- because once every position at depth L is the
    donor's, layers L+1..31 compute from the donor alone and the model simply becomes the donor.
    It is therefore a POSITIVE CONTROL, not a measurement: it confirms the patch propagates end to
    end and that no hidden path leaks the recipient's context past the hook. Do not read a layer
    profile off it; there is none to read.
    """
    def _apply(h):
        if donor.shape[1] != h.shape[1]:
            raise ValueError(f"donor has {donor.shape[1]} positions, recipient {h.shape[1]} — "
                             f"prompts must be token-aligned for patching to be meaningful")
        out = h.clone()
        # k = how many TRAILING positions to substitute. This is the axis that separates the two
        # explanations for a small effect: "the model does not read this layer" versus "one
        # position out of a thousand simply has little leverage". Sweeping k pins which it is.
        k = h.shape[1] if mode in ("all", -1) else (1 if mode == "last" else int(mode))
        k = max(1, min(k, h.shape[1]))
        out[:, -k:, :] = donor[:, -k:, :].to(h.dtype)
        return out

    def hook(_m, _in, out):
        if isinstance(out, (tuple, list)):
            return (_apply(out[0]),) + tuple(out[1:])
        return _apply(out)

    return _block(model, layer).register_forward_hook(hook)


@torch.no_grad()
def _next_token_dist(model, tok, prompt: str, device: str, max_seq_len: int):
    ids = [int(i) for i in list(tok.tokenize(prompt))[:max_seq_len]]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return F.softmax(logits[0, -1, :].float(), dim=-1), len(ids)


@torch.no_grad()
def _dist_with_capture(model, tok, prompt: str, device: str, max_seq_len: int, layer: int):
    store: dict = {}
    h = _capture_hook(model, layer, store)
    try:
        p, n = _next_token_dist(model, tok, prompt, device, max_seq_len)
    finally:
        h.remove()
    return p, n, store["h"]


@torch.no_grad()
def _dist_with_patch(model, tok, prompt: str, device: str, max_seq_len: int,
                     layer: int, donor: torch.Tensor, mode: str):
    h = _patch_hook(model, layer, donor, mode)
    try:
        p, _ = _next_token_dist(model, tok, prompt, device, max_seq_len)
    finally:
        h.remove()
    return p


def _alignment(p_dst, p_patched, p_src) -> float:
    """Fraction of the way the patched output moved FROM the recipient TOWARD the donor.

    0 = the patch did nothing; 1 = the recipient now behaves exactly like the donor. Negative means
    it moved away. This is the projection of the achieved change onto the desired change, which is
    the quantity a raw KL cannot give: KL says "something changed", this says "it changed into the
    donor" -- and a big KL with alignment ~0 is a disruption, not a transfer.
    """
    want = (p_src - p_dst)
    got = (p_patched - p_dst)
    denom = float((want * want).sum())
    return float((got * want).sum() / denom) if denom > 1e-12 else float("nan")


def _kl(p, q) -> float:
    p = p.clamp_min(1e-12); q = q.clamp_min(1e-12)
    return float((p * (p / q).log()).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE", "RIPP"])
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(0, 32, 2)))
    ap.add_argument("--mode", default="all",
                    help="'all' (saturates: positive control), 'last' (1 position), or an INTEGER "
                         "k = substitute the last k positions. Sweeping k separates 'this layer is "
                         "not read' from 'one position has too little leverage'.")
    ap.add_argument("--k-sweep", type=int, nargs="*", default=None,
                    help="Run several k values in one pass, e.g. --k-sweep 1 10 100 500.")
    ap.add_argument("--ctx-nt", type=int, default=1000)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/activation_patching.json"))
    args = ap.parse_args()

    byc: dict[str, list] = {c: [] for c in args.classes}
    for line in args.cores.open():
        r = json.loads(line)
        c = r.get("compound_class")
        if c in byc and len(r.get("sequence", "")) >= args.ctx_nt + 100:
            byc[c].append(r)
    rng = random.Random(args.seed)
    for c in args.classes:
        rng.shuffle(byc[c])
    print("[patch] pool: " + ", ".join(f"{c}={len(byc[c])}" for c in args.classes), flush=True)

    # Pairs: donor of class A, recipient of class B != A. The recipient's taxonomy tag is used for
    # BOTH prompts so the two token sequences have identical length -- position-aligned patching is
    # only meaningful if the positions correspond.
    pairs = []
    for i in range(args.n_pairs):
        ca = args.classes[i % len(args.classes)]
        cb = args.classes[(i + 1 + (i // len(args.classes))) % len(args.classes)]
        if ca == cb:
            cb = args.classes[(i + 2) % len(args.classes)]
        donor, recip = byc[ca][i % len(byc[ca])], byc[cb][i % len(byc[cb])]
        same = byc[cb][(i + 1) % len(byc[cb])]        # same-class-as-recipient control donor
        pairs.append((ca, cb, donor, recip, same))
    modes = args.k_sweep if args.k_sweep else [args.mode]
    print(f"[patch] {len(pairs)} pairs, layers={args.layers}, modes={modes}", flush=True)

    wrapper = load_evo2_wrapper_for_inference(Path(args.adapter) if args.adapter else None,
                                              device=args.device)
    model, tok = wrapper.model, wrapper.tokenizer
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    rows = []
    for pi, (ca, cb, donor, recip, same) in enumerate(pairs):
        tax = recip.get("taxonomic_tag", "") or ""
        p_donor_prompt = tax + donor["sequence"][: args.ctx_nt]
        p_recip_prompt = tax + recip["sequence"][: args.ctx_nt]
        p_same_prompt = tax + same["sequence"][: args.ctx_nt]

        p_dst, n_dst = _next_token_dist(model, tok, p_recip_prompt, args.device, args.max_seq_len)
        p_src, n_src = _next_token_dist(model, tok, p_donor_prompt, args.device, args.max_seq_len)
        if n_dst != n_src:
            print(f"[patch]   pair {pi}: token length {n_src} vs {n_dst} — skipped", flush=True)
            continue

        for L in args.layers:
            _, _, h_donor = _dist_with_capture(model, tok, p_donor_prompt, args.device,
                                               args.max_seq_len, L)
            _, _, h_same = _dist_with_capture(model, tok, p_same_prompt, args.device,
                                              args.max_seq_len, L)
            shuf = h_donor[:, torch.randperm(h_donor.shape[1], generator=gen), :]
            nrm = h_donor.float().norm(dim=-1, keepdim=True)
            noise = torch.randn(h_donor.shape, generator=gen).to(h_donor.device)
            noise = (noise / noise.float().norm(dim=-1, keepdim=True) * nrm).to(h_donor.dtype)

            for md in modes:
                for arm, dn in (("cross_class", h_donor), ("same_class", h_same),
                                ("shuffled", shuf), ("noise", noise)):
                    pp = _dist_with_patch(model, tok, p_recip_prompt, args.device,
                                          args.max_seq_len, L, dn, md)
                    rows.append({"pair": pi, "donor_class": ca, "recip_class": cb, "layer": L,
                                 "arm": arm, "mode": str(md),
                                 "kl_vs_recipient": _kl(pp, p_dst),
                                 "alignment_to_donor": _alignment(p_dst, pp, p_src)})
        print(f"[patch] pair {pi + 1}/{len(pairs)} ({ca}->{cb}) done", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=1))

    import statistics as st
    print("\n" + "=" * 92)
    print(f"ACTIVATION PATCHING (mode={args.mode}) — does substituting layer L's state change "
          f"what the model does next?")
    print("alignment: 0 = patch did nothing, 1 = recipient now behaves exactly like the donor.")
    print("=" * 92)
    arms = ["cross_class", "same_class", "shuffled", "noise"]
    for md in modes:
        print(f"\n--- mode/k = {md} ---")
        print(f"{'layer':>6} | " + " ".join(f"{a[:11]:>12}" for a in arms) + "   <- mean alignment")
        for L in args.layers:
            cells = []
            for a in arms:
                v = [r["alignment_to_donor"] for r in rows if r["layer"] == L and r["arm"] == a
                     and r["mode"] == str(md)
                     and r["alignment_to_donor"] == r["alignment_to_donor"]]
                cells.append(f"{st.mean(v):>12.3f}" if v else f"{'--':>12}")
            print(f"{L:>6} | " + " ".join(cells))
    print(f"\n{'layer':>6} | " + " ".join(f"{a[:11]:>12}" for a in arms) + "   <- mean KL vs recipient")
    for L in args.layers:
        cells = []
        for a in arms:
            v = [r["kl_vs_recipient"] for r in rows if r["layer"] == L and r["arm"] == a]
            cells.append(f"{st.mean(v):>12.4f}" if v else f"{'--':>12}")
        print(f"{L:>6} | " + " ".join(cells))

    cc = [r["alignment_to_donor"] for r in rows if r["arm"] == "cross_class"
          and r["alignment_to_donor"] == r["alignment_to_donor"]]
    nz = [r["alignment_to_donor"] for r in rows if r["arm"] == "noise"
          and r["alignment_to_donor"] == r["alignment_to_donor"]]
    print("\nHOW TO READ THIS.")
    print(f"  * cross-class alignment pooled over layers: {st.mean(cc):.3f}  "
          f"(noise control {st.mean(nz):.3f})")
    print("  * HIGH cross-class alignment => the model DOES read this layer; a real, in-distribution")
    print("    donor moves it where our rank-1 edit could not. Intervention reopens, with a")
    print("    transplant rather than a direction. Phase B then tests class transfer functionally.")
    print("  * NEAR-ZERO everywhere, including 'all' mode => the model does not consult these")
    print("    activations for the next base. Inference-time intervention is structurally dead.")
    print("  * cross_class ~ same_class means the patch perturbs without transferring CLASS; the")
    print("    gap between them is the only part attributable to class.")
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
