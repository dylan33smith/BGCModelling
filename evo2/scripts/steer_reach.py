#!/usr/bin/env python
"""DOES A STEERING EDIT REACH THE OUTPUT? A per-layer measurement, forward passes only.

THE HYPOTHESIS THIS TESTS. Phase 3 established that steering at layer 16 does not change what
the model writes. One explanation is DILUTION: the edit is made 16 blocks from the output and is
attenuated to nothing on the way. Measured on the activation cache the directions are fit from
(n=3,430 real cores, `acts_valtest_fit.npz`), the residual stream is not remotely scale-stable:

    layer         16      20      24      27       28        29        30
    mean ||h||    8.95    9.78    6.54    11.25    5.47e3    8.66e6    3.69e12
    class AUC     0.923   0.927   0.885   0.835    0.590     0.610     0.553

So L27 is the LAST layer where the class direction is still real, and the last one before the
residual explodes by eleven orders of magnitude. If dilution is the binding constraint, an edit
injected at L27 should move the output distribution far more than the same edit at L16.

WHAT IS MEASURED, per layer, on real held-out cores:

  reach  = mean KL(p_steered || p_base) over the scored positions. Purely "how much did the
           output distribution move", with no reference to class. This is the dilution number.
  gap    = (per-token loglik of the TRUE continuation when steering toward its OWN class)
           minus (the same when steering toward a DIFFERENT class). This is Phase 1's Test B
           statistic: it cancels the generic damage any residual perturbation causes, so what
           survives is the CLASS-SPECIFIC part of the effect.

Both are reported for the real directions and for `perm*` SHUFFLED-LABEL directions built by the
identical recipe on scrambled labels. Reach will be nonzero for shuffled directions too -- any
perturbation moves the distribution. Only the REAL-vs-SHUFFLED contrast carries information.

DOSE IS IN UNITS OF THE LOCAL RESIDUAL NORM (`--norm-frac`), never absolute. Holding ||delta||
fixed across layers whose ||h|| differs by 10^11 would confound the very quantity under test.
The table above is the mean-POOLED cache norm; measured live per position it is 6.69 at L16 and
31.97 at L27, so in the units that actually govern a token, one class-unit is 0.082*||h|| at L16
and 0.056*||h|| at L27.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402
from evo2_inference import count_prefix_tokens  # noqa: E402


def _norm_rel_add_hook(model, layer: int, unit: torch.Tensor, frac: float, start_pos: int):
    """h <- h + frac * ||h|| * unit, at positions >= start_pos only.

    The dose is recomputed from each POSITION's own residual norm, which is what makes one
    `frac` mean the same intervention at two layers with different scales. `start_pos` keeps the
    context untouched: we score a continuation under an intervention, so perturbing the
    conditioning would measure a different conditional distribution than the one reported.
    """
    block = dict(model.named_modules())[f"blocks.{layer}"]

    def _apply(h):
        if h.shape[1] <= start_pos:
            return h
        n = h.detach().float().norm(dim=-1, keepdim=True)      # (B, P, 1)
        d = (unit * (frac * n)).to(h.dtype)
        out = h.clone()
        out[:, start_pos:, :] = out[:, start_pos:, :] + d[:, start_pos:, :]
        return out

    def hook(_m, _in, out):
        if isinstance(out, (tuple, list)):
            return (_apply(out[0]),) + tuple(out[1:])
        return _apply(out)

    return block.register_forward_hook(hook)


@torch.no_grad()
def _logprobs(model, tok, context: str, target: str, max_seq_len: int, device: str):
    """(logprobs over scored positions, index of the first scored position, token ids)."""
    ids = list(tok.tokenize(context + target))[:max_seq_len]
    ids = [int(i) for i in ids]
    n_ctx = count_prefix_tokens(tok, context)
    if n_ctx >= len(ids) - 1:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)[0]     # (L-1, V)
    labels = x[0, 1:]
    keep = torch.arange(1, x.shape[1], device=device) >= n_ctx
    return lp[keep], labels[keep], n_ctx


def _load(npz: Path, layers, classes, n_perm_use: int):
    z = np.load(npz)
    avail = [int(m.group(1)) for m in (re.match(r"perm(\d+)_L", k) for k in z.files) if m]
    n_perm = max(avail) + 1 if avail else 0
    if n_perm < n_perm_use:
        raise SystemExit(f"[reach] ABORT: {npz} holds {n_perm} shuffled-label control sets, "
                         f"need {n_perm_use}. Without them, 'the edit moved the output' cannot "
                         f"be separated from 'any perturbation moves the output'.")
    real, perms, units = {}, collections.defaultdict(dict), {}
    missing = []
    for L in layers:
        for c in classes:
            k, ku = f"L{L}_{c}", f"classunit_L{L}_{c}"
            if k not in z.files or ku not in z.files:
                missing.append(k if k not in z.files else ku)
                continue
            v = z[k].astype(np.float64)
            real[(L, c)] = v / (np.linalg.norm(v) + 1e-12)
            units[(L, c)] = float(z[ku])
            for j in range(n_perm_use):
                pk = f"perm{j}_L{L}_{c}"
                if pk not in z.files:
                    missing.append(pk)
                    continue
                pv = z[pk].astype(np.float64)
                perms[j][(L, c)] = pv / (np.linalg.norm(pv) + 1e-12)
    if missing:
        raise SystemExit(f"[reach] ABORT: {len(missing)} arrays missing from {npz}, "
                         f"e.g. {missing[:4]}. Rebuild with build_steer_dirs.py --layers "
                         f"{' '.join(str(L) for L in layers)}.")
    print(f"[reach] {npz.name}: {len(real)} directions over layers {layers}, "
          f"{n_perm_use}/{n_perm} shuffled-label control sets in use")
    for L in layers:
        cu = [units[(L, c)] for c in classes if (L, c) in units]
        print(f"[reach]   L{L}: class-unit mean {np.mean(cu):.4f} "
              f"[{min(cu):.4f}..{max(cu):.4f}]")
    return real, perms, units


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768"
                                 "/checkpoints/step_1200"))
    ap.add_argument("--dirs-npz", type=Path, required=True)
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/valtest_eval.jsonl"))
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE", "RIPP"])
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24, 27])
    ap.add_argument("--norm-fracs", type=float, nargs="+", default=[0.061, 0.16],
                    help="Dose as a fraction of the LOCAL residual norm. 0.061 is what one "
                         "class-unit amounts to at L16 (the Phase 1-3 operating point); 0.16 is "
                         "what one class-unit amounts to at L27.")
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--ctx-nt", type=int, default=600)
    ap.add_argument("--score-nt", type=int, default=1200)
    ap.add_argument("--n-perm-use", type=int, default=3)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/steer_reach/reach.json"))
    args = ap.parse_args()

    real, perms, units = _load(args.dirs_npz, args.layers, args.classes, args.n_perm_use)

    byc: dict[str, list] = {c: [] for c in args.classes}
    for line in args.cores.open():
        r = json.loads(line)
        c = r.get("compound_class")
        if c in byc and len(r.get("sequence", "")) >= args.ctx_nt + 300:
            byc[c].append(r)
    rng = random.Random(args.seed)
    cores = []
    for c in args.classes:
        rng.shuffle(byc[c])
        cores.extend(byc[c][: args.n_per_class])
    rng.shuffle(cores)
    print(f"[reach] {len(cores)} cores: " + ", ".join(f"{c}={min(len(byc[c]), args.n_per_class)}"
                                                      for c in args.classes))

    wrapper = load_evo2_wrapper_for_inference(Path(args.adapter) if args.adapter else None,
                                              device=args.device)
    model, tok = wrapper.model, wrapper.tokenizer

    rows: list[dict] = []
    for i, r in enumerate(cores):
        cls, seq = r["compound_class"], r["sequence"]
        ctx = (r.get("taxonomic_tag", "") or "") + seq[: args.ctx_nt]
        tgt = seq[args.ctx_nt: args.ctx_nt + args.score_nt]
        if len(tgt) < 100:
            continue
        other = rng.choice([c for c in args.classes if c != cls])
        base = _logprobs(model, tok, ctx, tgt, args.max_seq_len, args.device)
        if base is None:
            continue
        lp0, labels, n_ctx = base
        p0 = lp0.exp()
        ll0 = float(lp0.gather(-1, labels.unsqueeze(-1)).mean())

        for L in args.layers:
            for frac in args.norm_fracs:
                for kind, dmap in ([("real", real)] +
                                   [(f"perm{j}", perms[j]) for j in range(args.n_perm_use)]):
                    rec = {"class": cls, "other": other, "layer": L, "frac": frac, "kind": kind,
                           "ll_base": ll0}
                    for tag, c in (("true", cls), ("wrong", other)):
                        u = torch.tensor(dmap[(L, c)], dtype=torch.float32, device=args.device)
                        h = _norm_rel_add_hook(model, L, u, frac, n_ctx)
                        try:
                            got = _logprobs(model, tok, ctx, tgt, args.max_seq_len, args.device)
                        finally:
                            h.remove()
                        if got is None:
                            continue
                        lp, lab, _ = got
                        rec[f"ll_{tag}"] = float(lp.gather(-1, lab.unsqueeze(-1)).mean())
                        # KL(p_steer || p_base), averaged over scored positions
                        rec[f"kl_{tag}"] = float((lp.exp() * (lp - lp0)).sum(-1).mean())
                    if "ll_true" in rec and "ll_wrong" in rec:
                        rec["gap"] = rec["ll_true"] - rec["ll_wrong"]
                        rec["reach"] = 0.5 * (rec["kl_true"] + rec["kl_wrong"])
                        rows.append(rec)
        if (i + 1) % 5 == 0:
            print(f"[reach] {i + 1}/{len(cores)} cores", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows))
    print(f"[reach] wrote {len(rows)} rows -> {args.out_json}\n")

    # ------------------------------------------------------------------ summary
    print("=" * 92)
    print("REACH = how far a steering edit moves the OUTPUT distribution (mean KL, nats/token)")
    print("GAP   = class-SPECIFIC part: loglik(true-class steer) - loglik(wrong-class steer)")
    print("A shuffled-label direction perturbs just as hard; only real-vs-shuffled carries info.")
    print("=" * 92)
    hdr = (f"{'layer':>5} {'frac':>6} {'n':>4} {'reach_real':>11} {'reach_perm':>11} "
           f"{'ratio':>6} {'gap_real':>10} {'gap_perm_mean':>14} {'gap_perm_sd':>12} {'z':>6} {'p':>6}")
    print(hdr)
    summary = []
    for L in args.layers:
        for frac in args.norm_fracs:
            sel = [r for r in rows if r["layer"] == L and r["frac"] == frac]
            rr = [r for r in sel if r["kind"] == "real"]
            if not rr:
                continue
            reach_real = st.mean(r["reach"] for r in rr)
            gap_real = st.mean(r["gap"] for r in rr)
            perm_reach, perm_gap = [], []
            for j in range(args.n_perm_use):
                pj = [r for r in sel if r["kind"] == f"perm{j}"]
                if pj:
                    perm_reach.append(st.mean(r["reach"] for r in pj))
                    perm_gap.append(st.mean(r["gap"] for r in pj))
            if not perm_gap:
                continue
            pm = st.mean(perm_gap)
            psd = st.pstdev(perm_gap) if len(perm_gap) > 1 else float("nan")
            # A z built on a 3-point sd is not a z. With few controls the sd is itself mostly
            # noise, so a tiny spread manufactures an enormous z from a negligible effect --
            # the same family of error as the retired "beat the max of the controls" rule,
            # which got STRICTER as controls were added. Report it only when the sd is
            # estimated from enough points to mean something; the permutation p is the
            # trustworthy statistic, and it is FLOORED at 1/(n_controls+1).
            z = ((gap_real - pm) / psd
                 if len(perm_gap) >= 5 and psd == psd and psd > 0 else float("nan"))
            # permutation p, floored at 1/(n+1) -- with 3 controls the best attainable is 0.250
            n_ge = sum(1 for g in perm_gap if g >= gap_real)
            p = (n_ge + 1) / (len(perm_gap) + 1)
            rp = st.mean(perm_reach)
            print(f"{L:>5} {frac:>6.3f} {len(rr):>4} {reach_real:>11.5f} {rp:>11.5f} "
                  f"{reach_real / rp if rp else float('nan'):>6.2f} {gap_real:>10.5f} "
                  f"{pm:>14.5f} {psd:>12.5f} {z:>6.2f} {p:>6.3f}")
            summary.append({"layer": L, "frac": frac, "n": len(rr), "reach_real": reach_real,
                            "reach_perm": rp, "gap_real": gap_real, "gap_perm_mean": pm,
                            "gap_perm_sd": psd, "z": z, "p": p})
    args.out_json.with_name(args.out_json.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\nREAD: if `reach_real` at L27 is much larger than at L16 for the SAME frac, the L16")
    print("edit was being attenuated on the way to the output -- dilution confirmed, and a later")
    print("injection point is the fix. If reach is comparable across layers, dilution is NOT the")
    print("binding constraint and the null is about what the model does with the edit, not")
    print("whether it receives it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
