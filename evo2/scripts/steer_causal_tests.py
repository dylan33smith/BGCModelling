#!/usr/bin/env python
"""PHASE 1 — does the model actually USE class, or merely reflect it?

THE QUESTION. A linear probe reads compound class off layer-16 activations at 0.91 (chance
0.09). That proves class is PRESENT. It does not prove the generator READS it. The state could
encode class the way your brain encodes a book's font size while reading — accurately present,
and irrelevant to what you predict comes next. If class is present-but-unused, activation
steering is writing into a variable nothing downstream consumes, and no fix to direction, dose,
layer or readout rescues it.

Two independent reasons to take that seriously, both measured:
  * class readability peaks at layer 16 (0.906), holds to layer 23, then COLLAPSES to 0.354 by
    the output layer — the network discards class in exactly the quarter that picks the tokens;
  * base Evo2, which has no class prior and never saw our data, reads class just as well as the
    fine-tuned model (0.911 vs 0.906).

THREE TESTS, all forward passes only (no sampling), all run in ONE model load.

  A  VALIDATE AGAINST WHAT ALREADY WORKS.  Seeding with a real exemplar demonstrably controls
     class (correct_class 0.283 vs a 0.033 floor), and on mismatched arms the output follows the
     SEED, not the tag. So: read the layer-L state of those already-generated sequences and ask
     which class direction they sit closest to. If seeding — the one intervention known to
     control class — moves the state along OUR directions, the directions are causally relevant.
     If seeding controls class without moving them, we are pushing on the wrong variable and
     everything downstream is void. Cheapest and most informative; run first.

  B  TWO-SIDED NUDGE.  Condition on the first half of a real held-out core, score the true second
     half. Nudge toward the TRUE class, then toward a DIFFERENT class. If the model uses class,
     the true-class nudge makes the true continuation MORE likely and the wrong-class nudge makes
     it LESS. The statistic is the GAP between the two, which cancels the generic damage that any
     residual-stream perturbation causes — that cancellation is what makes it trustworthy.

  C  ABLATION.  Instead of adding class, DELETE it: project the class direction out of the state
     and score the true continuation. If removing class information costs the model nothing, it
     was not using it.

Every test is read against `perm*` shuffled-label directions from build_steer_dirs.py, which are
built by the identical recipe on scrambled labels and therefore carry no class information. That
comparison is what separates "steering did something" from "perturbing the residual stream does
something".
"""
from __future__ import annotations

import argparse
import json
import collections
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any

import numpy as np
import re
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference, sequence_loglik  # noqa: E402


# ----------------------------------------------------------------------------- hooks
def _add_hook(model: Any, layer: int, delta: torch.Tensor, start_pos: int = 0):
    """Add a constant vector to block `layer`'s output, ONLY at positions >= start_pos.

    WHY THE GATE. During real generation we would steer the tokens the model WRITES, not the
    prompt it reads. An ungated hook perturbs the context too, which is a different intervention
    and corrupts the very context whose continuation we are scoring. start_pos = number of
    context tokens reproduces the generation-time semantics.
    """
    block = dict(model.named_modules())[f"blocks.{layer}"]

    def _apply(h):
        if start_pos <= 0:
            return h + delta.to(h.dtype)
        if h.shape[1] <= start_pos:
            return h                                  # entirely context: leave untouched
        out = h.clone()
        out[:, start_pos:, :] = out[:, start_pos:, :] + delta.to(h.dtype)
        return out

    def hook(_m, _in, out):
        if isinstance(out, (tuple, list)):
            return (_apply(out[0]),) + tuple(out[1:])
        return _apply(out)

    return block.register_forward_hook(hook)


def _project_out_hook(model: Any, layer: int, u: torch.Tensor, start_pos: int = 0):
    """REMOVE the component along unit vector `u` (the ablation), at positions >= start_pos.

    h <- h - (h . u) u . Deletes the class coordinate while leaving all others untouched.
    """
    block = dict(model.named_modules())[f"blocks.{layer}"]

    def strip(h):
        uu = u.to(h.dtype)
        d = (h * uu).sum(dim=-1, keepdim=True) * uu
        if start_pos <= 0:
            return h - d
        if h.shape[1] <= start_pos:
            return h
        out = h.clone()
        out[:, start_pos:, :] = out[:, start_pos:, :] - d[:, start_pos:, :]
        return out

    def hook(_m, _in, out):
        if isinstance(out, (tuple, list)):
            return (strip(out[0]),) + tuple(out[1:])
        return strip(out)

    return block.register_forward_hook(hook)


def _capture_hook(model: Any, layer: int, sink: dict):
    """Mean-pool the block's output over positions — matches how directions were built."""
    block = dict(model.named_modules())[f"blocks.{layer}"]

    def hook(_m, _in, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        sink["h"] = h[0].float().mean(dim=0).detach().cpu().numpy()

    return block.register_forward_hook(hook)


# ----------------------------------------------------------------------------- helpers
def _load_dirs(npz: Path, layer: int, classes: list[str], need_perm: int):
    """Load directions and FAIL LOUDLY on anything missing.

    WHY SO STRICT (2026-07-29). A silent fallback cost a run: `class_probe_sweep.py` auto-writes
    its own LEGACY sidecar to <stem>.dirs.npz and overwrote a real build. The legacy file has
    `L16_NRPS` (so the direction load succeeded) but no `classunit_*` and no `perm*`. The old
    code defaulted every class-unit to 1.0 and carried on with ZERO shuffled-label controls —
    which makes the whole experiment uninterpretable, since the null band is the only thing
    separating "steering worked" from "perturbing the residual stream does something". It printed
    "0 shuffled-label controls available, using 2" and kept going. Never again: missing controls
    or missing dose units are now hard errors.
    """
    z = np.load(npz)
    # Scan the file for however many controls it actually holds. Hardcoding this (it was 10)
    # silently capped a 30-permutation build at 10 and tripped the >= need_perm guard.
    avail = [int(m.group(1)) for m in (re.match(r"perm(\d+)_L", k) for k in z.files) if m]
    n_perm = max(avail) + 1 if avail else 0
    real, perms, units = {}, [], {}
    missing: list[str] = []
    for c in classes:
        k, ku = f"L{layer}_{c}", f"classunit_L{layer}_{c}"
        if k not in z:
            missing.append(k)
        else:
            real[c] = z[k].astype(np.float64)
        if ku not in z:
            missing.append(ku)
        else:
            units[c] = float(z[ku])
    for i in range(n_perm):
        d = {c: z[f"perm{i}_L{layer}_{c}"].astype(np.float64)
             for c in classes if f"perm{i}_L{layer}_{c}" in z}
        if len(d) == len(classes):
            perms.append(d)

    print(f"[dirs] {npz.name}  layer {layer}")
    print(f"[dirs]   directions      : {len(real)}/{len(classes)} classes")
    print(f"[dirs]   dose units      : {len(units)}/{len(classes)} classes"
          + ("" if len(units) == len(classes) else "   <-- MISSING"))
    print(f"[dirs]   shuffled nulls  : {len(perms)} complete sets (need >= {need_perm})")
    if missing:
        raise SystemExit(
            f"[dirs] ABORT: {len(missing)} required arrays missing, e.g. {missing[:4]}.\n"
            f"       This looks like the LEGACY sidecar written by class_probe_sweep.py, not a\n"
            f"       build_steer_dirs.py output. Rebuild to a *.steerdirs.npz path.")
    if len(perms) < need_perm:
        raise SystemExit(
            f"[dirs] ABORT: only {len(perms)} shuffled-label control sets, need {need_perm}.\n"
            f"       Without them a null result cannot be distinguished from a broken hook.")
    print(f"[dirs]   class-units     : " + ", ".join(f"{c}={units[c]:.3f}" for c in classes))
    if all(abs(units[c] - 1.0) < 1e-9 for c in classes):
        raise SystemExit("[dirs] ABORT: every class-unit is exactly 1.000 — that is the old "
                         "silent default, not a measured scale.")
    return real, perms, units


def _tag_of(rec: dict) -> str:
    return rec.get("taxonomic_tag", "") or ""


# ----------------------------------------------------------------------------- Test A
@torch.no_grad()
def test_a(model, tok, args, real, perms, classes):
    """Does the intervention that WORKS (seeding) move the state along our directions?"""
    arms = {
        "v2_tag":       "seeded, tag agrees  (works: correct_class 0.283)",
        "v2_mismatch":  "seeded, tag DISAGREES (output follows the SEED)",
        "v2_tag_shuf":  "seed codon-shuffled (broken: correct_class 0.000)",
    }
    out = {}
    sink: dict = {}
    handle = _capture_hook(model, args.layer, sink)
    try:
        for arm, blurb in arms.items():
            p = args.seed_dir / f"{arm}.jsonl"
            if not p.exists():
                print(f"[A] !! missing {p}; skipping", file=sys.stderr)
                continue
            recs = [json.loads(l) for l in p.open() if l.strip()]
            random.Random(args.seed).shuffle(recs)
            recs = [r for r in recs if r.get("seed_class") in classes][: args.n_per_arm]
            hits_seed = hits_tag = n = 0
            margins = []
            for r in recs:
                s = (r.get("sequence") or "")[: args.max_score_nt]
                if len(s) < 500:
                    continue
                ids = tok.tokenize(s)
                if not torch.is_tensor(ids):
                    ids = torch.LongTensor(list(ids))
                sink.clear()
                model(ids.to(args.device).view(1, -1))
                h = sink.get("h")
                if h is None:
                    continue
                scores = {c: float(h @ real[c]) for c in classes}
                pred = max(scores, key=scores.get)
                sc, tc = r.get("seed_class"), r.get("tag_class")
                n += 1
                hits_seed += int(pred == sc)
                hits_tag += int(pred == tc)
                if sc in scores and tc in scores and sc != tc:
                    margins.append(scores[sc] - scores[tc])
            if n:
                out[arm] = {
                    "blurb": blurb, "n": n,
                    "predicted_seed_class": round(hits_seed / n, 3),
                    "predicted_tag_class": round(hits_tag / n, 3),
                    "mean_seed_minus_tag_projection": round(st.mean(margins), 4) if margins else None,
                    "chance": round(1 / len(classes), 3),
                }
                print(f"[A] {arm:<13} n={n:<3} -> matches SEED class {out[arm]['predicted_seed_class']:.3f} | "
                      f"matches TAG class {out[arm]['predicted_tag_class']:.3f} | chance {out[arm]['chance']:.3f}")
    finally:
        handle.remove()
    return out


# ----------------------------------------------------------------------------- Tests B / C
@torch.no_grad()
def _score(model, tok, args, context: str, target: str, *, vec=None, ablate_u=None) -> float | None:
    """Per-token log-likelihood of `target` given `context`, optionally under an intervention
    applied ONLY to the target positions (never to the context)."""
    handle = None
    if vec is not None or ablate_u is not None:
        n_ctx = len(list(tok.tokenize(context)))
        if ablate_u is not None:
            handle = _project_out_hook(model, args.layer, ablate_u, start_pos=n_ctx)
        else:
            handle = _add_hook(model, args.layer, vec, start_pos=n_ctx)
    try:
        r = sequence_loglik(model, tok, context, target, args.max_seq_len, args.device, None)
    finally:
        if handle is not None:
            handle.remove()
    return None if r is None else r[0] / max(r[1], 1)      # per-token, so lengths are comparable


def _cores(args, classes) -> list[dict]:
    """Held-out real cores, windowed as a FRACTION of each core's own length.

    WHY FRACTIONS, not fixed nt. A fixed 1000+1000 window silently selects the longest cores:
    it admits 98% of NRPS but only 29% of PKS, 19% of TERPENE and 5% of ECTOINE (median core
    393 nt). Since LENGTH is the confound that wrecked the original directions, selecting cores
    ON length is the worst possible sampling rule. Conditioning on the first `split_frac` of each
    core and scoring the next `split_frac` keeps every class representable at its natural scale.
    """
    byc: dict[str, list] = {c: [] for c in classes}
    total = collections.Counter()
    with args.cores.open() as fh:
        for line in fh:
            r = json.loads(line)
            c = r.get("compound_class")
            if c in byc:
                total[c] += 1
                if len(r.get("sequence", "")) >= args.min_core_nt:
                    byc[c].append(r)
    rng = random.Random(args.seed)
    picked = []
    for c in classes:
        rng.shuffle(byc[c])
        picked.extend(byc[c][: args.n_per_class])
    print("[BC] cores (kept / available in split):")
    for c in classes:
        k = min(len(byc[c]), args.n_per_class)
        print(f"       {c:>9}: {k:>3} kept, {len(byc[c]):>5}/{total[c]:<5} cores clear the "
              f"{args.min_core_nt} nt floor ({100 * len(byc[c]) / max(total[c], 1):.0f}%)")
    rng.shuffle(picked)
    return picked


def _window(seq: str, frac: float) -> tuple[str, str]:
    """(context, scored) as fractions of this core's own length."""
    n = len(seq)
    a, b = int(n * frac), int(n * 2 * frac)
    return seq[:a], seq[a:b]


@torch.no_grad()
def test_bc(model, tok, args, real, perms, classes, units):
    """B: two-sided nudge, swept over DOSE.  C: ablation.  Plus a positive-control anchor.

    WHY A DOSE SWEEP. A single dose cannot distinguish "the model does not use class" from "we
    did not push hard enough". Measured at 1 class-unit: class-units are ~0.5, so the injected
    ||delta|| was ~0.5 against a coherence ceiling near 8 — a 0.45% shift in log-likelihood.
    A null at that dose says nothing. We sweep until either an effect appears or the model
    visibly degrades, which brackets the answer either way.

    WHY A POSITIVE CONTROL. If the true-vs-wrong gap is ~0 at every dose, that is only
    interpretable if the SAME measurement can detect a class effect that we know exists.
    Seeding is that effect: conditioning on a real exemplar of the true class versus one of a
    different class demonstrably controls generated class (0.283 vs 0.067). Scoring the same
    continuation under those two prefixes gives an anchor in identical units, so every steering
    gap can be reported as a FRACTION of a real, known-detectable class effect.
    """
    cores = _cores(args, classes)
    dev = args.device
    # exemplar pool for the positive control: real cores, one per class, held apart from `cores`
    pool: dict[str, str] = {}
    with args.cores.open() as fh:
        for line in fh:
            rr = json.loads(line)
            c = rr.get("compound_class")
            if c in classes and c not in pool and len(rr.get("sequence", "")) >= args.anchor_nt:
                pool[c] = rr["sequence"][: args.anchor_nt]
            if len(pool) == len(classes):
                break

    rng = random.Random(args.seed + 1)
    skipped = collections.Counter()
    rows: list[dict] = []
    for i, r in enumerate(cores):
        cls = r["compound_class"]
        seq = r["sequence"]
        prefix = _tag_of(r)                       # taxonomy ONLY — class must come from the nudge
        cond, tgt = _window(seq, args.split_frac)
        if len(tgt) < 100:
            continue
        other = rng.choice([c for c in classes if c != cls])

        # TWO CONTEXT CONDITIONS, scored side by side for every arm.
        #   "reinforce" = taxonomy + the true sequence's own first half, then score the second
        #                 half. The model can already SEE the class in that prefix, so an
        #                 intervention has little left to add — this is what silently flattened
        #                 both the anchor (-0.00126 +/- 0.00107, undetectable) and every
        #                 steering cell in the previous run.
        #   "create"    = taxonomy only. The intervention is the ONLY thing that could carry
        #                 class. This matches what we actually want to know: can a nudge make
        #                 the model PRODUCE a class, not merely continue one it can already see?
        ctxs = {k: v for k, v in (("create", prefix), ("reinforce", prefix + cond))
                if k in args.contexts}
        row: dict = {"class": cls, "other": other, "core_nt": len(seq), "scored_nt": len(tgt)}
        # Require BOTH contexts to score. `reinforce` prepends the core's own first 40%, so on
        # long cores context+target can exceed max_seq_len and sequence_loglik returns None.
        # Previously this `break`-ed, leaving rows with create keys but no reinforce keys — which
        # (a) crashed the summariser on KeyError and (b) silently made the two contexts
        # UNPAIRED (n=100 vs n=95), so create-vs-reinforce was not a like-for-like comparison.
        bases = {cn: _score(model, tok, args, ctx, tgt) for cn, ctx in ctxs.items()}
        if any(v is None for v in bases.values()):
            skipped["ctx_too_long"] += 1
            continue
        for cn, b in bases.items():
            row[f"{cn}_ll_base"] = b

        for cname, ctx in ctxs.items():
            # --- positive control: real exemplar of the true class vs of the wrong class ---
            if pool.get(cls) and pool.get(other):
                at = _score(model, tok, args, ctx + pool[cls], tgt)
                aw = _score(model, tok, args, ctx + pool[other], tgt)
                if at is not None and aw is not None:
                    row[f"{cname}_anchor_gap"] = at - aw
            # --- Test B: two-sided nudge, swept over dose ---
            for dose in args.doses:
                for kind, dmap in ([("real", real)] +
                                   [(f"perm{j}", p) for j, p in enumerate(perms[: args.n_perm_use])]):
                    scale = dose * units.get(cls, 1.0)
                    for tag, c in (("true", cls), ("wrong", other)):
                        v = torch.tensor(dmap[c] * scale, dtype=torch.float32, device=dev)
                        row[f"{cname}_d{dose}_ll_{kind}_{tag}"] = _score(
                            model, tok, args, ctx, tgt, vec=v)
            # --- Test C: ablation (dose-free projection) ---
            for kind, dmap in ([("real", real)] +
                               [(f"perm{j}", p) for j, p in enumerate(perms[: args.n_perm_use])]):
                u = torch.tensor(dmap[cls] / np.linalg.norm(dmap[cls]),
                                 dtype=torch.float32, device=dev)
                row[f"{cname}_ll_{kind}_ablate"] = _score(model, tok, args, ctx, tgt, ablate_u=u)
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[BC] scored {i + 1}/{len(cores)}", flush=True)
    if skipped:
        print(f"[BC] skipped {sum(skipped.values())} cores: {dict(skipped)} "
              f"(both contexts must score, so the two stay strictly paired)")
    return rows


def summarize_bc(rows, n_perm_use, doses, ctx):
    print("\n" + "#" * 78)
    print(f"##  CONTEXT = {ctx.upper()}" + ("   (taxonomy only -- can a nudge CREATE the class?)"
          if ctx == "create" else "   (taxonomy + the true first half -- mere reinforcement)"))
    print("#" * 78)
    out = {"n": len(rows)}
    kinds = ["real"] + [f"perm{j}" for j in range(n_perm_use)]

    # --- the anchor first: can this measurement see a class effect we KNOW exists? ---
    anc = [r[f"{ctx}_anchor_gap"] for r in rows if r.get(f"{ctx}_anchor_gap") is not None]
    anchor = st.mean(anc) if anc else None
    print("\n" + "=" * 78)
    print("POSITIVE CONTROL — real class-B exemplar vs class-A exemplar as prefix.")
    print("         This is the class effect we KNOW controls generation (0.283 vs 0.067).")
    print("=" * 78)
    if anchor is not None:
        ase = st.stdev(anc) / len(anc) ** 0.5 if len(anc) > 1 else 0.0
        print(f"  exemplar ONLY as context   : {anchor:+.5f} +/- {ase:.5f} log-lik/token  (n={len(anc)})")
        wc = [r["reinforce_anchor_gap"] for r in rows if r.get("reinforce_anchor_gap") is not None] if ctx == "create" else []
        if wc:
            wm = st.mean(wc); wse = st.stdev(wc) / len(wc) ** 0.5 if len(wc) > 1 else 0.0
            print(f"  same, WITH true prefix     : {wm:+.5f} +/- {wse:.5f}   <- the old, masked design")
            out["anchor_gap_withcond"], out["anchor_withcond_se"] = wm, wse
        ok = abs(anchor) > 2 * ase
        print(f"  {'DETECTABLE — the measurement can see a class effect, so a steering null MEANS something' if ok else 'NOT DETECTABLE — the instrument is too insensitive; no steering result here is interpretable'}")
        out["anchor_gap"], out["anchor_se"], out["anchor_detectable"] = anchor, ase, bool(ok)
    else:
        print("  (no anchor computed)")

    print("\n" + "=" * 78)
    print("TEST B — two-sided nudge, SWEPT OVER DOSE.")
    print("         GAP = log-lik(TRUE-class nudge) - log-lik(WRONG-class nudge), per token.")
    print("         POSITIVE => the model uses class. Generic damage cancels in the gap.")
    print("=" * 78)
    print(f"{'dose':>6} {'REAL gap':>11} {'+/-':>8} {'perm mean':>10} {'z':>6} {'perm p':>7} "
          f"{'exceed':>7} {'vs anch':>8}")
    out["by_dose"] = {}
    for dose in doses:
        gaps = {}
        for k in kinds:
            kt, kw = f"{ctx}_d{dose}_ll_{k}_true", f"{ctx}_d{dose}_ll_{k}_wrong"
            g = [r[kt] - r[kw] for r in rows if r.get(kt) is not None and r.get(kw) is not None]
            if g:
                gaps[k] = g
        if "real" not in gaps:
            continue
        rm = st.mean(gaps["real"])
        rse = st.stdev(gaps["real"]) / len(gaps["real"]) ** 0.5 if len(gaps["real"]) > 1 else 0.0
        nullm = [st.mean(gaps[k]) for k in gaps if k != "real"]
        # PROPER permutation statistics. The previous rule was "beat the MAX of the controls",
        # which gets STRICTER as controls are added -- so the same effect read "USES CLASS" with
        # 2 controls and "inside null" with 6. Backwards. Report instead:
        #   z    : how many control-distribution SDs the real effect sits above the controls
        #   p    : exact permutation p = (#controls >= real + 1) / (#controls + 1)
        # NOTE p is FLOORED at 1/(n_perm+1) -- with 6 controls the best attainable p is 0.143,
        # so a "non-significant" verdict may be a limit of the control count, not of the data.
        pm = st.mean(nullm) if nullm else 0.0
        ps = st.stdev(nullm) if len(nullm) > 1 else float("nan")
        z = (rm - pm) / ps if ps == ps and ps > 0 else float("nan")
        n_ex = sum(1 for v in nullm if v >= rm)
        pval = (n_ex + 1) / (len(nullm) + 1) if nullm else float("nan")
        floor = 1 / (len(nullm) + 1) if nullm else float("nan")
        frac = f"{rm / anchor:>8.2f}x" if anchor else "      n/a"
        note = "AT p-FLOOR" if abs(pval - floor) < 1e-9 else ""
        print(f"{dose:>6} {rm:>+11.5f} {rse:>8.5f} {pm:>+10.5f} {z:>6.1f} {pval:>7.3f} "
              f"{n_ex:>3}/{len(nullm):<3} {frac} {note}")
        out["by_dose"][str(dose)] = {"real": rm, "se": rse, "perm_mean": pm, "perm_sd": ps,
                                     "z": z, "perm_p": pval, "p_floor": floor,
                                     "n_controls": len(nullm)}
    # damage guard: how much is the model being perturbed at all, per dose?
    print(f"\n{'dose':>6} {'mean |shift| vs unsteered':>26}  (0 => the nudge is doing nothing)")
    for dose in doses:
        v = [abs(r[f"{ctx}_d{dose}_ll_real_true"] - r[f"{ctx}_ll_base"]) for r in rows
             if r.get(f"{ctx}_d{dose}_ll_real_true") is not None]
        if v:
            base = abs(st.mean([r[f"{ctx}_ll_base"] for r in rows]))
            print(f"{dose:>6} {st.mean(v):>26.5f}  ({100 * st.mean(v) / base:.2f}% of baseline)")
            out["by_dose"].setdefault(str(dose), {})["mean_abs_shift"] = st.mean(v)

    print("\n" + "=" * 78)
    print("TEST C — ablation.  Delta = log-lik(class direction DELETED) - log-lik(untouched).")
    print("         NEGATIVE => deleting class hurt => the model was using it.")
    print("=" * 78)
    b = [r[f"{ctx}_ll_base"] for r in rows]
    for k in kinds:
        d = [r[f"{ctx}_ll_{k}_ablate"] - r[f"{ctx}_ll_base"] for r in rows if r.get(f"{ctx}_ll_{k}_ablate") is not None]
        if d:
            m, se = st.mean(d), (st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0)
            lab = "REAL directions" if k == "real" else f"  shuffled {k}"
            print(f"  {lab:<20} delta = {m:+.5f} +/- {se:.5f}")
    out["ll_base_mean"] = st.mean(b) if b else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768/checkpoints/step_1200"))
    ap.add_argument("--dirs-npz", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/acts_v2.dirs_v2.npz"))
    ap.add_argument("--cores", type=Path, default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"),
                    help="HELD-OUT cores. The cached activations came from val.jsonl, so test.jsonl "
                         "keeps the causal tests independent of direction fitting.")
    ap.add_argument("--seed-dir", type=Path, default=Path("/data2/ds85/bgcmodel_runs/seed_deconfound"))
    ap.add_argument("--out-json", type=Path, default=Path("/data2/ds85/bgcmodel_runs/steer_causal/phase1.json"))
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE", "ECTOINE", "RIPP"])
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--tests", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--doses", type=float, nargs="+", default=[1, 2, 4, 8, 16],
                    help="Nudge sizes in CLASS-UNITS. Class-units are ~0.5 here, so this spans\n"
                         "||delta|| ~0.5 to ~8 — from barely-a-touch to the measured coherence\n"
                         "ceiling. A single dose cannot separate 'unused' from 'under-pushed'.")
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--n-per-arm", type=int, default=45)
    ap.add_argument("--n-perm-use", type=int, default=3, help="Shuffled-label controls to run (cost scales).")
    ap.add_argument("--contexts", nargs="+", default=["create", "reinforce"],
                    choices=["create", "reinforce"],
                    help="'create' (taxonomy only) is the question we care about; "
                         "'reinforce' is the diagnostic showing how much a true-sequence "
                         "prefix masks the effect. Drop it to halve cost.")
    ap.add_argument("--split-frac", type=float, default=0.4,
                    help="Condition on the first FRAC of each core, score the next FRAC. "
                         "Fractions, not fixed nt, so we do not select cores on length -- the "
                         "very confound that wrecked the original directions.")
    ap.add_argument("--min-core-nt", type=int, default=300)
    ap.add_argument("--anchor-nt", type=int, default=1000,
                    help="Exemplar length for the positive-control prefix.")
    ap.add_argument("--max-score-nt", type=int, default=4096)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    real, perms, units = _load_dirs(args.dirs_npz, args.layer, args.classes, args.n_perm_use)

    adapter = args.adapter
    if adapter and not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
        adapter = adapter / "adapter"
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)
    model, tok = wrapper.model, wrapper.tokenizer
    model.eval()

    result: dict = {"layer": args.layer, "doses_class_units": args.doses, "classes": args.classes}
    if "A" in args.tests:
        print("\n" + "=" * 78)
        print("TEST A — does SEEDING (the intervention that works) move the state along our")
        print("         directions? Read the class off the generated sequence's own state.")
        print("=" * 78)
        result["test_A"] = test_a(model, tok, args, real, perms, args.classes)
    if "B" in args.tests or "C" in args.tests:
        rows = test_bc(model, tok, args, real, perms, args.classes, units)
        result["test_BC_rows"] = rows
        result["test_BC_summary"] = {c: summarize_bc(rows, args.n_perm_use, args.doses, c)
                                     for c in args.contexts}

    args.out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
