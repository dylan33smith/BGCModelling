#!/usr/bin/env python3
"""GPU equivalence gate for BATCHED vs SEQUENTIAL generation.

Batched generation (scripts/generate_bgc.py --batch-size>1) left-pads prompts to
a uniform length so vortex actually batches them (vortex silently de-batches
mixed-length prompts, and right-pads with no attention mask). The open question
is whether the LEADING PAD perturbs the model's output vs. clean single-prompt
generation. This script answers it empirically before the faster batched path is
trusted in the auto-eval.

Two checks must BOTH pass:

  (1) PADDING EQUIVALENCE (greedy, top_k=1 ⇒ argmax, deterministic):
      take held-out prompts of DIFFERENT lengths (so padding is actually
      exercised — the gate fails if <2 distinct lengths), generate each the
      SEQUENTIAL way (one batch-of-1 call each, exactly what generate_one does),
      then generate them all in ONE left-padded batched call, and compare. Padding
      corruption shows up as an EARLY, severe divergence (the first generated
      tokens differ), whereas harmless batched-vs-unbatched floating-point jitter
      only diverges late — so the gate keys on an exact HEAD-token match.

  (2) SAMPLE INDEPENDENCE (stochastic, top_k=4 — the PRODUCTION regime):
      the same prompt repeated in a batch must still yield INDEPENDENT samples,
      not collapse to one sequence. Fails only when the sequential path stays
      diverse but the batched path collapses (robust to vortex reseeding).

Plus a real speedup (else batching is pointless). Writes a JSON verdict and exits
0 (PASS → use batched) or 1 (FAIL/error → keep sequential). The caller treats any
nonzero exit as "keep sequential", so this script never has to be trusted to be
bug-free to be SAFE.

Usage:
  python scripts/validate_batched_generation.py \
      --adapter /data2/.../checkpoints/step_250 \
      --from-jsonl /data2/.../splits_dedup/val.jsonl \
      --classes NRPS PKS PKS_NRPS_HYBRID --n-prompts 3 \
      --n-tokens 3000 --out /data2/.../batch_validation_step250.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_bgc as G  # noqa: E402


def _longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _pick_prompts(records: list[dict], classes: set[str], n: int,
                  rng: random.Random) -> list[dict]:
    """Pick n prompts, preferring DISTINCT prefix lengths so left-padding is
    genuinely exercised (a batch of equal-length prompts would not test padding)."""
    pool = [r for r in records if (not classes or r.get("compound_class") in classes)]
    rng.shuffle(pool)
    picked, seen_len = [], set()
    for r in pool:
        plen = len(G.build_prefix(r.get("compound_class", "UNKNOWN"),
                                  r.get("taxonomic_tag", "")))
        if plen in seen_len:
            continue
        seen_len.add(plen)
        picked.append({"compound_class": r.get("compound_class", "UNKNOWN"),
                       "taxonomic_tag": r.get("taxonomic_tag", "")})
        if len(picked) >= n:
            break
    # top up (allow duplicate lengths) if we couldn't find enough distinct ones
    if len(picked) < n:
        for r in pool:
            cand = {"compound_class": r.get("compound_class", "UNKNOWN"),
                    "taxonomic_tag": r.get("taxonomic_tag", "")}
            if cand not in picked:
                picked.append(cand)
            if len(picked) >= n:
                break
    return picked[:n]


def _generate(wrapper, prefixes: list[str], n_tokens: int,
              top_k: int = 1, temperature: float = 1.0, top_p: float = 1.0) -> list[str]:
    """One batched call over left-padded prefixes; returns cleaned nucleotide
    sequences (generation-only output → extract_sequence). Defaults to GREEDY
    (top_k=1) for the deterministic padding-equivalence probe; pass top_k>1 for a
    stochastic probe (the independence check)."""
    out = wrapper.generate(
        prompt_seqs=G.left_pad_to_uniform(prefixes), n_tokens=n_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p, batched=True,
        cached_generation=True, verbose=0,
    )
    return [G.extract_sequence(g)["sequence"] for g in G._gen_sequences(out)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Checkpoint dir (with adapter/) or run dir (uses best/).")
    ap.add_argument("--from-jsonl", type=Path, required=True)
    ap.add_argument("--classes", nargs="*", default=["NRPS", "PKS", "PKS_NRPS_HYBRID"])
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--n-tokens", type=int, default=3000,
                    help="Greedy tokens per sequence for the probe (short = fast; "
                         "padding corruption diverges within the first few).")
    ap.add_argument("--head-len", type=int, default=50,
                    help="Leading nucleotides that must match EXACTLY for a pass.")
    ap.add_argument("--min-speedup", type=float, default=1.5,
                    help="Required seq_time/bat_time for batched to be worth it.")
    ap.add_argument("--min-len", type=int, default=0,
                    help="Min cleaned length for a prompt to be judged (default: head-len).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    min_len = args.min_len or args.head_len

    verdict = {"passed": False, "reason": "init", "checkpoint": str(args.adapter),
               "n_tokens": args.n_tokens, "head_len": args.head_len,
               "min_speedup": args.min_speedup}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def emit(rc: int) -> int:
        args.out.write_text(json.dumps(verdict, indent=2) + "\n")
        print(f"[validate_batched] {'PASS' if verdict['passed'] else 'FAIL'}: "
              f"{verdict['reason']}", file=sys.stderr)
        return rc

    try:
        records = [json.loads(l) for l in args.from_jsonl.open()]
        prompts = _pick_prompts(records, set(args.classes), args.n_prompts,
                                random.Random(args.seed))
        if len(prompts) < 2:
            verdict["reason"] = f"need >=2 prompts, found {len(prompts)}"
            return emit(1)
        prefixes = [G.build_prefix(p["compound_class"], p["taxonomic_tag"]) for p in prompts]
        verdict["prefix_lens"] = [len(x) for x in prefixes]
        verdict["distinct_prefix_lens"] = len(set(verdict["prefix_lens"]))
        # Padding is only exercised when prompts actually differ in length. If they
        # do not, left_pad_to_uniform is a no-op and a PASS would be meaningless —
        # fail conservatively (caller keeps sequential).
        if verdict["distinct_prefix_lens"] < 2:
            verdict["reason"] = (f"padding not exercised: <2 distinct prompt lengths "
                                 f"({verdict['prefix_lens']})")
            return emit(1)

        from evo2_inference import load_evo2_wrapper_for_inference
        print(f"[validate_batched] loading {args.adapter} ...", file=sys.stderr, flush=True)
        wrapper = load_evo2_wrapper_for_inference(args.adapter, device=args.device)

        # warm up CUDA/kernels so neither timed path eats first-call overhead
        _ = _generate(wrapper, [prefixes[0]], 16)

        # SEQUENTIAL reference: one batch-of-1 call per prompt (== generate_one)
        t0 = time.perf_counter()
        seq = [_generate(wrapper, [p], args.n_tokens)[0] for p in prefixes]
        seq_time = time.perf_counter() - t0

        # BATCHED: one left-padded call over all prompts
        t0 = time.perf_counter()
        bat = _generate(wrapper, prefixes, args.n_tokens)
        bat_time = time.perf_counter() - t0

        per = []
        for p, s, b in zip(prompts, seq, bat):
            lcp = _longest_common_prefix(s, b)
            per.append({
                "class": p["compound_class"],
                "prefix_len": len(G.build_prefix(p["compound_class"], p["taxonomic_tag"])),
                "len_seq": len(s), "len_bat": len(b),
                "head_match": s[:args.head_len] == b[:args.head_len] and len(s) >= args.head_len,
                "lcp": lcp,
                "lcp_frac": round(lcp / max(1, min(len(s), len(b))), 4),
                "exact": s == b,
            })
        speedup = round(seq_time / bat_time, 3) if bat_time > 0 else 0.0
        all_nonempty = all(d["len_seq"] >= min_len and d["len_bat"] >= min_len for d in per)
        all_head = all(d["head_match"] for d in per)
        mean_lcp = round(sum(d["lcp_frac"] for d in per) / len(per), 4)

        # STOCHASTIC INDEPENDENCE: production decoding is stochastic (top_k=4), and
        # a batch can contain identical prompts (e.g. --n>1, or duplicate taxa).
        # Batched generation must still draw INDEPENDENT samples per row — it must
        # not collapse repeated prompts to one sequence. Probe: same prompt ×rep_n,
        # stochastic. Robust to vortex possibly reseeding per call — only a FAIL
        # when the sequential path stays diverse but the batched path collapses
        # (i.e. batching specifically destroys diversity sequential would have).
        rep_n = 3
        rep = [prefixes[0]] * rep_n
        seq_rep = [_generate(wrapper, [p], 300, top_k=4, temperature=1.0, top_p=1.0)[0]
                   for p in rep]
        bat_rep = _generate(wrapper, rep, 300, top_k=4, temperature=1.0, top_p=1.0)
        seq_rep_diverse = len(set(seq_rep)) > 1
        bat_rep_collapsed = len(set(bat_rep)) == 1
        indep_ok = not (seq_rep_diverse and bat_rep_collapsed)

        verdict.update({
            "seq_time_s": round(seq_time, 2), "bat_time_s": round(bat_time, 2),
            "speedup": speedup, "all_head_match": all_head,
            "all_nonempty": all_nonempty, "mean_lcp_frac": mean_lcp,
            "per_prompt": per,
            "independence": {"rep_n": rep_n, "seq_distinct": len(set(seq_rep)),
                             "bat_distinct": len(set(bat_rep)),
                             "seq_diverse": seq_rep_diverse,
                             "bat_collapsed": bat_rep_collapsed, "indep_ok": indep_ok},
        })

        reasons = []
        if not all_nonempty:
            reasons.append(f"some sequence shorter than min_len={min_len}")
        if not all_head:
            reasons.append(f"head ({args.head_len} nt) mismatch on >=1 prompt "
                           "→ padding perturbs output")
        if speedup < args.min_speedup:
            reasons.append(f"speedup {speedup}x < required {args.min_speedup}x")
        if not indep_ok:
            reasons.append("batched collapses repeated-prompt samples that sequential "
                           "keeps diverse → per-row sampling not independent")
        verdict["passed"] = not reasons
        verdict["reason"] = ("batched ≡ sequential on head tokens, samples stay "
                             "independent, and is faster") \
            if verdict["passed"] else "; ".join(reasons)
        return emit(0 if verdict["passed"] else 1)

    except Exception as e:  # any failure → FAIL → caller keeps sequential
        import traceback
        verdict["reason"] = f"exception: {type(e).__name__}: {e}"
        verdict["traceback"] = traceback.format_exc()
        return emit(1)


if __name__ == "__main__":
    raise SystemExit(main())
