#!/usr/bin/env python
"""DISCRIMINATOR-GUIDED DECODING — search output space instead of asking the model for a class.

WHY THIS IS NOT ANOTHER STEERING VARIANT. Everything this project has tried so far required the
GENERATOR to consume a class conditioner: a label token, a soft prefix, an activation edit. The
2026-08-10 programme established it does not -- the class coordinate is present at 0.911 probe
accuracy and behaviourally inert. Guided decoding needs none of that. It samples several
continuations, scores each with an EXTERNAL classifier, and keeps the best. The model never has
to read anything; it only has to produce class-X-ish output sometimes, and the scorer only has to
recognise it. That sidesteps the exact failure the whole programme ran into.

This is the chunk-level member of the FUDGE / GeDi / PPLM family (sample -> score -> select ->
continue), not per-token FUDGE. The reason is deliberate: per-token reweighting needs a one-token
lookahead forward pass per candidate per step with branch-and-rollback on the KV cache, and this
codebase has a documented history of vortex cache bugs (see bugs.md: seqlen_offset corruption on
resumed generate calls, silent de-batching). Chunk-level selection uses only plain, already-tested
generate calls. It tests the same question -- can an external class scorer steer the output? --
without betting the result on cache bookkeeping.

THREE ARMS, and the middle one is the one that matters:

  --select best    guided: keep the candidate the discriminator scores highest for the target
  --select random  CONTROL: generate the SAME candidates, keep a random one. This is the
                   shuffled-label analogue. Best-of-N with a USELESS scorer is exactly random
                   selection among N, so anything `best` does not beat `random` by is not the
                   discriminator doing work -- it is just resampling variance and chunk-boundary
                   effects. With a shared --seed the two arms see identical candidate sets at the
                   first chunk and diverge only because of the selection rule.
  --n-candidates 1 floor: no selection at all, same chunking and prompt handling otherwise.

CIRCULARITY -- THE TRAP THIS EXPERIMENT IS BUILT AROUND. We select on the discriminator's score.
Reporting that same score as the result would be guaranteed success by construction: we optimised
it. The discriminator's score is recorded here ONLY as a manipulation check ("did selection
actually move the quantity we selected on?"). The RESULT must come from instruments that had no
part in selection -- antiSMASH (gold standard, FPR 0.000) and the Pfam class markers. Every
downstream summary in this project must keep that separation, and the fields are named so a
reader cannot confuse them: `guide_*` for the thing we optimised, everything else for evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402
from generate_bgc import _gen_sequences, assemble_record, extract_sequence  # noqa: E402
from probe_score_generations import _embed  # noqa: E402  (same pooling recipe as the fit set)


def select_candidate(probs: np.ndarray, target_idx: int, usable: list[int],
                     objective: str, rule: str, rng):
    """Pick one candidate. Returns (index into the ORIGINAL cands list, objective values).

    `probs` has one row per USABLE candidate, in the order of `usable` -- so an argmax over it
    yields a LOCAL index that must be mapped back through `usable`. Getting that mapping wrong
    would silently select a different candidate than the one that scored best, i.e. turn the
    guided arm into the random arm while still labelling it "guided". Extracted here so it is
    unit-testable; see tests/test_guided_decoding.py.
    """
    tgt = probs[:, target_idx]
    if objective == "margin":
        other = np.delete(probs, target_idx, axis=1).max(axis=1)
        obj = tgt - other
    else:
        obj = np.log(np.clip(tgt, 1e-12, None))
    local = int(np.argmax(obj)) if rule == "best" else rng.randrange(len(usable))
    return usable[local], obj, local, tgt


def load_discriminator(path: Path):
    import joblib
    blob = joblib.load(path)
    print(f"[guide] discriminator {path.name}: layer {blob['layer']}, "
          f"grouped balanced acc {blob['grouped_balanced_acc']:.4f} "
          f"(chance {blob['chance']:.4f})", file=sys.stderr)
    print(f"[guide]   per prefix length: "
          + ", ".join(f"{k}={v:.3f}" for k, v in blob["per_prefix_len_acc"].items()),
          file=sys.stderr)
    return blob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/phase1_lora_prod_20260617_095202_L32768"
                                 "/checkpoints/step_1200"))
    ap.add_argument("--discriminator", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/class_probe_sweep/prefix_discriminator_L16.joblib"))
    ap.add_argument("--from-jsonl", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/valtest_eval.jsonl"),
                    help="Source of real taxonomic tags (HELD OUT; the discriminator is train-only).")
    ap.add_argument("--target-class", required=True,
                    help="Class to guide TOWARD. Also recorded as compound_class (what we asked for).")
    ap.add_argument("--tag-class", required=True,
                    help="Comma list of classes to draw taxonomy tags from. Give the SAME list to "
                         "every arm so tags are identical and arms are paired per taxon.")
    ap.add_argument("--seed-nt", type=int, default=0,
                    help="If >0, prompt with this many nt of a REAL core of --target-class before "
                         "generating. THIS IS USUALLY REQUIRED. Measured: the unseeded regime has "
                         "no floor to select from -- antiSMASH is_bgc 0/25 (Phase 2) and "
                         "correct_class 0/12 in every soft-prefix arm -- so best-of-N is choosing "
                         "among candidates none of which is a BGC, and can only return one. The "
                         "seeded regime has a measured floor (0.283 matched / 0.067 mismatched), "
                         "i.e. actual dynamic range for selection to act on.")
    ap.add_argument("--seed-src", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/valtest_eval.jsonl"),
                    help="Real cores to draw seed exemplars from (held out).")
    ap.add_argument("--n", type=int, default=6, help="Sequences to generate.")
    ap.add_argument("--n-candidates", type=int, default=4,
                    help="Candidates per chunk. 1 = no selection (the plain floor).")
    ap.add_argument("--chunk-nt", type=int, default=600)
    ap.add_argument("--n-chunks", type=int, default=5)
    ap.add_argument("--select", choices=["best", "random"], default="best",
                    help="'best' = guided. 'random' = the control that isolates the "
                         "discriminator's contribution from plain resampling variance.")
    ap.add_argument("--objective", choices=["logp", "margin"], default="logp",
                    help="logp = log P(target) (FUDGE's Bayes term). margin = P(target) minus the "
                         "best rival, which also penalises ambiguous candidates.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-nt", type=int, default=4096, help="Cap for the scoring forward pass.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    args = ap.parse_args()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.n_candidates < 1:
        ap.error("--n-candidates must be >= 1")
    if args.n_candidates == 1 and args.select == "best":
        print("[guide] NOTE: --n-candidates 1 means there is nothing to select between; this is "
              "the plain floor arm regardless of --select.", file=sys.stderr)

    disc = load_discriminator(args.discriminator)
    pipe, classes = disc["pipe"], list(disc["classes"])
    layer = int(disc["layer"])
    if args.target_class not in classes:
        raise SystemExit(f"[guide] ABORT: target {args.target_class!r} is not one of the "
                         f"discriminator's {len(classes)} classes -- it cannot be scored, so "
                         f"guidance toward it is undefined.")
    ci = {c: i for i, c in enumerate(classes)}

    adapter = args.adapter
    if adapter and not (adapter / "adapter_config.json").exists() and (adapter / "adapter").exists():
        adapter = adapter / "adapter"
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)

    want = [c.strip() for c in str(args.tag_class).replace(",", " ").split() if c.strip()]
    tags = sorted({r["taxonomic_tag"] for r in
                   (json.loads(l) for l in args.from_jsonl.open() if l.strip())
                   if r.get("compound_class") in want and r.get("taxonomic_tag")})
    if not tags:
        raise SystemExit(f"[guide] no taxonomic tags for {want} in {args.from_jsonl}")
    rng_tags = random.Random(args.seed)
    rng_tags.shuffle(tags)
    print(f"[guide] {len(tags)} candidate taxa from {want}; generating {args.n} "
          f"(identical taxa in every arm sharing this --seed/--tag-class)", file=sys.stderr)

    def score_batch(seqs: list[str]) -> np.ndarray:
        """P(class | mean-pooled layer-`layer` activations), same pooling as the fit set."""
        X = _embed(wrapper, seqs, [layer], args.device, args.max_nt)[layer]
        return pipe.predict_proba(X)

    seeds: list[dict] = []
    if args.seed_nt > 0:
        pool = [r for r in (json.loads(l) for l in args.seed_src.open() if l.strip())
                if r.get("compound_class") == args.target_class
                and len(r.get("sequence", "")) >= args.seed_nt + 500]
        if not pool:
            raise SystemExit(f"[guide] no {args.target_class} cores with >= {args.seed_nt + 500} nt "
                             f"in {args.seed_src}")
        pool.sort(key=lambda r: str(r.get("accession") or r.get("id") or ""))
        random.Random(args.seed).shuffle(pool)
        seeds = pool[: args.n]
        print(f"[guide] SEEDED: {len(seeds)} real {args.target_class} exemplars, "
              f"{args.seed_nt} nt each (identical across arms sharing --seed)", file=sys.stderr)

    class _A:
        temperature = args.temperature; top_k = args.top_k; top_p = args.top_p
        max_new_tokens = args.chunk_nt * args.n_chunks; max_n_frac = 0.10

    n_written = 0
    with args.out_jsonl.open("w") as fh:
        for i in range(args.n):
            tax = tags[i % len(tags)]
            seed_rec = seeds[i % len(seeds)] if seeds else None
            seed_dna = seed_rec["sequence"][: args.seed_nt] if seed_rec else ""
            if seed_rec is not None:
                # Taxonomy comes from the SEED's own record when seeded, so the exemplar and the
                # taxon are consistent -- exactly as Phase 3 did, keeping the two arms comparable
                # to the existing seeded baselines.
                tax = seed_rec.get("taxonomic_tag", "") or tax
            seq = ""
            trace = []
            rng_sel = random.Random(args.seed * 1000 + i)
            for chunk in range(args.n_chunks):
                cands: list[str] = []
                for j in range(args.n_candidates):
                    # Distinct seed per (sequence, chunk, candidate) so candidates differ, but
                    # DETERMINISTIC given --seed, so the `best` and `random` arms see identical
                    # candidate sets wherever their histories have not yet diverged.
                    torch.manual_seed(args.seed * 100003 + i * 997 + chunk * 31 + j)
                    out = wrapper.generate(prompt_seqs=[tax + seed_dna + seq], n_tokens=args.chunk_nt,
                                           temperature=args.temperature, top_k=args.top_k,
                                           top_p=args.top_p, cached_generation=True, verbose=0)
                    cands.append(extract_sequence(_gen_sequences(out)[0])["sequence"])
                usable = [k for k, c in enumerate(cands) if len(c) > 0]
                if not usable:
                    trace.append({"chunk": chunk, "aborted": "all candidates empty"})
                    break
                # Score the CONTINUATION so far, never the seed. The seed is 1000 nt of real
                # class-X DNA; including it would make every candidate look class-X regardless of
                # what was generated, collapsing the discrimination we are testing to zero.
                # (Cost: the discriminator was fit on prefixes FROM THE START of real cores, so a
                # continuation is a mid-sequence fragment -- a distribution shift it shares with
                # every other use of this probe on generations.)
                probs = score_batch([seq + cands[k] for k in usable])
                pick, obj, pick_local, tgt = select_candidate(
                    probs, ci[args.target_class], usable, args.objective, args.select, rng_sel)
                trace.append({
                    "chunk": chunk, "n_candidates": len(cands), "n_usable": len(usable),
                    "chosen": pick, "rule": args.select,
                    # the quantity we OPTIMISED -- a manipulation check, never the result
                    "guide_obj_all": [round(float(v), 5) for v in obj],
                    "guide_obj_chosen": round(float(obj[pick_local]), 5),
                    "guide_p_target_chosen": round(float(tgt[pick_local]), 5),
                    "guide_p_target_max": round(float(tgt.max()), 5),
                    "cand_lens": [len(c) for c in cands],
                })
                seq += cands[pick]
            ex = {"sequence": seq, "hit_eos": False}
            rec = assemble_record(args.target_class, tax, ex["sequence"], ex["hit_eos"], 1, _A)
            rec.update({
                "guided": True, "guide_target_class": args.target_class,
                "seeded": args.seed_nt > 0, "seed_nt": len(seed_dna),
                "seed_class": args.target_class if seed_rec else None,
                "seed_accession": (seed_rec.get("accession") or seed_rec.get("id")) if seed_rec else None,
                "scored_span": "continuation_only" if seed_rec else "full",
                "guide_select": args.select, "guide_objective": args.objective,
                "guide_n_candidates": args.n_candidates,
                "guide_chunk_nt": args.chunk_nt, "guide_n_chunks": args.n_chunks,
                "guide_discriminator": args.discriminator.name,
                "guide_trace": trace,
                "tag_class": ",".join(want), "tax_idx": i % len(tags),
                "prompt": ("taxonomy+seed+guided_decoding" if seed_rec
                           else "taxonomy_only+guided_decoding"),
                "accession": f"GD_{args.target_class}_{args.select}_{i}",
            })
            fh.write(json.dumps(rec) + "\n")
            n_written += 1
            print(f"[guide] {i + 1}/{args.n}  len={len(seq)}  "
                  f"mean guide_p_target={np.mean([t.get('guide_p_target_chosen', np.nan) for t in trace if 'guide_p_target_chosen' in t]):.4f}",
                  file=sys.stderr, flush=True)
    print(f"[guide] wrote {n_written} -> {args.out_jsonl}", file=sys.stderr)
    print(f"[guide] REMINDER: guide_* fields are what we SELECTED ON. Do not report them as the "
          f"result -- score these with antiSMASH / class markers.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
