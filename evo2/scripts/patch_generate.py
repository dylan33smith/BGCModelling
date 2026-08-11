#!/usr/bin/env python
"""PHASE B: does a transplanted activation transfer CLASS, or only local sequence?

WHAT PHASE A ESTABLISHED (activation_patching.py, k-sweep). The model DOES read mid-layer
activations. Substituting a real donor's state over just 10 of 1000 positions at layer 16 moves the
next-token distribution 41% of the way toward the donor (controls: same-class donor 0.128, shuffled
0.129, noise -0.099); at 200 positions it reaches 0.837. The earlier one-position null was a
LEVERAGE artifact, not blindness. This matters because our steering edit -- rank-1, every position,
2.8-11.4 class-units, provably landing far off the data manifold -- moved the output essentially not
at all. The difference is the SHAPE of the edit, which is what the ACE pre-check predicted.

WHY PHASE A CANNOT FINISH THE ARGUMENT. Evo2's vocabulary is bytes, so the next-token distribution
is over A/C/G/T and compound class is not legible in it. Worse, at small k the patched positions ARE
the immediate context of the next base, so a high alignment may be nothing more than ordinary local
sequence continuation. Distributional alignment cannot separate "the donor's CLASS transferred" from
"the donor's last few nucleotides transferred".

WHAT THIS SCRIPT DOES. Patch layer L's representation of the CONTEXT with a donor of a DIFFERENT
class, then let the model generate a full continuation and score it with instruments that took no
part in the intervention. The question is whether the continuation comes out as the donor's class or
the recipient's.

  * Continuation takes the DONOR's class -> a real, usable inference-time lever, and the first one
    this project has found. Class conditioning by transplant rather than by label or by direction.
  * Continuation stays the RECIPIENT's  -> the patch moves local sequence statistics but not class.
    Combined with Phase A that is a precise and final statement: the model reads these activations,
    but what it reads from them is not the class.

THE PATCH APPLIES TO THE PREFILL ONLY. Generated positions have no donor counterpart, so the hook
fires only on the pass whose length matches the donor's and passes everything else through
untouched. That is the intended semantics -- "replace the model's internal picture of the context,
then let it continue" -- and it is also why the hook must not raise on a length mismatch here, in
contrast to Phase A where a mismatch means genuinely misaligned prompts.

CONTROLS: an unpatched arm (the recipient's own floor) and a SAME-CLASS donor arm (a different real
core of the recipient's class), which absorbs "any real transplant perturbs generation".
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
sys.path.insert(0, str(REPO / "src"))

from activation_patching import _block, _capture_hook  # noqa: E402
from evo2_inference import load_evo2_wrapper_for_inference  # noqa: E402
from generate_bgc import assemble_record, extract_sequence  # noqa: E402


def _gen_sequences(out):
    seqs = getattr(out, "sequences", out)
    return list(seqs) if isinstance(seqs, (list, tuple)) else [seqs]


def _prefill_patch_hook(model, layer: int, donor: torch.Tensor, k: int, stats: dict):
    """Substitute the donor over the last k CONTEXT positions, on the prefill pass only."""
    n_donor = donor.shape[1]

    def _apply(h):
        if h.shape[1] != n_donor:      # a generation step, not the prefill: leave untouched
            return h
        kk = max(1, min(k, h.shape[1]))
        out = h.clone()
        out[:, -kk:, :] = donor[:, -kk:, :].to(h.dtype)
        stats["applied"] = stats.get("applied", 0) + 1
        stats["k"] = kk
        return out

    def hook(_m, _in, out):
        if isinstance(out, (tuple, list)):
            return (_apply(out[0]),) + tuple(out[1:])
        return _apply(out)

    return _block(model, layer).register_forward_hook(hook)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--classes", nargs="+", default=["NRPS", "PKS", "TERPENE", "RIPP"])
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 22])
    ap.add_argument("--k", type=int, nargs="+", default=[200])
    ap.add_argument("--ctx-nt", type=int, default=1000)
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/data2/ds85/bgcmodel_runs/patch_generate"))
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

    pairs = []
    for i in range(args.n_pairs):
        ca = args.classes[i % len(args.classes)]
        cb = args.classes[(i + 1 + (i // len(args.classes))) % len(args.classes)]
        if ca == cb:
            cb = args.classes[(i + 2) % len(args.classes)]
        pairs.append((ca, cb, byc[ca][i % len(byc[ca])], byc[cb][i % len(byc[cb])],
                      byc[cb][(i + 1) % len(byc[cb])]))
    print(f"[pg] {len(pairs)} pairs, layers={args.layers}, k={args.k}", flush=True)

    wrapper = load_evo2_wrapper_for_inference(Path(args.adapter) if args.adapter else None,
                                              device=args.device)
    model, tok = wrapper.model, wrapper.tokenizer

    class _A:   # decoding-metadata shim for assemble_record (matches seed_generate.py)
        temperature, top_k, top_p = args.temperature, args.top_k, args.top_p
        max_new_tokens = args.max_new_tokens
        max_n_frac = 0.10

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_recs: dict[str, list] = {}

    for pi, (ca, cb, donor, recip, same) in enumerate(pairs):
        tax = recip.get("taxonomic_tag", "") or ""
        p_recip = tax + recip["sequence"][: args.ctx_nt]
        donor_prompts = {"cross_class": tax + donor["sequence"][: args.ctx_nt],
                         "same_class": tax + same["sequence"][: args.ctx_nt]}

        def _gen(handle_factory, arm, layer, k):
            torch.manual_seed(args.seed * 977 + pi)
            stats: dict = {}
            h = handle_factory(stats) if handle_factory else None
            try:
                out = wrapper.generate(prompt_seqs=[p_recip], n_tokens=args.max_new_tokens,
                                       temperature=args.temperature, top_k=args.top_k,
                                       top_p=args.top_p, cached_generation=True, verbose=0)
            finally:
                if h is not None:
                    h.remove()
            ex = extract_sequence(_gen_sequences(out)[0])
            # compound_class is what we are TESTING FOR: the donor's class in the patched arms.
            # The recipient's class is recorded alongside so both directions can be scored.
            rec = assemble_record(ca if arm != "unpatched" else cb, tax,
                                  ex["sequence"], ex["hit_eos"], 1, _A)
            rec.update({"arm": arm, "layer": layer, "k": k, "pair": pi,
                        "donor_class": ca, "recip_class": cb,
                        "donor_accession": donor.get("accession") or donor.get("id"),
                        "recip_accession": recip.get("accession") or recip.get("id"),
                        "scored_span": "continuation_only", "seeded": True,
                        "seed_nt": args.ctx_nt, "seed_class": cb, "tax_idx": pi,
                        "patch_applied": stats.get("applied", 0), "patch_k": stats.get("k"),
                        "prompt": "recipient_context+activation_patch"})
            return rec

        out_recs.setdefault("unpatched", []).append(_gen(None, "unpatched", None, None))
        for L in args.layers:
            caps = {}
            for arm, dp in donor_prompts.items():
                store: dict = {}
                hc = _capture_hook(model, L, store)
                try:
                    ids = [int(x) for x in list(tok.tokenize(dp))[:args.max_seq_len]]
                    with torch.no_grad():
                        model(torch.tensor([ids], dtype=torch.long, device=args.device))
                finally:
                    hc.remove()
                caps[arm] = store["h"]
            for k in args.k:
                for arm, dn in caps.items():
                    key = f"{arm}_L{L}_k{k}"
                    rec = _gen(lambda s, _d=dn, _L=L, _k=k:
                               _prefill_patch_hook(model, _L, _d, _k, s), arm, L, k)
                    if rec["patch_applied"] == 0:
                        raise SystemExit(
                            f"[pg] ABORT: the patch hook never fired for {key}. The prefill length "
                            f"did not match the donor ({dn.shape[1]} positions), so this arm is "
                            f"silently IDENTICAL to unpatched and would read as 'patching does "
                            f"nothing'.")
                    out_recs.setdefault(key, []).append(rec)
        print(f"[pg] pair {pi + 1}/{len(pairs)} ({ca} -> {cb}) done", flush=True)

    for key, recs in out_recs.items():
        p = args.out_dir / f"pg_{key}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(f"[pg] wrote {p.name}: {len(recs)} records")
    print(f"\n[pg] score with:\n"
          f"  python evo2/scripts/score_generations_antismash.py {args.out_dir}/pg_*.jsonl "
          f"--expected compound_class --workers 10 --allow-legacy "
          f"--out-tsv {args.out_dir}/antismash.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
