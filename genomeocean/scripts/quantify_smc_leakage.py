#!/usr/bin/env python
"""QUANTIFY the unmeasured leakage between `bgcFM`'s training corpus and our `splits_core`.

WHY THIS GATES EVERYTHING ELSE ABOUT GENOMEOCEAN. `bgcFM` is GenomeOcean-4B further trained on
**1.72M deduplicated SMC BGCs (43.5 Gbp)**, and SMC is antiSMASH-derived — as is `splits_core`. The
comparison doc has carried "leakage against our splits_core is UNQUANTIFIED" as an open risk since
2026-07-27. Novelty is a HARD CONSTRAINT on every rung of the ladder, so an unmeasured overlap would
invalidate any capability claim made on this substrate before it was made.

WHY NOT JUST INTERSECT THE CORPORA. SMC is not on this machine and is 43.5 Gbp; downloading it to
answer a gating question is the expensive path. A model that has memorised a sequence will
RECONSTRUCT it, so memorisation can be measured from the model alone.

THE TEST. Prompt with the first `--prompt-nt` bases of a real held-out core, generate a
continuation, and measure k=21 containment of that continuation against the TRUE continuation. Two
controls, because the raw number is meaningless alone:

  MISMATCHED  the same generation scored against a DIFFERENT core's continuation.
              ⚠️ THIS FLOOR IS EXPECTED TO BE EXACTLY ZERO, and an earlier version of this file
              asserted the opposite ("BGC DNA shares motifs, so containment is not zero by
              construction"). That was WRONG and it caused a correct result to be thrown out as a
              broken instrument. Measured: two REAL unrelated BGC cores score 0.000000, and the
              chance rate at k=21 is 4.6e-10 (the space is 4.4e12 21-mers). Shared PROTEIN motifs
              do not survive as exact 21-NUCLEOTIDE matches. A zero floor is the correct floor.
  BASE MODEL  plain GenomeOcean-4B, which never saw the SMC BGC fine-tune. If base reconstructs our
              cores as well as bgcFM does, any signal is generic BGC-likeness, not SMC leakage.
  POSITIVE    the containment function applied to a sequence against ITSELF (must be 1.0) and
              against a 5%-mutated copy (must land intermediate). Since the floor is legitimately
              zero, the ONLY way to distinguish "no memorisation" from "dead measurement" is to
              demonstrate the instrument has dynamic range. That is what this checks.

GREEDY, NOT SAMPLED. The first run generated at temperature 1.0 / top_k 50. A model can have
memorised a sequence and still not reproduce it verbatim while sampling, so that run had weak
sensitivity for a reason unrelated to the floor. Memorisation probes decode greedily.

READING IT, fixed before running:
  bgcFM_true >> bgcFM_mismatched  AND  bgcFM_true >> base_true   -> real memorisation. Quantify it
      and treat every novelty claim on this substrate as suspect until the overlap is excluded.
  bgcFM_true ~= its controls                                     -> no detectable leakage at this
      sensitivity. Record the sensitivity; this bounds the risk, it does not prove zero overlap.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

MODELS = {
    "bgcFM": "pGenomeOcean/GenomeOcean-4B-bgcFM",
    "base": "pGenomeOcean/GenomeOcean-4B",
}


def containment(query: str, ref: str, k: int = 21) -> float:
    """Fraction of the QUERY's k-mers that also occur in REF — the same asymmetric measure the
    project's novelty guard uses, so the numbers are comparable to `PASS_novel` thresholds."""
    if len(query) < k or len(ref) < k:
        return 0.0
    q = {query[i:i + k] for i in range(len(query) - k + 1)}
    r = {ref[i:i + k] for i in range(len(ref) - k + 1)}
    return len(q & r) / len(q) if q else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--prompt-nt", type=int, default=500)
    ap.add_argument("--gen-nt", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=REPO / "genomeocean" / "experiments" / "smc_leakage.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    need = args.prompt_nt + args.gen_nt
    cores = []
    with open("/data2/ds85/bgcmodel_data/splits_core/test.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if len(r.get("sequence", "")) >= need:
                cores.append(r["sequence"][:need])
                if len(cores) >= args.n:
                    break
    if len(cores) < 4:
        raise SystemExit("[leak] ABORT: too few long test cores")
    prompts = [c[:args.prompt_nt] for c in cores]
    truths = [c[args.prompt_nt:] for c in cores]
    print(f"[leak] {len(cores)} held-out cores; prompt {args.prompt_nt} nt, generate {args.gen_nt} nt")

    results = {}
    for label, mid in MODELS.items():
        print(f"[leak] loading {label} ({mid}) …", flush=True)
        tok = PreTrainedTokenizerFast.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        gens = []
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.gen_nt // 4 + 32,
                                     do_sample=False,          # GREEDY: see module docstring
                                     pad_token_id=tok.pad_token_id)
            for j, o in enumerate(out):
                txt = tok.decode(o, skip_special_tokens=True).replace(" ", "").upper()
                gens.append(txt[len(chunk[j]):][:args.gen_nt])
            print(f"[leak]   {min(i+args.batch, len(prompts))}/{len(prompts)}", flush=True)
        true_c = [containment(g, t) for g, t in zip(gens, truths)]
        # MISMATCHED: rotate by one so every generation is scored against another core's truth.
        mism_c = [containment(g, truths[(i + 1) % len(truths)]) for i, g in enumerate(gens)]
        results[label] = {"true": true_c, "mismatched": mism_c}
        print(f"[leak]   {label}: true {st.mean(true_c):.4f}  mismatched {st.mean(mism_c):.4f}")
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 76)
    print("SMC LEAKAGE — does bgcFM RECONSTRUCT our held-out cores? (k=21 containment)")
    print("=" * 76)
    print(f"{'model':<8} {'vs TRUE continuation':>22} {'vs MISMATCHED (floor)':>23} {'excess':>10}")
    for label, r in results.items():
        t, m = st.mean(r["true"]), st.mean(r["mismatched"])
        print(f"{label:<8} {t:>22.4f} {m:>23.4f} {t - m:>+10.4f}")

    # ── POSITIVE CONTROL, BEFORE ANY VERDICT ────────────────────────────────────────────────
    # The floor here is LEGITIMATELY ZERO (see docstring), so "everything reads 0" cannot be
    # distinguished from "the measurement is dead" by looking at the floor — which is exactly the
    # mistake made on the first run, in both directions: the zero was first reported as a null, then
    # rejected as a broken instrument. Neither was justified without showing the function has
    # dynamic range. So demonstrate it, every run, on the real data being used.
    import random as _rnd
    _r = _rnd.Random(0)
    _ref = truths[0]
    _mut = list(_ref)
    for _i in range(len(_mut)):
        if _r.random() < 0.05:
            _mut[_i] = _r.choice("ACGT")
    self_c = containment(_ref, _ref)
    mut_c = containment("".join(_mut), _ref)
    print(f"\n  POSITIVE CONTROL  identical {self_c:.4f} (must be 1.0)   "
          f"5%-mutated {mut_c:.4f} (must be intermediate)")
    if self_c < 0.999 or not (0.2 < mut_c < 0.95):
        print("  !! INSTRUMENT DEAD — NO VERDICT. containment() has no dynamic range on this data.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"status": "INSTRUMENT_DEAD",
                                        "self": self_c, "mutated": mut_c}, indent=1))
        return 2
    print("  ⇒ instrument has dynamic range; a zero reading below is a real negative, and would")
    print("    still have caught a 5%-diverged memorised copy.")

    from scipy.stats import wilcoxon
    print()
    for label, r in results.items():
        try:
            _, p = wilcoxon([a - b for a, b in zip(r["true"], r["mismatched"])])
        except ValueError:
            p = float("nan")
        print(f"  {label}: true vs mismatched paired p = {p:.4f}")
    if "bgcFM" in results and "base" in results:
        ex_f = st.mean(results["bgcFM"]["true"]) - st.mean(results["bgcFM"]["mismatched"])
        ex_b = st.mean(results["base"]["true"]) - st.mean(results["base"]["mismatched"])
        print(f"\n  bgcFM excess {ex_f:+.4f} vs base excess {ex_b:+.4f}")
        if ex_f > 0.05 and ex_f > 2 * max(ex_b, 1e-9):
            print("  ⇒ MEMORISATION DETECTED. bgcFM reconstructs our held-out cores beyond both the")
            print("    chance floor and what the un-fine-tuned base does. Treat every novelty claim")
            print("    on this substrate as suspect until the SMC overlap is excluded directly.")
        else:
            print("  ⇒ NO memorisation detected AT THIS SENSITIVITY. That bounds the risk; it does")
            print("    NOT prove zero overlap. A model can be trained on a sequence without")
            print("    reconstructing it from a 500 nt prompt, so this is a floor on the evidence,")
            print("    not a clearance.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"\n[leak] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
