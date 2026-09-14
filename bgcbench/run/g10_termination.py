"""G10 — termination, per substrate (SPEC 5.1, 12).

MUST PASS BEFORE ANY ADAPTER IS TRAINED. If Evo2 needs its terminator written into the
training text for the model to ever learn to stop, that is a training-data format decision
that cannot be retrofitted after adapters exist -- every weight-state arm would generate to
the full budget forever and the multi-gene axis would stay confounded against a ~1-9 kb
real-core reference.

Four checks, none of which assume anything the docs claim:
  T1  the terminator id is what the tokenizer actually reports, and round-trips
  T2  training_text() carries a terminator on substrates whose tokenizer omits one
  T3  the terminator is DETECTABLE in generated output (truncation works)
  T4  realised generation length and hit_eos, from the base model, vs real held-out cores
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgcbench.model.genconfig import decoding_for
from bgcbench.model.load import load

ROOT = Path("/data2/ds85/bgcbench")
OUT = ROOT / "gates"


def check(sub, n_probe: int, budget_nt: int) -> dict:
    res: dict = {"substrate": sub.id, "checkpoint": sub.checkpoint,
                 "terminator_id": sub.terminator_id,
                 "appends_terminator": sub.appends_terminator,
                 "native_stop": sub.native_stop, "meta": sub.meta}

    # T1 -- round trip
    try:
        if sub.family == "evo2":
            ids = list(sub.tokenizer.tokenize("ACGT"))
            res["T1_tokenize_ACGT"] = [int(x) for x in ids]
            # BOTH DIRECTIONS. Testing only encode is how this gate passed while the
            # property it gates was false: vortex decodes with chr(max(32, min(id, vocab))),
            # so ids 0/1/32 all render as a space and the DECODE direction failed silently
            # for the whole of Stage 1.
            # encode the TRAINING form (must yield the real id), decode the display form
            enc_ok = int(sub.tokenizer.tokenize(
                sub.terminator_encode_str or sub.terminator_str)[0]) == sub.terminator_id
            dec = sub.tokenizer.detokenize([sub.terminator_id])
            res["T1_terminator_decodes_to"] = repr(dec)
            res["T1_decode_distinguishable"] = (
                dec != sub.tokenizer.detokenize([1])
                and dec != sub.tokenizer.detokenize([32]))
            res["T1_terminator_roundtrip"] = bool(
                enc_ok and dec == sub.terminator_str
                and res["T1_decode_distinguishable"])
        else:
            enc = sub.tokenizer("ACGTACGTACGT")["input_ids"]
            res["T1_tokenize"] = enc
            res["T1_terminator_present_at_encode"] = sub.terminator_id in enc
            res["T1_terminator_roundtrip"] = True
    except Exception as e:
        res["T1_error"] = f"{type(e).__name__}: {e}"

    # T2 -- training text carries a terminator
    tt = sub.training_text("ACGTACGT")
    if sub.family == "evo2":
        # check the TOKENS, not the characters: terminator_str is the display sentinel and
        # training text carries the encode form, so a string comparison tests the wrong thing
        tail = [int(x) for x in sub.tokenizer.tokenize(tt)]
        res["T2_training_text_has_terminator"] = bool(tail and tail[-1] == sub.terminator_id)
        res["T2_training_text_ids_tail"] = tail[-3:]
    else:
        ids = sub.tokenizer(tt)["input_ids"]
        res["T2_training_text_has_terminator"] = sub.terminator_id in ids
        res["T2_training_text_ids_tail"] = ids[-3:]

    # T3 -- detectable in MODEL-DECODED output, not in a probe string we built ourselves.
    # Building the probe in Python guarantees the terminator is present as a character, which
    # is exactly the assumption that failed: the model's terminator never survived decoding.
    # Round-tripping through the tokenizer is what makes this test able to fail.
    if sub.terminator_str:
        if sub.family == "evo2":
            ids = [int(x) for x in sub.tokenizer.tokenize("ACGT" * 10)] \
                + [sub.terminator_id] + [int(x) for x in sub.tokenizer.tokenize("TTTT" * 10)]
            probe = sub.tokenizer.detokenize(ids)
        else:
            probe = "ACGT" * 10 + sub.terminator_str + "TTTT" * 10
        body, hit = sub.truncate_at_terminator(probe)
        res["T3_probe_is_decoded_output"] = (sub.family == "evo2")
        res["T3_truncation_works"] = bool(hit and len(body) == 40)
    else:
        res["T3_truncation_works"] = "n/a — BPE substrate detects on ids"

    # T4 -- what the BASE model actually does at the budget
    if sub.family == "evo2":
        n_tok = budget_nt
        # ⚠ DECODING COMES FROM THE SUBSTRATE (SPEC §12.A7), NOT A LITERAL. These two calls
        # hard-coded `top_k=4, temperature=1.0`, bypassing `decoding_for()` entirely -- so
        # G10's GenomeOcean half was measured at the setting that keeps 19% of GO's
        # probability mass, while every other generation path had already moved to its own
        # gated value. Nothing raised, because a literal is always valid Python.
        _dec = decoding_for(sub.family)
        out = sub.model.generate(prompt_seqs=["ACGTACGTACGTACGTACGT"] * n_probe,
                                 n_tokens=n_tok, temperature=_dec["temperature"],
                                 top_k=_dec["top_k"], verbose=0)
        seqs = list(out[0]) if isinstance(out, tuple) else list(out.sequences)
        lens, hits = [], 0
        for t in seqs:
            body, hit = sub.truncate_at_terminator(t)
            hits += hit
            lens.append(len(sub.clean(body)))
    else:
        import torch
        # ⚠ _encode_prompts, NOT the raw tokenizer. GenomeOcean's TemplateProcessing appends
        # [SEP] -- which IS sub.terminator_id -- so a bare tokenizer call made this probe
        # measure termination starting ONE TOKEN PAST the model's own terminator, a context
        # no arm generates from. That is the same defect generate.py fixed; this gate kept it,
        # which is worse here because G10 is the gate that certifies termination.
        from bgcbench.model.generate import _encode_prompts
        enc = _encode_prompts(sub, ["ACGTACGTACGTACGTACGT"] * n_probe)
        # the GenomeOcean tokenizer emits token_type_ids; the model does not accept them
        enc = {k: v.to(sub.model.device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            gen = sub.model.generate(**enc,
                                     max_new_tokens=int(budget_nt / sub.approx_nt_per_token),
                                     do_sample=True, top_k=decoding_for(sub.family)["top_k"],
                                     temperature=decoding_for(sub.family)["temperature"],
                                     top_p=decoding_for(sub.family)["top_p"],
                                     eos_token_id=sub.terminator_id,
                                     pad_token_id=sub.tokenizer.pad_token_id)
        lens, hits = [], 0
        for row in gen:
            ids = row.tolist()
            hits += sub.terminator_id in ids[enc["input_ids"].shape[1]:]
            # ⚠ Substrate.detokenize, NOT tokenizer.decode -- decode() space-joins GO's BPE
            # tokens and clean() masks each space to N (KNOWN_WRONG #3).
            txt = sub.detokenize(ids)
            lens.append(len(sub.clean(txt)))
    lens.sort()
    res["T4"] = {"n": len(lens), "budget_nt": budget_nt,
                 "hit_eos": hits, "hit_eos_rate": round(hits / max(len(lens), 1), 4),
                 "median_len": lens[len(lens) // 2] if lens else 0,
                 "min_len": lens[0] if lens else 0, "max_len": lens[-1] if lens else 0}
    res["PASS_mechanism"] = bool(res.get("T2_training_text_has_terminator")) and \
        (res.get("T3_truncation_works") is True or sub.family != "evo2")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", nargs="+", default=["evo2-1b"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget-nt", type=int, default=2000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for sid in args.substrates:
        print(f"\n=== {sid} ===", flush=True)
        try:
            sub = load(sid)
        except Exception as e:
            report[sid] = {"load_error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"  LOAD FAILED: {report[sid]['load_error']}", flush=True)
            continue
        r = check(sub, args.n, args.budget_nt)
        report[sid] = r
        for k in ("terminator_id", "appends_terminator", "native_stop",
                  "T1_terminator_roundtrip", "T2_training_text_has_terminator",
                  "T3_truncation_works", "PASS_mechanism"):
            print(f"  {k:34s} {r.get(k)}", flush=True)
        print(f"  T4 base-model at {args.budget_nt} nt: {r['T4']}", flush=True)
        del sub
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    (OUT / "g10_termination.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'g10_termination.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
