#!/usr/bin/env python3
"""C3 — generate conditioned BGC sequences from the fine-tuned Evo2 model.

Loads base Evo2 + the trained LoRA adapter (merged), builds the Phase-1
conditioning prefix |COMPOUND_CLASS:{cls}|{tax}, samples a sequence with Evo2's
efficient cached generation, stops at the trained |END| marker, and writes a
FASTA (for the eval suite) + a JSONL of metadata.

Generation runs the full --max-new-tokens (vortex hardcodes stop_at_eos=False),
so we trim at |END| ourselves. For BGCs longer than one window, --max-windows>1
enables chained generation via the |CONTINUATION:{cls}|{tax} prefix + carried
overlap context (mirrors training; audit M11).

Prompts come from --class/--taxon (one explicit prompt) or --from-jsonl (sample
class+taxon prompts from a held-out split, e.g. for evaluation). With --adapter
omitted, the untouched base model is used — the M5 generation baseline.

NOTE: generation requires a GPU + evo2 weights + a trained checkpoint. The
pure post-processing (prefix building, EOS trimming, nucleotide sanitation,
FASTA) is unit-tested in tests/test_generation.py.

Examples:
  # generate from held-out val prompts (eval), 2 per class, with the best adapter
  python scripts/generate_bgc.py \
      --adapter /data2/ds85/bgcmodel_runs/<run>/checkpoints/best \
      --from-jsonl /data2/ds85/bgcmodel_data/splits_curated/val.jsonl \
      --per-class 2 --max-new-tokens 16384 \
      --out-fasta gen.fasta --out-jsonl gen.jsonl

  # one explicit prompt
  python scripts/generate_bgc.py --adapter <ckpt> \
      --class NRPS --taxon "|D__BACTERIA;P__ACTINOMYCETOTA;..." --n 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# Must match scripts/finetune_evo2_lora.py (asserted in tests/test_generation.py).
EOS_MARKER = "|END|"
CLASS_PREFIX_FMT = "|COMPOUND_CLASS:{cls}|{tax}"
CONT_PREFIX_FMT = "|CONTINUATION:{cls}|{tax}"
_NUC_RE = re.compile(r"[ACGTN]*")
# Stray non-nucleotide characters: mask to N (frame-preserving) rather than truncate.
# See extract_sequence for the measurement that forced this. bugs.md 2026-08-20.
JUNK_POLICY_DEFAULT = "mask"

# Left-pad byte for BATCHED generation. vortex only batches prompts of EQUAL
# length — otherwise it SILENTLY de-batches and loops one-at-a-time
# (vortex/model/generation.py:312) — and when it does batch it RIGHT-pads with
# no attention mask, which is wrong for causal generation. We therefore LEFT-pad
# the prompts to a uniform length ourselves so (a) vortex actually batches and
# (b) each real prompt stays adjacent to its own generation (vortex's internal
# right-pad then becomes a no-op). The left pad uses the tokenizer's pad byte
# (space ≈ pad_id 0). Whether the leading pad perturbs outputs vs. unpadded
# single-prompt generation is NOT assumed safe — it is verified on-GPU by
# scripts/validate_batched_generation.py before the batched path is trusted.
LEFT_PAD_CHAR = " "


# ── Pure-logic helpers (no torch; unit-tested) ──────────────────────────────

def build_prefix(cls: str, tax: str) -> str:
    return CLASS_PREFIX_FMT.format(cls=cls, tax=tax)


def build_continuation_prefix(cls: str, tax: str) -> str:
    return CONT_PREFIX_FMT.format(cls=cls, tax=tax)


def extract_sequence(generated: str, junk_policy: str = JUNK_POLICY_DEFAULT) -> dict:
    """Turn a raw generated string into a clean nucleotide sequence.

    Always trims at the first ``|END|`` (the EOS the model was trained to emit). What happens to a
    stray NON-nucleotide character elsewhere is the `junk_policy`:

    ``"mask"`` (default since 2026-08-20)
        Replace each non-ACGTN character with ``N`` and KEEP GOING. Frame is preserved, and the
        existing ``--max-n-frac`` filter still rejects genuinely degenerate output.

    ``"truncate"`` (the pre-2026-08-20 behaviour, kept for reproducing older runs)
        Keep only the LEADING run of ACGTN and discard everything after the first stray character.

    ⚠️ WHY THE DEFAULT CHANGED. Measured on 32 raw PKS-adapter generations: **23/32 contained a
    stray character**, almost always a single SPACE, and it landed at a codon boundary (``GGCTGA ``,
    ``GATTAA `` — TGA/TAA are stop codons). Under ``truncate`` we kept a median of 1,817 nt and threw
    away a median of **6,183 nt that was 99.9% ACGT — real sequence**. The effect is far stronger for
    fine-tuned adapters than for the base model (PKS: 45.5% of adapter records fell below the scoring
    window vs 2.0% of base-model records), so it biased precisely the treatment-vs-control comparison
    the phase exists to make.

    ⚠️ **THE CAUSE IS NOT IDENTIFIED.** What is established: fine-tuned adapters do it ~35x more than
    the base model (PKS 69.5% of records vs 2.0%), and the character lands at a codon boundary far
    more often than chance would suggest — ``GGCTGA ``, ``GATTAA ``, ``CGATGA `` (TGA/TAA are stop
    codons). **Ruled out:** the batched left-pad, even though ``LEFT_PAD_CHAR`` is itself a space —
    pad length does not predict truncation (Pearson r = +0.020 over 200 records, and no dose-response
    across pad buckets). **Still open:** whether this is a partially-learned terminator (each training
    record ends, so the adapter may have learned "stop here" and reached for the nearest
    boundary-like byte it ever saw) or simply a low-probability byte surfacing under top-k sampling.
    Against the "learned stop" reading: the model does not actually stop — the text after the
    character is 99.9% valid DNA and continues for a median of 6.2 kb.

    ⚠️ **DO NOT "STRIP" THE CHARACTER BY DELETING IT.** Deletion shifts the reading frame by one base
    and destroys every downstream ORF, and ORFs are what the entire scoring stack is built on.
    Masking to ``N`` keeps every base at its true coordinate.
    """
    if junk_policy not in ("mask", "truncate"):
        raise ValueError(f"junk_policy must be 'mask' or 'truncate', got {junk_policy!r}")
    hit_eos = EOS_MARKER in generated
    body = generated.split(EOS_MARKER, 1)[0] if hit_eos else generated
    up = body.upper()
    lead = _NUC_RE.match(up).group(0)
    n_junk = sum(1 for c in up if c not in "ACGTN")
    clean = lead if junk_policy == "truncate" else "".join(
        c if c in "ACGTN" else "N" for c in up)
    return {
        "sequence": clean,
        "hit_eos": hit_eos,
        "len": len(clean),
        "n_count": clean.count("N"),
        "junk_policy": junk_policy,
        "n_junk_chars": n_junk,
        "leading_run_len": len(lead),
        "discarded_by_truncate": len(body) - len(lead),
        "trailing_junk_trimmed": junk_policy == "truncate" and len(lead) < len(body),
    }


def n_fraction(seq: str) -> float:
    return (seq.upper().count("N") / len(seq)) if seq else 0.0


def to_fasta_record(seq_id: str, seq: str, **meta) -> str:
    tags = " ".join(f"{k}={v}" for k, v in meta.items())
    header = f">{seq_id} {tags}".rstrip()
    lines = [seq[i:i + 80] for i in range(0, len(seq), 80)] or [""]
    return header + "\n" + "\n".join(lines) + "\n"


def sample_prompts(records: list[dict], per_class: int, rng: random.Random) -> list[dict]:
    """Sample (compound_class, taxonomic_tag) prompts, up to per_class per class."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r.get("compound_class", "UNKNOWN")].append(r)
    prompts = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        rng.shuffle(pool)
        for r in pool[:per_class] if per_class > 0 else pool:
            prompts.append({"compound_class": cls,
                            "taxonomic_tag": r.get("taxonomic_tag", "")})
    return prompts


def left_pad_to_uniform(prefixes: list[str], pad: str = LEFT_PAD_CHAR) -> list[str]:
    """Left-pad every prefix with ``pad`` so all share the longest length.

    Required for vortex batched generation: it only batches prompts of EQUAL
    length (else silently de-batches). Left-padding (rather than vortex's
    internal right-pad) keeps each real prompt's final token adjacent to where
    generation begins, which is the causally-correct placement. Each returned
    string ENDS with its original prefix (so the conditioning is preserved); a
    single prefix (or already-uniform prefixes) is returned unchanged.
    """
    if not prefixes:
        return []
    width = max(len(p) for p in prefixes)
    return [pad * (width - len(p)) + p for p in prefixes]


def assemble_record(cls: str, tax: str, sequence: str, hit_eos: bool,
                    windows: int, args: Any) -> dict:
    """Build the per-record output dict. SINGLE source of truth shared by the
    sequential (``generate_one``) and batched (``generate_batch``) paths so both
    emit a byte-identical schema downstream (eval_suite_driver / quick_eval /
    FASTA all key off these fields)."""
    nfrac = n_fraction(sequence)
    return {
        "compound_class": cls,
        "taxonomic_tag": tax,
        "sequence": sequence,
        "length": len(sequence),
        "hit_eos": hit_eos,
        "windows": windows,
        "n_count": sequence.upper().count("N"),
        "n_fraction": round(nfrac, 5),
        "n_pass": nfrac <= args.max_n_frac,
        "decoding": {"temperature": args.temperature, "top_k": args.top_k,
                     "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
        "junk_policy": getattr(args, "junk_policy", JUNK_POLICY_DEFAULT),
        "base_model": getattr(args, "base_model", None),
    }


def should_batch(args: Any) -> bool:
    """Batched generation applies only for single-window generation. Chained
    long-seq generation (``--max-windows>1``) carries a per-sequence overlap
    seed, so it cannot share one batched call; fall back to sequential there."""
    if args.batch_size == 1:
        return False
    if args.max_windows > 1:
        print("WARNING: --batch-size>1 is not supported with --max-windows>1; "
              "using sequential generation.", file=sys.stderr)
        return False
    return True


# ── Generation (GPU; lazy heavy imports) ────────────────────────────────────

def _gen_sequences(out: Any) -> list[str]:
    seqs = getattr(out, "sequences", None)
    if seqs is None and isinstance(out, (tuple, list)):
        seqs = out[0]
    return list(seqs)


def generate_one(wrapper: Any, cls: str, tax: str, args) -> dict:
    """Generate one conditioned BGC (single window, optionally chained)."""
    prefix = build_prefix(cls, tax)
    out = wrapper.generate(
        prompt_seqs=[prefix], n_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        cached_generation=True, verbose=0,
    )
    info = extract_sequence(_gen_sequences(out)[0], args.junk_policy)
    full = info["sequence"]
    hit_eos = info["hit_eos"]
    windows = 1

    # Chained long-sequence generation: continue from the carried overlap until
    # the model emits EOS or we hit the window cap (audit M11 / long-seq).
    while (not hit_eos) and windows < args.max_windows and len(full) >= args.chunk_overlap:
        seed = full[-args.chunk_overlap:]
        out = wrapper.generate(
            prompt_seqs=[build_continuation_prefix(cls, tax) + seed],
            n_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            cached_generation=True, verbose=0,
        )
        cont = extract_sequence(_gen_sequences(out)[0], args.junk_policy)
        if cont["len"] == 0:
            break
        full += cont["sequence"]
        hit_eos = cont["hit_eos"]
        windows += 1

    return assemble_record(cls, tax, full, hit_eos, windows, args)


def generate_batch(wrapper: Any, prompts: list[dict], args) -> list[dict]:
    """Generate one conditioned BGC for EACH prompt in a single batched call.

    Single-window only (``should_batch`` guards chaining). All prefixes are
    left-padded to a uniform length so vortex batches them; the returned strings
    are generation-only (vortex strips the prompt), so each is fed straight to
    ``extract_sequence``. Output order matches the input ``prompts`` order, and
    records are byte-identical in schema to ``generate_one``.
    """
    prefixes = [build_prefix(p["compound_class"], p["taxonomic_tag"]) for p in prompts]
    padded = left_pad_to_uniform(prefixes)
    out = wrapper.generate(
        prompt_seqs=padded, n_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        batched=True, cached_generation=True, verbose=0,
    )
    gens = _gen_sequences(out)
    if len(gens) != len(prompts):
        raise RuntimeError(
            f"batched generate returned {len(gens)} sequences for {len(prompts)} "
            "prompts — batch alignment broken; refusing to mislabel records.")
    records = []
    for p, g in zip(prompts, gens):
        info = extract_sequence(g, args.junk_policy)
        records.append(assemble_record(
            p["compound_class"], p["taxonomic_tag"],
            info["sequence"], info["hit_eos"], 1, args))
    return records


def require_explicit_substrate(base_model: str | None) -> str:
    """Fail loudly when the substrate is not stated. [P3-B7]

    This script used to fall through to the 7B whenever `EVO2_BASE_MODEL` was unset, so a bare
    invocation silently generated from the wrong model and only failed if the adapter happened to
    be shape-incompatible. That cost 150 discarded control generations on 2026-08-17 (`bugs.md`).
    The substrate is not a default — it is part of the result, and an unstated one is a bug.
    """
    resolved = base_model or os.environ.get("EVO2_BASE_MODEL")
    if not resolved:
        raise SystemExit(
            "[generate_bgc] FATAL: no substrate. Set EVO2_BASE_MODEL (e.g. evo2_1b_base) or pass "
            "--base-model. This script previously defaulted to the 7B and silently produced "
            "generations from the wrong model -- see bugs.md [P3-B7]. Refusing to guess.")
    return resolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA checkpoint dir (with adapter/) or run dir (uses best/). "
                         "Omit to generate from the BASE model (M5 baseline).")
    # prompt source
    ap.add_argument("--from-jsonl", type=Path, default=None,
                    help="Sample class+taxon prompts from this split (e.g. val.jsonl).")
    ap.add_argument("--per-class", type=int, default=2,
                    help="Prompts sampled per class from --from-jsonl (0 = all).")
    ap.add_argument("--class", dest="cls", default=None, help="Explicit compound class.")
    ap.add_argument("--taxon", default=None, help="Explicit taxonomic_tag for --class.")
    ap.add_argument("--n", type=int, default=1, help="Samples per prompt.")
    # decoding
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16384,
                    help="Nucleotides generated per window (vortex runs the full count).")
    ap.add_argument("--max-windows", type=int, default=1,
                    help=">1 enables chained generation for BGCs longer than one window.")
    ap.add_argument("--chunk-overlap", type=int, default=2048,
                    help="Overlap (nt) carried as context into each chained window.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Sequences generated per batched call. 1 (default) = "
                         "sequential, one prompt at a time (unchanged behavior). "
                         ">1 left-pads prompts to uniform length and generates "
                         "that many at once; 0 = all in one batch. Single-window "
                         "only (ignored with --max-windows>1). Verify equivalence "
                         "with scripts/validate_batched_generation.py first.")
    ap.add_argument("--max-n-frac", type=float, default=0.01,
                    help="Max fraction of N for a sequence to be flagged n_pass=true.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--junk-policy", choices=("mask", "truncate"),
                    default=JUNK_POLICY_DEFAULT,
                    help="Stray non-ACGTN character handling. 'mask' (default) replaces it with N "
                         "and continues, preserving frame; 'truncate' is the pre-2026-08-20 "
                         "behaviour that discarded everything after it (and with it a median of "
                         "6.2 kb of real sequence). Part of the scoring config -- record it.")
    ap.add_argument("--base-model", default=None,
                    help="Evo2 substrate, e.g. evo2_1b_base. REQUIRED unless EVO2_BASE_MODEL is "
                         "set in the environment -- this script does not guess (bugs.md P3-B7).")
    ap.add_argument("--out-fasta", type=Path, default=Path("generated_bgcs.fasta"))
    ap.add_argument("--out-jsonl", type=Path, default=Path("generated_bgcs.jsonl"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the prompt/decoding plan without loading the model.")
    args = ap.parse_args()
    # Substrate must be explicit BEFORE any model load, and is stamped into every record.
    args.base_model = require_explicit_substrate(args.base_model)
    os.environ["EVO2_BASE_MODEL"] = args.base_model
    print(f"[generate_bgc] substrate: {args.base_model}", flush=True)

    # Resolve prompts.
    rng = random.Random(args.seed)
    if args.from_jsonl is not None:
        records = [json.loads(l) for l in args.from_jsonl.open()]
        prompts = sample_prompts(records, args.per_class, rng)
    elif args.cls is not None and args.taxon is not None:
        prompts = [{"compound_class": args.cls, "taxonomic_tag": args.taxon}]
    else:
        ap.error("provide --from-jsonl, or both --class and --taxon")

    total = len(prompts) * args.n
    print(f"Prompts: {len(prompts)}  x  {args.n} sample(s) = {total} sequences",
          file=sys.stderr)
    print(f"Decoding: temp={args.temperature} top_k={args.top_k} top_p={args.top_p} "
          f"max_new={args.max_new_tokens} max_windows={args.max_windows}", file=sys.stderr)
    print(f"Model: {'base Evo2 (baseline)' if args.adapter is None else args.adapter}",
          file=sys.stderr)
    if args.dry_run:
        for p in prompts[:10]:
            print(f"  prompt: class={p['compound_class']} tax={p['taxonomic_tag'][:50]}...",
                  file=sys.stderr)
        print("[dry-run] model not loaded.", file=sys.stderr)
        return

    from evo2_inference import load_evo2_wrapper_for_inference
    print("Loading Evo2 + adapter (merging)...", file=sys.stderr, flush=True)
    wrapper = load_evo2_wrapper_for_inference(args.adapter, device=args.device)

    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    # Flatten (prompt, sample) into instances; the id scheme gen_{pi:04d}_{si} and
    # iteration order are preserved identically by both the sequential and the
    # batched paths (consumers key records by this id).
    instances = [(pi, si, p) for pi, p in enumerate(prompts) for si in range(args.n)]
    n_done = n_eos = 0

    def _write(sid: str, rec: dict, fa, jl) -> None:
        nonlocal n_done, n_eos
        rec["id"] = sid
        jl.write(json.dumps(rec) + "\n")
        fa.write(to_fasta_record(
            sid, rec["sequence"], compound_class=rec["compound_class"],
            length=rec["length"], eos=rec["hit_eos"], windows=rec["windows"]))
        n_done += 1
        n_eos += int(rec["hit_eos"])
        if n_done % 10 == 0:
            print(f"  {n_done}/{total} generated ({n_eos} hit EOS)",
                  file=sys.stderr, flush=True)

    with args.out_fasta.open("w") as fa, args.out_jsonl.open("w") as jl:
        if should_batch(args):
            chunk = len(instances) if args.batch_size <= 0 else args.batch_size
            print(f"Batched generation: batch_size={chunk} "
                  f"({len(instances)} sequences, single window).", file=sys.stderr)
            for start in range(0, len(instances), chunk):
                group = instances[start:start + chunk]
                recs = generate_batch(wrapper, [g[2] for g in group], args)
                for (pi, si, _), rec in zip(group, recs):
                    _write(f"gen_{pi:04d}_{si}", rec, fa, jl)
        else:
            for pi, si, p in instances:
                rec = generate_one(wrapper, p["compound_class"], p["taxonomic_tag"], args)
                _write(f"gen_{pi:04d}_{si}", rec, fa, jl)

    print(f"\nDone: {n_done} sequences ({n_eos} terminated with EOS).", file=sys.stderr)
    print(f"  FASTA: {args.out_fasta}\n  JSONL: {args.out_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
