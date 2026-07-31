#!/usr/bin/env python3
"""Measure GenomeOcean's BPE tokenization against our splits_core BGC dataset.

The question this answers: **does our data fit GenomeOcean's context, and how
does that compare to Evo2?**

Evo2 uses a byte-level CharLevelTokenizer (1 token == 1 nucleotide), so its
production context of L=32,768 tokens buys exactly 32,768 bp. GenomeOcean uses a
4,096-entry BPE vocabulary, so its 10,240-token context buys however many bp the
compression ratio gives -- the paper claims ~5x (~51 kb), which we verify here on
*our* cores rather than trusting the number.

Outputs a JSON report + a human-readable table:
  - bp/token compression, overall and per compound class
  - fraction of cores that fit whole at each candidate context length
  - the same fractions for Evo2 at L=32,768, for a like-for-like comparison
  - megasynthase (NRPS/PKS/HYBRID) breakout, since those are the long cores that
    Evo2 could never fit whole (see docs/project_memory/progress.md)

Usage:
  python genomeocean/scripts/analyze_tokenization.py \
      --jsonl /data2/ds85/bgcmodel_data/splits_core/val.jsonl \
      --out genomeocean/experiments/tokenization_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# GenomeOcean-4B trained context (README) and the ceiling implied by
# config.json max_position_embeddings=32768 with rope_theta=1e6.
GO_TRAINED_CTX = 10_240
GO_ROPE_CEILING = 32_768
# Evo2 production shape from CLAUDE.md "Current Decisions".
EVO2_CTX_BP = 32_768

MEGA_CLASSES = {"NRPS", "PKS", "HYBRID"}


def load_records(path: Path, limit: int | None, seed: int) -> list[dict]:
    """Reservoir-sample `limit` records.

    splits_core JSONL is grouped by genome, so taking the head of the file gives a
    taxonomically skewed sample. Reservoir sampling keeps one streaming pass.
    """
    import random

    rng = random.Random(seed)
    reservoir: list[dict] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            if limit is None or len(reservoir) < limit:
                reservoir.append(rec)
            else:
                j = rng.randrange(i + 1)
                if j < limit:
                    reservoir[j] = rec
    return reservoir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, required=True,
                    help="splits_core JSONL (train/val/test).")
    ap.add_argument("--model", default="pGenomeOcean/GenomeOcean-4B-bgcFM",
                    help="HF model id or local path holding tokenizer.json.")
    ap.add_argument("--limit", type=int, default=2000,
                    help="Max records to tokenize (0 = all). Tokenizing is the slow part.")
    ap.add_argument("--seed", type=int, default=0, help="Reservoir-sampling seed.")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON report here.")
    args = ap.parse_args()

    from transformers import PreTrainedTokenizerFast

    # PreTrainedTokenizerFast, not AutoTokenizer: transformers>=5.0 routes
    # model_type=mistral through the MistralCommon backend, which needs a
    # tekken.json that GenomeOcean checkpoints do not ship. See the upstream
    # README migration note.
    tok = PreTrainedTokenizerFast.from_pretrained(args.model)

    recs = load_records(args.jsonl, args.limit or None, args.seed)
    if not recs:
        raise SystemExit(f"no records read from {args.jsonl}")

    rows = []
    for r in recs:
        seq = r["sequence"].upper()
        n_tok = len(tok.encode(seq, add_special_tokens=False))
        rows.append({
            "accession": r.get("accession"),
            "compound_class": r.get("compound_class", "UNKNOWN"),
            "bp": len(seq),
            # Full antiSMASH region the strict core was cut from. We only train on
            # the core today, but this is the length a "train on whole regions"
            # variant would need to fit -- the case where context length bites.
            "region_bp": r.get("region_len") or 0,
            "tokens": n_tok,
            "bp_per_token": len(seq) / n_tok if n_tok else 0.0,
        })

    ratios = [x["bp_per_token"] for x in rows]
    total_bp = sum(x["bp"] for x in rows)
    total_tok = sum(x["tokens"] for x in rows)

    def frac_fits(subset: list[dict], ctx_tokens: int) -> float:
        if not subset:
            return float("nan")
        return sum(1 for x in subset if x["tokens"] <= ctx_tokens) / len(subset)

    def frac_fits_evo2(subset: list[dict]) -> float:
        if not subset:
            return float("nan")
        return sum(1 for x in subset if x["bp"] <= EVO2_CTX_BP) / len(subset)

    mega = [x for x in rows if x["compound_class"] in MEGA_CLASSES]

    by_class: dict[str, list[dict]] = defaultdict(list)
    for x in rows:
        by_class[x["compound_class"]].append(x)

    report = {
        "model": args.model,
        "jsonl": str(args.jsonl),
        "n_records": len(rows),
        "compression": {
            "bp_per_token_overall": total_bp / total_tok,
            "bp_per_token_mean": statistics.mean(ratios),
            "bp_per_token_median": statistics.median(ratios),
            "bp_per_token_min": min(ratios),
            "bp_per_token_max": max(ratios),
        },
        "context_fit": {
            "genomeocean_trained_10240tok": {
                "all": frac_fits(rows, GO_TRAINED_CTX),
                "megasynthase": frac_fits(mega, GO_TRAINED_CTX),
            },
            "genomeocean_rope_ceiling_32768tok": {
                "all": frac_fits(rows, GO_ROPE_CEILING),
                "megasynthase": frac_fits(mega, GO_ROPE_CEILING),
            },
            "evo2_32768bp": {
                "all": frac_fits_evo2(rows),
                "megasynthase": frac_fits_evo2(mega),
            },
        },
        "by_class": {},
    }

    # Whole-REGION scenario: strict cores are short, so context length barely
    # separates the two models on today's data. It separates them a lot if we ever
    # train on full antiSMASH regions.
    regions = [x for x in rows if x["region_bp"] > 0]
    if regions:
        c_ratio = report["compression"]["bp_per_token_overall"]
        mega_regions = [x for x in regions if x["compound_class"] in MEGA_CLASSES]

        def frac_region_fits(subset, ctx_tokens=None, ctx_bp=None):
            if not subset:
                return float("nan")
            if ctx_bp is not None:
                return sum(1 for x in subset if x["region_bp"] <= ctx_bp) / len(subset)
            return sum(1 for x in subset
                       if x["region_bp"] / c_ratio <= ctx_tokens) / len(subset)

        report["whole_region_fit"] = {
            "note": ("region token counts are ESTIMATED as region_bp / measured "
                     "bp-per-token; the strict-core numbers above are exact."),
            "median_region_bp": statistics.median([x["region_bp"] for x in regions]),
            "median_mega_region_bp": (statistics.median([x["region_bp"] for x in mega_regions])
                                      if mega_regions else None),
            "genomeocean_trained_10240tok": {
                "all": frac_region_fits(regions, ctx_tokens=GO_TRAINED_CTX),
                "megasynthase": frac_region_fits(mega_regions, ctx_tokens=GO_TRAINED_CTX),
            },
            "evo2_32768bp": {
                "all": frac_region_fits(regions, ctx_bp=EVO2_CTX_BP),
                "megasynthase": frac_region_fits(mega_regions, ctx_bp=EVO2_CTX_BP),
            },
        }

    for cls, subset in sorted(by_class.items()):
        bp = [x["bp"] for x in subset]
        report["by_class"][cls] = {
            "n": len(subset),
            "median_bp": statistics.median(bp),
            "max_bp": max(bp),
            "median_tokens": statistics.median([x["tokens"] for x in subset]),
            "bp_per_token": sum(bp) / sum(x["tokens"] for x in subset),
            "fits_go_10240tok": frac_fits(subset, GO_TRAINED_CTX),
            "fits_evo2_32768bp": frac_fits_evo2(subset),
        }

    c = report["compression"]
    print(f"\n=== GenomeOcean BPE on {args.jsonl.name} (n={len(rows)}) ===")
    print(f"bp/token  overall {c['bp_per_token_overall']:.3f}  "
          f"median {c['bp_per_token_median']:.3f}  "
          f"range [{c['bp_per_token_min']:.2f}, {c['bp_per_token_max']:.2f}]")
    eff_kb = GO_TRAINED_CTX * c["bp_per_token_overall"] / 1000
    ceil_kb = GO_ROPE_CEILING * c["bp_per_token_overall"] / 1000
    print(f"effective context: {GO_TRAINED_CTX} tok ~ {eff_kb:.1f} kb "
          f"(RoPE ceiling {GO_ROPE_CEILING} tok ~ {ceil_kb:.1f} kb) "
          f"vs Evo2 {EVO2_CTX_BP/1000:.1f} kb")

    f = report["context_fit"]
    print("\nfraction of cores that fit WHOLE (no chunking):")
    print(f"  GenomeOcean @10,240 tok : all {f['genomeocean_trained_10240tok']['all']:.3f}   "
          f"mega {f['genomeocean_trained_10240tok']['megasynthase']:.3f}")
    print(f"  GenomeOcean @32,768 tok : all {f['genomeocean_rope_ceiling_32768tok']['all']:.3f}   "
          f"mega {f['genomeocean_rope_ceiling_32768tok']['megasynthase']:.3f}")
    print(f"  Evo2        @32,768 bp  : all {f['evo2_32768bp']['all']:.3f}   "
          f"mega {f['evo2_32768bp']['megasynthase']:.3f}")

    wr = report.get("whole_region_fit")
    if wr:
        print(f"\nif we trained on WHOLE antiSMASH regions instead of strict cores "
              f"(median region {wr['median_region_bp']:.0f} bp, "
              f"mega {wr['median_mega_region_bp']:.0f} bp):")
        print(f"  GenomeOcean @10,240 tok : all {wr['genomeocean_trained_10240tok']['all']:.3f}   "
              f"mega {wr['genomeocean_trained_10240tok']['megasynthase']:.3f}")
        print(f"  Evo2        @32,768 bp  : all {wr['evo2_32768bp']['all']:.3f}   "
              f"mega {wr['evo2_32768bp']['megasynthase']:.3f}")

    print(f"\n{'class':<14}{'n':>6}{'med_bp':>9}{'med_tok':>9}{'bp/tok':>8}"
          f"{'fit_GO':>9}{'fit_Evo2':>10}")
    for cls, d in sorted(report["by_class"].items(),
                         key=lambda kv: -kv[1]["median_bp"]):
        print(f"{cls:<14}{d['n']:>6}{d['median_bp']:>9.0f}{d['median_tokens']:>9.0f}"
              f"{d['bp_per_token']:>8.2f}{d['fits_go_10240tok']:>9.3f}"
              f"{d['fits_evo2_32768bp']:>10.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
