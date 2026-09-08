"""Build the whole SPEC 4 data layer, end to end.

    python -m bgcbench.run.build_data [--skip-extract]

Stages: corpus -> class splits (global, cluster-aware) -> verification -> negative
controls -> manifest. Every stage writes its numbers to the manifest; nothing is
reported that was not measured.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from bgcbench.data import manifest as mf
from bgcbench.data import mibig, negative, split, stratify, tar_index
from bgcbench.data.classmap import BENCHMARK_CLASSES, build_map, validate

ROOT = Path("/data2/ds85/bgcbench")
CORPUS = ROOT / "corpus" / "core_records.jsonl"
SPLITS = ROOT / "splits"
NEG = ROOT / "negatives"
WORK = ROOT / "work"
MANIFEST = ROOT / "manifest.json"

#: SPEC 4.7, fixed by G2 and by the MEASURED degradation curve. evo2-1b's config
#: max_seqlen is 8192; NLL of the last 1000 tokens of a prefix reads 0.805 at 8,192,
#: 1.040 at 12,000 and 1.239 at 15,900 against ln(4)=1.386 chance. A bound above the
#: usable context trains on record tails the model reads at near-chance.
MAX_LEN = 8192
COMMON_N = 0             # SPEC 4.4.3: 0 = derive from the smallest class's effective_n
NEG_PER_CLASS = 300      # SPEC 4.8: n>=300 bounds an observed-zero FPR below ~0.01


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--common-n", type=int, default=COMMON_N,
                    help="0 derives it from the smallest class's cluster count, so it "
                         "cannot be a stale literal from a previous bound")
    ap.add_argument("--neg-per-class", type=int, default=NEG_PER_CLASS)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--strata", action="store_true",
                    help="build the SPEC 4.5.2 RIPP strata. OFF by default: they are a "
                         "conditional sub-experiment gated on a de novo ladder gradient, "
                         "nothing in the main benchmark depends on them, and rebuilding "
                         "them on every pass costs ~37k records of work for nothing.")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    cm = build_map()
    validate(cm["mapping"])
    mf.update(MANIFEST, "_classmap", {k: v for k, v in cm.items() if k != "mapping"})
    print(f"class map: {cm['n_rules']} products -> {len(cm['classes'])} classes; "
          f"promotions {sorted(cm['promotions'])}", flush=True)

    print("\n== MiBIG external-validation partition ==", flush=True)
    all_recs = split.load_corpus(CORPUS, BENCHMARK_CLASSES, args.max_len)
    held = mibig.partition(all_recs, workdir=WORK, threads=args.threads)
    (ROOT / "mibig").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "mibig" / "external_validation.jsonl", "w") as fh:
        for r in all_recs:
            if r["accession"] in held:
                fh.write(json.dumps(r) + "\n")
    print(f"  {len(held)} of {len(all_recs)} corpus records match MiBIG -> "
          f"excluded from train/val/test, read ONCE at the end", flush=True)
    mf.update(MANIFEST, "_mibig", {"n_held": len(held),
                                   "of_records": len(all_recs),
                                   "source": str(mibig.MIBIG_GBK),
                                   "policy": "excluded from every training corpus; "
                                             "read once; never iterated against"})

    print("\n== splits ==", flush=True)
    rep = split.build(CORPUS, SPLITS, BENCHMARK_CLASSES, args.max_len,
                      args.common_n, workdir=WORK, threads=args.threads,
                      exclude=held)
    for cls, d in rep["classes"].items():
        print(f"  {cls:14s} clusters {d['clusters_used']}/{d['clusters_available']:6d}  "
              f"n={d['n']}  medlen={d['median_seq_len']:5d}  "
              f"multigene={d['multigene_frac']:.3f}  hybrid={d['hybrid_frac']:.3f}  "
              f"products={d['n_products']}", flush=True)

    print("\n== verification ==", flush=True)
    ver = split.verify(SPLITS, BENCHMARK_CLASSES, workdir=WORK, threads=args.threads)
    for cls, d in ver.items():
        print(f"  {cls:14s} genome_overlap={d['genome_overlap']} "
              f"neardup_fwd={d['neardup_fwd']} neardup_rc={d['neardup_revcomp']}",
              flush=True)

    if not args.strata:
        print("\n== within-RIPP strata: SKIPPED (SPEC 4.5.2, pass --strata) ==", flush=True)
        strat = None
    else:
        print("\n== within-RIPP multi-gene stratification (SPEC 4.5) ==", flush=True)
        strat = stratify.build(SPLITS, ROOT / "strata", "RIPP",
                               corpus_path=CORPUS, max_len=args.max_len,
                               exclude=held)
    for part, sr in (strat or {}).items():
        print(f"  {part:6s} matched single/multi = {sr['matched']['single']}/"
              f"{sr['matched']['multi']}  (available {sr['available']['single']}/"
              f"{sr['available']['multi']})  medlen {sr['median_len']['single']}/"
              f"{sr['median_len']['multi']}  medCDS {sr['median_cds']['single']}/"
              f"{sr['median_cds']['multi']}", flush=True)
    if strat:
        mf.update(MANIFEST, "_ripp_strata", strat)

    print("\n== negative controls ==", flush=True)
    NEG.mkdir(parents=True, exist_ok=True)
    index = tar_index.load_index()
    neg_report = {}
    for cls in BENCHMARK_CLASSES:
        recs = [json.loads(l) for l in open(SPLITS / cls / "train.jsonl")]
        targets = negative.length_targets(recs, args.neg_per_class)
        # draw from genomes NOT contributing this class's records, so a control can never
        # be a neighbour of a core in the same genome
        used = {r["genome_accession"] for r in recs}
        # SPREAD ACROSS THE WHOLE INDEX, not its alphabetical head. `sorted(index)` made
        # all five classes walk the same prefix: 312 distinct genomes across all 1,500
        # controls, pairwise overlap 288-295 of 300, so the five per-class FPRs were
        # approximately ONE measurement reported five times. The head of the index is also
        # GC-poor fungi and archaea, which is what produced the apparent core-vs-control
        # GC gap.
        import hashlib as _h
        pool = sorted(
            (g for g in index if g not in used),
            key=lambda g: _h.sha256(f"{cls}|{g}".encode()).hexdigest(),
        )
        # ONE interval per genome. Drawing all 300 from one genome would be
        # pseudo-replication, not 300 independent controls -- the first build did
        # exactly that and reported genomes=1.
        out, ti = [], list(targets)
        for g in pool:
            if not ti:
                break
            try:
                got = negative.sample_from_genome(g, index, ti[:1])
            except Exception:
                continue
            if got:
                r = dict(got[0])
                r["accession"] = f"{r['genome_accession']}.{r['locus']}.neg{len(out)}"
                r["target_class"] = cls
                out.append(r)
                ti.pop(0)
        with open(NEG / f"{cls}.jsonl", "w") as fh:
            for r in out:
                fh.write(json.dumps(r) + "\n")
        v = negative.verify(out, recs)
        neg_report[cls] = v
        print(f"  {cls:14s} n={v['n']:4d} medlen neg/pos={v['median_len_neg']}/"
              f"{v['median_len_pos']}  gc neg/pos={v['median_gc_neg']}/{v['median_gc_pos']}"
              f"  medCDS={v['median_cds_neg']}  genomes={v['genomes']}", flush=True)

    import hashlib
    h = hashlib.sha256()
    with open(CORPUS, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    # the DERIVED common_n, not the CLI flag: --common-n 0 means "derive", and recording
    # the 0 would have published a manifest saying the benchmark used n=0 per class
    mf.update(MANIFEST, "_build", {"max_len": args.max_len,
                                   "common_n": rep.get("common_n", args.common_n),
                                   "common_n_source": ("derived" if not args.common_n
                                                       else "cli"),
                                   "corpus": str(CORPUS),
                                   "corpus_sha256": h.hexdigest(),
                                   "corpus_records": sum(1 for _ in open(CORPUS))})
    for cls in BENCHMARK_CLASSES:
        mf.update(MANIFEST, cls, {"split": rep["classes"][cls],
                                  "verification": ver[cls],
                                  "negative_control": neg_report[cls],
                                  "clustering": rep["clustering"],
                                  "max_len": args.max_len})
    print(f"\nmanifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
