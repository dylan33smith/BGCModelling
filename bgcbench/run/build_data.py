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
from bgcbench.data import mibig, negative, split, tar_index
from bgcbench.data.classmap import BENCHMARK_CLASSES, build_map, validate

ROOT = Path("/data2/ds85/bgcbench")
CORPUS = ROOT / "corpus" / "core_records.jsonl"
SPLITS = ROOT / "splits"
NEG = ROOT / "negatives"
WORK = ROOT / "work"
MANIFEST = ROOT / "manifest.json"

MAX_LEN = 16000          # SPEC 4.7, fixed by G2
COMMON_N = 1224          # SPEC 4.4.3, ARYLPOLYENE binds
NEG_PER_CLASS = 300      # SPEC 4.8: n>=300 bounds an observed-zero FPR below ~0.01


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--common-n", type=int, default=COMMON_N)
    ap.add_argument("--neg-per-class", type=int, default=NEG_PER_CLASS)
    ap.add_argument("--threads", type=int, default=16)
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
        pool = [g for g in sorted(index) if g not in used]
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
                out.extend(got[:1])
                ti.pop(0)
        with open(NEG / f"{cls}.jsonl", "w") as fh:
            for r in out:
                fh.write(json.dumps(r) + "\n")
        v = negative.verify(out, recs)
        neg_report[cls] = v
        print(f"  {cls:14s} n={v['n']:4d} medlen neg/pos={v['median_len_neg']}/"
              f"{v['median_len_pos']}  gc neg/pos={v['median_gc_neg']}/{v['median_gc_pos']}"
              f"  medCDS={v['median_cds_neg']}  genomes={v['genomes']}", flush=True)

    mf.update(MANIFEST, "_build", {"max_len": args.max_len, "common_n": args.common_n,
                                   "corpus": str(CORPUS)})
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
