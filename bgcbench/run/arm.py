"""Run one arm coordinate end to end (SPEC 6, 7, 3).

generate -> antiSMASH -> class map -> novelty (per-arm AND corpus-level) -> scored records
-> endpoints. Exactly the path the oracle validated, with a model in place of a real core.

One arm here means one (substrate, arm, stage); it generates toward EVERY benchmark class,
because the K x K confusion matrix and the lift denominator both need the full row set.

⚠ DEGENERATE COORDINATES (SPEC 6.0). With no conditioning channel in the input, an arm
carrying no class-bearing factor produces the same generations regardless of target. Such
an arm is generated ONCE and its output serves every row, which is also exactly the
unconditioned marginal that lift divides by. Generating it five times would be five
samples of one distribution reported as five measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgcbench.data.classmap import BENCHMARK_CLASSES, build_map
from bgcbench.model.generate import ArmSpec, GenConfig, generate
from bgcbench.model.load import attach_adapter, load
from bgcbench.score import antismash
from bgcbench.score.endpoints import confusion, gene_count_profile, lift, rates, subclass_profile
from bgcbench.score.novelty import Reference, corpus_novelty, write_corpus_fasta
from bgcbench.score.record import build as build_records

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
RUNS = ROOT / "runs"
WORK = ROOT / "work"
CORPUS_FA = ROOT / "reference" / "corpus.fasta"


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def ensure_corpus_reference() -> Path:
    """SPEC 3.8 corpus-level reference: every eligible record, so the untrained arms are
    gateable at all."""
    if CORPUS_FA.exists():
        return CORPUS_FA
    recs = []
    seen = set()
    for cls in BENCHMARK_CLASSES:
        for part in ("train", "val", "test"):
            for r in _load(SPLITS / cls / f"{part}.jsonl"):
                if r["accession"] not in seen:
                    seen.add(r["accession"])
                    recs.append(r)
    return write_corpus_fasta(recs, CORPUS_FA)


def run_arm(sub, arm: ArmSpec, n: int, cfg: GenConfig, stage: str,
            class_bearing: bool, cpus: int = 16) -> dict:
    mapping = build_map()["mapping"]
    corpus_fa = ensure_corpus_reference()
    by_target: dict[str, list[dict]] = {}
    pooled_ref = None

    targets = list(BENCHMARK_CLASSES)
    if not class_bearing:
        targets = targets[:1]          # SPEC 6.0: generate once, serve every row

    for cls in targets:
        seed_pool = _load(SPLITS / cls / "test.jsonl") if arm.seeded else None
        gens = generate(sub, arm, cls, n, cfg, seed_pool=seed_pool, stage=stage)
        gens = [g for g in gens if g["sequence"]]
        if not gens:
            by_target[cls] = []
            continue

        verdicts = antismash.run([(g["generation_id"], g["sequence"]) for g in gens],
                                 workdir=WORK, cpus=cpus)
        cvs = corpus_novelty(gens, corpus_fa, workdir=WORK, threads=cpus)

        # SPEC 3.8: the per-arm reference is the split this arm actually trained on. An
        # untrained arm has none, and Reference([]) raises by design -- so it is gated on
        # the corpus-level reference alone, and that is stated rather than silently skipped.
        if arm.weight_state == "base":
            ref = pooled_ref or Reference(_load(SPLITS / cls / "train.jsonl"))
            pooled_ref = ref
        else:
            ref = Reference(_load(SPLITS / cls / "train.jsonl"))

        by_target[cls] = build_records(gens, verdicts, mapping, ref, arm=arm.arm_id,
                                       substrate=sub.id, stage=stage,
                                       corpus_verdicts=cvs)

    if not class_bearing:
        one = by_target[targets[0]]
        by_target = {c: one for c in BENCHMARK_CLASSES}

    classes = list(BENCHMARK_CLASSES)
    unconditioned = [r for r in by_target[classes[0]]] if not class_bearing else \
        [r for rows in by_target.values() for r in rows]
    report = {
        "arm": arm.arm_id, "substrate": sub.id, "stage": stage,
        "class_bearing": class_bearing, "n_per_class": n,
        "budget_nt": cfg.budget_nt,
        "per_class": {c: rates(by_target[c], c) for c in classes},
        "confusion": confusion(by_target, classes),
        "lift": lift(by_target, unconditioned, classes),
        "subclass": {c: subclass_profile(by_target[c], c, mapping) for c in classes},
        "gene_counts": {c: gene_count_profile(by_target[c], c) for c in classes},
        "novelty": {c: {
            "per_arm_gate": {g: sum(1 for r in by_target[c] if r["gate"] == g)
                             for g in {r["gate"] for r in by_target[c]}},
            "matched_known_bgc": sum(1 for r in by_target[c] if r["matched_known_bgc"]),
            "n": len(by_target[c]),
        } for c in classes},
        "hit_eos_rate": round(
            sum(1 for rows in by_target.values() for r in rows if r["hit_eos"])
            / max(sum(len(v) for v in by_target.values()), 1), 4),
        "median_len": sorted(r["seq_len"] for rows in by_target.values()
                             for r in rows)[max(0, sum(len(v) for v in by_target.values()) // 2)]
        if any(by_target.values()) else 0,
        "scoring_config": antismash.config_hash(),
    }
    d = RUNS / f"{stage}_{sub.id}_{arm.arm_id}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "scored.jsonl", "w") as fh:
        seen = set()
        for rows in by_target.values():
            for r in rows:
                if r["generation_id"] in seen:
                    continue
                seen.add(r["generation_id"])
                fh.write(json.dumps(r) + "\n")
    (d / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="evo2-1b")
    ap.add_argument("--arm", default="base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--seeded", action="store_true")
    ap.add_argument("--seed-len", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget-nt", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--cpus", type=int, default=16)
    args = ap.parse_args()

    sub = load(args.substrate)
    if args.adapter:
        sub = attach_adapter(sub, args.adapter)
        print(f"attached adapter {args.adapter}", flush=True)
    arm = ArmSpec(arm_id=args.arm,
                  weight_state="base" if args.adapter is None else args.arm,
                  seeded=args.seeded, seed_len_nt=args.seed_len,
                  adapter_path=args.adapter)
    # class-bearing iff something in the coordinate carries the class
    class_bearing = bool(args.adapter) or args.seeded
    cfg = GenConfig(budget_nt=args.budget_nt, batch_size=args.batch_size)

    rep = run_arm(sub, arm, args.n, cfg, args.stage, class_bearing, cpus=args.cpus)
    print(f"\narm={rep['arm']} substrate={rep['substrate']} "
          f"class_bearing={rep['class_bearing']} hit_eos={rep['hit_eos_rate']} "
          f"median_len={rep['median_len']}")
    for c, r in rep["per_class"].items():
        nv = rep["novelty"][c]
        print(f"  {c:14s} detect={r['detect_rate']} precision={r['precision']} "
              f"on_target={r['on_target_rate']}  known_bgc={nv['matched_known_bgc']}/{nv['n']} "
              f"gates={nv['per_arm_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
