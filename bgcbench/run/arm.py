"""Run one arm coordinate end to end (SPEC 6, 7, 3).

generate -> antiSMASH -> class map -> novelty (per-arm AND corpus-level) -> scored records
-> endpoints. Exactly the path the oracle validated, with a model in place of a real core.

One arm here means one (substrate, arm, stage); it generates toward EVERY benchmark class,
because the K x K confusion matrix and the lift denominator both need the full row set.

⚠ WHERE THE CLASS ENTERS DECIDES HOW MANY TIMES AN ARM GENERATES (SPEC 6.0).

  class enters at GENERATION time (seeded prompt, steering, iterative refine)
      -> the target varies the output. Generate once PER TARGET; fills a whole row set.

  class enters through the WEIGHTS (a per-class adapter)
      -> one adapter is one distribution. It cannot be "conditioned toward TERPENE": the
         adapter IS the conditioning. Generate ONCE and it fills exactly ONE row of the
         matrix -- its own class's. The five per-class adapters together fill the 5x5.

  class does not enter at all (base, pooled adapter, de novo)
      -> one distribution serving every row, which is also the unconditioned marginal
         that lift divides by.

Getting this wrong is not just wasteful. Generating a RIPP adapter five times "toward"
five different classes would be five samples of ONE distribution reported as five
independent measurements, and the confusion matrix built from them would be five identical
rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bgcbench.data.classmap import BENCHMARK_CLASSES, build_map
from bgcbench.model import genconfig as gc
from bgcbench.model.genconfig import FROZEN as GEN_FROZEN
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


def _sha_of(path: str | None) -> str | None:
    """Fingerprint the WEIGHTS. Without this, two runs of one arm at different checkpoints
    produce byte-identical provenance and are indistinguishable on disk."""
    if not path:
        return None
    p = Path(path)
    h = hashlib.sha256()
    files = sorted(p.rglob("*")) if p.is_dir() else [p]
    for f in files:
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _corpus_sha() -> str | None:
    """Which DATA an arm was run against. Two arms compared across a corpus rebuild are
    not comparable, and nothing else in the report would show it."""
    try:
        return json.loads((ROOT / "manifest.json").read_text())["_build"]["corpus_sha256"]
    except Exception:
        return None


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
            class_bearing: bool, cpus: int = 16, row_class: str | None = None) -> dict:
    mapping = build_map()["mapping"]
    corpus_fa = ensure_corpus_reference()
    by_target: dict[str, list[dict]] = {}
    empty_counts: dict[str, int] = {}
    pooled_ref = None

    targets = list(BENCHMARK_CLASSES)
    if row_class:
        targets = [row_class]          # weights carry the class: one adapter, one row
    elif not class_bearing:
        targets = targets[:1]          # nothing carries the class: one row set

    for cls in targets:
        seed_pool = _load(SPLITS / cls / "test.jsonl") if arm.seeded else None
        gens = generate(sub, arm, cls, n, cfg, seed_pool=seed_pool, stage=stage)

        # ⚠ EMPTY GENERATIONS STAY IN THE DENOMINATOR.
        # A draw whose terminator lands at position 0 cleans to "" -- a real outcome of a
        # model that has learned to stop. Dropping it before scoring inflates the rate and
        # is DIRECTIONALLY BIASED toward the hypothesis: only a model that emits its
        # terminator can lose a draw this way, so the deletion concentrates in the trained
        # arms and is absent from the base control they are compared against. Two arms with
        # identical biology, 70 on-target of 200 draws each, report 0.35 vs 0.50 if one had
        # 60 instant terminations. They are scored as non-detections, which is what they are.
        nonempty = [g for g in gens if g["sequence"]]
        n_empty = len(gens) - len(nonempty)
        verdicts = antismash.run([(g["generation_id"], g["sequence"]) for g in nonempty],
                                 workdir=WORK, cpus=cpus) if nonempty else {}
        for g in gens:
            if not g["sequence"]:
                verdicts[g["generation_id"]] = {
                    "scored_ok": True, "detected": False, "products": [],
                    "region_table": [], "n_cds": 0, "coding_density": 0.0,
                    "produced_core_genes": 0, "gc_content": None, "scored_len": 0}
        cvs = corpus_novelty(nonempty, corpus_fa, workdir=WORK, threads=cpus) \
            if nonempty else {}
        empty_counts[cls] = n_empty

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

    if row_class:
        # one adapter fills exactly its own row; the other rows belong to other adapters
        by_target = {c: (by_target[row_class] if c == row_class else [])
                     for c in BENCHMARK_CLASSES}
    elif not class_bearing:
        one = by_target[targets[0]]
        by_target = {c: one for c in BENCHMARK_CLASSES}

    classes = list(BENCHMARK_CLASSES)
    unconditioned = [r for r in by_target[classes[0]]] if not class_bearing else \
        [r for rows in by_target.values() for r in rows]
    realised = gc.realised(
        n_per_row=n, budget_nt=cfg.budget_nt, batch_size=cfg.batch_size,
        rng_seed=cfg.seed, temperature=arm.temperature, top_k=arm.top_k,
        top_p=arm.top_p, seeded=arm.seeded, seed_len_nt=arm.seed_len_nt,
        weight_state=arm.weight_state, adapter=arm.adapter_path,
        adapter_sha=_sha_of(arm.adapter_path), row_class=row_class or "ALLROWS",
        substrate=sub.id, checkpoint=sub.checkpoint,
        scoring_config=antismash.config_hash(), corpus_sha256=_corpus_sha(),
    )
    rhash = gc.realised_hash(realised)

    report = {
        "arm": arm.arm_id, "substrate": sub.id, "stage": stage,
        "realised": realised,
        "realised_config_hash": rhash,
        # which realised values differ from FROZEN; {} means a frozen-config run
        "off_frozen": gc.off_frozen(realised),
        "n_empty_generations": empty_counts,
        "class_bearing": class_bearing, "n_per_class": n,
        "budget_nt": cfg.budget_nt,
        "row_class": row_class,
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
        "generation_config_frozen": GEN_FROZEN,
    }
    # the config hash is IN the run directory name, so a run under different generation
    # settings cannot silently overwrite or be mistaken for this one
    # SPEC 9.5: <stage>_<SUBSTRATE>_<ARM>_<CLASS>, plus a hash OF THE RUN. Naming it with
    # the frozen-literal hash made all five per-class adapters resolve to one path, each
    # truncating the last -- and the survivor rendered the four destroyed rows as
    # {"n": 0, "detect_rate": null}, which is byte-identical to the SPEC 6.5
    # NOT-APPLICABLE encoding. Four deleted measurements would have read as four
    # structural absences.
    d = RUNS / f"{stage}_{sub.id}_{arm.arm_id}_{row_class or 'ALLROWS'}_{rhash}"
    if d.exists():
        raise RuntimeError(
            f"{d} already exists — refusing to overwrite a measurement. The only file in "
            f"this repo that had this guard (score_gates) learned it the hard way. Delete "
            f"it deliberately if you mean to replace it."
        )
    d.mkdir(parents=True, exist_ok=False)
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
    ap.add_argument("--seed-len", type=int, default=GEN_FROZEN["seed_len_nt"],
                    help="seeded-regime prompt length. Frozen: an unrecorded default of 0 "
                         "made `--seeded` without this flag a silent DE NOVO arm.")
    ap.add_argument("--n", type=int, default=GEN_FROZEN["n_per_row"],
                    help="OVERRIDES the frozen n. For smoke tests only: any run "
                         "that differs from the frozen config is flagged in its report.")
    ap.add_argument("--budget-nt", type=int, default=GEN_FROZEN["budget_nt"])
    ap.add_argument("--batch-size", type=int, default=GEN_FROZEN["batch_size"])
    ap.add_argument("--row-class", default=None,
                    help="the class this arm's WEIGHTS carry. Set for a per-class adapter: "
                         "it generates once and fills that one row, rather than pretending "
                         "to be conditioned toward five different targets.")
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--cpus", type=int, default=16)
    args = ap.parse_args()

    sub = load(args.substrate)
    adapter = args.adapter
    if adapter:
        # SPEC 6.4: evaluate at the BEST held-out checkpoint, not the last. Point at an
        # adapter DIRECTORY and this resolves to best/ automatically.
        p = Path(adapter)
        if (p / "BEST").exists():
            meta = json.loads((p / "BEST").read_text())
            if meta.get("path"):
                adapter = meta["path"]
                print(f"using BEST checkpoint (step {meta['step']}, "
                      f"val {meta['val_loss']}) rather than final", flush=True)
    if adapter:
        sub = attach_adapter(sub, adapter)
        print(f"attached adapter {adapter}", flush=True)
    arm = ArmSpec(arm_id=args.arm,
                  weight_state="base" if args.adapter is None else args.arm,
                  seeded=args.seeded, seed_len_nt=args.seed_len,
                  adapter_path=args.adapter)
    # class-bearing iff something in the coordinate carries the class
    # the class must enter at GENERATION time for a target to mean anything
    class_bearing = args.seeded or arm.inference_control != "none"
    cfg = GenConfig(budget_nt=args.budget_nt, batch_size=args.batch_size)


    rep = run_arm(sub, arm, args.n, cfg, args.stage, class_bearing, cpus=args.cpus,
                  row_class=args.row_class)
    if rep["off_frozen"]:
        print(f"⚠ NOT THE FROZEN CONFIG — {rep['off_frozen']}. Recorded in the report; "
              f"this run is not comparable to a frozen-config run.", flush=True)
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
