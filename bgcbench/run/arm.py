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

from bgcbench.data.classmap import BENCHMARK_CLASSES, build_map, mapping_hash
from bgcbench.model import genconfig as gc
from bgcbench.model.genconfig import FROZEN as GEN_FROZEN
from bgcbench.model.generate import ArmSpec, GenConfig, generate
from bgcbench.model.load import attach_adapter, attach_intervention, load
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


#: antiSMASH switches prodigal to a different gene-calling mode above this per-sequence
#: length. Crossing it would score two substrates with different gene callers, and the
#: difference would read as a substrate effect.
PRODIGAL_MODE_SWITCH_NT = 20000


def _assert_scorable(budget_nt: int, sub) -> None:
    if budget_nt >= PRODIGAL_MODE_SWITCH_NT:
        raise SystemExit(
            f"budget {budget_nt} >= {PRODIGAL_MODE_SWITCH_NT}: antiSMASH would switch "
            f"prodigal gene-calling mode, so arms above and below the threshold would be "
            f"scored by different callers."
        )
    ctx = (sub.meta or {}).get("max_seqlen")
    if ctx and budget_nt > ctx:
        raise SystemExit(
            f"budget {budget_nt} exceeds {sub.id}'s usable context {ctx}. Measured on "
            f"evo2-1b: NLL rises from 0.805 at 8,192 to 1.239 at 15,900, against a chance "
            f"level of 1.386 — the arm would generate near-random sequence."
        )


def _novelty_params() -> dict:
    """Gate parameters were module constants recorded nowhere; two arms scored across an
    edit would be gated differently with nothing showing it."""
    from bgcbench.score import novelty as nv
    return {"K": nv.K, "fail_at": nv.FAIL_AT, "warn_at": nv.WARN_AT,
            "min_ref_kmers": nv.MIN_REF_KMERS,
            "corpus_min_id": nv.CORPUS_MIN_ID, "corpus_cov": nv.CORPUS_COV}


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
            class_bearing: bool, cpus: int = 16, row_class: str | None = None,
            train_classes: list[str] | None = None,
            intervention: object | None = None) -> dict:
    mapping = build_map()["mapping"]
    corpus_fa = ensure_corpus_reference()
    by_target: dict[str, list[dict]] = {}
    empty_counts: dict[str, int] = {}
    ref_cache: dict[tuple, object] = {}
    ref_used: dict[str, list[str]] = {}

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

        # SPEC 3.8: the per-arm reference is THE SPLIT THIS ARM TRAINED ON, not the split
        # of whichever class is being scored. Keying it on the target class meant a pooled
        # arm was gated against one class's train set, and an untrained arm was gated
        # against TERPENE's -- so the same memorised output could pass or fail depending on
        # which row it landed in, and no artifact recorded which reference was used.
        ref_classes = train_classes if train_classes else [cls]
        key = tuple(sorted(ref_classes))
        if key not in ref_cache:
            recs = [r for c in ref_classes for r in _load(SPLITS / c / "train.jsonl")]
            ref_cache[key] = Reference(recs)
        ref = ref_cache[key]
        ref_used[cls] = list(ref_classes)

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
    # ⚠ THE LIFT DENOMINATOR DIFFERS BY ARM SHAPE, so it is NAMED, not left implicit under
    # one field. A per-class adapter has no unconditioned counterpart of its own -- the
    # honest denominator is the arm that shares its weights minus the class signal, which
    # only exists once the pooled arm is run. Publishing all three under "lift" would
    # compare quantities that are not the same quantity.
    if row_class:
        unconditioned, denom = [], "none — per-class adapter has no unconditioned twin"
    elif not class_bearing:
        unconditioned, denom = by_target[classes[0]], "own output (one distribution)"
    else:
        unconditioned = [r for rows in by_target.values() for r in rows]
        denom = "pooled across this arm's own targets"
    realised = gc.realised(
        n_per_row=n, budget_nt=cfg.budget_nt, batch_size=cfg.batch_size,
        rng_seed=cfg.seed, temperature=arm.temperature, top_k=arm.top_k,
        top_p=arm.top_p, seeded=arm.seeded, seed_len_nt=arm.seed_len_nt,
        weight_state=arm.weight_state, adapter=arm.adapter_path,
        adapter_sha=_sha_of(arm.adapter_path), row_class=row_class or "ALLROWS",
        substrate=sub.id, checkpoint=sub.checkpoint,
        scoring_config=antismash.config_hash(), corpus_sha256=_corpus_sha(),
        termination_mode=sub.termination_mode,
        # SPEC 6.5: an arm attached at 4 of 25 sites is not the same arm as one at 32 of
        # 32, and a W3 run at rank 16 is not the one at rank 64. Without these in the
        # REALISED hash both collide on one run directory and the loser is destroyed.
        intervention_method=("offset" if intervention is not None else None),
        intervention_rank=(getattr(intervention, "rank", None)
                           if intervention is not None else None),
        intervention_trainable=(intervention.n_trainable()
                                if intervention is not None else None),
        intervention_sites=((sub.meta or {}).get("intervention_sites")
                            if intervention is not None else None),
        intervention_attached=(bool(intervention.is_attached())
                               if intervention is not None else None),
        classmap_hash=mapping_hash(mapping),
        novelty=_novelty_params(),
    )
    rhash = gc.realised_hash(realised)
    # the instrument that ACTUALLY ran, only knowable after the first scoring call
    realised["antismash_observed_version"] = antismash.OBSERVED.get("version")

    report = {
        "arm": arm.arm_id, "substrate": sub.id, "stage": stage,
        "realised": realised,
        "realised_config_hash": rhash,
        # which realised values differ from FROZEN; {} means a frozen-config run
        "off_frozen": gc.off_frozen(realised),
        "n_empty_generations": empty_counts,
        "novelty_reference_classes": ref_used,
        "class_bearing": class_bearing, "n_per_class": n,
        "budget_nt": cfg.budget_nt,
        "row_class": row_class,
        "per_class": {c: rates(by_target[c], c) for c in classes},
        "confusion": confusion(by_target, classes),
        "lift": lift(by_target, unconditioned, classes) if unconditioned else None,
        "lift_denominator": denom,
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
    ap.add_argument("--train-classes", nargs="+", default=None,
                    help="the classes this arm's WEIGHTS were trained on. Sets the SPEC 3.8 "
                         "per-arm novelty reference. Omit for an untrained arm.")
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
        if p.is_dir() and (p / "BEST").exists():
            meta = json.loads((p / "BEST").read_text())
            if meta.get("path"):
                adapter = meta["path"]
                print(f"using BEST checkpoint (step {meta['step']}, "
                      f"val {meta['val_loss']}) rather than final", flush=True)
    intervention = None
    if adapter and str(adapter).endswith(".pt"):
        sub, intervention = attach_intervention(sub, adapter)
        print(f"attached intervention {adapter} "
              f"({sub.meta.get('intervention_sites', {}).get('n_attention_sites')} sites)",
              flush=True)
    elif adapter:
        sub = attach_adapter(sub, adapter)
        print(f"attached adapter {adapter}", flush=True)
    arm = ArmSpec(arm_id=args.arm,
                  weight_state="base" if args.adapter is None else args.arm,
                  seeded=args.seeded, seed_len_nt=args.seed_len,
                  adapter_path=args.adapter)
    # class-bearing iff something in the coordinate carries the class
    # the class must enter at GENERATION time for a target to mean anything
    class_bearing = args.seeded or arm.inference_control != "none"
    # A per-class adapter carries the class in its WEIGHTS. Run without --row-class it
    # would replicate one sample into all five confusion rows and self-divide lift to 1.0,
    # producing a report that is structurally indistinguishable from a real result.
    if args.adapter and not args.row_class and not class_bearing and not args.train_classes:
        raise SystemExit(
            "refusing to run: an adapter arm needs either --row-class (the class its "
            "weights carry, for a per-class adapter) or --train-classes (for a pooled "
            "adapter, whose weights carry no single class). Without one, five identical "
            "rows would be published as a confusion matrix."
        )
    cfg = GenConfig(budget_nt=args.budget_nt, batch_size=args.batch_size)
    _assert_scorable(cfg.budget_nt, sub)


    if intervention is not None:
        # a hook, not merged weights: if it is not attached for the whole of generation the
        # arm silently produces base-model output and reads as a null
        with intervention.attached():
            # recorded from INSIDE the context, so the artifact attests that the hooks were
            # live when the numbers were produced rather than that the code intended it
            rep = run_arm(sub, arm, args.n, cfg, args.stage, class_bearing, cpus=args.cpus,
                          row_class=args.row_class, train_classes=args.train_classes,
                          intervention=intervention)
    else:
        rep = run_arm(sub, arm, args.n, cfg, args.stage, class_bearing, cpus=args.cpus,
                      row_class=args.row_class, train_classes=args.train_classes)
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
