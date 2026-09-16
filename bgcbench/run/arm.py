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
from bgcbench.model.load import (attach_adapter, attach_direction,
                                 load, resolve_best)
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

    # THE LINEAGE POOL FOLLOWS THE ARM'S TRAINING DATA, and is built once outside the target
    # loop. For a per-class adapter that is its own class, matching what it trained on; for a
    # pooled arm the union of its classes; for the untrained floor, all of them. Built inside
    # the loop it would have drawn W0's lineages from whichever class sorted first, so the
    # floor would see a narrower organism distribution than the arms it is the floor for.
    tax_pool = None
    if arm.prefix == "taxonomy":
        from bgcbench.data import taxonomy
        pool_classes = ([row_class] if row_class
                        else (train_classes or list(BENCHMARK_CLASSES)))
        # EQUAL SHARE PER CLASS. A pooled arm draws n/len(classes) lineages from each class's
        # held-out set -- 50 each at n=200 -- taken from the HEAD of that class's
        # accession-sorted list, so they are a strict subset of the 200 that class's own
        # per-class arm uses. That keeps the pooled and per-class arms drawing from the same
        # organisms rather than two unrelated samples.
        per = n // len(pool_classes) if len(pool_classes) > 1 else n
        tax_pool = []
        for c in sorted(pool_classes):
            recs = sorted(_load(SPLITS / c / "test.jsonl"), key=lambda r: r["accession"])
            tax_pool.extend(recs[:per])                   # deterministic, seedless
        cov = taxonomy.attach(tax_pool)
        print(f"  lineage pool: {cov['n']} records "
              f"({per} from each of {sorted(pool_classes)}), "
              f"{cov['with_lineage']} with a lineage, widths {cov['realised_widths']}",
              flush=True)

    for cls in targets:
        seed_pool = _load(SPLITS / cls / "test.jsonl") if arm.seeded else None
        if arm.seeded and arm.prefix == "taxonomy":
            taxonomy.attach(seed_pool)     # seeded+prefixed: seeds keep their own lineages
        gens = generate(sub, arm, cls, n, cfg, seed_pool=(seed_pool or tax_pool),
                        stage=stage)

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
        top_p=arm.top_p, seeded=arm.seeded, seed_len_nt=arm.seed_len_nt, prefix=arm.prefix,
        weight_state=arm.weight_state, adapter=arm.adapter_path,
        adapter_sha=_sha_of(arm.adapter_path), row_class=row_class or "ALLROWS",
        substrate=sub.id, substrate_family=sub.family, checkpoint=sub.checkpoint,
        scoring_config=antismash.config_hash(), corpus_sha256=_corpus_sha(),
        termination_mode=sub.termination_mode,
        # SPEC 6.5: an arm attached at 4 of 25 sites is not the same arm as one at 32 of
        # 32, and a W3 run at rank 16 is not the one at rank 64. Without these in the
        # REALISED hash both collide on one run directory and the loser is destroyed.
        # ⚠ THE "offset" DEFAULT IS UNREACHABLE AND IS LEFT IN PLACE DELIBERATELY. W3 shared
        # this path until 2026-09-16; with it gone, `intervention` is set only by
        # attach_direction, which always writes intervention_kind ("i1" or "i1_random").
        # Measured across the 48 frozen arms: None 32, i1 65, i1_random 8 -- never "offset".
        # This expression is an input to the REALISED HASH that names every run directory, so
        # it is not edited during cleanup: a default that cannot fire costs nothing, and being
        # wrong about "cannot fire" would rename run dirs and orphan the results.
        intervention_method=((sub.meta or {}).get("intervention_kind", "offset")
                             if intervention is not None else None),
        # ⚠ ALPHA IS PART OF THE REALISED IDENTITY. I1 at alpha 1 and alpha 4 are different
        # arms; without alpha here they collide on one run directory and the second run
        # destroys the first -- the same failure the site/rank fields exist to prevent.
        intervention_alpha=(getattr(intervention, "alpha", None)
                            if intervention is not None else None),
        intervention_direction_class=((sub.meta or {}).get("intervention_direction_class")
                                      if intervention is not None else None),
        # ⚠ WHICH VECTORS. Without these, an I1 arm and its magnitude-matched random
        # control -- same alpha, same sites, same class -- produce the SAME realised hash
        # and land in the same run directory, so the control silently overwrites the arm
        # it exists to be compared against.
        intervention_direction_file=((sub.meta or {}).get("intervention")
                                     if intervention is not None else None),
        intervention_random_seed=((sub.meta or {}).get("intervention_random_seed")
                                  if intervention is not None else None),
        # ⚠ WHICH SITES WERE ACTUALLY STEERED. `intervention_sites` is the site_report and
        # says 4 of 25 for every I1 arm -- but a degenerate site is zeroed, so an arm can
        # steer 3. directions.py cites SPEC 6.5 ("an arm attached at 3 of 4 sites is not the
        # arm attached at 4") as the justification for zeroing, and that was the one
        # requirement the frozen record did not meet.
        intervention_active_sites=((sub.meta or {}).get("intervention_active_sites")
                                   if intervention is not None else None),
        intervention_site_subset=((sub.meta or {}).get("intervention_site_subset")
                                  if intervention is not None else None),
        intervention_degenerate_sites=((sub.meta or {}).get("intervention_degenerate_sites")
                                       if intervention is not None else None),
        # SPEC 6.4: a null is uninformative unless the intervention verifiably landed. The
        # verdict travels with the direction and is recorded, not merely printed at
        # derivation time and then lost.
        intervention_check_passed=((sub.meta or {}).get("intervention_check_passed")
                                   if intervention is not None else None),
        intervention_rank=(getattr(intervention, "rank", None)
                           if intervention is not None else None),
        # DirectionInjection has no trainable parameters and no n_trainable(); calling it
        # unconditionally raised AttributeError for every I1 arm.
        intervention_trainable=(intervention.n_trainable()
                                if intervention is not None
                                and hasattr(intervention, "n_trainable") else None),
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
    # ---- HITS LEDGER -------------------------------------------------------------------
    # One row per DETECTED generation, carrying the organism it was prompted with. The
    # per-generation record is the only place the lineage survives, so without this a
    # positive hit cannot be traced back to the taxon that produced it.
    hits = []
    for target, rows in by_target.items():
        for r in rows:
            if not r.get("detected"):
                continue
            hits.append({
                "arm": arm.arm_id,
                "substrate": sub.id,
                "stage": stage,
                "target_class": target,
                "called_classes": r.get("observed_classes"),
                "on_target": r.get("on_target"),
                "antismash_products": r.get("products"),
                "produced_core_genes": r.get("produced_core_genes"),
                "n_cds": r.get("n_cds"),
                "coding_density": r.get("coding_density"),
                "generation_id": r["generation_id"],
                "seq_len": r.get("seq_len"),
                "hit_eos": r.get("hit_eos"),
                # the organism whose lineage was prompted
                "prefix_kind": r.get("prefix_kind"),
                "prefix_tag": r.get("prefix_tag"),
                "prefix_source_accession": r.get("prefix_source_accession"),
                "prefix_source_genome": r.get("prefix_source_genome"),
                # seeded regime provenance, when present
                "seed_accession": r.get("seed_accession"),
                "novelty_gate": r.get("gate"),
                "containment_worst": r.get("containment_worst"),
                "novel": r.get("novel"),
                "realised_config_hash": rhash,
            })
    with open(d / "hits.jsonl", "w") as fh:
        for h in sorted(hits, key=lambda x: x["generation_id"]):
            fh.write(json.dumps(h) + "\n")
    report["n_hits_recorded"] = len(hits)
    print(f"  hits ledger: {len(hits)} detected generation(s) -> {d / 'hits.jsonl'}",
          flush=True)

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
    # FROZEN, not merely defaulted. Passing a different value is refused unless the run
    # also declares --off-frozen, because batch size changes the frozen hash and therefore
    # which runs are comparable to which. See genconfig.FROZEN["batch_size"].
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"frozen at {GEN_FROZEN['batch_size']}; sized from measured memory "
                         f"(2.08 GB model + 0.429 GB per in-flight sequence at the 8,192 nt "
                         f"budget). A different value requires --off-frozen.")
    ap.add_argument("--off-frozen", action="store_true",
                    help="acknowledge that this run departs from the frozen generation "
                         "config and is NOT comparable to frozen-config runs.")
    ap.add_argument("--train-classes", nargs="+", default=None,
                    help="the classes this arm's WEIGHTS were trained on. Sets the SPEC 3.8 "
                         "per-arm novelty reference. Omit for an untrained arm.")
    ap.add_argument("--row-class", default=None,
                    help="the class this arm's WEIGHTS carry. Set for a per-class adapter: "
                         "it generates once and fills that one row, rather than pretending "
                         "to be conditioned toward five different targets.")
    ap.add_argument("--prefix", choices=["none", "taxonomy"], default="none",
                    help="'taxonomy' prepends the target class's held-out GTDB lineage -- "
                         "Evo2's native pretraining format. It names an organism, never a "
                         "compound class. Recorded in the realised config and in the hash.")
    ap.add_argument("--direction", default=None,
                    help="I1: a .pt from run.derive_directions. Composes with --adapter.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="I1 injection magnitude. Required with --direction; swept by G9 "
                         "against the manipulation check, never the endpoint (SPEC 2.4).")
    ap.add_argument("--sites", nargs="+", type=int, default=None,
                    help="I1: restrict injection to these attention-site indices. The other "
                         "half of G9 (§6: site AND magnitude are swept together); without it "
                         "the site set is an unexamined default of 'all of them'.")
    ap.add_argument("--random-direction", type=int, default=None, metavar="SEED",
                    help="SPEC 6.3 control: magnitude-matched random vectors at the same "
                         "alpha, so anything I1 achieves that this does not is the "
                         "direction doing work rather than the push.")
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--cpus", type=int, default=16)
    args = ap.parse_args()

    sub = load(args.substrate)
    adapter = args.adapter
    if adapter:
        # SPEC 6.4: evaluate at the BEST held-out checkpoint, not the last. Point at an
        # adapter DIRECTORY and this resolves to best/ automatically. SHARED with
        # run.derive_directions via load.resolve_best, so an I1 direction and the arm it
        # steers cannot resolve to different checkpoints.
        adapter = resolve_best(adapter)
        if adapter != args.adapter:
            print(f"using BEST checkpoint {adapter} rather than final", flush=True)
    intervention = None
    # ⚠ ORDER: the weight state is attached FIRST and the direction SECOND, so the two
    # COMPOSE. attach_adapter MERGES the LoRA into the base weights; attaching a direction
    # first would hold hooks on the pre-merge module objects; and I1 is defined as steering
    # the model the arm actually runs, not the base model.
    #
    # ⚠ THESE ARE TWO SEPARATE `if`s, NOT AN if/elif CHAIN. As a chain, --direction and
    # --adapter were mutually exclusive while the help text advertised that they compose:
    # every I1 arm run on a trained weight state silently steered the BASE model and would
    # have reported a null as though steering had been tested on that arm. All 24 steering
    # arms depend on both firing.
    #
    # ⚠ A third branch here loaded a W3 conditioner from a `.pt` adapter, and a guard below
    # refused --direction alongside it because run_arm holds exactly ONE
    # intervention.attached() context. Both went with W3 on 2026-09-16; `intervention` is now
    # set only by attach_direction.
    if adapter:
        sub = attach_adapter(sub, adapter)
        print(f"attached adapter {adapter}", flush=True)
    if args.direction:
        if args.alpha is None:
            raise SystemExit("--direction needs --alpha; there is no default magnitude, "
                             "and an unstated one would be an undeclared free parameter")
        # ⚠ TELL attach_direction WHAT PREFIX THIS ARM GENERATES WITH, so it can refuse a
        # direction derived under a different one. Without this the cross-check reads None
        # and silently passes -- a guard that exists and does nothing.
        sub.meta["prefix_kind"] = args.prefix
        sub, intervention = attach_direction(sub, args.direction, args.alpha,
                                             randomise=args.random_direction,
                                             sites=args.sites)
        print(f"attached {sub.meta['intervention_kind']} alpha={args.alpha} "
              f"from {args.direction}"
              + (f" on top of {adapter}" if adapter else " on the BASE model"), flush=True)
    arm = ArmSpec(arm_id=args.arm,
                  weight_state="base" if args.adapter is None else args.arm,
                  seeded=args.seeded, seed_len_nt=args.seed_len, prefix=args.prefix,
                  adapter_path=args.adapter,
                  # ⚠ I1 IS A CLASS CHANNEL and must say so here. The direction is derived
                  # per target class, so an I1 arm carries the class at generation time.
                  # Left at the "none" default it would fall through to SPEC 6.0's
                  # degenerate collapse and fill every confusion row from ONE distribution
                  # -- reporting a class-conditional arm as though it were unconditioned.
                  inference_control=("steer" if args.direction else "none"))
    # SPEC 12.A7: decoding comes from the SUBSTRATE, not from a shared constant. This must
    # happen before both generation AND report building, so the `realised` block records the
    # values that were actually sampled with rather than the unresolved Nones.
    arm = arm.with_decoding(sub.family)
    print(f"decoding [{sub.family}]: temperature={arm.temperature} top_k={arm.top_k} "
          f"top_p={arm.top_p}", flush=True)
    # class-bearing iff something in the coordinate carries the class
    # the class must enter at GENERATION time for a target to mean anything
    # ⚠ A TAXONOMY PREFIX IS NOT A CLASS CHANNEL. It names an organism, not a compound
    # class, so it does NOT make an arm class-bearing. An earlier version set class_bearing
    # whenever a prefix was present; that broke SPEC 6.0's degenerate collapse, because an
    # arm carrying the class NOWHERE -- not in its weights, not in its input -- must fill
    # every row from ONE distribution rather than generating four times.
    #
    # Taxa do correlate with which BGC classes they carry, so a lineage is not perfectly
    # class-free in an information-theoretic sense. That is deliberately NOT treated as a
    # conditioning channel here: it is a property of the organism distribution, not a label
    # supplied to the model, and designing around it would mean discarding the model's own
    # native input format to chase an effect the benchmark does not measure.
    class_bearing = args.seeded or arm.inference_control != "none"
    # A per-class adapter carries the class in its WEIGHTS. Run without --row-class it
    # would replicate one sample into all five confusion rows and self-divide lift to 1.0,
    # producing a report that is structurally indistinguishable from a real result.
    # ⚠ AN I1 DIRECTION IS KEYED TO ONE CLASS. It is attached once, before the target loop,
    # and nothing about the loop variable reaches the model -- so an I1 arm without
    # --row-class generates the SAME distribution for every row (identical draws, since
    # _run re-seeds from the frozen rng_seed each call) and publishes them as four
    # class-conditional measurements with lift 1.000 by construction. The adapter guard
    # below cannot catch it: that one requires --adapter, and an I1 arm on base weights has
    # none, and requires `not class_bearing`, which steering makes False.
    if args.direction and not args.row_class:
        raise SystemExit(
            "--direction is keyed to a single class, so without --row-class every "
            "confusion row would be the same steered distribution relabelled, and lift "
            "would be 1.000 by construction rather than measured. Pass --row-class "
            "<the direction's target class>.")
    if args.adapter and not args.row_class and not class_bearing and not args.train_classes:
        raise SystemExit(
            "refusing to run: an adapter arm needs either --row-class (the class its "
            "weights carry, for a per-class adapter) or --train-classes (for a pooled "
            "adapter, whose weights carry no single class). Without one, five identical "
            "rows would be published as a confusion matrix."
        )
    # The frozen batch size is used unless the run explicitly declares otherwise. A silent
    # override would change the frozen hash -- and therefore the run directory and the
    # comparability of the result -- without anyone deciding to.
    batch_size = GEN_FROZEN["batch_size"] if args.batch_size is None else args.batch_size
    if batch_size != GEN_FROZEN["batch_size"] and not args.off_frozen:
        raise SystemExit(
            f"refusing to run: --batch-size {batch_size} differs from the frozen "
            f"{GEN_FROZEN['batch_size']}. Batch size is part of the frozen generation "
            f"config, so changing it changes the run hash and makes this run "
            f"non-comparable to every frozen-config arm. Pass --off-frozen to declare "
            f"that deliberately."
        )
    cfg = GenConfig(budget_nt=args.budget_nt, batch_size=batch_size)
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
