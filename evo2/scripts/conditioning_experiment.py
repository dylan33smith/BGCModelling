#!/usr/bin/env python3
"""Controlled conditioning experiments — does conditioning CAUSE the property?

Two experiments on a trained checkpoint (the audit's missing causal controls):

CLASS control (audit M2/M9): generate under each candidate class (same taxon,
matched seeds) and recover each generation's class. If conditioning is causal, the
requested class is recovered, and the requested-vs-recovered confusion diagonal
beats the majority-class prior (and, ideally, base Evo2). Class recovery uses the
likelihood classifier (always available; the fine-tuned model scores the sequence
under every class prefix) and, if installed, antiSMASH (independent gold standard).

TAXON control (E. coli generalization, audit C4/C5): generate the SAME class under
a target chassis taxon (E. coli) vs a native source taxon (matched seeds), then
compare GC / CAI (metric 7). If taxon conditioning generalizes to the unseen
(E.coli, class) cell, E.coli-tagged output is measurably more E.-coli-like (lower
GC, higher E. coli CAI) than source-tagged. Otherwise the structural prior
dominates and the chassis tag does not steer — answered with evidence, not assumed.

Generation needs a GPU + checkpoint. Plan-building + aggregation (confusion matrix,
GC/CAI deltas, steering verdict) are pure and unit-tested (tests/test_conditioning_experiment.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parents[1] / "src"))

# Default lineage tags (full lineage so the prefix matches training format).
ECOLI_TAXON = ("|D__BACTERIA;P__PSEUDOMONADOTA;C__GAMMAPROTEOBACTERIA;"
               "O__ENTEROBACTERALES;F__ENTEROBACTERIACEAE;G__ESCHERICHIA;S__ESCHERICHIA_COLI")
STREPTO_TAXON = ("|D__BACTERIA;P__ACTINOMYCETOTA;C__ACTINOMYCETES;"
                 "O__KITASATOSPORALES;F__STREPTOMYCETACEAE;G__STREPTOMYCES")
ECOLI_GC = 0.508  # E. coli K-12 genome GC


# ── Pure-logic: plan building ───────────────────────────────────────────────

def build_class_plan(taxa: list[str], classes: list[str], n_per: int) -> list[dict]:
    """One generation request per (taxon, class, rep). The 'requested_class' is the
    conditioning; recovery checks whether it is what comes out."""
    plan = []
    for taxon in taxa:
        for cls in classes:
            for rep in range(n_per):
                plan.append({"experiment": "class", "requested_class": cls,
                             "taxon": taxon, "rep": rep,
                             "id": f"class_{cls}_{_taxon_key(taxon)}_{rep}"})
    return plan


def build_taxon_plan(classes: list[str], target_taxon: str, source_taxon: str,
                     n_per: int) -> list[dict]:
    """Matched pairs: the same class+rep generated under target vs source taxon."""
    plan = []
    for cls in classes:
        for rep in range(n_per):
            for label, taxon in (("target", target_taxon), ("source", source_taxon)):
                plan.append({"experiment": "taxon", "requested_class": cls,
                             "taxon": taxon, "taxon_label": label, "rep": rep,
                             "id": f"taxon_{cls}_{label}_{rep}"})
    return plan


def _taxon_key(taxon: str) -> str:
    for part in taxon.replace("|", ";").split(";"):
        if part.startswith("G__"):
            return part[3:].lower()[:12]
    return "tax"


# ── Pure-logic: aggregation ─────────────────────────────────────────────────

def class_confusion(rows: list[dict], classes: list[str]) -> dict[str, Any]:
    """rows: [{requested_class, recovered_class}]. Confusion + diagonal accuracy vs
    the majority-class prior (the bar a non-conditioned model would hit)."""
    rows = [r for r in rows if r.get("recovered_class")]
    n = len(rows)
    conf: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for r in rows:
        conf[r["requested_class"]][r["recovered_class"]] += 1
        correct += int(r["requested_class"] == r["recovered_class"])
    rec_counts = Counter(r["recovered_class"] for r in rows)
    majority_prior = (max(rec_counts.values()) / n) if n else 0.0
    diag = correct / n if n else 0.0
    per_class = {}
    for c in classes:
        tot = sum(conf[c].values())
        per_class[c] = round(conf[c][c] / tot, 3) if tot else None
    return {"n": n, "diagonal_accuracy": round(diag, 3),
            "majority_class_prior": round(majority_prior, 3),
            "beats_prior": diag > majority_prior,
            "per_class_recall": per_class,
            "confusion": {r: dict(conf[r]) for r in sorted(conf)}}


def taxon_effect(rows: list[dict], ecoli_gc: float = ECOLI_GC,
                 gc_margin: float = 0.02) -> dict[str, Any]:
    """rows: [{requested_class, taxon_label('target'|'source'), gc, cai_ecoli}].
    Per class, does target(E.coli) output sit closer to E. coli GC than source, and
    have higher E. coli CAI? 'steers' if target GC is closer to ECOLI_GC by margin."""
    by = defaultdict(lambda: {"target": [], "source": []})
    for r in rows:
        if r.get("gc") is not None and r.get("taxon_label") in ("target", "source"):
            by[r["requested_class"]][r["taxon_label"]].append(r)

    def _mean(xs, key):
        vals = [x[key] for x in xs if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    per_class, steer_votes = {}, 0
    for cls, d in by.items():
        if not d["target"] or not d["source"]:
            continue
        gt, gs = _mean(d["target"], "gc"), _mean(d["source"], "gc")
        ct, cs = _mean(d["target"], "cai_ecoli"), _mean(d["source"], "cai_ecoli")
        closer = (gt is not None and gs is not None
                  and abs(gt - ecoli_gc) < abs(gs - ecoli_gc) - gc_margin)
        per_class[cls] = {
            "target_gc": round(gt, 4) if gt is not None else None,
            "source_gc": round(gs, 4) if gs is not None else None,
            "target_cai_ecoli": round(ct, 4) if ct is not None else None,
            "source_cai_ecoli": round(cs, 4) if cs is not None else None,
            "ecoli_tag_steers_gc_toward_ecoli": closer,
        }
        steer_votes += int(closer)
    return {"ecoli_gc_reference": ecoli_gc, "per_class": per_class,
            "n_classes_steered": steer_votes, "n_classes": len(per_class),
            "verdict": ("steers" if per_class and steer_votes >= max(1, len(per_class) // 2)
                        else "does_not_steer" if per_class else "no_data")}


# ── GPU generation + scoring (thin; reuses generate_bgc + evaluation) ────────

def _run(plan: list[dict], adapter: Path, classes: list[str], args) -> list[dict]:
    import generate_bgc as G
    from evo2_inference import load_evo2_wrapper_for_inference, sequence_loglik
    from bgc_pipeline.evaluation import (
        check_taxon_faithfulness, load_taxon_profiles, resolve_taxon_profile,
        ECOLI_PROFILE,
    )
    taxon_profiles = (load_taxon_profiles(args.taxon_profiles)
                      if args.taxon_profiles and Path(args.taxon_profiles).exists() else {})
    print(f"Loading model (adapter={adapter})...", file=sys.stderr, flush=True)
    wrapper = load_evo2_wrapper_for_inference(adapter, device=args.device)

    out = []
    for i, req in enumerate(plan):
        prefix = G.build_prefix(req["requested_class"], req["taxon"])
        gen = wrapper.generate(prompt_seqs=[prefix], n_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k,
                               top_p=args.top_p, cached_generation=True, verbose=0)
        seq = G.extract_sequence(G._gen_sequences(gen)[0])["sequence"]
        rec = dict(req); rec["gen_length"] = len(seq)
        if len(seq) < args.k:  # nothing usable
            rec["recovered_class"] = None
            out.append(rec); continue

        if req["experiment"] == "class":
            # likelihood class recovery: score the generation under each class prefix
            best_c, best_lp = None, float("-inf")
            for c in classes:
                r = sequence_loglik(wrapper.model, wrapper.tokenizer,
                                    G.build_prefix(c, req["taxon"]), seq,
                                    args.max_seq_len, device=args.device,
                                    score_len=args.score_len)
                if r is not None:
                    lp = r[0] / max(r[1], 1)
                    if lp > best_lp:
                        best_lp, best_c = lp, c
            rec["recovered_class"] = best_c
        else:  # taxon control -> taxon_faithfulness (GC + CAI vs E. coli)
            m7 = check_taxon_faithfulness(
                seq, taxon_profile=resolve_taxon_profile(req["taxon"], taxon_profiles),
                chassis_profile=ECOLI_PROFILE)
            ce = m7.get("chassis_expressibility", {})
            rec["gc"] = ce.get("gc_content")
            rec["cai_ecoli"] = ce.get("cai")
        out.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(plan)} generated+scored", file=sys.stderr, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True, help="LoRA checkpoint dir.")
    ap.add_argument("--experiment", choices=["class", "taxon", "both"], default="both")
    ap.add_argument("--classes", nargs="+",
                    default=["NRPS", "PKS", "TERPENE", "RIPP", "SIDEROPHORE", "OTHER"])
    ap.add_argument("--n-per", type=int, default=4)
    ap.add_argument("--class-taxa", nargs="+", default=[STREPTO_TAXON],
                    help="Taxa used in the CLASS control (held fixed while class varies).")
    ap.add_argument("--target-taxon", default=ECOLI_TAXON)
    ap.add_argument("--source-taxon", default=STREPTO_TAXON)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--score-len", type=int, default=8192)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--taxon-profiles", default="data/processed/taxon_profiles.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=Path("conditioning_experiment.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = []
    if args.experiment in ("class", "both"):
        plan += build_class_plan(args.class_taxa, args.classes, args.n_per)
    if args.experiment in ("taxon", "both"):
        plan += build_taxon_plan(args.classes, args.target_taxon, args.source_taxon, args.n_per)

    print(f"Experiment plan: {len(plan)} generations "
          f"(class={sum(1 for p in plan if p['experiment']=='class')}, "
          f"taxon={sum(1 for p in plan if p['experiment']=='taxon')})", file=sys.stderr)
    if args.dry_run:
        for p in plan[:8]:
            print(f"  {p['id']}: class={p['requested_class']} taxon={p['taxon'][:45]}...",
                  file=sys.stderr)
        print("[dry-run] model not loaded.", file=sys.stderr)
        return

    rows = _run(plan, args.adapter, args.classes, args)
    report: dict[str, Any] = {"adapter": str(args.adapter), "n_generations": len(rows)}
    cls_rows = [r for r in rows if r["experiment"] == "class"]
    tax_rows = [r for r in rows if r["experiment"] == "taxon"]
    if cls_rows:
        report["class_control"] = class_confusion(cls_rows, args.classes)
    if tax_rows:
        report["taxon_control"] = taxon_effect(tax_rows)
    report["rows"] = rows
    args.output.write_text(json.dumps(report, indent=2))

    print(file=sys.stderr)
    if "class_control" in report:
        c = report["class_control"]
        print(f"CLASS control: diagonal={c['diagonal_accuracy']} vs prior="
              f"{c['majority_class_prior']} -> beats_prior={c['beats_prior']}", file=sys.stderr)
    if "taxon_control" in report:
        t = report["taxon_control"]
        print(f"TAXON control (E. coli generalization): {t['verdict']} "
              f"({t['n_classes_steered']}/{t['n_classes']} classes steered)", file=sys.stderr)
    print(f"  wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
