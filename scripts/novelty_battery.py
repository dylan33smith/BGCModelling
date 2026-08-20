#!/usr/bin/env python
"""T3.2 / T3.3 / T6.1 — the novelty tests the old battery did NOT have.

See `docs/phase3_evaluation_battery.md`. Three tests, each closing a hole through which a model
could pass every check we previously ran while having invented nothing.

────────────────────────────────────────────────────────────────────────────────────────────
T3.2  PROTEIN-LEVEL NOVELTY
Our novelty guard compares DNA letter-for-letter (k=21 containment). DNA is redundant: many
letters can change while the encoded protein stays identical. So a model can copy a training
cluster, swap synonymous codons, score as fully novel at the nucleotide level, and have invented
nothing. This translates predicted ORFs and searches them against the proteins of the TRAINING
set, reporting best amino-acid identity (AAI).
Reference: the phage paper (Hie et al., Science 2026) reported AAI as low as 63% to natural
proteins as its novelty evidence — protein identity is the standard this field expects.

T3.3  INTRA-SET DIVERSITY
Every novelty check we own compares generations to TRAINING DATA. None compares generations to
EACH OTHER. A model that emits one good sequence 150 times passes all of them. Mode collapse was
invisible — and it becomes more likely the moment many generations share a seed, which is exactly
what the Phase-3 seeding arms introduce.

T6.1  JOINT PASS RATE
An arm can post 30% on-class and 100% novel while the on-class records are precisely the non-novel
ones. Marginal rates cannot see that; only the per-record intersection can. This is the analogue of
the phage paper's "302 candidates from hundreds of thousands" — the count that survives every
filter AT ONCE, which is the only number describing what could actually be taken forward.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

K = 21


# ─────────────────────── PER-CLASS METRIC POLICY ───────────────────────
def load_class_policy(cls: str) -> dict:
    """Which metrics mean what FOR THIS CLASS (`config/class_eval_policy.yaml`).

    The reporting set is class-agnostic; its interpretation is not. `bio_span_frac` reads 0.997
    on real PKS cores (saturated -- a megasynthase core is one long biosynthetic ORF) and
    `n_class_domains` inflates ~2.7x for PKS because several Pfam models cover one catalytic
    domain. Both are meaningful for RIPP. Prose cannot stop a void number being quoted, so the
    policy is machine-readable and travels inside every scored file.
    """
    import yaml
    path = REPO / "config" / "class_eval_policy.yaml"
    if not path.exists():
        return {}
    pol = yaml.safe_load(path.read_text()) or {}
    return pol.get(cls, {})


def strict_mode() -> bool:
    """Same switch the eval suite uses (`BGC_EVAL_STRICT`, default on)."""
    import os
    return os.environ.get("BGC_EVAL_STRICT", "1").strip().lower() not in ("0", "false", "no", "off")


# ──────────────────────────── shared ────────────────────────────
def kmers(seq: str, k: int = K) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else set()


def containment(q: str, r: str, k: int = K) -> float:
    a, b = kmers(q, k), kmers(r, k)
    return len(a & b) / len(a) if a else 0.0


def translate_orfs(seqs: list[str], min_aa: int = 30) -> list[list[str]]:
    from bgc_pipeline.evaluation import find_orfs
    out = []
    for s in seqs:
        try:
            out.append([o.aa_seq for o in find_orfs(s, min_aa=min_aa)])
        except Exception:
            out.append([])
    return out


# ──────────────────────────── T3.2 ────────────────────────────
def build_protein_db(train_jsonl: Path, out_fasta: Path, limit: int | None = None) -> Path:
    """Translate the TRAINING set's ORFs into one FASTA — the thing generations are novel *against*."""
    if out_fasta.exists() and out_fasta.stat().st_size:
        return out_fasta
    seqs = []
    for i, line in enumerate(train_jsonl.open()):
        if limit and i >= limit:
            break
        seqs.append(json.loads(line)["sequence"])
    print(f"[T3.2] translating {len(seqs):,} training records …", flush=True)
    prot = translate_orfs(seqs)
    n = 0
    with out_fasta.open("w") as w:
        for i, ps in enumerate(prot):
            for j, p in enumerate(ps):
                w.write(f">train_{i}_{j}\n{p}\n")
                n += 1
    print(f"[T3.2] wrote {n:,} training proteins -> {out_fasta}")
    return out_fasta


def protein_novelty(gen_seqs: list[str], db_fasta: Path, env: str = "bgcmodel",
                    threads: int = 16) -> list[float]:
    """Best AAI per GENERATION (max over its ORFs). 0.0 = no protein resembles anything in train."""
    prot = translate_orfs(gen_seqs)
    best = [0.0] * len(gen_seqs)
    with tempfile.TemporaryDirectory() as tmp:
        qf = Path(tmp) / "q.fa"
        with qf.open("w") as w:
            for i, ps in enumerate(prot):
                for j, p in enumerate(ps):
                    w.write(f">gen_{i}_{j}\n{p}\n")
        if qf.stat().st_size == 0:
            return best
        out = Path(tmp) / "hits.tsv"
        r = subprocess.run(
            ["micromamba", "run", "-n", env, "mmseqs", "easy-search",
             str(qf), str(db_fasta), str(out), str(Path(tmp) / "t"),
             "--format-output", "query,target,fident", "-e", "1e-3",
             "--threads", str(threads), "-s", "5.7"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print("[T3.2] mmseqs failed:", r.stderr[-500:])
            return best
        for line in out.open():
            q, _, fid = line.rstrip("\n").split("\t")[:3]
            i = int(q.split("_")[1])
            best[i] = max(best[i], float(fid))
    return best


# ─────────────────────── T3.0  PIPELINE INTEGRITY ───────────────────────
def exact_duplicate_audit(seqs: list[str]) -> dict:
    """EXACT byte-identical records within one generation set.

    This is NOT mode collapse and must never be reported as such. A model sampling at
    temperature 1.0 does not emit the same 8,000-nt sequence twice; byte-identical records
    mean the generation set was ASSEMBLED wrong -- the 2026-08-19 fan-out wrote four copies
    of the same 47 units because `seed_generate.py` takes no shard argument and every worker
    ran with the same --seed. Effective n was 47 while every table said 188.

    Rates survive uniform duplication; n, confidence intervals and p-values do not.
    """
    seen: dict[str, int] = {}
    for s in seqs:
        seen[s] = seen.get(s, 0) + 1
    dup_groups = {k: v for k, v in seen.items() if v > 1}
    return {
        "n": len(seqs),
        "n_unique": len(seen),
        "n_exact_duplicate_records": len(seqs) - len(seen),
        "largest_duplicate_group": max(dup_groups.values(), default=1),
        "effective_n": len(seen),
    }


# ──────────────────────────── T3.3 ────────────────────────────
def intra_set_diversity(seqs: list[str], thresh: float = 0.80) -> dict:
    """All-vs-all containment WITHIN one arm's output. Catches mode collapse."""
    ks = [kmers(s) for s in seqs]
    n = len(seqs)
    pair, dup = [], [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not ks[i] or not ks[j]:
                continue
            c = len(ks[i] & ks[j]) / min(len(ks[i]), len(ks[j]))
            pair.append(c)
            if c >= thresh:
                dup[i] = dup[j] = True
    # greedy distinct-cluster count at the same threshold
    reps: list[set] = []
    for kk in ks:
        if not kk:
            continue
        if not any(len(kk & r) / min(len(kk), len(r)) >= thresh for r in reps):
            reps.append(kk)
    return {
        "n": n,
        "median_pairwise_containment": st.median(pair) if pair else 0.0,
        "max_pairwise_containment": max(pair) if pair else 0.0,
        "n_distinct_clusters": len(reps),
        "frac_distinct": len(reps) / n if n else 0.0,
        "frac_with_a_near_duplicate": sum(dup) / n if n else 0.0,
    }


# ──────────────────────────── T6.1 ────────────────────────────
def joint_pass(on_class: list[bool], nt_containment: list[float], aai: list[float],
               distinct: list[bool], nt_thresh: float = 0.80,
               aai_thresh: float = 0.95) -> dict:
    """The per-record INTERSECTION. Marginal rates cannot detect an arm whose on-class records
    are exactly its non-novel ones."""
    n = len(on_class)
    rows = [(oc, c < nt_thresh, a < aai_thresh, d)
            for oc, c, a, d in zip(on_class, nt_containment, aai, distinct)]
    both = sum(1 for r in rows if all(r))
    return {
        "n": n,
        "on_class": sum(1 for r in rows if r[0]),
        "nt_novel": sum(1 for r in rows if r[1]),
        "protein_novel": sum(1 for r in rows if r[2]),
        "distinct": sum(1 for r in rows if r[3]),
        "JOINT_PASS": both,
        "joint_rate": both / n if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True, help="generations jsonl")
    ap.add_argument("--train", type=Path, required=True, help="the class TRAIN jsonl")
    ap.add_argument("--cls", default="RIPP")
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--db-fasta", type=Path,
                    default=Path("/data2/ds85/bgcmodel_data/splits_class/RIPP/train_proteins.fa"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gens = [json.loads(l).get("sequence", "")[: args.window] for l in args.gen.open()]
    gens = [g for g in gens if g]
    train = [json.loads(l)["sequence"] for l in args.train.open()]
    print(f"[battery] {len(gens)} generations, {len(train):,} training records, "
          f"window {args.window} nt\n")

    # on-class + nucleotide containment (existing instruments, re-used not reimplemented)
    from concurrent.futures import ProcessPoolExecutor

    from ladder_audit import one
    from bgc_pipeline.evaluation import OBLIGATE_DOMAINS
    with ProcessPoolExecutor(max_workers=24) as ex:
        scored = list(ex.map(one, [("b", g, args.cls, i) for i, g in enumerate(gens)]))

    # ON-CLASS IS CLASS-SPECIFIC. `one()["bio"]` is a bitscore against the GLOBAL ~91-model
    # biosynthetic set and ignores `cls` entirely, so `bio > 0` means "any biosynthetic domain",
    # NOT "a domain of this class". Reading it as on-class inverted the Phase-3 A0 result on
    # 2026-08-14 (the all-class adapter scored 0.080 generic but 0.000 RIPP-specific) -- the
    # conclusion flipped from "the specialist lost" to "the specialist is the only arm on target".
    # See bugs.md. --cls now actually gates the metric.
    marker_accs = set(OBLIGATE_DOMAINS.get(args.cls) or [])
    if not marker_accs:
        raise SystemExit(
            f"[battery] FATAL: OBLIGATE_DOMAINS has no markers for cls={args.cls!r}. "
            f"A class with no marker set cannot produce an on-class rate -- it would silently "
            f"read 0.000 for every arm. Known classes: {sorted(OBLIGATE_DOMAINS)}")
    on_class = [bool(set(s["bio_accs"]) & marker_accs) for s in scored]
    on_class_generic = [s["bio"] > 0 for s in scored]
    tk = [kmers(t) for t in train]
    nt_cont = []
    for g in gens:
        kg = kmers(g)
        nt_cont.append(max((len(kg & t) / len(kg) for t in tk if kg), default=0.0))

    db = build_protein_db(args.train, args.db_fasta)
    aai = protein_novelty(gens, db)
    policy = load_class_policy(args.cls)
    void = {k: v.get("reason", "") for k, v in (policy.get("metrics") or {}).items()
            if isinstance(v, dict) and v.get("status") == "void"}
    if policy:
        want_w = policy.get("window_nt")
        if want_w and want_w != args.window:
            print(f"  ⚠️ window {args.window} nt is NOT the registered window for {args.cls} "
                  f"({want_w} nt, config/class_eval_policy.yaml). These numbers are not "
                  f"comparable to that class's ceiling.")
    dupe = exact_duplicate_audit(gens)
    if dupe["n_exact_duplicate_records"]:
        msg = (f"[battery] EXACT-DUPLICATE RECORDS: {dupe['n_exact_duplicate_records']} of "
               f"{dupe['n']} records are byte-identical copies; effective n is "
               f"{dupe['effective_n']}, not {dupe['n']} (largest group "
               f"{dupe['largest_duplicate_group']}x). This is a GENERATION-PIPELINE BUG, not "
               f"mode collapse -- see bugs.md, fan-out shard collision. Every rate computed "
               f"here carries the WRONG n.")
        if strict_mode():
            raise SystemExit(msg + "\n[battery] refusing to emit a scored file with a false n. "
                             "Deduplicate the generation set, or set BGC_EVAL_STRICT=0 to "
                             "score it anyway with effective_n recorded.")
        print("  ⚠️ " + msg)
    div = intra_set_diversity(gens)

    # per-record distinctness for the joint test
    ks = [kmers(g) for g in gens]
    distinct = []
    for i, ki in enumerate(ks):
        near = any(ki and kj and len(ki & kj) / min(len(ki), len(kj)) >= 0.80
                   for j, kj in enumerate(ks) if i != j)
        distinct.append(not near)

    jp = joint_pass(on_class, nt_cont, aai, distinct)

    # LADDER SECONDARY OUTCOMES (terms.md; preregistration 3 -- reported always, decisive never).
    # A0 showed all four hits carrying ONE domain where real cores average 1.45, so a hit rate
    # alone cannot distinguish "made a cluster" from "made one enzyme". These measure that.
    import statistics as _st
    def _agg(key):
        v = [s[key] for s in scored]
        hv = [s[key] for s, oc in zip(scored, on_class) if oc]      # among ON-CLASS records only
        return {"mean": round(_st.mean(v), 4) if v else 0.0,
                "median": round(_st.median(v), 4) if v else 0.0,
                "mean_among_on_class": round(_st.mean(hv), 4) if hv else None,
                "n_on_class": len(hv)}
    ladder = {k: _agg(k) for k in
              ("n_orfs", "max_orf_aa", "any", "bio", "frac",
               "n_bio_orfs", "n_bio_domains", "bio_span_frac", "co_orient")}
    # class-specific domain COUNT (n_bio_domains is global); the cluster question, on-class
    n_class_domains = [len(set(s["bio_accs"]) & marker_accs) for s in scored]
    ladder["n_class_domains"] = {
        "mean_among_on_class": round(_st.mean([n for n, oc in zip(n_class_domains, on_class) if oc]), 4)
        if any(on_class) else None,
        "distribution": {str(k): n_class_domains.count(k) for k in sorted(set(n_class_domains))},
        "n_with_ge2": sum(1 for n in n_class_domains if n >= 2)}

    print("=" * 74)
    print(f"NOVELTY BATTERY — {args.gen.name}")
    print("=" * 74)
    _gen_k = sum(on_class_generic)
    print(f"T1.1 on_class_rate         {jp['on_class']}/{jp['n']} = {jp['on_class']/jp['n']:.3f}   "
          f"[{args.cls}-specific, {len(marker_accs)} accessions]")
    print(f"     (generic, ANY biosynthetic domain: {_gen_k}/{jp['n']} = {_gen_k/jp['n']:.3f} — NOT the endpoint)")
    print(f"T3.1 nucleotide novelty    max containment {max(nt_cont):.3f}  "
          f"median {st.median(nt_cont):.3f}   (FAIL >=0.95, WARN >=0.80)")
    print(f"T3.2 protein novelty       median best AAI {st.median(aai):.3f}  "
          f"max {max(aai):.3f}   ({sum(1 for a in aai if a>=0.98)} records >=0.98 = paraphrase)")
    print(f"T3.3 intra-set diversity   {div['n_distinct_clusters']}/{div['n']} distinct "
          f"({div['frac_distinct']:.2f})  median pairwise {div['median_pairwise_containment']:.3f}")
    print()
    print(f"T6.1 JOINT PASS            on-class {jp['on_class']}  nt-novel {jp['nt_novel']}  "
          f"protein-novel {jp['protein_novel']}  distinct {jp['distinct']}")
    print(f"     -> ALL FOUR AT ONCE:  {jp['JOINT_PASS']}/{jp['n']} = {jp['joint_rate']:.3f}")
    print()
    print("LADDER SECONDARY OUTCOMES  (reported always, decisive never)")
    print(f"  {'metric':<16}{'all records':>16}{'among on-class':>18}   what it measures")
    _what = {
        "n_class_domains": "DISTINCT RIPP markers -- >1 = a cluster, not one enzyme",
        "n_bio_domains":   "total biosynthetic domain hits (AUROC 0.919)",
        "n_bio_orfs":      "distinct ORFs carrying a biosynthetic domain",
        "bio_span_frac":   "how far apart they sit = IS IT A CLUSTER (AUROC 0.896)",
        "frac":            "biosynthetic_fraction -- specificity of the protein written",
        "co_orient":       "share on the majority strand (real cores median 1.000)",
        "n_orfs":          "genes called at all",
        "max_orf_aa":      "longest ORF (DEMOTED -- structural only, not capability)",
    }
    for k in ("n_class_domains", "n_bio_domains", "n_bio_orfs", "bio_span_frac",
              "frac", "co_orient", "n_orfs", "max_orf_aa"):
        d = ladder[k]
        allv = f"{d['mean']:.3f}" if "mean" in d else "--"
        onc = d.get("mean_among_on_class")
        onc = f"{onc:.3f}" if onc is not None else "n/a"
        print(f"  {k:<16}{allv:>16}{onc:>18}   {_what[k]}")
    if void:
        print()
        for k, why in void.items():
            print(f"  ⛔ VOID FOR {args.cls}: {k} — {' '.join(why.split())[:150]}")
        print(f"  ⛔ Do not quote the metric(s) above for {args.cls}. "
              f"See config/class_eval_policy.yaml.")
    d2 = ladder["n_class_domains"]
    # The real-core reference is PER CLASS -- a hardcoded RIPP number printed under a PKS report
    # is exactly the cross-class contamination config/class_eval_policy.yaml exists to prevent.
    _ref = {"RIPP": "8/50 = 0.160", "PKS": "42/50 = 0.840 by Pfam ACCESSION, but only 0.300 by "
                                           "catalytic unit and 0.060 by distinct ORF -- VOID, see policy",
            "TERPENE": "8/50 = 0.160"}.get(args.cls, "not measured for this class")
    print(f"  -> records with >=2 distinct {args.cls} markers: {d2['n_with_ge2']}/{jp['n']}"
          f"   (real {args.cls} cores: {_ref})")
    print()
    if jp["on_class"] and jp["JOINT_PASS"] < jp["on_class"]:
        print(f"  ⚠️ {jp['on_class'] - jp['JOINT_PASS']} on-class record(s) fail a novelty or")
        print("     diversity gate. Marginal rates would have hidden this.")
    # SCORING STAMP. A rate whose scoring config is unstated is not a result: the SAME key
    # `on_class` previously held both the global and the class-specific number, in files whose
    # names differed only by window. Every output now carries the config that produced it, and
    # the class goes in the FILENAME as well as the payload.
    res = {"scoring": {"metric": "best_bio_bits",
                       "endpoint": f"best_bio_bits > 0 @ OBLIGATE_DOMAINS[{args.cls}]",
                       "cls": args.cls,
                       "marker_accessions": sorted(marker_accs),
                       "n_marker_accessions": len(marker_accs),
                       "window_nt": args.window,
                       "gen_set": args.gen.name,
                       "train_set": str(args.train),
                       "n": len(gens),
                       "effective_n": dupe["effective_n"]},
           "ladder": ladder,
           "on_class": on_class, "on_class_generic": on_class_generic,
           "nt_containment": nt_cont, "aai": aai,
           "distinct": distinct, "diversity": div, "joint": jp,
           "integrity": dupe,
           "class_policy": {"loaded": bool(policy),
                            "registered_window_nt": policy.get("window_nt"),
                            "void_metrics": sorted(void)}}
    if args.out:
        out = args.out
        if args.cls.lower() not in out.stem.lower():
            out = out.with_name(f"{out.stem}_{args.cls}{out.suffix}")
            print(f"[battery] scoring set not in the filename — writing {out.name} instead")
        out.write_text(json.dumps(res, indent=1))
        print(f"[battery] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
