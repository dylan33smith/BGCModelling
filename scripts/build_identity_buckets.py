#!/usr/bin/env python
"""[P13-DAT-identitybuckets] Annotate a subclass split with phage-paper identity buckets.

Ports the conditioning scheme of Hie et al. (bioRxiv 2025.09.12.675911v1, Methods B.1.5): every
training record is tagged with its nucleotide identity to a single frozen reference, binned into
five buckets, and that bin becomes an atomic special token prepended after the class token.

Their reference was PhiX174. We have no canonical azole cluster, so the reference is DEFINED here
as the MEDOID of the train split -- a real sequence, reproducible from the split with no judgement.

Pre-registered in docs/phase13_IDENTITY_BUCKET_preregistration.md. The identity formula, the bucket
edges and Gate T0 are fixed there; this script must not silently deviate from them.
"""
import argparse, json, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# Pre-registered bucket edges (prereg 3.3). (token, lo, hi) with lo <= ani < hi, top bucket closed.
BUCKETS = [
    ("[ID_95_100]", 0.95, 1.01),
    ("[ID_80_95]",  0.80, 0.95),
    ("[ID_70_80]",  0.70, 0.80),
    ("[ID_50_70]",  0.50, 0.70),
    ("[ID_00_50]",  0.00, 0.50),
]
T0_MIN_TOP_BUCKET = 30  # prereg 3.4


def bucket_of(ani, edges=BUCKETS):
    for tok, lo, hi in edges:
        if lo <= ani < hi:
            return tok
    raise ValueError(f"unbucketed ani={ani}")


def write_fasta(recs, path):
    with open(path, "w") as fh:
        for i, r in enumerate(recs):
            fh.write(f">r{i}\n{r['sequence']}\n")


def all_vs_all(fasta, tmp, threads):
    """mmseqs easy-search self-vs-self -> {(q,t): ani} using the pre-registered formula."""
    out = Path(tmp) / "hits.tsv"
    cmd = ["mmseqs", "easy-search", str(fasta), str(fasta), str(out), str(Path(tmp) / "ms"),
           "--search-type", "3", "-s", "7.5", "-e", "10", "--max-seqs", "5000",
           "--threads", str(threads), "-v", "1",
           "--format-output", "query,target,fident,alnlen,qlen,tlen"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    best = defaultdict(float)
    with open(out) as fh:
        for line in fh:
            q, t, fident, alnlen, qlen, tlen = line.rstrip("\n").split("\t")
            # prereg 3.2: alignment-weighted identity, so a short local hit cannot pass as a copy
            ani = float(fident) * int(alnlen) / max(int(qlen), int(tlen))
            if ani > best[(q, t)]:
                best[(q, t)] = min(ani, 1.0)
    return best


def proteome_all_vs_all(recs, tmp, threads):
    """[AMENDMENT 2026-08-27] Record-level proteome AAI, {(qi,ti): aai}.

    DNA identity is the WRONG instrument for BGCs across genera -- 76.5% of azole records had zero
    alignable nucleotide identity to the DNA medoid (Gate T0 FAIL, prereg §7). BGC relatedness is
    conventionally measured at the protein level, which is why our own gate carries `protein_aai`
    alongside `containment`.

    `aai_to_ref` = coverage-weighted proteome reconstruction:
        sum over TARGET proteins of (best fident*alnlen/tlen from any query ORF) * tlen
        ------------------------------------------------------------------------------
                                  sum over TARGET proteins of tlen
    i.e. "how much of that record's proteome does this record reproduce, weighted by identity".
    Deliberately NOT `protein_aai` (= max over ORFs, `terms.md`): one shared enzyme is not a
    near-copy, and template fidelity has to be a proteome-coverage statement.
    """
    from novelty_battery import translate_orfs
    prot = translate_orfs([r["sequence"] for r in recs], min_aa=30)
    qf = Path(tmp) / "prot.fa"
    plen = {}
    with qf.open("w") as w:
        for i, ps in enumerate(prot):
            for j, p in enumerate(ps):
                w.write(f">r{i}_{j}\n{p}\n")
                plen[f"r{i}_{j}"] = len(p)
    n_orf = [len(p) for p in prot]
    print(f"[P13-DAT] ORFs (min_aa=30): total {sum(n_orf)}, median/record "
          f"{sorted(n_orf)[len(n_orf)//2]}, records with none: {sum(1 for x in n_orf if x==0)}")
    out = Path(tmp) / "phits.tsv"
    subprocess.run(["mmseqs", "easy-search", str(qf), str(qf), str(out), str(Path(tmp) / "pms"),
                    "-e", "1e-3", "-s", "5.7", "--max-seqs", "20000", "--threads", str(threads),
                    "-v", "1", "--format-output", "query,target,fident,alnlen"],
                   check=True, stdout=subprocess.DEVNULL)
    # best per (query RECORD, target PROTEIN)
    best = defaultdict(float)
    with open(out) as fh:
        for line in fh:
            q, t, fid, aln = line.rstrip("\n").split("\t")
            qi = int(q.split("_")[0][1:])
            cov = float(fid) * int(aln) / plen[t]
            k = (qi, t)
            if cov > best[k]:
                best[k] = min(cov, 1.0)
    tot_len = defaultdict(float)
    for pid, L in plen.items():
        tot_len[int(pid.split("_")[0][1:])] += L
    num = defaultdict(float)
    for (qi, t), v in best.items():
        num[(qi, int(t.split("_")[0][1:]))] += v * plen[t]
    return {k: (num[k] / tot_len[k[1]] if tot_len[k[1]] else 0.0) for k in num}, n_orf


def aai_against_ref(recs, ref_seq, tmp, threads):
    """Same coverage-weighted formula as proteome_all_vs_all, but against ONE frozen reference.

    Used for val/test, which must be bucketed on the reference chosen from TRAIN -- never on a
    reference re-derived from themselves, which would leak the split into its own conditioning.
    """
    from novelty_battery import translate_orfs
    tmp = Path(tmp)
    ref_prots = translate_orfs([ref_seq], min_aa=30)[0]
    if not ref_prots:
        return [0.0] * len(recs)
    rf, plen = tmp / "ref.fa", {}
    with rf.open("w") as w:
        for j, p in enumerate(ref_prots):
            w.write(f">ref_{j}\n{p}\n")
            plen[f"ref_{j}"] = len(p)
    prot = translate_orfs([r["sequence"] for r in recs], min_aa=30)
    qf = tmp / "q.fa"
    with qf.open("w") as w:
        for i, ps in enumerate(prot):
            for j, p in enumerate(ps):
                w.write(f">r{i}_{j}\n{p}\n")
    if qf.stat().st_size == 0:
        return [0.0] * len(recs)
    out = tmp / "refhits.tsv"
    subprocess.run(["mmseqs", "easy-search", str(qf), str(rf), str(out), str(tmp / "rms"),
                    "-e", "1e-3", "-s", "5.7", "--threads", str(threads), "-v", "1",
                    "--format-output", "query,target,fident,alnlen"],
                   check=True, stdout=subprocess.DEVNULL)
    best = defaultdict(float)
    with open(out) as fh:
        for line in fh:
            q, t, fid, aln = line.rstrip("\n").split("\t")
            qi = int(q.split("_")[0][1:])
            cov = min(float(fid) * int(aln) / plen[t], 1.0)
            if cov > best[(qi, t)]:
                best[(qi, t)] = cov
    denom = sum(plen.values())
    num = defaultdict(float)
    for (qi, t), v in best.items():
        num[qi] += v * plen[t]
    return [min(num[i] / denom, 1.0) for i in range(len(recs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True, help="dir with train.jsonl")
    ap.add_argument("--metric", choices=["dna", "protein"], default="dna",
                    help="dna = prereg 3.2 (nucleotide); protein = §7 amendment 2026-08-27")
    ap.add_argument("--out", type=Path, required=True, help="run dir to write into")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quintiles", action="store_true",
                    help="prereg 3.4 fallback: replace fixed edges with observed quintiles")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.split / "train.jsonl")]
    print(f"[P13-DAT] {len(recs)} train records from {args.split}")
    args.out.mkdir(parents=True, exist_ok=True)

    field = "ani_to_ref" if args.metric == "dna" else "aai_to_ref"
    n_orf = None
    with tempfile.TemporaryDirectory() as tmp:
        if args.metric == "dna":
            fa = Path(tmp) / "train.fna"
            write_fasta(recs, fa)
            print("[P13-DAT] all-vs-all NUCLEOTIDE search (medoid selection)...")
            pair = {(int(q[1:]), int(t[1:])): v for (q, t), v in
                    all_vs_all(fa, tmp, args.threads).items()}
        else:
            print("[P13-DAT] all-vs-all PROTEOME search (medoid selection)...")
            pair, n_orf = proteome_all_vs_all(recs, tmp, args.threads)

    # medoid = record maximising summed SYMMETRISED identity to every other record
    tot = defaultdict(float)
    for (q, t), v in pair.items():
        if q != t:
            tot[q] += v
            tot[t] += v
    if not tot:
        sys.exit("[P13-DAT] FATAL: no cross-record hits; cannot define a medoid.")
    ref_i = max(tot, key=tot.get)
    ref = recs[ref_i]
    print(f"[P13-DAT] REF = {ref['accession']}  (idx {ref_i}, summed {tot[ref_i]:.1f}, "
          f"{len(ref['sequence'])} nt)")

    vals = []
    for i in range(len(recs)):
        a = 1.0 if i == ref_i else max(pair.get((i, ref_i), 0.0), pair.get((ref_i, i), 0.0))
        vals.append(min(a, 1.0))

    edges = BUCKETS
    scheme = "phage-paper fixed thresholds"
    if args.quintiles:
        srt = sorted(vals)
        cuts = [srt[int(len(srt) * f)] for f in (0.2, 0.4, 0.6, 0.8)]
        edges = [("[ID_Q5]", cuts[3], 1.01), ("[ID_Q4]", cuts[2], cuts[3]),
                 ("[ID_Q3]", cuts[1], cuts[2]), ("[ID_Q2]", cuts[0], cuts[1]),
                 ("[ID_Q1]", 0.0, cuts[0])]
        scheme = f"observed quintiles, cuts={[round(c,4) for c in cuts]}"
        print(f"[P13-DAT] QUINTILE fallback active: {scheme}")

    hist = defaultdict(int)
    for i, (r, a) in enumerate(zip(recs, vals)):
        tok = bucket_of(a, edges)
        r[field], r["id_bucket"] = round(a, 6), tok
        hist[tok] += 1

    print(f"\n[P13-DAT] {field}: min {min(vals):.4f}  median "
          f"{sorted(vals)[len(vals)//2]:.4f}  max {max(vals):.4f}")
    print("[P13-DAT] BUCKET HISTOGRAM (prereg 3.4 -- published with the results):")
    for tok, lo, hi in edges:
        print(f"    {tok:<14} [{lo:.4f}, {hi:.4f})  n={hist[tok]:4d}  "
              f"{hist[tok]/len(recs):6.1%}")

    top = hist[edges[0][0]]
    t0 = top >= T0_MIN_TOP_BUCKET
    print(f"\n[P13-DAT] GATE T0: top bucket n={top} vs required >={T0_MIN_TOP_BUCKET} "
          f"-> {'PASS' if t0 else 'FAIL'}")
    if not t0 and not args.quintiles:
        print("[P13-DAT] ⛔ T0 FAILED. Per prereg 3.4, re-run with --quintiles and record the "
              "amendment in docs/phase13_IDENTITY_BUCKET_preregistration.md §7.")

    with open(args.out / "train_bucketed.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    # val/test bucketed against the SAME frozen reference chosen from train (no self-reference)
    val_hist = {}
    for name in ("val", "test"):
        src = args.split / f"{name}.jsonl"
        if not src.exists():
            continue
        vrecs = [json.loads(l) for l in open(src)]
        with tempfile.TemporaryDirectory() as tmp:
            vvals = (aai_against_ref(vrecs, ref["sequence"], tmp, args.threads)
                     if args.metric == "protein" else [0.0] * len(vrecs))
        vh = defaultdict(int)
        for r, a in zip(vrecs, vvals):
            tok = bucket_of(a, edges)
            r[field], r["id_bucket"] = round(a, 6), tok
            vh[tok] += 1
        with open(args.out / f"{name}_bucketed.jsonl", "w") as fh:
            for r in vrecs:
                fh.write(json.dumps(r) + "\n")
        val_hist[name] = dict(vh)
        print(f"[P13-DAT] {name}: n={len(vrecs)}  " +
              "  ".join(f"{t}={vh[t]}" for t, _, _ in edges))
    json.dump({"split": str(args.split), "n_train": len(recs), "scheme": scheme,
               "metric": args.metric, "field": field,
               "orfs_per_record_median": (sorted(n_orf)[len(n_orf)//2] if n_orf else None),
               "ref_accession": ref["accession"], "ref_index": ref_i,
               "ref_len_nt": len(ref["sequence"]),
               "id_min": min(vals), "id_median": sorted(vals)[len(vals) // 2],
               "id_max": max(vals),
               "edges": [[t, lo, hi] for t, lo, hi in edges],
               "histogram": dict(hist), "histogram_val_test": val_hist, "gate_T0_pass": t0,
               "T0_min_top_bucket": T0_MIN_TOP_BUCKET},
              open(args.out / "identity_bucket_report.json", "w"), indent=2)
    print(f"[P13-DAT] wrote {args.out}/train_bucketed.jsonl + identity_bucket_report.json")
    sys.exit(0 if t0 else 3)


if __name__ == "__main__":
    main()
