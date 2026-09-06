#!/usr/bin/env python
"""[P13-EVL-exchangerate] Assemble the fidelity-vs-novelty exchange rate across identity buckets.

Emits the Phase-3 reporting set with METRICS AS ROWS and ARMS AS COLUMNS (CLAUDE.md), every row
printed even when it did not move, and BOTH own-subclass denominators labelled -- the P11 entry
records a retraction caused by mixing "rate among detections" with "rate over all records".
"""
import json, math, sys
from pathlib import Path

RUN = Path("/data2/ds85/bgcmodel_runs/phase13_AZOLE_IDBUCKET")
P10 = Path("/data2/ds85/bgcmodel_runs/phase10_AZOLE_CONTAINING_RIPP")
OWN = "azole-containing-RiPP"
ARMS = [("p13_id95_100", "[ID_95_100]", 40), ("p13_id80_95", "[ID_80_95]", 115),
        ("p13_id70_80", "[ID_70_80]", 14), ("p13_id50_70", "[ID_50_70]", 2),
        ("p13_id00_50", "[ID_00_50]", 623), ("p13_nobucket", "none", None)]


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    def lc(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    n = a + b + c + d
    p0 = lc(a + b, a) + lc(c + d, c) - lc(n, a + c)
    tot = 0.0
    for i in range(max(0, a + c - (c + d)), min(a + b, a + c) + 1):
        p = lc(a + b, i) + lc(c + d, a + c - i) - lc(n, a + c)
        if p <= p0 + 1e-9:
            tot += math.exp(p)
    return min(1.0, tot)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def read_as(tsv):
    """-> (n_ran, detections, own_hits, [products...], n_rows)

    n_ran counts records antiSMASH actually processed; n_rows counts records submitted. They differ
    (P10: 1000 submitted, 998 ran) and a record that failed to run is NOT a success -- so rates are
    denominated on records GENERATED, with the shortfall printed as its own row.
    """
    if not Path(tsv).exists():
        return None
    ran = det = own = rows = 0
    prods = []
    for i, line in enumerate(open(tsv)):
        if i == 0:
            continue
        f = line.rstrip("\n").split("\t")
        rows += 1
        ran += int(f[2])
        if int(f[3]):
            det += 1
            ps = f[5] if len(f) > 5 else ""
            prods.append(ps)
            if OWN in ps:
                own += 1
    return ran, det, own, prods, rows


def read_bat(p):
    return json.load(open(p)) if Path(p).exists() else None


def lens(jsonl):
    if not Path(jsonl).exists():
        return None
    L = sorted(len(json.loads(l)["sequence"]) for l in open(jsonl))
    return L[len(L) // 2] if L else 0


def main():
    cols, missing = [], []
    for tag, tok, ntr in ARMS:
        a = read_as(RUN / f"as_{tag}_ml200.tsv")
        b = read_bat(RUN / f"{tag}_full_RIPP.json")
        if a is None or b is None:
            missing.append(tag)
            continue
        cols.append(dict(tag=tag, tok=tok, ntrain=ntr, ran=a[0], det=a[1], own=a[2],
                         prods=a[3], nrows=a[4], bat=b, med_nt=lens(RUN / f"{tag}.jsonl")))
    # comparison partner: [P10-TRN-azole], n=1000.
    # ⚠️ P10 split antiSMASH into pos/neg TSVs (158 Pfam-positive + 842 Pfam-negative = 1000).
    # Reading only `pos` would drop 842 records and inflate every P10 rate ~6x. Combine both.
    _p, _n = read_as(P10 / "as_azole_pos_ml200.tsv"), read_as(P10 / "as_azole_neg_ml200.tsv")
    pa = (tuple(_p[i] + _n[i] for i in range(3)) + (_p[3] + _n[3], _p[4] + _n[4])
          if (_p and _n) else None)
    if pa and pa[4] != 1000:
        print(f"⛔ P10 partner: combined rows={pa[4]}, expected 1000 — refusing to compare.")
        sys.exit(3)
    p10b = read_bat(P10 / "azole_denovo_full_RIPP.json")
    if missing:
        print(f"⛔ INCOMPLETE — no scores for: {missing}. Refusing to print a partial table.")
        sys.exit(3)

    W = 13
    def row(label, vals, note=""):
        print(f"  {label:<26}" + "".join(f"{v:>{W}}" for v in vals) + (f"   {note}" if note else ""))

    print("\n" + "=" * 118)
    print("[P13-EVL-exchangerate]  AZOLE identity-bucket sweep — metrics are ROWS, arms are COLUMNS")
    print("=" * 118)
    print(f"  {'':<26}" + "".join(f"{c['tok']:>{W}}" for c in cols))
    row("train records behind tok", [c["ntrain"] if c["ntrain"] is not None else "—" for c in cols])
    row("n generated", [c["bat"]["scoring"]["n"] for c in cols])
    print("  " + "-" * 114)
    print("  PRIMARY — Stage A, denominator = ALL generated records")
    row("own_subclass_rate_all*", [f"{c['own']}/{c['bat']['scoring']['n']}={c['own']/c['bat']['scoring']['n']:.3f}" for c in cols])
    row("  95% CI (Wilson)", [f"[{wilson(c['own'],c['bat']['scoring']['n'])[0]:.3f},{wilson(c['own'],c['bat']['scoring']['n'])[1]:.3f}]" for c in cols])
    row("antismash_detection_rate", [f"{c['det']}/{c['bat']['scoring']['n']}={c['det']/c['bat']['scoring']['n']:.3f}" for c in cols])
    row("  antismash_ran", [f"{c['ran']}/{c['nrows']}" + ("" if c['ran']==c['nrows'] else " ⚠") for c in cols])
    row("own_subclass|detected", [f"{c['own']}/{c['det']}={c['own']/c['det']:.3f}" if c["det"] else "0/0=n/a" for c in cols])
    print("  " + "-" * 114)
    print("  NOVELTY GATES (absolute; * = gate)")
    row("containment max*", [f"{max(c['bat']['nt_containment']):.3f}" for c in cols])
    row("containment n>=0.95 FAIL*", [f"{sum(1 for x in c['bat']['nt_containment'] if x>=0.95)}" for c in cols])
    row("containment n>=0.80 WARN*", [f"{sum(1 for x in c['bat']['nt_containment'] if x>=0.80)}" for c in cols])
    row("protein_aai max*", [f"{max(c['bat']['aai']):.3f}" for c in cols])
    row("protein_aai n>=0.98*", [f"{sum(1 for x in c['bat']['aai'] if x>=0.98)}" for c in cols])
    row("distinct*", [f"{sum(c['bat']['distinct'])}/{c['bat']['scoring']['n']}" for c in cols])
    row("JOINT_PASS", [f"{c['bat']['joint']['JOINT_PASS']}/{c['bat']['scoring']['n']}={c['bat']['joint']['JOINT_PASS']/c['bat']['scoring']['n']:.3f}" for c in cols])
    print("  " + "-" * 114)
    print("  CLUSTER STRUCTURE (Stage B — among Pfam on-class positives)")
    for k in ("n_bio_orfs", "n_bio_domains", "bio_span_frac", "n_orfs", "co_orient", "frac"):
        row(k, [f"{c['bat']['ladder'][k].get('mean_among_on_class', float('nan')):.3f}" for c in cols])
    row("  n_on_class (Pfam)", [f"{c['bat']['ladder']['n_orfs']['n_on_class']}" for c in cols])
    print("  " + "-" * 114)
    print("  CONTEXT")
    row("median length nt", [f"{c['med_nt']}" for c in cols])
    row("  fidelity vs 6,293", [f"{c['med_nt']/6293:.2f}x" for c in cols])
    print("=" * 118)

    if pa and p10b:
        n10 = p10b["scoring"]["n"]
        print(f"\n  COMPARISON PARTNER [P10-TRN-azole] (n={n10}, no bucket token):")
        print(f"    own_subclass_rate_all       {pa[2]}/{n10} = {pa[2]/n10:.4f}")
        print(f"    antismash_detection_rate    {pa[1]}/{n10} = {pa[1]/n10:.4f}"
              f"   (antismash ran {pa[0]}/{pa[4]})")
        print(f"    own_subclass|detected       {pa[2]}/{pa[1]} = {pa[2]/pa[1]:.4f}")
        print(f"    containment max             {max(p10b['nt_containment']):.3f}   "
              f">=0.95 FAIL {sum(1 for x in p10b['nt_containment'] if x>=0.95)}")
        print(f"    protein_aai max             {max(p10b['aai']):.3f}")
        print("\n  FISHER EXACT vs [P10], own_subclass over ALL records (the matched denominator):")
        for c in cols:
            n = c["bat"]["scoring"]["n"]
            p = fisher(c["own"], n - c["own"], pa[2], n10 - pa[2])
            print(f"    {c['tok']:<13} {c['own']}/{n} vs {pa[2]}/{n10}   p = {p:.3g}"
                  + ("   ***" if p < 0.05 else ""))

    print("\n  PRODUCTS SEEN (what it made when it was detected):")
    for c in cols:
        from collections import Counter
        cc = Counter(p for ps in c["prods"] for p in ps.split(";") if p)
        print(f"    {c['tok']:<13} " + (", ".join(f"{k}={v}" for k, v in cc.most_common(6)) or "(no detections)"))


if __name__ == "__main__":
    main()
