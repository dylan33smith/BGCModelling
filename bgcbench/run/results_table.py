#!/usr/bin/env python
"""Assemble the cross-substrate comparison from run directories.

⚠ EVERY TABLE CARRIES ITS HEADER (CLAUDE.md). A bare number or an internal arm name means
nothing to a reader who was not in the code, and this project has already shipped one table
that put Evo2's nats/NUCLEOTIDE beside GenomeOcean's nats/TOKEN in adjacent rows.

⚠ WHAT MAKES THESE ROWS COMPARABLE is not that the arms were configured alike -- they were
not, deliberately (SPEC 14A: rank, sites, alpha and decoding are per substrate). It is that
every row went through the SAME measurement: n=200 per class, one 8,192 nt budget, one frozen
antiSMASH scoring config, one novelty gate. Those four invariants are asserted below, and a
row that violates any of them is flagged rather than printed as if it were comparable.
"""
import json, glob, os, sys, collections

RUNS = "/data2/ds85/bgcbench/runs"

def load(d):
    p = os.path.join(d, "report.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None

def row(d, r):
    """One comparable row.

    ⚠ THE DENOMINATOR IS NOT THE SAME SHAPE FOR EVERY ARM, and summing per_class blindly gets
    it wrong. A CLASS-BEARING arm (a per-class adapter) fills exactly one row: 200 generations,
    one class. A POOLED arm (W0, W1n; class_bearing=False) generates 200 and arm.py scores that
    SAME set against all four class rows, so per_class holds n=200 four times and repeats the
    identical n_detected in each. Summing gives n=800 and det=4x the truth: Evo2's W1n reads
    3 detections as 12, and its rate as 3/800 = 0.0037 when it is 3/200 = 0.015 -- a 4x
    understatement set beside per-class arms whose denominator is honest. n_on_target is the
    one field that DOES sum correctly, because a generation is on-target in at most one row.
    """
    pc = r["per_class"]; nov = r.get("novelty") or {}
    pooled = not r.get("class_bearing") and not r.get("row_class")
    if pooled:
        live = [v for v in pc.values() if v.get("n")]
        n   = live[0]["n"] if live else 0                      # ONE set of generations
        det = max((v.get("n_detected", 0) for v in live), default=0)   # identical across rows
        ont = sum(v.get("n_on_target", 0) for v in pc.values())        # disjoint across rows
        npass = max(((nov.get(c, {}).get("per_arm_gate", {}) or {}).get("PASS", 0)
                     for c in pc), default=0)
    else:
        n   = sum(v.get("n", 0) for v in pc.values())
        det = sum(v.get("n_detected", 0) for v in pc.values())
        ont = sum(v.get("n_on_target", 0) for v in pc.values())
        npass = sum((nov.get(c, {}).get("per_arm_gate", {}) or {}).get("PASS", 0) for c in pc)
    L = [json.loads(l).get("seq_len") or 0 for l in open(os.path.join(d, "scored.jsonl"))]
    return {
        "arm": r["arm"], "substrate": r["substrate"],
        "row_class": r.get("row_class") or ("POOLED" if pooled else "ALLROWS"),
        "n": n, "detected": det, "on_target": ont, "novel_pass": npass,
        "over_budget": sum(1 for x in L if x > (r.get("budget_nt") or 8192)),
        "max_len": max(L) if L else 0, "hit_eos": r.get("hit_eos_rate"),
        "n_per_class": r.get("n_per_class"), "budget_nt": r.get("budget_nt"),
        "scoring": r.get("scoring_config"),
    }

def main():
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "stage1_*"))):
        r = load(d)
        if not r:
            continue
        if not (r["arm"].startswith("evo2-1b_") or r["arm"].startswith("go-4b_")):
            continue
        if "_a0." in r["arm"]:                      # alpha-sweep rungs are not endpoint arms
            continue
        if want and want not in r["arm"]:
            continue
        rows.append(row(d, r))

    # --- the comparability invariants, asserted not assumed ---
    inv = collections.defaultdict(set)
    for x in rows:
        inv["n_per_class"].add(x["n_per_class"]); inv["budget_nt"].add(x["budget_nt"])
        inv["scoring"].add(x["scoring"])
    print("COMPARABILITY INVARIANTS (all rows must share these or the table is not a comparison)")
    for k in ("n_per_class", "budget_nt", "scoring"):
        v = sorted(str(y) for y in inv[k])
        print(f"  {k:12s} {'OK  ' if len(v)==1 else 'MIXED!'} {v}")
    bad = [x["arm"] for x in rows if x["over_budget"]]
    print(f"  budget       {'OK  ' if not bad else 'VIOLATED!'} "
          f"{'all rows within budget' if not bad else bad}")
    nov = [x["arm"] for x in rows if x["novel_pass"] != x["n"]]
    print(f"  novelty gate {'OK  ' if not nov else 'FAILED!'} "
          f"{'every generation passed' if not nov else nov}")
    print()

    def emit(title, question, pred):
        sel = [x for x in rows if pred(x)]
        if not sel:
            return
        print(f"QUESTION  {question}")
        print( "MODEL     Evo2-1B (evo2-1b) and GenomeOcean-4B (go-4b), labelled per row")
        print(f"ARM       {title}")
        print( "METRIC    on-target = antiSMASH called the row's own class. Higher is better.")
        print( "UNIT      counts out of n. EVERY row is 200 generations: a per-class arm fills")
        print( "          one class row; a POOLED arm's 200 are scored against all four rows,")
        print( "          so its detect count is that of the single 200-generation set.")
        print(f"  {'model':7s} {'arm':34s} {'class':15s} {'n':>4s} {'det':>4s} {'on-tgt':>7s} {'rate':>7s} {'eos':>5s}")
        for x in sorted(sel, key=lambda y: (y["substrate"], y["arm"])):
            m = "Evo2" if x["substrate"].startswith("evo2") else "GO"
            nm = x["arm"].replace("evo2-1b_", "").replace("go-4b_", "")
            rate = x["on_target"] / x["n"] if x["n"] else 0
            print(f"  {m:7s} {nm[:34]:34s} {x['row_class'][:15]:15s} {x['n']:>4d} "
                  f"{x['detected']:>4d} {x['on_target']:>7d} {rate:>7.4f} "
                  f"{(x['hit_eos'] if x['hit_eos'] is not None else 0):>5.2f}")
        print()

    emit("weight state only, DE NOVO (no seed, no steering)",
         "how often does each weight state produce its own BGC class unaided?",
         lambda x: "_S1" not in x["arm"] and "I1" not in x["arm"])
    emit("weight state + SEEDING (S1): generation continues a real core prefix",
         "how much does seeding add on top of the weight state?",
         lambda x: "_S1" in x["arm"] and "I1" not in x["arm"])
    emit("I1 = derived-direction steering; I1rand = magnitude/site-matched random control; "
         "I1xS1 = steering composed with seeding",
         "does activation steering change yield, and is it the direction's CONTENT doing it?",
         lambda x: "I1" in x["arm"])

if __name__ == "__main__":
    main()
