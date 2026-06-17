#!/usr/bin/env python3
"""Calibration check for the recalibrated M2: do REAL held-out cores pass?

For a sample of splits_core/test cores per class, check whether each contains ANY
of its class's markers (the recalibrated OBLIGATE_DOMAINS) — i.e. M2's pass rule.
Real BGCs SHOULD pass at a high rate now (the goal of the recalibration). One
batched Pfam-A hmmsearch over all sampled ORFs for speed.
"""
from __future__ import annotations
import json, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bgc_pipeline.evaluation import find_orfs, OBLIGATE_DOMAINS  # noqa: E402
import pyhmmer  # noqa: E402
from pyhmmer.easel import Alphabet, TextSequence  # noqa: E402
from pyhmmer.plan7 import HMMFile  # noqa: E402

TEST = Path("/data2/ds85/bgcmodel_data/splits_core/test.jsonl")
PFAM = Path("/data2/ds85/pfam/Pfam-A.hmm")
PER_CLASS = 20

by_class = defaultdict(list)
for l in TEST.open():
    r = json.loads(l); by_class[r["compound_class"]].append(r["sequence"])
rng = random.Random(7)
alphabet = Alphabet.amino()
digital, name_cc = [], {}
n_cores = {}
for c, seqs in by_class.items():
    rng.shuffle(seqs); sample = seqs[:PER_CLASS]; n_cores[c] = len(sample)
    for ci, s in enumerate(sample):
        for oi, orf in enumerate(find_orfs(s, 50)):
            nm = f"{c}#{ci}#{oi}"; name_cc[nm] = (c, ci)
            digital.append(TextSequence(name=nm.encode(), sequence=orf.aa_seq).digitize(alphabet))
print(f"[calib] cores={sum(n_cores.values())} ORFs={len(digital)}; scanning ...", file=sys.stderr)

core_doms = defaultdict(set)   # (class,ci) -> set(pfam)
with HMMFile(str(PFAM)) as hf:
    for th in pyhmmer.hmmsearch(hf, digital, E=1e-10):
        acc = (th.query.accession.decode() if isinstance(th.query.accession, (bytes, bytearray))
               else str(th.query.accession or "")).split(".")[0]
        if not acc:
            continue
        for hit in th:
            if hit.included:
                hn = hit.name.decode() if isinstance(hit.name, (bytes, bytearray)) else str(hit.name)
                cc = name_cc.get(hn)
                if cc:
                    core_doms[cc].add(acc)

# per-class pass rate = fraction of cores with >=1 class marker
passed = defaultdict(int)
for (c, ci), doms in core_doms.items():
    if set(OBLIGATE_DOMAINS.get(c, [])) & doms:
        passed[c] += 1
tot_p = tot_n = 0
print(f"\n{'class':18}{'pass':>6}{'n':>5}{'rate':>7}")
for c in sorted(n_cores, key=lambda k: -n_cores[k]):
    p, n = passed[c], n_cores[c]; tot_p += p; tot_n += n
    print(f"{c:18}{p:>6}{n:>5}{(p/n if n else 0):>7.2f}")
print(f"\nOVERALL M2 pass on real held-out cores: {tot_p}/{tot_n} = {tot_p/tot_n:.2f}")
print("(was ~0 on many classes before recalibration)")
