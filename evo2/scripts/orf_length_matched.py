"""LENGTH-MATCHED ORF comparison. The earlier table compared de novo @6kb against real @3kb —
different windows, so the gap was not interpretable. A longest-ORF statistic is capped by the
window (a 2 kb window cannot hold an ORF above 666 aa), so real and generated must be truncated
to the SAME length before they can be compared."""
import json, sys, statistics as st
sys.path.insert(0,'/home/ds85/projects/BCGModelling/src')
from bgc_pipeline.evaluation import find_orfs
from concurrent.futures import ProcessPoolExecutor

def one(job):
    tag, seq, L = job
    s = seq[:L]
    orfs = find_orfs(s)
    lens = sorted((len(o.aa_seq) for o in orfs), reverse=True)
    return {"tag": tag, "L": L, "n_orfs": len(orfs),
            "max_aa": lens[0] if lens else 0,
            "sum_aa": sum(lens),
            "frac_of_window": (lens[0]*3/L) if lens else 0.0}

jobs=[]
# generated arms at their native lengths
for f,tag,L in [('steer_titration/L16_b0.jsonl','denovo',2000),
                ('steer_magnitude/L16_d0.jsonl','denovo',2000),
                ('steer_sweep/a0_control.jsonl','denovo',6000),
                ('guided_decoding/gd_NRPS_plain.jsonl','seeded',3000),
                ('guided_decoding/gd_PKS_plain.jsonl','seeded',3000),
                ('guided_decoding/gd_TERPENE_plain.jsonl','seeded',3000),
                ('guided_decoding/gd_RIPP_plain.jsonl','seeded',3000)]:
    for r in (json.loads(l) for l in open(f'/data2/ds85/bgcmodel_runs/{f}') if l.strip()):
        if len(r['sequence'])>=L: jobs.append((tag, r['sequence'], L))
# real cores truncated to EACH of the generated lengths
real=[]
for line in open('/data2/ds85/bgcmodel_data/splits_core/test.jsonl'):
    r=json.loads(line)
    if r.get('compound_class') in ('NRPS','PKS','TERPENE','RIPP') and len(r['sequence'])>=6000:
        real.append(r['sequence'])
        if len(real)>=40: break
for L in (2000,3000,6000):
    for s in real: jobs.append(('REAL', s, L))

with ProcessPoolExecutor(max_workers=12) as ex:
    rows=list(ex.map(one, jobs))

print(f"{'group':>8} {'window':>7} {'n':>4} {'max ORF aa':>11} {'ceiling aa':>11} "
      f"{'% of window':>12} {'n_orfs':>7}")
for L in (2000,3000,6000):
    for tag in ('REAL','seeded','denovo'):
        sub=[r for r in rows if r['tag']==tag and r['L']==L]
        if not sub: continue
        print(f"{tag:>8} {L:>7,} {len(sub):>4} {st.mean(r['max_aa'] for r in sub):>11.0f} "
              f"{L//3:>11,} {st.mean(r['frac_of_window'] for r in sub):>11.1%} "
              f"{st.mean(r['n_orfs'] for r in sub):>7.1f}")
    print()
print("RATIO de novo / REAL at the SAME window:")
for L in (2000,6000):
    d=[r['max_aa'] for r in rows if r['tag']=='denovo' and r['L']==L]
    rr=[r['max_aa'] for r in rows if r['tag']=='REAL' and r['L']==L]
    if d and rr: print(f"  {L:,} nt: {st.mean(d):.0f} / {st.mean(rr):.0f} = {st.mean(d)/st.mean(rr):.2f}")
d=[r['max_aa'] for r in rows if r['tag']=='seeded' and r['L']==3000]
rr=[r['max_aa'] for r in rows if r['tag']=='REAL' and r['L']==3000]
print(f"  seeded 3,000 nt: {st.mean(d):.0f} / {st.mean(rr):.0f} = {st.mean(d)/st.mean(rr):.2f}")
