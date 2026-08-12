"""Is de novo failure a CAPABILITY problem or an INSTRUMENT problem?
antiSMASH needs clustered biosynthetic genes. Softer instruments ask smaller questions:
does the DNA even code for proteins, and do those proteins resemble anything known?"""
import json, sys
from pathlib import Path
sys.path.insert(0,'/home/ds85/projects/BCGModelling/src')
from bgc_pipeline.evaluation import check_coding_sanity, check_class_markers
from concurrent.futures import ProcessPoolExecutor
PFAM=Path('/data2/ds85/pfam/Pfam-A.hmm')

def one(job):
    tag, seq, cls = job
    cs = check_coding_sanity(seq)
    mk = check_class_markers(seq, expected_class=cls, pfam_hmm_path=PFAM)
    return {"tag":tag,"nt":len(seq),
            "coding_density":cs.get("coding_density"),
            "n_orfs":cs.get("n_orfs") or cs.get("num_orfs"),
            "longest_orf_aa":cs.get("longest_orf_aa") or cs.get("max_orf_aa"),
            "any_pfam":bool(mk.get("any_pfam_hit") if mk.get("any_pfam_hit") is not None
                            else mk.get("n_pfam_hits",0)),
            "n_pfam":mk.get("n_pfam_hits"),
            "markers":bool(mk.get("markers_present")) if not mk.get("skipped") else None}

jobs=[]
for f,tag in [('steer_sweep/a0_control.jsonl','denovo_6k'),
              ('steer_titration/L16_b0.jsonl','denovo_2k'),
              ('steer_magnitude/L16_d0.jsonl','denovo_2k'),
              ('guided_decoding/gd_NRPS_plain.jsonl','seeded_3k'),
              ('guided_decoding/gd_PKS_plain.jsonl','seeded_3k'),
              ('guided_decoding/gd_TERPENE_plain.jsonl','seeded_3k')]:
    for r in (json.loads(l) for l in open(f'/data2/ds85/bgcmodel_runs/{f}') if l.strip()):
        jobs.append((tag, r['sequence'], r.get('compound_class') or 'NRPS'))
# real cores at matched length, as the ceiling
for line in open('/data2/ds85/bgcmodel_data/splits_core/test.jsonl'):
    r=json.loads(line)
    if r.get('compound_class') in ('NRPS','PKS','TERPENE','RIPP') and len(r['sequence'])>=3000:
        jobs.append(('REAL_3k', r['sequence'][:3000], r['compound_class']))
        if sum(1 for j in jobs if j[0]=='REAL_3k')>=30: break
print(f"{len(jobs)} sequences", flush=True)
rows=[]
with ProcessPoolExecutor(max_workers=12) as ex:
    for i,res in enumerate(ex.map(one, jobs)):
        rows.append(res)
        if (i+1)%25==0: print(f"  {i+1}/{len(jobs)}", flush=True)
json.dump(rows, open('/data2/ds85/bgcmodel_runs/soft_instruments.json','w'), indent=1)
import statistics as st
print(f"\n{'group':>12} {'n':>4} {'coding dens':>12} {'longest ORF aa':>15} {'any Pfam':>9} {'class markers':>14}")
for g in ['denovo_2k','denovo_6k','seeded_3k','REAL_3k']:
    sub=[r for r in rows if r['tag']==g]
    if not sub: continue
    cd=[r['coding_density'] for r in sub if r['coding_density'] is not None]
    lo=[r['longest_orf_aa'] for r in sub if r['longest_orf_aa'] is not None]
    ap=[r['any_pfam'] for r in sub]
    mk=[r['markers'] for r in sub if r['markers'] is not None]
    print(f"{g:>12} {len(sub):>4} {st.mean(cd):>12.3f} {(st.mean(lo) if lo else float('nan')):>15.0f} "
          f"{st.mean(ap):>9.3f} {(st.mean(mk) if mk else float('nan')):>14.3f}")
