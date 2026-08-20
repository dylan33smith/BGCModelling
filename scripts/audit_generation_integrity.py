"""Generation-set integrity audit + recomputation on unique records.

Written 2026-08-19 after the fan-out shard collision (`bugs.md`): four workers launched with the
same `--seed` wrote byte-identical shards, so five Phase-4/5 arms carried an effective n of 47-141
while every table said 188. Rates survive uniform duplication; n, CIs and p-values do not.

Recomputes, on DEDUPLICATED records (first occurrence wins, order preserved): the Pfam endpoint,
the antiSMASH-corrected rate, the novelty gates, and the Holm-corrected WIDE contrasts.

`scripts/novelty_battery.py` now refuses to score a duplicated set, so this is a forensic tool for
the sets produced BEFORE that guard existed -- and a template for auditing any new one.
"""
import json, csv
from pathlib import Path
from scipy.stats import fisher_exact

RUNS = Path('/data2/ds85/bgcmodel_runs')
CMP  = RUNS / 'phase3_RIPP_widecmp'

ARMS = {  # arm -> (generation jsonl, score json w2000)
 'W1_seeded': (RUNS/'phase3_RIPP_wide/W1_seeded.jsonl',            CMP/'W1_seeded_w2000_RIPP.json'),
 'W2_seeded': (RUNS/'phase3_RIPP_strictmatched/W2_seeded.jsonl',   CMP/'W2_seeded_w2000_RIPP.json'),
 'W1_8k':     (RUNS/'phase3_RIPP_wide/W1_seeded8k.jsonl',          CMP/'W1_8k_w2000_RIPP.json'),
 'W2_8k':     (RUNS/'phase3_RIPP_strictmatched/W2_seeded8k.jsonl', CMP/'W2_8k_w2000_RIPP.json'),
 'SF_8k':     (RUNS/'phase3_RIPP/SF_seeded8k.jsonl',               CMP/'SF_8k_w2000_RIPP.json'),
}

def seqs_of(p):
    return [json.loads(l)['sequence'] for l in open(p)]

def dedup_idx(seqs):
    seen, keep = set(), []
    for i, s in enumerate(seqs):
        if s not in seen:
            seen.add(s); keep.append(i)
    return keep

# ---- antiSMASH TSV, aligned to the as_*.jsonl files by row order -------------
tsv = list(csv.DictReader(open(CMP/'antismash_widecmp.tsv'), delimiter='\t'))
by_arm = {}
for r in tsv:
    by_arm.setdefault(r['arm'], []).append(r)

def as_rate(tag):
    """(unique detections, unique n, raw detections, raw n) for as_<tag>.jsonl"""
    jl = CMP/f'as_{tag}.jsonl'
    if not jl.exists(): return None
    s = seqs_of(jl); rows = by_arm.get(f'as_{tag}', [])
    assert len(rows) == len(s), f'{tag}: {len(rows)} tsv vs {len(s)} jsonl'
    for i, r in enumerate(rows):                     # alignment check
        assert int(r['length']) == len(s[i]), f'{tag} row {i} length mismatch'
    keep = dedup_idx(s)
    det_u = sum(int(rows[i]['is_bgc']) for i in keep)
    det_r = sum(int(r['is_bgc']) for r in rows)
    return det_u, len(keep), det_r, len(rows)

print('=== ARM-LEVEL: raw vs deduplicated ===')
out = {}
for arm, (gen, sc) in ARMS.items():
    s = seqs_of(gen); keep = dedup_idx(s)
    d = json.load(open(sc))
    oc = d['on_class']
    assert len(oc) == len(s)
    P_raw, n_raw = sum(oc), len(oc)
    P_u,   n_u   = sum(oc[i] for i in keep), len(keep)
    pos = as_rate(f'{arm}_pos'); neg = as_rate(f'{arm}_neg')
    rp_u = pos[0]/pos[1] if pos and pos[1] else 0.0
    rn_u = neg[0]/neg[1] if neg and neg[1] else 0.0
    corr_u = (P_u/n_u)*rp_u + (1 - P_u/n_u)*rn_u
    out[arm] = dict(P_raw=P_raw, n_raw=n_raw, P_u=P_u, n_u=n_u,
                    rp_u=rp_u, rn_u=rn_u, corr_u=corr_u,
                    pos_u=pos, neg_u=neg)
    print(f'{arm:10s} Pfam {P_raw:3d}/{n_raw:3d}={P_raw/n_raw:.3f}  ->  '
          f'UNIQUE {P_u:3d}/{n_u:3d}={P_u/n_u:.3f}   '
          f'corrected {corr_u:.3f}   rp={rp_u:.3f}(n={pos[1] if pos else 0}) '
          f'rn={rn_u:.3f}(n={neg[1] if neg else 0})')

print()
print('=== THE CONTRASTS, recomputed on unique records (Fisher exact, Pfam counts) ===')
def fish(a, b):
    A, B = out[a], out[b]
    raw = fisher_exact([[A['P_raw'], A['n_raw']-A['P_raw']],
                        [B['P_raw'], B['n_raw']-B['P_raw']]])[1]
    uni = fisher_exact([[A['P_u'],   A['n_u']  -A['P_u']],
                        [B['P_u'],   B['n_u']  -B['P_u']]])[1]
    return raw, uni

tests = [('W1_seeded','W2_seeded','WIDE vs STRICT-matched @2.2kb'),
         ('W1_8k','W2_8k','WIDE vs STRICT-matched @8kb'),
         ('W2_8k','SF_8k','STRICT-matched vs STRICT-full (dataset size)')]
res = []
for a, b, lab in tests:
    raw, uni = fish(a, b); res.append((lab, raw, uni))

# Holm over the 3 tests, on each family separately
for name, idx in (('RAW (as published)', 1), ('UNIQUE (corrected)', 2)):
    ps = sorted(((r[idx], r[0]) for r in res))
    m = len(ps); adj = {}
    running = 0.0
    for k, (p, lab) in enumerate(ps):
        v = max(running, min(1.0, (m-k)*p)); running = v; adj[lab] = v
    print(f'-- {name}')
    for lab, raw, uni in res:
        p = raw if idx == 1 else uni
        print(f'   {lab:46s} p={p:.3g}  Holm={adj[lab]:.3g}  '
              f'{"SIG" if adj[lab] < 0.05 else "n.s."}')

# ─── extra rows needed for the consolidated Phase-3 reporting set ────────────
print()
print('=== NOVELTY + LADDER on unique records ===')
for arm, (gen, sc) in ARMS.items():
    s = seqs_of(gen); keep = dedup_idx(s)
    d = json.load(open(sc))
    cont = [d['nt_containment'][i] for i in keep]
    aai  = [d['aai'][i] for i in keep]
    oc   = [d['on_class'][i] for i in keep]
    # intra-set distinctness must be RECOMPUTED on the deduplicated set: the
    # published value was 0 by construction.
    lad = d['ladder']
    print(f"{arm:10s} n={len(keep):3d}  max_cont={max(cont):.3f}  max_aai={max(aai):.3f}  "
          f"aai_on_class={max([a for a,o in zip(aai,oc) if o], default=0):.3f}  "
          f"n_bio_domains|on={lad['n_bio_domains'].get('mean_among_on_class')}  "
          f"n_class_ge2={lad['n_class_domains']['n_with_ge2']}  "
          f"co_orient={lad['co_orient']['mean']:.3f}  n_orfs={lad['n_orfs']['mean']:.2f}")
