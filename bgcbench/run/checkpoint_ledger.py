#!/usr/bin/env python
"""Record WHICH CHECKPOINT produced each result, without touching the realised hash.

⚠ WHY THIS EXISTS. A run's `realised` block records the adapter DIRECTORY
(.../go-4b_W1n_fx), not the checkpoint inside it that was actually loaded. Which one that is
depends on `resolve_best()`'s preference order, and that order CHANGED on 2026-09-14 from
BEST-first to EPOCH-first. So two runs with byte-identical `adapter` fields can have been
generated from different weights, and the artifact alone cannot tell them apart.

Rather than change the realised block (which would change every run hash and make new runs
non-comparable to today's), this reconstructs the mapping from the phase logs, which print
`attached adapter <resolved path>` immediately before each arm's `arm=<name>` summary line.

Rebuilt from the logs each time, so it is idempotent and self-correcting.
"""
import json, re, glob, os
from pathlib import Path

LOGS = glob.glob("/data2/ds85/bgcbench/work/pipeline/*.log") + \
       glob.glob("/data2/ds85/bgcbench/work/*.log")
OUT = Path("/data2/ds85/bgcbench/runs/CHECKPOINT_LEDGER.json")
A = Path("/data2/ds85/bgcbench/adapters")

ledger = {}
for lg in LOGS:
    try: lines = Path(lg).read_text(errors="replace").splitlines()
    except Exception: continue
    pending = None            # the most recent "attached adapter" path
    for ln in lines:
        m = re.search(r"attached (?:adapter|intervention) (\S+)", ln)
        if m:
            pending = m.group(1); continue
        m = re.match(r"arm=(\S+) substrate=(\S+)", ln)
        if m:
            arm, sub = m.group(1), m.group(2)
            ck = pending
            pending = None
            entry = {"substrate": sub, "checkpoint_used": ck, "from_log": os.path.basename(lg)}
            if ck and "/" in ck:
                adir, leaf = os.path.split(ck)
                entry["adapter_dir"] = adir
                entry["checkpoint"] = leaf              # 'epoch' | 'best' | 'final' | '*.pt'
                marker = {"epoch": "EPOCH", "best": "BEST"}.get(leaf)
                mf = Path(adir) / marker if marker else None
                if mf and mf.exists():
                    try:
                        md = json.loads(mf.read_text())
                        entry["step"] = md.get("step")
                        entry["val_loss"] = md.get("val_loss")
                        entry["steps_per_epoch"] = md.get("steps_per_epoch")
                        if md.get("step") and md.get("steps_per_epoch"):
                            entry["epochs"] = round(md["step"] / md["steps_per_epoch"], 3)
                        # the road not taken, so the trade-off is visible in the record
                        other = Path(adir) / ("BEST" if marker == "EPOCH" else "EPOCH")
                        if other.exists():
                            od = json.loads(other.read_text())
                            entry["alternative"] = {"checkpoint": other.name.lower(),
                                                    "step": od.get("step"),
                                                    "val_loss": od.get("val_loss")}
                    except Exception: pass
            else:
                entry["checkpoint"] = "base (no adapter)" if ck is None else ck
            ledger[arm] = entry

# attach the run directory and headline counts, so one file answers "which weights, what result"
for d in glob.glob("/data2/ds85/bgcbench/runs/stage1_*"):
    p = os.path.join(d, "report.json")
    if not os.path.exists(p): continue
    try: r = json.load(open(p))
    except Exception: continue
    e = ledger.get(r.get("arm"))
    if e is None: continue
    e["run_dir"] = os.path.basename(d)
    pc = {c: v for c, v in (r.get("per_class") or {}).items() if v.get("n")}
    rc = r.get("row_class")
    if rc and rc in pc:
        e["result"] = f"{pc[rc]['n_on_target']}/{pc[rc]['n']} on-target ({rc})"
    elif pc:
        e["result"] = f"{max(v['n_detected'] for v in pc.values())}/{max(v['n'] for v in pc.values())} detected (pooled)"
    e["median_len"] = r.get("median_len")

OUT.write_text(json.dumps(ledger, indent=1, sort_keys=True))
print(f"{len(ledger)} arms recorded -> {OUT}\n")
rows = [(a, e) for a, e in sorted(ledger.items()) if e.get("run_dir") and "_fx" in a]
print(f"{'arm':30s} {'checkpoint':10s} {'step':>6s} {'epochs':>7s} {'val':>9s}  result")
for a, e in rows:
    vl = e.get("val_loss")
    vs = f"{vl:.4f}" if isinstance(vl, (int, float)) else "-"
    print(f"{a:30s} {str(e.get('checkpoint')):10s} {str(e.get('step','-')):>6s} "
          f"{str(e.get('epochs','-')):>7s} {vs:>9s}  {e.get('result','-')}")
