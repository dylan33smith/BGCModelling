"""G9: the I1 injection-magnitude sweep.

⚠ THE CRITERION IS NOT THE ENDPOINT (SPEC 2.4, and §6's G9 row). It is "the largest α at
which generation quality is not degraded beyond a stated tolerance, read as per-token
likelihood on the model's own output and coding density".

So this script reads exactly three quantities from each arm: the model's own per-token NLL on
what it wrote, the coding density prodigal finds in it, and the realised length. It NEVER
reads `detected`, `on_target`, `products` or `observed_classes` -- those fields exist in the
scored records it walks past, and selecting on them would tune the arm against the metric
that scores it. The selection is recorded with the fields it used so the claim is auditable.

TOLERANCE, stated in advance: α is admissible while median coding density stays within
`--tol-coding` (default 0.10, relative) of the α=0 arm AND median self-NLL stays within
`--tol-nll` (default 0.10, relative). The chosen α is the LARGEST admissible one -- steering
as hard as possible without breaking generation, which is what the arm is trying to test.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path("/data2/ds85/bgcbench")
OUT = ROOT / "g9"


def _self_nll(sub, seqs: list[str], device: str = "cuda:0") -> list[float]:
    """Per-token NLL of the model's OWN output -- 'is it confident in what it wrote'."""
    import torch.nn.functional as F

    from bgcbench.model.train import _unwrap
    base = sub.model.model if sub.family == "evo2" else sub.model
    out = []
    with torch.no_grad():
        for s in seqs:
            if len(s) < 2:
                continue
            ids = [int(x) for x in sub.tokenizer.tokenize(s)]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            lo = _unwrap(base(x))
            lp = F.log_softmax(lo.float(), dim=-1)[:, :-1]
            tgt = x[:, 1:]
            out.append(float(-lp.gather(-1, tgt.unsqueeze(-1)).mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="evo2-1b")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--row-class", required=True)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tol-coding", type=float, default=0.10)
    ap.add_argument("--tol-nll", type=float, default=0.10)
    ap.add_argument("--prefix", default="taxonomy")
    ap.add_argument("--tag", default="G9")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in args.alphas:
        arm = f"{args.tag}_a{a}_{args.row_class}"
        cmd = [sys.executable, "-m", "bgcbench.run.arm", "--arm", arm,
               "--row-class", args.row_class, "--n", str(args.n),
               "--prefix", args.prefix, "--stage", "stage1", "--off-frozen",
               "--batch-size", "50"]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        if a > 0:
            cmd += ["--direction", args.direction, "--alpha", str(a)]
        print(f"\n=== alpha {a}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"alpha {a} failed rc={rc}", flush=True)
            rows.append({"alpha": a, "rc": rc})
            continue
        d = sorted(ROOT.glob(f"runs/stage1_{args.substrate}_{arm}_*"),
                   key=lambda p: p.stat().st_mtime)[-1]
        # ⚠ ONLY these fields. detected / on_target / products are deliberately not read.
        cd, ln, eos, seqs = [], [], [], []
        for line in open(d / "scored.jsonl"):
            r = json.loads(line)
            if r.get("coding_density") is not None:
                cd.append(r["coding_density"])
            ln.append(r.get("seq_len") or 0)
            eos.append(bool(r.get("hit_eos")))
            seqs.append(r.get("sequence") or "")
        rows.append({"alpha": a, "rc": 0, "run_dir": str(d), "n": len(ln),
                     "median_coding_density": st.median(cd) if cd else None,
                     "median_len": st.median(ln) if ln else None,
                     "hit_eos_rate": (sum(eos) / len(eos)) if eos else None,
                     "_seqs": seqs[:16]})

    # self-NLL, one model load for all arms
    from bgcbench.model.load import attach_adapter, load, resolve_best
    sub = load(args.substrate)
    if args.adapter:
        sub = attach_adapter(sub, resolve_best(args.adapter))
    for r in rows:
        s = r.pop("_seqs", [])
        v = _self_nll(sub, s) if s else []
        r["median_self_nll"] = st.median(v) if v else None

    base = next((r for r in rows if r["alpha"] == 0.0 and r.get("rc") == 0), None)
    chosen, why = None, "no alpha=0 baseline; cannot judge degradation"
    if base and base["median_coding_density"] and base["median_self_nll"]:
        adm = []
        for r in rows:
            if r.get("rc") != 0 or r["alpha"] == 0.0:
                continue
            dc = abs(r["median_coding_density"] - base["median_coding_density"]) / base["median_coding_density"]
            dn = abs(r["median_self_nll"] - base["median_self_nll"]) / base["median_self_nll"]
            r["rel_coding_change"], r["rel_nll_change"] = dc, dn
            r["admissible"] = bool(dc <= args.tol_coding and dn <= args.tol_nll)
            if r["admissible"]:
                adm.append(r["alpha"])
        chosen = max(adm) if adm else None
        why = (f"largest alpha with |Δcoding| <= {args.tol_coding} and |Δself-NLL| <= "
               f"{args.tol_nll}, both relative to alpha=0")

    art = {"gate": "G9", "row_class": args.row_class, "direction": args.direction,
           "adapter": args.adapter, "alphas": args.alphas, "n_per_alpha": args.n,
           "tolerance": {"coding": args.tol_coding, "nll": args.tol_nll},
           "selection_fields": ["median_coding_density", "median_self_nll"],
           "endpoint_fields_read": [],
           "criterion": why, "chosen_alpha": chosen, "rows": rows}
    p = OUT / f"{args.tag}_{args.row_class}.json"
    p.write_text(json.dumps(art, indent=1))
    print(f"\n{'alpha':>6s} {'coding':>8s} {'self-NLL':>9s} {'med len':>8s} {'eos':>6s} {'admissible':>11s}")
    for r in rows:
        if r.get("rc") != 0:
            print(f"{r['alpha']:6.2f}  FAILED"); continue
        print(f"{r['alpha']:6.2f} {r['median_coding_density'] or 0:8.4f} "
              f"{r['median_self_nll'] or 0:9.4f} {r['median_len'] or 0:8.0f} "
              f"{r['hit_eos_rate'] or 0:6.3f} {str(r.get('admissible','baseline')):>11s}")
    print(f"\nCHOSEN ALPHA: {chosen}  ({why})")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
