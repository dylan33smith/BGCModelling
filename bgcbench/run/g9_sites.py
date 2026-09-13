"""G9, the site half: WHICH attention sites should the direction be injected at?

⚠ THIS HALF WAS NEVER RUN. §6 specifies "injection site and magnitude, swept together by
Gate G9". Only α was ever swept; the site set sat at an unexamined default of "every
attention site that is not degenerate", on both substrates. Matching a second substrate to
that default would propagate an unmeasured choice and call the result a controlled
comparison.

⚠ THE CRITERION IS NOT THE ENDPOINT (§2.4). Two endpoint-free quantities per candidate set:

  reach   mean KL(steered ‖ unsteered) over next-token distributions -- how hard the
          intervention lands through those sites. This is SPEC §6's own check (b), the half
          that licenses reading a null, reused as a selection statistic.
  fidelity the mean train/val cosine over the steered sites -- whether the direction at
          those sites is real class content rather than noise (check (c)).

A site set is ADMISSIBLE when fidelity clears the same bar the manipulation check uses
(mean > 0.3, no steered site anti-aligned); among admissible sets the chosen one has the
highest reach. Reach is maximised rather than minimised because α is what controls damage,
and α is swept afterwards against generation health -- so this step asks "through which
sites does the direction travel best", and the α sweep then asks "how hard can we push".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bgcbench.model import directions as D
from bgcbench.model.interventions import attention_sites, matched_depth_subset
from bgcbench.model.load import attach_adapter, load, resolve_best
from bgcbench.provenance import code_version, corpus_max_len

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
OUT = ROOT / "g9sites"


def _load(p: Path, cls: str) -> list[dict]:
    out = []
    for line in open(p):
        r = json.loads(line)
        r["split_class"] = cls
        out.append(r)
    return out


def candidate_sets(n: int) -> dict:
    """Site sets to try, parameterised by how many the substrate exposes.

    Enumerating all subsets is only feasible for Evo2 (4 sites, 15 non-empty). For a
    24-layer decoder it is not, so the candidates are singles spread through depth,
    contiguous thirds, the Evo2-matched relative depths, and all-of-them.
    """
    if n <= 6:
        sets = {}
        for i in range(n):
            sets[f"single_{i}"] = [i]
        for i in range(n - 1):
            sets[f"pair_{i}{i+1}"] = [i, i + 1]
        sets["all"] = list(range(n))
        if n >= 3:
            sets["early2"] = [0, 1]
            sets["late2"] = [n - 2, n - 1]
        return sets
    third = n // 3
    sets = {"all": list(range(n)),
            "early_third": list(range(0, third)),
            "middle_third": list(range(third, 2 * third)),
            "late_third": list(range(2 * third, n)),
            "evo2_matched": matched_depth_subset(n)}
    for i in sorted({0, third // 2, third, n // 2, 2 * third, n - 1}):
        sets[f"single_{i}"] = [i]
    return sets


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="evo2-1b")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--prefix", default="taxonomy")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="FIXED probe magnitude for the site comparison. Not the generation "
                         "alpha -- that is swept afterwards, at the winning site set.")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--max-len-nt", type=int, default=corpus_max_len())
    ap.add_argument("--tag", default="G9SITES")
    args = ap.parse_args()

    sub = load(args.substrate)
    adapter = None
    if args.adapter:
        adapter = resolve_best(args.adapter)
        sub = attach_adapter(sub, adapter)
    base = sub.model.model if sub.family == "evo2" else sub.model
    n_sites = len(attention_sites(base))

    art = torch.load(args.direction, map_location="cpu", weights_only=False)
    d_all = art["directions"]
    cos = (art.get("manipulation_check") or {}).get("cosine_train_val") or []
    if d_all.shape[0] != n_sites:
        raise SystemExit(f"direction has {d_all.shape[0]} sites, model exposes {n_sites}")

    others = [c for c in ("TERPENE", "RIPP", "ARYLPOLYENE", "REDOX_COFACTOR")
              if c != args.target]
    va_t = _load(SPLITS / args.target / "val.jsonl", args.target)
    if args.prefix == "taxonomy":
        from bgcbench.data import taxonomy
        taxonomy.attach(va_t, taxonomy.load_table())

    cands = candidate_sets(n_sites)
    print(f"{args.substrate}: {n_sites} attention sites, {len(cands)} candidate sets, "
          f"probe alpha {args.alpha}\n", flush=True)
    print(f"{'site set':16s} {'sites':22s} {'reach (KL)':>11s} {'fidelity':>9s} {'min cos':>8s} {'adm':>5s}")
    rows = []
    for name, idx in cands.items():
        d = d_all.clone()
        for i in range(d.shape[0]):
            if i not in set(idx):
                d[i] = 0.0
        live = [i for i in idx if float(d_all[i].norm()) > 0]
        if not live:
            print(f"{name:16s} {str(idx):22s} {'--':>11s} {'--':>9s} {'--':>8s} "
                  f"{'no live site':>5s}")
            continue
        kl = D.kl_vs_unsteered(sub, va_t, d, args.prefix, args.max_len_nt,
                               alpha=args.alpha, limit=args.limit)
        fid = [cos[i] for i in live] if cos else []
        mean_fid = sum(fid) / len(fid) if fid else None
        min_fid = min(fid) if fid else None
        adm = bool(mean_fid is not None and mean_fid > 0.3 and min_fid > 0.0)
        rows.append({"name": name, "sites": idx, "live_sites": live,
                     "reach_kl_nats": kl["mean_kl_nats"],
                     "frac_argmax_changed": kl["frac_argmax_changed"],
                     "fidelity_mean_cosine": mean_fid, "min_cosine": min_fid,
                     "admissible": adm})
        print(f"{name:16s} {str(idx):22s} {kl['mean_kl_nats']:11.5f} "
              f"{(mean_fid if mean_fid is not None else 0):9.3f} "
              f"{(min_fid if min_fid is not None else 0):8.3f} {str(adm):>5s}")

    adm = [r for r in rows if r["admissible"]]
    best = max(adm, key=lambda r: r["reach_kl_nats"]) if adm else None
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"gate": "G9-sites", "substrate": args.substrate, "target": args.target,
           "adapter": adapter, "probe_alpha": args.alpha, "n_attention_sites": n_sites,
           "selection_fields": ["reach_kl_nats", "fidelity_mean_cosine", "min_cosine"],
           "endpoint_fields_read": [],
           "criterion": ("admissible = mean steered-site train/val cosine > 0.3 and no "
                         "steered site anti-aligned; among admissible, the highest reach "
                         "(mean KL against unsteered). Never the endpoint (§2.4)."),
           "chosen": best["name"] if best else None,
           "chosen_sites": best["sites"] if best else None,
           "rows": rows, "code_version": code_version()}
    p = OUT / f"{args.tag}_{args.substrate}_{args.target}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nCHOSEN SITE SET: {out['chosen']} -> {out['chosen_sites']}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
