"""G9, the site half: WHICH attention sites should the direction be injected at?

⚠ THIS HALF WAS NEVER RUN. §6 specifies "injection site and magnitude, swept together by
Gate G9". Only α was ever swept; the site set sat at an unexamined default of "every
attention site that is not degenerate", on both substrates. Matching a second substrate to
that default would propagate an unmeasured choice and call the result a controlled
comparison.

⚠ EACH SUBSTRATE IS PROBED IN ITS OWN WORKING REGIME, AND NOTHING IS MATCHED ACROSS THEM.
The two models are treated differently and given the same TASK; what is compared is the task
outcome, never the configuration. So the site set, the injection magnitude and the probe
magnitude are each chosen per substrate, by measurement, and a candidate motivated by
resembling the other model is not privileged.

Concretely: `evo2_matched` appears in GenomeOcean's candidate list only because an earlier
plan would have imposed Evo2's relative depths on it. Measured, it reaches 0.023 against
`early_third`'s 1.093 — a ~47x penalty — because Evo2's 4-of-25 spacing spreads injection
across depth while GenomeOcean wants it concentrated early. It is retained as evidence that
matching would have been wrong, never as a contender.

⚠ AND THE PROBE MUST SIT INSIDE THE SUBSTRATE'S OWN HEALTH CEILING. Ranking site sets at a
magnitude the model cannot actually be run at ranks them in a regime its generation does not
survive. Evo2's ceiling is α = 0.3 (G9, FINDINGS §15), so Evo2 is probed there. A substrate
whose ceiling is not yet measured is probed provisionally (`--target-kl`) and re-confirmed
once its own α sweep has run.

⚠ THE CRITERION IS NOT THE ENDPOINT (§2.4). Two endpoint-free quantities per candidate set:

  reach   mean KL(steered ‖ unsteered) over next-token distributions -- how hard the
          intervention lands through those sites. This is SPEC §6's own check (b), the half
          that licenses reading a null, reused as a selection statistic.
  fidelity the mean train/val cosine over the steered sites -- whether the direction at
          those sites is real class content rather than noise (check (c)).

A site set is ADMISSIBLE when fidelity clears the same bar the manipulation check uses
(mean > 0.3, no steered site anti-aligned). Among admissible sets, REACH GATES and FIDELITY
CHOOSES: eligible = reach ≥ 10% of the best admissible set's reach, and among those the
highest fidelity wins.

⚠ REACH USED TO BE MAXIMISED, AND THAT SELECTED DEPTH RATHER THAN CLASS CONTENT (§12.A9).
Perturbing an early site changes the input to every site downstream of it, so reach falls
monotonically with depth for mechanical reasons: measured on GenomeOcean at α=1, every one of
the four classes reads early_third > middle_third > late_third (5.900/3.500/0.226,
8.176/1.719/0.146, 3.697/0.525/0.034, 4.022/1.604/0.063 nats/token). Meanwhile the smallest
relative class-difference norm sits in the EARLY band for all four classes (0.051, 0.140,
0.092, 0.052) -- the rule was picking the sites carrying least class signal. On Evo2's 4 sites
it barely moves; on a 24-layer stack it decided the answer.

The reach bar is RELATIVE to the best admissible set, not an absolute nats/token floor, so it
is invariant to the probe magnitude and carries no unit across substrates -- a fixed floor
would mean different things on a 1 nt token and a 4.8 nt one.

⚠ SELECTION READS val_A; THE SPEC 6.4 CHECK READS val_B. This used to select on the per-site
cosines inside `manipulation_check` -- the reported evidence -- so the site set was chosen to
maximise the number that then licensed the arm. `--finalize` re-reads all three checks at the
chosen sites on val_B and writes them into the direction artifact, which is what
`attach_direction` gates on.
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
    ap.add_argument("--prefix", choices=["none", "taxonomy"], default=None,
                    help="omit to use the substrate's own setting (SPEC 14A): evo2 -> taxonomy, "
                         "genomeocean -> none. ⚠ THIS DEFAULTED TO 'taxonomy' FOR BOTH.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="probe magnitude for the site comparison. Omit to use the top rung "
                         "of this substrate's own generation grid. ⚠ DEFAULTED TO 1.0 FOR "
                         "BOTH SUBSTRATES; SPEC §14A names probe magnitude a TREATMENT "
                         "parameter, and comparing site sets at a magnitude no arm generates "
                         "at picks the sites that win in a regime nothing runs in.")
    ap.add_argument("--target-kl", type=float, default=None,
                    help="calibrate the probe alpha so mean KL lands here, instead of using "
                         "--alpha literally. ⚠ The same alpha means different things on "
                         "different substrates: at alpha=1 Evo2 sits at KL 0.197 and GO at "
                         "3.4-7.8 with ~98%% of tokens flipped. Probing both at one magnitude "
                         "compares a working model with a wrecked one (§20.1).")
    ap.add_argument("--limit", type=int, default=16,
                    help="records per candidate set for the reach (KL) measurement.")
    ap.add_argument("--check-limit", type=int, default=64,
                    help="records per side for the --finalize SPEC 6.4 re-read. Matches "
                         "run.derive_directions' --limit so the chosen-site check is read at "
                         "the same sample size as the all-sites one it replaces.")
    ap.add_argument("--max-len-nt", type=int, default=corpus_max_len())
    ap.add_argument("--reach-floor-frac", type=float, default=0.10,
                    help="a candidate set must reach at least this FRACTION of the best "
                         "admissible set's reach to be eligible. Relative rather than "
                         "absolute so it is invariant to the probe magnitude and carries no "
                         "unit across substrates.")
    ap.add_argument("--finalize", action="store_true",
                    help="after choosing, re-read the SPEC 6.4 check on val_B RESTRICTED to "
                         "the chosen sites and write the result back into the direction "
                         "artifact. Without this the artifact's check_all_pass still "
                         "describes the all-sites configuration, which is not what the arm "
                         "runs -- the defect that refused all four GenomeOcean directions.")
    ap.add_argument("--tag", default="G9SITES")
    args = ap.parse_args()

    sub = load(args.substrate)
    if args.prefix is None:
        from bgcbench.model.substrate_config import for_substrate
        args.prefix = for_substrate(sub.family)["prefix"]
        print(f"prefix resolved from substrate ({sub.family}): {args.prefix}", flush=True)
    adapter = None
    if args.adapter:
        adapter = resolve_best(args.adapter)
        sub = attach_adapter(sub, adapter)
    # ⚠ THE WEIGHT STATE MUST BE THE ONE THE DIRECTION WAS DERIVED ON, and --finalize makes
    # this load-bearing: it REWRITES `check_all_pass` in the artifact, so a mismatch here
    # would stamp a gate measured on one model onto a direction belonging to another. A
    # direction describes its own model's activation geometry; `attach_direction` already
    # refuses that mismatch at generation time, but by then the artifact would carry a pass
    # that was never true of it. Checked before any measurement so a wrong --adapter costs
    # seconds rather than a full sweep.
    _art_adapter = None
    base = sub.model.model if sub.family == "evo2" else sub.model
    n_sites = len(attention_sites(base))

    art = torch.load(args.direction, map_location="cpu", weights_only=False)
    _art_adapter = art.get("adapter")
    if (_art_adapter or None) != (adapter or None):
        raise SystemExit(
            f"direction was derived on weight state {_art_adapter!r} but g9_sites loaded "
            f"{adapter!r}. Those are different activation spaces, and --finalize would write "
            f"a gate measured on the wrong model into the direction artifact. Pass the "
            f"--adapter the direction came from.")
    d_all = art["directions"]
    if d_all.shape[0] != n_sites:
        raise SystemExit(f"direction has {d_all.shape[0]} sites, model exposes {n_sites}")

    # ⚠ SELECTION READS val_A; THE CHECK READS val_B. This used to read the per-site cosines
    # straight out of `manipulation_check`, which is the reported SPEC 6.4 evidence -- so the
    # site set was chosen to maximise the very numbers that then licensed the arm. Refusing a
    # stale artifact rather than falling back is deliberate: a silent fallback to
    # `manipulation_check` restores the circularity and nothing downstream could see it.
    selr = art.get("selection_readout") or {}
    cos = selr.get("cosine_train_val") or []
    if not cos:
        raise SystemExit(
            f"{args.direction} has no `selection_readout` -- it predates the val A/B split "
            f"(2026-09-15). Selecting on `manipulation_check` means choosing the site set to "
            f"maximise the same fold the check is read on. Re-run run.derive_directions.")
    if selr.get("fold") != "val_A":
        raise SystemExit(f"selection_readout is on fold {selr.get('fold')!r}, expected 'val_A'")

    others = [c for c in ("TERPENE", "RIPP", "ARYLPOLYENE", "REDOX_COFACTOR")
              if c != args.target]
    va_t_a, va_t_b = D.val_halves(_load(SPLITS / args.target / "val.jsonl", args.target))
    if args.prefix == "taxonomy":
        from bgcbench.data import taxonomy
        _tbl = taxonomy.load_table()
        taxonomy.attach(va_t_a, _tbl)
        taxonomy.attach(va_t_b, _tbl)
    va_t = va_t_a          # every candidate-set measurement below is on half A

    from bgcbench.model.substrate_config import alpha_grid, max_alpha
    grid = alpha_grid(sub.family)
    if args.alpha is None:
        args.alpha = max_alpha(sub.family)
        print(f"probe alpha resolved from substrate ({sub.family}) generation grid: "
              f"{args.alpha}", flush=True)
    probe_alpha = args.alpha
    calib = None
    if args.target_kl is not None:
        calib = D.alpha_for_target_kl(sub, va_t, d_all, args.prefix, args.max_len_nt,
                                      target_kl=args.target_kl, limit=min(8, args.limit))
        probe_alpha = calib["alpha"]
        print(f"probe alpha calibrated to KL~{args.target_kl}: alpha={probe_alpha:.4f} "
              f"(realised KL {calib['realised_kl']:.3f})", flush=True)

    cands = candidate_sets(n_sites)
    print(f"{args.substrate}: {n_sites} attention sites, {len(cands)} candidate sets, "
          f"probe alpha {probe_alpha:.4f}\n", flush=True)
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
                               alpha=probe_alpha, limit=args.limit)
        fid = [cos[i] for i in live] if cos else []
        mean_fid = sum(fid) / len(fid) if fid else None
        min_fid = min(fid) if fid else None
        adm = bool(mean_fid is not None and mean_fid > 0.3 and min_fid > 0.0)
        rows.append({"name": name, "sites": idx, "live_sites": live,
                     "reach_kl_nats": kl["mean_kl_nats"],
                     "reach_kl_nats_per_nt": kl["mean_kl_nats_per_nt"],
                     "frac_argmax_changed": kl["frac_argmax_changed"],
                     "fidelity_mean_cosine": mean_fid, "min_cosine": min_fid,
                     "admissible": adm})
        print(f"{name:16s} {str(idx):22s} {kl['mean_kl_nats']:11.5f} "
              f"{(mean_fid if mean_fid is not None else 0):9.3f} "
              f"{(min_fid if min_fid is not None else 0):8.3f} {str(adm):>5s}")

    # ⚠ MAXIMISING REACH PICKS THE INPUT END OF THE STACK, NOT THE CLASS SIGNAL. Reach is
    # mean KL against unsteered, and perturbing an early site changes the input to every site
    # downstream of it -- so reach falls monotonically with depth for mechanical reasons that
    # have nothing to do with class content. Measured on GenomeOcean at alpha=1:
    # early_third 3.7-8.2, middle_third 0.5-1.7, late_third 0.03-0.15 nats/token, for every
    # one of the four classes. Meanwhile the RELATIVE class-difference norms are WEAKEST at
    # the early sites (0.05-0.22) and strongest at sites 6-13 (0.43-1.09). "Highest reach
    # among admissible" therefore selected against the sites actually carrying the class.
    # On Evo2's 4 sites the rule barely moves; on a 24-layer stack it decided the answer.
    #
    # The rule now is: among admissible sets that DEMONSTRABLY REACH THE OUTPUT, take the one
    # carrying the most class content. The reach bar is relative to the best admissible set,
    # so it is invariant to the probe magnitude and carries no unit across substrates -- an
    # absolute nats/token floor would mean different things on a 1 nt token and a 4.8 nt one.
    adm = [r for r in rows if r["admissible"]]
    best = None
    reach_floor = None
    if adm:
        top_reach = max(r["reach_kl_nats"] for r in adm)
        reach_floor = args.reach_floor_frac * top_reach
        eligible = [r for r in adm if r["reach_kl_nats"] >= reach_floor]
        # Falling back to the whole admissible set would silently restore the old rule.
        # It cannot happen -- the set achieving `top_reach` always clears its own fraction --
        # but assert it rather than trust it.
        if not eligible:
            raise SystemExit("no admissible set cleared the relative reach floor; that is "
                             "arithmetically impossible and means `rows` is malformed")
        best = max(eligible, key=lambda r: r["fidelity_mean_cosine"])
        for r in rows:
            r["eligible"] = bool(r["admissible"] and r["reach_kl_nats"] >= reach_floor)
        print(f"\nreach floor {args.reach_floor_frac:.2f} x {top_reach:.4f} = "
              f"{reach_floor:.4f} nats/token -> {len(eligible)}/{len(adm)} admissible sets "
              f"eligible; choosing on fidelity among those")

    OUT.mkdir(parents=True, exist_ok=True)
    out = {"gate": "G9-sites", "substrate": args.substrate, "target": args.target,
           "adapter": adapter, "probe_alpha": probe_alpha,
           "probe_alpha_requested": args.alpha, "probe_alpha_calibration": calib,
           "n_attention_sites": n_sites,
           "selection_fold": "val_A",
           "check_fold": "val_B",
           "reach_floor_frac": args.reach_floor_frac,
           "reach_floor_nats": reach_floor,
           "selection_fields": ["reach_kl_nats", "fidelity_mean_cosine", "min_cosine"],
           "endpoint_fields_read": [],
           "criterion": ("measured on val_A. admissible = mean steered-site train/val cosine "
                         "> 0.3 and no steered site anti-aligned; eligible = admissible AND "
                         "reach >= reach_floor_frac x the best admissible reach; among "
                         "eligible, the HIGHEST FIDELITY (mean cosine). Reach gates, fidelity "
                         "chooses -- maximising reach selects depth, not class content. "
                         "Never the endpoint (§2.4)."),
           "chosen": best["name"] if best else None,
           "chosen_sites": best["sites"] if best else None,
           "rows": rows, "code_version": code_version()}
    p = OUT / f"{args.tag}_{args.substrate}_{args.target}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nCHOSEN SITE SET: {out['chosen']} -> {out['chosen_sites']}")
    print(f"wrote {p}")

    if args.finalize:
        if best is None:
            raise SystemExit("--finalize with no admissible site set: there is nothing to "
                             "write back, and the direction stays failed.")
        _finalize(sub, args, art, best, va_t_b, others, grid)
    return 0


def _finalize(sub, args, art, best, va_t_b, others, grid) -> None:
    """Re-read SPEC 6.4 at the CHOSEN sites, on val_B, and write it into the artifact.

    ⚠ THIS IS THE STEP WHOSE ABSENCE REFUSED ALL FOUR GENOMEOCEAN DIRECTIONS. `attach_direction`
    gates generation on `check_all_pass` stored in the direction artifact, and that field was
    written once, for the all-24-sites configuration, before the site sweep ran. `chosen_sites`
    lived in a separate JSON that nothing read back. So TERPENE was refused at generation time
    on the strength of a configuration it does not use, while the 8-site configuration that
    passes sat in a file beside it.
    """
    import torch
    sites = best["sites"]
    per = max(1, args.check_limit // len(others))
    va_o_b: list[dict] = []
    for c in others:
        _a, b = D.val_halves(_load(SPLITS / c / "val.jsonl", c))
        va_o_b += b[:per]
    if args.prefix == "taxonomy":
        from bgcbench.data import taxonomy
        taxonomy.attach(va_o_b, taxonomy.load_table())

    print(f"\n--- SPEC 6.4 re-read at chosen sites {sites} on val_B ---", flush=True)
    d_all = art["directions"]
    chk = D.manipulation_check(sub, va_t_b, va_o_b, d_all, args.prefix, args.max_len_nt,
                               limit=args.check_limit, alpha=args.alpha, sites=sites)
    chk["fold"] = "val_B"
    mono = D.projection_vs_alpha(sub, va_t_b, d_all, args.prefix, args.max_len_nt,
                                 alphas=grid, limit=min(args.check_limit, 16), sites=sites)
    kl = D.kl_vs_unsteered(sub, va_t_b, d_all, args.prefix, args.max_len_nt,
                           alpha=args.alpha, limit=min(args.check_limit, 16), sites=sites)
    art["chosen_sites"] = list(sites)
    art["chosen_site_set"] = best["name"]
    art["manipulation_check"] = chk
    art["check_a_monotone"] = mono
    art["check_b_kl"] = kl
    art["check_alpha"] = float(args.alpha)
    art["alpha_grid"] = list(grid)
    art["check_all_pass"] = bool(mono["passes"] and kl["passes"] and chk["passes"])
    art["check_configuration"] = {
        "sites": list(sites), "site_set": best["name"], "alpha": float(args.alpha),
        "alpha_grid": list(grid), "selection_fold": "val_A", "check_fold": "val_B",
        "note": ("the check is read at the configuration the arm generates with. An "
                 "all-sites check on a subset-steered arm describes a different arm."),
    }
    print(f"(a) monotone at first steered site : "
          f"{'PASS' if mono['passes'] else 'FAIL'} "
          f"({mono['n_sites_monotone']}/{mono['n_sites_steered']} steered sites monotone)")
    print(f"(b) KL vs unsteered : {'PASS' if kl['passes'] else 'FAIL'}  "
          f"{kl['mean_kl_nats']:.5f} nats/token = {kl['mean_kl_nats_per_nt']:.5f} nats/NT, "
          f"argmax changed {kl['frac_argmax_changed']:.3f}")
    print(f"(c) reproduces on val_B : {'PASS' if chk['passes'] else 'FAIL'}  "
          f"mean cos {chk['mean_cosine']:.3f} min {chk['min_active_cosine']:+.3f}")
    print(f"MANIPULATION CHECK : {'PASS' if art['check_all_pass'] else 'FAIL'} "
          f"(all three required)")
    torch.save(art, args.direction)
    meta = {k: v for k, v in art.items() if k != "directions"}
    Path(args.direction).with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"rewrote {args.direction} with the chosen-site check", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
