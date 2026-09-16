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
import json as _json
import statistics as st
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path("/data2/ds85/bgcbench")
OUT = ROOT / "g9"


def _wait_for_gpu(need_mib: int, tag: str, poll_s: int = 60) -> None:
    """Block until `need_mib` is free on the card.

    ⚠ PER ARM, NOT PER PHASE. The pipeline checked free memory once before the sweep, but
    every alpha spawns a FRESH subprocess that reloads the model, so the memory checked for
    was long gone by the third arm. Measured consequence on 2026-09-15: all 24 Evo2 alpha
    arms died with `OutOfMemoryError: tried to allocate 4.26 GiB, 1.72 GiB free` while two
    other processes held 60.7 GB. The sweep then had no alpha=0 baseline and correctly
    refused to choose an alpha -- a whole phase lost to a guard at the wrong granularity.
    """
    import subprocess as _sp
    import time as _t
    while True:
        try:
            free = int(_sp.run(["nvidia-smi", "--query-gpu=memory.free",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True).stdout.split()[0])
        except Exception:
            return                      # no nvidia-smi: do not block a CPU-only environment
        if free >= need_mib:
            return
        print(f"  waiting for GPU: need {need_mib} MiB, {free} free ({tag})", flush=True)
        _t.sleep(poll_s)


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
            # ⚠ TOKENISE PER FAMILY. `sub.tokenizer.tokenize` is vortex's byte-level API and
            # returns ints; GenomeOcean's HF tokenizer returns STRING tokens, so int() raises.
            # directions.py and g2_health.py both branch here; this one did not.
            if sub.family == "evo2":
                ids = [int(x) for x in sub.tokenizer.tokenize(s)]
            else:
                ids = sub.tokenizer(s)["input_ids"]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            lo = _unwrap(base(x))
            lp = F.log_softmax(lo.float(), dim=-1)[:, :-1]
            tgt = x[:, 1:]
            out.append(float(-lp.gather(-1, tgt.unsqueeze(-1)).mean()))
    return out


def _same_config(d, alpha: float, want_sites, want_adapter) -> bool:
    """Does this run directory hold a measurement of the configuration we are about to run?

    ⚠ MATCHING ON THE ARM NAME IS NOT ENOUGH, AND THE FIRST VERSION OF THE REUSE PATH DID
    EXACTLY THAT. The glob `runs/stage1_<substrate>_<arm>_*` ends in a wildcard that swallows
    the realised-config HASH, so two runs of the same arm name at DIFFERENT injection sites
    look identical to it. The patch shipped with a comment asserting the hash made collision
    impossible; the glob it shipped with ignored the hash. Measured consequence: Evo2's phase C
    "completed" in 2.5 minutes on 2026-09-15 by reusing the superseded sweep -- RIPP's alpha=0.4
    directory records `intervention_site_subset=[0]`, the site set §12.A9 replaced, while the
    corrected protocol had chosen `all`=[0,1,2,3]. Every alpha it reported was an alpha for a
    different intervention, and phase D then ran on them.

    The fields compared are the ones that make it a different arm: the injection sites, the
    magnitude, and the weight state. A run whose report cannot be read is not reused.
    """
    try:
        rel = _json.loads((d / "report.json").read_text())["realised"]
    except Exception:
        return False
    got_alpha = rel.get("intervention_alpha")
    if alpha == 0.0:
        # ⚠ AT ZERO MAGNITUDE THERE IS NO INJECTION, SO THE SITE SET IS NOT A PROPERTY OF THE
        # ARM. g9_alpha passes no --direction for this rung, so the run records
        # intervention_site_subset=None however many sites the sweep steers at alpha>0.
        # Comparing sites here rejected every alpha=0 baseline on disk -- the exact case the
        # reuse path was written for -- and all four Evo2 classes came back
        # "no alpha=0 baseline; cannot judge degradation" after a 1h54m sweep whose alpha>0
        # rungs were all fine. The site check belongs only where a direction is attached.
        if got_alpha not in (None, 0.0):
            return False
    else:
        if got_alpha is None or abs(float(got_alpha) - float(alpha)) > 1e-9:
            return False
        got_sites = rel.get("intervention_site_subset")
        got_sites = sorted(int(i) for i in got_sites) if got_sites else None
        if got_sites != want_sites:
            return False
    if (rel.get("adapter") or None) != (want_adapter or None):
        return False
    return True


def _read_arm(d, a: float) -> dict:
    """Health fields for one sweep rung, read off a run directory.

    ⚠ ONLY these fields. detected / on_target / products are deliberately not read (§2.4):
    alpha is chosen against generation health, never against the benchmark endpoint.
    """
    cd, ln, eos, seqs = [], [], [], []
    for line in open(d / "scored.jsonl"):
        r = json.loads(line)
        if r.get("coding_density") is not None:
            cd.append(r["coding_density"])
        ln.append(r.get("seq_len") or 0)
        eos.append(bool(r.get("hit_eos")))
        seqs.append(r.get("sequence") or "")
    return {"alpha": a, "rc": 0, "run_dir": str(d), "n": len(ln),
            "median_coding_density": st.median(cd) if cd else None,
            "median_len": st.median(ln) if ln else None,
            "hit_eos_rate": (sum(eos) / len(eos)) if eos else None,
            "_seqs": seqs[:16]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="evo2-1b")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--row-class", required=True)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4],
                    help="GRID LOWERED 2026-09-15 from [0, 0.5, 1, 2, 4, 8]. The old grid "
                         "bottomed out ABOVE the usable range on every class that produced a "
                         "readable sweep: against a healthy alpha=0 baseline (coding density "
                         "0.9967 on ARYLPOLYENE, matching real cores) the SMALLEST rung, 0.5, "
                         "already dropped coding to 0.1678 and killed termination outright "
                         "(hit_eos 0.68 -> 0.00). REDOX agreed on a clean 6/6 sweep. So the "
                         "sweep could only ever report no-alpha, whether or not steering works "
                         "at all here. 0.0 stays as the mandatory baseline: without it the gate "
                         "cannot judge degradation and returns None.")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tol-coding", type=float, default=0.10)
    ap.add_argument("--tol-nll", type=float, default=0.10)
    ap.add_argument("--tol-eos", type=float, default=0.15,
                    help="max ABSOLUTE DROP in hit_eos rate. One-sided: steering that stops "
                         "the model terminating has broken generation even if density holds, "
                         "but terminating more often is not damage.")
    ap.add_argument("--prefix", choices=["none", "taxonomy"], default=None,
                    help="omit to use the substrate's own setting (SPEC 14A): evo2 -> taxonomy, "
                         "genomeocean -> none. ⚠ THIS DEFAULTED TO 'taxonomy' FOR BOTH, so a "
                         "manual GenomeOcean invocation that forgot the flag silently fed it a "
                         "GTDB lineage -- a format it was never pretrained on (SPEC 15.7) -- and "
                         "nothing raised.")
    ap.add_argument("--need-mib", type=int, default=None,
                    help="free GPU memory required before EACH alpha arm is launched. "
                         "Checked per arm because every alpha reloads the model in a new "
                         "process; a once-per-phase check let all 24 arms OOM on 2026-09-15. "
                         "RAISED 26000 -> 34000: the guard must reserve for the arm's PEAK, not "
                         "its entry footprint. At 26000 it admitted RIPP's sweep, which then grew "
                         "to 24.15 GiB and OOMed with 3.45 free, losing all 6 arms. Measured peak "
                         "for an Evo2 generation arm: 29,690 MiB. ⚠ THEN MADE PER SUBSTRATE: "
                         "34,000 was Evo2's number applied to both, a 2.5x over-estimate for "
                         "GenomeOcean (measured 13,430-16,830 MiB). On a card where another user "
                         "holds ~40 GB, that stalls a GenomeOcean sweep between arms waiting for "
                         "memory it never needed. Omit to use the substrate's measured value.")
    ap.add_argument("--sites", nargs="+", type=int, default=None,
                    help="⚠ INJECTION SITES, forwarded to run.arm. G9 is site AND magnitude "
                         "swept together (SPEC 6); an alpha measured at the default "
                         "every-site subset is an alpha for a DIFFERENT intervention than "
                         "one measured at the sites G9's site half chose.")
    ap.add_argument("--random-direction", type=int, default=None, metavar="SEED",
                    help="sweep the SPEC 6.3 random control instead of the derived direction. "
                         "⚠ The control needs its OWN ceiling: at TERPENE's alpha the random "
                         "vector broke generation while the real one did not (FINDINGS 17.3), "
                         "so a control run at the arm's alpha is not health-matched and its "
                         "zero cannot be read.")
    ap.add_argument("--tag", default="G9")
    args = ap.parse_args()
    # ⚠ see derive_directions: defaulting to "taxonomy" for both substrates silently fed
    # GenomeOcean a lineage. This script shells out to run.arm before loading a model, so the
    # family is resolved from the substrate id rather than a loaded Substrate.
    if args.prefix is None:
        from bgcbench.model.substrate_config import for_substrate
        fam = "evo2" if args.substrate.startswith("evo2") else "genomeocean"
        args.prefix = for_substrate(fam)["prefix"]
        print(f"prefix resolved from substrate ({fam}): {args.prefix}", flush=True)

    if args.need_mib is None:
        from bgcbench.model.substrate_config import gpu_mib
        fam = "evo2" if args.substrate.startswith("evo2") else "genomeocean"
        args.need_mib = gpu_mib(fam)
        print(f"gpu threshold resolved from substrate ({fam}): {args.need_mib} MiB", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in args.alphas:
        arm = f"{args.tag}_a{a}_{args.row_class}"
        # ⚠ REUSE AN EXISTING MEASUREMENT RATHER THAN DYING ON IT. `run_arm` refuses to
        # overwrite a run directory -- a deliberate guard -- so a rung already measured makes
        # the subprocess exit 1, the row degrades to {"alpha": a, "rc": 1}, and if that rung
        # is alpha=0 the whole sweep returns chosen_alpha=None with "no alpha=0 baseline".
        # That is exactly what happened to all four GenomeOcean classes: their alpha=0 arms
        # were measured at 16:00-16:15 during the attempt whose steered rungs were refused by
        # the (then broken) §6.4 gate, so the baseline was on disk and unreadable.
        #
        # ⚠ SAFE BECAUSE THE DIRECTORY NAME CARRIES THE REALISED CONFIG HASH. `run_arm` builds
        # it from the realised block, which includes `intervention_active_sites`,
        # `intervention_site_subset` and the alpha -- so a different site set or magnitude
        # CANNOT collide with this name. The alpha=0 rung attaches no direction at all, which
        # is why its hash is identical across site sets: at zero magnitude there is no
        # injection, so the site set is not a property of that arm.
        want_sites = sorted(int(i) for i in args.sites) if args.sites else None
        prior = [d for d in ROOT.glob(f"runs/stage1_{args.substrate}_{arm}_*")
                 if (d / "scored.jsonl").exists()
                 and _same_config(d, a, want_sites, args.adapter)]
        if prior:
            d = sorted(prior, key=lambda p: p.stat().st_mtime)[-1]
            print(f"\n=== alpha {a}: REUSING existing measurement {d}", flush=True)
            rows.append(_read_arm(d, a))
            continue
        _wait_for_gpu(args.need_mib, f"{args.tag} alpha={a}")
        # ⚠ --substrate MUST BE FORWARDED. run.arm defaults it to "evo2-1b", so without this
        # every GenomeOcean alpha sweep launched EVO2 carrying a GenomeOcean adapter. It
        # fails loudly at peft attach rather than producing a wrong number, but phase C could
        # never produce a GenomeOcean alpha at all.
        cmd = [sys.executable, "-m", "bgcbench.run.arm",
               "--substrate", args.substrate, "--arm", arm,
               "--row-class", args.row_class, "--n", str(args.n),
               "--prefix", args.prefix, "--stage", "stage1", "--off-frozen",
               "--batch-size", "50"]
        if args.sites:
            cmd += ["--sites"] + [str(i) for i in args.sites]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        if a > 0:
            cmd += ["--direction", args.direction, "--alpha", str(a)]
            if args.random_direction is not None:
                cmd += ["--random-direction", str(args.random_direction)]
        print(f"\n=== alpha {a}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"alpha {a} failed rc={rc}", flush=True)
            rows.append({"alpha": a, "rc": rc})
            continue
        d = sorted(ROOT.glob(f"runs/stage1_{args.substrate}_{arm}_*"),
                   key=lambda p: p.stat().st_mtime)[-1]
        rows.append(_read_arm(d, a))

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
            # ⚠ CODING DENSITY IS ONE-SIDED. Only a DROP is degradation; more gene-like
            # output is not a reason to reject an alpha. The first version of this used
            # abs(), and it rejected W0's alpha=4 for coding density 0.3153 against a
            # baseline of 0.2044 -- i.e. for being 54% BETTER.
            dc = (base["median_coding_density"] - r["median_coding_density"]) / base["median_coding_density"]
            dc = max(0.0, dc)
            # self-NLL stays two-sided: a rise is incoherence, a collapse is degeneracy.
            dn = abs(r["median_self_nll"] - base["median_self_nll"]) / base["median_self_nll"]
            # termination is a third health axis -- on W2 the alpha=0 arm stops 84% of the
            # time and alpha=0.5 stops 4%, which no coding-density tolerance would catch on
            # its own.
            # ⚠ ONE-SIDED, like coding density. Degradation is the model FAILING to stop
            # and running to the budget -- measured on W2, alpha=0.5 takes hit_eos 0.840 to
            # 0.040 with 90% of generations at the full 8,192. Terminating MORE is not
            # damage: at alpha=0.3 hit_eos is 1.000 with no truncation (min length 1,002,
            # nothing under 500 nt), median 1,934 against real TERPENE cores at 1,375, and
            # the highest coding density in the sweep (0.986). A two-sided rule rejected
            # that arm -- the healthiest one measured -- for being too good.
            de = max(0.0, (base.get("hit_eos_rate") or 0) - (r.get("hit_eos_rate") or 0))
            r["rel_coding_drop"], r["rel_nll_change"], r["eos_drop"] = dc, dn, de
            r["admissible"] = bool(dc <= args.tol_coding and dn <= args.tol_nll
                                   and de <= args.tol_eos)
            if r["admissible"]:
                adm.append(r["alpha"])
        chosen = max(adm) if adm else None
        why = (f"largest alpha with coding-density DROP <= {args.tol_coding} (one-sided), "
               f"|Δself-NLL| <= {args.tol_nll} (relative) and hit_eos DROP <= {args.tol_eos} "
               f"(absolute, one-sided), all against alpha=0")

    art = {"gate": "G9", "row_class": args.row_class, "direction": args.direction,
           "random_direction_seed": args.random_direction,
           "adapter": args.adapter, "alphas": args.alphas, "n_per_alpha": args.n,
           "tolerance": {"coding": args.tol_coding, "nll": args.tol_nll},
           "selection_fields": ["median_coding_density", "median_self_nll",
                                "hit_eos_rate", "distinct_21mer_frac"],
           "endpoint_fields_read": [],
           "criterion": why, "chosen_alpha": chosen, "sites": args.sites,
           "rows": rows}
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
