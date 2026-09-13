"""G11: select a substrate's DECODING configuration (SPEC §12.A7, §12.A8).

⚠ THE CRITERION IS NOT THE ENDPOINT (SPEC §2.4). This script never runs antiSMASH and
never reads `detected`, `on_target`, `products` or `observed_classes`. It asks only whether
generated sequence has the STRUCTURAL STATISTICS of real sequence:

    distinct 21-mer fraction   a pure string property; collapses under truncation
    coding density (prodigal)  ORF coverage -- no cluster rule, no marker set
    median ORF length          ditto

The reference is REAL HELD-OUT SEQUENCE, never another arm and never the prior
implementation's preset. The selected configuration is the one whose medians are closest to
real, as mean absolute relative deviation over those three. Ties break toward the LESS
restrictive setting: a restriction that buys nothing is a confound with no benefit.

Why this gate exists: `top_k=4` was frozen once and applied to both substrates. Over Evo2's
4-letter alphabet it keeps 0.9999 of the probability mass; over GenomeOcean's 4,096-token
BPE vocabulary it keeps 0.1937, discarding 81% of the distribution at every step.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/data2/ds85/bgcbench")
SPLITS = ROOT / "splits"
OUT = ROOT / "g11"

K = 21
#: The three statistics the selection reads. Named here so the record of what was selected
#: on cannot drift from what was actually compared.
SELECTION_FIELDS = ("distinct_21mer_fraction", "coding_density", "median_orf_nt")


def _prodigal() -> str:
    """⚠ NOT the bare name. prodigal lives in this environment's bin, which is on PATH for
    an interactive shell but not necessarily for a subprocess launched from a script."""
    p = shutil.which("prodigal") or str(Path(sys.executable).parent / "prodigal")
    if not Path(p).exists():
        raise SystemExit(f"prodigal not found (tried PATH and {p})")
    return p


def distinct_kmer_fraction(seq: str, k: int = K) -> float | None:
    """Distinct k-mers / total k-mer positions. 1.0 = never repeats, low = degenerate.

    This is the statistic that top-k truncation destroys: sampling from 4 of 4,096 tokens
    forces the model back through the same few tokens and the output starts repeating.

    ⚠ IT IS A PATHOLOGY DETECTOR, NOT A FIDELITY MEASURE, and the selection rule depends on
    knowing the difference. Measured on 800 nt: uniform-random 1.000, a real TERPENE core
    1.000, `ATGC` repeated 0.005. Real and random are INDISTINGUISHABLE here, so this field
    contributes nothing between two healthy configurations and everything against a
    degenerate one -- coding density and ORF length are what separate the healthy ones.
    """
    s = seq.upper()
    n = len(s) - k + 1
    if n <= 0:
        return None
    return len({s[i:i + k] for i in range(n)}) / n


def prodigal_stats(seqs: list[str]) -> dict:
    """Coding density and ORF lengths from prodigal alone -- NOT from antiSMASH.

    `-p meta` because these are short anonymous fragments, not whole genomes; prodigal's
    single-genome training mode needs ~20 kb and would refuse or train badly on a 1 kb
    generation, which would read as "no ORFs" and look like a decoding failure.
    """
    with tempfile.TemporaryDirectory() as td:
        fna = Path(td) / "in.fna"
        with fna.open("w") as fh:
            for i, s in enumerate(seqs):
                if len(s) >= 60:                       # prodigal's own floor
                    fh.write(f">s{i}\n{s}\n")
        gff = Path(td) / "out.gff"
        r = subprocess.run([_prodigal(), "-i", str(fna), "-o", str(gff), "-f", "gff",
                            "-p", "meta", "-q"], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"prodigal failed: {r.stderr[:400]}")
        per: dict[str, int] = {}
        lens: list[int] = []
        for line in gff.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t")
            if len(f) < 5 or f[2] != "CDS":
                continue
            L = int(f[4]) - int(f[3]) + 1
            lens.append(L)
            per[f[0]] = per.get(f[0], 0) + L
    total_nt = sum(len(s) for s in seqs if len(s) >= 60)
    return {"coding_density": (sum(per.values()) / total_nt) if total_nt else None,
            "median_orf_nt": st.median(lens) if lens else None,
            "n_orfs": len(lens)}


def summarise(seqs: list[str], hits: list[bool] | None = None) -> dict:
    dk = [x for x in (distinct_kmer_fraction(s) for s in seqs) if x is not None]
    out = {"n": len(seqs),
           "median_len_nt": st.median([len(s) for s in seqs]) if seqs else None,
           "distinct_21mer_fraction": st.median(dk) if dk else None}
    out.update(prodigal_stats(seqs))
    if hits is not None:
        out["hit_eos_rate"] = sum(hits) / len(hits) if hits else None
    return out


def deviation(cand: dict, real: dict) -> float | None:
    """Mean absolute RELATIVE deviation over SELECTION_FIELDS. Relative, because coding
    density (~0.9) and ORF length (~500 nt) cannot be averaged on an absolute scale."""
    ds = []
    for f in SELECTION_FIELDS:
        c, r = cand.get(f), real.get(f)
        if c is None or not r:
            return None
        ds.append(abs(c - r) / abs(r))
    return sum(ds) / len(ds)


#: (temperature, top_k, top_p). The CURRENT frozen Evo2 value is included as the first rung
#: so the damage it does on this substrate is measured rather than assumed.
DEFAULT_GRID = [(1.0, 4, 1.0), (1.0, 64, 1.0), (1.0, 256, 1.0),
                (1.0, 0, 1.0), (1.0, 0, 0.95), (0.9, 0, 1.0)]


def restrictiveness(cfg: tuple) -> tuple:
    """Lower is LESS restrictive. Used only to break ties (§12.A8)."""
    t, k, p = cfg
    return (0 if k == 0 else 1, -k if k else 0, -p, abs(t - 1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="go-4b")
    ap.add_argument("--adapter", default=None,
                    help="weight state to sweep on. Use the POOLED arm so no class is "
                         "favoured, exactly as G6/G6b were run.")
    ap.add_argument("--class-for-reference", default="TERPENE",
                    help="which held-out split supplies the REAL reference statistics")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--prefix", choices=["none", "taxonomy"], default="none")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from bgcbench.model.generate import ArmSpec, GenConfig, _run
    from bgcbench.model.load import attach_adapter, load, resolve_best
    from bgcbench.provenance import code_version

    sub = load(args.substrate)
    adapter = None
    if args.adapter:
        adapter = resolve_best(args.adapter)
        sub = attach_adapter(sub, adapter)
        print(f"swept on adapter {adapter}", flush=True)

    real_recs = [json.loads(l) for l in
                 open(SPLITS / args.class_for_reference / "val.jsonl")][:args.n]
    real = summarise([r["sequence"] for r in real_recs])
    print(f"REAL {args.class_for_reference} val (n={real['n']}): "
          f"distinct21={real['distinct_21mer_fraction']:.4f} "
          f"coding={real['coding_density']:.4f} orf={real['median_orf_nt']} nt", flush=True)

    cfg = GenConfig.frozen()
    cfg.batch_size = args.batch_size
    rows = []
    for (t, k, p) in DEFAULT_GRID:
        arm = ArmSpec(arm_id=f"G11_t{t}_k{k}_p{p}", weight_state="sweep",
                      prefix=args.prefix, adapter_path=adapter,
                      temperature=t, top_k=k, top_p=p)
        seqs, hits = _run(sub, arm, [""] * args.n, cfg)
        seqs = [sub.clean(sub.truncate_at_terminator(s)[0]) for s in seqs]
        r = summarise(seqs, hits)
        r.update({"temperature": t, "top_k": k, "top_p": p})
        r["deviation_from_real"] = deviation(r, real)
        rows.append(r)
        print(f"  t={t} k={k:<4} p={p}  len={r['median_len_nt']:<6} "
              f"distinct21={(r['distinct_21mer_fraction'] or 0):.4f} "
              f"coding={(r['coding_density'] or 0):.4f} "
              f"orf={r['median_orf_nt']}  dev={r['deviation_from_real']}", flush=True)

    ok = [r for r in rows if r["deviation_from_real"] is not None]
    if not ok:
        raise SystemExit("no configuration produced measurable statistics")
    best = min(ok, key=lambda r: (round(r["deviation_from_real"], 4),
                                  restrictiveness((r["temperature"], r["top_k"], r["top_p"]))))
    art = {"gate": "G11", "substrate": args.substrate, "substrate_family": sub.family,
           "adapter": adapter, "prefix": args.prefix, "n_per_config": args.n,
           "reference_class": args.class_for_reference, "reference_real": real,
           "selection_fields": list(SELECTION_FIELDS),
           "selection_rule": "min mean |relative deviation| from REAL held-out sequence; "
                             "ties break toward the less restrictive setting",
           "endpoint_read": "NONE — antiSMASH is never invoked by this gate (SPEC 2.4)",
           "grid": rows, "selected": {k: best[k] for k in ("temperature", "top_k", "top_p")},
           "selected_deviation": best["deviation_from_real"],
           "code_version": code_version()}
    OUT.mkdir(parents=True, exist_ok=True)
    name = args.name or f"G11_{args.substrate}"
    (OUT / f"{name}.json").write_text(json.dumps(art, indent=1))
    print(f"\nSELECTED temperature={best['temperature']} top_k={best['top_k']} "
          f"top_p={best['top_p']}  (deviation {best['deviation_from_real']:.4f})")
    print(f"wrote {OUT / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
