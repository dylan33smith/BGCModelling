#!/usr/bin/env python
"""Score generated sequences with antiSMASH — the gold-standard is_bgc / correct_class gate.

CPU-only and parallel, so it can run alongside a GPU job.

WHICH RUNS ARE WORTH SCORING. Only generations made with CORRECTED directions
(`steer_scaling == "class_units"`, i.e. build_steer_dirs.py output). The earlier
`steer_titration` (beta_raw_diffmean) and `steer_magnitude` (delta_norm_absolute on the legacy
sidecar) used vectors that are ~the sequence-length axis, sign-inverted for NRPS<->PKS and
ECTOINE<->TERPENE. Scoring those would measure "does the WRONG vector produce class" — already
answered 0/30 by the alpha sweep — and would put uninterpretable rows next to real ones in the
same table, which is how the original 0/30 misled this project for weeks. `--allow-legacy`
overrides, but the run is then labelled `legacy_direction=True` in the output.

DATABASE PATH. antiSMASH's prerequisite DBs are NOT in the package directory (that holds only a
README); they live at /data2/ds85/antismash_db. Calling check_antismash() without
`databases_dir` silently returns skipped=True with "prerequisite DBs not downloaded", which looks
exactly like a broken install. Always pass it.

FIELD SEMANTICS (from evaluation.py):
  is_bgc        <- antismash.detected
  correct_class <- antismash.class_match (expected class present in mapped_classes)
  skipped=True  <- antiSMASH could not run; NOT a negative result. Counted separately.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DB_DEFAULT = "/data2/ds85/antismash_db"


_CLASS_MAP = None


def _get_class_map(path: str):
    """antiSMASH emits product names like 'T1PKS'/'lanthipeptide-class-i'; our vocabulary is
    'PKS'/'RIPP'. Without this map, `mapped_classes` is just the raw product uppercased and
    `class_match` silently fails for most classes — measured: real held-out cores score
    correct_class 0.19-0.25 WITHOUT the map vs the ~0.97 the suite was calibrated to WITH it.
    Forgetting it looks exactly like a model that cannot produce the right class."""
    global _CLASS_MAP
    if _CLASS_MAP is None:
        # Use the project's own loader. A hand-rolled YAML read here looked for a "map" key
        # while the file uses "mappings", silently yielding a 2-entry map and turning
        # correct_class into noise. Reuse beats reimplementation for exactly this reason.
        from bgc_pipeline.class_map import load_class_map
        _CLASS_MAP, _ = load_class_map(Path(path))
    return _CLASS_MAP


def _score_one(payload):
    """Worker: (idx, sequence, expected_class, db, class_map_path) -> verdict dict."""
    from bgc_pipeline.evaluation import check_antismash, EvalResourceError
    i, seq, expected, db, cmpath = payload
    if not seq or len(seq) < 200:
        return {"i": i, "too_short": True, "skipped": True}
    try:
        r = check_antismash(seq, expected_class=expected or "", databases_dir=db,
                            class_map=_get_class_map(cmpath))
    except EvalResourceError:
        # NEVER swallow this. EvalResourceError subclasses RuntimeError, so a bare
        # `except Exception` catches it and converts a CONFIGURATION failure back into a silent
        # per-sequence skip -- exactly the behaviour the strict-resource work was added to stop.
        # A misconfigured run must die on the first sequence, not print 0.000 for every arm.
        raise
    except Exception as e:                      # a crash on ONE sequence must not kill the sweep
        return {"i": i, "skipped": True, "error": str(e)[:200]}
    return {"i": i, "skipped": bool(r.get("skipped", False)),
            "detected": bool(r.get("detected", False)),
            "class_match": bool(r.get("class_match", False)),
            "mapped_classes": r.get("mapped_classes", []),
            "reason": r.get("reason", "")}


def _expected_class(rec: dict, mode: str) -> str:
    """Which class counts as 'correct' for this record.

    For a cross-class override run the answer is the STEER TARGET, not `compound_class`
    (which carries the seed/tag class). Getting this wrong silently scores the experiment
    against the class we were steering AWAY from.
    """
    if mode == "target":
        return rec.get("steer_target_class") or rec.get("compound_class") or ""
    if mode == "seed":
        return rec.get("seed_class") or rec.get("compound_class") or ""
    return rec.get("compound_class") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path, nargs="+", help="Generated .jsonl file(s).")
    ap.add_argument("--databases-dir", default=DB_DEFAULT)
    ap.add_argument("--class-map", default=str(REPO / "config" / "compound_class_map.yaml"),
                    help="antiSMASH product -> our class vocabulary. REQUIRED for a meaningful "
                         "correct_class; without it real cores score ~0.2 instead of ~0.97.")
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--expected", choices=["compound_class", "target", "seed"],
                    default="compound_class",
                    help="'target' = steer_target_class (use for cross-class override runs).")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--allow-legacy", action="store_true",
                    help="Score runs built with the legacy (length-axis) directions anyway.")
    args = ap.parse_args()

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    tasks, meta = [], []
    for p in args.jsonl:
        recs = [json.loads(l) for l in p.open() if l.strip()]
        if not recs:
            print(f"[as] {p.name}: empty, skipping", file=sys.stderr)
            continue
        scaling = recs[0].get("steer_scaling")
        legacy = scaling in ("beta_raw_diffmean", "alpha_refnorm", "delta_norm_absolute")
        if legacy and not args.allow_legacy:
            print(f"[as] REFUSING {p.name}: steer_scaling={scaling} is a LEGACY (length-axis) "
                  f"direction. Scoring it produces uninterpretable rows. Use --allow-legacy to "
                  f"override.", file=sys.stderr)
            continue
        for r in recs:
            exp = _expected_class(r, args.expected)
            tasks.append((len(tasks), r.get("sequence", ""), exp, args.databases_dir,
                          args.class_map))
            meta.append({"arm": p.stem, "expected": exp, "legacy_direction": legacy,
                         "seed_class": r.get("seed_class", ""),
                         "target_class": r.get("steer_target_class", ""),
                         "compound_class": r.get("compound_class", ""),
                         "dose": r.get("steer_class_units", r.get("steer_beta",
                                       r.get("steer_delta_norm", ""))),
                         "dir_prefix": r.get("steer_dir_prefix", "real") or "real",
                         "length": len(r.get("sequence", "") or "")})
    if not tasks:
        print("[as] nothing to score", file=sys.stderr)
        return 1

    print(f"[as] scoring {len(tasks)} sequences with {args.workers} workers "
          f"(~5 s each serial)", flush=True)
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, out in enumerate(ex.map(_score_one, tasks, chunksize=1)):
            results[out["i"]] = out
            if (k + 1) % 25 == 0:
                print(f"[as]   {k + 1}/{len(tasks)}", flush=True)

    with args.out_tsv.open("w") as fh:
        fh.write("arm\tdose\tdir_prefix\texpected\tseed_class\ttarget_class\tlength\t"
                 "skipped\tis_bgc\tcorrect_class\tmapped_classes\tlegacy_direction\n")
        for m, r in zip(meta, results):
            fh.write(f"{m['arm']}\t{m['dose']}\t{m['dir_prefix']}\t{m['expected']}\t"
                     f"{m['seed_class']}\t{m['target_class']}\t{m['length']}\t"
                     f"{int(r['skipped'])}\t{int(r.get('detected', False))}\t"
                     f"{int(r.get('class_match', False))}\t"
                     f"{','.join(r.get('mapped_classes', []))}\t{int(m['legacy_direction'])}\n")

    # ---- summary. SKIPPED sequences are reported separately, never counted as negatives. ----
    agg = collections.defaultdict(lambda: {"n": 0, "skip": 0, "bgc": 0, "cls": 0})
    for m, r in zip(meta, results):
        a = agg[m["arm"]]
        a["n"] += 1
        if r["skipped"]:
            a["skip"] += 1
            continue
        a["bgc"] += int(r.get("detected", False))
        a["cls"] += int(r.get("class_match", False))
    print(f"\n{'arm':>16} {'n':>5} {'scored':>7} {'skipped':>8} {'is_bgc':>9} {'correct_class':>15}")
    for arm in sorted(agg):
        a = agg[arm]
        ok = a["n"] - a["skip"]
        print(f"{arm:>16} {a['n']:>5} {ok:>7} {a['skip']:>8} "
              f"{(a['bgc'] / ok if ok else 0):>9.3f} {(a['cls'] / ok if ok else 0):>15.3f}")
    if any(a["skip"] for a in agg.values()):
        print("\n  NOTE: skipped = antiSMASH could not run. NOT a negative result — excluded from")
        print("        the rates above rather than counted as failures.")
    print(f"\n[as] per-sequence -> {args.out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
