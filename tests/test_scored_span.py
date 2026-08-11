"""Guard WHAT SPAN GETS SCORED — is the seed inside the sequence antiSMASH sees?

WHY THIS EXISTS. On 2026-08-11 the guided-decoding result was reported with the conclusion that
`correct_class` on seeded generations was confounded: the seed is a real fragment of a
target-class cluster, so antiSMASH would recognise the seed and report its class regardless of
what the model wrote. The evidence offered was that `correct_class` agreed with `is_bgc` on
117/120 records.

THAT CONCLUSION WAS WRONG, and the error survived a five-reviewer adversarial pass. Both
generators score the CONTINUATION ONLY:

  * `guided_generate.py` starts `seq = ""` and only ever does `seq += cands[pick]`; the seed goes
    into `prompt_seqs` and never into the stored record.
  * `seed_generate.py` stores `extract_sequence(...)`, which is the generation with the prompt
    already stripped.

Checked against every seeded run on disk: **0 of 1512 records begin with their seed**, and only
8 contain even a single 60-mer of it. The confound does not exist and never did.

The real reason `correct_class` tracked `is_bgc` so closely is a fact about the GENERATIONS, not
the instrument: on real held-out cores truncated to 3 kb, 31.4% of antiSMASH detections are the
WRONG class, so the "detected but off-class" cell is well populated and the metric has real
discriminative power. Our generations simply land on-class when they land at all.

So this file asserts the property that was assumed, argued about, and never checked. A one-line
change in either generator -- prepending the seed for context, say -- would silently reintroduce
the confound that was hypothesised here, and every downstream class number would inflate with no
error anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evo2" / "scripts"))

RUNS = Path("/data2/ds85/bgcmodel_runs")
SPLITS = Path("/data2/ds85/bgcmodel_data/splits_core")


def _load_cores():
    cores = {}
    for split in ("val", "test", "train"):
        p = SPLITS / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            k = r.get("accession") or r.get("id")
            if k:
                cores[k] = r["sequence"].upper()
    return cores


def test_generators_never_put_the_seed_into_the_stored_sequence():
    """Source-level guard: works with no run artifacts present."""
    gg = (REPO / "evo2" / "scripts" / "guided_generate.py").read_text()
    assert 'seq = ""' in gg, "guided_generate no longer starts the scored sequence empty"
    assert "seq += cands[pick]" in gg, (
        "guided_generate no longer builds the scored sequence purely from generated chunks — if "
        "the seed is now concatenated in, every class number it produces is confounded")
    assert "tax + seed_dna + seq" in gg, (
        "the seed is expected to reach the model through the PROMPT only; that construction is gone")

    sg = (REPO / "evo2" / "scripts" / "seed_generate.py").read_text()
    assert "extract_sequence(_gen_sequences(out)[0])" in sg, (
        "seed_generate no longer stores the prompt-stripped generation")
    print("PASS scored-span: both generators build the scored sequence from generated text only")


def test_no_stored_record_begins_with_its_seed():
    """Data-level guard against every seeded run on disk. Skips if the runs are absent."""
    cores = _load_cores()
    if not cores or not RUNS.exists():
        print("SKIP scored-span: no run artifacts / splits on this host")
        return
    n = begins = contains = 0
    offenders = []
    for f in sorted(RUNS.glob("*/*.jsonl")):
        try:
            recs = [json.loads(l) for l in f.open() if l.strip()]
        except (json.JSONDecodeError, OSError):
            continue
        for r in recs:
            acc, sn = r.get("seed_accession"), r.get("seed_nt") or 0
            if acc not in cores or sn <= 0:
                continue
            n += 1
            seed, gen = cores[acc][:sn], r.get("sequence", "").upper()
            if gen[:sn] == seed:
                begins += 1
                offenders.append(f"{f.parent.name}/{f.name}")
            if any(seed[i:i + 60] in gen for i in range(0, max(1, len(seed) - 60), 60)):
                contains += 1
    if n == 0:
        print("SKIP scored-span: no seeded records found on disk")
        return
    assert begins == 0, (
        f"{begins}/{n} stored sequences BEGIN with their own seed (in {sorted(set(offenders))}). "
        f"antiSMASH would then be scoring real class-X DNA we supplied, and every correct_class "
        f"number from those runs is confounded.")
    # Verbatim carry-over of a seed fragment is a separate, milder issue (novelty covers it), but
    # a sudden jump would mean the model started copying rather than continuing.
    frac = contains / n
    assert frac < 0.05, (f"{contains}/{n} = {frac:.1%} of generations contain a 60-mer of their "
                         f"seed — copying, not continuing")
    print(f"PASS scored-span: 0/{n} stored sequences contain their seed as a prefix "
          f"({contains} contain any 60-mer, {frac:.1%})")


def main() -> int:
    for t in (test_generators_never_put_the_seed_into_the_stored_sequence,
              test_no_stored_record_begins_with_its_seed):
        t()
    print("\nALL SCORED-SPAN TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
