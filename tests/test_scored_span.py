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
    n = begins = contains = coincidental = eligible = 0
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
                # SHORT SEEDS COINCIDE BY CHANCE. This guard was written when seeds were ~500 nt,
                # where a prefix match is impossible unless the seed was prepended. Phase-3 uses
                # 4-8 nt seeds (L*=8), and both seeds and generations start at a gene, so both are
                # ATG-enriched: at 4 nt, 3/50 matched on 2026-08-17 with agreement running to nt
                # 5, 4, 4 -- i.e. exactly the seed and no further. Real leakage runs for hundreds.
                # So the discriminator is HOW FAR the agreement extends, not whether it starts.
                src = cores[acc]
                k = 0
                while k < len(src) and k < len(gen) and src[k] == gen[k]:
                    k += 1
                if k >= max(sn + 30, 60):
                    begins += 1
                    offenders.append(f"{f.parent.name}/{f.name} (agrees to nt {k}, seed {sn})")
                else:
                    coincidental += 1
            # 60-MER CARRY-OVER — only meaningful when the seed actually contains a 60-mer.
            # With `max(1, len(seed)-60)` a short seed degenerates to "is this k-mer anywhere in
            # 2,200 nt", which is ~certain for k=4. Measured 2026-08-17: 49/50 at L=4 and 4/50 at
            # L=8 (chance), vs 0/50 at L=20/100/500 where the test is real. Restrict to sn >= 60
            # so the guard keeps its original strength for long seeds and stops manufacturing
            # hits for short ones.
            if sn >= 60:
                eligible += 1
                if any(seed[i:i + 60] in gen for i in range(0, len(seed) - 60 + 1, 60)):
                    contains += 1
    if n == 0:
        print("SKIP scored-span: no seeded records found on disk")
        return
    assert begins == 0, (
        f"{begins}/{n} stored sequences begin with their own seed AND keep agreeing well past it "
        f"(in {sorted(set(offenders))}). antiSMASH would then be scoring real class-X DNA we "
        f"supplied, and every correct_class number from those runs is confounded.")
    if coincidental:
        print(f"     ({coincidental}/{n} short-seed prefix coincidences, agreement not extending "
              f"past the seed — expected at 4-8 nt, not leakage)")
    # Verbatim carry-over of a seed fragment is a separate, milder issue (novelty covers it), but
    # a sudden jump would mean the model started copying rather than continuing.
    frac = (contains / eligible) if eligible else 0.0
    assert frac < 0.05, (f"{contains}/{eligible} = {frac:.1%} of generations with a >=60 nt seed "
                         f"contain a 60-mer of it — copying, not continuing")
    print(f"PASS scored-span: 0/{n} stored sequences begin with their seed and keep agreeing; "
          f"{contains}/{eligible} of >=60 nt-seed records carry a 60-mer ({frac:.1%})")


def main() -> int:
    for t in (test_generators_never_put_the_seed_into_the_stored_sequence,
              test_no_stored_record_begins_with_its_seed):
        t()
    print("\nALL SCORED-SPAN TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
