"""Generation harness (SPEC 6, 7).

⚠ SPEC 9 RULE 2: THIS FILE CONTAINS NO ARM NAMES. It takes an arm *coordinate* as data.
An arm-specific branch here is how two arms come to be generated differently without
anyone deciding to, which is the failure the whole benchmark design exists to prevent.

Two things this handles that are easy to get wrong:

* **Budgets are in NUCLEOTIDES.** GenomeOcean is BPE at a measured ~4.8 nt/token, so a
  token budget would hand the substrates different amounts of sequence (SPEC 5.1).
* **The seed is never scored.** Evo2 returns only the continuation; HuggingFace returns
  prompt+continuation. Getting that wrong silently scores the seed as model output, which
  would make every seeded arm look extraordinary.
"""
from __future__ import annotations

from contextlib import contextmanager

from dataclasses import dataclass, field
from typing import Any

from bgcbench.model.genconfig import FROZEN
from bgcbench.model.load import EVO2, Substrate


@dataclass
class ArmSpec:
    """One coordinate in the SPEC 6 grid, as data."""
    arm_id: str
    weight_state: str                 # which adapter/prefix to load; "base" for none
    seeded: bool = False
    seed_len_nt: int = 0              # seeded regime; length fixed by gate G3
    inference_control: str = "none"   # "none" | "steer" | "refine"
    #: "none" -> bare input (SPEC 4.3 default). "taxonomy" -> prepend the target class's
    #: held-out GTDB lineage, Evo2's native pretraining format. Names an ORGANISM, never a
    #: compound class, so it does not hand the model the answer the benchmark asks for.
    prefix: str = "none"
    adapter_path: str | None = None
    steer: dict[str, Any] = field(default_factory=dict)
    temperature: float = FROZEN["temperature"]
    top_k: int = FROZEN["top_k"]
    top_p: float = FROZEN["top_p"]


@dataclass
class GenConfig:
    """Instantiated from the FROZEN config (SPEC 7). Do not construct one by hand for a
    benchmark run -- use `GenConfig.frozen()`, so the values and the hash cannot diverge."""
    budget_nt: int = FROZEN["budget_nt"]
    batch_size: int = FROZEN["batch_size"]
    seed: int = FROZEN["rng_seed"]
    min_new_tokens: int = FROZEN["min_new_tokens"]

    @classmethod
    def frozen(cls) -> "GenConfig":
        return cls(budget_nt=FROZEN["budget_nt"], batch_size=FROZEN["batch_size"],
                   seed=FROZEN["rng_seed"], min_new_tokens=FROZEN["min_new_tokens"])


def _seed_text(rec: dict, n_nt: int) -> str:
    """Seeded regime: the first `n_nt` nt of a held-out core of the target class.

    Callers must supply records at least `n_nt` long -- see `usable_seed_pool`. A short
    record silently yields a SHORT PROMPT, and ragged prompts split Evo2 between batched
    and sequential decoding (vortex batches only when all prompts are the same length).
    That split correlates with the confusion-matrix row, so a decoding-mode difference
    would read as a class effect.
    """
    s = (rec["sequence"] or "")[:n_nt]
    if len(s) != n_nt:
        raise ValueError(
            f"{rec.get('accession')} yields a {len(s)} nt seed, not {n_nt}. Ragged prompts "
            f"change the decode path; filter the pool with usable_seed_pool() first."
        )
    return s


def usable_seed_pool(records: list[dict], n_nt: int) -> list[dict]:
    """Records long enough to give a full-length seed, in a deterministic order.

    Ordering by accession (not file order) and filtering BEFORE the modulo means the seed
    set is a function of the pool and n_nt alone -- previously it drifted with n and with
    whatever order the split file happened to be written in.
    """
    return sorted((r for r in records if (r.get("seq_len") or 0) >= n_nt),
                  key=lambda r: r["accession"])


def generate(sub: Substrate, arm: ArmSpec, target_class: str, n: int,
             cfg: GenConfig, seed_pool: list[dict] | None = None,
             stage: str = "stage1") -> list[dict]:
    """Return generation dicts ready for `record.build`. One per requested sequence."""
    if arm.seeded:
        if not seed_pool:
            raise ValueError(f"{arm.arm_id} is seeded but no seed pool was supplied")
        if arm.seed_len_nt <= 0:
            raise ValueError(
                f"{arm.arm_id} is seeded with seed_len_nt={arm.seed_len_nt}: an empty "
                f"prompt makes it a DE NOVO arm wearing a seeded label."
            )
        seed_pool = usable_seed_pool(seed_pool, arm.seed_len_nt)
        if not seed_pool:
            raise ValueError(
                f"no record in the pool is >= {arm.seed_len_nt} nt; a short seed would "
                f"change the decode path for this row only."
            )

    # TAXONOMY PREFIX POOL. Lineages are drawn from HELD-OUT records of the target class, so
    # a prompt is never a lineage the arm trained on for that particular record. The lineage
    # names an organism, not a compound class.
    tax_recs: list[dict] = []
    tax_pool: list[str] = []
    if arm.prefix == "taxonomy":
        tax_recs = [r for r in (seed_pool or []) if r.get("tax_tag")]
        tax_pool = [r["tax_tag"] for r in tax_recs]
        if not tax_pool:
            raise ValueError(
                "prefix='taxonomy' but no record in the pool carries a tax_tag; call "
                "bgcbench.data.taxonomy.attach() on the pool first. Falling back to a bare "
                "prompt would silently run a DIFFERENT arm than the one requested."
            )

    prompts: list[str] = []
    seeds: list[dict | None] = []
    prefixes: list[str] = []
    prefix_srcs: list[str | None] = []
    prefix_genomes: list[str | None] = []
    for i in range(n):
        pre_i = (i % len(tax_pool)) if tax_pool else None
        pre = tax_pool[pre_i] if tax_pool else ""
        pre_rec = tax_recs[pre_i] if tax_pool else None
        prefixes.append(pre)
        prefix_srcs.append(pre_rec["accession"] if pre_rec else None)
        prefix_genomes.append(pre_rec.get("genome_accession") if pre_rec else None)
        if arm.seeded:
            rec = seed_pool[i % len(seed_pool)]
            prompts.append(pre + _seed_text(rec, arm.seed_len_nt))
            seeds.append(rec)
        else:
            # SPEC 4.3 default: bare sequence input, nothing to condition on. With
            # prefix="taxonomy" the prompt is the lineage alone.
            prompts.append(pre)
            seeds.append(None)

    texts, hits = _run(sub, arm, prompts, cfg)

    out: list[dict] = []
    for i, (txt, hit) in enumerate(zip(texts, hits)):
        rec = seeds[i]
        out.append({
            "generation_id": f"{arm.arm_id}::{sub.id}::{target_class}::{i:05d}",
            "stage": stage,
            "target_class": target_class,
            "sequence": txt,
            "hit_eos": hit,
            "seed_accession": rec["accession"] if rec else None,
            "seed_core_gene_count": rec["core_gene_count"] if rec else None,
            "seed_seq_len": rec["seq_len"] if rec else None,
            # WHICH LINEAGE produced this generation, so a hit can be traced to its organism
            "prefix_kind": arm.prefix,
            "prefix_tag": prefixes[i],
            "prefix_source_accession": prefix_srcs[i],
            "prefix_source_genome": prefix_genomes[i],
        })
    return out


def _seed_everything(seed: int) -> None:
    """SPEC 7.3 requires an identical RNG seed across arms. `GenConfig.seed` was declared,
    stamped into every report as `rng_seed: 0`, and READ BY NOTHING -- generation was
    unseeded on both substrate paths while every artifact asserted otherwise. There was no
    truthful field anywhere to contradict the claim."""
    import random as _r

    import numpy as _np
    import torch as _t
    _r.seed(seed)
    _np.random.seed(seed)
    _t.manual_seed(seed)
    if _t.cuda.is_available():
        _t.cuda.manual_seed_all(seed)


def _run(sub: Substrate, arm: ArmSpec, prompts: list[str],
         cfg: GenConfig) -> tuple[list[str], list[bool]]:
    _seed_everything(cfg.seed)
    if sub.family == EVO2:
        return _run_evo2(sub, arm, prompts, cfg)
    return _run_hf(sub, arm, prompts, cfg)


#: Masking value for a suppressed token. NOT -inf: vortex's `sample()` runs
#: `torch.where(logits == -inf, 0, logits)`, which would turn a -inf mask into a logit of
#: ZERO -- i.e. it would make the suppressed token MORE likely, not less. A large finite
#: negative survives that rewrite and is removed by top-k filtering.
SUPPRESSED_LOGIT = -1e9


@contextmanager
def suppress_terminator(sub, n_positions: int):
    """Forbid the terminator for the first `n_positions` sampled tokens (Evo2 only).

    ⚠ WHY THIS IS NEEDED. Measured once the terminator was made visible: from the bare de
    novo prompt Evo2 emits its stop token after TWO nucleotides -- 56% of base generations
    and 91% of fine-tuned ones, first terminator at index 2. Unconditioned de novo
    generation on this substrate collapses immediately, so without a floor there is nothing
    to score and the arm cannot produce a benchmark measurement at all.

    This is a DECODING POLICY, not a conditioning channel: it is one number, identical for
    every arm and every class, and it injects no class information. It is the same fix the
    prior project applied to GenomeOcean, where EOS firing straight after the seed produced
    61/200 empty generations and a min-token floor took the arm from 0.400 to 0.580.

    vortex offers no logits-processor hook, so this swaps the module-level `sample` that
    `generation.py` imported at line 9. The swap is restored on exit even if the body
    raises; a leaked patch would silently alter every later arm in the same process.
    """
    if n_positions <= 0 or sub.family != EVO2:
        yield {"suppressed_positions": 0}
        return
    import vortex.model.generation as vg
    orig = vg.sample
    eos = int(sub.terminator_id)
    state = {"calls": 0, "suppressed_positions": 0}

    def wrapped(logits, **kw):
        if state["calls"] < n_positions:
            # CLONE FIRST. vortex hands back an inference tensor and an in-place write to
            # one raises outside InferenceMode; the clone is an ordinary tensor.
            logits = logits.clone()
            logits[..., eos] = SUPPRESSED_LOGIT
            state["suppressed_positions"] += 1
        state["calls"] += 1
        return orig(logits, **kw)

    vg.sample = wrapped
    try:
        yield state
    finally:
        vg.sample = orig


def _run_evo2(sub, arm, prompts, cfg):
    """Generate, BUCKETING BY PROMPT LENGTH so every call is actually batched.

    ⚠ vortex batches only when every prompt in a call has the same length
    (`uniform_lengths = all(len(s) == len(prompt_seqs[0]))`, generation.py:313). Otherwise it
    writes "WARNING: Batched generation is turned off" to stderr and generates ONE SEQUENCE
    AT A TIME. That fallback is silent in any log that filters warnings, and it is
    catastrophic rather than merely slow: measured twice on real GTDB lineage prompts, 42
    prompts carry 19-21 distinct lengths, so a single call becomes 19-21 near-sequential
    generations -- hours of wall time at ~40% GPU where one batch takes minutes.

    De novo prompts (all "A") and seeded prompts (all `seed_len_nt`) are uniform by
    construction, so this changes nothing for them. It matters the moment any variable-length
    prompt is used, which is why it belongs here rather than in a caller.

    ⚠ ORDER IS PRESERVED BY INDEX. Results are scattered back to their original positions:
    generation `i` must correspond to prompt `i`, or every per-generation field recorded
    downstream -- seed accession, seed length, the confusion-matrix row -- is attached to the
    wrong sequence. Appending bucket by bucket would silently permute them.
    """
    prepared = [p if p else "A" for p in prompts]
    buckets: dict[int, list[int]] = {}
    for i, p in enumerate(prepared):
        buckets.setdefault(len(p), []).append(i)
    if len(buckets) > 1:
        print(f"  prompts span {len(buckets)} lengths -> {len(buckets)} bucket(s), "
              f"sizes {sorted((len(v) for v in buckets.values()), reverse=True)[:8]}",
              flush=True)

    texts: list[str | None] = [None] * len(prepared)
    hits: list[bool | None] = [None] * len(prepared)
    for _L, idxs in sorted(buckets.items()):
        for j in range(0, len(idxs), cfg.batch_size):
            sl = idxs[j:j + cfg.batch_size]
            with suppress_terminator(sub, cfg.min_new_tokens):
                out = sub.model.generate(prompt_seqs=[prepared[k] for k in sl],
                                         n_tokens=cfg.budget_nt,
                                         temperature=arm.temperature, top_k=arm.top_k,
                                         top_p=arm.top_p, verbose=0)
            seqs = list(out[0]) if isinstance(out, tuple) else list(out.sequences)
            for k, sq in zip(sl, seqs):
                # Evo2 returns ONLY the continuation, so there is no prompt to strip.
                body, hit = sub.truncate_at_terminator(sq)
                texts[k] = sub.clean(body)
                hits[k] = bool(hit)
    if any(t is None for t in texts):
        raise RuntimeError("a prompt produced no generation; bucketing lost a record")
    return texts, hits


def _run_hf(sub, arm, prompts, cfg):
    import torch
    texts, hits = [], []
    max_new = max(1, int(cfg.budget_nt / sub.approx_nt_per_token))
    for i in range(0, len(prompts), cfg.batch_size):
        chunk = [p if p else "A" for p in prompts[i:i + cfg.batch_size]]
        # LEFT padding. A decoder-only model conditions on the token immediately before
        # the first generated position; with right padding that token is [PAD] for every
        # row that is not the longest in its batch, so the first sampled token of those
        # rows is drawn from a corrupted context. The corruption is invisible in the output.
        prev_side = getattr(sub.tokenizer, "padding_side", "right")
        sub.tokenizer.padding_side = "left"
        enc = sub.tokenizer(chunk, return_tensors="pt", padding=True)
        sub.tokenizer.padding_side = prev_side
        enc = {k: v.to(sub.model.device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        plen = enc["input_ids"].shape[1]
        # ⚠ THE MIN-TOKEN FLOOR APPLIES HERE TOO, AND IT WAS MISSING. Evo2 gets a floor of
        # `cfg.min_new_tokens` via `suppress_terminator`, which returns early for every other
        # family -- so the HF path had no floor at all. GenomeOcean terminates natively and
        # eagerly: the prior project measured EOS firing straight after the seed, 61/200
        # empty generations, and a min-token floor taking that arm from 0.400 to 0.580.
        # Without this, a cross-substrate comparison would score GO on truncated output and
        # attribute the deficit to the substrate.
        #
        # ⚠ THE FLOOR IS IN NUCLEOTIDES, CONVERTED PER SUBSTRATE. `min_new_tokens` is 1,000
        # and Evo2 is byte-level, so for Evo2 it is 1,000 nt. Passing 1,000 TOKENS to a BPE
        # model at ~4.8 nt/token would demand ~4,800 nt -- 4.8x the sequence, which is not
        # the same floor. Both substrates must clear the same nucleotide bar.
        min_new = max(1, int(cfg.min_new_tokens / sub.approx_nt_per_token))
        min_new = min(min_new, max_new)
        with torch.no_grad():
            gen = sub.model.generate(**enc, max_new_tokens=max_new,
                                     min_new_tokens=min_new, do_sample=True,
                                     temperature=arm.temperature, top_k=arm.top_k,
                                     top_p=arm.top_p,
                                     eos_token_id=sub.terminator_id,
                                     pad_token_id=sub.tokenizer.pad_token_id)
        for row in gen:
            ids = row.tolist()[plen:]          # STRIP THE PROMPT — never score the seed
            hits.append(sub.terminator_id in ids)
            # sub.detokenize, NOT tokenizer.decode -- decode() separates BPE tokens
            # with a space, which clean() masks to N. See Substrate.detokenize.
            txt = sub.detokenize(ids)
            texts.append(sub.clean(txt))
    return texts, hits
