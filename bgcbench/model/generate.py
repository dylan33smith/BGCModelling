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

    @classmethod
    def frozen(cls) -> "GenConfig":
        return cls(budget_nt=FROZEN["budget_nt"], batch_size=FROZEN["batch_size"],
                   seed=FROZEN["rng_seed"])


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

    prompts: list[str] = []
    seeds: list[dict | None] = []
    for i in range(n):
        if arm.seeded:
            rec = seed_pool[i % len(seed_pool)]
            prompts.append(_seed_text(rec, arm.seed_len_nt))
            seeds.append(rec)
        else:
            # SPEC 4.3: bare sequence input. With no conditioning channel there is nothing
            # to put in a de novo prompt, so it is empty and the model is unconstrained.
            prompts.append("")
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


def _run_evo2(sub, arm, prompts, cfg):
    texts, hits = [], []
    # Evo2 batches only when prompts are uniform length; de novo prompts are all empty and
    # seeded prompts are all seed_len_nt, so batching is available in both regimes.
    for i in range(0, len(prompts), cfg.batch_size):
        chunk = prompts[i:i + cfg.batch_size]
        # a truly empty prompt has nothing to condition on; vortex needs at least one token
        chunk = [p if p else "A" for p in chunk]
        out = sub.model.generate(prompt_seqs=chunk, n_tokens=cfg.budget_nt,
                                 temperature=arm.temperature, top_k=arm.top_k,
                                 top_p=arm.top_p, verbose=0)
        seqs = list(out[0]) if isinstance(out, tuple) else list(out.sequences)
        for s in seqs:
            # Evo2 returns ONLY the continuation, so there is no prompt to strip.
            body, hit = sub.truncate_at_terminator(s)
            texts.append(sub.clean(body))
            hits.append(bool(hit))
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
        with torch.no_grad():
            gen = sub.model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                     temperature=arm.temperature, top_k=arm.top_k,
                                     top_p=arm.top_p,
                                     eos_token_id=sub.terminator_id,
                                     pad_token_id=sub.tokenizer.pad_token_id)
        for row in gen:
            ids = row.tolist()[plen:]          # STRIP THE PROMPT — never score the seed
            hits.append(sub.terminator_id in ids)
            txt = sub.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(sub.clean(txt))
    return texts, hits
