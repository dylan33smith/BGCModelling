"""Substrate loading and a uniform generation interface (SPEC 5).

The two substrates differ in every way that matters for this benchmark and fail in
OPPOSITE places, so the abstraction has to be explicit about each:

                    terminator   auto-appended?   stops on it?     tokenisation
  Evo2              byte 0       NO               NO (hardcoded)   byte-level, 1 tok ~ 1 nt
  GenomeOcean       [SEP] = 2    YES              only if told     BPE, vocab 4096

⚠ EVO2 CANNOT STOP AT ALL, AND THE REASON IS NOT WHAT IT LOOKS LIKE.

The weights are local; nothing here is a remote service. Three layers were checked:

  1. `Evo2.generate()` exposes no stop parameter.
  2. `vortex.model.generation.generate()` calls the inner `Generator` with a hardcoded
     `stop_at_eos=False`, forwarding `**kwargs` AFTER it — so passing `stop_at_eos=True`
     is a duplicate-keyword error, not an override.
  3. `Generator.generate()` DOES take `stop_at_eos`, defaulting to True — but it is DEAD
     CODE. The entire implementation is:

         if stop_at_eos and (generation[0, -1:] == eos_token_ids).all():
             print("Stopping generation at EOS")

     It prints and does not break. The only `break` in the function is inside a `verbose`
     display block, unrelated. Generation always runs to `num_tokens`.

  And even if it did break, it inspects `generation[0]` — row 0 only — so under batched
  generation the whole batch would be cut the moment the FIRST sequence emitted EOS,
  truncating every other row mid-sequence.

⇒ POST-HOC TRUNCATION IS THE ONLY CORRECT MECHANISM, not a workaround for a missing
feature. Generate to budget, truncate each row at its own terminator, record `hit_eos`.
A block-wise early exit (generate in blocks, stop when ALL rows have terminated) would
recover the wasted compute and is deferred until a fine-tuned model is measured to stop
early enough to be worth it.

So termination is handled UNIFORMLY and post hoc: generate to the budget, then truncate at
the first terminator occurrence, and record whether one appeared. That works identically on
both substrates and does not depend on vendor stop support. Where a substrate DOES support
native stopping (GenomeOcean, via `eos_token_id=2`), it is used as well and the two are
cross-checked in G10.

⚠ BUDGETS ARE IN NUCLEOTIDES, NEVER TOKENS. GenomeOcean is BPE, so a token budget would
hand the two substrates different amounts of sequence.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("HF_HOME", "/data2/ds85/hf_cache")

EVO2 = "evo2"
GENOMEOCEAN = "genomeocean"


@dataclass
class Substrate:
    id: str
    family: str
    checkpoint: str
    terminator_id: int
    terminator_str: str
    appends_terminator: bool          # does the tokenizer add it at encode time?
    native_stop: bool                 # can generation stop on it without post-processing?
    approx_nt_per_token: float
    model: Any = None
    tokenizer: Any = None
    meta: dict = field(default_factory=dict)

    # ---- training text -------------------------------------------------------------
    def training_text(self, sequence: str) -> str:
        """SPEC 4.3: bare sequence, plus a terminator where the tokenizer will not add one.

        A model that never sees a terminator in training will never emit one, so this is
        not optional for Evo2 -- and it is not a conditioning channel, it is the model's
        own native token.
        """
        if self.appends_terminator:
            return sequence
        return sequence + self.terminator_str

    # ---- termination ---------------------------------------------------------------
    def truncate_at_terminator(self, text: str) -> tuple[str, bool]:
        """Return (sequence up to the first terminator, whether one was found)."""
        i = text.find(self.terminator_str) if self.terminator_str else -1
        if i < 0:
            return text, False
        return text[:i], True

    @staticmethod
    def clean(text: str) -> str:
        """Keep only nucleotide characters. Applied AFTER terminator detection, never
        before -- stripping first would delete the terminator and make hit_eos always
        False."""
        return "".join(c for c in text.upper() if c in "ACGTN")


def load(substrate_id: str, device: str = "cuda:0") -> Substrate:
    if substrate_id.startswith("evo2"):
        from evo2 import Evo2
        name = {"evo2-1b": "evo2_1b_base", "evo2-7b": "evo2_7b"}.get(substrate_id,
                                                                     "evo2_1b_base")
        m = Evo2(name)
        tok = m.tokenizer
        eos = int(getattr(tok, "eos_id", 0))
        return Substrate(
            id=substrate_id, family=EVO2, checkpoint=name,
            terminator_id=eos, terminator_str=chr(eos),
            appends_terminator=False,       # verified: tokenize("ACGT") -> [65,67,71,84]
            native_stop=False,              # vortex hardcodes stop_at_eos=False
            approx_nt_per_token=1.0,
            model=m, tokenizer=tok,
            meta={"vocab_size": getattr(tok, "vocab_size", None)},
        )

    if substrate_id in ("go-4b", "bgcfm") or "genomeocean" in substrate_id:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = {"go-4b": "pGenomeOcean/GenomeOcean-4B",
                "bgcfm": "pGenomeOcean/GenomeOcean-4B-bgcFM"}.get(
                    substrate_id, "pGenomeOcean/GenomeOcean-4B")
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        # GenomeOcean does not DECLARE an eos_token, but its tokenizer appends [SEP]=2.
        # Take the id from the tokenizer's own special-token table, never a literal.
        sep = tok.convert_tokens_to_ids("[SEP]")
        model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        return Substrate(
            id=substrate_id, family=GENOMEOCEAN, checkpoint=name,
            terminator_id=int(sep), terminator_str="",   # BPE: detect on ids, not text
            appends_terminator=True,
            native_stop=True,               # generate(eos_token_id=sep)
            # MEASURED in G10, not assumed: 1,000 tokens decoded to ~4,792 nt. An
            # estimate of 4.0 would hand GenomeOcean ~20% more sequence than the
            # nucleotide budget asks for, which is not a fair budget across substrates.
            approx_nt_per_token=4.8,
            model=model, tokenizer=tok,
            meta={"vocab_size": tok.vocab_size,
                  "declares_eos": tok.eos_token_id is not None},
        )

    raise ValueError(f"unknown substrate {substrate_id!r}")
