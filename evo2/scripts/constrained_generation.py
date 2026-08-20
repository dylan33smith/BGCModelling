#!/usr/bin/env python
"""Token-id-aware generation for Evo2: early stopping, constrained decoding, snip-and-replace.

WHY THIS EXISTS (2026-08-20)
----------------------------
Three separate needs all reduce to one missing capability -- **seeing the sampled token ids**.
vortex's `generate()` returns `logits` / `logprobs_mean` / `sequences` and never the ids, and ids
**0 (EOS), 1 (PAD) and 32 (space) all detokenize to the same character**, so the decoded string
cannot tell "the model stopped" from "the model emitted junk". Everything below follows from
recovering the ids.

1. **EARLY STOPPING.** ⚠️ vortex's own `stop_at_eos` is BROKEN: `generation.py:208` checks for EOS
   and then only `print`s -- there is no `break` -- and it inspects `generation[0]`, batch row 0
   only. Measured cost: the PKS adapter stops at a median ~1,750 nt of 8,000 requested, so ~75% of
   that arm's generation compute ran after the model had finished.
2. **CONSTRAINED DECODING.** The adapter puts ~1.8% of its probability mass on non-nucleotide
   tokens. Masking every id outside {A,C,G,T,N,EOS} to -inf makes the stray byte impossible by
   construction. Measured: 0/48 stop events under a full non-ACGTN mask, vs 36/48 unmasked.
3. **SNIP-AND-REPLACE.** Degenerate records (the model collapsing to a ~uniform distribution over
   all 512 ids) are a SEPARATE failure from termination, and constraining does not fix them --
   masking at a uniform-distribution position just forces an arbitrary nucleotide. They have to be
   detected and regenerated.

HOW: vortex's loop cannot be broken from outside, so `sample()` is wrapped. The wrapper masks the
logits, records the sampled ids, and raises `_AllRowsDone` once every row has emitted EOS; the
caller catches it and reconstructs the sequences from the recorded ids. Nothing depends on vortex's
return value, so the early exit costs nothing.
"""
from __future__ import annotations

import torch

EOS_ID, PAD_ID, SPACE_ID = 0, 1, 32
NUCLEOTIDE_IDS = (65, 67, 71, 84, 78)          # A C G T N
DEFAULT_ALLOWED = NUCLEOTIDE_IDS + (EOS_ID,)   # nucleotides + the real stop token
VOCAB = 512


class _AllRowsDone(Exception):
    """Raised inside the patched sampler once every row has emitted EOS, to exit vortex's loop."""


class TokenRecorder:
    """Wraps `vortex.model.sample.sample` to mask, record ids, and stop early.

    `allowed_ids=None` disables constrained decoding (records + early-stops only).
    """

    def __init__(self, batch_size: int, allowed_ids=DEFAULT_ALLOWED, early_stop: bool = True):
        self.batch = batch_size
        self.allowed = None if allowed_ids is None else sorted(set(allowed_ids))
        self.early_stop = early_stop
        self.ids: list[list[int]] = [[] for _ in range(batch_size)]
        self.done = [False] * batch_size
        self.steps = 0
        self._mask = None

    def _banned_mask(self, logits):
        if self._mask is None:
            m = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            m[:] = True
            m[torch.tensor(self.allowed, device=logits.device)] = False
            self._mask = m
        return self._mask

    def wrap(self, original):
        def _sample(logits, top_k=1, top_p=0.0, temperature=1.0):
            if self.allowed is not None:
                logits = logits.clone()
                logits[..., self._banned_mask(logits)] = float("-inf")
            new_idx = original(logits, top_k=top_k, top_p=top_p, temperature=temperature)
            self.steps += 1
            flat = new_idx.reshape(-1).tolist()
            for r, t in enumerate(flat[: self.batch]):
                if not self.done[r]:
                    self.ids[r].append(int(t))
                    if int(t) == EOS_ID:
                        self.done[r] = True
            if self.early_stop and all(self.done):
                raise _AllRowsDone
            return new_idx
        return _sample

    # ── readout ────────────────────────────────────────────────────────────────
    def sequences(self):
        """Per row: (nucleotide string up to EOS, hit_eos, n_non_nucleotide_ids)."""
        out = []
        for r in range(self.batch):
            ids = self.ids[r]
            hit = bool(ids) and ids[-1] == EOS_ID
            body = ids[:-1] if hit else ids
            seq = "".join(chr(t) if t in NUCLEOTIDE_IDS else "N" for t in body)
            junk = sum(1 for t in body if t not in NUCLEOTIDE_IDS)
            out.append((seq, hit, junk))
        return out


def is_degenerate(ids, window: int = 200, threshold: float = 0.5) -> bool:
    """Has the model left the nucleotide alphabet? -- the snip-and-replace trigger.

    Degeneracy is measured on a TRAILING WINDOW, not over the whole record: a record that wrote
    6 kb of clean DNA and then collapsed must be caught, and a whole-record fraction dilutes exactly
    that case. ⚠️ Under constrained decoding this can never fire (every id is a nucleotide by
    construction) -- which is the point of `[X1d]`: constraining hides the collapse rather than
    fixing it, so a length/quality gate is still required.
    """
    if not ids:
        return True                                    # empty generation is a failure too
    tail = [t for t in ids[-window:] if t != EOS_ID]
    if not tail:
        return False
    bad = sum(1 for t in tail if t not in NUCLEOTIDE_IDS)
    return bad / len(tail) >= threshold


def generate_recorded(wrapper, prompt_seqs, n_tokens, *, allowed_ids=DEFAULT_ALLOWED,
                      early_stop=True, temperature=1.0, top_k=4, top_p=1.0, batched=True):
    """Generate with ids recorded, optional constrained decoding, and real early stopping.

    Returns (rows, recorder) where rows is a list of (sequence, hit_eos, n_junk_ids).
    """
    import vortex.model.generation as VG
    import vortex.model.sample as VS

    rec = TokenRecorder(len(prompt_seqs), allowed_ids=allowed_ids, early_stop=early_stop)
    orig_vs, orig_vg = VS.sample, VG.sample
    VS.sample = VG.sample = rec.wrap(orig_vs)
    try:
        wrapper.generate(prompt_seqs=prompt_seqs, n_tokens=n_tokens,
                         temperature=temperature, top_k=top_k, top_p=top_p,
                         batched=batched, cached_generation=True, verbose=0)
    except _AllRowsDone:
        pass                                            # every row finished; the loop exited early
    finally:
        VS.sample, VG.sample = orig_vs, orig_vg
    return rec.sequences(), rec
