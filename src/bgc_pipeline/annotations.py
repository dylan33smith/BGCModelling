"""Attach per-position annotations (domain weights, codon phase) to a training WINDOW.

WHY THIS IS ITS OWN MODULE. The annotations live in nucleotide coordinates on the FULL record, but
training sees a chunked WINDOW preceded by a text prefix. Two offsets therefore have to be applied
in the right order, and both fail silently:

  1. WINDOW OFFSET. A record is chunked into `[nt_start, nt_end)` windows. Annotations must be
     sliced to the window and re-based to 0, or every span lands in the wrong place — and for codon
     phase in particular, computing the mask per-chunk instead of slicing a whole-record mask gets
     the phase wrong at every chunk boundary while still producing a mask of the right shape.
  2. PREFIX OFFSET. The tokenised row is `prefix + sequence`, and only the sequence half is
     supervised. Annotation index i of the window corresponds to token index
     `prefix_token_count + i`. Off by the prefix length and the weights decorate the wrong bases.

This is exact rather than approximate only because Evo2's `CharLevelTokenizer` is one token per
byte. A BPE substrate (GenomeOcean) would need overlap-weighted assignment instead, since a token
spans ~5 bases.
"""
from __future__ import annotations

from typing import Optional

import torch

from .objective import build_frame_mask, build_position_weights


def window_annotations(
    record_length: int,
    genes: Optional[list],
    domain_spans: Optional[list],
    nt_start: int,
    nt_end: int,
    prefix_token_count: int,
    total_tokens: int,
    domain_weight: float = 1.0,
    normalise_per_record: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (position_weights, frame_phase) aligned to the tokenised row.

    Both are length `total_tokens`. Prefix positions get weight 1.0 (they are masked from the loss
    anyway) and phase -1 (never penalised).

    NOTE ON NORMALISATION SCOPE: weights are normalised over the WHOLE RECORD before slicing, not
    over the window. Normalising per window would make a window that happens to be all-domain
    identical to one that is all-linker — destroying exactly the contrast the weighting exists to
    create.
    """
    w_full = build_position_weights(record_length, domain_spans or [], domain_weight,
                                    normalise_per_record=normalise_per_record)
    ph_full = build_frame_mask(record_length, genes or [])

    s = max(0, min(nt_start, record_length))
    e = max(s, min(nt_end, record_length))
    w_win, ph_win = w_full[s:e], ph_full[s:e]

    weights = torch.ones(total_tokens, dtype=torch.float32)
    phase = torch.full((total_tokens,), -1, dtype=torch.int8)
    n = min(len(w_win), max(0, total_tokens - prefix_token_count))
    if n > 0:
        weights[prefix_token_count:prefix_token_count + n] = w_win[:n]
        phase[prefix_token_count:prefix_token_count + n] = ph_win[:n]
    return weights, phase


def load_annotation_index(spans_jsonl) -> dict:
    """row index -> {'length', 'genes', 'spans'} from a `*.domain_spans.jsonl` sidecar.

    KEYED BY ROW, NOT ACCESSION. 5,219 accessions are shared by 12,217 training records with
    DIFFERENT sequences, so an accession-keyed join attaches the wrong record's spans — silently,
    and with plausible-looking output.
    """
    import json
    from pathlib import Path
    idx: dict = {}
    p = Path(spans_jsonl)
    if not p.exists():
        raise SystemExit(
            f"[annot] ABORT: no annotation sidecar at {p}. Build it with "
            f"`python scripts/build_domain_spans.py --splits train`. Running the weighted or "
            f"frame-aware objective without it would silently fall back to uniform weights and a "
            f"never-firing penalty — i.e. the baseline, reported as the treatment.")
    for n, line in enumerate(p.open()):
        r = json.loads(line)
        key = r.get("row", n)
        idx[key] = {"length": r.get("length", 0),
                    "genes": r.get("genes") or [],
                    "spans": r.get("spans") or [],
                    "accession": r.get("accession")}
    if not idx:
        raise SystemExit(f"[annot] ABORT: {p} parsed to zero records")
    return idx
