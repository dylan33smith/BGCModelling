"""Model-agnostic core of the compound-class linear probe.

The probe asks one question: **is `compound_class` linearly decodable from a frozen
model's hidden states?** Everything here is the part that must be *identical* across
models for the comparison to mean anything — sampling of cores, the classifier, the
cross-validation scheme, and the shuffled-label control. Only the embedding extraction
is model-specific, and that lives in the per-model scripts:

  - Evo2:        evo2/scripts/class_probe.py        (forward hooks on `blocks.N`)
  - GenomeOcean: genomeocean/scripts/class_probe_go.py (HF `output_hidden_states=True`)

Extracted from the Evo2 script so both tracks share one statistical protocol rather than
two hand-matched copies. `evo2/scripts/class_probe.py` still carries its own copy and
should be migrated to this module.

Interpretation:
  * balanced_acc >> shuffled_acc  -> the model ENCODES class; the failure is
    decoding/steering, and cheap fixes are viable (guided decoding, steering vectors,
    soft prompts, CFG with a trained null).
  * balanced_acc ~= shuffled_acc  -> class is NOT represented; a representation has to be
    INSTALLED by training. Only that justifies heavy compute.
"""
from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

import numpy as np


_PHYLUM_RE = re.compile(r"p__([^;|]+)")


def phylum_of(rec: dict) -> str | None:
    """Phylum from a `|d__...;p__...;c__...|` GTDB tag, or None."""
    m = _PHYLUM_RE.search(rec.get("taxonomic_tag") or "")
    return m.group(1) if m else None


def sample_cores(jsonl: Path, per_class: int, min_class: int, max_nt: int, seed: int,
                 label_field: str = "compound_class",
                 restrict_phylum: str | None = None
                 ) -> tuple[list[tuple[str, str]], list[str]]:
    """Balanced sample of real BGC cores as (label, raw_nucleotides).

    RAW nucleotides only — no prefix, no class tag. If the class tag were present the
    probe would trivially read it back off the tag tokens and report ~1.0, telling us
    nothing about the sequence representation.

    TAXON CONFOUND CONTROLS. Compound class correlates with taxonomy (Streptomyces is
    high-GC and PKS-rich), and nucleotide composition encodes taxonomy — so a strong
    "class" probe can be a taxon probe wearing a hat. Two knobs to separate them:

      label_field="phylum"      -> probe TAXON instead of class from the same activations.
                                   A class probe is only interesting if it beats this.
      restrict_phylum="Actinomycetota" -> probe class WITHIN one phylum, where taxonomy is
                                   held roughly constant. If accuracy survives, the signal
                                   is class; if it collapses, it was taxonomy.
    """
    rng = random.Random(seed)
    byc: dict[str, list[str]] = {}
    for line in jsonl.open():
        r = json.loads(line)
        s = r.get("sequence", "")
        if len(s) < 200:
            continue
        if restrict_phylum and phylum_of(r) != restrict_phylum:
            continue
        c = phylum_of(r) if label_field == "phylum" else r.get(label_field)
        if c:
            byc.setdefault(c, []).append(s)
    classes = sorted(c for c, v in byc.items() if len(v) >= min_class)
    data: list[tuple[str, str]] = []
    for c in classes:
        pool = byc[c][:]
        rng.shuffle(pool)
        for s in pool[:per_class]:
            data.append((c, s[:max_nt]))
    rng.shuffle(data)
    return data, classes


def phylum_counts(jsonl: Path) -> collections.Counter:
    """Records per phylum — use it to pick a `restrict_phylum` with enough data."""
    c: collections.Counter = collections.Counter()
    for line in jsonl.open():
        p = phylum_of(json.loads(line))
        if p:
            c[p] += 1
    return c


def probe(Xi: np.ndarray, y: np.ndarray, seed: int,
          classes: list[str]) -> tuple[float, float, dict[str, float]]:
    """Cross-validated logistic probe + a shuffled-label control on the same folds.

    Returns (balanced_acc, shuffled_acc, per_class_recall). The shuffled control is the
    honest baseline: with 22 imbalanced classes and a high-dimensional X, raw accuracy can
    look impressive purely from overfitting, so the gap to shuffled is what matters.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(pipe, Xi, y, cv=skf)
    acc = balanced_accuracy_score(y, pred)

    ysh = y.copy()
    np.random.RandomState(seed).shuffle(ysh)
    predsh = cross_val_predict(pipe, Xi, ysh, cv=skf)
    accsh = balanced_accuracy_score(ysh, predsh)

    per_class = dict(zip(
        classes,
        np.round(recall_score(y, pred, labels=classes, average=None,
                              zero_division=0), 3).tolist(),
    ))
    return float(acc), float(accsh), per_class
