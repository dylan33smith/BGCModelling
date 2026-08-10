"""Acceptance suite for generated BGC sequences — named CHECKS → QUESTIONS.

First-principles structure (2026-06-17, see docs/archive/REDESIGN_PLAN.md). Two layers, not a
flat numbered list:

CHECKS (compute units; one consistent gene caller, Prodigal/pyrodigal):
  coding_sanity        gene-rich, complete coding DNA (vs degenerate junk)
  antismash            antiSMASH detection + classification (detected, class_match)
  class_markers        per-class Pfam markers (fast proxy for antiSMASH class)
  kmer_novelty         anti-memorization k-mer containment vs training
  protein_homology     MMseqs2 homology of predicted proteins to known enzymes
  module_architecture  ordered assembly-line modules (from class_markers positions)
  taxon_faithfulness   codon/GC faithfulness to the conditioned taxon
  protein_foldability  ESMFold pLDDT — OPTIONAL, opt-in (GPU-expensive)

QUESTIONS (what we actually want to know; derived by combining checks):
  is_bgc                coding_sanity ∧ antismash.detected (markers proxy)   [GATE]
  correct_class         antismash.class_match (class_markers proxy)          [GATE]
  novel                 kmer_novelty                                         [GATE]
  proteins_plausible    protein_homology                                     [diag]
  complete              module_architecture                                  [diag]
  conditioning_faithful taxon_faithfulness                                   [diag]

antiSMASH is the gold-standard BGC detector/classifier; class_markers is the fast
Pfam proxy for quick-eval. RETIRED: synthesis feasibility, Evo2 perplexity,
BiG-SCAPE (all removed in the rewrite). Each check returns a dict; external tools
that are not installed are gracefully skipped.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# E. coli K-12 highly expressed gene codon usage (per 1000 codons)
# Source: Kazusa codon usage database, E. coli K-12 substr. MG1655
_ECOLI_CODON_FREQ: dict[str, float] = {
    "TTT": 22.0, "TTC": 16.2, "TTA": 13.8, "TTG": 13.4,
    "CTT": 11.0, "CTC": 10.9, "CTA": 3.9,  "CTG": 52.6,
    "ATT": 30.1, "ATC": 24.5, "ATA": 4.5,  "ATG": 27.8,
    "GTT": 18.3, "GTC": 15.1, "GTA": 10.8, "GTG": 25.9,
    "TCT": 8.5,  "TCC": 8.8,  "TCA": 7.2,  "TCG": 8.8,
    "CCT": 7.0,  "CCC": 5.5,  "CCA": 8.4,  "CCG": 23.2,
    "ACT": 9.0,  "ACC": 23.0, "ACA": 7.1,  "ACG": 14.4,
    "GCT": 15.5, "GCC": 25.4, "GCA": 20.0, "GCG": 33.6,
    "TAT": 16.2, "TAC": 12.1, "TAA": 2.0,  "TAG": 0.3,
    "CAT": 12.8, "CAC": 9.7,  "CAA": 15.3, "CAG": 28.8,
    "AAT": 17.6, "AAC": 21.5, "AAA": 33.6, "AAG": 10.1,
    "GAT": 32.2, "GAC": 19.0, "GAA": 39.4, "GAG": 17.8,
    "TGT": 5.2,  "TGC": 6.5,  "TGA": 1.0,  "TGG": 15.2,
    "CGT": 20.9, "CGC": 21.8, "CGA": 3.6,  "CGG": 5.6,
    "AGT": 8.8,  "AGC": 16.0, "AGA": 2.1,  "AGG": 1.2,
    "GGT": 24.6, "GGC": 29.4, "GGA": 8.0,  "GGG": 11.1,
}

# ---------------------------------------------------------------------------
# Reference organism profiles for the taxon_faithfulness check
# ---------------------------------------------------------------------------
#
# taxon_faithfulness answers two DIFFERENT questions that must not be conflated:
#   (a) Faithfulness  — does the generated sequence match the codon/GC/dinucleotide
#       signature of the ORGANISM IT WAS CONDITIONED ON? (graded vs that taxon)
#   (b) Expressibility — would it express in the wet-lab synthesis chassis (E. coli)?
# The model is conditioned on the source taxon (97.6% bacteria; Actinomycetota
# GC ~0.71), so a *correct* model reproduces high-GC, non-E.coli statistics.
# Grading faithfulness against E. coli therefore auto-fails a good model. We keep
# E. coli as an explicit CHASSIS reference for (b), reported separately, and grade
# the verdict on (a) against the conditioned taxon's profile when one is supplied.

@dataclass
class ReferenceProfile:
    """Codon/GC/dinucleotide reference for one organism or synthesis chassis."""

    name: str
    target_gc: float
    codon_freq: dict[str, float]          # per-1000 codon usage (for CAI)
    dinuc_ratios: dict[str, float]        # observed/expected dinucleotide ratios
    gc_tol: float = 0.10
    cai_threshold: float = 0.7
    dinuc_threshold: float = 0.15


# E. coli K-12 dinucleotide observed/expected ratios (synthesis chassis reference)
_ECOLI_DINUC_RATIOS: dict[str, float] = {
    "AA": 1.024, "AC": 0.867, "AG": 1.013, "AT": 1.098,
    "CA": 1.094, "CC": 0.934, "CG": 1.066, "CT": 1.013,
    "GA": 1.069, "GC": 1.098, "GG": 0.934, "GT": 0.867,
    "TA": 0.811, "TC": 1.069, "TG": 1.094, "TT": 1.024,
}

# Default synthesis chassis (the wet-lab expression host).
ECOLI_PROFILE = ReferenceProfile(
    name="ecoli_k12",
    target_gc=0.508,
    codon_freq=_ECOLI_CODON_FREQ,
    dinuc_ratios=_ECOLI_DINUC_RATIOS,
)


def build_profile_from_sequences(
    name: str,
    sequences: Iterable[str],
    gc_tol: float = 0.10,
    cai_threshold: float = 0.7,
    dinuc_threshold: float = 0.15,
) -> ReferenceProfile:
    """Derive a ReferenceProfile empirically from reference DNA sequences.

    Aggregates GC content, codon usage (per-1000), and observed/expected
    dinucleotide ratios across all provided sequences. Run this over the training
    BGCs of a given taxon so taxon_faithfulness can grade faithfulness to the organism the
    model was actually conditioned on. See scripts/build_taxon_profiles.py.
    """
    gc_count = 0
    base_total = 0
    codon_counts: Counter = Counter()
    mono: Counter = Counter()
    di: Counter = Counter()
    di_total = 0
    for seq in sequences:
        s = seq.upper()
        if not s:
            continue
        gc_count += s.count("G") + s.count("C")
        base_total += len(s)
        mono.update(s)
        for i in range(len(s) - 1):
            di[s[i:i + 2]] += 1
        di_total += max(len(s) - 1, 0)
        for i in range(0, len(s) - 2, 3):
            codon = s[i:i + 3]
            if codon in _CODON_TABLE:
                codon_counts[codon] += 1

    target_gc = gc_count / base_total if base_total else 0.0
    total_codons = sum(codon_counts.values())
    codon_freq = {
        c: (codon_counts.get(c, 0) / total_codons * 1000.0) if total_codons else 0.0
        for c in _CODON_TABLE
    }
    dinuc_ratios: dict[str, float] = {}
    for d, n in di.items():
        if len(d) == 2 and mono[d[0]] and mono[d[1]] and base_total:
            expected = (mono[d[0]] / base_total) * (mono[d[1]] / base_total) * di_total
            if expected > 0:
                dinuc_ratios[d] = n / expected

    return ReferenceProfile(
        name=name,
        target_gc=round(target_gc, 4),
        codon_freq=codon_freq,
        dinuc_ratios=dinuc_ratios,
        gc_tol=gc_tol,
        cai_threshold=cai_threshold,
        dinuc_threshold=dinuc_threshold,
    )


# Class-characteristic Pfam markers per COMPOUND_CLASS — DATA-DRIVEN (2026-06-17):
# derived by scanning splits_core cores per class for Pfams that are frequent
# (>=30% of the class's cores) AND enriched (>=4x vs background), via
# scripts/derive_class_markers.py (-> data/.../class_markers.json). This replaces
# the old textbook-only list, which missed real subtype diversity (e.g. carotenoid
# "terpene" clusters use SQS_PSY/polyprenyl_synt, not the classic terpene cyclases).
# M2 passes if a generation contains ANY of its class's markers ("has class-defining
# machinery"); module COMPLETENESS/ordering is M11 (MODULE_PATTERNS), not M2.
OBLIGATE_DOMAINS: dict[str, list[str]] = {
    # rule: keep Pfams (freq>=0.3 & enr>=4) OR (freq>=0.08 & enr>=8) — the latter
    # adds rare-but-highly-specific SUBTYPE markers (e.g. type-III PKS chalcone
    # synthase, lanthipeptide RiPP), so the gate passes valid subtypes too.
    "ALKALOID": ["PF11991", "PF12902"],
    "ARYLPOLYENE": ["PF22817", "PF00109", "PF02801", "PF13561"],
    "BETALACTAM": ["PF00733", "PF13537", "PF02668", "PF00491", "PF13522", "PF09147"],
    "BETALACTONE": ["PF00682", "PF00501", "PF08502", "PF16177", "PF22617", "PF13193", "PF00126", "PF03466"],
    "BUTYROLACTONE": ["PF03756"],
    "CDPS": ["PF16715"],
    "ECTOINE": ["PF06339"],
    "FURAN": ["PF05655", "PF12710"],
    "HSERLACTONE": ["PF00765", "PF13444"],
    "MELANIN": ["PF06236", "PF00264"],
    "NRPS": ["PF00501", "PF00668", "PF00975"],                       # A, C, TE
    "NUCLEOSIDE": ["PF04055", "PF01142", "PF00275", "PF13640", "PF00591", "PF07831"],
    "PHENAZINE": ["PF03284", "PF12680", "PF03472"],
    "PHOSPHONATE": ["PF13714", "PF02775", "PF00266", "PF12804", "PF00155", "PF02776", "PF00171", "PF01066"],
    "PKS": ["PF00698", "PF02801", "PF00109", "PF21089", "PF14765", "PF16197", "PF00195", "PF08392"],  # +type-III
    "PKS_NRPS_HYBRID": ["PF00668", "PF00698", "PF16197", "PF22621", "PF00550", "PF00109", "PF02801", "PF00501"],
    "PUFA": ["PF07977", "PF00698", "PF16197", "PF08659", "PF02801", "PF00109", "PF00550"],
    "RESORCINOL": ["PF14399", "PF16169", "PF01061", "PF12698", "PF08541", "PF13304", "PF22818", "PF13279"],
    "RIPP": ["PF02624", "PF04055", "PF05402", "PF00881", "PF03070", "PF13353", "PF14028", "PF05114"],  # lanthi/radSAM/etc
    "SACCHARIDE": ["PF24621", "PF01761", "PF06722", "PF00201", "PF00534", "PF21036", "PF13692", "PF01041"],
    "SIDEROPHORE": ["PF00668", "PF00425", "PF00857", "PF00550", "PF13193", "PF00501", "PF00975", "PF01497"],
    "TERPENE": ["PF00348", "PF00494", "PF19086", "PF13243", "PF13249", "PF01593", "PF13450"],  # carotenoid+classic
    "OTHER": [],                                                      # catch-all; no marker
}


# ── Acceptance suite: named CHECKS → QUESTIONS ──────────────────────────────
# First-principles structure (2026-06-17, see docs/archive/REDESIGN_PLAN.md). Two layers, NOT a
# flat numbered list (the old metric_1..metric_11 numbering is gone):
#   CHECKS    = the compute units. One consistent gene caller (Prodigal/pyrodigal)
#               feeds the protein-based ones. Each returns a result dict that may
#               carry several signals (e.g. `antismash` reports both `detected` and
#               `class_match`).
#   QUESTIONS = what we actually want to know, each DERIVED by combining checks.
#               GATE questions decide accept/reject; DIAGNOSTIC ones are informative.
#
#   QUESTION               derived from                                       role
#   is_bgc                 coding_sanity ∧ antismash.detected (markers proxy)  GATE
#   correct_class          antismash.class_match (class_markers proxy)         GATE
#   novel                  kmer_novelty                                        GATE
#   proteins_plausible     protein_homology                                    diag
#   complete               module_architecture                                 diag
#   conditioning_faithful  taxon_faithfulness                                  diag
#
# antiSMASH is the GOLD-STANDARD BGC detector/classifier (owns is_bgc +
# correct_class). class_markers (Pfam) is the fast PROXY used when antiSMASH is
# skipped (quick-eval). RETIRED: synthesis feasibility, Evo2 perplexity, BiG-SCAPE.
# taxon_faithfulness REMOVED from the suite 2026-08-10. It produced no_verdict on 870/870
# records for its entire history (a CWD-relative profile path plus a case-sensitive phylum
# match), and once wired its dinucleotide sub-check rejects ~63% of REAL held-out cores, so it
# cannot be passed even by a perfect model. It also measures taxon conditioning, which is not
# what this project is testing. The check_taxon_faithfulness FUNCTION is retained for
# evo2/scripts/conditioning_experiment.py, which uses it deliberately for that other question.
CHECKS = ("coding_sanity", "antismash", "class_markers", "kmer_novelty",
          "protein_homology", "module_architecture")
# Opt-in checks. `class_probe` needs per-record scores from a MODEL-SPECIFIC probe, which the
# suite deliberately does not compute itself -- keeping evaluation.py model-agnostic is what lets
# Evo2 and GenomeOcean be graded on one instrument.
OPTIONAL_CHECKS = ("protein_foldability", "class_probe")

QUESTIONS = {
    "is_bgc":                {"gate": True},   # real coding DNA + a biosynthetic cluster
    "correct_class":         {"gate": True},   # cluster type matches the conditioned class
    "novel":                 {"gate": True},   # not memorized from training
    "proteins_plausible":    {"gate": False},  # proteins resemble known enzymes
    "complete":              {"gate": False},  # complete, correctly-ordered modules
    # CONTINUOUS class readout. Diagnostic by construction and never a gate: the probe has no
    # "not a BGC" class, reads the generator's own hidden states rather than biology, and is fit
    # on val+test. It exists to see effects too small for a threshold gate, not to judge validity.
    "class_probe_agrees":    {"gate": False},  # the class probe's argmax == the conditioned class
}
GATE_QUESTIONS = tuple(q for q, m in QUESTIONS.items() if m["gate"])
DIAGNOSTIC_QUESTIONS = tuple(q for q, m in QUESTIONS.items() if not m["gate"])

# M11 module patterns: the ordered obligate domains that form one assembly-line
# MODULE (collinearity). M11 counts modules + whether they appear IN ORDER —
# upgrading M2's binary "domains present?" to "complete, correctly-ordered modules?".
MODULE_PATTERNS = {"NRPS": ["PF00668", "PF00501", "PF00550"],   # C - A - T
                   "PKS":  ["PF00109", "PF00698", "PF00550"]}   # KS - AT - ACP
STOP_CODONS = {"TAA", "TAG", "TGA"}


@dataclass
class ORF:
    """A predicted open reading frame."""
    start: int
    end: int
    strand: int  # +1 or -1
    nt_seq: str
    aa_seq: str
    partial: bool = False  # Prodigal-flagged edge-truncated gene (incomplete)


# Gene caller: Prodigal (pyrodigal) — the standard prokaryotic gene caller, and the
# one antiSMASH itself uses, so every protein-based metric (M2/M8/M10/M11) shares one
# accurate, consistent caller. It models real gene structure (proper starts; no
# ATG-only / six-frame fragmentation that split megasynthases) and flags partial
# (edge-truncated) genes — a free completeness signal for M10. meta=True avoids
# per-sequence training so it works on a single BGC of any length.
_GENE_CALLER_ERROR: Optional[str] = None
try:  # pragma: no cover - import guard
    import pyrodigal as _pyrodigal
    _GENE_FINDER = _pyrodigal.GeneFinder(meta=True)
except Exception as _gene_caller_exc:  # keep BROAD: an ABI/API break must not crash the import,
    # but it must not be indistinguishable from "pyrodigal absent" either -- record WHY, and let
    # find_orfs() raise with the reason. A bare `except Exception` that silently swapped the gene
    # caller was the most damaging instance of this bug class in the repo (see find_orfs).
    _pyrodigal = None
    _GENE_FINDER = None
    _GENE_CALLER_ERROR = f"{type(_gene_caller_exc).__name__}: {_gene_caller_exc}"


def find_orfs(sequence: str, min_aa: int = 50) -> list[ORF]:
    """Predict genes with Prodigal (pyrodigal); return ORFs >= ``min_aa`` residues.

    Proper Prodigal gene calls (real starts, strand, no six-frame fragmentation).
    ``ORF.partial`` is True when Prodigal flags the gene as edge-truncated. Falls
    back to the legacy six-frame scanner only if pyrodigal is unavailable.
    """
    seq = sequence.upper()
    if _GENE_FINDER is None:
        # THE GENE CALLER IS A GATING RESOURCE. find_orfs feeds check_coding_sanity (whose `pass`
        # IS the is_bgc verdict whenever antiSMASH is skipped), class_markers, foldability and
        # homology. Silently swapping Prodigal for the RETIRED six-frame scanner changes the
        # instrument mid-series with no marker. Measured on one real 9.4 kb PKS core:
        #   coding_density         0.9736 -> 1.0
        #   n_orfs                 9      -> 35     (megasynthase fragmentation — the exact
        #                                            failure the 2026-06-17 rewrite retired)
        #   complete_gene_fraction 0.889  -> 1.0    (PINNED: the six-frame path never sets
        #                                            ORF.partial, so 100% of genes read complete)
        # None of that was recorded anywhere in the result dict.
        if _strict_resources():
            raise EvalResourceError(
                "find_orfs: pyrodigal gene caller unavailable"
                + (f" ({_GENE_CALLER_ERROR})" if _GENE_CALLER_ERROR else "")
                + ". Falling back to the RETIRED six-frame scanner would change coding_density, "
                  "n_orfs and pin complete_gene_fraction to 1.0 with no marker. Install pyrodigal, "
                  "or set BGC_EVAL_STRICT=0 to accept the degraded caller."
            )
        return _find_orfs_sixframe(seq, min_aa)
    orfs: list[ORF] = []
    for gene in _GENE_FINDER.find_genes(seq.encode()):
        aa = gene.translate().rstrip("*")
        if len(aa) < min_aa:
            continue
        start, end = gene.begin - 1, gene.end  # pyrodigal coords are 1-based inclusive
        nt = seq[start:end]
        if gene.strand == -1:
            nt = str(Seq(nt).reverse_complement())
        partial = bool(gene.partial_begin or gene.partial_end)
        orfs.append(ORF(start, end, gene.strand, nt, aa, partial))
    return orfs


def _find_orfs_sixframe(sequence: str, min_aa: int = 50) -> list[ORF]:
    """Legacy six-frame ATG->stop scanner. Fallback only when pyrodigal is absent."""
    seq = sequence.upper()
    orfs: list[ORF] = []
    for strand, nuc in [(+1, seq), (-1, str(Seq(seq).reverse_complement()))]:
        for frame in range(3):
            i = frame
            while i + 3 <= len(nuc):
                codon = nuc[i : i + 3]
                if codon == "ATG":
                    # Start codon found, scan for stop
                    aa_list: list[str] = []
                    j = i
                    while j + 3 <= len(nuc):
                        c = nuc[j : j + 3]
                        aa = _CODON_TABLE.get(c, "X")
                        if aa == "*":
                            break
                        aa_list.append(aa)
                        j += 3
                    if len(aa_list) >= min_aa:
                        nt = nuc[i : j + 3] if j + 3 <= len(nuc) else nuc[i:j]
                        if strand == +1:
                            orfs.append(ORF(i, j + 3, strand, nt, "".join(aa_list)))
                        else:
                            real_start = len(seq) - (j + 3)
                            real_end = len(seq) - i
                            orfs.append(ORF(real_start, real_end, strand, nt, "".join(aa_list)))
                    i = j + 3
                    continue
                i += 3
    # Deduplicate overlapping ORFs — keep longest per region
    orfs.sort(key=lambda o: -(o.end - o.start))
    kept: list[ORF] = []
    used: set[tuple[int, int]] = set()
    for o in orfs:
        key = (o.start // 100, o.strand)
        if key not in used:
            kept.append(o)
            used.add(key)
    return kept


# ---------------------------------------------------------------------------
# Check: antismash — BGC detection + class (is_bgc / correct_class)
# ---------------------------------------------------------------------------


class EvalResourceError(RuntimeError):
    """A required evaluation resource is missing (database, HMM file, class map, binary).

    THIS IS A CONFIGURATION BUG, NOT A PROPERTY OF THE SEQUENCE. It must never be reported as a
    measurement. Measured 2026-07-31, three instances in one afternoon, each producing output
    indistinguishable from a real research result:

      * check_antismash without `databases_dir`      -> every sequence "skipped"; looked like a
        broken antiSMASH install and nearly triggered a 15 GB re-download.
      * check_antismash without `class_map`          -> antiSMASH emits "T1PKS"/"lanthipeptide-
        class-i" while our vocabulary is "PKS"/"RIPP", so class_match compares raw strings. REAL
        held-out cores scored correct_class 0.125 instead of 0.750 -- i.e. exactly like "the model
        cannot produce the right class", the central claim of this project.
      * check_class_markers without `pfam_hmm_path`  -> every sequence skipped; a caller doing
        bool(r.get("markers_present")) turned "could not measure" into "measured absent" and
        produced an all-zero Phase 3 table.

    A crash announces itself. A silent skip that a caller coerces to False does not -- and this
    project already lost weeks to a 0/30 that was an instrument artifact. Hence: raise.

    Escape hatch: BGC_EVAL_STRICT=0 restores the old skip-and-continue behaviour globally, for
    the deliberately-optional checks or for triage on a partially-provisioned host.
    """


def _strict_resources() -> bool:
    return os.environ.get("BGC_EVAL_STRICT", "1").strip().lower() not in ("0", "false", "no", "off")


def _resource_missing(result: dict[str, Any], reason: str, *, gating: bool = True) -> dict[str, Any]:
    """Handle a missing resource.

    `gating=True`  -> the check owns a GATE (is_bgc / correct_class). Raise: a gate that cannot
                      be evaluated must stop the run, never quietly become a negative.
    `gating=False` -> a diagnostic/opt-in check (foldability, homology). Skip, but tag
                      `skip_kind="resource"` so a caller can tell "not configured" apart from
                      "measured absent".
    """
    if gating and _strict_resources():
        raise EvalResourceError(
            f"{result.get('check', 'check')}: {reason}. This is a missing resource, not a "
            f"result -- fix the configuration, or set BGC_EVAL_STRICT=0 to downgrade to a skip."
        )
    result["skipped"] = True
    result["skip_kind"] = "resource"
    result["reason"] = reason
    return result


def check_class_probe(
    probe_scores: Optional[dict[str, float]],
    expected_class: str = "",
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """CONTINUOUS class readout: what a linear class probe makes of this sequence.

    WHY THIS CHECK EXISTS. Every other class readout in the suite is BINARY -- antiSMASH called a
    cluster of type X, or a class-defining Pfam domain was found. Both are threshold instruments
    and both are insensitive at the lengths we generate. Measured 2026-08-10:

        class_markers TPR on real class DNA at 3 kb : 0.717   (NRPS .867 PKS .600 TERP .867 RIPP .533)
        antiSMASH is_bgc on seeded 3 kb generations : ~0.33   (so a steering arm's CEILING is 0.33)

    Against instruments like that an intervention can move the model substantially toward a class
    and still score exactly 0.000, because "somewhat more PKS-like" is not a domain. A null from a
    threshold gate bounds a LARGE effect and is silent on a small one -- which is precisely where a
    weak-but-real conditioning effect would live. This check reports a PROBABILITY instead, so a
    partial shift is visible.

    IT IS A DIAGNOSTIC AND MUST NEVER GATE. Three independent reasons, all load-bearing:

      1. NO NEGATIVE CLASS. The probe is trained on BGC classes only, so it has no way to say
         "this is not a BGC at all" -- it always returns a distribution over BGC classes and
         always has an argmax. MEASURED 2026-08-10 (evo2/scripts/calibrate_class_probe.py) on 25
         real non-BGC genomic windows vs 60 real cores, both at 3 kb:

             argmax == true class on real cores  : 0.900   (the marker gate manages 0.717)
             mean argmax confidence, real cores  : 0.986
             mean argmax confidence, NON-BGC DNA : 0.900   <-- 24/25 are >= 0.5 confident
             where non-BGC DNA lands             : RIPP 14, NRPS 5, BETALACTONE 4, PHOSPHONATE 2

         It is nearly as confident on ordinary bacterial DNA as on real clusters. It measures
         "which class does this most RESEMBLE", never "is this a cluster". Note also that RIPP
         is the default attractor for unremarkable DNA, so an arm targeting RIPP gets an
         inflated p_expected -- harmless in a paired comparison, misleading as a bare number.
      2. IT READS THE MODEL, NOT BIOLOGY. The scores come from the generator's own hidden states.
         A model can write DNA its own probe calls PKS while antiSMASH finds nothing. antiSMASH
         is the ground truth for validity; this is a sensitivity instrument, not a verdict.
      3. LEAKAGE. The probe in use is fit on val+test (the standing debt in decisions.md).

    Its real power is PAIRED comparison -- same seed exemplar, steered vs unsteered vs
    shuffled-label -- where the shared bias cancels and only the difference is read.

    `probe_scores` maps class name -> probability and is computed UPSTREAM by the model-specific
    scorer (`evo2/scripts/probe_score_generations.py`), which keeps this suite model-agnostic:
    each track supplies scores from its own probe and is graded here identically.
    """
    result: dict[str, Any] = {"check": "class_probe"}
    if not probe_scores:
        # Diagnostic, so a missing score SKIPS rather than raising -- but it is tagged
        # `resource`, never silently folded into a FAIL.
        return _resource_missing(
            result,
            "no probe scores supplied for this record (run "
            "evo2/scripts/probe_score_generations.py --emit-sidecar and pass --probe-scores)",
            gating=False)

    scores = {str(k): float(v) for k, v in probe_scores.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    result["n_classes"] = len(scores)
    result["argmax"] = ranked[0][0]
    result["argmax_p"] = ranked[0][1]
    result["top3"] = [{"class": c, "p": p} for c, p in ranked[:3]]
    # Entropy in bits, normalised by log2(n): 0 = certain, 1 = uniform. A probe that is uniform
    # on everything is not reading anything, and that is invisible if only argmax is reported.
    n = max(len(scores), 2)
    ent = -sum(p * math.log2(p) for _, p in ranked if p > 0)
    result["entropy_norm"] = ent / math.log2(n)

    if not expected_class:
        result["pass"] = None
        result["no_expected_class"] = True
        result["p_expected"] = None
        return result
    if expected_class not in scores:
        # An OOV class cannot be scored. Say so rather than reporting p=0, which reads as
        # "the probe is certain it is not this class".
        result["unknown_class"] = expected_class
        result["pass"] = None
        result["p_expected"] = None
        result["reason"] = (f"{expected_class} is not one of the probe's {len(scores)} classes "
                            f"— not scoreable, which is not the same as scoring zero")
        return result

    p_exp = scores[expected_class]
    best_other = max((p for c, p in scores.items() if c != expected_class), default=0.0)
    result["p_expected"] = p_exp
    result["p_chance"] = 1.0 / len(scores)
    result["margin"] = p_exp - best_other          # >0 iff the expected class is the argmax
    result["argmax_matches"] = result["argmax"] == expected_class
    result["pass"] = bool(result["argmax_matches"] and result["margin"] >= min_margin)
    return result


def check_antismash(
    sequence: str,
    accession: str = "query",
    expected_class: str = "",
    class_map: Optional[dict[str, str]] = None,
    timeout: int = 600,
    databases_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Run antiSMASH on a sequence and check predicted BGC class.

    ``databases_dir`` points antiSMASH at a databases root installed outside the
    package default (we keep them on /data2; see evo2/scripts/run_eval.sh). When None,
    antiSMASH uses its built-in default path.
    """
    result: dict[str, Any] = {"check": "antismash"}

    # PREFLIGHT. Decide input validity HERE, not by pattern-matching antiSMASH's stderr after
    # the fact. The first attempt at this matched only the "smaller than minimum length" message,
    # so all-N and non-nucleotide generations -- both real model failure modes -- still raised
    # EvalResourceError, and with the misleading text "prerequisite DBs missing" even when the
    # databases were supplied and fine. Classifying the input up front makes the rule exact:
    #   invalid input  -> a MEASURED negative (the model produced something unscoreable)
    #   valid input, non-zero exit -> a genuine CONFIG/exec failure -> raise
    _seq = (sequence or "").upper()
    _acgt = sum(_seq.count(b) for b in "ACGT")
    _why = None
    if len(_seq) < 1000:
        _why = f"sequence is {len(_seq)} nt; antiSMASH requires >= 1000"
    elif _acgt < 0.8 * len(_seq):
        _why = (f"only {_acgt}/{len(_seq)} = {_acgt / max(len(_seq), 1):.0%} of characters are "
                f"A/C/G/T (all-N or non-nucleotide output)")
    if _why:
        result.update({"detected": False, "class_match": False if expected_class else None,
                       "pass": False if expected_class else None,
                       "input_rejected": True, "reason": f"not scoreable: {_why}"})
        if not expected_class:
            result["no_expected_class"] = True
        return result

    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / f"{accession}.fasta"
        fasta.write_text(f">{accession}\n{sequence}\n")
        outdir = Path(tmp) / "antismash_out"

        cmd = [
            "antismash",
            str(fasta),
            "--output-dir", str(outdir),
            "--genefinding-tool", "prodigal",
            "--minimal",
        ]
        if databases_dir:
            cmd += ["--databases", str(databases_dir)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False,
            )
        except FileNotFoundError:
            return _resource_missing(result, "antismash not installed", gating=True)
        except subprocess.TimeoutExpired:
            # Tag it like every other failure path. An untagged skip is indistinguishable from a
            # measured negative downstream, and a timeout silently swaps the is_bgc verdict from
            # antiSMASH to the much looser Pfam proxy MID-SERIES in eval_track.jsonl.
            result["skip_kind"] = "timeout"
            result["skipped"] = True
            result["reason"] = f"timeout after {timeout}s"
            return result

        result["returncode"] = proc.returncode

        # Parse results JSON
        results_json = outdir / f"{accession}.json"
        if not results_json.exists():
            # Try finding any json
            jsons = list(outdir.glob("*.json"))
            if jsons:
                results_json = jsons[0]

        if results_json.exists():
            try:
                as_data = json.loads(results_json.read_text())
                all_products: list[str] = []
                for record in as_data.get("records", []):
                    for region in record.get("areas", record.get("regions", [])):
                        products = region.get("products", region.get("product", []))
                        if isinstance(products, str):
                            products = [products]
                        all_products.extend(products)
                result["predicted_products"] = all_products
                # is_bgc signal: antiSMASH detected at least one biosynthetic cluster.
                result["detected"] = len(all_products) > 0

                # map antiSMASH products -> our class vocabulary (keys are lowercased
                # in load_class_map; ANY mapped product matching = right class).
                if class_map:
                    # An antiSMASH upgrade that adds product rules makes the NEW products
                    # silently map to OTHER, so genuine PKS/RIPP regions score class_match=False
                    # and correct_class deflates for exactly those classes -- which reads as a
                    # biological finding ("the model handles PKS but not RIPP") rather than a
                    # stale YAML. Record what fell through so it is visible in the result.
                    unmapped = sorted({p for p in all_products if p.lower() not in class_map})
                    if unmapped:
                        result["unmapped_products"] = unmapped
                    mapped = {class_map.get(p.lower(), "OTHER") for p in all_products}
                elif expected_class:
                    # NO SKIP HERE -- this branch silently returns WRONG classes. antiSMASH emits
                    # "T1PKS"/"lanthipeptide-class-i"; our vocabulary is "PKS"/"RIPP". Comparing
                    # raw uppercased products scored REAL held-out cores at correct_class 0.125
                    # instead of 0.750 -- indistinguishable from "the model cannot produce the
                    # right class". Only raise when a class verdict was actually requested;
                    # detection-only callers legitimately need no map.
                    return _resource_missing(
                        result,
                        "expected_class was requested but no class_map supplied — mapped_classes "
                        "would be raw antiSMASH products and class_match would be silently wrong. "
                        "Load config/compound_class_map.yaml via bgc_pipeline.class_map.load_class_map",
                        gating=True)
                else:
                    mapped = {p.upper() for p in all_products}
                result["mapped_classes"] = sorted(mapped)

                # correct_class signal: does the cluster type match the conditioned class?
                if expected_class:
                    exp = expected_class.upper()
                    if exp in ("PKS_NRPS_HYBRID", "HYBRID"):
                        # a hybrid = both PKS and NRPS machinery present in the region(s)
                        result["class_match"] = ("PKS_NRPS_HYBRID" in mapped
                                                 or {"PKS", "NRPS"} <= mapped)
                    else:
                        result["class_match"] = exp in mapped
                    result["pass"] = result["class_match"]   # back-compat alias
            except (json.JSONDecodeError, KeyError) as e:
                # Malformed/unexpected antiSMASH JSON: the tool gave no usable verdict.
                # SKIP (not FAIL) so derive_questions falls back to the class_markers
                # proxy CONSISTENTLY for both is_bgc and correct_class — never a
                # half-state where detected=False but class_match is left unset.
                result["parse_error"] = str(e)
                result["skipped"] = True
        else:
            result["no_results_json"] = True
            # Distinguish "antismash ran but found nothing" (rc==0 → genuinely NOT a
            # BGC: detected=False) from "antismash could not run" (rc!=0, no JSON —
            # almost always missing DBs). The latter taught us nothing → SKIP.
            if proc.returncode != 0:
                if proc.stderr:
                    result["stderr_tail"] = proc.stderr.strip()[-500:]
                # INPUT REJECTION IS A MEASUREMENT, NOT A CONFIGURATION ERROR. antiSMASH exits 1
                # when every input record is under its 1000 nt minimum -- which is exactly what a
                # model emitting short/degenerate output produces. Measured: all 102 historical
                # antismash "skips" across 25 reports were this, with sequence lengths min=0,
                # median=117, max=990. Raising here would make a MODEL FAILURE MODE crash its own
                # evaluation, and would have aborted 11.7% of the records on record.
                _err = (proc.stderr or "").lower()
                if ("smaller than minimum length" in _err or "no valid records" in _err
                        or "input records smaller" in _err):
                    result["detected"] = False
                    result["class_match"] = False
                    result["pass"] = False
                    result["input_rejected"] = True
                    result["reason"] = ("antiSMASH rejected the input (below its minimum record "
                                        "length) — scored as NOT a BGC, which is the correct "
                                        "verdict for a too-short generation")
                    return result
                return _resource_missing(
                    result,
                    f"antismash exited {proc.returncode} without results — prerequisite DBs "
                    f"missing. Pass databases_dir=/data2/ds85/antismash_db (they are NOT in the "
                    f"package directory, which holds only a README)",
                    gating=True)
            else:
                result["detected"] = False
                result["class_match"] = False
                result["pass"] = False

    # Default: ran but no verdict set → not detected. `pass` is the class_match alias, so it is
    # only meaningful when a class was REQUESTED.
    #
    # `setdefault("pass", False)` fabricated a measured NEGATIVE for unconditional runs: with
    # expected_class="" a genuinely DETECTED cluster reported pass=False -> verdict FAIL.
    # Measured in shipped reports (GenomeOcean zero-shot, expected_class empty on every record):
    # per_check antismash 0 PASS / 216 FAIL / pass_rate 0.000 while antismash.detected was True
    # on 27/216 -- and class_markers in the SAME report correctly reported no_verdict, so two
    # rows of one report contradicted each other. Mirror class_markers instead.
    if not result.get("skipped"):
        result.setdefault("detected", False)
        if expected_class:
            result.setdefault("pass", False)
        else:
            result["pass"] = None
            result["no_expected_class"] = True

    return result


# ---------------------------------------------------------------------------
# Check: class_markers — functional domain recovery (pyhmmer + Pfam)
# ---------------------------------------------------------------------------

def check_class_markers(
    sequence: str,
    expected_class: str = "",
    pfam_hmm_path: Optional[Path] = None,
    evalue_threshold: float = 1e-10,
) -> dict[str, Any]:
    """Scan predicted ORFs against Pfam using pyhmmer."""
    result: dict[str, Any] = {"check": "class_markers"}

    try:
        import pyhmmer
        from pyhmmer.easel import TextSequence, Alphabet
        from pyhmmer.plan7 import HMMFile
    except ImportError:
        return _resource_missing(result, "pyhmmer not installed", gating=True)

    if pfam_hmm_path is None or not Path(pfam_hmm_path).exists():
        return _resource_missing(
            result,
            f"Pfam HMM not found at {pfam_hmm_path} (expected /data2/ds85/pfam/Pfam-A.hmm). "
            f"Without it EVERY sequence skips, and a caller coercing markers_present to bool "
            f"reads that as 'markers absent'",
            gating=True)

    orfs = find_orfs(sequence)
    result["orf_count"] = len(orfs)

    if not orfs:
        result["domains_found"] = []
        result["pass"] = False
        result["reason"] = "no ORFs found"
        return result

    # Build digital sequences for pyhmmer
    alphabet = Alphabet.amino()
    sequences = []
    for i, orf in enumerate(orfs):
        name = f"orf_{i}".encode()
        ts = TextSequence(name=name, sequence=orf.aa_seq)
        sequences.append(ts.digitize(alphabet))

    # Scan against Pfam
    domains_found: list[dict[str, Any]] = []
    domain_accessions: set[str] = set()

    with HMMFile(str(pfam_hmm_path)) as hmm_file:
        for top_hits in pyhmmer.hmmsearch(hmm_file, sequences, E=evalue_threshold):
            query_hmm = top_hits.query
            raw_acc = query_hmm.accession
            acc = raw_acc.decode() if isinstance(raw_acc, bytes) else str(raw_acc) if raw_acc else ""
            raw_name = query_hmm.name
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name) if raw_name else ""
            # Strip version from accession (PF00109.29 -> PF00109)
            acc_base = acc.split(".")[0] if acc else ""
            for hit in top_hits:
                if hit.included:
                    for domain in hit.domains:
                        if domain.included:
                            raw_hit_name = hit.name
                            hit_name = raw_hit_name.decode() if isinstance(raw_hit_name, bytes) else str(raw_hit_name)
                            domains_found.append({
                                "pfam_accession": acc_base,
                                "pfam_name": name,
                                "target": hit_name,
                                "evalue": float(domain.i_evalue),
                                "score": float(domain.score),
                                "env_from": int(getattr(domain, "env_from", 0) or 0),  # within-ORF aa pos (for M11 ordering)
                            })
                            domain_accessions.add(acc_base)

    result["domains_found"] = domains_found
    result["unique_domain_accessions"] = sorted(domain_accessions)
    result["domain_count"] = len(domains_found)
    # ORF coordinates (for M11 module-architecture ordering across ORFs)
    result["orfs_meta"] = [{"i": i, "start": o.start, "end": o.end, "aa_len": len(o.aa_seq)}
                           for i, o in enumerate(orfs)]

    # Check class markers (data-derived, see OBLIGATE_DOMAINS). ANY-of semantics for
    # ALL classes: a valid member has at least one class-characteristic enzyme. Module
    # COMPLETENESS + ordering is M11's job (MODULE_PATTERNS), not this gate. The
    # graded "fraction of markers present" (obligate_domains − missing_obligate) is
    # the climb signal surfaced by quick_eval.
    if expected_class:
        # An OOV class name (typo, new class, wrong vocabulary) yields [] exactly like the
        # deliberately-empty "OTHER" entry, so it silently becomes no_verdict -- which
        # eval_suite_driver then counts in the denominator of correct_class.
        if expected_class not in OBLIGATE_DOMAINS:
            result["unknown_class"] = expected_class
        markers = OBLIGATE_DOMAINS.get(expected_class, [])
        if markers:
            present = [d for d in markers if d in domain_accessions]
            missing = [d for d in markers if d not in domain_accessions]
            result["obligate_domains"] = markers          # = class markers
            result["missing_obligate"] = missing
            result["markers_present"] = present
            result["pass"] = len(present) >= 1            # ANY class marker present
        else:
            result["obligate_domains"] = []
            result["pass"] = None  # No markers defined for this class (e.g. OTHER)
            result["note"] = f"No class markers defined for class {expected_class}"
    else:
        # No expected_class was supplied, so no class verdict is possible. Say so explicitly:
        # the previous silent `pass=None` with NO markers_present key meant a caller doing
        # bool(r.get("markers_present")) read "no class requested" as "markers absent".
        result["pass"] = None
        result["no_expected_class"] = True
        result["markers_present"] = None      # explicit "not evaluated", not an empty list

    return result


# ---------------------------------------------------------------------------
# Check: protein_foldability — ESMFold (OPTIONAL, opt-in)
# ---------------------------------------------------------------------------

def check_protein_foldability(
    sequence: str,
    max_orfs: int = 5,
    foldseek_db: Optional[str] = None,
) -> dict[str, Any]:
    """Predict protein structures with ESMFold and assess foldability.

    Uses the HuggingFace transformers EsmForProteinFolding implementation
    (transformers==4.46.3), which does not require OpenFold as a dependency.
    Requires: pip install transformers==4.46.3 accelerate torch==2.5.1+cu124
    """
    result: dict[str, Any] = {"check": "protein_foldability"}

    try:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError:
        return _resource_missing(result, "transformers not installed (pip install transformers==4.46.3 accelerate)", gating=False)

    orfs = find_orfs(sequence)[:max_orfs]
    result["orf_count"] = len(orfs)

    if not orfs:
        result["pass"] = False
        result["reason"] = "no ORFs found"
        return result

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        )
        model = model.to(device).eval()
    except Exception as e:
        return _resource_missing(result, f"ESMFold model load failed: {e}", gating=False)

    orf_results: list[dict[str, Any]] = []
    for i, orf in enumerate(orfs):
        aa = orf.aa_seq[:1000]  # ESMFold limit
        try:
            inputs = tokenizer([aa], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            # out.plddt shape: (batch, seq_len, 37 atoms) — mean over atoms per residue
            plddt_per_res = out.plddt[0].mean(dim=-1).cpu().numpy()
            plddt_mean = float(plddt_per_res.mean()) * 100.0  # scale 0-1 → 0-100
            orf_results.append({
                "orf_index": i,
                "length_aa": len(orf.aa_seq),
                "plddt_mean": round(plddt_mean, 2),
                "passes_plddt_70": plddt_mean > 70.0,
            })
        except Exception as e:
            orf_results.append({"orf_index": i, "error": str(e)})

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result["orf_results"] = orf_results
    passing = [o for o in orf_results if o.get("passes_plddt_70")]
    # Per-ORF exceptions (CUDA OOM, tokenizer error) were appended to orf_results and counted
    # in the DENOMINATOR, so "ESMFold could not run here" was reported as "these proteins do not
    # fold". Score only ORFs that actually produced a pLDDT.
    scored = [o for o in orf_results if "error" not in o]
    result["n_orfs_errored"] = len(orf_results) - len(scored)
    if not scored:
        result["skipped"] = True
        result["skip_kind"] = "runtime"
        result["reason"] = f"all {len(orf_results)} ORFs errored during folding; no pLDDT obtained"
        return result
    result["pass"] = len(passing) > len(scored) // 2

    return result


# Metrics 4 (synthesis feasibility), 5 (Evo2 base perplexity) and 6 (BiG-SCAPE)
# were RETIRED in the first-principles eval rewrite (2026-06-17):
#   M4 = a wet-lab MANUFACTURING axis (out of scope here);
#   M5 = base-model perplexity of fine-tune output was near-circular / uninformative;
#   M6 = an unimplemented stub that never produced a verdict.
# Novelty is owned by M9; protein plausibility by M8. See docs/archive/REDESIGN_PLAN.md.


def check_kmer_novelty(
    novelty: Optional[dict[str, Any]],
    fail_threshold: float = 0.95,
    warn_threshold: float = 0.80,
) -> dict[str, Any]:
    """Nucleotide novelty / anti-memorization GATE (audit C6/M13).

    Novelty is a query-vs-corpus operation, so it is computed in BATCH by the eval
    driver (one streaming pass over the reference corpus via
    ``scripts/memorization_check.scan_corpus``) and the per-sequence result passed
    in here. Gates on max canonical-k-mer containment to the nearest reference BGC:
      - FAIL_memorized if containment >= fail_threshold,
      - WARN          if >= warn_threshold,
      - PASS_novel    otherwise.
    If no novelty scan is supplied the metric is SKIPPED — and a skipped novelty
    gate means novelty is UNVERIFIED; it must never be read as a pass.
    """
    result: dict[str, Any] = {"check": "kmer_novelty"}
    if not novelty:
        result["skipped"] = True
        result["reason"] = ("no novelty scan supplied; run memorization_check.scan_corpus "
                            "over the full reference corpus and pass the per-sequence result")
        return result
    # A GATE MUST NOT DEFAULT TO ITS PASSING VALUE. `.get("max_containment", 0.0)` returned 0.0
    # -- maximal novelty -- for a novelty record that exists but lacks the key (schema drift, a
    # truncated memorization report, a partially-written scan). A memorized sequence would be
    # certified novel. Absence of evidence must be a skip, never a pass.
    if "max_containment" not in novelty:
        result["skipped"] = True
        result["skip_kind"] = "malformed"
        result["reason"] = ("novelty record present but has no 'max_containment' key — the "
                            "anti-memorization gate cannot be evaluated; refusing to default to PASS")
        return result
    cont = float(novelty["max_containment"])
    result["max_containment"] = round(cont, 4)
    result["nearest_accession"] = novelty.get("nearest_accession")
    if cont >= fail_threshold:
        result["verdict"], result["pass"] = "FAIL_memorized", False
    elif cont >= warn_threshold:
        result["verdict"], result["pass"] = "WARN", None
    else:
        result["verdict"], result["pass"] = "PASS_novel", True
    return result


# ---------------------------------------------------------------------------
# Check: taxon_faithfulness — organism compatibility (CAI + GC + dinucleotide stats)
# ---------------------------------------------------------------------------

def _score_against_profile(seq: str, profile: ReferenceProfile) -> dict[str, Any]:
    """Score GC / CAI / dinucleotide agreement of a sequence against one profile."""
    block: dict[str, Any] = {"reference": profile.name}

    # GC content
    gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0
    gc_dev = abs(gc - profile.target_gc)
    block["gc_content"] = round(gc, 4)
    block["gc_target"] = profile.target_gc
    block["gc_deviation"] = round(gc_dev, 4)
    block["gc_pass"] = gc_dev < profile.gc_tol

    # Codon Adaptation Index vs this reference's codon usage
    cai = _compute_cai(seq, profile.codon_freq)
    block["cai"] = round(cai, 4) if cai is not None else None
    block["cai_pass"] = cai is not None and cai > profile.cai_threshold

    # Dinucleotide RMSD vs this reference's observed/expected ratios
    obs = _dinucleotide_frequencies(seq)
    sq, n = 0.0, 0
    for d, ref in profile.dinuc_ratios.items():
        if d in obs:
            sq += (obs[d] - ref) ** 2
            n += 1
    rmsd = math.sqrt(sq / n) if n > 0 else float("inf")
    block["dinucleotide_rmsd"] = round(rmsd, 4)
    block["dinucleotide_pass"] = rmsd < profile.dinuc_threshold

    passes = [block["gc_pass"], block["cai_pass"], block["dinucleotide_pass"]]
    block["composite_pass"] = sum(1 for p in passes if p is True) >= 2  # >=2 of 3
    return block


def check_taxon_faithfulness(
    sequence: str,
    taxon_profile: Optional[ReferenceProfile] = None,
    chassis_profile: Optional[ReferenceProfile] = ECOLI_PROFILE,
) -> dict[str, Any]:
    """Organism compatibility via CAI + GC + dinucleotide statistics.

    Two distinct sub-scores are reported and never conflated:

    - ``faithfulness`` — agreement with ``taxon_profile``, the organism the
      sequence was conditioned on. This is what the metric's PASS verdict
      reflects. If no ``taxon_profile`` is supplied the verdict is ``None``
      (``no_verdict``) — a correctly-trained, non-E.coli model must NOT be
      auto-failed merely because we lack a taxon reference.
    - ``chassis_expressibility`` — agreement with ``chassis_profile`` (E. coli by
      default), the wet-lab expression host. Reported for information only; it
      never gates the verdict, because the Phase-1 objective is faithful
      generation, not E. coli codon-optimisation (that is a separate recoding
      step). See docs/archive/AUDIT_FINDINGS.md C4.
    """
    result: dict[str, Any] = {"check": "taxon_faithfulness"}
    seq = sequence.upper()

    # (a) Faithfulness to the conditioned taxon — drives the verdict.
    if taxon_profile is not None:
        faith = _score_against_profile(seq, taxon_profile)
        result["faithfulness"] = faith
        result["pass"] = faith["composite_pass"]
    else:
        # No taxon reference available: report raw stats but withhold a verdict
        # rather than grade against the wrong organism.
        result["faithfulness"] = None
        result["pass"] = None
        result["verdict_note"] = (
            "no taxon_profile supplied; faithfulness ungraded "
            "(supply the conditioned taxon's profile for a PASS/FAIL verdict)"
        )

    # (b) Chassis expressibility — informational only, never gates the verdict.
    if chassis_profile is not None:
        result["chassis_expressibility"] = _score_against_profile(seq, chassis_profile)

    return result


def _compute_cai(
    sequence: str,
    codon_freq: Optional[dict[str, float]] = None,
) -> Optional[float]:
    """Compute Codon Adaptation Index relative to a reference codon-usage table.

    Defaults to E. coli K-12 for backward compatibility; pass a taxon's
    ``codon_freq`` (e.g. from a ReferenceProfile) to score faithfulness to the
    conditioned organism instead.
    """
    if codon_freq is None:
        codon_freq = _ECOLI_CODON_FREQ
    seq = sequence.upper()
    # Group codons by amino acid to find max frequency per AA
    aa_to_codons: dict[str, list[str]] = {}
    for codon, aa in _CODON_TABLE.items():
        if aa != "*":
            aa_to_codons.setdefault(aa, []).append(codon)

    # Compute relative adaptiveness (w) for each codon
    w: dict[str, float] = {}
    for aa, codons in aa_to_codons.items():
        max_freq = max(codon_freq.get(c, 0.0) for c in codons)
        if max_freq == 0:
            continue
        for c in codons:
            freq = codon_freq.get(c, 0.0)
            w[c] = freq / max_freq if max_freq > 0 else 0.0

    # Score the sequence
    log_sum = 0.0
    n_codons = 0
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in w and w[codon] > 0:
            log_sum += math.log(w[codon])
            n_codons += 1

    if n_codons == 0:
        return None
    return math.exp(log_sum / n_codons)


def _dinucleotide_frequencies(seq: str) -> dict[str, float]:
    """Compute observed/expected dinucleotide ratios."""
    if len(seq) < 2:
        return {}
    # Mono counts
    mono = Counter(seq)
    total = len(seq)
    # Di counts
    di: Counter[str] = Counter()
    for i in range(len(seq) - 1):
        di[seq[i : i + 2]] += 1
    total_di = len(seq) - 1

    ratios: dict[str, float] = {}
    for d in di:
        if len(d) == 2:
            expected = (mono[d[0]] / total) * (mono[d[1]] / total) * total_di
            if expected > 0:
                ratios[d] = di[d] / expected
    return ratios


# ---------------------------------------------------------------------------
# Check: protein_homology — protein sequence homology (MMseqs2)
# ---------------------------------------------------------------------------

def check_protein_homology(
    sequence: str,
    accession: str = "query",
    db_path: Optional[str] = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Search predicted ORFs against a protein database using MMseqs2."""
    result: dict[str, Any] = {"check": "protein_homology"}

    try:
        _probe = subprocess.run(["mmseqs", "version"], capture_output=True, timeout=10, check=False)
        if _probe.returncode != 0:
            # Present on PATH but broken (bad build, missing shared lib). Discarding the return
            # code let the suite proceed and then report "no ORF resembles a known enzyme".
            return _resource_missing(
                result, f"mmseqs present but `mmseqs version` exited {_probe.returncode}",
                gating=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _resource_missing(result, "mmseqs2 not installed", gating=False)

    orfs = find_orfs(sequence)
    result["orf_count"] = len(orfs)
    if not orfs:
        result["hits"] = []
        return result

    with tempfile.TemporaryDirectory() as tmp:
        # Write ORF protein sequences
        query_fasta = Path(tmp) / "query.fasta"
        with query_fasta.open("w") as f:
            for i, orf in enumerate(orfs[:20]):  # Limit to top 20 ORFs
                f.write(f">orf_{i}_len{len(orf.aa_seq)}\n{orf.aa_seq}\n")

        if db_path is None:
            return _resource_missing(result, "no database path provided (need UniRef50 MMseqs2 DB)", gating=False)

        result_tsv = Path(tmp) / "result.tsv"
        tmpdir = Path(tmp) / "mmseqs_tmp"
        tmpdir.mkdir()

        cmd = [
            "mmseqs", "easy-search",
            str(query_fasta), db_path, str(result_tsv), str(tmpdir),
            "--format-output", "query,target,pident,evalue,bits",
            "-e", "1e-5",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            # Tag it like every other failure path. An untagged skip is indistinguishable from a
            # measured negative downstream, and a timeout silently swaps the is_bgc verdict from
            # antiSMASH to the much looser Pfam proxy MID-SERIES in eval_track.jsonl.
            result["skip_kind"] = "timeout"
            result["skipped"] = True
            result["reason"] = f"timeout after {timeout}s"
            return result

        if proc.returncode != 0:
            # The return code was assigned and never read: the function branched only on whether
            # the TSV existed. A failed search (bad DB, out of disk) produced no TSV, fell to the
            # `else`, and reported hits=[] -- "no ORF resembles any known enzyme".
            return _resource_missing(
                result,
                f"mmseqs easy-search exited {proc.returncode}: "
                f"{(proc.stderr or '').strip()[-200:]}",
                gating=False)

        if result_tsv.exists():
            hits: list[dict[str, Any]] = []
            for line in result_tsv.read_text().splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    hits.append({
                        "query": parts[0],
                        "target": parts[1],
                        "pident": float(parts[2]),
                        "evalue": float(parts[3]),
                        "bitscore": float(parts[4]),
                    })
            result["hits"] = hits[:50]  # Keep top 50
            if hits:
                pidents = [h["pident"] for h in hits]
                result["max_pident"] = max(pidents)
                result["mean_pident"] = round(sum(pidents) / len(pidents), 2)
                # NOTE: this is a PLAUSIBILITY signal (do the ORFs resemble known
                # enzymes?), NOT a novelty/memorization signal. High identity to a
                # *reference protein DB* is EXPECTED for a plausible novel BGC, since
                # biosynthetic enzyme families are conserved. Anti-memorization is
                # owned solely by kmer_novelty (k-mer containment vs the TRAINING set).
                result["high_identity_to_known"] = max(pidents) > 95.0
                n_queried = min(len(orfs), 20)  # only the first 20 ORFs are searched
                hit_orfs = {h["query"] for h in hits}
                result["homology_fraction"] = (
                    round(len(hit_orfs) / n_queried, 3) if n_queried else 0.0
                )
                # Diagnostic (non-gating) verdict: do most ORFs look like real proteins?
                result["pass"] = result["homology_fraction"] >= 0.5
            else:
                # ZERO hits is the WORST outcome for proteins_plausible, not an absent one.
                # Leaving `pass` unset made pass_rate a mean over only the sequences that
                # already had a hit: 9 hit-less + 1 homologous reported 1.000 instead of 0.100.
                result["max_pident"] = 0.0
                result["homology_fraction"] = 0.0
                result["pass"] = False
        else:
            result["hits"] = []
            result["homology_fraction"] = 0.0
            result["pass"] = False

    return result



# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Configuration for the evaluation pipeline."""
    pfam_hmm_path: Optional[Path] = None
    mibig_gbk_dir: Optional[Path] = None
    mmseqs2_db: Optional[str] = None
    antismash_db_dir: Optional[str] = None
    class_map: Optional[dict[str, str]] = None
    antismash_timeout: int = 600
    skip_checks: list[str] = field(default_factory=list)   # check NAMES to skip (see CHECKS)
    run_protein_foldability: bool = False  # ESMFold: GPU-expensive, off by default (opt-in spot-check)
    # taxon_faithfulness: organism-compatibility references (see docs/archive/AUDIT_FINDINGS.md C4).
    # `chassis_profile` is the wet-lab host (E. coli) used for the informational
    # expressibility sub-score. `taxon_profiles` maps a taxon key (e.g. a phylum
    # token like "P__ACTINOMYCETOTA") to the empirical profile used to grade
    # faithfulness. Populate via load_taxon_profiles(); empty disables the M7
    # verdict (reported as no_verdict rather than a wrong-organism FAIL).
    chassis_profile: Optional["ReferenceProfile"] = field(default_factory=lambda: ECOLI_PROFILE)
    taxon_profiles: dict[str, "ReferenceProfile"] = field(default_factory=dict)
    # kmer_novelty: nucleotide novelty / anti-memorization gate (audit C6/M13). A
    # generated sequence whose max k-mer containment to the nearest reference BGC
    # is >= fail is treated as memorized (FAIL); >= warn is WARN. Calibrate these
    # from the (de-leaked) positive-control distribution — see memorization_check.py.
    novelty_fail_threshold: float = 0.95
    novelty_warn_threshold: float = 0.80


def phylum_token(taxonomic_tag: str) -> Optional[str]:
    """Extract the phylum token (e.g. 'P__ACTINOMYCETOTA') from a taxonomic tag."""
    for part in (taxonomic_tag or "").replace("|", ";").split(";"):
        part = part.strip()
        # GTDB tags are lowercase ("p__Actinomycetota") but this matched "P__", so
        # conditioning_faithful was null on EVERY row of every eval_track.jsonl.
        if part.upper().startswith("P__") and len(part) > 3:
            return part
    return None


def resolve_taxon_profile(
    expected_taxon: str,
    taxon_profiles: dict[str, ReferenceProfile],
) -> Optional[ReferenceProfile]:
    """Resolve the faithfulness reference for a sequence's conditioned taxon.

    Tries an exact key match first, then falls back to the phylum token, so a
    caller may pass a full taxonomic_tag or a bare phylum key.
    """
    if not expected_taxon or not taxon_profiles:
        return None
    # CASE-INSENSITIVE. build_taxon_profiles.py writes UPPERCASE keys ("P__ACTINOMYCETOTA") while
    # GTDB tags are lowercase ("p__Actinomycetota"), so every exact lookup missed and
    # conditioning_faithful was null on EVERY row of every eval_track.jsonl.
    upper = {k.upper(): v for k, v in taxon_profiles.items()}
    if expected_taxon in taxon_profiles:
        return taxon_profiles[expected_taxon]
    if expected_taxon.upper() in upper:
        return upper[expected_taxon.upper()]
    phy = phylum_token(expected_taxon)
    if phy and phy in taxon_profiles:
        return taxon_profiles[phy]
    if phy and phy.upper() in upper:
        return upper[phy.upper()]
    return None


def load_taxon_profiles(path: Path) -> dict[str, ReferenceProfile]:
    """Load taxon ReferenceProfiles from a JSON file written by build_taxon_profiles.py."""
    data = json.loads(Path(path).read_text())
    profiles: dict[str, ReferenceProfile] = {}
    for key, p in data.items():
        profiles[key] = ReferenceProfile(
            name=p.get("name", key),
            target_gc=p["target_gc"],
            codon_freq=p["codon_freq"],
            dinuc_ratios=p["dinuc_ratios"],
            gc_tol=p.get("gc_tol", 0.10),
            cai_threshold=p.get("cai_threshold", 0.7),
            dinuc_threshold=p.get("dinuc_threshold", 0.15),
        )
    return profiles


def _dinuc_entropy(seq: str) -> float:
    """Shannon entropy (bits) of the dinucleotide distribution. Real coding DNA is
    near-maximal (~3.7-4.0); degenerate output (e.g. the greedy all-GC collapse
    'GCGCGC...' or a homopolymer) collapses toward ~0-1 bit. Robust, length-stable
    junk detector — unlike gene-completeness, it does NOT penalise a legitimate core
    that is one big edge-truncated megasynthase gene."""
    s = (seq or "").upper()
    if len(s) < 2:
        return 0.0
    counts: Counter = Counter(s[i:i + 2] for i in range(len(s) - 1))
    total = sum(counts.values())
    ent = 0.0
    for v in counts.values():
        p = v / total
        ent -= p * math.log2(p)
    return ent


def check_coding_sanity(sequence: str, min_aa: int = 50) -> dict[str, Any]:
    """is_bgc FLOOR: is this real, gene-rich coding DNA (vs the degenerate collapse)?
    From the Prodigal gene calls: coding density (union gene coverage), gene count,
    sizes, complete-gene fraction; plus a dinucleotide-entropy COMPLEXITY guard. This
    is a sanity floor — the authoritative "is it a BGC" verdict is antiSMASH detection
    (see derive_questions); coding_sanity is the proxy floor when antiSMASH is skipped
    (quick-eval / diagnostics) and catches the greedy all-GC / homopolymer collapse.

    NOTE: it deliberately does NOT require a COMPLETE gene — a legitimate strict core
    is often one big edge-truncated megasynthase (Prodigal flags it partial); the
    complexity guard, not gene-completeness, is what rejects degenerate junk."""
    res: dict[str, Any] = {"check": "coding_sanity"}
    L = len(sequence or "")
    if L == 0:
        res.update({"pass": False, "coding_density": 0.0, "n_orfs": 0, "reason": "empty"})
        return res
    orfs = find_orfs(sequence, min_aa)
    covered, ce = 0, -1                       # union of gene spans (avoid double-counting overlaps)
    for s, e in sorted((o.start, o.end) for o in orfs):
        if s > ce:
            covered += e - s; ce = e
        elif e > ce:
            covered += e - ce; ce = e
    aa = [len(o.aa_seq) for o in orfs]
    complete = sum(1 for o in orfs if not o.partial)   # full genes (not edge-truncated)
    coding_density = round(covered / L, 4)
    max_aa = max(aa) if aa else 0
    dinuc_entropy = round(_dinuc_entropy(sequence), 3)
    res.update({
        "coding_density": coding_density, "n_orfs": len(orfs),
        "mean_orf_aa": round(sum(aa) / len(aa), 1) if aa else 0, "max_orf_aa": max_aa,
        # complete_gene_fraction is only meaningful under Prodigal, which flags edge-truncated
        # genes. The six-frame fallback never sets ORF.partial, so this would be exactly 1.0 for
        # every sequence -- a fabricated "every gene is complete". Report None instead, and stamp
        # the caller so a reader can never mistake one instrument's numbers for the other's.
        "complete_gene_fraction": (round(complete / len(orfs), 3) if orfs else 0.0)
                                  if _GENE_FINDER is not None else None,
        "gene_caller": "prodigal" if _GENE_FINDER is not None else "sixframe_DEGRADED",
        "dinuc_entropy": dinuc_entropy,
        # gene-rich + a substantial gene + non-degenerate sequence complexity.
        "pass": coding_density >= 0.5 and len(orfs) >= 1 and max_aa >= 100 and dinuc_entropy >= 2.5,
    })
    return res


def _count_ordered_modules(types: list[str], pattern: list[str]) -> int:
    """Greedy count of non-overlapping in-order occurrences of `pattern` within the
    position-ordered `types` stream (other domains already filtered out)."""
    count, pi = 0, 0
    for t in types:
        if t == pattern[pi]:
            pi += 1
            if pi == len(pattern):
                count += 1; pi = 0
    return count


def check_module_architecture(m2: Optional[dict[str, Any]], expected_class: str) -> dict[str, Any]:
    """DIAGNOSTIC (M11): from M2's positioned domain hits, order the obligate domains
    along the sequence and count assembly-line MODULES + whether they are IN ORDER
    (collinearity). Upgrades M2's binary presence check. Only applicable to module
    classes (NRPS, PKS, PKS_NRPS_HYBRID); else no_verdict."""
    res: dict[str, Any] = {"check": "module_architecture"}
    if expected_class == "PKS_NRPS_HYBRID":
        patterns = [("NRPS", MODULE_PATTERNS["NRPS"]), ("PKS", MODULE_PATTERNS["PKS"])]
    elif expected_class in MODULE_PATTERNS:
        patterns = [(expected_class, MODULE_PATTERNS[expected_class])]
    else:
        patterns = []
    if not m2 or m2.get("skipped") or not patterns:
        res.update({"applicable": False, "pass": None, "module_count": 0, "ordered_module_count": 0})
        return res
    orfs_meta = {o["i"]: o for o in (m2.get("orfs_meta") or [])}
    relevant = {p for _, pat in patterns for p in pat}
    stream = []
    for d in (m2.get("domains_found") or []):
        acc = d.get("pfam_accession")
        if acc not in relevant:
            continue
        try:
            oi = int(str(d.get("target", "")).split("_")[1])
        except (IndexError, ValueError):
            continue
        om = orfs_meta.get(oi)
        if not om:
            continue
        gpos = om["start"] + int(d.get("env_from", 0)) * 3   # approx nt position of the domain
        stream.append((gpos, acc))
    types = [acc for _, acc in sorted(stream)]
    counts = Counter(types)
    per, total_mod, total_ord = {}, 0, 0
    for pname, pat in patterns:
        module_count = min(counts.get(p, 0) for p in pat)    # modules' worth of each domain
        ordered = _count_ordered_modules(types, pat)
        per[pname] = {"module_count": module_count, "ordered_module_count": ordered}
        total_mod += module_count; total_ord += ordered
    res.update({
        "applicable": True, "patterns": [p for p, _ in patterns], "per_pattern": per,
        "module_count": total_mod, "ordered_module_count": total_ord,
        "in_order_fraction": round(total_ord / total_mod, 3) if total_mod else None,
        "pass": total_ord >= 1,    # diagnostic: at least one complete, correctly-ordered module
    })
    return res


def _verdict_from_pass(r: Optional[dict[str, Any]]) -> str:
    """Map a check result's ``pass`` field to PASS/FAIL/no_verdict/skipped."""
    if not r or r.get("skipped"):
        return "skipped"
    p = r.get("pass")
    if p is None:
        return "no_verdict"
    return "PASS" if p else "FAIL"


# Union of every class's obligate markers: the Pfam accessions that actually indicate
# biosynthetic machinery, as opposed to any protein domain whatsoever.
_BIOSYNTHETIC_PFAMS = {a for _v in OBLIGATE_DOMAINS.values() for a in (_v or [])}


def derive_questions(results: dict[str, Any]) -> dict[str, str]:
    """Combine CHECK results into question-level verdicts (PASS/FAIL/no_verdict/
    skipped). antiSMASH owns is_bgc + correct_class; class_markers is the fast
    PROXY used when antiSMASH was skipped (e.g. quick-eval). See QUESTIONS."""
    def g(name: str) -> dict[str, Any]:
        return results.get(name) or {}
    cs, asr, cm = g("coding_sanity"), g("antismash"), g("class_markers")
    as_ran = bool(asr) and not asr.get("skipped")
    cm_ran = bool(cm) and not cm.get("skipped")
    q: dict[str, str] = {}

    # is_bgc — antiSMASH detection is AUTHORITATIVE when it ran (the gold-standard BGC
    # detector; a real cluster shouldn't fail on a coding-sanity technicality such as
    # an edge-truncated megasynthase). Without antiSMASH, fall back to the coding_sanity
    # floor + a domain-count proxy (quick-eval / diagnostics).
    # PROVENANCE IS PART OF THE VERDICT. Measured on 768 paired records where BOTH antiSMASH and
    # the Pfam proxy ran, the proxy's agreement with the gold standard on correct_class is:
    #     precision 0.366   recall 0.972
    #     proxy PASS 191/768 (0.249)  vs  antiSMASH PASS 72/768 (0.094)
    # i.e. the proxy INFLATES correct_class by ~2.6x. Substituting it silently means a number
    # produced with antiSMASH skipped is not comparable to one produced with it, and nothing in
    # the output said which you were looking at. Record the source alongside every gate verdict.
    prov: dict[str, str] = {}
    sane = cs.get("pass")
    if as_ran and "detected" in asr:
        q["is_bgc"] = "PASS" if asr["detected"] else "FAIL"
        prov["is_bgc"] = "antismash"
    elif cm_ran:
        # The proxy must count BIOSYNTHETIC domains, not ANY Pfam hit -- the comment said the
        # former, the code did the latter. The false positives were generic housekeeping families
        # (AMP-binding, DAO, Amino_oxidase, adh_short, NAD_binding_8...). Measured on the 768
        # records where antiSMASH also ran:
        #     any Pfam >= 1  (old):  sens 1.000  spec 0.598  PPV 0.330
        #     biosynth >= 2  (new):  sens 0.882  spec 0.878  PPV 0.589
        # Free -- the scan already ran; we just stop counting irrelevant hits.
        _bio = set(cm.get("unique_domain_accessions") or []) & _BIOSYNTHETIC_PFAMS
        detected = len(_bio) >= 2
        if cs and not cs.get("skipped"):
            q["is_bgc"] = "PASS" if (detected and sane) else "FAIL"
        else:
            q["is_bgc"] = "PASS" if detected else "FAIL"
        prov["is_bgc"] = "class_markers_proxy"
    elif cs and not cs.get("skipped"):
        q["is_bgc"] = "PASS" if sane else "FAIL"           # weak: coding floor only
        prov["is_bgc"] = "coding_floor_only"
    else:
        q["is_bgc"] = "skipped"
        prov["is_bgc"] = "none"

    # correct_class = antiSMASH class match (class_markers proxy when AS skipped).
    if as_ran and asr.get("class_match") is not None:
        cls = asr["class_match"]
        prov["correct_class"] = "antismash"
    elif cm_ran:
        cls = cm.get("pass")
        prov["correct_class"] = "class_markers_proxy"   # precision 0.366 — inflates ~2.6x
    else:
        cls = None
        prov["correct_class"] = "none"
    q["correct_class"] = ("skipped" if (not as_ran and not cm_ran)
                          else "no_verdict" if cls is None
                          else "PASS" if cls else "FAIL")

    # the remaining questions are one check → one verdict.
    q["novel"] = _verdict_from_pass(g("kmer_novelty"))
    q["proteins_plausible"] = _verdict_from_pass(g("protein_homology"))
    q["complete"] = _verdict_from_pass(g("module_architecture"))
    q["class_probe_agrees"] = _verdict_from_pass(g("class_probe"))
    if q["class_probe_agrees"] != "skipped":
        prov["class_probe_agrees"] = "class_probe_DIAGNOSTIC_never_a_gate"
    # Not a question — leading underscore keeps it out of any QUESTIONS-keyed tally.
    q["_verdict_source"] = prov
    return q


def evaluate_bgc(
    sequence: str,
    accession: str = "query",
    expected_class: str = "",
    config: Optional[EvalConfig] = None,
    expected_taxon: str = "",
    novelty: Optional[dict[str, Any]] = None,
    probe_scores: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Run the eval CHECKS on a BGC sequence and derive the QUESTION verdicts.

    ``expected_taxon`` is the taxon the sequence was conditioned on (a full
    taxonomic_tag or a phylum token); it selects the taxon_faithfulness reference
    from ``config.taxon_profiles``. ``novelty`` is the precomputed memorization-scan
    result for THIS sequence (from memorization_check.scan_corpus), driving the
    kmer_novelty check.

    Returns a dict with one key per CHECK run, a per-check ``checks`` verdict map,
    and a ``questions`` map (also aliased as ``summary``) — see CHECKS / QUESTIONS.
    """
    if config is None:
        config = EvalConfig()
    skip = set(config.skip_checks)

    results: dict[str, Any] = {
        "accession": accession,
        "expected_class": expected_class,
        "expected_taxon": expected_taxon,
        "sequence_length": len(sequence),
    }

    # is_bgc + correct_class: antiSMASH (gold standard) + class_markers (Pfam proxy).
    if "antismash" not in skip:
        results["antismash"] = check_antismash(
            sequence, accession, expected_class, config.class_map,
            config.antismash_timeout, config.antismash_db_dir,
        )
    if "class_markers" not in skip:
        results["class_markers"] = check_class_markers(
            sequence, expected_class, config.pfam_hmm_path,
        )
    # is_bgc floor: gene-rich, complete coding DNA (catches degenerate output).
    if "coding_sanity" not in skip:
        results["coding_sanity"] = check_coding_sanity(sequence)
    # proteins_plausible: homology to known enzymes.
    if "protein_homology" not in skip:
        results["protein_homology"] = check_protein_homology(
            sequence, accession, config.mmseqs2_db,
        )
    # novel: anti-memorization k-mer novelty (a gate — memorized => FAIL).
    if "kmer_novelty" not in skip:
        results["kmer_novelty"] = check_kmer_novelty(
            novelty, config.novelty_fail_threshold, config.novelty_warn_threshold,
        )
    # class_probe_agrees: CONTINUOUS class readout. Opt-in and diagnostic-only -- it runs when
    # the caller supplies scores from a model-specific probe, and it never gates (see
    # check_class_probe for why: no negative class, reads the model not biology, leakage debt).
    if "class_probe" not in skip and probe_scores is not None:
        results["class_probe"] = check_class_probe(probe_scores, expected_class)
    # complete: module architecture, derived from class_markers domain positions.
    if "module_architecture" not in skip:
        results["module_architecture"] = check_module_architecture(
            results.get("class_markers"), expected_class,
        )
    # conditioning_faithful: codon/GC vs the conditioned taxon (E. coli sub-score
    # is informational only and does not gate).
    if "taxon_faithfulness" not in skip:
        results["taxon_faithfulness"] = check_taxon_faithfulness(
            sequence,
            taxon_profile=resolve_taxon_profile(expected_taxon, config.taxon_profiles),
            chassis_profile=config.chassis_profile,
        )
    # OPTIONAL: ESMFold foldability — opt-in only (GPU-expensive).
    if config.run_protein_foldability and "protein_foldability" not in skip:
        results["protein_foldability"] = check_protein_foldability(sequence)

    # per-check verdicts + derived per-question verdicts.
    checks_summary: dict[str, str] = {}
    for name in CHECKS + OPTIONAL_CHECKS:
        if name in skip:
            checks_summary[name] = "skipped"
        elif name in results:
            checks_summary[name] = _verdict_from_pass(results[name])
    results["checks"] = checks_summary
    results["questions"] = derive_questions(results)
    results["summary"] = results["questions"]   # alias: the headline/accept source
    return results
