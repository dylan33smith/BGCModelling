"""Eight-metric evaluation suite for generated BGC sequences.

Implements the evaluation framework from Section 5 of the research plan:

Tier 1 (Primary):
  1. AntiSMASH BGC class identification
  2. Functional domain recovery (pyhmmer + Pfam 37.0)
  3. Protein foldability + structural homology (ESMFold + Foldseek)
  4. Synthesis feasibility (DNA Chisel)

Tier 2 (Secondary):
  5. Sequence naturalness (Evo2 perplexity)
  6. Structural novelty + coherence (BiG-SCAPE 2.0)
  7. Organism compatibility (CAI + GC + dinucleotide stats)

Tier 3 (Descriptive):
  8. Protein sequence homology (MMseqs2 vs UniRef50)

Each metric function accepts a sequence (and metadata) and returns a dict of results.
External tools that are not installed are gracefully skipped.
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
# Reference organism profiles for Metric 7 (organism compatibility)
# ---------------------------------------------------------------------------
#
# Metric 7 answers two DIFFERENT questions that must not be conflated:
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
    BGCs of a given taxon so Metric 7 can grade faithfulness to the organism the
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


# Obligate Pfam domains per COMPOUND_CLASS (accession-based for pyhmmer)
# These are the minimum domains expected in a valid BGC of each class.
OBLIGATE_DOMAINS: dict[str, list[str]] = {
    "PKS": ["PF00109", "PF00698", "PF00550"],       # KS (ketosynthase), AT (acyltransferase), ACP (acyl carrier)
    "NRPS": ["PF00668", "PF00501", "PF00550"],      # C (condensation), A (adenylation/AMP-binding), T/PCP (carrier)
    "TERPENE": ["PF03936", "PF19086", "PF01397"],     # Terpene_synth_C, Terpene_syn_C_2, or Terpene_synth (any one suffices)
    "RIPP": [],                                       # RiPPs are diverse; no universal obligate domain
    "SACCHARIDE": ["PF00534"],                        # Glycos_transf_1 (glycosyltransferase)
    "OTHER": [],                                      # Too diverse for obligate domain list
    "PKS_NRPS_HYBRID": ["PF00109", "PF00668"],       # KS + C domains
    "SIDEROPHORE": ["PF04183"],                       # IucA_IucC
    "ALKALOID": [],
    "BETALACTONE": [],
}


@dataclass
class ORF:
    """A predicted open reading frame."""
    start: int
    end: int
    strand: int  # +1 or -1
    nt_seq: str
    aa_seq: str


def find_orfs(sequence: str, min_aa: int = 50) -> list[ORF]:
    """Simple six-frame ORF finder. Returns ORFs >= min_aa amino acids."""
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
# Metric 1: AntiSMASH BGC Class Identification
# ---------------------------------------------------------------------------

def metric_1_antismash(
    sequence: str,
    accession: str = "query",
    expected_class: str = "",
    class_map: Optional[dict[str, str]] = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run antiSMASH on a sequence and check predicted BGC class."""
    result: dict[str, Any] = {"metric": 1, "name": "antismash_bgc_class", "tier": 1}

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
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False,
            )
        except FileNotFoundError:
            result["skipped"] = True
            result["reason"] = "antismash not installed"
            return result
        except subprocess.TimeoutExpired:
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
                regions = []
                for record in as_data.get("records", []):
                    for region in record.get("areas", record.get("regions", [])):
                        products = region.get("products", region.get("product", []))
                        if isinstance(products, str):
                            products = [products]
                        regions.append({"products": products})
                result["regions"] = regions
                all_products = []
                for r in regions:
                    all_products.extend(r["products"])
                result["predicted_products"] = all_products

                # Check class match
                if expected_class and class_map:
                    mapped = [class_map.get(p.lower(), "OTHER") for p in all_products]
                    result["mapped_classes"] = mapped
                    result["pass"] = expected_class in mapped
                elif expected_class:
                    result["pass"] = expected_class.lower() in [
                        p.lower() for p in all_products
                    ]
            except (json.JSONDecodeError, KeyError) as e:
                result["parse_error"] = str(e)
        else:
            result["no_results_json"] = True
            result["pass"] = False

    # Default: if no pass verdict was set but we ran, set to False
    if "pass" not in result and not result.get("skipped"):
        result["pass"] = False

    return result


# ---------------------------------------------------------------------------
# Metric 2: Functional Domain Recovery (pyhmmer + Pfam)
# ---------------------------------------------------------------------------

def metric_2_domain_recovery(
    sequence: str,
    expected_class: str = "",
    pfam_hmm_path: Optional[Path] = None,
    evalue_threshold: float = 1e-10,
) -> dict[str, Any]:
    """Scan predicted ORFs against Pfam using pyhmmer."""
    result: dict[str, Any] = {"metric": 2, "name": "domain_recovery", "tier": 1}

    try:
        import pyhmmer
        from pyhmmer.easel import TextSequence, Alphabet
        from pyhmmer.plan7 import HMMFile
    except ImportError:
        result["skipped"] = True
        result["reason"] = "pyhmmer not installed"
        return result

    if pfam_hmm_path is None or not pfam_hmm_path.exists():
        result["skipped"] = True
        result["reason"] = f"Pfam HMM not found at {pfam_hmm_path}"
        return result

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
                            })
                            domain_accessions.add(acc_base)

    result["domains_found"] = domains_found
    result["unique_domain_accessions"] = sorted(domain_accessions)
    result["domain_count"] = len(domains_found)

    # Check obligate domains
    # For most classes: ALL obligate domains must be present.
    # For TERPENE: ANY ONE of the listed domains suffices (they're alternative families).
    _ANY_ONE_CLASSES = {"TERPENE"}

    if expected_class:
        obligate = OBLIGATE_DOMAINS.get(expected_class, [])
        if obligate:
            missing = [d for d in obligate if d not in domain_accessions]
            result["obligate_domains"] = obligate
            result["missing_obligate"] = missing
            if expected_class in _ANY_ONE_CLASSES:
                result["pass"] = any(d in domain_accessions for d in obligate)
            else:
                result["pass"] = len(missing) == 0
        else:
            result["obligate_domains"] = []
            result["pass"] = None  # No obligate domains defined for this class
            result["note"] = f"No obligate domain set defined for class {expected_class}"
    else:
        result["pass"] = None

    return result


# ---------------------------------------------------------------------------
# Metric 3: Protein Foldability (ESMFold + Foldseek)
# ---------------------------------------------------------------------------

def metric_3_esmfold(
    sequence: str,
    max_orfs: int = 5,
    foldseek_db: Optional[str] = None,
) -> dict[str, Any]:
    """Predict protein structures with ESMFold and assess foldability.

    Uses the HuggingFace transformers EsmForProteinFolding implementation
    (transformers==4.46.3), which does not require OpenFold as a dependency.
    Requires: pip install transformers==4.46.3 accelerate torch==2.5.1+cu124
    """
    result: dict[str, Any] = {"metric": 3, "name": "protein_foldability", "tier": 1}

    try:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError:
        result["skipped"] = True
        result["reason"] = "transformers not installed (pip install transformers==4.46.3 accelerate)"
        return result

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
        result["skipped"] = True
        result["reason"] = f"ESMFold model load failed: {e}"
        return result

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
    result["pass"] = len(passing) > len(orf_results) // 2

    return result


# ---------------------------------------------------------------------------
# Metric 4: Synthesis Feasibility (DNA Chisel)
# ---------------------------------------------------------------------------

def metric_4_synthesis_feasibility(
    sequence: str,
    window_size: int = 50,
) -> dict[str, Any]:
    """Check synthesis constraints per Twist Bioscience guidelines."""
    result: dict[str, Any] = {"metric": 4, "name": "synthesis_feasibility", "tier": 1}
    seq = sequence.upper()

    if not re.fullmatch(r"[ACGT]+", seq):
        result["pass"] = False
        result["reason"] = "non-ACGT characters present"
        return result

    checks: dict[str, Any] = {}

    # Global GC content: 25-65%
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    checks["global_gc"] = {"value": round(gc, 4), "pass": 0.25 <= gc <= 0.65}

    # Local GC in 50 bp windows: 35-65%  (warn but don't hard-fail)
    violations_local = 0
    for i in range(0, len(seq) - window_size + 1, window_size):
        w = seq[i : i + window_size]
        wgc = (w.count("G") + w.count("C")) / len(w)
        if not (0.35 <= wgc <= 0.65):
            violations_local += 1
    total_windows = max(1, (len(seq) - window_size + 1) // window_size)
    checks["local_gc_violations"] = {
        "count": violations_local,
        "total_windows": total_windows,
        "fraction": round(violations_local / total_windows, 4),
    }

    # Homopolymer runs >= 10 bp
    max_homo = 0
    if seq:
        cur = 1
        for j in range(1, len(seq)):
            if seq[j] == seq[j - 1]:
                cur += 1
                max_homo = max(max_homo, cur)
            else:
                cur = 1
        max_homo = max(max_homo, cur)
    checks["max_homopolymer"] = {"value": max_homo, "pass": max_homo < 10}

    # Direct repeats > 20 bp (sample check — full check is O(n^2))
    has_long_repeat = False
    if len(seq) <= 200_000:  # Only check shorter sequences
        for k in range(21, 31):
            kmers: set[str] = set()
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                if kmer in kmers:
                    has_long_repeat = True
                    break
                kmers.add(kmer)
            if has_long_repeat:
                break
    checks["direct_repeat_gt20"] = {"detected": has_long_repeat}

    # DNA Chisel full check (if installed)
    try:
        from dnachisel import DnaOptimizationProblem
        from dnachisel.builtin_specifications import (
            EnforceGCContent,
            AvoidPattern,
        )

        problem = DnaOptimizationProblem(
            sequence=seq[:50000] if len(seq) > 50000 else seq,  # Limit for speed
            constraints=[
                EnforceGCContent(mini=0.25, maxi=0.65),
                AvoidPattern("10xN"),  # No homopolymer >= 10
            ],
        )
        checks["dnachisel_all_pass"] = problem.all_constraints_pass()
        checks["dnachisel_summary"] = problem.constraints_text_summary()
    except ImportError:
        checks["dnachisel"] = "not installed"

    all_pass = (
        checks["global_gc"]["pass"]
        and checks["max_homopolymer"]["pass"]
    )
    result["checks"] = checks
    result["pass"] = all_pass

    return result


# ---------------------------------------------------------------------------
# Metric 5: Sequence Naturalness (Evo2 Perplexity)
# ---------------------------------------------------------------------------

def metric_5_evo2_perplexity(
    sequence: str,
    model_name: str = "evo2_7b_262k",
) -> dict[str, Any]:
    """Compute per-nucleotide perplexity under the pretrained Evo2 base model.

    Uses the evo2 PyPI package (evo2==0.5.5) with the arcinstitute/evo2_7b_262k
    checkpoint. score_sequences() returns mean log-likelihood (higher = more
    probable); we negate it to get mean NLL and exponentiate for perplexity.

    Requires: pip install evo2==0.5.5 flash-attn==2.7.4.post1 torch==2.5.1+cu124
    GPU strongly recommended (loads ~14 GB of weights).
    """
    result: dict[str, Any] = {"metric": 5, "name": "evo2_perplexity", "tier": 2}

    try:
        from evo2 import Evo2
    except ImportError:
        result["skipped"] = True
        result["reason"] = "evo2 not installed (pip install evo2==0.5.5)"
        return result

    try:
        model = Evo2(model_name)
        # score_sequences returns mean log-likelihood per sequence (higher = more probable)
        scores = model.score_sequences([sequence], batch_size=1)
        mean_log_likelihood = float(scores[0])
        mean_nll = -mean_log_likelihood
        perplexity = math.exp(mean_nll) if mean_nll < 700 else float("inf")
        result["mean_log_likelihood"] = round(mean_log_likelihood, 4)
        result["perplexity"] = round(perplexity, 4)
        result["nucleotides_scored"] = len(sequence)
        # Reference: MIBiG BGCs typically score between -0.3 and -0.8 log-likelihood
        # Shuffled sequences score much lower (more negative, higher perplexity)
        result["pass"] = mean_log_likelihood > -2.0
    except Exception as e:
        result["skipped"] = True
        result["reason"] = f"Evo2 inference failed: {e}"

    return result


# ---------------------------------------------------------------------------
# Metric 6: Structural Novelty + Coherence (BiG-SCAPE 2.0)
# ---------------------------------------------------------------------------

def metric_6_bigscape(
    sequence: str,
    accession: str = "query",
    mibig_gbk_dir: Optional[Path] = None,
    timeout: int = 1200,
) -> dict[str, Any]:
    """Run BiG-SCAPE to assess cluster coherence vs MIBiG.

    AUDIT C6: this is a STUB — BiG-SCAPE distance parsing is not implemented, so it
    never produces a verdict and must NOT be relied on for novelty. Nucleotide
    novelty / anti-memorization is gated by ``metric_9_novelty`` instead. This
    function is kept only to run BiG-SCAPE for future coherence analysis.
    """
    result: dict[str, Any] = {"metric": 6, "name": "bigscape_coherence",
                              "tier": 2, "stub_not_implemented": True, "pass": None}

    # Check bigscape is installed
    try:
        subprocess.run(["bigscape", "--version"], capture_output=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result["skipped"] = True
        result["reason"] = "bigscape not installed"
        return result

    if mibig_gbk_dir is None or not mibig_gbk_dir.is_dir():
        result["skipped"] = True
        result["reason"] = "MIBiG GBK directory not provided"
        return result

    with tempfile.TemporaryDirectory() as tmp:
        # Write query as a minimal GenBank file
        query_gbk = Path(tmp) / "query" / f"{accession}.gbk"
        query_gbk.parent.mkdir()
        rec = SeqRecord(Seq(sequence), id=accession, description="generated BGC")
        with query_gbk.open("w") as f:
            SeqIO.write(rec, f, "genbank")

        outdir = Path(tmp) / "bigscape_out"
        cmd = [
            "bigscape", "cluster",
            "--input-dir", str(query_gbk.parent),
            "--mibig-dir", str(mibig_gbk_dir),
            "--output-dir", str(outdir),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False,
            )
            result["returncode"] = proc.returncode
            if proc.returncode != 0:
                result["stderr_tail"] = proc.stderr[-1000:] if proc.stderr else ""
        except subprocess.TimeoutExpired:
            result["skipped"] = True
            result["reason"] = f"bigscape timeout after {timeout}s"
            return result

        # Parse distance matrix if available
        # BiG-SCAPE output format varies by version; this is a STUB (pass stays None).
        result["note"] = ("STUB: BiG-SCAPE ran but distance parsing is not implemented; "
                          "this metric does NOT assess novelty — see metric_9_novelty.")

    return result


def metric_9_novelty(
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
    result: dict[str, Any] = {"metric": 9, "name": "nucleotide_novelty", "tier": 1}
    if not novelty:
        result["skipped"] = True
        result["reason"] = ("no novelty scan supplied; run memorization_check.scan_corpus "
                            "over the full reference corpus and pass the per-sequence result")
        return result
    cont = float(novelty.get("max_containment", 0.0))
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
# Metric 7: Organism Compatibility (CAI + GC + dinucleotide stats)
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


def metric_7_organism_compatibility(
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
      step). See AUDIT_FINDINGS.md C4.
    """
    result: dict[str, Any] = {"metric": 7, "name": "organism_compatibility", "tier": 2}
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
# Metric 8: Protein Sequence Homology (MMseqs2)
# ---------------------------------------------------------------------------

def metric_8_mmseqs2(
    sequence: str,
    accession: str = "query",
    db_path: Optional[str] = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Search predicted ORFs against a protein database using MMseqs2."""
    result: dict[str, Any] = {"metric": 8, "name": "mmseqs2_homology", "tier": 3}

    try:
        subprocess.run(["mmseqs", "version"], capture_output=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result["skipped"] = True
        result["reason"] = "mmseqs2 not installed"
        return result

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
            result["skipped"] = True
            result["reason"] = "no database path provided (need UniRef50 MMseqs2 DB)"
            return result

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
            result["skipped"] = True
            result["reason"] = f"timeout after {timeout}s"
            return result

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
                result["memorisation_flag"] = max(pidents) > 95.0
        else:
            result["hits"] = []

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
    class_map: Optional[dict[str, str]] = None
    antismash_timeout: int = 600
    skip_metrics: list[int] = field(default_factory=list)
    # Metric 7: organism-compatibility references (see AUDIT_FINDINGS.md C4).
    # `chassis_profile` is the wet-lab host (E. coli) used for the informational
    # expressibility sub-score. `taxon_profiles` maps a taxon key (e.g. a phylum
    # token like "P__ACTINOMYCETOTA") to the empirical profile used to grade
    # faithfulness. Populate via load_taxon_profiles(); empty disables the M7
    # verdict (reported as no_verdict rather than a wrong-organism FAIL).
    chassis_profile: Optional["ReferenceProfile"] = field(default_factory=lambda: ECOLI_PROFILE)
    taxon_profiles: dict[str, "ReferenceProfile"] = field(default_factory=dict)
    # Metric 9: nucleotide novelty / anti-memorization gate (audit C6/M13). A
    # generated sequence whose max k-mer containment to the nearest reference BGC
    # is >= fail is treated as memorized (FAIL); >= warn is WARN. Calibrate these
    # from the (de-leaked) positive-control distribution — see memorization_check.py.
    novelty_fail_threshold: float = 0.95
    novelty_warn_threshold: float = 0.80


def phylum_token(taxonomic_tag: str) -> Optional[str]:
    """Extract the phylum token (e.g. 'P__ACTINOMYCETOTA') from a taxonomic tag."""
    for part in (taxonomic_tag or "").replace("|", ";").split(";"):
        part = part.strip()
        if part.startswith("P__") and len(part) > 3:
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
    if expected_taxon in taxon_profiles:
        return taxon_profiles[expected_taxon]
    phy = phylum_token(expected_taxon)
    if phy and phy in taxon_profiles:
        return taxon_profiles[phy]
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


def evaluate_bgc(
    sequence: str,
    accession: str = "query",
    expected_class: str = "",
    config: Optional[EvalConfig] = None,
    expected_taxon: str = "",
    novelty: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the evaluation metrics on a BGC sequence.

    ``expected_taxon`` is the taxon the sequence was conditioned on (a full
    taxonomic_tag or a phylum token); it selects the Metric 7 faithfulness
    reference from ``config.taxon_profiles``.

    ``novelty`` is the precomputed memorization-scan result for THIS sequence
    (from memorization_check.scan_corpus, computed in batch by the driver); it
    drives Metric 9, the nucleotide novelty / anti-memorization gate (audit C6).

    Returns a dict with top-level keys for each metric and a summary.
    """
    if config is None:
        config = EvalConfig()
    skip = set(config.skip_metrics)

    results: dict[str, Any] = {
        "accession": accession,
        "expected_class": expected_class,
        "expected_taxon": expected_taxon,
        "sequence_length": len(sequence),
    }

    # Tier 1
    if 1 not in skip:
        results["metric_1"] = metric_1_antismash(
            sequence, accession, expected_class, config.class_map, config.antismash_timeout,
        )
    if 2 not in skip:
        results["metric_2"] = metric_2_domain_recovery(
            sequence, expected_class, config.pfam_hmm_path,
        )
    if 3 not in skip:
        results["metric_3"] = metric_3_esmfold(sequence)
    if 4 not in skip:
        results["metric_4"] = metric_4_synthesis_feasibility(sequence)

    # Tier 2
    if 5 not in skip:
        results["metric_5"] = metric_5_evo2_perplexity(sequence)
    if 6 not in skip:
        results["metric_6"] = metric_6_bigscape(
            sequence, accession, config.mibig_gbk_dir,
        )
    if 7 not in skip:
        results["metric_7"] = metric_7_organism_compatibility(
            sequence,
            taxon_profile=resolve_taxon_profile(expected_taxon, config.taxon_profiles),
            chassis_profile=config.chassis_profile,
        )

    # Tier 3
    if 8 not in skip:
        results["metric_8"] = metric_8_mmseqs2(
            sequence, accession, config.mmseqs2_db,
        )

    # Novelty / anti-memorization gate (audit C6/M13) — a first-class, gating
    # metric in the scored summary so a memorized sequence FAILS the suite.
    if 9 not in skip:
        results["metric_9"] = metric_9_novelty(
            novelty, config.novelty_fail_threshold, config.novelty_warn_threshold,
        )

    # Summary
    summary: dict[str, Any] = {}
    for i in range(1, 10):
        key = f"metric_{i}"
        if i in skip:
            summary[key] = "skipped"
        elif key not in results:
            summary[key] = "skipped"
        else:
            m = results[key]
            if m.get("skipped"):
                summary[key] = "skipped"
            elif "pass" in m:
                if m["pass"] is None:
                    summary[key] = "no_verdict"
                else:
                    summary[key] = "PASS" if m["pass"] else "FAIL"
            else:
                summary[key] = "no_verdict"
    results["summary"] = summary

    return results
