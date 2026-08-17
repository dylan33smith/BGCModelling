#!/usr/bin/env python
"""Verify the documentation framework — migration safety + the ongoing docs contract.

WHY THIS EXISTS. Documents decay; tests do not. This repo has already lost a headline result to a
metric whose name was stable while the quantity behind it silently changed, and half-closed a
research track whose treatment never landed. `terms.md` and `data.md` only stop that if something
mechanically checks them against the code and the disk.

TWO JOBS.

  MIGRATION  — nothing is lost in the docs overhaul. Every pre-framework document is either still
               present or demonstrably archived, and every load-bearing claim survives somewhere.

  CONTRACT   — the ongoing invariants. Every metric the scoring code computes has a `terms.md`
               entry with a stated scoring config; every dataset path in `data.md` exists on disk
               with the record count it claims; no DEPRECATED dataset or DEMOTED metric is being
               used as a live endpoint; every run directory is registered.

RUN
    python tests/test_docs_contract.py            # standalone, exit 0/1
    python tests/test_docs_contract.py -v         # show every passing check
    pytest tests/test_docs_contract.py            # also works under pytest

PRE- AND POST-CUTOVER. The framework files are located automatically: `docs/framework/` while the
overhaul is staged, then their final locations after. Checks that need `/data2` SKIP (not fail)
when it is not mounted, so this stays runnable off-host.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data2/ds85/bgcmodel_data")
RUNS_ROOT = Path("/data2/ds85/bgcmodel_runs")

FRAMEWORK_FILES = ("CLAUDE.md", "plan.md", "memory.md", "terms.md", "data.md", "bugs.md")

# ── result collection ───────────────────────────────────────────────────────

_RESULTS: list[tuple[str, str, str, str]] = []   # (status, category, name, detail)
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


def _rec(status, category, name, detail=""):
    _RESULTS.append((status, category, name, detail))


def ok(cat, name, detail=""):
    _rec("PASS", cat, name, detail)


def fail(cat, name, detail):
    _rec("FAIL", cat, name, detail)


def warn(cat, name, detail):
    _rec("WARN", cat, name, detail)


def skip(cat, name, detail):
    _rec("SKIP", cat, name, detail)


# ── locating the framework files ────────────────────────────────────────────

def locate() -> dict[str, Path]:
    """Find each framework file, staged or cut-over. Returns {name: path} for those that exist."""
    candidates = {
        "CLAUDE.md": [REPO / "docs/framework/CLAUDE.md", REPO / "CLAUDE.md"],
        "plan.md":   [REPO / "docs/framework/plan.md", REPO / "docs/plan.md",
                      REPO / "docs/project_memory/plan.md"],
        "memory.md": [REPO / "docs/framework/memory.md", REPO / "docs/memory.md",
                      REPO / "docs/project_memory/memory.md"],
        "terms.md":  [REPO / "docs/framework/terms.md", REPO / "docs/terms.md",
                      REPO / "docs/project_memory/terms.md"],
        "data.md":   [REPO / "docs/framework/data.md", REPO / "docs/data.md",
                      REPO / "docs/project_memory/data.md"],
        "bugs.md":   [REPO / "docs/framework/bugs.md", REPO / "docs/bugs.md",
                      REPO / "docs/project_memory/bugs.md"],
    }
    found = {}
    for name, paths in candidates.items():
        for p in paths:
            if p.exists():
                found[name] = p
                break
    return found


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════════
# MIGRATION — nothing lost
# ═══════════════════════════════════════════════════════════════════════════

def check_all_files_present(F):
    cat = "migration"
    missing = [n for n in FRAMEWORK_FILES if n not in F]
    if missing:
        fail(cat, "all six framework files exist", f"missing: {', '.join(missing)}")
    else:
        ok(cat, "all six framework files exist", ", ".join(str(F[n].relative_to(REPO))
                                                          for n in FRAMEWORK_FILES))


def check_history_preserved(F):
    """The pre-framework docs must still be readable somewhere. Losing them is the one
    irreversible failure mode of this overhaul."""
    cat = "migration"
    legacy = {
        "progress.md": ["docs/project_memory/progress.md", "docs/archive/pre-framework/progress.md"],
        "decisions.md": ["docs/project_memory/decisions.md",
                         "docs/archive/pre-framework/decisions.md"],
        "bugs.md(legacy)": ["docs/project_memory/bugs.md", "docs/archive/pre-framework/bugs.md"],
    }
    lost = []
    for name, paths in legacy.items():
        if not any((REPO / p).exists() for p in paths):
            lost.append(name)
    if lost:
        fail(cat, "pre-framework history preserved",
             f"NOT FOUND at either original or archive path: {', '.join(lost)}")
    else:
        ok(cat, "pre-framework history preserved", f"{len(legacy)} legacy documents readable")


def check_bugs_content_carried(F):
    """bugs.md is carried forward verbatim, so its subject sections must all survive."""
    cat = "migration"
    old = next((p for p in (REPO / "docs/archive/pre-framework/bugs.md",
                            REPO / "docs/project_memory/bugs.md") if p.exists()), None)
    if "bugs.md" not in F or old is None:
        skip(cat, "bugs.md sections carried forward", "legacy bugs.md not present")
        return
    new_txt, old_txt = read(F["bugs.md"]), read(old)
    if F["bugs.md"].resolve() == old.resolve():
        skip(cat, "bugs.md sections carried forward", "same file (not yet migrated)")
        return
    old_secs = set(re.findall(r"^## (.+)$", old_txt, re.M))
    new_secs = set(re.findall(r"^## (.+)$", new_txt, re.M))
    dropped = old_secs - new_secs
    if dropped:
        fail(cat, "bugs.md sections carried forward",
             f"{len(dropped)} section(s) lost: {sorted(dropped)[:3]}")
    else:
        ok(cat, "bugs.md sections carried forward", f"{len(old_secs)} sections intact")


LOAD_BEARING = [
    # (short label, regex that must appear somewhere in the new framework docs)
    ("ladder AUROC 0.950",            r"0\.950"),
    ("max_orf_aa demoted",            r"max_orf_aa.{0,400}?DEMOT"),
    ("proxy inflates correct_class",  r"2\.6"),
    ("splits_combined leaked",        r"94\.6"),
    ("class tag worth ~0 nats",       r"0\.0006"),
    ("de novo P(detect) 0.012",       r"0\.012"),
    ("A0 rate 4/150",                 r"4/150"),
    ("real-core ceiling 0.440",       r"0\.440"),
    ("novelty FAIL threshold 0.95",   r"0\.95"),
    ("RIPP is the target",            r"RIPP"),
    ("1B needs Transformer Engine",   r"Transformer Engine"),
    ("weighted arm never landed",     r"never landed"),
    ("MiBIG held out",                r"MiBIG"),
    ("bs=1 ga=128 only shape",        r"grad-accum 128|ga=128|grad_accum 128"),
]


def check_load_bearing_claims(F):
    """Every fact the project would be damaged by losing must survive the rewrite."""
    cat = "migration"
    corpus = "\n".join(read(F[n]) for n in FRAMEWORK_FILES if n in F)
    missing = [label for label, pat in LOAD_BEARING
               if not re.search(pat, corpus, re.S | re.I)]
    if missing:
        fail(cat, "load-bearing claims survive", f"absent from all framework docs: {missing}")
    else:
        ok(cat, "load-bearing claims survive", f"{len(LOAD_BEARING)}/{len(LOAD_BEARING)} present")


def check_governor_size(F):
    cat = "migration"
    if "CLAUDE.md" not in F:
        return
    n = len(read(F["CLAUDE.md"]).splitlines())
    if n > 150:
        fail(cat, "CLAUDE.md under 150 lines", f"{n} lines — it is auto-loaded every session")
    else:
        ok(cat, "CLAUDE.md under 150 lines", f"{n} lines")


def check_governor_has_no_findings(F):
    """The governor is a contract. Numeric results in it are exactly the drift this overhaul
    is meant to end."""
    cat = "migration"
    if "CLAUDE.md" not in F:
        return
    txt = read(F["CLAUDE.md"])
    # Strip the provenance EXAMPLE line, which legitimately contains figures.
    txt = "\n".join(l for l in txt.splitlines() if "provenance:" not in l)
    hits = re.findall(r"\b(?:AUROC|p\s*=\s*0\.\d|\d+/\d+\s*=\s*0\.\d|0\.\d{3})\b", txt)
    if hits:
        fail(cat, "CLAUDE.md carries no findings",
             f"result-like figures found: {sorted(set(hits))[:5]} — move them to memory.md")
    else:
        ok(cat, "CLAUDE.md carries no findings", "contract only")


def check_internal_links(F):
    cat = "migration"
    broken = []
    for name, p in F.items():
        for target in re.findall(r"\]\(([^)#][^)]*)\)", read(p)):
            if target.startswith(("http", "mailto:", "#")):
                continue
            t = target.split("#")[0].strip()
            if not t:
                continue
            resolved = (p.parent / t).resolve()
            # Staged files reference post-cutover locations; also try repo-root relative.
            if not resolved.exists() and not (REPO / t).exists():
                broken.append(f"{name} → {t}")
    if broken:
        warn(cat, "internal links resolve",
             f"{len(broken)} unresolved (expected while staged): {broken[:4]}")
    else:
        ok(cat, "internal links resolve", "all targets exist")


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACT — terms.md
# ═══════════════════════════════════════════════════════════════════════════

TERM_FIELDS = ("Is:", "Computed by:", "Status:")

# Metrics the scoring code actually produces. Keyed by the canonical doc name; the value is the
# code-level key, which is deliberately allowed to differ (that difference is itself documented).
CANONICAL_METRICS = {
    "best_bio_bits": "bio",
    "best_any_bits": "any",
    "biosynthetic_fraction": "frac",
    "n_bio_domains": "n_bio_domains",
    "bio_span_frac": "bio_span_frac",
    "max_orf_aa": "max_orf_aa",
    "correct_class": "correct_class",
    "is_bgc": "is_bgc",
    "novel": "novel",
    "class_probe": "class_probe",
    "coding_density": "coding_density",
    "containment": "max_containment",
}


def parse_terms(txt: str) -> dict[str, str]:
    """{identifier: body} for each '### <name>' entry. Fenced blocks are stripped first so the
    entry-schema template is not itself parsed as an entry."""
    txt = re.sub(r"^```.*?^```", "", txt, flags=re.M | re.S)
    entries = {}
    parts = re.split(r"^### ", txt, flags=re.M)[1:]
    for part in parts:
        head = part.split("\n", 1)[0]
        body = part
        # identifier = first backticked token, else the leading words before a tag/arrow
        m = re.search(r"`([^`]+)`", head)
        name = m.group(1) if m else re.split(r"\s+[\[→]", head)[0].strip()
        entries[name.strip().strip("`")] = body
    return entries


def check_every_metric_defined(F):
    cat = "terms"
    if "terms.md" not in F:
        fail(cat, "terms.md present", "not found")
        return
    txt = read(F["terms.md"])
    entries = parse_terms(txt)
    keys = set(entries)
    missing = []
    for doc_name in CANONICAL_METRICS:
        if doc_name in keys:
            continue
        # allow the entry to be titled with the name anywhere in its heading
        if any(doc_name in k for k in keys):
            continue
        if re.search(rf"^### .*{re.escape(doc_name)}", txt, re.M):
            continue
        missing.append(doc_name)
    if missing:
        fail(cat, "every computed metric has an entry", f"undefined: {missing}")
    else:
        ok(cat, "every computed metric has an entry", f"{len(CANONICAL_METRICS)} metrics")


def check_term_schema(F):
    cat = "terms"
    if "terms.md" not in F:
        return
    entries = parse_terms(read(F["terms.md"]))
    bad = []
    for name, body in entries.items():
        if name.upper() == name and " " in name:      # section-ish heading, e.g. "THE LADDER"
            continue
        if "see **" in body or "→  see" in body:      # pure alias pointer
            continue
        # A RETIRED metric needs only its Status — its computation is gone by definition.
        if re.search(r"Status:\*{0,2}\s*\*{0,2}(RETIRED)", body):
            required = ("Status:",)
        else:
            required = TERM_FIELDS
        missing = [f for f in required if f"**{f.rstrip(':')}:**" not in body
                   and f not in body]
        if missing:
            bad.append(f"{name} (missing {missing})")
    if bad:
        fail(cat, "entries carry required fields", f"{len(bad)}: {bad[:4]}")
    else:
        ok(cat, "entries carry required fields", f"{len(entries)} entries")


def check_primary_metric_states_config(F):
    """The A0 inversion in one check: the primary endpoint must say what changes its meaning."""
    cat = "terms"
    if "terms.md" not in F:
        return
    entries = parse_terms(read(F["terms.md"]))
    body = entries.get("best_bio_bits", "")
    if not body:
        fail(cat, "best_bio_bits states its scoring config", "no entry")
        return
    needs = ["CHANGES MEANING WITH", "OBLIGATE_DOMAINS"]
    missing = [n for n in needs if n not in body]
    if missing:
        fail(cat, "best_bio_bits states its scoring config",
             f"missing {missing} — this is the field that would have caught the A0 inversion")
    else:
        ok(cat, "best_bio_bits states its scoring config", "global vs class-specific documented")


def check_retired_metrics_not_live(F):
    """A DEMOTED or RETIRED metric must not reappear as a live endpoint in plan.md."""
    cat = "terms"
    if "terms.md" not in F or "plan.md" not in F:
        return
    entries = parse_terms(read(F["terms.md"]))
    dead = [n for n, b in entries.items()
            if re.search(r"\*\*Status:\*\*.{0,80}(RETIRED|DEMOTED)", b, re.S)
            or re.search(r"Status:.{0,80}(RETIRED|DEMOTED)", b, re.S)]
    plan = read(F["plan.md"])
    # only the endpoint column / "Primary endpoint:" lines count as live use
    endpoint_lines = [l for l in plan.splitlines()
                      if "Primary endpoint" in l or "| Endpoint" in l
                      or re.match(r"^\|\s*P\d", l)]
    offenders = [d for d in dead if any(d in l for l in endpoint_lines)]
    if offenders:
        fail(cat, "no demoted/retired metric used as an endpoint", f"{offenders}")
    else:
        ok(cat, "no demoted/retired metric used as an endpoint",
           f"{len(dead)} inactive metric(s) tracked: {sorted(dead)[:4]}")


def check_aliases_flagged(F):
    """Known synonyms must be listed as aliases so a search for either name lands here."""
    cat = "terms"
    if "terms.md" not in F:
        return
    txt = read(F["terms.md"])
    required_alias_pairs = [
        ("best_bio_bits", "bio"),
        ("biosynthetic_fraction", "frac"),
        ("containment", "kmer_novelty"),
    ]
    missing = [f"{doc}←{alias}" for doc, alias in required_alias_pairs
               if not re.search(rf"###\s*`?{re.escape(doc)}`?.*?Aliases:.*?{re.escape(alias)}",
                                txt, re.S)]
    if missing:
        fail(cat, "code-key aliases documented", f"unlinked: {missing}")
    else:
        ok(cat, "code-key aliases documented", "code keys map to canonical names")


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACT — data.md against disk
# ═══════════════════════════════════════════════════════════════════════════

def check_dataset_paths(F):
    cat = "data"
    if "data.md" not in F:
        return
    if not DATA_ROOT.exists():
        skip(cat, "dataset paths exist on disk", f"{DATA_ROOT} not mounted")
        return
    txt = read(F["data.md"])
    # Only paths presented as LIVE (section 3). Deprecated ones are expected to be absent.
    live_sec = txt.split("## 4.")[0]
    paths = set(re.findall(r"`(/data2/ds85/[^`]+)`", live_sec))
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        fail(cat, "live dataset paths exist on disk", f"{missing}")
    else:
        ok(cat, "live dataset paths exist on disk", f"{len(paths)} paths verified")


def check_record_counts(F):
    """Record counts in data.md must match the files. A stale count is how a leaked or
    partially-rebuilt split goes unnoticed."""
    cat = "data"
    if "data.md" not in F:
        return
    if not DATA_ROOT.exists():
        skip(cat, "record counts match disk", f"{DATA_ROOT} not mounted")
        return
    txt = read(F["data.md"])
    expected = {}
    for fname, count in re.findall(r"\|\s*`(train|val|test)\.jsonl`\s*\|\s*([\d,]+)", txt):
        expected[f"splits_core/{fname}.jsonl"] = int(count.replace(",", ""))
    # RIPP row from the per-class table
    m = re.search(r"\|\s*\*\*RIPP\*\*\s*\|\s*[\d,]+\s*\|\s*\*\*([\d,]+)\*\*\s*\|"
                  r"\s*([\d,]+)\s*\|\s*([\d,]+)", txt)
    if m:
        for split, val in zip(("train", "val", "test"), m.groups()):
            expected[f"splits_class/RIPP/{split}.jsonl"] = int(val.replace(",", ""))
    bad = []
    for rel, want in expected.items():
        p = DATA_ROOT / rel
        if not p.exists():
            bad.append(f"{rel}: MISSING")
            continue
        with p.open("rb") as fh:
            got = sum(1 for _ in fh)
        if got != want:
            bad.append(f"{rel}: doc says {want:,}, disk has {got:,}")
    if bad:
        fail(cat, "record counts match disk", "; ".join(bad))
    else:
        ok(cat, "record counts match disk", f"{len(expected)} splits verified")


def check_manifest_coverage(F):
    """Every built per-class split must have a provenance record in the manifest."""
    cat = "data"
    mf = DATA_ROOT / "splits_class/manifest.json"
    if not mf.exists():
        skip(cat, "per-class splits are all in the manifest", "manifest not present")
        return
    manifest = set(json.loads(read(mf)))
    built = {d.name for d in (DATA_ROOT / "splits_class").iterdir()
             if d.is_dir() and (d / "train.jsonl").exists()}
    orphans = built - manifest
    if orphans:
        # This must be visible in data.md, or it is silent.
        txt = read(F["data.md"]) if "data.md" in F else ""
        if all(o in txt for o in orphans) and "orphan" in txt.lower():
            warn(cat, "per-class splits are all in the manifest",
                 f"{len(orphans)} orphaned but DOCUMENTED: {sorted(orphans)}")
        else:
            fail(cat, "per-class splits are all in the manifest",
                 f"built with no provenance record and not documented: {sorted(orphans)}")
    else:
        ok(cat, "per-class splits are all in the manifest", f"{len(built)} classes")


def check_run_registry(F):
    cat = "data"
    if "data.md" not in F:
        return
    if not RUNS_ROOT.exists():
        skip(cat, "run directories are registered", f"{RUNS_ROOT} not mounted")
        return
    txt = read(F["data.md"])
    on_disk = {d.name for d in RUNS_ROOT.iterdir() if d.is_dir()}
    # A run counts as registered if its name appears anywhere in data.md.
    unregistered = sorted(n for n in on_disk if n not in txt)
    if unregistered:
        fail(cat, "run directories are registered",
             f"{len(unregistered)} unregistered: {unregistered[:6]}")
    else:
        ok(cat, "run directories are registered", f"{len(on_disk)} runs")


def check_scored_outputs_stamp_their_config(F):
    """A rate whose scoring config is unstated is not a result.

    This is the check that would have caught the A0 inversion. The SAME key `on_class` held both
    the global-biosynthetic and the class-specific number, in files whose names differed only by
    window, so nothing on disk said which you were reading. Every scored output must now carry a
    `scoring` block naming the metric, the class, the marker set and the window.
    """
    cat = "data"
    if not RUNS_ROOT.exists():
        skip(cat, "scored outputs stamp their scoring config", f"{RUNS_ROOT} not mounted")
        return
    stamped, unstamped = [], []
    for p in RUNS_ROOT.glob("*/*_w[0-9]*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "on_class" not in d:
            continue
        s = d.get("scoring")
        if isinstance(s, dict) and {"cls", "window_nt", "marker_accessions"} <= set(s):
            stamped.append(p.name)
        else:
            unstamped.append(p.name)
    if unstamped:
        warn(cat, "scored outputs stamp their scoring config",
             f"{len(unstamped)} pre-fix file(s) carry a bare on_class: {sorted(unstamped)[:4]} "
             f"— rescore with novelty_battery.py to stamp them")
    elif stamped:
        ok(cat, "scored outputs stamp their scoring config", f"{len(stamped)} stamped")
    else:
        skip(cat, "scored outputs stamp their scoring config", "no scored outputs yet")


def check_scorer_is_class_gated(F):
    """The scorer must intersect with OBLIGATE_DOMAINS[cls], not just read `bio > 0`."""
    cat = "code"
    nb = REPO / "scripts/novelty_battery.py"
    if not nb.exists():
        skip(cat, "on_class is class-gated in the scorer", "novelty_battery.py missing")
        return
    src = read(nb)
    m = re.search(r"^\s*on_class\s*=\s*(.+)$", src, re.M)
    if not m:
        fail(cat, "on_class is class-gated in the scorer", "could not find the assignment")
        return
    expr = m.group(1)
    if "bio_accs" in expr and "marker_accs" in expr:
        ok(cat, "on_class is class-gated in the scorer", "intersects OBLIGATE_DOMAINS[cls]")
    elif '"bio"' in expr or "'bio'" in expr:
        fail(cat, "on_class is class-gated in the scorer",
             f"reads the GLOBAL bio score: {expr.strip()} — this is the A0 inversion bug (bugs.md)")
    else:
        warn(cat, "on_class is class-gated in the scorer", f"unrecognised form: {expr.strip()[:60]}")


def check_case_collisions(F):
    """Two run dirs differing only in case destroy each other on a careless copy."""
    cat = "data"
    if not RUNS_ROOT.exists():
        skip(cat, "no case-colliding run directories", f"{RUNS_ROOT} not mounted")
        return
    names = [d.name for d in RUNS_ROOT.iterdir() if d.is_dir()]
    seen: dict[str, list[str]] = {}
    for n in names:
        seen.setdefault(n.lower(), []).append(n)
    collisions = {k: v for k, v in seen.items() if len(v) > 1}
    if collisions:
        txt = read(F["data.md"]) if "data.md" in F else ""
        if "CASE COLLISION" in txt:
            warn(cat, "no case-colliding run directories",
                 f"DOCUMENTED collision: {list(collisions.values())}")
        else:
            fail(cat, "no case-colliding run directories", f"{list(collisions.values())}")
    else:
        ok(cat, "no case-colliding run directories", f"{len(names)} runs")


def check_deprecated_not_referenced(F):
    """A deprecated dataset must never appear in the active plan."""
    cat = "data"
    if "data.md" not in F or "plan.md" not in F:
        return
    dtxt = read(F["data.md"])
    dep_sec = dtxt.split("## 4.")[1].split("## 5.")[0] if "## 4." in dtxt else ""
    deprecated = set(re.findall(r"`([a-z_0-9/]*splits[a-z_0-9]*)/`", dep_sec))
    live = {"splits_core", "splits_class"}
    deprecated = {d for d in deprecated if d.rsplit("/", 1)[-1] not in live}
    plan = read(F["plan.md"])
    offenders = sorted(d for d in deprecated if d in plan)
    if offenders:
        fail(cat, "plan.md references no deprecated dataset", f"{offenders}")
    else:
        ok(cat, "plan.md references no deprecated dataset",
           f"{len(deprecated)} deprecated dataset(s) tracked")


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACT — plan.md discipline
# ═══════════════════════════════════════════════════════════════════════════

def check_plan_has_current_state(F):
    cat = "plan"
    if "plan.md" not in F:
        return
    txt = read(F["plan.md"])
    problems = []
    if "## Current State" not in txt:
        problems.append("no Current State section")
    if not re.search(r"\*\*Last updated:\*\*\s*20\d\d-\d\d-\d\d", txt):
        problems.append("no dated 'Last updated'")
    if "Phase Ledger" not in txt:
        problems.append("no Phase Ledger")
    if problems:
        fail(cat, "plan.md has the required sections", "; ".join(problems))
    else:
        ok(cat, "plan.md has the required sections", "Current State + Ledger + date")


def check_ledger_rows_have_provenance(F):
    """A ledger row without n, or a block without provenance, is not a result."""
    cat = "plan"
    if "plan.md" not in F:
        return
    txt = read(F["plan.md"])
    if "Provenance" not in txt and "provenance" not in txt:
        fail(cat, "ledger states provenance", "no provenance block under the Phase Ledger")
        return
    rows = [l for l in txt.splitlines() if re.match(r"^\|\s*P\d[\w-]*\s*\|", l)]
    if not rows:
        warn(cat, "ledger states provenance", "no ledger rows yet")
        return
    bad = [r.split("|")[1].strip() for r in rows if len(r.split("|")) < 7]
    if bad:
        fail(cat, "ledger states provenance", f"rows missing columns: {bad}")
    else:
        ok(cat, "ledger states provenance", f"{len(rows)} ledger rows, provenance block present")


def check_manipulation_check_required(F):
    """The Phase-2 lesson, enforced: the template must demand a manipulation check."""
    cat = "plan"
    if "plan.md" not in F:
        return
    txt = read(F["plan.md"])
    if "MANIPULATION CHECK" not in txt.upper():
        fail(cat, "intervention template demands a manipulation check",
             "absent — this is what made the Phase-2 weighted-arm null uninterpretable")
    else:
        ok(cat, "intervention template demands a manipulation check", "present")


def check_novelty_guard_present(F):
    cat = "plan"
    if "plan.md" not in F:
        return
    txt = read(F["plan.md"]).lower()
    if "novelty" not in txt:
        fail(cat, "novelty guard appears in the plan", "no mention — it gates every rung")
    else:
        ok(cat, "novelty guard appears in the plan", "present")


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACT — memory.md discipline
# ═══════════════════════════════════════════════════════════════════════════

def check_correction_format(F):
    """Corrections must be paired: an [INCORRECT] line and a dated [CORRECTION - ...] below it."""
    cat = "memory"
    if "memory.md" not in F:
        return
    txt = read(F["memory.md"])
    # Only line-initial markers count — the rules section quotes them inline as prose.
    inc = len(re.findall(r"^\[INCORRECT\]", txt, re.M))
    cor = len(re.findall(r"^\[CORRECTION - 20\d\d-\d\d-\d\d\]", txt, re.M))
    if inc == 0 and cor == 0:
        warn(cat, "corrections are well-formed", "none recorded yet")
    elif inc != cor:
        fail(cat, "corrections are well-formed",
             f"{inc} [INCORRECT] vs {cor} dated [CORRECTION] — they must pair")
    else:
        ok(cat, "corrections are well-formed", f"{inc} correction pair(s)")


def check_memory_append_marker(F):
    cat = "memory"
    if "memory.md" not in F:
        return
    if "APPEND NEW ENTRIES BELOW" not in read(F["memory.md"]):
        warn(cat, "memory.md has an append marker", "no explicit append point")
    else:
        ok(cat, "memory.md has an append marker", "present")


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACT — code ↔ docs
# ═══════════════════════════════════════════════════════════════════════════

def check_code_keys_documented(F):
    """Every dict key ladder_audit produces must be reachable from terms.md by name or alias."""
    cat = "code"
    la = REPO / "evo2/scripts/ladder_audit.py"
    if "terms.md" not in F or not la.exists():
        skip(cat, "ladder_audit keys are documented", "source or terms.md missing")
        return
    src = read(la)
    m = re.search(r'base\s*=\s*\{(.*?)\}\s*\n', src, re.S)
    if not m:
        skip(cat, "ladder_audit keys are documented", "could not parse the result dict")
        return
    keys = set(re.findall(r'"(\w+)"\s*:', m.group(1)))
    keys -= {"tag", "cls", "key"}          # identifiers, not metrics
    terms = read(F["terms.md"])
    undocumented = sorted(k for k in keys if k not in terms)
    if undocumented:
        fail(cat, "ladder_audit keys are documented",
             f"produced but absent from terms.md: {undocumented}")
    else:
        ok(cat, "ladder_audit keys are documented", f"{len(keys)} result keys")


def check_obligate_domains_classes(F):
    """Classes named in data.md must exist in OBLIGATE_DOMAINS, or class-specific scoring
    silently falls back to the global set — the A0 failure mode."""
    cat = "code"
    ev = REPO / "src/bgc_pipeline/evaluation.py"
    if "data.md" not in F or not ev.exists():
        skip(cat, "per-class targets exist in OBLIGATE_DOMAINS", "source missing")
        return
    src = read(ev)
    blk = src.split("OBLIGATE_DOMAINS: dict[str, list[str]] = {", 1)
    if len(blk) < 2:
        skip(cat, "per-class targets exist in OBLIGATE_DOMAINS", "could not parse")
        return
    known = set(re.findall(r'"([A-Z_]+)":', blk[1][:4000]))
    dtxt = read(F["data.md"])
    named = set(re.findall(r"\|\s*\*{0,2}([A-Z][A-Z_]{3,})\*{0,2}\s*\|", dtxt))
    named &= {"RIPP", "PKS", "TERPENE", "HSERLACTONE", "BUTYROLACTONE", "ECTOINE", "MELANIN",
              "NRPS", "SIDEROPHORE", "SACCHARIDE", "ALKALOID"}
    missing = sorted(named - known)
    if missing:
        fail(cat, "per-class targets exist in OBLIGATE_DOMAINS", f"{missing}")
    else:
        ok(cat, "per-class targets exist in OBLIGATE_DOMAINS",
           f"{len(named)} classes cross-checked against {len(known)} defined")


# ═══════════════════════════════════════════════════════════════════════════

CHECKS = [
    check_all_files_present, check_history_preserved, check_bugs_content_carried,
    check_load_bearing_claims, check_governor_size, check_governor_has_no_findings,
    check_internal_links,
    check_every_metric_defined, check_term_schema, check_primary_metric_states_config,
    check_retired_metrics_not_live, check_aliases_flagged,
    check_dataset_paths, check_record_counts, check_manifest_coverage, check_run_registry,
    check_case_collisions, check_deprecated_not_referenced,
    check_scored_outputs_stamp_their_config, check_scorer_is_class_gated,
    check_plan_has_current_state, check_ledger_rows_have_provenance,
    check_manipulation_check_required, check_novelty_guard_present,
    check_correction_format, check_memory_append_marker,
    check_code_keys_documented, check_obligate_domains_classes,
]


def run() -> int:
    F = locate()
    for fn in CHECKS:
        try:
            fn(F)
        except Exception as e:                                    # a check must never crash the run
            fail("internal", fn.__name__, f"{type(e).__name__}: {e}")

    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    C = {"PASS": "\033[32m", "FAIL": "\033[31m", "WARN": "\033[33m", "SKIP": "\033[90m",
         "R": "\033[0m", "B": "\033[1m"} if use_color else dict.fromkeys(
        ["PASS", "FAIL", "WARN", "SKIP", "R", "B"], "")
    mark = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP"}

    print(f"\n{C['B']}Documentation framework verifier{C['R']}")
    print(f"repo: {REPO}\n")
    last = None
    for status, cat, name, detail in _RESULTS:
        if status == "PASS" and not VERBOSE:
            continue
        if cat != last:
            print(f"{C['B']}[{cat}]{C['R']}")
            last = cat
        print(f"  {C[status]}{mark[status]}{C['R']}  {name}")
        if detail:
            print(f"        {detail}")

    n = {k: sum(1 for r in _RESULTS if r[0] == k) for k in mark}
    print(f"\n{C['B']}{n['PASS']} passed, {n['FAIL']} failed, "
          f"{n['WARN']} warnings, {n['SKIP']} skipped{C['R']}")
    if n["FAIL"]:
        print(f"{C['FAIL']}The docs and the repo disagree. Fix the docs or fix the code — "
              f"do not quote a number until this passes.{C['R']}\n")
        return 1
    if not VERBOSE and n["PASS"]:
        print("(-v to see passing checks)")
    print()
    return 0


# pytest entry points ────────────────────────────────────────────────────────
def test_docs_contract():
    assert run() == 0, "documentation contract violated — see output above"


if __name__ == "__main__":
    sys.exit(run())
