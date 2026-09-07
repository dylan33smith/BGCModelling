"""Product -> class map, REGENERATED from antiSMASH's own rule definitions.

SPEC 3.3, 10.3: regenerated, never copied, then diffed against any prior map. Copying
would inherit hand overrides of unknown provenance; regenerating and diffing shows what
they were.

THE CLASS LEVEL IS A CHOICE. antiSMASH supplies two levels and the benchmark sits between
them:

  CATEGORY  7 values (PKS, NRPS, RiPP, terpene, saccharide, alkaloid, other). Too coarse:
            `other` is an explicit grab-bag of ~30 unrelated products and is the largest
            group in the corpus.
  PRODUCT   ~103 values. Too fine: a class becomes a single product, which erases the
            class/subclass distinction that SPEC 3.6 depends on.

  class = CATEGORY, except for the products in PROMOTIONS below.

⚠ A DERIVED PROMOTION RULE WAS ATTEMPTED AND ABANDONED -- recorded so it is not retried.
The intended criterion was "promote a product whose trigger set fires no sibling rule in
its category". Implemented against the rule algebra it is **vacuous**: antiSMASH rules are
built from largely disjoint profile-HMM families, so every one of the 103 rules passes.
Size-only promotion was the other derived option; at the benchmark's common_n it promotes
13 products and drives the class level down to the product level, which removes subclass
structure everywhere.

So promotion is HAND-SPECIFIED, deliberately small, and each entry carries its reason.
This is a design choice (SPEC [C]) and the paper reports it as one. The empirical check is
the real-core confusion matrix (SPEC 3.5): if a promoted class is not separable from its
parent category, the off-diagonal mass shows it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

RULE_DIR = Path(
    "/home/ds85/.local/share/mamba/envs/bgcmodel/lib/python3.12/site-packages/"
    "antismash/detection/hmm_detection/cluster_rules"
)
_COMMENT = re.compile(r"#.*")

#: product -> why it is lifted out of its antiSMASH category. Additions need a reason.
PROMOTIONS: dict[str, str] = {
    "arylpolyene":
        "category PKS is 42.8% multi-gene and cannot express arylpolyene's 66.9%; "
        "its APE_KS trigger is a fatty-acid-like polyene synthase, not a canonical "
        "modular or iterative polyketide.",
    "betalactone":
        "category `other` is a declared grab-bag that antiSMASH does not assert is "
        "coherent, so membership carries no grouping information to preserve.",
}

#: the five classes the benchmark is defined over (SPEC 4.4.2). validate() enforces them.
BENCHMARK_CLASSES = ("TERPENE", "NRPS", "RIPP", "ARYLPOLYENE", "BETALACTONE")


@dataclass
class Rule:
    name: str
    category: str


def load_rules(rule_dir: Path = RULE_DIR) -> dict[str, Rule]:
    text = "\n".join(sorted(p.read_text() for p in rule_dir.glob("*.txt")))
    rules: dict[str, Rule] = {}
    for block in re.split(r"\nRULE ", text)[1:]:
        name = block.split()[0]
        cat = re.search(r"^\s*CATEGORY\s+(\S+)", block, re.M)
        if cat:
            rules[name] = Rule(name, cat.group(1))
    if not rules:
        raise RuntimeError(f"no rules parsed from {rule_dir} — refusing to build an "
                           f"empty class map")
    return rules


def build_map(rules: dict[str, Rule] | None = None) -> dict:
    rules = rules or load_rules()
    unknown = set(PROMOTIONS) - set(rules)
    if unknown:
        raise ValueError(f"PROMOTIONS names products antiSMASH does not define: "
                         f"{sorted(unknown)} — the rule set changed, revisit SPEC 4.4.2")
    mapping = {
        name: (name.upper() if name in PROMOTIONS else r.category.upper())
        for name, r in rules.items()
    }
    return {
        "mapping": mapping,
        "promotions": PROMOTIONS,
        "n_rules": len(rules),
        "classes": sorted(set(mapping.values())),
        "unmapped_policy":
            "a product with no CATEGORY in the installed rule files is recorded as "
            "UNMAPPED and EXCLUDED, never silently folded into OTHER",
    }


def mapping_hash(mapping: dict[str, str]) -> str:
    """The map is REBUILT from the installed antiSMASH on every run and never recorded.
    A within-major upgrade that renamed a product or moved a category would silently
    redefine what the benchmark measures, and no artifact would show the change. Hashing it
    makes the redefinition visible in every run directory that used it."""
    import hashlib
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True).encode()).hexdigest()[:12]


def classify(products: list[str], mapping: dict[str, str]) -> list[str]:
    """SPEC 3.3 hybrid rule: a record counts for EVERY class its products map to."""
    return sorted({mapping.get(p, "UNMAPPED") for p in products})


def validate(mapping: dict[str, str],
             classes: tuple[str, ...] = BENCHMARK_CLASSES) -> None:
    """Fail loudly if the benchmark's classes did not survive map regeneration.

    The map is rebuilt from the *installed* antiSMASH; a version bump that renamed a
    product or moved a category would otherwise silently redefine what the benchmark
    measures. That must be an error, not a surprise in a results table.
    """
    produced = set(mapping.values())
    missing = [c for c in classes if c not in produced]
    if missing:
        raise RuntimeError(
            f"benchmark classes absent from the regenerated map: {missing}. "
            f"The installed antiSMASH rule set no longer yields them; SPEC 4.4.2 must be "
            f"revisited before any corpus is built."
        )
    for p in PROMOTIONS:
        if mapping.get(p) != p.upper():
            raise RuntimeError(f"promotion of {p!r} did not take effect (got "
                               f"{mapping.get(p)!r})")
    # a promoted product must not also survive under its old category name by accident
    for p in PROMOTIONS:
        siblings = [k for k, v in mapping.items() if v == p.upper()]
        if siblings != [p]:
            raise RuntimeError(f"class {p.upper()} is not exactly one product: {siblings}")


if __name__ == "__main__":
    m = build_map()
    validate(m["mapping"])
    print(json.dumps({k: v for k, v in m.items() if k != "mapping"}, indent=2))
    print(f"\n{len(m['mapping'])} products -> {len(m['classes'])} classes")
    for c in BENCHMARK_CLASSES:
        members = sorted(k for k, v in m["mapping"].items() if v == c)
        print(f"  {c:14s} <- {len(members):2d} product(s): {members[:6]}"
              f"{' ...' if len(members) > 6 else ''}")
