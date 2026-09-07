"""THE SINGLE antiSMASH INVOCATION SITE (SPEC 3.1, 9.1).

There is exactly one function in this repository that runs antiSMASH, and it takes no
per-arm or per-class argument. That is structural, not stylistic: the benchmark's whole
claim rests on every arm being scored identically, and a second invocation site is how two
arms come to be scored differently without anyone deciding to do that.

The configuration below is FROZEN and hashed. The hash goes into every scored artifact's
filename (SPEC 9.3), so two scorings of one generation set can never collide.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

ANTISMASH = "/home/ds85/.local/share/mamba/envs/bgcmodel/bin/antismash"
ENV_BIN = "/home/ds85/.local/share/mamba/envs/bgcmodel/bin"
DATABASES = "/data2/ds85/antismash_db"

#: SPEC 3.1. `minlength=1` is deliberate and measured: the default of 1000 filters the
#: INPUT RECORD length, not the cluster length, and on extracted cores it silently drops
#: 72% of TERPENE and 48% of RIPP while dropping nothing from the long classes. A dropped
#: record is not a non-detection.
FROZEN = {
    "tool": "antismash",
    "version_expected": "8.0.4",
    "minimal": True,
    "genefinding_tool": "prodigal",
    "minlength": 1,
    "databases": DATABASES,
}

_LOC = re.compile(r"\[(\d+):(\d+)\]\(([+-])\)")

#: The version that ACTUALLY ran, filled in on first invocation. FROZEN records the
#: version we expect and the check compares only the MAJOR component, so a point release
#: with changed rules or HMMs would flip verdicts while every artifact still attested
#: "8.0.4". An artifact must record the instrument that produced it, not the one intended.
OBSERVED: dict = {"version": None}


def config_hash() -> str:
    return hashlib.sha256(
        json.dumps(FROZEN, sort_keys=True).encode()
    ).hexdigest()[:12]


def _parse_loc(loc: str) -> tuple[int, int, int]:
    m = _LOC.search(loc or "")
    if not m:
        return 0, 0, 1
    return int(m.group(1)), int(m.group(2)), (1 if m.group(3) == "+" else -1)


def run(records: list[tuple[str, str]], workdir: Path | None = None,
        cpus: int = 8) -> dict[str, dict]:
    """Score (accession, sequence) pairs. Returns accession -> verdict.

    EVERY SUBMITTED SEQUENCE MUST COME BACK. A shortfall between submitted and scored is
    raised, never absorbed: counting a dropped record as a non-detection makes the
    denominator silently input-dependent (SPEC 3.1, KNOWN_WRONG #5).
    """
    if not records:
        return {}
    accs = [a for a, _ in records]
    if len(accs) != len(set(accs)):
        from collections import Counter
        dup = [a for a, n in Counter(accs).items() if n > 1]
        raise ValueError(
            f"{len(dup)} duplicate accessions submitted (e.g. {dup[:3]}). antiSMASH "
            f"renames a duplicate to '<id>_0', so the totality check would still pass "
            f"while one record's verdict silently served for two."
        )
    import os

    env = dict(os.environ)
    env["PATH"] = ENV_BIN + os.pathsep + env.get("PATH", "")
    with tempfile.TemporaryDirectory(dir=str(workdir) if workdir else None) as td:
        tmp = Path(td)
        # OPAQUE POSITIONAL IDS. antiSMASH SANITISES record ids -- it silently strips
        # colons, so "oracle::TERPENE::GCF_x.region2" comes back as
        # "oracleTERPENEGCF_x.region2" and every join misses. Rather than enumerate which
        # characters survive, submit ids we control completely and map back.
        fa = tmp / "in.fasta"
        safe = {f"s{i:07d}": acc for i, (acc, _) in enumerate(records)}
        with open(fa, "w") as fh:
            for sid, (acc, seq) in zip(safe, records):
                fh.write(f">{sid}\n{seq}\n")
        cmd = [ANTISMASH, "--minimal", "--genefinding-tool", FROZEN["genefinding_tool"],
               "--minlength", str(FROZEN["minlength"]), "--databases", FROZEN["databases"],
               "--cpus", str(cpus), "--output-dir", str(tmp / "out"), str(fa)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"antiSMASH failed (rc={proc.returncode}):\n"
                               f"{proc.stderr[-3000:]}")
        jsons = list((tmp / "out").glob("*.json"))
        if not jsons:
            raise RuntimeError("antiSMASH produced no JSON output")
        out: dict[str, dict] = {}
        for jp in jsons:
            doc = json.loads(jp.read_text())
            ver = doc.get("version")
            OBSERVED["version"] = ver          # what actually ran, not what we expected
            if ver and not str(ver).startswith(FROZEN["version_expected"].split(".")[0]):
                raise RuntimeError(f"antiSMASH major version {ver} != "
                                   f"{FROZEN['version_expected']}; the frozen scoring "
                                   f"config no longer describes the instrument")
            for rec in doc.get("records", []):
                rid = rec.get("id")
                acc = safe.get(rid)
                if acc is None:
                    raise RuntimeError(
                        f"antiSMASH returned record id {rid!r}, which was not submitted. "
                        f"Ids are opaque positional keys precisely so this cannot happen "
                        f"silently."
                    )
                feats = rec.get("features", [])
                regions, cds = [], []
                for f in feats:
                    if f.get("type") == "region":
                        s, e, _ = _parse_loc(f.get("location"))
                        q = f.get("qualifiers", {})
                        regions.append({
                            "start": s, "end": e,
                            "products": list(q.get("product", [])),
                            "rules": list(q.get("rules", [])),
                            "contig_edge": (q.get("contig_edge", ["False"])[0] == "True"),
                        })
                    elif f.get("type") == "CDS":
                        s, e, st = _parse_loc(f.get("location"))
                        kind = (f.get("qualifiers", {})
                                 .get("gene_kind", ["(none)"])[0])
                        cds.append((s, e, st, kind))
                # antiSMASH 8 emits seq as {"data": ..., "alphabet": ...}, so len() on
                # the dict returns 2 and coding_density came out in the hundreds.
                raw = rec.get("seq")
                seq_str = raw.get("data", "") if isinstance(raw, dict) else (raw or "")
                seq_len = len(seq_str)
                cov = sum(e - s for s, e, _, _ in cds)
                # symmetric with the corpus definition (SPEC 4.2): the genes that
                # satisfied the detection rule, not every CDS. Available under --minimal.
                produced_core = sum(1 for _, _, _, k in cds if k == "biosynthetic")
                out[acc] = {
                    "scored_ok": True,
                    "detected": bool(regions),
                    "products": sorted({p for r in regions for p in r["products"]}),
                    "region_table": regions,
                    "n_cds": len(cds),
                    "produced_core_genes": produced_core,
                    "coding_density": round(cov / seq_len, 4) if seq_len else 0.0,
                    "gc_content": rec.get("gc_content"),
                    "scored_len": seq_len,
                }
    submitted = {a for a, _ in records}
    missing = submitted - set(out)
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(submitted)} submitted sequences received no verdict "
            f"(e.g. {sorted(missing)[:3]}). A dropped record is not a non-detection; "
            f"scoring must be total (SPEC 3.1)."
        )
    return out
