"""Random access into the raw GenBank tar.

The raw corpus is a single ~173 GB tar of per-genome gzipped GenBank files. Tar is a
sequential format, so pulling N specific genomes costs a full scan unless member offsets
are known. This builds that offset index once; every later pass seeks directly.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

TAR = Path("/data2/ds85/asdb5_gbks/asdb5_gbks.tar")


def build_index(tar_path: Path = TAR, out: Path | None = None) -> dict[str, dict]:
    """One sequential pass; record (offset_data, size) per genome accession."""
    out = out or tar_path.with_suffix(".index.json")
    idx: dict[str, dict] = {}
    with tarfile.open(tar_path, "r|") as tf:          # streaming mode, no seeking
        for m in tf:
            if not m.isfile():
                continue
            acc = Path(m.name).name
            for suffix in (".gbk.gz", ".gbff.gz", ".gbk", ".gbff"):
                if acc.endswith(suffix):
                    acc = acc[: -len(suffix)]
                    break
            idx[acc] = {"name": m.name, "offset": m.offset_data, "size": m.size}
    out.write_text(json.dumps(idx))
    return idx


def load_index(tar_path: Path = TAR) -> dict[str, dict]:
    p = tar_path.with_suffix(".index.json")
    if not p.exists():
        raise FileNotFoundError(f"index not built: {p} — run build_index() first")
    return json.loads(p.read_text())


def read_member(acc: str, index: dict[str, dict], tar_path: Path = TAR) -> str:
    """Return the decompressed GenBank text for one genome accession."""
    ent = index[acc]
    with open(tar_path, "rb") as fh:
        fh.seek(ent["offset"])
        raw = fh.read(ent["size"])
    if ent["name"].endswith(".gz"):
        return gzip.decompress(raw).decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    idx = build_index()
    print(f"indexed {len(idx)} genomes in {time.time()-t0:.0f}s", file=sys.stderr)
