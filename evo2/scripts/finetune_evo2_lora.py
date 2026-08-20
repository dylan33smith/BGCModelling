#!/usr/bin/env python3
"""Fine-tune Evo2 on BGC data using LoRA adapters (PEFT).

⚠️ THE PARAMETER COUNTS IN THIS HEADER ARE FOR THE 7B. Phases 3/6/7 train the **1B**
(`EVO2_BASE_MODEL=evo2_1b_base`, 25 blocks, 1,118,465,280 params, native context **8,192** — a hard
model limit). On the 1B the same LoRA config yields **10,475,520 trainable params = 0.937%**, 208
tensors. The run's own log line is authoritative; this header is kept for the 7B sizing argument.

CONDITIONING PREFIX — still present, and a deliberate no-op for per-class adapters.
Every record's `training_text` is `|COMPOUND_CLASS:X|` + `|taxonomic_tag|` + sequence, built once at
dataset-build time. Two facts about it:
  * The prefix is **MASKED FROM THE LOSS** (H3, below), so it contributes no gradient. It is context
    the model may attend to, not a target.
  * Inside a per-class adapter the class tag has exactly **ONE distinct value**, so it carries zero
    information. Phase 1 measured the tag at **−0.0006 nats** — no effect — and that is precisely
    why Phase 3 stopped conditioning on labels and started routing to per-class adapters instead.
  * The **taxonomic tag is NOT a no-op**: 1,260–1,494 distinct values per class. It is real varying
    conditioning and is the bulk of the prefix (~110 of ~130 characters).
⇒ It is retained for **train/generation consistency** — `seed_generate.py` prepends the same prefix
  (`class_tag=True`) — not because the class tag does anything. Removing it would invalidate
  comparison with every Phase-3 arm. Cost: ~130 characters, 6.8% of a median RIPP record, 10.1% for
  PKS, **14.2% for TERPENE**.

Why LoRA (not full fine-tune)
-----------------------------
A full-parameter fine-tune of Evo2 7B needs at minimum:
    bf16 weights (14 GB) + grads (14 GB) + AdamW states (56 GB) = 84 GB
before any activations. That exceeds every single GPU we have or have
had access to:
  - trojai (4× A40 48 GB): the 84 GB base is only reachable by sharding
    optimizer + grads via DeepSpeed ZeRO-2 across 4 ranks. Smoke testing
    showed StripedHyena activations (~14 GB at L=1024) still push each
    rank over the 46 GB A40 budget on `optimizer.step()`. Details in
    `docs/trojai/FINETUNE_GUIDE.md` §12.
  - gputee (1× H100 80 GB): there is no second GPU to shard to. 84 GB
    > 80 GB even before activations.

LoRA eliminates the optimizer-state term entirely: frozen base weights
carry no grads and no optimizer state, and the ~28 M adapter params
need only ~336 MB of AdamW state total. Forward/backward still run
through the frozen base (so activation memory is unchanged), but the
critical OOM point (`optimizer.step()`) goes away.

LoRA targets (all nn.Linear layers in Evo2):
  - Attention blocks (every ~8th of 32): Wqkv, out_proj
  - All blocks: out_filter_dense (Hyena dense output projection)
  - All blocks MLP: l1, l2, l3

Trainable parameter budget (r=16):
  - Attention (4 blocks × Wqkv + out_proj):   ~1.6 M
  - Hyena filter output (28 blocks):           ~3.7 M
  - MLP (32 blocks × l1,l2,l3):              ~23.2 M
  - Total:                                   ~28.5 M  (0.44% of 6.48 B)

Launch
------
    # gputee, single H100 — note output-dir on /data2 (/home is tight)
    micromamba activate bgcmodel
    export HF_HOME=/data2/ds85/hf_cache
    deepspeed --num_gpus=1 \\
        scripts/finetune_evo2_lora.py \\
        --train         data/processed/splits_combined/train.jsonl \\
        --val           data/processed/splits_combined/val.jsonl \\
        --output-dir    /data2/ds85/bgcmodel_runs/phase1_lora \\
        --grad-accum    32                        \\
        --wandb-project bcg-evo2-phase1

Notes:
  - `--grad-accum`: default (8) targets world_size=4 for an effective batch
    of 128. On a single H100 use `--grad-accum 32` to keep effective batch
    at 128. See docs/gputee/FINETUNE_GUIDE.md §4.
  - `--output-dir` on `/data2`: on gputee `/home` is near-full (~30 GB free
    as of 2026-04-22). Checkpoints, samples, offline wandb logs, and plots
    all go under the output dir. `/data2` has ~1.5 TB free. See
    docs/gputee/FINETUNE_GUIDE.md §2 (storage layout).
  - Checkpoints exclude frozen base-model weights (~25 GB/ckpt that would
    otherwise be redundant with the HF cache). See §11 of the guide.

Full documentation: docs/gputee/FINETUNE_GUIDE.md §12.

Project design notes:
  - Phase 1 is class-only. COMPOUND token stripped during merge step.
  - Goal is transposition, not invention (FINETUNE_GUIDE.md §10).
  - Reproducibility: config + data fingerprint + pip freeze saved at start.
  - Three parallel viz paths: WandB, JSONL logs, PNG plots.
  - Adapter weights saved in peft format (adapter_config.json + adapter_model.safetensors)
    alongside DeepSpeed optimizer/scheduler state, so checkpoints are
    loadable both via peft and DeepSpeed resume.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
from functools import partial
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────

# Base Evo2 checkpoint. Overridable via env so training AND eval (evo2_inference
# imports this constant) switch together — e.g. EVO2_BASE_MODEL=evo2_1b_base for the
# fast 1B/8k probe loop (cap --max-seq-len 8192 for that model). Default = production 7B.
EVO2_MODEL_NAME = os.environ.get("EVO2_BASE_MODEL", "evo2_7b_262k")
PAD_TOKEN_ID    = 0
IGNORE_INDEX    = -100

# Phase-2 custom objectives (frame-aware / domain-weighted). Model-agnostic, in the shared package
# so the 1B track uses exactly the same code path.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bgc_pipeline.objective import custom_lm_loss  # noqa: E402

# End-of-BGC sentinel (audit M11 / long-sequence support). Appended — supervised,
# NOT loss-masked — to the *final* window of a BGC so the model learns where a
# cluster ends and can terminate generation. "END" contains non-nucleotide chars,
# so "|END|" can never occur inside an ACGTN sequence (unambiguous stop string).
EOS_MARKER = "|END|"

# ⚠️ THE REAL EOS IS A SINGLE TOKEN AND WE NEVER TRAINED IT (2026-08-20).
# `EOS_MARKER` is a 5-BYTE STRING (124,69,78,68,124) and `hit_eos` has read 0 in every arm ever
# generated. Meanwhile `CharLevelTokenizer.eos` is **token id 0**, which Evo2's own pretraining
# already uses as a sequence separator: measured on real held-out cores, P(id 0) at the true end vs
# mid-core is 40.9x for the BASE model and 2,100x for our adapter. And it is CAUSAL -- masking id 0
# to -inf at generation restores the median length from 4,583 to the full 8,000 requested tokens.
# ⚠️ Evo2's tokenizer appends NOTHING automatically (unlike GenomeOcean's BPE tokenizer, which wraps
# every sequence BOS=1 ... EOS=2). So the id must be appended by hand, after tokenisation.
EOS_TOKEN_ID = 0

# Base hyperparameters — same schedule as full fine-tune
DEFAULTS = dict(
    max_seq_len       = 32768,
    batch_size        = 4,
    grad_accum        = 8,
    lr                = 5e-5,   # higher than full FT (1e-5) — LoRA adapters can use more LR
    lr_min_ratio      = 0.1,
    warmup_steps      = 200,
    max_epochs        = 2,
    weight_decay      = 0.01,   # lower WD for LoRA (adapters are tiny; heavy WD hurts)
    grad_clip         = 1.0,
    beta1             = 0.9,
    beta2             = 0.95,
    seed              = 42,
    log_every         = 10,
    val_every         = 250,
    save_every        = 500,
    val_max_batches   = 500,
    keep_last_ckpts   = 5,
    # audit m3: also rotate emergency checkpoints (step_N_oom/_interrupted/_final).
    # With C6 auto-resume these can be written on every preemption; without a cap
    # they accumulate unbounded and refill the disk. Keep the newest few.
    keep_special_ckpts = 2,
    # Early stopping on validation loss (audit M7). 0 disables; otherwise stop
    # after this many consecutive validations with no improvement >= min_delta.
    early_stopping_patience  = 0,
    early_stopping_min_delta = 0.0,
)

# Validation length-bucket boundaries (nt) for stratified val-loss reporting
# (audit M2). First-window validation makes the loss comparable to inference;
# stratifying by full BGC length separates "whole short BGC" from "prefix of a
# long BGC" so a length-correlated regression is visible.
VAL_LENGTH_BOUNDS = [16384, 32768, 65536, 131072]

# LoRA-specific defaults
LORA_DEFAULTS = dict(
    lora_r           = 16,    # adapter rank
    lora_alpha       = 32,    # scaling = alpha / r = 2.0
    lora_dropout     = 0.05,  # light regularisation
)

# LoRA target modules — suffix-matched against all named Linear layers
# Covers attention (Wqkv, out_proj), Hyena dense (out_filter_dense), MLP (l1,l2,l3)
LORA_TARGET_MODULES = [
    "Wqkv",
    "out_proj",
    "out_filter_dense",
    "l1",
    "l2",
    "l3",
]


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--train",      type=Path, required=True)
    p.add_argument("--val",        type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name",   type=str,  default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    for k, v in LORA_DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    p.add_argument("--lora-targets", type=str, nargs="+",
                   default=LORA_TARGET_MODULES,
                   help="Linear module name suffixes to add LoRA adapters to")
    p.add_argument("--lora-target-parameters", type=str, nargs="*", default=[],
                   help="Raw parameter-name suffixes to adapt via peft "
                        "target_parameters (e.g. 'projections.weight' for the "
                        "Hyena input projection — a TELinear nn.Module that "
                        "target_modules cannot reach). Empty = disabled. "
                        "[2026-07-03 diagnosis: the long-range Hyena mixing "
                        "input projection is otherwise frozen.]")
    ac = p.add_mutually_exclusive_group()
    ac.add_argument("--activation-checkpointing", dest="activation_checkpointing",
                    action="store_true",
                    help="Enable block-level activation checkpointing "
                         "(default: enabled).")
    ac.add_argument("--no-activation-checkpointing", dest="activation_checkpointing",
                    action="store_false",
                    help="Disable block-level activation checkpointing.")
    p.set_defaults(activation_checkpointing=True)
    # ── Phase-2 custom objective ────────────────────────────────────────────
    # Defaults are the IDENTITY: weight 1.0 and lambda 0.0 reproduce causal_lm_loss bit for bit,
    # so an un-flagged run is exactly the control cell of the factorial.
    p.add_argument("--domain-weight", type=float, default=1.0,
                   help="Up-weight class-defining domain positions by this factor. 1.0 = off. "
                        "Weights are normalised PER RECORD so every core contributes the same "
                        "total regardless of coverage (78.6%% in cores <1 kb vs 25.1%% >50 kb); an "
                        "unnormalised weight silently becomes a LENGTH weighting.")
    p.add_argument("--frame-lambda", type=float, default=0.0,
                   help="Weight on the in-gene stop-completion penalty. 0.0 = off. Penalises "
                        "probability mass on a base that would close a stop codon at codon phase 2 "
                        "inside an annotated gene; real gene termini are exempt.")
    p.add_argument("--annotations", type=Path, default=None,
                   help="`*.domain_spans.jsonl` sidecar (scripts/build_domain_spans.py). REQUIRED "
                        "when --domain-weight != 1.0 or --frame-lambda > 0.")
    # EOS / continuation conditioning (audit M11 + long-sequence support).
    eg = p.add_mutually_exclusive_group()
    eg.add_argument("--eos-token", dest="eos_token", action="store_true",
                    help="Append a supervised |END| token to the final window of "
                         "each BGC so the model learns to terminate (default: on).")
    eg.add_argument("--no-eos-token", dest="eos_token", action="store_false",
                    help="Do not append an end-of-BGC token.")
    p.set_defaults(eos_token=True)
    eg.add_argument("--eos-mode", choices=("token", "marker", "both"), default="token",
                    help="HOW the end-of-record signal is written. 'token' (DEFAULT since "
                         "2026-08-20) appends the tokenizer's real single-token EOS (id 0) after "
                         "tokenisation -- the token Evo2 pretrained with and the one the model "
                         "already emits. 'marker' is the pre-2026-08-20 behaviour: the 5-byte "
                         "STRING '|END|', which never once fired at generation. 'both' writes the "
                         "marker then the token. Recorded in the run config -- it is part of what "
                         "the model was trained to do.")
    cg = p.add_mutually_exclusive_group()
    cg.add_argument("--continuation-prefix", dest="continuation_prefix",
                    action="store_true",
                    help="Label interior chunk windows with a distinct "
                         "|CONTINUATION:…| prefix instead of the class-start "
                         "prefix (default: on; resolves audit M11).")
    cg.add_argument("--no-continuation-prefix", dest="continuation_prefix",
                    action="store_false",
                    help="Use the class prefix on all windows (legacy M11 behaviour).")
    p.set_defaults(continuation_prefix=True)
    p.add_argument("--gene-aware-chunking", action="store_true",
                   help="Snap chunk window boundaries to gene gaps so complete "
                        "genes/modules aren't split across windows (needs "
                        "'core_gene_bounds' per record; falls back to arithmetic "
                        "tiling where absent or for single genes > window budget).")
    p.add_argument("--smoke-pad-to-max-seq-len", action="store_true",
                   help="For train batches only: pad every micro-batch to a fixed "
                        "length of --max-seq-len (pad token + ignored labels on "
                        "padding). Use for GPU memory sweeps so peak memory reflects "
                        "the requested L even when dataset samples are shorter. "
                        "Validation still uses natural-length collation.")
    p.add_argument(
        "--long-seq-strategy",
        type=str,
        default="truncate",
        choices=["truncate", "chunk"],
        help="truncate: tokenize full training_text then clip to --max-seq-len "
             "(legacy). chunk: slice each record's nucleotide sequence into "
             "windows with --chunk-overlap; nucleotide budget per chunk is "
             "max_seq_len minus the prefix token count (see --auto-prefix-budget).",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=2048,
        help="Nucleotide overlap between consecutive chunks (--long-seq-strategy chunk). "
             "Stride = (max_seq_len - prefix_token_cap) - chunk_overlap.",
    )
    p.add_argument(
        "--prefix-budget",
        type=int,
        default=256,
        help="Fixed tokenizer-token allowance for the conditioning prefix when "
             "--no-auto-prefix-budget is set (chunk mode). Ignored when auto scan is on.",
    )
    apb = p.add_mutually_exclusive_group()
    apb.add_argument(
        "--auto-prefix-budget",
        dest="auto_prefix_budget",
        action="store_true",
        help="Chunk mode (default): scan each JSONL once with the Evo2 tokenizer, "
             "store max prefix token count in <stem>.lengths.meta.json, and use "
             "max_seq_len - max_prefix_tokens for nucleotide windows (canonical "
             "prefix from COMPOUND_CLASS + taxonomic_tag). Falls back to "
             "--prefix-budget if scan is disabled.",
    )
    apb.add_argument(
        "--no-auto-prefix-budget",
        dest="auto_prefix_budget",
        action="store_false",
        help="Chunk mode: use fixed --prefix-budget for every row (legacy).",
    )
    p.set_defaults(auto_prefix_budget=True)
    p.add_argument(
        "--prefix-slack-tokens",
        type=int,
        default=0,
        help="Chunk mode: extra token headroom subtracted from seq_budget so "
             "len(tokenize(prefix + sub)) is guaranteed <= max_seq_len even if "
             "the tokenizer ever introduces a seam token (H6). Stored in the "
             "lengths sidecar meta; --auto-prefix-budget scans for empirical "
             "overflow and uses max(--prefix-slack-tokens, measured).",
    )
    p.add_argument(
        "--lengths-cache-dir",
        type=Path,
        default=None,
        help="Directory for <jsonl_stem>.lengths.npy sidecars (default: parent of --train).",
    )
    p.add_argument("--max-steps",  type=int, default=0,
                   help="If >0, stop after this many optimizer steps (smoke tests)")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Adapter checkpoint directory to resume from")
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()
    # A REQUESTED OBJECTIVE THAT SILENTLY DOES NOTHING IS THE WORST OUTCOME: the run completes,
    # the curve looks normal, and the "treatment" arm is in fact the control.
    if (args.domain_weight != 1.0 or args.frame_lambda > 0) and args.annotations is None:
        raise SystemExit(
            "ABORT: --domain-weight/--frame-lambda were set but --annotations was not. The "
            "objective would fall back to uniform weights and a never-firing penalty — i.e. the "
            "baseline, reported as the treatment. Build the sidecar with "
            "`python scripts/build_domain_spans.py --splits train`.")
    return args


# ────────────────────────────────────────────────────────────────────────
# Distributed helpers  (identical to finetune_evo2.py)
# ────────────────────────────────────────────────────────────────────────

def ddp_rank() -> int:        return int(os.environ.get("RANK", 0))
def ddp_local_rank() -> int:  return int(os.environ.get("LOCAL_RANK", 0))
def ddp_world_size() -> int:  return int(os.environ.get("WORLD_SIZE", 1))
def is_main() -> bool:        return ddp_rank() == 0

def rank0_print(*msg, **kw) -> None:
    if is_main():
        print(f"[rank0 {datetime.now().strftime('%H:%M:%S')}]", *msg, **kw, flush=True)


# ────────────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # audit m2: use deterministic kernels where available; warn_only=True so the
    # StripedHyena FFT long-convs / flash-attention (which have no deterministic
    # CUDA kernel) emit a one-time warning instead of crashing. Resume reproduces
    # the data stream + optimizer state exactly, but loss values are NOT bit-exact.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _seed_dataloader_worker(worker_id: int) -> None:
    """Deterministically seed each DataLoader worker's RNG (audit: worker seeding).

    Without this, the two workers' numpy/python RNGs are seeded from entropy, so
    any future in-worker randomness (augmentation, sampling) would not be
    reproducible across a resume. The data ORDER is already deterministic via the
    DistributedSampler; this hardens everything else in the worker.
    """
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed((base + worker_id) % (2 ** 32))
    random.seed(base + worker_id)


def wandb_log_safe(run: Any, data: dict, step: int) -> None:
    """Log to WandB without ever letting a logging error stall/kill training."""
    if run is None:
        return
    try:
        run.log(data, step=step)
    except Exception as e:
        rank0_print(f"  (wandb.log failed, continuing: {e})")


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_config(args: argparse.Namespace, output_dir: Path) -> None:
    cfg = {
        "hyperparameters": {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()},
        "git_commit":      git_commit_hash(),
        "hostname":        os.uname().nodename,
        "start_time_utc":  datetime.now(timezone.utc).isoformat(),
        "torch_version":   torch.__version__,
        "cuda_version":    torch.version.cuda,
        "world_size":      ddp_world_size(),
        "gpus":            [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())],
        "mode":            "lora",
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def _file_fingerprint(path: Path) -> dict:
    """Full-file sha256 + size + line count (audit m1 — was first-100-lines only)."""
    st = path.stat()
    n, hasher = 0, hashlib.sha256()
    with path.open("rb") as f:
        for line in f:
            n += 1
            hasher.update(line)
    return {"path": str(path), "size_bytes": st.st_size, "lines": n,
            "sha256_full": hasher.hexdigest()}


def save_data_fingerprint(args: argparse.Namespace, output_dir: Path) -> None:
    """Fingerprint train/val on first run; on resume (file exists), COMPARE and
    warn loudly if the data changed since the checkpoint started training on it.

    The old version hashed only the first 100 lines and never compared on resume,
    so a regenerated split would pass silently (audit m1).
    """
    path = output_dir / "data_fingerprint.json"
    current = {"train": _file_fingerprint(args.train), "val": _file_fingerprint(args.val)}
    if path.exists():
        try:
            saved = json.loads(path.read_text())
        except Exception:
            saved = {}
        for split in ("train", "val"):
            old = saved.get(split, {})
            new = current[split]
            if old.get("sha256_full") and old["sha256_full"] != new["sha256_full"]:
                rank0_print(
                    f"  ⚠️  DATA FINGERPRINT MISMATCH ({split}): the data CHANGED since "
                    f"this run started (sha {old['sha256_full'][:12]} → "
                    f"{new['sha256_full'][:12]}, lines {old.get('lines')} → {new['lines']}). "
                    f"Resuming on different data than the checkpoint trained on corrupts "
                    f"the run — verify {new['path']} before continuing."
                )
        # Do NOT overwrite the original fingerprint on resume.
    else:
        path.write_text(json.dumps(current, indent=2))


def save_env(output_dir: Path) -> None:
    try:
        pf = subprocess.check_output(["pip", "freeze"],
                                     stderr=subprocess.DEVNULL).decode()
        (output_dir / "env.txt").write_text(pf)
    except Exception as e:
        (output_dir / "env.txt").write_text(f"pip freeze failed: {e}\n")


# ────────────────────────────────────────────────────────────────────────
# Dataset  (identical to finetune_evo2.py)
# ────────────────────────────────────────────────────────────────────────


def _lengths_sidecar_paths(cache_dir: Path, jsonl_path: Path) -> tuple[Path, Path]:
    stem = jsonl_path.stem
    return cache_dir / f"{stem}.lengths.npy", cache_dir / f"{stem}.lengths.meta.json"


def _lengths_meta_matches(meta: dict[str, Any], jsonl_path: Path) -> bool:
    stat = jsonl_path.stat()
    return (
        meta.get("path") == str(jsonl_path.resolve())
        and meta.get("size") == stat.st_size
        and meta.get("mtime") == int(stat.st_mtime)
    )


def load_or_build_lengths_sidecar(jsonl_path: Path, cache_dir: Path) -> np.ndarray:
    """Load int32 per-record sequence lengths; build sidecar if missing or stale."""
    npy_path, meta_path = _lengths_sidecar_paths(cache_dir, jsonl_path)
    stem = jsonl_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    if npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if _lengths_meta_matches(meta, jsonl_path):
            arr = np.load(npy_path)
            if int(meta.get("count", -1)) == len(arr):
                return arr.astype(np.int32, copy=False)

    lengths: list[int] = []
    with jsonl_path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            lengths.append(len(rec["sequence"]))
    arr = np.asarray(lengths, dtype=np.int32)
    stat = jsonl_path.stat()
    meta = {
        "path": str(jsonl_path.resolve()),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "count": len(arr),
    }
    tmp_npy = cache_dir / f"{stem}.lengths.pending-{os.getpid()}.npy"
    tmp_meta = meta_path.with_suffix(".json.pending")
    np.save(tmp_npy, arr)
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_npy, npy_path)
    os.replace(tmp_meta, meta_path)
    return arr


def canonical_phase1_prefix(rec: dict[str, Any]) -> str:
    """Rebuild Phase 1 conditioning prefix from JSON fields (merge pipeline format).

    This is the FIRST-window prefix: it marks the start of a real cluster.
    """
    cc = str(rec.get("compound_class", ""))
    tax = str(rec.get("taxonomic_tag", ""))
    return f"|COMPOUND_CLASS:{cc}|{tax}"


def continuation_phase1_prefix(rec: dict[str, Any]) -> str:
    """Interior-window (continuation) prefix (audit M11 + chained generation).

    Distinct from the first-window prefix so the model can tell "start of an
    NRPS cluster" from "continue an NRPS cluster mid-way". This keeps the
    `|COMPOUND_CLASS:…|` token meaning crisp (it only ever marks a real start)
    and gives chained long-sequence generation a clean continuation mode. It is
    strictly shorter than the first-window prefix (CONTINUATION < COMPOUND_CLASS),
    so the prefix-token budget computed from the first-window form bounds it.
    """
    cc = str(rec.get("compound_class", ""))
    tax = str(rec.get("taxonomic_tag", ""))
    return f"|CONTINUATION:{cc}|{tax}"


def training_prefix_for_chunking(rec: dict[str, Any]) -> str:
    """Prefix string before ``sequence``; prefer canonical assembly, else slice."""
    seq = rec.get("sequence") or ""
    tt = rec.get("training_text") or ""
    canon = canonical_phase1_prefix(rec)
    if seq and canon + seq == tt:
        return canon
    if seq and tt.endswith(seq):
        return tt[: len(tt) - len(seq)]
    raise ValueError(
        "Cannot derive training prefix: need training_text to end with sequence "
        "and (optional) canonical |COMPOUND_CLASS:…| + taxonomic_tag match."
    )


def count_prefix_tokens(tokenizer: Any, prefix: str) -> int:
    toks = tokenizer.tokenize(prefix)
    return len(toks) if isinstance(toks, (list, tuple)) else len(list(toks))


def assert_pad_token_safe(tokenizer: Any) -> None:
    """Fail fast if ``PAD_TOKEN_ID`` collides with any character we'd train on.

    The collate fills tail positions with ``PAD_TOKEN_ID`` (and marks
    those positions IGNORE in labels). If the tokenizer happens to map a
    real character (nucleotide or prefix punctuation) to the same id, a
    real position would be indistinguishable from padding and the loss
    mask would ignore real content. CharLevelTokenizer encodes characters
    by ord(), so PAD_TOKEN_ID=0 is safe — this check just keeps the
    invariant explicit (audit M4).
    """
    test_chars = "ACGTUNacgtun|:_-"
    bad: list[str] = []
    for c in test_chars:
        try:
            toks = tokenizer.tokenize(c)
        except Exception:
            continue
        if not toks:
            continue
        first = toks[0] if isinstance(toks, (list, tuple)) else next(iter(toks), None)
        if first is None:
            continue
        if int(first) == PAD_TOKEN_ID:
            bad.append(c)
    if bad:
        raise ValueError(
            f"PAD_TOKEN_ID={PAD_TOKEN_ID} collides with character(s) {bad!r} "
            f"under the active tokenizer; the labels mask would silently drop "
            f"real content. Change PAD_TOKEN_ID or use a different tokenizer."
        )


def scan_jsonl_max_prefix_tokens(jsonl_path: Path, tokenizer: Any) -> tuple[int, int]:
    """Return (max_prefix_token_count, n_rows_where_canonical_mismatch)."""
    max_tok = 0
    n_mismatch = 0
    with jsonl_path.open("rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            rec = json.loads(raw)
            seq = rec.get("sequence") or ""
            tt = rec.get("training_text") or ""
            canon = canonical_phase1_prefix(rec)
            if seq and canon + seq != tt:
                n_mismatch += 1
            prefix = training_prefix_for_chunking(rec)
            max_tok = max(max_tok, count_prefix_tokens(tokenizer, prefix))
    return max_tok, n_mismatch


def compute_prefix_slack_tokens(
    jsonl_path: Path,
    tokenizer: Any,
    max_seq_len: int,
    prefix_token_cap: int,
    n_samples: int = 64,
) -> int:
    """Sample records and measure worst-case tokeniser-vs-character overflow.

    For Evo2's CharLevelTokenizer this returns 0 because
    ``len(tokenize(prefix + sub)) == len(prefix) + len(sub)`` exactly. The
    function exists to make any future tokenizer change (BPE, sentencepiece,
    etc.) fail loudly during sidecar build instead of silently clipping the
    tail of long chunks at training time (audit item H6).

    Returns ``max(observed_overflow) + 4`` so the chunked ``seq_budget`` can
    be shrunk to keep every record under ``max_seq_len`` even when the
    tokenizer merges across the prefix↔sequence seam.
    """
    seq_budget = max_seq_len - prefix_token_cap
    if seq_budget <= 0:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) must exceed prefix_token_cap ({prefix_token_cap})"
        )
    max_overflow = 0
    seen = 0
    with jsonl_path.open("rb") as f:
        for raw in f:
            if seen >= n_samples:
                break
            if not raw.strip():
                continue
            rec = json.loads(raw)
            seq = rec.get("sequence") or ""
            if not seq:
                continue
            prefix = training_prefix_for_chunking(rec)
            sub = seq[: min(seq_budget, len(seq))]
            text = prefix + sub
            toks = tokenizer.tokenize(text)
            n = len(toks) if isinstance(toks, (list, tuple)) else len(list(toks))
            if n > max_seq_len:
                max_overflow = max(max_overflow, n - max_seq_len)
            seen += 1
    return max_overflow + 4 if max_overflow > 0 else 0


def ensure_max_prefix_tokens_in_meta(
    jsonl_path: Path,
    cache_dir: Path,
    tokenizer: Any,
    evo_model_tag: str,
    max_seq_len: int,
    cli_prefix_slack_tokens: int = 0,
) -> tuple[int, int]:
    """Write ``max_prefix_tokens`` + ``prefix_slack_tokens`` (+ evo tag) into lengths meta.

    ``cli_prefix_slack_tokens`` acts as a floor; the empirically-measured
    overflow slack (from :func:`compute_prefix_slack_tokens`) overrides if
    larger. Returns ``(max_prefix_tokens, prefix_slack_tokens)``.
    """
    _, meta_path = _lengths_sidecar_paths(cache_dir, jsonl_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing lengths meta: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if not _lengths_meta_matches(meta, jsonl_path):
        raise ValueError(f"Stale lengths meta for {jsonl_path}; rebuild sidecars first.")
    cached_ok = (
        meta.get("max_prefix_tokens_evo_tag") == evo_model_tag
        and "max_prefix_tokens" in meta
        and "prefix_slack_tokens" in meta
        and int(meta["prefix_slack_tokens"]) >= int(cli_prefix_slack_tokens)
    )
    if cached_ok:
        v = int(meta["max_prefix_tokens"])
        s = int(meta["prefix_slack_tokens"])
        rank0_print(
            f"  {jsonl_path.name}: max_prefix_tokens={v} "
            f"slack={s} (from meta)"
        )
        return v, s
    rank0_print(f"  {jsonl_path.name}: scanning max prefix token length…")
    max_tok, n_bad = scan_jsonl_max_prefix_tokens(jsonl_path, tokenizer)
    if n_bad:
        rank0_print(
            f"  WARNING: {n_bad:,} rows where canonical prefix+sequence != "
            f"training_text; used training_text[:-len(sequence)] for those."
        )
    measured_slack = compute_prefix_slack_tokens(
        jsonl_path, tokenizer, max_seq_len, int(max_tok),
    )
    slack = max(int(cli_prefix_slack_tokens), measured_slack)
    meta["max_prefix_tokens"] = int(max_tok)
    meta["prefix_slack_tokens"] = int(slack)
    meta["max_prefix_tokens_evo_tag"] = evo_model_tag
    meta_path.write_text(json.dumps(meta, indent=2))
    rank0_print(
        f"  {jsonl_path.name}: max_prefix_tokens={max_tok} "
        f"slack={slack} (written to meta; measured_overflow_slack={measured_slack})"
    )
    return int(max_tok), int(slack)


def load_max_prefix_tokens_from_meta(
    jsonl_path: Path, cache_dir: Path, evo_model_tag: str
) -> tuple[int, int]:
    """Return ``(max_prefix_tokens, prefix_slack_tokens)``.

    Missing ``prefix_slack_tokens`` (legacy meta written before audit item
    H6) is treated as 0 with a one-line note, so existing sidecars keep
    working until the next ``--auto-prefix-budget`` scan refreshes them.
    """
    _, meta_path = _lengths_sidecar_paths(cache_dir, jsonl_path)
    meta = json.loads(meta_path.read_text())
    if not _lengths_meta_matches(meta, jsonl_path):
        raise ValueError(f"Stale lengths meta for {jsonl_path}")
    if meta.get("max_prefix_tokens_evo_tag") != evo_model_tag:
        raise ValueError(
            f"{meta_path}: max_prefix_tokens_evo_tag mismatch "
            f"(expected {evo_model_tag!r}). Re-run on rank 0 to refresh."
        )
    if "max_prefix_tokens" not in meta:
        raise KeyError(f"{meta_path} missing max_prefix_tokens; run training on rank 0.")
    max_tok = int(meta["max_prefix_tokens"])
    if "prefix_slack_tokens" not in meta:
        rank0_print(
            f"  {meta_path.name}: legacy meta without prefix_slack_tokens; "
            f"defaulting to 0 (will be added on next --auto-prefix-budget run)."
        )
        slack = 0
    else:
        slack = int(meta["prefix_slack_tokens"])
    return max_tok, slack


def build_nt_chunk_spans(
    seq_len_nt: int,
    max_seq_len: int,
    prefix_token_cap: int,
    chunk_overlap_nt: int,
    slack_tokens: int = 0,
    eos_reserve: int = 0,
) -> list[tuple[int, int]]:
    """Return [(nt_start, nt_end), ...] covering [0, seq_len_nt) in nucleotide space.

    ``slack_tokens`` shrinks ``seq_budget`` so that
    ``len(tokenize(prefix + sub))`` is guaranteed to fit in ``max_seq_len``
    even if the tokenizer ever introduces a seam token (audit item H6).
    For Evo2's CharLevelTokenizer the empirical slack is 0.

    ``eos_reserve`` reserves tokens for the ``EOS_MARKER`` appended to the final
    window (audit M11 / long-seq support), so ``prefix + sub + EOS`` still fits in
    ``max_seq_len``. Reserved for every window (only the last actually uses it),
    which is conservative but negligible (~5 tokens out of 32k).
    """
    seq_budget = max_seq_len - prefix_token_cap - slack_tokens - eos_reserve
    if seq_budget <= 0:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) must exceed prefix_token_cap "
            f"({prefix_token_cap}) + slack_tokens ({slack_tokens})"
        )
    if chunk_overlap_nt >= seq_budget:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap_nt}) must be < sequence budget "
            f"(max_seq_len - prefix_token_cap - slack_tokens) = {seq_budget}"
        )
    stride = seq_budget - chunk_overlap_nt
    if stride <= 0:
        raise ValueError(
            "stride = max_seq_len - prefix_token_cap - slack_tokens - chunk_overlap "
            "must be > 0"
        )
    if seq_len_nt <= seq_budget:
        return [(0, seq_len_nt)]
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + seq_budget, seq_len_nt)
        spans.append((start, end))
        if end >= seq_len_nt:
            break
        start += stride
    return spans


def build_gene_aware_chunk_spans(
    seq_len_nt: int,
    gene_bounds: list | None,
    max_seq_len: int,
    prefix_token_cap: int,
    chunk_overlap_nt: int,
    slack_tokens: int = 0,
    eos_reserve: int = 0,
) -> list[tuple[int, int]]:
    """Like build_nt_chunk_spans but snap each window END to a gene boundary so a
    complete gene (and the modules inside it) is never split across a window edge.

    Greedily packs complete genes into each window up to ``seq_budget``; the window
    ends at the end of the last gene that fits. When a single gene is larger than the
    budget (unavoidable), fall back to an arithmetic cut at ``start + seq_budget``.
    ``gene_bounds`` = [(start, end), ...] nt offsets within the sequence. Falls back
    entirely to arithmetic tiling when ``gene_bounds`` is empty/None.
    """
    seq_budget = max_seq_len - prefix_token_cap - slack_tokens - eos_reserve
    if seq_budget <= 0:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) must exceed prefix_token_cap "
            f"({prefix_token_cap}) + slack_tokens ({slack_tokens})"
        )
    if seq_len_nt <= seq_budget:
        return [(0, seq_len_nt)]
    if not gene_bounds:
        return build_nt_chunk_spans(
            seq_len_nt, max_seq_len, prefix_token_cap, chunk_overlap_nt,
            slack_tokens, eos_reserve,
        )
    genes = sorted((int(g[0]), int(g[1])) for g in gene_bounds)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < seq_len_nt:
        if seq_len_nt - start <= seq_budget:
            spans.append((start, seq_len_nt))
            break
        limit = start + seq_budget
        end = None
        for gs, ge in genes:
            if gs < start:
                continue
            if gs >= limit:
                break            # genes sorted; nothing else fits
            if ge <= limit:
                end = ge         # extend window to this gene's end
        if end is None or end <= start:
            end = limit          # single gene exceeds budget → forced mid-gene cut
        spans.append((start, end))
        # Advance with overlap context; but if the window was shorter than the
        # overlap (small gene, next gene didn't fit), stepping back by the overlap
        # would regress — jump to ``end`` instead so we always advance by >= the
        # window size (avoids a 1-nt crawl / window explosion).
        nxt = end - chunk_overlap_nt
        start = nxt if nxt > start else end
    return spans


def build_all_chunk_indices(
    lengths: np.ndarray,
    max_seq_len: int,
    prefix_token_cap: int,
    chunk_overlap_nt: int,
    slack_tokens: int = 0,
    eos_reserve: int = 0,
    gene_bounds_list: list | None = None,
) -> list[tuple[int, int, int]]:
    """Flat index: (record_idx, nt_start, nt_end). When ``gene_bounds_list`` is
    given (one entry per record, aligned to ``lengths``), snap cuts to gene gaps."""
    chunks: list[tuple[int, int, int]] = []
    for rec_idx, slen in enumerate(lengths.astype(int).tolist()):
        if gene_bounds_list is not None:
            spans = build_gene_aware_chunk_spans(
                slen, gene_bounds_list[rec_idx], max_seq_len, prefix_token_cap,
                chunk_overlap_nt, slack_tokens, eos_reserve,
            )
        else:
            spans = build_nt_chunk_spans(
                slen, max_seq_len, prefix_token_cap, chunk_overlap_nt, slack_tokens,
                eos_reserve,
            )
        for nt0, nt1 in spans:
            chunks.append((rec_idx, nt0, nt1))
    return chunks


class BGCTextDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: Any,
        max_seq_len: int,
        *,
        long_seq_strategy: str = "truncate",
        chunk_overlap: int = 2048,
        prefix_token_cap: int = 256,
        slack_tokens: int = 0,
        lengths_cache_dir: Path | None = None,
        first_window_only: bool = False,
        append_eos: bool = False,
        eos_mode: str = "token",
        continuation_prefix: bool = False,
        gene_aware_chunking: bool = False,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.long_seq_strategy = long_seq_strategy
        self.chunk_overlap = chunk_overlap
        self.prefix_token_cap = prefix_token_cap
        self.slack_tokens = slack_tokens
        self.lengths_cache_dir = lengths_cache_dir
        self.first_window_only = first_window_only
        self.append_eos = append_eos
        self.eos_mode = eos_mode
        self.continuation_prefix = continuation_prefix
        self.gene_aware_chunking = gene_aware_chunking
        # Tokens reserved for the EOS_MARKER on the final window (M11). Computed
        # from the tokenizer so the chunk budget below leaves room for it.
        # Reserve exactly what will be appended: 5 tokens for the string marker, 1 for the real
        # EOS id, 6 for both. Over-reserving silently shortens every training window.
        _marker_toks = count_prefix_tokens(tokenizer, EOS_MARKER)
        self.eos_reserve = 0 if not append_eos else (
            1 if eos_mode == "token" else
            _marker_toks if eos_mode == "marker" else _marker_toks + 1)

        self.offsets: list[int] = []
        with jsonl_path.open("rb") as f:
            offset = 0
            while True:
                line = f.readline()
                if not line:
                    break
                self.offsets.append(offset)
                offset += len(line)
        rank0_print(f"  {jsonl_path.name}: indexed {len(self.offsets):,} records")

        self.chunks: list[tuple[int, int, int]] | None = None

        if long_seq_strategy == "chunk":
            if lengths_cache_dir is None:
                raise ValueError("lengths_cache_dir is required for long_seq_strategy=chunk")
            lengths = load_or_build_lengths_sidecar(jsonl_path, lengths_cache_dir)
            if len(lengths) != len(self.offsets):
                raise ValueError(
                    f"Lengths sidecar count {len(lengths)} != jsonl line count "
                    f"{len(self.offsets)} for {jsonl_path}"
                )
            gene_bounds_list = None
            if gene_aware_chunking:
                gene_bounds_list = []
                with jsonl_path.open() as gbf:
                    for line in gbf:
                        gene_bounds_list.append(json.loads(line).get("core_gene_bounds"))
                if len(gene_bounds_list) != len(lengths):
                    raise ValueError(
                        f"core_gene_bounds count {len(gene_bounds_list)} != lengths "
                        f"{len(lengths)} for {jsonl_path}"
                    )
                n_missing = sum(1 for b in gene_bounds_list if not b)
                rank0_print(
                    f"  {jsonl_path.name}: gene-aware chunking "
                    f"({len(gene_bounds_list) - n_missing:,} with bounds, "
                    f"{n_missing:,} fall back to arithmetic)"
                )
            self.chunks = build_all_chunk_indices(
                lengths, max_seq_len, prefix_token_cap, chunk_overlap, slack_tokens,
                self.eos_reserve, gene_bounds_list=gene_bounds_list,
            )
            if first_window_only:
                # Validation (audit M2): keep only the prefix-aligned first
                # window (nt_start == 0) of each record — exactly one per BGC,
                # matching the inference regime (prefix → start of cluster).
                # Interior windows (nt_start > 0) carry a class/taxon prefix the
                # local sequence does not justify and never occur at inference.
                self.chunks = [c for c in self.chunks if c[1] == 0]
            rank0_print(
                f"  {jsonl_path.name}: chunk mode → {len(self.chunks):,} windows "
                f"(overlap_nt={chunk_overlap}, prefix_token_cap={prefix_token_cap}, "
                f"slack={slack_tokens}"
                f"{', first-window-only' if first_window_only else ''})"
            )
            # H7: the legacy `_assert_prefix_token_cap_on_sample` only
            # checked record 0; the broader `prefix_mask_sanity_check`
            # called from main() (H3) samples several rows and validates
            # prefix < length on each, so the record-0-only assertion has
            # been removed.

    def __len__(self) -> int:
        if self.chunks is not None:
            return len(self.chunks)
        return len(self.offsets)

    def prefix_mask_sanity_check(self, n: int = 4) -> list[dict[str, int]]:
        """Read a few samples and confirm every row has supervised tokens.

        Used at startup (rank 0) to fail fast if H3 prefix masking would
        leave a row with zero unmasked labels. Returns the per-sample
        stats so the caller can log them.
        """
        stats: list[dict[str, int]] = []
        for i in range(min(n, len(self))):
            item = self[i]
            L = int(item["length"])
            p = int(item["prefix_token_count"])
            if p >= L:
                raise ValueError(
                    f"Prefix-mask sanity failed at idx={i}: prefix_token_count="
                    f"{p} >= length={L}; no sequence tokens to train on."
                )
            stats.append({"idx": i, "length": L, "prefix": p, "supervised": L - p})
        return stats

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.chunks is None:
            record_idx = idx
            chunk_idx = idx
        else:
            record_idx, nt_start, nt_end = self.chunks[idx]
            chunk_idx = idx

        with self.jsonl_path.open("rb") as f:
            f.seek(self.offsets[record_idx])
            line = f.readline()
        rec = json.loads(line)
        seq = rec["sequence"]
        training_text = rec["training_text"]

        # Derive the prefix string in both paths so we can mask prefix
        # positions from the loss downstream (H3). `training_prefix_for_chunking`
        # raises if neither canonical assembly nor `training_text[:-len(seq)]`
        # is consistent, which gives the truncate (legacy) path the same
        # integrity check as chunk mode.
        # First-window (class) prefix; also integrity-checks training_text == prefix+seq.
        first_prefix = training_prefix_for_chunking(rec)
        if self.chunks is None:
            # Truncate (legacy): single window = whole record, token-clipped below.
            # EOS is appended; if the record is actually truncated the clip removes
            # it (and the tail), correctly signalling "not the real end".
            nt_start, nt_end = 0, len(seq)
            win_prefix = first_prefix
            _want_marker = self.append_eos and self.eos_mode in ("marker", "both")
            text = training_text + (EOS_MARKER if _want_marker else "")
        else:
            if not training_text.endswith(seq):
                raise ValueError(
                    f"Record {record_idx}: training_text does not end with sequence"
                )
            is_first = (nt_start == 0)
            is_last = (nt_end >= len(seq))  # final window covers the BGC end
            # Interior windows get the distinct continuation prefix (M11) so the
            # class-start token (`|COMPOUND_CLASS:…|`) stays unambiguous.
            if is_first or not self.continuation_prefix:
                win_prefix = first_prefix
            else:
                win_prefix = continuation_phase1_prefix(rec)
            sub = seq[nt_start:nt_end]
            # EOS_MARKER is appended (supervised, after the prefix) only on the
            # window that contains the true end of the BGC (M11 / termination).
            tail = EOS_MARKER if (self.append_eos and is_last
                                  and self.eos_mode in ("marker", "both")) else ""
            text = win_prefix + sub + tail

        tokens = self.tokenizer.tokenize(text)
        ids = list(tokens) if isinstance(tokens, (list, tuple)) else [int(t) for t in tokens]
        ids = [int(t) for t in ids]

        # ── APPEND THE REAL EOS TOKEN (id 0) ───────────────────────────────────────────────
        # Evo2's CharLevelTokenizer maps bytes and appends NOTHING, so the id has to be added by
        # hand -- and it must be added HERE, after tokenisation, because there is no character
        # that encodes to it (ids 0, 1 and 32 all DEtokenize to ' ', but ' ' encodes to 32).
        # Appended AFTER the prefix, so it stays supervised by the masking below; and appended
        # only on the window carrying the record's true end, exactly like the string marker.
        _is_last_window = True if self.chunks is None else (nt_end >= len(seq))
        _want_token = (self.append_eos and _is_last_window
                       and self.eos_mode in ("token", "both"))
        if _want_token:
            ids.append(EOS_TOKEN_ID)

        # Count prefix tokens for label-masking. Relies on Evo2's
        # CharLevelTokenizer where `tokenize(prefix + sub)` is the concatenation
        # of `tokenize(prefix)` and `tokenize(sub)`. `win_prefix` (class on the
        # first window, continuation on interior windows) is what gets masked;
        # the appended EOS is AFTER the prefix, so it stays supervised.
        ptoks = self.tokenizer.tokenize(win_prefix)
        prefix_ids = list(ptoks) if isinstance(ptoks, (list, tuple)) else [int(t) for t in ptoks]
        prefix_token_count = len(prefix_ids)
        # audit B1 (seam guard): the masking is only correct if the full
        # tokenisation starts with exactly the prefix tokens. If the tokenizer
        # ever merges across the prefix↔sequence seam, fail loudly rather than
        # silently mask the wrong span.
        if ids[:prefix_token_count] != prefix_ids:
            raise ValueError(
                f"Record {record_idx} (chunk_idx={idx}): tokenizer merged across the "
                f"prefix↔sequence seam — prefix masking would be wrong "
                f"(prefix_token_count={prefix_token_count}). The label-mask invariant "
                f"is violated; a different masking strategy is required."
            )

        # H6: chunk-mode tiling chooses nt windows so that
        #     len(prefix) + (nt_end - nt_start) + slack <= max_seq_len
        # for any reasonable tokenizer. If we still overflow here something
        # is wrong (stale sidecar meta, tokenizer change, bad input row).
        # Fail loudly rather than silently truncating the tail.
        if len(ids) > self.max_seq_len:
            if self.chunks is not None:
                raise ValueError(
                    f"Record {record_idx} (chunk_idx={idx}, nt_start={nt_start}, "
                    f"nt_end={nt_end}): tokenised length {len(ids)} > max_seq_len "
                    f"{self.max_seq_len}. The sidecar meta or prefix-slack scan is "
                    f"stale; re-run `python scripts/build_chunk_index.py` and / or "
                    f"a fresh `--auto-prefix-budget` scan."
                )
            # Truncate-mode (legacy) keeps the original clip-tail behaviour.
            ids = ids[: self.max_seq_len]

        if prefix_token_count >= len(ids):
            raise ValueError(
                f"Record {record_idx} (chunk_idx={idx}): prefix tokenises to "
                f"{prefix_token_count} tokens, but sample only has {len(ids)} "
                f"tokens; no sequence tokens left to train on. "
                f"Increase --prefix-budget or refresh the auto-prefix-budget scan."
            )

        # Phase-2: per-position annotations for the custom objective, sliced to THIS window and
        # offset past the prefix. Built here because window bounds and record identity are only
        # available at this point; skipped entirely when the objective does not use them.
        pos_w = frame_ph = None
        if getattr(self, "annotation_index", None) is not None:
            from bgc_pipeline.annotations import window_annotations
            a = self.annotation_index.get(record_idx)
            # Cross-check the join rather than trusting it: if the sidecar row does not name the
            # same accession, the sidecar was built from a DIFFERENT split file and every weight
            # would land on the wrong sequence.
            if a is not None and a.get("accession") not in (None, rec.get("accession")):
                raise ValueError(
                    f"Annotation row {record_idx} names accession {a.get('accession')!r} but the "
                    f"training record is {rec.get('accession')!r} — the sidecar does not match "
                    f"this split file.")
            if a is not None:
                pos_w, frame_ph = window_annotations(
                    record_length=a["length"] or len(seq),
                    genes=a["genes"], domain_spans=a["spans"],
                    nt_start=nt_start, nt_end=nt_end,
                    prefix_token_count=prefix_token_count, total_tokens=len(ids),
                    domain_weight=getattr(self, "domain_weight", 1.0),
                )

        out: dict[str, Any] = {
            "position_weights": pos_w,
            "frame_phase": frame_ph,
            "input_ids": ids,
            "length": len(ids),
            "prefix_token_count": prefix_token_count,
            "accession": rec.get("accession", ""),
            "class": rec.get("compound_class", ""),
            "record_idx": record_idx,
            "chunk_idx": idx,
            "nt_start": nt_start if self.chunks is not None else 0,
            "nt_end": nt_end if self.chunks is not None else len(seq),
            "full_seq_len": len(seq),  # full BGC length (for val length-stratification)
        }
        return out


def collate_pad(batch: list[dict[str, Any]],
                pad_id: int = PAD_TOKEN_ID) -> dict[str, Any]:
    # Labels are initialised to IGNORE_INDEX, then only the **sequence**
    # token positions (after the prefix, before the pad) are copied from
    # input_ids. The CE loss in `causal_lm_loss` calls cross_entropy with
    # ignore_index=IGNORE_INDEX, so the model is never penalised for
    # predicting the prefix or the pad — matching the "prompt → generate"
    # training objective. (H3 in the audit plan.)
    content_max = max(b["length"] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, content_max), pad_id, dtype=torch.long)
    labels    = torch.full((B, content_max), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["length"]
        p = int(b["prefix_token_count"])
        ids = torch.tensor(b["input_ids"], dtype=torch.long)
        input_ids[i, :L] = ids
        if p < L:
            labels[i, p:L] = ids[p:]
    # Phase-2 annotations, padded to the same width. Weight 1.0 and phase -1 are the identity:
    # padded tail positions are ignored by the loss and never penalised.
    pos_w = frame_ph = None
    if batch[0].get("position_weights") is not None:
        pos_w = torch.ones((B, content_max), dtype=torch.float32)
        frame_ph = torch.full((B, content_max), -1, dtype=torch.int8)
        for i, b in enumerate(batch):
            L = b["length"]
            if b.get("position_weights") is not None:
                pos_w[i, :L] = b["position_weights"][:L]
                frame_ph[i, :L] = b["frame_phase"][:L]

    first = batch[0]
    return {
        "input_ids":     input_ids,
        "labels":        labels,
        "position_weights": pos_w,
        "frame_phase":   frame_ph,
        "content_max_len": content_max,
        "first_record_idx": int(first["record_idx"]),
        "first_chunk_idx": int(first["chunk_idx"]),
        "first_nt_start": int(first["nt_start"]),
        "first_nt_end": int(first["nt_end"]),
        "first_prefix_token_count": int(first["prefix_token_count"]),
        "full_seq_lens": [int(b["full_seq_len"]) for b in batch],
    }


def collate_pad_to_max(batch: list[dict[str, Any]], max_seq_len: int,
                       pad_id: int = PAD_TOKEN_ID) -> dict[str, Any]:
    """Pad every row to `max_seq_len` tokens (train memory sweeps).

    Dataset items are already truncated to <= max_seq_len. Padding tail
    uses `pad_id`; labels on padded positions are IGNORE_INDEX so loss
    does not train on filler tokens. Prefix positions are likewise
    IGNORE_INDEX (H3 prefix masking).
    """
    content_max = max(b["length"] for b in batch)
    if content_max > max_seq_len:
        raise ValueError(
            f"collate_pad_to_max: content_max={content_max} > max_seq_len={max_seq_len}"
        )
    B = len(batch)
    input_ids = torch.full((B, max_seq_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_seq_len), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["length"]
        p = int(b["prefix_token_count"])
        ids = torch.tensor(b["input_ids"], dtype=torch.long)
        input_ids[i, :L] = ids
        if p < L:
            labels[i, p:L] = ids[p:]
    first = batch[0]
    return {
        "input_ids":       input_ids,
        "labels":          labels,
        "content_max_len": content_max,
        "first_record_idx": int(first["record_idx"]),
        "first_chunk_idx": int(first["chunk_idx"]),
        "first_nt_start": int(first["nt_start"]),
        "first_nt_end": int(first["nt_end"]),
        "first_prefix_token_count": int(first["prefix_token_count"]),
        "full_seq_lens": [int(b["full_seq_len"]) for b in batch],
    }


# ────────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────────

def causal_lm_loss(logits: torch.Tensor,
                   labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


# ────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────

def _length_bucket_label(n: int, bounds: list[int]) -> str:
    prev = 0
    for b in bounds:
        if n <= b:
            return f"<={b // 1000}k"
        prev = b
    return f">{prev // 1000}k"


@torch.no_grad()
def run_validation(engine, val_loader,
                   local_rank: int, max_batches: int,
                   length_bounds: list[int] | None = None) -> dict[str, Any]:
    """Token-weighted validation loss.

    With a first-window val dataset (audit M2) the loss is computed on the
    prefix-aligned start of each held-out BGC — the same regime as inference —
    instead of mislabelled interior chunk windows. Returns the overall loss plus
    a per-length-bucket breakdown (bucket stats are rank-local; the overall loss
    is all-reduced across ranks for the early-stopping signal).
    """
    engine.eval()
    total_loss = 0.0
    n_tokens   = 0
    buckets: dict[str, list[float]] = {}  # label -> [loss_sum, n_tokens]
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
        labels    = batch["labels"].to(local_rank, non_blocking=True)
        logits, _ = engine(input_ids)
        loss = causal_lm_loss(logits, labels)
        n = (labels[..., 1:] != IGNORE_INDEX).sum().item()
        total_loss += loss.item() * n
        n_tokens   += n
        if length_bounds is not None and n > 0:
            fsl = batch.get("full_seq_lens")
            rep = max(fsl) if fsl else int(input_ids.shape[1])
            slot = buckets.setdefault(_length_bucket_label(rep, length_bounds), [0.0, 0])
            slot[0] += loss.item() * n
            slot[1] += n
    if torch.distributed.is_initialized():
        t = torch.tensor([total_loss, n_tokens], dtype=torch.float64,
                         device=f"cuda:{local_rank}")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total_loss, n_tokens = t.tolist()
    engine.train()
    return {
        "overall": total_loss / max(n_tokens, 1),
        "by_length": {
            lab: {"loss": round(s / max(nt, 1), 6), "n_tokens": int(nt)}
            for lab, (s, nt) in sorted(buckets.items())
        },
    }


# ────────────────────────────────────────────────────────────────────────
# DeepSpeed config  (ZeRO-2, same as full FT; optimizer state is tiny now)
# ────────────────────────────────────────────────────────────────────────

def build_ds_config(args: argparse.Namespace, world_size: int,
                    total_steps: int) -> dict[str, Any]:
    effective_batch = args.batch_size * world_size * args.grad_accum
    return {
        "train_batch_size":               effective_batch,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps":    args.grad_accum,
        "gradient_clipping":              args.grad_clip,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage":                2,
            "allgather_partitions": True,
            "reduce_scatter":       True,
            "overlap_comm":         True,
            "contiguous_gradients": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size":    5e8,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr":           args.lr,
                "betas":        [args.beta1, args.beta2],
                "eps":          1e-8,
                "weight_decay": args.weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps":  total_steps,
                "warmup_num_steps": args.warmup_steps,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio":    args.lr_min_ratio,
                "warmup_type":      "linear",
            },
        },
        "steps_per_print":       max(args.log_every, 1),
        "wall_clock_breakdown":  False,
    }


# ────────────────────────────────────────────────────────────────────────
# LoRA application
# ────────────────────────────────────────────────────────────────────────

def apply_lora(model: torch.nn.Module, args: argparse.Namespace,
               ) -> torch.nn.Module:
    """Wrap model with peft LoRA adapters and freeze base weights."""
    from peft import LoraConfig, get_peft_model

    # Verify requested targets exist in the model
    found = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            for t in args.lora_targets:
                if name.endswith(t):
                    found.add(t)
    missing = set(args.lora_targets) - found
    if missing:
        rank0_print(f"  WARNING: LoRA targets not found in model: {missing}")

    # Verify requested target_parameters exist (nn.Parameter suffix match).
    # These reach non-Linear layers (e.g. the Hyena projections TELinear) that
    # target_modules cannot wrap; peft adapts the raw weight tensor in place.
    tparams = list(getattr(args, "lora_target_parameters", []) or [])
    if tparams:
        pfound, pcount = set(), 0
        for pname, _ in model.named_parameters():
            for t in tparams:
                if pname.endswith(t):
                    pfound.add(t); pcount += 1
        pmissing = set(tparams) - pfound
        if pmissing:
            rank0_print(f"  WARNING: LoRA target_parameters not found: {pmissing}")
        rank0_print(f"  target_parameters resolved: {sorted(pfound)} "
                    f"({pcount} tensors matched)")

    # Fix 1: peft's BaseTuner.get_model_config() calls model.config.to_dict().
    # Evo2's vortex dotdict returns None for missing attribute lookups
    # (rather than raising AttributeError), so hasattr() lies and
    # model.config.to_dict is None → 'NoneType' object is not callable.
    # Always inject the shim unconditionally via dict-key assignment.
    try:
        model.config["to_dict"] = lambda: {
            k: v for k, v in model.config.items()
            if not callable(v)  # skip any function/class values
        }
    except Exception:
        pass  # last-resort: if config isn't mutable, peft may still work

    config_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        bias="none",
        inference_mode=False,
        # task_type omitted: CAUSAL_LM adds PeftModelForCausalLM wrapper
        # which expects HF-style attributes Evo2 doesn't have.
    )
    if tparams:
        # peft 0.19: adapt raw nn.Parameters (Hyena projections) that
        # target_modules can't reach. Only one adapter per model may use this.
        config_kwargs["target_parameters"] = tparams
    config = LoraConfig(**config_kwargs)
    # Fix 2: peft 0.19 tries to auto-cast adapters to float8_e8m0fnu,
    # which doesn't exist in torch 2.5. Disable with autocast_adapter_dtype=False.
    peft_model = get_peft_model(model, config, autocast_adapter_dtype=False)

    # Summary
    trainable, total = 0, 0
    for p in peft_model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    rank0_print(f"  LoRA applied: r={args.lora_r}  alpha={args.lora_alpha}  "
                f"dropout={args.lora_dropout}")
    rank0_print(f"  Trainable params: {trainable:,} / {total:,}  "
                f"({100*trainable/total:.3f}%)")
    rank0_print(f"  Target modules:   {args.lora_targets}")
    if tparams:
        rank0_print(f"  Target params:    {tparams}")

    return peft_model


# ────────────────────────────────────────────────────────────────────────
# Block-level activation checkpointing
# ────────────────────────────────────────────────────────────────────────

def _find_stripedhyena(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the underlying StripedHyena instance (the one that owns
    `.blocks`) underneath any peft / DeepSpeed wrappers.

    Walks a small set of well-known wrapper attributes and stops at the
    first module that exposes a `.blocks` attribute. Raises if it can't
    find one — we'd rather fail loudly than silently no-op.
    """
    candidates: list[torch.nn.Module] = [model]
    visited: set[int] = set()
    # Common wrapper attribute paths we may need to drill through:
    #   PeftModel.base_model.model
    #   LoraModel.model
    #   DeepSpeedEngine.module
    while candidates:
        m = candidates.pop(0)
        if id(m) in visited:
            continue
        visited.add(id(m))
        if hasattr(m, "blocks") and isinstance(getattr(m, "blocks"), torch.nn.ModuleList):
            return m
        for attr in ("module", "base_model", "model"):
            if hasattr(m, attr):
                child = getattr(m, attr)
                if isinstance(child, torch.nn.Module):
                    candidates.append(child)
    raise RuntimeError(
        "Could not locate a module with a `.blocks: nn.ModuleList` attribute. "
        "Activation checkpointing requires access to the StripedHyena block list."
    )


def enable_block_activation_checkpointing(model: torch.nn.Module) -> int:
    """Wrap each StripedHyena block's `forward` with
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.

    Why use_reentrant=False:
      - The legacy reentrant variant does not forward keyword arguments
        (block forward takes `padding_mask=` and `inference_params=`).
      - With LoRA dropout active, the recompute on backward must replay
        the same RNG state; the non-reentrant variant snapshots/restores
        RNG correctly.
      - It is also the API direction PyTorch is moving toward.

    Returns the number of blocks wrapped.

    Note: we override `forward` on the block instance. `nn.Module.__call__`
    looks up `self.forward` via normal attribute resolution, so the instance
    attribute shadows the class method. peft only rewrites inner Linear
    modules and never touches the block list, so this stays valid through
    LoRA application. DeepSpeed's wrapping happens above this layer and
    does not interpose on per-block forward.
    """
    from torch.utils.checkpoint import checkpoint

    sh = _find_stripedhyena(model)
    n_wrapped = 0
    for block in sh.blocks:
        original_forward = block.forward  # bound method on this block instance

        def make_checkpointed(orig):
            def checkpointed_forward(*args, **kwargs):
                return checkpoint(orig, *args, use_reentrant=False, **kwargs)
            return checkpointed_forward

        block.forward = make_checkpointed(original_forward)
        n_wrapped += 1
    return n_wrapped


# ────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (LoRA-specific: save adapter + DS state separately)
# ────────────────────────────────────────────────────────────────────────

def gather_rng_state(device: torch.device | None = None) -> dict[str, Any]:
    """Snapshot Python / NumPy / Torch (CPU + CUDA) RNG state.

    Returned dict is per-rank: each rank has its own RNG sequence in
    distributed training. Serialised via DeepSpeed's ``client_state``
    (which is per-rank), so RNG sequences are restored exactly on resume.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state(device) if device is not None
            else torch.cuda.get_rng_state()
        )
    return state


def set_rng_state(state: dict[str, Any] | None,
                  device: torch.device | None = None) -> bool:
    """Restore RNG state captured by :func:`gather_rng_state`.

    Best-effort: missing or unparseable fields are skipped with a warning
    so a corrupt RNG snapshot can never block resume. Returns True if at
    least one component was restored.
    """
    if not state:
        return False
    restored = False
    try:
        if state.get("python") is not None:
            random.setstate(state["python"])
            restored = True
    except Exception as e:
        rank0_print(f"  WARNING: failed to restore Python RNG: {e}")
    try:
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
            restored = True
    except Exception as e:
        rank0_print(f"  WARNING: failed to restore NumPy RNG: {e}")
    try:
        if state.get("torch_cpu") is not None:
            torch.set_rng_state(state["torch_cpu"])
            restored = True
    except Exception as e:
        rank0_print(f"  WARNING: failed to restore torch CPU RNG: {e}")
    try:
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            if device is not None:
                torch.cuda.set_rng_state(state["torch_cuda"], device)
            else:
                torch.cuda.set_rng_state(state["torch_cuda"])
            restored = True
    except Exception as e:
        rank0_print(f"  WARNING: failed to restore torch CUDA RNG: {e}")
    return restored


def make_client_state(
    *,
    step: int,
    best_val_loss: float,
    epoch: int,
    micro_step_in_epoch: int,
    grad_accum: int = 1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Build the dict persisted as ``client_state`` by DeepSpeed.

    Includes everything needed for a faithful (data-stream + RNG) resume
    (audit H1). Capture is wrapped in try/except so a transient RNG-gather
    failure never blocks a checkpoint write (e.g. during the OOM
    emergency-save path).

    audit M1: ``micro_step_in_epoch`` is snapped DOWN to the last completed
    grad-accumulation boundary. ``step`` (DeepSpeed ``global_steps``) only
    advances at accumulation boundaries, so a checkpoint written mid-accumulation
    (OOM / interrupted / final saves) would otherwise persist a micro_step that
    is inconsistent with ``step`` — on resume the partially-accumulated
    micro-batches (whose gradients were never applied) would be skipped entirely.
    Aligning makes resume re-process them. Periodic checkpoints already land on a
    boundary, so this is a no-op for them.
    """
    ga = max(int(grad_accum), 1)
    aligned_micro = (int(micro_step_in_epoch) // ga) * ga
    cs: dict[str, Any] = {
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "epoch": int(epoch),
        "micro_step_in_epoch": aligned_micro,
    }
    try:
        cs["rng_state"] = gather_rng_state(device)
    except Exception as e:
        rank0_print(f"  WARNING: gather_rng_state failed ({e}); "
                    f"resume will fall back to seed-only RNG")
    return cs


def save_lora_checkpoint(model_engine, args: argparse.Namespace,
                         ckpt_root: Path, tag: str,
                         client_state: dict) -> None:
    """Save both the peft adapter (human-readable) and the DeepSpeed
    optimizer/scheduler state (needed for exact resume)."""
    ckpt_dir = ckpt_root / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. DeepSpeed checkpoint — saves scheduler + client_state and ZeRO-2
    #    optimizer shards for the LoRA-trainable params only. Frozen base
    #    model weights are intentionally excluded: they are already on disk
    #    in the HF cache (evo2_7b.pt, ~14 GB) and are reloaded on every
    #    run via Evo2("evo2_7b"). Re-serialising them into
    #    mp_rank_00_model_states.pt would add ~25 GB per checkpoint for
    #    no resume-time benefit. See docs/gputee/FINETUNE_GUIDE.md §11.
    model_engine.save_checkpoint(str(ckpt_root), tag=tag,
                                 client_state=client_state,
                                 exclude_frozen_parameters=True)

    # 2. peft adapter weights — rank 0 only (ZeRO-2 keeps full weights on each rank)
    if is_main():
        adapter_dir = ckpt_dir / "adapter"
        # model_engine.module is the peft model
        model_engine.module.save_pretrained(str(adapter_dir))
        rank0_print(f"  Adapter saved → {adapter_dir}")


def load_lora_checkpoint(
    model_engine, resume_from: Path, device: torch.device | None = None,
) -> dict[str, Any]:
    """Load DeepSpeed optimizer state for resume and restore RNG.

    Returns a dict with ``step``, ``best_val_loss``, ``epoch``,
    ``micro_step_in_epoch``. RNG state, if present, is restored in-place
    so the data stream is reproducible from the checkpoint forward
    (audit H1).

    The checkpoint was saved with ``exclude_frozen_parameters=True``, so
    ``mp_rank_00_model_states.pt`` only contains LoRA-trainable params.
    The frozen base-model weights are already loaded from the HF cache,
    and the LoRA adapter weights are already restored via
    ``PeftModel.from_pretrained``. We pass ``load_module_strict=False``
    so DeepSpeed doesn't error on the missing frozen-param keys — the
    optimizer state and client_state (step / best_val_loss / epoch /
    micro_step / RNG) are the only things we need from this checkpoint.

    Legacy checkpoints (saved before H1) that lack ``epoch`` /
    ``micro_step_in_epoch`` / ``rng_state`` keys load with epoch=0,
    micro_step_in_epoch=0, and seed-only RNG. ``step`` still applies, so
    the optimizer / LR schedule pick up where they left off; only the
    data-stream-from-resume invariant is weaker for legacy resumes.
    """
    _, client_state = model_engine.load_checkpoint(
        str(resume_from.parent), tag=resume_from.name,
        load_module_strict=False,
    )
    if client_state is None:
        client_state = {}
    out: dict[str, Any] = {
        "step": int(client_state.get("step", 0)),
        "best_val_loss": float(client_state.get("best_val_loss", float("inf"))),
        "epoch": int(client_state.get("epoch", 0)),
        "micro_step_in_epoch": int(client_state.get("micro_step_in_epoch", 0)),
    }
    rng_state = client_state.get("rng_state")
    if rng_state:
        restored = set_rng_state(rng_state, device)
        if restored:
            rank0_print("  RNG state restored from checkpoint")
        else:
            rank0_print("  WARNING: checkpoint had rng_state but none could be applied")
    else:
        rank0_print(
            "  Note: checkpoint has no rng_state (legacy pre-H1 ckpt). "
            "Data stream will be reproduced from seed only."
        )
    return out


# ────────────────────────────────────────────────────────────────────────
# Plot generation  (identical to finetune_evo2.py)
# ────────────────────────────────────────────────────────────────────────

def generate_plots(output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    train_log = output_dir / "train_log.jsonl"
    val_log   = output_dir / "val_log.jsonl"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    if not train_log.exists():
        return

    train = [json.loads(l) for l in train_log.open()]
    val   = [json.loads(l) for l in val_log.open()] if val_log.exists() else []
    if not train:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([e["step"] for e in train], [e["train_loss"] for e in train],
            label="train", alpha=0.8, linewidth=1)
    if val:
        ax.plot([e["step"] for e in val], [e["val_loss"] for e in val],
                label="val", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training / Validation Loss (LoRA)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots_dir / "loss.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot([e["step"] for e in train], [e["lr"] for e in train], color="C2")
    ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.set_title("LR Schedule")
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots_dir / "lr.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot([e["step"] for e in train], [e["grad_norm"] for e in train],
            color="C3", alpha=0.7, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("grad norm"); ax.set_title("Gradient Norm")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots_dir / "grad_norm.png", dpi=100); plt.close(fig)

    tps = [e.get("tokens_per_sec", 0) for e in train]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot([e["step"] for e in train], tps, color="C4", alpha=0.7, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("tokens/sec"); ax.set_title("Throughput")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots_dir / "throughput.png", dpi=100); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax = axes[0, 0]
    ax.plot([e["step"] for e in train], [e["train_loss"] for e in train],
            label="train", alpha=0.6, linewidth=1)
    if val:
        ax.plot([e["step"] for e in val], [e["val_loss"] for e in val],
                label="val", linewidth=2, marker="o", markersize=4)
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("step")
    axes[0,1].plot([e["step"] for e in train], [e["lr"] for e in train], color="C2")
    axes[0,1].set_title("LR (log)"); axes[0,1].set_yscale("log"); axes[0,1].grid(alpha=0.3)
    axes[0,1].set_xlabel("step")
    axes[1,0].plot([e["step"] for e in train], [e["grad_norm"] for e in train],
                   color="C3", alpha=0.6, linewidth=1)
    axes[1,0].set_title("Gradient norm"); axes[1,0].grid(alpha=0.3); axes[1,0].set_xlabel("step")
    axes[1,1].plot([e["step"] for e in train], tps, color="C4", alpha=0.6, linewidth=1)
    axes[1,1].set_title("Throughput (tok/s)"); axes[1,1].grid(alpha=0.3); axes[1,1].set_xlabel("step")
    fig.suptitle(f"LoRA training — {output_dir.name}", y=0.995)
    fig.tight_layout(); fig.savefig(plots_dir / "summary.png", dpi=110); plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Logging helpers
# ────────────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def gpu_memory_gb() -> list[float]:
    return [torch.cuda.max_memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())]

# Only "step_<digits>" (pure numeric) checkpoints are eligible for
# cleanup. Special-purpose checkpoints — `best/`, `step_N_interrupted/`,
# `step_N_oom/`, `step_N_final/` — are deliberately preserved (audit C5).
_STEP_CKPT_RE = re.compile(r"^step_(\d+)$")
_SPECIAL_CKPT_RE = re.compile(r"^step_(\d+)_(oom|interrupted|final)$")


def cleanup_old_checkpoints(ckpt_root: Path, keep_last: int,
                            keep_special: int = 2) -> None:
    """Rotate old checkpoints.

    - Pure ``step_N`` checkpoints: keep the newest ``keep_last`` (by step).
    - Emergency ``step_N_{oom,interrupted,final}`` checkpoints: keep the newest
      ``keep_special`` (by step) and delete older ones (audit m3 — these are
      written on every preemption under C6 auto-resume and otherwise accumulate
      unbounded). ``keep_special <= 0`` disables their rotation (keep all).
    - ``best/`` is ALWAYS preserved.
    """
    if not ckpt_root.exists():
        return
    if keep_last <= 0:
        rank0_print(
            f"  cleanup_old_checkpoints: keep_last={keep_last} <= 0; "
            f"refusing to delete (would wipe every periodic checkpoint)."
        )
        return
    numeric: list[tuple[int, Path]] = []
    special: list[tuple[int, Path]] = []
    for p in ckpt_root.iterdir():
        if not p.is_dir():
            continue
        if _STEP_CKPT_RE.match(p.name):
            numeric.append((int(_STEP_CKPT_RE.match(p.name).group(1)), p))
        elif _SPECIAL_CKPT_RE.match(p.name):
            special.append((int(_SPECIAL_CKPT_RE.match(p.name).group(1)), p))
        # `best/` and anything unrecognised are left untouched.
    numeric.sort(key=lambda t: t[0])
    special.sort(key=lambda t: t[0])

    stale: list[Path] = list(p for _, p in numeric[:-keep_last])
    if keep_special > 0 and len(special) > keep_special:
        stale += [p for _, p in special[:-keep_special]]
    for path in stale:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            rank0_print(f"  cleanup: failed to remove {path}: {e}")
    if stale:
        rank0_print(f"  cleanup: removed {len(stale)} old checkpoint(s); "
                    f"kept newest {keep_last} step_N + {keep_special} emergency + best/")


# ────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ────────────────────────────────────────────────────────────────────────

_SHOULD_STOP = False

def _set_stop_flag(signum, frame):
    global _SHOULD_STOP
    _SHOULD_STOP = True
    rank0_print(f"Signal {signum} received; will checkpoint after current step")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Per-rank GPU masking (see FINETUNE_GUIDE.md §12.1 for explanation) ──
    # Evo2's StripedHyena loader auto-shards layers across all visible CUDA
    # devices. Restrict each worker to its own GPU before any CUDA init.
    local_rank_env = ddp_local_rank()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank_env)
    os.environ["LOCAL_RANK"] = "0"   # DS reads this for device_id in init_distributed
    local_rank = 0
    args.local_rank = 0              # DS sanity-checks args.local_rank == env LOCAL_RANK

    seed_everything(args.seed + ddp_rank())

    import deepspeed
    import evo2

    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    # Run setup
    if args.run_name is None:
        args.run_name = f"{args.output_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)

    if is_main():
        save_config(args, args.output_dir)
        save_data_fingerprint(args, args.output_dir)
        save_env(args.output_dir)
        rank0_print(f"Output dir: {args.output_dir}")
        rank0_print(f"Run name:   {args.run_name}")
        rank0_print(f"World size: {ddp_world_size()}  (RANK={ddp_rank()}  LOCAL_RANK={local_rank})")
        rank0_print(f"Mode:       LoRA  r={args.lora_r}  alpha={args.lora_alpha}")

    # WandB
    wandb_run = None
    if is_main() and args.wandb_project:
        import wandb
        # Try the requested mode; if online init fails (no network/login on a
        # long shared-GPU run), fall back to offline so logs are still captured
        # locally rather than lost. Never let WandB block training.
        modes = [args.wandb_mode]
        if args.wandb_mode == "online":
            modes.append("offline")
        for mode in modes:
            try:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=args.run_name,
                    config=vars(args),
                    mode=mode,
                    resume="allow",
                    dir=str(args.output_dir),
                    tags=["lora", f"r{args.lora_r}"],
                )
                if mode != args.wandb_mode:
                    rank0_print(f"WandB: '{args.wandb_mode}' init failed; fell back to '{mode}'")
                rank0_print(f"WandB: {wandb_run.url}")
                break
            except Exception as e:
                rank0_print(f"WandB init ({mode}) failed: {e}")
                wandb_run = None

    signal.signal(signal.SIGTERM, _set_stop_flag)
    signal.signal(signal.SIGINT,  _set_stop_flag)

    # ── Load Evo2 base model ──────────────────────────────────────────
    rank0_print("Loading Evo2 7B base model + tokenizer...")
    t0 = time.time()
    evo_wrapper = evo2.Evo2(EVO2_MODEL_NAME)
    model     = evo_wrapper.model
    tokenizer = evo_wrapper.tokenizer
    rank0_print(f"  Loaded in {time.time()-t0:.0f} s")
    rank0_print(f"  Base params: {sum(p.numel() for p in model.parameters()):,}")

    # M4: pad-token vs nucleotide-token collision check.
    assert_pad_token_safe(tokenizer)

    # M7: fail loud if --resume-from is set but the directory or its
    # `adapter/` subdir is missing. The old behaviour silently fell
    # through to a fresh-adapter init, which trained from scratch while
    # the user thought they were resuming.
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(
                f"--resume-from {args.resume_from} does not exist; refusing "
                f"to silently train from scratch."
            )
        adapter_dir = args.resume_from / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"--resume-from {args.resume_from} has no adapter/ subdir; "
                f"checkpoint is incomplete or wrong path. Pass the directory "
                f"that contains both `adapter/` and `mp_rank_00_model_states.pt`."
            )

    # Fix non-contiguous Wqkv tensors (FINETUNE_GUIDE.md §12.1 Bug 2)
    # AND: clone any inference-mode tensors so backward can save them.
    # Evo2 config sets inference_mode=True, which tags some tensors (e.g.
    # RMSNorm scale) as inference tensors. Autograd cannot save these for
    # backward, raising "Inference tensors cannot be saved for backward."
    # clone() produces a normal (non-inference-mode) tensor.
    n_fixed = 0
    n_cloned = 0
    with torch.no_grad():
        for p in model.parameters():
            needs_clone = hasattr(p.data, "is_inference") and p.data.is_inference()
            if not p.is_contiguous() or needs_clone:
                p.data = p.data.clone().contiguous()
                n_fixed += 1
                if needs_clone:
                    n_cloned += 1
        # Buffers: need to replace via the owning module
        for mod_name, mod in model.named_modules():
            for buf_name, buf in list(mod.named_buffers(recurse=False)):
                if buf is None:
                    continue
                needs_clone = hasattr(buf, "is_inference") and buf.is_inference()
                if (not buf.is_contiguous()) or needs_clone:
                    setattr(mod, buf_name, buf.clone().contiguous())
                    n_fixed += 1
                    if needs_clone:
                        n_cloned += 1
    rank0_print(f"  Fixed {n_fixed} tensors ({n_cloned} cloned out of inference mode)")
    # Ensure the model is in training mode (Evo2 loads in eval/inference mode)
    model.train()

    # ── Apply LoRA ────────────────────────────────────────────────────
    # Load adapter from checkpoint if resuming, otherwise apply fresh config.
    if args.resume_from is not None and (args.resume_from / "adapter").exists():
        from peft import PeftModel
        # Fix: peft's BaseTuner.get_model_config() calls model.config.to_dict().
        # Evo2's vortex dotdict returns None for missing attribute lookups
        # (same shim as apply_lora Fix 1 — needed on the from_pretrained path too).
        try:
            model.config["to_dict"] = lambda: {
                k: v for k, v in model.config.items()
                if not callable(v)
            }
        except Exception:
            pass
        # Fix 3: peft 0.19's set_peft_model_state_dict tries to import
        # transformers.integrations.tensor_parallel (ALL_PARALLEL_STYLES,
        # ColwiseParallel, EmbeddingParallel, RowwiseParallel), which only
        # exists in transformers >= 4.50. We're pinned to 4.46.3. Provide
        # a stub module with dummy names so the import doesn't crash. The
        # TP sharding logic never fires in our single-GPU setup — it only
        # activates when layers have `_hf_tp_plan` / `_hf_device_mesh`.
        import types, sys as _sys
        _tp_mod_name = "transformers.integrations.tensor_parallel"
        if _tp_mod_name not in _sys.modules:
            _stub = types.ModuleType(_tp_mod_name)
            _stub.ALL_PARALLEL_STYLES = {}
            _stub.ColwiseParallel = None
            _stub.EmbeddingParallel = None
            _stub.RowwiseParallel = None
            _stub.gather_state_dict_for_save = None
            _sys.modules[_tp_mod_name] = _stub

        rank0_print(f"  Loading LoRA adapter from {args.resume_from / 'adapter'}")
        # Fix 2 (resume path): peft 0.19 tries to auto-cast adapters to
        # float8_e8m0fnu, which doesn't exist in torch 2.5.
        model = PeftModel.from_pretrained(
            model, str(args.resume_from / "adapter"), is_trainable=True,
            autocast_adapter_dtype=False,
        )
        rank0_print("  Adapter loaded (weights restored)")
    else:
        model = apply_lora(model, args)

    # ── Optional block-level activation checkpointing ─────────────────
    # See docs/gputee/FINETUNE_GUIDE.md §12.7. Default-on; opt out via
    # `--no-activation-checkpointing`.
    if args.activation_checkpointing:
        n_wrapped = enable_block_activation_checkpointing(model)
        rank0_print(
            f"  Activation checkpointing ENABLED: wrapped {n_wrapped} blocks "
            f"(use_reentrant=False)"
        )
    else:
        rank0_print("  Activation checkpointing: disabled")

    # ── Datasets ──────────────────────────────────────────────────────
    cache_dir = (
        args.lengths_cache_dir
        if args.lengths_cache_dir is not None
        else args.train.parent
    )
    if args.long_seq_strategy == "chunk":
        if is_main():
            rank0_print(
                "Ensuring nucleotide length sidecars for chunk mode "
                f"(cache dir: {cache_dir})..."
            )
            load_or_build_lengths_sidecar(args.train, cache_dir)
            load_or_build_lengths_sidecar(args.val, cache_dir)
            if args.auto_prefix_budget:
                ensure_max_prefix_tokens_in_meta(
                    args.train, cache_dir, tokenizer, EVO2_MODEL_NAME,
                    args.max_seq_len, args.prefix_slack_tokens,
                )
                ensure_max_prefix_tokens_in_meta(
                    args.val, cache_dir, tokenizer, EVO2_MODEL_NAME,
                    args.max_seq_len, args.prefix_slack_tokens,
                )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    rank0_print("Indexing datasets...")
    train_cap = args.prefix_budget
    val_cap = args.prefix_budget
    train_slack = args.prefix_slack_tokens
    val_slack = args.prefix_slack_tokens
    if args.long_seq_strategy == "chunk" and args.auto_prefix_budget:
        train_cap, train_slack = load_max_prefix_tokens_from_meta(
            args.train, cache_dir, EVO2_MODEL_NAME)
        val_cap, val_slack = load_max_prefix_tokens_from_meta(
            args.val, cache_dir, EVO2_MODEL_NAME)
    elif args.long_seq_strategy == "chunk" and is_main():
        rank0_print(
            f"  Chunk mode: fixed prefix_token_cap={args.prefix_budget} "
            f"slack={args.prefix_slack_tokens} (--no-auto-prefix-budget)"
        )

    ds_common: dict[str, Any] = {
        "long_seq_strategy": args.long_seq_strategy,
        "chunk_overlap": args.chunk_overlap,
        "lengths_cache_dir": cache_dir if args.long_seq_strategy == "chunk" else None,
        "append_eos": args.eos_token,
        "eos_mode": args.eos_mode,
        "continuation_prefix": args.continuation_prefix,
        "gene_aware_chunking": args.gene_aware_chunking,
    }
    train_ds = BGCTextDataset(
        args.train,
        tokenizer,
        args.max_seq_len,
        prefix_token_cap=train_cap,
        slack_tokens=train_slack,
        **ds_common,
    )
    val_ds = BGCTextDataset(
        args.val,
        tokenizer,
        args.max_seq_len,
        prefix_token_cap=val_cap,
        slack_tokens=val_slack,
        first_window_only=True,  # audit M2: validate on prefix-aligned starts only
        **ds_common,
    )

    # Phase-2: load the annotation sidecar onto the TRAIN dataset only. Validation keeps the
    # plain objective so val loss stays comparable across cells of the factorial — a weighted val
    # loss would move for reasons unrelated to model quality and make the arms incomparable.
    if args.annotations is not None:
        from bgc_pipeline.annotations import load_annotation_index
        _idx = load_annotation_index(args.annotations)
        train_ds.annotation_index = _idx
        train_ds.domain_weight = args.domain_weight
        _hit = sum(1 for a in list(_idx)[:0]) if False else None
        if is_main():
            rank0_print(f"  objective: domain_weight={args.domain_weight} "
                        f"frame_lambda={args.frame_lambda} "
                        f"annotations={len(_idx):,} records from {args.annotations.name}")
            # COVERAGE CHECK: annotations keyed by accession must actually match the training
            # records, or the objective silently degrades to the baseline on the misses.
            import json as _json
            _acc = []
            with args.train.open() as _f:
                for _i, _l in enumerate(_f):
                    if _i >= 200:
                        break
                    _acc.append((_i, _json.loads(_l).get("accession")))
            _cov = sum(1 for _i, a in _acc
                       if _i in _idx and _idx[_i].get("accession") in (None, a)) / max(len(_acc), 1)
            rank0_print(f"  objective: annotation coverage on the first {len(_acc)} train "
                        f"records = {_cov:.1%}")
            if _cov < 0.9:
                raise SystemExit(
                    f"ABORT: only {_cov:.1%} of training records have annotations. The objective "
                    f"would be the baseline on the rest, making the arm a blend of treatment and "
                    f"control. Rebuild the sidecar for THIS split.")

    # H3 prefix-mask sanity: confirm a few rows of each split have >0
    # supervised (non-prefix, non-pad) tokens. Cheap rank-0-only check.
    if is_main():
        for split_name, ds in (("train", train_ds), ("val", val_ds)):
            stats = ds.prefix_mask_sanity_check(n=4)
            for s in stats:
                rank0_print(
                    f"  prefix-mask {split_name}: idx={s['idx']} "
                    f"length={s['length']} prefix={s['prefix']} "
                    f"supervised={s['supervised']}"
                )

    world_size    = ddp_world_size()
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=ddp_rank(), shuffle=True, seed=args.seed)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                       rank=ddp_rank(), shuffle=False)

    if args.smoke_pad_to_max_seq_len:
        train_collate = partial(collate_pad_to_max, max_seq_len=args.max_seq_len)
        rank0_print(
            f"  Train collation: pad every micro-batch to --max-seq-len={args.max_seq_len} "
            f"(smoke memory sweep; val still natural-length)"
        )
    else:
        train_collate = collate_pad

    _loader_gen = torch.Generator()
    _loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=train_collate, num_workers=2, pin_memory=True, drop_last=True,
        worker_init_fn=_seed_dataloader_worker, generator=_loader_gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=collate_pad, num_workers=2, pin_memory=True, drop_last=False,
        worker_init_fn=_seed_dataloader_worker, generator=_loader_gen,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps     = steps_per_epoch * args.max_epochs
    rank0_print(f"Train batches/rank: {len(train_loader):,}")
    rank0_print(f"Steps per epoch:    {steps_per_epoch:,}  (grad_accum={args.grad_accum})")
    rank0_print(f"Total steps:        {total_steps:,}  (epochs={args.max_epochs})")

    # audit A4: the cosine LR horizon is `total_steps`. The restored optimizer
    # advances along the schedule originally created with the run's total_steps;
    # if a resume recomputes a DIFFERENT total_steps (e.g. different data,
    # --max-epochs, or grad-accum), the schedule won't line up. Persist it on the
    # first run and warn loudly on a resume mismatch.
    if is_main():
        sched_path = args.output_dir / "schedule.json"
        sched_now = {"total_steps": total_steps, "warmup_steps": args.warmup_steps,
                     "steps_per_epoch": steps_per_epoch, "max_epochs": args.max_epochs,
                     "grad_accum": args.grad_accum}
        if args.resume_from is not None and sched_path.exists():
            try:
                prev = json.loads(sched_path.read_text())
            except Exception:
                prev = {}
            if prev.get("total_steps") not in (None, total_steps):
                rank0_print(
                    f"  ⚠️  LR-SCHEDULE HORIZON CHANGED on resume: total_num_steps "
                    f"{prev.get('total_steps')} → {total_steps} (steps/epoch "
                    f"{prev.get('steps_per_epoch')} → {steps_per_epoch}). The restored "
                    f"cosine schedule will not align with this horizon; LR will be wrong. "
                    f"Resume with the SAME data / --max-epochs / --grad-accum as run start."
                )
        else:
            sched_path.write_text(json.dumps(sched_now, indent=2))

    # ── DeepSpeed init ────────────────────────────────────────────────
    # Only trainable (adapter) parameters are passed — frozen base params
    # are excluded, so ZeRO-2 only shards the ~28 M adapter gradients/states.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    rank0_print(f"Trainable tensors passed to DeepSpeed: {len(trainable_params)}")

    ds_config = build_ds_config(args, world_size, total_steps)
    if is_main():
        (args.output_dir / "deepspeed_config.json").write_text(
            json.dumps(ds_config, indent=2))

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=trainable_params,
        config=ds_config,
    )

    # ── Resume optimizer state ────────────────────────────────────────
    start_step    = 0
    best_val_loss = float("inf")
    start_epoch   = 0
    skip_micro    = 0
    cuda_device   = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
    if args.resume_from is not None and args.resume_from.exists():
        rank0_print(f"Loading DeepSpeed checkpoint from {args.resume_from}")
        resume_state = load_lora_checkpoint(
            model_engine, args.resume_from, device=cuda_device,
        )
        start_step    = resume_state["step"]
        best_val_loss = resume_state["best_val_loss"]
        start_epoch   = resume_state["epoch"]
        skip_micro    = resume_state["micro_step_in_epoch"]
        # If the saved micro_step_in_epoch landed past the end of the
        # epoch (e.g. checkpoint written at the boundary), roll over to
        # the next epoch with no skip.
        if skip_micro >= len(train_loader):
            start_epoch += 1
            skip_micro = 0
        rank0_print(
            f"  Resumed at step {start_step} epoch={start_epoch} "
            f"micro_step_in_epoch={skip_micro} best_val_loss={best_val_loss:.4f}"
        )

    # ── Training loop ─────────────────────────────────────────────────
    rank0_print(f"Starting LoRA training from step {start_step}")
    model_engine.train()

    step             = start_step
    epoch            = start_epoch
    micro_step       = max(skip_micro - 1, 0)
    log_buffer_loss  = 0.0
    log_buffer_count = 0
    last_log_time    = time.time()
    last_log_tokens  = 0
    evals_no_improve  = 0       # consecutive validations with no improvement (M7)
    should_early_stop = False

    train_log_path = args.output_dir / "train_log.jsonl"
    val_log_path   = args.output_dir / "val_log.jsonl"
    ckpt_root      = args.output_dir / "checkpoints"

    try:
        for epoch in range(start_epoch, args.max_epochs):
            train_sampler.set_epoch(epoch)
            rank0_print(f"── Epoch {epoch+1}/{args.max_epochs} ──")

            # H1: faithful resume. For the first epoch after resume, skip
            # forward through the dataloader by the same number of
            # micro-batches we'd already consumed when the checkpoint was
            # written. Skipping still triggers dataset reads + tokenisation
            # (no fast-forward in DistributedSampler), but it's bounded by
            # `save_every * grad_accum` micro-batches per resume, which is
            # acceptable next to the cost of an unfaithful data stream.
            this_epoch_skip = skip_micro if epoch == start_epoch else 0
            micro_iter = enumerate(train_loader)
            if this_epoch_skip > 0:
                rank0_print(
                    f"  Resume skip-ahead: advancing "
                    f"{this_epoch_skip:,} micro-batches in epoch {epoch+1}"
                )
                t_skip = time.time()
                advanced = 0
                for _ in micro_iter:
                    advanced += 1
                    if advanced >= this_epoch_skip:
                        break
                rank0_print(
                    f"  Skip complete ({advanced:,} micro-batches in "
                    f"{time.time() - t_skip:.1f}s); resuming at "
                    f"micro_step={this_epoch_skip}"
                )

            for micro_step, batch in micro_iter:
                content_max_len = batch.pop("content_max_len", None)
                first_record_idx = batch.pop("first_record_idx", None)
                first_chunk_idx = batch.pop("first_chunk_idx", None)
                first_nt_start = batch.pop("first_nt_start", None)
                first_nt_end = batch.pop("first_nt_end", None)
                first_prefix_token_count = batch.pop(
                    "first_prefix_token_count", None
                )
                input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
                labels    = batch["labels"].to(local_rank, non_blocking=True)
                collated_seq_len = int(input_ids.shape[1])

                logits, _ = model_engine(input_ids)
                # CUSTOM OBJECTIVE (Phase 2). With --domain-weight 1.0 and --frame-lambda 0.0
                # this is bit-identical to causal_lm_loss — the factorial's control cell must be
                # the current objective exactly, or no delta is attributable. Pinned by
                # tests/test_objective.py.
                pw = batch.get("position_weights")
                fp = batch.get("frame_phase")
                loss, loss_comp = custom_lm_loss(
                    logits, labels,
                    input_ids=input_ids if args.frame_lambda > 0 else None,
                    position_weights=pw.to(local_rank) if pw is not None else None,
                    frame_phase=fp.to(local_rank) if fp is not None else None,
                    frame_lambda=args.frame_lambda,
                )

                model_engine.backward(loss)
                model_engine.step()

                # Average loss across ranks so logged train_loss reflects
                # the true mean over all GPUs' data shards (not rank-0 only).
                loss_scalar = loss.detach()
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(loss_scalar,
                                                 op=torch.distributed.ReduceOp.SUM)
                    loss_scalar = loss_scalar / world_size
                log_buffer_loss  += loss_scalar.item()
                log_buffer_count += 1
                last_log_tokens  += input_ids.numel() * world_size

                # DS's `global_steps` is the authoritative optimizer-step counter:
                # it is incremented exactly once per accumulation boundary, inside
                # `_take_model_step`, AFTER optimizer.step() and scheduler.step().
                # DO NOT use `is_gradient_accumulation_boundary()` here — that check
                # returns True one micro-batch EARLY when called after engine.step()
                # (formula: (micro_steps + 1) % ga == 0, designed for pre-step() use).
                # At ga=1 the two coincide (always True), but at ga>1 it would cause
                # logged grad_norm/lr/checkpoints to reflect the previous opt step.
                if model_engine.global_steps != step:
                    step = model_engine.global_steps

                    # ── Logging
                    if step % args.log_every == 0:
                        avg_loss  = log_buffer_loss / max(log_buffer_count, 1)
                        lr        = scheduler.get_last_lr()[0] if scheduler else args.lr
                        grad_norm = float(model_engine.get_global_grad_norm() or 0.0)
                        elapsed   = time.time() - last_log_time
                        tps       = last_log_tokens / max(elapsed, 1e-3)

                        if is_main():
                            entry = {
                                "step":           step,
                                "epoch":          round(step / max(steps_per_epoch,1), 4),
                                "train_loss":     round(avg_loss, 6),
                                # Log the objective's COMPONENTS, not just the total. A stop
                                # penalty that is silently zero (bad annotations, wrong phase) or
                                # one that swamps the CE is invisible in a single number — and
                                # either would make the arm uninterpretable.
                                "loss_ce":        round(loss_comp.get("ce", float("nan")), 6),
                                "loss_stop_pen":  round(loss_comp.get("stop_penalty", 0.0), 6),
                                "domain_weight":  args.domain_weight,
                                "frame_lambda":   args.frame_lambda,
                                "lr":             lr,
                                "grad_norm":      round(grad_norm, 4),
                                "gpu_mem_gb":     [round(x, 2) for x in gpu_memory_gb()],
                                "tokens_per_sec": int(tps),
                                "elapsed_sec":    int(time.time() - t0),
                                "collated_seq_len": collated_seq_len,
                                "content_max_len": int(content_max_len)
                                if content_max_len is not None else collated_seq_len,
                                "first_record_idx": first_record_idx,
                                "first_chunk_idx": first_chunk_idx,
                                "first_nt_start": first_nt_start,
                                "first_nt_end": first_nt_end,
                                "first_prefix_token_count": first_prefix_token_count,
                            }
                            append_jsonl(train_log_path, entry)
                            wandb_log_safe(
                                wandb_run,
                                {k: v for k, v in entry.items()
                                 if not isinstance(v, list)}, step)
                            if step % (args.log_every * 10) == 0:
                                rank0_print(
                                    f"step {step:5d} | ep {entry['epoch']:.2f} | "
                                    f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                                    f"gn {grad_norm:.3f} | {int(tps):,} tok/s"
                                )

                        log_buffer_loss  = 0.0
                        log_buffer_count = 0
                        last_log_time    = time.time()
                        last_log_tokens  = 0

                    # ── Validation (audit M2: first-window, length-stratified)
                    if step % args.val_every == 0:
                        prev_best = best_val_loss
                        val_stats = run_validation(
                            model_engine, val_loader, local_rank,
                            args.val_max_batches, VAL_LENGTH_BOUNDS)
                        val_loss = val_stats["overall"]
                        if is_main():
                            ventry = {
                                "step":     step,
                                "val_loss": round(val_loss, 6),
                                "val_ppl":  round(math.exp(min(val_loss, 20)), 4),
                                "val_by_length": val_stats["by_length"],
                            }
                            append_jsonl(val_log_path, ventry)
                            wandb_log_safe(
                                wandb_run,
                                {"val_loss": ventry["val_loss"],
                                 "val_ppl": ventry["val_ppl"]}, step)
                            bucket_str = "  ".join(
                                f"{lab}:{d['loss']:.3f}"
                                for lab, d in val_stats["by_length"].items())
                            rank0_print(
                                f"  VAL @ step {step}: loss {val_loss:.4f}  "
                                f"ppl {math.exp(min(val_loss,20)):.3f}  "
                                f"(best: {best_val_loss:.4f})  [{bucket_str}]"
                            )
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                rank0_print("  NEW best val_loss → saving best/ checkpoint")

                        # Broadcast best_val_loss to all ranks
                        if torch.distributed.is_initialized():
                            t_tensor = torch.tensor(
                                [best_val_loss, val_loss],
                                dtype=torch.float64, device=f"cuda:{local_rank}")
                            torch.distributed.broadcast(t_tensor, 0)
                            best_val_loss = t_tensor[0].item()
                            current_val   = t_tensor[1].item()
                        else:
                            current_val = val_loss

                        if current_val <= best_val_loss + 1e-9:
                            save_lora_checkpoint(
                                model_engine, args, ckpt_root, "best",
                                make_client_state(
                                    step=step, best_val_loss=best_val_loss,
                                    epoch=epoch, micro_step_in_epoch=micro_step + 1, grad_accum=args.grad_accum,
                                    device=cuda_device,
                                ),
                            )

                        # ── Early stopping (audit M7): stop when val stops
                        # improving. `best/` already holds the best checkpoint,
                        # and the `finally` block exports the final adapter.
                        if args.early_stopping_patience > 0:
                            if current_val < prev_best - args.early_stopping_min_delta:
                                evals_no_improve = 0
                            else:
                                evals_no_improve += 1
                                rank0_print(
                                    f"  No val improvement "
                                    f"({evals_no_improve}/{args.early_stopping_patience}; "
                                    f"need Δ>{args.early_stopping_min_delta:g} vs "
                                    f"best {best_val_loss:.4f})"
                                )
                            if evals_no_improve >= args.early_stopping_patience:
                                rank0_print(
                                    f"  EARLY STOP at step {step}: no improvement in "
                                    f"{evals_no_improve} validations "
                                    f"(patience={args.early_stopping_patience}); "
                                    f"best val_loss={best_val_loss:.4f}."
                                )
                                should_early_stop = True

                    # ── Periodic checkpoint
                    if step % args.save_every == 0:
                        if is_main():
                            rank0_print(f"  Saving checkpoint step_{step}...")
                        save_lora_checkpoint(
                            model_engine, args, ckpt_root, f"step_{step}",
                            make_client_state(
                                step=step, best_val_loss=best_val_loss,
                                epoch=epoch, micro_step_in_epoch=micro_step + 1, grad_accum=args.grad_accum,
                                device=cuda_device,
                            ),
                        )
                        if is_main():
                            cleanup_old_checkpoints(ckpt_root, keep_last=args.keep_last_ckpts,
                                                    keep_special=args.keep_special_ckpts)
                            generate_plots(args.output_dir)
                            rank0_print("  Checkpoint saved; plots refreshed")

                    # ── Smoke-test exit
                    if args.max_steps and step >= args.max_steps:
                        rank0_print(f"Reached --max-steps={args.max_steps}; exiting")
                        if is_main():
                            generate_plots(args.output_dir)
                        return

                    # ── Graceful shutdown
                    if _SHOULD_STOP:
                        rank0_print("Graceful shutdown → saving interrupted checkpoint")
                        save_lora_checkpoint(
                            model_engine, args, ckpt_root, f"step_{step}_interrupted",
                            make_client_state(
                                step=step, best_val_loss=best_val_loss,
                                epoch=epoch, micro_step_in_epoch=micro_step + 1, grad_accum=args.grad_accum,
                                device=cuda_device,
                            ),
                        )
                        if is_main():
                            generate_plots(args.output_dir)
                        return

                    if should_early_stop or step >= total_steps:
                        break
            if should_early_stop or step >= total_steps:
                break

    except torch.cuda.OutOfMemoryError as e:
        rank0_print(f"CUDA OOM at step {step}: {e}")
        # H9: free as much GPU memory as we can before the emergency
        # save. save_lora_checkpoint allocates buffers for the ZeRO-
        # sharded optimizer state and is itself OOM-prone immediately
        # after a forward/backward failure.
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        rank0_print("Attempting emergency checkpoint (full DS + adapter)...")
        try:
            save_lora_checkpoint(
                model_engine, args, ckpt_root, f"step_{step}_oom",
                make_client_state(
                    step=step, best_val_loss=best_val_loss,
                    epoch=epoch, micro_step_in_epoch=micro_step + 1, grad_accum=args.grad_accum,
                    device=cuda_device,
                ),
            )
            rank0_print("  Emergency checkpoint succeeded")
        except Exception as ee:
            rank0_print(f"  Full emergency save failed: {ee}")
            # Last-ditch fallback: just dump the adapter on rank 0.
            # peft's save_pretrained is CPU-side, so it's the most
            # likely thing to still succeed after the GPU is wedged.
            if is_main():
                try:
                    adapter_dir = ckpt_root / f"step_{step}_oom" / "adapter"
                    adapter_dir.mkdir(parents=True, exist_ok=True)
                    model_engine.module.save_pretrained(str(adapter_dir))
                    rank0_print(f"  Adapter-only fallback saved → {adapter_dir}")
                except Exception as ee2:
                    rank0_print(f"  Adapter-only save also failed: {ee2}")
        raise

    finally:
        # Final adapter export (clean peft format, no DS wrapper)
        if step > start_step:
            rank0_print(f"Saving final checkpoint at step {step}")
            try:
                save_lora_checkpoint(
                    model_engine, args, ckpt_root, f"step_{step}_final",
                    make_client_state(
                        step=step, best_val_loss=best_val_loss,
                        epoch=epoch, micro_step_in_epoch=micro_step + 1, grad_accum=args.grad_accum,
                        device=cuda_device,
                    ),
                )
                if is_main():
                    # Clean peft-format adapter at <output_dir>/final_adapter/
                    # is the documented load path (docs/gputee/FINETUNE_GUIDE.md §11).
                    # save_lora_checkpoint already wrote the same bytes under
                    # checkpoints/step_N_final/adapter/ via peft's save_pretrained,
                    # so we copy that tree rather than re-serialising through peft.
                    final_adapter = args.output_dir / "final_adapter"
                    src_adapter   = ckpt_root / f"step_{step}_final" / "adapter"
                    if src_adapter.exists():
                        if final_adapter.exists():
                            shutil.rmtree(final_adapter)
                        shutil.copytree(src_adapter, final_adapter)
                        rank0_print(f"Final adapter exported → {final_adapter}")
                    else:
                        rank0_print(
                            f"WARNING: expected adapter at {src_adapter} but it "
                            "does not exist; skipping final_adapter export")
            except Exception as e:
                rank0_print(f"Final checkpoint/export failed: {e}")
            if is_main():
                generate_plots(args.output_dir)

        if wandb_run:
            wandb_run.finish()
        rank0_print(f"Done. step={step}  best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
