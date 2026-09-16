"""Substrate loading and a uniform generation interface (SPEC 5).

The two substrates differ in every way that matters for this benchmark and fail in
OPPOSITE places, so the abstraction has to be explicit about each:

                    terminator   auto-appended?   stops on it?     tokenisation
  Evo2              byte 0       NO               NO (hardcoded)   byte-level, 1 tok ~ 1 nt
  GenomeOcean       [SEP] = 2    YES              only if told     BPE, vocab 4096

⚠ EVO2 CANNOT STOP AT ALL, AND THE REASON IS NOT WHAT IT LOOKS LIKE.

The weights are local; nothing here is a remote service. Three layers were checked:

  1. `Evo2.generate()` exposes no stop parameter.
  2. `vortex.model.generation.generate()` calls the inner `Generator` with a hardcoded
     `stop_at_eos=False`, forwarding `**kwargs` AFTER it — so passing `stop_at_eos=True`
     is a duplicate-keyword error, not an override.
  3. `Generator.generate()` DOES take `stop_at_eos`, defaulting to True — but it is DEAD
     CODE. The entire implementation is:

         if stop_at_eos and (generation[0, -1:] == eos_token_ids).all():
             print("Stopping generation at EOS")

     It prints and does not break. The only `break` in the function is inside a `verbose`
     display block, unrelated. Generation always runs to `num_tokens`.

  And even if it did break, it inspects `generation[0]` — row 0 only — so under batched
  generation the whole batch would be cut the moment the FIRST sequence emitted EOS,
  truncating every other row mid-sequence.

⇒ POST-HOC TRUNCATION IS THE ONLY CORRECT MECHANISM, not a workaround for a missing
feature. Generate to budget, truncate each row at its own terminator, record `hit_eos`.
A block-wise early exit (generate in blocks, stop when ALL rows have terminated) would
recover the wasted compute and is deferred until a fine-tuned model is measured to stop
early enough to be worth it.

So termination is handled UNIFORMLY and post hoc: generate to the budget, then truncate at
the first terminator occurrence, and record whether one appeared. That works identically on
both substrates and does not depend on vendor stop support. Where a substrate DOES support
native stopping (GenomeOcean, via `eos_token_id=2`), it is used as well and the two are
cross-checked in G10.

⚠ BUDGETS ARE IN NUCLEOTIDES, NEVER TOKENS. GenomeOcean is BPE, so a token budget would
hand the two substrates different amounts of sequence.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("HF_HOME", "/data2/ds85/hf_cache")

EVO2 = "evo2"
GENOMEOCEAN = "genomeocean"


@dataclass
class Substrate:
    id: str
    family: str
    checkpoint: str
    terminator_id: int
    #: How the terminator LOOKS IN DECODED OUTPUT, for detection. For Evo2 this is a
    #: sentinel, because chr(0) cannot survive vortex's decoder (see
    #: `_install_terminator_shim`). NEVER append this to training text.
    terminator_str: str
    appends_terminator: bool          # does the tokenizer add it at encode time?
    native_stop: bool                 # can generation stop on it without post-processing?
    approx_nt_per_token: float
    #: What training text appends so the tokenizer emits `terminator_id`. This MUST encode
    #: to the real token: appending the DISPLAY sentinel instead would teach the model to
    #: emit '*' (id 42) rather than its own stop token (id 0). Empty when the tokenizer
    #: appends a terminator itself.
    terminator_encode_str: str = ""
    model: Any = None
    tokenizer: Any = None
    meta: dict = field(default_factory=dict)

    @property
    def termination_mode(self) -> str:
        """How this substrate stops. BOTH terminate and both produce terminator-truncated
        output; only the internal compute differs, and the difference is recorded rather
        than assumed because it is not the same between arms on different substrates.

        native_eos    the sampler halts on the terminator (HuggingFace `eos_token_id`)
        post_hoc      generation runs to budget and output is truncated at the first
                      terminator. Evo2 has no alternative: vortex's `stop_at_eos` prints
                      and does not break. A block-wise early exit would recover the wasted
                      compute but costs more than it saves at a 16 kb budget -- re-prompting
                      each block reprocesses the accumulated prefix -- unless termination is
                      very early, which base models do not do (measured 0/12).
        """
        return "native_eos" if self.native_stop else "post_hoc_truncation"


    # ---- training text -------------------------------------------------------------
    def training_text(self, sequence: str, prefix: str = "") -> str:
        """SPEC 4.3: bare sequence, plus a terminator where the tokenizer will not add one.

        A model that never sees a terminator in training will never emit one, so this is
        not optional for Evo2 -- and it is not a conditioning channel, it is the model's
        own native token.
        """
        if self.appends_terminator:
            return prefix + sequence
        # the ENCODE form, not the display sentinel -- see the field comments above
        return prefix + sequence + (self.terminator_encode_str or self.terminator_str)

    # ---- termination ---------------------------------------------------------------
    def truncate_at_terminator(self, text: str) -> tuple[str, bool]:
        """Return (sequence up to the first terminator, whether one was found)."""
        i = text.find(self.terminator_str) if self.terminator_str else -1
        if i < 0:
            return text, False
        return text[:i], True

    # ---- detokenisation ------------------------------------------------------------
    def detokenize(self, ids) -> str:
        """Token ids -> nucleotide text, WITHOUT a separator between tokens.

        ⚠ DO NOT REPLACE THIS WITH `tokenizer.decode()`. GenomeOcean's fast tokenizer has
        no `backend_tokenizer.decoder`, so HuggingFace falls back to `" ".join(tokens)` and
        puts a SPACE between every BPE token. `clean()` then masks each space to N, which
        at ~4.8 nt/token is an N every ~5 bases: no 21-mer is N-free, so the novelty gate's
        k-mer set is empty and it raises (KNOWN_WRONG #3 -- it fails closed, which is how
        this was caught), and had it passed, antiSMASH would have been handed sequence with
        every ORF destroyed and GenomeOcean would have scored ~0 for a reason that has
        nothing to do with the model.

        Measured: 'ATGCGG...ACGGGG' (63 nt) decodes to 75 characters via `decode()` and to
        the original 63 via this method.

        Evo2 never reaches here -- vortex's generate returns text, not ids -- but the
        dispatch is explicit rather than implied, because a silent fallthrough is what
        produced the bug above.
        """
        if self.family == EVO2:
            return self.tokenizer.detokenize(list(ids))
        toks = self.tokenizer.convert_ids_to_tokens(list(ids))
        special = set(self.tokenizer.all_special_tokens)
        return "".join(t for t in toks if t not in special)

    @staticmethod
    def clean(text: str) -> str:
        """MASK non-nucleotide characters to N. Never delete them.

        ⚠ This used to delete, and deletion SHIFTS THE READING FRAME by one base, destroying
        every ORF downstream of the deleted character -- and ORFs are what the whole scoring
        stack is built on. Measured on the frozen Stage 1 bundle: 1,422 of 1,600 generations
        (88.9%) lost at least one character this way, mean 3.13, up to 13. Every one of those
        characters was the model's OWN TERMINATOR (id 0), recovered by patching the decoder
        and regenerating: 231 of 231 dropped characters across 60 records were id 0.

        Masking preserves coordinates, so a stray byte costs one codon rather than the rest
        of the sequence."""
        return "".join(c if c in "ACGTN" else "N" for c in text.upper())



#: A character that cannot occur in a nucleotide string, used to make Evo2's terminator
#: visible after decoding. See `_install_terminator_shim`.
EVO2_TERMINATOR_SENTINEL = "*"


def _install_terminator_shim(tok, eos_id: int, sentinel: str) -> bool:
    """Make Evo2's terminator survive detokenisation.

    ⚠ vortex's CharLevelTokenizer decodes with `chr(max(32, min(id, vocab)))`, so ids 0
    (EOS), 1 (PAD) and 32 (space) ALL render as a space and are indistinguishable. Searching
    the decoded string for chr(0) therefore could never match: `hit_eos` was 0.0 in all 13
    Stage 1 run reports, and that was a structural zero of the METRIC, not a property of the
    model. Measured after this shim: the fine-tuned de novo arms emit their first terminator
    at index 2 in 49 of 60 records, so every frozen artifact is ~8,190 nt of POST-termination
    sampling.

    Gate G10 missed it because T1 only tested the encode direction and T3 tested truncation
    on a probe string built in Python, never on model output.

    `detokenize` is `"".join(map(self.decode_token, ids))`, so overriding `decode_token` on
    the instance is sufficient and is not monkeypatching library internals beyond that call.
    """
    orig = getattr(tok, "decode_token", None)
    if orig is None:
        return False

    def decode_token(token, _orig=orig, _eos=int(eos_id), _s=sentinel):
        return _s if int(token) == _eos else _orig(token)

    tok.decode_token = decode_token
    return True

def _evo2_max_seqlen(m) -> int | None:
    try:
        cfg = m.model.config
        return int(cfg["max_seqlen"] if isinstance(cfg, dict) else cfg.max_seqlen)
    except Exception:
        return None


def _de_inference(module) -> int:
    """Evo2's checkpoint load creates some parameters under torch.inference_mode(), and an
    inference tensor cannot be saved for backward -- training dies at the first norm layer
    with "Inference tensors cannot be saved for backward". Measured on evo2-1b: 28 of 265
    parameters. Cloning them OUTSIDE inference mode clears the flag; cloning inside would
    just make more inference tensors."""
    import torch
    n = 0
    with torch.inference_mode(False):
        for mod in module.modules():
            for name, prm in list(mod.named_parameters(recurse=False)):
                if prm.is_inference():
                    n += 1
                setattr(mod, name, torch.nn.Parameter(prm.data.clone(),
                                                      requires_grad=prm.requires_grad))
            for name, buf in list(mod.named_buffers(recurse=False)):
                if buf is not None:
                    mod.register_buffer(name, buf.data.clone())
    return n


def load(substrate_id: str, device: str = "cuda:0",
         trainable: bool = False) -> Substrate:
    if substrate_id.startswith("evo2"):
        from evo2 import Evo2
        name = {"evo2-1b": "evo2_1b_base", "evo2-7b": "evo2_7b"}.get(substrate_id,
                                                                     "evo2_1b_base")
        m = Evo2(name)
        tok = m.tokenizer
        eos = int(getattr(tok, "eos_id", 0))
        shimmed = _install_terminator_shim(tok, eos, EVO2_TERMINATOR_SENTINEL)
        n_inf = _de_inference(m.model) if trainable else 0
        return Substrate(
            id=substrate_id, family=EVO2, checkpoint=name,
            # the SENTINEL, not chr(eos): chr(0) cannot survive vortex's decoder
            terminator_id=eos,
            terminator_str=(EVO2_TERMINATOR_SENTINEL if shimmed else chr(eos)),
            terminator_encode_str=chr(eos),
            appends_terminator=False,       # verified: tokenize("ACGT") -> [65,67,71,84]
            native_stop=False,              # vortex hardcodes stop_at_eos=False
            approx_nt_per_token=1.0,
            model=m, tokenizer=tok,
            meta={"vocab_size": getattr(tok, "vocab_size", None),
                  "de_inferenced_params": n_inf,
                  # the model's own configured context. Measured degradation past it:
                  # NLL 0.805 at 8,192 -> 1.040 at 12,000 -> 1.239 at 15,900 (chance 1.386)
                  "max_seqlen": _evo2_max_seqlen(m),
                  "terminator_shim": shimmed},
        )

    if substrate_id in ("go-4b", "bgcfm") or "genomeocean" in substrate_id:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = {"go-4b": "pGenomeOcean/GenomeOcean-4B",
                "bgcfm": "pGenomeOcean/GenomeOcean-4B-bgcFM"}.get(
                    substrate_id, "pGenomeOcean/GenomeOcean-4B")
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        # GenomeOcean does not DECLARE an eos_token, but its tokenizer appends [SEP]=2.
        # Take the id from the tokenizer's own special-token table, never a literal.
        sep = tok.convert_tokens_to_ids("[SEP]")
        model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        return Substrate(
            id=substrate_id, family=GENOMEOCEAN, checkpoint=name,
            terminator_id=int(sep), terminator_str="",   # BPE: detect on ids, not text
            appends_terminator=True,
            native_stop=True,               # generate(eos_token_id=sep)
            # MEASURED in G10, not assumed: 1,000 tokens decoded to ~4,792 nt. An
            # estimate of 4.0 would hand GenomeOcean ~20% more sequence than the
            # nucleotide budget asks for, which is not a fair budget across substrates.
            approx_nt_per_token=4.8,
            model=model, tokenizer=tok,
            meta={"vocab_size": tok.vocab_size,
                  "declares_eos": tok.eos_token_id is not None},
        )

    raise ValueError(f"unknown substrate {substrate_id!r}")


def resolve_best(adapter_path: str) -> str:
    """A checkpoint DIRECTORY resolves to its best held-out checkpoint (SPEC 6.4).

    ⚠ SHARED so the derivation and the arm cannot diverge. `run.arm` resolved BEST while
    `run.derive_directions` did not, so a direction derived from a directory came from the
    FINAL checkpoint while the arm generated from BEST -- two different models, and the
    mismatch would have been invisible in both reports.

    ⚠ PREFERENCE ORDER CHANGED 2026-09-13: `epoch/` BEFORE `best/`. Early stopping was
    shipping adapters that had seen 5-40% of ONE epoch (15 of 17 arms, both substrates), so
    "best held-out checkpoint" meant "best of a handful of checkpoints taken before the
    model had read the data once". The arms now generate from the FULL-EPOCH checkpoint.
    `best/` is still written and still resolvable, so the two remain comparable.
    """
    import json as _json
    from pathlib import Path as _Path
    p = _Path(adapter_path)
    if p.is_dir():
        for marker in ("EPOCH", "BEST"):
            if not (p / marker).exists():
                continue
            meta = _json.loads((p / marker).read_text())
            if meta.get("path"):
                return meta["path"]
    return str(adapter_path)


def attach_adapter(sub: Substrate, adapter_path: str) -> Substrate:
    """Load a LoRA adapter and MERGE it into the base weights.

    Merging rather than wrapping: generation goes through the vendor's own path
    (`Evo2.generate` -> vortex), and handing that path a PeftModel wrapper risks a
    compatibility failure that would be silent or obscure. A merged model is a plain model
    with the adaptation baked in, so the generation path is byte-for-byte the one the base
    arm uses -- which is what makes the arms comparable.

    ⚠ Returns a Substrate whose weights are NO LONGER the base model. Reload for a
    different arm; do not attach twice.
    """
    from peft import PeftModel

    base = sub.model.model if sub.family == EVO2 else sub.model
    cfgobj = getattr(base, "config", None)
    if cfgobj is not None and not callable(getattr(cfgobj, "to_dict", None)):
        try:
            cfgobj.to_dict = lambda _c=cfgobj: {}
        except Exception:
            pass
    peft_model = PeftModel.from_pretrained(base, adapter_path,
                                           autocast_adapter_dtype=False)
    merged = peft_model.merge_and_unload()
    if sub.family == EVO2:
        sub.model.model = merged
    else:
        sub.model = merged
    sub.meta["adapter"] = adapter_path
    return sub


def attach_direction(sub: Substrate, path: str, alpha: float,
                     randomise: int | None = None,
                     sites: list[int] | None = None) -> tuple[Substrate, object]:
    """Attach an I1 derived-direction injection for generation (SPEC 6, gate G9).

    `randomise` builds SPEC 6.3's magnitude-matched random control instead, at the same
    alpha and the same unit norm, so the only difference from the real arm is whether the
    direction carries class content.

    Returns the substrate and the LIVE intervention. This is
    a hook, not merged weights. If the caller does not hold `attached()` for the duration
    of generation the arm silently generates from the BASE MODEL and reads as a null.
    """
    import torch

    from bgcbench.model.interventions import (DirectionInjection,
                                              random_direction_control)
    base = sub.model.model if sub.family == EVO2 else sub.model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(ck["hidden"])
    # ⚠ THE PREFIX MUST MATCH TOO, AND IT WAS WRITTEN BUT NEVER READ. `derive()` records
    # `prefix` in the artifact and `save()` persists it, yet nothing in the package compared
    # it back -- so a direction derived under a GTDB lineage could be attached to an arm
    # generating from bare sequence, or the reverse, with no error. The activation geometry
    # of "lineage + core" is not that of "core", so the vector would be from a different
    # input distribution and the arm would measure nothing in particular. This is the same
    # class of mismatch the weight-state check above catches, and it went unguarded because
    # the field existed and looked like it was doing work.
    want_p = ck.get("prefix")
    got_p = sub.meta.get("prefix_kind")
    if want_p is not None and got_p is not None and want_p != got_p:
        raise ValueError(
            f"direction was derived with prefix={want_p!r} but the arm generates with "
            f"prefix={got_p!r}. Those are different input distributions, so the direction "
            f"describes activations the arm never produces. Re-derive at the arm's prefix.")

    # ⚠ SITE SELECTION IS THE OTHER HALF OF G9. §6 specifies "injection site AND magnitude,
    # swept together"; only α was ever swept, and the site set sat at an unexamined default
    # of "every attention site". `sites` restricts injection to chosen indices by zeroing
    # every other row, which keeps the direction tensor the same shape as the site list the
    # model exposes -- so the realised coverage is still recorded honestly (§6.5).
    # ⚠ THE SITE SET DEFAULTS TO THE ONE THE CHECK WAS READ AT, NOT TO "all of them".
    # `g9_sites --finalize` writes `chosen_sites` into the artifact alongside a
    # `manipulation_check` read at exactly those sites. A caller that passed nothing used to
    # get every site -- a configuration the check does not describe and no gate ever saw.
    chosen = ck.get("chosen_sites")
    if sites is None and chosen is not None:
        sites = [int(i) for i in chosen]
        print(f"site set resolved from the direction artifact: {sites} "
              f"({ck.get('chosen_site_set')})", flush=True)
    elif (sites is not None and chosen is not None
          and sorted(int(i) for i in sites) != sorted(int(i) for i in chosen)):
        # Not a warning. The artifact's check_all_pass describes `chosen`; running `sites`
        # means the gate that licensed this arm was read on a different arm.
        raise ValueError(
            f"--sites {sorted(int(i) for i in sites)} does not match the site set this "
            f"direction's SPEC 6.4 check was read at ({sorted(int(i) for i in chosen)}). "
            f"The check would be describing a different configuration than the one "
            f"generating. Re-run g9_sites --finalize at the site set you want, or drop "
            f"--sites to use the checked one.")
    elif sites is not None and chosen is None:
        # ⚠ THE GUARD ABOVE WAS ONE-SIDED. An artifact with no `chosen_sites` predates
        # `g9_sites --finalize`, so its check was read at ALL sites; running a SUBSET against
        # it is the same mismatch, just in the direction the elif above does not cover. This
        # is not hypothetical -- every Evo2 direction currently on disk is in exactly this
        # state, and phaseC/phaseD pass --sites from the g9sites JSON, so REDOX_COFACTOR and
        # RIPP were already generating at 1 of 4 sites against a 4-site check.
        raise ValueError(
            f"--sites {sorted(int(i) for i in sites)} was given, but {path} carries no "
            f"`chosen_sites`: its SPEC 6.4 check was read at ALL sites, not at this subset. "
            f"A check read on a different configuration cannot license this arm. Re-run "
            f"run.derive_directions then g9_sites --finalize for this direction, or drop "
            f"--sites to generate at the configuration the check actually describes.")
    if sites is not None:
        keep = set(int(i) for i in sites)
        d0 = ck["directions"]
        if any(i < 0 or i >= d0.shape[0] for i in keep):
            raise ValueError(f"site indices {sorted(keep)} out of range for {d0.shape[0]}")
        d0 = d0.clone()
        for i in range(d0.shape[0]):
            if i not in keep:
                d0[i] = 0.0
        ck = dict(ck)
        ck["directions"] = d0
        ck["site_subset"] = sorted(keep)
    dev = next(base.parameters()).device
    if randomise is not None:
        iv = random_direction_control(base, hidden, seed=int(randomise), alpha=alpha)
        # ⚠ THE CONTROL MUST STEER THE SAME SITES. A site the real arm leaves alone because
        # it carries no class signal must be left alone here too, or the control pushes on
        # an axis the arm never touches and stops being magnitude-matched.
        #
        # ⚠ THAT USED TO MEAN DEGENERATE SITES ONLY, WHICH WAS COMPLETE ONLY WHILE EVERY ARM
        # STEERED EVERY SITE. Once a direction carries `chosen_sites`, the real arm steers a
        # SUBSET and this control still pushed on all of them -- on GenomeOcean's TERPENE
        # that is 8 sites against 24, so I1-vs-random would have compared a subset-steered
        # arm with a control pushing 3x as many sites at the same alpha. The comparison SPEC
        # 6.3 exists to make is "same magnitude, same sites, different content".
        deg = list(ck.get("degenerate_sites") or [])
        subset = ck.get("site_subset")
        silent = set(deg)
        if subset is not None:
            silent |= {i for i in range(iv.directions.shape[0]) if i not in set(subset)}
        if silent:
            with torch.no_grad():
                for i in sorted(silent):
                    iv.directions[i] = 0.0
        sub.meta["intervention_kind"] = "i1_random"
        sub.meta["intervention_random_seed"] = int(randomise)
        sub.meta["intervention_degenerate_sites"] = list(deg)
        sub.meta["intervention_active_sites"] = sorted(
            i for i in range(iv.directions.shape[0]) if i not in silent)
    else:
        d = ck["directions"]
        if d.shape[-1] != hidden:
            raise ValueError(f"directions are {d.shape[-1]}-dim against hidden {hidden}")
        iv = DirectionInjection(base, hidden, d, alpha)
        sub.meta["intervention_kind"] = "i1"
        sub.meta["intervention_degenerate_sites"] = list(ck.get("degenerate_sites") or [])
        # ⚠ READ OFF THE TENSOR, NOT OFF `active_sites`. The derivation's `active_sites` is
        # every non-degenerate site; a subset-steered arm touches fewer. Reporting the
        # derivation's list would overstate the realised coverage that SPEC 6.5 requires,
        # and by exactly the amount the site sweep narrowed it (24 -> 8 on GO TERPENE).
        sub.meta["intervention_active_sites"] = [i for i in range(d.shape[0])
                                                 if float(d[i].norm()) > 0]
    iv = iv.to(dev)
    # ⚠ THE DIRECTION MUST MATCH THE WEIGHT STATE IT STEERS. A direction derived on a LoRA
    # describes that model's activation geometry; injected into the base model, or into a
    # different adapter, it is a vector from another space and the arm silently measures
    # nothing in particular. Nothing else in the pipeline compares the two.
    want = ck.get("adapter")
    got = sub.meta.get("adapter")
    if (want or None) != (got or None):
        raise ValueError(
            f"direction was derived on weight state {want!r} but the arm is running "
            f"{got!r}. Re-derive against the arm's checkpoint, or run the arm on the "
            f"checkpoint the direction came from -- they are different activation spaces.")
    # ⚠ SPEC 6.4 TRAVELS WITH THE DIRECTION. Nothing previously read `manipulation_check`
    # outside the derivation script, so an arm could be run -- and a null reported -- from a
    # direction whose check had FAILED, with no trace of that in the frozen record. Refuse
    # by default; `BGCBENCH_ALLOW_FAILED_CHECK=1` permits it deliberately for a diagnostic
    # run, and the record still says the check failed.
    import os as _os
    chk = ck.get("manipulation_check") or {}
    # All THREE parts are required (SPEC 6, 12.A4). `check_all_pass` is written by
    # run.derive_directions; a direction predating it falls back to part (c) alone.
    passed = ck.get("check_all_pass")
    if passed is None:
        passed = chk.get("passes")
    sub.meta["intervention_check_passed"] = passed
    if passed is False and _os.environ.get("BGCBENCH_ALLOW_FAILED_CHECK") != "1":
        raise ValueError(
            f"{path} failed its SPEC 6.4 manipulation check "
            f"(mean steered-site cosine {chk.get('mean_cosine')!r}, min "
            f"{chk.get('min_active_cosine')!r}). A null from this arm would be "
            f"UNINFORMATIVE, not negative. Re-derive with more records, or set "
            f"BGCBENCH_ALLOW_FAILED_CHECK=1 to run it as a deliberate diagnostic.")
    sub.meta["intervention"] = path
    sub.meta["intervention_alpha"] = float(alpha)
    sub.meta["intervention_site_subset"] = ck.get("site_subset")
    sub.meta["intervention_sites"] = ck.get("sites")
    sub.meta["intervention_direction_class"] = ck.get("target_class")
    return sub, iv
