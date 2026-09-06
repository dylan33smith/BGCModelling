# Phase 13 — Identity-bucket conditioning on AZOLE. Pre-registration.

**Opened 2026-08-27.** Written BEFORE any Phase-13 arm generates (Standing Constraint 4). Endpoints
in this file do not change mid-phase; deviations are recorded as dated amendments at the bottom,
never by editing the original text.

Companions: `docs/phase3_preregistration.md`, `docs/phase8_GENOMEOCEAN_preregistration.md`.
Its comparison partner is **`phase10_AZOLE_CONTAINING_RIPP/` (`[P10-TRN-azole]`) and nothing else.**

---

## 1. The one question this phase exists to answer

`[P11]` established **r(log10 target length, own-subclass rate) = −0.933** and read it as a
**compositional capacity limit**: past ~3 kb the method writes RiPP-like DNA but not the requested
chemistry. `[P13-ANL-phagegap]` (`memory.md` 2026-08-27) then showed that the phage paper's
multi-gene 5.4 kb success was obtained **under explicit template-fidelity conditioning** — the `∼`
token, "95–100% identity to ΦX174" — and that its viable outputs are **93.0–98.8% identical to a
training sequence**.

> **Is the 6.3 kb collapse a hard capability boundary, or is it the far end of an EXCHANGE RATE
> between template fidelity and novelty that we have only ever sampled at one extreme?**

Every arm this project has run sits at the maximum-novelty end of that dial, because the dial does
not exist in our conditioning. This phase builds it and measures the whole curve.

## 2. Why this is the right experiment, and what it is NOT

**It is not an attempt to beat the phage paper at its own task.** Generating a near-copy of a known
azole cluster is not a result; Standing Constraint 1 exists precisely to stop us reporting one.

**The result is the CURVE, not any single point.** Two outcomes, both publishable, and they are
distinguishable only by running the full sweep:

| if… | then the [P11] reading is… | and the paper says |
|---|---|---|
| own-subclass rate rises sharply toward fidelity **and** the novelty gate fails in lockstep | **intact** — we bought the chemistry by copying, which was always available | the boundary is real; here is exactly what it costs to cross it |
| rate rises **while `containment`/`protein_aai` stay inside the gate** | **weakened** — a reachable middle regime exists | 6.3 kb multi-gene design is achievable de novo; [P11] mapped a conditioning artefact, not a ceiling |

⚠️ **We EXPECT the top bucket to fail the novelty gate.** That is the measurement, not an accident,
and it must not be reported as memorisation-by-surprise. The pre-registered deliverable is the
**bucket at which the gate stops passing**, quoted against the rate at that same bucket.

## 3. Design

### 3.1 The reference — our analogue of ΦX174

They had a canonical family member. We do not, so one is **defined and frozen here before training**:

> **`REF_AZOLE` = the MEDOID of `splits_subclass/AZOLE_CONTAINING_RIPP/train.jsonl`** — the record
> maximising summed nucleotide identity to all other train records.

Chosen over "the longest", "the best-annotated", or a consensus because a medoid is (a) a real
sequence, so identity to it is well-defined, and (b) reproducible from the split with no judgement.
**It is a TRAIN record**, so identity to it is identity to training data — which is what the gate
measures, and what their 93.0–98.8% figure measures.

### 3.2 The identity metric — pre-registered formula

For train record *q* against `REF_AZOLE` *t*, from `mmseqs easy-search --search-type 3`:

```
ani_to_ref = fident * alnlen / max(qlen, tlen)
```

Alignment-weighted, so a short high-identity local hit cannot masquerade as a near-copy. Records
with no hit score **0.0**. ⚠️ **Not comparable to `containment` (k=21)** — different instrument;
never quote one against the other.

### 3.3 The buckets

Their thresholds exactly, as five atomic special tokens (the `[CLS_...]` mechanism from `[P8-T4]`,
which survives a tokenizer round trip as one id and is masked from the loss):

| bucket token | `ani_to_ref` | phage-paper symbol |
|---|---|---|
| `[ID_95_100]` | 0.95–1.00 | `∼` |
| `[ID_80_95]` | 0.80–0.95 | `^` |
| `[ID_70_80]` | 0.70–0.80 | `#` |
| `[ID_50_70]` | 0.50–0.70 | `$` |
| `[ID_00_50]` | < 0.50 | `!` |

Training prefix: `[CLS_AZOLE_CONTAINING_RIPP][ID_xx_yy]<sequence>` — the class token is retained so
the ONLY delta vs `[P10-TRN-azole]` is the added bucket token.

### 3.4 ⛔ GATE T0 — THE MANIPULATION CHECK, RUN BEFORE TRAINING

Standing Constraint 5: a null is interpretable only if the intervention is verified to have landed.
The way this experiment dies quietly is **an empty top bucket** — if almost no azole record is ≥0.95
to the medoid, `[ID_95_100]` is a token with no training signal and the sweep measures nothing.

> **T0 PASSES if the top bucket holds ≥30 train records.**
> **If it FAILS, the fixed thresholds are replaced by QUINTILES of the observed `ani_to_ref`
> distribution**, recorded as a dated amendment in §7, tokens renamed `[ID_Q1]`…`[ID_Q5]`, and the
> phage-paper threshold correspondence is explicitly dropped from all reporting.

⚠️ **Either way the bucket histogram is published with the results.** A rate at a bucket holding 4
training records is not a rate.

### 3.5 Arms

| arm | conditioning | n | purpose |
|---|---|---|---|
| `[P13-A-q5]` … `[P13-A-q1]` | class + each bucket token | 200 each | **the curve** — 5 points |
| `[P13-C-nobucket]` | class token only, this adapter | 200 | isolates the token from the retrain |
| `[P10-TRN-azole]` | class token only, prior adapter | 1,000 (existing) | the published comparison partner |

Total new generation: **1,200 records.**

## 4. Endpoints — fixed

**PRIMARY (Stage A, rate over ALL generated records, per bucket):**
`own_subclass_rate` = antiSMASH full mode `--minlength 200` assigns `azole-containing-RiPP`.

**GATES (`*`, absolute, per bucket, per record):** `containment` (k=21, FAIL ≥0.95, WARN ≥0.80) and
`protein_aai`, both reported against the **real-core reference on this arm's own positives**.

**SECONDARY (Stage B, positives only):** `n_bio_orfs` (ceiling: real RIPP cores **1.454** ⚠️[SUPERSEDED 2026-09-01 — that is the class-level 50-core sample; real AZOLE cores carry **3.378**]),
`length fidelity` vs the 6,293 nt target (`[P10]` reached 0.54x), `JOINT_PASS`, `n_orfs`,
`bio_span_frac`. Full Phase-3 reporting set, every row, every bucket.

**THE HEADLINE NUMBER:** the **highest-novelty bucket whose `own_subclass_rate` significantly
exceeds `[P10-TRN-azole]`'s 2/65 = 0.031** (Fisher exact, unpaired), with its gate status attached.

## 5. Scoring — no new configuration

FULL-LENGTH scoring (the window is retired). antiSMASH 8.0.4 `--minlength 200`. Emitted by
`scripts/novelty_battery.py`. Ceiling = the existing `phase10_AZOLE_CONTAINING_RIPP/
as_realtest_ml200.tsv` (45/45 = 1.000). Floor = the existing base-model control. **No ceiling or
floor is re-derived**, so every number here is directly comparable to `[P10]` and `[P11]`.

## 6. What would falsify the phase

- **T0 fails and quintiles are also degenerate** (top quintile median `ani_to_ref` < 0.5) ⇒ azole
  has no template-fidelity structure to condition on; report as **uninformative**, not negative.
- **Flat across all five buckets** ⇒ the bucket token did not land. Check the token is atomic and
  the histogram is non-degenerate before writing a null.
- **Rate rises but `[P13-C-nobucket]` rises equally** ⇒ the effect is the retrain, not the token.

## 7. Amendments

### [AMENDMENT 2026-08-27 -a] ⛔ GATE T0 FAILED ON DNA. The identity metric is changed to PROTEIN.

`[P13-DAT-identitybuckets]`, `--metric dna`, `phase13_AZOLE_IDBUCKET/`. DNA medoid
`GCF_009707405.1.region4` (6,298 nt). **Top bucket n=3, required ≥30 ⇒ T0 FAIL.**

| bucket | n | % |
|---|---|---|
| `[ID_95_100]` | **3** | 0.4% |
| `[ID_80_95]` | 56 | 7.0% |
| `[ID_70_80]` | 3 | 0.4% |
| `[ID_50_70]` | 28 | 3.5% |
| `[ID_00_50]` | **709** | 88.7% |

**`ani_to_ref` median 0.0000 — 76.5% of azole records have ZERO alignable nucleotide identity to
the medoid.** The §3.4 quintile fallback is also **degenerate**: cuts = `[0.0, 0.0, 0.0, 0.2021]`,
so Q1–Q3 are indistinguishable.
⚠️ **This is NOT an instrument failure — dynamic range was demonstrated** (self-identity 1.0000; 59
records ≥0.80; consistent with the independent `train_frac_distinct` clustering, which found azole's
largest 80%-identity cluster at 74 records). **The data really is that diverged at DNA level.**

⇒ ★ **This is itself a finding, and it reframes the phage comparison.** *Microviridae* is a
**taxonomic family** — a real sequence family with a canonical member, so DNA identity to ΦX174 is a
meaningful axis. **"Azole-containing RiPP" is a CHEMICAL annotation over genomically unrelated
clusters**, and has no canonical member at the nucleotide level. Their conditioning axis does not
exist in our data at the level they used it.

**AMENDMENT:** `ani_to_ref` (nucleotide) is replaced by **`aai_to_ref` (coverage-weighted proteome
identity)**, defined in `terms.md`. Justification: BGC relatedness is conventionally measured at the
protein level, which is why this project's own novelty gate already carries `protein_aai` beside
`containment`. Bucket EDGES, Gate T0 and every endpoint in §4 are **unchanged**; only the axis the
buckets are cut on changes. The medoid is re-derived under the new metric so reference and metric
agree.

### [AMENDMENT 2026-08-27 -b] ✅ T0 PASSES on protein. Two buckets are declared UNDERPOWERED.

`--metric protein`. Proteome medoid **`GCF_025536415.1.region1`** (idx 428, 6,303 nt). ORFs
`min_aa=30`: 5,441 total, median 4/record, 0 records with none. **Top bucket n=40 ≥ 30 ⇒ T0 PASS.**

| bucket | n | % | status |
|---|---|---|---|
| `[ID_95_100]` | **40** | 5.0% | ✅ carries the curve |
| `[ID_80_95]` | **115** | 14.4% | ✅ carries the curve |
| `[ID_70_80]` | **14** | 1.8% | ⚠️ **UNDERPOWERED — diagnostic only** |
| `[ID_50_70]` | **2** | 0.3% | ⚠️ **UNDERPOWERED — diagnostic only** |
| `[ID_00_50]` | **628** | 78.6% | ✅ carries the curve |

★ **The distribution is BIMODAL, not a continuum** — ~155 records form a proteome-near-copy
population and 628 are essentially unrelated, with a near-empty middle (16 records across two
buckets). ⇒ Azole is **not one family with a diversity gradient**; it is a tight family plus a
grab-bag sharing only a chemical annotation.

**DECLARED BEFORE GENERATION, so it is not a post-hoc rescue:** all five tokens are trained exactly
as pre-registered (T0 passed — thresholds are NOT moved after seeing outcomes), but
**`[ID_70_80]` and `[ID_50_70]` are reported as DIAGNOSTIC ONLY and no claim rests on them.** The
curve is carried by the three supported buckets: **40 / 115 / 628.** Their per-bucket `n` is printed
in every table (Standing Constraint: a rate at n=2 is not a rate).

⇒ ⚠️ **§2's two-outcome table is now a THREE-point curve, not five.** The phase remains
interpretable — the extremes (`[ID_95_100]` vs `[ID_00_50]`) are the contrast that matters and both
are well supported.

### [AMENDMENT 2026-08-27 -c] Arm tags and generation config, fixed before generation.

§3.5 named the arms `[P13-A-q5]`…`[P13-A-q1]` before the metric amendment renamed the buckets. The
arms AS RUN, with their on-disk tags in `phase13_AZOLE_IDBUCKET/`:

| arm tag | prompt | train records behind the token |
|---|---|---|
| `p13_id95_100` | `BOS [CLS_AZOLE_CONTAINING_RIPP] [ID_95_100]` | 40 |
| `p13_id80_95` | `BOS [CLS_…] [ID_80_95]` | 115 |
| `p13_id70_80` | `BOS [CLS_…] [ID_70_80]` | ⚠️ 14 — diagnostic only |
| `p13_id50_70` | `BOS [CLS_…] [ID_50_70]` | ⚠️ 2 — diagnostic only |
| `p13_id00_50` | `BOS [CLS_…] [ID_00_50]` | 623 |
| `p13_nobucket` | `BOS [CLS_…]` — no bucket token | (control) |

n=200 each, distinct `--seed` per arm (101–106). **Decoding is byte-identical to `[P10]`'s
`azole_denovo.jsonl`:** temperature 0.9, top_p 1.0, top_k 0, repetition_penalty 1.2,
`max_new_tokens` 1600, `min_new_tokens` 0, batch 20.

**Training config is byte-identical to `[P10-TRN-azole]`** — r=16, α=32, 10 epochs, `seq_len` 10240,
lr 5e-5, warmup 50, eval/save every 50, early stopping OFF — on **P10's exact 794 train / 44 val
records** with `id_bucket` joined on (0 join misses). Trainable rises 55,449,600 → **55,480,320**;
the +30,720 is exactly the five new embedding rows. Class token id 4096 unchanged; bucket tokens
4097–4101, all verified atomic at load.

⚠️ **`p13_nobucket` is OFF-DISTRIBUTION** — every training record carried a bucket token, so
omitting it at generation is a prompt the model never saw. It still does its declared job (if every
arm including this one moves equally against `[P10]`, the effect is the retrain and not the token),
but **it is not a clean "P10 re-run" and must not be described as one.** `[P10-TRN-azole]` remains
the comparison partner of record.

⚠️ **Trained under third-party GPU contention** (99% utilisation from another user on this shared
host). Irrelevant to the endpoints, which are rates — but **no throughput or wall-clock number from
this run is quotable.**
