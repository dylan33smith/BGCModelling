# TPU Grant — Working Document (BCGModelling)

> **Status — 2026-06-10.** Active direction: a **TPU-native** program — **O1 BGC-Bench + O2 BGC-MaxText baseline + O3 Tunix-GRPO**, with **O8** education bundle in the separate Track-2 lane. **Zero custom kernels.**
>
> The earlier "port StripedHyena's CUDA/FFT kernels to XLA" pitch is **RETIRED** (see Appendix B). Expert Google TPU reviewers read single-student kernel-porting as low-feasibility; explicitly *declining* the port is now a credibility signal in the proposal.
>
> **Deadline:** June 30 2026 (rolling decisions; reply 4-6 weeks). Requires Apache-2.0/MIT release + quarterly updates. Applicant: PhD researcher + faculty supervisor; the gift goes to the university (no overhead).
>
> **The reframe that unlocks everything:** the project's moat is the **leakage-free dataset + the 8-metric eval suite + the conditioning recipe** — *not* Evo2. Evo2/StripedHyena is a swappable component, so TPUs become the natural tool (train a TPU-native generator, benchmark it, RL-optimize it against the eval suite) instead of something to fight.
>
> **How to use this doc:** Part 1 is the decision menu (8 options + ranking + recommendation + decision guide). Appendix A is the TPU-compatibility audit (the evidence for declining the port). Appendix B is the retired v1 proposal, kept for history — **do not submit it.**

---

# Part 1 — Strategy & Decision Menu

# BGC-on-TPU Grant Decision Menu
### 2026 Google Awards for ML Research & Education with TPUs — strategist's recommendation for [PhD student + supervisor]

The 17 raw framings collapse into **8 distinct options** plus one **anti-option** (the kernel port) that the menu explicitly retires. The collapse logic is at the end. Throughout, remember the two assets that actually win this grant: **(1) the leakage-free curated dataset and (2) the validated 8-metric eval suite.** Several options are really just different ways of monetizing those two assets on TPU. The eval suite is confirmed present and substantial (`src/bgc_pipeline/evaluation.py`, 1100 lines; ESMFold at `metric_3`, line 446).

---

## The 8 options at a glance

| # | Option | Feasibility | Grant-fit | Sci. value | Effort | Evo2 |
|---|--------|:---:|:---:|:---:|:---:|:---:|
| **O1** | **BGC-Bench** — frozen benchmark + leaderboard + TPU-native MaxText baseline | **H** | **H** | M-H | 4–6mo | scorer/baseline |
| **O2** | **BGC-MaxText** — TPU-native long-context transformer generator (Splash/CP) | M-H | **H** | M-H | 6–9mo | teacher/baseline |
| **O3** | **Eval-as-Reward (GRPO)** — Tunix verifiable-reward RL on the 8 metrics | M | **H** | **H** | 6–9mo | teacher/KL-anchor |
| **O4** | **Evo2-Distill** — cache GPU-Evo2 logits, distill into TPU student via Tunix | M-H | M-H | M-H | 3–6mo | **preserved (soft)** |
| **O5** | **BGC-SSM/Mamba2** — JAX SSD genomic model + Pallas selective-scan kernel | M (sys L) | M-H | **H** | 8–14mo | dropped/teacher |
| **O6** | **Generate+Validate Factory** — vLLM-on-TPU mass sampling + at-scale folding | M | M | M | 4–8mo | GPU generator OK |
| **O7** | **Guided-Decoding Critics** — fast TPU verifier heads + best-of-n reranking | M-H | M | M | 4–6mo | **fully preserved** |
| **O8** | **Education bundle** — plug-and-play Tunix lab + grad module + workshop | **H** | **H (T2)** | n/a (teaching) | 2–6mo | scorer only |
| ~~A~~ | ~~Port StripedHyena FFT kernels to Pallas~~ — **RETIRED, do not pitch** | **L** | L | L | 12–24mo | literal weights |

---

## Option detail

### O1 — BGC-Bench: the benchmark + leaderboard + TPU-native baseline
*(merges raw "BGC-Bench" + the "(D)/factory" leaderboard-feeder role)*

- **Description.** Freeze the genome-keyed held-out test split + the conditioning protocol + the 8-metric scorer into a versioned, Apache-2.0 benchmark with one headline number ("valid-and-novel BGC rate") plus per-criterion breakdown. Ship a **TPU-native reference baseline** (Gemma-shaped decoder in MaxText, Splash + Context-Parallel attention default-on, Tunix LoRA/SFT) so the leaderboard is reproducible on TPU from day one. Host a leaderboard; Evo2-7B-LoRA enters as a strong GPU reference, not the deliverable.
- **TPU role.** Native and unforced: the baseline is a vanilla decoder transformer — exactly what MaxText exists to train. Splash (Pallas) + Context Parallelism shard the long DNA sequence across the pod (the same machinery AlphaGenome rides on TPU). The leaderboard's distinctive promise is reproducible-on-TPU end-to-end, which a GPU-only Evo2 stack cannot offer.
- **Feasibility: H.** The 80% that sinks benchmark papers (leakage-free data + validated metrics + conditioning recipe) already exists. Remaining work is a supported MaxText train + harness packaging + a leaderboard. No kernels, no FFT, no custom architecture.
- **Benefits.** Leads with the true moat, not a systems lift reviewers distrust. Benchmarks are durable, high-citation, and the most defensible single-student deliverable. Multi-objective + memorization-gated scoring is genuinely novel. Quarterly leaderboard cadence maps onto the grant's quarterly-update requirement. Nothing built so far is wasted.
- **Drawbacks.** A benchmark needs adopters — requires a flagship result + outreach. MaxText accepts only curated architectures (baseline must be Gemma/Llama-shaped). Heavy scorers (antiSMASH/ESMFold/BiG-SCAPE) need caching/batching. Harsh reviewers may rate "benchmark" below "model" on technical merit — defended by the TPU-native baseline + the novel metric.
- **Evo2 tradeoff.** Fully preserves Evo2's knowledge: it stays on GPU as a leaderboard entry and the metric-5 perplexity scorer. The TPU baseline trains fresh but only needs to be a *credible reference*, not SOTA.
- **Grant-fit.** Track 1 Theme 2 (Applied Science) + strong Theme 3 (Tool Builder). Wins on open-source impact, feasibility, and TPU relevance simultaneously.
- **Effort:** 4–6mo. **Novelty:** genuinely novel as a benchmark (no public benchmark jointly scores class + domain-completeness + novelty + synthesizability + expressibility with a memorization gate). Protein-design precedent shows benchmarks are top-tier-publishable.

### O2 — BGC-MaxText: TPU-native long-context conditioned generator
*(merges raw "BGC-MaxText" + "BGC-LongGen" + "(B)")*

- **Description.** Train a decoder-only nucleotide transformer in MaxText (~300M–1B, scale toward larger if TRC allows), reusing the `|COMPOUND_CLASS:..|+taxonomy` prefix verbatim. Long context (32k–100kb BGCs) comes from **Context-Parallel + Splash attention as config flags, not kernels you write**. The science deliverable is the first TPU-native *generative* genomic FM + a controlled study: how close does a clean transformer get to Evo2-7B-LoRA on the 8 metrics, and how far does sequence-parallelism push BGC context on a pod?
- **TPU role.** The canonical MaxText workload. Long-context sequence-parallel attention on an ICI-connected pod is the native way to reach kilobase DNA windows — not a GPU ring-attention hack, not a port.
- **Feasibility: M-H.** Systems path is fully paved. The genuine risk is **scientific, not systems**: you will not replicate Evo2's ~2.4T-token pretraining budget on academic TPU, so a from-scratch base may underperform Evo2 on novelty/foldability. Mitigate by continue-training an open base rather than starting cold, and by **scoping the claim to "native conditioned generation with rigorous eval," not "beat Evo2."**
- **Benefits.** Maximal TPU-ecosystem relevance (rides Google's flagship stack as intended). Fills the documented gap. Reuses 100% of the moat unchanged. Clean Apache-2.0 release (model + DNA-MaxText config + harness). Maps perfectly to the "Model Architect: port a frontier *capability* to XLA" archetype.
- **Drawbacks.** Gives up Evo2's pretraining in the weights. Quadratic attention is arguably the wrong inductive bias for ultra-long DNA vs Evo2's long-conv/SSM. 18K BGCs alone can't pretrain — needs a bacterial-genome corpus (data-engineering scope). A null result ("transformer ≈ Evo2") is less exciting than a win.
- **Evo2 tradeoff.** Gives up the weights but keeps Evo2 on GPU as teacher + perplexity scorer + headline baseline; knowledge re-enters as supervision, not parameters.
- **Grant-fit.** Track 1 Theme 2 primary + Theme 3 secondary. Wins on all four criteria *if scoped honestly.*
- **Effort:** 6–9mo scoped (9–12 if a competitive base needs real pretraining). **Novelty:** high as a capability/system claim; must come from the *TPU-native + conditioning + eval-rigor combination*, not raw architecture (transformer-DNA is crowded: GENA-LM, HybriDNA, JanusDNA).

### O3 — Eval-as-Reward: verifiable-reward GRPO with Tunix
*(merges the two near-identical raw GRPO framings)*

- **Description.** Wrap the 8-metric suite as a Tunix verifiable reward (the documented `grpo_demo` pattern accepts arbitrary non-differentiable Python reward callables). Tier the reward: a **cheap dense inner loop** (pyhmmer domains, DNA-Chisel, CAI, k-mer novelty gate — all CPU-fast, every rollout) and an **expensive sparse outer signal** (antiSMASH, ESMFold, BiG-SCAPE — subsampled/async/held-out). GRPO's group-relative advantage (group = fixed prefix) needs no value network. Policy is a TPU-native Gemma/MaxText DNA decoder; Evo2 on GPU supplies the perplexity reward + a KL anchor to keep the policy on the natural-sequence manifold.
- **TPU role.** The single most TPU-native framing: RL rollouts are large-batch sampling — exactly the TPU-economical regime — and Tunix is Google's JAX-native, TPU-built post-training library designed for precisely this loop. The Track-1 research version of the grant's own "Fine-tuning Gemma with Tunix on TPUs" archetype.
- **Feasibility: M.** Tunix GRPO runs end-to-end on v5e/v6e today and the eval suite already returns per-metric score dicts. Risks are RL-specific, not TPU-specific: **reward hacking** (gate aggressively — the k-mer novelty gate + antiSMASH class check are built-in anti-hacking), **reward latency** (heavy metrics can't run inner-loop — managed by the tiering above), and **RL stability**. The eval suite + conditioning recipe already existing removes the usual cold-start risk.
- **Benefits.** *Closes the loop:* directly optimizes the 5 acceptance criteria that SFT cannot target (novelty, synthesizability, domain-completeness are constraints, not next-token likelihoods). Turns the eval moat into a *training* signal. Highest technical-merit ceiling. Produces an open "biological-verifier reward library for Tunix."
- **Drawbacks.** Highest research risk: RL may yield only marginal gains over SFT, or gains that don't survive the held-out heavy-metric check (a gap you must report honestly). Depends on having a competent TPU policy first — **overlaps O2's baseline work** (a feature, not a bug; see Combinations). Cross-hardware scoring interface (Evo2-GPU ↔ policy-TPU) needs clean glue.
- **Evo2 tradeoff.** Tightest integration of the GPU asset: Evo2 shapes training as perplexity-reward + KL anchor without ever being ported.
- **Grant-fit.** Track 1 Theme 2 + credible Theme 1 (RL-with-verifiable-domain-rewards as method). Best fit to the named recommended stack. Wins on innovation + TPU relevance; defend feasibility carefully.
- **Effort:** 6–9mo. **Novelty:** genuinely novel for BGC DNA (no prior verifiable-reward RL for biosynthetic clusters). Honest caveat: GRPO-for-bio-sequence exists (ProteinZero, 2025) — the contribution is the *genomic domain + the 8-metric multi-verifier + the TPU/Tunix engineering*, not the RL algorithm.

### O4 — Evo2-Distill: preserve Evo2's knowledge on TPU without porting kernels

- **Description.** Run frozen Evo2-7B on the existing H100 purely as a **teacher**: cache top-k logits over a genomic+BGC corpus to disk — the teacher never runs on TPU, sidestepping the entire kernel problem. Train a TPU student (O2 transformer or O5 SSM) with Tunix's built-in distillation trainers. Because Evo2 uses single-nucleotide tokenization, a vocab-matched student uses plain KL (no cross-tokenizer loss needed). Finish with SFT on the 18K BGCs; score with the suite.
- **TPU role.** Tunix is co-designed with MaxText for exactly this; KD is a first-class trainer. TPU does high-throughput student training while the un-portable Hyena/FFT teacher stays on GPU — the canonical "right hardware for each role" design, showcasing a flagship Tunix capability.
- **Feasibility: M-H.** Distillation is built into Tunix; offline logit caching removes co-location; vocab match removes cross-tokenizer complexity. Main risks are storage/IO for caching logits over long DNA windows and the student-capacity ceiling — engineering, not research.
- **Benefits.** **The only option that explicitly preserves Evo2's 9.3T-bp knowledge** while requiring zero kernel porting. Most compute-efficient route to a strong TPU model. Architecture-agnostic — composes with O2 or O5 as the student.
- **Drawbacks.** Student quality is upper-bounded by the teacher and student size — you approach Evo2 more cheaply, you don't beat it. Logit caching is storage-heavy. Tunix KD examples are same-family (Gemma→Gemma); cross-architecture Hyena→transformer via feature projection is documented but less battle-tested (expect glue code). Reviewers may read it as "applied distillation."
- **Evo2 tradeoff.** Its entire reason to exist: transfers Evo2's learned distribution via soft targets, not weights. Cost is a distillation gap.
- **Grant-fit.** Track 1 Theme 2 + Theme 3; strongest "uses the recommended stack as intended" story (Tunix KD + MaxText student). Wins on feasibility + TPU relevance; lower on raw innovation.
- **Effort:** 3–6mo. **Novelty:** moderate — novel as the *first distillation of a frontier generative genomic model across the GPU→TPU and Hyena→transformer boundary*. Frame as "preserving a foundation model's knowledge on TPU without porting its kernels."

### O5 — BGC-SSM / Mamba2: the XLA-tractable long-range architecture + a real kernel
*(merges raw "BGC-SSM" + "(C) Mamba2 hybrid")*

- **Description.** A causal Mamba-2 / Transformer-Mamba2 hybrid (HybriDNA is direct 2025 precedent) genomic decoder in JAX. Mamba2's selective scan reduces to a chunked associative-scan **matmul** that XLA/Pallas handle far better than FlashFFTConv — Evo2's hard part (FFT long-conv) is **dropped, not ported**. The systems deliverable is a clean, benchmarked **Pallas selective-scan kernel** released open-source — literally the grant's "Kernel Engineer (Mamba/SSM)" archetype.
- **TPU role.** Google explicitly names Mamba/SSM under Kernel Engineer, so a TPU-native SSD scan is a *sanctioned, high-merit* target — one well-defined associative-scan primitive, not a kernel zoo.
- **Feasibility: M, systems-L.** JAX Mamba references exist; HybriDNA proves SSMs model DNA. But hitting competitive TPU MFU on the scan has a long debugging tail, Mamba isn't first-class in MaxText (transformer-only), and from-scratch genomic quality risk is the same as O2. Two hard problems at once (new kernel + new bio model) is a lot for one student.
- **Benefits.** Highest technical-merit ceiling — hits the named archetype dead-center. Cleanest "TPU-native, not a port" architecture story. Strongest ultra-long-DNA (>>100kb) narrative. Two stackable artifacts (reusable kernel + generative SSM).
- **Drawbacks.** Highest systems risk of the generative options. Builds outside MaxText's best-supported path. Highest "ambitious but might not land" exposure (kernel never reaches MFU, or the small SSM generates poor BGCs).
- **Evo2 tradeoff.** Drops Evo2's weights (architecture-incompatible); recovers the *capability* architecturally. Best paired with O4 distillation.
- **Grant-fit.** Track 1 — the only option credibly touching all three research themes (Theme 1 Foundational + Theme 2 Applied + Theme 3 Kernel Engineer). Wins hardest on innovation + open-source; feasibility is its weakest card.
- **Effort:** 8–14mo. **Novelty:** highest of all options; also highest chance to under-deliver on one axis.

### O6 — Generate + Validate Factory: TPU throughput as the contribution
*(merges raw "TPU-scale factory" + "(D) vLLM-on-TPU serve")*

- **Description.** Two-stage TPU pipeline. Stage A: mass conditional sampling from a TPU-compatible generator (Gemma / O2 / O5 — **not Evo2; vLLM-on-TPU can't serve StripedHyena**) via vLLM-on-TPU (ragged paged attention, ~5× throughput). Stage B: at-scale ESMFold/folding + the fast metrics over millions of ORFs. Deliverable: an open TPU generation+validation service + the largest in-silico-validated synthetic BGC library + a generation-frontier study (how valid-and-novel rate trades off vs conditioning strength, temperature, taxon).
- **TPU role.** The most defensible "TPUs are the natural tool" story because it's a **throughput/economics** argument, not a kernel argument: large-batch generation + large-batch folding are where TPUs win on perf-per-dollar. Compute scales with the science, so the grant is self-justifying.
- **Feasibility: M.** Generation-at-scale is High. **The weak link is ESMFold-on-TPU** — the reference impl is PyTorch; no production JAX ESMFold exists. Scope Stage B as "batched structure inference with a PyTorch-on-TPU fallback," not a from-scratch JAX folder.
- **Benefits.** Strongest pure-TPU economic narrative, no kernel work. Tangible citable artifact (largest validated library). Reuses the suite directly. Becomes the engine that feeds O1's leaderboard.
- **Drawbacks.** ESMFold-on-TPU engineering risk. CPU-heavy validators (antiSMASH/BiG-SCAPE) create a cost/storage tail. "We generated a lot" reads as engineering unless a falsifiable hypothesis anchors it. If the TPU generator underperforms, library quality suffers.
- **Evo2 tradeoff.** Flexible; cleanest hybrid keeps Evo2 (GPU) as generator and TPU as the validation factory — but that makes the TPU role validation-only.
- **Grant-fit.** Track 1 Theme 2 + Tool Builder. Wins on TPU relevance + feasibility-of-impact; weakest on innovation unless the frontier study is real science.
- **Effort:** 4–8mo. **Novelty:** moderate (scale, not method).

### O7 — Guided-Decoding Critics: the low-risk enabler
- **Description.** Train small JAX/TPU classifier/value heads that cheaply predict the slow eval verdicts (class, domain-completeness, synthesizability, novelty). At inference, use them for best-of-n / verifier-reranked / FUDGE-style guided decoding: any generator (Evo2-GPU included) proposes, TPU critics reweight toward the 5 constraints. The heads double as **O3's dense inner-loop reward**, making GRPO affordable.
- **TPU role.** Small critics + best-of-n reranking are embarrassingly parallel scoring — a clean TPU throughput story via vLLM/JAX inference.
- **Feasibility: M-H.** Critic heads + reranking are straightforward supervised training on labels the suite already produces. Tight token-level guidance for discrete DNA is fiddlier (thinner literature).
- **Benefits.** Lowest training risk (no RL tricks, no pretraining). Generator-agnostic — ships value with Evo2-GPU before any TPU base exists. **Directly de-risks O3.**
- **Drawbacks.** Lowest standalone novelty (guided decoding is established). Surrogate critics can be miscalibrated vs real tools (surrogate-gaming). On its own produces no new model.
- **Evo2 tradeoff.** Most Evo2-friendly: keeps Evo2 fully intact on GPU as proposer; TPU adds complementary critics.
- **Grant-fit.** Track 1 Theme 3 (Tool Builder). Best as O3's bundled enabler or a fast-follow release; weak solo flagship.
- **Effort:** 4–6mo. **Novelty:** moderate, honestly incremental.

### O8 — Education bundle: plug-and-play lab + grad module + workshop
*(merges the three Track-2 framings)*

- **Description.** A composable Track-2 deliverable, sized to ambition: **(a)** a 3-notebook plug-and-play lab — the genomics analog of Google's named "Fine-tuning Gemma with Tunix on TPUs," running free-tier Colab v5e-1, with the 8-metric suite as a science-grounded **auto-grader** (the differentiator — most ML-bio labs grade with toy accuracy); **(b)** a 4–6 week graduate module "Generative Genomics on TPUs" teaching JAX/MaxText/Tunix, GSPMD sharding, Splash/Ring attention, with the project as the running case study and the data-leakage + novelty-gate labs teaching scientific rigor; **(c)** a half-day workshop + permanent "GPU→TPU for genomic generation" tutorial as the reach multiplier.
- **TPU role.** TPUs are the *subject*, not a port target. Tunix LoRA runs free on Colab TPU today; the module teaches the GPU→TPU programming-model shift using the project's own H100/DeepSpeed stack as the explicit "before."
- **Feasibility: H.** Every component exists; main work is pedagogical engineering (sequencing labs, autograders, a small-but-faithful JAX LM). The grader must ship a CPU/GPU subset or precomputed cache so heavy deps (antiSMASH/ESMFold/MMseqs2) don't break the plug-and-play promise.
- **Benefits.** Maps 1:1 onto the cited Track-2 archetype — reviewers instantly recognize it. Reuses the eval suite as a *novel teaching artifact*. Zero-cost reach (free TPU). Lab is shippable in a term; module/workshop scale up.
- **Drawbacks.** Small student model produces weak BGCs — framing must foreground "learn the pipeline + eval loop," not output quality. Lowest novelty as research. Module is real curriculum work atop a dissertation.
- **Evo2 tradeoff.** Sidesteps Evo2 on the training path (students fine-tune a small TPU transformer); Evo2 appears only as an optional GPU perplexity scorer.
- **Grant-fit.** **Track 2 Education**, all three focus areas across the bundle. Wins reach + open-source impact + feasibility. The *only* family that naturally yields a teaching deliverable — and Track 2 is a separate scoring lane, so this composes with any Track-1 option at near-zero opportunity cost.
- **Effort:** 2–3mo (lab) / 4–6mo (full module). **Novelty:** moderate as research, high as a teaching artifact.

### ~~Anti-option A — Port StripedHyena FFT kernels to Pallas. RETIRED.~~
Keep it on the menu only as the thing you **explicitly decline**. TPUs/MXUs are built for dense matmul; Evo2's data-dependent FFT long-conv is the single op that maps *worst* to TPU. Single-student parity port = 18–24 engineer-months with high odds of no converged result, and even on success the deliverable is "same model, different chip" — weak on the innovation criterion and exactly the pitch expert TPU reviewers discount. **Naming this as the obvious-but-wrong path and rejecting it is itself a credibility signal in the proposal.**

---

## Ranking — combined (feasibility × grant-fit × scientific value) for *this* applicant

> Weighting reflects the applicant's reality: a PhD student + supervisor, a 1-year-ish window, a 1–2 page proposal judged on TPU-relevance / technical-merit / feasibility / open-source-impact, and a moat that lives in **data + eval**, not systems.

1. **O1 — BGC-Bench.** Best risk-adjusted score. It is the only option that is *simultaneously* high on all three axes, because it leads with the moat and demands no scientific miracle (the baseline only needs to be credible). Maximal open-source impact, quarterly cadence built in, and it makes every other option's model a leaderboard entry. Lowest chance of a year-end "we have nothing."
2. **O3 — Eval-as-Reward (GRPO).** Highest *technical-merit-per-unit-feasibility* of the model options. It uses Tunix exactly as Google intends, converts the eval moat into a training signal, and tells a clean before/after story. Ranks below O1 only because RL carries real "might yield marginal gains" risk — which is why it should ride on O1's baseline rather than stand alone.
3. **O2 — BGC-MaxText.** The honest, feasible flagship *model* option and the literal "first TPU-native generative genomic FM" claim. Ranks below O3 because, scoped honestly, its headline risks being "transformer ≈ Evo2" (a systems/benchmark result, not a capability win). It is also the substrate O1 and O3 both need, so it rarely stands alone — it's the backbone.
4. **O4 — Evo2-Distill.** Highest feasibility among knowledge-preserving model options and the best "uses the recommended stack as intended" narrative. Slightly lower scientific ceiling (you approach, not beat, Evo2) and a touch more "applied distillation" framing risk than O2/O3.
5. **O8 — Education bundle.** Ranked here as a **near-free multiplier, not a competitor**: it scores in a separate Track-2 lane, so pairing it with any Track-1 option strictly increases total grant surface. As a *standalone research* pitch it would rank last — its value is additive.
6. **O6 — Generate+Validate Factory.** Strong TPU-economics story and a tangible artifact, but the ESMFold-on-TPU weak link and the "scale ≠ science" perception cap it. Best as O1's engine.
7. **O7 — Guided-Decoding Critics.** Safe and useful but lowest standalone novelty; its real worth is de-risking O3. Don't lead with it.
8. **O5 — BGC-SSM/Mamba2.** Highest novelty ceiling and the only all-three-themes option, but **highest systems risk and longest timeline** — two hard problems for one student. Ranks last on *risk-adjusted* value despite topping raw novelty. Excellent as a stretch arm or a follow-on, not the primary 1-year bet.
9. ~~Kernel port~~ — off the board.

---

## Recommendation

**Primary pick: O1 (BGC-Bench) as the spine, with O2 (BGC-MaxText) as its built-in TPU-native baseline.** This single coherent proposal maximizes all four review criteria at once: TPU-relevance (MaxText + Splash + Context-Parallel, used as intended), technical merit (the novel multi-objective memorization-gated metric *is* the science), feasibility (the moat exists; the baseline needn't be SOTA), and open-source impact (frozen Apache-2.0 harness + public leaderboard + reusable DNA-MaxText config). It is the proposal least able to fail, and benchmarks are durable, high-citation outputs ideal for a PhD timeline.

**Strong runner-up: O3 (Eval-as-Reward GRPO), pitched as the research-forward stretch built on the O1/O2 baseline.** If the applicant wants the higher technical-merit ceiling and a fresher publishable angle (first verifiable-reward RL for BGC DNA), this is the one — and because it sits on top of the same baseline, it shares the O1/O2 prerequisite rather than competing for it.

**Plus O8 (Education lab) bolted on regardless.** It's a separate scoring lane at near-zero opportunity cost and turns the same assets into a Track-2 win.

---

## Decision guide

- **"Keep Evo2 / waste nothing built so far"** → **O4 (Distill)** is the purist answer (preserves the 9.3T-bp knowledge as soft targets). **O1** and **O7** also keep Evo2 fully intact on GPU as scorer/baseline/proposer.
- **"Minimize risk / guarantee something ships"** → **O1**, with **O8** as the floor. Both ride existing assets and the blessed stack; neither needs a scientific miracle.
- **"Maximize novelty / strongest technical-merit"** → **O3** (verifiable-reward RL) for the best risk-adjusted novelty; **O5** (SSM + Pallas kernel) for the highest raw ceiling if the applicant has genuine kernel appetite and a longer horizon.
- **"Also teach / want a Track-2 deliverable"** → **O8** — and it composes with any of the above for free.
- **"Best pure-TPU 'this needs a pod' economic argument"** → **O6** (mass generate + fold), ideally feeding O1's leaderboard.
- **"Strongest single 'uses the recommended stack as intended' story"** → **O4** (Tunix KD) or **O3** (Tunix GRPO).

---

## What combines well

These are not 8 mutually exclusive bets — several snap together into one coherent program:

- **The flagship program (recommended):** **O2 (TPU-native baseline) → O1 (benchmark + leaderboard) → O3 (GRPO on the same policy).** One backbone model, three escalating contributions, one before/after leaderboard story. This is the "TPU-native generator + the benchmark + Tunix-RL constraint optimization as one coherent program" the brief hints at, and it reads as a single thesis arc.
- **O3 + O7:** the guided-decoding critics *are* O3's dense inner-loop reward — building O7 makes GRPO affordable and de-risks it. Bundle them.
- **O4 into O2 or O5:** distillation supplies the student architecture's missing genomic prior. If O2's from-scratch quality worries reviewers, fold O4 in — Evo2's knowledge enters by soft targets, solving the cold-start problem.
- **O6 + O1:** the generation+validation factory is the engine that mass-produces leaderboard submissions and stress-tests the benchmark at scale.
- **O8 + anything:** the education bundle is the public face of whatever research option is funded, and the workshop/tutorial double as dissemination for it.
- **O5 + O4:** the SSM recovers long-range *capability* architecturally while distillation recovers Evo2's *knowledge* — together they neutralize the SSM's cold-start risk, though at the cost of stacking two hard problems.

---

## Positioning the applicant's own idea (MaxText + Ring Attention)

The applicant's idea **is O2** (its "(B)" form) and partly **O1's baseline** — a decoder-only transformer in MaxText using **Splash/Ring (context-parallel) attention** for long DNA. Where it ranks and why, honestly:

- **It ranks #3 on its own**, and it is the single most *useful* idea on the menu because it's the **shared substrate** O1, O3, and O6 all need. It is not a wasted instinct — it's the backbone. Its great virtue is that it is **genuinely TPU-native, not a forced port**: Splash attention is a shipped JAX/Pallas TPU kernel and Ring/context-parallel attention is the documented JAX sequence-parallel recipe — no kernel writing, exactly what MaxText is for. That is precisely the "TPUs are the natural tool" framing the applicant set out to find.
- **Its honest weakness is the headline, not the engineering.** As a *standalone* pitch ("train a transformer DNA LM in MaxText"), reviewers can read it as incremental versus HyenaDNA/Evo2/GENA-LM/HybriDNA, and its likeliest scoped outcome is "TPU transformer ≈ Evo2," a null-ish result that lands as a systems/benchmark paper rather than a capability win. The fix is **not to lead with the model** — wrap it in O1 (the benchmark gives it a novel, defensible contribution that doesn't depend on beating Evo2) and/or push it forward with O3 (RL gives it a fresh method claim). Ring/context-parallel attention also lets you make the crisp, TPU-flattering empirical claim ("how far can sequence-parallelism push BGC context length on a pod?") that turns the long-context machinery into a result.
- **Bottom line for the applicant:** your instinct is correct and central — keep it — but **demote it from "the pitch" to "the engine," and let the benchmark (O1) and/or the verifiable-reward RL (O3) carry the novelty.** That converts your idea from a contestable "another DNA transformer" into the reproducible-on-TPU backbone of a program that scores on all four review criteria.

---

## TPU native evo2 model
We propose Evo2-XLA: an open-source, hardware-agnostic re-implementation of the StripedHyena 2 architecture optimized for the XLA compiler stack. By porting the underlying long-context implicit convolutions and hybrid attention layers to native JAX/Pallas primitives, we will unlock the pre-trained, 7B/40B parameter biological weights of Evo2 for high-performance training and inference on Google TPU Pods. We will validate numerical parity against the reference PyTorch implementation before leveraging this newly unlocked compute scale to fine-tune the model for constrained, synthesis-ready Biosynthetic Gene Cluster (BGC) generation.


This approach avoids the massive compute expense of pre-training from scratch, retains the power of Evo2's training data, and gives Google a highly valuable open-source tool that makes one of the world's best genomic models run on their silicon.

To learn more about the engineering background and the broader capabilities of the model you are porting, watch the team's detailed breakdown in Building Evo 2: A Frontier DNA Language Model. This video explains the architecture, training dataset, and biological design capabilities directly from the primary authors of Evo2, providing excellent context for your re-implementation strategy.

Phase 1: Re-implementing the Math Frontend (StripedHyena 2 in JAX)Your first task is to write a clean implementation of the StripedHyena 2 architecture using pure JAX or PyTorch/XLA. The model alternates between rotary attention and gated convolutions.The Attention Layers: Standard self-attention layers map cleanly to JAX/XLA out of the box. You will just need to implement the Rotary Position Embedding (RoPE) using native tensor slicing and transformations instead of the GPU-specific flash_attn normalizations.The Convolution Layers: This is the primary hurdle. Evo2 uses custom CUDA code (flash_fft_conv) to compute massive 1-million context convolutions. To make this work on a TPU, you must implement the convolution in the frequency domain using XLA-compatible primitives:Pythonimport jax.numpy as jnp

def tpu_hyena_conv(x, kernel_filter):
    # Move inputs to the frequency domain natively on the TPU
    x_fft = jnp.fft.rfft(x, axis=-1)
    k_fft = jnp.fft.rfft(kernel_filter, axis=-1)

    # Element-wise multiplication (equivalent to time-domain convolution)
    out_fft = x_fft * k_fft

    # Transform back to the time/sequence domain
    return jnp.fft.irfft(out_fft, axis=-1)
Wrap this entire forward pass in @jax.jit, and the XLA compiler will optimize the memory layouts automatically for the TPU.Phase 2: Weight Translation and Parity ValidationOnce the structural blueprint is written in JAX, you have to load Evo2’s 7-billion (or 40-billion) parameters into it.Extract the State Dict: Use a standard CPU script to read the original PyTorch weights into memory.The Key-Mapping Script: Write a dictionary mapping function. For example, if the original Nvidia repository names a weight mixer.layers.0.hyena_proj.weight, you must map that exact tensor to your corresponding JAX variable name (e.g., params['layer_0']['hyena_proj']['w']).Numerical Parity Validation (Crucial): Before running massive BGC fine-tuning, you must prove your TPU code is mathematically identical to the original code. You pass an identical 1,000-base-pair test sequence through both the original GPU model and your new TPU model. You then compare the intermediate hidden states and output logits. If they match down to a tight numerical tolerance (e.g., $10^{-5}$), your port is 100% successful, and the pre-trained biological knowledge remains fully intact.Phase 3: Native TPU ShardingEvo2 relies on Megatron-LM for sequence parallelism on GPUs. You will completely strip this out.Instead, you will use JAX’s native jax.sharding API or PyTorch/XLA's GSPMD. You define a logical TPU mesh and tell the runtime to shard the 1-million token sequence dimension across the available TPU cores. XLA will handle the underlying collective communications automatically, allowing you to scale the context window dynamically across a TPU Pod slice.

---


### Appendix — how the 17 raw framings collapsed to 8
- **O1** ← "BGC-Bench" + leaderboard-feeder role of "(D)".
- **O2** ← "BGC-MaxText" + "BGC-LongGen" + "(B)" (all = decoder transformer in MaxText w/ Splash/CP/Ring).
- **O3** ← "Eval-suite-as-reward" + "BGC-GRPO" (the two are the same GRPO option).
- **O4** ← "Evo2-Distill" (standalone).
- **O5** ← "BGC-SSM (Mamba-2)" + "(C) Mamba2/hybrid" (same architecture + Pallas scan).
- **O6** ← "TPU-scale generation + validation factory" + serving role of "(D) vLLM-on-TPU".
- **O7** ← "BGC-Guide (classifier/verifier-guided decoding)".
- **O8** ← "Plug-and-Play Lab" + "Graduate Module" + "Workshop/Tutorial" (one composable Track-2 bundle).
- **Anti-option** ← "(A) Port StripedHyena kernels" — retained only as the explicit decline.

(Resolved 2026-06-11: the suite is an **8-metric acceptance suite** — antiSMASH class, Pfam/pyhmmer domains, ESMFold foldability, DNA-Chisel synthesizability, Evo2 perplexity, CAI/taxon, MMseqs2 homology, and the k-mer novelty gate. Metric 6 (BiG-SCAPE) is an unimplemented stub, **excluded** from scoring and reserved only as optional downstream characterization.)

---

# Appendix A — TPU-compatibility audit (why the port is retired)

Evo2-7B runs on the `vortex` StripedHyena-2 runtime; every compute-heavy operator is a hand-written CUDA/Triton kernel with no XLA/TPU backend:

| Bottleneck | What it is | TPU status |
|---|---|---|
| **FlashFFTConv** | long FFT convolutions (the defining Hyena operator) | custom CUDA |
| **flash_attn_2_cuda** | attention blocks (5 of 32 layers) | CUDA-only |
| **causal_conv1d** | short causal depthwise conv | CUDA `.cu` + Triton |
| **hyena_se Toeplitz** | short-explicit operator | Triton (no TPU backend) |
| **transformer_engine** | FP8 path | NVIDIA-only |
| **DeepSpeed ZeRO-2** | training engine | GPU-centric |

A single-student parity port — especially the data-dependent **FFT long-conv**, the op that maps *worst* to the TPU MXU — is ~18-24 engineer-months with high odds of no converged result, and even on success the deliverable is "same model, different chip" (weak on the innovation criterion). **That is why it is retired.**

Verified config (for the record): `hidden_size 4096`, `num_attention_heads 32` -> `head_dim 128`; attention at layers `[3,10,17,24,31]`; 262k base / 1M stretch context; `use_flash_attn: True`.

*Honest upside, kept as a detail not a flagship:* `vortex` ships pure-PyTorch fallbacks (`torch.fft.rfft/irfft`, `F.conv1d`) that specify the numerics and lower to XLA — so a *reference* port is tractable and is exactly what makes **O4 (Evo2-Distill)** feasible (teacher stays on GPU; student is TPU-native).

---

# Appendix B — RETIRED v1 proposal (kernel port) — DO NOT SUBMIT

> Superseded 2026-06-10 by the TPU-native program in Part 1. Kept for history; Appendix A is the audit that justifies retiring it. The biology/impact framing here is still reusable; the *engineering* framing (porting kernels) is not.

# Programmable Biology on TPUs: An XLA-Native Generative Stack for Constrained Biosynthetic Gene Cluster Design

**Track 1 Research Award — Theme 2 (Applied Science) & Theme 3 (Open-Source Contributions)**
**Faculty Supervisor / Applicant:** [Faculty Name], [Department], [University Name] (recipient institution contact) · **PhD Researcher:** [PhD Student Name]

---

## 1. Introduction & Scientific Impact

Genomic language models have learned to *read* DNA — predicting variant effects, expression, and chromatin state (AlphaGenome, Nucleotide Transformer, Caduceus). The next frontier is *writing*: generating functional DNA on demand. This matters most for **natural-product drug discovery**: silent/cryptic biosynthetic gene clusters (BGCs) — the genomic blueprints for antibiotics — vastly outnumber the characterized active ones, yet the antibiotic pipeline has stagnated on constant rediscovery even as antimicrobial resistance escalates.

We propose to attack this bottleneck by **generating synthesis-ready BGC DNA on demand**, conditioned on compound class and taxonomy. The core scientific problem is *constrained* generation: a useful BGC must simultaneously be (1) correctly classified, (2) domain-complete, (3) **novel** (not memorized), (4) synthesizable, and (5) heterologously expressible. Evo2 already emits genome-scale DNA, but *unconstrained*, and prior BGC-specific generative work operates at the protein-domain level — no synthesis-ready DNA, no synthesizability/novelty/expressibility checks. Constrained, multi-criteria, class- and taxonomy-conditioned, anti-memorization-gated BGC generation is therefore an open niche. We have already built the infrastructure to validate it: a leakage-free genome-keyed data pipeline, a working H100 LoRA fine-tune of Evo2-7B, and an 8-metric in-silico acceptance suite (antiSMASH class, Pfam obligate-domain recovery, ESMFold foldability, DNA-Chisel synthesizability, Evo2 perplexity, taxon-faithful CAI, MMseqs2 homology, and a k-mer-containment novelty gate).

## 2. TPU Relevance & Open-Source Contribution

**AlphaGenome (JAX, TPU v3+, *Nature*, open-sourced Jan 2026) proves Google backs TPU genomics — but it *reads*.** The generative side has no XLA execution path today, and our project fills exactly that gap. Evo2-7B runs on the `vortex` StripedHyena-2 runtime, whose token-mixing operators are hand-written CUDA/Triton with no TPU backend: **FlashAttention**, **FlashFFTConv** (the long implicit Hyena convolution), and **`causal_conv1d` + `hyena_se` Toeplitz** short operators. Training relies on **DeepSpeed ZeRO-2** and CUDA-bound activation checkpointing — both GPU-centric.

We will build a **JAX/Flax + Pallas** port (`torch_xla` tracked as upside, not a dependency), framed as a **risk-tiered** program rather than a flat port:

- **Lower-risk assembly** — `vortex` ships pure-framework reference paths (`torch.fft.rfft/irfft`, `F.conv1d`) that fix the numerics and lower cleanly to XLA. Hyena-SE/MR map to `lax.conv_general_dilated` + chunked Toeplitz `einsum` (MXU-native); attention maps to Pallas Splash Attention. The model's `head_dim = 4096/32 = 128` already satisfies the MXU 128-lane constraint — a verified de-risking point, not an open risk.
- **Medium-risk systems work** — replacing DeepSpeed ZeRO-2 with XLA-native sharding (FSDPv2/GSPMD via `shard_map`, MaxText/Tunix patterns), porting CUDA checkpointing to `jax.remat`, and porting the autoregressive **generation path** (modal-FFT prefill + KV cache).
- **High-risk novel kernel** — among the Hyena operators, **only the long implicit convolution structurally needs FFT**. A naive `jnp.fft` long-conv carries a large peak-memory and a sharding-defeating all-gather penalty at long context (to be quantified against the GPU baseline in P0). Overcoming this with a **chunked/Monarch-decomposed Pallas long-conv kernel** is the genuinely novel deliverable.

This hits the Theme-3 archetypes by name: **Model Architects** (StripedHyena-2 → XLA), **Kernel Engineers** (Pallas FFT-conv + Toeplitz), **Tool Builders** (parity harness + eval suite). As of mid-2026, we are aware of no open-source TPU/Pallas long-FFT-conv kernel for StripedHyena-2-class models; the closest prior art, `irhum/hyena` (MIT JAX/Flax), demonstrates the FFT-conv pattern but lacks training-scale sharding, a chunked long-conv kernel, and generative caching.

## 3. Milestones

The plan interleaves engineering and biology, each gated by a measurable exit; validation is a tiered numerical→functional→empirical ladder.

| Phase / Qtr | Engineering exit | Biology exit |
|---|---|---|
| **P0 — Spec lock & parity harness** (Q3 2026) | Per-block operator map documented; naive-`jnp.fft` memory/sharding penalty benchmarked vs baseline; Tier-0 harness green on PyTorch self-consistency | GPU baseline eval-suite numbers (5 criteria) recorded as the parity target on frozen leakage-free split |
| **P1 — XLA reference port** (Q4 2026) | **M1:** per-token max-abs logit delta + held-out ΔPPL within pre-registered tolerance vs `vortex`; reference port + tests released | First-window-aligned PPL parity confirmed generation-relevant |
| **P2 — Pallas kernel + sharding** (Q1 2027) | **M2:** Pallas long-conv kernel cuts peak memory + retains sharding at ≥128k vs naive `jnp.fft`; full-model step under FSDPv2/GSPMD | — (systems phase) |
| **P3 — TPU LoRA + Tier-1 parity** (Q2 2027) | **M3:** TPU LoRA (r=16, bf16, L=32k) converges; generation path ported | All 5 criteria + conditioning adherence (top-k, MRR) match GPU baseline within CI |
| **P4 — Paper, release, capstone** (Q3 2027+) | **M4:** hardened TPU-native stack released | Stretch: wet-lab *E. coli*/*Streptomyces* expressibility of a novelty-gated panel |

**Tier-0** is numerical (logit/ΔPPL parity, pre-registered tolerances, tightest on fp32 FFT-conv); **Tier-1** is functional (the five criteria on a matched generation panel, run on TPU-agnostic host tools as an independent check); **Tier-2** (wet-lab) is a non-gating capstone, so feasibility is never hostage to strain/BSL logistics. A staged fallback protects the schedule: ship the FFT-free attention + short/medium operators first and keep Hyena-LI on padded `jnp.fft` at base context for the fine-tune milestone, staging the production long-conv kernel in afterward — so a hard kernel never blocks training/eval.

## 4. Deliverables & Venues

All code released under **Apache-2.0/MIT**: the **first XLA-native generative StripedHyena-2 (Evo2-7B) reference port**; a **Pallas long-convolution kernel** (chunked/Monarch FFT-conv) for the 262k base context and 1M stretch; a **Pallas Toeplitz / short-causal-conv operator library** reusable by any TPU SSM/long-conv genomic LM; an **XLA sharding + Tunix LoRA recipe** (FSDPv2/GSPMD replacing DeepSpeed ZeRO-2) for 7B at L=32k on a v6e slice; a **GPU↔TPU parity harness** with pre-registered tolerances and golden tensors; and the **multi-criteria BGC generation + evaluation pipeline**.

**Target venues (deadlines approximate, each following its enabling phase):** a **NeurIPS 2026 workshop** (~Sep 2026) or **MLCB 2027** for the operator-port + Tier-0 parity result; **MLSys 2027** (~Oct 2026) or **ICML 2027** (~Jan 2027) for the Pallas-kernel + sharding systems paper; **RECOMB 2027 / ISMB-ECCB 2027** or a 2027 *Nature Methods*/NAR software paper for the biology + parity protocol.

## 5. Resource Request & Budget

**TPU request (start small, ramp).** **Phase A** (P0–P1, operator port + Tier-0 parity) on **v5e-8 / v6e-8**, ~1,500–3,000 chip-hours, largely covered by **TRC free quota**. **Phase B** (P2–P4, full XLA training + Tier-1 parity) on **v6e-16 → v6e-32** (Trillium, 32 GB HBM/chip): our H100 job (eff. batch 128 at L=32768, batch 1 × grad-accum 128 on 1×H100, ~45 GB *before* sharding) maps onto a v6e slice once FSDPv2/GSPMD shards the footprint across chips; projected ~8–12 h/epoch (vs ~72 h on H100) is **contingent on the Pallas kernel reaching MXU throughput** and will be re-baselined at M2. Budget ~5,000–15,000 chip-hours; optional **v5p** for continued-context work toward the 1M stretch. We apply to **TRC in parallel**; the paid line below is a deadline-burst backstop, returned if TRC fully covers compute.

**Unrestricted gift — $60,000** (no overhead/indirect; ~100% to research):

| Line | Amount | Note |
|---|---|---|
| PhD researcher stipend + fringe | $42,000 (70%) | [PhD Student Name], full support (12 mo) at the department's standard annual stipend + fringe; the person doing the XLA/Pallas port + TPU training |
| Cloud overflow / paid v6e burst | $9,000 (15%) | ~3,000–4,000 chip-hrs at preemptible/spot v6e (~$2.5/chip-hr); contingent, returnable if TRC suffices |
| Conference travel (1–2 trips) | $6,000 (10%) | NeurIPS / ICML / MLSys / MLCB |
| Contingency | $3,000 (5%) | Pricing/quota variance |
| **Total** | **$60,000** | Within the $25k–$100k envelope |

**Why fundable now:** the numerics are de-risked by `vortex`'s own torch fallbacks, the eval suite and GPU baseline already exist, and the Tier-0 parity milestone gives concrete early evidence a single student can land the port — while producing durable open-source infrastructure (Pallas kernels, XLA StripedHyena-2, parity harness) that any TPU genomic-LM effort can reuse.
