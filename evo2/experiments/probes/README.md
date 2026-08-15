# Probe & experiment runners (lab-box / single-H100 reference)

These are the exact scripts used for the 2026-07 probe program on the lab H100
(`gputee`). Paths are hardcoded to the lab box (`/data2/ds85/...`); they are kept
as **reference** for what was run, not for re-execution on Quartz. See
`docs/plan.md` (current) and `docs/archive/pre-framework/progress.md` for results and
`docs/quartz_setup.md` for the Quartz long-context plan.

| script | what it ran | result |
|---|---|---|
| run_probe_chain.sh | P0 / B / C / D sweep (350 steps, L=16384) | C flickered, rest flat |
| run_ptag.sh | P-tag: constant class tag vs `\|CONTINUATION\|` | ruled out |
| run_geneaware_ab.sh | blind vs gene-aware chunking A/B | ruled out |
| run_concentration.sh | mega_all + n=15 re-eval of P0/C | C's win was n=6 noise |
| run_ranksweep.sh | LoRA r=64 / r=128 | capacity ruled out |
| run_optA.sh | real mega-only whole-core run (L=32768, self-gating) | auto-killed at epoch 4 |
