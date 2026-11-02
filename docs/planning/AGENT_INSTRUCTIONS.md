# CLAUDE.md — Agent instructions for the Causally-Useful-Compression project

You are the executing agent for Nura's research project. **`MASTER_PLAN.md` is the single source of truth** — it supersedes `COMPRESSION_INTELLIGENCE_RESEARCH_PLAN (1).md` (the v2 you have been following). Read it in full before acting. This file tells you what changed, what code was written outside your session, and what to do next.

## Your operating rules (non-negotiable)
1. **Gates are hard.** Never enter a phase whose gate hasn't passed. Never reinterpret a gate criterion after seeing results. Pre-registrations are frozen once written.
2. **Family (+seed) is the independent unit.** Never treat generated grids as samples. Cluster bootstraps and hierarchical models throughout.
3. **No proxy-switching.** One MDL proxy project-wide. If a result needs a different metric to appear, it isn't a result.
4. **Negative outcomes are deliverables.** Every branch in the master plan ships something; report failures at the same fidelity as successes.
5. **Replication data is never development data.** Phase 5 / R.6 / U.4 run the frozen pipeline with zero tuning.
6. **Do not cut P3.2 (rLLC under do(C)/do(U)) or pre-registration steps under time pressure.** These are the highest-novelty and highest-integrity items respectively.
7. When you finish a work item, update the status ledger in `MASTER_PLAN.md` (✅/🔄/⏳) and commit. Keep the immutable JSONL run-record discipline for every training/eval run, including Track R.
8. Report style: lead with the decision-relevant number, then per-family breakdown, then caveats. Flag anything that smells like leakage or a broken generator immediately — do not paper over it with more models.

## What changed vs the v2 plan you were following
- **P1.2 stays 4 baselines for the in-flight sweep** (array 48067). Do not resubmit. The 5th baseline — **CompressARC's per-puzzle MDL objective run inside our harness** — is now **P2.0**, first item of Phase 2.
- **Phase 4 symbolic arm = MADIL** (Ferré 2025, open source), not a hand-rolled DSL; fallback to typed-DSL search only if MADIL can't be adapted in ~3 days. H3 is now CompressARC↔MADIL semantic convergence.
- **P3.2 upgraded:** plain LLC → **refined LLC (rLLC) measured w.r.t. the do(C) vs do(U) distributions** (Wang et al. ICLR 2025; `devinterp` tooling; validate on known-RLCT toys first).
- **Three new workstreams:** Track R (RSMI on generators — code handed off below), Track U (task universality classes — rides on Phase 2 data), B-6.5 (selective c-function — Phase 6, gated).
- **Ledger schema:** add nullable `rsmi_relevance` column `{unconditional, conditional, coverage_conditioned, positions_aggregated}` at the next schema touch.
- Framing corrections for any text you draft: never claim "compression ≠ generalization" as novel (Ferré owns it); identifiability is cited, never claimed; the physics motivation is never a premise.

## Immediate task queue (in order)
1. **When array 48067 lands:** aggregate → run the pre-registered GATE 1 driver (family-clustered bootstrap) → report PASS/FAIL per criterion with per-family plots. If FAIL: diagnose which families/criteria, branch B-1.1 (fix the *generator definition*, loop GATE 0), never add models to rescue a broken family.
2. **Import Track R** into the repo under `trackR/` (files + patched RSMI-NE vendored or pinned as a submodule with the two patches recorded — see below). Add its deps as a separate env (`env-trackR`, TF-based; do not mix with the torch `agi` env).
3. **Track R next steps** (all model-free, can run on CPU or spare cluster nodes, parallel to Phase 2):
   a. Coverage-conditioned scorer (spec below).
   b. Multi-position × ≥5-seed protocol; aggregation = mean for global vars, coverage-conditioned max for local vars.
   c. **Write and freeze `trackR/GATE_R_PREREGISTRATION.md` before any run you intend to believe** — criteria: on ≥2 designed families, long-range causal variable ranks above local nuisance, direction + 95% CI excluding zero across seeds; geometry sweep (block size, buffer, env) pre-specified.
   d. R.5 conditional RSMI (spec below) — Track R's most novel measurement.
4. **Phase 2 kickoff** (after GATE 1 passes): P2.0 CompressARC-in-harness; P2.1 expand families; P2.2 scrubbing design; Track U signature matrix (free — the harness already emits per-family intervention-response vectors).

## Track R code handoff (written in a chat session on 15 Jul; smoke-tested end to end)

### Files
- **`track_r_adapter.py`** — the core adapter.
  - `MirrorMarkerFamily` — planted family designed for GATE R: `axis` (causal, long-range mirror symmetry), `marker_col` (causal, local single cell), `noise_seed` (nuisance, local salt-and-pepper), `bg_col` (nuisance, spatially global — the deliberate physics-vs-task trap). `LEDGER` dict annotates `role` × `spatial` per variable; `do(p, var)` resamples one variable for matched intervention pairs.
  - `arc_to_potts(grids)` — `(N,L,L)` colors 0–9 → `(N,L,L,9)` one-hot. **Convention: Nq−1 = 9 channels; the 10th state is the all-zeros redundant one.**
  - `make_rsmi_dataset(grids, ll, index, buffer_size, env_size)` — wraps `rsmine.coarsegrainer.build_dataset.dataset` with in-memory configs. **Critical details:** pass `visible_dim=1` and explicit `shape=(L,L,9)`; `cap = ll + 2*(buffer+env)` and the assert keeps the window inside the grid because `partition_x` wrap-pads (periodic BC — wrong for ARC; the cap sizing is the mitigation, and it constrains valid block positions to the grid interior).
  - `train_filter(V, E, dp, ...)` — calls `cg_opt.train_RSMI_optimiser`. **`CG_params` must include `"nonlinearCG": None`** (undocumented required key). `Nq=None` in CG_params = binary coarse variable (sufficient for the pilot).
  - `coarse_output(filt, grids, ll, index)` — replicates the package's exact contraction `einsum('tijad,ijab->tbd')`: the spatial filter applies **per one-hot channel**, so the coarse variable's deterministic part is a 9-vector of logits per sample.
  - `rsmi_relevance(family, filt, ll, index)` — the third ledger column, pilot version: mean normalized L2 shift of the coarse logit vector under matched `do(var)` pairs.
- **`ising_calibration.py`** — R.1. Wolff cluster sampler at T_c (correct at criticality) + vectorized checkerboard Metropolis for T=5 (Wolff decorrelates too slowly at high T — that bug produced spurious MI before the fix). **Calibration PASSED:** 3/3 seeds recover the uniform-sign Kadanoff block-spin filter at T_c (CV 0.02–0.18); MI(T_c) > MI(T=5). **Binding lessons:** (1) absolute MI values are meaningless — estimator floor ~0.4 nats observed; only matched-settings comparisons count; (2) always verify sample decorrelation for MC data (ARC generators are i.i.d., so exempt). Rerun this calibration whenever RSMI hyperparameters change.
- **`multiposition_sweep.py`** — 9-position sweep with `MirrorMarkerFamilyV2` (adds `marker_pos` as a local nuisance; marker position randomized in the interior). **Results (single seed — instrument-building, not evidence):** `axis` ≈ 0.56 kept at every position; `bg_col` ≈ 1.39 dominates everywhere (unconditional RSMI tracks *spatial range*, not task role — this motivates R.5); `marker_col` ≈ 0.01 — **diluted, not measured**: a 2×2 block covers a random marker in ~1.6% of samples, so mean-over-pairs washes out rare-coverage local variables.

### Environment & patches (pin these)
- Separate TF env: `tensorflow-cpu` (2.21 used), `tensorflow-probability`, `tf-keras`, `networkx`, `tqdm`, `wandb`; run everything with `WANDB_MODE=disabled`.
- RSMI-NE (github.com/RSMI-NE/RSMI-NE) needs two local patches: (1) in `rsmine/coarsegrainer/cg_optimisers.py`, make the `WandbCallback` import optional (upstream imports a removed `wandb.keras` path); (2) callers must supply `CG_params["nonlinearCG"] = None`. Vendor the patched copy or maintain a patch file; record the upstream commit hash in the pinning manifest.

### Specs for the next Track R implementations
- **Coverage-conditioned scorer (R.3a).** For a variable with spatial support S(p) (e.g., the marker cell): partition do-pairs into covered (S ∩ block ≠ ∅ in *both* base and intervened renders) vs uncovered. Report `coverage_rate` and `relevance | covered` separately; the ledger column stores both. For variables without localizable support (seeds, global colors), coverage = 1 by definition. Acceptance test: with the scorer, `marker_col`'s covered-relevance must be measurable (nonzero or credibly zero with tight CI) rather than dilution-zero — that distinction is what GATE R needs ("RSMI discards it" vs "block never saw it").
- **Conditional RSMI (R.5).** Two pre-registered variants, compared: (a) joint encoding — append the task output grid as extra channels/sites to the configuration and rerun RSMI (information shared with environment *and* output); (b) conditional — restrict the ensemble to a fixed output equivalence class and compare filters across classes. Deliverable: per-variable `(unconditional, conditional)` relevance pairs; the gap is the physics-vs-task decomposition. Prediction to test first: `bg_col`'s relevance should collapse under (b) if it is truly task-irrelevant; `axis`'s should not.
- **Aggregation (R.3b).** Per-position filters (the package's disordered mode); ≥5 seeds; report mean over positions for global variables, coverage-conditioned max for local ones; MI stability across positions as a sanity diagnostic (observed spread 1.66–1.99 nats in the pilot).

## Cluster notes you already know (kept for continuity)
tcsh login shell → `bash -s` heredocs for remote commands; cu126 wheels only, `--no-cache-dir` (pip cache silently restores cu130); QOS 8 jobs/user → pack array tasks; `GROUPS` is a bash read-only special variable. Track R jobs are CPU-friendly — schedule them on non-GPU capacity.
