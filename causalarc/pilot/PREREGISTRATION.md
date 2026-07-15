# Phase 1 Pre-Registration (P1.4)

**Frozen: 2026-07-15, before observing any iid/shifted accuracy or OOD-gap result.**
At freeze time the only pilot output seen was that the oracle's rule code linearly
encodes C (causal-probe ≈ 1.0) — a wiring sanity check, not a hypothesis test.

This document fixes the hypotheses, analyses, exclusions, and thresholds for the
Phase 1 feasibility pilot so the GATE 1 decision cannot be moved after the fact.
Thresholds here are mirrored as constants in `gate1.py`.

## Design (fixed)

- **Track A:** 10 planted families (F1–F10), each with a C/U causal ledger that
  passed GATE 0 (interventions valid, mask recovered, shortcut detectable).
- **Task:** few-shot — k=3 demonstration pairs reveal the latent rule C; the model
  applies it to a query input and predicts the query output grid.
- **OOD split (do(U)+recombination):** a spurious barcode in the top border row
  encodes the joint causal index at train/iid and is resampled independently at
  shifted-test. This is the nuisance whose train-correlation is broken at test.
- **Baselines (4):** `oracle` (given C), `raw` (full grids), `object_centric`
  (border stripped → nuisance removed), `cond_ib` (variational bottleneck, β=1e-2).
- **Seeds:** 10 per (family, baseline). Independent unit = **family** (never grids).
- **Primary metric:** exact-match grid accuracy (ARC-faithful). Cell accuracy is
  secondary/descriptive.
- **Causal-score metric:** mean linear-probe accuracy decoding the C latents from
  the rule code z (nuisance-probe on U reported alongside).

## Hypotheses & GATE 1 items (confirmatory)

**H(pilot)-1 — an OOD gap exists.** The raw baseline drops from iid to shifted.
- PASS iff pooled raw (iid_exact − shifted_exact) ≥ **0.15**, with 95% family-cluster
  bootstrap lower bound > **0.05**, AND ≥ **70%** of families show raw shifted < iid.

**H(pilot)-2 — knowing C helps under shift.** The oracle beats raw at shifted-test.
- PASS iff pooled (oracle_shifted − raw_shifted) ≥ **0.15**, 95% family-cluster
  bootstrap CI excluding 0.

**H(pilot)-3 — a causal-score is stable, not seed-noise.**
- PASS iff mean cross-seed std of the causal-probe ≤ **0.10** AND mean causal-probe
  ≥ **0.50** (well above per-family chance).

**GATE 1 passes iff all three hold.** All are pre-registered; the β sweep,
criticality, and LLC analyses are explicitly *exploratory* and out of scope here.

## Analyses (fixed)

- Family-clustered percentile bootstrap (10,000 resamples), family = unit.
- Report raw per-family effects alongside pooled estimates.
- object_centric vs raw at shifted-test is reported **descriptively** (supports the
  nuisance-invariance reading) but is **not** a gate item.

## Exclusions (fixed)

- A run is excluded only if it errors or the oracle fails to fit its own family
  (oracle iid_exact < 0.50) — that indicates the task/architecture cannot express
  the rule, so the family is uninformative for H2 and returns to P0/P1 (the plan's
  guardrail branch), rather than being interpreted.
- No metric tuning, no threshold changes, no proxy switching after this freeze.

## Branch map (from the research plan)

- **All three PASS →** signal exists; proceed to Phase 2 (core causal-compression
  test on 20–30 families).
- **Gate fails on a broken generator →** B-1.1 repair, loop through GATE 0.
- **Irreparable (generators can't license a causal DOF) →** LEAF-N0 methods note.
