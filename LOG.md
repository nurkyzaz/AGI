# Decisions & Results Log — causally useful compression on ARC

Dated entries, newest first. Retractions are marked RETRACTED, never silently
edited. Every gate evaluation also has an immutable JSONL record in
`runs_causalarc/`.

---

## 2026-07-15 — B-1.1 repair executed; v2 wave launched (SLURM 48093).

**Capacity fix validated by smoke (gate: oracle iid >= 0.50):** F6_rotate
0.18 -> 0.89 PASS with the deeper decoder (2 attention blocks, hidden 64,
4000 steps; F1 control 1.00). **F4_gravity failed its gate** (0.03 -> 0.22,
cell 0.988): gravity is per-column content *sorting*, not a fixed spatial
permutation — one more attention hop wasn't enough for exact-match. Per the
frozen prereg v2 rule the family is REPLACED, not retuned: **F4_cycle**
(cyclic whole-grid shift; C = direction x step; a fixed permutation the
architecture can express). GravityDrop stays in code, unregistered, as a
future hard-family candidate. F4_cycle passed GATE 0 in full
(record 20260715-151238, mask P=R=1.0) and its oracle smoke hit 1.00.

**Probe estimator v2:** probe_n 512->2048, 5-fold CV x 3 probe seeds.
Thresholds unchanged (prereg v2). v2 wave = 10 families x 10 seeds x 4
baselines, tag pilot2, array 48093. A second item-3 failure will be reported
as a finding, not looped again.

## 2026-07-15 — GATE 1: FAIL (item 3), items 1–2 PASS decisively. Branch: B-1.1 repair.

**Pilot:** 400 runs (10 families × 10 seeds × 4 baselines), SLURM array 48067
(5×A6000, ~1.7 h). Pre-registration frozen before results
(`causalarc/pilot/PREREGISTRATION.md`); thresholds mirrored in `gate1.py`.
Record: `runs_causalarc/20260715-145732_gate1.jsonl`.

**Item 1 (OOD gap exists): PASS.** Raw baseline iid→shifted drop = 0.268,
95% family-cluster CI [0.12, 0.44], 100% of families drop.
**Item 2 (knowing C helps): PASS.** Oracle−raw shifted margin = 0.453,
CI [0.25, 0.66].
**Item 3 (causal score stable across seeds): FAIL.** Mean cross-seed std of the
raw-model causal-probe = 0.144 > 0.10 (mean probe 0.567 ≥ 0.50 ok).

**Pre-registered exclusion applied:** F4_gravity (oracle iid 0.03) and F6_rotate
(oracle iid 0.18) — the oracle cannot fit its own family, so these are
uninformative about representations and return to P0/P1 (architecture/budget
repair, not science). Item 3 still fails on the 8 informative families
(mean std 0.132).

**Diagnosis (two separable causes):**
1. *Estimator under-sampling.* The probe uses 512 samples, single 50/50 split,
   one probe seed — the causal-probe number itself is noisy independent of the
   model. Stable families exist (F7 std 0.049, F10 0.045, F9 0.083), so the
   metric is not intrinsically broken.
2. *Genuine solution bimodality on geometric families* (F1 0.179, F3 0.228,
   F8 0.230): across seeds the raw model lands either on the barcode shortcut or
   on partial geometry — high probe variance there is real signal about raw
   models, not just measurement noise. (Consistent with item-1's large gaps on
   exactly those families.)

**Descriptive (not gated):** object-centric shifted 0.47 vs raw shifted 0.29 —
the nuisance-invariant representation is more robust, previewing H1.

**Decision:** B-1.1 repair loop, two changes, each pre-registered before re-run:
(a) fix F4/F6 learnability (more steps and/or capacity; if the oracle still
can't fit, replace the families and re-pass GATE 0); (b) strengthen the
causal-score estimator (larger probe_n, k-fold cross-validation, average over
probe seeds) — an *estimator* fix chosen for measurement quality, explicitly not
tuned on outcomes. Amended thresholds go into PREREGISTRATION.md as v2 with the
v1 result recorded above. GATE 1 re-run required in full.

## 2026-07-15 — Phase 1 pilot built; do(U) mechanism validated single-seed.

F1_translate: oracle 0.96→0.96, object-centric 0.82→0.86, raw 0.69→0.02
(gap +0.68), cond-IB 0.46→0.01. The spurious-barcode recombination works
exactly as designed. Full sweep launched.

## 2026-07-15 — GATE 0: PASS (all three items).

5 planted families; every C variable moves the output (rates 0.375–1.0), every
U variable exactly inert (0.0); planted shortcut 1.0/1.0/0.0 train/iid/shifted
with causal probe 1.0 everywhere; causal-mask recovery P=R=1.0 on all families.
Adversarial negative controls (leaky nuisance, dead causal) are caught by the
harness — 9/9 tests. Record: `runs_causalarc/20260715-113506_gate0.jsonl`.
Phase 0 harness committed and pushed (github.com/nurkyzaz/AGI).
