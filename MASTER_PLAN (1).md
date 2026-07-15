# MASTER PLAN — Causally Useful Compression on ARC
**Single source of truth. Supersedes `COMPRESSION_INTELLIGENCE_RESEARCH_PLAN (1).md` (v2) and the v3/physics-track documents by merging them.**
Last updated 15 July 2026, ~evening HKT. Status marks: ✅ done · 🔄 running · ⏳ next · ▢ later.

---

## 0. The bet, and what it is not

**The bet (shippable, falsifiable):** A representation that preserves the variables which *causally determine* an ARC output — and is invariant to nuisance interventions — generalizes better than equally accurate alternatives; low-dimensional neural representations and short symbolic programs recover the *same interventional semantics* when both generalize.

**Framing correction (load-bearing):** "Compression ≠ generalization on ARC" is *published prior art* (Ferré 2021–2025: the most-compressive model sometimes fails on test). Never claim that observation. The contribution is the **diagnosis**: isolate the *causal component* of compression via interventions + shortcut reversal, and show that component — tied to selective rate allocation and effective DOF — predicts generalization. Ferré noticed the gap; we explain it.

**Not a leaderboard entry.** 2025's competitive theme is refinement loops (TTT, evolutionary synthesis). This is a mechanism study. Use the ARC Prize report's framing: accuracy gap = engineering; efficiency gap = "science and ideas" — we are the latter. Also: ARC-AGI-1 is contaminated by LLM memorization; audited procedural generators with planted ground truth sidestep contamination by construction. Say so in the intro.

**The private north star** ("intelligence is compression", "IB=RG in neural nets", "information becomes law") is the *source of hypotheses*, never a claim. The Intellectual-Motivation appendix never ships as-is. Scope facts: IB=RG (Gordon et al. 2021, PRL 126.240601) is proven **for field theories only**; whether analogous coarse-graining happens in learned systems is the open empirical question Tracks R and Phase 6 test.

**Identifiability posture:** CRL identifiability (Ahuja ICML 2023; Squires/Uhler; von Kügelgen; Li–Kaba–Ravanbakhsh AISTATS 2025 — defines identifiability for exactly our contrastive before/after design) is cited as *license* in one paragraph. We never claim, prove, or extend identifiability theory.

**If HR + H1 + HU all land, the flagship narrative:** *three independent definitions of relevance — field-theoretic (RSMI), interventional (causal ledger), and learned (network representation) — select the same variables on ARC, and their agreement predicts generalization.* The earned form of "information becomes law."

---

## 1. Status ledger (15 Jul 2026)

| Item | Status |
|---|---|
| **P0 harness, 10 families' ledgers, GATE 0** | ✅ passed (incl. planted-shortcut detection, causal-mask recovery) |
| **P1.1** 10 planted families (F1–F10) | ✅ all pass GATE 0 machinery; latent polyomino-placement bug found & fixed |
| **do(U)+recombination mechanism** | ✅ dataset-level spurious barcode (couples to joint causal code at train, recombined at shift); validated on F1_translate: oracle 0.96→0.96, raw 0.69→0.02, object-centric 0.82→0.86, probes 1.00/0.39/0.98 |
| **P1.2** 4-baseline × 10-seed × 10-family sweep (400 runs) | 🔄 SLURM array **48067**, 5 tasks × 2 families, ETA ~90 min from submission |
| **P1.3** ID estimators (TwoNN, Levina–Bickel) validated on known manifolds | ✅ incl. documented high-d underestimation (cube_d8 → ~6.6) |
| **P1.4** pre-registration frozen before gap results | ✅ `causalarc/pilot/PREREGISTRATION.md` |
| **GATE 1 decision** | ⏳ aggregate sweep → family-clustered bootstrap → pre-registered criteria |
| **Track R adapter + Ising calibration + 9-position sweep** | ✅ built & smoke-run (chat session, files below); calibration PASS |
| **Track R: coverage-conditioned scorer, conditional RSMI, GATE R pre-reg, multi-seed** | ⏳ next in Track R |
| Infra: repo `github.com/nurkyzaz/AGI`; SLURM cluster (A6000s, cu126); conda env `agi`, torch 2.6.0+cu126 | ✅ live |

**Reconciliation note (v2→master):** the in-flight sweep follows v2's 4-baseline P1.2. Do **not** kill it — GATE 1's criteria don't require CompressARC. The 5th baseline (CompressARC-in-harness) moves to **P2.0** below.

---

## 2. The tree at a glance

```
P0  Specification & audit ✅──────────────► GATE 0 ✅
      │
P1  Feasibility pilot (10 families) 🔄────► GATE 1: signal exists at all
      │            ║
      │            ║  TRACK R (parallel, model-free): RSMI on generators
      │            ║    R.1 Ising calibration ✅ · R.2 geometry spec ⏳
      │            ║    R.3 relevance column ⏳ · GATE R ⏳ · R.5 conditional ▢
      ├─ FAIL ─► B-1.1 repair ─► loop P0 · (irreparable) ─► LEAF-N0
      │
P2  Core causal-compression test ─────────► GATE 2 (H1): causal score predicts
      │   (20–30 fams; +CompressARC P2.0;    shifted accuracy, survives
      │    +Track U signature matrix)        shortcut reversal
      ├─ H1+ ──────────────────────────────────────────────────┐
      ├─ H1− but MDL predicts OOD ─► B-2.1 ─► LEAF-A           │
      └─ H1−, nothing predicts OOD ─► B-2.2 ─► LEAF-N1         │
                                                               │
P3  Precision allocation & effective DOF ◄─────────────────────┘
      │   H2 + rLLC w.r.t. do(C)/do(U) ───► GATE 3
      ├─ H2+ ─► LEAF-P strengthens · H2− ─► B-3.1
      │
P4  Symbolic arm (MADIL) & convergence ───► GATE 4 (H3): CompressARC ↔ MADIL
      ├─ CONVERGE ─► LEAF-P · DIVERGE ─► B-4.1 taxonomy ─► LEAF-A/P
      │
P5  External validity (RE-ARC/ARC-TGI; ────► GATE 5: Track-A effects replicate
      │   ConceptARC opt.; R.6, U.4)
      │
P6  ONLY IF core positive: physics & action
      ├─ B-6.1 IB phase transitions      ├─ B-6.2 RG-like layers (+profiles)
      ├─ B-6.3 LLC developmental traj.   ├─ B-6.5 selective c-function
      └─ B-6.4 ARC-AGI-3 active extension
```

---

## PHASE 0 — ✅ done. Invariants that remain binding
- Frozen: ledger schema, DSL grammar, seeds, metrics, immutable JSONL run records (generator hash, split hash, seed, model config, compute, per-intervention predictions).
- **Schema amendment (do at next ledger touch):** add nullable `rsmi_relevance` column (Track R's third column) with fields `{unconditional, conditional, coverage_conditioned, positions_aggregated}`.
- One MDL proxy for the whole project (locked at P0.4). Pin exact revisions of RE-ARC, `arc-dsl`, ARC-TGI, **CompressARC, MADIL, RSMI-NE (with local patches — see infra appendix)**.
- GATE 0 criteria (all held): interventions pass generator tests; planted shortcut invisible to i.i.d. but detected under reversal; causal-mask precision/recall ≈ 1.

## PHASE 1 — 🔄 sweep in flight
- **GATE 1 (pre-registered, decide when array 48067 lands):** (1) raw baseline drops meaningfully under do(U)+recombination; (2) oracle beats raw under shift; (3) ≥1 causal-score metric stable across seeds. Family-clustered bootstrap; family = independent unit.
- Branches: FAIL → B-1.1 generator repair (loop GATE 0), irreparable → LEAF-N0 ("generators define labels, not identifiable causal factors").
- Preview evidence (F1_translate, single-family — not a gate result): mechanism works, probe separates baselines cleanly.

## TRACK R — RSMI on ARC generators (parallel; model-free; TF env)
**HR (relevance alignment):** RSMI-selected DOF ≈ ledger `C`; per-family alignment predicts that family's model generalization.
Three worlds, all publishable: alignment (IB=RG-in-learning at the data level) · structured divergence (spatial-range vs task-role — a nameable phenomenon) · no alignment (honest boundary on IB=RG universality → LEAF-RN).

**Done (this session, code in repo `trackR/` after import):**
- `track_r_adapter.py` — ARC=10-state Potts encoding, (V,E) extraction, filter training, relevance scorer. End-to-end smoke passed.
- `ising_calibration.py` — **calibration PASS**: 3/3 seeds recover uniform Kadanoff block-spin filter at T_c (CV 0.02–0.18); MI(T_c) > MI(T=5) after fixing hot-arm decorrelation.
- `multiposition_sweep.py` — 9-position sweep on `MirrorMarkerFamilyV2`; stable results: `axis` (causal, long-range) kept everywhere (~0.56); `bg_col` (global **nuisance**) dominates (~1.39) → unconditional RSMI tracks spatial range, not task role; `marker_col` diluted to ~0.01 by rare block coverage → scorer bug identified.

**Calibration lessons (binding rules):** report matched-settings MI comparisons only (never absolute values; estimator floor ~0.4 nats observed); verify sample decorrelation for any MC-sampled data (ARC generators are i.i.d. — automatic).

**Next (in order):**
- **R.3a Coverage-conditioned scorer** — for local variables, compute relevance only over do-pairs where the block covers the variable's support; report `(coverage_rate, conditional_relevance)` pairs. Required before GATE R can distinguish "RSMI discards it" from "block never saw it".
- **R.3b Multi-position, multi-seed protocol** — per-position filters (disordered mode), ≥5 seeds, aggregate mean (global vars) + coverage-conditioned max (local vars).
- **GATE R (pre-register BEFORE running):** on ≥2 designed families, RSMI must rank a long-range causal variable above a local nuisance (direction + CI excluding zero, seeds as replicates). Fail after pre-registered geometry sweep → LEAF-RN.
- **R.5 Conditional RSMI** — condition on (or jointly encode) the task output; the unconditional-vs-conditional relevance gap **is** the physics-vs-task decomposition (motivated directly by the `bg_col` result). This is Track R's most novel measurement.
- **R.6** frozen pipeline on 10–20 audited RE-ARC families (Phase 5; replication only).
- **Headline analysis:** per-family RSMI↔ledger alignment vs per-family generalization of Phase-2 models; one scatter, family-clustered bootstrap.

## PHASE 2 — Core test (after GATE 1)
> **H1:** at matched i.i.d. accuracy and resource budget, causal-subspace score predicts shifted-test accuracy.
- **P2.0 (v3 addition):** run **CompressARC's per-puzzle MDL objective inside the harness** on planted families — does its compression preferentially preserve `C` over `U`? Co-opts the nearest competitor as a datapoint. Mine its VAE-as-MDL derivation for the cond-IB baseline. Also: track down the invariance-augmentation ARC transformer (via arXiv:2606.12847) before finalizing equivariant augmentations.
- **P2.1** expand to 20–30 families; sweep β/capacity; checkpoints matched on train performance.
- **P2.2** mechanism, not mediation: randomize compression/prior/capacity/seed; **scrub** causal subspace vs dimension-and-norm-matched nuisance subspace; repeat after do(U). Scrubbing-causal-hurts-shift is the evidence.
- **P2.3** family-cluster bootstrap (10k) + hierarchical model; per-family plots alongside pooled.
- **P2.4** draft related work in parallel (CompressARC / ARC-TGI-as-infrastructure / Cluster-D CRL / Ferré) — the "cite or die" set.
- **Track U starts here (free):** stack per-family intervention-response vectors → signature matrix; cluster (pre-register method + gap statistic; ARI vs design templates). **U.2** transfer submatrix (~12–15 families, checkpoint evals, no retraining): does class predict transfer beyond surface similarity? **U.3** the risky prediction — hold out 3–5 families, predict their transfer profile from generator-side signatures alone. Pre-register with P1.4-style discipline.
- **GATE 2 (pre-registered):** causal-score effect on do(U)+recombination accuracy (controls: train acc, family, output entropy, model class) exceeds +5pp/SD with 95% cluster-bootstrap CI excluding it, and survives shortcut reversal.
- Branches: H1+ → P3 · B-2.1 (MDL predicts OOD without causal DOF → LEAF-A; partial convergence with Ferré — frame as explaining his boundary) · B-2.2 (nothing predicts, oracle does → LEAF-N1: bottleneck is inductive bias) · guardrail: no model incl. oracle shows a gap → generator invalid, return to P0/P1, do not interpret.

## PHASE 3 — Precision & effective DOF (overlaps P2)
> **H2:** extra rate goes to `C` not `U`, and this allocation predicts OOD beyond total rate.
- **P3.1** per-factor conditional rate R_C − R_U via intervention pairs; categorical/group-structured latent (Gaussian-variance semantics on discrete grids is a reviewer trap).
- **P3.2 — do not cut under time pressure.** **rLLC per intervention distribution** (Wang et al., ICLR 2025 Spotlight): estimate each component's rLLC w.r.t. do(C) vs do(U). High-under-do(C), low-under-do(U) = parameter-space signature of causal selectivity — the project's highest-novelty single measurement and the Timaeus scoop-defense. Tooling: `devinterp`; validate on shipped known-RLCT toys + rescaling-invariance check first. Pre-register as descriptive/corroborating. Cross-check: lower train-LLC ↔ higher causal score across families (novel SLT↔causal bridge either way).
- **GATE 3:** R_C − R_U > 0 reliably AND differential rate predicts OOD beyond total rate (nested comparison, family-clustered).
- Branches: H2+ → "selective causal rate allocation, not compression volume" headline · B-3.1 → report plainly, no proxy-switching; rLLC result stands alone in both branches.

## PHASE 4 — Symbolic arm & semantic convergence
> **H3 (sharpened):** the two independently developed MDL solvers — **CompressARC (neural)** and **MADIL (symbolic, Ferré 2025)** — agree on intervention-response signatures when both generalize.
- **P4.1** adopt **MADIL** in the frozen harness (saves weeks; non-strawman; two *real* systems). Fallback if not adaptable within ~3 days: v2's bounded typed-DSL search with explicit `L(P)+L(D|P)`; never start with full Hodel DSL (PeARL's warning). Cite Ferré 2021→IDA 2024→MADIL 2025 + DreamCoder. **Verify the "92%/72%" Ferré figure against the primary PDF before quoting.**
- **P4.2** coordinate-free convergence: intervention-response signature · causal-influence mask (P/R vs annotated) · probes/CCA (descriptive only). Never compare "latent axis 3" to "program arg 3".
- **P4.3** ≥2 DSL encodings; DSL library cost held constant or included.
- **GATE 4:** both select the same interventionally-relevant equivalence class on **held-out** interventions. Branches: CONVERGE → LEAF-P · B-4.1 taxonomy (geometric-not-programmatic / programmatic-not-geometric / both) → LEAF-A/P.

## PHASE 5 — External validity
- **P5.1** frozen harness on 30–50 audited RE-ARC/ARC-TGI families — replication, zero tuning. **P5.2** second annotator on 20% of ledgers; unknowns never force-labeled. **P5.3 (H4)** distill only after H1; student must preserve intervention signature + resource frontier. **P5.4 (opt.)** one ConceptARC descriptive pass. **R.6 + U.4** replication passes.
- **GATE 5:** Track-A effects reproduce in direction and rough magnitude; transfer holds out families by rule template. PARTIAL/FAILS → scope to planted families honestly; the gap is a finding about real-ARC ambiguity.

## PHASE 6 — Physics & action (gate: H1+ survived shortcut reversal; otherwise forbidden)
- **B-6.1** IB phase transitions: reproducible rate–distortion kink + coincident causal-score jump; "edge of chaos" needs its own dynamical observable.
- **B-6.2** RG-like layers, descriptive analogy only (read Saxe et al. 2018 first); absorbs the layerwise causal/nuisance sensitivity **profiles** (shapes, within-class vs between-class consistency by permutation test — never fitted exponents; the exponent idea is demoted permanently unless profiles are strikingly log-linear).
- **B-6.3** LLC developmental trajectory (Hoogland-style stages) — cross-wire with B-6.5.
- **B-6.5 Selective c-function.** Prior art: Erdmenger–Grosvenor–Jefferson (2107.06898) tried a relative-entropy monotone in DNNs, found it insensitive to phase structure, called for "more refined probes" — this is that probe. HC: nuisance-information decreases monotonically along depth/training while causal-information is preserved. Measurement (pre-registered, both reported): held-out probe accuracy per variable per layer (V-information footing) AND scrub-based damage. Never naive MI on deterministic activations (Saxe's swamp). Hunt violations honestly (nuisance re-entry at late layers). Scope sentence: "c-theorem-*inspired*; no theorem, charge, or fixed point claimed."
- **B-6.4** ARC-AGI-3 active extension: info-seeking actions beat matched random/exploitative per interaction. Separate follow-up; never a clause "confirmed" by the core.

---

## 3. Leaves & venues
| Leaf | Trigger | Venue |
|---|---|---|
| **LEAF-P** flagship | H1+ ∧ (H2+ ∨ H3 converge) ∧ P5 replicates | Primary **NeurIPS/ICML 2027**; ICLR 2027 (~24 Sep) stretch only if Gates 0–4 clear by ~5 Sep; ML4PS staging |
| **LEAF-A** refinement | H1− but MDL predicts OOD, or clean H3 divergence | Workshop / short paper / **ARC Prize 2026 paper track** |
| **LEAF-N0** methods | generators can't license a ledger | Workshop methods note |
| **LEAF-N1** negative | nothing predicts, oracle does | Workshop / reproducibility |
| **LEAF-RN** Track-R negative | GATE R fails after pre-registered sweep | Workshop: boundary on IB=RG universality |
| **Tool release** | GATE 0 ✅ | **Open-source the harness now** — primary scoop insurance (repo already public) |

Standing commitments: harness public at GATE 0 (✅ effectively done); workshop version submitted when GATE 2 resolves *either way*.

## 4. Analysis & reporting invariants
Family (+seed) = independent unit; cluster/hierarchical CIs; never grids-as-samples. Transfer claims hold out by rule template. Report all seeds, failures, revisions, probe calibration, per-family plots. Confirmatory H1–H4/HR/HU separated from exploratory β/criticality/LLC. One MDL proxy; DSL cost held constant. Every claim tagged proven / conjectured-empirical / speculative. Identifiability cited, never claimed. Motivation appendix never ships. MI: matched-settings comparisons only; decorrelation verified.

## 5. Literature anchors (merged; verified 13–15 Jul 2026)
**Substrate:** RE-ARC 2404.07353 · ARC-TGI 2603.05099 (infrastructure, not competitor) · Hodel arc-dsl · ARC Prize 2025 report 2601.10904 (efficiency-gap framing; ARC-AGI-1 contamination) · Living Survey 2603.13372 · ConceptARC.
**Co-opted competitors:** CompressARC 2512.06104 (paper to beat AND harness model; VAE-as-MDL) · **Ferré lineage (must-cite):** 2112.00848; IDA 2024 2311.00545 (verify figures vs primary); **MADIL 2505.01081** · SOAR/TRM/NVARC (context) · invariance-augmentation transformer via 2606.12847 (find primary).
**CRL license & antecedents (cite or die):** Locatello 2019 · Ahuja et al. ICML 2023 · **Li–Kaba–Ravanbakhsh AISTATS 2025** (read before freezing interventions) · Ahuja et al. 2022 IB×invariance · NuRD · IST 2306.11890 (falsification epistemics precedent) · KamonBench, CITRIS · Schölkopf 2021 · 2603.11907 = confirmed false alarm (multi-treatment ITE; cite once, differentiate in one sentence).
**SLT/DOF:** Watanabe · Lau et al. AISTATS 2025 · **Wang et al. rLLC ICLR 2025** (the instrument) · Hoogland 2024 · devinterp · Lehalleur 2025.
**Physics bridge (scope, not license):** Gordon 2021 PRL 126.240601 · Koch-Janusz & Ringel 2018 · Lenggenhager 2020 · Gökmen 2021 PRL/PRE (**RSMI-NE = Track R's instrument**), 2024 Nat. Comms. · Saxe 2018 + Mehta–Schwab graveyard · **Erdmenger–Grosvenor–Jefferson 2107.06898** (B-6.5 prior art + hook) · **Peraza Coppola–Helias–Ringel 2510.25553** (must-cite; differentiate: learning-curve RG vs task-causal structure).
**Universality neighbors (Track U):** Huh et al. 2024 Platonic · SAE mechanistic universality ICLR 2025 · universal object dims 2605.13675 · robustness-universality critique 2510.19427 — all model-side; Track U is task-side with ground truth.
**Compression/repr theory:** MDL generalization 2402.03254 · IB transitions 2001.01878 · intrinsic dimension 1905.12784 · manifold-dim scaling JMLR 23.

## 6. Pre-submission checklist
▢ CLeaR 2026 PMLR full pass ("causal + compression/MDL + generalization/reasoning") — first pass 15 Jul clean · ▢ ICML 2026 accepted list, same keywords · ▢ title-skim of the ~90 ARC Prize 2025 papers ("causal/intervention/disentangle/bottleneck") · ▢ verify Ferré 92%/72% vs primary PDF · ▢ re-verify 2026 arXiv IDs (ARC-TGI, KamonBench) · ▢ find primary source behind 2606.12847's augmentation reference · ▢ watch ARC Prize 2026 paper track for late causal/MDL entries.

## 7. Infrastructure appendix (state as built)
- **Repo:** `github.com/nurkyzaz/AGI` (public; PDFs + nested `cascadearc` gitignored). Track R files to be imported under `trackR/`.
- **Cluster:** SLURM, A6000 nodes (gpu:2/4/8), driver CUDA 12.6 → **torch builds must be cu126** (cu130 fails; pip cache silently reinstalls it — use `--no-cache-dir` + pinned build). Conda env `agi`: torch 2.6.0+cu126. Login shell **tcsh** → remote commands via explicit `bash -s` heredocs. QOS cap 8 jobs/user → pack families per array task. Known footgun (fixed): `GROUPS` is a bash read-only special variable.
- **Track R env (separate, TF-based):** tensorflow-cpu 2.21, tensorflow-probability, tf-keras, networkx, tqdm, wandb (run with `WANDB_MODE=disabled`). **RSMI-NE local patches to pin:** (1) `cg_optimisers.py` wandb import made optional (upstream uses removed `wandb.keras` path); (2) `CG_params` requires `"nonlinearCG": None`. **Data conventions:** configs `(N, L, L, Nq−1)` one-hot (last Potts state redundant); `visible_dim=1` with explicit `shape=(L, L, Nq−1)` to `rsmi_data`; `partition_x` wrap-pads → size `cap = ll + 2(buffer+env)` to fit inside the grid; filter applied as `einsum('tijad,ijab->tbd')` (spatial weights per one-hot channel → coarse variable is an (Nq−1)-vector of logits).
