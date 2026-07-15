# Causally Useful Compression on ARC — A Phased Research Tree

**Version 2 · restructured as a decision tree with phases, branches, and gates**
Last updated 13 July 2026.

---

## 0. Read this first: the one-sentence bet, and what it is *not*

**The bet (shippable, falsifiable):** A representation that preserves the variables which *causally determine* an ARC output — and is invariant to nuisance interventions — generalizes better than equally accurate alternatives; low-dimensional neural representations and short symbolic programs recover the *same interventional semantics* when both generalize.

**What this is NOT (the north star, kept private):** "Intelligence is compression," "IB = RG in neural nets," "information becomes law." Those are the *source of hypotheses*, not claims this project asserts. The published unit stands or falls on the interventional experiment below, never on the intuition that generated it.

**Why the distinction is load-bearing.** A zip file compresses. A lookup-table memorizer compresses. Neither is intelligent. The testable object is *resource-bounded, task-relevant compression under interventions* — not compression per se. Every gate below exists to stop the north star from smuggling itself back in as an unfalsifiable headline.

**Scope honesty on the physics bridge.** The IB=RG equivalence (Gordon et al. 2021, PRL 126, 240601) is *proven for field theories*, where IB-relevant modes = lowest-scaling-dimension operators. It is **not** a theorem about ARC networks. Whether an analogous coarse-graining happens in an ARC-trained net is an *open empirical question* this plan tests in Phase 5 — it is never a premise.

---

## 1. How to read the tree

```
PHASE  = a block of work with a single purpose and a hard gate at the end.
GATE   = a pre-registered pass/fail decision. You are NOT allowed to enter
         the next phase without meeting it. This is the anti-self-deception
         mechanism.
BRANCH = a fork taken CONDITIONAL on a gate outcome. Named B-x.y.
LEAF   = a terminal deliverable (a paper, a negative-result note, a tool release).
```

The spine is Phases 0→6. Most phases fan out into 2–3 branches depending on what the gate returns. **A "negative" gate is not failure — several branches lead to publishable negative or methodological results.** The plan is designed so that *every leaf ships something*.

---

## 2. The tree at a glance

```
P0  Specification & audit ───────────────► GATE 0: harness is trustworthy
      │
P1  Feasibility pilot (10 families) ──────► GATE 1: signal exists at all
      │
      ├─ FAIL ─► B-1.1  Repair generators / redefine causal ledger ─► loop to P0
      │          (if irreparable) ─► LEAF-N0: "ARC generators don't
      │                              license clean causal DOF" (methods note)
      │
P2  Core causal-compression test ─────────► GATE 2 (H1): causal score
      │   (20–30 families)                   predicts shifted accuracy,
      │                                       survives shortcut reversal
      │
      ├─ H1 POSITIVE ─────────────────────────────────────────────┐
      │                                                            │
      ├─ H1 NEGATIVE but MDL predicts OOD ─► B-2.1 ─► LEAF-A       │
      │                                                            │
      └─ H1 NEGATIVE, nothing predicts OOD ─► B-2.2 ─► LEAF-N1     │
                                                                   │
P3  Precision-allocation & effective-DOF ◄─────────────────────────┘
      │   (H2 + LLC/SLT instrument) ───────► GATE 3: rate & effective
      │                                       DOF go to C not U
      │
      ├─ H2 POSITIVE ─► main paper strengthens ─► toward LEAF-P (flagship)
      ├─ H2 NEGATIVE ─► B-3.1: "relevance matters, precision story is wrong"
      │
P4  Symbolic arm & semantic convergence ──► GATE 4 (H3): neural & symbolic
      │                                       agree on intervention signatures
      │
      ├─ CONVERGE ─► strong "same object" evidence ─► LEAF-P
      ├─ DIVERGE  ─► B-4.1: taxonomy of failure modes ─► LEAF-A/P
      │
P5  External validity + efficiency ───────► GATE 5: Track-A effects replicate
      │   (RE-ARC / ARC-TGI, distillation)   on audited real generators
      │
P6  ONLY IF core positive: physics & action
      ├─ B-6.1 IB phase transitions (β sweep, reproducible kink)
      ├─ B-6.2 RG-like layerwise coarse-graining (descriptive)
      ├─ B-6.3 LLC developmental trajectory over training
      └─ B-6.4 Active/interactive extension on ARC-AGI-3
```

---

## PHASE 0 — Specification & audit (Week 1)

**Purpose.** Build a harness you can trust *before* you have any result to be excited about. Every later gate is only as honest as this phase.

### Work items
- **P0.1** Freeze the causal-ledger schema, DSL grammar, seeds, metric definitions, and the immutable JSONL run-record format. Each record carries: generator hash, split hash, seed, model config, train compute, inference/search compute, predictions for *every* intervention, and all metrics.
- **P0.2** Implement 5 planted families. Each ships with a test verifying every declared intervention actually changes (or preserves) the target, and labels each variable relevant vs nuisance.
- **P0.3** Pin exact revisions of RE-ARC, `arc-dsl`, ARC-TGI. Inspect 10 families and complete their ledgers: `family | generator vars | candidate C/U | valid interventions | reference program | rule type | unresolved ambiguity`.
- **P0.4** Lock the *single* MDL proxy you will use throughout (recommend: LZ-78 on fixed-precision amplitude/state bitstrings, pre-registered). Neural KL, program tokens, and zipped weights are **not** interchangeable absolute bit-units — this decision prevents you from silently switching proxies to chase a result.

### ► GATE 0 (harness is trustworthy) — ALL must hold:
1. Every declared intervention passes its generator test.
2. A **planted shortcut** (a nuisance perfectly correlated with the answer at train) is invisible to i.i.d. accuracy but *fails under correlation reversal at test*. If your harness can't detect this on a case you planted yourself, it cannot be trusted on real data.
3. The metric code recovers the known causal mask on the planted families (precision/recall ≈ 1 on ground truth you inserted).

**If GATE 0 fails:** you have a tooling bug, not a science result. Fix the harness. Do not proceed.

---

## PHASE 1 — Feasibility pilot (Weeks 2–3)

**Purpose.** Establish that there is *any* signal worth scaling to 30 families, before spending the compute.

### Work items
- **P1.1** Expand Track A to 10 planted families.
- **P1.2** Run four baselines across 10 seeds each: raw grid transformer/CNN, object-centric/equivariant, rate–distortion (conditional-IB) model, and an **oracle** that gets `C` directly (upper bound).
- **P1.3** Validate every intrinsic-dimension estimator (TwoNN, Levina–Bickel MLE, local ID) on synthetic manifolds of *known* dimension — spheres, tori, product-categorical factors. Report where each estimator breaks. ID is a *descriptor*, never ground truth.
- **P1.4** **Pre-register** final hypotheses, analyses, exclusions, and the practical effect threshold (default +5pp shifted-test accuracy per SD of causal score) — *after* the pilot design is fixed, *before* seeing final results.

### ► GATE 1 (signal exists at all) — ALL must hold:
1. Interventions produce a nontrivial OOD gap (raw baseline drops meaningfully under `do(U)` + recombination).
2. Oracle beats the raw baseline under shifts (if knowing `C` doesn't help, your task family has no causal structure to recover).
3. At least one causal-score metric is stable across seeds (not seed-noise).

### Branches
- **B-1.1 (repair):** If a gate item fails because a generator is ill-posed (redundant parameters, invalid interventions), fix the *generator definition* and loop back through GATE 0. Do **not** add models to paper over a broken task.
- **LEAF-N0 (methods note):** If, after honest repair attempts, ARC generators *cannot* be made to license a clean causal DOF ledger — that itself is a publishable methodological contribution: *"Procedural ARC generators define labels, not identifiable causal factors."* Venue: ML4PS / a workshop. Small but real, and it saves the field time.

---

## PHASE 2 — Core causal-compression test (Weeks 4–6)

**Purpose.** The decisive test of **H1**. This is the phase the whole project is built around.

> **H1 (causal compression):** At matched in-distribution accuracy *and* matched resource budget, a representation's causal-subspace score predicts its shifted-test accuracy.

### Work items
- **P2.1** Expand to 20–30 families. Sweep β and capacity, saving checkpoints matched on train performance (so you compare representations at equal fit, not equal training time).
- **P2.2** **Do not use ordinary mediation as the headline.** "description length → recovered DOF → generalization" is *not* identified by correlational mediation — architecture, difficulty, optimization, and supervision confound all three arms. Instead:
  - Randomize compression strength, inductive prior, capacity, seed.
  - **Scrub** the learned causal subspace → measure output drop under shift.
  - Scrub a nuisance-predictive subspace *matched for dimension and norm* (the control).
  - Repeat all of it after `do(U)`.
  - If scrubbing *causal* info (but not nuisance info) reliably harms shifted performance, that is mechanism-level evidence, not a regression coefficient.
- **P2.3** Family-cluster bootstrap (10,000 resamples) + hierarchical model with task-family random intercept. Report raw per-family effects *alongside* the pooled estimate. **The independent unit is the family, not the thousands of generated grids** — this is the single most common way ARC-style studies fool themselves on significance.

### ► GATE 2 (H1) — pre-registered:
**H1 clears** iff the causal-subspace score's effect on `do(U)`+recombination accuracy (controlling for train accuracy, family, output entropy, model class) exceeds the pre-specified +5pp threshold with a 95% cluster-bootstrap interval excluding it, **and** the effect survives shortcut reversal.

### Branches
- **H1 POSITIVE →** proceed to Phase 3. The flagship paper is now live.
- **B-2.1 (H1 negative, but MDL still predicts OOD) → LEAF-A:** Algorithmic simplicity helps generalization *without* recovering your annotated causal factorization. This refines rather than kills the thesis — inspect generator ambiguity and compositional variables (a rule may be a short *program* over many variables, not a low-dim manifold). Publishable: *"MDL predicts ARC OOD generalization, but not via recoverable causal DOF."*
- **B-2.2 (H1 negative, nothing predicts OOD, oracle does) → LEAF-N1:** Model priors/optimization fail to recover recoverable structure. The hypothesis stays testable but unsupported; you publish an honest negative with the oracle gap as proof the structure was there to find. This is a *real* contribution — it tells the field the bottleneck is inductive bias, not information.
- **Guardrail branch (do not interpret):** If *no* model, including the oracle, shows a shift gap — the generator is too easy or its interventions are invalid. Do not interpret. Return to P0/P1 on that family.

---

## PHASE 3 — Precision allocation & effective degrees of freedom (Weeks 6–7, overlaps P2)

**Purpose.** Test **H2**, and — the v2 addition — bring in a *principled* effective-DOF instrument instead of relying on representation-geometry ID alone.

> **H2 (precision allocation):** Extra rate/precision is assigned to `C`, not `U`, and this allocation predicts OOD performance *beyond* total rate.

### Work items
- **P3.1** Estimate per-factor conditional rate `R_C − R_U` via intervention pairs on the stochastic latent. Prefer a categorical / group-structured latent — asserting that Gaussian variance carries semantic meaning on discrete grids is a modeling error waiting to be exploited by a reviewer.
- **P3.2 — NEW INSTRUMENT (singular learning theory).** Add the **local learning coefficient (LLC)** as a second, independent effective-dimension measure. Rationale: your ID estimators measure the geometry of the *representation*; the LLC (Watanabe's RLCT, operationalized by Lau–Murfet et al.) measures the effective dimensionality of the *parameter-space solution the learner actually settled into*. These are different claims. Having both is a sharper instrument than either.
  - Tooling: Timaeus `devinterp` library (SGLD-based LLC estimation).
  - Pre-register: LLC is *descriptive/corroborating*, not a replacement for the causal-scrubbing test. It is easy to over-claim here — SLT's guarantees are asymptotic and its estimators are finicky at small scale. Validate the LLC estimator on toy models with known RLCT first (the `devinterp` docs ship these).
  - The interesting cross-check: **does lower LLC coincide with higher causal-subspace score across families?** If effective DOF and *causal* DOF move together, that is a genuinely novel bridge between SLT and causal representation quality. If they *don't*, that's equally interesting and worth reporting.

### ► GATE 3 (precision & DOF):
**H2 clears** iff `R_C − R_U > 0` reliably *and* this differential rate predicts OOD beyond total rate (nested model comparison, family-clustered).

### Branches
- **H2 POSITIVE →** the flagship claim strengthens to *"selective, causal rate allocation — not compression volume — explains robust generalization."* Strongest possible headline.
- **B-3.1 (H2 negative):** Relevant representation matters (H1 held) but the VAE/KL precision story is *not* the mechanism. Report it plainly; the paper is still strong on H1. Do not resuscitate the precision story with proxy-switching.

---

## PHASE 4 — Symbolic arm & semantic convergence (Weeks 7–8)

**Purpose.** Test **H3** — whether the neural and symbolic compressions are the *same object* semantically.

> **H3 (semantic convergence):** Neural and symbolic solutions agree on intervention-response signatures, especially when both generalize.

### Work items
- **P4.1** Bounded enumerative search in a small, fixed, typed DSL with an explicit two-part code `L(P) + L(D|P)`, time-limited. **Start with a small compositional DSL — do not begin by searching the full Hodel DSL.**
- **P4.2** Measure convergence via three *coordinate-free* objects (never compare "latent axis 3" to "program argument 3"):
  1. **Intervention-response signature:** does the neural output change the same way as the reference program under every valid perturbation?
  2. **Causal-influence mask:** which interventions move the output? Precision/recall vs the annotated mask.
  3. **Subspace-to-semantics:** held-out nonlinear probes / CCA, best permutation-invariant alignment, *secondary/descriptive only*.
- **P4.3** Repeat under ≥2 DSL encodings. A shorter program in a richer DSL is not evidence unless the DSL's library cost is held constant or included.

### ► GATE 4 (H3):
"Convergence" = both methods select the *same interventionally-relevant equivalence class* on **held-out** interventions (not just fitting the demonstrations). Coordinate/surface-form identity is explicitly **not** required.

### Branches
- **CONVERGE →** strong "same object" evidence. Feeds LEAF-P.
- **B-4.1 (DIVERGE) →** both generalize but disagree. This *refines* the thesis, doesn't kill it. Produce a taxonomy of families: geometric-but-not-programmatic, programmatic-but-not-geometric, both. That taxonomy is itself a paper-worthy contribution (LEAF-A or a section of LEAF-P).

---

## PHASE 5 — External validity & efficiency (Weeks 9–10)

**Purpose.** Show the planted-family effects aren't artifacts of your own generators.

### Work items
- **P5.1** Apply the *frozen* harness to 30–50 audited RE-ARC / ARC-TGI families. **Replication, not development data** — no metric tuning here.
- **P5.2** Second annotator reviews 20% of ledgers; report agreement. Unclassifiable variables are marked *unknown*, never force-labeled.
- **P5.3** **H4 (bounded compressor):** distill the teacher *only after* it has demonstrated H1. Measure whether the student preserves the *intervention signature* and the resource frontier — not merely whether it matches teacher outputs on training samples.

### ► GATE 5 (external validity):
Track-A effects reproduce in direction and rough magnitude on audited real generators, with the family as the unit. Transfer statements hold out families by *rule template*, not just sampled parameters.

### Branches
- **REPLICATES →** LEAF-P is fully supported. Ship.
- **PARTIAL / FAILS →** scope the claim to planted families honestly and report the transfer gap as a finding about real-ARC ambiguity. Still publishable, just narrower.

---

## PHASE 6 — Physics & action (ONLY after a positive core result)

**Gate to even enter Phase 6:** H1 positive and survived shortcut reversal. Otherwise these branches are cosmic drift and are explicitly forbidden.

- **B-6.1 IB phase transitions.** Repeat β sweeps from independent inits. Claim a transition *only* with a reproducible rate–distortion kink/curvature criterion *and* a coincident causal-score jump. "Edge of chaos" needs its own dynamical observable — never inferred from an accuracy peak.
- **B-6.2 RG-like layers.** Test whether layerwise changes in causal/nuisance information and ID are consistent across families. Evidence for a *useful coarse-graining analogy* — **not** a derivation of RG from an ARC transformer. Cite Gordon et al. for what *is* proven and state clearly you are outside its field-theory scope.
- **B-6.3 LLC developmental trajectory.** Track the LLC over training (the Timaeus developmental-interpretability move). Does effective-DOF collapse coincide with the causal-score rise / an IB transition? This ties Phases 3, 5, 6 together and is the most differentiated physics-side result available to you.
- **B-6.4 Active extension (ARC-AGI-3).** New hypothesis: actions maximizing expected reduction in uncertainty about the task mechanism beat matched random/exploitative actions per interaction. Static ARC-2 *cannot* test this — reserve it as a genuinely separate follow-up, not a clause "confirmed" by the core work.

---

## 3. Leaves (what actually ships, per branch outcome)

| Leaf | Trigger | Deliverable | Venue fit |
|---|---|---|---|
| **LEAF-P (flagship)** | H1+ and (H2+ or H3 converge) and P5 replicates | "Causally useful compression predicts ARC generalization" — full paper | ICLR 2027 main (deadline ~24 Sep 2026) or NeurIPS 2027; ML4PS as fallback |
| **LEAF-A (refinement)** | H1− but MDL predicts OOD, or H3 diverges cleanly | "MDL predicts ARC OOD without recoverable causal DOF" / convergence taxonomy | Workshop (ML4PS, AI4Science) or short paper |
| **LEAF-N0 (methods)** | Generators can't license a causal ledger | "ARC generators define labels, not identifiable causal factors" | Workshop methods note |
| **LEAF-N1 (negative)** | H1−, nothing predicts OOD, oracle does | Honest negative: bottleneck is inductive bias, not information | Workshop; strong for reproducibility tracks |
| **Tool release** | GATE 0 passed | The planted-generator + intervention harness, open-sourced | Accompanies any leaf; raises citation floor |

**Every path ships.** That is the design goal.

---

## 4. Analysis & reporting rules (unchanged from v1, restated as invariants)

- Independent unit = task family (+ seed). Hierarchical or family-cluster bootstrap CIs. Never treat generated grids as independent samples.
- Hold families out by rule template for any transfer claim.
- Report all seeds, failed runs, code/data revisions, probe calibration, raw per-family plots.
- Separate confirmatory H1–H4 from exploratory β/criticality/LLC analyses in the writeup.
- One fixed MDL proxy throughout. A shorter program in a richer DSL is not evidence unless DSL cost is held constant.
- Every claim tagged: **proven / conjectured-empirical / speculative**. The physics bridge is conjectured-empirical *at best* on ARC.

---

## 5. Literature anchors (verified 13 July 2026)

**Core substrate / assets**
- RE-ARC — procedural generators for all 400 ARC-1 training tasks: arxiv.org/abs/2404.07353
- ARC-TGI — 461 validated task-family generators (incl. 66 ARC-AGI-2): arxiv.org/abs/2603.05099
- Hodel `arc-dsl`: github.com/michaelhodel/arc-dsl (training solutions are hand-designed: openreview.net/pdf?id=JlSyXwCEIQ)
- ARC Prize 2025 results & cost constraint: arcprize.org/competitions/2025 · ARC-AGI-3 interactive design: arcprize.org/arc-agi/3

**The physics bridge — cite for scope, not as license**
- Gordon, Banerjee, Koch-Janusz, Ringel (2021), *Phys. Rev. Lett.* 126, 240601 — IB=RG relevance, **proven for field theories only**: arxiv.org/abs/2012.01447
- Gökmen, Ringel, Huber, Koch-Janusz (2021), real-space mutual-information neural estimation (the numerical machinery behind the above): *Phys. Rev. E* 104, 064106
- **Cautionary case:** Saxe et al. (2018), critique of the IB-of-deep-learning story — read this so you don't repeat it.

**Effective DOF / singular learning theory (v2 addition)**
- Watanabe — singular learning theory / RLCT (the theory)
- Lau, Murfet et al. — local learning coefficient estimation; Timaeus `devinterp` library (the tool + developmental interpretability)

**Compression / representation theory**
- Locatello et al. (2019) — non-identifiability of unsupervised disentanglement (*the* reason you use labels + interventions): proceedings.mlr.press/v97/locatello19a.html
- MDL generalization guarantees for representation learning: arxiv.org/abs/2402.03254
- IB phase transitions (motivates the β sweep): arxiv.org/abs/2001.01878
- Intrinsic dimension in trained nets (diagnostic, not ground truth): arxiv.org/abs/1905.12784
- Scaling laws from data-manifold dimension (empirical, not a guaranteed ARC exponent): jmlr.org/papers/v23/20-1111.html

---

## 6. First action tomorrow (unchanged — this is still the right first move)

Implement the **5-family planted generator suite and its intervention tests** (P0.2 + GATE 0 items). Do **not** yet train a large VAE, build a leaderboard cascade, add the LLC instrument, or look for a critical point. GATE 0 is the only thing standing between you and a harness you can trust. Everything downstream is worthless without it.

In two weeks, GATE 1 tells you whether there's a real causal-compression signal worth scaling. Until then, you are building instruments, not testing hypotheses.
