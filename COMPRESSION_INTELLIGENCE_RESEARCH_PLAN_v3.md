# Causally Useful Compression on ARC — A Phased Research Tree

**Version 3 · incorporates the competitor review + addendum (both 13 Jul 2026) and a fresh blind-spot pass (15 Jul 2026)**
Last updated 15 July 2026.

**Changelog v2 → v3 (summary):**
1. §0 reframed per the Ferré correction: "compression ≠ generalization on ARC" is *published prior art* — the claim is now "*which* compression matters, isolated under interventions."
2. CompressARC added as a **model inside the harness** (P1.2) — co-opt the nearest competitor as a datapoint.
3. P3.2 upgraded from plain LLC to the **rLLC measured w.r.t. do(C) vs do(U) distributions** (Wang et al., ICLR 2025) — the flagged novel measurement and the Timaeus scoop-defense.
4. Phase 4 symbolic arm rebased on **MADIL (Ferré 2025)**; H3 sharpened to *CompressARC (neural MDL) ↔ MADIL (symbolic MDL)* semantic convergence.
5. Identifiability posture made explicit: cite CRL theory as *license* in one paragraph; never contribute identifiability theory.
6. Timeline re-cut for reality: ICLR 2027 demoted to stretch goal; primary target re-set.
7. New §7 pre-submission checklist (CLeaR/ICML 2026 passes, 90-paper title skim, Ferré figure verification).
8. ARC-AGI-1 contamination folded into positioning; ConceptARC added as optional Phase-5 testbed.

---

## 0. Read this first: the one-sentence bet, and what it is *not*

**The bet (shippable, falsifiable):** A representation that preserves the variables which *causally determine* an ARC output — and is invariant to nuisance interventions — generalizes better than equally accurate alternatives; low-dimensional neural representations and short symbolic programs recover the *same interventional semantics* when both generalize.

**The framing correction (v3, load-bearing).** The observation that raw compression is insufficient for ARC generalization is **already in the literature**: Ferré's symbolic MDL-on-ARC program (2021–2025) reports directly that the most-compressive model sometimes fails on test examples. Do **not** frame the contribution as "we show compression alone isn't enough" — that is known. The contribution is the **diagnosis**: we isolate the *causal component* of a representation's compression via interventions and shortcut reversal, and show *that* component — tied to selective rate allocation and effective DOF — is what predicts generalization. Ferré noticed the gap; we explain it. Cite Ferré as independent support, then move past it.

**What this is NOT (the north star, kept private):** "Intelligence is compression," "IB = RG in neural nets," "information becomes law." Those are the *source of hypotheses*, not claims this project asserts. The intellectual-motivation appendix is a private document; its language ("proven mathematical equivalence," "cornerstone") must **never** appear in the paper — any physics content in the writeup is drafted fresh under the scope rules below. The published unit stands or falls on the interventional experiment, never on the intuition that generated it.

**What this is also NOT (v3): a leaderboard entry.** The 2025 competitive landscape is refinement loops (TTT, evolutionary synthesis — NVARC at ~24% ARC-AGI-2). This project is a *mechanism study*, not a solver. State this explicitly in the paper to preempt "why isn't your Kaggle score higher." Use the ARC Prize 2025 report's own framing: the accuracy gap is "bottlenecked by engineering," the efficiency gap is "bottlenecked by science and ideas" — this work is squarely the latter.

**Positioning gift (v3):** ARC-AGI-1 is now widely regarded as contaminated by LLM memorization, and the field is moving to procedural/interactive tasks. A mechanism study on **audited procedural generators with planted ground truth** sidesteps contamination by construction. Say so in the intro.

**Why the distinction is load-bearing.** A zip file compresses. A lookup-table memorizer compresses. Neither is intelligent. The testable object is *resource-bounded, task-relevant compression under interventions* — not compression per se. Every gate below exists to stop the north star from smuggling itself back in as an unfalsifiable headline.

**Scope honesty on the physics bridge.** The IB=RG equivalence (Gordon et al. 2021, PRL 126, 240601) is *proven for field theories*. It is **not** a theorem about ARC networks. Whether an analogous coarse-graining happens in an ARC-trained net is an *open empirical question* tested in Phase 6 — never a premise.

**Scope honesty on identifiability (v3).** The identifiability of causal factors from interventions is a mature, crowded theory lane (Ahuja et al. ICML 2023; Squires/Uhler; von Kügelgen; Li–Kaba–Ravanbakhsh AISTATS 2025 — the last defines identifiability for exactly our contrastive before/after design). This project **applies** those results as the license for the interventional design, cited in one paragraph. It does **not** claim, prove, or extend identifiability theory. If a reviewer from that community reads us as claiming novel identifiability, we lose; if they read us as applying their results to a new empirical question, we win.

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

The spine is Phases 0→6. Most phases fan out into 2–3 branches depending on what the gate returns. **A "negative" gate is not failure — several branches lead to publishable negative or methodological results.** Every leaf ships something.

---

## 2. The tree at a glance

```
P0  Specification & audit ───────────────► GATE 0: harness is trustworthy
      │
P1  Feasibility pilot (10 families) ──────► GATE 1: signal exists at all
      │   (baselines now include CompressARC)
      ├─ FAIL ─► B-1.1  Repair generators / redefine causal ledger ─► loop to P0
      │          (if irreparable) ─► LEAF-N0
      │
P2  Core causal-compression test ─────────► GATE 2 (H1): causal score
      │   (20–30 families)                   predicts shifted accuracy,
      │                                       survives shortcut reversal
      ├─ H1 POSITIVE ──────────────────────────────────────────────┐
      ├─ H1 NEGATIVE but MDL predicts OOD ─► B-2.1 ─► LEAF-A       │
      └─ H1 NEGATIVE, nothing predicts OOD ─► B-2.2 ─► LEAF-N1     │
                                                                   │
P3  Precision-allocation & effective-DOF ◄─────────────────────────┘
      │   (H2 + rLLC w.r.t. do(C)/do(U)) ──► GATE 3: rate & effective
      │                                       DOF go to C not U
      ├─ H2 POSITIVE ─► toward LEAF-P (flagship)
      ├─ H2 NEGATIVE ─► B-3.1
      │
P4  Symbolic arm (MADIL) & convergence ───► GATE 4 (H3): CompressARC and
      │                                       MADIL agree on intervention
      │                                       signatures
      ├─ CONVERGE ─► LEAF-P
      ├─ DIVERGE  ─► B-4.1: taxonomy ─► LEAF-A/P
      │
P5  External validity + efficiency ───────► GATE 5: Track-A effects replicate
      │   (RE-ARC / ARC-TGI; ConceptARC opt.) on audited real generators
      │
P6  ONLY IF core positive: physics & action
      ├─ B-6.1 IB phase transitions
      ├─ B-6.2 RG-like layerwise coarse-graining (descriptive)
      ├─ B-6.3 LLC developmental trajectory over training
      └─ B-6.4 Active/interactive extension on ARC-AGI-3
```

---

## PHASE 0 — Specification & audit (Week 1: 15–22 Jul)

**Purpose.** Build a harness you can trust *before* you have any result to be excited about. Every later gate is only as honest as this phase.

### Work items
- **P0.1** Freeze the causal-ledger schema, DSL grammar, seeds, metric definitions, and the immutable JSONL run-record format. Each record carries: generator hash, split hash, seed, model config, train compute, inference/search compute, predictions for *every* intervention, and all metrics.
- **P0.2** Implement 5 planted families. Each ships with a test verifying every declared intervention actually changes (or preserves) the target, and labels each variable relevant vs nuisance.
- **P0.3** Pin exact revisions of RE-ARC, `arc-dsl`, ARC-TGI — and **(v3)** CompressARC and MADIL, since both now run *inside* the harness. Inspect 10 families and complete their ledgers: `family | generator vars | candidate C/U | valid interventions | reference program | rule type | unresolved ambiguity`.
- **P0.4** Lock the *single* MDL proxy used throughout (recommend: LZ-78 on fixed-precision bitstrings, pre-registered). Neural KL, program tokens, and zipped weights are **not** interchangeable absolute bit-units.
- **P0.5 (v3)** Write the identifiability-license paragraph now (Ahuja 2023; Li–Kaba–Ravanbakhsh 2025; Locatello 2019 as the reason interventions are necessary). Getting it on paper early prevents scope creep later.

### ► GATE 0 (harness is trustworthy) — ALL must hold:
1. Every declared intervention passes its generator test.
2. A **planted shortcut** (a nuisance perfectly correlated with the answer at train) is invisible to i.i.d. accuracy but *fails under correlation reversal at test*. If the harness can't detect this on a planted case, it cannot be trusted on real data.
3. The metric code recovers the known causal mask on the planted families (precision/recall ≈ 1 on inserted ground truth).

**If GATE 0 fails:** tooling bug, not a science result. Fix the harness. Do not proceed.

---

## PHASE 1 — Feasibility pilot (Weeks 2–3)

**Purpose.** Establish that there is *any* signal worth scaling to 30 families, before spending the compute.

### Work items
- **P1.1** Expand Track A to 10 planted families.
- **P1.2** Run **five** baselines across 10 seeds each *(v3: was four)*:
  1. raw grid transformer/CNN;
  2. object-centric/equivariant — **(v3)** before finalizing the augmentation design, track down the invariance-augmentation ARC transformer referenced via arXiv:2606.12847 (group symmetries, grid traversals, automata perturbations) and align or differentiate;
  3. rate–distortion (conditional-IB) model;
  4. **(v3) CompressARC's per-puzzle MDL objective**, run inside the harness on the planted families. This turns the nearest competitor into a datapoint: does its compression preferentially preserve `C` over `U`? Its VAE-loss-as-MDL derivation also directly informs baseline 3;
  5. an **oracle** that gets `C` directly (upper bound).
- **P1.3** Validate every intrinsic-dimension estimator (TwoNN, Levina–Bickel MLE, local ID) on synthetic manifolds of *known* dimension. Report where each breaks. ID is a *descriptor*, never ground truth.
- **P1.4** **Pre-register** final hypotheses, analyses, exclusions, and the practical effect threshold (default +5pp shifted-test accuracy per SD of causal score) — *after* the pilot design is fixed, *before* seeing final results.

### ► GATE 1 (signal exists at all) — ALL must hold:
1. Interventions produce a nontrivial OOD gap (raw baseline drops meaningfully under `do(U)` + recombination).
2. Oracle beats the raw baseline under shifts (if knowing `C` doesn't help, the family has no causal structure to recover).
3. At least one causal-score metric is stable across seeds (not seed-noise).

### Branches
- **B-1.1 (repair):** If a gate item fails because a generator is ill-posed, fix the *generator definition* and loop back through GATE 0. Do **not** add models to paper over a broken task.
- **LEAF-N0 (methods note):** If, after honest repair, ARC generators *cannot* license a clean causal-DOF ledger — publishable methodological contribution: *"Procedural ARC generators define labels, not identifiable causal factors."* Venue: ML4PS / workshop.

---

## PHASE 2 — Core causal-compression test (Weeks 4–6)

**Purpose.** The decisive test of **H1**. This is the phase the whole project is built around.

> **H1 (causal compression):** At matched in-distribution accuracy *and* matched resource budget, a representation's causal-subspace score predicts its shifted-test accuracy.

### Work items
- **P2.1** Expand to 20–30 families. Sweep β and capacity, saving checkpoints matched on train performance (compare representations at equal fit, not equal training time).
- **P2.2** **Do not use ordinary mediation as the headline.** "Description length → recovered DOF → generalization" is *not* identified by correlational mediation. Instead:
  - Randomize compression strength, inductive prior, capacity, seed.
  - **Scrub** the learned causal subspace → measure output drop under shift.
  - Scrub a nuisance-predictive subspace *matched for dimension and norm* (the control).
  - Repeat all of it after `do(U)`.
  - If scrubbing *causal* info (but not nuisance info) reliably harms shifted performance, that is mechanism-level evidence, not a regression coefficient.
- **P2.3** Family-cluster bootstrap (10,000 resamples) + hierarchical model with task-family random intercept. Report raw per-family effects alongside pooled estimates. **The independent unit is the family, not the generated grids.**
- **P2.4 (v3)** Related-work drafting in parallel with the runs: the differentiation from CompressARC ("does compression work" vs "*what kind* of compression, and why"), ARC-TGI (infrastructure, not competitor), Cluster-D CRL (method antecedents: IST epistemics, KamonBench, CITRIS, NuRD, Ahuja 2022 IB×invariance), and Ferré (independent prior support for compression ≠ generalization). These are the "cite or die" citations; writing them early prevents the desk-reject failure mode.

### ► GATE 2 (H1) — pre-registered:
**H1 clears** iff the causal-subspace score's effect on `do(U)`+recombination accuracy (controlling for train accuracy, family, output entropy, model class) exceeds the pre-specified +5pp threshold with a 95% cluster-bootstrap interval excluding it, **and** the effect survives shortcut reversal.

### Branches
- **H1 POSITIVE →** Phase 3. The flagship paper is now live.
- **B-2.1 (H1 negative, MDL still predicts OOD) → LEAF-A:** Algorithmic simplicity helps generalization *without* recovering the annotated causal factorization. Publishable: *"MDL predicts ARC OOD generalization, but not via recoverable causal DOF."* **(v3)** Note this outcome would *partially converge with Ferré's observation* — frame it as explaining the boundary of his finding.
- **B-2.2 (H1 negative, nothing predicts OOD, oracle does) → LEAF-N1:** honest negative; the oracle gap proves the structure was there to find. The bottleneck is inductive bias, not information.
- **Guardrail branch (do not interpret):** If *no* model, including the oracle, shows a shift gap — generator too easy or interventions invalid. Return to P0/P1 on that family.

---

## PHASE 3 — Precision allocation & effective degrees of freedom (Weeks 6–7, overlaps P2)

**Purpose.** Test **H2**, with a principled effective-DOF instrument.

> **H2 (precision allocation):** Extra rate/precision is assigned to `C`, not `U`, and this allocation predicts OOD performance *beyond* total rate.

### Work items
- **P3.1** Estimate per-factor conditional rate `R_C − R_U` via intervention pairs on the stochastic latent. Prefer a categorical / group-structured latent — asserting Gaussian variance carries semantic meaning on discrete grids is a modeling error waiting to be exploited by a reviewer.
- **P3.2 — UPGRADED INSTRUMENT (v3): the refined LLC (rLLC), measured per intervention distribution.**
  - v2 planned a plain LLC as a second effective-dimension measure. The sharper move, per Wang et al. (ICLR 2025 Spotlight): the **rLLC measures a component's complexity w.r.t. an arbitrary data distribution that can differ from training**. That is tailor-made for the causal ledger:
  - **Core measurement:** for each layer/component, estimate rLLC w.r.t. the `do(C)` distribution and w.r.t. the `do(U)` distribution. High rLLC w.r.t. `do(C)` + low w.r.t. `do(U)` = a *parameter-space signature of causal selectivity* — a measurement nobody has made, and the explicit defense against the Timaeus overlap. **This is the highest-novelty single measurement in the project; do not cut it under time pressure.**
  - Tooling: Timaeus `devinterp` (SGLD-based). Validate on the known-RLCT toy models the docs ship, and run the rescaling-invariance sanity check, *before* trusting any ARC number.
  - Pre-register: rLLC is *descriptive/corroborating*, not a replacement for the causal-scrubbing test. SLT guarantees are asymptotic; estimators are finicky at small scale.
  - Cross-check: **does lower (train-distribution) LLC coincide with higher causal-subspace score across families?** If effective DOF and *causal* DOF move together, that is a genuinely novel SLT↔causal-quality bridge. If not, equally worth reporting.

### ► GATE 3 (precision & DOF):
**H2 clears** iff `R_C − R_U > 0` reliably *and* this differential rate predicts OOD beyond total rate (nested model comparison, family-clustered).

### Branches
- **H2 POSITIVE →** flagship claim strengthens to *"selective, causal rate allocation — not compression volume — explains robust generalization."*
- **B-3.1 (H2 negative):** relevant representation matters (H1 held) but the precision story is not the mechanism. Report plainly. Do not resuscitate with proxy-switching. **(v3)** The rLLC-per-distribution result stands on its own either way — report it in both branches.

---

## PHASE 4 — Symbolic arm & semantic convergence (Weeks 7–8)

**Purpose.** Test **H3** — whether the neural and symbolic compressions are the *same object* semantically.

> **H3 (semantic convergence, sharpened v3):** The two independently developed MDL solvers for ARC — **CompressARC (neural MDL)** and **MADIL (symbolic MDL, Ferré 2025)** — agree on intervention-response signatures, especially when both generalize. Two real systems, opposite sides of the neural/symbolic divide, both explicitly minimizing description length, never tested against each other for semantic agreement.

### Work items
- **P4.1 (rebased, v3)** Adopt **MADIL** (open-source, object-centric MDL, built on RE-ARC-adjacent structure) as the primary symbolic arm, run inside the frozen harness. This (a) saves weeks over rolling a DSL from scratch, (b) gives a citable, non-strawman symbolic competitor, (c) makes H3 a comparison between two *real published systems* rather than one real system and a homemade baseline.
  - **Fallback:** if MADIL cannot be adapted to the planted families within ~3 days of engineering, revert to the v2 plan — bounded enumerative search in a small, fixed, typed DSL with explicit two-part code `L(P)+L(D|P)`, time-limited. Do **not** start with the full Hodel DSL either way (PeARL's honest negative is the warning).
  - Cite Ferré 2021 → IDA 2024 → MADIL 2025 as the established symbolic-MDL-on-ARC lineage, and DreamCoder as the library-learning-as-MDL tradition it sits in. **Verify the secondary "92%/72%" Ferré-2024 figure against the primary PDF before quoting any number.**
- **P4.2** Measure convergence via three *coordinate-free* objects (never compare "latent axis 3" to "program argument 3"):
  1. **Intervention-response signature:** does the neural output change the same way as the reference program under every valid perturbation?
  2. **Causal-influence mask:** which interventions move the output? Precision/recall vs the annotated mask.
  3. **Subspace-to-semantics:** held-out nonlinear probes / CCA, best permutation-invariant alignment, *secondary/descriptive only*.
- **P4.3** Repeat under ≥2 DSL encodings. A shorter program in a richer DSL is not evidence unless the DSL's library cost is held constant or included. (Using MADIL + one small typed DSL satisfies this naturally.)

### ► GATE 4 (H3):
"Convergence" = both methods select the *same interventionally-relevant equivalence class* on **held-out** interventions (not just fitting the demonstrations). Coordinate/surface-form identity is explicitly **not** required.

### Branches
- **CONVERGE →** strong "same object" evidence. Feeds LEAF-P.
- **B-4.1 (DIVERGE) →** both generalize but disagree. This *refines* the thesis. Produce a taxonomy of families: geometric-but-not-programmatic, programmatic-but-not-geometric, both. Paper-worthy on its own (LEAF-A or a section of LEAF-P).

---

## PHASE 5 — External validity & efficiency (Weeks 9–10)

**Purpose.** Show the planted-family effects aren't artifacts of our own generators.

### Work items
- **P5.1** Apply the *frozen* harness to 30–50 audited RE-ARC / ARC-TGI families. **Replication, not development data** — no metric tuning here.
- **P5.2** Second annotator reviews 20% of ledgers; report agreement. Unclassifiable variables are marked *unknown*, never force-labeled.
- **P5.3** **H4 (bounded compressor):** distill the teacher *only after* it has demonstrated H1. Measure whether the student preserves the *intervention signature* and the resource frontier — not merely teacher-output matching on training samples.
- **P5.4 (optional, v3)** If time permits, one **ConceptARC** pass as a human-structured generalization testbed beyond the procedural generators. Strictly descriptive; not gated.

### ► GATE 5 (external validity):
Track-A effects reproduce in direction and rough magnitude on audited real generators, with the family as the unit. Transfer statements hold out families by *rule template*, not just sampled parameters.

### Branches
- **REPLICATES →** LEAF-P fully supported. Ship.
- **PARTIAL / FAILS →** scope the claim to planted families honestly and report the transfer gap as a finding about real-ARC ambiguity. Still publishable, narrower.

---

## PHASE 6 — Physics & action (ONLY after a positive core result)

**Gate to even enter Phase 6:** H1 positive and survived shortcut reversal. Otherwise these branches are cosmic drift and are explicitly forbidden.

- **B-6.1 IB phase transitions.** Repeat β sweeps from independent inits. Claim a transition *only* with a reproducible rate–distortion kink/curvature criterion *and* a coincident causal-score jump. "Edge of chaos" needs its own dynamical observable — never inferred from an accuracy peak.
- **B-6.2 RG-like layers.** Test whether layerwise changes in causal/nuisance information and ID are consistent across families. Evidence for a *useful coarse-graining analogy* — **not** a derivation of RG from an ARC transformer. Cite Gordon et al. for what *is* proven; state clearly this is outside its field-theory scope. Read Saxe et al. 2018 before writing a word of this section.
- **B-6.3 LLC developmental trajectory.** Track the LLC over training (Hoogland et al. developmental stages). Does effective-DOF collapse coincide with the causal-score rise / an IB transition? Ties Phases 3, 5, 6 together.
- **B-6.4 Active extension (ARC-AGI-3, released early 2026).** New hypothesis: actions maximizing expected reduction in uncertainty about the task mechanism beat matched random/exploitative actions per interaction. Static ARC-2 *cannot* test this — a genuinely separate follow-up, never a clause "confirmed" by the core work.

---

## 3. Leaves & venues (revised v3 — see timeline honesty below)

| Leaf | Trigger | Deliverable | Venue fit (v3) |
|---|---|---|---|
| **LEAF-P (flagship)** | H1+ and (H2+ or H3 converge) and P5 replicates | "Causally useful compression predicts ARC generalization" — full paper | **Primary: NeurIPS 2027** (or ICML 2027, ~Jan deadline). **Stretch: ICLR 2027 (~24 Sep 2026)** only if Gates 0–4 are all cleared by ~5 Sep. ML4PS / AI4Science workshop as staging |
| **LEAF-A (refinement)** | H1− but MDL predicts OOD, or H3 diverges cleanly | "MDL predicts ARC OOD without recoverable causal DOF" / convergence taxonomy | Workshop (ML4PS, AI4Science) or short paper; **ARC Prize 2026 paper track** is a natural fit |
| **LEAF-N0 (methods)** | Generators can't license a causal ledger | "ARC generators define labels, not identifiable causal factors" | Workshop methods note |
| **LEAF-N1 (negative)** | H1−, nothing predicts OOD, oracle does | Honest negative: bottleneck is inductive bias, not information | Workshop; reproducibility tracks |
| **Tool release** | GATE 0 passed | Planted-generator + intervention harness, open-sourced **early** — this is the primary scoop insurance | Accompanies any leaf; raises citation floor |

**Timeline honesty (v3).** From 15 July, Weeks 1–10 end ~23 September — the ICLR deadline with zero writing/buffer time, while carrying coursework, the CUHK-Shenzhen transfer, and other commitments. Plan for **NeurIPS/ICML 2027 as the flagship target**, with two hard interim commitments that keep scoop risk managed: (a) **open-source the harness the week GATE 0 passes**, (b) submit a workshop version (ML4PS or ARC Prize 2026 paper track) as soon as GATE 2 resolves *either way*. If everything lands early, ICLR remains available — but nothing in the plan may be cut to chase it, least of all P3.2 or the pre-registration steps.

**Every path ships.** That is the design goal.

---

## 4. Analysis & reporting rules (invariants)

- Independent unit = task family (+ seed). Hierarchical or family-cluster bootstrap CIs. Never treat generated grids as independent samples.
- Hold families out by rule template for any transfer claim.
- Report all seeds, failed runs, code/data revisions, probe calibration, raw per-family plots.
- Separate confirmatory H1–H4 from exploratory β/criticality/LLC analyses in the writeup.
- One fixed MDL proxy throughout. A shorter program in a richer DSL is not evidence unless DSL cost is held constant.
- Every claim tagged: **proven / conjectured-empirical / speculative**. The physics bridge is conjectured-empirical *at best* on ARC.
- **(v3)** Identifiability claims are cited, never made. The intellectual-motivation appendix never ships as-is.

---

## 5. Literature anchors (verified 13–15 July 2026)

**Core substrate / assets**
- RE-ARC — procedural generators for all 400 ARC-1 training tasks: arxiv.org/abs/2404.07353
- ARC-TGI — 461 validated task-family generators: arxiv.org/abs/2603.05099 *(frame as infrastructure built on, never a competitor beaten)*
- Hodel `arc-dsl`: github.com/michaelhodel/arc-dsl
- ARC Prize 2025 Technical Report — arxiv.org/abs/2601.10904 *(the "efficiency gap = science" framing; ARC-AGI-1 contamination)*
- "The ARC of Progress towards AGI: A Living Survey" — arxiv.org/abs/2603.13372
- ConceptARC (Moskvichev, Odouard, Mitchell 2023) — optional Phase-5 testbed

**Direct competitors / co-opted systems (v3)**
- **CompressARC** — Liao & Gu, arXiv:2512.06104. The paper to beat *and* a model in the harness. Mine its VAE-as-MDL derivation for the conditional-IB baseline.
- **Ferré lineage (MUST-CITE, v3):** Ferré 2021 (arXiv:2112.00848); object-centric MDL, IDA 2024 (arXiv:2311.00545 — verify any quoted figures against the primary PDF); **MADIL 2025 (arXiv:2505.01081)** — the symbolic arm.
- SOAR (ICML 2025), TRM, NVARC — landscape context; the refinement-loop theme this project is *not*.
- Invariance-augmentation ARC transformer (referenced via arXiv:2606.12847) — track down primary source for the Phase-1 equivariant baseline.

**Causal representation learning (license + antecedents — cite or die)**
- Locatello et al. 2019, ICML — non-identifiability without supervision/interventions (why the design is necessary).
- Ahuja, Mahajan, Wang, Bengio — Interventional CRL, ICML 2023 (the identifiability license).
- **Li, Kaba, Ravanbakhsh — Identifiability of Causal Abstractions, AISTATS 2025 (MUST-CITE, v3)** — defines identifiability for exactly our contrastive before/after intervention design. Read before finalizing the intervention suite.
- Ahuja et al. 2022 — "Invariance principle meets information bottleneck for OOD" (the IB×causal antecedent of H1/H2; read in full).
- NuRD (Puli et al.) — the rigorous version of the correlation-reversal control.
- Interventional Style Transfer, arXiv:2306.11890 — the falsification epistemics, in print; cite as precedent.
- KamonBench, CITRIS/TRIS — planted-factor benchmark antecedents.
- Schölkopf et al. 2021 — CRL manifesto.
- arXiv:2603.11907 ("CRL with Optimal Compression under Complex Treatments") — **false alarm, confirmed 15 Jul**: multi-treatment ITE estimation; cite once, differentiate in one sentence.

**Effective DOF / singular learning theory**
- Watanabe — SLT / RLCT (theory).
- Lau et al. — LLC, AISTATS 2025 (canonical).
- **Wang et al. — rLLC, ICLR 2025 Spotlight (v3: the instrument to build on)** — per-component, per-distribution complexity; the `do(C)`/`do(U)` measurement lives here.
- Hoogland et al. 2024 — developmental stages via LLC (Phase 6 B-6.3).
- Timaeus `devinterp` — tooling; validate on shipped known-RLCT toys first.
- Lehalleur et al. 2025 position paper — data-structure ↔ model-structure; the SLT-side statement of the thesis.

**The physics bridge — cite for scope, not as license**
- Gordon, Banerjee, Koch-Janusz, Ringel 2021, PRL 126, 240601 — IB=RG, **field theories only**: arxiv.org/abs/2012.01447
- Koch-Janusz & Ringel 2018 (Nat. Phys.); Lenggenhager et al. 2020 (PRX); Gökmen et al. 2021 (PRL/PRE), 2024 (Nat. Comms.) — the lineage.
- Saxe et al. 2018 — the IB-of-deep-learning deflation; the Mehta–Schwab "DL=RG" graveyard. Read before Phase 6.

**Symbolic / neuro-symbolic tradition**
- DreamCoder (Ellis et al. 2021) — library learning as MDL; the H3 framing gift.
- PeARL / Sci. Reports 2024 — the realistic bar; the don't-start-with-full-Hodel-DSL warning.
- NSA (arXiv:2501.04424), ARGA (AAAI 2022), GPAR (AAAI 2024), VSA-ARC (arXiv:2511.08747) — map, lower priority.

**Compression / representation theory**
- MDL generalization guarantees: arxiv.org/abs/2402.03254
- IB phase transitions: arxiv.org/abs/2001.01878
- Intrinsic dimension in trained nets: arxiv.org/abs/1905.12784
- Scaling laws from data-manifold dimension: jmlr.org/papers/v23/20-1111.html

---

## 6. First action tomorrow (unchanged in substance)

Implement the **5-family planted generator suite and its intervention tests** (P0.2 + GATE 0). Do **not** yet train a large VAE, wire in MADIL, add the rLLC instrument, or look for a critical point. GATE 0 is the only thing standing between you and a harness you can trust.

One addition (v3): while generator tests run, do the two short reads that change design decisions — **Li–Kaba–Ravanbakhsh (AISTATS 2025)** before freezing the intervention suite, and **CompressARC's VAE-as-MDL derivation** before freezing the conditional-IB baseline.

---

## 7. Pre-submission checklist (v3 — the residual blind spots, tracked)

- [ ] **CLeaR 2026 (PMLR) full proceedings pass** for "causal + compression/MDL + generalization/reasoning." *First pass done 15 Jul 2026: nothing found binding causal recovery + rate/MDL + effective DOF on ARC; nearest hits are bivariate MDL causal discovery and the 2603.11907 ITE false alarm. Repeat once more at submission time.*
- [ ] **ICML 2026 accepted-papers list** pass, same keywords (decisions may post-date 13 Jul search).
- [ ] **Title-skim of the ~90 ARC Prize 2025 papers** for "causal / intervention / disentangle / bottleneck."
- [ ] **Verify the Ferré-2024 "92%/72%" figure** against the primary PDF before quoting any number.
- [ ] Double-check exact arXiv IDs for 2026 preprints (ARC-TGI 2603.05099, KamonBench 2605.13322) before citing.
- [ ] Confirm the primary source behind the invariance-augmentation transformer reference (arXiv:2606.12847).
- [ ] Watch the **ARC Prize 2026** paper track (relaunched on ARC-AGI-2) for late-breaking causal/MDL entries.
