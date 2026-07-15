# Literature Review — Addendum: closing the two gaps

**Companion to `LITERATURE_REVIEW_competitors.md`.** Compiled 13 July 2026.
Covers (1) non-arXiv 2026 causal-representation-learning proceedings, (2) the ARC Prize 2025 Kaggle winners' solutions, and (3) **a cluster the first review missed entirely** — the symbolic MDL-on-ARC lineage, which changes one of my earlier claims.

---

## 0. Headline of the second pass

The white space in §8 of the main review **survives**, but with one important correction: **you are not the first to observe that compression alone ≠ generalization on ARC.** Sébastien Ferré's MDL-on-ARC program (2021–2025) found exactly this, empirically, and it is the single most relevant prior-art cluster I missed. It doesn't scoop you — Ferré has no causal ledger, no interventions, no rate/DOF decomposition — but it means your framing must change from "we show compression alone isn't enough" (already known) to "**we show *which* compression matters, by isolating the causal component under interventions and tying it to rate and effective DOF.**" That is a sharper and still-defensible claim.

Neither gap turned up a direct scoop. But both turned up things you must cite.

---

## GAP 1 — Causal representation learning proceedings (non-arXiv, 2025–2026)

### 1.1 The false alarm (differentiate, don't panic)
**"Causal Representation Learning with Optimal Compression under Complex Treatments"** (arXiv:2603.11907, May 2026). The title reads like a direct hit. It is not: it's **individual treatment-effect (ITE) estimation** in multi-treatment settings — balancing weights, a multi-treatment generalization bound, Wasserstein-geodesic-preserving generative architecture. "Compression" there means dimensionality reduction over the treatment space, not MDL/rate on a representation, and there is no ARC, no reasoning, no causal-factor-recovery-predicts-OOD test. Cite it once to show you checked; differentiate in one sentence.

### 1.2 The real state of the CRL field — and a strategic warning
The identifiability side of "recover causal factors from interventions" is a **crowded, theoretically mature lane** owned by strong groups. A representative (non-exhaustective) map from this pass:

- **Ahuja, Mahajan, Wang, Bengio — Interventional Causal Representation Learning**, ICML 2023. The backbone result.
- **Squires, Seigal, Bhate, Uhler — Linear causal disentanglement via interventions**, ICML 2023; and **Zhang, Greenewald, Squires, Srivastava, Shanmugam, Uhler — Identifiability from soft interventions**, NeurIPS 2023. A single intervention per latent suffices (linear/soft-intervention regimes).
- **Varıcı, Acartürk, Shanmugam, Kumar, Tajer — General Identifiability and Achievability for CRL**, AISTATS 2024; score-based CRL, and **Linear CRL from Unknown Multi-node Interventions** (2024). The score-based line.
- **Buchholz, Rajendran, Rosenfeld, Aragam, Schölkopf, Ravikumar — Learning linear causal representations from interventions under general nonlinear mixing**, NeurIPS 2023.
- **Zhang, Xie, Ng, Zheng — CRL from multiple distributions: a general setting**, ICML 2024.
- **Li, Kaba, Ravanbakhsh — On the Identifiability of Causal Abstractions**, AISTATS 2025 (PMLR 258). Contrastive before/after unknown interventions on *arbitrary subsets* of latents — the most general contrastive-CRL identifiability to date. **Read this**: your intervention suite is a contrastive before/after design, and this paper defines what is and isn't identifiable in exactly that setting.
- **Brehmer, De Haan, Lippe, Cohen — Weakly supervised causal representation learning** (contrastive CRL, NeurIPS 2022) — the antecedent Li et al. generalize.
- **von Kügelgen et al.** — Nonparametric Identifiability of Causal Representations from Unknown Interventions (NeurIPS 2023); Interaction Asymmetry (ICLR 2025); a March-2026 "Empirical Bayes for CRL" preprint. This group is the center of gravity.
- **Squires/Zhang & collaborators, "Causal Triplet"** (CLeaR 2023) — an *intervention-centric CRL benchmark*, the closest thing to a "planted-factors + interventions" evaluation platform in CRL proper (analogous in spirit to your Track A, different domain).

**Venue to watch:** **CLeaR 2026** (5th Conference on Causal Learning and Reasoning, Cambridge MA, 6–8 April 2026). Its proceedings (PMLR) are the place a "causal factors on grids" paper would most naturally land. As of this search I could not pull the full CLeaR 2026 accepted list (the site is a JS app that didn't render); **do one manual pass over the CLeaR 2026 PMLR volume and the ICML 2026 accepted list before you submit** — that's the residual blind spot.

**The strategic warning (this is the useful part):** do **not** try to contribute *identifiability theory*. That lane is saturated with mathematicians who will out-theorem you, and it is not your comparative advantage. Your contribution is **empirical, on ARC, and bound to compression/DOF** — a place none of these groups work. Lead with the binding (causal recovery × rate × LLC × neural-vs-symbolic) and treat their identifiability results as the *license* for your interventional design, cited in one paragraph, not as terrain you're competing on. If a reviewer from this community sees you claiming novel identifiability, you lose; if they see you *applying* their identifiability results to a new empirical question, you win.

---

## GAP 2 — ARC Prize 2025 Kaggle winners' solutions

I checked the official results, the technical report, and the winners' write-ups. Summary: **the competitive Kaggle landscape is orthogonal to your angle.** No top solution does causal-factor recovery or an MDL-mechanism study.

- **Kaggle top *score* (SOTA): NVARC** (NVIDIA — Ivan Sorokin, Jean-François Puget). ~24% private (27.64% semi-private public leaderboard) on ARC-AGI-2 at ~$0.20/task, via **synthetic data generation + test-time training on a fine-tuned 4B model**. This is a bespoke-refinement / TTT solution — no compression-mechanism, no causal analysis. It's the "engineering closes the accuracy gap" datapoint the ARC Prize report describes.
- **Paper-prize winners** (already in the main review): **SOAR** (2nd — evolutionary program synthesis, self-improving LLM on its own traces), **CompressARC** (3rd — neural MDL, your main competitor), and **TRM** (Tiny Recursive Model, 7M params, recursive latent refinement).
- **Dominant 2025 theme:** the **"refinement loop"** — per-task iterative program optimization guided by a feedback signal, in both program space (evolutionary synthesis) and weight space (zero-pretrain TTT). Your work is *not* a refinement-loop solver, which is a feature: you're a mechanism study, not a leaderboard entry. Say so explicitly to preempt "why isn't your Kaggle score higher."

**Context facts worth citing accurately:** ARC Prize 2025 ran on ARC-AGI-2, 1,455 teams / 15,154 entries, $700K grand prize (≥85%) unclaimed, 90 papers submitted. All winning solutions are open-sourced (Kaggle rules require it) — so if you want to inspect a specific solver for hidden causal tricks, the code is public. I spot-checked the themes; none advertise causal-factor analysis.

**Residual blind spot:** I did not read all 90 paper submissions line-by-line, nor every open-sourced notebook. The *themes* are clearly TTT / evolutionary synthesis / small-net compression, so the risk is low, but if you want zero residual risk, skim the titles of the 90 ARC Prize 2025 papers (they're linked from the results page) for "causal," "intervention," "disentangle," "information bottleneck."

---

## GAP 3 (the one I missed) — The symbolic MDL-on-ARC lineage

This is the most important addition from the second pass. The first review treated CompressARC (Dec 2025) as *the* MDL-on-ARC work. That was incomplete: there is a **multi-year symbolic MDL-on-ARC program that predates it**, and it is directly relevant to your symbolic arm and your core thesis.

### Ferré's line (Univ Rennes)
- **Ferré 2021** — *First Steps… Descriptive Grid Models and the MDL Principle*, arXiv:2112.00848. Grid models that parse and generate grids; MDL guides the search toward maximally compressive models. Solved ~29/400 training tasks in 30s/task, and — key — **outputs an intelligible model and explanation**, not just a prediction.
- **Ferré 2023/2024** — *Tackling ARC with Object-centric Models and the MDL Principle*, arXiv:2311.00545, published at **IDA 2024 (Springer LNCS 14641)**. Object-centric models producing **joint descriptions of input/output pairs**; MDL searches the model space; "the learned models are similar to the natural programs" humans write. (A secondary summary reports 92% train / 72% eval "generalization"; I could **not verify those exact figures** against the primary PDF — treat as unconfirmed and check before citing a number.)
- **MADIL — Ferré 2025**, arXiv:2505.01081. "MDL-based AI," pattern-based decomposition for structured generalization; ~7% at ARC Prize 2024. The current incarnation of the program.

### Why this matters to you specifically — three things

1. **Your symbolic baseline is a Ferré-family method.** Your Phase-4 two-part code `L(P)+L(D|P)` over an object-centric DSL is essentially MADIL's recipe. Cite Ferré as the established object-centric-MDL-on-ARC baseline; consider using MADIL (open, and built exactly on RE-ARC-adjacent structure) as your symbolic arm rather than rolling your own from scratch. That saves weeks and gives you a citable, non-strawman symbolic competitor.

2. **Ferré already found "compression ≠ generalization" on ARC — cite it as support, and move past it.** In the object-centric paper he notes the tension directly: MDL learning finds the *most compressive* model, but in prediction mode the input model must be *as general as possible while still capturing the information needed to generate the output* — the most-compressive model sometimes **fails on test examples**. That is an independent, published, ARC-specific instance of your thesis that raw compression is the wrong objective. **This is good for you** (independent support) but it also means you cannot claim the observation as novel. Your novelty is the *diagnosis*: you isolate the causal component under interventions and show *that* is what predicts generalization, with rate and LLC as the instruments. Ferré noticed the gap; you explain it.

3. **It sharpens your H3 (neural↔symbolic convergence) into something unusually clean.** You now have two *independently developed* MDL solvers for ARC on opposite sides of the neural/symbolic divide — **CompressARC (neural MDL)** and **Ferré/MADIL (symbolic MDL)** — both explicitly minimizing description length, neither tested against the other for semantic agreement. "Do the neural-MDL and symbolic-MDL compressions of the same task recover the same interventional semantics?" is now a concrete, well-posed question with two real systems to plug in, not a hypothetical. That is a stronger H3 than the main review implied.

### Adjacent symbolic/neuro-symbolic ARC work surfaced (map, lower priority)
- **NSA — Neuro-symbolic ARC** (Batorski et al., Jan 2025, arXiv:2501.04424): transformer proposes DSL primitives to prune search; reported ~27% gain over prior SOTA on the eval set.
- **Bober-Irizar & Banerjee** — "Neural networks for abstraction and reasoning" (*Scientific Reports* 2024): the PeARL/DreamCoder-on-ARC study already in the main review.
- **ARGA** (Xu, Khalil, Sanner, AAAI 2022): graph-abstraction DSL + constraint search — the object-centric-graph baseline.
- **GPAR** (Generalized Planning for ARC, AAAI 2024): ARC as PDDL planning.
- **Vector Symbolic Algebras for ARC** (arXiv:2511.08747, Nov 2025): VSA/hyperdimensional approach.
- **Benchmarks for generalization *within* ARC:** **ConceptARC** (Moskvichev, Odouard, Mitchell 2023) and **LARC** (Language-annotated ARC). If you want a real-ARC generalization testbed with human structure beyond RE-ARC/ARC-TGI, ConceptARC is the one to know.
- **An invariance-augmentation ARC transformer**: recent work (referenced in arXiv:2606.12847, June 2026) describes an ARC-AGI-2 transformer using **group symmetries, grid traversals, and automata perturbations to enforce invariance to representation changes** — directly relevant to your equivariant/object baseline (Phase 2, model #2). Worth tracking down the primary source for your augmentation design.

---

## 4. Net effect on your positioning (updated)

The main-review white-space claim, re-stated with the corrections from this pass:

> The observation that **raw compression is not sufficient for ARC generalization** is already in the literature (Ferré). The identifiability of causal factors from interventions is already established theory (Ahuja, Squires/Uhler, von Kügelgen, Li-Ravanbakhsh). What is **not** done by anyone: (a) building a per-generator **causal ledger with interventions** on ARC-style generators, (b) measuring whether the **causal** component of a representation's compression — separated from the nuisance component by `do(U)` and shortcut reversal — predicts OOD, (c) tying that to **selective rate allocation** and **effective DOF via the LLC**, and (d) testing whether the two real MDL solvers on ARC (**CompressARC neural**, **MADIL symbolic**) converge on the same interventional semantics.

Every clause of (a)–(d) is defensible as unoccupied after two search passes. The binding is still the paper.

**Two must-do citations you'd have missed:** Ferré (2021, 2024) / MADIL (2025) on the symbolic-MDL side; Li-Kaba-Ravanbakhsh (AISTATS 2025) + Ahuja (ICML 2023) on the interventional-identifiability side. Omitting either would read as not knowing the field.

---

## 5. Residual blind spots (be honest about these before submission)

1. **CLeaR 2026 + ICML 2026 accepted lists** — I could not fully enumerate them (CLeaR site didn't render; ICML 2026 decisions may post-date this search). Do one manual pass for "causal + compression/MDL + generalization/reasoning."
2. **The 90 ARC Prize 2025 papers** — themes are clearly TTT / evolutionary synthesis / small-net compression, but I didn't read all titles. One title-skim for "causal / intervention / disentangle / bottleneck" closes it.
3. **Unverified number:** the Ferré-2024 "92%/72%" figure from a secondary summary — verify against the primary PDF before quoting.
4. **Not found (good for you):** after two passes, still no paper binding causal-factor recovery + rate/MDL + effective-DOF on ARC, and no neural-MDL-vs-symbolic-MDL semantic-convergence test. If clauses 1–2 above come back clean, the white space is real.
