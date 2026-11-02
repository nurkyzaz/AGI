# Literature Review: Closest Prior Work & Competitors

**For the "causally useful compression on ARC" project.**
Compiled 13 July 2026. Every reference below was checked against a primary or near-primary source during this search; where I could not verify a claim I say so.

---

## 0. The one-paragraph verdict (read this first)

Your project sits at an intersection that, as far as this search can tell, **no single paper currently occupies** — but every *edge* of it is occupied, some very recently, and two of them closely. The compression-on-ARC edge is owned by **CompressARC (Liao & Gu, Dec 2025)**, which is the paper you will be compared to first and hardest. The "planted causal factors + interventions → does recovery predict OOD" methodology is **already a mature paradigm in causal representation learning** (CITRIS, KamonBench, the single-cell IST work, Locatello, Ahuja) — just *not on ARC and not tied to compression*. The IB=RG bridge is a **closed, physics-scoped result** (Koch-Janusz–Ringel–Gökmen–Gordon) that has been tried-and-mostly-failed as a literal "deep learning = RG" claim. Your defensible white space is the **binding**: causal-factor recovery ↔ rate allocation ↔ effective degrees of freedom (LLC), tested on ARC generators, with a neural↔symbolic convergence check. Nobody owns that binding yet. But you are not the first to do interventional causal-representation tests, and writing as if you were is the fastest way to get desk-rejected. Position accordingly.

---

## 1. Competitor tiers — who you're actually up against

| Tier | Work | What it does | Threat level to you |
|---|---|---|---|
| **Beat / differentiate** | **CompressARC** (Liao & Gu 2025) | Pure MDL, 76K params, per-puzzle test-time compression solves 20% ARC-AGI-1 | **Highest.** Owns "compression is intelligence, demonstrated on ARC." You must state clearly how you differ. |
| **Beat / differentiate** | **ARC-TGI** (2026) | Resampleable ARC task-family generators; nuisance sweeps, within/across-family OOD | **High.** This is *your substrate* AND a diagnostic paper doing nuisance/OOD analysis. Overlap risk. |
| **Don't-get-scooped (methodology)** | CITRIS/TRIS, KamonBench, IST single-cell, "Beyond the Doors of Perception" | Planted causal factors + interventions → test whether recovery predicts OOD | **High.** Your *method* already exists in other domains. Cite or die. |
| **Scope-guard (theory)** | Koch-Janusz–Ringel–Gökmen–Gordon (2018–2024) | IB/RSMI = RG relevance, **proven for field theories** | Medium. Not a competitor; a boundary you must not overstep. |
| **Differentiator instrument** | Lau et al. LLC (2025), Wang et al. rLLC (ICLR 2025) | Effective DOF of a *learner* via singular learning theory | Low as competitor, **high as your edge** — nobody has put LLC on ARC. |
| **Symbolic arm** | DreamCoder + PeARL (Ellis 2021; Scientific Reports 2024) | Neuro-symbolic program synthesis with MDL library learning | Medium. Your symbolic baseline lives in this tradition. |

---

## 2. Cluster A — Compression / MDL on ARC (the edge you must beat)

### CompressARC — the direct competitor
Liao & Gu, *ARC-AGI Without Pretraining*, arXiv:2512.06104 (Dec 2025); ARC Prize 2025 Paper Award, 3rd place.

What it is, precisely: a 76K-parameter network, **no pretraining, no training set, no search** — it trains at inference time on the single target puzzle, minimizing description length. Their key technical move is deriving that **a standard VAE loss with decoder regularization can substitute for the combinatorial search** implied by MDL code-golfing. Result: ~20% ARC-AGI-1 eval, ~34.75% train, ~4% ARC-AGI-2, ~20 min/puzzle on one RTX 4070. Their framing question is verbatim yours: *can lossless compression by itself produce intelligent behavior?* They cite the same Hutter/Legg-Hutter/Mahoney lineage you do.

**Why this is the paper to beat, and how you actually differ (be explicit in your related-work):**
- CompressARC is a *solver / existence proof*: "MDL compression can produce ARC-solving behavior." It does **not** ask *which variables* the compression preserves, does **not** have causal ground truth, does **not** test nuisance-intervention invariance, and does **not** separate causal from nuisance rate. It is a "does compression work" paper; yours is a "**what kind of compression, and why**" paper.
- Your causal ledger + `do(U)` interventions + shortcut-reversal is exactly the mechanism analysis CompressARC lacks. You can even **use CompressARC as a model in your harness** — run its per-puzzle MDL objective on your planted families and measure whether its compression preferentially preserves `C` over `U`. That turns your nearest competitor into a datapoint. Strongly recommend.
- Their VAE-loss-as-MDL derivation is directly relevant to your rate–distortion neural model (Phase 2–3). Read their derivation carefully; your conditional-IB objective is a supervised, causal-aware cousin of it.

### The two other ARC Prize 2025 paper-award systems (context, lower threat)
- **TRM (Tiny Recursive Model)**, Jolicoeur-Martineau 2025 — 7M params, 45% ARC-AGI-1 via recursive latent refinement. Parameter-efficiency story, not a compression-mechanism story. Relevant only as "small models can do ARC."
- **SOAR** (Pourcel, Colas, Oudeyer, ICML 2025) — self-improving evolutionary program synthesis, LLM fine-tuned on its own search traces, no human DSL. This is a *program-synthesis* competitor for your symbolic arm's "you don't need a hand-built DSL" claim.

### The ecosystem documents you should own
- **ARC Prize 2025 Technical Report**, arXiv:2601.10904 — the official landscape. Key line for your framing: they assess the ARC-AGI-1/2 accuracy gap as now "bottlenecked by engineering" while the **efficiency gap is "bottlenecked by science and ideas."** Your MDL/DOF angle is squarely an *efficiency/science* contribution — use their own framing to position it.
- **"The ARC of Progress towards AGI: A Living Survey"**, arXiv:2603.13372 (Mar 2026) — living survey; cite for landscape and to show you know the field.

---

## 3. Cluster B — ARC benchmark & diagnostic tooling (your substrate, and an overlap risk)

- **RE-ARC** (Hodel), arXiv:2404.07353 — procedural example generators for all 400 ARC-1 training tasks. Your Track-B substrate.
- **ARC-TGI**, arXiv:2603.05099 (Mar 2026) — **read this one closely, it is the biggest overlap risk after CompressARC.** It reframes ARC into resampleable *task families with aligned reasoning traces*, and explicitly supports "**robustness sweeps over nuisance variation, within- vs. across-family generalization studies, and controlled comparisons between inductive program-based solvers and transductive direct-prediction approaches.**" That is a large chunk of your experimental design, already built as infrastructure.
  - **How you differ:** ARC-TGI is a *benchmark/diagnostic platform* — it measures LLM in-vs-out-of-distribution gaps. It does **not** build a per-variable causal ledger, does **not** run representation-scrubbing on `C` vs `U` subspaces, does **not** measure rate allocation or LLC, and does **not** test neural↔symbolic semantic convergence. Your contribution is the *causal-mechanism instrumentation on top of* their (or RE-ARC's) generators. Frame ARC-TGI as **infrastructure you build on**, not a competitor you beat — and cite it as the reason you don't need to hand-build all 30 families.

---

## 4. Cluster C — IB = RG (scope boundary, and a cautionary "DL = RG" graveyard)

The lineage, in order (all verified):
1. **Koch-Janusz & Ringel 2018**, *Nature Physics* 14, 578 — RSMI: neural nets identify relevant DOF for real-space RG. The origin.
2. **Lenggenhager, Gökmen, Ringel, Huber, Koch-Janusz 2020**, *Phys. Rev. X* 10, 011037 — optimal RG transformation from information theory; RG-as-compression made precise.
3. **Gökmen et al. 2021**, *Phys. Rev. Lett.* 127, 240603 + *Phys. Rev. E* 104, 064106 — RSMI neural estimation, symmetries/phase diagrams.
4. **Gordon, Banerjee, Koch-Janusz, Ringel 2021**, *Phys. Rev. Lett.* 126, 240601 (arXiv:2012.01447) — **the one your motivation doc cites.** Proves IB-relevant modes = lowest-scaling-dimension operators **for statistical systems described by a field theory.**
5. **Gökmen et al. 2024**, *Nature Communications* 15, 10214 — compression theory for inhomogeneous systems (the frontier of this program).

**The hard boundary you must respect.** Every one of these is about *physical systems / field theories with a known RG structure*. The equivalence is a theorem *there*. It is **not** a theorem about an ARC-trained transformer. Your motivation doc's sentence "the question is whether the same principle operates in neural networks" is legitimate **as an open question**, but the moment it reads as "IB=RG, therefore my ARC net does RG," a physics-literate reviewer stops reading.

**The cautionary graveyard — "is deep learning an RG flow?"** This literal question has been asked repeatedly and has *not* produced a clean, accepted result:
- Mehta & Schwab 2014 — the original "exact mapping between variational RG and deep learning" claim (RBMs). Influential but contested.
- Schwab & Mehta 2016 self-comment; de Mello Koch et al. 2019 "Is deep learning a renormalization group flow?"; Iso, Shiba, Yokoo — scale-invariant feature extraction.
- This is the same genre of "beautiful framing, contested empirics" as the Tishby IB-of-deep-learning story that **Saxe et al. (2018)** deflated. **Read Saxe et al. before you write Phase 6.** Your plan already treats layerwise RG as a *descriptive analogy, not a derivation* — keep it that way; this cluster is why.

---

## 5. Cluster D — Causal representation learning with interventions (**your biggest scoop risk**)

This is the cluster I most want you to internalize, because **your core methodology already exists — in other domains.** The paradigm "generate data from planted latent causal factors, intervene, and test whether a representation that recovers the causal factors generalizes OOD better" is *the* central program of modern causal representation learning. You are importing a mature method to ARC + compression, which is a legitimate and valuable move — but you must cite these or reviewers will think you don't know the field.

**The epistemic core — already stated in print (this is almost your thesis sentence):**
- **Interventional Style Transfer / single-cell microscopy**, arXiv:2306.11890 — states explicitly that *the hypothesis that a representation is causal (invariant over nuisances) cannot be falsified using i.i.d. hold-out data; instead you need generalization to OOD interventions as the empirical measure of causal learning.* That is your Section-2 epistemics, already published. Cite it as precedent, and differentiate on domain (grids/reasoning) + the compression binding.

**Planted-factor synthetic benchmarks (closest in spirit to your Track A):**
- **KamonBench**, arXiv:2605.13322 — grammar-based dataset with 3 labeled factors per image and "factor-aware diagnostics" for compositional factor recovery in VLMs. This is *structurally* your planted-generator idea, minus ARC and minus compression.
- **CITRIS / TRIS / SMS-TRIS** (video, temporal interventions) — synthetic frames rendered from latent causal factors that evolve via mechanisms with occasional interventions; tests whether disentanglement improves few-shot/OOD transfer. Your interventional design in the temporal setting.
- **"Beyond the Doors of Perception"**, arXiv:2406.15955 — a whole section titled *"Disentanglement Correlates with Generalization Performance"* on ViTs with held-out color/shape combinations. Directly your H1-shaped claim, on vision.

**Nuisance-invariance formalizations (your `do(U)` machinery):**
- **NuRD (Nuisance-Randomized Distillation)**, Puli et al., arXiv:2107.00520 / AAAI — formalizes the nuisance-randomized distribution and proves a minimax-optimality property. This is the rigorous version of your "break the nuisance-label correlation at test" control.
- **"Invariance principle meets information bottleneck for OOD generalization"**, Ahuja et al. 2022 — **directly couples IB with invariance for OOD**, which is precisely your IB × causal angle. This one is close enough that you should read it in full and cite it as the theoretical antecedent of your H1/H2.

**Identifiability theory (why interventions are necessary — your Locatello point, sharpened):**
- **Locatello et al. 2019**, ICML — unsupervised disentanglement is non-identifiable without inductive bias/supervision. You already cite this; it's the reason your labels+interventions design is *necessary*, not optional.
- **Ahuja, Mahajan, Wang, Bengio, "Interventional Causal Representation Learning"**, ICML 2023 — identifiability *from interventions*. The theoretical license for your claim that interventions recover a preferred factorization.
- **Squires et al., "Linear causal disentanglement via interventions"**, ICML 2023, and **"Identifiability from soft interventions"**, arXiv:2307.06250 — a single intervention per latent suffices for identifiability (linear case). Cite to justify your intervention-suite design.
- **Schölkopf et al., "Towards Causal Representation Learning"**, arXiv:2102.11107 — the canonical CRL manifesto. Your framing's home base.

**Your honest differentiation from all of Cluster D:** they recover causal factors and correlate with OOD; **none of them tie causal recovery to a compression/rate/MDL account, none use effective-DOF (LLC), and none run on ARC-style discrete reasoning generators.** That triple — *causal recovery × rate allocation × effective DOF, on ARC* — is your defensible novelty. Say it in exactly those terms.

---

## 6. Cluster E — Effective degrees of freedom / SLT / LLC (your differentiator instrument)

This is where you have the clearest open lane. The LLC machinery is mature, has a maintained library (`devinterp`), and — critically — **nobody has connected it to ARC or to causal-representation quality.**

- **Lau, Furman, Wang, Murfet, Wei, "The Local Learning Coefficient: A Singularity-Aware Complexity Measure"**, AISTATS 2025 — the canonical LLC paper. LLC = effective dimensionality of the *parameter-space solution*, and crucially they show degeneracy in DNNs **cannot** be captured by counting flat directions. This is the principled effective-DOF measure your v1 plan proxied with TwoNN/Levina-Bickel.
- **Wang et al., "Differentiation and Specialization of Attention Heads via the refined LLC (rLLC)"**, ICLR 2025 (Spotlight) — **this is the one to build on.** The rLLC measures the complexity of a *component* (layer, head) **with respect to an arbitrary data distribution that can differ from the training distribution.** That is tailor-made for your causal ledger: measure the rLLC of an ARC layer *with respect to your `do(C)` distribution vs your `do(U)` distribution.* If a layer's rLLC is high w.r.t. `do(C)` and low w.r.t. `do(U)`, that is a parameter-space signature of causal selectivity — a genuinely novel measurement nobody has made.
- **Hoogland et al. 2024** — developmental stages via LLC over training; the basis for your Phase 6 "LLC developmental trajectory" branch.
- **"Estimating the LLC at Scale"** (devinterp) — LLC measured on deep linear nets to 100M params; shows the rescaling-invariance sanity check you must run.
- **Lehalleur et al., position paper (Feb 2025)** — argues for the relationship between structure in the *data distribution* and structure in the *model*. This is the SLT-side statement of your entire thesis; cite it to show your compression-of-structure intuition has a rigorous home.

**Caveat to keep you honest:** SLT guarantees are asymptotic (Bayesian, large-data), LLC estimators are SGLD-based and finicky at small scale, and there's active debate about estimator reliability. Validate on the known-RLCT toy models `devinterp` ships *before* trusting any ARC number. Do not let LLC become a second unfalsifiable headline.

---

## 7. Cluster F — Neuro-symbolic ARC (your symbolic arm's tradition)

- **DreamCoder**, Ellis et al. 2021, *Phil. Trans. R. Soc. A* — wake-sleep program synthesis where **library learning is literally MDL compression over the program corpus.** Your symbolic baseline's two-part code `L(P)+L(D|P)` is in this exact lineage; DreamCoder's whole "abstraction sleep" phase *is* compression. Cite it as the symbolic counterpart to CompressARC's neural compression — and note the framing gift: **the two nearest solvers on your two arms (CompressARC neural, DreamCoder symbolic) are BOTH compression systems, yet nobody has tested whether they recover the same causal semantics.** That's your H3 in one sentence.
- **PeARL / "Neural networks for abstraction and reasoning"**, *Scientific Reports* 2024 (arXiv:2402.03507) — adapts DreamCoder to ARC with a perceptual DSL; honest finding that hand-crafted DSL search (Icecuber) still beats it. Sets the realistic bar for your symbolic arm and warns you not to start with the full Hodel DSL (your plan already says this).
- **Banburski et al., "Dreaming with ARC"** (CBMM) — earlier DreamCoder-on-ARC; useful for the compression-drives-abstraction argument.
- **SOAR** (above) — the "no hand-built DSL needed" competitor.

---

## 8. The white space — your defensible contribution, stated precisely

Overlay the six clusters and the empty cell is specific:

> **No existing work tests whether task-relevant, nuisance-invariant compression — measured jointly as (a) causal-factor recovery under interventions, (b) selective rate allocation to causal vs nuisance factors, and (c) effective degrees of freedom via the LLC — predicts out-of-distribution generalization on ARC-style reasoning, or whether neural and symbolic compressors converge on the same interventional semantics.**

Each competitor owns *one* term and misses the binding:
- CompressARC: compression on ARC, **no causal/rate/DOF decomposition.**
- CRL (CITRIS/KamonBench/IST/Ahuja): causal recovery → OOD, **no compression/rate/DOF, not ARC.**
- IB=RG: relevance = compression, **field theory only, not learned nets.**
- SLT/LLC: effective DOF of learners, **never applied to ARC or causal recovery.**
- DreamCoder: symbolic MDL compression, **no causal-recovery test, no neural convergence check.**

The binding is the paper. Hold that line and don't let any single term inflate into the whole claim.

---

## 9. Threat assessment — who could scoop you, and the defense

| Who | Scoop scenario | Your defense (build it in now) |
|---|---|---|
| **Liao & Gu (CompressARC team)** | They add a causal/mechanism analysis to CompressARC | Your planted causal ledger + `do(U)` + shortcut reversal is a whole apparatus, not a bolt-on. Ship the harness early; make the ledger the contribution. Consider running *their* method inside *your* harness — co-opt, don't collide. |
| **ARC-TGI authors** | They extend nuisance sweeps into a causal-factor recovery study | You add the representation-level instrumentation (scrubbing, rate, rLLC) they don't have. Cite them as substrate, move up the stack to *mechanism*. |
| **Timaeus / devinterp** | They put rLLC on a reasoning benchmark | Your causal ledger + rate binding is not their agenda (they do interpretability of LLMs). Move fast on the `rLLC-w.r.t.-do(C)-vs-do(U)` measurement — that's the overlap point. |
| **A CRL group (Ahuja/Schölkopf-adjacent)** | They run their interventional-CRL method on ARC generators | Your compression/MDL + LLC binding and the neural↔symbolic convergence are outside standard CRL scope. Lead with those, not with "we recover causal factors" (which they'd do better). |

**Overall scoop risk: moderate, and rising.** The individual ingredients are hot (CompressARC Dec 2025, ARC-TGI Mar 2026, rLLC ICLR 2025 all landed inside the last ~14 months). The binding is still open, but the window is not indefinite. Your Phase-0 harness being fast and public is your best insurance.

---

## 10. Reading priority (do these in order)

**Tier 1 — read in full this week (direct competitors / method antecedents):**
1. Liao & Gu, CompressARC — arXiv:2512.06104. *The paper to beat; also mine its VAE-as-MDL derivation.*
2. ARC-TGI — arXiv:2603.05099. *Your substrate and biggest design overlap.*
3. Ahuja et al., "Invariance principle meets information bottleneck for OOD" (2022). *The IB×causal antecedent of your H1/H2.*
4. Wang et al., refined LLC (rLLC), ICLR 2025. *Your differentiator instrument; the `do(C)`/`do(U)` measurement lives here.*

**Tier 2 — read closely (methodology you're importing):**
5. Interventional Style Transfer, single-cell — arXiv:2306.11890. *Your falsification epistemics, in print.*
6. KamonBench — arXiv:2605.13322. *Closest planted-factor benchmark in spirit.*
7. Ahuja et al., Interventional Causal Representation Learning, ICML 2023. *Identifiability license for interventions.*
8. Lau et al., LLC, AISTATS 2025. *The effective-DOF measure.*

**Tier 3 — scope & cautionary (so you don't overstep):**
9. Gordon et al., IB=RG, PRL 126, 240601. *Cite for scope, never as license.*
10. Saxe et al. 2018 (IB-of-deep-learning critique). *Read before writing Phase 6.*
11. DreamCoder, Ellis et al. 2021. *Your symbolic arm's home.*
12. ARC Prize 2025 Technical Report — arXiv:2601.10904. *Landscape + the "efficiency gap = science" framing.*

---

## 11. Confidence & gaps in this review

- **High confidence:** CompressARC, ARC-TGI, the IB=RG lineage, and the LLC/rLLC papers are verified against primary sources with correct venues/IDs.
- **Medium confidence:** exact arXiv IDs for a few 2026 preprints (KamonBench 2605.13322, ARC-TGI 2603.05099) are as returned by search; double-check the ID before citing in a submission.
- **Known gap:** I did not exhaustively search closed-venue (non-arXiv) causal-representation-learning proceedings from 2026, nor Kaggle notebook write-ups (which occasionally contain unpublished-but-real methods). Before submission, do one pass over the **ICLR 2026 / NeurIPS 2025 accepted-papers lists** for the keywords "causal + compression + generalization" and one pass over the **ARC Prize 2025 Kaggle winners' solution threads** — a solver that quietly does causal-factor analysis could hide there.
- **Not found (a good sign for you):** no paper combining causal-factor recovery + rate allocation + LLC on ARC. If that remains true after the two passes above, the white space in §8 is real.
