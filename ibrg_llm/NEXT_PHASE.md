# Next phase — rigorously testing IB=RG (from r=0.68 to the real number)

## Where we are
E1 (headline): **selectivity** (rule − shortcut decodability) predicts **generalization**
(rule-following), **r=0.68** across 15 model×seed runs; a graded causal hint from ablation
(shortcut-subspace ablation → +0.018 rule-following vs +0.001 random). Limitations: mostly a
*between-model/scale* effect, purely *correlational*, small effects (these LLMs barely take the
shortcut). Now unbound from generative models, we can test IB=RG far more directly.

## The central IB=RG claim, sharpened
*A representation generalizes iff it selectively compresses — preserves causal (C) information
and discards nuisance (U) information.* Sub-claims (from the original MASTER_PLAN program):
- **S1** selective representation **predicts** generalization
- **S2** nuisance is **discarded with depth/compression**, causal preserved (RG flow / selective c-function)
- **S3** more **rate** goes to causal than nuisance (R_C > R_U)
- **S4** compression has **phase transitions** where relevant features are abruptly discovered (IB β-transition)
- **Causal**: *intervening* on selectivity **changes** generalization

We now attack these on **two complementary tracks**.

---

# TRACK B — Controlled IB=RG with a from-scratch VIB *(the direct test; CPU-cheap; NEW)*
Train a **Variational Information Bottleneck** model where **we set the compression β
ourselves** — the motivation's central object. Full control + ground-truth C/U ⇒ the cleanest
possible IB=RG test, and the CPU-feasible realization of the original Phase-6 physics program.

- **Data:** synthetic (or reuse `causalarc` grids). x = [causal dims that determine y] +
  [nuisance dims] + [a spurious shortcut dim correlated with y in train, **decorrelated at test**
  = the `do(U)` recombination]. Ground-truth C/U known by construction.
- **Model:** encoder q(z|x) → bottleneck z → classifier p(y|z). Loss = CE + **β·KL(z)**.
- **β-sweep** (the core experiment): train across β ∈ [0 … large], and per β measure
  - **Rate** R = KL(z) and **distortion** = task loss → the **rate–distortion curve**;
  - **Selectivity in z**: I(z;C) vs I(z;U) (or held-out probes) — what the bottleneck keeps;
  - **OOD generalization**: accuracy under `do(U)` recombination (shortcut broken).
- **IB=RG predictions, each falsifiable here:**
  - **S4** the R–D curve shows a **kink / phase transition** where the model "discovers" C;
  - **S2/S3** as β↑, **U-info drops before C-info** (nuisance discarded first) and **R_C > R_U**;
  - **S1** the β that best **preserves C / discards U** is the β that **generalizes best OOD** —
    IB=RG's core claim, with full control and no scale confound.
- **Developmental (LLC-lite, orig. B-6.3):** at fixed β, track I(z;C), I(z;U) over training
  epochs — does the net **discard U over training** while building C? (selective c-function in time).
- *All CPU: a small MLP VIB trains in seconds; a full β×seed×task sweep is a fast SLURM array.*

---

# TRACK A — IB=RG in real LLMs *(what we have; make it rigorous)*
### Tier 1 — Nail E1 (is it selectivity, or just scale/competence?)
1. **Power + de-confound:** {0.5B,1.5B,3B,7B}×10 seeds → r with CI; **within-model cross-seed**
   E1; **partial correlation controlling for aligned-accuracy** (competence).
2. **Causal centerpiece — bidirectional mediation:** graded ablation of the **shortcut** subspace
   (↑selectivity ⇒ should ↑generalization) and of the **rule** subspace (↓selectivity ⇒ ↓);
   show the interventions move points **along the E1 line** (causal, not correlational).
3. **Non-circular within-model manipulation:** change selectivity via instruction/demo-order, not
   the cue statistics, and check generalization tracks it.
### Tier 2 — Original measurements on the LLM
4. **Selective c-function (orig. B-6.5):** held-out probe (V-information) for rule vs shortcut at
   every layer×position — does shortcut-info fall with depth while rule-info is preserved? Causal
   control: make the "color" cue causal → its depth profile should **flip**.
5. **Rate allocation (orig. H2):** effective-dim / description-length of rule vs shortcut subspaces
   per layer; more rate to C, more so with scale?

---

# TRACK C — Bridges *(mixed compute)*
6. **RSMI relevance alignment (orig. Track R / HR):** run RSMI-NE on inputs; does field-theoretic
   relevance select the C DOF? The "three definitions agree" test. *TF env — stretch.*
7. **`causalarc` reconnection:** run both a VIB (Track B) and the LLM on the planted C/U grid
   families; does causal-score predict OOD across all three (RSMI ≈ ledger ≈ representation)?
8. **rLLC (orig. P3.2)** w.r.t. do(C)/do(U). *`devinterp`/training → GPU, parked.*

---

## Regime fix (do first for Track A) 
Make the shortcut easier and the rule harder so reliance spans 0→1 (currently hugs 0.5),
giving every Track-A effect real variance to explain.

## Priority
**Track B β-sweep** (fast, cleanest, most direct IB=RG test) → **Track A Tier 1** (turn r=0.68
into a causal, de-confounded claim) → Track A Tier 2 + Track C bridges.

## Why Track B is the stronger IB=RG test
Track A probes a fixed pretrained model — we can only *observe* + ablate. Track B lets us **dial
the compression (β)** and **watch selective preservation, phase transitions, and generalization
co-vary** with ground-truth C/U — which is literally the "information becomes law" mechanism the
motivation describes, testable and falsifiable. IB=RG stays *motivation*; we report the R–D
curves, effect sizes, and any nulls honestly. All of Track A/B is CPU-only, laptop-independent.
