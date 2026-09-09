# Brainstorm — mapping our approach onto Neel's recommended problems

**Our approach, in one line:** a *ground-truth-controlled instrument* for the
question "which information does a model actually use vs. discard, and does that
predict how it generalizes?" — planted **causal (C)** vs **nuisance (U)** factors,
`do(U)` recombination to break spurious correlations, layer×position **linear
probes**, and **causal ablation/steering** of the directions we find. This is the
honest, empirical, real-model form of the IB=RG north star: a model generalizes
iff it compresses to the *relevant* variables and throws the rest away.

Below, each of Neel's recommended areas → the concrete question our instrument
could answer → an honest fit call.

---

### 1. Science of Generalization  — **BEST FIT**
Neel's framing (verbatim theme): *"why do models generalize to one solution over
another when both perform well on training data — emergent misalignment is just a
very clean example of this being weird."*

**Our question:** *In-context, when a nuisance feature is spuriously predictive of
the answer in the few-shot demos, does the model latch onto the shortcut or infer
the true rule — and is that choice visible and steerable in the residual stream?*

- We **plant** the shortcut (a top-row barcode that perfectly predicts the answer
  in the demos) and break it at the query with `do(U)` recombination. We *know* the
  causal rule and the shortcut by construction — the controlled analog of
  "general vs narrow solution," but with ground truth Neel's emergent-misalignment
  setups lack.
- Behavioral (does accuracy collapse under recombination?) + representational (is
  "using the shortcut" decodable?) + causal (ablate the shortcut direction → does
  it revert to the rule?).
- **Why it fits:** hits his flagship live interest, is causal not just
  correlational (he weights this heavily), and has ground truth + baselines.

### 2. Applied Interpretability → Monitoring probes — **STRONG, "useful" payoff**
Neel: *"probing is SOTA for cheap monitoring… how can probes be improved? cases
where information is spread across tokens, or long context with false positives."*

**Our question:** *Does reading a probe at the layer where the model has
discarded nuisance give more nuisance-robust monitoring — fewer false positives
under `do(U)` — than an early-layer probe?* Our C/U selectivity map tells us
*which layer* to read; the win is a concrete monitoring improvement, measured
against the obvious early-layer / last-layer baselines.

### 3. Circuit analysis → automate probing baselines — **GOOD, methodological**
Neel: *"can we automate and scale the process of testing many linear probes, at
all appropriate layers / token positions, for a given task?"*

**Our question:** *An automated causal-vs-nuisance probe sweep with ground truth:
where (layer, token position) is a latent rule best decoded, and how does that
differ between an inferred causal variable and a surface nuisance one?* Our sweep
*is* this, and the planted labels let us score it honestly (not just "a probe
exists").

### 4. Basic Science / computation — **PUREST IB=RG, but "basic science" risk**
**Our question:** *Does a transformer do RG-like selective coarse-graining —
discarding surface nuisance with depth while building up the inferred rule — and
can we flip the profile with a causal control (promote the nuisance to causal)?*
This is the cleanest test of the north star, but Neel says he's now lukewarm on
pure basic science, so it should ride *inside* #1/#2, not stand alone.

### 5. Concept representations / model biology — **WEAK FIT**
Probing truth/deception/uncertainty is on his list, but those aren't where our
controlled-generator + `do(U)` machinery has an edge; skip.

---

## Recommendation

**Spine = #1 (shortcut-vs-rule generalization), mechanism = #4 (depth
selectivity), payoff = #2 (nuisance-robust monitoring).** One coherent story:

> *A real LLM's in-context generalization reflects whether it compressed to the
> causal rule or a spurious nuisance. We can (a) measure the choice behaviorally
> via `do(U)` recombination, (b) read it from the residual stream, (c) show the
> nuisance is selectively discarded with depth (IB=RG signature) and flip that
> with a causal control, and (d) causally steer the choice by ablating the
> shortcut direction.*

Scoped to 2 days, the non-negotiable core is (a)+(b)+(c-ablation); (depth
selectivity) and (monitoring) are the stretch. See PLAN.md.
