# The no-deadline research program — IB=RG on real LLMs, scaled on CPU

**Substrate:** the in-context shortcut-vs-rule task (a spurious "color" cue vs the
true "shape" rule). Everything below is **inference / probing / activation
steering** — all CPU-feasible and parallelizable across many SLURM jobs.
Fine-tuning is NOT CPU-feasible → parked for GPU.

Each experiment is tagged with the **IB=RG idea** it operationalizes and the
**Neel recommendation** it serves.

## A — Scale & robustness  *(CPU ✓, no new code — just parameters)*
- **A1 Model-scale sweep** {0.5B, 1.5B, 3B, (7B if disk)} × seeds, full H1–H5.
  *Does shortcut-reliance, cue-selectivity, and steerability change with scale?*
  IB: does more capacity mean cleaner compression to the causal variable?
  Neel: science of generalization, scaling.
- **A2 Many seeds (≥20)** on the base config → tight bootstrap CIs.
- **A3 Task-structure generality** — cue types beyond shape/color (position,
  count, first-letter, parity). Does the phenomenon replicate across rules?

## B — The IB phase diagram  *(CPU ✓, cheap logit-only; highest IB=RG alignment)*
- **B1** 2-D sweep: **shortcut strength** (how perfectly the cue predicts in the
  demos: 100/90/80/70%) × **#demos** → shortcut_pref. The motivation's *IB phase
  transition* (β × evidence) made concrete — look for a sharp reliance boundary.
- **B2** add model scale as a 3rd axis → does the transition sharpen with scale?

## C — Causal mechanism & steering  *(CPU ✓, most Neel-aligned)*
- **C1 Causal layer-sweep** — ablate the shortcut subspace at **each** layer;
  which layer is load-bearing for the behavior? (localizes the shortcut mechanism)
- **C2 Minimal direction** — 1-D vs full-subspace ablation → minimal sufficient
  shortcut direction.
- **C3 Bidirectional dose-response** — *add* the shortcut direction (push toward
  shortcut) vs *ablate* (push toward rule), swept over magnitude.
- **C4 Activation patching** — transplant the shortcut representation from one
  task into another; does the shortcut behavior transfer? (causal, not correlational)
  IB: steering the compression. Neel: steering / causal intervention / patching.

## D — Selective coarse-graining (RG flow)  *(CPU ✓)*
- **D1** full layer × token-position × cue decodability tensor.
- **D2 Nuisance→causal flip** — make the color cue causal; does its depth profile
  flip from discarded→preserved? (the MASTER_PLAN "selective c-function" test)
- **D3** does nuisance info actually get *discarded* with depth (RG prediction) or
  retained? Our earlier grid pilot hinted **retained** — test rigorously here.
  IB: layerwise coarse-graining. Neel: model biology of ICL.

## E — The headline IB=RG claim  *(CPU ✓)*
- **E1 Selectivity-predicts-generalization** — across conditions/models, correlate
  probe **selectivity** (rule-decodability − shortcut-decodability) with **OOD
  behavior** (rule-following on conflict/recombined queries). *Does compressing to
  the causal variable predict generalization?* — the core thesis, one scatter plot.

## F — Applied monitoring  *(CPU ✓, Neel applied-interp)*
- **F1** shortcut-probe as a **monitor**: train on some tasks, predict
  shortcut-reliance on held-out tasks; is the coarse-grained layer the best read?

## Parked (needs GPU — not CPU)
Fine-tuning model organisms, synthetic-document finetuning, training-dynamics /
IB-transition *during training*. Queue when a GPU frees.

## Queue order
- **Wave 1 (now, reuses runner):** A1 scale sweep + A2 seeds.
- **Wave 2:** B (phase diagram) + C1 (layer-sweep ablation) — small runner flags.
- **Wave 3:** C3/C4, D, E1, F1.
Every wave = a CPU-only `sbatch` array (schedules instantly, laptop-independent),
results aggregated across seeds into one figure per experiment.
