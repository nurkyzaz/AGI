# Next phase — the foliation and its criticality
### (the empirical core of "information becomes law")

## Synthesis: the two proposals are one experiment
- **Proposal 1 (foliation).** Causal vs spurious directions are **indistinguishable by any
  observational/geometric statistic** (variance, curvature, IB-relevance) but **separated by an
  invariance/response score under do(·)**.
- **Proposal 2 (criticality).** The **interventional response χ(β) peaks at the IB critical β\***
  — where compression is optimized and susceptibility is maximal.

They compose into one claim, testable in a single β-sweep:

> **The causal/spurious foliation is invisible to observation and revealed only by interventional
> response — and that response signal is maximal at the compression critical point β\*.**
> Relevance (compression/RG) finds the manifold; response (intervention) reveals the law; both
> peak at criticality. Where response = fluctuation (FDT) is where correlation = causation; where
> they diverge is where the law lives.

## The necessary experiments (pruned to the core)
Substrate: the **controlled VIB** (Track B) — it has ground-truth causal (`v_C`) and spurious
(`v_S`) directions and a tunable compression β. Reuses `vib.py`/`boundary.py` machinery. CPU-only.

### E1 — The foliation *(THE necessary core)*
On the trained representation z, for the ground-truth causal and spurious directions:
- **Observational stats** (predict: `v_C ≈ v_S`, cannot separate):
  Y-decodability from the 1-D projection ⟨z,v⟩ on train · variance/energy along v · (curvature — *dropped*, see below).
- **Interventional stats** (predict: `v_C ≠ v_S`, cleanly separate):
  response to `do(z += ε·v)` measured on **recombined/OOD** data · cross-environment invariance of
  the Y↔v relationship (aligned vs `do(U)` environment).
- **Quantify the foliation.** Build many directions (seeds × causal/shortcut sub-blocks), then
  score **observational-AUC** (can observational stats classify causal-vs-spurious? → expect ~0.5)
  vs **interventional-AUC** (→ expect ≫0.5). The **gap = the foliation, made quantitative** — the
  manifold analog of "you can't read the law off the fluctuations."

### E2 — Criticality of the response *(makes it "intervene at criticality")*
Sweep β; at each β compute the **interventional separation** (response gap between `v_C` and `v_S`)
and the **generalization curve** (to locate β\* / the rate–distortion kink).
- **Predict:** the response gap **peaks at β\***, coinciding with the generalization optimum. This
  is "the optimal, minimal-cost point to intervene is the compression phase transition."

### E3 — Fluctuation–dissipation *(theoretical glue; descriptive)*
Per β: interventional response **χ(β)** and observational fluctuation **Var(z)**.
- **Predict:** χ ≈ Var near β\* (FDT holds → correlation ≈ causation there); they **diverge** away
  from β\*, and the divergence tracks the causal/spurious separability and the generalization jump
  (the FDT-violation → generalization story, arXiv:2607.04135).

## Pruned, and why
- **Manifold curvature / full geometric estimators** — high-variance on a small z, marginal signal
  over variance + decodability. Drop (keep variance + IB-relevance as the observational stats).
- **Multiple task families, the real-LLM (Track A) version** — corroboration, not core. Defer;
  the controlled VIB is where the geometry and criticality are cleanly measurable.
- **RSMI / full-DAG identifiability** — out of scope for this phase.

## The one figure that states the whole thesis
A single panel, x = β:
- **generalization** (locates β\*),
- **observational discriminability** of causal-vs-spurious → flat at chance,
- **interventional discriminability** → peaks at β\*.
If the interventional curve peaks at the generalization optimum while the observational curve stays
at chance, that plot *is* the thesis: **causation is invisible to observation, visible to
intervention, and maximally so at criticality.**

## Compute
One CPU-only SLURM array over seeds (VIB is seconds/seed), laptop-independent via `sbatch` +
orchestrator. `crit.py` implements E1–E3 in one β-sweep; `aggregate.py` gets `fig_foliation`
(the thesis panel) + `fig_fdt`.
