# HANDOFF — IB=RG on real systems: status + the single forward plan
**The one plan for the next agent.** Raw chronological log: `DECISIONS.md`. Original motivation &
prior program: `../ Intellectual Motivation.md`, `../MASTER_PLAN (1).md`, `../LITERATURE_REVIEW_*.md`.

---

## 1. The thesis (what we are testing)
**Information becomes law through *selective compression* + *intervention*.**
- **Compression (IB=RG)** selectively preserves causal information and discards nuisance — but on a
  single observational distribution it *cannot* separate a causal feature from a cheap spurious one
  (both are valid minimal sufficient statistics; it even prefers the cheaper spurious one).
- **Intervention (`do(U)`)** supplies the missing causal information: causal = invariant under
  intervention / across environments.
- **Geometry vs response:** the causal/spurious distinction is *invisible to observation/geometry*
  and *visible only to interventional response* — and is (hypothesised) *maximal at the compression
  critical point* (criticality ⇒ max susceptibility).

Grounding: Causal Information Bottleneck ([2410.00535]), IRM ([1907.02893]), IMA + Manifold
Hypothesis ([2312.13438]), IB phase transitions ([2001.01878]), Physics of DL & Brains
([2509.22649], the motivation paper), FDT-violation → generalization ([2607.04135]).
IB=RG stays *motivation*, never asserted. Report effect sizes and nulls honestly.

## 2. What's DONE (with file + result pointers)
### Track A — real LLM (Qwen2.5 {0.5,1.5,3}B), in-context shortcut-vs-rule
Code: `shortcut_task.py`, `runner.py` (H1–H5), `phase.py` (B+D), `ablsweep.py` (C1/C3).
Results: `out/shortcut_seed*`, `shortcut_q{05,15}b_s*`, `phase_seed*`, `abl_seed*`.
Figures: `fig_base_H1_H4`, `fig_scale`, `fig_phase`, `fig_ablsweep`, `fig_selectivity`, `fig_monitor`.
- **H1**: 3B is ~rule-robust (shortcut_pref 0.49; aligned acc 0.997). **Scale**: reliance falls with
  size (0.55→0.50→0.49). **H4/C1**: shortcut-subspace ablation causally shifts behaviour, localised
  to mid-layers, small (+0.018 vs random +0.001). **B**: reliance ↑ with shortcut strength, ↓ with
  #demos. **E1 (headline)**: selectivity predicts generalization **r=0.68** (largely *between-model*).
  **F**: probe-monitor is null (r=−0.34).
### Track B — from-scratch VIB (controlled, ground-truth C/U, tunable β)
Code: `vib.py` (nuisance), `boundary.py` (spurious-shortcut vs intervention). Results: `out/vib_seed*`,
`boundary_seed*`. Figures: `fig_vib`, `fig_boundary`.
- **VIB**: as β↑, nuisance is discarded (reconstruction 0.63→0.00) while causal is preserved
  (~0.90); generalization gain is weak (nuisance didn't hurt OOD).
- **BOUNDARY (headline)**: compression alone **cannot** fix a cheap spurious shortcut (OOD ~chance at
  every β; shortcut retention = 1.0), but **intervention `do(U)`** does (OOD **0.46 → 0.90**). The
  earned "compression ≠ generalization; interventions are required."

## 3. NEXT — the forward plan (priority order)
### P1 — Foliation & criticality  *(do first; the empirical capstone; CPU)*
Build `crit.py`; one β-sweep on the VIB producing three measurements per β:
- **E1 foliation** — for ground-truth causal (`v_C`) vs spurious (`v_S`) directions in z:
  *observational* score (Y-decodability of ⟨z,v⟩ on train; variance along v) and *interventional*
  score (response to `do(z += εv)` on recombined/OOD data; cross-environment invariance). Over many
  directions (seeds × sub-blocks) report **observational-AUC (predict ≈0.5)** vs
  **interventional-AUC (predict ≫0.5)** — the gap is the foliation, quantified.
- **E2 criticality** — the interventional separation (`v_C` vs `v_S` response gap) **peaks at β\***
  (the generalization optimum / rate–distortion kink). Locate β\* from the generalization curve.
- **E3 fluctuation–dissipation** *(descriptive)* — response χ(β) vs fluctuation Var(z): coincide near
  β\*, diverge where generalization jumps.
- **Deliverable**: `fig_foliation` — the thesis panel (x=β: generalization locates β\*;
  observational-discriminability flat at chance; interventional-discriminability peaks at β\*) + `fig_fdt`.

### P2 — Harden E1 (make r=0.68 causal & scale-free)  *(CPU)*
Add to `runner.py`/a new driver: **bidirectional ablation mediation** (ablate shortcut → ↑selectivity
→ predict ↑generalization; ablate rule → ↓selectivity → ↓); **within-model cross-seed** E1;
**partial correlation** of selectivity↔generalization controlling for aligned-accuracy. Turns the
between-model correlation into a de-confounded, causal claim.

### P3 — Optional corroboration / bridges
- **Selective c-function + rate allocation on the LLM** (layerwise: does nuisance-info fall with
  depth while causal is preserved?). - **LLM corroboration of the foliation** (obs-invisible /
  int-visible in a real model). - **RSMI relevance-alignment** (Track R, TF env — heavy);
  **`causalarc` grid reconnection**; **rLLC** (needs training → GPU).

## 4. PRUNED / abandoned (and why — don't redo these)
- **Serialized-grid ARC task** (`step0.py`, `stimuli.py`, `colab_ibrg.py`): the 3B is behaviourally
  dead on text-serialized grids (exact=0, wellformed=0) → replaced by the text ICL task. Legacy, kept.
- **Old causalarc GATE program / physics Phase-6 on grid CNNs** (`../causalarc`, `../MASTER_PLAN`):
  superseded by the LLM+VIB approach; GATE-1 v2 never landed. Historical only.
- **Manifold curvature, multi-task-family, full-DAG/RSMI inside P1**: high cost, low marginal signal
  → deferred (variance + IB-relevance are the observational stats we keep).

## 5. Infrastructure (how to run)
- **Cluster**: `ssh gpus` (physics dept, key auth; tcsh login → send remote cmds via `bash -s`
  heredocs). Env python: `/home/user/nurkyz/miniconda3/envs/agi/bin/python` (torch 2.6+cu126,
  transformers, scikit-learn, matplotlib). Repo `~/agi`, `PYTHONPATH=$HOME/agi`.
- **Compute**: GPUs are contended (jobs pend for days); **CPU-only `sbatch` arrays schedule instantly**
  and are laptop-independent. **QOS cap = 8 submitted jobs** — submit in ≤5-task chunks;
  `orchestrate.sh` sequences experiments within the cap. VIB seeds are seconds each.
- **Aggregate**: `python -m ibrg_llm.aggregate` → all figures into `out/`. Add new `fig_*` functions
  there for P1/P2.
- **Discipline**: smoke-test every new script at tiny N before an array (two real bugs were caught
  this way); assert-check label/probe wiring; keep the correlational / causal / descriptive split.
