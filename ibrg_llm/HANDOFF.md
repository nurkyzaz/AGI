# HANDOFF — IB=RG on real systems: status + the single forward plan
**The one plan for the next agent.** Raw chronological log: `DECISIONS.md`. Original motivation &
prior program: `../ Intellectual Motivation.md`, `../MASTER_PLAN (1).md`, `../LITERATURE_REVIEW_*.md`.

---

## 0. Where everything lives  (read first)
- **GitHub (source of truth):** `github.com/nurkyzaz/AGI`, branch **`claude/agi-research-review-1244c8`**,
  directory **`ibrg_llm/`**. The IB=RG work is on this branch, **not `main`**. Get it with:
  `git clone https://github.com/nurkyzaz/AGI && cd AGI && git checkout claude/agi-research-review-1244c8`
- **Code:** `ibrg_llm/*.py` (task/model/experiment modules — see §2 for what each does).
- **Plan + log:** `ibrg_llm/HANDOFF.md` (this file) + `ibrg_llm/DECISIONS.md`.
- **Results & figures:** `ibrg_llm/out/*.json` (per run) + `ibrg_llm/out/fig_*.png` — committed to the
  branch; regenerate any figure with `python -m ibrg_llm.aggregate`.
- **Cluster working copy:** `gpus:~/agi/ibrg_llm/` is **loose scp'd files, NOT under git**. Treat
  GitHub as authoritative: `scp` code up before running, `scp` results back (results also live at
  `gpus:~/agi/ibrg_llm/out/`). The cluster `~/agi` is *not* a git repo.

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
### Track B — Foliation & criticality (`crit.py`)  *(P1, DONE 2026-09-13; CPU-local, 5 seeds)*
Code: `crit.py` (runs `aligned` + `recomb`/do(U) conditions across a β-sweep). Results: `out/crit_seed*`.
Figures: `fig_foliation`, `fig_fdt`.
- **E1 foliation (SUPPORTED)**: on the training distribution a rule direction (`v_C`) and a shortcut
  direction (`v_S`) decode the label **equally** (aligned β=0: both 1.00 → observational separation-AUC
  **0.48**, i.e. observation can't tell them apart). Cross-environment invariance (min-over-env
  label-decodability — the IRM criterion, *not* the literal do() response, see design note) **separates**
  them: rule 0.83 vs shortcut 0.73; inv-AUC up to **0.86–0.91** (recomb, low β). "Rule invisible to
  observation, visible to intervention."
- **E2 criticality (NULL — reported, not manufactured)**: the `v_C`–`v_S` separation is **largest at the
  weakest compression** (β≈0.001) and shrinks as β↑; it does **not** peak at the generalization-optimal
  β* (recomb OOD optimum ≈0.1). "Susceptibility peaks at β*" is not supported in this VIB. Reason: a
  cheap shortcut is kept at every β (no robust interior β* inverted-U from pure compression — the
  boundary phenomenon), so β* was taken as the OOD optimum and the null reported honestly.
- **E3 fluctuation–dissipation (descriptive)**: χ(β) (the do(z+=εv) response) and Var(z) both fall
  monotonically with β (χ 3.99→0.001, Var 253→0.02), no shared peak.
- **Design note (why crit.py is shaped this way; verify before extending)**: (1) obs-AUC≈0.5 needs the
  shortcut to be as Y-predictive as the rule on train → the **aligned** condition; under recomb the rule
  is genuinely more readable (obs-AUC≈0.75), so both conditions are run. (2) The literal `do(z+=εv)`
  response is ~environment-invariant for a *fixed* model (both v_C and v_S), so it can't be the E1
  discriminator — cross-environment invariance of the label relation is. Kept do() only for χ (E3).

## 3. NEXT — the forward plan (priority order)
### P1 — Foliation & criticality  ✔ **DONE (2026-09-13)** — see §2 "Track B — Foliation & criticality"
`crit.py` built and run (5 seeds). **E1 foliation SUPPORTED** (obs-AUC 0.48; inv-AUC up to 0.86–0.91).
**E2 criticality is a NULL** (separation peaks at weak β, not at β*). **E3 FDT** descriptive (χ, Var(z)
fall together, no peak). Deliverables `fig_foliation`, `fig_fdt` shipped. **→ Start with P2 below.**

### P2 — Harden E1 (make r=0.68 causal & scale-free)  *(do first now; CPU)*
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
