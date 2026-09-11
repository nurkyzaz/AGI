# PLAN v2 — Shortcut vs. rule in in-context learning: reading and steering how an LLM generalizes
**The earned, real-model form of IB = RG.** Application project for MATS 12.0 (Neel Nanda).

## Intellectual motivation, made testable
North star (IB = RG): a system generalizes by **selective compression** — keeping the
variables that *causally* determine the output and discarding nuisance, the same
coarse-graining RG performs in physics (Gordon et al. 2021 — proven for field
theories only). The open empirical question the motivation doc poses: *does a
learned network actually do this?* We test it on a **real LLM** with ground-truth
**causal (rule)** vs **nuisance (spurious shortcut)** cues, in-context.

We turn four ideas from the motivation into cheap, real-model measurements, plus
the causal test that makes it Neel-shaped:

| # | IB=RG idea (from motivation) | Measurement | Neel recommendation it serves |
|---|---|---|---|
| H1 | selective preservation **predicts generalization** | behavioral: does the model follow the shortcut or the rule on conflict queries? | **science of generalization** (why one solution over another) |
| H2 | **selective preservation** of causal vs nuisance | probe: which cue is linearly decodable, rule vs shortcut? | monitoring probes / concept representations |
| H3 | **layerwise coarse-graining (RG flow)** | probe rule vs nuisance decodability **across depth** | model biology of ICL |
| H4 | **steering the compression** | **ablate the shortcut subspace** → does generalization shift to the rule? (vs random-direction control) | **steering / causal intervention** (his favorite) |
| H5 | **IB phase transition** (β drives abrupt feature discovery) | k-sweep: as demos increase, does reliance flip shortcut→rule, and is it abrupt? | why one solution over another |

## Task (text ICL on a real model)
Each demo pairs a nonsense **shape** word and **color** word with a single-token
label. Within a task the **shape→label** map is the true rule; **color** is a
spurious cue perfectly aligned with the label in the demos. The query is a
**conflict**: its shape implies one label, its color another. Which label the
model prefers reveals whether it generalized via the rule or the shortcut.
Nonsense words + digit labels force genuine ICL (no lexical prior) and a clean
logit readout. Model: **Qwen2.5-3B-Instruct** (CPU-run; upgradeable).

## Metrics
- **H1** shortcut_pref = p(shortcut label)/(p(shortcut)+p(rule)) on conflict; hard-vote too. Sanity: aligned-query accuracy (did the model learn the mapping at all?).
- **H2/H3** balanced-accuracy probes of shape-class (rule) and color-class (shortcut) per layer, scaled-feature logistic regression, vs shuffled-label floor. Chance = 1/3.
- **H4** Δshortcut_pref after ablating the color (shortcut) subspace, **minus the random-subspace control**; contrast with ablating the shape (rule) subspace.
- **H5** shortcut_pref vs demos-per-class k ∈ {1,2,3,4}.
- Independent unit = **seed** (5 seeds); report mean ± cluster/seed bootstrap.

## Baselines & controls (Neel screens hard)
Random-direction ablation of equal rank (H4 control) · shuffled-label probe floor
(H2/H3) · aligned-accuracy sanity (model must be able to learn the mapping) ·
shape-subspace ablation contrast · "hard vote" alongside the soft preference.

## Sanity checks (in the write-up)
Print label token ids + model top-5 on an aligned prompt (already catches the
tokenization bug) · read raw prompts/completions · confirm conflict truly
conflicts · re-derive the headline shortcut_pref with a fresh one-liner · assert
label ids distinct.

## The CPU job set (exactly what runs — matches this plan)
- `ibrg_llm/shortcut.sbatch`: **CPU-only** SLURM array, **seeds 0–4**, each task runs
  the *full* pipeline **H1 + H2 + H3 + H4 + H5** and saves `out/shortcut_seed{S}.json`.
- CPU-only ⇒ **schedules immediately** despite GPU contention; `sbatch` ⇒ SLURM
  controller owns it ⇒ **runs to completion after the laptop closes**.
- After: aggregate 5 seeds → mean ± CI per hypothesis; one figure per H.

## Honest caveats (state up front)
- A 3B may prefer the shortcut *trivially*; the interesting, causal result is
  whether H4 can **steer it back** to the rule — that's the load-bearing claim.
- ICL cue-competition is a proxy for "compression to causal variables," not the
  physics claim; IB=RG stays motivation, never asserted.
- Single model, single task family, CPU (fp32) — we report it as a focused study,
  not a universal law, and list the obvious next steps (bigger model, more
  families, token-position sweep).
