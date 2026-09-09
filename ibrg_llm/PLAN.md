# PLAN — In-context shortcut vs. rule: reading and steering how an LLM generalizes

**Application project for MATS 12.0 (Neel Nanda).** Grounded in the Step-0 checks
in DECISIONS.md. This is the honest, real-model form of the IB=RG hypothesis:
*a model generalizes iff it compresses to the causal variables and discards the
nuisance ones.* We test that on a real LLM with ground-truth causal/nuisance
labels from the `causalarc` planted generators.

## Research question
When a nuisance feature is spuriously predictive of the answer in few-shot
demonstrations, does an LLM rely on the shortcut or infer the true rule — and can
we **read** that choice from its residual stream and **causally control** it?

## Hypotheses (confirmatory unless marked)
- **H1 (phenomenon is real — behavioral).** With a planted shortcut aligned in the
  demos, query exact-match accuracy drops under `do(U)` recombination relative to a
  no-shortcut control. *If it doesn't drop, there's no shortcut reliance to study —
  strengthen the shortcut or report the negative honestly.*
- **H2 (representational).** The rule (C) and the shortcut/nuisance (U) are linearly
  decodable from residual-stream activations before the answer token; decodability
  varies systematically by layer and token position.
- **H3 (selective coarse-graining — the IB=RG signature).** Surface-nuisance
  decodability *declines* with depth while inferred-rule decodability *rises*
  (a crossover). **Causal control:** promoting the nuisance to be causal flips its
  depth profile from discarded to preserved.
- **H4 (causal / steering).** Ablating the shortcut direction (difference-of-means
  or probe weight) reduces behavioral shortcut reliance **vs. a random-direction
  ablation of equal norm** (the key baseline).
- **H5 (applied, stretch).** A probe read at the coarse-grained layer yields fewer
  `do(U)` false positives than an early-layer probe.

## Technical setup — what we quantify and how
- **Model.** A current, capable, *dense instruct* model — target **Qwen2.5-7B-Instruct**
  (fallback 3–4B if compute-tight; parameterized by name so we can swap to
  whatever's current, e.g. a Qwen3 dense checkpoint). Deliberately **not** GPT-2 /
  Gemma-2 (Neel warns against old/tiny models). Greedy decoding, temperature 0.
- **Activations.** Residual stream per layer via HF `output_hidden_states=True`
  (or `nnsight` if we need sublayer sites). Probe positions: the last token of the
  query input, mean over query-input tokens, and the ":" before the answer.
- **Stimuli.** `causalarc` families, serialized by `ibrg_llm/stimuli.py`:
  - `F3_reflect` — **binary** causal `axis` (chance = 50%, ideal probe target),
    nuisance = `frame_color`, `distractor_color`.
  - `F2_recolor_parity` — causal = output color; doubles as the **nuisance→causal
    control** (a color that is causal here vs. nuisance in F3).
  - k = 3 demos; the rule is fixed within a task and must be *inferred* from demos.
  - Shortcut arm: top-row barcode = joint causal index in demos; recombined at the
    query for the `do(U)` / shifted split.
- **Metrics.**
  - *Behavioral:* exact-match query-grid accuracy; **shortcut-reliance = acc(aligned) − acc(recombined)**.
  - *Probe:* balanced accuracy per (layer, position, factor), 5-fold CV, averaged
    over ≥3 probe seeds; report per-factor curves over depth.
  - *Ablation:* change in shortcut-reliance after removing the direction, **minus**
    the random-direction control; effect with bootstrap CI.
- **Independent unit / stats.** Task (rule instance) is the unit; cluster-bootstrap
  CIs over tasks; never treat cells or tokens as independent samples (carried over
  from the causalarc discipline).

## Baselines & controls (Neel screens hard for these)
- Random-direction ablation of matched norm (the H4 control).
- Shuffled-label probe → chance floor for every probe number.
- "Just ask the model" behavioral baseline (no intervention).
- No-shortcut control condition (rule inferable, no barcode) — the ceiling.
- Nuisance→causal flip (the H3 mechanism control).

## Sanity checks (documented in the write-up)
- Read raw prompts **and model completions** (started in Step 0a; found the color
  collision). Confirm the shortcut is actually predictive in demos and that
  recombination truly breaks it (compute the correlation, don't assume).
- Hand-check a sample of probe "positives" — are they really that class?
- Confirm the rule is inferable at all (strong-model / oracle can do it), else H1
  is vacuous.
- Re-derive the headline shortcut-reliance number with a fresh one-liner.

## Pre-run gate items (must clear before believing anything)
1. **Fix the color collision** (DECISIONS.md, Step 0a): nuisance colors disjoint
   from the signal color(s) per task.
2. **Step 0 go/no-go on the cluster:** (a) the model infers ≥1 rule few-shot above
   chance with the shortcut absent; (b) at least one C factor is decodable above
   its shuffled-label floor. If (a) fails for grids, switch to a text-token analog
   with identical C/U/shortcut structure (substrate swap, same design). If (b)
   fails, the representational arm is dead — report that and pivot to behavioral.

## Two-day timeline (today = Sep 9; ext. deadline Sep 11)
- **Day 1 AM** — cluster env; fix colors; **Step 0 go/no-go**; pick the working
  family + model size.
- **Day 1 PM** — H1 behavioral shortcut-reliance; H2 probe sweep (layer×position) →
  the headline selectivity graph.
- **Day 2 AM** — H3 depth crossover + nuisance→causal control; H4 ablation vs
  random-direction.
- **Day 2 PM** — sanity checks, randomly-sampled qualitative examples, write-up +
  executive summary + graphs. (H5 only if time.)
- Fallback ladder if we run short: ship H1+H2 with clean sanity checks and an
  honest "what we'd do next" — a well-analyzed partial result beats an
  over-claimed full one.

## Compute needed (the ask)
- **One GPU with ≥24 GB** (an A6000 is plenty). **No training** — inference +
  activation extraction only, so this is light: a few thousand forward passes over
  2 families × a few conditions is minutes-to-an-hour.
- **Env:** `torch` (the cluster `agi` env already has 2.6+cu126) **+ `transformers`
  + `accelerate` + `scikit-learn`** (+ optional `nnsight`). HF hub access to
  download a 4–8B instruct checkpoint (~8–15 GB disk).
- **Nothing exotic:** no multi-GPU, no fine-tuning, no long jobs.
