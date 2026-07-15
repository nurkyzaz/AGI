# ARC-AGI Benchmark Landscape (as of July 2026) — Target Selection

## The three benchmarks

| | ARC-AGI-1 (2019) | ARC-AGI-2 (2025) | ARC-AGI-3 (Mar 2026) |
|---|---|---|---|
| Format | Static grid puzzles, 400 eval | Static grid puzzles, harder; 1000 train / 120 public eval / private eval | **Interactive** turn-based game environments; agent explores, no instructions |
| Human perf | ~85–100% | 66% avg individual, 100% panel | 100% of levels |
| Frontier AI (unconstrained) | ~87%+ (saturated) | 37.6% Opus 4.5; 54% Poetiq/Gemini-3; newer claims ~77–85% at high $/task | **<1%** (Gemini 3.1 Pro 0.37%, GPT-5.4 0.26%, Opus 4.6 0.25%) |
| Constrained SOTA (Kaggle) | ~81% (ARChitects 2024) | **24.03% (NVARC, 2025 winner)** | — (new; agent toolkit/API) |
| Status | Effectively solved / calibration only | **Active: ARC Prize 2026 Kaggle track, 85% grand-prize threshold** | Active: $700K for 100%; milestones Jun 30 / Sep 30, 2026 |

## Key facts driving the decision

1. **The Kaggle ARC-AGI-2 track is our regime.** Rules: no internet during evaluation
   (no GPT/Claude APIs), fixed Kaggle GPU budget, open-source required. This *enforces*
   the efficiency thesis — a frontier-LLM harness like Berman's is ineligible; a
   cascade of cheap tiers + small local LLM is exactly what can win here.
2. **The constrained-vs-unconstrained gap is the opportunity.** Public frontier scores
   (~54–85%) vs. Kaggle winner 24% means the prize-relevant frontier is wide open, and
   the winning 2025 recipe (NVARC = synthetic data + test-time training + TRM
   ensemble) is literally our T2 tier. Beating 24% under Kaggle limits = top of field.
3. **Our paper stack is validated by the 2025 Paper Prizes:** TRM won 1st ($50k),
   SOAR 2nd ($20k), CompressARC 3rd ($5k). We're building on the community-vetted
   best ideas. (Note: CompressARC's 20% was ARC-1-eval only; it was never run on
   ARC-2 — porting its MDL objective to ARC-2 is untested territory.)
4. **ARC-AGI-3 is where the field is going, but it's a different problem.** Exploration,
   goal acquisition, world models, long-horizon planning — an agent loop, not a
   puzzle-compression loop. Frontier <1% means enormous headroom, but our current
   toolkit (MDL, DSL search, TTT, sketch evolution) addresses per-turn reasoning, not
   the exploration scaffold. Entering seriously would be a second project.
5. **ARC-AGI-1 remains useful only as a cheap calibration set** (fast signal, known
   baselines for every method we reimplement).

## Re-evaluation addendum (2026-07-10, second pass)

The "<1% frontier" figure understates ARC-3 tractability: the **community
leaderboard** (harnesses allowed) shows purpose-built agents at **63.7%**
(Rodionov's "baseline1", $350/run — a coding agent that builds and *verifies an
executable Python world model*, then plans through it), 63.1% ($4,788/run,
continual-learning multimodal agent), 50.2%, 43.9% ($1,406/run). Three points:

1. ARC-3 yields to harness engineering — but every strong agent runs on
   **frontier APIs at $350–$4,800 per run**, exactly the cost regime this project
   exists to escape. An *efficient small-model* ARC-3 agent is unexplored — a real
   future opportunity, not a reason to switch now.
2. The best ARC-3 recipe (executable world model + verification + planning) is
   our cascade philosophy transposed: shortest verified world-model ≈ MDL; our T3
   sketch→expander loop generalizes to "propose world-model program → verify
   against observed transitions". The skills we build for ARC-2 transfer.
3. Still ARC-2 first: our validated paper stack, a beatable constrained SOTA
   (24%), one shared GPU, and one Nov 2 deadline — splitting focus across two
   benchmark formats risks losing both.

**Verdict: primary target unchanged (ARC-AGI-2, Kaggle constraints). ARC-3 spike
upgraded from "optional" to a planned Phase-5 decision point (~Sept 2026):
reuse the T3 sketch/world-model machinery with a small local model on the 3
public games.**

## Decision

- **Primary target: ARC-AGI-2, optimized under ARC Prize 2026 Kaggle constraints.**
  Metrics: pass@2 on ARC-2 public eval (120 tasks) + cost-per-task; design everything
  to be Kaggle-submittable (offline, ≤Kaggle GPU/runtime, small local models only).
  Re-anchored goal: **≥24% (beat NVARC) under Kaggle limits**; stretch 30%+.
- **Secondary: ARC-AGI-1 + ConceptARC for calibration** of every tier against
  published numbers.
- **Exploratory (Phase 5, optional): ARC-AGI-3 agent prototype** — wrap the cascade's
  perception + hypothesis-search core in an exploration agent using the ARC-AGI-3
  toolkit (docs.arcprize.org). Decision point after Phase 3: if cascade research is
  on track by ~Sept 2026, spike 2 weeks on AGI-3; the Sep 30 milestone prize is the
  motivation. Do not let it cannibalize the main track.

## Consequences for the plan (deltas to RESEARCH_PLAN.md)

- T3's sketcher LLM must be **local and Kaggle-deployable** (quantized 7–14B fits
  Kaggle's 4×L4 96GB; ARChitects 2024 proved finetuned-LLM submissions work). No
  frontier-API dependency anywhere in the solve path; API models allowed only for
  offline seed-trace generation (distillation) before submission.
- Add Kaggle runtime budget (~12h for the full private eval set) as a hard constraint
  in the harness cost model: ≈ **5–6 min/task max**, cascade median must be far below.
- Success metric updated: primary = private-eval-style score under Kaggle limits;
  the $/task frontier plot remains the paper's headline figure.

Sources: [ARC Prize 2026](https://arcprize.org/competitions/2026), [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis), [ARC-AGI-3](https://arcprize.org/arc-agi/3), [ARC-AGI-3 launch](https://arcprize.org/blog/arc-agi-3-launch), [Kaggle ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard)
