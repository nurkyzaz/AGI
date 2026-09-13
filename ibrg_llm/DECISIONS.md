# IB=RG on a real LLM — decision & execution log

Running log for the MATS 12.0 (Neel Nanda) application project. Newest entries
appended at the bottom. This file documents *what I did and why* — every design
choice, dead end, and pivot — so the reasoning is auditable.

**One-line project:** Does a *learned* LLM do RG-like selective coarse-graining —
preserving causally-relevant factors of an input while discarding nuisance
factors across depth — and does that selectivity predict shortcut-reliance /
OOD generalization? This is the honest, falsifiable, real-model form of the
"IB = RG in neural nets" hypothesis (the north star), using ground-truth
causal/nuisance labels from the existing `causalarc` planted generators.

---

## 2026-09-09 — Environment probe (feasibility gate for *where* work runs)

Probed the machine this agent runs on:
- **MacBook Pro (M1), 8 GB RAM, 8 cores, 31 GB free disk.**
- Python 3.9.6; `numpy` 2.0, `sklearn` 1.6 present. **No `torch`, `transformers`,
  `nnsight`, `huggingface_hub`.** Network to huggingface.co OK (200).

**Decision — split the work by machine:**
- The **real experiment** (capable model: Qwen-4B/9B-class, per Neel's rec) must
  run on the user's **SLURM A6000 cluster** — an 8 GB M1 cannot hold a 4B model.
- On *this* machine I will: reuse `causalarc` (numpy-only) to build stimuli,
  read the serialized prompts by hand, and — timeboxed — install torch(CPU) +
  transformers and run a **tiny** model (~0.5 B) purely to *validate the
  pipeline* (serialize → tokenize → hidden states → linear probe → selectivity
  curve). Pipeline/setup validation on a small model is explicitly fine under
  Neel's time rules and is not the scientific result.
- The tiny smoke-test model is **not** the subject of the study; the cluster
  script is parameterized by model name.

**Why reuse `causalarc` as the stimulus source:** it already emits, per few-shot
task, ground-truth causal codes (`c_code`), nuisance codes (`u_code`), and the
`do(U)`+recombination barcode split (`datasets.sample_task`). That is exactly the
labelled C/U supervision the probing study needs — no new generator required.

## 2026-09-09 — Step 0a: read the serialized data (`ibrg_llm/stimuli.py`)

Built the serializer (grid -> rows of digits; few-shot Input/Output blocks) and
printed real prompts. Findings from eyeballing:
- **Serialization is clean and the task is well-posed as text.** k=3 few-shot is
  ~1150 tokens — comfortable for a 4-8B model.
- **BUG FOUND — color collision (must fix before any run):** the signal object's
  color (F3/F4 use `1 + seed%8`) can equal the nuisance `frame_color` /
  `distractor_color` (sampled from 1-9). In the very first F3 sample the query
  object was color 6 and the frame was also 6, camouflaging the object against
  the border *and* entangling the `frame_color` nuisance label with the signal.
  This would confound both the model's ability to do the task and the
  cleanliness of the U-probe. **Fix:** constrain nuisance colors to be disjoint
  from the signal color(s) used in a task (or canonicalize the object to a
  reserved color and exclude it from the nuisance palette). Deferred to the
  implementation pass; tracked as a pre-run gate item in PLAN.md.

## 2026-09-09 — Compute decision

Real runs need a GPU. This M1 cannot hold a 4-8B model, so Step 0's *decodability*
check runs on the **cluster** (user is arranging access), not here. On this
machine I finish: stimulus code, activation-extraction + probing code (unit-
tested against mock tensors), and the runbook. See PLAN.md for the exact ask.

## 2026-09-10 — Cluster access + Step-0 launch (CUHK physics `gpus`)

Got working passwordless access via the existing `gpus` alias
(`gpus.phy.cuhk.edu.hk`, user `nurkyz`) — the same SLURM box the AGI pilot uses.
Recon: partitions `normal`/`debug`/`a`/`b`/`c`, gpu:2-6 per node; `agi` conda env
has torch 2.6+cu126; login node has internet. **Remote commands must be piped to
`bash -s` via heredoc** — the tcsh login shell chokes on `$(...)` in argv.

Setup done: pip-installed `transformers 5.17` + `accelerate` into the `agi` env;
downloaded `Qwen/Qwen2.5-3B-Instruct` (6.0 G) into `~/.cache/huggingface`.
**Disk note:** home quota is 150 G soft / 160 G hard, was 136 G used -> the model
took us to ~142 G. That is why I chose 3B over 7B (7B would breach the soft
quota); will delete the cache after. `/tmp` is big but node-local (not shared to
GPU nodes), so home is the only shared cache.

**Color fix shipped** (`ibrg_llm/stimuli.py`): object canonicalized to color 7,
nuisance colors drawn disjoint from it -> 0/200 collisions, no camouflage.

**Step 0 launched** (`ibrg_llm/step0.py`): F3_reflect, k=3,
n_probe=300, n_behav=40; probes `axis` (C, binary) and `frame_color` (U) per
layer vs shuffled-label floors, plus a behavioral generation check with
copy-input / all-zero baselines. Model subject = Qwen2.5-3B-Instruct (smoke-test
size; the real study can scale up if disk is freed). Awaiting the go/no-go.

## 2026-09-11 — GPU contention + robustness (deadline day)

**All ~56 cluster GPUs are allocated** (checked GresUsed==Gres on every node;
a3 down). So Step 0 is blocked purely on a GPU freeing — ETA unknown. `normal`
is *low priority* ("nodes reserved for higher priority partitions"), so I submit
across `a,b,c,normal` to grab the first freed GPU at best priority.

**Converted srun -> `sbatch` (job 52054)** so the run is owned by the SLURM
controller and completes whenever a GPU frees **independent of the user's laptop
/ my session** — results persist to disk (`ibrg_llm/out/*.json,*.png`, log
`logs/step0_sbatch.out`). The monitoring poller and any follow-on steps run from
the laptop session, so they pause when the laptop sleeps and resume when it's
back; the experiment itself does not depend on them. `step0.py` now saves the
per-layer selectivity curve (PNG) + numbers so one GPU-grab yields the
deliverable without re-queuing. matplotlib installed into the `agi` env.

## 2026-09-13 — P1 done: foliation & criticality (`crit.py`), with an honest null

(Note: the 2026-09-12 Track-B work — `vib.py`, `boundary.py` — is logged in commits and in
HANDOFF §2, not here; this entry picks up at P1.)

Built `crit.py` (the P1 capstone the handoff only *planned*) and ran it **CPU-local** — the
laptop has torch 2.11 + sklearn + matplotlib, and P1 is seconds/seed, so no cluster was needed.
5 seeds, 2 conditions (`aligned`, `recomb`/do(U)) × 8 βs.

**Design decisions, each forced by a probe I ran before writing the production script (scratch
probes, not committed — re-derive if extending):**
- A clean low-rate shortcut is **never** dropped by compression — OOD stays at chance for all β
  (reproduces the boundary result; this IS the thesis "IB can't separate a rule from a cheap
  spurious, it even prefers it"). So there is **no robust interior β\*** inverted-U from pure
  compression in this VIB. I did **not** manufacture one; I report the null.
- The foliation's obs-AUC≈0.5 needs the shortcut to be **as** Y-predictive as the rule on the
  training distribution → the **aligned** condition (shortcut==y on all train). Under recomb the
  shortcut is only weakly predictive, so the rule is genuinely more observationally readable
  (obs-AUC≈0.75). Hence crit.py runs BOTH conditions, mirroring boundary.py.
- First E1 metric attempt used the **literal do(z+=εv) response** and FAILED (obs-AUC 0.875,
  int-AUC 0.50). Root cause: for a *fixed* model the do() response is ~environment-invariant for
  BOTH v_C and v_S (so it can't discriminate), and my ε was large enough to saturate the softmax.
  Fix: the E1 discriminator is **cross-environment invariance of the label relation** — min over
  environments of the a-vs-b decodability of ⟨z,v⟩ (the IRM criterion). Kept do(z+=εv) only for
  the descriptive susceptibility χ (E3).

**Results (`out/crit_seed*.json`, 5 seeds; `fig_foliation`, `fig_fdt`):**
- **E1 foliation — SUPPORTED.** Aligned β=0: rule and shortcut directions read off the training
  label EQUALLY (both 1.00 → observational separation-AUC **0.48**). Cross-environment invariance
  separates them: rule predicts the label in every environment (0.83), shortcut only where it
  lines up (0.73); inv-AUC up to **0.86–0.91** (recomb, low β).
- **E2 criticality — NULL.** The v_C–v_S separation is largest at the *weakest* compression
  (β≈0.001, gap 0.195) and shrinks with β; it does NOT peak at the generalization-optimal β
  (recomb OOD optimum ≈0.1).
- **E3 FDT — descriptive.** χ(β) and Var(z) both fall monotonically with β (χ 3.99→0.001,
  Var 253→0.02); no shared peak.

Added `fig_foliation` + `fig_fdt` to `aggregate.py`. Committed crit.py + results + figures and
**fast-forwarded `main`** (which previously lacked `ibrg_llm/`) so the code has a clean public
URL: github.com/nurkyzaz/AGI/tree/main/ibrg_llm. Also pushed the same to the research branch.

**Non-research (same session):** wrote the MATS application from the committed results —
`APPLICATION_EXEC_SUMMARY.md` (research write-up: question → 8 numbered experiments → conclusion →
limitations, each figure explained; public-safe, committed) and, kept **local/uncommitted** because
`main` is public, `APPLICATION_ANSWERS.md` + `APPLICATION_DOC.docx` (personal application answers +
the figure-embedded Word doc). The .docx is built by a scratchpad script from the two .md files.

**NEXT:** P1 is done → the forward plan is now **P2** (harden E1: within-model cross-seed,
partial correlation controlling for aligned-accuracy, bidirectional ablation mediation) then P3.
See HANDOFF §3.
