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

**Step 0 launched** (`ibrg_llm/step0.py`, job on `debug`): F3_reflect, k=3,
n_probe=300, n_behav=40; probes `axis` (C, binary) and `frame_color` (U) per
layer vs shuffled-label floors, plus a behavioral generation check with
copy-input / all-zero baselines. Model subject = Qwen2.5-3B-Instruct (smoke-test
size; the real study can scale up if disk is freed). Currently PENDING on GPU
resources (cluster busy). Awaiting the go/no-go.
