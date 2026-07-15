# causalarc — Phase 0 harness

The trustworthy instrument for *"Causally Useful Compression on ARC"*
(see [`../COMPRESSION_INTELLIGENCE_RESEARCH_PLAN (1).md`](../COMPRESSION_INTELLIGENCE_RESEARCH_PLAN%20(1).md)).

**Phase 0 = specification & audit.** Build a harness you can trust *before* you
have any result to be excited about. This package implements the 5 planted
generator families, their intervention tests, the locked MDL proxy, and the
GATE 0 decision. It is CPU-only and dependency-light (stdlib + numpy). No
training happens here — the GPU cluster is a Phase 1 concern.

## GATE 0 (all three must hold to enter Phase 1)

| # | Check | Where | Status |
|---|-------|-------|--------|
| 1 | Every declared intervention passes its test (C can change the output; U never does) | `interventions.intervention_tests` | ✅ PASS |
| 2 | A planted shortcut is invisible i.i.d. but fails under correlation reversal | `splits` + `gate0.check_shortcut` | ✅ PASS |
| 3 | Metric recovers the known causal mask (precision/recall = 1) | `interventions.recover_causal_mask` | ✅ PASS |

Reproduce:

```bash
python -m causalarc.gate0            # runs all three checks, writes an immutable record
python -m pytest causalarc/tests -q  # incl. adversarial negative controls
```

The negative-control tests are the point: they plant bugs (a nuisance that
secretly moves the output; a declared cause that is inert) and assert the
harness *fails* them. A harness that can't fail on planted bugs can't be trusted
to pass on real data.

## The 5 planted families

Each family splits a **signal canvas** (governed by `content_seed` + the C
latents) from **nuisance decoration** (governed by the U latents). The output is
computed from the signal alone, so U latents are output-irrelevant *by
construction* — holding `content_seed` fixed, do(U) leaves the output
byte-identical while do(C) changes it.

| Family | rule_type | Causal (C) | Nuisance (U) |
|--------|-----------|------------|--------------|
| F1 TranslateObject | geometric | `dy`, `dx` | frame, distractor |
| F2 RecolorByParity | recolor | `color_even`, `color_odd` | frame, distractor |
| F3 ReflectByAxis | geometric | `axis` | frame, distractor |
| F4 GravityDrop | geometric | `direction` | frame, distractor |
| F5 ShortcutRecolor | recolor + shortcut | `color_c0`, `color_c1` | marker, frame |

F5 additionally drives the GATE 0 item-2 shortcut experiment: its corner marker
can be made to correlate with the class label (train/iid) or anti-correlate
(shifted), and two oracle probes (`shortcut_predict`, `causal_predict`)
demonstrate the shift plumbing works.

## Locked decisions (do not silently change)

- **MDL proxy = LZ-78 on fixed-precision row-major state bitstrings** (`mdl.py`).
  The single proxy used throughout; neural KL / program tokens / zipped weights
  are *not* interchangeable bit-units.
- **Independent unit = task family** (not generated grids). Enforced in later
  phases' statistics; noted here so it is never forgotten.
- Every GATE run appends one line to `../runs_causalarc/*.jsonl` (append-only,
  content-hashed, records config + ledger hashes + git rev).

## Not yet done (Phase 0 → Phase 1 boundary)

- **git**: `agi/` is not yet a git repo, so records log `git_rev: no-git`.
  Initialize version control before any confirmatory run so code revision is
  pinned into every record.
- **P1.4 pre-registration** of hypotheses/thresholds happens *after* the Phase 1
  pilot design is fixed, before seeing final results.
- Phase 1 (feasibility pilot: 4 baselines incl. an oracle over 10 seeds) is the
  first GPU workload — runs on the `gpus` cluster.
