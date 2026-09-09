"""Serialize `causalarc` planted-family tasks into few-shot text prompts for an
LLM, carrying the ground-truth causal (C) and nuisance (U) labels needed for the
layerwise probing study.

Design (IB=RG selectivity):
  * The RULE (a C latent, e.g. F3's reflection `axis`) is *latent*: it is only
    inferable from the demo input->output pairs, so decoding it from activations
    tests whether the model has *built up* the task-relevant variable.
  * A NUISANCE latent (e.g. `frame_color`) is *surface*: it is printed in every
    grid, so it is trivially decodable early; the IB=RG question is whether the
    model *discards* it with depth while amplifying the rule.

We deliberately expose two knobs the study needs:
  * `barcode`: optional spurious shortcut painted into the top border row that
    equals the joint causal index (train) or is resampled (shifted / do(U)).
    Off by default for the clean selectivity arm; on for the shortcut arm.
  * `promote_frame_to_causal`: control condition — makes the OUTPUT depend on the
    frame color, so `frame_color` becomes causal. If the depth profile of
    `frame_color` flips (discarded -> preserved) when it becomes causal, that is
    mechanism, not correlation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from causalarc.families import FAMILIES
from causalarc.grid import Grid
from causalarc.ledger import Role


def grid_to_text(g: Grid) -> str:
    """One grid -> rows of space-separated digits."""
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


@dataclass
class Stimulus:
    prompt: str                 # the full few-shot text prompt (ends before the answer)
    target_text: str            # the correct query-output grid, serialized
    c_labels: Dict[str, Any]    # ground-truth causal latent values (the rule)
    u_labels: Dict[str, Any]    # ground-truth nuisance latent values (surface)
    barcode: Optional[int]      # spurious shortcut value if painted, else None
    family: str


_HEADER = (
    "You are given input/output grid pairs that all follow one hidden rule.\n"
    "Grids are rows of digits 0-9 (0 = background). Infer the rule from the\n"
    "examples, then produce the output grid for the final input.\n"
)


def _paint_barcode_row0(g: Grid, value: int, n_cells: int) -> None:
    """Encode `value` base-9 into the top border row (colors 1-9), mirroring
    causalarc.pilot.datasets so the shortcut arm matches the pilot design."""
    v = value
    for i in range(n_cells):
        g[0, 1 + i] = (v % 9) + 1
        v //= 9


def make_stimulus(
    family_name: str,
    rng: np.random.Generator,
    k: int = 3,
    barcode: bool = False,
    shifted: bool = False,
) -> Stimulus:
    """Build one few-shot task. The rule (C latents) is fixed across the k demos
    and the query; content (the object drawn) varies per example."""
    fam = FAMILIES[family_name]
    ledger = fam.ledger()
    c_names = [v.name for v in ledger.variables if v.role is Role.C]
    u_names = [v.name for v in ledger.variables if v.role is Role.U]

    latents = fam.sample_latents(rng)
    seeds = [int(rng.integers(1, 2**31)) for _ in range(k + 1)]

    # optional spurious barcode = joint causal index (train) or random (shifted)
    bc_val = None
    if barcode:
        # small fixed width is enough for these low-cardinality families
        n_cells = 2
        if shifted:
            bc_val = int(rng.integers(0, 9**n_cells))
        else:
            # deterministic function of the C latents (a perfect train shortcut)
            bc_val = abs(hash(tuple(sorted((k, str(latents[k])) for k in c_names)))) % (9**n_cells)

    def render(seed: int) -> Tuple[Grid, Grid]:
        gi, go = fam.render(latents, seed)
        if barcode:
            _paint_barcode_row0(gi, bc_val, 2)
        return gi, go

    parts: List[str] = [_HEADER]
    for i in range(k):
        gi, go = render(seeds[i])
        parts.append(f"\nExample {i+1}\nInput:\n{grid_to_text(gi)}\nOutput:\n{grid_to_text(go)}")
    qi, qo = render(seeds[k])
    parts.append(f"\nNow the final input.\nInput:\n{grid_to_text(qi)}\nOutput:\n")

    return Stimulus(
        prompt="".join(parts),
        target_text=grid_to_text(qo),
        c_labels={n: latents[n] for n in c_names},
        u_labels={n: latents[n] for n in u_names},
        barcode=bc_val,
        family=family_name,
    )


if __name__ == "__main__":
    # eyeball a couple of serialized prompts (Neel rule 1: read your data)
    rng = np.random.default_rng(0)
    for fam in ["F3_reflect", "F2_recolor_parity"]:
        s = make_stimulus(fam, rng, k=2)
        print("=" * 70)
        print(f"FAMILY {fam}   C={s.c_labels}   U={s.u_labels}")
        print("=" * 70)
        print(s.prompt + s.target_text)
        print()
