"""Few-shot task sampling + the do(U)+recombination OOD split.

An ARC rule is *latent*: the transformation C is not visible in a single grid,
so a task is few-shot — k demonstration pairs reveal the rule, and the model
applies it to a query input. This is what makes C necessary and lets the oracle
(handed C directly) be a genuine upper bound.

The OOD split is created by a *spurious barcode*: a 1-2 cell code painted into
the top border row of every INPUT grid.

    train / iid  : barcode = the true joint causal index  (a perfect shortcut)
    shifted      : barcode = an independent random index   (do(U): recombined)

A raw model can read the barcode and skip inferring the rule from demos — it
then collapses when the correlation is broken at shifted-test. The object-centric
baseline crops the border away (barcode gone); the oracle never sees grids. The
barcode lives at the dataset level, NOT inside the pure families, so GATE 0 —
which asserts the families' declared U are output-irrelevant — remains valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..families import Family
from ..grid import Grid
from ..ledger import Role

BARCODE_ROW = 0  # outermost border row; crop-to-content discards it


# ---- canonical C / U encodings ---------------------------------------------

def causal_spec(family: Family) -> List[Tuple[str, Tuple]]:
    """Ordered [(name, domain)] for the family's causal variables."""
    return [(v.name, v.domain) for v in family.ledger().variables if v.role is Role.C]


def nuisance_spec(family: Family) -> List[Tuple[str, Tuple]]:
    return [(v.name, v.domain) for v in family.ledger().variables if v.role is Role.U]


def causal_code(family: Family, latents: Dict) -> List[int]:
    """Per-variable index vector for the C latents (probe / oracle target)."""
    out = []
    for name, domain in causal_spec(family):
        val = latents[name]
        idx = next(i for i, d in enumerate(domain) if _eq(d, val))
        out.append(idx)
    return out


def nuisance_code(family: Family, latents: Dict) -> List[int]:
    out = []
    for name, domain in nuisance_spec(family):
        val = latents[name]
        idx = next(i for i, d in enumerate(domain) if _eq(d, val))
        out.append(idx)
    return out


def causal_cardinalities(family: Family) -> List[int]:
    return [len(domain) for _, domain in causal_spec(family)]


def joint_causal_index(family: Family, latents: Dict) -> int:
    """Mixed-radix flatten of the causal code into a single integer."""
    idx = 0
    for c, card in zip(causal_code(family, latents), causal_cardinalities(family)):
        idx = idx * card + c
    return idx


def joint_causal_size(family: Family) -> int:
    n = 1
    for card in causal_cardinalities(family):
        n *= card
    return n


def _eq(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


# ---- barcode overlay --------------------------------------------------------

def _barcode_cells(J: int) -> int:
    """Number of border cells needed to encode indices in [0, J) base-9."""
    n, cap = 1, 9
    while cap < J:
        cap *= 9
        n += 1
    return n


def _paint_barcode(inp: Grid, value: int, n_cells: int) -> None:
    """Encode `value` base-9 into border row cells [1 .. n_cells] as colors 1-9."""
    v = value
    for i in range(n_cells):
        digit = v % 9
        v //= 9
        inp[BARCODE_ROW, 1 + i] = digit + 1  # colors 1..9


# ---- task sampling ----------------------------------------------------------

@dataclass
class Task:
    demo_inputs: np.ndarray   # (k, H, W) int8
    demo_outputs: np.ndarray  # (k, H, W) int8
    query_input: np.ndarray   # (H, W) int8
    query_output: np.ndarray  # (H, W) int8
    c_code: List[int]         # causal latent indices
    u_code: List[int]         # nuisance latent indices
    barcode: int              # the value painted (shortcut signal)


def sample_task(
    family: Family, rng: np.random.Generator, k: int, split: str
) -> Task:
    """Sample one few-shot task. split in {train, iid, shifted}."""
    latents = family.sample_latents(rng)
    J = joint_causal_size(family)
    n_cells = _barcode_cells(J)
    true_idx = joint_causal_index(family, latents)
    if split == "shifted":
        barcode = int(rng.integers(0, 9 ** n_cells))  # decoupled from C (recombination)
    else:
        barcode = true_idx  # perfect shortcut at train / iid

    content_seeds = [int(rng.integers(1, 2 ** 31)) for _ in range(k + 1)]
    demo_in, demo_out = [], []
    for cs in content_seeds[:k]:
        gi, go = family.render(latents, cs)
        _paint_barcode(gi, barcode, n_cells)
        demo_in.append(gi)
        demo_out.append(go)
    qi, qo = family.render(latents, content_seeds[k])
    _paint_barcode(qi, barcode, n_cells)

    return Task(
        demo_inputs=np.stack(demo_in),
        demo_outputs=np.stack(demo_out),
        query_input=qi,
        query_output=qo,
        c_code=causal_code(family, latents),
        u_code=nuisance_code(family, latents),
        barcode=barcode,
    )


def sample_batch(
    family: Family, rng: np.random.Generator, n: int, k: int, split: str
) -> List[Task]:
    return [sample_task(family, rng, k, split) for _ in range(n)]


def strip_border(grid: np.ndarray, depth: int = 1) -> np.ndarray:
    """Object-centric canonicalization: zero out the border ring (removes the
    frame nuisance AND the spurious barcode), keeping the interior signal."""
    g = grid.copy()
    g[:depth, :] = 0
    g[-depth:, :] = 0
    g[:, :depth] = 0
    g[:, -depth:] = 0
    return g
