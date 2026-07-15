"""Grid type and primitive operations.

A Grid is a numpy int8 array of shape (H, W) with cell values 0-9 (ARC palette;
0 = background/black by convention). We use numpy rather than list-of-lists so
that hashing, equality, and the LZ-78 bitstring serialization are unambiguous
and fast. Conversion helpers to/from the official JSON list format live here.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

Grid = np.ndarray  # dtype int8, shape (H, W)

MAX_COLOR = 9
BACKGROUND = 0


def new_grid(h: int, w: int, fill: int = BACKGROUND) -> Grid:
    return np.full((h, w), fill, dtype=np.int8)


def grids_equal(a: Grid, b: Grid) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def grid_key(g: Grid) -> Tuple:
    """A hashable, canonical key for a grid (shape + bytes)."""
    return (g.shape, g.astype(np.int8).tobytes())


def to_json(g: Grid) -> List[List[int]]:
    return g.astype(int).tolist()


def from_json(rows: List[List[int]]) -> Grid:
    return np.asarray(rows, dtype=np.int8)


def crop_to_content(g: Grid, background: int = BACKGROUND) -> Grid:
    """Bounding-box crop around all non-background cells. Empty -> 1x1 bg."""
    mask = g != background
    if not mask.any():
        return new_grid(1, 1, background)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def translate(g: Grid, dy: int, dx: int, background: int = BACKGROUND) -> Grid:
    """Shift content by (dy, dx); cells shifted off-grid are dropped, vacated
    cells filled with background. Same shape as input."""
    out = new_grid(g.shape[0], g.shape[1], background)
    h, w = g.shape
    for r in range(h):
        nr = r + dy
        if nr < 0 or nr >= h:
            continue
        for c in range(w):
            nc = c + dx
            if 0 <= nc < w:
                out[nr, nc] = g[r, c]
    return out


def reflect(g: Grid, axis: str) -> Grid:
    if axis == "horizontal":  # flip left-right (mirror across vertical axis)
        return g[:, ::-1].copy()
    if axis == "vertical":  # flip top-bottom
        return g[::-1, :].copy()
    raise ValueError(f"unknown axis {axis!r}")


def recolor(g: Grid, mapping: dict, background: int = BACKGROUND) -> Grid:
    """Map non-background colors via `mapping` (color->color); unlisted kept."""
    out = g.copy()
    for src, dst in mapping.items():
        out[g == src] = dst
    return out
