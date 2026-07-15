"""Shortcut splits and probes (GATE 0 item 2).

We build a dataset where a nuisance *marker* is perfectly correlated with the
label at train and iid-test, but the correlation is REVERSED at shifted-test
(this is do(U) on the shortcut). Two probe predictors then demonstrate the
harness's shift machinery works:

  * `shortcut_predict` decides the class from the marker (the nuisance) — high
    on train/iid, must collapse on shifted.
  * `causal_predict` decides the class from the object geometry (the true cause)
    — high on every split.

If a shortcut we planted ourselves does NOT collapse under reversal, the harness
cannot be trusted to detect shortcuts on real data, and GATE 0 fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .families import ShortcutRecolor
from .grid import BACKGROUND, Grid, grids_equal, new_grid

# Fixed latents for the shortcut experiment: distinct class colors, a non-5
# frame (so the object, drawn in color 5, is unambiguous), marker enabled.
FIXED_LATENTS = {
    "color_c0": 2,
    "color_c1": 4,
    "marker_on": True,
    "frame_color": 1,
}
_MARKER_PALETTE = ShortcutRecolor._marker_palette  # (class0_color, class1_color)


@dataclass
class Instance:
    input: Grid
    output: Grid
    y: int


def make_shortcut_splits(
    n_per_split: int = 120, seed: int = 0
) -> Dict[str, List[Instance]]:
    """train/iid share hint=y (marker tells the truth); shifted uses hint=1-y."""
    fam = ShortcutRecolor()

    def build(n: int, reverse: bool, base: int) -> List[Instance]:
        out = []
        for i in range(n):
            cs = base + i
            inp, tgt, y = fam.render_labeled(FIXED_LATENTS, cs, hint=None)  # honest marker = y
            if reverse:
                inp, tgt, y = fam.render_labeled(FIXED_LATENTS, cs, hint=1 - y)
            out.append(Instance(inp, tgt, y))
        return out

    return {
        "train": build(n_per_split, reverse=False, base=1_000_000),
        "iid": build(n_per_split, reverse=False, base=2_000_000),
        "shifted": build(n_per_split, reverse=True, base=3_000_000),
    }


def _object_mask(inp: Grid) -> np.ndarray:
    """The signal object is the set of cells equal to 5 (marker color used to
    draw the shape); frame and shortcut-marker use other colors."""
    return inp == 5


def _recolor_object_to(inp: Grid, color: int) -> Grid:
    out = new_grid(*inp.shape, fill=BACKGROUND)
    out[_object_mask(inp)] = color
    return out


def shortcut_predict(inp: Grid, latents: Dict = FIXED_LATENTS) -> Grid:
    """Decide class from the corner marker (the nuisance)."""
    marker = int(inp[1, 1])
    cls = 1 if marker == _MARKER_PALETTE[1] else 0
    color = latents["color_c1"] if cls == 1 else latents["color_c0"]
    return _recolor_object_to(inp, color)


def causal_predict(inp: Grid, latents: Dict = FIXED_LATENTS) -> Grid:
    """Decide class from object geometry (taller-than-wide == class 1)."""
    mask = _object_mask(inp)
    if not mask.any():
        cls = 0
    else:
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        h = rows[-1] - rows[0] + 1
        w = cols[-1] - cols[0] + 1
        cls = 1 if h > w else 0
    color = latents["color_c1"] if cls == 1 else latents["color_c0"]
    return _recolor_object_to(inp, color)


def accuracy(predict: Callable[[Grid], Grid], instances: List[Instance]) -> float:
    if not instances:
        return 0.0
    hits = sum(1 for ins in instances if grids_equal(predict(ins.input), ins.output))
    return hits / len(instances)
