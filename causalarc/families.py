"""The 5 planted generator families (P0.2).

Every family separates a *signal canvas* (governed by content_seed and the C
latents) from *nuisance decoration* (governed by the U latents). The output is
computed from the signal canvas alone:

    signal = draw_signal(content_seed)
    output = rule(signal, C-latents)
    input  = compose(signal, U-latents)     # frame + distractor, absent in output

Because the rule never reads the nuisance decoration, U latents are provably
output-irrelevant *by construction* — holding content_seed fixed, intervening
on any U latent leaves `output` byte-identical, while intervening on a C latent
changes it. GATE 0 checks that this designed-in property is what the metric code
actually recovers.

Families:
    F1 TranslateObject   geometric  — C: (dy, dx)
    F2 RecolorByParity   recolor    — C: (color_even, color_odd)
    F3 ReflectByAxis     geometric  — C: axis
    F4 GravityDrop       geometric  — C: direction
    F5 ShortcutRecolor   recolor+shortcut — C: (color_c0, color_c1); planted
                          nuisance marker used for the GATE-0 shortcut test.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .grid import BACKGROUND, Grid, new_grid, reflect, translate
from .ledger import CausalLedger, Family, Role, Variable

# Colors usable for objects / decoration (exclude 0 = background).
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
CANVAS = 12  # H = W; large enough that translations don't clip the signal.
MARGIN = 3  # keep the signal object away from the border (frame lives there).


# --------------------------------------------------------------------------
# shared substrate helpers
# --------------------------------------------------------------------------

def _rng(content_seed: int) -> np.random.Generator:
    return np.random.default_rng(content_seed)


def _random_polyomino(rng: np.random.Generator, n_cells: int) -> Set[Tuple[int, int]]:
    """Random connected set of `n_cells` cells via random growth from origin."""
    cells: Set[Tuple[int, int]] = {(0, 0)}
    frontier = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while len(cells) < n_cells:
        rng.shuffle(frontier)
        grew = False
        for _ in range(len(frontier)):
            cand = frontier.pop(0)
            frontier.append(cand)
            if cand in cells:
                continue
            cells.add(cand)
            r, c = cand
            frontier.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
            grew = True
            break
        if not grew:  # pragma: no cover - frontier always has room here
            break
    # normalize so min row/col is 0
    rs = min(r for r, _ in cells)
    cs = min(c for _, c in cells)
    return {(r - rs, c - cs) for r, c in cells}


def _blit(canvas: Grid, cells: Set[Tuple[int, int]], top: int, left: int, color: int) -> None:
    for r, c in cells:
        rr, cc = top + r, left + c
        if 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1]:
            canvas[rr, cc] = color


def _draw_signal(content_seed: int, color: int, n_cells_range=(3, 7)) -> Grid:
    """One object of `color` placed with margin on a CANVAS x CANVAS grid."""
    rng = _rng(content_seed)
    n = int(rng.integers(n_cells_range[0], n_cells_range[1] + 1))
    cells = _random_polyomino(rng, n)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    top = int(rng.integers(MARGIN, CANVAS - MARGIN - h + 1))
    left = int(rng.integers(MARGIN, CANVAS - MARGIN - w + 1))
    canvas = new_grid(CANVAS, CANVAS)
    _blit(canvas, cells, top, left, color)
    return canvas


def _add_frame(inp: Grid, frame_color: int) -> None:
    """Draw a 1-cell decorative border (nuisance; never in output)."""
    inp[0, :] = frame_color
    inp[-1, :] = frame_color
    inp[:, 0] = frame_color
    inp[:, -1] = frame_color


def _add_distractor(inp: Grid, distractor_color: int, content_seed: int) -> None:
    """Place a tiny nuisance blob just inside a corner (never in output)."""
    rng = _rng(content_seed * 7919 + 13)
    corner = int(rng.integers(0, 4))
    r = 1 if corner in (0, 1) else CANVAS - 2
    c = 1 if corner in (0, 2) else CANVAS - 2
    inp[r, c] = distractor_color


# The nuisance kit shared by F1-F4: two U variables, uniformly defined.
_NUISANCE_VARS = (
    Variable("frame_color", Role.U, PALETTE,
             "decorative border; rule ignores it, absent from output"),
    Variable("distractor_color", Role.U, PALETTE,
             "corner nuisance blob; rule ignores it, absent from output"),
)


def _compose_input(signal: Grid, latents: Dict[str, Any], content_seed: int) -> Grid:
    inp = signal.copy()
    _add_frame(inp, int(latents["frame_color"]))
    _add_distractor(inp, int(latents["distractor_color"]), content_seed)
    return inp


def _sample_nuisance(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "frame_color": int(rng.choice(PALETTE)),
        "distractor_color": int(rng.choice(PALETTE)),
    }


# --------------------------------------------------------------------------
# F1 — TranslateObject
# --------------------------------------------------------------------------

class TranslateObject(Family):
    name = "F1_translate"
    _shifts = (-2, -1, 1, 2)

    def ledger(self) -> CausalLedger:
        return CausalLedger(
            family=self.name,
            variables=(
                Variable("dy", Role.C, self._shifts, "vertical shift of the object"),
                Variable("dx", Role.C, self._shifts, "horizontal shift of the object"),
                *_NUISANCE_VARS,
            ),
            reference_rule="output = translate(signal_object, dy, dx)",
            rule_type="geometric",
        )

    def sample_latents(self, rng):
        return {
            "dy": int(rng.choice(self._shifts)),
            "dx": int(rng.choice(self._shifts)),
            **_sample_nuisance(rng),
        }

    def render(self, latents, content_seed):
        color = 1 + (content_seed % 8)  # deterministic object color from content
        signal = _draw_signal(content_seed, color)
        output = translate(signal, int(latents["dy"]), int(latents["dx"]))
        inp = _compose_input(signal, latents, content_seed)
        return inp, output


# --------------------------------------------------------------------------
# F2 — RecolorByParity
# --------------------------------------------------------------------------

class RecolorByParity(Family):
    name = "F2_recolor_parity"
    _colors = (1, 2, 3, 4, 6, 7, 8, 9)  # exclude 5 (marker) and 0 (bg)
    _marker = 5

    def ledger(self) -> CausalLedger:
        return CausalLedger(
            family=self.name,
            variables=(
                Variable("color_even", Role.C, self._colors,
                         "output color when object cell-count is even"),
                Variable("color_odd", Role.C, self._colors,
                         "output color when object cell-count is odd"),
                *_NUISANCE_VARS,
            ),
            reference_rule="output = recolor(signal, color_even if |cells| even else color_odd)",
            rule_type="recolor",
        )

    def sample_latents(self, rng):
        ce, co = rng.choice(self._colors, size=2, replace=False)
        return {"color_even": int(ce), "color_odd": int(co), **_sample_nuisance(rng)}

    def render(self, latents, content_seed):
        signal = _draw_signal(content_seed, self._marker)
        n_cells = int((signal == self._marker).sum())
        target = int(latents["color_even"]) if n_cells % 2 == 0 else int(latents["color_odd"])
        output = signal.copy()
        output[signal == self._marker] = target
        inp = _compose_input(signal, latents, content_seed)
        return inp, output


# --------------------------------------------------------------------------
# F3 — ReflectByAxis
# --------------------------------------------------------------------------

class ReflectByAxis(Family):
    name = "F3_reflect"
    _axes = ("horizontal", "vertical")

    def ledger(self) -> CausalLedger:
        return CausalLedger(
            family=self.name,
            variables=(
                Variable("axis", Role.C, self._axes, "reflection axis"),
                *_NUISANCE_VARS,
            ),
            reference_rule="output = reflect(signal, axis)",
            rule_type="geometric",
        )

    def sample_latents(self, rng):
        return {"axis": str(rng.choice(self._axes)), **_sample_nuisance(rng)}

    def _asymmetric_signal(self, content_seed: int) -> Grid:
        """Draw a signal that is not symmetric under either reflection, so the
        axis intervention is guaranteed to change the output."""
        color = 1 + (content_seed % 8)
        s = content_seed
        for _ in range(64):
            signal = _draw_signal(s, color)
            if not np.array_equal(signal, reflect(signal, "horizontal")) and not np.array_equal(
                signal, reflect(signal, "vertical")
            ):
                return signal
            s = s * 2654435761 % (2**31)
        return signal  # pragma: no cover

    def render(self, latents, content_seed):
        signal = self._asymmetric_signal(content_seed)
        output = reflect(signal, str(latents["axis"]))
        inp = _compose_input(signal, latents, content_seed)
        return inp, output


# --------------------------------------------------------------------------
# F4 — GravityDrop
# --------------------------------------------------------------------------

class GravityDrop(Family):
    name = "F4_gravity"
    _dirs = ("down", "up", "left", "right")

    def ledger(self) -> CausalLedger:
        return CausalLedger(
            family=self.name,
            variables=(
                Variable("direction", Role.C, self._dirs, "gravity direction"),
                *_NUISANCE_VARS,
            ),
            reference_rule="output = each non-bg cell moved maximally toward `direction`",
            rule_type="geometric",
        )

    def sample_latents(self, rng):
        return {"direction": str(rng.choice(self._dirs)), **_sample_nuisance(rng)}

    def _scatter_signal(self, content_seed: int) -> Grid:
        rng = _rng(content_seed)
        canvas = new_grid(CANVAS, CANVAS)
        n = int(rng.integers(4, 8))
        for _ in range(n):
            r = int(rng.integers(MARGIN, CANVAS - MARGIN))
            c = int(rng.integers(MARGIN, CANVAS - MARGIN))
            canvas[r, c] = int(rng.choice(PALETTE))
        return canvas

    @staticmethod
    def _drop(signal: Grid, direction: str) -> Grid:
        out = new_grid(*signal.shape)
        h, w = signal.shape
        if direction in ("down", "up"):
            for c in range(w):
                col = [signal[r, c] for r in range(h) if signal[r, c] != BACKGROUND]
                if direction == "down":
                    for i, val in enumerate(reversed(col)):
                        out[h - 1 - i, c] = val
                else:
                    for i, val in enumerate(col):
                        out[i, c] = val
        else:
            for r in range(h):
                row = [signal[r, c] for c in range(w) if signal[r, c] != BACKGROUND]
                if direction == "right":
                    for i, val in enumerate(reversed(row)):
                        out[r, w - 1 - i] = val
                else:
                    for i, val in enumerate(row):
                        out[r, i] = val
        return out

    def render(self, latents, content_seed):
        signal = self._scatter_signal(content_seed)
        output = self._drop(signal, str(latents["direction"]))
        inp = _compose_input(signal, latents, content_seed)
        return inp, output


# --------------------------------------------------------------------------
# F5 — ShortcutRecolor (drives the GATE-0 shortcut test)
# --------------------------------------------------------------------------

class ShortcutRecolor(Family):
    """A 2-class recolor task whose true label y is set by object geometry
    (tall vs wide), plus a planted *marker* nuisance whose color can be made to
    correlate with y. splits.py sets the marker = y (train / iid) or = 1 - y
    (shifted) to build the shortcut. The honest `render` (marker = y) is used by
    the causal-mask metric; the labeled variant exposes y and the marker knob.
    """

    name = "F5_shortcut"
    _colors = (1, 2, 3, 4, 6, 7, 8, 9)
    _marker_palette = (3, 8)  # two visually distinct marker colors, one per class

    def ledger(self) -> CausalLedger:
        return CausalLedger(
            family=self.name,
            variables=(
                Variable("color_c0", Role.C, self._colors, "output color for class 0 (wide)"),
                Variable("color_c1", Role.C, self._colors, "output color for class 1 (tall)"),
                Variable("marker_on", Role.U, (True, False),
                         "whether the shortcut marker is drawn (nuisance; not read by rule)"),
                Variable("frame_color", Role.U, PALETTE, "decorative border"),
            ),
            reference_rule="y = 1 if object taller than wide else 0; output = recolor(object, color_c[y])",
            rule_type="recolor",
        )

    def sample_latents(self, rng):
        c0, c1 = rng.choice(self._colors, size=2, replace=False)
        return {
            "color_c0": int(c0),
            "color_c1": int(c1),
            "marker_on": bool(rng.integers(0, 2)),
            "frame_color": int(rng.choice(PALETTE)),
        }

    # --- geometry that determines the true (causal) label ----------------
    def _labeled_signal(self, content_seed: int) -> Tuple[Grid, int]:
        """Return (signal, y) where y=1 iff the object is taller than wide."""
        rng = _rng(content_seed)
        marker = 5
        for _ in range(64):
            n = int(rng.integers(4, 8))
            cells = _random_polyomino(rng, n)
            h = max(r for r, _ in cells) + 1
            w = max(c for _, c in cells) + 1
            if h == w:
                continue
            y = 1 if h > w else 0
            top = int(rng.integers(MARGIN, CANVAS - MARGIN - h + 1))
            left = int(rng.integers(MARGIN, CANVAS - MARGIN - w + 1))
            canvas = new_grid(CANVAS, CANVAS)
            _blit(canvas, cells, top, left, marker)
            return canvas, y
        raise RuntimeError("could not sample an unequal-aspect object")  # pragma: no cover

    def render_labeled(
        self, latents: Dict[str, Any], content_seed: int, hint: Optional[int] = None
    ) -> Tuple[Grid, Grid, int]:
        """Full render exposing y and the shortcut `hint`.

        hint = the class the marker color advertises. Set hint=y for an honest /
        train-correlated instance, hint=1-y to reverse the correlation. If the
        marker is disabled (marker_on False) no marker is drawn.
        """
        signal, y = self._labeled_signal(content_seed)
        target = int(latents["color_c1"]) if y == 1 else int(latents["color_c0"])
        output = signal.copy()
        output[signal == 5] = target

        inp = signal.copy()
        _add_frame(inp, int(latents["frame_color"]))
        if latents.get("marker_on", True):
            h = hint if hint is not None else y
            inp[1, 1] = self._marker_palette[h]  # corner marker encodes the hint
        return inp, output, y

    def render(self, latents, content_seed):
        inp, output, _ = self.render_labeled(latents, content_seed, hint=None)
        return inp, output


# Registry --------------------------------------------------------------------
FAMILIES: Dict[str, Family] = {
    f.name: f
    for f in (
        TranslateObject(),
        RecolorByParity(),
        ReflectByAxis(),
        GravityDrop(),
        ShortcutRecolor(),
    )
}
