"""GATE 0 tests, including adversarial negative controls.

The positive tests confirm the real harness passes. The negative-control tests
are the ones that make a PASS meaningful: we deliberately plant bugs (a declared
nuisance that actually moves the output; a declared-causal variable that is
actually inert) and assert the harness *catches them*. A harness that cannot
fail on planted bugs cannot be trusted to pass on real data.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from causalarc.families import FAMILIES, RecolorByParity, TranslateObject
from causalarc.gate0 import run_gate0
from causalarc.interventions import (intervention_tests, recover_causal_mask,
                                     variable_effect_rate)
from causalarc.ledger import CausalLedger, Role, Variable
from causalarc.mdl import lz78_bits, mdl_bits_of_grid
from causalarc.grid import new_grid


# ---- positive: the real harness passes -------------------------------------

def test_gate0_overall_passes():
    r = run_gate0(n_content=30, n_base=3, n_split=80, seed=0, write=False)
    assert r["overall_pass"], r


def test_all_families_intervention_tests_pass():
    for fam in FAMILIES.values():
        ok, detail = intervention_tests(fam, n_content=30, n_base=2, seed=1)
        assert ok, (fam.name, detail)


def test_all_families_mask_recovery_perfect():
    for fam in FAMILIES.values():
        res = recover_causal_mask(fam, n_content=30, n_base=2, seed=2)
        assert res.perfect, (fam.name, res)


def test_nuisance_variables_are_exactly_inert():
    """U variables must have effect rate == 0.0 exactly, not merely small."""
    for fam in FAMILIES.values():
        for v in fam.ledger().variables:
            if v.role is Role.U:
                rate = variable_effect_rate(fam, v.name, n_content=30, n_base=2, seed=3)
                assert rate == 0.0, (fam.name, v.name, rate)


# ---- negative controls: the harness must FAIL on planted bugs --------------

class LeakyNuisance(TranslateObject):
    """A broken F1 whose 'frame_color' (declared U) secretly shifts the output —
    i.e. a nuisance that is actually causal. The harness must catch this."""

    name = "BROKEN_leaky_nuisance"

    def render(self, latents, content_seed):
        inp, out = super().render(latents, content_seed)
        # BUG: leak the nuisance into the output (tie output to frame_color).
        from causalarc.grid import translate
        out = translate(out, int(latents["frame_color"]) % 3, 0)
        return inp, out


class DeadCausal(RecolorByParity):
    """A broken F2 whose declared-C 'color_odd' never affects the output
    (rule always uses color_even). The harness must catch the missed cause."""

    name = "BROKEN_dead_causal"

    def render(self, latents, content_seed):
        from causalarc.grid import new_grid as _ng  # noqa
        signal = self._draw = None
        # reuse parent draw but force target = color_even always
        from causalarc.families import _draw_signal, _compose_input
        sig = _draw_signal(content_seed, self._marker)
        out = sig.copy()
        out[sig == self._marker] = int(latents["color_even"])  # BUG: ignores parity/color_odd
        inp = _compose_input(sig, latents, content_seed)
        return inp, out


def test_negative_control_leaky_nuisance_is_caught():
    fam = LeakyNuisance()
    ok, detail = intervention_tests(fam, n_content=30, n_base=2, seed=4)
    assert not ok, detail  # a leaked nuisance must break item-1
    res = recover_causal_mask(fam, n_content=30, n_base=2, seed=4)
    assert not res.perfect  # frame_color would be (correctly) flagged causal here


def test_negative_control_dead_causal_is_caught():
    fam = DeadCausal()
    ok, detail = intervention_tests(fam, n_content=40, n_base=3, seed=5)
    assert not ok, detail  # color_odd is declared C but inert -> item-1 fails
    res = recover_causal_mask(fam, n_content=40, n_base=3, seed=5)
    assert res.recall < 1.0  # a real cause was missed


# ---- MDL proxy sanity -------------------------------------------------------

def test_lz78_deterministic_and_nonnegative():
    data = b"the quick brown fox" * 5
    assert lz78_bits(data) == lz78_bits(data)
    assert lz78_bits(data) >= 0.0
    assert lz78_bits(b"") == 0.0


def test_lz78_structure_compresses_below_random():
    rng = np.random.default_rng(0)
    structured = new_grid(12, 12, 0)
    structured[3:6, 3:6] = 4  # a simple block
    noise = rng.integers(0, 10, size=(12, 12)).astype(np.int8)
    assert mdl_bits_of_grid(structured) < mdl_bits_of_grid(noise)


def test_families_render_is_deterministic():
    for fam in FAMILIES.values():
        rng = np.random.default_rng(7)
        latents = fam.sample_latents(rng)
        a_in, a_out = fam.render(latents, 123)
        b_in, b_out = fam.render(latents, 123)
        assert np.array_equal(a_in, b_in) and np.array_equal(a_out, b_out)
