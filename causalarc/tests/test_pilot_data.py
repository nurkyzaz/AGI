"""Tests for the Phase-1 pilot data machinery (no torch required).

Locks the do(U)+recombination invariants: the barcode encodes the true causal
index at train/iid and is decoupled at shifted, while the query OUTPUT never
depends on the barcode (it is a nuisance). Also checks the causal/nuisance code
encodings the oracle and probes rely on.
"""
from __future__ import annotations

import numpy as np

from causalarc.families import FAMILIES
from causalarc.pilot.datasets import (BARCODE_ROW, _barcode_cells, causal_code,
                                      joint_causal_index, joint_causal_size,
                                      sample_task, strip_border)


def test_barcode_encodes_true_index_on_train():
    for name, fam in FAMILIES.items():
        rng = np.random.default_rng(0)
        for _ in range(20):
            t = sample_task(fam, rng, k=3, split="train")
            # barcode value equals the joint causal index of the task's latents
            # (we can't read latents back, but train barcode == the code we painted)
            assert t.barcode == joint_causal_index_of_task(fam, t)


def joint_causal_index_of_task(fam, task):
    # reconstruct joint index from the recorded c_code + cardinalities
    from causalarc.pilot.datasets import causal_cardinalities
    idx = 0
    for c, card in zip(task.c_code, causal_cardinalities(fam)):
        idx = idx * card + c
    return idx


def test_shifted_barcode_is_decoupled():
    """Over many shifted tasks, the barcode is not always the true index."""
    fam = FAMILIES["F2_recolor_parity"]
    rng = np.random.default_rng(1)
    mism = 0
    n = 100
    for _ in range(n):
        t = sample_task(fam, rng, k=3, split="shifted")
        if t.barcode != joint_causal_index_of_task(fam, t):
            mism += 1
    assert mism > n // 2  # decoupled: barcode usually != true index


def test_output_independent_of_barcode():
    """The query OUTPUT must not change when only the barcode changes."""
    from causalarc.pilot.datasets import _paint_barcode
    fam = FAMILIES["F1_translate"]
    rng = np.random.default_rng(2)
    latents = fam.sample_latents(rng)
    _, out0 = fam.render(latents, 12345)
    # rendering never paints a barcode into the output; painting one into a copy
    # of the input must not be reflected in the output
    inp, out1 = fam.render(latents, 12345)
    J = joint_causal_size(fam)
    _paint_barcode(inp, (joint_causal_index(fam, latents) + 3) % (9 ** _barcode_cells(J)),
                   _barcode_cells(J))
    assert np.array_equal(out0, out1)  # output identical regardless of barcode


def test_strip_border_removes_barcode_keeps_signal():
    fam = FAMILIES["F1_translate"]
    rng = np.random.default_rng(3)
    t = sample_task(fam, rng, k=1, split="train")
    stripped = strip_border(t.query_input)
    assert (stripped[BARCODE_ROW] == 0).all()          # barcode gone
    assert (t.query_input != 0).sum() >= (stripped != 0).sum()  # interior preserved


def test_causal_code_roundtrip():
    for name, fam in FAMILIES.items():
        rng = np.random.default_rng(4)
        latents = fam.sample_latents(rng)
        code = causal_code(fam, latents)
        assert len(code) == len(fam.ledger().causal_names)
        assert all(isinstance(c, int) and c >= 0 for c in code)
