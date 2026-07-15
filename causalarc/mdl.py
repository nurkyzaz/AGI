"""The single, pre-registered MDL proxy (P0.4): LZ-78 on fixed-precision
state bitstrings.

Per the plan, this is the *one* description-length proxy used throughout the
project. Neural KL, program-token counts, and zipped weights are NOT
interchangeable absolute bit-units; fixing LZ-78 here prevents silently
switching proxies to chase a result. The estimate is the LZ-78 dictionary size
times the bits-per-phrase code length — a standard, deterministic universal-code
length that needs no training and no external library.

Determinism and reproducibility are the whole point: `lz78_bits` is a pure
function of the byte string, and `mdl_bits_of_grid` serializes a grid to a fixed
row-major state bitstring before measuring.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .grid import Grid


def lz78_bits(data: bytes) -> float:
    """LZ-78 codelength in bits.

    Parse `data` into LZ-78 phrases (the classic incremental-dictionary parse).
    A dictionary of D phrases costs sum_i ceil(log2(i)) bits for the phrase
    pointers plus one literal symbol (8 bits) per phrase. This is the standard
    LZ-78 description length and is a deterministic function of the input.
    """
    if not data:
        return 0.0
    dictionary = {b"": 0}
    next_index = 1
    phrases = 0
    current = b""
    pointer_bits = 0.0
    for byte in data:
        current = current + bytes([byte])
        if current not in dictionary:
            # emit a phrase: pointer to prefix (log2 of current dict size) + 8-bit literal
            pointer_bits += math.log2(max(1, next_index))
            dictionary[current] = next_index
            next_index += 1
            phrases += 1
            current = b""
    if current:  # trailing partial phrase points at an existing dictionary entry
        pointer_bits += math.log2(max(1, next_index))
        phrases += 1
    literal_bits = 8.0 * phrases
    return pointer_bits + literal_bits


def grid_to_bitstring(g: Grid) -> bytes:
    """Fixed row-major state serialization: shape header + one byte per cell.
    The shape header makes the code prefix-free across differently-sized grids."""
    h, w = g.shape
    header = bytes([h & 0xFF, w & 0xFF])
    return header + g.astype(np.uint8).tobytes()


def mdl_bits_of_grid(g: Grid) -> float:
    return lz78_bits(grid_to_bitstring(g))


def mdl_bits_of_grids(grids: Iterable[Grid]) -> float:
    """MDL of a concatenated sequence of grids (e.g., all outputs of a task)."""
    buf = b"".join(grid_to_bitstring(g) for g in grids)
    return lz78_bits(buf)
