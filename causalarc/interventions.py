"""Interventions, effect measurement, and causal-mask recovery.

do(var=value) means: swap a single latent and re-render with the *same*
content_seed and all other latents fixed. Any change in the OUTPUT is then
attributable to that one latent. This module turns that primitive into two
GATE-0 instruments:

  * `variable_effect_rate`  — the fraction of (content, alternative-value) pairs
    on which intervening on a variable changes the output. C variables should be
    > 0 (they *can* move the target); U variables should be exactly 0.
  * `recover_causal_mask`   — classifies each variable causal/nuisance purely
    from effect rates, then scores precision/recall against the ledger's
    ground-truth roles (GATE 0 item 3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .grid import grids_equal
from .ledger import Family, Role


def variable_effect_rate(
    family: Family,
    var_name: str,
    n_content: int = 40,
    n_base: int = 3,
    seed: int = 0,
) -> float:
    """Fraction of (base-latents, content, alternative-value) triples on which
    do(var_name=alt) changes the output. 0.0 == provably inert here."""
    rng = np.random.default_rng(seed)
    ledger = family.ledger()
    var = ledger.var(var_name)
    content_seeds = [int(rng.integers(1, 2**31)) for _ in range(n_content)]

    changed = 0
    total = 0
    for _ in range(n_base):
        latents = family.sample_latents(rng)
        alts = var.other_values(latents[var_name])
        for cs in content_seeds:
            _, base_out = family.render(latents, cs)
            for alt in alts:
                inter = family.with_intervention(latents, var_name, alt)
                _, alt_out = family.render(inter, cs)
                total += 1
                if not grids_equal(base_out, alt_out):
                    changed += 1
    return changed / total if total else 0.0


@dataclass
class MaskResult:
    family: str
    effect_rates: Dict[str, float]
    ground_truth_causal: List[str]
    predicted_causal: List[str]
    precision: float
    recall: float

    @property
    def perfect(self) -> bool:
        return self.precision == 1.0 and self.recall == 1.0


def recover_causal_mask(
    family: Family,
    n_content: int = 40,
    n_base: int = 3,
    seed: int = 0,
    threshold: float = 0.0,
) -> MaskResult:
    """Recover which variables are causal from effect rates alone and score it
    against the ledger. A variable is predicted causal iff its effect rate
    strictly exceeds `threshold` (default 0 — U variables are exactly inert by
    construction, so any nonzero rate is a true causal signal)."""
    ledger = family.ledger()
    rates: Dict[str, float] = {}
    for v in ledger.variables:
        rates[v.name] = variable_effect_rate(
            family, v.name, n_content=n_content, n_base=n_base, seed=seed
        )

    predicted = [name for name, r in rates.items() if r > threshold]
    truth = ledger.causal_names

    pred_set, truth_set = set(predicted), set(truth)
    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set) if pred_set else 1.0
    recall = tp / len(truth_set) if truth_set else 1.0

    return MaskResult(
        family=family.name,
        effect_rates=rates,
        ground_truth_causal=truth,
        predicted_causal=predicted,
        precision=precision,
        recall=recall,
    )


def intervention_tests(
    family: Family, n_content: int = 40, n_base: int = 3, seed: int = 0
) -> Tuple[bool, Dict[str, Any]]:
    """GATE 0 item 1: every C variable can change the target (rate > 0) and
    every U variable never does (rate == 0). Returns (passed, detail)."""
    ledger = family.ledger()
    detail: Dict[str, Any] = {}
    passed = True
    for v in ledger.variables:
        rate = variable_effect_rate(family, v.name, n_content=n_content, n_base=n_base, seed=seed)
        if v.role is Role.C:
            ok = rate > 0.0
        else:
            ok = rate == 0.0
        detail[v.name] = {"role": v.role.value, "effect_rate": rate, "ok": ok}
        passed = passed and ok
    return passed, detail
