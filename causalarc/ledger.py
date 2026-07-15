"""Causal-ledger schema and the Family abstract base class.

The ledger is the load-bearing annotation of Phase 0: for every planted family
it records which latent variables are CAUSAL (C — they determine the output
rule) versus NUISANCE (U — they change appearance but the rule is invariant to
them). GATE 0 is, in essence, the claim that this annotation is *correct* and
*machine-recoverable* on families we planted ourselves.

A Family separates two sources of randomness on purpose:

    latents       : the generative parameters we annotate as C or U.
    content_seed  : the per-pair substrate the rule acts on (the "data").

Holding `content_seed` fixed while we intervene on a single latent is what makes
an intervention test clean: any change in the OUTPUT is attributable to that one
latent, not to resampled content.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np

from .grid import Grid


class Role(str, Enum):
    C = "C"  # causal / relevant: the output rule depends on it
    U = "U"  # nuisance: appearance changes, output rule is invariant


@dataclass(frozen=True)
class Variable:
    name: str
    role: Role
    domain: Tuple[Any, ...]  # finite set of legal values (interventions draw here)
    description: str = ""

    def other_values(self, current: Any) -> List[Any]:
        """Legal intervention targets: every domain value except the current one.
        Values compared structurally so tuples/lists work as domain members."""
        out = []
        for v in self.domain:
            if not _val_equal(v, current):
                out.append(v)
        return out


def _val_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


@dataclass(frozen=True)
class CausalLedger:
    """Immutable, hashable annotation for one family."""

    family: str
    variables: Tuple[Variable, ...]
    reference_rule: str  # human-readable description of output = f(input, C)
    rule_type: str  # e.g. "geometric", "recolor", "object-select"

    @property
    def causal_names(self) -> List[str]:
        return [v.name for v in self.variables if v.role is Role.C]

    @property
    def nuisance_names(self) -> List[str]:
        return [v.name for v in self.variables if v.role is Role.U]

    def var(self, name: str) -> Variable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "reference_rule": self.reference_rule,
            "rule_type": self.rule_type,
            "variables": [
                {
                    "name": v.name,
                    "role": v.role.value,
                    "domain": _jsonable(v.domain),
                    "description": v.description,
                }
                for v in self.variables
            ],
        }

    def hash(self) -> str:
        """Stable content hash — pins the ledger into the run record."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.astype(int).tolist()
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


class Family(ABC):
    """A planted ARC-style generator with a declared causal ledger.

    Subclasses implement `ledger()`, `sample_latents()`, and `render()`.
    The base class provides the machinery every family shares.
    """

    name: str = "family"

    @abstractmethod
    def ledger(self) -> CausalLedger: ...

    @abstractmethod
    def sample_latents(self, rng: np.random.Generator) -> Dict[str, Any]: ...

    @abstractmethod
    def render(self, latents: Dict[str, Any], content_seed: int) -> Tuple[Grid, Grid]:
        """Deterministically produce (input_grid, output_grid) where
        output = f(input, {C latents}) and U latents touch only nuisance
        regions absent from the output."""

    # ---- shared helpers -------------------------------------------------

    def with_intervention(
        self, latents: Dict[str, Any], name: str, value: Any
    ) -> Dict[str, Any]:
        """Return a copy of `latents` with `name` set to `value` — this is do(name=value)."""
        out = dict(latents)
        out[name] = value
        return out

    def content_pairs(self, latents: Dict[str, Any], seeds: List[int]) -> List[Tuple[Grid, Grid]]:
        return [self.render(latents, s) for s in seeds]
