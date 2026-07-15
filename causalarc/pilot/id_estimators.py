"""Intrinsic-dimension estimators + validation on known-dimension manifolds (P1.3).

ID is a *descriptor, never ground truth* (the plan's words). Before trusting any
ID number on a learned representation, we validate the estimators on synthetic
manifolds whose dimension we know, and report where each one breaks (all local
ID estimators underestimate at high d and are sensitive to curvature/sampling).

Implements:
  * TwoNN (Facco et al. 2017) — ratio of 2nd/1st nearest-neighbor distances.
  * Levina-Bickel MLE (2004) — maximum-likelihood over k neighbors.
Dependency-light: numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


def _knn_dists(X: np.ndarray, k: int) -> np.ndarray:
    """Return sorted distances to the k nearest neighbors (excluding self)."""
    # pairwise squared distances
    sq = np.sum(X**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 0.0)
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    nn = np.take_along_axis(d2, idx, axis=1)
    nn = np.sort(nn, axis=1)
    return np.sqrt(nn)


def twonn(X: np.ndarray, discard_frac: float = 0.1) -> float:
    """TwoNN estimator. Uses mu = r2/r1; fits d via the empirical CDF line
    -log(1 - F(mu)) = d * log(mu) through the origin, discarding the top tail."""
    d = _knn_dists(X, 2)
    r1, r2 = d[:, 0], d[:, 1]
    good = r1 > 0
    mu = r2[good] / r1[good]
    mu = np.sort(mu)
    n = len(mu)
    keep = int(n * (1 - discard_frac))
    mu = mu[:keep]
    F = (np.arange(1, keep + 1)) / (n + 1)
    x = np.log(mu)
    y = -np.log(1 - F)
    # least-squares slope through origin
    d_hat = float(np.sum(x * y) / np.sum(x * x))
    return d_hat


def mle_levina_bickel(X: np.ndarray, k1: int = 10, k2: int = 20) -> float:
    """Levina-Bickel MLE, averaged over k in [k1, k2] and over points."""
    d = _knn_dists(X, k2 + 1)
    ests = []
    for k in range(k1, k2 + 1):
        Tk = d[:, k - 1:k]                    # distance to k-th neighbor
        Tj = d[:, : k - 1]                    # distances to 1..k-1
        with np.errstate(divide="ignore"):
            logs = np.log(Tk / Tj)
        m = (k - 2) / np.sum(logs, axis=1)    # per-point inverse-mean-log
        ests.append(np.mean(m[np.isfinite(m)]))
    return float(np.mean(ests))


# ---- known-dimension manifolds ---------------------------------------------

def sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on S^d embedded in R^(d+1); intrinsic dim = d."""
    X = rng.standard_normal((n, d + 1))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform in [0,1]^d; intrinsic dim = d."""
    return rng.random((n, d))


def torus(n: int, rng: np.random.Generator, R: float = 2.0, r: float = 1.0) -> np.ndarray:
    """2-torus embedded in R^3; intrinsic dim = 2."""
    u = rng.random(n) * 2 * np.pi
    v = rng.random(n) * 2 * np.pi
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.stack([x, y, z], axis=1)


@dataclass
class IDValidation:
    manifold: str
    true_dim: int
    twonn: float
    mle: float


def validate_estimators(n: int = 2000, seed: int = 0) -> Dict[str, IDValidation]:
    """Run both estimators on manifolds of known dimension; report where they
    diverge from truth (expect high-d underestimation)."""
    rng = np.random.default_rng(seed)
    cases: Dict[str, Tuple[np.ndarray, int]] = {
        "sphere_S2": (sphere(n, 2, rng), 2),
        "sphere_S5": (sphere(n, 5, rng), 5),
        "sphere_S10": (sphere(n, 10, rng), 10),
        "cube_d3": (hypercube(n, 3, rng), 3),
        "cube_d8": (hypercube(n, 8, rng), 8),
        "torus_2d": (torus(n, rng), 2),
    }
    out = {}
    for name, (X, td) in cases.items():
        out[name] = IDValidation(name, td, round(twonn(X), 2), round(mle_levina_bickel(X), 2))
    return out


def _print_validation(res: Dict[str, IDValidation]) -> None:
    print(f"{'manifold':12s} {'true':>4s} {'TwoNN':>7s} {'MLE':>7s}")
    for r in res.values():
        print(f"{r.manifold:12s} {r.true_dim:>4d} {r.twonn:>7.2f} {r.mle:>7.2f}")


if __name__ == "__main__":
    _print_validation(validate_estimators())
