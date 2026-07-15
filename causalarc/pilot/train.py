"""Train one baseline on one family/seed; evaluate iid vs shifted; probe z.

Data is sampled online from the planted generators (the supply is effectively
infinite), so there is no train-set memorization confound. We train on the
`train` split (barcode coupled to C), then measure exact-match grid accuracy on
`iid` (coupled) and `shifted` (do(U): barcode recombined). The OOD gap is
iid_acc - shifted_acc. Finally we freeze the encoder and fit linear probes that
decode each causal and nuisance latent from z — the pilot's causal-score metric.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..families import Family
from .datasets import (causal_cardinalities, nuisance_spec, sample_batch)
from .models import Batch, BaselineModel


@dataclass
class PilotConfig:
    k: int = 3
    steps: int = 1500
    batch: int = 64
    lr: float = 2e-3
    z_dim: int = 64
    hidden: int = 48
    beta: float = 1e-2          # cond-IB KL weight
    eval_n: int = 256
    probe_n: int = 512
    probe_steps: int = 300
    device: str = "auto"


def _device(cfg: PilotConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def c_onehot_dim(family: Family) -> int:
    return int(sum(causal_cardinalities(family)))


def collate(tasks, family: Family) -> Batch:
    cards = causal_cardinalities(family)
    demo_in = torch.tensor(np.stack([t.demo_inputs for t in tasks]), dtype=torch.long)
    demo_out = torch.tensor(np.stack([t.demo_outputs for t in tasks]), dtype=torch.long)
    query_in = torch.tensor(np.stack([t.query_input for t in tasks]), dtype=torch.long)
    query_out = torch.tensor(np.stack([t.query_output for t in tasks]), dtype=torch.long)
    barcode = torch.tensor([t.barcode for t in tasks], dtype=torch.long)
    oh = []
    for t in tasks:
        vec = []
        for idx, card in zip(t.c_code, cards):
            e = np.zeros(card, dtype=np.float32); e[idx] = 1.0
            vec.append(e)
        oh.append(np.concatenate(vec))
    c_onehot = torch.tensor(np.stack(oh), dtype=torch.float32)
    return Batch(demo_in, demo_out, query_in, query_out, c_onehot, barcode)


def _seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _exact_match(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)                       # (B,H,W)
    per_grid = (pred == target).all(dim=(-1, -2))     # (B,)
    return per_grid.float().mean().item()


def _cell_acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


@torch.no_grad()
def evaluate(model, family, rng, cfg, split: str, device) -> Dict[str, float]:
    model.eval()
    tasks = sample_batch(family, rng, cfg.eval_n, cfg.k, split)
    b = collate(tasks, family).to(device)
    logits, _ = model(b)
    return {"exact": _exact_match(logits, b.query_out),
            "cell": _cell_acc(logits, b.query_out)}


def _linear_probe(Z: torch.Tensor, y: torch.Tensor, n_classes: int, steps: int, device) -> float:
    """Fit a linear classifier Z->y, return held-out accuracy (50/50 split)."""
    n = Z.shape[0]; ntr = n // 2
    clf = torch.nn.Linear(Z.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    Ztr, ytr = Z[:ntr], y[:ntr]
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(clf(Ztr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (clf(Z[ntr:]).argmax(1) == y[ntr:]).float().mean().item()
    return acc


@torch.no_grad()
def _collect_codes(model, family, rng, cfg, device):
    tasks = sample_batch(family, rng, cfg.probe_n, cfg.k, "iid")
    b = collate(tasks, family).to(device)
    z, _ = model.encode(b)
    c_codes = np.array([t.c_code for t in tasks])   # (N, nC)
    u_codes = np.array([t.u_code for t in tasks])   # (N, nU)
    return z.detach(), c_codes, u_codes


def probe_causal_score(model, family, rng, cfg, device) -> Dict[str, float]:
    """Mean linear-probe accuracy for causal vs nuisance latents from z."""
    model.eval()
    z, c_codes, u_codes = _collect_codes(model, family, rng, cfg, device)
    cards = causal_cardinalities(family)
    c_accs = []
    for j, card in enumerate(cards):
        y = torch.tensor(c_codes[:, j], dtype=torch.long, device=device)
        c_accs.append(_linear_probe(z, y, card, cfg.probe_steps, device))
    u_accs = []
    for j, (_, dom) in enumerate(nuisance_spec(family)):
        y = torch.tensor(u_codes[:, j], dtype=torch.long, device=device)
        if len(torch.unique(y)) < 2:
            continue
        u_accs.append(_linear_probe(z, y, len(dom), cfg.probe_steps, device))
    return {
        "causal_probe": float(np.mean(c_accs)) if c_accs else 0.0,
        "nuisance_probe": float(np.mean(u_accs)) if u_accs else 0.0,
    }


def train_one(kind: str, family: Family, seed: int, cfg: PilotConfig) -> Dict:
    _seed_all(seed)
    device = _device(cfg)
    rng = np.random.default_rng(seed)
    model = BaselineModel(kind, c_onehot_dim(family), cfg.z_dim, cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for step in range(cfg.steps):
        tasks = sample_batch(family, rng, cfg.batch, cfg.k, "train")
        b = collate(tasks, family).to(device)
        logits, kl = model(b)
        loss = F.cross_entropy(logits, b.query_out)
        if kl is not None:
            loss = loss + cfg.beta * kl
        opt.zero_grad(); loss.backward(); opt.step()

    eval_rng = np.random.default_rng(seed + 10_000)
    iid = evaluate(model, family, eval_rng, cfg, "iid", device)
    shifted = evaluate(model, family, eval_rng, cfg, "shifted", device)
    probe = probe_causal_score(model, family, eval_rng, cfg, device)

    return {
        "kind": kind, "family": family.name, "seed": seed,
        "iid_exact": iid["exact"], "iid_cell": iid["cell"],
        "shifted_exact": shifted["exact"], "shifted_cell": shifted["cell"],
        "ood_gap": iid["exact"] - shifted["exact"],
        "causal_probe": probe["causal_probe"], "nuisance_probe": probe["nuisance_probe"],
        "device": str(device),
    }
