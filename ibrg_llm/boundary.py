"""Track B — the boundary: compression vs. intervention.

The clean, controlled statement of the whole IB=RG phase:
  * IB compression **discards a nuisance** (shown in vib.py) — good.
  * IB compression **cannot fix a cheap spurious shortcut**: a class-level cue is a
    low-rate, valid minimal-sufficient statistic on the training distribution, so the
    bottleneck KEEPS it and OOD stays broken at every β.
  * An **intervention** — do(U) recombination in *training* (environments where the
    shortcut is decorrelated from the label) — is what breaks the shortcut and
    restores causal generalization.

Task: x = [causal block | shortcut block].  y depends only on the causal block
(moderately noisy → learnable but not trivial).  The shortcut block is a strong,
easy one-hot cue.
  condition "spurious":      train shortcut = y (perfect cue);  test shortcut random.
  condition "intervention":  train shortcut = y on half the data, random on the other
                             half (the do(U) recombination);    test shortcut random.
We sweep β for BOTH and compare OOD test accuracy and shortcut-reliance.
Prediction: spurious → test stays low at every β (compression can't help);
intervention → test high (causal recovered). CPU-only.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from ibrg_llm.vib import VIB

BETAS = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def make_data(rng, n, protos, n_classes, shortcut):
    """shortcut in {'aligned','random','recombined'}. Returns X, y, and the shortcut class."""
    n_c = protos.shape[1]
    y = rng.integers(0, n_classes, n)
    Xc = protos[y] + 0.9 * rng.standard_normal((n, n_c))       # causal: learnable-but-noisy
    if shortcut == "aligned":
        sc = y.copy()
    elif shortcut == "random":
        sc = rng.integers(0, n_classes, n)
    else:                                                       # recombined: half aligned, half random
        sc = y.copy()
        half = rng.random(n) < 0.5
        sc[half] = rng.integers(0, n_classes, int(half.sum()))
    Xs = np.zeros((n, n_classes), np.float32)
    Xs[np.arange(n), sc] = 4.0                                  # strong, easy one-hot cue
    X = np.concatenate([Xc, Xs], axis=1).astype(np.float32)
    return X, y.astype(np.int64), sc.astype(np.int64)


def sweep(condition, rng, protos, n_classes, cfg, seed):
    # train distribution differs by condition; test is always shortcut-random (OOD)
    tr_short = "aligned" if condition == "spurious" else "recombined"
    Xtr, ytr, sctr = make_data(rng, cfg["n_train"], protos, n_classes, tr_short)
    Xte, yte, scte = make_data(rng, cfg["n_test"], protos, n_classes, "random")
    out = []
    for beta in BETAS:
        torch.manual_seed(seed)
        m = VIB(Xtr.shape[1], cfg["z_dim"], n_classes)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
        for _ in range(cfg["epochs"]):
            m.train(); opt.zero_grad()
            lg, kl, _ = m(xt)
            (F.cross_entropy(lg, yt) + beta * kl.mean()).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lg_tr, kl_tr, _ = m(xt, sample=False)
            lg_te, _, mu_te = m(torch.tensor(Xte), sample=False)
        Zte = mu_te.numpy(); h = len(Zte) // 2
        sc_ret = float(balanced_accuracy_score(
            scte[h:], LogisticRegression(max_iter=2000).fit(Zte[:h], scte[:h]).predict(Zte[h:])))
        out.append({
            "beta": beta, "rate": float(kl_tr.mean()),
            "train_acc": float((lg_tr.argmax(1).numpy() == ytr).mean()),
            "test_acc": float((lg_te.argmax(1).numpy() == yte).mean()),   # OOD, clean causal
            "shortcut_retention": sc_ret,
        })
        print(f"[{condition}] beta={beta:<6} test={out[-1]['test_acc']:.3f} "
              f"train={out[-1]['train_acc']:.3f} shortcut_ret={sc_ret:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_classes", type=int, default=4)
    ap.add_argument("--n_c", type=int, default=4)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    rng = np.random.default_rng(args.seed)
    protos = rng.standard_normal((args.n_classes, args.n_c)) * 2.0
    cfg = {"z_dim": args.z_dim, "epochs": args.epochs, "n_train": args.n_train, "n_test": args.n_test}
    res = {c: sweep(c, np.random.default_rng(args.seed + 1), protos, args.n_classes, cfg, args.seed)
           for c in ("spurious", "intervention")}
    sp = max(r["test_acc"] for r in res["spurious"])
    iv = max(r["test_acc"] for r in res["intervention"])
    print(f"[boundary] best OOD test — spurious(compression only)={sp:.3f}  intervention={iv:.3f}", flush=True)
    odir = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(odir, exist_ok=True)
    json.dump({"seed": args.seed, "cfg": vars(args), "conditions": res},
              open(os.path.join(odir, f"boundary_{tag}.json"), "w"), indent=2)
    print(f"[saved] out/boundary_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
